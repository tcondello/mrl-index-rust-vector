//! Shared query orchestration: plan → scan → exact-IP rerank.
//!
//! [`IndexEngine`] and [`crate::retrieval_bench`] both drive the same scan kernel
//! through [`QueryPipeline`] so timing / shortlist identity stay aligned.

use std::time::Instant;

use crate::block::{PaxBlock64, VectorId, VECTORS_PER_BLOCK};
use crate::config::{EngineConfig, PrefixCodec};
use crate::eden::EdenFit;
use crate::embed::dot;
use crate::scan::{
    scan_block, scan_blocks, scan_blocks_stats, QueryPlan, ScanStats, ScoredHit, Shortlist,
};
use crate::segment::SegmentReader;

/// Thin façade over plan + scan + exact-IP helpers.
#[derive(Debug, Clone, Copy)]
pub struct QueryPipeline<'a> {
    pub cfg: &'a EngineConfig,
    pub eden: Option<&'a EdenFit>,
}

/// Result of a plan+scan phase (no fetch / neural rerank).
#[derive(Debug, Clone)]
pub struct ScanPhase {
    pub plan: QueryPlan,
    pub hits: Vec<ScoredHit>,
    pub scan_ms: f64,
    pub blocks_scanned: usize,
}

impl<'a> QueryPipeline<'a> {
    pub fn new(cfg: &'a EngineConfig) -> Self {
        Self { cfg, eden: None }
    }

    pub fn with_eden(cfg: &'a EngineConfig, fit: &'a EdenFit) -> Self {
        Self {
            cfg,
            eden: Some(fit),
        }
    }

    pub fn build_plan(&self, query: &[f32]) -> QueryPlan {
        debug_assert_eq!(query.len(), self.cfg.embed_dims);
        match (self.cfg.prefix_codec, self.eden) {
            (PrefixCodec::EdenProd, Some(fit)) => {
                QueryPlan::from_query_eden(query, self.cfg.embed_dims, fit)
            }
            _ => QueryPlan::from_query(query, self.cfg.embed_dims),
        }
    }

    /// Contiguous PaxBlock slab (mmap retrieval bench / in-memory corpora).
    pub fn scan_slab(
        &self,
        query: &[f32],
        blocks: &[PaxBlock64],
        time_lo: Option<u32>,
        time_hi: Option<u32>,
        mut stats: Option<&mut ScanStats>,
    ) -> ScanPhase {
        let plan = self.build_plan(query);
        let t = Instant::now();
        let hits = if let Some(s) = stats.as_deref_mut() {
            scan_blocks_stats(blocks, &plan, self.cfg, time_lo, time_hi, Some(s))
        } else {
            scan_blocks(blocks, &plan, self.cfg, time_lo, time_hi)
        };
        ScanPhase {
            plan,
            hits,
            scan_ms: t.elapsed().as_secs_f64() * 1000.0,
            blocks_scanned: blocks.len(),
        }
    }

    /// Segment-backed scan used by [`crate::engine::IndexEngine`].
    ///
    /// Fills `shortlist` from mapped slabs or per-block loads. Caller merges ring
    /// hits afterward, then calls [`Shortlist::into_sorted`].
    pub fn scan_segments_into(
        &self,
        plan: &QueryPlan,
        reader: &SegmentReader,
        shortlist: &mut Shortlist,
        time_lo: Option<u32>,
        time_hi: Option<u32>,
    ) -> usize {
        let n_blocks = reader.block_count();
        if let Some(blocks) = reader.as_slice() {
            for hit in scan_blocks(blocks, plan, self.cfg, time_lo, time_hi) {
                shortlist.push(hit);
            }
        } else {
            for i in 0..n_blocks {
                if let Some(block) = reader.block(i) {
                    scan_block(
                        block,
                        i as u32,
                        plan,
                        self.cfg,
                        shortlist,
                        time_lo,
                        time_hi,
                    );
                }
            }
        }
        n_blocks
    }

    /// Plan + timed segment scan (no ring merge).
    pub fn scan_segments(
        &self,
        query: &[f32],
        reader: &SegmentReader,
        time_lo: Option<u32>,
        time_hi: Option<u32>,
    ) -> (QueryPlan, Shortlist, f64, usize) {
        let plan = self.build_plan(query);
        let mut shortlist = Shortlist::new(self.cfg.shortlist_cand);
        let t = Instant::now();
        let n_blocks =
            self.scan_segments_into(&plan, reader, &mut shortlist, time_lo, time_hi);
        (
            plan,
            shortlist,
            t.elapsed().as_secs_f64() * 1000.0,
            n_blocks,
        )
    }

    /// Exact inner-product rerank over flat corpus row ids (retrieval bench path).
    pub fn rerank_exact_rows<'b>(
        query: &[f32],
        candidate_ids: &[usize],
        corpus_row: impl Fn(usize) -> &'b [f32],
    ) -> Vec<(usize, f32)> {
        let mut ranked: Vec<(usize, f32)> = candidate_ids
            .iter()
            .copied()
            .map(|i| (i, dot(query, corpus_row(i))))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }
}

/// Flatten scan hits to dense corpus row ids (`block * 64 + slot`).
///
/// Skips ring sentinel hits (`block == u32::MAX`).
pub fn scored_hits_to_row_ids(hits: &[ScoredHit]) -> Vec<usize> {
    hits.iter()
        .filter(|h| h.id.block != u32::MAX)
        .map(|h| h.id.block as usize * VECTORS_PER_BLOCK + h.id.slot as usize)
        .collect()
}

/// Inverse of [`scored_hits_to_row_ids`] for synthetic tests.
pub fn row_id_to_vector_id(row: usize) -> VectorId {
    VectorId::new(
        (row / VECTORS_PER_BLOCK) as u32,
        (row % VECTORS_PER_BLOCK) as u16,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::encode_block;
    use crate::StrataDepth;

    fn l2_normalize(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= n;
        }
    }

    #[test]
    fn pipeline_scan_slab_matches_scan_blocks() {
        let dims = 768;
        let mut rows = Vec::new();
        for i in 0..64usize {
            let mut v = vec![0.0f32; dims];
            for d in 0..dims {
                v[d] = ((i * 13 + d * 7) % 97) as f32 * 0.01 - 0.4;
            }
            l2_normalize(&mut v);
            rows.push(v);
        }
        let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        let ts: Vec<u32> = (0..64).map(|i| i as u32).collect();
        let src = vec![1u32; 64];
        let block = encode_block(&refs, &ts, &src, dims);

        let mut cfg = EngineConfig::default();
        cfg.embed_dims = dims;
        cfg.strata_depth = StrataDepth::Full64b;
        cfg.shortlist_cand = 16;
        cfg.numa_sharding = false;
        cfg.temporal_filter_enable = false;

        let pipe = QueryPipeline::new(&cfg);
        let phase = pipe.scan_slab(&rows[0], &[block], None, None, None);
        let direct = {
            let plan = QueryPlan::from_query(&rows[0], dims);
            scan_blocks(&[block], &plan, &cfg, None, None)
        };
        assert_eq!(phase.hits.len(), direct.len());
        for (a, b) in phase.hits.iter().zip(direct.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.score.to_bits(), b.score.to_bits());
        }

        let ids = scored_hits_to_row_ids(&phase.hits);
        let ranked = QueryPipeline::rerank_exact_rows(&rows[0], &ids, |i| rows[i].as_slice());
        assert_eq!(ranked.len(), ids.len());
        assert!(ranked.windows(2).all(|w| w[0].1 >= w[1].1));
    }
}
