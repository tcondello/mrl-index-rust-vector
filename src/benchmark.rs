//! Reproducible recall / latency / ingest benchmark suite.
//!
//! Used to establish a baseline before algorithmic fixes and to measure deltas.

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use crate::block::{source, PaxBlock64, VECTORS_PER_BLOCK};
use crate::config::{EngineConfig, SlabResidency, StrataDepth};
use crate::embed::dot;
use crate::engine::IndexEngine;
use crate::quantize::encode_block;
use crate::scan::{scan_blocks, QueryPlan, ScoredHit};

#[derive(Debug, Clone, Serialize)]
pub struct ScanRecallReport {
    pub label: String,
    pub n_vectors: usize,
    pub n_queries: usize,
    pub dims: usize,
    pub shortlist: usize,
    pub strata: String,
    pub recall_at_10: f64,
    pub recall_at_shortlist: f64,
    pub mean_scan_ms: f64,
    pub mvecs_per_sec: f64,
    pub mean_exact_overlap_at_10: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct IngestReport {
    pub label: String,
    pub n_docs: usize,
    pub dims: usize,
    pub total_ms: f64,
    pub docs_per_sec: f64,
    pub blocks_written: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchSuiteReport {
    pub label: String,
    pub scan: ScanRecallReport,
    pub ingest: IngestReport,
}

fn strata_name(s: StrataDepth) -> String {
    match s {
        StrataDepth::L0Only => "l0_only".into(),
        StrataDepth::L1Rabitq => "l1_rabitq".into(),
        StrataDepth::Full64b => "full_64b".into(),
    }
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v {
        *x /= n;
    }
}

fn random_unit(rng: &mut ChaCha8Rng, dims: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
    l2_normalize(&mut v);
    v
}

fn make_corpus(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| random_unit(&mut rng, dims)).collect()
}

fn encode_corpus(vecs: &[Vec<f32>], dims: usize) -> Vec<PaxBlock64> {
    let mut blocks = Vec::new();
    for chunk in vecs.chunks(VECTORS_PER_BLOCK) {
        let refs: Vec<&[f32]> = chunk.iter().map(|v| v.as_slice()).collect();
        let ts: Vec<u32> = (0..chunk.len()).map(|i| i as u32).collect();
        let src = vec![source::DOC; chunk.len()];
        blocks.push(encode_block(&refs, &ts, &src, dims));
    }
    blocks
}

fn exact_topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, dot(query, v)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    scored
}

fn hit_ids(hits: &[ScoredHit]) -> Vec<usize> {
    hits.iter()
        .map(|h| h.id.block as usize * VECTORS_PER_BLOCK + h.id.slot as usize)
        .collect()
}

fn recall_overlap(truth: &[(usize, f32)], pred_ids: &[usize], k: usize) -> f64 {
    let want: std::collections::HashSet<usize> =
        truth.iter().take(k).map(|(i, _)| *i).collect();
    if want.is_empty() {
        return 1.0;
    }
    let got: usize = pred_ids.iter().filter(|id| want.contains(id)).count();
    // Cap by min(k, shortlist size) for fair rate.
    let denom = want.len().min(pred_ids.len()).max(1);
    got as f64 / denom as f64
}

/// Scan recall + latency against exact brute-force top-k on synthetic unit vectors.
pub fn run_scan_recall_bench(
    label: &str,
    n_vectors: usize,
    n_queries: usize,
    dims: usize,
    shortlist: usize,
    strata: StrataDepth,
    seed: u64,
) -> ScanRecallReport {
    let corpus = make_corpus(n_vectors, dims, seed);
    let blocks = encode_corpus(&corpus, dims);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xDEAD_BEEF);

    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.shortlist_cand = shortlist;
    cfg.strata_depth = strata;
    cfg.temporal_filter_enable = false;

    let mut recall10 = 0.0;
    let mut recall_sl = 0.0;
    let mut scan_ms_sum = 0.0;

    for _ in 0..n_queries {
        let q = random_unit(&mut rng, dims);
        let truth = exact_topk(&q, &corpus, shortlist.max(10));
        let plan = QueryPlan::from_query(&q, dims);

        let t = Instant::now();
        let hits = scan_blocks(&blocks, &plan, &cfg, None, None);
        scan_ms_sum += t.elapsed().as_secs_f64() * 1000.0;

        let ids = hit_ids(&hits);
        recall10 += recall_overlap(&truth, &ids, 10);
        recall_sl += recall_overlap(&truth, &ids, shortlist);
    }

    let mean_scan_ms = scan_ms_sum / n_queries as f64;
    let mvecs = (n_vectors as f64 / (mean_scan_ms / 1000.0)) / 1e6;

    ScanRecallReport {
        label: label.into(),
        n_vectors,
        n_queries,
        dims,
        shortlist,
        strata: strata_name(strata),
        recall_at_10: recall10 / n_queries as f64,
        recall_at_shortlist: recall_sl / n_queries as f64,
        mean_scan_ms,
        mvecs_per_sec: mvecs,
        mean_exact_overlap_at_10: recall10 / n_queries as f64,
    }
}

/// End-to-end ingest timing (exposes quadratic segment reload cost).
pub fn run_ingest_bench(label: &str, n_docs: usize, dims: usize) -> IngestReport {
    let dir = std::env::temp_dir().join(format!(
        "mrl-bench-ingest-{}-{}",
        std::process::id(),
        label
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.slab_residency = SlabResidency::RamLocked;
    cfg.strata_depth = StrataDepth::L1Rabitq;

    let mut engine = IndexEngine::create(&dir, cfg).expect("create");
    let items: Vec<(String, u32, u32)> = (0..n_docs)
        .map(|i| {
            (
                format!("benchmark document {i} about retrieval systems and bitplanes"),
                i as u32,
                source::DOC,
            )
        })
        .collect();

    let t = Instant::now();
    engine.ingest_batch(&items).expect("ingest");
    engine.flush().expect("flush");
    let total_ms = t.elapsed().as_secs_f64() * 1000.0;
    let blocks_written = engine.block_count();
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);

    IngestReport {
        label: label.into(),
        n_docs,
        dims,
        total_ms,
        docs_per_sec: n_docs as f64 / (total_ms / 1000.0),
        blocks_written,
    }
}

/// Full suite used for baseline → fix → compare.
pub fn run_suite(label: &str) -> BenchSuiteReport {
    // Sized to finish in seconds on a laptop while stressing pruning + ingest.
    let scan = run_scan_recall_bench(
        label,
        8_192, // 128 PaxBlocks
        32,
        384,
        64,
        StrataDepth::L1Rabitq,
        42,
    );
    let ingest = run_ingest_bench(label, 4_096, 384);
    BenchSuiteReport {
        label: label.into(),
        scan,
        ingest,
    }
}

impl BenchSuiteReport {
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize")
    }

    pub fn print_summary(&self) {
        println!("=== mrl-index bench suite [{}] ===", self.label);
        println!(
            "scan:  n={} q={}  Recall@10={:.4}  Recall@SL={:.4}  scan={:.3} ms  ({:.2} Mvecs/s)",
            self.scan.n_vectors,
            self.scan.n_queries,
            self.scan.recall_at_10,
            self.scan.recall_at_shortlist,
            self.scan.mean_scan_ms,
            self.scan.mvecs_per_sec
        );
        println!(
            "ingest: docs={}  blocks={}  total={:.1} ms  ({:.0} docs/s)",
            self.ingest.n_docs,
            self.ingest.blocks_written,
            self.ingest.total_ms,
            self.ingest.docs_per_sec
        );
    }
}
