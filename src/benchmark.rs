//! Reproducible recall / latency / ingest benchmark suite.
//!
//! Primary path: Modal-generated **MRL** embeddings from
//! `astr010/sec-10k-lsh-chunks` (`data/sec10k_mrl/`). Isotropic random vectors
//! are available only via `--synthetic` for plumbing tests — they do **not**
//! evaluate the MRL-prefix design.

use std::path::Path;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use crate::block::{source, PaxBlock64, VECTORS_PER_BLOCK};
use crate::config::{EngineConfig, SlabResidency, StrataDepth};
use crate::corpus::MrlCorpus;
use crate::embed::dot;
use crate::engine::IndexEngine;
use crate::quantize::encode_block;
use crate::scan::{scan_blocks, QueryPlan, ScoredHit};

#[derive(Debug, Clone, Serialize)]
pub struct ScanRecallReport {
    pub label: String,
    pub corpus: String,
    pub mrl: bool,
    pub model: Option<String>,
    pub n_vectors: usize,
    pub n_queries: usize,
    pub dims: usize,
    pub shortlist: usize,
    pub strata: String,
    pub recall_at_10: f64,
    pub recall_at_shortlist: f64,
    pub mean_scan_ms: f64,
    pub mvecs_per_sec: f64,
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
    let denom = want.len().min(pred_ids.len()).max(1);
    got as f64 / denom as f64
}

fn run_scan_on_vectors(
    label: &str,
    corpus_name: &str,
    mrl: bool,
    model: Option<String>,
    vectors: &[Vec<f32>],
    query_indices: &[usize],
    shortlist: usize,
    strata: StrataDepth,
) -> ScanRecallReport {
    let dims = vectors[0].len();
    let n_vectors = vectors.len();
    let n_queries = query_indices.len();
    let blocks = encode_corpus(vectors, dims);

    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.shortlist_cand = shortlist;
    cfg.strata_depth = strata;
    cfg.temporal_filter_enable = false;

    let mut recall10 = 0.0;
    let mut recall_sl = 0.0;
    let mut scan_ms_sum = 0.0;

    for &qi in query_indices {
        let q = &vectors[qi];
        let truth = exact_topk(q, vectors, shortlist.max(10));
        // Exclude the query doc itself from truth for retrieval-style scoring.
        let truth: Vec<(usize, f32)> = truth.into_iter().filter(|(i, _)| *i != qi).collect();
        let plan = QueryPlan::from_query(q, dims);

        let t = Instant::now();
        let hits = scan_blocks(&blocks, &plan, &cfg, None, None);
        scan_ms_sum += t.elapsed().as_secs_f64() * 1000.0;

        let ids: Vec<usize> = hit_ids(&hits).into_iter().filter(|i| *i != qi).collect();
        recall10 += recall_overlap(&truth, &ids, 10);
        recall_sl += recall_overlap(&truth, &ids, shortlist);
    }

    let mean_scan_ms = scan_ms_sum / n_queries as f64;
    let mvecs = (n_vectors as f64 / (mean_scan_ms / 1000.0)) / 1e6;

    ScanRecallReport {
        label: label.into(),
        corpus: corpus_name.into(),
        mrl,
        model,
        n_vectors,
        n_queries,
        dims,
        shortlist,
        strata: strata_name(strata),
        recall_at_10: recall10 / n_queries as f64,
        recall_at_shortlist: recall_sl / n_queries as f64,
        mean_scan_ms,
        mvecs_per_sec: mvecs,
    }
}

/// Primary bench: real MRL embeddings from Modal (`data/sec10k_mrl`).
pub fn run_scan_recall_mrl(
    label: &str,
    embeddings_dir: &Path,
    n_queries: usize,
    shortlist: usize,
    strata: StrataDepth,
    seed: u64,
) -> Result<ScanRecallReport, crate::corpus::CorpusError> {
    let corpus = MrlCorpus::load(embeddings_dir)?;
    let vectors = corpus.all_vectors();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = vectors.len();
    let n_queries = n_queries.min(n.max(1));
    let mut query_indices = Vec::with_capacity(n_queries);
    while query_indices.len() < n_queries {
        let i = rng.gen_range(0..n);
        if !query_indices.contains(&i) {
            query_indices.push(i);
        }
    }
    Ok(run_scan_on_vectors(
        label,
        &format!("{} / {}", corpus.manifest.dataset, embeddings_dir.display()),
        true,
        Some(corpus.manifest.model.clone()),
        &vectors,
        &query_indices,
        shortlist,
        strata,
    ))
}

/// Plumbing-only: isotropic random unit vectors (NOT an MRL evaluation).
pub fn run_scan_recall_synthetic(
    label: &str,
    n_vectors: usize,
    n_queries: usize,
    dims: usize,
    shortlist: usize,
    strata: StrataDepth,
    seed: u64,
) -> ScanRecallReport {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| {
            let mut v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
            l2_normalize(&mut v);
            v
        })
        .collect();
    let query_indices: Vec<usize> = (0..n_queries.min(n_vectors)).collect();
    run_scan_on_vectors(
        label,
        "synthetic_isotropic_unit",
        false,
        None,
        &vectors,
        &query_indices,
        shortlist,
        strata,
    )
}

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

/// Full suite against Modal MRL artifacts.
pub fn run_suite_mrl(
    label: &str,
    embeddings_dir: &Path,
) -> Result<BenchSuiteReport, crate::corpus::CorpusError> {
    let scan = run_scan_recall_mrl(
        label,
        embeddings_dir,
        64,
        64,
        StrataDepth::L1Rabitq,
        42,
    )?;
    let ingest = run_ingest_bench(label, 4_096, scan.dims);
    Ok(BenchSuiteReport {
        label: label.into(),
        scan,
        ingest,
    })
}

impl BenchSuiteReport {
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize")
    }

    pub fn print_summary(&self) {
        println!("=== mrl-index bench suite [{}] ===", self.label);
        println!(
            "corpus: {}  mrl={}  model={}",
            self.scan.corpus,
            self.scan.mrl,
            self.scan.model.as_deref().unwrap_or("-")
        );
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
