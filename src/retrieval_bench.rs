//! Clean MS MARCO / EmbeddingGemma-2 retrieval benchmark.
//!
//! Uses **separate query embeddings** + **precomputed exact neighbors** + **qrels**.
//! This is not corpus self-retrieval and does not recompute brute-force GT at runtime.
//!
//! Corpus embeddings are **mmapped** so full-scale suites (~27 GiB at 8.8M × 768)
//! fit on ~16 GiB hosts; only PaxBlocks (~4 KB / 64 vecs) stay resident in RAM.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use memmap2::{Advice, Mmap, MmapOptions};
use serde::{Deserialize, Serialize};

use crate::block::{source, PaxBlock64, VECTORS_PER_BLOCK};
use crate::config::{EngineConfig, StrataDepth};
use crate::corpus::CorpusError;
use crate::embed::dot;
use crate::quantize::encode_block;
use crate::scan::{scan_blocks, QueryPlan, ScoredHit};

#[derive(Debug, Clone, Deserialize)]
pub struct RetrievalManifest {
    pub dataset: String,
    pub model: String,
    pub dims: usize,
    pub n: usize,
    pub mrl: bool,
    pub embeddings_file: String,
    #[serde(default)]
    pub n_queries: Option<usize>,
    #[serde(default)]
    pub n_qrels: Option<usize>,
    #[serde(default)]
    pub k_max: Option<usize>,
    #[serde(default)]
    pub query_embeddings_file: Option<String>,
    #[serde(default)]
    pub queries_file: Option<String>,
    #[serde(default)]
    pub neighbors_file: Option<String>,
    #[serde(default)]
    pub qrels_file: Option<String>,
    #[serde(default)]
    pub scale: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct NeighborRow {
    query_index: usize,
    neighbor_row_ids: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct QrelRow {
    query_index: usize,
    row_id: usize,
    score: i32,
}

/// Corpus float matrix backed by an immutable mmap (little-endian f32, row-major).
pub struct MmapF32Matrix {
    mmap: Mmap,
    rows: usize,
    dims: usize,
}

impl MmapF32Matrix {
    pub fn open(path: &Path, rows: usize, dims: usize) -> Result<Self, CorpusError> {
        let expected = rows
            .checked_mul(dims)
            .and_then(|n| n.checked_mul(4))
            .ok_or(CorpusError::CorruptSize {
                expected: 0,
                got: 0,
            })?;
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        if mmap.len() != expected {
            return Err(CorpusError::CorruptSize {
                expected,
                got: mmap.len(),
            });
        }
        // bytemuck cast needs align(4); mmap offset is page-aligned so OK.
        if mmap.as_ptr() as usize % std::mem::align_of::<f32>() != 0 {
            return Err(CorpusError::CorruptSize {
                expected,
                got: mmap.len(),
            });
        }
        Ok(Self { mmap, rows, dims })
    }

    pub fn advise_sequential(&self) {
        let _ = self.mmap.advise(Advice::Sequential);
    }

    pub fn advise_random(&self) {
        let _ = self.mmap.advise(Advice::Random);
    }

    pub fn as_slice(&self) -> &[f32] {
        bytemuck::cast_slice(&self.mmap)
    }

    pub fn row(&self, i: usize) -> &[f32] {
        debug_assert!(i < self.rows);
        let d = self.dims;
        &self.as_slice()[i * d..(i + 1) * d]
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn dims(&self) -> usize {
        self.dims
    }
}

pub struct RetrievalBenchData {
    pub dir: PathBuf,
    pub manifest: RetrievalManifest,
    pub corpus_embeddings: MmapF32Matrix,
    pub query_embeddings: Vec<f32>,
    pub neighbors: Vec<Vec<usize>>,
    pub qrels: HashMap<usize, HashSet<usize>>,
}

impl RetrievalBenchData {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, CorpusError> {
        let dir = dir.as_ref().to_path_buf();
        let manifest_path = dir.join("manifest.json");
        if !manifest_path.exists() {
            return Err(CorpusError::MissingManifest(manifest_path));
        }
        let manifest: RetrievalManifest =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
        if !manifest.mrl {
            return Err(CorpusError::NotMrl);
        }
        let n_queries = match manifest.n_queries {
            Some(n) if n > 0 => n,
            _ => {
                return Err(CorpusError::MissingManifest(
                    dir.join("manifest.json (n_queries)"),
                ))
            }
        };

        let q_emb_name = manifest
            .query_embeddings_file
            .as_deref()
            .ok_or_else(|| CorpusError::MissingManifest(dir.join("query_embeddings.bin")))?;
        let neighbors_name = manifest
            .neighbors_file
            .as_deref()
            .ok_or_else(|| CorpusError::MissingManifest(dir.join("neighbors.jsonl")))?;
        let qrels_name = manifest
            .qrels_file
            .as_deref()
            .unwrap_or("qrels.jsonl");

        eprintln!(
            "mmap corpus embeddings {} (n={} dims={}) …",
            manifest.embeddings_file, manifest.n, manifest.dims
        );
        let corpus_embeddings = MmapF32Matrix::open(
            &dir.join(&manifest.embeddings_file),
            manifest.n,
            manifest.dims,
        )?;
        let query_embeddings =
            read_f32_matrix(&dir.join(q_emb_name), n_queries, manifest.dims)?;

        let mut neighbors = vec![Vec::new(); n_queries];
        let nf = File::open(dir.join(neighbors_name))?;
        for line in BufReader::new(nf).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row: NeighborRow = serde_json::from_str(&line)?;
            if row.query_index >= n_queries {
                continue;
            }
            neighbors[row.query_index] = row.neighbor_row_ids;
        }

        let mut qrels: HashMap<usize, HashSet<usize>> = HashMap::new();
        let qf = dir.join(qrels_name);
        if qf.exists() {
            let file = File::open(qf)?;
            for line in BufReader::new(file).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let row: QrelRow = serde_json::from_str(&line)?;
                if row.score > 0 {
                    qrels.entry(row.query_index).or_default().insert(row.row_id);
                }
            }
        }

        Ok(Self {
            dir,
            manifest,
            corpus_embeddings,
            query_embeddings,
            neighbors,
            qrels,
        })
    }

    pub fn dims(&self) -> usize {
        self.manifest.dims
    }

    pub fn n_corpus(&self) -> usize {
        self.manifest.n
    }

    pub fn n_queries(&self) -> usize {
        self.manifest.n_queries.unwrap_or(0)
    }

    pub fn corpus_vector(&self, i: usize) -> &[f32] {
        self.corpus_embeddings.row(i)
    }

    pub fn query_vector(&self, i: usize) -> &[f32] {
        let d = self.dims();
        &self.query_embeddings[i * d..(i + 1) * d]
    }
}

fn read_f32_matrix(path: &Path, rows: usize, dims: usize) -> Result<Vec<f32>, CorpusError> {
    let expected = rows * dims * 4;
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    if buf.len() != expected {
        return Err(CorpusError::CorruptSize {
            expected,
            got: buf.len(),
        });
    }
    let mut out = vec![0.0f32; rows * dims];
    for (i, chunk) in buf.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

fn encode_corpus_mmap(emb: &MmapF32Matrix) -> Vec<PaxBlock64> {
    use rayon::prelude::*;

    let n = emb.rows();
    let dims = emb.dims();
    let slice = emb.as_slice();
    let n_blocks = n.div_ceil(VECTORS_PER_BLOCK);
    (0..n_blocks)
        .into_par_iter()
        .map(|bi| {
            let start = bi * VECTORS_PER_BLOCK;
            let end = (start + VECTORS_PER_BLOCK).min(n);
            let refs: Vec<&[f32]> = (start..end)
                .map(|j| &slice[j * dims..(j + 1) * dims])
                .collect();
            let ts: Vec<u32> = (start..end).map(|j| j as u32).collect();
            let src = vec![source::DOC; end - start];
            encode_block(&refs, &ts, &src, dims)
        })
        .collect()
}

fn hit_ids(hits: &[ScoredHit]) -> Vec<usize> {
    hits.iter()
        .map(|h| h.id.block as usize * VECTORS_PER_BLOCK + h.id.slot as usize)
        .collect()
}

fn recall_at_k(truth: &HashSet<usize>, pred: &[usize], k: usize) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let got = pred.iter().take(k).filter(|id| truth.contains(id)).count();
    got as f64 / truth.len().min(k) as f64
}

fn ndcg_at_k(truth: &HashSet<usize>, pred: &[usize], k: usize) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let mut dcg = 0.0;
    for (i, id) in pred.iter().take(k).enumerate() {
        if truth.contains(id) {
            dcg += 1.0 / ((i as f64) + 2.0).log2();
        }
    }
    let ideal_n = truth.len().min(k);
    let mut idcg = 0.0;
    for i in 0..ideal_n {
        idcg += 1.0 / ((i as f64) + 2.0).log2();
    }
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalBenchReport {
    pub label: String,
    pub corpus: String,
    pub model: String,
    pub scale: Option<String>,
    pub mrl: bool,
    pub n_vectors: usize,
    pub n_queries: usize,
    pub dims: usize,
    pub shortlist: usize,
    pub strata: String,
    pub corpus_mmap: bool,
    #[serde(default)]
    pub stratum4_weight: f32,
    #[serde(default)]
    pub cs_tolerance: f32,
    pub shortlist_coverage_at_10: f64,
    pub reranked_recall_at_10: f64,
    pub reranked_recall_at_100: f64,
    pub qrels_recall_at_10: f64,
    pub qrels_ndcg_at_10: f64,
    pub mean_scan_ms: f64,
    pub mvecs_per_sec: f64,
    pub encode_ms: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RetrievalBenchParams {
    pub shortlist: usize,
    pub strata: StrataDepth,
    pub stratum4_weight: f32,
    pub cs_tolerance: f32,
}

impl Default for RetrievalBenchParams {
    fn default() -> Self {
        Self {
            shortlist: 256,
            strata: StrataDepth::L1Rabitq,
            stratum4_weight: crate::rotate::DEFAULT_STRATUM4_WEIGHT,
            cs_tolerance: 1.0,
        }
    }
}

fn strata_name(s: StrataDepth) -> String {
    match s {
        StrataDepth::L0Only => "l0_only".into(),
        StrataDepth::L1Rabitq => "l1_rabitq".into(),
        StrataDepth::Full64b => "full_64b".into(),
    }
}

pub fn run_retrieval_bench(
    label: &str,
    data_dir: &Path,
    n_queries: Option<usize>,
    shortlist: usize,
    strata: StrataDepth,
) -> Result<RetrievalBenchReport, CorpusError> {
    run_retrieval_bench_params(
        label,
        data_dir,
        n_queries,
        RetrievalBenchParams {
            shortlist,
            strata,
            ..RetrievalBenchParams::default()
        },
    )
}

pub fn run_retrieval_bench_params(
    label: &str,
    data_dir: &Path,
    n_queries: Option<usize>,
    params: RetrievalBenchParams,
) -> Result<RetrievalBenchReport, CorpusError> {
    let data = RetrievalBenchData::load(data_dir)?;
    let dims = data.dims();
    let n = data.n_corpus();
    let q_total = data.n_queries();
    let n_queries = n_queries.unwrap_or(q_total).min(q_total);

    data.corpus_embeddings.advise_sequential();
    let t_enc = Instant::now();
    eprintln!("encode PaxBlocks n={n} dims={dims} …");
    let blocks = encode_corpus_mmap(&data.corpus_embeddings);
    let encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "encode done: {} blocks in {:.1} s",
        blocks.len(),
        encode_ms / 1000.0
    );
    data.corpus_embeddings.advise_random();

    Ok(eval_retrieval_on_blocks(
        label, &data, &blocks, n_queries, params, encode_ms,
    ))
}

/// Encode PaxBlocks once, then evaluate many `(w_s4, cs_tolerance, shortlist)` cells.
/// Encoded bits are invariant to positive fold scaling, so this is a pure query-side sweep.
///
/// When `out_dir` is set, each cell JSON is written immediately (crash-safe).
pub fn run_retrieval_sweep(
    data_dir: &Path,
    n_queries: Option<usize>,
    strata: StrataDepth,
    shortlists: &[usize],
    w_s4_values: &[f32],
    cs_values: &[f32],
    label_prefix: &str,
    out_dir: Option<&Path>,
) -> Result<Vec<RetrievalBenchReport>, CorpusError> {
    let data = RetrievalBenchData::load(data_dir)?;
    let dims = data.dims();
    let n = data.n_corpus();
    let q_total = data.n_queries();
    let n_queries = n_queries.unwrap_or(q_total).min(q_total);

    data.corpus_embeddings.advise_sequential();
    let t_enc = Instant::now();
    eprintln!("encode PaxBlocks once n={n} dims={dims} (sweep reuses blocks) …");
    let blocks = encode_corpus_mmap(&data.corpus_embeddings);
    let encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "encode done: {} blocks in {:.1} s — {} cells",
        blocks.len(),
        encode_ms / 1000.0,
        shortlists.len() * w_s4_values.len() * cs_values.len()
    );
    data.corpus_embeddings.advise_random();

    if let Some(dir) = out_dir {
        std::fs::create_dir_all(dir)?;
    }

    let mut reports = Vec::new();
    let mut cell_i = 0usize;
    for &sl in shortlists {
        for &w in w_s4_values {
            for &cs in cs_values {
                let label = format!("{label_prefix}-sl{sl}-w{w:.4}-cs{cs:.2}");
                eprintln!("=== sweep cell {label} ===");
                let params = RetrievalBenchParams {
                    shortlist: sl,
                    strata,
                    stratum4_weight: w,
                    cs_tolerance: cs,
                };
                let report = eval_retrieval_on_blocks(
                    &label, &data, &blocks, n_queries, params, encode_ms,
                );
                report.print_summary();
                if let Some(dir) = out_dir {
                    let path = dir.join(format!("cell_{cell_i:03}_{label}.json"));
                    let _ = std::fs::write(&path, report.to_json_pretty());
                    eprintln!("wrote {}", path.display());
                }
                reports.push(report);
                cell_i += 1;
            }
        }
    }
    Ok(reports)
}

fn eval_retrieval_on_blocks(
    label: &str,
    data: &RetrievalBenchData,
    blocks: &[PaxBlock64],
    n_queries: usize,
    params: RetrievalBenchParams,
    encode_ms: f64,
) -> RetrievalBenchReport {
    let dims = data.dims();
    let n = data.n_corpus();

    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.shortlist_cand = params.shortlist;
    cfg.strata_depth = params.strata;
    cfg.stratum4_weight = params.stratum4_weight;
    cfg.cs_tolerance = params.cs_tolerance;
    cfg.temporal_filter_enable = false;

    let mut coverage10 = 0.0;
    let mut rerank10 = 0.0;
    let mut rerank100 = 0.0;
    let mut qrels_r10 = 0.0;
    let mut qrels_ndcg = 0.0;
    let mut qrels_n = 0usize;
    let mut scan_ms_sum = 0.0;

    let progress_every = (n_queries / 20).max(1);
    for qi in 0..n_queries {
        let q = data.query_vector(qi);
        let truth10: HashSet<usize> = data.neighbors[qi].iter().take(10).copied().collect();
        let truth100: HashSet<usize> = data.neighbors[qi].iter().take(100).copied().collect();
        let plan = QueryPlan::from_query(q, dims);

        let t = Instant::now();
        let hits = scan_blocks(blocks, &plan, &cfg, None, None);
        scan_ms_sum += t.elapsed().as_secs_f64() * 1000.0;
        let ids = hit_ids(&hits);

        let covered = truth10.iter().filter(|i| ids.contains(i)).count() as f64 / 10.0;
        coverage10 += covered;

        let mut reranked: Vec<(usize, f32)> = ids
            .iter()
            .copied()
            .map(|i| (i, dot(q, data.corpus_vector(i))))
            .collect();
        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let rerank_ids: Vec<usize> = reranked.into_iter().map(|(i, _)| i).collect();

        rerank10 += recall_at_k(&truth10, &rerank_ids, 10);
        rerank100 += recall_at_k(&truth100, &rerank_ids, 100);

        if let Some(rels) = data.qrels.get(&qi) {
            qrels_r10 += recall_at_k(rels, &rerank_ids, 10);
            qrels_ndcg += ndcg_at_k(rels, &rerank_ids, 10);
            qrels_n += 1;
        }

        if (qi + 1) % progress_every == 0 || qi + 1 == n_queries {
            let done = qi + 1;
            let mean = scan_ms_sum / done as f64;
            eprintln!(
                "  queries {done}/{n_queries}  mean_scan={mean:.1} ms/q  eta≈{:.0}s",
                mean * (n_queries - done) as f64 / 1000.0
            );
        }
    }

    let mean_scan_ms = scan_ms_sum / n_queries as f64;
    let mvecs = (n as f64 / (mean_scan_ms / 1000.0)) / 1e6;
    let qrels_denom = qrels_n.max(1) as f64;

    RetrievalBenchReport {
        label: label.into(),
        corpus: format!("{} / {}", data.manifest.dataset, data.dir.display()),
        model: data.manifest.model.clone(),
        scale: data.manifest.scale.clone(),
        mrl: true,
        n_vectors: n,
        n_queries,
        dims,
        shortlist: params.shortlist,
        strata: strata_name(params.strata),
        corpus_mmap: true,
        stratum4_weight: params.stratum4_weight,
        cs_tolerance: params.cs_tolerance,
        shortlist_coverage_at_10: coverage10 / n_queries as f64,
        reranked_recall_at_10: rerank10 / n_queries as f64,
        reranked_recall_at_100: rerank100 / n_queries as f64,
        qrels_recall_at_10: qrels_r10 / qrels_denom,
        qrels_ndcg_at_10: qrels_ndcg / qrels_denom,
        mean_scan_ms,
        mvecs_per_sec: mvecs,
        encode_ms,
    }
}

impl RetrievalBenchReport {
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize")
    }

    pub fn print_summary(&self) {
        println!("=== mrl-index retrieval bench [{}] ===", self.label);
        println!(
            "corpus: {}  scale={}  model={}  mrl={}  mmap={}",
            self.corpus,
            self.scale.as_deref().unwrap_or("-"),
            self.model,
            self.mrl,
            self.corpus_mmap
        );
        println!(
            "scan: n={} q={} dims={} shortlist={} strata={}",
            self.n_vectors, self.n_queries, self.dims, self.shortlist, self.strata
        );
        println!(
            "ANN:  shortlist_cov@10={:.4}  reranked_R@10={:.4}  reranked_R@100={:.4}",
            self.shortlist_coverage_at_10, self.reranked_recall_at_10, self.reranked_recall_at_100
        );
        println!(
            "IR:   qrels_R@10={:.4}  qrels_nDCG@10={:.4}",
            self.qrels_recall_at_10, self.qrels_ndcg_at_10
        );
        println!(
            "perf: encode={:.1} ms  scan={:.3} ms/q  ({:.2} Mvecs/s)",
            self.encode_ms, self.mean_scan_ms, self.mvecs_per_sec
        );
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PrecisionRerankReport {
    pub label: String,
    pub corpus: String,
    pub n_vectors: usize,
    pub n_queries: usize,
    pub shortlist: usize,
    pub precision: String,
    pub bytes_per_vec: usize,
    pub reranked_recall_at_10: f64,
    pub qrels_recall_at_10: f64,
    pub qrels_ndcg_at_10: f64,
    /// Fraction of fp32 R@10 retained (1.0 = perfect parity).
    pub recall_parity_vs_fp32: f64,
    pub mean_rerank_us: f64,
    pub mean_scan_ms: f64,
    pub encode_ms: f64,
}

/// Scan once per query; rerank the same shortlist at fp32 / fp16 / int8 and compare.
pub fn run_precision_rerank_bench(
    label_prefix: &str,
    data_dir: &Path,
    n_queries: Option<usize>,
    shortlist: usize,
    strata: StrataDepth,
) -> Result<Vec<PrecisionRerankReport>, CorpusError> {
    use crate::lowprec::{rerank_dot, RerankPrecision};

    let data = RetrievalBenchData::load(data_dir)?;
    let dims = data.dims();
    if dims != 768 {
        return Err(CorpusError::CorruptSize {
            expected: 768,
            got: dims,
        });
    }
    let n = data.n_corpus();
    let q_total = data.n_queries();
    let n_queries = n_queries.unwrap_or(q_total).min(q_total);

    data.corpus_embeddings.advise_sequential();
    let t_enc = Instant::now();
    eprintln!("encode PaxBlocks n={n} dims={dims} …");
    let blocks = encode_corpus_mmap(&data.corpus_embeddings);
    let encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
    data.corpus_embeddings.advise_random();

    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.shortlist_cand = shortlist;
    cfg.strata_depth = strata;
    cfg.temporal_filter_enable = false;

    let precisions = [
        RerankPrecision::Fp32,
        RerankPrecision::Fp16,
        RerankPrecision::Int8,
    ];
    let mut r10 = [0.0f64; 3];
    let mut qrels_r10 = [0.0f64; 3];
    let mut ndcg = [0.0f64; 3];
    let mut qrels_n = 0usize;
    let mut rerank_us = [0.0f64; 3];
    let mut scan_ms_sum = 0.0;

    let progress_every = (n_queries / 20).max(1);
    for qi in 0..n_queries {
        let q = data.query_vector(qi);
        let truth10: HashSet<usize> = data.neighbors[qi].iter().take(10).copied().collect();
        let plan = QueryPlan::from_query(q, dims);

        let t = Instant::now();
        let hits = scan_blocks(&blocks, &plan, &cfg, None, None);
        scan_ms_sum += t.elapsed().as_secs_f64() * 1000.0;
        let ids = hit_ids(&hits);

        for (pi, &prec) in precisions.iter().enumerate() {
            let t_r = Instant::now();
            let mut reranked: Vec<(usize, f32)> = ids
                .iter()
                .copied()
                .map(|i| (i, rerank_dot(prec, q, data.corpus_vector(i))))
                .collect();
            reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            rerank_us[pi] += t_r.elapsed().as_secs_f64() * 1e6;
            let rerank_ids: Vec<usize> = reranked.into_iter().map(|(i, _)| i).collect();
            r10[pi] += recall_at_k(&truth10, &rerank_ids, 10);
            if let Some(rels) = data.qrels.get(&qi) {
                if pi == 0 {
                    qrels_n += 1;
                }
                qrels_r10[pi] += recall_at_k(rels, &rerank_ids, 10);
                ndcg[pi] += ndcg_at_k(rels, &rerank_ids, 10);
            }
        }

        if (qi + 1) % progress_every == 0 || qi + 1 == n_queries {
            eprintln!(
                "  precision-bench queries {}/{}",
                qi + 1,
                n_queries
            );
        }
    }

    let qrels_denom = qrels_n.max(1) as f64;
    let fp32_r10 = r10[0] / n_queries as f64;
    let mean_scan_ms = scan_ms_sum / n_queries as f64;
    let mut out = Vec::new();
    for (pi, &prec) in precisions.iter().enumerate() {
        let rr = r10[pi] / n_queries as f64;
        out.push(PrecisionRerankReport {
            label: format!("{label_prefix}-{}", prec.as_str()),
            corpus: format!("{} / {}", data.manifest.dataset, data.dir.display()),
            n_vectors: n,
            n_queries,
            shortlist,
            precision: prec.as_str().into(),
            bytes_per_vec: prec.bytes_per_vec(),
            reranked_recall_at_10: rr,
            qrels_recall_at_10: qrels_r10[pi] / qrels_denom,
            qrels_ndcg_at_10: ndcg[pi] / qrels_denom,
            recall_parity_vs_fp32: if fp32_r10 > 0.0 {
                rr / fp32_r10
            } else {
                1.0
            },
            mean_rerank_us: rerank_us[pi] / n_queries as f64,
            mean_scan_ms,
            encode_ms,
        });
    }
    Ok(out)
}

impl PrecisionRerankReport {
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize")
    }

    pub fn print_summary(&self) {
        println!(
            "=== precision rerank [{}] {}  B/vec={}  R@10={:.4}  parity={:.4}  rerank={:.1} µs ===",
            self.label,
            self.precision,
            self.bytes_per_vec,
            self.reranked_recall_at_10,
            self.recall_parity_vs_fp32,
            self.mean_rerank_us
        );
    }
}
