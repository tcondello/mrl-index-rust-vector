//! Clean MS MARCO / EmbeddingGemma-2 retrieval benchmark.
//!
//! Uses **separate query embeddings** + **precomputed exact neighbors** + **qrels**.
//! This is not corpus self-retrieval and does not recompute brute-force GT at runtime.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
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

#[derive(Debug, Clone)]
pub struct RetrievalBenchData {
    pub dir: PathBuf,
    pub manifest: RetrievalManifest,
    pub corpus_embeddings: Vec<f32>,
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

        let corpus_embeddings = read_f32_matrix(
            &dir.join(&manifest.embeddings_file),
            manifest.n,
            manifest.dims,
        )?;
        let query_embeddings = read_f32_matrix(
            &dir.join(q_emb_name),
            n_queries,
            manifest.dims,
        )?;

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
        let d = self.dims();
        &self.corpus_embeddings[i * d..(i + 1) * d]
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

fn encode_corpus_flat(emb: &[f32], n: usize, dims: usize) -> Vec<PaxBlock64> {
    let n_blocks = n.div_ceil(VECTORS_PER_BLOCK);
    (0..n_blocks)
        .into_par_iter()
        .map(|b| {
            let i = b * VECTORS_PER_BLOCK;
            let end = (i + VECTORS_PER_BLOCK).min(n);
            let mut refs: [&[f32]; VECTORS_PER_BLOCK] = [&[]; VECTORS_PER_BLOCK];
            let mut ts = [0u32; VECTORS_PER_BLOCK];
            let src = [source::DOC; VECTORS_PER_BLOCK];
            let count = end - i;
            for k in 0..count {
                let j = i + k;
                refs[k] = &emb[j * dims..(j + 1) * dims];
                ts[k] = j as u32;
            }
            encode_block(&refs[..count], &ts[..count], &src[..count], dims)
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
    pub shortlist_coverage_at_10: f64,
    pub reranked_recall_at_10: f64,
    pub reranked_recall_at_100: f64,
    pub qrels_recall_at_10: f64,
    pub qrels_ndcg_at_10: f64,
    pub mean_scan_ms: f64,
    pub mvecs_per_sec: f64,
    pub encode_ms: f64,
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
    let data = RetrievalBenchData::load(data_dir)?;
    let dims = data.dims();
    let n = data.n_corpus();
    let q_total = data.n_queries();
    let n_queries = n_queries.unwrap_or(q_total).min(q_total);

    let t_enc = Instant::now();
    let blocks = encode_corpus_flat(&data.corpus_embeddings, n, dims);
    let encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.shortlist_cand = shortlist;
    cfg.strata_depth = strata;
    cfg.temporal_filter_enable = false;

    let mut coverage10 = 0.0;
    let mut rerank10 = 0.0;
    let mut rerank100 = 0.0;
    let mut qrels_r10 = 0.0;
    let mut qrels_ndcg = 0.0;
    let mut qrels_n = 0usize;
    let mut scan_ms_sum = 0.0;

    for qi in 0..n_queries {
        let q = data.query_vector(qi);
        let truth10: HashSet<usize> = data.neighbors[qi].iter().take(10).copied().collect();
        let truth100: HashSet<usize> = data.neighbors[qi].iter().take(100).copied().collect();
        let plan = QueryPlan::from_query(q, dims);

        let t = Instant::now();
        let hits = scan_blocks(&blocks, &plan, &cfg, None, None);
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
    }

    let mean_scan_ms = scan_ms_sum / n_queries as f64;
    let mvecs = (n as f64 / (mean_scan_ms / 1000.0)) / 1e6;
    let qrels_denom = qrels_n.max(1) as f64;

    Ok(RetrievalBenchReport {
        label: label.into(),
        corpus: format!(
            "{} / {}",
            data.manifest.dataset,
            data_dir.display()
        ),
        model: data.manifest.model.clone(),
        scale: data.manifest.scale.clone(),
        mrl: true,
        n_vectors: n,
        n_queries,
        dims,
        shortlist,
        strata: strata_name(strata),
        shortlist_coverage_at_10: coverage10 / n_queries as f64,
        reranked_recall_at_10: rerank10 / n_queries as f64,
        reranked_recall_at_100: rerank100 / n_queries as f64,
        qrels_recall_at_10: qrels_r10 / qrels_denom,
        qrels_ndcg_at_10: qrels_ndcg / qrels_denom,
        mean_scan_ms,
        mvecs_per_sec: mvecs,
        encode_ms,
    })
}

impl RetrievalBenchReport {
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize")
    }

    pub fn print_summary(&self) {
        println!("=== mrl-index retrieval bench [{}] ===", self.label);
        println!(
            "corpus: {}  scale={}  model={}  mrl={}",
            self.corpus,
            self.scale.as_deref().unwrap_or("-"),
            self.model,
            self.mrl
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
