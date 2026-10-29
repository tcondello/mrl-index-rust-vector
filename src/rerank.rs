//! Single-batch JIT neural (or local) reranker (spec §4.2).

use serde::Serialize;

use crate::embed::dot;

/// Pluggable on-device embedder (MLX / CUDA / hashing demo).
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Vec<Vec<f32>>;
    fn dimensions(&self) -> usize;
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankHit {
    pub passage_id: u64,
    pub text: String,
    pub scan_score: f32,
    pub exact_score: f32,
}

/// Re-embed candidate texts in one batch and rank by exact query·doc dots.
pub fn rerank_batch(
    embedder: &dyn Embedder,
    query_text: &str,
    candidates: &[(u64, String, f32)],
    top_k: usize,
) -> Vec<RerankHit> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut texts: Vec<String> = Vec::with_capacity(candidates.len() + 1);
    texts.push(query_text.to_string());
    for (_, t, _) in candidates {
        texts.push(t.clone());
    }

    let embeds = embedder.embed(&texts);
    let q = &embeds[0];

    let mut hits: Vec<RerankHit> = candidates
        .iter()
        .enumerate()
        .map(|(i, (id, text, scan))| RerankHit {
            passage_id: *id,
            text: text.clone(),
            scan_score: *scan,
            exact_score: dot(q, &embeds[i + 1]),
        })
        .collect();

    hits.sort_by(|a, b| b.exact_score.partial_cmp(&a.exact_score).unwrap());
    hits.truncate(top_k.max(1));
    hits
}

/// Re-embed candidate texts in one batch against an already-computed `query_vec`
/// (avoids re-embedding the query a second time during `IndexEngine::query_filtered`).
pub fn rerank_with_query_vec(
    embedder: &dyn Embedder,
    query_vec: &[f32],
    candidates: &[(u64, String, f32)],
    top_k: usize,
) -> Vec<RerankHit> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let texts: Vec<String> = candidates.iter().map(|(_, t, _)| t.clone()).collect();
    let embeds = embedder.embed(&texts);

    let mut hits: Vec<RerankHit> = candidates
        .iter()
        .enumerate()
        .map(|(i, (id, text, scan))| RerankHit {
            passage_id: *id,
            text: text.clone(),
            scan_score: *scan,
            exact_score: dot(query_vec, &embeds[i]),
        })
        .collect();

    hits.sort_by(|a, b| b.exact_score.partial_cmp(&a.exact_score).unwrap());
    hits.truncate(top_k.max(1));
    hits
}
