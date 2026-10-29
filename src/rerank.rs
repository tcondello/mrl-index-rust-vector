//! Single-batch JIT neural (or local) reranker (spec §4.2).

use serde::Serialize;

use crate::block::PaxBlock64;
use crate::embed::dot;

/// Pluggable on-device embedder (MLX / CUDA / hashing demo).
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Vec<Vec<f32>>;
    fn dimensions(&self) -> usize;
}

/// Pinned query state containing both full-precision and speculative Layer-8 prefixes.
#[derive(Debug, Clone)]
pub struct QuerySession {
    /// Full 768-dimensional L2-normalized query vector for `QueryPlan` and final exact dot.
    pub query_f32: [f32; 768],
    /// Speculative 64-dimensional MRL prefix emitted early at Layer 8.
    pub query_early_64: [f32; 64],
}

/// Contiguous ragged token batch avoiding string allocations and `<pad>` tokens.
#[derive(Debug, Clone, Copy)]
pub struct PackedTokenBatch<'a> {
    /// Contiguous 1D slice of token IDs (supports Gemma-2's 262,144 SentencePiece vocabulary).
    pub tokens: &'a [u32],
    /// Cumulative sequence lengths of length `num_sequences + 1`:
    /// sequence `i` spans `tokens[cu_seqlens[i] as usize .. cu_seqlens[i+1] as usize]`.
    pub cu_seqlens: &'a [u32],
}

impl<'a> PackedTokenBatch<'a> {
    #[inline]
    pub fn num_sequences(&self) -> usize {
        self.cu_seqlens.len().saturating_sub(1)
    }

    #[inline]
    pub fn sequence(&self, i: usize) -> &'a [u32] {
        let start = self.cu_seqlens[i] as usize;
        let end = self.cu_seqlens[i + 1] as usize;
        &self.tokens[start..end]
    }
}

/// Unified execution interface bridging the local Gemma-2 Apple Silicon model and `mrl-index`.
pub trait UnifiedEmbedder: Send + Sync {
    /// Tokenize raw UTF-8 text directly into `u32` token IDs (up to 262,144 vocab).
    fn tokenize_u32(&self, text: &str) -> Vec<u32>;

    /// Initiate query embedding: computes `query_early_64` at Layer 8 and completes
    /// `query_f32` using the pinned instruction prefix KV-cache (`PrefixKVCache`).
    fn begin_query(&self, query_text: &str) -> QuerySession;

    /// Direct zero-copy passage ingestion: executes ragged prefill and transposes
    /// bitplanes directly into pre-allocated `PaxBlock64` slices without runtime FWHT.
    fn ingest_packed_to_paxblocks(
        &self,
        batch: PackedTokenBatch<'_>,
        timestamps: &[u32],
        sources: &[u32],
        out_blocks: &mut [PaxBlock64],
    );

    /// Diagnostic / ring-buffer passage ingestion: returns L2-normalized 768-d vectors.
    fn embed_packed(&self, batch: PackedTokenBatch<'_>) -> Vec<[f32; 768]>;

    /// In-register shortlist scoring: executes 2-stage ragged prefill (early-exit prune
    /// against `session.query_early_64`, then full-layer exact dot against `session.query_f32`),
    /// returning the top-`k` `(candidate_index, exact_score)` pairs sorted descending.
    fn score_shortlist_packed(
        &self,
        session: &QuerySession,
        batch: PackedTokenBatch<'_>,
        prior_scores: &[f32],
        top_k: usize,
    ) -> Vec<(usize, f32)>;
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
