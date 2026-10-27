//! Deterministic local embedding for demos (no GPU required).
//!
//! Uses feature hashing + lightly structured Matryoshka-friendly decay so the
//! MRL prefix carries most of the signal. Swap this for MLX/CUDA EmbeddingGemma
//! in production via the [`crate::rerank::Embedder`] trait.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::rerank::Embedder;

#[derive(Debug, Clone)]
pub struct HashingEmbedder {
    dims: usize,
}

impl HashingEmbedder {
    pub fn new(dims: usize) -> Self {
        assert!(dims >= 192);
        Self { dims }
    }

    pub fn dims(&self) -> usize {
        self.dims
    }
}

impl Embedder for HashingEmbedder {
    fn embed(&self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| embed_one(t, self.dims)).collect()
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

fn embed_one(text: &str, dims: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dims];
    let tokens = tokenize(text);
    if tokens.is_empty() {
        // Non-zero tiny bias so norms stay well-defined.
        v[0] = 1e-3;
        return l2_normalize(v);
    }

    for (pos, tok) in tokens.iter().enumerate() {
        let mut h = DefaultHasher::new();
        tok.hash(&mut h);
        let hv = h.finish();
        let idx = (hv as usize) % dims;
        let sign = if hv & 1 == 0 { 1.0f32 } else { -1.0f32 };
        // Positional soft decay favors earlier tokens in leading dims.
        let pos_w = 1.0 / (1.0 + 0.05 * pos as f32);
        v[idx] += sign * pos_w;

        // Bigram feature
        if pos + 1 < tokens.len() {
            let mut h2 = DefaultHasher::new();
            tok.hash(&mut h2);
            tokens[pos + 1].hash(&mut h2);
            let hv2 = h2.finish();
            let idx2 = (hv2 as usize) % dims;
            let sign2 = if hv2 & 1 == 0 { 1.0f32 } else { -1.0f32 };
            v[idx2] += 0.5 * sign2 * pos_w;
        }
    }

    // Matryoshka-style spectral decay: amplify leading coordinates.
    for (i, x) in v.iter_mut().enumerate() {
        let decay = (1.0 / (1.0 + 0.01 * i as f32)).sqrt();
        *x *= decay;
    }

    l2_normalize(v)
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let mut n2 = 0.0f32;
    for &x in &v {
        n2 += x * x;
    }
    let n = n2.sqrt().max(1e-12);
    for x in &mut v {
        *x /= n;
    }
    v
}

/// Exact f32 dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
