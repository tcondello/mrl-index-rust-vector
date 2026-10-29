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

/// Hash whitespace/alnum tokens into stable `u16` IDs for `passages.bin` v2 storage.
///
/// Production swaps this for the real SentencePiece / BPE vocabulary IDs from the
/// embedder; the demo hashing embedder uses the same tokenizer alphabet.
pub fn tokenize_u16(text: &str) -> Vec<u16> {
    tokenize(text)
        .into_iter()
        .map(|tok| {
            let mut h = DefaultHasher::new();
            tok.hash(&mut h);
            (h.finish() & 0xffff) as u16
        })
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

/// Exact f32 dot product (dispatches to the unrolled 768-d kernel when possible).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if a.len() == 768 {
        // SAFETY: lengths checked; arrays are plain f32 with no niche.
        let a768: &[f32; 768] = a.try_into().expect("len 768");
        let b768: &[f32; 768] = b.try_into().expect("len 768");
        return dot_768(a768, b768);
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 8-way unrolled dot for EmbeddingGemma @768 (`24 × 32` floats, no remainder).
///
/// Independent accumulators keep Apple Silicon / x86 FMA pipelines fed.
#[inline]
pub fn dot_768(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;
    let mut i = 0usize;
    while i < 768 {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        acc4 += a[i + 4] * b[i + 4];
        acc5 += a[i + 5] * b[i + 5];
        acc6 += a[i + 6] * b[i + 6];
        acc7 += a[i + 7] * b[i + 7];
        i += 8;
    }
    acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7
}

/// Dot `q` (fp32) against an `fp16` corpus vector with 8-way unrolled promote-mul-add.
///
/// Portable scalar path; ARMv8.2-FP16 / x86 F16C can replace the promote later.
#[inline]
pub fn dot_f16_768(q: &[f32; 768], v: &[half::f16; 768]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;
    let mut i = 0usize;
    while i < 768 {
        acc0 += q[i] * v[i].to_f32();
        acc1 += q[i + 1] * v[i + 1].to_f32();
        acc2 += q[i + 2] * v[i + 2].to_f32();
        acc3 += q[i + 3] * v[i + 3].to_f32();
        acc4 += q[i + 4] * v[i + 4].to_f32();
        acc5 += q[i + 5] * v[i + 5].to_f32();
        acc6 += q[i + 6] * v[i + 6].to_f32();
        acc7 += q[i + 7] * v[i + 7].to_f32();
        i += 8;
    }
    acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7
}

/// Dot `q` (fp32) against symmetric int8 codes: `⟨q, scale · codes⟩`.
#[inline]
pub fn dot_i8_768(q: &[f32; 768], codes: &[i8; 768], scale: f32) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;
    let mut i = 0usize;
    while i < 768 {
        acc0 += q[i] * codes[i] as f32;
        acc1 += q[i + 1] * codes[i + 1] as f32;
        acc2 += q[i + 2] * codes[i + 2] as f32;
        acc3 += q[i + 3] * codes[i + 3] as f32;
        acc4 += q[i + 4] * codes[i + 4] as f32;
        acc5 += q[i + 5] * codes[i + 5] as f32;
        acc6 += q[i + 6] * codes[i + 6] as f32;
        acc7 += q[i + 7] * codes[i + 7] as f32;
        i += 8;
    }
    (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7) * scale
}
