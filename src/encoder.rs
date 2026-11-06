//! Composable encode primitives: float vectors / fused tokens → [`PaxBlock64`].
//!
//! Ring flush, mmap retrieval benches, and synthetic scan benches share
//! [`Encoder`] + [`encode_f32_corpus_par`]. The Gemma fused path uses
//! [`FusedPackedEncoder`] (implemented for any [`UnifiedEmbedder`]).

use std::sync::Arc;

use crate::block::{PaxBlock64, VECTORS_PER_BLOCK};
use crate::eden::EdenFit;
use crate::quantize::{
    encode_block, encode_block_from_augmented, encode_block_with_fit, encode_block_with_fits,
};
use crate::rerank::{PackedTokenBatch, UnifiedEmbedder};
use crate::rotate::AUGMENTED_FULL_DIMS;

/// Encode up to 64 L2-normalized float embeddings into one [`PaxBlock64`].
pub trait Encoder: Send + Sync {
    fn embed_dims(&self) -> usize;

    fn encode_block(
        &self,
        vectors: &[&[f32]],
        timestamps: &[u32],
        sources: &[u32],
    ) -> PaxBlock64;
}

/// Default encoder: runtime mid-band FWHT + tail fold via [`encode_block`].
#[derive(Debug, Clone, Copy)]
pub struct QuantizeEncoder {
    pub dims: usize,
}

impl QuantizeEncoder {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Encoder for QuantizeEncoder {
    fn embed_dims(&self) -> usize {
        self.dims
    }

    fn encode_block(
        &self,
        vectors: &[&[f32]],
        timestamps: &[u32],
        sources: &[u32],
    ) -> PaxBlock64 {
        encode_block(vectors, timestamps, sources, self.dims)
    }
}

/// Encoder that writes EDEN-prod codes into the 4-bit MRL prefix planes.
#[derive(Debug, Clone)]
pub struct EdenPrefixEncoder {
    pub dims: usize,
    pub fit: Arc<EdenFit>,
}

impl EdenPrefixEncoder {
    pub fn new(dims: usize, fit: EdenFit) -> Self {
        Self {
            dims,
            fit: Arc::new(fit),
        }
    }
}

impl Encoder for EdenPrefixEncoder {
    fn embed_dims(&self) -> usize {
        self.dims
    }

    fn encode_block(
        &self,
        vectors: &[&[f32]],
        timestamps: &[u32],
        sources: &[u32],
    ) -> PaxBlock64 {
        encode_block_with_fit(vectors, timestamps, sources, self.dims, Some(&self.fit))
    }
}

/// EDEN-prod prefix (b=4) + EDEN-prod mid-band (b=1).
#[derive(Debug, Clone)]
pub struct EdenDualEncoder {
    pub dims: usize,
    pub prefix: Arc<EdenFit>,
    pub mid: Arc<EdenFit>,
}

impl EdenDualEncoder {
    pub fn new(dims: usize, prefix: EdenFit, mid: EdenFit) -> Self {
        assert_eq!(prefix.dim, crate::block::MRL_PREFIX_DIMS);
        assert_eq!(prefix.bits, 4);
        assert_eq!(mid.dim, crate::block::RABITQ_DIMS);
        assert_eq!(mid.bits, 1);
        Self {
            dims,
            prefix: Arc::new(prefix),
            mid: Arc::new(mid),
        }
    }
}

impl Encoder for EdenDualEncoder {
    fn embed_dims(&self) -> usize {
        self.dims
    }

    fn encode_block(
        &self,
        vectors: &[&[f32]],
        timestamps: &[u32],
        sources: &[u32],
    ) -> PaxBlock64 {
        encode_block_with_fits(
            vectors,
            timestamps,
            sources,
            self.dims,
            Some(&self.prefix),
            Some(&self.mid),
        )
    }
}

/// Offline-fusion encoder: pre-rotated augmented 960-d rows → PaxBlock
/// (no runtime `rotate_mid_band` / `rotate_tail_fold`).
#[derive(Debug, Clone, Copy, Default)]
pub struct AugmentedEncoder;

impl AugmentedEncoder {
    pub fn encode_block(
        &self,
        augmented: &[[f32; AUGMENTED_FULL_DIMS]],
        timestamps: &[u32],
        sources: &[u32],
    ) -> PaxBlock64 {
        encode_block_from_augmented(augmented, timestamps, sources)
    }
}

/// Fused token → PaxBlock path used by Gemma@768 ingest.
pub trait FusedPackedEncoder: Send + Sync {
    fn encode_packed_blocks(
        &self,
        batch: PackedTokenBatch<'_>,
        timestamps: &[u32],
        sources: &[u32],
        out_blocks: &mut [PaxBlock64],
    );
}

impl<T: UnifiedEmbedder + ?Sized> FusedPackedEncoder for T {
    fn encode_packed_blocks(
        &self,
        batch: PackedTokenBatch<'_>,
        timestamps: &[u32],
        sources: &[u32],
        out_blocks: &mut [PaxBlock64],
    ) {
        self.ingest_packed_to_paxblocks(batch, timestamps, sources, out_blocks);
    }
}

/// Encode a row-major `n × dims` f32 matrix into PaxBlocks (serial).
pub fn encode_f32_corpus(
    encoder: &dyn Encoder,
    matrix: &[f32],
    n: usize,
    default_source: u32,
) -> Vec<PaxBlock64> {
    let dims = encoder.embed_dims();
    assert_eq!(matrix.len(), n * dims);
    let n_blocks = n.div_ceil(VECTORS_PER_BLOCK);
    let mut out = Vec::with_capacity(n_blocks);
    for bi in 0..n_blocks {
        let start = bi * VECTORS_PER_BLOCK;
        let end = (start + VECTORS_PER_BLOCK).min(n);
        let refs: Vec<&[f32]> = (start..end)
            .map(|j| &matrix[j * dims..(j + 1) * dims])
            .collect();
        let ts: Vec<u32> = (start..end).map(|j| j as u32).collect();
        let src = vec![default_source; end - start];
        out.push(encoder.encode_block(&refs, &ts, &src));
    }
    out
}

/// Parallel encode of a row-major `n × dims` f32 matrix (mmap retrieval benches).
pub fn encode_f32_corpus_par(
    encoder: &dyn Encoder,
    matrix: &[f32],
    n: usize,
    default_source: u32,
) -> Vec<PaxBlock64> {
    use rayon::prelude::*;

    let dims = encoder.embed_dims();
    assert_eq!(matrix.len(), n * dims);
    let n_blocks = n.div_ceil(VECTORS_PER_BLOCK);
    (0..n_blocks)
        .into_par_iter()
        .map(|bi| {
            let start = bi * VECTORS_PER_BLOCK;
            let end = (start + VECTORS_PER_BLOCK).min(n);
            let refs: Vec<&[f32]> = (start..end)
                .map(|j| &matrix[j * dims..(j + 1) * dims])
                .collect();
            let ts: Vec<u32> = (start..end).map(|j| j as u32).collect();
            let src = vec![default_source; end - start];
            encoder.encode_block(&refs, &ts, &src)
        })
        .collect()
}

/// Encode owned row vectors (synthetic / in-memory benches).
pub fn encode_f32_rows(encoder: &dyn Encoder, rows: &[Vec<f32>], default_source: u32) -> Vec<PaxBlock64> {
    let dims = encoder.embed_dims();
    let n = rows.len();
    let n_blocks = n.div_ceil(VECTORS_PER_BLOCK);
    let mut out = Vec::with_capacity(n_blocks);
    for bi in 0..n_blocks {
        let start = bi * VECTORS_PER_BLOCK;
        let end = (start + VECTORS_PER_BLOCK).min(n);
        let refs: Vec<&[f32]> = rows[start..end].iter().map(|r| {
            assert_eq!(r.len(), dims);
            r.as_slice()
        }).collect();
        let ts: Vec<u32> = (start..end).map(|j| j as u32).collect();
        let src = vec![default_source; end - start];
        out.push(encoder.encode_block(&refs, &ts, &src));
    }
    out
}

/// Convenience: flush a ring-style pending batch.
///
/// When `eden` is set, the 64-d prefix uses EDEN-prod; otherwise min–max 4-bit.
pub fn encode_pending_embeddings(
    embeddings: &[&[f32]],
    timestamps: &[u32],
    sources: &[u32],
) -> PaxBlock64 {
    encode_pending_embeddings_eden(embeddings, timestamps, sources, None)
}

pub fn encode_pending_embeddings_eden(
    embeddings: &[&[f32]],
    timestamps: &[u32],
    sources: &[u32],
    eden: Option<&EdenFit>,
) -> PaxBlock64 {
    assert!(!embeddings.is_empty());
    let dims = embeddings[0].len();
    match eden {
        Some(fit) => EdenPrefixEncoder {
            dims,
            fit: Arc::new(fit.clone()),
        }
        .encode_block(embeddings, timestamps, sources),
        None => QuantizeEncoder::new(dims).encode_block(embeddings, timestamps, sources),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::source;
    use crate::quantize::encode_block;

    fn l2_normalize(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= n;
        }
    }

    #[test]
    fn quantize_encoder_matches_encode_block() {
        let dims = 768;
        let mut rows = Vec::new();
        for i in 0..8usize {
            let mut v = vec![0.0f32; dims];
            for d in 0..dims {
                v[d] = ((i * 11 + d * 3) % 89) as f32 * 0.01 - 0.4;
            }
            l2_normalize(&mut v);
            rows.push(v);
        }
        let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        let ts: Vec<u32> = (0..8).map(|i| i as u32).collect();
        let src = vec![source::DOC; 8];

        let via_trait = QuantizeEncoder::new(dims).encode_block(&refs, &ts, &src);
        let direct = encode_block(&refs, &ts, &src, dims);
        assert_eq!(via_trait.as_bytes(), direct.as_bytes());
    }

    #[test]
    fn encode_f32_corpus_par_roundtrip_len() {
        let dims = 384;
        let n = 130usize; // 3 blocks (64+64+2)
        let mut matrix = vec![0.0f32; n * dims];
        for (i, x) in matrix.iter_mut().enumerate() {
            *x = ((i * 17) % 101) as f32 * 0.01 - 0.5;
        }
        for r in 0..n {
            l2_normalize(&mut matrix[r * dims..(r + 1) * dims]);
        }
        let enc = QuantizeEncoder::new(dims);
        let blocks = encode_f32_corpus_par(&enc, &matrix, n, source::DOC);
        assert_eq!(blocks.len(), 3);
        let serial = encode_f32_corpus(&enc, &matrix, n, source::DOC);
        assert_eq!(blocks.len(), serial.len());
        for (a, b) in blocks.iter().zip(serial.iter()) {
            assert_eq!(a.as_bytes(), b.as_bytes());
        }
    }
}
