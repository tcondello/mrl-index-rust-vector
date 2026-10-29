//! Low-precision shortlist / ring representations for Step 4.
//!
//! - `fp16`: `[f16; 768]` — 1,536 B/vec (2× smaller than fp32)
//! - `int8`: `[i8; 768]` + `f16` scale — 770 B/vec (≈4× smaller)

use half::f16;

use crate::embed::{dot_768, dot_f16_768, dot_i8_768};

pub const EMBED_DIMS_768: usize = 768;
/// fp32 bytes per 768-d vector.
pub const BYTES_FP32_768: usize = EMBED_DIMS_768 * 4;
/// fp16 bytes per 768-d vector.
pub const BYTES_FP16_768: usize = EMBED_DIMS_768 * 2;
/// int8 payload + f16 scale.
pub const BYTES_INT8_768: usize = EMBED_DIMS_768 + 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankPrecision {
    Fp32,
    Fp16,
    Int8,
}

impl RerankPrecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
        }
    }

    pub fn bytes_per_vec(self) -> usize {
        match self {
            Self::Fp32 => BYTES_FP32_768,
            Self::Fp16 => BYTES_FP16_768,
            Self::Int8 => BYTES_INT8_768,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Fp16Vec768 {
    pub data: [f16; EMBED_DIMS_768],
}

impl Fp16Vec768 {
    pub fn from_f32(v: &[f32]) -> Self {
        assert_eq!(v.len(), EMBED_DIMS_768);
        let mut data = [f16::ZERO; EMBED_DIMS_768];
        for i in 0..EMBED_DIMS_768 {
            data[i] = f16::from_f32(v[i]);
        }
        Self { data }
    }

    #[inline]
    pub fn dot_query_f32(&self, q: &[f32; EMBED_DIMS_768]) -> f32 {
        dot_f16_768(q, &self.data)
    }
}

#[derive(Debug, Clone)]
pub struct Int8Vec768 {
    pub codes: [i8; EMBED_DIMS_768],
    pub scale: f16,
}

impl Int8Vec768 {
    /// Symmetric per-vector int8: `x ≈ scale * code`, `scale = max|x| / 127`.
    pub fn from_f32(v: &[f32]) -> Self {
        assert_eq!(v.len(), EMBED_DIMS_768);
        let mut max_abs = 0.0f32;
        for &x in v {
            max_abs = max_abs.max(x.abs());
        }
        let scale = if max_abs < 1e-12 {
            0.0
        } else {
            max_abs / 127.0
        };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut codes = [0i8; EMBED_DIMS_768];
        for i in 0..EMBED_DIMS_768 {
            let q = (v[i] * inv).round().clamp(-127.0, 127.0) as i32;
            codes[i] = q as i8;
        }
        Self {
            codes,
            scale: f16::from_f32(scale),
        }
    }

    #[inline]
    pub fn dot_query_f32(&self, q: &[f32; EMBED_DIMS_768]) -> f32 {
        dot_i8_768(q, &self.codes, self.scale.to_f32())
    }
}

/// Dot `query` (fp32) against a corpus vector at the requested precision.
#[inline]
pub fn rerank_dot(precision: RerankPrecision, query: &[f32], corpus: &[f32]) -> f32 {
    debug_assert_eq!(query.len(), EMBED_DIMS_768);
    debug_assert_eq!(corpus.len(), EMBED_DIMS_768);
    let q: &[f32; EMBED_DIMS_768] = query.try_into().expect("768");
    match precision {
        RerankPrecision::Fp32 => {
            let c: &[f32; EMBED_DIMS_768] = corpus.try_into().expect("768");
            dot_768(q, c)
        }
        RerankPrecision::Fp16 => {
            let c = Fp16Vec768::from_f32(corpus);
            c.dot_query_f32(q)
        }
        RerankPrecision::Int8 => {
            let c = Int8Vec768::from_f32(corpus);
            c.dot_query_f32(q)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::dot_768;

    fn unitish(seed: u64) -> [f32; 768] {
        let mut v = [0.0f32; 768];
        let mut x = seed;
        for i in 0..768 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            v[i] = ((x >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0;
        }
        let n: f32 = v.iter().map(|t| t * t).sum::<f32>().sqrt().max(1e-12);
        for t in &mut v {
            *t /= n;
        }
        v
    }

    #[test]
    fn fp16_dot_near_fp32() {
        let a = unitish(1);
        let b = unitish(2);
        let exact = dot_768(&a, &b);
        let approx = Fp16Vec768::from_f32(&b).dot_query_f32(&a);
        assert!((exact - approx).abs() < 1e-2, "{exact} vs {approx}");
    }

    #[test]
    fn int8_dot_near_fp32() {
        let a = unitish(3);
        let b = unitish(4);
        let exact = dot_768(&a, &b);
        let approx = Int8Vec768::from_f32(&b).dot_query_f32(&a);
        assert!((exact - approx).abs() < 5e-2, "{exact} vs {approx}");
    }
}
