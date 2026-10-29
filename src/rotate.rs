//! Structured orthogonal rotation for RaBitQ mid-band + tail fold (FWHT).
//!
//! Per-coordinate Rademacher flips are a no-op for sign-agreement tests
//! (`sign(x·s) == sign(q·s)` ⟺ `sign(x) == sign(q)`). A real Walsh–Hadamard
//! transform mixes coordinates so the 1-bit estimator is non-trivial.
//!
//! Step 3: the 576-d tail (`192..768`) is folded into 64 dims via block
//! Rademacher signs + FWHT, then stored as 1-bit codes in `stratum4_mid`.

use crate::block::{MRL_PREFIX_DIMS, RABITQ_DIMS, TAIL_FOLD_BLOCKS, TAIL_FOLD_DIMS, TAIL_START};

/// In-place orthonormal Fast Walsh–Hadamard Transform (length must be 2^k).
pub fn fwht_inplace(a: &mut [f32]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1usize;
    while h < n {
        let step = h * 2;
        for i in (0..n).step_by(step) {
            for j in i..i + h {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h = step;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for x in a.iter_mut() {
        *x *= scale;
    }
}

/// Rotate the 128-dim mid-band with FWHT; returns orthonormal coordinates.
pub fn rotate_mid_band(mid: &[f32]) -> [f32; RABITQ_DIMS] {
    assert_eq!(mid.len(), RABITQ_DIMS);
    let mut buf = [0.0f32; RABITQ_DIMS];
    buf.copy_from_slice(mid);
    fwht_inplace(&mut buf);
    buf
}

/// Expected |N(0, 1/D)| scale used to map ±1 codes back toward IP units.
#[inline]
pub fn rabitq_bit_scale() -> f32 {
    // E[|g|] for g ~ N(0, 1/D) ≈ √(2/π) / √D
    (2.0 / std::f32::consts::PI).sqrt() / (RABITQ_DIMS as f32).sqrt()
}

/// Deterministic ±1 Rademacher for tail-block `(b, d)`.
#[inline]
pub fn tail_rademacher(block: usize, dim: usize) -> f32 {
    // SplitMix64-style mix — not a cryptographic hash; just stable ±1 noise.
    let mut x = (block as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(dim as u64)
        .wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 27;
    if x & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Fold dims `[TAIL_START..]` into 64 coords: `u_d = Σ_b s_{b,d} x_{192+64b+d}`.
///
/// MMSE `1/√B` shrinkage is applied query-side (`EngineConfig::stratum4_weight`),
/// not here — encode only keeps `sign(FWHT(u))`, which is scale-invariant.
pub fn fold_tail(vec: &[f32]) -> [f32; TAIL_FOLD_DIMS] {
    let mut u = [0.0f32; TAIL_FOLD_DIMS];
    if vec.len() <= TAIL_START {
        return u;
    }
    let available = vec.len() - TAIL_START;
    let n_blocks = (available / MRL_PREFIX_DIMS).min(TAIL_FOLD_BLOCKS);
    if n_blocks == 0 {
        return u;
    }
    for b in 0..n_blocks {
        let base = TAIL_START + b * MRL_PREFIX_DIMS;
        for d in 0..TAIL_FOLD_DIMS {
            u[d] += tail_rademacher(b, d) * vec[base + d];
        }
    }
    // Intentionally unscaled: encode takes `sign(FWHT(u))`, which is invariant to
    // any positive scale. Stratum-4 shrinkage is applied query-side via
    // `EngineConfig::stratum4_weight` (locked default `0.20` from Stage 2 sweep).
    u
}

/// FWHT of the folded tail — used for both encode (sign bits) and query weights.
pub fn rotate_tail_fold(vec: &[f32]) -> [f32; TAIL_FOLD_DIMS] {
    let mut buf = fold_tail(vec);
    fwht_inplace(&mut buf);
    buf
}

/// Default MMSE (Wiener) shrinkage for folding `B` Rademacher blocks into 64 signs.
///
/// Folding `B` blocks introduces `B(B-1)` cross-block collision terms and raises
/// estimator noise variance by `B`. Scaling the query-side Stratum-4 contribution
/// by `1/√B` applies the exact `1/B` MMSE weight (for Gemma@768, `B=9` → `1/3`).
#[inline]
pub fn mmse_stratum4_weight(n_blocks: usize) -> f32 {
    if n_blocks == 0 {
        1.0
    } else {
        1.0 / (n_blocks as f32).sqrt()
    }
}

/// Default Stratum-4 weight for EmbeddingGemma-2 @768.
///
/// Locked from Stage 2 Pareto sweep: `0.20` beats pure MMSE `1/√9 ≈ 0.333`
/// on both 1M (R@10 0.881 vs 0.857) and 8.8M (R@10 0.899 vs 0.881) at sl=256.
pub const DEFAULT_STRATUM4_WEIGHT: f32 = 0.20;

/// IP scale for 64-d folded-tail 1-bit codes (same derivation as RaBitQ).
#[inline]
pub fn tail_fold_bit_scale() -> f32 {
    (2.0 / std::f32::consts::PI).sqrt() / (TAIL_FOLD_DIMS as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_preserves_l2() {
        let mut v: Vec<f32> = (0..128).map(|i| ((i * 17) % 97) as f32 * 0.01 - 0.5).collect();
        let n0: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        fwht_inplace(&mut v);
        let n1: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - n1).abs() < 1e-4, "{n0} vs {n1}");
    }

    #[test]
    fn rademacher_would_be_noop_but_fwht_is_not() {
        // FWHT mixes energy across dims — first coord changes vs identity.
        let mid: Vec<f32> = (0..128).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let rot = rotate_mid_band(&mid);
        let changed = rot.iter().skip(1).any(|&x| x.abs() > 1e-6);
        assert!(changed, "FWHT should spread a spike across dimensions");
    }

    #[test]
    fn fold_tail_768_is_nontrivial() {
        let mut v = vec![0.0f32; 768];
        for (i, x) in v.iter_mut().enumerate() {
            *x = if i >= 192 { ((i * 3) % 17) as f32 * 0.01 - 0.08 } else { 0.0 };
        }
        let folded = fold_tail(&v);
        let energy: f32 = folded.iter().map(|x| x * x).sum();
        assert!(energy > 1e-6, "folded tail should carry energy");
        let rot = rotate_tail_fold(&v);
        let n0: f32 = folded.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n1: f32 = rot.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - n1).abs() < 1e-4);
    }
}
