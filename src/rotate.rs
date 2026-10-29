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

/// Output dimensionality of the offline-folded sketch operator `T_norm`:
/// `64 (prefix) + 128 (rotated mid) + 64 (folded+rotated tail) = 256`.
pub const FUSED_SKETCH_DIMS: usize = MRL_PREFIX_DIMS + RABITQ_DIMS + TAIL_FOLD_DIMS;

/// Output dimensionality of the augmented projection matrix `W_full`:
/// `768 (full unrotated embedding) + 128 (rotated mid) + 64 (folded+rotated tail) = 960`.
pub const AUGMENTED_FULL_DIMS: usize = 768 + RABITQ_DIMS + TAIL_FOLD_DIMS;

/// Fold a model's output projection matrix `w_proj` (`[768, d_model]` row-major)
/// offline into `W_fused_norm = T_norm · W_proj` (`[256, d_model]` row-major).
///
/// Because `T_norm = diag(I_64, H_128, (1/√9) H_64 S)` is linear and column-wise
/// decomposable, we apply `T_norm` to each of the `d_model` columns of `w_proj`.
/// The resulting `[256, d_model]` matrix has condition number `κ(T_norm) = 1.0`
/// (exact partial isometry) and cuts output projection FLOPs by 66.7%.
pub fn fold_projection_weights(w_proj: &[f32], d_model: usize) -> Vec<f32> {
    assert_eq!(
        w_proj.len(),
        768 * d_model,
        "w_proj must have shape [768, d_model]"
    );
    let mut w_fused = vec![0.0f32; FUSED_SKETCH_DIMS * d_model];
    let inv_sqrt_b = 1.0 / (TAIL_FOLD_BLOCKS as f32).sqrt(); // 1/3 for B=9

    let mut col = [0.0f32; 768];
    for c in 0..d_model {
        for r in 0..768 {
            col[r] = w_proj[r * d_model + c];
        }
        // 1. Rows 0..64: Identity on MRL prefix
        for d in 0..MRL_PREFIX_DIMS {
            w_fused[d * d_model + c] = col[d];
        }
        // 2. Rows 64..192: Orthonormal 128-d FWHT on mid-band
        let rot_mid = rotate_mid_band(&col[MRL_PREFIX_DIMS..MRL_PREFIX_DIMS + RABITQ_DIMS]);
        for d in 0..RABITQ_DIMS {
            w_fused[(MRL_PREFIX_DIMS + d) * d_model + c] = rot_mid[d];
        }
        // 3. Rows 192..256: Normalized (1/√9) Rademacher fold + 64-d FWHT on tail
        let rot_tail = rotate_tail_fold(&col);
        for d in 0..TAIL_FOLD_DIMS {
            w_fused[(MRL_PREFIX_DIMS + RABITQ_DIMS + d) * d_model + c] = rot_tail[d] * inv_sqrt_b;
        }
    }
    w_fused
}

/// Build the augmented `[960, d_model]` projection matrix `W_full`:
/// - Rows `0..768`: original `W_proj` (produces exact 768-d embedding + norms + 64-d prefix)
/// - Rows `768..896`: `H_128 · W_mid` (pre-rotated Stratum-1 RaBitQ coordinates)
/// - Rows `896..960`: `(1/3) H_64 · S · W_tail` (pre-rotated Stratum-4 folded tail coordinates)
pub fn augment_projection_weights(w_proj: &[f32], d_model: usize) -> Vec<f32> {
    assert_eq!(
        w_proj.len(),
        768 * d_model,
        "w_proj must have shape [768, d_model]"
    );
    let mut w_full = vec![0.0f32; AUGMENTED_FULL_DIMS * d_model];
    w_full[..768 * d_model].copy_from_slice(w_proj);

    let w_fused = fold_projection_weights(w_proj, d_model);
    // Copy rotated mid (rows 64..192 of w_fused) into rows 768..896 of w_full
    w_full[768 * d_model..(768 + RABITQ_DIMS) * d_model]
        .copy_from_slice(&w_fused[MRL_PREFIX_DIMS * d_model..(MRL_PREFIX_DIMS + RABITQ_DIMS) * d_model]);
    // Copy folded+rotated tail (rows 192..256 of w_fused) into rows 896..960 of w_full
    w_full[(768 + RABITQ_DIMS) * d_model..AUGMENTED_FULL_DIMS * d_model]
        .copy_from_slice(&w_fused[(MRL_PREFIX_DIMS + RABITQ_DIMS) * d_model..FUSED_SKETCH_DIMS * d_model]);

    w_full
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

    #[test]
    fn fold_projection_weights_matches_runtime_rotation() {
        let d_model = 32usize;
        let mut w_proj = vec![0.0f32; 768 * d_model];
        for (i, w) in w_proj.iter_mut().enumerate() {
            *w = ((i * 13 + 7) % 101) as f32 * 0.01 - 0.5;
        }
        let mut h = vec![0.0f32; d_model];
        for (i, x) in h.iter_mut().enumerate() {
            *x = ((i * 29 + 3) % 73) as f32 * 0.02 - 0.7;
        }

        // Unfused: z = W_proj · h, then rotate_mid_band & rotate_tail_fold
        let mut z = vec![0.0f32; 768];
        for r in 0..768 {
            let mut sum = 0.0f32;
            for c in 0..d_model {
                sum += w_proj[r * d_model + c] * h[c];
            }
            z[r] = sum;
        }
        let exp_mid = rotate_mid_band(&z[64..192]);
        let exp_tail = rotate_tail_fold(&z);

        // Fused: y = W_fused · h
        let w_fused = fold_projection_weights(&w_proj, d_model);
        let mut y = vec![0.0f32; FUSED_SKETCH_DIMS];
        for r in 0..FUSED_SKETCH_DIMS {
            let mut sum = 0.0f32;
            for c in 0..d_model {
                sum += w_fused[r * d_model + c] * h[c];
            }
            y[r] = sum;
        }

        for d in 0..64 {
            assert!((y[d] - z[d]).abs() < 1e-4);
        }
        for d in 0..128 {
            assert!((y[64 + d] - exp_mid[d]).abs() < 1e-4);
        }
        for d in 0..64 {
            assert!((y[192 + d] - exp_tail[d] / 3.0).abs() < 1e-4);
            assert_eq!(y[192 + d] >= 0.0, exp_tail[d] >= 0.0);
        }
    }
}
