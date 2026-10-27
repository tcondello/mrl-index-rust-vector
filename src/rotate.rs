//! Structured orthogonal rotation for RaBitQ mid-band (FWHT).
//!
//! Per-coordinate Rademacher flips are a no-op for sign-agreement tests
//! (`sign(x·s) == sign(q·s)` ⟺ `sign(x) == sign(q)`). A real Walsh–Hadamard
//! transform mixes coordinates so the 1-bit estimator is non-trivial.

use crate::block::RABITQ_DIMS;

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
}
