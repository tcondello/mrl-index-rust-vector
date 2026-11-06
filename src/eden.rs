//! EDEN-prod style prefix codebook (unbiased Lloyd–Max after center/normalize/rotate).
//!
//! Pipeline on a d-dim slice (default: MRL prefix d=64):
//!   Center → L2-normalize → structured Hadamard (Rademacher ∘ FWHT) →
//!   b-bit Lloyd–Max N(0,1) levels with unbiased IP scale S = ‖x‖² / ⟨x, codeword⟩.
//!
//! Score estimate for original (pre-normalize) vectors:
//!   ⟨q, x⟩ ≈ ⟨q, μ⟩ + ‖x_c‖ · S · ⟨R(q − μ), codeword⟩
//! where queries are *not* renormalized (asymmetric doc-side normalize).

use crate::block::MRL_PREFIX_DIMS;
use crate::rotate::fwht_inplace;

/// Fixed Lloyd–Max N(0,1) centroids for b=1..=4 (enough for our probes).
const B1: [f32; 2] = [-0.79788456, 0.79788456];
const B2: [f32; 4] = [-1.51041761, -0.45278003, 0.45278003, 1.51041761];
const B3: [f32; 8] = [
    -2.15194570,
    -1.34390928,
    -0.75600528,
    -0.24509418,
    0.24509418,
    0.75600528,
    1.34390928,
    2.15194570,
];
const B4: [f32; 16] = [
    -2.73258958,
    -2.06901724,
    -1.61804640,
    -1.25623121,
    -0.94234046,
    -0.65675912,
    -0.38804830,
    -0.12839503,
    0.12839503,
    0.38804830,
    0.65675912,
    0.94234046,
    1.25623121,
    1.61804640,
    2.06901724,
    2.73258958,
];

#[inline]
pub fn lloyd_max_normal(bits: u8) -> &'static [f32] {
    match bits {
        1 => &B1,
        2 => &B2,
        3 => &B3,
        4 => &B4,
        _ => panic!("lloyd_max_normal: bits must be 1..=4, got {bits}"),
    }
}

#[inline]
pub fn nearest_centroid(codebook: &[f32], z: f32) -> usize {
    match codebook.binary_search_by(|c| c.partial_cmp(&z).unwrap()) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) if i >= codebook.len() => codebook.len() - 1,
        Err(i) => {
            if (z - codebook[i - 1]).abs() <= (codebook[i] - z).abs() {
                i - 1
            } else {
                i
            }
        }
    }
}

/// Deterministic ±1 diagonal for structured Hadamard (seed mixes dim index).
#[inline]
fn rademacher_diag(seed: u64, i: usize) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(i as u64)
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

/// In-place structured orthogonal mix: diag(±1) then FWHT. `d` must be 2^k.
pub fn structured_hadamard_inplace(v: &mut [f32], seed: u64) {
    let d = v.len();
    debug_assert!(d.is_power_of_two());
    for i in 0..d {
        v[i] *= rademacher_diag(seed, i);
    }
    fwht_inplace(v);
}

/// Fitted corpus mean over the coded slice (prefix by default).
#[derive(Debug, Clone)]
pub struct EdenFit {
    pub dim: usize,
    pub bits: u8,
    pub seed: u64,
    pub mean: Vec<f32>,
}

impl EdenFit {
    pub fn fit_prefix(corpus_rows: impl Iterator<Item = impl AsRef<[f32]>>, bits: u8, seed: u64) -> Self {
        Self::fit_slice(corpus_rows, MRL_PREFIX_DIMS, bits, seed)
    }

    pub fn fit_slice(
        corpus_rows: impl Iterator<Item = impl AsRef<[f32]>>,
        dim: usize,
        bits: u8,
        seed: u64,
    ) -> Self {
        Self::fit_band(corpus_rows, 0, dim, bits, seed)
    }

    /// Fit mean on `row[start..start+dim]` for each corpus row.
    pub fn fit_band(
        corpus_rows: impl Iterator<Item = impl AsRef<[f32]>>,
        start: usize,
        dim: usize,
        bits: u8,
        seed: u64,
    ) -> Self {
        assert!((1..=4).contains(&bits));
        assert!(dim.is_power_of_two(), "EDEN slice dim must be 2^k, got {dim}");
        let mut mean = vec![0.0f32; dim];
        let mut n = 0u64;
        for row in corpus_rows {
            let x = row.as_ref();
            assert!(x.len() >= start + dim);
            for d in 0..dim {
                mean[d] += x[start + d];
            }
            n += 1;
        }
        assert!(n > 0, "empty corpus for EdenFit");
        let inv = 1.0 / n as f32;
        for m in mean.iter_mut() {
            *m *= inv;
        }
        Self {
            dim,
            bits,
            seed,
            mean,
        }
    }
}

/// Per-vector EDEN code for one slice.
#[derive(Debug, Clone)]
pub struct EdenCode {
    /// Lloyd–Max level per coordinate (`0..2^bits`).
    pub levels: Vec<u8>,
    /// ‖x − μ‖₂ before normalize.
    pub centered_norm: f32,
    /// Unbiased dequant scale S on the unit sphere.
    pub scale_s: f32,
}

impl EdenCode {
    /// Combined doc-side multiplier ‖x_c‖ · S used in asymmetric IP estimates.
    #[inline]
    pub fn alpha(&self) -> f32 {
        self.centered_norm * self.scale_s
    }
}

/// Encode one slice with a fitted [`EdenFit`].
pub fn encode_slice(fit: &EdenFit, x: &[f32]) -> EdenCode {
    let d = fit.dim;
    assert!(x.len() >= d);
    let codebook = lloyd_max_normal(fit.bits);
    let sqrt_d = (d as f32).sqrt();
    let inv_sqrt_d = 1.0 / sqrt_d;

    let mut centered = vec![0.0f32; d];
    let mut norm_sq = 0.0f32;
    for i in 0..d {
        let v = x[i] - fit.mean[i];
        centered[i] = v;
        norm_sq += v * v;
    }
    let centered_norm = norm_sq.sqrt().max(1e-12);
    let inv_norm = 1.0 / centered_norm;
    for v in centered.iter_mut() {
        *v *= inv_norm;
    }

    structured_hadamard_inplace(&mut centered, fit.seed);

    let mut levels = vec![0u8; d];
    let mut codeword = vec![0.0f32; d];
    let mut ip = 0.0f32;
    for i in 0..d {
        let level = nearest_centroid(codebook, centered[i] * sqrt_d);
        levels[i] = level as u8;
        let cw = codebook[level] * inv_sqrt_d;
        codeword[i] = cw;
        ip += centered[i] * cw;
    }
    let scale_s = if ip.abs() > 1e-12 {
        // Unbiased: ‖u‖² / ⟨u, cw⟩ with ‖u‖=1 → 1/ip
        1.0 / ip
    } else {
        0.0
    };

    EdenCode {
        levels,
        centered_norm,
        scale_s,
    }
}

/// Forward structured Hadamard on a raw query slice (for recon-faithful scoring).
///
/// Reconstruction is `μ + α R⁻¹(cw)`. With our orthogonal `R = FWHT ∘ diag`,
/// `⟨q, R⁻¹(cw)⟩ = ⟨R(q), cw⟩`, so we rotate `q` (not `q−μ`).
pub fn rotate_query(fit: &EdenFit, q: &[f32]) -> Vec<f32> {
    let d = fit.dim;
    assert!(q.len() >= d);
    let mut v = q[..d].to_vec();
    structured_hadamard_inplace(&mut v, fit.seed);
    v
}

/// Alias kept for call sites that historically centered first.
pub fn rotate_query_centered(fit: &EdenFit, q: &[f32]) -> Vec<f32> {
    rotate_query(fit, q)
}

/// IP estimate ⟨q, x̂⟩ matching [`reconstruct`]: ⟨q, μ⟩ + α ⟨R(q), cw⟩.
pub fn score_ip(fit: &EdenFit, q: &[f32], q_rot: &[f32], code: &EdenCode) -> f32 {
    let d = fit.dim;
    let codebook = lloyd_max_normal(fit.bits);
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    let mut dot_cw = 0.0f32;
    for i in 0..d {
        dot_cw += q_rot[i] * codebook[code.levels[i] as usize] * inv_sqrt_d;
    }
    let mut mean_ip = 0.0f32;
    for i in 0..d {
        mean_ip += q[i] * fit.mean[i];
    }
    mean_ip + code.alpha() * dot_cw
}

/// Reconstruct approximate centered unit vector · α + μ (for MSE probes).
pub fn reconstruct(fit: &EdenFit, code: &EdenCode) -> Vec<f32> {
    let d = fit.dim;
    let codebook = lloyd_max_normal(fit.bits);
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    let mut cw = vec![0.0f32; d];
    for i in 0..d {
        cw[i] = codebook[code.levels[i] as usize] * inv_sqrt_d;
    }
    // Inverse structured Hadamard = same transform (FWHT is involution up to
    // the diag signs applied again in the same order: diag then FWHT; inverse
    // is FWHT then diag).
    fwht_inplace(&mut cw);
    for i in 0..d {
        cw[i] *= rademacher_diag(fit.seed, i);
    }
    let alpha = code.alpha();
    for i in 0..d {
        cw[i] = fit.mean[i] + alpha * cw[i];
    }
    cw
}

/// Current production min–max 4-bit prefix encode (for A/B).
#[derive(Debug, Clone)]
pub struct MinMax4Code {
    pub levels: [u8; MRL_PREFIX_DIMS],
    pub scale: f32,
    pub offset: f32,
}

pub fn encode_minmax4_prefix(x: &[f32]) -> MinMax4Code {
    let prefix = &x[..MRL_PREFIX_DIMS];
    let min_v = prefix.iter().copied().fold(f32::INFINITY, f32::min);
    let max_v = prefix.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_v - min_v).max(1e-8);
    let scale = range / 15.0;
    let offset = min_v;
    let mut levels = [0u8; MRL_PREFIX_DIMS];
    for (i, &v) in prefix.iter().enumerate() {
        levels[i] = ((v - offset) / scale).round().clamp(0.0, 15.0) as u8;
    }
    MinMax4Code {
        levels,
        scale,
        offset,
    }
}

pub fn score_minmax4_prefix(q: &[f32], code: &MinMax4Code) -> f32 {
    let mut s = 0.0f32;
    for d in 0..MRL_PREFIX_DIMS {
        s += q[d] * (code.offset + code.scale * code.levels[d] as f32);
    }
    s
}

pub fn reconstruct_minmax4_prefix(code: &MinMax4Code) -> [f32; MRL_PREFIX_DIMS] {
    let mut out = [0.0f32; MRL_PREFIX_DIMS];
    for d in 0..MRL_PREFIX_DIMS {
        out[d] = code.offset + code.scale * code.levels[d] as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lloyd_b4_sorted_symmetric() {
        let c = lloyd_max_normal(4);
        assert_eq!(c.len(), 16);
        assert!(c.windows(2).all(|w| w[0] < w[1]));
        for (a, b) in c.iter().zip(c.iter().rev()) {
            assert!((a + b).abs() < 1e-5);
        }
    }

    #[test]
    fn eden_unbiased_slope_near_one_on_gaussian() {
        // Synthetic unit-ish rows: EDEN IP estimates should track true dots.
        let dim = 64usize;
        let bits = 4u8;
        let seed = 1u64;
        let mut rows = Vec::new();
        for n in 0..128u64 {
            let mut v = vec![0.0f32; dim];
            for i in 0..dim {
                // deterministic pseudo-gaussian
                let mut x = n
                    .wrapping_mul(0x9E37)
                    .wrapping_add(i as u64)
                    .wrapping_mul(0xBF58);
                x ^= x >> 17;
                v[i] = ((x as i64 as f32) / (u64::MAX as f32)) * 2.0 - 1.0;
            }
            let nrm: f32 = v.iter().map(|t| t * t).sum::<f32>().sqrt().max(1e-6);
            for t in v.iter_mut() {
                *t /= nrm;
            }
            rows.push(v);
        }
        let fit = EdenFit::fit_slice(rows.iter().map(|r| r.as_slice()), dim, bits, seed);
        let codes: Vec<_> = rows.iter().map(|r| encode_slice(&fit, r)).collect();
        let q = &rows[0];
        let q_rot = rotate_query_centered(&fit, q);
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (row, code) in rows.iter().zip(&codes) {
            let exact: f32 = q.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
            let est = score_ip(&fit, q, &q_rot, code);
            num += est * exact;
            den += exact * exact;
        }
        let slope = num / den.max(1e-8);
        assert!(
            (slope - 1.0).abs() < 0.35,
            "unbiased slope {slope} not near 1"
        );
    }
}
