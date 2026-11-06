//! Affine 4-bit MRL prefix + RaBitQ mid-band + folded-tail stratum4 quantization.
//!
//! Optional EDEN-prod prefix via [`encode_block_with_fit`] / [`quantize_vector_eden`].

use crate::block::{
    Bitplane64, PaxBlock64, RabitqPlane, VectorMeta, VECTORS_PER_BLOCK, MRL_PREFIX_DIMS,
    RABITQ_DIMS, TAIL_FOLD_DIMS, TAIL_START,
};
use crate::eden::{encode_slice, EdenFit};
use crate::rotate::{rotate_mid_band, rotate_tail_fold, AUGMENTED_FULL_DIMS};

/// Build a filled PaxBlock64 from up to 64 float vectors (row-major, `dims` each).
pub fn encode_block(
    vectors: &[&[f32]],
    timestamps: &[u32],
    sources: &[u32],
    dims: usize,
) -> PaxBlock64 {
    encode_block_with_fit(vectors, timestamps, sources, dims, None)
}

/// Like [`encode_block`], but when `eden` is set the 64-d prefix uses EDEN-prod
/// (`meta.scale = α = ‖x−μ‖·S`, `meta.offset = 0`, levels in the 4 bitplanes).
pub fn encode_block_with_fit(
    vectors: &[&[f32]],
    timestamps: &[u32],
    sources: &[u32],
    dims: usize,
    eden: Option<&EdenFit>,
) -> PaxBlock64 {
    assert!(!vectors.is_empty() && vectors.len() <= VECTORS_PER_BLOCK);
    assert_eq!(timestamps.len(), vectors.len());
    assert_eq!(sources.len(), vectors.len());
    assert!(dims >= MRL_PREFIX_DIMS + RABITQ_DIMS);
    if let Some(fit) = eden {
        assert_eq!(fit.dim, MRL_PREFIX_DIMS);
        assert_eq!(fit.bits, 4, "Pax 4-bit planes require EDEN bits=4");
    }

    let mut block = PaxBlock64::zeroed();
    let n = vectors.len();

    let mut min_ts = u32::MAX;
    let mut max_ts = 0u32;
    let mut source_mask = 0u32;

    for (i, (vec, (&ts, &src))) in vectors
        .iter()
        .zip(timestamps.iter().zip(sources.iter()))
        .enumerate()
    {
        assert_eq!(vec.len(), dims);
        min_ts = min_ts.min(ts);
        max_ts = max_ts.max(ts);
        source_mask |= src;

        let (meta, bits4, rabitq, fold_bits) = match eden {
            Some(fit) => quantize_vector_eden(fit, vec),
            None => quantize_vector(vec),
        };
        block.meta[i] = meta;
        write_4bit_planes(&mut block, i, &bits4);
        write_rabitq_plane(&mut block.stratum1_rabitq, i, &rabitq);
        write_tail_fold_plane(&mut block.stratum4_mid, i, &fold_bits);
    }

    // Unused slots are tombstoned so scans skip them.
    let mut tomb = 0u64;
    for i in n..VECTORS_PER_BLOCK {
        tomb |= 1u64 << i;
        block.meta[i] = VectorMeta::from_f32(0.0, 0.0, 0.0, 0);
    }

    block.header.min_timestamp = if n == 0 { 0 } else { min_ts };
    block.header.max_timestamp = max_ts;
    block.header.source_mask = source_mask;
    block.header.tombstone_mask = tomb;
    block.recompute_crc();
    block
}

/// Build a `PaxBlock64` directly from up to 64 pre-rotated augmented vectors
/// (`[f32; 960]` produced by `W_full = augment_projection_weights(W_proj)`)
/// without running `rotate_mid_band` or `rotate_tail_fold` at runtime.
///
/// Layout of each 960-d slice `aug`:
/// - `aug[0..768]`: L2-normalized 768-d embedding (prefix `0..64`, mid `64..192`, tail `192..768`)
/// - `aug[768..896]`: pre-rotated 128-d FWHT mid-band (`H_128 · x_mid`)
/// - `aug[896..960]`: pre-rotated 64-d folded tail (`(1/3) H_64 · S · x_tail`)
pub fn encode_block_from_augmented(
    augmented_vecs: &[[f32; AUGMENTED_FULL_DIMS]],
    timestamps: &[u32],
    sources: &[u32],
) -> PaxBlock64 {
    assert!(!augmented_vecs.is_empty() && augmented_vecs.len() <= VECTORS_PER_BLOCK);
    assert_eq!(timestamps.len(), augmented_vecs.len());
    assert_eq!(sources.len(), augmented_vecs.len());

    let mut block = PaxBlock64::zeroed();
    let n = augmented_vecs.len();

    let mut min_ts = u32::MAX;
    let mut max_ts = 0u32;
    let mut source_mask = 0u32;

    for (i, (aug, (&ts, &src))) in augmented_vecs
        .iter()
        .zip(timestamps.iter().zip(sources.iter()))
        .enumerate()
    {
        min_ts = min_ts.min(ts);
        max_ts = max_ts.max(ts);
        source_mask |= src;

        let prefix = &aug[..MRL_PREFIX_DIMS];
        let mid = &aug[MRL_PREFIX_DIMS..MRL_PREFIX_DIMS + RABITQ_DIMS];
        let high_tail = &aug[TAIL_START..768];
        let rot_mid = &aug[768..768 + RABITQ_DIMS];
        let rot_tail = &aug[768 + RABITQ_DIMS..AUGMENTED_FULL_DIMS];

        let min_v = prefix.iter().copied().fold(f32::INFINITY, f32::min);
        let max_v = prefix.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_v - min_v).max(1e-8);
        let scale = range / 15.0;
        let offset = min_v;

        let mut bits4 = [0u8; MRL_PREFIX_DIMS];
        for (d, &x) in prefix.iter().enumerate() {
            bits4[d] = ((x - offset) / scale).round().clamp(0.0, 15.0) as u8;
        }

        let mut rabitq = [0u8; RABITQ_DIMS];
        for d in 0..RABITQ_DIMS {
            rabitq[d] = if rot_mid[d] >= 0.0 { 1 } else { 0 };
        }

        let mut fold_bits = [0u8; TAIL_FOLD_DIMS];
        for d in 0..TAIL_FOLD_DIMS {
            fold_bits[d] = if rot_tail[d] >= 0.0 { 1 } else { 0 };
        }

        let mid_sq: f32 = mid.iter().map(|&x| x * x).sum();
        let high_sq: f32 = high_tail.iter().map(|&x| x * x).sum();
        let rest_norm = (mid_sq + high_sq).sqrt();
        let high_norm = high_sq.sqrt();
        let flags = VectorMeta::pack_rho_flags(high_norm, rest_norm);

        block.meta[i] = VectorMeta::from_f32(scale, offset, rest_norm, flags);
        write_4bit_planes(&mut block, i, &bits4);
        write_rabitq_plane(&mut block.stratum1_rabitq, i, &rabitq);
        write_tail_fold_plane(&mut block.stratum4_mid, i, &fold_bits);
    }

    let mut tomb = 0u64;
    for i in n..VECTORS_PER_BLOCK {
        tomb |= 1u64 << i;
        block.meta[i] = VectorMeta::from_f32(0.0, 0.0, 0.0, 0);
    }

    block.header.min_timestamp = if n == 0 { 0 } else { min_ts };
    block.header.max_timestamp = max_ts;
    block.header.source_mask = source_mask;
    block.header.tombstone_mask = tomb;
    block.recompute_crc();
    block
}

fn residual_norms(vec: &[f32]) -> (f32, f32, u8) {
    let mid = &vec[MRL_PREFIX_DIMS..MRL_PREFIX_DIMS + RABITQ_DIMS];
    let high_tail = if vec.len() > TAIL_START {
        &vec[TAIL_START..]
    } else {
        &[][..]
    };
    let mut mid_sq = 0.0f32;
    for &x in mid {
        mid_sq += x * x;
    }
    let mut high_sq = 0.0f32;
    for &x in high_tail {
        high_sq += x * x;
    }
    let rest_norm = (mid_sq + high_sq).sqrt();
    let high_norm = high_sq.sqrt();
    let flags = VectorMeta::pack_rho_flags(high_norm, rest_norm);
    (rest_norm, high_norm, flags)
}

fn mid_and_fold_bits(vec: &[f32]) -> ([u8; RABITQ_DIMS], [u8; TAIL_FOLD_DIMS]) {
    let mid = &vec[MRL_PREFIX_DIMS..MRL_PREFIX_DIMS + RABITQ_DIMS];
    let rotated = rotate_mid_band(mid);
    let mut rabitq = [0u8; RABITQ_DIMS];
    for i in 0..RABITQ_DIMS {
        rabitq[i] = if rotated[i] >= 0.0 { 1 } else { 0 };
    }
    let folded = rotate_tail_fold(vec);
    let mut fold_bits = [0u8; TAIL_FOLD_DIMS];
    for i in 0..TAIL_FOLD_DIMS {
        fold_bits[i] = if folded[i] >= 0.0 { 1 } else { 0 };
    }
    (rabitq, fold_bits)
}

fn quantize_vector(
    vec: &[f32],
) -> (
    VectorMeta,
    [u8; MRL_PREFIX_DIMS],
    [u8; RABITQ_DIMS],
    [u8; TAIL_FOLD_DIMS],
) {
    let prefix = &vec[..MRL_PREFIX_DIMS];

    let min_v = prefix.iter().copied().fold(f32::INFINITY, f32::min);
    let max_v = prefix.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_v - min_v).max(1e-8);
    let scale = range / 15.0;
    let offset = min_v;

    let mut bits4 = [0u8; MRL_PREFIX_DIMS];
    for (i, &x) in prefix.iter().enumerate() {
        let q = ((x - offset) / scale).round().clamp(0.0, 15.0) as u8;
        bits4[i] = q;
    }

    let (rabitq, fold_bits) = mid_and_fold_bits(vec);
    let (rest_norm, _high_norm, flags) = residual_norms(vec);
    let meta = VectorMeta::from_f32(scale, offset, rest_norm, flags);
    (meta, bits4, rabitq, fold_bits)
}

/// EDEN-prod prefix + unchanged RaBitQ mid / folded tail.
pub fn quantize_vector_eden(
    fit: &EdenFit,
    vec: &[f32],
) -> (
    VectorMeta,
    [u8; MRL_PREFIX_DIMS],
    [u8; RABITQ_DIMS],
    [u8; TAIL_FOLD_DIMS],
) {
    let code = encode_slice(fit, vec);
    let mut bits4 = [0u8; MRL_PREFIX_DIMS];
    bits4.copy_from_slice(&code.levels[..MRL_PREFIX_DIMS]);
    let (rabitq, fold_bits) = mid_and_fold_bits(vec);
    let (rest_norm, _high_norm, flags) = residual_norms(vec);
    // Pack α into scale; offset unused (0).
    let meta = VectorMeta::from_f32(code.alpha(), 0.0, rest_norm, flags);
    (meta, bits4, rabitq, fold_bits)
}

fn write_4bit_planes(block: &mut PaxBlock64, slot: usize, bits4: &[u8; MRL_PREFIX_DIMS]) {
    let mask = 1u64 << slot;
    for d in 0..MRL_PREFIX_DIMS {
        let q = bits4[d];
        // bit3 = MSB / sign-analogue in unsigned 4-bit grid
        if q & 0b1000 != 0 {
            block.stratum0_sign.dim_bits[d] |= mask;
        }
        if q & 0b0100 != 0 {
            block.stratum2_msb1.dim_bits[d] |= mask;
        }
        if q & 0b0010 != 0 {
            block.stratum3_msb2.dim_bits[d] |= mask;
        }
        if q & 0b0001 != 0 {
            block.stratum5_lsb.dim_bits[d] |= mask;
        }
    }
}

fn write_rabitq_plane(plane: &mut RabitqPlane, slot: usize, bits: &[u8; RABITQ_DIMS]) {
    let mask = 1u64 << slot;
    for d in 0..RABITQ_DIMS {
        if bits[d] != 0 {
            plane.dim_bits[d] |= mask;
        }
    }
}

fn write_tail_fold_plane(plane: &mut Bitplane64, slot: usize, bits: &[u8; TAIL_FOLD_DIMS]) {
    let mask = 1u64 << slot;
    for d in 0..TAIL_FOLD_DIMS {
        if bits[d] != 0 {
            plane.dim_bits[d] |= mask;
        }
    }
}

/// Reconstruct approximate prefix contribution for tests / debugging.
pub fn dequantize_prefix(meta: VectorMeta, bits4: [u8; MRL_PREFIX_DIMS]) -> [f32; MRL_PREFIX_DIMS] {
    let scale = meta.scale_f32();
    let offset = meta.offset_f32();
    let mut out = [0.0f32; MRL_PREFIX_DIMS];
    for i in 0..MRL_PREFIX_DIMS {
        out[i] = bits4[i] as f32 * scale + offset;
    }
    out
}

/// Extract 4-bit codes for one slot from a block (test helper).
pub fn extract_bits4(block: &PaxBlock64, slot: usize) -> [u8; MRL_PREFIX_DIMS] {
    let mask = 1u64 << slot;
    let mut bits4 = [0u8; MRL_PREFIX_DIMS];
    for d in 0..MRL_PREFIX_DIMS {
        let mut q = 0u8;
        if block.stratum0_sign.dim_bits[d] & mask != 0 {
            q |= 0b1000;
        }
        if block.stratum2_msb1.dim_bits[d] & mask != 0 {
            q |= 0b0100;
        }
        if block.stratum3_msb2.dim_bits[d] & mask != 0 {
            q |= 0b0010;
        }
        if block.stratum5_lsb.dim_bits[d] & mask != 0 {
            q |= 0b0001;
        }
        bits4[d] = q;
    }
    bits4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{source, EMBED_DIMS_768};

    #[test]
    fn encode_roundtrip_bits_and_crc() {
        let dims = 256;
        let mut v = vec![0.0f32; dims];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f32 * 0.07) % 2.0) - 1.0;
        }
        let block = encode_block(&[&v], &[42], &[source::CODE], dims);
        assert!(block.verify_crc());
        assert_eq!(block.header.min_timestamp, 42);
        assert_eq!(block.active_count(), 1);
        assert!(block.is_tombstoned(1));

        let bits = extract_bits4(&block, 0);
        let approx = dequantize_prefix(block.meta[0], bits);
        let mut err = 0.0f32;
        for i in 0..MRL_PREFIX_DIMS {
            err += (approx[i] - v[i]).abs();
        }
        assert!(
            err / (MRL_PREFIX_DIMS as f32) < 0.2,
            "mean abs err too high: {}",
            err / 64.0
        );
    }

    #[test]
    fn encode_768_packs_rho_and_stratum4() {
        let mut v = vec![0.0f32; EMBED_DIMS_768];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f32 * 0.013) % 2.0) - 1.0;
        }
        // L2 normalize so norms are well-defined.
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= n;
        }
        let block = encode_block(&[&v], &[1], &[source::DOC], EMBED_DIMS_768);
        let meta = block.meta[0];
        assert!(meta.flags > 0, "768-d vectors should pack a non-zero ρ");
        let mid = meta.mid_norm_f32();
        let high = meta.high_tail_norm_f32();
        let rest = meta.tail_norm_f32();
        assert!((mid * mid + high * high - rest * rest).abs() < 1e-3);

        // stratum4 should have a mix of 0/1 bits (not all-zero for random-ish input).
        let mut ones = 0u32;
        for d in 0..TAIL_FOLD_DIMS {
            if block.stratum4_mid.dim_bits[d] & 1 != 0 {
                ones += 1;
            }
        }
        assert!(ones > 5 && ones < 60, "unexpected stratum4 popcount {ones}");
    }

    #[test]
    fn encode_block_from_augmented_is_bit_identical_to_encode_block() {
        use crate::rotate::augment_projection_weights;

        let d_model = 16usize;
        let mut w_proj = vec![0.0f32; 768 * d_model];
        for (i, w) in w_proj.iter_mut().enumerate() {
            *w = ((i * 19 + 5) % 97) as f32 * 0.01 - 0.48;
        }
        let w_full = augment_projection_weights(&w_proj, d_model);

        let mut vecs_768 = Vec::new();
        let mut vecs_960 = Vec::new();
        let mut ts = Vec::new();
        let mut srcs = Vec::new();

        for v_idx in 0..8usize {
            let mut h = vec![0.0f32; d_model];
            for c in 0..d_model {
                h[c] = (((v_idx * 31 + c * 17 + 3) % 89) as f32) * 0.02 - 0.85;
            }
            let mut raw768 = [0.0f32; 768];
            for r in 0..768 {
                let mut s = 0.0f32;
                for c in 0..d_model {
                    s += w_proj[r * d_model + c] * h[c];
                }
                raw768[r] = s;
            }
            let norm = raw768.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let inv_norm = 1.0 / norm;
            for x in &mut raw768 {
                *x *= inv_norm;
            }

            let mut aug960 = [0.0f32; AUGMENTED_FULL_DIMS];
            for r in 0..AUGMENTED_FULL_DIMS {
                let mut s = 0.0f32;
                for c in 0..d_model {
                    s += w_full[r * d_model + c] * h[c];
                }
                aug960[r] = s * inv_norm;
            }

            vecs_768.push(raw768.to_vec());
            vecs_960.push(aug960);
            ts.push(1000 + v_idx as u32);
            srcs.push(source::DOC);
        }

        let refs_768: Vec<&[f32]> = vecs_768.iter().map(|v| v.as_slice()).collect();
        let block_std = encode_block(&refs_768, &ts, &srcs, 768);
        let block_aug = encode_block_from_augmented(&vecs_960, &ts, &srcs);

        assert_eq!(block_std.as_bytes(), block_aug.as_bytes());
        assert_eq!(block_std.header.crc32c, block_aug.header.crc32c);
    }
}
