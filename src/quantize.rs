//! Affine 4-bit MRL prefix + RaBitQ-style 1-bit mid-band quantization.

use crate::block::{
    Bitplane64, PaxBlock64, RabitqPlane, VectorMeta, VECTORS_PER_BLOCK, MRL_PREFIX_DIMS,
    RABITQ_DIMS,
};
use crate::rotate::rotate_mid_band;

/// Build a filled PaxBlock64 from up to 64 float vectors (row-major, `dims` each).
pub fn encode_block(
    vectors: &[&[f32]],
    timestamps: &[u32],
    sources: &[u32],
    dims: usize,
) -> PaxBlock64 {
    assert!(!vectors.is_empty() && vectors.len() <= VECTORS_PER_BLOCK);
    assert_eq!(timestamps.len(), vectors.len());
    assert_eq!(sources.len(), vectors.len());
    assert!(dims >= MRL_PREFIX_DIMS + RABITQ_DIMS);

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

        let (meta, bits4, rabitq) = quantize_vector(vec);
        block.meta[i] = meta;
        write_4bit_planes(&mut block, i, &bits4);
        write_rabitq_plane(&mut block.stratum1_rabitq, i, &rabitq);
        write_mid_plane(&mut block.stratum4_mid, i, vec);
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

fn quantize_vector(vec: &[f32]) -> (VectorMeta, [u8; MRL_PREFIX_DIMS], [u8; RABITQ_DIMS]) {
    let prefix = &vec[..MRL_PREFIX_DIMS];
    let mid = &vec[MRL_PREFIX_DIMS..MRL_PREFIX_DIMS + RABITQ_DIMS];
    let tail = &vec[MRL_PREFIX_DIMS + RABITQ_DIMS..];

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

    // Real orthogonal mix (FWHT). Per-coordinate ±1 flips are a no-op for
    // sign-agreement tests and must not be used here.
    let rotated = rotate_mid_band(mid);
    let mut rabitq = [0u8; RABITQ_DIMS];
    for i in 0..RABITQ_DIMS {
        rabitq[i] = if rotated[i] >= 0.0 { 1 } else { 0 };
    }

    let mut tail_sq = 0.0f32;
    for &x in tail {
        tail_sq += x * x;
    }

    let meta = VectorMeta::from_f32(scale, offset, tail_sq.sqrt(), 0);
    (meta, bits4, rabitq)
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

fn write_mid_plane(plane: &mut Bitplane64, slot: usize, vec: &[f32]) {
    let mask = 1u64 << slot;
    let start = MRL_PREFIX_DIMS;
    for d in 0..MRL_PREFIX_DIMS {
        // 1-bit refinement of dims 64..127 (MSB of unsigned mid-grid relative to 0)
        if vec[start + d] >= 0.0 {
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
    use crate::block::source;

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
}
