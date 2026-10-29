//! 4 KB page-aligned stratified macro-block (`PaxBlock64`).
//!
//! Layout (spec §3.1):
//!   header 64 + meta 448 + S0 512 + S1 1024 + S2..S5 512×4 = 4096 bytes.

use bytemuck::{Pod, Zeroable};
use half::f16;

pub const VECTORS_PER_BLOCK: usize = 64;
pub const BLOCK_BYTES: usize = 4096;
pub const MRL_PREFIX_DIMS: usize = 64;
pub const RABITQ_DIMS: usize = 128;
pub const MID_PREFIX_DIMS: usize = 64;
/// Start of the high-dim tail folded into `stratum4_mid` (after prefix+RaBitQ).
pub const TAIL_START: usize = MRL_PREFIX_DIMS + RABITQ_DIMS; // 192
/// Folded tail sketch width (reclaims the previously-idle 512 B stratum4).
pub const TAIL_FOLD_DIMS: usize = 64;
/// Max 64-d blocks in the tail for EmbeddingGemma @768 (`(768-192)/64 = 9`).
pub const TAIL_FOLD_BLOCKS: usize = 9;
/// Canonical embedding width for Step-3 unrolled kernels.
pub const EMBED_DIMS_768: usize = 768;

/// Source channel bits in `BlockHeader::source_mask`.
pub mod source {
    pub const SLACK: u32 = 1 << 0;
    pub const EMAIL: u32 = 1 << 1;
    pub const CODE: u32 = 1 << 2;
    pub const DOC: u32 = 1 << 3;
}

/// 64-byte cache-line header with temporal / tombstone metadata.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlockHeader {
    pub min_timestamp: u32,
    pub max_timestamp: u32,
    pub source_mask: u32,
    pub coarse_cluster_id: u16,
    pub block_flags: u16,
    pub tombstone_mask: u64,
    pub crc32c: u32,
    pub reserved_a: [u8; 32],
    pub reserved_b: [u8; 4],
}

const _: () = assert!(std::mem::size_of::<BlockHeader>() == 64);

/// Per-vector affine scale / offset / tail L2 (7 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VectorMeta {
    pub scale: u16,     // f16 bits
    pub offset: u16,    // f16 bits
    pub tail_norm: u16, // f16 bits
    pub flags: u8,
}

const _: () = assert!(std::mem::size_of::<VectorMeta>() == 7);

impl VectorMeta {
    pub fn scale_f32(self) -> f32 {
        f16::from_bits(self.scale).to_f32()
    }
    pub fn offset_f32(self) -> f32 {
        f16::from_bits(self.offset).to_f32()
    }
    /// ‖x[64:]‖₂ (mid + high-dim tail residual after the 4-bit MRL prefix).
    pub fn tail_norm_f32(self) -> f32 {
        f16::from_bits(self.tail_norm).to_f32()
    }

    /// `flags/255 ≈ ‖x[192:]‖₂ / ‖x[64:]‖₂` (Step-3 mid/tail energy split).
    #[inline]
    pub fn rho_ratio(self) -> f32 {
        self.flags as f32 / 255.0
    }

    /// ‖x[64..192]‖₂ recovered from `tail_norm` and `flags`.
    #[inline]
    pub fn mid_norm_f32(self) -> f32 {
        let t = self.tail_norm_f32().max(0.0);
        let r = self.rho_ratio().clamp(0.0, 1.0);
        ((1.0 - r * r).max(0.0).sqrt()) * t
    }

    /// ‖x[192:]‖₂ recovered from `tail_norm` and `flags`.
    #[inline]
    pub fn high_tail_norm_f32(self) -> f32 {
        self.rho_ratio() * self.tail_norm_f32().max(0.0)
    }

    pub fn from_f32(scale: f32, offset: f32, tail_norm: f32, flags: u8) -> Self {
        Self {
            scale: f16::from_f32(scale).to_bits(),
            offset: f16::from_f32(offset).to_bits(),
            tail_norm: f16::from_f32(tail_norm).to_bits(),
            flags,
        }
    }

    /// Pack ρ = round(255 · ‖tail‖ / ‖rest‖) into `flags` (0 when no high-dim tail).
    pub fn pack_rho_flags(high_tail_norm: f32, rest_norm: f32) -> u8 {
        if rest_norm <= 1e-12 {
            return 0;
        }
        let r = (high_tail_norm / rest_norm).clamp(0.0, 1.0);
        (r * 255.0).round().clamp(0.0, 255.0) as u8
    }
}

/// Stratum 0 / 2 / 3 / 5: 64 dims × 64 vectors as transposed u64 planes.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Bitplane64 {
    /// `dim_bits[d]` bit `v` = bitplane value of vector `v` at dimension `d`.
    pub dim_bits: [u64; MRL_PREFIX_DIMS],
}

const _: () = assert!(std::mem::size_of::<Bitplane64>() == 512);

/// Stratum 1: RaBitQ rotated 1-bit plane over 128 dims.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RabitqPlane {
    pub dim_bits: [u64; RABITQ_DIMS],
}

const _: () = assert!(std::mem::size_of::<RabitqPlane>() == 1024);

/// Full 4 KB PaxBlock64.
#[repr(C, align(4096))]
#[derive(Clone, Copy, Debug)]
pub struct PaxBlock64 {
    pub header: BlockHeader,
    pub meta: [VectorMeta; VECTORS_PER_BLOCK],
    pub stratum0_sign: Bitplane64,
    pub stratum1_rabitq: RabitqPlane,
    pub stratum2_msb1: Bitplane64,
    pub stratum3_msb2: Bitplane64,
    pub stratum4_mid: Bitplane64,
    pub stratum5_lsb: Bitplane64,
}

const _: () = assert!(std::mem::size_of::<PaxBlock64>() == BLOCK_BYTES);

unsafe impl Zeroable for PaxBlock64 {}
unsafe impl Pod for PaxBlock64 {}

impl Default for PaxBlock64 {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl PaxBlock64 {
    pub fn zeroed() -> Self {
        bytemuck::Zeroable::zeroed()
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<&Self> {
        if bytes.len() < BLOCK_BYTES {
            return None;
        }
        bytemuck::try_from_bytes(&bytes[..BLOCK_BYTES]).ok()
    }

    pub fn from_bytes_mut(bytes: &mut [u8]) -> Option<&mut Self> {
        if bytes.len() < BLOCK_BYTES {
            return None;
        }
        bytemuck::try_from_bytes_mut(&mut bytes[..BLOCK_BYTES]).ok()
    }

    pub fn active_count(&self) -> u32 {
        (!self.header.tombstone_mask).count_ones()
    }

    pub fn is_tombstoned(&self, slot: usize) -> bool {
        slot < VECTORS_PER_BLOCK && (self.header.tombstone_mask & (1u64 << slot)) != 0
    }

    pub fn recompute_crc(&mut self) {
        self.header.crc32c = 0;
        let bytes = bytemuck::bytes_of(self);
        self.header.crc32c = crc32fast::hash(bytes);
    }

    pub fn verify_crc(&self) -> bool {
        let mut copy = *self;
        let expected = copy.header.crc32c;
        copy.header.crc32c = 0;
        crc32fast::hash(bytemuck::bytes_of(&copy)) == expected
    }
}

/// Global vector identity: block index × slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorId {
    pub block: u32,
    pub slot: u16,
}

impl VectorId {
    pub fn new(block: u32, slot: u16) -> Self {
        debug_assert!((slot as usize) < VECTORS_PER_BLOCK);
        Self { block, slot }
    }

    pub fn as_u64(self) -> u64 {
        ((self.block as u64) << 16) | self.slot as u64
    }

    pub fn from_u64(v: u64) -> Self {
        Self {
            block: (v >> 16) as u32,
            slot: (v & 0xffff) as u16,
        }
    }
}
