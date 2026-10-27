//! Vertical bitplane scan with Cauchy–Schwarz early rejection (spec §2, §4).
//!
//! Scoring reconstructs an approximate inner product from the **full** 4-bit
//! MRL prefix (strata 0/2/3/5) plus a scaled RaBitQ mid-band (stratum 1).
//! Progressive strata are used for *pruning bounds*; ranking always uses the
//! best available reconstruction for the configured depth. Using MSB-only for
//! ranking collapses shortlist recall (≈7% coverage of exact@10 on nomic MRL).

use crate::block::{
    PaxBlock64, VectorId, MRL_PREFIX_DIMS, RABITQ_DIMS, VECTORS_PER_BLOCK,
};
use crate::config::{EngineConfig, StrataDepth};
use crate::rotate::{rabitq_bit_scale, rotate_mid_band};

/// Precomputed query side for bitplane scoring.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub dims: usize,
    pub prefix: [f32; MRL_PREFIX_DIMS],
    /// Per-dim weights for bits 0..3: `q_d * 2^k`.
    pub bit_weights: [[f32; 4]; MRL_PREFIX_DIMS],
    pub rabitq_q: [f32; RABITQ_DIMS],
    pub rabitq_scale: f32,
    pub mid_prefix: [f32; MRL_PREFIX_DIMS],
    pub q_tail_norm: f32,
    /// ‖q[64..]‖₂ — remainder after leading 64 dims.
    pub q_after_prefix: f32,
    pub q_after_s1: f32,
    /// Σ |q_d| over prefix — for bitplane residual slack.
    pub prefix_l1: f32,
}

impl QueryPlan {
    pub fn from_query(query: &[f32], dims: usize) -> Self {
        assert_eq!(query.len(), dims);
        assert!(dims >= MRL_PREFIX_DIMS + RABITQ_DIMS);

        let mut prefix = [0.0f32; MRL_PREFIX_DIMS];
        prefix.copy_from_slice(&query[..MRL_PREFIX_DIMS]);

        let mut bit_weights = [[0.0f32; 4]; MRL_PREFIX_DIMS];
        let mut prefix_l1 = 0.0f32;
        for d in 0..MRL_PREFIX_DIMS {
            let q = prefix[d];
            prefix_l1 += q.abs();
            bit_weights[d][0] = q * 1.0;
            bit_weights[d][1] = q * 2.0;
            bit_weights[d][2] = q * 4.0;
            bit_weights[d][3] = q * 8.0;
        }

        let mid = &query[MRL_PREFIX_DIMS..MRL_PREFIX_DIMS + RABITQ_DIMS];
        let rabitq_q = rotate_mid_band(mid);
        let rabitq_scale = rabitq_bit_scale();

        let mut mid_prefix = [0.0f32; MRL_PREFIX_DIMS];
        mid_prefix.copy_from_slice(&query[MRL_PREFIX_DIMS..MRL_PREFIX_DIMS + MRL_PREFIX_DIMS]);

        let mut q_tail_sq = 0.0f32;
        for &x in &query[MRL_PREFIX_DIMS + RABITQ_DIMS..] {
            q_tail_sq += x * x;
        }
        let q_tail_norm = q_tail_sq.sqrt();

        let mut after_pfx = q_tail_sq;
        for i in 0..RABITQ_DIMS {
            after_pfx += query[MRL_PREFIX_DIMS + i].powi(2);
        }
        let q_after_prefix = after_pfx.sqrt();

        Self {
            dims,
            prefix,
            bit_weights,
            rabitq_q,
            rabitq_scale,
            mid_prefix,
            q_tail_norm,
            q_after_prefix,
            q_after_s1: q_tail_norm,
            prefix_l1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScoredHit {
    pub id: VectorId,
    pub score: f32,
}

/// Fixed-capacity min-heap of the best `capacity` scores (keeps highest).
pub struct Shortlist {
    capacity: usize,
    heap: Vec<ScoredHit>,
}

impl Shortlist {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            heap: Vec::with_capacity(capacity),
        }
    }

    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.capacity {
            f32::NEG_INFINITY
        } else {
            self.heap
                .iter()
                .map(|h| h.score)
                .fold(f32::INFINITY, f32::min)
        }
    }

    pub fn push(&mut self, hit: ScoredHit) {
        if self.heap.len() < self.capacity {
            self.heap.push(hit);
            return;
        }
        if let Some((idx, worst)) = self
            .heap
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap())
        {
            if hit.score > worst.score {
                self.heap[idx] = hit;
            }
        }
    }

    pub fn into_sorted(mut self) -> Vec<ScoredHit> {
        self.heap
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        self.heap
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }
}

/// Accumulate one transposed bitplane into parallel scores for live slots.
#[inline]
fn accum_plane(plane: u64, live: u64, weight: f32, scores: &mut [f32; VECTORS_PER_BLOCK]) {
    if weight == 0.0 {
        return;
    }
    let mut bits = plane & live;
    while bits != 0 {
        let v = bits.trailing_zeros() as usize;
        scores[v] += weight;
        bits &= bits - 1;
    }
}

/// Scan one PaxBlock64; push survivors into `shortlist`.
pub fn scan_block(
    block: &PaxBlock64,
    block_idx: u32,
    plan: &QueryPlan,
    cfg: &EngineConfig,
    shortlist: &mut Shortlist,
    time_lo: Option<u32>,
    time_hi: Option<u32>,
) {
    if cfg.temporal_filter_enable {
        if let Some(lo) = time_lo {
            if block.header.max_timestamp < lo {
                return;
            }
        }
        if let Some(hi) = time_hi {
            if block.header.min_timestamp > hi {
                return;
            }
        }
    }

    let tau = shortlist.threshold();
    let alpha = cfg.cs_tolerance;
    let active = !block.header.tombstone_mask;
    if active == 0 {
        return;
    }

    // --- Reconstruct 4-bit codes · q  (code contribution; scale applied per vec) ---
    // Always use all four bitplanes for ranking quality. StrataDepth only gates
    // how much mid-band / refinement we add and how tight CS pruning is.
    let mut code_dot = [0.0f32; VECTORS_PER_BLOCK];
    for d in 0..MRL_PREFIX_DIMS {
        // bit3 MSB (stratum 0), bit2 (s2), bit1 (s3), bit0 LSB (s5)
        accum_plane(
            block.stratum0_sign.dim_bits[d],
            active,
            plan.bit_weights[d][3],
            &mut code_dot,
        );
        if !matches!(cfg.strata_depth, StrataDepth::L0Only) {
            // Even L1_RABITQ must use full 4-bit for ranking — planes are on disk.
            accum_plane(
                block.stratum2_msb1.dim_bits[d],
                active,
                plan.bit_weights[d][2],
                &mut code_dot,
            );
            accum_plane(
                block.stratum3_msb2.dim_bits[d],
                active,
                plan.bit_weights[d][1],
                &mut code_dot,
            );
            accum_plane(
                block.stratum5_lsb.dim_bits[d],
                active,
                plan.bit_weights[d][0],
                &mut code_dot,
            );
        }
    }

    let mut scores = [0.0f32; VECTORS_PER_BLOCK];
    let mut survivors = 0u64;
    for v in 0..VECTORS_PER_BLOCK {
        let bit = 1u64 << v;
        if active & bit == 0 {
            continue;
        }
        let scale = block.meta[v].scale_f32();
        let offset = block.meta[v].offset_f32();
        // q·(scale·code + offset·1) = scale·(q·code) + offset·Σq
        let mut prefix_ip = code_dot[v] * scale;
        for d in 0..MRL_PREFIX_DIMS {
            prefix_ip += plan.prefix[d] * offset;
        }
        scores[v] = prefix_ip;

        // CS: prefix_ip + ‖q[64:]‖₂ ‖x[64:]‖₂ ≥ true IP.
        let x_rest = block.meta[v].tail_norm_f32().max(0.0);
        let ub = prefix_ip + alpha * plan.q_after_prefix * x_rest;
        if ub >= tau {
            survivors |= bit;
        }
    }

    if survivors == 0 {
        return;
    }

    if matches!(cfg.strata_depth, StrataDepth::L0Only) {
        finish_push(block_idx, survivors, &scores, shortlist);
        return;
    }

    // --- Stratum 1: RaBitQ 1-bit mid-band (dims 64..191), IP-scaled ---
    let alpha_bit = plan.rabitq_scale;
    for d in 0..RABITQ_DIMS {
        let plane = block.stratum1_rabitq.dim_bits[d];
        let q_rot = plan.rabitq_q[d];
        let pos = q_rot * alpha_bit;
        let neg = -pos;
        let mut bits = plane & survivors;
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            scores[v] += pos;
            bits &= bits - 1;
        }
        let mut bits = (!plane) & survivors;
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            scores[v] += neg;
            bits &= bits - 1;
        }
    }

    // Post-S1: residual after prefix+mid is dims ≥ 192.
    // We stored ‖x[64:]‖; conservative: allow full residual as UB slack.
    let mut survivors2 = 0u64;
    for v in 0..VECTORS_PER_BLOCK {
        let bit = 1u64 << v;
        if survivors & bit == 0 {
            continue;
        }
        let ub = scores[v] + alpha * plan.q_after_s1 * block.meta[v].tail_norm_f32();
        if ub >= tau {
            survivors2 |= bit;
        }
    }
    survivors = survivors2;
    if survivors == 0 {
        return;
    }

    if matches!(cfg.strata_depth, StrataDepth::L1Rabitq) {
        finish_push(block_idx, survivors, &scores, shortlist);
        return;
    }

    // Stratum 4 is retained on disk for future mid-prefix refinement; do not
    // perturb the RaBitQ-calibrated score here (empirically hurts ranking).

    finish_push(block_idx, survivors, &scores, shortlist);
}

fn finish_push(
    block_idx: u32,
    survivors: u64,
    scores: &[f32; VECTORS_PER_BLOCK],
    shortlist: &mut Shortlist,
) {
    let mut bits = survivors;
    while bits != 0 {
        let v = bits.trailing_zeros() as usize;
        shortlist.push(ScoredHit {
            id: VectorId::new(block_idx, v as u16),
            score: scores[v],
        });
        bits &= bits - 1;
    }
}

/// Scan a contiguous slice of blocks.
pub fn scan_blocks(
    blocks: &[PaxBlock64],
    plan: &QueryPlan,
    cfg: &EngineConfig,
    time_lo: Option<u32>,
    time_hi: Option<u32>,
) -> Vec<ScoredHit> {
    let mut shortlist = Shortlist::new(cfg.shortlist_cand);
    for (i, block) in blocks.iter().enumerate() {
        scan_block(
            block,
            i as u32,
            plan,
            cfg,
            &mut shortlist,
            time_lo,
            time_hi,
        );
    }
    shortlist.into_sorted()
}
