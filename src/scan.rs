//! Vertical bitplane scan with Cauchy–Schwarz early rejection (spec §2, §4).

use crate::block::{
    PaxBlock64, VectorId, MRL_PREFIX_DIMS, RABITQ_DIMS, VECTORS_PER_BLOCK,
};
use crate::config::{EngineConfig, StrataDepth};
use crate::rotate::{rabitq_bit_scale, rotate_mid_band};

/// Precomputed query side for bitplane scoring.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub dims: usize,
    /// Query values for dims 0..63.
    pub prefix: [f32; MRL_PREFIX_DIMS],
    /// Quantized 4-bit LUT weights: for each dim, contribution of bit k ∈ {0..3}.
    /// score += scale * Σ_k bit_k * weight[d][k]  + offset terms handled separately.
    pub bit_weights: [[f32; 4]; MRL_PREFIX_DIMS],
    /// FWHT-rotated query mid-band (dims 64..191), L2-preserving.
    pub rabitq_q: [f32; RABITQ_DIMS],
    /// Maps ±1 codes toward IP units (√(2/π)/√D).
    pub rabitq_scale: f32,
    /// Mid-prefix query dims 64..127 for stratum 4.
    pub mid_prefix: [f32; MRL_PREFIX_DIMS],
    /// ‖q_tail‖₂ for Cauchy–Schwarz remainder (dims ≥ 192).
    pub q_tail_norm: f32,
    /// ‖q_unvisited‖ envelope helpers per stratum checkpoint.
    pub q_after_s0: f32,
    pub q_after_s1: f32,
}

impl QueryPlan {
    pub fn from_query(query: &[f32], dims: usize) -> Self {
        assert_eq!(query.len(), dims);
        assert!(dims >= MRL_PREFIX_DIMS + RABITQ_DIMS);

        let mut prefix = [0.0f32; MRL_PREFIX_DIMS];
        prefix.copy_from_slice(&query[..MRL_PREFIX_DIMS]);

        // Bit weights for unsigned 4-bit grid: value = Σ_k bit_k * 2^k
        // Inner product contribution per dim: q_d * (scale * code + offset)
        // During scan we accumulate q_d * code; scale/offset applied per-vector.
        let mut bit_weights = [[0.0f32; 4]; MRL_PREFIX_DIMS];
        for d in 0..MRL_PREFIX_DIMS {
            let q = prefix[d];
            bit_weights[d][0] = q * 1.0; // LSB
            bit_weights[d][1] = q * 2.0;
            bit_weights[d][2] = q * 4.0;
            bit_weights[d][3] = q * 8.0; // MSB
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

        // Remainder norms for progressive UB (conservative envelopes).
        let mut rem_s0 = q_tail_sq;
        for i in 0..RABITQ_DIMS {
            rem_s0 += query[MRL_PREFIX_DIMS + i].powi(2);
        }
        // After S0 we still owe mid+tail; after S1 only true tail (+ mid refinement slack).
        let q_after_s0 = rem_s0.sqrt();
        let q_after_s1 = q_tail_norm;

        Self {
            dims,
            prefix,
            bit_weights,
            rabitq_q,
            rabitq_scale,
            mid_prefix,
            q_tail_norm,
            q_after_s0,
            q_after_s1,
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
    // Min-heap by score among survivors.
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
        // Replace current worst if better.
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

    // --- Stratum 0: MSB plane → coarse code contribution ---
    let mut scores = [0.0f32; VECTORS_PER_BLOCK];
    let mut offset_dot = [0.0f32; VECTORS_PER_BLOCK];

    // Pre-sum q·offset for live slots (affine term).
    for v in 0..VECTORS_PER_BLOCK {
        if (active & (1u64 << v)) == 0 {
            continue;
        }
        let off = block.meta[v].offset_f32();
        let mut s = 0.0f32;
        for d in 0..MRL_PREFIX_DIMS {
            s += plan.prefix[d] * off;
        }
        offset_dot[v] = s;
    }

    // Accumulate MSB (bit3) contributions via transposed POPCNT-friendly AND.
    for d in 0..MRL_PREFIX_DIMS {
        let plane = block.stratum0_sign.dim_bits[d];
        let w = plan.bit_weights[d][3];
        if w == 0.0 {
            continue;
        }
        let live = plane & active;
        let mut bits = live;
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            scores[v] += w;
            bits &= bits - 1;
        }
    }

    // Bound check 0: drop cold vectors.
    let mut survivors = 0u64;
    for v in 0..VECTORS_PER_BLOCK {
        let bit = 1u64 << v;
        if active & bit == 0 {
            continue;
        }
        let scale = block.meta[v].scale_f32();
        let approx = scores[v] * scale + offset_dot[v];
        let ub = approx + alpha * plan.q_after_s0 * block.meta[v].tail_norm_f32().max(1.0);
        // Conservative: also allow remaining bitplane capacity (±15*scale*|q|_1 slack loosely).
        let bit_slack = 7.0 * scale * plan.prefix.iter().map(|x| x.abs()).sum::<f32>();
        let ub = ub + alpha * bit_slack * 0.25;
        if ub >= tau {
            survivors |= bit;
            scores[v] = approx; // store scaled partial for refinement path
        }
    }

    if survivors == 0 {
        return;
    }

    match cfg.strata_depth {
        StrataDepth::L0Only => {
            finish_push(block, block_idx, survivors, &scores, shortlist);
            return;
        }
        _ => {}
    }

    // --- Stratum 1: RaBitQ 1-bit estimator (scaled into IP units) ---
    // Previously added raw ±|q_d| which dwarfed the S0 IP estimate, inflated τ,
    // and caused false CS pruning on later blocks. Score with FWHT(q) · (±α).
    let z = cfg.rabitq_confidence;
    let alpha_bit = plan.rabitq_scale;
    for d in 0..RABITQ_DIMS {
        let plane = block.stratum1_rabitq.dim_bits[d];
        let q_rot = plan.rabitq_q[d];
        let pos = q_rot * alpha_bit;
        let neg = -pos;
        let live = plane & survivors;
        let mut bits = live;
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            scores[v] += pos;
            bits &= bits - 1;
        }
        let disagree = (!plane) & survivors;
        let mut bits = disagree;
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            scores[v] += neg;
            bits &= bits - 1;
        }
        let _ = z; // confidence reserved for soft CI widening in FULL mode
    }

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
        finish_push(block, block_idx, survivors, &scores, shortlist);
        return;
    }

    // --- Strata 2,3,5: refine leading 4-bit codes ---
    for d in 0..MRL_PREFIX_DIMS {
        let scale_plane = [
            (block.stratum5_lsb.dim_bits[d], plan.bit_weights[d][0]),
            (block.stratum3_msb2.dim_bits[d], plan.bit_weights[d][1]),
            (block.stratum2_msb1.dim_bits[d], plan.bit_weights[d][2]),
        ];
        for (plane, w) in scale_plane {
            if w == 0.0 {
                continue;
            }
            let live = plane & survivors;
            let mut bits = live;
            while bits != 0 {
                let v = bits.trailing_zeros() as usize;
                let scale = block.meta[v].scale_f32();
                scores[v] += w * scale;
                bits &= bits - 1;
            }
        }
    }

    // --- Stratum 4: mid-prefix sign refinement ---
    for d in 0..MRL_PREFIX_DIMS {
        let plane = block.stratum4_mid.dim_bits[d];
        let q = plan.mid_prefix[d];
        let live = plane & survivors;
        let mut bits = live;
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            scores[v] += q; // bit=1 ⇒ positive mid
            bits &= bits - 1;
        }
        let neg = (!plane) & survivors;
        let mut bits = neg;
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            scores[v] -= q.abs() * 0.25; // mild penalty when mid bit disagrees
            bits &= bits - 1;
        }
    }

    finish_push(block, block_idx, survivors, &scores, shortlist);
}

fn finish_push(
    _block: &PaxBlock64,
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

/// Scan a contiguous slice of blocks (serial; caller may shard with rayon).
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
