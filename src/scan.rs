//! Vertical bitplane scan with Cauchy–Schwarz early rejection (spec §2, §4).
//!
//! Scoring reconstructs an approximate inner product from the **full** 4-bit
//! MRL prefix (strata 0/2/3/5) plus a scaled RaBitQ mid-band (stratum 1).
//! Progressive strata are used for *pruning bounds*; ranking always uses the
//! best available reconstruction for the configured depth. Using MSB-only for
//! ranking collapses shortlist recall (≈7% coverage of exact@10 on nomic MRL).

use rayon::prelude::*;

use crate::block::{
    PaxBlock64, VectorId, MRL_PREFIX_DIMS, RABITQ_DIMS, TAIL_FOLD_DIMS, TAIL_START,
    VECTORS_PER_BLOCK,
};
use crate::config::{EngineConfig, StrataDepth};
use crate::rotate::{rabitq_bit_scale, rotate_mid_band, rotate_tail_fold, tail_fold_bit_scale};

/// Precomputed query side for bitplane scoring.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub dims: usize,
    pub prefix: [f32; MRL_PREFIX_DIMS],
    /// Per-dim weights for bits 0..3: `q_d * 2^k`.
    pub bit_weights: [[f32; 4]; MRL_PREFIX_DIMS],
    pub rabitq_q: [f32; RABITQ_DIMS],
    pub rabitq_scale: f32,
    /// `2.0 * rabitq_q[d] * rabitq_scale` for single-pass bit=1 accumulation.
    pub rabitq_weights2: [f32; RABITQ_DIMS],
    /// `- Σ_d rabitq_q[d] * rabitq_scale` baseline when all bits are 0.
    pub rabitq_base: f32,
    /// 4-dim LUTs for dense Stratum-1 evaluation (`32 × 16` floats = 2 KiB).
    pub rabitq_lut: [[f32; 16]; RABITQ_DIMS / 4],
    /// FWHT(fold(q[192:])) — Step-3 stratum4 query side.
    pub tail_fold_q: [f32; TAIL_FOLD_DIMS],
    pub tail_fold_weights2: [f32; TAIL_FOLD_DIMS],
    pub tail_fold_base: f32,
    pub has_tail_fold: bool,
    /// ‖q[192:]‖₂ — high-dim tail only (post-S1 residual).
    pub q_tail_norm: f32,
    /// ‖q[64:]‖₂ — remainder after leading 64 dims.
    pub q_after_prefix: f32,
    /// Alias of `q_tail_norm` (post-S1 residual energy).
    pub q_after_s1: f32,
    /// Σ |q_d| over prefix — for bitplane residual slack.
    pub prefix_l1: f32,
    /// Σ q_d over prefix — for O(1) affine offset dot product.
    pub prefix_sum: f32,
}

impl QueryPlan {
    pub fn from_query(query: &[f32], dims: usize) -> Self {
        assert_eq!(query.len(), dims);
        assert!(dims >= MRL_PREFIX_DIMS + RABITQ_DIMS);

        let mut prefix = [0.0f32; MRL_PREFIX_DIMS];
        prefix.copy_from_slice(&query[..MRL_PREFIX_DIMS]);

        let mut bit_weights = [[0.0f32; 4]; MRL_PREFIX_DIMS];
        let mut prefix_l1 = 0.0f32;
        let mut prefix_sum = 0.0f32;
        for d in 0..MRL_PREFIX_DIMS {
            let q = prefix[d];
            prefix_l1 += q.abs();
            prefix_sum += q;
            bit_weights[d][0] = q * 1.0;
            bit_weights[d][1] = q * 2.0;
            bit_weights[d][2] = q * 4.0;
            bit_weights[d][3] = q * 8.0;
        }

        let mid = &query[MRL_PREFIX_DIMS..MRL_PREFIX_DIMS + RABITQ_DIMS];
        let rabitq_q = rotate_mid_band(mid);
        let rabitq_scale = rabitq_bit_scale();

        let mut rabitq_weights2 = [0.0f32; RABITQ_DIMS];
        let mut rabitq_base = 0.0f32;
        for d in 0..RABITQ_DIMS {
            let pos = rabitq_q[d] * rabitq_scale;
            rabitq_base -= pos;
            rabitq_weights2[d] = 2.0 * pos;
        }

        let mut rabitq_lut = [[0.0f32; 16]; RABITQ_DIMS / 4];
        for g in 0..(RABITQ_DIMS / 4) {
            let p0 = rabitq_q[4 * g] * rabitq_scale;
            let p1 = rabitq_q[4 * g + 1] * rabitq_scale;
            let p2 = rabitq_q[4 * g + 2] * rabitq_scale;
            let p3 = rabitq_q[4 * g + 3] * rabitq_scale;
            for mask in 0..16usize {
                let s0 = if (mask & 1) != 0 { p0 } else { -p0 };
                let s1 = if (mask & 2) != 0 { p1 } else { -p1 };
                let s2 = if (mask & 4) != 0 { p2 } else { -p2 };
                let s3 = if (mask & 8) != 0 { p3 } else { -p3 };
                rabitq_lut[g][mask] = s0 + s1 + s2 + s3;
            }
        }

        let mut q_tail_sq = 0.0f32;
        for &x in &query[TAIL_START..] {
            q_tail_sq += x * x;
        }
        let q_tail_norm = q_tail_sq.sqrt();

        let mut after_pfx = q_tail_sq;
        for i in 0..RABITQ_DIMS {
            after_pfx += query[MRL_PREFIX_DIMS + i].powi(2);
        }
        let q_after_prefix = after_pfx.sqrt();

        let has_tail_fold = dims > TAIL_START;
        let tail_fold_q = if has_tail_fold {
            rotate_tail_fold(query)
        } else {
            [0.0f32; TAIL_FOLD_DIMS]
        };
        let tail_fold_scale = tail_fold_bit_scale();
        let mut tail_fold_weights2 = [0.0f32; TAIL_FOLD_DIMS];
        let mut tail_fold_base = 0.0f32;
        if has_tail_fold {
            for d in 0..TAIL_FOLD_DIMS {
                let pos = tail_fold_q[d] * tail_fold_scale;
                tail_fold_base -= pos;
                tail_fold_weights2[d] = 2.0 * pos;
            }
        }

        Self {
            dims,
            prefix,
            bit_weights,
            rabitq_q,
            rabitq_scale,
            rabitq_weights2,
            rabitq_base,
            rabitq_lut,
            tail_fold_q,
            tail_fold_weights2,
            tail_fold_base,
            has_tail_fold,
            q_tail_norm,
            q_after_prefix,
            q_after_s1: q_tail_norm,
            prefix_l1,
            prefix_sum,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScoredHit {
    pub id: VectorId,
    pub score: f32,
}

/// Fixed-capacity top-K tracker with cached O(1) minimum threshold.
#[derive(Debug, Clone)]
pub struct Shortlist {
    capacity: usize,
    heap: Vec<ScoredHit>,
    min_idx: usize,
    min_score: f32,
    floor_tau: f32,
}

impl Shortlist {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            heap: Vec::with_capacity(capacity),
            min_idx: 0,
            min_score: f32::INFINITY,
            floor_tau: f32::NEG_INFINITY,
        }
    }

    pub fn with_floor(capacity: usize, floor_tau: f32) -> Self {
        Self {
            capacity: capacity.max(1),
            heap: Vec::with_capacity(capacity),
            min_idx: 0,
            min_score: f32::INFINITY,
            floor_tau,
        }
    }

    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.capacity {
            self.floor_tau
        } else {
            self.min_score.max(self.floor_tau)
        }
    }

    #[inline]
    pub fn push(&mut self, hit: ScoredHit) {
        if hit.score < self.floor_tau {
            return;
        }
        if self.heap.len() < self.capacity {
            let idx = self.heap.len();
            if hit.score < self.min_score {
                self.min_score = hit.score;
                self.min_idx = idx;
            }
            self.heap.push(hit);
            return;
        }
        if hit.score > self.min_score {
            self.heap[self.min_idx] = hit;
            self.recompute_min();
        }
    }

    #[inline]
    fn recompute_min(&mut self) {
        let mut min_idx = 0;
        let mut min_score = f32::INFINITY;
        for (i, h) in self.heap.iter().enumerate() {
            if h.score < min_score {
                min_score = h.score;
                min_idx = i;
            }
        }
        self.min_idx = min_idx;
        self.min_score = min_score;
    }

    pub fn merge(&mut self, other: &Shortlist) {
        for &hit in &other.heap {
            self.push(hit);
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

/// SWAR 4-bit nibble unpack mask (selects 1 bit per byte across 8 bytes).
const SWAR_M8: u64 = 0x0101_0101_0101_0101;

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

    let mut scores = [0.0f32; VECTORS_PER_BLOCK];
    let mut survivors = 0u64;

    if matches!(cfg.strata_depth, StrataDepth::L0Only) {
        let mut code_dot = [0.0f32; VECTORS_PER_BLOCK];
        for d in 0..MRL_PREFIX_DIMS {
            accum_plane(
                block.stratum0_sign.dim_bits[d],
                active,
                plan.bit_weights[d][3],
                &mut code_dot,
            );
        }
        for v in 0..VECTORS_PER_BLOCK {
            let bit = 1u64 << v;
            if active & bit == 0 {
                continue;
            }
            let scale = block.meta[v].scale_f32();
            let offset = block.meta[v].offset_f32();
            let prefix_ip = code_dot[v] * scale + offset * plan.prefix_sum;
            scores[v] = prefix_ip;

            let x_rest = block.meta[v].tail_norm_f32().max(0.0);
            let ub = prefix_ip + alpha * plan.q_after_prefix * x_rest;
            if ub >= tau {
                survivors |= bit;
            }
        }
        if survivors != 0 {
            finish_push(block_idx, survivors, &scores, shortlist);
        }
        return;
    }

    // --- Branchless SWAR + SIMD 4-bit MRL prefix reconstruction (strata 0/2/3/5) ---
    // `acc[s][k]` holds the unscaled 4-bit dot product for vector `v = 8 * k + s`.
    let mut acc = [[0.0f32; 8]; 8];
    for d in 0..MRL_PREFIX_DIMS {
        let q = plan.prefix[d];
        if q == 0.0 {
            continue;
        }
        let b0 = block.stratum5_lsb.dim_bits[d];
        let b1 = block.stratum3_msb2.dim_bits[d];
        let b2 = block.stratum2_msb1.dim_bits[d];
        let b3 = block.stratum0_sign.dim_bits[d];

        for s in 0..8 {
            let packed = ((b0 >> s) & SWAR_M8)
                | (((b1 >> s) & SWAR_M8) << 1)
                | (((b2 >> s) & SWAR_M8) << 2)
                | (((b3 >> s) & SWAR_M8) << 3);
            let bytes = packed.to_le_bytes();
            for k in 0..8 {
                acc[s][k] += (bytes[k] as f32) * q;
            }
        }
    }

    for v in 0..VECTORS_PER_BLOCK {
        let bit = 1u64 << v;
        if active & bit == 0 {
            continue;
        }
        let scale = block.meta[v].scale_f32();
        let offset = block.meta[v].offset_f32();
        let code_dot_v = acc[v & 7][v >> 3];
        let prefix_ip = code_dot_v * scale + offset * plan.prefix_sum;
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

    // Precompute per-survivor mid/high norms once (flags → f16 decode + sqrt).
    let mut mid_n = [0.0f32; VECTORS_PER_BLOCK];
    let mut high_n = [0.0f32; VECTORS_PER_BLOCK];
    {
        let mut bits = survivors;
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            mid_n[v] = block.meta[v].mid_norm_f32().max(1e-6);
            high_n[v] = block.meta[v].high_tail_norm_f32().max(0.0);
            bits &= bits - 1;
        }
    }

    // --- Stratum 1: RaBitQ 1-bit mid-band (dims 64..191), IP-scaled ---
    // Using identity: Σ (2*bit_d - 1)*pos_d = rabitq_base + Σ bit_d * (2*pos_d)
    // Accumulate unscaled bitplane sum into `s1_dot`, then multiply once per survivor by `mid_n[v]`.
    let mut s1_dot = [plan.rabitq_base; VECTORS_PER_BLOCK];
    for d in 0..RABITQ_DIMS {
        let mut bits = block.stratum1_rabitq.dim_bits[d] & survivors;
        let w = plan.rabitq_weights2[d];
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            s1_dot[v] += w;
            bits &= bits - 1;
        }
    }
    {
        let mut s_bits = survivors;
        while s_bits != 0 {
            let v = s_bits.trailing_zeros() as usize;
            scores[v] += s1_dot[v] * mid_n[v];
            s_bits &= s_bits - 1;
        }
    }

    // Post-S1 CS: residual is the high-dim tail only (dims ≥ 192).
    let mut survivors2 = 0u64;
    let mut check_bits = survivors;
    while check_bits != 0 {
        let v = check_bits.trailing_zeros() as usize;
        let ub = scores[v] + alpha * plan.q_after_s1 * high_n[v];
        if ub >= tau {
            survivors2 |= 1u64 << v;
        }
        check_bits &= check_bits - 1;
    }
    survivors = survivors2;
    if survivors == 0 {
        return;
    }

    // --- Stratum 4: folded high-dim tail (dims 192..768) → 64-bit FWHT signs ---
    // Applied for both L1 and Full64b whenever the query has a tail (Step 3).
    if plan.has_tail_fold {
        let mut s4_dot = [plan.tail_fold_base; VECTORS_PER_BLOCK];
        for d in 0..TAIL_FOLD_DIMS {
            let mut bits = block.stratum4_mid.dim_bits[d] & survivors;
            let w = plan.tail_fold_weights2[d];
            while bits != 0 {
                let v = bits.trailing_zeros() as usize;
                s4_dot[v] += w;
                bits &= bits - 1;
            }
        }
        let w_s4 = cfg.stratum4_weight;
        let mut s_bits = survivors;
        while s_bits != 0 {
            let v = s_bits.trailing_zeros() as usize;
            scores[v] += s4_dot[v] * high_n[v].max(1e-6) * w_s4;
            s_bits &= s_bits - 1;
        }
    }

    // Full64b and L1Rabitq share the same ranking path now that stratum4
    // carries the folded high-dim tail (Step 3 bit-budget reclaim).
    let _ = matches!(cfg.strata_depth, StrataDepth::Full64b);
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

/// Scan a contiguous slice of blocks (parallelized across Rayon shards when `numa_sharding` is enabled).
pub fn scan_blocks(
    blocks: &[PaxBlock64],
    plan: &QueryPlan,
    cfg: &EngineConfig,
    time_lo: Option<u32>,
    time_hi: Option<u32>,
) -> Vec<ScoredHit> {
    let mut shortlist = Shortlist::new(cfg.shortlist_cand);
    if blocks.is_empty() {
        return Vec::new();
    }

    // Parallel shard path when numa_sharding is enabled and block count warrants it.
    const PAR_THRESHOLD: usize = 64;
    if cfg.numa_sharding && blocks.len() >= PAR_THRESHOLD {
        // Warm up initial threshold τ on the first few blocks so every parallel worker
        // starts with an effective Cauchy–Schwarz pruning bound even at shortlist 256..1024.
        let warmup = (cfg.shortlist_cand / VECTORS_PER_BLOCK + 1)
            .min(32)
            .min(blocks.len());
        for i in 0..warmup {
            scan_block(
                &blocks[i],
                i as u32,
                plan,
                cfg,
                &mut shortlist,
                time_lo,
                time_hi,
            );
        }
        let rest = &blocks[warmup..];
        let shards = (rayon::current_num_threads() * 2).max(8);
        let chunk_size = ((rest.len() + shards - 1) / shards).max(16);
        let floor_tau = shortlist.threshold();
        let partials: Vec<Shortlist> = rest
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(ci, chunk)| {
                let mut local = Shortlist::with_floor(cfg.shortlist_cand, floor_tau);
                let base_idx = (warmup + ci * chunk_size) as u32;
                for (j, block) in chunk.iter().enumerate() {
                    scan_block(
                        block,
                        base_idx + j as u32,
                        plan,
                        cfg,
                        &mut local,
                        time_lo,
                        time_hi,
                    );
                }
                local
            })
            .collect();

        for p in &partials {
            shortlist.merge(p);
        }
        return shortlist.into_sorted();
    }

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
