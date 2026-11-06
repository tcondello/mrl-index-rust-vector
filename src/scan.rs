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
use crate::config::{EngineConfig, PrefixCodec, ScanConfig};
use crate::eden::{lloyd_max_normal, rotate_query, EdenFit};
use crate::rotate::{rabitq_bit_scale, rotate_mid_band, rotate_tail_fold, tail_fold_bit_scale};

/// Precomputed query side for bitplane scoring.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub dims: usize,
    pub prefix_codec: PrefixCodec,
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
    /// ‖q[64..192]‖₂ — mid-band energy for split residual CS.
    pub q_mid_norm: f32,
    /// ‖q[64:]‖₂ — remainder after leading 64 dims (loose single-term CS).
    pub q_after_prefix: f32,
    /// Alias of `q_tail_norm` (post-S1 residual energy).
    pub q_after_s1: f32,
    /// Σ_d |tail_fold_q[d] · bit_scale| — max |s4_dot| over bit assignments.
    pub s4_abs_budget: f32,
    /// Σ |q_d| over prefix — for bitplane residual slack.
    pub prefix_l1: f32,
    /// Σ q_d over prefix — for O(1) affine offset dot product.
    pub prefix_sum: f32,
    /// EDEN: R(q_prefix) in codebook domain.
    pub eden_q_rot: [f32; MRL_PREFIX_DIMS],
    /// EDEN: ⟨q_prefix, μ⟩.
    pub eden_mean_ip: f32,
    /// EDEN: 1/√d for Lloyd–Max codewords.
    pub eden_inv_sqrt_d: f32,
    /// EDEN: per-level centroid values (length 16 for b=4).
    pub eden_codebook: [f32; 16],
}

impl QueryPlan {
    pub fn from_query(query: &[f32], dims: usize) -> Self {
        Self::from_query_codec(query, dims, PrefixCodec::MinMax4, None)
    }

    pub fn from_query_eden(query: &[f32], dims: usize, fit: &EdenFit) -> Self {
        Self::from_query_codec(query, dims, PrefixCodec::EdenProd, Some(fit))
    }

    pub fn from_query_codec(
        query: &[f32],
        dims: usize,
        prefix_codec: PrefixCodec,
        eden: Option<&EdenFit>,
    ) -> Self {
        assert_eq!(query.len(), dims);
        assert!(dims >= MRL_PREFIX_DIMS + RABITQ_DIMS);
        if prefix_codec == PrefixCodec::EdenProd {
            let fit = eden.expect("EdenProd requires EdenFit");
            assert_eq!(fit.dim, MRL_PREFIX_DIMS);
            assert_eq!(fit.bits, 4);
        }

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

        let mut eden_q_rot = [0.0f32; MRL_PREFIX_DIMS];
        let mut eden_mean_ip = 0.0f32;
        let mut eden_inv_sqrt_d = 1.0 / (MRL_PREFIX_DIMS as f32).sqrt();
        let mut eden_codebook = [0.0f32; 16];
        if let Some(fit) = eden {
            let rot = rotate_query(fit, query);
            eden_q_rot.copy_from_slice(&rot);
            for d in 0..MRL_PREFIX_DIMS {
                eden_mean_ip += query[d] * fit.mean[d];
            }
            eden_inv_sqrt_d = 1.0 / (fit.dim as f32).sqrt();
            let cb = lloyd_max_normal(fit.bits);
            for (i, &c) in cb.iter().enumerate() {
                eden_codebook[i] = c;
            }
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

        let mut q_mid_sq = 0.0f32;
        for i in 0..RABITQ_DIMS {
            q_mid_sq += query[MRL_PREFIX_DIMS + i].powi(2);
        }
        let q_mid_norm = q_mid_sq.sqrt();
        let q_after_prefix = (q_mid_sq + q_tail_sq).sqrt();

        let has_tail_fold = dims > TAIL_START;
        let tail_fold_q = if has_tail_fold {
            rotate_tail_fold(query)
        } else {
            [0.0f32; TAIL_FOLD_DIMS]
        };
        let tail_fold_scale = tail_fold_bit_scale();
        let mut tail_fold_weights2 = [0.0f32; TAIL_FOLD_DIMS];
        let mut tail_fold_base = 0.0f32;
        let mut s4_abs_budget = 0.0f32;
        if has_tail_fold {
            for d in 0..TAIL_FOLD_DIMS {
                let pos = tail_fold_q[d] * tail_fold_scale;
                tail_fold_base -= pos;
                tail_fold_weights2[d] = 2.0 * pos;
                s4_abs_budget += pos.abs();
            }
        }

        Self {
            dims,
            prefix_codec,
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
            q_mid_norm,
            q_after_prefix,
            q_after_s1: q_tail_norm,
            s4_abs_budget,
            prefix_l1,
            prefix_sum,
            eden_q_rot,
            eden_mean_ip,
            eden_inv_sqrt_d,
            eden_codebook,
        }
    }
}

/// Optional aggregate counters for diagnosing prune effectiveness.
#[derive(Debug, Clone, Default)]
pub struct ScanStats {
    pub blocks: u64,
    pub active_vecs: u64,
    pub after_prefix: u64,
    pub after_s1: u64,
    pub after_s4_budget: u64,
    pub pushed: u64,
}

impl ScanStats {
    pub fn merge(&mut self, other: &ScanStats) {
        self.blocks += other.blocks;
        self.active_vecs += other.active_vecs;
        self.after_prefix += other.after_prefix;
        self.after_s1 += other.after_s1;
        self.after_s4_budget += other.after_s4_budget;
        self.pushed += other.pushed;
    }

    pub fn print_summary(&self) {
        let a = self.active_vecs.max(1) as f64;
        println!(
            "scan-stats: blocks={} active={}  keep_prefix={:.2}%  keep_s1={:.2}%  keep_s4bud={:.2}%  pushed={:.2}%",
            self.blocks,
            self.active_vecs,
            100.0 * self.after_prefix as f64 / a,
            100.0 * self.after_s1 as f64 / a,
            100.0 * self.after_s4_budget as f64 / a,
            100.0 * self.pushed as f64 / a,
        );
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

/// Per-block mutable state shared across composable scan stages.
#[derive(Clone, Debug)]
pub struct BlockScanState {
    pub scores: [f32; VECTORS_PER_BLOCK],
    pub survivors: u64,
    pub mid_n: [f32; VECTORS_PER_BLOCK],
    pub high_n: [f32; VECTORS_PER_BLOCK],
    pub active: u64,
}

impl BlockScanState {
    pub fn new(active: u64) -> Self {
        Self {
            scores: [0.0; VECTORS_PER_BLOCK],
            survivors: 0,
            mid_n: [0.0; VECTORS_PER_BLOCK],
            high_n: [0.0; VECTORS_PER_BLOCK],
            active,
        }
    }
}

/// Immutable inputs for one block scan (stages borrow this).
#[derive(Clone, Copy, Debug)]
pub struct BlockScanCtx<'a> {
    pub block: &'a PaxBlock64,
    pub plan: &'a QueryPlan,
    pub scan: ScanConfig,
    pub tau: f32,
    pub alpha: f32,
}

/// Stage: SWAR 4-bit prefix IP + split residual CS prune.
///
/// `ub = prefix_ip + α (‖q_mid‖ ‖x_mid‖ + ‖q_tail‖ ‖x_tail‖)`.
pub fn stage_prefix_ip_and_cs_split(ctx: &BlockScanCtx<'_>, state: &mut BlockScanState) {
    let use_eden = ctx.plan.prefix_codec == PrefixCodec::EdenProd;
    let mut acc = [[0.0f32; 8]; 8];
    for d in 0..MRL_PREFIX_DIMS {
        let weight = if use_eden {
            ctx.plan.eden_q_rot[d] * ctx.plan.eden_inv_sqrt_d
        } else {
            ctx.plan.prefix[d]
        };
        if weight == 0.0 {
            continue;
        }
        let b0 = ctx.block.stratum5_lsb.dim_bits[d];
        let b1 = ctx.block.stratum3_msb2.dim_bits[d];
        let b2 = ctx.block.stratum2_msb1.dim_bits[d];
        let b3 = ctx.block.stratum0_sign.dim_bits[d];

        for s in 0..8 {
            let packed = ((b0 >> s) & SWAR_M8)
                | (((b1 >> s) & SWAR_M8) << 1)
                | (((b2 >> s) & SWAR_M8) << 2)
                | (((b3 >> s) & SWAR_M8) << 3);
            let bytes = packed.to_le_bytes();
            if use_eden {
                for k in 0..8 {
                    let level = bytes[k] as usize;
                    acc[s][k] += ctx.plan.eden_codebook[level] * weight;
                }
            } else {
                for k in 0..8 {
                    acc[s][k] += (bytes[k] as f32) * weight;
                }
            }
        }
    }

    let mut survivors = 0u64;
    for v in 0..VECTORS_PER_BLOCK {
        let bit = 1u64 << v;
        if state.active & bit == 0 {
            continue;
        }
        state.mid_n[v] = ctx.block.meta[v].mid_norm_f32().max(1e-6);
        state.high_n[v] = ctx.block.meta[v].high_tail_norm_f32().max(0.0);
        let scale = ctx.block.meta[v].scale_f32();
        let code_dot_v = acc[v & 7][v >> 3];
        let prefix_ip = if use_eden {
            // meta.scale holds α = ‖x−μ‖·S; offset unused.
            ctx.plan.eden_mean_ip + scale * code_dot_v
        } else {
            let offset = ctx.block.meta[v].offset_f32();
            code_dot_v * scale + offset * ctx.plan.prefix_sum
        };
        state.scores[v] = prefix_ip;

        let ub = prefix_ip
            + ctx.alpha
                * (ctx.plan.q_mid_norm * state.mid_n[v]
                    + ctx.plan.q_tail_norm * state.high_n[v]);
        if ub >= ctx.tau {
            survivors |= bit;
        }
    }
    state.survivors = survivors;
}

/// Stage: RaBitQ 1-bit mid-band IP for current survivors.
pub fn stage_rabitq_ip(ctx: &BlockScanCtx<'_>, state: &mut BlockScanState) {
    let mut s1_dot = [ctx.plan.rabitq_base; VECTORS_PER_BLOCK];
    for d in 0..RABITQ_DIMS {
        let mut bits = ctx.block.stratum1_rabitq.dim_bits[d] & state.survivors;
        let w = ctx.plan.rabitq_weights2[d];
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            s1_dot[v] += w;
            bits &= bits - 1;
        }
    }
    let mut s_bits = state.survivors;
    while s_bits != 0 {
        let v = s_bits.trailing_zeros() as usize;
        state.scores[v] += s1_dot[v] * state.mid_n[v];
        s_bits &= s_bits - 1;
    }
}

/// Stage: post-S1 true-residual CS prune (`dims ≥ 192`).
pub fn stage_cs_tail(ctx: &BlockScanCtx<'_>, state: &mut BlockScanState) {
    let mut survivors2 = 0u64;
    let mut check_bits = state.survivors;
    while check_bits != 0 {
        let v = check_bits.trailing_zeros() as usize;
        let ub = state.scores[v] + ctx.alpha * ctx.plan.q_after_s1 * state.high_n[v];
        if ub >= ctx.tau {
            survivors2 |= 1u64 << v;
        }
        check_bits &= check_bits - 1;
    }
    state.survivors = survivors2;
}

/// Stage: pre-S4 ranking budget prune (skip S4 bitplanes that cannot beat τ).
pub fn stage_s4_abs_budget(ctx: &BlockScanCtx<'_>, state: &mut BlockScanState) {
    if !ctx.scan.strata_depth.includes_tail_fold() {
        return;
    }
    if !(ctx.plan.has_tail_fold && ctx.plan.s4_abs_budget > 0.0) {
        return;
    }
    let w_s4 = ctx.scan.stratum4_weight;
    let mut survivors3 = 0u64;
    let mut bits = state.survivors;
    while bits != 0 {
        let v = bits.trailing_zeros() as usize;
        let ub = state.scores[v]
            + ctx.alpha * state.high_n[v].max(1e-6) * w_s4 * ctx.plan.s4_abs_budget;
        if ub >= ctx.tau {
            survivors3 |= 1u64 << v;
        }
        bits &= bits - 1;
    }
    state.survivors = survivors3;
}

/// Stage: folded-tail 1-bit IP into scores for survivors.
pub fn stage_tail_fold_ip(ctx: &BlockScanCtx<'_>, state: &mut BlockScanState) {
    if !ctx.scan.strata_depth.includes_tail_fold() {
        return;
    }
    if !ctx.plan.has_tail_fold {
        return;
    }
    let w_s4 = ctx.scan.stratum4_weight;
    let mut s4_dot = [ctx.plan.tail_fold_base; VECTORS_PER_BLOCK];
    for d in 0..TAIL_FOLD_DIMS {
        let mut bits = ctx.block.stratum4_mid.dim_bits[d] & state.survivors;
        let w = ctx.plan.tail_fold_weights2[d];
        while bits != 0 {
            let v = bits.trailing_zeros() as usize;
            s4_dot[v] += w;
            bits &= bits - 1;
        }
    }
    let mut s_bits = state.survivors;
    while s_bits != 0 {
        let v = s_bits.trailing_zeros() as usize;
        state.scores[v] += s4_dot[v] * state.high_n[v].max(1e-6) * w_s4;
        s_bits &= s_bits - 1;
    }
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

/// Default Gemma@768 scan kernel: compose score + prune stages gated by [`StrataDepth`].
fn run_default_scan_kernel(
    ctx: &BlockScanCtx<'_>,
    state: &mut BlockScanState,
    stats: &mut Option<&mut ScanStats>,
) {
    stage_prefix_ip_and_cs_split(ctx, state);
    if let Some(s) = stats.as_deref_mut() {
        s.after_prefix += state.survivors.count_ones() as u64;
    }
    if state.survivors == 0 {
        return;
    }

    if ctx.scan.strata_depth.includes_rabitq() {
        stage_rabitq_ip(ctx, state);
        stage_cs_tail(ctx, state);
    }
    if let Some(s) = stats.as_deref_mut() {
        s.after_s1 += state.survivors.count_ones() as u64;
    }
    if state.survivors == 0 {
        return;
    }

    if ctx.scan.strata_depth.includes_tail_fold() {
        stage_s4_abs_budget(ctx, state);
        if let Some(s) = stats.as_deref_mut() {
            s.after_s4_budget += state.survivors.count_ones() as u64;
        }
        if state.survivors == 0 {
            return;
        }
        stage_tail_fold_ip(ctx, state);
    } else if let Some(s) = stats.as_deref_mut() {
        // No S4 stage: budget keep-rate == post-S1 keep-rate.
        s.after_s4_budget += state.survivors.count_ones() as u64;
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
    scan_block_stats(
        block, block_idx, plan, cfg, shortlist, time_lo, time_hi, None,
    );
}

/// Like [`scan_block`], optionally accumulating prune counters into `stats`.
pub fn scan_block_stats(
    block: &PaxBlock64,
    block_idx: u32,
    plan: &QueryPlan,
    cfg: &EngineConfig,
    shortlist: &mut Shortlist,
    time_lo: Option<u32>,
    time_hi: Option<u32>,
    mut stats: Option<&mut ScanStats>,
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

    let scan = cfg.scan_config();
    let tau = shortlist.threshold();
    let alpha = scan.cs_tolerance;
    let active = !block.header.tombstone_mask;
    if active == 0 {
        return;
    }
    if let Some(s) = stats.as_deref_mut() {
        s.blocks += 1;
        s.active_vecs += active.count_ones() as u64;
    }

    let mut state = BlockScanState::new(active);
    let ctx = BlockScanCtx {
        block,
        plan,
        scan,
        tau,
        alpha,
    };

    if !scan.strata_depth.includes_prefix_4bit() {
        let mut code_dot = [0.0f32; VECTORS_PER_BLOCK];
        for d in 0..MRL_PREFIX_DIMS {
            accum_plane(
                block.stratum0_sign.dim_bits[d],
                active,
                plan.bit_weights[d][3],
                &mut code_dot,
            );
        }
        let mut survivors = 0u64;
        for v in 0..VECTORS_PER_BLOCK {
            let bit = 1u64 << v;
            if active & bit == 0 {
                continue;
            }
            let scale = block.meta[v].scale_f32();
            let offset = block.meta[v].offset_f32();
            let prefix_ip = code_dot[v] * scale + offset * plan.prefix_sum;
            state.scores[v] = prefix_ip;
            let x_rest = block.meta[v].tail_norm_f32().max(0.0);
            let ub = prefix_ip + alpha * plan.q_after_prefix * x_rest;
            if ub >= tau {
                survivors |= bit;
            }
        }
        if survivors != 0 {
            if let Some(s) = stats.as_deref_mut() {
                let n = survivors.count_ones() as u64;
                s.after_prefix += n;
                s.after_s1 += n;
                s.after_s4_budget += n;
                s.pushed += n;
            }
            finish_push(block_idx, survivors, &state.scores, shortlist);
        }
        return;
    }

    run_default_scan_kernel(&ctx, &mut state, &mut stats);
    if state.survivors != 0 {
        if let Some(s) = stats.as_deref_mut() {
            s.pushed += state.survivors.count_ones() as u64;
        }
        finish_push(block_idx, state.survivors, &state.scores, shortlist);
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
    scan_blocks_stats(blocks, plan, cfg, time_lo, time_hi, None)
}

/// Like [`scan_blocks`], optionally accumulating prune counters (serial path).
pub fn scan_blocks_stats(
    blocks: &[PaxBlock64],
    plan: &QueryPlan,
    cfg: &EngineConfig,
    time_lo: Option<u32>,
    time_hi: Option<u32>,
    mut stats: Option<&mut ScanStats>,
) -> Vec<ScoredHit> {
    let mut shortlist = Shortlist::new(cfg.shortlist_cand);
    if blocks.is_empty() {
        return Vec::new();
    }

    const PAR_THRESHOLD: usize = 64;
    if cfg.numa_sharding && blocks.len() >= PAR_THRESHOLD && stats.is_none() {
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
        scan_block_stats(
            block,
            i as u32,
            plan,
            cfg,
            &mut shortlist,
            time_lo,
            time_hi,
            stats.as_deref_mut(),
        );
    }
    shortlist.into_sorted()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::encode_block;
    use crate::StrataDepth;

    fn l2_normalize(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= n;
        }
    }

    fn synth_block(seed: u32, n: usize, dims: usize) -> (PaxBlock64, Vec<Vec<f32>>) {
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = vec![0.0f32; dims];
            for d in 0..dims {
                let x = ((seed.wrapping_mul(1103515245).wrapping_add(12345)
                    .wrapping_add((i as u32).wrapping_mul(9973))
                    .wrapping_add((d as u32).wrapping_mul(7919)))
                    % 10_007) as f32
                    / 5003.5
                    - 1.0;
                v[d] = x;
            }
            l2_normalize(&mut v);
            rows.push(v);
        }
        let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        let ts: Vec<u32> = (0..n as u32).collect();
        let src: Vec<u32> = vec![1; n];
        let block = encode_block(&refs, &ts, &src, dims);
        (block, rows)
    }

    #[test]
    fn default_kernel_stages_narrow_monotonically() {
        let dims = 768;
        let (block, rows) = synth_block(42, 64, dims);
        let plan = QueryPlan::from_query(&rows[0], dims);
        let cfg = EngineConfig::default();
        let tau = -1.0f32; // keep almost everything so stages still run
        let mut state = BlockScanState::new(!block.header.tombstone_mask);
        let ctx = BlockScanCtx {
            block: &block,
            plan: &plan,
            scan: cfg.scan_config(),
            tau,
            alpha: cfg.cs_tolerance,
        };

        stage_prefix_ip_and_cs_split(&ctx, &mut state);
        let after_prefix = state.survivors.count_ones();
        assert!(after_prefix > 0);

        stage_rabitq_ip(&ctx, &mut state);
        stage_cs_tail(&ctx, &mut state);
        let after_s1 = state.survivors.count_ones();
        assert!(after_s1 <= after_prefix);

        let before_s4 = state.survivors;
        stage_s4_abs_budget(&ctx, &mut state);
        assert_eq!(state.survivors & !before_s4, 0); // only narrows
        let after_budget = state.survivors.count_ones();
        assert!(after_budget <= after_s1);

        stage_tail_fold_ip(&ctx, &mut state);
        assert!(state.survivors.count_ones() > 0);
    }

    #[test]
    fn scan_blocks_deterministic_and_stats_consistent() {
        let dims = 768;
        let (block, rows) = synth_block(7, 32, dims);
        let plan = QueryPlan::from_query(&rows[3], dims);
        let mut cfg = EngineConfig::default();
        cfg.embed_dims = dims;
        cfg.shortlist_cand = 16;
        cfg.numa_sharding = false;

        let a = scan_blocks(&[block], &plan, &cfg, None, None);
        let b = scan_blocks(&[block], &plan, &cfg, None, None);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.score.to_bits(), y.score.to_bits());
        }

        let mut stats = ScanStats::default();
        let c = scan_blocks_stats(&[block], &plan, &cfg, None, None, Some(&mut stats));
        assert_eq!(a.len(), c.len());
        for (x, y) in a.iter().zip(c.iter()) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.score.to_bits(), y.score.to_bits());
        }
        assert_eq!(stats.blocks, 1);
        assert!(stats.active_vecs >= 32);
        assert!(stats.after_prefix <= stats.active_vecs);
        assert!(stats.after_s1 <= stats.after_prefix);
        assert!(stats.after_s4_budget <= stats.after_s1);
        assert!(stats.pushed <= stats.after_s4_budget);
        // `pushed` counts survivors offered to the shortlist; capacity may truncate.
        assert!(c.len() <= cfg.shortlist_cand);
        assert!(c.len() as u64 <= stats.pushed);
    }

    #[test]
    fn s4_abs_budget_is_positive_for_768d_query() {
        let mut q = vec![0.0f32; 768];
        for (i, x) in q.iter_mut().enumerate() {
            *x = ((i * 17) % 101) as f32 * 0.01 - 0.5;
        }
        l2_normalize(&mut q);
        let plan = QueryPlan::from_query(&q, 768);
        assert!(plan.has_tail_fold);
        assert!(plan.s4_abs_budget > 0.0);
        assert!(plan.q_mid_norm > 0.0);
        assert!(plan.q_tail_norm > 0.0);
        assert_eq!(plan.q_after_s1, plan.q_tail_norm);
    }

    #[test]
    fn strata_depth_gates_s4_scores() {
        let dims = 768;
        let (block, rows) = synth_block(99, 64, dims);
        let plan = QueryPlan::from_query(&rows[0], dims);

        let mut cfg_full = EngineConfig::default();
        cfg_full.embed_dims = dims;
        cfg_full.strata_depth = StrataDepth::Full64b;
        cfg_full.shortlist_cand = 64;
        cfg_full.numa_sharding = false;

        let mut cfg_l1 = cfg_full.clone();
        cfg_l1.strata_depth = StrataDepth::L1Rabitq;

        let full = scan_blocks(&[block], &plan, &cfg_full, None, None);
        let l1 = scan_blocks(&[block], &plan, &cfg_l1, None, None);
        assert!(!full.is_empty());
        assert_eq!(full.len(), l1.len());
        // Same survivor set order may differ if scores change; at least one
        // top hit score must differ when S4 contributes.
        let full_scores: Vec<u32> = full.iter().map(|h| h.score.to_bits()).collect();
        let l1_scores: Vec<u32> = l1.iter().map(|h| h.score.to_bits()).collect();
        assert_ne!(
            full_scores, l1_scores,
            "Full64b must apply S4; L1Rabitq must skip it"
        );
    }
}
