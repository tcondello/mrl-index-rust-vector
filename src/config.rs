//! Production tuning parameters (spec §5).

use serde::{Deserialize, Serialize};

use crate::geometry::SketchGeometry;

/// How the leading MRL prefix (64-d, 4 bitplanes) is coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PrefixCodec {
    /// Per-vector affine min–max on raw prefix coords (production default).
    #[default]
    MinMax4,
    /// EDEN-prod: center → normalize → structured Hadamard → Lloyd–Max (unbiased).
    EdenProd,
}

/// How the 128-d mid-band (stratum 1) is coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MidCodec {
    /// FWHT + 1-bit RaBitQ signs (production default).
    #[default]
    Rabitq,
    /// EDEN-prod b=1 on the mid-band (levels in stratum1; α packed in `meta.offset` when
    /// prefix is also EDEN).
    EdenB1,
}

/// How many bitplane strata are evaluated during scan ranking.
///
/// Encode always packs the full `PaxBlock64`; this enum only gates which
/// score / prune stages run at query time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StrataDepth {
    /// Sign plane only (MSB of the 4-bit MRL prefix) + loose residual CS.
    L0Only,
    /// 4-bit MRL prefix + RaBitQ mid-band (+ post-S1 CS). Skips folded S4.
    L1Rabitq,
    /// Full production sketch: prefix + RaBitQ + folded S4 (+ S4 abs-budget prune).
    ///
    /// Default for Gemma@768 quality (matches the historical `l1_rabitq` bench
    /// path, which already scored S4 despite the name).
    #[default]
    Full64b,
}

impl StrataDepth {
    /// Whether the 4-bit MRL prefix (strata 0/2/3/5) is scored.
    #[inline]
    pub fn includes_prefix_4bit(self) -> bool {
        !matches!(self, Self::L0Only)
    }

    /// Whether RaBitQ mid-band IP + post-S1 CS run.
    #[inline]
    pub fn includes_rabitq(self) -> bool {
        matches!(self, Self::L1Rabitq | Self::Full64b)
    }

    /// Whether folded-tail S4 abs-budget prune + S4 IP run.
    #[inline]
    pub fn includes_tail_fold(self) -> bool {
        matches!(self, Self::Full64b)
    }
}

/// Page-cache / DRAM residency strategy for segment slabs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SlabResidency {
    RamLocked,
    #[default]
    MmapWillneed,
    DirectNvme,
}

/// Scan-time policy knobs (subset of [`EngineConfig`] consumed by the scan kernel).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScanConfig {
    pub shortlist_cand: usize,
    pub cs_tolerance: f32,
    pub stratum4_weight: f32,
    pub strata_depth: StrataDepth,
    pub temporal_filter_enable: bool,
    pub numa_sharding: bool,
}

impl ScanConfig {
    pub fn from_engine(cfg: &EngineConfig) -> Self {
        Self {
            shortlist_cand: cfg.shortlist_cand,
            cs_tolerance: cfg.cs_tolerance,
            stratum4_weight: cfg.stratum4_weight,
            strata_depth: cfg.strata_depth,
            temporal_filter_enable: cfg.temporal_filter_enable,
            numa_sharding: cfg.numa_sharding,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(16..=4096).contains(&self.shortlist_cand) {
            return Err("shortlist_cand must be in [16, 4096]".into());
        }
        if !(0.80..=1.20).contains(&self.cs_tolerance) {
            return Err("cs_tolerance must be in [0.80, 1.20]".into());
        }
        if !(0.05..=1.50).contains(&self.stratum4_weight) {
            return Err("stratum4_weight must be in [0.05, 1.50]".into());
        }
        Ok(())
    }
}

impl From<&EngineConfig> for ScanConfig {
    fn from(cfg: &EngineConfig) -> Self {
        Self::from_engine(cfg)
    }
}

/// Full engine configuration across the four tuning axes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    // Axis 1 — Memory & Storage
    pub strata_depth: StrataDepth,
    pub slab_residency: SlabResidency,
    pub numa_sharding: bool,

    // Axis 2 — Pruning & Recall
    pub shortlist_cand: usize,
    pub cs_tolerance: f32,
    /// Query-side Stratum-4 shrinkage weight (default `0.20` from Stage 2 sweep;
    /// pure MMSE `1/√B ≈ 0.333` is available via `mmse_stratum4_weight`).
    /// Encoded `PaxBlock64` bits are invariant to positive fold scaling, so this
    /// can be swept without re-encoding.
    pub stratum4_weight: f32,
    pub rabitq_confidence: f32,

    // Axis 3 — Latency & Batch
    pub jit_batch_size: usize,
    pub io_read_workers: usize,
    pub max_latency_budget_ms: u64,

    // Axis 4 — Ingestion & Lifecycle
    pub ring_buffer_size: usize,
    pub tombstone_compact_ratio: f32,
    pub temporal_filter_enable: bool,

    // Embedding geometry (fixed for a given index)
    pub embed_dims: usize,
    pub mrl_prefix: usize,
    pub final_top_k: usize,
    /// Opt-in flag to use local `google/embeddinggemma-2` (768-d) instead of the default demo `HashingEmbedder`.
    #[serde(default)]
    pub use_gemma2: bool,
    /// Prefix codebook (min–max vs EDEN-prod). Default keeps historical bits.
    #[serde(default)]
    pub prefix_codec: PrefixCodec,
    /// Structured-Hadamard seed when `prefix_codec == EdenProd`.
    #[serde(default = "default_eden_seed")]
    pub eden_seed: u64,
    /// Mid-band codebook. `EdenB1` is experimental and expects EDEN prefix packing.
    #[serde(default)]
    pub mid_codec: MidCodec,
}

fn default_eden_seed() -> u64 {
    1
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            // Full64b matches the historical production path (S4 scored under the
            // old `l1_rabitq` label). L1Rabitq is now the true S1-without-S4 ablation.
            strata_depth: StrataDepth::Full64b,
            slab_residency: SlabResidency::MmapWillneed,
            numa_sharding: true,
            shortlist_cand: 64,
            cs_tolerance: 1.0,
            stratum4_weight: crate::rotate::DEFAULT_STRATUM4_WEIGHT,
            rabitq_confidence: 2.576,
            jit_batch_size: 64,
            io_read_workers: 4,
            max_latency_budget_ms: 100,
            ring_buffer_size: 1024,
            tombstone_compact_ratio: 0.25,
            temporal_filter_enable: true,
            embed_dims: 384,
            mrl_prefix: 64,
            final_top_k: 10,
            use_gemma2: false,
            // EDEN-prod prefix b=4 won the 1M MS MARCO probe (+0.25pp R@10).
            // Mid-band stays RaBitQ (dual EDEN mid regressed on 1M).
            prefix_codec: PrefixCodec::EdenProd,
            eden_seed: 1,
            mid_codec: MidCodec::Rabitq,
        }
    }
}

impl EngineConfig {
    pub fn validate(&self) -> Result<(), String> {
        ScanConfig::from_engine(self).validate()?;
        self.geometry().validate()?;
        if self.use_gemma2 && self.embed_dims != 768 {
            return Err("use_gemma2 (--gemma2) requires embed_dims == 768".into());
        }
        if self.mrl_prefix != 64 {
            return Err("mrl_prefix must be 64 for PaxBlock64 layout".into());
        }
        if self.ring_buffer_size < 64 {
            return Err("ring_buffer_size must be >= 64".into());
        }
        Ok(())
    }

    /// Sketch geometry implied by `embed_dims` / `mrl_prefix`.
    pub fn geometry(&self) -> SketchGeometry {
        let mut g = SketchGeometry::for_embed_dims(self.embed_dims);
        g.mrl_prefix = self.mrl_prefix;
        g
    }

    /// Scan-kernel view of this config.
    pub fn scan_config(&self) -> ScanConfig {
        ScanConfig::from_engine(self)
    }

    /// Strata ids touched by the scan kernel for this depth.
    ///
    /// `0/2/3/5` = 4-bit prefix planes, `1` = RaBitQ, `4` = folded tail.
    pub fn active_strata(&self) -> &'static [u8] {
        match self.strata_depth {
            StrataDepth::L0Only => &[0],
            StrataDepth::L1Rabitq => &[0, 1, 2, 3, 5],
            StrataDepth::Full64b => &[0, 1, 2, 3, 4, 5],
        }
    }
}
