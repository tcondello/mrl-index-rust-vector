//! Production tuning parameters (spec §5).

use serde::{Deserialize, Serialize};

/// How many bitplane strata are materialised / scanned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StrataDepth {
    /// Sign plane only (~16 B/vec effective hot path).
    L0Only,
    /// Sign + RaBitQ 1-bit plane (default balanced).
    #[default]
    L1Rabitq,
    /// Full 64 B/vec PaxBlock64 strata.
    Full64b,
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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            strata_depth: StrataDepth::L1Rabitq,
            slab_residency: SlabResidency::MmapWillneed,
            numa_sharding: true,
            shortlist_cand: 64,
            cs_tolerance: 1.0,
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
        }
    }
}

impl EngineConfig {
    pub fn validate(&self) -> Result<(), String> {
        // Upper bound raised for large-corpus MRL (EmbeddingGemma @768): shortlist
        // 64–256 leaves substantial exact@10 on the table; 512–1024 recovers most of it.
        if !(16..=4096).contains(&self.shortlist_cand) {
            return Err("shortlist_cand must be in [16, 4096]".into());
        }
        if !(0.80..=1.20).contains(&self.cs_tolerance) {
            return Err("cs_tolerance must be in [0.80, 1.20]".into());
        }
        if self.embed_dims < 192 {
            return Err("embed_dims must be >= 192 (prefix 64 + mid 128)".into());
        }
        if self.mrl_prefix != 64 {
            return Err("mrl_prefix must be 64 for PaxBlock64 layout".into());
        }
        if self.ring_buffer_size < 64 {
            return Err("ring_buffer_size must be >= 64".into());
        }
        Ok(())
    }

    /// Strata actually evaluated during a scan given depth setting.
    pub fn active_strata(&self) -> &'static [u8] {
        match self.strata_depth {
            StrataDepth::L0Only => &[0],
            StrataDepth::L1Rabitq => &[0, 1],
            StrataDepth::Full64b => &[0, 1, 2, 3, 4, 5],
        }
    }
}
