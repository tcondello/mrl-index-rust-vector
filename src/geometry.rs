//! First-class sketch geometry for PaxBlock64 encode / plan / scan.
//!
//! Layout constants still live in [`crate::block`]; this type groups them so
//! query planning and scan kernels do not hard-code `64/128/192/768/B=9`.

use crate::block::{
    EMBED_DIMS_768, MRL_PREFIX_DIMS, RABITQ_DIMS, TAIL_FOLD_BLOCKS, TAIL_FOLD_DIMS, TAIL_START,
};
use crate::rotate::DEFAULT_STRATUM4_WEIGHT;

/// Embedding + bitplane sketch geometry for one index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SketchGeometry {
    pub embed_dims: usize,
    pub mrl_prefix: usize,
    pub rabitq_dims: usize,
    pub tail_start: usize,
    pub tail_fold_dims: usize,
    /// Max Rademacher blocks folded into stratum-4 (`(embed_dims - tail_start) / 64`).
    pub tail_fold_blocks: usize,
}

impl SketchGeometry {
    /// Canonical EmbeddingGemma-2 @768 PaxBlock64 geometry.
    pub const fn gemma768() -> Self {
        Self {
            embed_dims: EMBED_DIMS_768,
            mrl_prefix: MRL_PREFIX_DIMS,
            rabitq_dims: RABITQ_DIMS,
            tail_start: TAIL_START,
            tail_fold_dims: TAIL_FOLD_DIMS,
            tail_fold_blocks: TAIL_FOLD_BLOCKS,
        }
    }

    /// Geometry for an arbitrary embed width (≥ prefix + RaBitQ).
    pub fn for_embed_dims(embed_dims: usize) -> Self {
        let available_tail = embed_dims.saturating_sub(TAIL_START);
        let tail_fold_blocks = (available_tail / MRL_PREFIX_DIMS).min(TAIL_FOLD_BLOCKS);
        Self {
            embed_dims,
            mrl_prefix: MRL_PREFIX_DIMS,
            rabitq_dims: RABITQ_DIMS,
            tail_start: TAIL_START,
            tail_fold_dims: TAIL_FOLD_DIMS,
            tail_fold_blocks,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.mrl_prefix != MRL_PREFIX_DIMS {
            return Err(format!(
                "mrl_prefix must be {MRL_PREFIX_DIMS} for PaxBlock64 layout"
            ));
        }
        if self.rabitq_dims != RABITQ_DIMS {
            return Err(format!(
                "rabitq_dims must be {RABITQ_DIMS} for PaxBlock64 layout"
            ));
        }
        if self.tail_start != TAIL_START {
            return Err(format!(
                "tail_start must be {TAIL_START} for PaxBlock64 layout"
            ));
        }
        if self.tail_fold_dims != TAIL_FOLD_DIMS {
            return Err(format!(
                "tail_fold_dims must be {TAIL_FOLD_DIMS} for PaxBlock64 layout"
            ));
        }
        if self.embed_dims < self.tail_start {
            return Err(format!(
                "embed_dims ({}) must be >= tail_start ({})",
                self.embed_dims, self.tail_start
            ));
        }
        let max_blocks = (self.embed_dims - self.tail_start) / MRL_PREFIX_DIMS;
        if self.tail_fold_blocks > max_blocks.min(TAIL_FOLD_BLOCKS) {
            return Err(format!(
                "tail_fold_blocks {} exceeds available {} for embed_dims={}",
                self.tail_fold_blocks, max_blocks, self.embed_dims
            ));
        }
        Ok(())
    }

    /// Locked Stage-2 default when geometry is Gemma@768; otherwise MMSE `1/√B`.
    pub fn default_stratum4_weight(&self) -> f32 {
        if self.embed_dims == EMBED_DIMS_768 && self.tail_fold_blocks == TAIL_FOLD_BLOCKS {
            DEFAULT_STRATUM4_WEIGHT
        } else if self.tail_fold_blocks == 0 {
            1.0
        } else {
            crate::rotate::mmse_stratum4_weight(self.tail_fold_blocks)
        }
    }

    pub fn has_tail_fold(&self) -> bool {
        self.embed_dims > self.tail_start && self.tail_fold_blocks > 0
    }
}

impl Default for SketchGeometry {
    fn default() -> Self {
        Self::gemma768()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma768_matches_block_consts() {
        let g = SketchGeometry::gemma768();
        g.validate().unwrap();
        assert_eq!(g.embed_dims, 768);
        assert_eq!(g.tail_start, 192);
        assert_eq!(g.tail_fold_blocks, 9);
        assert!((g.default_stratum4_weight() - DEFAULT_STRATUM4_WEIGHT).abs() < 1e-6);
    }

    #[test]
    fn for_embed_dims_384_has_three_fold_blocks() {
        let g = SketchGeometry::for_embed_dims(384);
        g.validate().unwrap();
        assert_eq!(g.tail_fold_blocks, 3); // (384-192)/64
        assert!(g.has_tail_fold());
    }
}
