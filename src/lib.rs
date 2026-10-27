//! **mrl-index** — local-first bitplane-transposed vector retrieval engine.
//!
//! Architecture (v3.5):
//! - 4 KB `PaxBlock64` macro-blocks (64 B/vec) with progressive bitplane strata
//! - Vertical POPCNT-friendly scans + Cauchy–Schwarz pruning
//! - Zero full-precision L2 store on disk; JIT batch rerank from `passages.jsonl`
//! - Append-only ring buffer for 0 ms ingest visibility

#![allow(clippy::needless_range_loop)]

pub mod benchmark;
pub mod block;
pub mod config;
pub mod corpus;
pub mod embed;
pub mod engine;
pub mod passage;
pub mod quantize;
pub mod rerank;
pub mod ring;
pub mod rotate;
pub mod scan;
pub mod segment;

pub use block::{PaxBlock64, VectorId, VECTORS_PER_BLOCK, BLOCK_BYTES};
pub use config::{EngineConfig, SlabResidency, StrataDepth};
pub use embed::HashingEmbedder;
pub use engine::{EngineError, IndexEngine, QueryResult, QueryStats};
pub use passage::{PassageIndexEntry, PassageRecord, PassageStore};
pub use rerank::{Embedder, RerankHit};
