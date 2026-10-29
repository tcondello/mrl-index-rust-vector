//! Append-only ingestion ring buffer with 64-vector flush (spec §1, §5).

use crate::block::{source, PaxBlock64, VECTORS_PER_BLOCK};
use crate::embed::dot;
use crate::quantize::encode_block;

#[derive(Debug, Clone)]
pub struct PendingVector {
    pub passage_id: u64,
    pub embedding: Vec<f32>,
    pub timestamp: u32,
    pub source: u32,
    pub text: String,
}

#[derive(Debug)]
pub struct RingBuffer {
    capacity: usize,
    /// Hot unflushed vectors (queryable with 0 ms build latency).
    pub(crate) pending: Vec<PendingVector>,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(VECTORS_PER_BLOCK),
            pending: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Instant availability: vector is queryable from the hot ring before flush.
    pub fn push(&mut self, item: PendingVector) -> Option<PaxBlock64> {
        self.pending.push(item);
        if self.pending.len() >= VECTORS_PER_BLOCK {
            Some(self.flush_block())
        } else {
            None
        }
    }

    /// Force-flush a partial block (pads with tombstones).
    pub fn flush_partial(&mut self) -> Option<PaxBlock64> {
        if self.pending.is_empty() {
            None
        } else {
            Some(self.flush_block())
        }
    }

    fn flush_block(&mut self) -> PaxBlock64 {
        let take = self.pending.len().min(VECTORS_PER_BLOCK);
        let batch: Vec<PendingVector> = self.pending.drain(..take).collect();
        let dims = batch[0].embedding.len();
        let refs: Vec<&[f32]> = batch.iter().map(|p| p.embedding.as_slice()).collect();
        let timestamps: Vec<u32> = batch.iter().map(|p| p.timestamp).collect();
        let sources: Vec<u32> = batch
            .iter()
            .map(|p| if p.source == 0 { source::DOC } else { p.source })
            .collect();
        encode_block(&refs, &timestamps, &sources, dims)
    }

    /// Brute-force scan of unflushed embeddings (exact dots) for 0 ms build latency.
    pub fn search_exact(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32, &PendingVector)> {
        let mut scored: Vec<(usize, f32, &PendingVector)> = self
            .pending
            .iter()
            .enumerate()
            .map(|(i, p)| (i, dot(&p.embedding, query), p))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_k);
        scored
    }
}
