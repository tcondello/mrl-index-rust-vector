//! End-to-end mrl-index engine: ingest → PaxBlock64 → scan → fetch → rerank.

use std::path::{Path, PathBuf};
use std::time::Instant;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::block::{source, VectorId, VECTORS_PER_BLOCK};
use crate::config::EngineConfig;
use crate::embed::HashingEmbedder;
use crate::passage::{PassageRecord, PassageStore};
use crate::rerank::{rerank_with_query_vec, Embedder, RerankHit};
use crate::ring::{PendingVector, RingBuffer};
use crate::scan::{scan_block, scan_blocks, QueryPlan, Shortlist};
use crate::segment::{SegmentReader, SegmentWriter};

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("config: {0}")]
    Config(String),
    #[error(transparent)]
    Passage(#[from] crate::passage::PassageError),
    #[error(transparent)]
    Segment(#[from] crate::segment::SegmentError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexManifest {
    version: String,
    config: EngineConfig,
    next_passage_id: u64,
    blocks_written: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryStats {
    pub embed_ms: f64,
    pub scan_ms: f64,
    pub fetch_ms: f64,
    pub rerank_ms: f64,
    pub total_ms: f64,
    pub blocks_scanned: usize,
    pub shortlist: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub hits: Vec<RerankHit>,
    pub stats: QueryStats,
}

pub struct IndexEngine {
    dir: PathBuf,
    config: EngineConfig,
    passages: PassageStore,
    segment_writer: Mutex<SegmentWriter>,
    segment_reader: SegmentReader,
    ring: Mutex<RingBuffer>,
    /// Maps (block, slot) → passage_id for flushed vectors.
    id_map: Mutex<Vec<u64>>,
    embedder: Box<dyn Embedder>,
}

impl IndexEngine {
    pub fn create(dir: impl AsRef<Path>, config: EngineConfig) -> Result<Self, EngineError> {
        config
            .validate()
            .map_err(EngineError::Config)?;
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let passages = PassageStore::open(&dir)?;
        let seg_path = dir.join("segments").join("000001.pax");
        let segment_writer = SegmentWriter::create(&seg_path)?;
        let segment_reader = SegmentReader::open(&seg_path, config.slab_residency)?;
        let ring = RingBuffer::new(config.ring_buffer_size);
        let embedder = Box::new(HashingEmbedder::new(config.embed_dims));

        let engine = Self {
            dir: dir.clone(),
            config: config.clone(),
            passages,
            segment_writer: Mutex::new(segment_writer),
            segment_reader,
            ring: Mutex::new(ring),
            id_map: Mutex::new(Vec::new()),
            embedder,
        };
        engine.write_manifest(0)?;
        Ok(engine)
    }

    pub fn open(dir: impl AsRef<Path>) -> Result<Self, EngineError> {
        let dir = dir.as_ref().to_path_buf();
        let manifest: IndexManifest =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
        let config = manifest.config;
        config
            .validate()
            .map_err(EngineError::Config)?;

        let passages = PassageStore::open(&dir)?;
        let seg_path = dir.join("segments").join("000001.pax");
        let segment_writer = SegmentWriter::open_append(&seg_path)?;
        let segment_reader = SegmentReader::open(&seg_path, config.slab_residency)?;
        let ring = RingBuffer::new(config.ring_buffer_size);
        let embedder = Box::new(HashingEmbedder::new(config.embed_dims));

        // Rebuild id_map from live slots: passages are assigned in flush order.
        let mut id_map =
            Vec::with_capacity(segment_reader.block_count() * VECTORS_PER_BLOCK);
        let mut next_pid = 0u64;
        for i in 0..segment_reader.block_count() {
            let block = segment_reader.block(i).expect("block in range");
            for slot in 0..VECTORS_PER_BLOCK {
                if block.is_tombstoned(slot) {
                    id_map.push(u64::MAX);
                } else {
                    id_map.push(next_pid);
                    next_pid += 1;
                }
            }
        }
        if next_pid > passages.len() {
            return Err(EngineError::Config(format!(
                "segment/passage mismatch: {next_pid} live slots vs {} passages",
                passages.len()
            )));
        }

        Ok(Self {
            dir,
            config,
            passages,
            segment_writer: Mutex::new(segment_writer),
            segment_reader,
            ring: Mutex::new(ring),
            id_map: Mutex::new(id_map),
            embedder,
        })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn passage_count(&self) -> u64 {
        self.passages.len()
    }

    pub fn block_count(&self) -> usize {
        self.segment_reader.block_count()
    }

    fn write_manifest(&self, blocks_written: u32) -> Result<(), EngineError> {
        let m = IndexManifest {
            version: "3.5.0".into(),
            config: self.config.clone(),
            next_passage_id: self.passages.len(),
            blocks_written,
        };
        std::fs::write(
            self.dir.join("manifest.json"),
            serde_json::to_string_pretty(&m)?,
        )?;
        Ok(())
    }

    /// Ingest a text passage. Instantly searchable via ring; flushes every 64.
    pub fn ingest(
        &mut self,
        text: impl Into<String>,
        timestamp: u32,
        source_mask: u32,
    ) -> Result<u64, EngineError> {
        let text = text.into();
        let src = if source_mask == 0 {
            source::DOC
        } else {
            source_mask
        };
        let embeds = self.embedder.embed(&[text.clone()]);
        let embedding = embeds.into_iter().next().unwrap();
        let tokens = crate::embed::tokenize_u16(&text);

        let id = self.passages.append_with_tokens(
            PassageRecord {
                id: 0,
                text: text.clone(),
                timestamp,
                source: src,
                meta: serde_json::json!({}),
                tokens: Vec::new(),
            },
            &tokens,
        )?;

        let pending = PendingVector::new(id, embedding, timestamp, src, text);

        let maybe_block = {
            let mut ring = self.ring.lock();
            ring.push(pending)
        };

        if let Some(block) = maybe_block {
            self.commit_block_inner(block, true)?;
        }
        Ok(id)
    }

    pub fn ingest_batch(
        &mut self,
        items: &[(String, u32, u32)],
    ) -> Result<Vec<u64>, EngineError> {
        let mut ids = Vec::with_capacity(items.len());
        let batch_sz = self.config.jit_batch_size.max(64);
        let mut committed_any = false;

        for chunk in items.chunks(batch_sz) {
            let texts: Vec<String> = chunk.iter().map(|(t, _, _)| t.clone()).collect();
            let embeds = self.embedder.embed(&texts);
            for ((text, ts, src_raw), embedding) in chunk.iter().zip(embeds.into_iter()) {
                let src = if *src_raw == 0 {
                    source::DOC
                } else {
                    *src_raw
                };
                let tokens = crate::embed::tokenize_u16(text);
                let id = self.passages.append_with_tokens(
                    PassageRecord {
                        id: 0,
                        text: text.clone(),
                        timestamp: *ts,
                        source: src,
                        meta: serde_json::Value::Null,
                        tokens: Vec::new(),
                    },
                    &tokens,
                )?;
                ids.push(id);

                let pending = PendingVector::new(id, embedding, *ts, src, text.clone());
                let maybe_block = {
                    let mut ring = self.ring.lock();
                    ring.push(pending)
                };
                if let Some(block) = maybe_block {
                    self.commit_block_inner(block, false)?;
                    committed_any = true;
                }
            }
        }

        if committed_any {
            self.passages.flush()?;
            let blocks_written = self.segment_writer.lock().block_count();
            self.write_manifest(blocks_written)?;
        }
        Ok(ids)
    }

    fn commit_block_inner(
        &mut self,
        block: crate::block::PaxBlock64,
        sync_metadata: bool,
    ) -> Result<(), EngineError> {
        let active = block.active_count() as usize;
        // Passage ids for this block are the last `active` passages written.
        let end = self.passages.len();
        let start = end - active as u64;
        {
            let mut map = self.id_map.lock();
            for pid in start..end {
                map.push(pid);
            }
            // Tombstoned slots still occupy layout positions — pad map to 64-aligned.
            while map.len() % VECTORS_PER_BLOCK != 0 {
                map.push(u64::MAX);
            }
        }

        let blocks_written = {
            let mut w = self.segment_writer.lock();
            w.append_block(&block)?;
            w.block_count()
        };

        // O(1) live append — do not remap/recopy the whole segment (was O(N²)).
        self.segment_reader.append_live(block);
        if sync_metadata {
            self.passages.flush()?;
            self.write_manifest(blocks_written)?;
        }
        Ok(())
    }

    /// Flush any partial ring contents into a tombstone-padded block.
    pub fn flush(&mut self) -> Result<(), EngineError> {
        let block = {
            let mut ring = self.ring.lock();
            ring.flush_partial()
        };
        if let Some(block) = block {
            self.commit_block_inner(block, true)?;
        } else {
            self.passages.flush()?;
            let blocks_written = self.segment_writer.lock().block_count();
            self.write_manifest(blocks_written)?;
        }
        Ok(())
    }

    pub fn query(&self, query_text: &str) -> Result<QueryResult, EngineError> {
        self.query_filtered(query_text, None, None)
    }

    pub fn query_filtered(
        &self,
        query_text: &str,
        time_lo: Option<u32>,
        time_hi: Option<u32>,
    ) -> Result<QueryResult, EngineError> {
        let t0 = Instant::now();

        let q_embed = self.embedder.embed(&[query_text.to_string()]);
        let query = &q_embed[0];
        let embed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t_scan = Instant::now();
        let plan = QueryPlan::from_query(query, self.config.embed_dims);
        let mut shortlist = Shortlist::new(self.config.shortlist_cand);
        let n_blocks = self.segment_reader.block_count();

        if let Some(blocks) = self.segment_reader.as_slice() {
            for hit in scan_blocks(blocks, &plan, &self.config, time_lo, time_hi) {
                shortlist.push(hit);
            }
        } else {
            for i in 0..n_blocks {
                if let Some(block) = self.segment_reader.block(i) {
                    scan_block(
                        block,
                        i as u32,
                        &plan,
                        &self.config,
                        &mut shortlist,
                        time_lo,
                        time_hi,
                    );
                }
            }
        }

        // Merge exact ring hits (0 ms index latency path).
        {
            let ring = self.ring.lock();
            for (local_i, score, pending) in ring.search_exact(query, self.config.shortlist_cand) {
                // Encode ring hits with high block id sentinel via passage id path later.
                shortlist.push(crate::scan::ScoredHit {
                    id: VectorId::new(u32::MAX, local_i as u16),
                    score,
                });
                let _ = pending; // used via ring index
            }
        }

        let scanned = shortlist.into_sorted();
        let scan_ms = t_scan.elapsed().as_secs_f64() * 1000.0;

        let t_fetch = Instant::now();
        let id_map = self.id_map.lock();
        let ring = self.ring.lock();

        let mut candidates: Vec<(u64, String, f32)> = Vec::with_capacity(scanned.len());
        let mut disk_pids: Vec<u64> = Vec::with_capacity(scanned.len());
        let mut disk_scores: Vec<f32> = Vec::with_capacity(scanned.len());

        for hit in &scanned {
            if hit.id.block == u32::MAX {
                if let Some(p) = ring.pending.get(hit.id.slot as usize) {
                    candidates.push((p.passage_id, p.text.clone(), hit.score));
                }
                continue;
            }
            let ord = hit.id.block as usize * VECTORS_PER_BLOCK + hit.id.slot as usize;
            let Some(&pid) = id_map.get(ord) else {
                continue;
            };
            if pid == u64::MAX {
                continue;
            }
            disk_pids.push(pid);
            disk_scores.push(hit.score);
        }
        drop(ring);
        drop(id_map);

        if !disk_pids.is_empty() {
            if let Ok(fetched) = self.passages.fetch_many(&disk_pids) {
                for ((pid, text), score) in fetched.into_iter().zip(disk_scores.into_iter()) {
                    candidates.push((pid, text, score));
                }
            }
        }
        let fetch_ms = t_fetch.elapsed().as_secs_f64() * 1000.0;

        let t_rr = Instant::now();
        let hits = rerank_with_query_vec(
            self.embedder.as_ref(),
            query,
            &candidates,
            self.config.final_top_k,
        );
        let rerank_ms = t_rr.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(QueryResult {
            hits,
            stats: QueryStats {
                embed_ms,
                scan_ms,
                fetch_ms,
                rerank_ms,
                total_ms,
                blocks_scanned: n_blocks,
                shortlist: candidates.len(),
            },
        })
    }
}
