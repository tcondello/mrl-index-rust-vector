//! Binary length-prefixed passage store: `passages.bin` + packed `passages.idx` (spec §3.2).
//!
//! Layout per record in `passages.bin`:
//!   `[id: u64, timestamp: u32, source: u32, text_len: u32, meta_len: u32]` (24 B header)
//!   followed by `text_bytes[text_len]` (raw UTF-8) and `meta_bytes[meta_len]`.
//!
//! `PassageIndexEntry` stores `(text_offset: u64, text_len: u32)` pointing directly
//! at `text_bytes` so `fetch_text` performs a single zero-parse `pread` (`read_at`).

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::block::VectorId;

const RECORD_HEADER_BYTES: u64 = 24;
const AUTO_FLUSH_BYTES: usize = 256 * 1024;

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PassageIndexEntry {
    pub file_offset: u64,
    pub text_len: u32,
}

const _: () = assert!(std::mem::size_of::<PassageIndexEntry>() == 12);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassageRecord {
    pub id: u64,
    pub text: String,
    pub timestamp: u32,
    pub source: u32,
    #[serde(default)]
    pub meta: serde_json::Value,
}

#[derive(Debug, Error)]
pub enum PassageError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("utf8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("missing passage for vector {0:?}")]
    Missing(VectorId),
    #[error("corrupt index: entry {0}")]
    CorruptIndex(u64),
}

pub struct PassageStore {
    dir: PathBuf,
    data: File,
    idx: File,
    entries: Vec<PassageIndexEntry>,
    flushed_offset: u64,
    unflushed_data: Vec<u8>,
    unflushed_idx: Vec<u8>,
}

impl PassageStore {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, PassageError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let data_path = dir.join("passages.bin");
        let idx_path = dir.join("passages.idx");

        let data = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&data_path)?;
        let mut idx = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&idx_path)?;

        let flushed_offset = data.metadata()?.len();
        let mut idx_bytes = Vec::new();
        idx.seek(SeekFrom::Start(0))?;
        idx.read_to_end(&mut idx_bytes)?;
        let n_entries = idx_bytes.len() / 12;
        let mut entries = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let start = i * 12;
            let entry: PassageIndexEntry = *bytemuck::from_bytes(&idx_bytes[start..start + 12]);
            entries.push(entry);
        }

        Ok(Self {
            dir,
            data,
            idx,
            entries,
            flushed_offset,
            unflushed_data: Vec::with_capacity(64 * 1024),
            unflushed_idx: Vec::with_capacity(4 * 1024),
        })
    }

    pub fn path(&self) -> &Path {
        &self.dir
    }

    pub fn len(&self) -> u64 {
        self.entries.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Flush buffered passage and index records to the OS file descriptors.
    pub fn flush(&mut self) -> Result<(), PassageError> {
        if !self.unflushed_data.is_empty() {
            self.data.write_all(&self.unflushed_data)?;
            self.flushed_offset += self.unflushed_data.len() as u64;
            self.unflushed_data.clear();
        }
        if !self.unflushed_idx.is_empty() {
            self.idx.write_all(&self.unflushed_idx)?;
            self.unflushed_idx.clear();
        }
        Ok(())
    }

    /// Append a passage; returns the sequential passage id (= vector ordinal).
    pub fn append(&mut self, mut record: PassageRecord) -> Result<u64, PassageError> {
        let id = self.entries.len() as u64;
        record.id = id;

        let record_offset = self.flushed_offset + self.unflushed_data.len() as u64;
        let text_offset = record_offset + RECORD_HEADER_BYTES;
        let text_bytes = record.text.as_bytes();
        let text_len = text_bytes.len() as u32;

        let meta_empty = record.meta.is_null()
            || record
                .meta
                .as_object()
                .map_or(false, |obj| obj.is_empty());
        let meta_bytes = if meta_empty {
            Vec::new()
        } else {
            serde_json::to_vec(&record.meta)?
        };
        let meta_len = meta_bytes.len() as u32;

        self.unflushed_data.extend_from_slice(&id.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&record.timestamp.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&record.source.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&text_len.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&meta_len.to_le_bytes());
        self.unflushed_data.extend_from_slice(text_bytes);
        if !meta_bytes.is_empty() {
            self.unflushed_data.extend_from_slice(&meta_bytes);
        }

        let entry = PassageIndexEntry {
            file_offset: text_offset,
            text_len,
        };
        self.entries.push(entry);
        self.unflushed_idx
            .extend_from_slice(bytemuck::bytes_of(&entry));

        if self.unflushed_data.len() >= AUTO_FLUSH_BYTES {
            self.flush()?;
        }

        Ok(id)
    }

    #[inline]
    pub fn entry(&self, id: u64) -> Result<PassageIndexEntry, PassageError> {
        self.entries
            .get(id as usize)
            .copied()
            .ok_or(PassageError::CorruptIndex(id))
    }

    #[inline]
    fn read_range(&self, offset: u64, len: usize) -> Result<Vec<u8>, PassageError> {
        if offset >= self.flushed_offset {
            let local_start = (offset - self.flushed_offset) as usize;
            let local_end = local_start + len;
            if local_end <= self.unflushed_data.len() {
                return Ok(self.unflushed_data[local_start..local_end].to_vec());
            }
        }
        let mut buf = vec![0u8; len];
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.data.read_exact_at(&mut buf, offset)?;
        }
        #[cfg(not(unix))]
        {
            let mut file = &self.data;
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut buf)?;
        }
        Ok(buf)
    }

    pub fn fetch_text(&self, id: u64) -> Result<String, PassageError> {
        let entry = self.entry(id)?;
        let buf = self.read_range(entry.file_offset, entry.text_len as usize)?;
        Ok(String::from_utf8(buf)?)
    }

    pub fn fetch_record(&self, id: u64) -> Result<PassageRecord, PassageError> {
        let entry = self.entry(id)?;
        let record_offset = entry
            .file_offset
            .checked_sub(RECORD_HEADER_BYTES)
            .ok_or(PassageError::CorruptIndex(id))?;
        let header = self.read_range(record_offset, RECORD_HEADER_BYTES as usize)?;
        let rec_id = u64::from_le_bytes(header[0..8].try_into().unwrap());
        let timestamp = u32::from_le_bytes(header[8..12].try_into().unwrap());
        let source = u32::from_le_bytes(header[12..16].try_into().unwrap());
        let text_len = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
        let meta_len = u32::from_le_bytes(header[20..24].try_into().unwrap()) as usize;

        let payload = self.read_range(entry.file_offset, text_len + meta_len)?;
        let text = String::from_utf8(payload[..text_len].to_vec())?;
        let meta = if meta_len == 0 {
            serde_json::json!({})
        } else {
            serde_json::from_slice(&payload[text_len..text_len + meta_len])?
        };

        Ok(PassageRecord {
            id: rec_id,
            text,
            timestamp,
            source,
            meta,
        })
    }

    /// Batch fetch texts for shortlist vector ordinals, sorted by ascending file offset (LBA order).
    pub fn fetch_many(&self, ids: &[u64]) -> Result<Vec<(u64, String)>, PassageError> {
        let mut reqs: Vec<(usize, u64, PassageIndexEntry)> = Vec::with_capacity(ids.len());
        for (orig_idx, &id) in ids.iter().enumerate() {
            reqs.push((orig_idx, id, self.entry(id)?));
        }
        // Sort by ascending file offset for sequential NVMe / page-cache prefetching.
        reqs.sort_unstable_by_key(|r| r.2.file_offset);

        let mut results: Vec<(usize, u64, String)> = Vec::with_capacity(reqs.len());
        for (orig_idx, id, entry) in reqs {
            let buf = self.read_range(entry.file_offset, entry.text_len as usize)?;
            results.push((orig_idx, id, String::from_utf8(buf)?));
        }
        results.sort_unstable_by_key(|r| r.0);
        Ok(results.into_iter().map(|(_, id, text)| (id, text)).collect())
    }
}
