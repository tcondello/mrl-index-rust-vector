//! Binary length-prefixed passage store: `passages.bin` + packed `passages.idx`.
//!
//! ## v2 layout (Step 5)
//! Record header (28 B):
//!   `[id: u64, timestamp: u32, source: u32, text_len: u32, meta_len: u32,
//!     token_len: u16, flags: u16]`
//! Payload: `text_bytes[text_len]` + `meta_bytes[meta_len]` + `tokens[token_len] as u16 LE`.
//!
//! `PassageIndexEntry` (24 B) stores text + optional token span offsets so
//! `fetch_tokens_many` can read compact `u16` IDs without touching UTF-8, while
//! `fetch_many` / `fetch_text` still serve raw text for the final `top_k` hits.
//!
//! Segment / token slabs may be padded to Apple Silicon's 16 KiB VM page size
//! via [`align_page`] / [`PAGE_ALIGN`].

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::block::VectorId;

/// Apple Silicon VM page size (`4 × PaxBlock64`).
pub const PAGE_ALIGN: usize = 16_384;

/// Round `n` up to the next multiple of [`PAGE_ALIGN`].
#[inline]
pub fn align_page(n: usize) -> usize {
    align(n, PAGE_ALIGN)
}

/// Generic power-of-two (or arbitrary) alignment helper.
#[inline]
pub fn align(n: usize, alignment: usize) -> usize {
    debug_assert!(alignment > 0);
    (n + alignment - 1) / alignment * alignment
}

const RECORD_HEADER_BYTES: u64 = 28;
const AUTO_FLUSH_BYTES: usize = 256 * 1024;
const INDEX_ENTRY_BYTES: usize = 24;

/// Flag bit: record carries a non-empty token span.
pub const FLAG_HAS_TOKENS: u16 = 1;
/// Flag bit: token span is stored as little-endian `u32` (for Gemma-2 262,144 vocab).
pub const FLAG_TOKENS_U32: u16 = 2;

/// High bit in `PassageIndexEntry::token_len` indicating 4-byte `u32` token storage.
const TOKEN_LEN_U32_BIT: u32 = 1 << 31;
const TOKEN_LEN_MASK: u32 = 0x7fff_ffff;

/// Contiguous ragged `u32` token batch read in LBA order (`cu_seqlens` layout).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OwnedPackedTokenBatch {
    /// Passage IDs corresponding to each sequence `i` in `0..num_sequences()`.
    pub ids: Vec<u64>,
    /// Flattened 1D token stream across all sequences (zero `<pad>` tokens).
    pub tokens: Vec<u32>,
    /// Cumulative sequence offsets of length `ids.len() + 1`:
    /// sequence `i` spans `tokens[cu_seqlens[i] as usize .. cu_seqlens[i+1] as usize]`.
    pub cu_seqlens: Vec<u32>,
}

impl OwnedPackedTokenBatch {
    pub fn num_sequences(&self) -> usize {
        self.ids.len()
    }

    pub fn sequence(&self, i: usize) -> &[u32] {
        let start = self.cu_seqlens[i] as usize;
        let end = self.cu_seqlens[i + 1] as usize;
        &self.tokens[start..end]
    }

    pub fn as_view(&self) -> crate::rerank::PackedTokenBatch<'_> {
        crate::rerank::PackedTokenBatch {
            tokens: &self.tokens,
            cu_seqlens: &self.cu_seqlens,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PassageIndexEntry {
    /// Absolute file offset of UTF-8 text bytes.
    pub file_offset: u64,
    pub text_len: u32,
    /// Number of token IDs (low 31 bits); bit 31 = 1 if stored as `u32`, 0 if `u16`.
    pub token_len: u32,
    /// Absolute file offset of the token span (0 if `token_count() == 0`).
    pub token_offset: u64,
}

impl PassageIndexEntry {
    #[inline]
    pub fn token_count(self) -> usize {
        (self.token_len & TOKEN_LEN_MASK) as usize
    }

    #[inline]
    pub fn is_u32_tokens(self) -> bool {
        (self.token_len & TOKEN_LEN_U32_BIT) != 0
    }
}

const _: () = assert!(std::mem::size_of::<PassageIndexEntry>() == INDEX_ENTRY_BYTES);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassageRecord {
    pub id: u64,
    pub text: String,
    pub timestamp: u32,
    pub source: u32,
    #[serde(default)]
    pub meta: serde_json::Value,
    /// Optional pre-tokenized `u16` IDs (SentencePiece / BPE / hashing vocab).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tokens: Vec<u16>,
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
    #[error("corrupt token span for passage {0}")]
    CorruptTokens(u64),
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
        if idx_bytes.len() % INDEX_ENTRY_BYTES != 0 {
            return Err(PassageError::CorruptIndex(u64::MAX));
        }
        let n_entries = idx_bytes.len() / INDEX_ENTRY_BYTES;
        let mut entries = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let start = i * INDEX_ENTRY_BYTES;
            let entry: PassageIndexEntry =
                *bytemuck::from_bytes(&idx_bytes[start..start + INDEX_ENTRY_BYTES]);
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

    /// Pad the data file to the next 16 KiB boundary (segment / token slab helper).
    pub fn pad_to_page(&mut self) -> Result<(), PassageError> {
        self.flush()?;
        let cur = self.flushed_offset as usize;
        let target = align_page(cur);
        if target > cur {
            let pad = vec![0u8; target - cur];
            self.data.write_all(&pad)?;
            self.flushed_offset = target as u64;
        }
        Ok(())
    }

    /// Append a passage; returns the sequential passage id (= vector ordinal).
    pub fn append(&mut self, record: PassageRecord) -> Result<u64, PassageError> {
        let tokens = record.tokens.clone();
        self.append_with_tokens(record, &tokens)
    }

    /// Append with an explicit token span (preferred ingest path for Step 5).
    pub fn append_with_tokens(
        &mut self,
        mut record: PassageRecord,
        tokens: &[u16],
    ) -> Result<u64, PassageError> {
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
        let token_len = tokens.len() as u16;
        let flags = if token_len > 0 {
            FLAG_HAS_TOKENS
        } else {
            0
        };

        self.unflushed_data.extend_from_slice(&id.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&record.timestamp.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&record.source.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&text_len.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&meta_len.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&token_len.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&flags.to_le_bytes());
        self.unflushed_data.extend_from_slice(text_bytes);
        if !meta_bytes.is_empty() {
            self.unflushed_data.extend_from_slice(&meta_bytes);
        }
        let token_offset = if tokens.is_empty() {
            0
        } else {
            let off = self.flushed_offset + self.unflushed_data.len() as u64;
            for &t in tokens {
                self.unflushed_data.extend_from_slice(&t.to_le_bytes());
            }
            off
        };

        let entry = PassageIndexEntry {
            file_offset: text_offset,
            text_len,
            token_len: tokens.len() as u32,
            token_offset,
        };
        self.entries.push(entry);
        self.unflushed_idx
            .extend_from_slice(bytemuck::bytes_of(&entry));

        if self.unflushed_data.len() >= AUTO_FLUSH_BYTES {
            self.flush()?;
        }

        Ok(id)
    }

    /// Append with an explicit `u32` token span (supports Gemma-2's 262,144 SentencePiece vocabulary).
    pub fn append_with_tokens_u32(
        &mut self,
        mut record: PassageRecord,
        tokens: &[u32],
    ) -> Result<u64, PassageError> {
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
        let token_len_u16 = tokens.len().min(u16::MAX as usize) as u16;
        let flags = if !tokens.is_empty() {
            FLAG_HAS_TOKENS | FLAG_TOKENS_U32
        } else {
            0
        };

        self.unflushed_data.extend_from_slice(&id.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&record.timestamp.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&record.source.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&text_len.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&meta_len.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&token_len_u16.to_le_bytes());
        self.unflushed_data
            .extend_from_slice(&flags.to_le_bytes());
        self.unflushed_data.extend_from_slice(text_bytes);
        if !meta_bytes.is_empty() {
            self.unflushed_data.extend_from_slice(&meta_bytes);
        }
        let token_offset = if tokens.is_empty() {
            0
        } else {
            let off = self.flushed_offset + self.unflushed_data.len() as u64;
            for &t in tokens {
                self.unflushed_data.extend_from_slice(&t.to_le_bytes());
            }
            off
        };

        let encoded_token_len = if tokens.is_empty() {
            0
        } else {
            (tokens.len() as u32 & TOKEN_LEN_MASK) | TOKEN_LEN_U32_BIT
        };

        let entry = PassageIndexEntry {
            file_offset: text_offset,
            text_len,
            token_len: encoded_token_len,
            token_offset,
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

    /// Fetch pre-tokenized `u16` IDs for one passage (empty if none stored).
    pub fn fetch_tokens(&self, id: u64) -> Result<Vec<u16>, PassageError> {
        let entry = self.entry(id)?;
        let count = entry.token_count();
        if count == 0 {
            return Ok(Vec::new());
        }
        if entry.is_u32_tokens() {
            let nbytes = count * 4;
            let buf = self.read_range(entry.token_offset, nbytes)?;
            if buf.len() != nbytes {
                return Err(PassageError::CorruptTokens(id));
            }
            let mut out = Vec::with_capacity(count);
            for chunk in buf.chunks_exact(4) {
                let v = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push((v & 0xffff) as u16);
            }
            return Ok(out);
        }
        let nbytes = count * 2;
        let buf = self.read_range(entry.token_offset, nbytes)?;
        if buf.len() != nbytes {
            return Err(PassageError::CorruptTokens(id));
        }
        let mut out = Vec::with_capacity(count);
        for chunk in buf.chunks_exact(2) {
            out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        Ok(out)
    }

    /// Fetch pre-tokenized `u32` IDs for one passage (seamlessly promotes `u16` spans if stored as v2 `u16`).
    pub fn fetch_tokens_u32(&self, id: u64) -> Result<Vec<u32>, PassageError> {
        let entry = self.entry(id)?;
        self.read_entry_tokens_u32(id, entry)
    }

    #[inline]
    fn read_entry_tokens_u32(
        &self,
        id: u64,
        entry: PassageIndexEntry,
    ) -> Result<Vec<u32>, PassageError> {
        let count = entry.token_count();
        if count == 0 {
            return Ok(Vec::new());
        }
        if entry.is_u32_tokens() {
            let nbytes = count * 4;
            let buf = self.read_range(entry.token_offset, nbytes)?;
            if buf.len() != nbytes {
                return Err(PassageError::CorruptTokens(id));
            }
            let mut out = Vec::with_capacity(count);
            for chunk in buf.chunks_exact(4) {
                out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        } else {
            let nbytes = count * 2;
            let buf = self.read_range(entry.token_offset, nbytes)?;
            if buf.len() != nbytes {
                return Err(PassageError::CorruptTokens(id));
            }
            let mut out = Vec::with_capacity(count);
            for chunk in buf.chunks_exact(2) {
                out.push(u16::from_le_bytes([chunk[0], chunk[1]]) as u32);
            }
            Ok(out)
        }
    }

    /// Batch fetch token spans sorted by ascending `token_offset` (LBA order).
    pub fn fetch_tokens_many(&self, ids: &[u64]) -> Result<Vec<(u64, Vec<u16>)>, PassageError> {
        let mut reqs: Vec<(usize, u64, PassageIndexEntry)> = Vec::with_capacity(ids.len());
        for (orig_idx, &id) in ids.iter().enumerate() {
            reqs.push((orig_idx, id, self.entry(id)?));
        }
        reqs.sort_unstable_by_key(|r| r.2.token_offset);

        let mut results: Vec<(usize, u64, Vec<u16>)> = Vec::with_capacity(reqs.len());
        for (orig_idx, id, entry) in reqs {
            let count = entry.token_count();
            let tokens = if count == 0 {
                Vec::new()
            } else if entry.is_u32_tokens() {
                let nbytes = count * 4;
                let buf = self.read_range(entry.token_offset, nbytes)?;
                let mut out = Vec::with_capacity(count);
                for chunk in buf.chunks_exact(4) {
                    let v = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    out.push((v & 0xffff) as u16);
                }
                out
            } else {
                let nbytes = count * 2;
                let buf = self.read_range(entry.token_offset, nbytes)?;
                let mut out = Vec::with_capacity(count);
                for chunk in buf.chunks_exact(2) {
                    out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
                out
            };
            results.push((orig_idx, id, tokens));
        }
        results.sort_unstable_by_key(|r| r.0);
        Ok(results.into_iter().map(|(_, id, t)| (id, t)).collect())
    }

    /// Batch fetch `u32` token spans in LBA order and pack them into a single zero-padding
    /// [`OwnedPackedTokenBatch`] (`tokens` + `cu_seqlens`) in caller ID order.
    pub fn fetch_tokens_packed(&self, ids: &[u64]) -> Result<OwnedPackedTokenBatch, PassageError> {
        let mut reqs: Vec<(usize, u64, PassageIndexEntry)> = Vec::with_capacity(ids.len());
        let mut total_tokens = 0usize;
        for (orig_idx, &id) in ids.iter().enumerate() {
            let entry = self.entry(id)?;
            total_tokens += entry.token_count();
            reqs.push((orig_idx, id, entry));
        }
        reqs.sort_unstable_by_key(|r| r.2.token_offset);

        let mut results: Vec<(usize, u64, Vec<u32>)> = Vec::with_capacity(reqs.len());
        for (orig_idx, id, entry) in reqs {
            let toks = self.read_entry_tokens_u32(id, entry)?;
            results.push((orig_idx, id, toks));
        }
        results.sort_unstable_by_key(|r| r.0);

        let mut out_ids = Vec::with_capacity(ids.len());
        let mut tokens = Vec::with_capacity(total_tokens);
        let mut cu_seqlens = Vec::with_capacity(ids.len() + 1);
        cu_seqlens.push(0);
        for (_, id, toks) in results {
            out_ids.push(id);
            tokens.extend_from_slice(&toks);
            cu_seqlens.push(tokens.len() as u32);
        }

        Ok(OwnedPackedTokenBatch {
            ids: out_ids,
            tokens,
            cu_seqlens,
        })
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
        let token_len = u16::from_le_bytes(header[24..26].try_into().unwrap()) as usize;

        let payload = self.read_range(entry.file_offset, text_len + meta_len)?;
        let text = String::from_utf8(payload[..text_len].to_vec())?;
        let meta = if meta_len == 0 {
            serde_json::json!({})
        } else {
            serde_json::from_slice(&payload[text_len..text_len + meta_len])?
        };
        let tokens = if token_len == 0 {
            Vec::new()
        } else {
            self.fetch_tokens(id)?
        };

        Ok(PassageRecord {
            id: rec_id,
            text,
            timestamp,
            source,
            meta,
            tokens,
        })
    }

    /// Batch fetch texts for shortlist vector ordinals, sorted by ascending file offset (LBA order).
    pub fn fetch_many(&self, ids: &[u64]) -> Result<Vec<(u64, String)>, PassageError> {
        let mut reqs: Vec<(usize, u64, PassageIndexEntry)> = Vec::with_capacity(ids.len());
        for (orig_idx, &id) in ids.iter().enumerate() {
            reqs.push((orig_idx, id, self.entry(id)?));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_align_16kib() {
        assert_eq!(align_page(0), 0);
        assert_eq!(align_page(1), PAGE_ALIGN);
        assert_eq!(align_page(PAGE_ALIGN), PAGE_ALIGN);
        assert_eq!(align_page(PAGE_ALIGN + 1), PAGE_ALIGN * 2);
        assert_eq!(align(100, 64), 128);
    }
}
