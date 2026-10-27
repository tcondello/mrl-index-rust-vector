//! Raw text store: `passages.jsonl` + packed `passages.idx` (spec §3.2).

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::block::VectorId;

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
    #[error("missing passage for vector {0:?}")]
    Missing(VectorId),
    #[error("corrupt index: entry {0}")]
    CorruptIndex(u64),
}

pub struct PassageStore {
    dir: PathBuf,
    jsonl: File,
    idx: File,
    next_id: u64,
}

impl PassageStore {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, PassageError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let jsonl_path = dir.join("passages.jsonl");
        let idx_path = dir.join("passages.idx");

        let jsonl = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&jsonl_path)?;
        let idx = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&idx_path)?;

        let idx_len = idx.metadata()?.len();
        let next_id = idx_len / 12;

        Ok(Self {
            dir,
            jsonl,
            idx,
            next_id,
        })
    }

    pub fn path(&self) -> &Path {
        &self.dir
    }

    pub fn len(&self) -> u64 {
        self.next_id
    }

    pub fn is_empty(&self) -> bool {
        self.next_id == 0
    }

    /// Append a passage; returns the sequential passage id (= vector ordinal).
    pub fn append(&mut self, mut record: PassageRecord) -> Result<u64, PassageError> {
        let id = self.next_id;
        record.id = id;
        let offset = self.jsonl.seek(SeekFrom::End(0))?;
        let line = serde_json::to_string(&record)?;
        self.jsonl.write_all(line.as_bytes())?;
        self.jsonl.write_all(b"\n")?;
        self.jsonl.flush()?;

        let entry = PassageIndexEntry {
            file_offset: offset,
            text_len: record.text.len() as u32,
        };
        self.idx.write_all(bytemuck::bytes_of(&entry))?;
        self.idx.flush()?;
        self.next_id += 1;
        Ok(id)
    }

    pub fn entry(&self, id: u64) -> Result<PassageIndexEntry, PassageError> {
        if id >= self.next_id {
            return Err(PassageError::CorruptIndex(id));
        }
        let mut file = File::open(self.dir.join("passages.idx"))?;
        file.seek(SeekFrom::Start(id * 12))?;
        let mut buf = [0u8; 12];
        file.read_exact(&mut buf)?;
        Ok(*bytemuck::from_bytes(&buf))
    }

    pub fn fetch_text(&self, id: u64) -> Result<String, PassageError> {
        let entry = self.entry(id)?;
        let mut file = File::open(self.dir.join("passages.jsonl"))?;
        file.seek(SeekFrom::Start(entry.file_offset))?;
        // Read one JSONL line (text_len is payload length inside JSON; parse line instead).
        let mut line = String::new();
        use std::io::BufRead;
        let mut reader = std::io::BufReader::new(file);
        reader.read_line(&mut line)?;
        let rec: PassageRecord = serde_json::from_str(line.trim_end())?;
        Ok(rec.text)
    }

    pub fn fetch_record(&self, id: u64) -> Result<PassageRecord, PassageError> {
        let entry = self.entry(id)?;
        let mut file = File::open(self.dir.join("passages.jsonl"))?;
        file.seek(SeekFrom::Start(entry.file_offset))?;
        use std::io::BufRead;
        let mut reader = std::io::BufReader::new(file);
        let mut line = String::new();
        reader.read_line(&mut line)?;
        Ok(serde_json::from_str(line.trim_end())?)
    }

    /// Batch fetch texts for shortlist vector ordinals.
    pub fn fetch_many(&self, ids: &[u64]) -> Result<Vec<(u64, String)>, PassageError> {
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            out.push((id, self.fetch_text(id)?));
        }
        Ok(out)
    }
}
