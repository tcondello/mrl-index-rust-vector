//! On-disk segment of contiguous `PaxBlock64` macro-blocks.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};
use thiserror::Error;

use crate::block::{PaxBlock64, BLOCK_BYTES};
use crate::config::SlabResidency;

#[derive(Debug, Error)]
pub enum SegmentError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt segment size {0} (not multiple of 4096)")]
    CorruptSize(u64),
}

pub struct SegmentWriter {
    path: PathBuf,
    file: File,
    blocks: u32,
}

impl SegmentWriter {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, SegmentError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(&path)?;
        Ok(Self {
            path,
            file,
            blocks: 0,
        })
    }

    pub fn open_append(path: impl AsRef<Path>) -> Result<Self, SegmentError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        let len = file.metadata()?.len();
        if len % BLOCK_BYTES as u64 != 0 {
            return Err(SegmentError::CorruptSize(len));
        }
        Ok(Self {
            path,
            file,
            blocks: (len / BLOCK_BYTES as u64) as u32,
        })
    }

    pub fn append_block(&mut self, block: &PaxBlock64) -> Result<u32, SegmentError> {
        let idx = self.blocks;
        self.file.write_all(block.as_bytes())?;
        self.file.flush()?;
        self.blocks += 1;
        Ok(idx)
    }

    pub fn block_count(&self) -> u32 {
        self.blocks
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

pub struct SegmentReader {
    path: PathBuf,
    mmap: Option<Mmap>,
    owned: Vec<PaxBlock64>,
    residency: SlabResidency,
}

impl SegmentReader {
    pub fn open(path: impl AsRef<Path>, residency: SlabResidency) -> Result<Self, SegmentError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Ok(Self {
                path,
                mmap: None,
                owned: Vec::new(),
                residency,
            });
        }
        let file = File::open(&path)?;
        let len = file.metadata()?.len();
        if len % BLOCK_BYTES as u64 != 0 {
            return Err(SegmentError::CorruptSize(len));
        }

        match residency {
            SlabResidency::RamLocked | SlabResidency::MmapWillneed => {
                let mmap = unsafe { MmapOptions::new().map(&file)? };
                #[cfg(unix)]
                {
                    // Advise sequential / willneed; ignore errors on unsupported FS.
                    let _ = mmap.advise(memmap2::Advice::WillNeed);
                }
                let mut owned = Vec::new();
                if matches!(residency, SlabResidency::RamLocked) {
                    // Materialise into process heap (mlock alternative without privileges).
                    let n = (len as usize) / BLOCK_BYTES;
                    owned.reserve(n);
                    for i in 0..n {
                        let start = i * BLOCK_BYTES;
                        let block = PaxBlock64::from_bytes(&mmap[start..start + BLOCK_BYTES])
                            .expect("aligned block")
                            .clone();
                        owned.push(block);
                    }
                    return Ok(Self {
                        path,
                        mmap: None,
                        owned,
                        residency,
                    });
                }
                Ok(Self {
                    path,
                    mmap: Some(mmap),
                    owned: Vec::new(),
                    residency,
                })
            }
            SlabResidency::DirectNvme => {
                // Portable fallback: read into owned buffer (true O_DIRECT is platform-specific).
                let mut buf = vec![0u8; len as usize];
                let mut f = file;
                f.seek(SeekFrom::Start(0))?;
                f.read_exact(&mut buf)?;
                let n = buf.len() / BLOCK_BYTES;
                let mut owned = Vec::with_capacity(n);
                for i in 0..n {
                    let start = i * BLOCK_BYTES;
                    let block = PaxBlock64::from_bytes(&buf[start..start + BLOCK_BYTES])
                        .expect("block")
                        .clone();
                    owned.push(block);
                }
                Ok(Self {
                    path,
                    mmap: None,
                    owned,
                    residency,
                })
            }
        }
    }

    pub fn block_count(&self) -> usize {
        if !self.owned.is_empty() {
            return self.owned.len();
        }
        self.mmap
            .as_ref()
            .map(|m| m.len() / BLOCK_BYTES)
            .unwrap_or(0)
    }

    pub fn block(&self, idx: usize) -> Option<&PaxBlock64> {
        if !self.owned.is_empty() {
            return self.owned.get(idx);
        }
        let mmap = self.mmap.as_ref()?;
        let start = idx * BLOCK_BYTES;
        if start + BLOCK_BYTES > mmap.len() {
            return None;
        }
        PaxBlock64::from_bytes(&mmap[start..start + BLOCK_BYTES])
    }

    /// Contiguous slice of blocks when backed by `owned` or 4 KiB-aligned `mmap`.
    pub fn as_slice(&self) -> Option<&[PaxBlock64]> {
        if !self.owned.is_empty() {
            return Some(&self.owned);
        }
        let mmap = self.mmap.as_ref()?;
        bytemuck::try_cast_slice(&mmap[..]).ok()
    }

    pub fn for_each_block(&self, mut f: impl FnMut(u32, &PaxBlock64)) {
        let n = self.block_count();
        for i in 0..n {
            if let Some(b) = self.block(i) {
                f(i as u32, b);
            }
        }
    }

    pub fn residency(&self) -> SlabResidency {
        self.residency
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a newly written block in O(1) without remapping the whole segment.
    ///
    /// Previously `commit_block` reopened + recopied the entire file each flush
    /// (O(N²) over ingest). For mmap-backed readers we materialise once into
    /// `owned`, then push.
    pub fn append_live(&mut self, block: PaxBlock64) {
        if self.owned.is_empty() {
            if let Some(mmap) = self.mmap.take() {
                let n = mmap.len() / BLOCK_BYTES;
                self.owned.reserve(n + 1);
                for i in 0..n {
                    let start = i * BLOCK_BYTES;
                    let b = PaxBlock64::from_bytes(&mmap[start..start + BLOCK_BYTES])
                        .expect("aligned block")
                        .clone();
                    self.owned.push(b);
                }
            }
        }
        self.owned.push(block);
    }
}
