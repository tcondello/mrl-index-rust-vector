//! Load Modal-generated MRL embedding artifacts (`data/sec10k_mrl/`).

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CorpusError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("corrupt embeddings: expected {expected} bytes, got {got}")]
    CorruptSize { expected: usize, got: usize },
    #[error("manifest missing at {0}")]
    MissingManifest(PathBuf),
    #[error("corpus is not MRL (manifest.mrl != true) — refuse isotropic stand-ins")]
    NotMrl,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CorpusManifest {
    pub dataset: String,
    pub model: String,
    pub dims: usize,
    pub n: usize,
    pub mrl: bool,
    pub embeddings_file: String,
    pub texts_file: String,
    #[serde(default)]
    pub truncate_dim: Option<usize>,
    #[serde(default)]
    pub query_prefix: Option<String>,
    #[serde(default)]
    pub document_prefix: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CorpusText {
    pub id: usize,
    pub text: String,
    #[serde(default)]
    pub ticker: Option<String>,
    #[serde(default)]
    pub fiscal_year: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct MrlCorpus {
    pub dir: PathBuf,
    pub manifest: CorpusManifest,
    pub texts: Vec<CorpusText>,
    /// Row-major f32, length = n * dims.
    pub embeddings: Vec<f32>,
}

impl MrlCorpus {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, CorpusError> {
        let dir = dir.as_ref().to_path_buf();
        let manifest_path = dir.join("manifest.json");
        if !manifest_path.exists() {
            return Err(CorpusError::MissingManifest(manifest_path));
        }
        let manifest: CorpusManifest =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
        if !manifest.mrl {
            return Err(CorpusError::NotMrl);
        }

        let mut texts = Vec::with_capacity(manifest.n);
        let texts_path = dir.join(&manifest.texts_file);
        let file = File::open(&texts_path)?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            texts.push(serde_json::from_str(&line)?);
        }

        let emb_path = dir.join(&manifest.embeddings_file);
        let expected = manifest.n * manifest.dims * 4;
        let mut buf = Vec::new();
        File::open(&emb_path)?.read_to_end(&mut buf)?;
        if buf.len() != expected {
            return Err(CorpusError::CorruptSize {
                expected,
                got: buf.len(),
            });
        }
        let mut embeddings = vec![0.0f32; manifest.n * manifest.dims];
        for (i, chunk) in buf.chunks_exact(4).enumerate() {
            embeddings[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        Ok(Self {
            dir,
            manifest,
            texts,
            embeddings,
        })
    }

    pub fn dims(&self) -> usize {
        self.manifest.dims
    }

    pub fn len(&self) -> usize {
        self.manifest.n
    }

    pub fn is_empty(&self) -> bool {
        self.manifest.n == 0
    }

    pub fn vector(&self, i: usize) -> &[f32] {
        let d = self.dims();
        &self.embeddings[i * d..(i + 1) * d]
    }

    pub fn all_vectors(&self) -> Vec<Vec<f32>> {
        (0..self.len()).map(|i| self.vector(i).to_vec()).collect()
    }
}
