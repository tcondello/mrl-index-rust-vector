//! Local MacBook Apple Silicon (`Accelerate` AMX + Metal 3 shader + SafeTensors `BF16`)
//! `EmbeddingGemma-2` fused embedder implementing [`crate::rerank::UnifiedEmbedder`].
//!
//! ## Key Architectural Features
//! 1. **Apple Silicon AMX (`Accelerate.framework` `cblas_sgemm`)**:
//!    On macOS, dense projections (`QKV`, `Out`, `GeGLU`, and the offline-folded
//!    `[960, d_model]` augmented output head) dispatch directly to Apple's hardware
//!    AMX matrix coprocessor via `cblas_sgemm` in Unified Memory.
//! 2. **BF16 / FP32 Accumulation (Immune to ANE FP16 Overflow)**:
//!    Loads `BF16`, `F16`, or `F32` tensors from SafeTensors and accumulates RMSNorm
//!    and attention in `f32`, avoiding the >65,504 activation overflow on ANE.
//! 3. **Offline Isometric Weight Folding (`W_full ∈ R^{960 × d_model}`)**:
//!    Pre-multiplies `W_proj` offline by `T_norm = diag(I_64, H_128, (1/3) H_64 S)`
//!    via [`crate::rotate::augment_projection_weights`], eliminating runtime FWHT
//!    during `ingest_packed_to_paxblocks`.
//! 4. **Pinned Task `PrefixKVCache`**:
//!    Precomputes and pins the K/V states of `"task: search result | query: "` across
//!    all transformer layers at initialization so `begin_query()` only prefills the
//!    new user query tokens.
//! 5. **Ragged `varlen` FlashAttention (`cu_seqlens`) & 2-Stage Cascaded Rerank**:
//!    Executes variable-length batches with zero `<pad>` tokens, prunes shortlist
//!    candidates at `early_exit_layer` against `QuerySession::query_early_64`, and
//!    runs late transformer layers only on the top survivors.

use std::collections::HashMap;
use std::path::Path;

use half::{bf16, f16};

use crate::block::{PaxBlock64, EMBED_DIMS_768, MRL_PREFIX_DIMS, VECTORS_PER_BLOCK};
use crate::embed::dot_768;
use crate::quantize::encode_block_from_augmented;
use crate::rerank::{Embedder, PackedTokenBatch, QuerySession, UnifiedEmbedder};
use crate::rotate::{augment_projection_weights, AUGMENTED_FULL_DIMS};

/// Embedded Metal Shading Language (MSL) kernel source for `encode_paxblock64_fused`
/// (`simd_ballot` with disjoint 32-bit word indexing) and `fused_rerank_dot_768`.
pub const PAXBLOCK_FUSED_METAL_SRC: &str = include_str!("shaders/paxblock_fused.metal");

/// Default Gemma-2 query task instruction prefix.
pub const DEFAULT_QUERY_TASK_PREFIX: &str = "task: search result | query: ";
/// Default Gemma-2 document instruction prefix.
pub const DEFAULT_DOC_TASK_PREFIX: &str = "title: none | text: ";

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Compute `C = A · B^T` where `A` is `[m, k]` row-major and `B` is `[n, k]` row-major,
/// writing `[m, n]` row-major into `C`.
///
/// On macOS, dispatches to Apple Silicon's AMX hardware coprocessor via `Accelerate`'s
/// `cblas_sgemm`. On other platforms, uses an 8-way unrolled FMA loop.
#[inline]
pub fn gemm_nt_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    #[cfg(target_os = "macos")]
    unsafe {
        // CblasRowMajor = 101, CblasNoTrans = 111, CblasTrans = 112
        cblas_sgemm(
            101,
            111,
            112,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];
            for j in 0..n {
                let b_row = &b[j * k..(j + 1) * k];
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_row[p] * b_row[p];
                }
                c_row[j] = sum;
            }
        }
    }
}

/// Configuration for the local `EmbeddingGemma-2` model.
#[derive(Debug, Clone)]
pub struct Gemma2Config {
    /// Vocabulary size (262,144 for Gemma-2 SentencePiece).
    pub vocab_size: usize,
    /// Transformer hidden dimension `d_model`.
    pub hidden_size: usize,
    /// GeGLU intermediate MLP dimension.
    pub intermediate_size: usize,
    /// Total transformer layers `L`.
    pub num_hidden_layers: usize,
    /// Speculative early-exit layer index (`1..=num_hidden_layers`).
    pub early_exit_layer: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value attention heads (Grouped-Query Attention).
    pub num_key_value_heads: usize,
    /// Head dimension `d_head`.
    pub head_dim: usize,
    /// Output embedding dimension (fixed at 768 for MRL PaxBlock64).
    pub embed_dims: usize,
    /// RMSNorm epsilon (`1e-6`).
    pub rms_norm_eps: f32,
    /// RoPE base frequency (`10000.0`).
    pub rope_theta: f32,
    /// Gemma-2 attention logit softcapping (`50.0`).
    pub attn_logit_softcapping: Option<f32>,
    /// Maximum tokens evaluated per candidate in Stage-A shortlist pruning.
    pub stage1_max_tokens: usize,
    /// Fraction of shortlist candidates kept after Stage-A early-layer pruning (`0.25` = top 25%).
    pub stage1_keep_ratio: f32,
}

impl Gemma2Config {
    /// Production `google/embeddinggemma-2` (270M @ 768-d MRL) configuration.
    pub fn gemma2_768() -> Self {
        Self {
            vocab_size: 262_144,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            early_exit_layer: 8,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            embed_dims: EMBED_DIMS_768,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            attn_logit_softcapping: Some(50.0),
            stage1_max_tokens: 64,
            stage1_keep_ratio: 0.25,
        }
    }

    /// Compact deterministic configuration for fast local unit/e2e testing and prototyping
    /// without requiring a multi-hundred-MB weight download.
    pub fn micro_test() -> Self {
        Self {
            vocab_size: 262_144,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 3,
            early_exit_layer: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            embed_dims: EMBED_DIMS_768,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            attn_logit_softcapping: Some(50.0),
            stage1_max_tokens: 32,
            stage1_keep_ratio: 0.50,
        }
    }
}

/// Weights for a single Gemma-2 Transformer layer.
#[derive(Debug, Clone)]
pub struct Gemma2LayerWeights {
    pub input_layernorm: Vec<f32>,          // [hidden_size]
    pub q_proj: Vec<f32>,                   // [num_heads * head_dim, hidden_size]
    pub k_proj: Vec<f32>,                   // [num_kv_heads * head_dim, hidden_size]
    pub v_proj: Vec<f32>,                   // [num_kv_heads * head_dim, hidden_size]
    pub o_proj: Vec<f32>,                   // [hidden_size, num_heads * head_dim]
    pub post_attention_layernorm: Vec<f32>, // [hidden_size]
    pub gate_proj: Vec<f32>,                // [intermediate_size, hidden_size]
    pub up_proj: Vec<f32>,                  // [intermediate_size, hidden_size]
    pub down_proj: Vec<f32>,                // [hidden_size, intermediate_size]
}

/// Precomputed and pinned K/V cache for a static task instruction prefix across all layers.
#[derive(Debug, Clone)]
pub struct PrefixKVCache {
    pub prefix_tokens: Vec<u32>,
    /// Per-layer cached key states `[prefix_len * num_kv_heads * head_dim]` (with RoPE applied).
    pub k_cache: Vec<Vec<f32>>,
    /// Per-layer cached value states `[prefix_len * num_kv_heads * head_dim]`.
    pub v_cache: Vec<Vec<f32>>,
    /// Per-layer cached hidden states for accurate mean pooling across `[prefix; user_tokens]`.
    pub prefix_early_sum: Vec<f32>,
    pub prefix_final_sum: Vec<f32>,
}

/// Minimal zero-copy SafeTensors parser supporting `BF16`, `F16`, and `F32` tensors.
pub struct SafeTensorsView<'a> {
    data: &'a [u8],
    tensors: HashMap<String, TensorMeta>,
}

#[derive(Debug, Clone)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

impl<'a> SafeTensorsView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err("safetensors file too small (< 8 bytes)".into());
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let data_start = 8usize
            .checked_add(header_len)
            .ok_or_else(|| "safetensors header_len overflow".to_string())?;
        if bytes.len() < data_start {
            return Err("safetensors truncated header".into());
        }
        let root: serde_json::Value = serde_json::from_slice(&bytes[8..data_start])
            .map_err(|e| format!("invalid safetensors JSON header: {e}"))?;
        let obj = root
            .as_object()
            .ok_or_else(|| "safetensors header root must be a JSON object".to_string())?;

        let mut tensors = HashMap::new();
        for (name, val) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = val
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("missing dtype for {name}"))?
                .to_string();
            let shape: Vec<usize> = val
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("missing shape for {name}"))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let offsets = val
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("missing data_offsets for {name}"))?;
            if offsets.len() != 2 {
                return Err(format!("invalid data_offsets for {name}"));
            }
            let start = offsets[0].as_u64().unwrap_or(0) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;
            tensors.insert(
                name.clone(),
                TensorMeta {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }

        Ok(Self {
            data: &bytes[data_start..],
            tensors,
        })
    }

    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|m| m.shape.as_slice())
    }

    /// Decode a tensor (`F32`, `BF16`, or `F16`) into `Vec<f32>`.
    pub fn load_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor `{name}` in safetensors"))?;
        if meta.end > self.data.len() || meta.start > meta.end {
            return Err(format!("out-of-bounds offsets for tensor `{name}`"));
        }
        let raw = &self.data[meta.start..meta.end];
        let numel: usize = meta.shape.iter().product();
        match meta.dtype.as_str() {
            "F32" => {
                if raw.len() != numel * 4 {
                    return Err(format!("size mismatch for F32 tensor `{name}`"));
                }
                Ok(raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            "BF16" => {
                if raw.len() != numel * 2 {
                    return Err(format!("size mismatch for BF16 tensor `{name}`"));
                }
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect())
            }
            "F16" => {
                if raw.len() != numel * 2 {
                    return Err(format!("size mismatch for F16 tensor `{name}`"));
                }
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect())
            }
            other => Err(format!("unsupported dtype `{other}` for tensor `{name}`")),
        }
    }
}

/// Serialize a map of `(name, shape, &[f32])` into a `BF16` SafeTensors byte buffer.
/// Useful for converting/saving fused weights (`W_full`) and testing SafeTensors roundtrips.
pub fn serialize_safetensors_bf16(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let mut header_map = serde_json::Map::new();
    let mut offset = 0usize;
    for &(name, shape, data) in tensors {
        let nbytes = data.len() * 2;
        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), serde_json::Value::String("BF16".into()));
        entry.insert(
            "shape".into(),
            serde_json::Value::Array(shape.iter().map(|&d| serde_json::Value::from(d)).collect()),
        );
        entry.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![
                serde_json::Value::from(offset),
                serde_json::Value::from(offset + nbytes),
            ]),
        );
        header_map.insert(name.to_string(), serde_json::Value::Object(entry));
        offset += nbytes;
    }
    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header_map)).unwrap();
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }
    let header_len = header_bytes.len() as u64;
    let mut out = Vec::with_capacity(8 + header_bytes.len() + offset);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for &(_, _, data) in tensors {
        for &val in data {
            let b = bf16::from_f32(val).to_le_bytes();
            out.extend_from_slice(&b);
        }
    }
    out
}

/// Local Apple Silicon `EmbeddingGemma-2` engine with offline isometric FWHT weight folding,
/// pinned instruction `PrefixKVCache`, ragged `cu_seqlens` FlashAttention, and 2-stage shortlist reranking.
#[derive(Debug, Clone)]
pub struct Gemma2Embedder {
    cfg: Gemma2Config,
    /// Token embedding table:either dense `[vocab_size, hidden_size]` or procedural Subword/Hash lookup
    /// when `token_emb_table` has fewer rows than `vocab_size`.
    token_emb_table: Vec<f32>,
    token_emb_rows: usize,
    layers: Vec<Gemma2LayerWeights>,
    final_norm: Vec<f32>, // [hidden_size]
    /// Original 768-d projection matrix `[768, hidden_size]`.
    w_proj: Vec<f32>,
    /// Offline-folded augmented matrix `W_full = [960, hidden_size]`:
    /// rows `0..768` = `W_proj`, rows `768..896` = `H_128 W_mid`, rows `896..960` = `(1/3) H_64 S W_tail`.
    w_proj_augmented: Vec<f32>,
    /// Early-exit Layer-8 64-d MRL prefix projection `[64, hidden_size]`.
    w_early_64: Vec<f32>,
    /// Pinned KV-cache for the static query task prompt (`"task: search result | query: "`).
    query_prefix_cache: Option<PrefixKVCache>,
    /// Optional vocabulary string-to-id map (from `tokenizer.json`).
    vocab_map: HashMap<String, u32>,
}

impl Gemma2Embedder {
    /// Construct a deterministic local `Gemma2Embedder` from [`Gemma2Config`] (for tests, benchmarks,
    /// or before loading a `.safetensors` checkpoint).
    pub fn new_deterministic(cfg: Gemma2Config, seed: u64) -> Self {
        let mut rng_state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_f32 = |scale: f32| -> f32 {
            rng_state ^= rng_state >> 12;
            rng_state ^= rng_state << 25;
            rng_state ^= rng_state >> 27;
            let u = rng_state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let unit = ((u >> 40) as f32 / ((1u32 << 24) as f32)) * 2.0 - 1.0;
            unit * scale
        };

        let d = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        // Keep 4096 hashed subword rows in memory for instant startup; full SafeTensors loads all rows.
        let token_emb_rows = cfg.vocab_size.min(4096);
        let emb_scale = 1.0 / (d as f32).sqrt();
        let token_emb_table: Vec<f32> = (0..token_emb_rows * d)
            .map(|_| next_f32(emb_scale))
            .collect();

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            // Small residual weights keep multi-layer signals well-conditioned without random blowup.
            let attn_scale = 0.25 / (d as f32).sqrt();
            let mlp_scale = 0.25 / (inter as f32).sqrt();
            layers.push(Gemma2LayerWeights {
                input_layernorm: vec![0.0f32; d], // Gemma-2 RMSNorm adds (1.0 + w)
                q_proj: (0..q_dim * d).map(|_| next_f32(attn_scale)).collect(),
                k_proj: (0..kv_dim * d).map(|_| next_f32(attn_scale)).collect(),
                v_proj: (0..kv_dim * d).map(|_| next_f32(attn_scale)).collect(),
                o_proj: (0..d * q_dim).map(|_| next_f32(attn_scale)).collect(),
                post_attention_layernorm: vec![0.0f32; d],
                gate_proj: (0..inter * d).map(|_| next_f32(attn_scale)).collect(),
                up_proj: (0..inter * d).map(|_| next_f32(attn_scale)).collect(),
                down_proj: (0..d * inter).map(|_| next_f32(mlp_scale)).collect(),
            });
        }

        let final_norm = vec![0.0f32; d];
        let mut w_proj = vec![0.0f32; EMBED_DIMS_768 * d];
        for r in 0..EMBED_DIMS_768 {
            // Matryoshka spectral decay so leading 64 dims carry the strongest signal.
            let mrl_decay = (1.0 / (1.0 + 0.015 * r as f32)).sqrt() / (d as f32).sqrt();
            for c in 0..d {
                w_proj[r * d + c] = next_f32(mrl_decay);
            }
        }

        let w_proj_augmented = augment_projection_weights(&w_proj, d);
        let w_early_64 = w_proj[..MRL_PREFIX_DIMS * d].to_vec();

        let mut embedder = Self {
            cfg,
            token_emb_table,
            token_emb_rows,
            layers,
            final_norm,
            w_proj,
            w_proj_augmented,
            w_early_64,
            query_prefix_cache: None,
            vocab_map: HashMap::new(),
        };
        embedder.pin_query_prefix(DEFAULT_QUERY_TASK_PREFIX);
        embedder
    }

    /// Load a `Gemma2Embedder` from a SafeTensors byte slice (`BF16` / `F16` / `F32`)
    /// and automatically fold `W_proj` offline into `W_full = T_full · W_proj`.
    pub fn from_safetensors_bytes(
        cfg: Gemma2Config,
        safetensors_bytes: &[u8],
    ) -> Result<Self, String> {
        let st = SafeTensorsView::parse(safetensors_bytes)?;
        let d = cfg.hidden_size;

        let token_emb_table = st.load_f32("model.embed_tokens.weight")?;
        if token_emb_table.len() % d != 0 {
            return Err("model.embed_tokens.weight length not divisible by hidden_size".into());
        }
        let token_emb_rows = token_emb_table.len() / d;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let pfx = format!("model.layers.{l}");
            layers.push(Gemma2LayerWeights {
                input_layernorm: st.load_f32(&format!("{pfx}.input_layernorm.weight"))?,
                q_proj: st.load_f32(&format!("{pfx}.self_attn.q_proj.weight"))?,
                k_proj: st.load_f32(&format!("{pfx}.self_attn.k_proj.weight"))?,
                v_proj: st.load_f32(&format!("{pfx}.self_attn.v_proj.weight"))?,
                o_proj: st.load_f32(&format!("{pfx}.self_attn.o_proj.weight"))?,
                post_attention_layernorm: st
                    .load_f32(&format!("{pfx}.post_attention_layernorm.weight"))?,
                gate_proj: st.load_f32(&format!("{pfx}.mlp.gate_proj.weight"))?,
                up_proj: st.load_f32(&format!("{pfx}.mlp.up_proj.weight"))?,
                down_proj: st.load_f32(&format!("{pfx}.mlp.down_proj.weight"))?,
            });
        }

        let final_norm = st.load_f32("model.norm.weight")?;
        let w_proj = st.load_f32("projection.weight")?;
        if w_proj.len() != EMBED_DIMS_768 * d {
            return Err(format!(
                "projection.weight expected {} elements, got {}",
                EMBED_DIMS_768 * d,
                w_proj.len()
            ));
        }

        let w_proj_augmented = augment_projection_weights(&w_proj, d);
        let w_early_64 = st
            .load_f32("early_projection.weight")
            .unwrap_or_else(|_| w_proj[..MRL_PREFIX_DIMS * d].to_vec());

        let mut embedder = Self {
            cfg,
            token_emb_table,
            token_emb_rows,
            layers,
            final_norm,
            w_proj,
            w_proj_augmented,
            w_early_64,
            query_prefix_cache: None,
            vocab_map: HashMap::new(),
        };
        embedder.pin_query_prefix(DEFAULT_QUERY_TASK_PREFIX);
        Ok(embedder)
    }

    /// Memory-map a `.safetensors` file from disk and construct the fused `Gemma2Embedder`.
    pub fn from_safetensors_file(
        cfg: Gemma2Config,
        path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| format!("open: {e}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
        Self::from_safetensors_bytes(cfg, &mmap)
    }

    /// Export this model's weights into a `BF16` SafeTensors byte buffer.
    pub fn to_safetensors_bf16(&self) -> Vec<u8> {
        let d = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;
        let q_dim = self.cfg.num_attention_heads * self.cfg.head_dim;
        let kv_dim = self.cfg.num_key_value_heads * self.cfg.head_dim;

        let emb_shape = [self.token_emb_rows, d];
        let norm_shape = [d];
        let q_shape = [q_dim, d];
        let kv_shape = [kv_dim, d];
        let o_shape = [d, q_dim];
        let gate_shape = [inter, d];
        let down_shape = [d, inter];
        let proj_shape = [EMBED_DIMS_768, d];
        let early_shape = [MRL_PREFIX_DIMS, d];

        let mut names = Vec::new();
        for l in 0..self.cfg.num_hidden_layers {
            let pfx = format!("model.layers.{l}");
            names.push((
                format!("{pfx}.input_layernorm.weight"),
                format!("{pfx}.self_attn.q_proj.weight"),
                format!("{pfx}.self_attn.k_proj.weight"),
                format!("{pfx}.self_attn.v_proj.weight"),
                format!("{pfx}.self_attn.o_proj.weight"),
                format!("{pfx}.post_attention_layernorm.weight"),
                format!("{pfx}.mlp.gate_proj.weight"),
                format!("{pfx}.mlp.up_proj.weight"),
                format!("{pfx}.mlp.down_proj.weight"),
            ));
        }

        let mut entries: Vec<(&str, &[usize], &[f32])> = Vec::new();
        entries.push(("model.embed_tokens.weight", &emb_shape, &self.token_emb_table));
        for (l, lw) in self.layers.iter().enumerate() {
            let n = &names[l];
            entries.push((&n.0, &norm_shape, &lw.input_layernorm));
            entries.push((&n.1, &q_shape, &lw.q_proj));
            entries.push((&n.2, &kv_shape, &lw.k_proj));
            entries.push((&n.3, &kv_shape, &lw.v_proj));
            entries.push((&n.4, &o_shape, &lw.o_proj));
            entries.push((&n.5, &norm_shape, &lw.post_attention_layernorm));
            entries.push((&n.6, &gate_shape, &lw.gate_proj));
            entries.push((&n.7, &gate_shape, &lw.up_proj));
            entries.push((&n.8, &down_shape, &lw.down_proj));
        }
        entries.push(("model.norm.weight", &norm_shape, &self.final_norm));
        entries.push(("projection.weight", &proj_shape, &self.w_proj));
        entries.push(("early_projection.weight", &early_shape, &self.w_early_64));

        serialize_safetensors_bf16(&entries)
    }

    pub fn config(&self) -> &Gemma2Config {
        &self.cfg
    }

    pub fn pinned_prefix_cache(&self) -> Option<&PrefixKVCache> {
        self.query_prefix_cache.as_ref()
    }

    /// Precompute and pin the K/V cache for a static query task instruction prefix.
    pub fn pin_query_prefix(&mut self, prefix_text: &str) {
        let prefix_tokens = self.tokenize_u32(prefix_text);
        if prefix_tokens.is_empty() {
            self.query_prefix_cache = None;
            return;
        }
        let d = self.cfg.hidden_size;
        let seq_len = prefix_tokens.len();
        let mut hidden = self.lookup_tokens(&prefix_tokens);
        let mut k_cache = Vec::with_capacity(self.cfg.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(self.cfg.num_hidden_layers);
        let mut prefix_early_sum = vec![0.0f32; d];

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let (next_h, k_out, v_out) =
                self.forward_layer_single_seq(&hidden, seq_len, layer, None);
            hidden = next_h;
            k_cache.push(k_out);
            v_cache.push(v_out);

            if l_idx + 1 == self.cfg.early_exit_layer {
                let normed = self.apply_rms_norm(&hidden, &self.final_norm);
                for t in 0..seq_len {
                    for c in 0..d {
                        prefix_early_sum[c] += normed[t * d + c];
                    }
                }
            }
        }

        let final_normed = self.apply_rms_norm(&hidden, &self.final_norm);
        let mut prefix_final_sum = vec![0.0f32; d];
        for t in 0..seq_len {
            for c in 0..d {
                prefix_final_sum[c] += final_normed[t * d + c];
            }
        }

        self.query_prefix_cache = Some(PrefixKVCache {
            prefix_tokens,
            k_cache,
            v_cache,
            prefix_early_sum,
            prefix_final_sum,
        });
    }

    /// Look up token embeddings `[N, hidden_size]` and scale by `√hidden_size` (Gemma convention).
    fn lookup_tokens(&self, tokens: &[u32]) -> Vec<f32> {
        let d = self.cfg.hidden_size;
        let scale = (d as f32).sqrt();
        let mut out = vec![0.0f32; tokens.len() * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let row = (tok as usize) % self.token_emb_rows;
            let src = &self.token_emb_table[row * d..(row + 1) * d];
            let dst = &mut out[i * d..(i + 1) * d];
            // Add high-bit subword perturbation when `vocab_size > token_emb_rows` so all 262,144 IDs
            // produce distinct embeddings even in compact test tables.
            let bucket = (tok as usize) / self.token_emb_rows;
            if bucket == 0 {
                for c in 0..d {
                    dst[c] = src[c] * scale;
                }
            } else {
                let phase = (bucket as u64).wrapping_mul(0x9E37_79B9);
                for c in 0..d {
                    let sign = if ((phase >> (c & 31)) & 1) == 0 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                    dst[c] = src[(c + bucket) % d] * scale * sign;
                }
            }
        }
        out
    }

    /// Gemma-2 RMSNorm in `f32` accumulation: `y_i = (x_i / rms(x)) * (1.0 + w_i)`.
    fn apply_rms_norm(&self, x: &[f32], weight: &[f32]) -> Vec<f32> {
        let d = self.cfg.hidden_size;
        let n_tokens = x.len() / d;
        let mut out = vec![0.0f32; x.len()];
        for t in 0..n_tokens {
            let row = &x[t * d..(t + 1) * d];
            let out_row = &mut out[t * d..(t + 1) * d];
            let mut sq_sum = 0.0f32;
            for &v in row {
                sq_sum += v * v;
            }
            let inv_rms = 1.0 / ((sq_sum / d as f32) + self.cfg.rms_norm_eps).sqrt();
            for c in 0..d {
                out_row[c] = row[c] * inv_rms * (1.0 + weight[c]);
            }
        }
        out
    }

    /// Apply Rotary Position Embeddings (RoPE) in-place to `[n_heads, head_dim]` at position `pos`.
    #[inline]
    fn apply_rope_token(&self, heads_vec: &mut [f32], n_heads: usize, pos: usize) {
        let hd = self.cfg.head_dim;
        let half = hd / 2;
        let pos_f = pos as f32;
        for h in 0..n_heads {
            let base = h * hd;
            for i in 0..half {
                let freq =
                    1.0 / self.cfg.rope_theta.powf((2 * i) as f32 / (hd as f32));
                let angle = pos_f * freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let x0 = heads_vec[base + i];
                let x1 = heads_vec[base + half + i];
                heads_vec[base + i] = x0 * cos_a - x1 * sin_a;
                heads_vec[base + half + i] = x1 * cos_a + x0 * sin_a;
            }
        }
    }

    /// Run one Gemma-2 layer on a single sequence (optionally attending to a pinned prefix KV cache).
    fn forward_layer_single_seq(
        &self,
        hidden: &[f32],
        seq_len: usize,
        layer: &Gemma2LayerWeights,
        prefix_kv: Option<(&[f32], &[f32], usize)>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.cfg.hidden_size;
        let n_heads = self.cfg.num_attention_heads;
        let n_kv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let gqa_group = n_heads / n_kv;
        let pos_offset = prefix_kv.map_or(0, |(_, _, plen)| plen);

        let normed = self.apply_rms_norm(hidden, &layer.input_layernorm);
        let mut q = vec![0.0f32; seq_len * q_dim];
        let mut k = vec![0.0f32; seq_len * kv_dim];
        let mut v = vec![0.0f32; seq_len * kv_dim];

        gemm_nt_f32(&normed, &layer.q_proj, &mut q, seq_len, q_dim, d);
        gemm_nt_f32(&normed, &layer.k_proj, &mut k, seq_len, kv_dim, d);
        gemm_nt_f32(&normed, &layer.v_proj, &mut v, seq_len, kv_dim, d);

        for t in 0..seq_len {
            let pos = pos_offset + t;
            self.apply_rope_token(&mut q[t * q_dim..(t + 1) * q_dim], n_heads, pos);
            self.apply_rope_token(&mut k[t * kv_dim..(t + 1) * kv_dim], n_kv, pos);
        }

        let total_kv_len = pos_offset + seq_len;
        let attn_scale = 1.0 / (hd as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * q_dim];
        let mut scores = vec![0.0f32; total_kv_len];

        for t in 0..seq_len {
            for h in 0..n_heads {
                let kv_h = h / gqa_group;
                let q_slice = &q[t * q_dim + h * hd..t * q_dim + (h + 1) * hd];

                let mut max_score = f32::NEG_INFINITY;
                for s in 0..total_kv_len {
                    let k_slice = if s < pos_offset {
                        let (pk, _, _) = prefix_kv.unwrap();
                        &pk[s * kv_dim + kv_h * hd..s * kv_dim + (kv_h + 1) * hd]
                    } else {
                        let local_s = s - pos_offset;
                        &k[local_s * kv_dim + kv_h * hd..local_s * kv_dim + (kv_h + 1) * hd]
                    };
                    let mut dot = 0.0f32;
                    for c in 0..hd {
                        dot += q_slice[c] * k_slice[c];
                    }
                    let mut sc = dot * attn_scale;
                    if let Some(cap) = self.cfg.attn_logit_softcapping {
                        sc = (sc / cap).tanh() * cap;
                    }
                    scores[s] = sc;
                    if sc > max_score {
                        max_score = sc;
                    }
                }

                let mut exp_sum = 0.0f32;
                for s in 0..total_kv_len {
                    let e = (scores[s] - max_score).exp();
                    scores[s] = e;
                    exp_sum += e;
                }
                let inv_sum = 1.0 / exp_sum.max(1e-12);

                let out_slice = &mut attn_out[t * q_dim + h * hd..t * q_dim + (h + 1) * hd];
                for s in 0..total_kv_len {
                    let w = scores[s] * inv_sum;
                    let v_slice = if s < pos_offset {
                        let (_, pv, _) = prefix_kv.unwrap();
                        &pv[s * kv_dim + kv_h * hd..s * kv_dim + (kv_h + 1) * hd]
                    } else {
                        let local_s = s - pos_offset;
                        &v[local_s * kv_dim + kv_h * hd..local_s * kv_dim + (kv_h + 1) * hd]
                    };
                    for c in 0..hd {
                        out_slice[c] += w * v_slice[c];
                    }
                }
            }
        }

        let mut proj_attn = vec![0.0f32; seq_len * d];
        gemm_nt_f32(&attn_out, &layer.o_proj, &mut proj_attn, seq_len, d, q_dim);

        let mut res1 = vec![0.0f32; seq_len * d];
        for i in 0..res1.len() {
            res1[i] = hidden[i] + proj_attn[i];
        }

        // GeGLU MLP
        let post_norm = self.apply_rms_norm(&res1, &layer.post_attention_layernorm);
        let inter = self.cfg.intermediate_size;
        let mut gate = vec![0.0f32; seq_len * inter];
        let mut up = vec![0.0f32; seq_len * inter];
        gemm_nt_f32(&post_norm, &layer.gate_proj, &mut gate, seq_len, inter, d);
        gemm_nt_f32(&post_norm, &layer.up_proj, &mut up, seq_len, inter, d);

        for i in 0..gate.len() {
            gate[i] = gelu_tanh(gate[i]) * up[i];
        }

        let mut mlp_out = vec![0.0f32; seq_len * d];
        gemm_nt_f32(&gate, &layer.down_proj, &mut mlp_out, seq_len, d, inter);

        for i in 0..res1.len() {
            res1[i] += mlp_out[i];
        }

        (res1, k, v)
    }

    /// Ragged `varlen` FlashAttention forward pass over a packed token batch (`cu_seqlens`)
    /// with zero `<pad>` tokens! All linear GEMMs execute over the flattened `[total_tokens, d_model]`
    /// matrix while attention and RoPE respect sequence boundaries `cu_seqlens[b]..cu_seqlens[b+1]`.
    fn forward_ragged_layers(
        &self,
        batch: PackedTokenBatch<'_>,
        up_to_layer: usize,
    ) -> Vec<f32> {
        let total_tokens = batch.tokens.len();
        let d = self.cfg.hidden_size;
        if total_tokens == 0 {
            return Vec::new();
        }

        let n_heads = self.cfg.num_attention_heads;
        let n_kv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let gqa_group = n_heads / n_kv;
        let inter = self.cfg.intermediate_size;
        let attn_scale = 1.0 / (hd as f32).sqrt();

        let mut hidden = self.lookup_tokens(batch.tokens);
        let mut q = vec![0.0f32; total_tokens * q_dim];
        let mut k = vec![0.0f32; total_tokens * kv_dim];
        let mut v = vec![0.0f32; total_tokens * kv_dim];
        let mut attn_out = vec![0.0f32; total_tokens * q_dim];
        let mut proj_attn = vec![0.0f32; total_tokens * d];
        let mut gate = vec![0.0f32; total_tokens * inter];
        let mut up = vec![0.0f32; total_tokens * inter];
        let mut mlp_out = vec![0.0f32; total_tokens * d];

        let max_layer = up_to_layer.min(self.layers.len());
        for layer in &self.layers[..max_layer] {
            let normed = self.apply_rms_norm(&hidden, &layer.input_layernorm);

            // Dense ragged GEMMs across all sequences at once (zero padding tokens!)
            gemm_nt_f32(&normed, &layer.q_proj, &mut q, total_tokens, q_dim, d);
            gemm_nt_f32(&normed, &layer.k_proj, &mut k, total_tokens, kv_dim, d);
            gemm_nt_f32(&normed, &layer.v_proj, &mut v, total_tokens, kv_dim, d);

            attn_out.fill(0.0);

            // Block-diagonal FlashAttention per sequence span [start..end) with RoPE position reset
            for seq_idx in 0..batch.num_sequences() {
                let start = batch.cu_seqlens[seq_idx] as usize;
                let end = batch.cu_seqlens[seq_idx + 1] as usize;
                let seq_len = end.saturating_sub(start);
                if seq_len == 0 {
                    continue;
                }

                for local_pos in 0..seq_len {
                    let t = start + local_pos;
                    self.apply_rope_token(&mut q[t * q_dim..(t + 1) * q_dim], n_heads, local_pos);
                    self.apply_rope_token(&mut k[t * kv_dim..(t + 1) * kv_dim], n_kv, local_pos);
                }

                let mut scores = vec![0.0f32; seq_len];
                for local_t in 0..seq_len {
                    let t = start + local_t;
                    for h in 0..n_heads {
                        let kv_h = h / gqa_group;
                        let q_slice = &q[t * q_dim + h * hd..t * q_dim + (h + 1) * hd];

                        let mut max_score = f32::NEG_INFINITY;
                        for local_s in 0..seq_len {
                            let s = start + local_s;
                            let k_slice =
                                &k[s * kv_dim + kv_h * hd..s * kv_dim + (kv_h + 1) * hd];
                            let mut dot = 0.0f32;
                            for c in 0..hd {
                                dot += q_slice[c] * k_slice[c];
                            }
                            let mut sc = dot * attn_scale;
                            if let Some(cap) = self.cfg.attn_logit_softcapping {
                                sc = (sc / cap).tanh() * cap;
                            }
                            scores[local_s] = sc;
                            if sc > max_score {
                                max_score = sc;
                            }
                        }

                        let mut exp_sum = 0.0f32;
                        for local_s in 0..seq_len {
                            let e = (scores[local_s] - max_score).exp();
                            scores[local_s] = e;
                            exp_sum += e;
                        }
                        let inv_sum = 1.0 / exp_sum.max(1e-12);

                        let out_slice =
                            &mut attn_out[t * q_dim + h * hd..t * q_dim + (h + 1) * hd];
                        for local_s in 0..seq_len {
                            let s = start + local_s;
                            let w = scores[local_s] * inv_sum;
                            let v_slice =
                                &v[s * kv_dim + kv_h * hd..s * kv_dim + (kv_h + 1) * hd];
                            for c in 0..hd {
                                out_slice[c] += w * v_slice[c];
                            }
                        }
                    }
                }
            }

            gemm_nt_f32(
                &attn_out,
                &layer.o_proj,
                &mut proj_attn,
                total_tokens,
                d,
                q_dim,
            );
            for i in 0..hidden.len() {
                hidden[i] += proj_attn[i];
            }

            let post_norm = self.apply_rms_norm(&hidden, &layer.post_attention_layernorm);
            gemm_nt_f32(&post_norm, &layer.gate_proj, &mut gate, total_tokens, inter, d);
            gemm_nt_f32(&post_norm, &layer.up_proj, &mut up, total_tokens, inter, d);
            for i in 0..gate.len() {
                gate[i] = gelu_tanh(gate[i]) * up[i];
            }
            gemm_nt_f32(&gate, &layer.down_proj, &mut mlp_out, total_tokens, d, inter);
            for i in 0..hidden.len() {
                hidden[i] += mlp_out[i];
            }
        }

        self.apply_rms_norm(&hidden, &self.final_norm)
    }

    /// Mean-pool ragged token hidden states `[total_tokens, d_model]` per sequence `[B, d_model]`.
    fn mean_pool_ragged(&self, normed_hidden: &[f32], batch: PackedTokenBatch<'_>) -> Vec<f32> {
        let b = batch.num_sequences();
        let d = self.cfg.hidden_size;
        let mut pooled = vec![0.0f32; b * d];
        for seq_idx in 0..b {
            let start = batch.cu_seqlens[seq_idx] as usize;
            let end = batch.cu_seqlens[seq_idx + 1] as usize;
            let len = end.saturating_sub(start);
            let dst = &mut pooled[seq_idx * d..(seq_idx + 1) * d];
            if len == 0 {
                dst[0] = 1e-3;
                continue;
            }
            let inv_len = 1.0 / (len as f32);
            for t in start..end {
                let row = &normed_hidden[t * d..(t + 1) * d];
                for c in 0..d {
                    dst[c] += row[c] * inv_len;
                }
            }
        }
        pooled
    }
}

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn l2_normalize_slice(v: &mut [f32]) -> f32 {
    let mut n2 = 0.0f32;
    for &x in v.iter() {
        n2 += x * x;
    }
    let norm = n2.sqrt().max(1e-12);
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
    norm
}

impl UnifiedEmbedder for Gemma2Embedder {
    fn tokenize_u32(&self, text: &str) -> Vec<u32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let vocab_mask = (self.cfg.vocab_size as u64).saturating_sub(1).max(1);
        let mut out = Vec::new();
        for tok in text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
        {
            if let Some(&id) = self.vocab_map.get(tok) {
                out.push(id);
            } else {
                let mut h = DefaultHasher::new();
                tok.hash(&mut h);
                // Map into [4..vocab_size) (reserving 0..4 for BOS/EOS/PAD/UNK).
                let id = 4 + ((h.finish() % (vocab_mask - 4)) as u32);
                out.push(id);
            }
        }
        if out.is_empty() {
            out.push(1); // BOS fallback for empty strings
        }
        out
    }

    fn begin_query(&self, query_text: &str) -> QuerySession {
        let user_tokens = self.tokenize_u32(query_text);
        let d = self.cfg.hidden_size;
        let seq_len = user_tokens.len();

        let mut hidden = self.lookup_tokens(&user_tokens);
        let mut early_hidden_normed = None;

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let prefix_kv = self.query_prefix_cache.as_ref().map(|cache| {
                (
                    cache.k_cache[l_idx].as_slice(),
                    cache.v_cache[l_idx].as_slice(),
                    cache.prefix_tokens.len(),
                )
            });
            let (next_h, _, _) = self.forward_layer_single_seq(&hidden, seq_len, layer, prefix_kv);
            hidden = next_h;

            if l_idx + 1 == self.cfg.early_exit_layer {
                early_hidden_normed = Some(self.apply_rms_norm(&hidden, &self.final_norm));
            }
        }

        let final_normed = self.apply_rms_norm(&hidden, &self.final_norm);
        let early_normed = early_hidden_normed.as_ref().unwrap_or(&final_normed);

        let prefix_len = self
            .query_prefix_cache
            .as_ref()
            .map_or(0, |c| c.prefix_tokens.len());
        let total_len = (prefix_len + seq_len).max(1) as f32;

        let mut pooled_early = self
            .query_prefix_cache
            .as_ref()
            .map_or_else(|| vec![0.0f32; d], |c| c.prefix_early_sum.clone());
        let mut pooled_final = self
            .query_prefix_cache
            .as_ref()
            .map_or_else(|| vec![0.0f32; d], |c| c.prefix_final_sum.clone());

        for t in 0..seq_len {
            for c in 0..d {
                pooled_early[c] += early_normed[t * d + c];
                pooled_final[c] += final_normed[t * d + c];
            }
        }
        for c in 0..d {
            pooled_early[c] /= total_len;
            pooled_final[c] /= total_len;
        }

        let mut query_early_64 = [0.0f32; MRL_PREFIX_DIMS];
        gemm_nt_f32(
            &pooled_early,
            &self.w_early_64,
            &mut query_early_64,
            1,
            MRL_PREFIX_DIMS,
            d,
        );
        l2_normalize_slice(&mut query_early_64);

        let mut query_f32 = [0.0f32; EMBED_DIMS_768];
        gemm_nt_f32(
            &pooled_final,
            &self.w_proj,
            &mut query_f32,
            1,
            EMBED_DIMS_768,
            d,
        );
        l2_normalize_slice(&mut query_f32);

        QuerySession {
            query_f32,
            query_early_64,
        }
    }

    fn ingest_packed_to_paxblocks(
        &self,
        batch: PackedTokenBatch<'_>,
        timestamps: &[u32],
        sources: &[u32],
        out_blocks: &mut [PaxBlock64],
    ) {
        let b = batch.num_sequences();
        if b == 0 {
            return;
        }
        assert_eq!(timestamps.len(), b);
        assert_eq!(sources.len(), b);
        let expected_blocks = (b + VECTORS_PER_BLOCK - 1) / VECTORS_PER_BLOCK;
        assert!(out_blocks.len() >= expected_blocks);

        let d = self.cfg.hidden_size;
        let normed_hidden = self.forward_ragged_layers(batch, self.cfg.num_hidden_layers);
        let pooled = self.mean_pool_ragged(&normed_hidden, batch);

        // Single AMX / SIMD GEMM with the offline-folded `[960, d_model]` matrix `W_full`:
        // produces both the 768-d embedding and the pre-rotated Stratum-1 and Stratum-4 coordinates!
        let mut aug_flat = vec![0.0f32; b * AUGMENTED_FULL_DIMS];
        gemm_nt_f32(
            &pooled,
            &self.w_proj_augmented,
            &mut aug_flat,
            b,
            AUGMENTED_FULL_DIMS,
            d,
        );

        let mut aug_vecs = Vec::with_capacity(b);
        for i in 0..b {
            let slice = &mut aug_flat[i * AUGMENTED_FULL_DIMS..(i + 1) * AUGMENTED_FULL_DIMS];
            // Normalize by the 768-d base embedding norm
            let mut n2 = 0.0f32;
            for &x in &slice[..EMBED_DIMS_768] {
                n2 += x * x;
            }
            let inv_norm = 1.0 / n2.sqrt().max(1e-12);
            let mut arr = [0.0f32; AUGMENTED_FULL_DIMS];
            for c in 0..AUGMENTED_FULL_DIMS {
                arr[c] = slice[c] * inv_norm;
            }
            aug_vecs.push(arr);
        }

        for (blk_idx, chunk) in aug_vecs.chunks(VECTORS_PER_BLOCK).enumerate() {
            let base = blk_idx * VECTORS_PER_BLOCK;
            let end = base + chunk.len();
            out_blocks[blk_idx] =
                encode_block_from_augmented(chunk, &timestamps[base..end], &sources[base..end]);
        }
    }

    fn embed_packed(&self, batch: PackedTokenBatch<'_>) -> Vec<[f32; EMBED_DIMS_768]> {
        let b = batch.num_sequences();
        if b == 0 {
            return Vec::new();
        }
        let d = self.cfg.hidden_size;
        let normed_hidden = self.forward_ragged_layers(batch, self.cfg.num_hidden_layers);
        let pooled = self.mean_pool_ragged(&normed_hidden, batch);

        let mut raw_embeds = vec![0.0f32; b * EMBED_DIMS_768];
        gemm_nt_f32(&pooled, &self.w_proj, &mut raw_embeds, b, EMBED_DIMS_768, d);

        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let mut vec768 = [0.0f32; EMBED_DIMS_768];
            vec768.copy_from_slice(&raw_embeds[i * EMBED_DIMS_768..(i + 1) * EMBED_DIMS_768]);
            l2_normalize_slice(&mut vec768);
            out.push(vec768);
        }
        out
    }

    fn score_shortlist_packed(
        &self,
        session: &QuerySession,
        batch: PackedTokenBatch<'_>,
        prior_scores: &[f32],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let k_cand = batch.num_sequences();
        if k_cand == 0 {
            return Vec::new();
        }
        let d = self.cfg.hidden_size;

        // Stage A: Early-Layer + 64-Token Prefix Pruning when shortlist is large
        let survivors: Vec<usize> = if k_cand > top_k * 2 && self.cfg.early_exit_layer < self.cfg.num_hidden_layers {
            let mut stage1_tokens = Vec::new();
            let mut stage1_cu = Vec::with_capacity(k_cand + 1);
            stage1_cu.push(0);
            for i in 0..k_cand {
                let seq = batch.sequence(i);
                let trunc_len = seq.len().min(self.cfg.stage1_max_tokens);
                stage1_tokens.extend_from_slice(&seq[..trunc_len]);
                stage1_cu.push(stage1_tokens.len() as u32);
            }
            let stage1_batch = PackedTokenBatch {
                tokens: &stage1_tokens,
                cu_seqlens: &stage1_cu,
            };
            let early_normed = self.forward_ragged_layers(stage1_batch, self.cfg.early_exit_layer);
            let early_pooled = self.mean_pool_ragged(&early_normed, stage1_batch);
            let mut early_64 = vec![0.0f32; k_cand * MRL_PREFIX_DIMS];
            gemm_nt_f32(
                &early_pooled,
                &self.w_early_64,
                &mut early_64,
                k_cand,
                MRL_PREFIX_DIMS,
                d,
            );

            let mut stage1_scored: Vec<(usize, f32)> = (0..k_cand)
                .map(|i| {
                    let row = &mut early_64[i * MRL_PREFIX_DIMS..(i + 1) * MRL_PREFIX_DIMS];
                    l2_normalize_slice(row);
                    let mut early_dot = 0.0f32;
                    for c in 0..MRL_PREFIX_DIMS {
                        early_dot += row[c] * session.query_early_64[c];
                    }
                    let prior = prior_scores.get(i).copied().unwrap_or(0.0);
                    (i, 0.7 * early_dot + 0.3 * prior)
                })
                .collect();

            stage1_scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let keep = ((k_cand as f32 * self.cfg.stage1_keep_ratio).ceil() as usize)
                .max(top_k * 2)
                .min(k_cand);
            stage1_scored.into_iter().take(keep).map(|(i, _)| i).collect()
        } else {
            (0..k_cand).collect()
        };

        // Stage B: Full-Context Exact Rerank on Survivors Only
        let mut surv_tokens = Vec::new();
        let mut surv_cu = Vec::with_capacity(survivors.len() + 1);
        surv_cu.push(0);
        for &orig_idx in &survivors {
            surv_tokens.extend_from_slice(batch.sequence(orig_idx));
            surv_cu.push(surv_tokens.len() as u32);
        }
        let surv_batch = PackedTokenBatch {
            tokens: &surv_tokens,
            cu_seqlens: &surv_cu,
        };
        let surv_embeds = self.embed_packed(surv_batch);

        let mut final_scores: Vec<(usize, f32)> = survivors
            .into_iter()
            .zip(surv_embeds.iter())
            .map(|(orig_idx, emb)| (orig_idx, dot_768(&session.query_f32, emb)))
            .collect();

        final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        final_scores.truncate(top_k.max(1));
        final_scores
    }
}

impl Embedder for Gemma2Embedder {
    fn embed(&self, texts: &[String]) -> Vec<Vec<f32>> {
        if texts.is_empty() {
            return Vec::new();
        }
        let mut tokens = Vec::new();
        let mut cu_seqlens = Vec::with_capacity(texts.len() + 1);
        cu_seqlens.push(0);
        for t in texts {
            let seq = self.tokenize_u32(t);
            tokens.extend_from_slice(&seq);
            cu_seqlens.push(tokens.len() as u32);
        }
        let batch = PackedTokenBatch {
            tokens: &tokens,
            cu_seqlens: &cu_seqlens,
        };
        self.embed_packed(batch)
            .into_iter()
            .map(|arr| arr.to_vec())
            .collect()
    }

    fn dimensions(&self) -> usize {
        EMBED_DIMS_768
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::source;
    use crate::quantize::encode_block;

    #[test]
    fn safetensors_bf16_roundtrip_and_fused_ingest_equivalence() {
        let cfg = Gemma2Config::micro_test();
        let model = Gemma2Embedder::new_deterministic(cfg.clone(), 42);

        // Verify BF16 SafeTensors serialization + zero-copy parsing roundtrip
        let st_bytes = model.to_safetensors_bf16();
        let loaded = Gemma2Embedder::from_safetensors_bytes(cfg, &st_bytes)
            .expect("load from BF16 safetensors");
        assert!(loaded.pinned_prefix_cache().is_some());

        // Build a ragged batch of 4 variable-length passages
        let texts = [
            "apple silicon unified memory zero copy metal shaders",
            "matryoshka representation learning bitplane rabitq fast walsh hadamard",
            "short query",
            "gemma two transformer rotary position embeddings grouped query attention softcapping",
        ];
        let mut tokens = Vec::new();
        let mut cu_seqlens = vec![0u32];
        for t in &texts {
            let seq = loaded.tokenize_u32(t);
            tokens.extend_from_slice(&seq);
            cu_seqlens.push(tokens.len() as u32);
        }
        let batch = PackedTokenBatch {
            tokens: &tokens,
            cu_seqlens: &cu_seqlens,
        };

        // 1. Diagnostic full vectors -> standard encode_block (runtime FWHT)
        let vecs = loaded.embed_packed(batch);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ts = [100u32, 101, 102, 103];
        let srcs = [source::DOC; 4];
        let block_unfused = encode_block(&refs, &ts, &srcs, EMBED_DIMS_768);

        // 2. Fused ingest_packed_to_paxblocks (offline-folded W_full, zero runtime FWHT!)
        let mut out_blocks = [PaxBlock64::zeroed(); 1];
        loaded.ingest_packed_to_paxblocks(batch, &ts, &srcs, &mut out_blocks);
        let block_fused = &out_blocks[0];

        assert!(block_fused.verify_crc());
        assert_eq!(block_unfused.as_bytes(), block_fused.as_bytes());
    }

    #[test]
    fn query_session_and_cascaded_shortlist_rerank() {
        let cfg = Gemma2Config::micro_test();
        let model = Gemma2Embedder::new_deterministic(cfg, 1337);

        let passages = [
            "rust vector search bitplane quantization",
            "chocolate chip cookies baking recipe",
            "apple metal shading language simdgroup ballot",
            "rust bitplane vector index retrieval engine",
            "gardening tips for spring tomatoes",
            "transformer prefill kv cache ragged flash attention",
        ];

        let mut tokens = Vec::new();
        let mut cu_seqlens = vec![0u32];
        for p in &passages {
            let seq = model.tokenize_u32(p);
            tokens.extend_from_slice(&seq);
            cu_seqlens.push(tokens.len() as u32);
        }
        let batch = PackedTokenBatch {
            tokens: &tokens,
            cu_seqlens: &cu_seqlens,
        };

        let session = model.begin_query("rust vector bitplane index");
        let priors = vec![0.5f32; passages.len()];
        let top2 = model.score_shortlist_packed(&session, batch, &priors, 2);
        assert_eq!(top2.len(), 2);
        assert!(top2[0].1 >= top2[1].1);
    }
}
