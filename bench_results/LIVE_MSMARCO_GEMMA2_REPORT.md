# Live MS MARCO + `google/embeddinggemma-2` End-to-End Benchmark (Pure Rust + Apple Silicon AMX)

## 1. Provenance & Methodology (Zero Pre-Baked Embeddings)

| Provenance Field | Value |
|---|---|
| `model` | `google/embeddinggemma-2` (`EmbeddingGemma2TextModel`, 24 layers, `hidden_size = 512`, `intermediate_size = 2048`, `output_dim = 768`, 262,144-vocab SentencePiece BPE with 514,906 merges, Per-Layer Embeddings `PLE`, hybrid sliding/full attention on layers `5, 11, 17, 23`) |
| `compute_unit` | `CPU + Apple Silicon AMX (Accelerate.framework cblas_sgemm + vForce vvtanhf + Rayon; no CoreML/ANE, no Metal GPU dispatch)` |
| `dtype` | `fp32 (activations, RMSNorm, RoPE, attention, MLP, and final 768-d embeddings)` |
| `weights_dtype` | `bf16 on disk (model.safetensors, 622 MB) -> dequantized to fp32 in RAM at load` |
| `reference_dtype` | `bf16 forward pass on NVIDIA A100-SXM4-40GB -> stored as fp32 in embeddings.bin` |
| `ingest_batch_size` / `query_batch_size` | `64` (`VECTORS_PER_BLOCK = 64`) / `128` (128 dev queries packed via `cu_seqlens`) |
| `strata` / `stratum4_weight` / `cs_tolerance` | `l1_rabitq` / `0.2` / `1.0` |
| `rerank_mode` | `in_memory_dot_768` (bench harness caches live-embedded `[f32; 768]` in RAM for fast multi-shortlist sweep across `sl = 64, 256, 512, 1024`; production `IndexEngine::query` uses zero-`f32`-storage JIT re-embed over packed `u32` token spans) |

- **Offline Operator Folding**: At model load time (`625.1 ms`), the 768-d output projection matrix $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{768 \times 512}$, the 128-d Stratum-1 FWHT ($\frac{1}{\sqrt{128}}\mathbf{H}_{128}$), and the `576 -> 64` Stratum-4 tail fold + FWHT ($\frac{1}{3\sqrt{64}}\mathbf{H}_{64}\mathbf{S}_{\text{fold}}$) are fused into $\mathbf{W}_{\text{full}} \in \mathbb{R}^{960 \times 512}$.
- **Corpus & Queries**: `8,192` raw MS MARCO passages (`texts.jsonl`) and `128` MS MARCO dev queries (`queries.jsonl`) with `137` human relevance judgments (`qrels.jsonl` from `BeIR/msmarco-qrels`), tokenized, embedded, and indexed **from scratch** during the benchmark run.

---

## 2. Ingest, Tokenization & Numerical Fidelity (Before vs. After Speedup)

| Metric | Baseline (Unoptimized) | Optimized (AMX Strided Attn + `vForce` + `rayon`) | Speedup |
|---|---:|---:|---:|
| Model `.safetensors` + 262k BPE `tokenizer.json` load + offline FWHT fold | 598.1 ms | **625.1 ms** | **0.96×** |
| Tokenize 8,192 MS MARCO passages & persist `u32` token spans (`passages.bin` v2) | 3,749.9 ms (`2,185 docs/s`) | **477.3 ms** (**`17,162 docs/s`**) | **7.86×** |
| Live 24-layer embed + fused `PaxBlock64` index (8,192 passages, 128 blocks) | 1,618.0 s (`5.06 docs/s`) | **169.2 s** (**`48.43 docs/s`**) | **9.56×** |
| Live 24-layer query embedding (128 MS MARCO dev queries, batched) | 5.83 s (`45.51 ms/query`) | **0.59 s** (**`4.60 ms/query`**, `217.4 q/s`) | **9.89×** |
| Live 24-layer single-query sequential latency (`batch_size = 1`, warm) | 45.51 ms/query | **30.01 ms/query** (`33.3 q/s`) | **1.52×** |
| Fused vs. unfused `PaxBlock64` bit-identical blocks (`CRC32` + byte equality) | 128 / 128 (100.00%) | **128 / 128 (100.00%)** | — |
| Fused ingest vs. unfused query `scan_blocks` bit-identical queries (`id` + `score.to_bits()`) | 128 / 128 (100.00%) | **128 / 128 (100.00%)** | — |
| Cosine similarity vs. PyTorch BF16 A100 reference (8,192 passages: mean / min) | 0.999960 / 0.999732 | **0.999960 / 0.999732** | — |
| Cosine similarity vs. PyTorch BF16 A100 reference (128 queries: mean / min) | 0.999968 / 0.999927 | **0.999968 / 0.999927** | — |

### Batched Throughput Sweep (`batch_size = 1, 8, 32, 64` on 64 MS MARCO Queries & Passages)

| Batch Size | Query Latency (`ms/query`) | Query Throughput (`queries/s`) | Passage Latency (`ms/doc`) | Passage Throughput (`passages/s`) |
|---:|---:|---:|---:|---:|
| **1** | 31.22 ms | 32.0 q/s | 59.73 ms | 16.7 docs/s |
| **8** | 9.33 ms | 107.2 q/s | 34.17 ms | 29.3 docs/s |
| **32** | 7.80 ms | 128.1 q/s | 23.30 ms | 42.9 docs/s |
| **64** | **5.20 ms** | **192.4 q/s** | **21.94 ms** | **45.6 docs/s** |

---

## 3. Retrieval Quality & Latency Across Shortlist Sizes

- **Exact EmbeddingGemma-2 Oracle Ceiling on Corpus**:
  - `qrels Recall@10 = 0.9844`
  - `qrels nDCG@10 = 0.9531`
  - `qrels MRR@10 = 0.9466`

| Shortlist ($K$) | `PaxBlock64` Scan (`ms/q`) | ANN `R@1` | ANN `R@10` | BeIR `qrels R@10` | BeIR `nDCG@10` | BeIR `MRR@10` | Deferred Top-10 Text Fetch (`µs/q`) | Full-Shortlist Text Fetch (`µs/q`) | Text I/O Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **64** | 0.310 | 0.9922 | 0.7453 | 0.9766 | 0.9453 | 0.9388 | 5.0 µs | 34.9 µs | **7.0×** |
| **256** | 0.619 | **1.0000** | 0.9156 | **0.9844** | **0.9531** | **0.9466** | 4.7 µs | 128.2 µs | **27.2×** |
| **512** | 1.335 | **1.0000** | 0.9539 | **0.9844** | **0.9531** | **0.9466** | 5.3 µs | 263.2 µs | **49.4×** |
| **1024** | 2.966 | **1.0000** | **0.9844** | **0.9844** | **0.9531** | **0.9466** | 7.5 µs | 553.6 µs | **73.8×** |

---

## 4. Reproducibility & Opt-In `--gemma2` CLI Surface

### One-Command Cold Mac Benchmark Reproduction
```bash
huggingface-cli download google/embeddinggemma-2 model.safetensors tokenizer.json config.json && \
  cargo run --release --example bench_msmarco_live -- data/msmarco_gemma2/smoke_bin 8192 && \
  cat bench_results/live_msmarco_gemma2.json
```

### Opt-In `--gemma2` Flag in `mrl-index` CLI
`Gemma2Embedder` is opt-in behind `--gemma2` (default remains lightweight 384-d `HashingEmbedder` when `--gemma2` is not passed):
```bash
# Initialize an index configured for 768-d google/embeddinggemma-2 (persists "use_gemma2": true in manifest.json)
cargo run --release --bin mrl-index -- init /tmp/my_gemma2_idx --gemma2

# Ingest passages (automatically uses Gemma2Embedder when manifest.json has use_gemma2=true or --gemma2 is passed)
cargo run --release --bin mrl-index -- ingest /tmp/my_gemma2_idx --text "Matryoshka Representation Learning enables multi-granularity retrieval." --flush

# Query with zero-f32-storage JIT token-span reranking
cargo run --release --bin mrl-index -- query /tmp/my_gemma2_idx "multi-granularity vector search"
```
