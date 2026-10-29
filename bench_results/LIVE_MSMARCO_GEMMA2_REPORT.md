# Live MS MARCO + `google/embeddinggemma-2` End-to-End Benchmark (Pure Rust + Apple Silicon AMX)

## 1. Methodology (Zero Pre-Baked Embeddings)

- **Model Checkpoint**: `google/embeddinggemma-2` (`EmbeddingGemma2TextModel`, 24 layers, `hidden_size = 512`, `intermediate_size = 2048`, `output_dim = 768`, 262,144-vocab SentencePiece BPE with 514,906 merges, Per-Layer Embeddings `PLE`, and hybrid sliding/full attention on layers `5, 11, 17, 23`).
- **Execution Engine**: [`Gemma2Embedder`](../src/gemma_embed.rs) running locally in pure Rust on Apple M3 Pro (12 cores) linked against macOS `Accelerate.framework` (`cblas_sgemm` hardware AMX coprocessor + `vForce` `vvtanhf`) and `rayon`.
- **Offline Operator Folding**: At model load time (`575.7 ms`), the 768-d output projection matrix $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{768 \times 512}$, the 128-d Stratum-1 FWHT ($\frac{1}{\sqrt{128}}\mathbf{H}_{128}$), and the `576 -> 64` Stratum-4 tail fold + FWHT ($\frac{1}{3\sqrt{64}}\mathbf{H}_{64}\mathbf{S}_{\text{fold}}$) are fused into $\mathbf{W}_{\text{full}} \in \mathbb{R}^{960 \times 512}$.
- **Corpus & Queries**: `8,192` raw MS MARCO passages (`texts.jsonl`) and `128` MS MARCO dev queries (`queries.jsonl`) with `137` human relevance judgments (`qrels.jsonl` from `BeIR/msmarco-qrels`), tokenized, embedded, and indexed **from scratch** during the benchmark run.

---

## 2. Ingest, Tokenization & Numerical Fidelity (Before vs. After Speedup)

| Metric | Baseline (Unoptimized) | Optimized (AMX Strided Attn + `vForce` + `rayon`) | Speedup |
|---|---:|---:|---:|
| Model `.safetensors` + 262k BPE `tokenizer.json` load + offline FWHT fold | 598.1 ms | **575.7 ms** | **1.04×** |
| Tokenize 8,192 MS MARCO passages & persist `u32` token spans (`passages.bin` v2) | 3,749.9 ms (`2,185 docs/s`) | **540.9 ms** (**`15,145 docs/s`**) | **6.93×** |
| Live 24-layer embed + fused `PaxBlock64` index (8,192 passages, 128 blocks) | 1,618.0 s (`5.06 docs/s`) | **164.5 s** (**`49.80 docs/s`**) | **9.84×** |
| Live 24-layer query embedding (128 MS MARCO dev queries, batched) | 5.83 s (`45.51 ms/query`) | **0.50 s** (**`3.94 ms/query`**, `253.8 q/s`) | **11.55×** |
| Live 24-layer single-query sequential latency | 45.51 ms/query | **29.41 ms/query** (`34.0 q/s`) | **1.55×** |
| Fused vs. unfused `PaxBlock64` bit-identical blocks (`CRC32` + byte equality) | 128 / 128 (100.00%) | **128 / 128 (100.00%)** | — |
| Mean cosine similarity vs. PyTorch A100 reference (8,192 passages) | 0.999960 | **0.999960** | — |
| Mean cosine similarity vs. PyTorch A100 reference (128 queries) | 0.999968 | **0.999968** | — |

---

## 3. Retrieval Quality & Latency Across Shortlist Sizes

- **Exact EmbeddingGemma-2 Oracle Ceiling on Corpus**:
  - `qrels Recall@10 = 0.9844`
  - `qrels nDCG@10 = 0.9531`
  - `qrels MRR@10 = 0.9466`

| Shortlist ($K$) | `PaxBlock64` Scan (`ms/q`) | ANN `R@1` | ANN `R@10` | BeIR `qrels R@10` | BeIR `nDCG@10` | BeIR `MRR@10` | Deferred Top-10 Text Fetch (`µs/q`) | Full-Shortlist Text Fetch (`µs/q`) | Text I/O Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **64** | 0.319 | 0.9922 | 0.7453 | 0.9766 | 0.9453 | 0.9388 | 4.7 µs | 35.5 µs | **7.5×** |
| **256** | 0.627 | **1.0000** | 0.9156 | **0.9844** | **0.9531** | **0.9466** | 4.9 µs | 134.4 µs | **27.3×** |
| **512** | 1.335 | **1.0000** | 0.9539 | **0.9844** | **0.9531** | **0.9466** | 5.3 µs | 269.3 µs | **50.4×** |
| **1024** | 2.959 | **1.0000** | **0.9844** | **0.9844** | **0.9531** | **0.9466** | 6.9 µs | 562.5 µs | **81.9×** |
