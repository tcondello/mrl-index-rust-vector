# Step 4 — Low-Precision Shortlist Rerank (`fp32` / `fp16` / `int8`)

**Date:** 2026-10-29 (America/New_York)  
**Defaults:** `stratum4_weight = 0.20` (Stage 2 lock), `cs_tolerance = 1.0`  
**Model / corpus:** EmbeddingGemma-2 @768 · BeIR/msmarco (`1m_bin`, `full_bin`) · 6,980 queries

## Verdict

| Precision | Bytes/vec | vs fp32 | ANN R@10 parity | Notes |
|-----------|----------:|---------|-----------------|-------|
| **fp32** | 3,072 | — | baseline | |
| **fp16** | 1,536 (2×) | **100.0%** on all 4 configs | Bit-identical top-10 vs fp32 |
| **int8** | 770 (≈4×) | **98.0–99.1%** | Slightly under 99.5% target; IR (qrels) essentially unchanged |

## Results

| Corpus | sl | Prec | B/vec | ANN R@10 | qrels R@10 | nDCG@10 | parity | mean rerank µs |
|--------|---:|------|------:|---------:|-----------:|--------:|-------:|---------------:|
| 1M | 256 | fp32 | 3072 | 0.8812 | 0.8111 | 0.6316 | 1.0000 | 411 |
| 1M | 256 | fp16 | 1536 | 0.8812 | 0.8111 | 0.6316 | **1.0000** | 616 |
| 1M | 256 | int8 | 770 | 0.8728 | 0.8114 | 0.6313 | 0.9905 | 630 |
| 1M | 1024 | fp32 | 3072 | 0.9541 | 0.8224 | 0.6380 | 1.0000 | 1171 |
| 1M | 1024 | fp16 | 1536 | 0.9541 | 0.8224 | 0.6380 | **1.0000** | 2613 |
| 1M | 1024 | int8 | 770 | 0.9402 | 0.8224 | 0.6377 | 0.9855 | 2596 |
| 8.8M | 256 | fp32 | 3072 | 0.8988 | 0.5284 | 0.3319 | 1.0000 | 9052* |
| 8.8M | 256 | fp16 | 1536 | 0.8988 | 0.5284 | 0.3319 | **1.0000** | 646 |
| 8.8M | 256 | int8 | 770 | 0.8862 | 0.5287 | 0.3314 | 0.9860 | 618 |
| 8.8M | 1024 | fp32 | 3072 | 0.9633 | 0.5390 | 0.3365 | 1.0000 | 37080* |
| 8.8M | 1024 | fp16 | 1536 | 0.9633 | 0.5390 | 0.3365 | **1.0000** | 2634 |
| 8.8M | 1024 | int8 | 770 | 0.9441 | 0.5391 | 0.3358 | 0.9801 | 2644 |

\*fp32 rerank µs on 8.8M is inflated by cold mmap faults (fp32 runs first per query); warm `fp16`/`int8` paths better reflect steady-state convert+dot cost.

## Artifacts

- `bench_results/step4/{1m,8.8m}_sl{256,1024}/{label}-{fp32,fp16,int8}.json`
- `bench_results/step4/*/summary.json`

## Implementation

- `src/lowprec.rs` — `Fp16Vec768`, `Int8Vec768` (symmetric per-vector scale)
- `src/embed.rs` — `dot_f16_768`, `dot_i8_768`
- `src/ring.rs` — optional fp16/int8 mirrors on `PendingVector`
- CLI: `mrl-index bench-precision-rerank`
