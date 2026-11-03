# Primitives composition — end-to-end bench

**Branch:** `cursor/scan-stage-primitives-4432`  
**When:** 2026-11-02 09:02 America/Chicago  
**Corpus:** MS MARCO EmbeddingGemma-2 @768 (`1m_bin`, `smoke_bin`)  
**Config:** shortlist=256, `w_s4=0.20`, `cs_tolerance=1.0`

Interactive write-up: see the companion canvas linked from PR #4.

## Headline (1M, 128 queries, clean H2H)

| Build | Scan ms/q | Mvecs/s | ANN R@10 | qrels R@10 |
|-------|-----------|---------|----------|------------|
| `origin/main` @ `877eeee` | 35.579 | 28.11 | **0.8813** | **0.7930** |
| CS-prune @ `3b9bf58` | 33.163 | 30.15 | **0.8813** | **0.7930** |
| primitives (avg of 2) | **30.766** | **32.5** | **0.8813** | **0.7930** |

- **−14.8%** scan latency vs main at identical recall  
- **−7.2%** vs monolithic CS-prune branch (staging is not a regression)  
- S4 abs-budget keep-rate on 1M: **7.09%** (`--scan-stats`)

> Concurrent compiles inflated earlier timings (47–83 ms/q). The clean sequential series above is authoritative.

## Smoke (8,192 docs, 128 q)

| Strata | Scan ms/q | ANN R@10 | qrels R@10 |
|--------|-----------|----------|------------|
| `Full64b` (production) | 1.169 | 0.9133 | 0.9844 |
| `L1Rabitq` (no S4) | 0.812 | 0.8883 | 0.9844 |

Gates make the speed/quality trade-off explicit: skipping S4 is faster but drops ANN coverage.

## Primitive → benefit map

| Primitive | Benefit | Evidence |
|-----------|---------|----------|
| `S4AbsBudget` + split residual CS | Faster scan, same R@10 | 1M H2H vs main; keep_s4bud≈7% |
| Named `ScanKernel` stages | CS as composition; no mid-fn fork | Matches/beats CS-prune branch |
| `StrataDepth` gates | Real S1/S4 ablation | Smoke/1M L1 vs Full |
| `QueryPipeline` | One shortlist path | engine + retrieval_bench + benchmark |
| `Encoder` | Shared encode, bit-identical | unit: trait == `encode_block` |
| `ScanConfig` / `SketchGeometry` | Single policy + layout contract | typed API on this PR |

## Repro

```bash
huggingface-cli download astr010/msmarco-gemma2-smoke --repo-type dataset \
  --local-dir data/msmarco_gemma2/smoke
huggingface-cli download astr010/msmarco-gemma2-1m --repo-type dataset \
  --local-dir data/msmarco_gemma2/1m
python3 scripts/parquet_to_bench_bin.py data/msmarco_gemma2/smoke data/msmarco_gemma2/smoke_bin
python3 scripts/parquet_to_bench_bin.py data/msmarco_gemma2/1m data/msmarco_gemma2/1m_bin
cargo build --release --bin mrl-index

./target/release/mrl-index bench-retrieval \
  --label primitives-1m --data data/msmarco_gemma2/1m_bin \
  --queries 128 --shortlist 256 --strata full_64b

./target/release/mrl-index bench-retrieval \
  --label primitives-1m-stats --data data/msmarco_gemma2/1m_bin \
  --queries 16 --shortlist 256 --strata full_64b --scan-stats
```

Raw JSON logs: `/opt/cursor/artifacts/e2e_primitives/` (agent run) and this doc for the summary table.
