# Baseline vs Optimized (SWAR/SIMD + Rayon + mmap)

Config: EmbeddingGemma-2 @768, shortlist=64, strata=l1_rabitq, dual GT (neighbors+qrels).

Speedup columns: `mean_scan_ms` / `encode_ms` = baseline÷opt; `mvecs_per_sec` = opt÷baseline.

## 1m

| metric | baseline | optimized | speedup / Δ |
|--------|----------|-----------|-------------|
| mean_scan_ms | 183.828 | 23.649 | 7.77x |
| mvecs_per_sec | 5.440 | 42.285 | 7.77x |
| encode_ms | 2393.598 | 1030.527 | 2.32x |
| shortlist_coverage_at_10 | 0.585344 | 0.585344 | +0.000000 |
| reranked_recall_at_10 | 0.585344 | 0.585344 | +0.000000 |
| qrels_recall_at_10 | 0.716070 | 0.716070 | +0.000000 |
| qrels_ndcg_at_10 | 0.574708 | 0.574708 | +0.000000 |

Quality match (bit-identical metrics): **PASS**

## 8.8m

| metric | baseline | optimized | speedup / Δ |
|--------|----------|-----------|-------------|
| mean_scan_ms | 1734.792 | 208.612 | 8.32x |
| mvecs_per_sec | 5.097 | 42.384 | 8.32x |
| encode_ms | 54857.287 | 37989.184 | 1.44x |
| shortlist_coverage_at_10 | 0.597421 | 0.597421 | +0.000000 |
| reranked_recall_at_10 | 0.597421 | 0.597421 | +0.000000 |
| qrels_recall_at_10 | 0.431280 | 0.431280 | +0.000000 |
| qrels_ndcg_at_10 | 0.285516 | 0.285516 | +0.000000 |

Quality match (bit-identical metrics): **PASS**

