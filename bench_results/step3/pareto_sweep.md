# Stage 2 — Query-Only Stratum-4 MMSE Weight & CS-Tolerance Pareto Sweep

**Branch:** `cursor/overnight-step3to5-4432`  
**Date:** 2026-10-29 (America/New_York)  
**Setup:** Encode PaxBlocks **once**; sweep query-side `EngineConfig::stratum4_weight` (`w_s4`) and `cs_tolerance`. Encoded bits are invariant to positive fold scaling.

## Locked default

| Parameter | Old (MMSE `1/√9`) | **Locked** |
|-----------|------------------|------------|
| `DEFAULT_STRATUM4_WEIGHT` | `0.3333` | **`0.20`** |
| `cs_tolerance` | `1.0` | **`1.0`** (inert across `{1.00, 1.05, 1.10}`) |

**Why 0.20:** On both corpora at `sl=256`, lighter Stratum-4 weight beats pure Wiener `1/√9` — S4 noise still over-weights the ranker at 0.333. Goal `8.8M @ sl=256 ≥ 0.887` is met at **0.899**.

## 1M @ `sl=256` (full grid: 5 × 3 = 15 cells)

`cs_tolerance` produced **identical** ANN/IR scores within each `w_s4` (only scan ms jitter). Collapsed by weight:

| `w_s4` | ANN R@10 | ANN R@100 | qrels R@10 | nDCG@10 | scan ms/q |
|-------:|---------:|----------:|-----------:|--------:|----------:|
| **0.20** | **0.8812** | 0.5360 | 0.8111 | 0.6316 | ~27.3 |
| 0.25 | 0.8751 | 0.5277 | 0.8095 | 0.6308 | ~26.8 |
| 0.30 | 0.8656 | 0.5146 | 0.8078 | 0.6296 | ~27.0 |
| 0.3333 | 0.8571 | 0.5043 | 0.8068 | 0.6286 | ~27.0 |
| 0.40 | 0.8376 | 0.4813 | 0.8023 | 0.6263 | ~26.8 |

Artifacts: `bench_results/step3/sweep_1m/`.

## 8.8M @ `sl=256` (`cs=1.00`; encode-once)

| `w_s4` | ANN R@10 | vs target ≥0.887 | scan ms/q |
|-------:|---------:|:----------------:|----------:|
| **0.20** | **0.8988** | **PASS** (+0.012) | 263.9 |
| 0.25 | 0.8952 | PASS | 263.3 |
| 0.30 | 0.8871 | PASS (bare) | 261.7 |
| 0.3333 | 0.8809 | Stage-1 baseline | 262.0 |
| 0.40 | 0.8643 | miss | 262.1 |

Artifacts: `bench_results/step3/sweep_8.8m/cell_000`…`cell_004` (sl=256 complete).  
`sl=384` / `sl=512` grid was started then stopped after locking `w=0.20` so Step 4 benches could run; Stage 1 already covers `sl=512/1024` at the old weight.

## Notes

1. Moving MMSE out of `fold_tail` into `stratum4_weight` preserves encode bits and enables this zero-re-encode sweep.
2. `w_s4=0.3333` on the new path bit-matches Stage 1 R@10 (1M 0.8571; 8.8M 0.8809).
3. CS early-reject tolerance does not change top-k ranking here — shortlists fill before CS culls quality-critical survivors.
