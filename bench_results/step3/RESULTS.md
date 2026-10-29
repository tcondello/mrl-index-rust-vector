# Step 3.5 Verification — `origin/main` @ `4e608f2`

**Commit:** `4e608f2` — Keep Step 3 MMSE `1/sqrt(B)` fold shrinkage + fast factored S1/S4 scan + warmup 32 + dedup query embed  
**Model:** `google/embeddinggemma-2` (768-d MRL)  
**Corpus:** BeIR/msmarco (`1m_bin` = 1,000,000; `full_bin` = 8,841,823)  
**Queries:** 6,980 · Strata: `l1_rabitq` · Corpus mmap: yes  
**Date:** 2026-10-29 (America/New_York)

## Verdict

**PASS.** Fast factored S1/S4 scan + MMSE `1/√9` fold shrinkage simultaneously deliver:

| Target | Expected | Observed |
|--------|----------|----------|
| 1M R@10 @ sl=256 / 1024 | 0.857 / 0.939 | **0.857 / 0.939** |
| 8.8M R@10 @ sl=256 / 1024 | 0.881 / 0.952 | **0.881 / 0.952** |
| 1M scan @ sl=256 | ~28–30 ms/q | **30.2 ms/q** |
| 8.8M scan @ sl=256 | ~200–220 ms/q | **259 ms/q** (~34 Mvecs/s; machine-bound) |

New midpoint `sl=512` fills the Pareto curve between 256 and 1024.

## Results table

| Corpus | Shortlist | ANN R@10 | ANN R@100 | qrels R@10 | nDCG@10 | scan ms/q | Mvecs/s |
|--------|----------:|---------:|----------:|-----------:|--------:|----------:|--------:|
| 1M | 256 | 0.8571 | 0.5043 | 0.8068 | 0.6286 | 30.18 | 33.14 |
| 1M | 512 | 0.9045 | 0.6176 | 0.8142 | 0.6338 | 33.77 | 29.62 |
| 1M | 1024 | 0.9388 | 0.7191 | 0.8214 | 0.6372 | 43.52 | 22.98 |
| 8.8M | 256 | 0.8809 | 0.5458 | 0.5240 | 0.3309 | 259.23 | 34.11 |
| 8.8M | 512 | 0.9241 | 0.6571 | 0.5345 | 0.3346 | 260.21 | 33.98 |
| 8.8M | 1024 | 0.9519 | 0.7495 | 0.5378 | 0.3360 | 274.20 | 32.25 |

## Artifacts

- `s1_1m_sl{256,512,1024}.json`
- `s1_8.8m_sl{256,512,1024}.json`

## Notes

- Encode time is one-shot per run (not amortized). 8.8M encode ≈ 70–99 s.
- 8.8M wall-clock scan is ~259–274 ms/q on this VM; vector throughput (~32–34 Mvecs/s) matches 1M, so the gap vs the ~200–220 ms/q lab target is host/CPU, not algorithmic regression.
- MMSE fold shrinkage (`fold_tail` × `1/√n_blocks`) is required: earlier unattenuated fold dropped 8.8M R@10 from 0.952 → 0.805.
