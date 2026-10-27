# Why Recall@10 was ~0.39 — and the fix

## Root cause

`L1_RABITQ` ranked candidates using **only the MSB bitplane** of the 4-bit MRL prefix.
Oracle on the SEC / nomic MRL corpus:

| Scorer | Shortlist@64 coverage of exact@10 |
|--------|-----------------------------------|
| MSB-only 4-bit | **~7%** |
| Full 4-bit recon | **~78%** |
| Exact prefix-192 IP | **~97.5%** |

Production path was effectively MSB-only → ~39% end-to-end.

## Fix

- Always accumulate **all four** prefix bitplanes for ranking (planes are always stored).
- Keep RaBitQ mid-band IP-scaled; store residual ‖x[64:]‖ for CS bounds.
- Drop harmful stratum-4 score jitter.
- Bench reports **shortlist coverage@10** and **reranked R@10** (design-faithful).

## Result (8192 SEC chunks, nomic-embed-text-v1.5 @384-d MRL)

| Metric | Before | After (sl=128, L1) |
|--------|--------|---------------------|
| shortlist_cov@10 | ~0.39 | **0.977** |
| reranked_R@10 | ~0.39 | **0.977** |
