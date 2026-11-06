# Note for collaborators

## What this is

A compact MRL-aware scan index: richer codes on the first 64 dims, coarser codes further out, skip candidates that cannot make the shortlist, then exact-embed rerank.

## What we found (and shipped)

On MS MARCO with EmbeddingGemma-2 (768-d, 1M docs), replacing the old min–max 4-bit head with **EDEN-prod (4-bit Lloyd–Max after rotation)** improved ranking slightly (R@10 **0.8837 vs 0.8812**) and did not slow the scan. That path is now the **default**.

Trying the same EDEN idea on the mid-band looked great on a tiny smoke set but **lost** ~1.7 points of recall on 1M, so it stays off.

## How to poke it

```bash
cargo build --release
./target/release/mrl-index demo --docs 2048
# or
./target/release/mrl-index bench-retrieval --data data/msmarco_gemma2/smoke_bin --shortlist 256
```

Start at [`docs/OVERVIEW.md`](OVERVIEW.md) and [`bench_results/eden/SUMMARY.md`](../bench_results/eden/SUMMARY.md).
