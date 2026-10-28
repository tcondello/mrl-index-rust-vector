# Why MS MARCO recall / nDCG looked bad — and what fixes it

## What we measured (shortlist **64**, L1)

| Scale | ANN R@10 | qrels R@10 | qrels nDCG@10 |
|-------|----------|------------|---------------|
| 1M | 0.585 | 0.716 | 0.575 |
| 8.8M | 0.597 | 0.431 | 0.286 |

Reranked R@10 == shortlist coverage@10 in every run → **the shortlist is the bottleneck**, not exact-IP rerank.

## Two separate ceilings

### 1. Index shortlist (fixable)

Design-faithful SEC / nomic benches used **shortlist 128**. We ran Gemma2@768 at **64**.

| 1M shortlist | ANN R@10 | qrels R@10 | nDCG@10 |
|--------------|----------|------------|---------|
| 64 | 0.585 | 0.716 | 0.575 |
| 128 | 0.675 | 0.753 | 0.596 |
| 256 | 0.753 | 0.780 | 0.612 |
| **1024** | **0.873** | **0.806** | **0.629** |

`full_64b` vs `l1_rabitq` at shortlist 256: **identical** (L1 already ranks with full 4-bit prefix + RaBitQ).

### 2. Embedding → qrels oracle (not an index bug)

Exact IP top-k in embedding space vs BeIR qrels:

| Scale | oracle qrels R@10 | oracle hit@100 |
|-------|-------------------|----------------|
| smoke | 0.984 | 1.000 |
| 1M | **0.829** | 0.967 |
| 8.8M | **0.547** | 0.849 |

On the full 8.8M corpus, **even a perfect ANN + perfect rerank of the embedding neighbors only recovers ~55% of qrels @10**. Dense EmbeddingGemma-2@768 simply does not put every MS MARCO relevant passage in the top-10 over 8.8M distractors. Sparse labels (~1 relevant / query) also make nDCG@10 look harsh.

So published **8.8M qrels nDCG@10 ≈ 0.29 at shortlist 64** was mostly (a) undersized shortlist and (b) model/corpus IR ceiling — not a regression from the SWAR/SIMD merge (metrics matched bit-identically).

## Config changes

- `EngineConfig.shortlist_cand` allowed range: `16..=4096` (was `..=256`)
- `bench-retrieval` default shortlist: **256** (was 64)

Prefer `--shortlist 1024` when optimizing for quality on Gemma2@768.

## Full 8.8M @ shortlist 1024 (quality config)

| Metric | sl64 | **sl1024** | embedding oracle |
|--------|------|------------|------------------|
| ANN R@10 | 0.597 | **0.887** | 1.000 |
| qrels R@10 | 0.431 | **0.522** | 0.547 |
| qrels nDCG@10 | 0.286 | **0.330** | — |
| scan ms/q | 208.6 | 222.2 | — |

At shortlist 1024, qrels R@10 reaches **95%** of the EmbeddingGemma-2 oracle ceiling on the full corpus.
