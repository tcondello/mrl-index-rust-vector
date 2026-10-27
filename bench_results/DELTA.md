# Benchmark delta: baseline → post_fix

Suite: 8192 vectors × 32 queries × 384-d, shortlist=64, `L1_RABITQ`; ingest 4096 docs.

| Metric | Baseline (buggy) | Post-fix | Δ |
|--------|------------------|------------|---|
| Recall@10 | 0.1281 | 0.2000 | **+56.1%** |
| Recall@shortlist | 0.0869 | 0.1509 | **+73.6%** |
| Mean scan latency | 0.659 ms | 1.300 ms | +97% (more survivors pass CS; expected) |
| Scan throughput | 12.43 Mvecs/s | 6.30 Mvecs/s | — |
| Ingest 4096 docs | 15.3 ms | 16.7 ms | ~flat at this size (embed-bound) |

## Fixes applied after baseline

1. **Stratum-1 scale** — RaBitQ contribution is `FWHT(q)·(±√(2/π)/√D)` instead of raw `±|q_d|`, so shortlist τ stays in IP units and CS pruning no longer false-drops later blocks.
2. **Rotation** — replaced per-coordinate Rademacher (no-op for sign agreement) with orthonormal FWHT on the 128-d mid-band.
3. **Ingest reload** — `SegmentReader::append_live` is O(1) per flush; removed full-segment remap/recopy in `commit_block` (was O(N²)).

Reproduce:

```bash
cargo build --release
./target/release/mrl-index bench-suite --label run --out bench_results/run.json
```
