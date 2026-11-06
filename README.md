# mrl-index

Local-first **bitplane-transposed** vector retrieval for [Matryoshka](https://arxiv.org/abs/2205.13147) embeddings.

Keep a compact in-memory / NVMe scan index (~**64 B/vector**), store **zero** full-precision L2 embeddings on disk, and rerank a short candidate list with one batched local embedding pass over raw text.

**License:** MIT (`LICENSE`)

## Idea

MRL models put most of the useful signal in the first dimensions. We spend more bits on the head of the vector and fewer on the noisy tail. At query time we score the head first; if a document already cannot finish in the shortlist—even if the remaining bits were as good as possible—we skip reading the rest of its code.

Default prefix codebook is **EDEN-prod** (4-bit Lloyd–Max after center / normalize / structured Hadamard). Mid-band stays 1-bit RaBitQ; the high-dim tail is a folded 1-bit sketch. See [`docs/OVERVIEW.md`](docs/OVERVIEW.md).

## Pipeline

```
Query ─► embed ─► PaxBlock64 bitplane scan ─► top K_cand
              ─► passages.idx O(1) text fetch ─► 1× batched embed rerank ─► top-K
```

Ingest goes through an append-only ring buffer (queryable immediately) and flushes every 64 vectors into a 4 KiB page-aligned `PaxBlock64` macro-block. Fitted EDEN means live in `eden_prefix.json` next to `manifest.json`.

## Quick start

```bash
cargo build --release
./target/release/mrl-index demo --docs 2048
./target/release/mrl-index bench --blocks 2048
```

### Library / CLI

```bash
# create an index (default: EDEN-prod prefix @ 4 bits)
mrl-index init ./my-index --dims 384 --strata full_64b

# ingest lines from a file (one passage per line)
mrl-index ingest ./my-index --file notes.txt --flush

# query
mrl-index query ./my-index "bitplane scan latency"
```

## Layout

| Path | Role |
|------|------|
| `manifest.json` | Config + segment bookkeeping |
| `eden_prefix.json` | EDEN-prod mean / seed / bit-depth (when enabled) |
| `passages.bin` / `.idx` | Packed text + O(1) offsets |
| `segments/000001.pax` | Contiguous `PaxBlock64` (4096 B each) |

Each `PaxBlock64` holds 64 vectors at **64 B/vec**: header, f16 meta, and progressive bitplane strata.

## Results snapshot

On MS MARCO + EmbeddingGemma-2 @768 (1M docs, shortlist 256):

- EDEN-prod prefix vs legacy min–max: **R@10 0.8837 vs 0.8812**, scan ~**0.9×** latency
- Dual EDEN on the mid-band **regressed** on 1M (−1.7pp) — left off by default

Artifacts: `bench_results/eden/`.

## Benchmarks (MRL embeddings required)

Preferred suite: **MS MARCO** + **EmbeddingGemma 2** @ 768-d:

```bash
# Convert Hub parquet → *_bin (see data/msmarco_gemma2/README.md)
python3 scripts/parquet_to_bench_bin.py data/msmarco_gemma2/smoke data/msmarco_gemma2/smoke_bin

./target/release/mrl-index bench-retrieval \
  --label smoke --data data/msmarco_gemma2/smoke_bin \
  --shortlist 256 --out reports/bench_smoke.json
```

## Tuning

See `EngineConfig` (`src/config.rs`):

- **Codec:** `prefix_codec` (`eden_prod` default, or `minmax4`), `mid_codec` (`rabitq` default)
- **Memory:** `strata_depth`, `slab_residency`, `numa_sharding`
- **Pruning:** `shortlist_cand`, `cs_tolerance`
- **Latency:** `jit_batch_size`, `io_read_workers`, `max_latency_budget_ms`

## Embeddings

The default `HashingEmbedder` is a deterministic CPU stand-in so the full scan → fetch → rerank loop runs without GPU drivers. Swap in MLX / CUDA / ONNX by implementing the `Embedder` trait (`src/rerank.rs`). Optional local Gemma-2 path: `--gemma2` / `GEMMA2_SNAPSHOT`.

## Docs

| Doc | Topic |
|-----|--------|
| [`docs/OVERVIEW.md`](docs/OVERVIEW.md) | Architecture for collaborators |
| [`docs/PRIMITIVES_AND_COMPOSITION.md`](docs/PRIMITIVES_AND_COMPOSITION.md) | Scan stages / pipeline composition |
| [`bench_results/eden/SUMMARY.md`](bench_results/eden/SUMMARY.md) | EDEN-prod probe decision record |
