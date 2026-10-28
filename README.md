# mrl-index

Local-first **bitplane-transposed** vector retrieval engine (spec v3.5).

Designed for personal workstations: keep a compact in-memory / NVMe scan index (16–64 bytes per vector), store **zero** full-precision L2 embeddings on disk, and rerank a short candidate list with one batched local embedding pass over raw text.

## Pipeline

```
Query ─► embed ─► PaxBlock64 bitplane scan (XOR/POPCNT-style) ─► top K_cand
              ─► passages.idx O(1) text fetch ─► 1× batched embed rerank ─► top-K
```

Ingest goes through an append-only ring buffer (queryable immediately) and flushes every 64 vectors into a 4 KiB page-aligned `PaxBlock64` macro-block.

## Quick start

```bash
cargo build --release
./target/release/mrl-index demo --docs 2048
./target/release/mrl-index bench --blocks 2048
```

### Library / CLI

```bash
# create an index
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
| `passages.jsonl` | Append-only raw text (primary store) |
| `passages.idx` | Packed `{offset:u64, len:u32}` × N |
| `segments/000001.pax` | Contiguous `PaxBlock64` (4096 B each) |

Each `PaxBlock64` holds 64 vectors at **64 B/vec**: header, f16 scale/offset/tail_norm, and six progressive bitplane strata (MRL sign → RaBitQ 1-bit → MSB refinements).

## Tuning

See `EngineConfig` (`src/config.rs`) for the four axes from the design doc:

- **Memory:** `strata_depth`, `slab_residency`, `numa_sharding`
- **Pruning:** `shortlist_cand`, `cs_tolerance`, `rabitq_confidence`
- **Latency:** `jit_batch_size`, `io_read_workers`, `max_latency_budget_ms`
- **Lifecycle:** `ring_buffer_size`, `tombstone_compact_ratio`, `temporal_filter_enable`

## Embeddings

The default `HashingEmbedder` is a deterministic CPU stand-in so the full scan → fetch → rerank loop runs without GPU drivers. Swap in MLX / CUDA / ONNX by implementing the `Embedder` trait (`src/rerank.rs`).

## Benchmarks (MRL embeddings required)

The design assumes **Matryoshka** embeddings. The preferred large-scale suite is
**MS MARCO** embedded with **EmbeddingGemma 2** @ 768-d (parquet + dual ground truth):

```bash
pip install -r modal/requirements.txt
./scripts/check_modal_auth.sh
modal run modal/embed_msmarco_gemma2.py --scale 1m --download-dir data/msmarco_gemma2/1m
modal run modal/embed_msmarco_gemma2.py --scale full --download-dir data/msmarco_gemma2/full
python3 scripts/validate_msmarco_bench.py data/msmarco_gemma2/1m
```

- Dataset: [`BeIR/msmarco`](https://huggingface.co/datasets/BeIR/msmarco) + [`BeIR/msmarco-qrels`](https://huggingface.co/datasets/BeIR/msmarco-qrels)
- Model: [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) (768-d MRL)
- Artifacts: `corpus` / `queries` / `qrels` / `neighbors` parquet — see `data/msmarco_gemma2/README.md`
- Note: MS MARCO has **8.84M** passages (no native 10M split); `full` uses the entire corpus

Legacy SEC 10-K @ nomic 384-d path (still supported by `bench-suite`):

```bash
modal run modal/embed_sec10k_mrl.py --limit 50000 --download-dir data/sec10k_mrl
cargo build --release
./target/release/mrl-index bench-suite \
  --embeddings data/sec10k_mrl \
  --out bench_results/mrl_sec10k.json
```

Plumbing-only isotropic vectors (not an MRL evaluation):

```bash
./target/release/mrl-index bench-suite --synthetic --out bench_results/synthetic.json
```

See `bench_results/DELTA.md` for the earlier baseline → fix comparison (synthetic).

## Develop

```bash
cargo test
```

## License

MIT
