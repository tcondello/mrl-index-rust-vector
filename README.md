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

The design assumes **Matryoshka** embeddings. The primary suite loads real MRL
vectors for SEC 10-K chunks, generated on Modal:

```bash
pip install -r modal/requirements.txt
./scripts/check_modal_auth.sh   # MODAL_* in Cursor Environment secrets; start a new agent run
modal run modal/embed_sec10k_mrl.py --limit 50000 --download-dir data/sec10k_mrl

# or CPU fallback:
pip install -r modal/requirements-embed.txt
python3 scripts/embed_sec10k_local.py --limit 8192 --out data/sec10k_mrl

cargo build --release
./target/release/mrl-index bench-suite \
  --embeddings data/sec10k_mrl \
  --out bench_results/mrl_sec10k.json
```

- Dataset: [`astr010/sec-10k-lsh-chunks`](https://huggingface.co/datasets/astr010/sec-10k-lsh-chunks)
- Model: [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (MRL, truncated to 384-d)

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
