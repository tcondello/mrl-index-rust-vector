# SEC 10-K MRL embeddings

Artifacts are **not** committed (too large). Generate them with Modal:

```bash
pip install 'modal>=0.73'
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
modal run modal/embed_sec10k_mrl.py --limit 50000 --download-dir data/sec10k_mrl
```

This downloads:

| File | Contents |
|------|----------|
| `manifest.json` | model, dims=384, `mrl: true`, dataset id |
| `texts.jsonl` | chunk text + ticker / year metadata |
| `embeddings.bin` | row-major `f32[n, 384]` from **nomic-embed-text-v1.5** (Matryoshka truncate) |

Dataset: [astr010/sec-10k-lsh-chunks](https://huggingface.co/datasets/astr010/sec-10k-lsh-chunks)  
Model: [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (true MRL)

Then:

```bash
./target/release/mrl-index bench-suite --embeddings data/sec10k_mrl --out bench_results/mrl_sec10k.json
```
