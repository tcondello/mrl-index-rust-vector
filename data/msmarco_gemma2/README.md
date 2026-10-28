# MS MARCO + EmbeddingGemma 2 benchmark corpora

Proper retrieval benchmark artifacts for `mrl-index` (not SEC self-retrieval).

| Scale | Corpus size | Notes |
|-------|-------------|--------|
| `smoke` | 2,048 | CI / pipeline check |
| `1m` | 1,000,000 | All MS MARCO dev qrels docs + distractors |
| `full` | 8,841,823 | Entire BeIR/msmarco passage corpus (no native 10M split) |

## Contents (per scale directory)

| File | Role |
|------|------|
| `corpus.parquet` | `row_id`, `corpus_id`, `title`, `text`, `is_qrels_relevant`, `embedding[768]` |
| `queries.parquet` | MS MARCO dev queries with `embedding[768]` |
| `qrels.parquet` | Human relevance labels (`query_id`, `corpus_id`, `row_id`, `score`) |
| `neighbors.parquet` | Exact top-100 IP neighbors (ANN ground truth) |
| `manifest.json` | Model / dims / counts / prompts / seeds |

- **Model:** [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) @ **768-d** (MRL-capable)
- **Corpus / queries:** [`BeIR/msmarco`](https://huggingface.co/datasets/BeIR/msmarco)
- **Qrels:** [`BeIR/msmarco-qrels`](https://huggingface.co/datasets/BeIR/msmarco-qrels) `validation` (dev)
- **Metric:** inner product on L2-normalized vectors (= cosine)

## Generate (Modal GPU)

```bash
pip install -r modal/requirements.txt
./scripts/check_modal_auth.sh

# smoke
modal run modal/embed_msmarco_gemma2.py \
  --scale smoke --download-dir data/msmarco_gemma2/smoke

# 1M
modal run modal/embed_msmarco_gemma2.py \
  --scale 1m --download-dir data/msmarco_gemma2/1m

# full ~8.84M (large; multi-hour)
modal run modal/embed_msmarco_gemma2.py \
  --scale full --download-dir data/msmarco_gemma2/full
```

Re-download only (if volume already has finals):

```bash
modal run modal/embed_msmarco_gemma2.py --scale 1m --skip-run \
  --download-dir data/msmarco_gemma2/1m
```

## Validate

```bash
pip install -r modal/requirements-bench-data.txt
python3 scripts/validate_msmarco_bench.py data/msmarco_gemma2/1m
```

## Ground truth layers

1. **IR qrels** — official MS MARCO relevance for nDCG / Recall against human labels.
2. **Exact neighbors** — offline brute-force top-100 in embedding space for ANN/index recall (shortlist coverage, reranked R@k).
