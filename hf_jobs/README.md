# Hugging Face Jobs — MS MARCO + EmbeddingGemma-2 (option A resume)

Resume the **full** (~8.84M) corpus after Modal credits ran out.

## Work / output datasets

| Repo | Role |
|------|------|
| `astr010/msmarco-gemma2-full-work` | `meta/`, `text_shards/`, `emb_shards/` (salvaged + completed) |
| `astr010/msmarco-gemma2-full` | Final `corpus` / `queries` / `qrels` / `neighbors` / `manifest` |

## Steps

```bash
export HF_TOKEN=...   # Jobs + dataset write

# 1) Upload Modal salvage (after local download completes)
python3 hf_jobs/upload_salvage_to_hub.py \
  --repo astr010/msmarco-gemma2-full-work \
  --local data/msmarco_gemma2/full_modal_salvage

# 2) Embed missing shards (142..176) then finalize on GPU Jobs
python3 hf_jobs/launch_resume_full.py \
  --work-repo astr010/msmarco-gemma2-full-work \
  --out-repo astr010/msmarco-gemma2-full \
  --flavor a10g-large \
  --max-parallel 8 \
  --finalize-flavor a100-large

# 3) Download final suite locally
huggingface-cli download astr010/msmarco-gemma2-full \
  --repo-type dataset --local-dir data/msmarco_gemma2/full
python3 scripts/validate_msmarco_bench.py data/msmarco_gemma2/full
```

Single-shard manual job:

```bash
hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN \
  hf_jobs/embed_msmarco_gemma2_shard.py \
  --repo astr010/msmarco-gemma2-full-work --shard-idx 142
```
