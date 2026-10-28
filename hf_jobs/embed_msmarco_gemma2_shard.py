# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.7.1",
#   "torchvision==0.22.1",
#   "torchaudio==2.7.1",
#   "sentence-transformers[image]>=5.1.0",
#   "transformers>=4.56.0",
#   "datasets>=3.2.0",
#   "huggingface-hub>=0.30.0",
#   "hf_transfer>=0.1.9",
#   "pyarrow>=17.0.0",
#   "numpy",
#   "einops",
#   "accelerate",
#   "protobuf",
#   "pillow",
# ]
# ///
"""Embed one MS MARCO text shard with EmbeddingGemma-2 and upload to Hub.

Designed for Hugging Face Jobs (+ resume after Modal salvage).

Example:
  hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN \\
    hf_jobs/embed_msmarco_gemma2_shard.py \\
    --repo astr010/msmarco-gemma2-full-work \\
    --shard-idx 142
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"
EMBED_DIMS = 768
DOC_BATCH = 64


def load_model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID,
        truncate_dim=EMBED_DIMS,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": "bfloat16"},
        trust_remote_code=True,
    )
    model.eval()
    return model


def encode_documents(model, titles: list[str], texts: list[str]):
    """Encode docs as plain text.

    Important: do **not** pass `prompt=` into multimodal EmbeddingGemma-2 for
    raw MS MARCO passages — URLs in the text get treated as images to fetch.
    Bake the retrieval document instruction into the string instead.
    """
    import numpy as np

    inputs = [
        f"title: {(t or '').strip() or 'none'} | text: {x}"
        for t, x in zip(titles, texts)
    ]
    emb = model.encode(
        inputs,
        batch_size=DOC_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    arr = np.asarray(emb, dtype=np.float32)
    if arr.shape[1] != EMBED_DIMS:
        arr = arr[:, :EMBED_DIMS]
        norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
        arr = arr / norms
    return arr


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hub dataset repo for work artifacts")
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="Re-embed even if emb shard exists")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    shard_name = f"shard_{args.shard_idx:05d}.parquet"
    emb_path_in_repo = f"emb_shards/{shard_name}"

    if not args.force:
        try:
            files = api.list_repo_files(args.repo, repo_type="dataset")
            if emb_path_in_repo in files:
                print(f"skip existing {emb_path_in_repo}")
                return
        except Exception as e:
            print("list_repo_files warn:", e)

    text_local = hf_hub_download(
        repo_id=args.repo,
        filename=f"text_shards/{shard_name}",
        repo_type="dataset",
        token=token,
    )
    table = pq.read_table(text_local)
    titles = table.column("title").to_pylist()
    texts = table.column("text").to_pylist()
    row_ids = table.column("row_id").to_pylist()
    corpus_ids = table.column("corpus_id").to_pylist()
    flags = table.column("is_qrels_relevant").to_pylist()
    print(f"embedding shard {args.shard_idx} n={len(texts)}")

    t0 = time.time()
    model = load_model()
    emb = encode_documents(model, titles, texts)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s ({len(texts)/max(elapsed,1e-6):.1f} docs/s)")

    out_dir = Path("/tmp/emb_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / shard_name
    emb_list = [emb[i].tolist() for i in range(emb.shape[0])]
    out_table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.int32()),
            "corpus_id": pa.array(corpus_ids, type=pa.string()),
            "title": pa.array(titles, type=pa.string()),
            "text": pa.array(texts, type=pa.string()),
            "is_qrels_relevant": pa.array(flags, type=pa.bool_()),
            "embedding": pa.array(emb_list, type=pa.list_(pa.float32(), EMBED_DIMS)),
        }
    )
    pq.write_table(out_table, out_path, compression="zstd")
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=emb_path_in_repo,
        repo_id=args.repo,
        repo_type="dataset",
        token=token,
    )
    print(f"uploaded {emb_path_in_repo}")


if __name__ == "__main__":
    main()
