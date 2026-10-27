"""
Embed astr010/sec-10k-lsh-chunks with a true Matryoshka model on Modal GPUs.

Model: nomic-ai/nomic-embed-text-v1.5 (MRL; truncate_dim=384 to match mrl-index).
Dataset: https://huggingface.co/datasets/astr010/sec-10k-lsh-chunks

Auth:
  export MODAL_TOKEN_ID=...
  export MODAL_TOKEN_SECRET=...
  pip install 'modal>=0.73'
  modal run modal/embed_sec10k_mrl.py --limit 50000

Outputs (./data/sec10k_mrl/):
  manifest.json   — model / dims / count / MRL flags
  texts.jsonl     — id, text, ticker, …
  embeddings.bin  — row-major f32 [n, dims]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "mrl-index-sec10k-embed"
DATASET_ID = "astr010/sec-10k-lsh-chunks"
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
TRUNCATE_DIM = 384
BATCH_SIZE = 64
GPU = "A10G"

VOL = modal.Volume.from_name("mrl-index-sec10k-embeddings", create_if_missing=True)
OUT_DIR = "/vol/sec10k_mrl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "sentence-transformers==3.3.1",
        "transformers==4.46.3",
        "datasets==3.2.0",
        "huggingface-hub==0.26.2",
        "einops",
        "numpy",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=GPU,
    timeout=60 * 60 * 6,
    memory=32768,
    volumes={"/vol": VOL},
)
def prepare_and_embed(
    limit: int = 50_000,
    seed: int = 42,
    skip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Sample non-boilerplate 10-K chunks, embed with MRL model, write volume."""
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print(f"loading {DATASET_ID} (streaming until {limit} content chunks)…")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    rows: list[dict[str, Any]] = []
    for ex in ds:
        if skip_boilerplate and ex.get("is_boilerplate"):
            continue
        text = (ex.get("text") or "").strip()
        if len(text) < 80:
            continue
        rows.append(
            {
                "id": len(rows),
                "ticker": ex.get("ticker"),
                "cik": ex.get("cik"),
                "fiscal_year": ex.get("fiscal_year"),
                "chunk_id": ex.get("chunk_id"),
                "is_table": bool(ex.get("is_table")),
                "token_count": ex.get("token_count"),
                "text": text[:4000],
            }
        )
        if len(rows) >= limit:
            break

    if not rows:
        raise RuntimeError("no rows sampled from dataset")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    rows = [rows[int(i)] for i in order]
    # Re-id after shuffle so id == row index in embeddings.bin
    for i, r in enumerate(rows):
        r["id"] = i

    print(f"loading MRL model {MODEL_ID} truncate_dim={TRUNCATE_DIM}")
    model = SentenceTransformer(
        MODEL_ID,
        trust_remote_code=True,
        truncate_dim=TRUNCATE_DIM,
    )
    model.eval()

    texts = [f"search_document: {r['text']}" for r in rows]
    print(f"encoding {len(texts)} documents…")
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    arr = np.asarray(emb, dtype=np.float32)
    if arr.shape[1] != TRUNCATE_DIM:
        # Fallback truncate + renormalize if truncate_dim was ignored.
        arr = arr[:, :TRUNCATE_DIM]
        norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
        arr = arr / norms
    assert arr.shape == (len(rows), TRUNCATE_DIM), arr.shape

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "embeddings.bin").write_bytes(arr.tobytes(order="C"))
    with (out / "texts.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": DATASET_ID,
        "model": MODEL_ID,
        "truncate_dim": TRUNCATE_DIM,
        "dims": TRUNCATE_DIM,
        "n": len(rows),
        "skip_boilerplate": skip_boilerplate,
        "seed": seed,
        "mrl": True,
        "embeddings_file": "embeddings.bin",
        "texts_file": "texts.jsonl",
        "dtype": "f32",
        "layout": "row_major",
        "document_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    VOL.commit()
    print(f"wrote {OUT_DIR} n={len(rows)} dims={TRUNCATE_DIM}")
    return manifest


@app.function(volumes={"/vol": VOL}, timeout=60 * 30, memory=8192)
def export_bytes() -> dict[str, bytes]:
    out = Path(OUT_DIR)
    return {
        name: (out / name).read_bytes()
        for name in ("manifest.json", "texts.jsonl", "embeddings.bin")
    }


@app.local_entrypoint()
def main(limit: int = 50_000, seed: int = 42, download_dir: str = "data/sec10k_mrl"):
    """
      modal run modal/embed_sec10k_mrl.py --limit 50000
      modal run modal/embed_sec10k_mrl.py --limit 5000 --download-dir data/sec10k_mrl
    """
    manifest = prepare_and_embed.remote(limit=limit, seed=seed)
    print("remote manifest:", json.dumps(manifest, indent=2))

    dest = Path(download_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in export_bytes.remote().items():
        path = dest / name
        path.write_bytes(data)
        print(f"downloaded {path} ({len(data):,} bytes)")
    print("done — run: mrl-index bench-suite --embeddings", dest)
