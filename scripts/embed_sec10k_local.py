#!/usr/bin/env python3
"""Local CPU fallback — same artifacts as modal/embed_sec10k_mrl.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASET_ID = "astr010/sec-10k-lsh-chunks"
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
TRUNCATE_DIM = 384


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/sec10k_mrl"))
    args = p.parse_args()

    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print(f"loading {DATASET_ID} (streaming, limit={args.limit})…", flush=True)
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    rows = []
    for ex in ds:
        if ex.get("is_boilerplate"):
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
        if len(rows) >= args.limit:
            break

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    rows = [rows[int(i)] for i in order]
    for i, r in enumerate(rows):
        r["id"] = i

    print(f"loading {MODEL_ID} truncate_dim={TRUNCATE_DIM}…", flush=True)
    model = SentenceTransformer(
        MODEL_ID, trust_remote_code=True, truncate_dim=TRUNCATE_DIM
    )
    texts = [f"search_document: {r['text']}" for r in rows]
    print(f"encoding {len(texts)}…", flush=True)
    emb = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    arr = np.asarray(emb, dtype=np.float32)
    if arr.shape[1] != TRUNCATE_DIM:
        arr = arr[:, :TRUNCATE_DIM]
        norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
        arr = arr / norms

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "embeddings.bin").write_bytes(arr.tobytes(order="C"))
    with (args.out / "texts.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": DATASET_ID,
        "model": MODEL_ID,
        "truncate_dim": TRUNCATE_DIM,
        "dims": TRUNCATE_DIM,
        "n": len(rows),
        "skip_boilerplate": True,
        "seed": args.seed,
        "mrl": True,
        "embeddings_file": "embeddings.bin",
        "texts_file": "texts.jsonl",
        "dtype": "f32",
        "layout": "row_major",
        "document_prefix": "search_document: ",
        "query_prefix": "search_query: ",
        "generated_by": "scripts/embed_sec10k_local.py",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done → {args.out} n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
