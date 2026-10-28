# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.7.1",
#   "torchvision==0.22.1",
#   "torchaudio==2.7.1",
#   "sentence-transformers[image]>=5.1.0",
#   "transformers>=4.56.0",
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
"""Assemble final MS MARCO + EmbeddingGemma-2 parquet suite on HF Jobs.

Reads work repo (text/emb shards + meta), writes final/{corpus,queries,qrels,neighbors,manifest}.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"
EMBED_DIMS = 768
QUERY_BATCH = 64
K_MAX = 100


def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_ID,
        truncate_dim=EMBED_DIMS,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": "bfloat16"},
        trust_remote_code=True,
    )


def fixed_list_to_numpy(col):
    import numpy as np

    chunked = col.combine_chunks()
    flat = chunked.values.to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float32).reshape(len(chunked), EMBED_DIMS)


def main() -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--work-repo", required=True)
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--k-max", type=int, default=K_MAX)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.out_repo, repo_type="dataset", exist_ok=True, private=False)

    work_dir = Path(
        snapshot_download(
            repo_id=args.work_repo,
            repo_type="dataset",
            token=token,
            allow_patterns=["meta/*", "emb_shards/*"],
        )
    )
    prepare = json.loads((work_dir / "meta" / "prepare_manifest.json").read_text())
    n = int(prepare["n_corpus"])
    n_shards = int(prepare["n_shards"])
    print(f"finalize n={n} shards={n_shards}")

    # Verify all emb shards present.
    missing = []
    for s in range(n_shards):
        p = work_dir / "emb_shards" / f"shard_{s:05d}.parquet"
        if not p.exists():
            missing.append(s)
    if missing:
        raise SystemExit(f"missing emb shards ({len(missing)}): {missing[:20]}...")

    out = Path("/tmp/final")
    out.mkdir(parents=True, exist_ok=True)

    # Stream-concat corpus
    corpus_path = out / "corpus.parquet"
    writer = None
    total = 0
    try:
        for s in range(n_shards):
            t = pq.read_table(work_dir / "emb_shards" / f"shard_{s:05d}.parquet")
            if writer is None:
                writer = pq.ParquetWriter(corpus_path, t.schema, compression="zstd")
            writer.write_table(t)
            total += t.num_rows
            print(f"append shard {s} rows={t.num_rows}")
            del t
    finally:
        if writer is not None:
            writer.close()
    assert total == n, (total, n)

    # Queries
    qtext = pq.read_table(work_dir / "meta" / "queries_text.parquet")
    qids = qtext.column("query_id").to_pylist()
    texts = qtext.column("text").to_pylist()
    print(f"embedding {len(texts)} queries…")
    model = load_model()
    # Bake query instruction into the string to avoid multimodal URL/image fetch.
    q_inputs = [f"task: search result | query: {t}" for t in texts]
    emb = model.encode(
        q_inputs,
        batch_size=QUERY_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    qemb = np.asarray(emb, dtype=np.float32)
    if qemb.shape[1] != EMBED_DIMS:
        qemb = qemb[:, :EMBED_DIMS]
        norms = np.linalg.norm(qemb, axis=1, keepdims=True).clip(min=1e-12)
        qemb = qemb / norms
    pq.write_table(
        pa.table(
            {
                "query_id": pa.array(qids, type=pa.string()),
                "text": pa.array(texts, type=pa.string()),
                "embedding": pa.array(qemb.tolist(), type=pa.list_(pa.float32(), EMBED_DIMS)),
            }
        ),
        out / "queries.parquet",
        compression="zstd",
    )

    # Copy qrels
    (out / "qrels.parquet").write_bytes((work_dir / "meta" / "qrels.parquet").read_bytes())

    # Exact neighbors
    print("loading corpus embeddings for GT…")
    corpus = np.zeros((n, EMBED_DIMS), dtype=np.float32)
    id_map: dict[int, str] = {}
    for s in range(n_shards):
        shard = pq.read_table(
            work_dir / "emb_shards" / f"shard_{s:05d}.parquet",
            columns=["row_id", "corpus_id", "embedding"],
        )
        row_ids = shard.column("row_id").to_numpy()
        emb_np = fixed_list_to_numpy(shard.column("embedding"))
        corpus[row_ids] = emb_np
        for rid, cid in zip(row_ids.tolist(), shard.column("corpus_id").to_pylist()):
            id_map[int(rid)] = cid
        print(f"loaded emb shard {s}/{n_shards}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_t = torch.from_numpy(qemb).to(device)
    k_max = args.k_max
    block = 200_000 if n >= 200_000 else n
    top_scores = torch.full((len(qids), k_max), -1e9, device=device)
    top_indices = torch.full((len(qids), k_max), -1, device=device, dtype=torch.long)
    print(f"exact top-{k_max} Q={len(qids)} N={n} block={block}")
    for start in range(0, n, block):
        end = min(start + block, n)
        c_t = torch.from_numpy(corpus[start:end]).to(device)
        scores = q_t @ c_t.T
        cat_scores = torch.cat([top_scores, scores], dim=1)
        cat_indices = torch.cat(
            [
                top_indices,
                torch.arange(start, end, device=device).unsqueeze(0).expand(len(qids), -1),
            ],
            dim=1,
        )
        top_scores, sel = torch.topk(cat_scores, k=k_max, dim=1)
        top_indices = torch.gather(cat_indices, 1, sel)
        del c_t, scores, cat_scores, cat_indices, sel
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  scanned {end}/{n}")

    top_scores_np = top_scores.float().cpu().numpy()
    top_indices_np = top_indices.cpu().numpy()
    neighbor_ids, neighbor_cids, neighbor_scores = [], [], []
    for i in range(len(qids)):
        ids = [int(x) for x in top_indices_np[i].tolist()]
        neighbor_ids.append(ids)
        neighbor_cids.append([id_map[j] for j in ids])
        neighbor_scores.append([float(x) for x in top_scores_np[i].tolist()])
    pq.write_table(
        pa.table(
            {
                "query_id": pa.array(qids, type=pa.string()),
                "neighbor_row_ids": pa.array(neighbor_ids, type=pa.list_(pa.int32())),
                "neighbor_corpus_ids": pa.array(neighbor_cids, type=pa.list_(pa.string())),
                "neighbor_scores": pa.array(neighbor_scores, type=pa.list_(pa.float32())),
            }
        ),
        out / "neighbors.parquet",
        compression="zstd",
    )

    manifest = {
        **prepare,
        "files": {
            "corpus": "corpus.parquet",
            "queries": "queries.parquet",
            "qrels": "qrels.parquet",
            "neighbors": "neighbors.parquet",
        },
        "dtype": "f32",
        "embedding_column": "embedding",
        "completed_at_unix": int(time.time()),
        "producer": "hf_jobs/finalize_msmarco_gemma2.py",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"uploading final suite → {args.out_repo}")
    api.upload_folder(
        folder_path=str(out),
        repo_id=args.out_repo,
        repo_type="dataset",
        token=token,
    )
    print("done", args.out_repo)


if __name__ == "__main__":
    main()
