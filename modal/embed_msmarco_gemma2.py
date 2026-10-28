"""
Build MS MARCO retrieval benchmark corpora with EmbeddingGemma 2 (768-d) on Modal.

Produces locally reusable parquet artifacts with:
  - corpus embeddings (documents)
  - query embeddings (MS MARCO queries that appear in qrels)
  - IR qrels (human relevance labels)
  - exact top-K neighbor ground truth (inner product on L2-normalized vectors)

Scales:
  - 1m: 1_000_000 docs (all qrels-relevant docs included, then filled with distractors)
  - full: entire BeIR/msmarco corpus (~8.84M; MS MARCO has no 10M corpus)

Auth:
  export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
  pip install -r modal/requirements.txt
  ./scripts/check_modal_auth.sh

Examples:
  modal run modal/embed_msmarco_gemma2.py --scale smoke --download-dir data/msmarco_gemma2/smoke
  modal run modal/embed_msmarco_gemma2.py --scale 1m --download-dir data/msmarco_gemma2/1m
  modal run modal/embed_msmarco_gemma2.py --scale full --download-dir data/msmarco_gemma2/full
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Literal

import modal

APP_NAME = "mrl-index-msmarco-gemma2"
CORPUS_DATASET = "BeIR/msmarco"
QRELS_DATASET = "BeIR/msmarco-qrels"
MODEL_ID = "google/embeddinggemma-2"
EMBED_DIMS = 768
QRELS_SPLIT = "validation"  # MS MARCO "dev" in BEIR
K_MAX = 100
SEED = 42
SHARD_SIZE = 50_000
DOC_BATCH = 64
QUERY_BATCH = 64
GPU = "A100-40GB"

Scale = Literal["smoke", "1m", "full"]

SCALE_N: dict[str, int | None] = {
    # Must be >= #unique MS MARCO dev qrels docs (~7.4k), or use qrels subsample below.
    "smoke": 8_192,
    "1m": 1_000_000,
    "full": None,  # entire corpus
}
# For tiny smokes, keep only this many qrels queries (0 = keep all).
SMOKE_MAX_QUERIES = 128

VOL = modal.Volume.from_name("mrl-index-msmarco-gemma2", create_if_missing=True)
VOL_ROOT = Path("/vol/msmarco_gemma2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        "sentence-transformers[image]>=5.1.0",
        "transformers>=4.56.0",
        "datasets>=3.2.0",
        "huggingface-hub>=0.30.0",
        "pyarrow>=17.0.0",
        "numpy",
        "einops",
        "accelerate",
        "protobuf",
        "pillow",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
        }
    )
)

app = modal.App(APP_NAME, image=image)


def _scale_dir(scale: str) -> Path:
    return VOL_ROOT / scale


def _load_model():
    """Text-only EmbeddingGemma 2 @ 768-d (skip vision/audio towers)."""
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


def _encode_documents(model, titles: list[str], texts: list[str], batch_size: int):
    import numpy as np

    # Title-aware document prompt (EmbeddingGemma 2 retrieval convention).
    prompts = [f"title: {t.strip() or 'none'} | text: " for t in titles]
    # sentence-transformers applies a single prompt string per call; encode one
    # batch at a time when titles differ, falling back to batched same-prompt path.
    out = np.zeros((len(texts), EMBED_DIMS), dtype=np.float32)
    i = 0
    while i < len(texts):
        j = min(i + batch_size, len(texts))
        # Group consecutive identical prompts for throughput.
        k = i + 1
        while k < j and prompts[k] == prompts[i]:
            k += 1
        emb = model.encode(
            texts[i:k],
            prompt=prompts[i],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(emb, dtype=np.float32)
        if arr.shape[1] != EMBED_DIMS:
            arr = arr[:, :EMBED_DIMS]
            norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
            arr = arr / norms
        out[i:k] = arr
        i = k
    return out


def _encode_queries(model, texts: list[str], batch_size: int):
    import numpy as np

    emb = model.encode(
        texts,
        prompt_name="Retrieval-query",
        batch_size=batch_size,
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


@app.function(
    timeout=60 * 60 * 3,
    memory=65536,
    cpu=8,
    volumes={"/vol": VOL},
)
def prepare_corpus(scale: str = "1m", seed: int = SEED) -> dict[str, Any]:
    """Select corpus ids (qrels-relevant first), materialize shard manifests."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset

    if scale not in SCALE_N:
        raise ValueError(f"unknown scale {scale}; expected one of {list(SCALE_N)}")

    out = _scale_dir(scale)
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    print(f"loading qrels {QRELS_DATASET} split={QRELS_SPLIT}")
    qrels = load_dataset(QRELS_DATASET, split=QRELS_SPLIT)
    # Normalize id types to str.
    qrels_rows = [
        {
            "query_id": str(r["query-id"]),
            "corpus_id": str(r["corpus-id"]),
            "score": int(r["score"]),
        }
        for r in qrels
    ]
    rng = np.random.default_rng(seed)
    if scale == "smoke" and SMOKE_MAX_QUERIES > 0:
        all_q = sorted({r["query_id"] for r in qrels_rows})
        keep_q = set(rng.choice(all_q, size=min(SMOKE_MAX_QUERIES, len(all_q)), replace=False).tolist())
        qrels_rows = [r for r in qrels_rows if r["query_id"] in keep_q]
        print(f"smoke: subsampled qrels to {len(keep_q)} queries ({len(qrels_rows)} pairs)")

    relevant_ids = sorted({r["corpus_id"] for r in qrels_rows})
    query_ids = sorted({r["query_id"] for r in qrels_rows})
    print(f"qrels: {len(qrels_rows)} pairs, {len(query_ids)} queries, {len(relevant_ids)} relevant docs")

    print(f"loading queries {CORPUS_DATASET}")
    queries_ds = load_dataset(CORPUS_DATASET, "queries", split="queries")
    qid_set = set(query_ids)
    query_rows = []
    for r in queries_ds:
        qid = str(r["_id"])
        if qid in qid_set:
            query_rows.append({"query_id": qid, "text": (r.get("text") or "").strip()})
    query_rows.sort(key=lambda x: x["query_id"])
    missing_q = qid_set - {q["query_id"] for q in query_rows}
    if missing_q:
        raise RuntimeError(f"missing {len(missing_q)} queries from corpus queries split")

    print(f"loading corpus {CORPUS_DATASET} (this can take a few minutes)…")
    corpus_ds = load_dataset(CORPUS_DATASET, "corpus", split="corpus")
    n_total = len(corpus_ds)
    target_n = SCALE_N[scale]
    if target_n is None:
        target_n = n_total
    if target_n > n_total:
        raise RuntimeError(f"scale {scale} wants {target_n} > corpus size {n_total}")
    if len(relevant_ids) > target_n:
        raise RuntimeError(
            f"need {len(relevant_ids)} qrels docs but target_n={target_n}"
        )

    relevant_set = set(relevant_ids)

    # Decide which source indices to keep.
    # Always include every qrels-relevant doc; fill the rest with deterministic distractors.
    # Full scale: keep every passage in original corpus order.
    print("selecting corpus indices…")
    if target_n == n_total:
        selected_src_indices = list(range(n_total))
        # Dense row order = original corpus order for full scale.
        chosen_ids_in_order: list[str] = []
        for i in range(n_total):
            chosen_ids_in_order.append(str(corpus_ds[i]["_id"]))
            if (i + 1) % 1_000_000 == 0:
                print(f"  read ids {i+1}/{n_total}")
    else:
        # Map id -> source index (one linear pass).
        id_to_idx: dict[str, int] = {}
        for i in range(n_total):
            id_to_idx[str(corpus_ds[i]["_id"])] = i
            if (i + 1) % 1_000_000 == 0:
                print(f"  indexed {i+1}/{n_total}")
        missing_docs = [d for d in relevant_ids if d not in id_to_idx]
        if missing_docs:
            raise RuntimeError(f"{len(missing_docs)} qrels corpus ids missing from corpus")

        selected_src_indices = [id_to_idx[cid] for cid in relevant_ids]
        selected_set = set(selected_src_indices)
        need = target_n - len(selected_src_indices)
        print(f"sampling {need} distractor indices (seed={seed})…")
        all_indices = np.arange(n_total, dtype=np.int64)
        rng.shuffle(all_indices)
        for idx in all_indices:
            ii = int(idx)
            if ii in selected_set:
                continue
            selected_src_indices.append(ii)
            selected_set.add(ii)
            if len(selected_src_indices) >= target_n:
                break
        assert len(selected_src_indices) == target_n
        chosen_ids_in_order = [str(corpus_ds[i]["_id"]) for i in selected_src_indices]

    corpus_id_to_row = {cid: i for i, cid in enumerate(chosen_ids_in_order)}

    # Materialize selected corpus rows into shard parquet (text only; embeddings later).
    shards_dir = out / "text_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    n_shards = math.ceil(target_n / SHARD_SIZE)
    shard_paths: list[str] = []
    print(f"writing {n_shards} text shards…")
    for s in range(n_shards):
        start = s * SHARD_SIZE
        end = min(start + SHARD_SIZE, target_n)
        row_ids: list[int] = []
        corpus_ids: list[str] = []
        titles: list[str] = []
        texts: list[str] = []
        flags: list[bool] = []
        for row_id in range(start, end):
            src_i = selected_src_indices[row_id]
            src = corpus_ds[src_i]
            cid = str(src["_id"])
            row_ids.append(row_id)
            corpus_ids.append(cid)
            titles.append((src.get("title") or "").strip())
            texts.append((src.get("text") or "").strip())
            flags.append(cid in relevant_set)
        path = shards_dir / f"shard_{s:05d}.parquet"
        table = pa.table(
            {
                "row_id": pa.array(row_ids, type=pa.int32()),
                "corpus_id": pa.array(corpus_ids, type=pa.string()),
                "title": pa.array(titles, type=pa.string()),
                "text": pa.array(texts, type=pa.string()),
                "is_qrels_relevant": pa.array(flags, type=pa.bool_()),
            }
        )
        pq.write_table(table, path, compression="zstd")
        shard_paths.append(str(path))
        print(f"  wrote {path.name} rows={end-start}")

    # Remap qrels to dense row ids; drop pairs whose docs somehow missing (shouldn't happen).
    qrels_mapped = []
    for r in qrels_rows:
        if r["corpus_id"] not in corpus_id_to_row:
            continue
        qrels_mapped.append(
            {
                "query_id": r["query_id"],
                "corpus_id": r["corpus_id"],
                "row_id": corpus_id_to_row[r["corpus_id"]],
                "score": r["score"],
            }
        )

    pq.write_table(
        pa.table(
            {
                "query_id": [r["query_id"] for r in qrels_mapped],
                "corpus_id": [r["corpus_id"] for r in qrels_mapped],
                "row_id": pa.array([r["row_id"] for r in qrels_mapped], type=pa.int32()),
                "score": pa.array([r["score"] for r in qrels_mapped], type=pa.int32()),
            }
        ),
        meta / "qrels.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.table(
            {
                "query_id": [q["query_id"] for q in query_rows],
                "text": [q["text"] for q in query_rows],
            }
        ),
        meta / "queries_text.parquet",
        compression="zstd",
    )

    manifest = {
        "scale": scale,
        "dataset_corpus": CORPUS_DATASET,
        "dataset_qrels": QRELS_DATASET,
        "qrels_split": QRELS_SPLIT,
        "model": MODEL_ID,
        "dims": EMBED_DIMS,
        "mrl": True,
        "mrl_truncation_dims": [768, 512, 256, 128],
        "n_corpus": target_n,
        "n_corpus_available": n_total,
        "n_queries": len(query_rows),
        "n_qrels": len(qrels_mapped),
        "n_relevant_docs_included": len(relevant_ids),
        "k_max": K_MAX,
        "seed": seed,
        "shard_size": SHARD_SIZE,
        "n_shards": n_shards,
        "document_prompt": "title: {title|none} | text: ",
        "query_prompt_name": "Retrieval-query",
        "metric": "inner_product_on_l2_normalized",
        "note": (
            "MS MARCO BeIR corpus has 8,841,823 passages; "
            "'full' uses all of them (no native 10M split)."
        ),
    }
    (meta / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (meta / "shard_paths.json").write_text(json.dumps(shard_paths, indent=2) + "\n")
    VOL.commit()
    print("prepare done", json.dumps(manifest, indent=2))
    return manifest


@app.function(
    gpu=GPU,
    timeout=60 * 60 * 6,
    memory=65536,
    volumes={"/vol": VOL},
)
def embed_text_shard(scale: str, shard_idx: int) -> dict[str, Any]:
    """Embed one prepared text shard → emb_shards/shard_XXXXX.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = _scale_dir(scale)
    text_path = out / "text_shards" / f"shard_{shard_idx:05d}.parquet"
    emb_dir = out / "emb_shards"
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_path = emb_dir / f"shard_{shard_idx:05d}.parquet"
    if emb_path.exists():
        t = pq.read_table(emb_path, columns=["row_id"])
        return {"shard_idx": shard_idx, "rows": t.num_rows, "skipped": True, "path": str(emb_path)}

    table = pq.read_table(text_path)
    row_ids = table.column("row_id").to_pylist()
    corpus_ids = table.column("corpus_id").to_pylist()
    titles = table.column("title").to_pylist()
    texts = table.column("text").to_pylist()
    flags = table.column("is_qrels_relevant").to_pylist()

    print(f"embedding shard {shard_idx} n={len(texts)} on {GPU}")
    t0 = time.time()
    model = _load_model()
    emb = _encode_documents(model, titles, texts, DOC_BATCH)
    elapsed = time.time() - t0
    print(f"shard {shard_idx} done in {elapsed:.1f}s ({len(texts)/max(elapsed,1e-6):.1f} docs/s)")

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
    pq.write_table(out_table, emb_path, compression="zstd")
    VOL.commit()
    return {
        "shard_idx": shard_idx,
        "rows": len(texts),
        "skipped": False,
        "path": str(emb_path),
        "docs_per_sec": len(texts) / max(elapsed, 1e-6),
    }


@app.function(
    gpu=GPU,
    timeout=60 * 60 * 2,
    memory=65536,
    volumes={"/vol": VOL},
)
def embed_queries(scale: str) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = _scale_dir(scale)
    meta = out / "meta"
    qpath = meta / "queries_text.parquet"
    dest = out / "final"
    dest.mkdir(parents=True, exist_ok=True)
    final_path = dest / "queries.parquet"
    if final_path.exists():
        t = pq.read_table(final_path, columns=["query_id"])
        return {"n": t.num_rows, "skipped": True, "path": str(final_path)}

    table = pq.read_table(qpath)
    qids = table.column("query_id").to_pylist()
    texts = table.column("text").to_pylist()
    print(f"embedding {len(texts)} queries…")
    model = _load_model()
    emb = _encode_queries(model, texts, QUERY_BATCH)
    emb_list = [emb[i].tolist() for i in range(emb.shape[0])]
    out_table = pa.table(
        {
            "query_id": pa.array(qids, type=pa.string()),
            "text": pa.array(texts, type=pa.string()),
            "embedding": pa.array(emb_list, type=pa.list_(pa.float32(), EMBED_DIMS)),
        }
    )
    pq.write_table(out_table, final_path, compression="zstd")
    VOL.commit()
    return {"n": len(qids), "skipped": False, "path": str(final_path)}


@app.function(
    gpu=GPU,
    timeout=60 * 60 * 6,
    memory=131072,
    volumes={"/vol": VOL},
)
def compute_neighbors(scale: str, k_max: int = K_MAX) -> dict[str, Any]:
    """Exact top-k neighbors via blocked GPU GEMM (inner product)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    out = _scale_dir(scale)
    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    dest = final / "neighbors.parquet"
    if dest.exists():
        t = pq.read_table(dest, columns=["query_id"])
        return {"n": t.num_rows, "skipped": True, "path": str(dest)}

    qtable = pq.read_table(final / "queries.parquet")
    qids = qtable.column("query_id").to_pylist()
    qemb = np.asarray(qtable.column("embedding").to_pylist(), dtype=np.float32)
    assert qemb.shape[1] == EMBED_DIMS

    # Load corpus embeddings in row_id order from shards.
    prepare = json.loads((out / "meta" / "prepare_manifest.json").read_text())
    n = int(prepare["n_corpus"])
    n_shards = int(prepare["n_shards"])
    corpus = np.zeros((n, EMBED_DIMS), dtype=np.float32)
    for s in range(n_shards):
        shard = pq.read_table(out / "emb_shards" / f"shard_{s:05d}.parquet", columns=["row_id", "embedding"])
        row_ids = shard.column("row_id").to_numpy()
        emb = np.asarray(shard.column("embedding").to_pylist(), dtype=np.float32)
        corpus[row_ids] = emb
        print(f"loaded emb shard {s}/{n_shards}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_t = torch.from_numpy(qemb).to(device)
    # Block over corpus to bound memory.
    block = 200_000 if n >= 200_000 else n
    top_scores = torch.full((len(qids), k_max), -1e9, device=device)
    top_indices = torch.full((len(qids), k_max), -1, device=device, dtype=torch.long)

    print(f"exact top-{k_max} for {len(qids)} queries over {n} docs (block={block})")
    for start in range(0, n, block):
        end = min(start + block, n)
        c_t = torch.from_numpy(corpus[start:end]).to(device)
        # scores: [Q, block]
        scores = q_t @ c_t.T
        # merge with running top-k
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
        torch.cuda.empty_cache()
        print(f"  scanned {end}/{n}")

    top_scores_np = top_scores.float().cpu().numpy()
    top_indices_np = top_indices.cpu().numpy()

    # Map row_id -> corpus_id for convenience.
    id_map = {}
    for s in range(n_shards):
        shard = pq.read_table(
            out / "emb_shards" / f"shard_{s:05d}.parquet",
            columns=["row_id", "corpus_id"],
        )
        for rid, cid in zip(shard.column("row_id").to_pylist(), shard.column("corpus_id").to_pylist()):
            id_map[int(rid)] = cid

    neighbor_ids = []
    neighbor_corpus_ids = []
    neighbor_scores = []
    for i in range(len(qids)):
        ids = [int(x) for x in top_indices_np[i].tolist()]
        neighbor_ids.append(ids)
        neighbor_corpus_ids.append([id_map[j] for j in ids])
        neighbor_scores.append([float(x) for x in top_scores_np[i].tolist()])

    pq.write_table(
        pa.table(
            {
                "query_id": pa.array(qids, type=pa.string()),
                "neighbor_row_ids": pa.array(neighbor_ids, type=pa.list_(pa.int32())),
                "neighbor_corpus_ids": pa.array(neighbor_corpus_ids, type=pa.list_(pa.string())),
                "neighbor_scores": pa.array(neighbor_scores, type=pa.list_(pa.float32())),
            }
        ),
        dest,
        compression="zstd",
    )
    VOL.commit()
    return {"n": len(qids), "k_max": k_max, "skipped": False, "path": str(dest)}


@app.function(
    timeout=60 * 60 * 4,
    memory=131072,
    cpu=8,
    volumes={"/vol": VOL},
)
def assemble_final(scale: str) -> dict[str, Any]:
    """Concatenate emb shards → final/corpus.parquet + copy qrels + write manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = _scale_dir(scale)
    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    prepare = json.loads((out / "meta" / "prepare_manifest.json").read_text())
    n_shards = int(prepare["n_shards"])

    corpus_path = final / "corpus.parquet"
    if not corpus_path.exists():
        writer = None
        total_rows = 0
        try:
            for s in range(n_shards):
                p = out / "emb_shards" / f"shard_{s:05d}.parquet"
                if not p.exists():
                    raise FileNotFoundError(f"missing emb shard {p}")
                table = pq.read_table(p)
                # Shards are already contiguous in row_id order.
                if writer is None:
                    writer = pq.ParquetWriter(corpus_path, table.schema, compression="zstd")
                writer.write_table(table)
                total_rows += table.num_rows
                print(f"append shard {s} rows={table.num_rows}")
                del table
        finally:
            if writer is not None:
                writer.close()
        print(f"wrote {corpus_path} rows={total_rows}")
    else:
        total_rows = pq.read_metadata(corpus_path).num_rows
        print(f"corpus exists rows={total_rows}")

    # Copy qrels into final/
    qrels_src = out / "meta" / "qrels.parquet"
    qrels_dst = final / "qrels.parquet"
    if not qrels_dst.exists():
        qrels_dst.write_bytes(qrels_src.read_bytes())

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
    }
    (final / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    VOL.commit()
    return manifest


@app.function(timeout=60 * 60 * 6, memory=8192, volumes={"/vol": VOL})
def list_final_files(scale: str) -> list[dict[str, Any]]:
    final = _scale_dir(scale) / "final"
    files = []
    for p in sorted(final.glob("*")):
        if p.is_file():
            files.append({"name": p.name, "size": p.stat().st_size})
    return files


@app.function(timeout=60 * 60 * 2, memory=8192, volumes={"/vol": VOL})
def read_final_chunk(scale: str, name: str, offset: int, length: int) -> bytes:
    path = _scale_dir(scale) / "final" / name
    with path.open("rb") as f:
        f.seek(offset)
        return f.read(length)


@app.function(
    timeout=60 * 60 * 24,
    memory=8192,
    volumes={"/vol": VOL},
)
def run_pipeline(scale: str = "1m", seed: int = SEED, k_max: int = K_MAX) -> dict[str, Any]:
    """Prepare → embed all shards (parallel) → queries → neighbors → assemble."""
    prep = prepare_corpus.remote(scale=scale, seed=seed)
    n_shards = int(prep["n_shards"])
    print(f"embedding {n_shards} shards in parallel…")
    results = list(
        embed_text_shard.map([scale] * n_shards, list(range(n_shards)))
    )
    q = embed_queries.remote(scale=scale)
    nb = compute_neighbors.remote(scale=scale, k_max=k_max)
    manifest = assemble_final.remote(scale=scale)
    return {
        "prepare": prep,
        "shards": results,
        "queries": q,
        "neighbors": nb,
        "manifest": manifest,
    }


def _download_scale(scale: str, dest: Path, chunk_size: int = 32 * 1024 * 1024) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    files = list_final_files.remote(scale=scale)
    if not files:
        raise RuntimeError(f"no final files on volume for scale={scale}")
    for info in files:
        name = info["name"]
        size = int(info["size"])
        path = dest / name
        print(f"downloading {name} ({size:,} bytes)…")
        with path.open("wb") as f:
            offset = 0
            while offset < size:
                n = min(chunk_size, size - offset)
                data = read_final_chunk.remote(scale, name, offset, n)
                if not data:
                    raise RuntimeError(f"empty chunk at offset={offset} for {name}")
                f.write(data)
                offset += len(data)
                print(f"  {name}: {offset:,}/{size:,}")
        print(f"wrote {path}")


@app.local_entrypoint()
def main(
    scale: str = "smoke",
    seed: int = SEED,
    k_max: int = K_MAX,
    download_dir: str = "data/msmarco_gemma2/smoke",
    skip_run: bool = False,
):
    """
    modal run modal/embed_msmarco_gemma2.py --scale smoke
    modal run modal/embed_msmarco_gemma2.py --scale 1m --download-dir data/msmarco_gemma2/1m
    modal run modal/embed_msmarco_gemma2.py --scale full --download-dir data/msmarco_gemma2/full
    """
    if scale not in SCALE_N:
        raise SystemExit(f"scale must be one of {list(SCALE_N)}, got {scale!r}")

    if not skip_run:
        summary = run_pipeline.remote(scale=scale, seed=seed, k_max=k_max)
        print("pipeline summary:", json.dumps(summary, indent=2, default=str)[:4000])
    else:
        print("skip_run=True — downloading existing volume artifacts only")

    _download_scale(scale, Path(download_dir))
    print("done — artifacts in", download_dir)
