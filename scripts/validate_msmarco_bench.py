#!/usr/bin/env python3
"""Validate MS MARCO + EmbeddingGemma-2 parquet bench artifacts (streaming-safe)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _emb_matrix(col) -> np.ndarray:
    return np.asarray(col.to_pylist(), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path, help="Directory with manifest.json + parquet files")
    ap.add_argument("--spot-check", type=int, default=5, help="Recompute exact top-10 for N queries")
    ap.add_argument(
        "--norm-sample",
        type=int,
        default=2048,
        help="How many corpus rows to sample for norm/finite checks",
    )
    args = ap.parse_args()

    manifest = json.loads((args.dir / "manifest.json").read_text())
    dims = int(manifest["dims"])
    n = int(manifest["n_corpus"])
    k_max = int(manifest["k_max"])

    corpus_path = args.dir / "corpus.parquet"
    queries_path = args.dir / "queries.parquet"
    qrels_path = args.dir / "qrels.parquet"
    neighbors_path = args.dir / "neighbors.parquet"

    corpus_meta = pq.read_metadata(corpus_path)
    assert corpus_meta.num_rows == n, (corpus_meta.num_rows, n)

    # Stream a small corpus sample for dtype / norm sanity (avoid loading all embeddings).
    sample_n = min(args.norm_sample, n)
    pf = pq.ParquetFile(corpus_path)
    sample_batch = next(pf.iter_batches(columns=["embedding"], batch_size=sample_n))
    cemb_sample = _emb_matrix(sample_batch.column("embedding"))
    assert cemb_sample.shape[1] == dims, cemb_sample.shape
    assert np.isfinite(cemb_sample).all()
    norms = np.linalg.norm(cemb_sample, axis=1)
    assert np.all(norms > 0.9) and np.all(norms < 1.1), (float(norms.min()), float(norms.max()))
    sample_n = int(cemb_sample.shape[0])

    queries = pq.read_table(queries_path)
    qrels = pq.read_table(qrels_path)
    neighbors = pq.read_table(neighbors_path)

    qemb = _emb_matrix(queries.column("embedding"))
    assert qemb.shape[1] == dims
    assert queries.num_rows == int(manifest["n_queries"])
    assert neighbors.num_rows == queries.num_rows
    assert qrels.num_rows == int(manifest["n_qrels"])

    row_ids = np.asarray(qrels.column("row_id").to_pylist(), dtype=np.int32)
    assert row_ids.min() >= 0 and row_ids.max() < n

    # Verify qrels row_id ↔ corpus_id on the touched rows only.
    needed = sorted(set(row_ids.tolist()))
    # Read full id columns only (cheap vs embeddings).
    id_table = pq.read_table(corpus_path, columns=["row_id", "corpus_id"])
    corpus_row_ids = id_table.column("row_id").to_pylist()
    corpus_ids = id_table.column("corpus_id").to_pylist()
    id_by_row = {int(r): c for r, c in zip(corpus_row_ids, corpus_ids)}
    for rid, cid in zip(row_ids.tolist(), qrels.column("corpus_id").to_pylist()):
        assert id_by_row[rid] == cid

    # Spot-check neighbors against a streamed corpus matmul in blocks.
    qids = queries.column("query_id").to_pylist()
    qid_to_i = {q: i for i, q in enumerate(qids)}
    nb_qids = neighbors.column("query_id").to_pylist()
    nb_rows = neighbors.column("neighbor_row_ids").to_pylist()
    nb_scores = neighbors.column("neighbor_scores").to_pylist()

    n_check = min(args.spot_check, len(nb_qids))
    check_q = np.stack([qemb[qid_to_i[nb_qids[i]]] for i in range(n_check)], axis=0)  # [Q, D]
    # Running top-10 per checked query.
    top_scores = np.full((n_check, 10), -1e9, dtype=np.float32)
    top_indices = np.full((n_check, 10), -1, dtype=np.int64)

    pf = pq.ParquetFile(corpus_path)
    for batch in pf.iter_batches(columns=["row_id", "embedding"], batch_size=8192):
        rids = np.asarray(batch.column("row_id").to_pylist(), dtype=np.int64)
        emb = _emb_matrix(batch.column("embedding"))
        scores = check_q @ emb.T  # [Q, B]
        for qi in range(n_check):
            cat_s = np.concatenate([top_scores[qi], scores[qi]])
            cat_i = np.concatenate([top_indices[qi], rids])
            sel = np.argpartition(-cat_s, kth=9)[:10]
            sel = sel[np.argsort(-cat_s[sel])]
            top_scores[qi] = cat_s[sel]
            top_indices[qi] = cat_i[sel]

    for qi in range(n_check):
        got = np.asarray(nb_rows[qi][:10], dtype=np.int64)
        exp = top_indices[qi]
        if not np.array_equal(exp, got):
            # Tolerate rare float ties by comparing score multisets.
            if not np.allclose(np.sort(top_scores[qi]), np.sort(nb_scores[qi][:10]), atol=1e-3):
                print("neighbor mismatch for", nb_qids[qi], "expected", exp, "got", got)
                return 1
        assert len(nb_scores[qi]) == k_max

    print(
        "OK",
        json.dumps(
            {
                "dir": str(args.dir),
                "n_corpus": n,
                "n_queries": queries.num_rows,
                "n_qrels": qrels.num_rows,
                "dims": dims,
                "model": manifest.get("model"),
                "scale": manifest.get("scale"),
                "emb_norm_range_sample": [float(norms.min()), float(norms.max())],
                "norm_sample_n": sample_n,
                "spot_checked_queries": n_check,
            },
            indent=2,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
