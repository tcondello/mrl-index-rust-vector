#!/usr/bin/env python3
"""Validate MS MARCO + EmbeddingGemma-2 parquet bench artifacts."""

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
    args = ap.parse_args()

    manifest = json.loads((args.dir / "manifest.json").read_text())
    dims = int(manifest["dims"])
    n = int(manifest["n_corpus"])
    k_max = int(manifest["k_max"])

    corpus = pq.read_table(args.dir / "corpus.parquet")
    queries = pq.read_table(args.dir / "queries.parquet")
    qrels = pq.read_table(args.dir / "qrels.parquet")
    neighbors = pq.read_table(args.dir / "neighbors.parquet")

    assert corpus.num_rows == n, (corpus.num_rows, n)
    assert "embedding" in corpus.column_names
    cemb = _emb_matrix(corpus.column("embedding"))
    assert cemb.shape == (n, dims), cemb.shape
    norms = np.linalg.norm(cemb, axis=1)
    assert np.isfinite(cemb).all()
    assert np.all(norms > 0.9) and np.all(norms < 1.1), (norms.min(), norms.max())

    qemb = _emb_matrix(queries.column("embedding"))
    assert qemb.shape[1] == dims
    assert queries.num_rows == int(manifest["n_queries"])
    assert neighbors.num_rows == queries.num_rows
    assert qrels.num_rows == int(manifest["n_qrels"])

    # Every qrels row_id must be in range and match corpus_id.
    row_ids = np.asarray(qrels.column("row_id").to_pylist(), dtype=np.int32)
    assert row_ids.min() >= 0 and row_ids.max() < n
    corpus_ids = corpus.column("corpus_id").to_pylist()
    for rid, cid in zip(row_ids.tolist(), qrels.column("corpus_id").to_pylist()):
        assert corpus_ids[rid] == cid

    # Spot-check exact neighbor GT.
    qids = queries.column("query_id").to_pylist()
    qid_to_i = {q: i for i, q in enumerate(qids)}
    nb_qids = neighbors.column("query_id").to_pylist()
    nb_rows = neighbors.column("neighbor_row_ids").to_pylist()
    nb_scores = neighbors.column("neighbor_scores").to_pylist()

    n_check = min(args.spot_check, len(nb_qids))
    for qi in range(n_check):
        qid = nb_qids[qi]
        qvec = qemb[qid_to_i[qid]]
        scores = cemb @ qvec
        top = np.argpartition(-scores, kth=min(9, n - 1))[:10]
        top = top[np.argsort(-scores[top])]
        got = np.asarray(nb_rows[qi][:10], dtype=np.int64)
        if not np.array_equal(top, got):
            # Allow tiny float ties by comparing score sets.
            if not np.allclose(sorted(scores[top]), sorted(scores[got]), atol=1e-4):
                print("neighbor mismatch for", qid, "expected", top, "got", got)
                return 1
        assert len(nb_scores[qi]) == k_max

    print("OK", json.dumps({
        "dir": str(args.dir),
        "n_corpus": n,
        "n_queries": queries.num_rows,
        "n_qrels": qrels.num_rows,
        "dims": dims,
        "model": manifest.get("model"),
        "scale": manifest.get("scale"),
        "emb_norm_range": [float(norms.min()), float(norms.max())],
        "spot_checked_queries": n_check,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
