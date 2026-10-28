#!/usr/bin/env python3
"""Convert MS MARCO parquet suites into mrl-index binary bench layout.

Writes a directory the Rust loader can use for a *clean* retrieval bench:
  embeddings.bin       — corpus row-major f32 [n, dims]
  query_embeddings.bin — query row-major f32 [q, dims]
  texts.jsonl          — corpus texts (id == row_id)
  queries.jsonl        — query_id + text
  neighbors.jsonl      — exact top-k neighbor row ids/scores per query
  qrels.jsonl          — IR relevance labels mapped to row_id
  manifest.json        — mrl-index compatible + MS MARCO extensions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def fixed_list_to_numpy(col) -> np.ndarray:
    chunked = col.combine_chunks()
    flat = chunked.values.to_numpy(zero_copy_only=False)
    n = len(chunked)
    d = flat.size // n
    return np.asarray(flat, dtype=np.float32).reshape(n, d)


def convert(src: Path, dst: Path) -> None:
    src_manifest = json.loads((src / "manifest.json").read_text())
    dims = int(src_manifest["dims"])
    n = int(src_manifest["n_corpus"])
    n_queries = int(src_manifest["n_queries"])
    dst.mkdir(parents=True, exist_ok=True)

    print(f"reading corpus {src/'corpus.parquet'} …", flush=True)
    # Stream row groups to bound peak RAM during conversion.
    pf = pq.ParquetFile(src / "corpus.parquet")
    emb_path = dst / "embeddings.bin"
    texts_path = dst / "texts.jsonl"
    with emb_path.open("wb") as eb, texts_path.open("w", encoding="utf-8") as tf:
        written = 0
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(
                rg, columns=["row_id", "corpus_id", "title", "text", "embedding"]
            )
            row_ids = table.column("row_id").to_pylist()
            corpus_ids = table.column("corpus_id").to_pylist()
            titles = table.column("title").to_pylist()
            texts = table.column("text").to_pylist()
            emb = fixed_list_to_numpy(table.column("embedding"))
            assert emb.shape[1] == dims, emb.shape
            # Ensure row_id order within file is ascending for dense layout.
            order = np.argsort(np.asarray(row_ids, dtype=np.int64))
            emb = emb[order]
            eb.write(emb.tobytes(order="C"))
            for i in order.tolist():
                rec = {
                    "id": int(row_ids[i]),
                    "corpus_id": corpus_ids[i],
                    "title": titles[i],
                    "text": texts[i],
                }
                tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += emb.shape[0]
            print(f"  corpus rows {written}/{n}", flush=True)
    if written != n:
        raise SystemExit(f"corpus rows {written} != manifest n {n}")

    print("reading queries …", flush=True)
    qtable = pq.read_table(src / "queries.parquet")
    qids = qtable.column("query_id").to_pylist()
    qtexts = qtable.column("text").to_pylist()
    qemb = fixed_list_to_numpy(qtable.column("embedding"))
    assert qemb.shape == (n_queries, dims), qemb.shape
    (dst / "query_embeddings.bin").write_bytes(qemb.tobytes(order="C"))
    with (dst / "queries.jsonl").open("w", encoding="utf-8") as f:
        for i, (qid, text) in enumerate(zip(qids, qtexts)):
            f.write(json.dumps({"i": i, "query_id": qid, "text": text}, ensure_ascii=False) + "\n")

    print("reading neighbors …", flush=True)
    ntable = pq.read_table(src / "neighbors.parquet")
    # Map query_id -> dense query index
    qid_to_i = {qid: i for i, qid in enumerate(qids)}
    with (dst / "neighbors.jsonl").open("w", encoding="utf-8") as f:
        for qid, rows, scores in zip(
            ntable.column("query_id").to_pylist(),
            ntable.column("neighbor_row_ids").to_pylist(),
            ntable.column("neighbor_scores").to_pylist(),
        ):
            f.write(
                json.dumps(
                    {
                        "query_index": qid_to_i[qid],
                        "query_id": qid,
                        "neighbor_row_ids": [int(x) for x in rows],
                        "neighbor_scores": [float(x) for x in scores],
                    }
                )
                + "\n"
            )

    print("reading qrels …", flush=True)
    rtable = pq.read_table(src / "qrels.parquet")
    with (dst / "qrels.jsonl").open("w", encoding="utf-8") as f:
        for qid, cid, rid, score in zip(
            rtable.column("query_id").to_pylist(),
            rtable.column("corpus_id").to_pylist(),
            rtable.column("row_id").to_pylist(),
            rtable.column("score").to_pylist(),
        ):
            f.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "query_index": qid_to_i[qid],
                        "corpus_id": cid,
                        "row_id": int(rid),
                        "score": int(score),
                    }
                )
                + "\n"
            )

    out_manifest = {
        "dataset": src_manifest.get("dataset_corpus", "BeIR/msmarco"),
        "dataset_qrels": src_manifest.get("dataset_qrels", "BeIR/msmarco-qrels"),
        "model": src_manifest["model"],
        "dims": dims,
        "n": n,
        "n_queries": n_queries,
        "n_qrels": int(src_manifest.get("n_qrels", 0)),
        "k_max": int(src_manifest.get("k_max", 100)),
        "mrl": True,
        "truncate_dim": dims,
        "embeddings_file": "embeddings.bin",
        "texts_file": "texts.jsonl",
        "query_embeddings_file": "query_embeddings.bin",
        "queries_file": "queries.jsonl",
        "neighbors_file": "neighbors.jsonl",
        "qrels_file": "qrels.jsonl",
        "dtype": "f32",
        "layout": "row_major",
        "metric": "inner_product_on_l2_normalized",
        "seed": src_manifest.get("seed", 42),
        "scale": src_manifest.get("scale"),
        "source_parquet_manifest": src_manifest,
    }
    (dst / "manifest.json").write_text(json.dumps(out_manifest, indent=2) + "\n")
    print(f"wrote {dst} n={n} q={n_queries} dims={dims}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="parquet suite dir (smoke/1m/full)")
    ap.add_argument("dst", type=Path, help="output binary bench dir")
    args = ap.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
