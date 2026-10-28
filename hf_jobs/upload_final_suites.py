#!/usr/bin/env python3
"""Upload smoke / 1m / full final parquet suites to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


REQUIRED = {
    "corpus.parquet",
    "queries.parquet",
    "qrels.parquet",
    "neighbors.parquet",
    "manifest.json",
}


def upload_scale(api: HfApi, scale: str, root: Path) -> None:
    repo = f"astr010/msmarco-gemma2-{scale}"
    local = root / scale
    if not local.exists():
        raise SystemExit(f"missing local dir {local}")
    present = {p.name for p in local.iterdir() if p.is_file()}
    missing = REQUIRED - present
    if missing:
        raise SystemExit(f"{local} missing required files: {missing}")

    manifest = json.loads((local / "manifest.json").read_text())
    print(
        f"UPLOAD {repo} n_corpus={manifest.get('n_corpus')} "
        f"n_queries={manifest.get('n_queries')} dims={manifest.get('dims')}",
        flush=True,
    )
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Add MS MARCO EmbeddingGemma-2 {scale} benchmark suite",
    )
    files = set(api.list_repo_files(repo, repo_type="dataset"))
    still = REQUIRED - files
    if still:
        raise SystemExit(f"{repo} still missing after upload: {still}")
    print(f"OK {repo} files={sorted(files)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/msmarco_gemma2"))
    ap.add_argument(
        "--scales",
        nargs="+",
        default=["smoke", "1m", "full"],
        choices=["smoke", "1m", "full"],
    )
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")
    api = HfApi(token=token)
    for scale in args.scales:
        upload_scale(api, scale, args.root)
    print("ALL_UPLOADS_OK", flush=True)


if __name__ == "__main__":
    main()
