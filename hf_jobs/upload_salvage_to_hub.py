#!/usr/bin/env python3
"""Create Hub work dataset and upload Modal-salvaged shards + meta + text."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="astr010/msmarco-gemma2-full-work")
    ap.add_argument(
        "--local",
        type=Path,
        default=Path("data/msmarco_gemma2/full_modal_salvage"),
    )
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")
    if not args.local.exists():
        raise SystemExit(f"missing local salvage dir {args.local}")

    api = HfApi(token=token)
    api.create_repo(
        args.repo,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )
    readme = args.local / "README.md"
    if not readme.exists():
        readme.write_text(
            "---\n"
            "license: mit\n"
            "task_categories:\n"
            "  - text-retrieval\n"
            "---\n\n"
            "# MS MARCO EmbeddingGemma-2 full work shards\n\n"
            "Intermediate text + embedding shards for assembling the full "
            "(~8.84M) MS MARCO + `google/embeddinggemma-2` @ 768-d benchmark.\n\n"
            "Salvaged from Modal (`emb_shards` partial) and completed via Hugging Face Jobs.\n"
        )

    print(f"uploading {args.local} → {args.repo}")
    # Upload subfolders separately so a failure can resume without redoing everything.
    for sub in ("meta", "text_shards", "emb_shards"):
        src = args.local / sub
        if not src.exists():
            print(f"skip missing {src}")
            continue
        print(f"→ {sub}/")
        api.upload_folder(
            folder_path=str(src),
            path_in_repo=sub,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"Add/update {sub} for MS MARCO Gemma2 full work",
        )
    readme_path = args.local / "README.md"
    if readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    print("upload complete")


if __name__ == "__main__":
    main()
