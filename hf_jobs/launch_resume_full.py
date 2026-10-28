#!/usr/bin/env python3
"""Launch HF Jobs to embed missing MS MARCO full shards, then finalize.

Assumes work dataset already has meta/ + text_shards/ + salvaged emb_shards/.

Usage:
  export HF_TOKEN=...
  python3 hf_jobs/launch_resume_full.py \\
    --work-repo astr010/msmarco-gemma2-full-work \\
    --out-repo astr010/msmarco-gemma2-full \\
    --flavor a10g-large \\
    --max-parallel 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
SHARD_SCRIPT = ROOT / "hf_jobs" / "embed_msmarco_gemma2_shard.py"
FINAL_SCRIPT = ROOT / "hf_jobs" / "finalize_msmarco_gemma2.py"


def _stage(api: HfApi, job_id: str) -> str:
    info = api.inspect_job(job_id)
    return str(info.status.stage)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-repo", default="astr010/msmarco-gemma2-full-work")
    ap.add_argument("--out-repo", default="astr010/msmarco-gemma2-full")
    ap.add_argument("--flavor", default="a10g-large")
    ap.add_argument("--timeout", default="2h")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--finalize-flavor", default="a100-large")
    ap.add_argument("--finalize-timeout", default="6h")
    ap.add_argument("--skip-finalize", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")
    api = HfApi(token=token)

    files = set(api.list_repo_files(args.work_repo, repo_type="dataset"))
    # Discover n_shards from prepare manifest if present locally or on hub.
    n_shards = None
    for candidate in (
        ROOT / "data/msmarco_gemma2/full_modal_salvage/meta/prepare_manifest.json",
        Path("/tmp/prepare_manifest.json"),
    ):
        if candidate.exists():
            n_shards = int(json.loads(candidate.read_text())["n_shards"])
            break
    if n_shards is None:
        # Fallback: infer from text shards on hub.
        text = sorted(f for f in files if f.startswith("text_shards/shard_") and f.endswith(".parquet"))
        if not text:
            raise SystemExit("cannot determine n_shards; upload prepare_manifest.json first")
        n_shards = len(text)

    missing = [
        i
        for i in range(n_shards)
        if f"emb_shards/shard_{i:05d}.parquet" not in files
    ]
    print(f"work={args.work_repo} n_shards={n_shards} missing={len(missing)}")
    if missing:
        print("missing sample:", missing[:20], ("..." if len(missing) > 20 else ""))

    jobs = []
    for idx in missing:
        if args.dry_run:
            print(f"DRY would launch shard {idx}")
            continue
        # Simple concurrency gate.
        while True:
            running_n = sum(
                1
                for j in jobs
                if _stage(api, j.id) in {"RUNNING", "PENDING", "SCHEDULED", "STARTING"}
            )
            if running_n < args.max_parallel:
                break
            time.sleep(20)

        job = api.run_uv_job(
            str(SHARD_SCRIPT),
            script_args=["--repo", args.work_repo, "--shard-idx", str(idx)],
            flavor=args.flavor,
            timeout=args.timeout,
            secrets=["HF_TOKEN"],
            name=f"msmarco-gemma2-shard-{idx:05d}",
        )
        jobs.append(job)
        print(f"launched shard {idx}: {job.id} {getattr(job, 'url', '')}")

    # Wait for shard jobs.
    if jobs and not args.dry_run:
        print(f"waiting on {len(jobs)} shard jobs…")
        for j in jobs:
            info = api.wait_for_job(j.id, timeout=60 * 60 * 6)
            stage = info.status.stage
            print(f"job {j.id} -> {stage}")
            if stage != "COMPLETED":
                raise SystemExit(f"shard job failed: {j.id} {stage} {getattr(j, 'url', '')}")

    if args.skip_finalize or args.dry_run:
        print("skip finalize")
        return

    final = api.run_uv_job(
        str(FINAL_SCRIPT),
        script_args=["--work-repo", args.work_repo, "--out-repo", args.out_repo],
        flavor=args.finalize_flavor,
        timeout=args.finalize_timeout,
        secrets=["HF_TOKEN"],
        name="msmarco-gemma2-finalize",
    )
    print(f"finalize job: {final.id} {getattr(final, 'url', '')}")
    info = api.wait_for_job(final.id, timeout=60 * 60 * 12)
    print(f"finalize -> {info.status.stage}")
    if info.status.stage != "COMPLETED":
        raise SystemExit(f"finalize failed: {final.id}")


if __name__ == "__main__":
    main()
