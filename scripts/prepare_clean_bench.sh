#!/usr/bin/env bash
# Prepare a clean (no-shortcuts) MS MARCO / EmbeddingGemma-2 retrieval bench env.
#
# Clean means:
#   - query embeddings (not corpus self-retrieval)
#   - precomputed exact IP neighbors as ANN ground truth
#   - BeIR/msmarco-qrels for IR metrics
#   - full 768-d MRL (no truncate, no --synthetic)
#   - release binary
#
# Hub backup (canonical):
#   astr010/msmarco-gemma2-{smoke,1m,full}
#   astr010/msmarco-gemma2-full-work  (177 shard intermediates)
#
# Usage:
#   scripts/prepare_clean_bench.sh              # smoke + 1m bins + release build
#   scripts/prepare_clean_bench.sh smoke        # smoke only
#   scripts/prepare_clean_bench.sh 1m
#   SCALE=full scripts/prepare_clean_bench.sh   # full bin (needs ~30Gi RAM or mmap later)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCALE="${1:-all}"
PARQUET_ROOT="${PARQUET_ROOT:-data/msmarco_gemma2}"
BIN_ROOT="${BIN_ROOT:-data/msmarco_gemma2}"

need_suite() {
  local s="$1"
  local d="$PARQUET_ROOT/$s"
  if [[ ! -f "$d/corpus.parquet" || ! -f "$d/manifest.json" ]]; then
    echo "missing $d — download from Hub, e.g.:"
    echo "  huggingface-cli download astr010/msmarco-gemma2-$s --repo-type dataset --local-dir $d"
    exit 1
  fi
}

convert_one() {
  local s="$1"
  need_suite "$s"
  local dst="$BIN_ROOT/${s}_bin"
  if [[ -f "$dst/embeddings.bin" && -f "$dst/query_embeddings.bin" && -f "$dst/neighbors.jsonl" && -f "$dst/manifest.json" ]]; then
    echo "skip convert $s → $dst (already present)"
  else
    echo "convert $s → $dst"
    python3 scripts/parquet_to_bench_bin.py "$PARQUET_ROOT/$s" "$dst"
  fi
}

echo "== cargo release =="
cargo build --release

case "$SCALE" in
  smoke) convert_one smoke ;;
  1m) convert_one 1m ;;
  full) convert_one full ;;
  all)
    convert_one smoke
    convert_one 1m
    ;;
  *)
    echo "unknown scale: $SCALE (smoke|1m|full|all)"
    exit 1
    ;;
esac

echo
echo "Ready. Clean smoke (all queries):"
echo "  ./target/release/mrl-index bench-retrieval --label smoke-clean \\"
echo "    --data ${BIN_ROOT}/smoke_bin --shortlist 64 --strata l1_rabitq \\"
echo "    --out reports/bench_smoke_clean.json"
echo
echo "Clean 1M (all ${BIN_ROOT}/1m_bin queries; expect ~20–40 min):"
echo "  ./target/release/mrl-index bench-retrieval --label 1m-clean \\"
echo "    --data ${BIN_ROOT}/1m_bin --shortlist 64 --strata l1_rabitq \\"
echo "    --out reports/bench_1m_clean.json"
echo
echo "Do NOT use: bench-suite --synthetic  (isotropic, not MRL)"
echo "Do NOT omit neighbors/qrels  (self-retrieval is not a clean IR bench)"
