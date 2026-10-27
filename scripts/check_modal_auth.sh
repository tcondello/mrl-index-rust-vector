#!/usr/bin/env bash
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
if [[ -z "${MODAL_TOKEN_ID:-}" || -z "${MODAL_TOKEN_SECRET:-}" ]]; then
  echo "MODAL_TOKEN_ID / MODAL_TOKEN_SECRET not set in this agent VM." >&2
  echo "Add them in Cursor Environment secrets, then start a NEW agent run (secrets inject at boot)." >&2
  exit 1
fi
modal profile current
echo "Modal auth OK"
