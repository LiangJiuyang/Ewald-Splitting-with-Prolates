#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE_DIR="${ROOT_DIR}/force_eval_100conf"

"${FORCE_DIR}/run_force_eval_100conf.sh"

if [[ "${SKIP_SUMMARY:-0}" != "1" ]]; then
  python3 "${FORCE_DIR}/compute_force_errors.py"
fi
