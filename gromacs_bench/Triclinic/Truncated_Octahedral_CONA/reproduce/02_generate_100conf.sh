#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# > 0 )); then
  exec "${ROOT_DIR}/run_100k_equil_then_dump.sh" "$@"
fi

TPR="${TPR:-${ROOT_DIR}/cona_em.tpr}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/100-conf}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/runs/$(basename "${TPR}" .tpr)_eq100k_prod100k}"

exec "${ROOT_DIR}/run_100k_equil_then_dump.sh" \
  --tpr "${TPR}" \
  --out-dir "${OUT_DIR}" \
  --work-dir "${WORK_DIR}"
