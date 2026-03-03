#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${ROOT_DIR}/force_eval_first100_esp_pme_pmeref.sh" "$@"
