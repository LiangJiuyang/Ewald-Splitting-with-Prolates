#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAME="$(basename "${ROOT_DIR}")"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RELEASE_ROOT="${ROOT_DIR}/release"
PKG_DIR="${RELEASE_ROOT}/${DATASET_NAME}_release_${STAMP}"
TAR_PATH="${PKG_DIR}.tar.gz"

cd "${ROOT_DIR}"

copy_rel() {
  local rel src dst
  for rel in "$@"; do
    src="${ROOT_DIR}/${rel}"
    if [[ ! -e "${src}" ]]; then
      echo "Error: missing required path: ${rel}" >&2
      exit 1
    fi
    dst="${PKG_DIR}/${rel}"
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
    printf '%s\n' "${rel}" >> "${PKG_DIR}/RELEASE_INCLUDED.txt"
  done
}

mkdir -p "${RELEASE_ROOT}"
if [[ -e "${PKG_DIR}" || -e "${TAR_PATH}" ]]; then
  echo "Error: release target already exists for STAMP=${STAMP}" >&2
  exit 1
fi

mkdir -p "${PKG_DIR}"
: > "${PKG_DIR}/RELEASE_INCLUDED.txt"

copy_rel \
  README.md \
  SOURCE.txt \
  download_link.txt \
  prepare_tpr.sh \
  run_100k_equil_then_dump.sh \
  reproduce \
  mdp \
  system \
  force_eval_100conf/run_force_eval_100conf.sh \
  force_eval_100conf/compute_force_errors.py

copy_rel timeperformance/*.mdp
copy_rel timeperformance/*.sh
copy_rel timeperformance/*.slurm

cp -a "${ROOT_DIR}/pack_release.sh" "${PKG_DIR}/pack_release.sh"
printf '%s\n' "pack_release.sh" >> "${PKG_DIR}/RELEASE_INCLUDED.txt"

tar -czf "${TAR_PATH}" -C "${RELEASE_ROOT}" "$(basename "${PKG_DIR}")"

echo "Release directory: ${PKG_DIR}"
echo "Release archive:   ${TAR_PATH}"
