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
  1-plain-md \
  2-ss-refinement \
  3-ensemble-refinement \
  top \
  mdp \
  scripts \
  reproduce \
  force_eval_first100_esp_pme_pmeref.sh \
  run_esp_pme_np_scaling_param_sweep_repeat5_npme0.sh \
  run_pme_only_mpi_repeat5_npme0_from_mdp.sh \
  recompute_first100_summary.py \
  equilibration/em.mdp \
  equilibration/nvt.mdp \
  equilibration/npt.mdp \
  equilibration/npt_post100k_dump1000.mdp \
  equilibration/npt_from_nvt50k_forcetest.gro \
  equilibration/npt_from_nvt50k_forcetest.cpt \
  equilibration/run_full_equilibration.sh \
  equilibration/run_npt_post100k_dump100.sh

cp -a "${ROOT_DIR}/pack_release.sh" "${PKG_DIR}/pack_release.sh"
printf '%s\n' "pack_release.sh" >> "${PKG_DIR}/RELEASE_INCLUDED.txt"

tar -czf "${TAR_PATH}" -C "${RELEASE_ROOT}" "$(basename "${PKG_DIR}")"

echo "Release directory: ${PKG_DIR}"
echo "Release archive:   ${TAR_PATH}"
