#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EQ_DIR="${ROOT_DIR}/equilibration"

GMX_BIN="${GMX_BIN:-/mnt/home/jliang/local/gromacs_pme_forcetest/gromacs/build_gpu/bin/gmx_mpi}"
MDP_FILE="${MDP_FILE:-${EQ_DIR}/npt_post100k_dump1000.mdp}"
START_GRO="${START_GRO:-${EQ_DIR}/npt_from_nvt50k_forcetest.gro}"
START_CPT="${START_CPT:-${EQ_DIR}/npt_from_nvt50k_forcetest.cpt}"
REF_GRO="${REF_GRO:-${ROOT_DIR}/1-plain-md/conf.gro}"
TOP_FILE="${TOP_FILE:-${ROOT_DIR}/top/topol.top}"
NDX_FILE="${NDX_FILE:-${ROOT_DIR}/top/index.ndx}"

TPR_FILE="${TPR_FILE:-${EQ_DIR}/npt_post100k_dump1000.tpr}"
DEFFNM="${DEFFNM:-${EQ_DIR}/npt_post100k_dump1000}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/100-conf}"
RAW_DIR="${RAW_DIR:-${EQ_DIR}/raw_frames_npt_post100k_dump1000_$(date +%Y%m%d_%H%M%S)}"
FRAME_COUNT="${FRAME_COUNT:-100}"

for req in "${GMX_BIN}" "${MDP_FILE}" "${START_GRO}" "${START_CPT}" "${REF_GRO}" "${TOP_FILE}" "${NDX_FILE}"; do
  if [[ ! -e "${req}" ]]; then
    echo "Error: required file not found: ${req}" >&2
    exit 1
  fi
done

if find "${OUT_DIR}" -maxdepth 1 -type f -name 'frame_*.gro' -print -quit 2>/dev/null | grep -q .; then
  echo "Error: ${OUT_DIR} already contains frame_*.gro; please use an empty folder." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${RAW_DIR}"

maybe_load_modules() {
  if [[ "${SKIP_MODULES:-0}" == "1" ]]; then
    return 0
  fi
  if type module >/dev/null 2>&1; then
    module load cuda openblas openmpi gcc fftw
  else
    echo "[info] module command not found; assuming runtime dependencies are already available" >&2
  fi
}

maybe_load_modules

echo "[1/4] grompp from finished NPT state"
"${GMX_BIN}" grompp \
  -f "${MDP_FILE}" \
  -c "${START_GRO}" \
  -t "${START_CPT}" \
  -r "${REF_GRO}" \
  -p "${TOP_FILE}" \
  -n "${NDX_FILE}" \
  -o "${TPR_FILE}"

echo "[2/4] mdrun 100000 steps on GPU"
mpirun -np 1 "${GMX_BIN}" mdrun \
  -s "${TPR_FILE}" \
  -deffnm "${DEFFNM}" \
  -ntomp 8 \
  -nb gpu \
  -pme gpu \
  -bonded gpu \
  -update cpu

echo "[3/4] extract GRO frames"
printf '%s\n' "System" | "${GMX_BIN}" trjconv \
  -f "${DEFFNM}.xtc" \
  -s "${TPR_FILE}" \
  -o "${RAW_DIR}/frame_.gro" \
  -sep \
  >/dev/null

mapfile -t RAW_FRAMES < <(find "${RAW_DIR}" -maxdepth 1 -type f -name 'frame_*.gro' | sort -V)
RAW_COUNT="${#RAW_FRAMES[@]}"

if (( RAW_COUNT < FRAME_COUNT )); then
  echo "Error: extracted only ${RAW_COUNT} frames, less than ${FRAME_COUNT}." >&2
  exit 1
fi

START_INDEX=0
if (( RAW_COUNT > FRAME_COUNT )); then
  START_INDEX=$(( RAW_COUNT - FRAME_COUNT ))
fi

echo "[4/4] write final ${FRAME_COUNT} files into ${OUT_DIR}"
for (( i = 0; i < FRAME_COUNT; i++ )); do
  src="${RAW_FRAMES[$(( START_INDEX + i ))]}"
  dst="${OUT_DIR}/frame_$(printf '%03d' "$(( i + 1 ))").gro"
  cp "${src}" "${dst}"
done

echo "Done."
echo "  out_dir  : ${OUT_DIR}"
echo "  tpr      : ${TPR_FILE}"
echo "  trajectory: ${DEFFNM}.xtc"
