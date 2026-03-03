#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_100k_equil_then_dump.sh --tpr INPUT.tpr --out-dir OUTPUT_DIR [options]

Required:
  --tpr PATH            Input TPR built from your original MDP settings.
  --out-dir PATH        Directory to store final 100 GRO files.

Options:
  --work-dir PATH       Working directory for intermediate files.
                        Default: runs/<tpr_basename>_eq100k_prod100k
  --gmx-bin PATH        GROMACS binary. Default is auto-selected by tpr name:
                        *esp* -> /mnt/home/jliang/local/gromacs_new/build_force_test/bin/gmx_mpi_d
                        else  -> /mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi_d
  --gmx-prefix STRING   Prefix for non-mdrun commands (convert-tpr/dump/trjconv).
                        Example: "mpirun --map-by socket:pe=1 -np 1"
  --mdrun-prefix STRING Prefix for mdrun commands.
                        Example: "mpirun --map-by socket:pe=48 -np 1"
  --ntomp INT           OpenMP threads for mdrun. Default: 48
  --equil-steps INT     Equilibration steps. Default: 100000
  --prod-steps INT      Additional production steps. Default: 100000
  --dump-interval INT   Desired interval (in MD steps) between dumped frames.
                        Default: 1000
  --frame-count INT     Number of final GRO files to keep. Default: 100
  --trj-group NAME      trjconv output group. Default: System
  -h, --help            Show help

Notes:
  1) Physics parameters come from your original TPR/MDP; this script only changes
     step count using convert-tpr.
  2) The script expects trajectory output frequency to be compatible with
     dump interval (dump_interval must be a multiple of nstxout-compressed).
EOF
}

run_with_prefix() {
  local prefix="$1"
  shift
  if [[ -n "${prefix}" ]]; then
    # shellcheck disable=SC2206
    local parts=( ${prefix} )
    "${parts[@]}" "$@"
  else
    "$@"
  fi
}

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "$(pwd)/${path}"
  fi
}

extract_tpr_int() {
  local tpr_file="$1"
  local key="$2"
  run_with_prefix "${GMX_PREFIX}" "${GMX_BIN}" dump -s "${tpr_file}" 2>/dev/null \
    | awk -F'=' -v key="${key}" '
        {
          lhs=$1
          gsub(/[[:space:]]/, "", lhs)
          if (lhs == key) {
            rhs=$2
            gsub(/[[:space:]]/, "", rhs)
            if (rhs ~ /^[0-9]+$/) {
              print rhs
              exit
            }
          }
        }
      '
}

TPR=""
OUT_DIR=""
WORK_DIR=""

GMX_BIN="${GMX_BIN:-}"
GMX_PREFIX="${GMX_PREFIX:-}"
MDRUN_PREFIX="${MDRUN_PREFIX:-}"

NTOMP="${NTOMP:-48}"
EQUIL_STEPS=100000
PROD_STEPS=100000
DUMP_INTERVAL=1000
FRAME_COUNT=100
TRJ_GROUP="System"

while (( $# > 0 )); do
  case "$1" in
    --tpr)
      TPR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --gmx-bin)
      GMX_BIN="$2"
      shift 2
      ;;
    --gmx-prefix)
      GMX_PREFIX="$2"
      shift 2
      ;;
    --mdrun-prefix)
      MDRUN_PREFIX="$2"
      shift 2
      ;;
    --ntomp)
      NTOMP="$2"
      shift 2
      ;;
    --equil-steps)
      EQUIL_STEPS="$2"
      shift 2
      ;;
    --prod-steps)
      PROD_STEPS="$2"
      shift 2
      ;;
    --dump-interval)
      DUMP_INTERVAL="$2"
      shift 2
      ;;
    --frame-count)
      FRAME_COUNT="$2"
      shift 2
      ;;
    --trj-group)
      TRJ_GROUP="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${TPR}" || -z "${OUT_DIR}" ]]; then
  echo "Error: --tpr and --out-dir are required." >&2
  usage
  exit 1
fi

if [[ ! -f "${TPR}" ]]; then
  echo "Error: TPR not found: ${TPR}" >&2
  exit 1
fi

TPR="$(abs_path "${TPR}")"
OUT_DIR="$(abs_path "${OUT_DIR}")"

if [[ -z "${GMX_BIN}" ]]; then
  case "$(basename "${TPR}")" in
    *esp*)
      GMX_BIN="/mnt/home/jliang/local/gromacs_esp_gpu/build_cpu_double/bin/gmx_mpi_d"
      ;;
    *)
      GMX_BIN="/mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi_d"
      ;;
  esac
fi

if [[ ! -x "${GMX_BIN}" ]]; then
  echo "Error: GROMACS binary is not executable: ${GMX_BIN}" >&2
  exit 1
fi

if [[ -z "${WORK_DIR}" ]]; then
  base="$(basename "${TPR}" .tpr)"
  WORK_DIR="runs/${base}_eq100k_prod100k"
fi
WORK_DIR="$(abs_path "${WORK_DIR}")"

if find "${OUT_DIR}" -maxdepth 1 -type f -name 'frame_*.gro' -print -quit 2>/dev/null | grep -q .; then
  echo "Error: ${OUT_DIR} already contains frame_*.gro. Use an empty output folder." >&2
  exit 1
fi

mkdir -p "${WORK_DIR}" "${OUT_DIR}"

EQ_TPR="${WORK_DIR}/eq.tpr"
PROD_TPR="${WORK_DIR}/prod.tpr"
EQ_DEFFNM="${WORK_DIR}/eq"
PROD_DEFFNM="${WORK_DIR}/prod"
RAW_FRAME_DIR="${WORK_DIR}/raw_frames"

echo "[1/6] Build equilibration tpr (${EQUIL_STEPS} steps)"
run_with_prefix "${GMX_PREFIX}" "${GMX_BIN}" convert-tpr -s "${TPR}" -o "${EQ_TPR}" -nsteps "${EQUIL_STEPS}"

echo "[2/6] Run equilibration mdrun"
run_with_prefix "${MDRUN_PREFIX}" "${GMX_BIN}" mdrun \
  -s "${EQ_TPR}" \
  -deffnm "${EQ_DEFFNM}" \
  -cpo "${EQ_DEFFNM}.cpt" \
  -ntomp "${NTOMP}"

if [[ ! -f "${EQ_DEFFNM}.cpt" ]]; then
  echo "Error: checkpoint not found after equilibration: ${EQ_DEFFNM}.cpt" >&2
  exit 1
fi

echo "[3/6] Build production tpr (+${PROD_STEPS} steps)"
run_with_prefix "${GMX_PREFIX}" "${GMX_BIN}" convert-tpr -s "${EQ_TPR}" -o "${PROD_TPR}" -extend "${PROD_STEPS}"

echo "[4/6] Continue from checkpoint and run production"
run_with_prefix "${MDRUN_PREFIX}" "${GMX_BIN}" mdrun \
  -s "${PROD_TPR}" \
  -deffnm "${PROD_DEFFNM}" \
  -cpi "${EQ_DEFFNM}.cpt" \
  -cpo "${PROD_DEFFNM}.cpt" \
  -noappend \
  -ntomp "${NTOMP}"

NSTX_COMP="$(extract_tpr_int "${PROD_TPR}" "nstxout-compressed" || true)"
NSTX_COMP="${NSTX_COMP:-0}"
if [[ ! "${NSTX_COMP}" =~ ^[0-9]+$ ]]; then
  NSTX_COMP=0
fi
if (( NSTX_COMP <= 0 )); then
  echo "Error: nstxout-compressed <= 0 in tpr; cannot extract .gro from compressed trajectory." >&2
  exit 1
fi

if (( DUMP_INTERVAL % NSTX_COMP != 0 )); then
  echo "Error: dump interval (${DUMP_INTERVAL}) is not divisible by nstxout-compressed (${NSTX_COMP})." >&2
  exit 1
fi
SKIP=$(( DUMP_INTERVAL / NSTX_COMP ))
if (( SKIP < 1 )); then
  SKIP=1
fi

TRAJ_FILE="${PROD_DEFFNM}.xtc"
if [[ ! -f "${TRAJ_FILE}" ]]; then
  echo "Error: expected trajectory not found: ${TRAJ_FILE}" >&2
  exit 1
fi

mkdir -p "${RAW_FRAME_DIR}"

echo "[5/6] Extract frames from production trajectory"
printf '%s\n' "${TRJ_GROUP}" | run_with_prefix "${GMX_PREFIX}" "${GMX_BIN}" trjconv \
  -f "${TRAJ_FILE}" \
  -s "${PROD_TPR}" \
  -o "${RAW_FRAME_DIR}/frame_.gro" \
  -sep \
  -skip "${SKIP}" \
  >/dev/null

mapfile -t RAW_FRAMES < <(find "${RAW_FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.gro' | sort -V)
RAW_COUNT="${#RAW_FRAMES[@]}"

if (( RAW_COUNT < FRAME_COUNT )); then
  echo "Error: only ${RAW_COUNT} frame files extracted, fewer than requested ${FRAME_COUNT}." >&2
  exit 1
fi

START_INDEX=0
if (( RAW_COUNT > FRAME_COUNT )); then
  # Keep the latest frames to avoid an extra initial frame from restart boundaries.
  START_INDEX=$(( RAW_COUNT - FRAME_COUNT ))
fi

echo "[6/6] Write final ${FRAME_COUNT} GRO files to ${OUT_DIR}"
for (( i = 0; i < FRAME_COUNT; i++ )); do
  src="${RAW_FRAMES[$((START_INDEX + i))]}"
  dst="${OUT_DIR}/frame_$(printf '%03d' "$((i + 1))").gro"
  cp "${src}" "${dst}"
done

echo "Done."
echo "  work dir : ${WORK_DIR}"
echo "  out dir  : ${OUT_DIR}"
echo "  gmx bin  : ${GMX_BIN}"
echo "  interval : every ${DUMP_INTERVAL} steps"
echo "  frames   : ${FRAME_COUNT}"
