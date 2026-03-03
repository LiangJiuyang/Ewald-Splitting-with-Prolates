#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

GMX="${GMX:-/mnt/home/jliang/local/gromacs_esp_gpu/build_gpu/bin/gmx_mpi}"
EM_MDP="${EM_MDP:-${ROOT_DIR}/equilibration/em.mdp}"
NVT_MDP="${NVT_MDP:-${ROOT_DIR}/equilibration/nvt.mdp}"
NPT_MDP="${NPT_MDP:-${ROOT_DIR}/equilibration/npt.mdp}"
CONF_GRO="${CONF_GRO:-${ROOT_DIR}/1-plain-md/conf.gro}"
TOP_FILE="${TOP_FILE:-${ROOT_DIR}/top/topol.top}"
INDEX_FILE="${INDEX_FILE:-${ROOT_DIR}/top/index.ndx}"
EM_DEFFNM="${EM_DEFFNM:-${ROOT_DIR}/equilibration/em_full}"
NVT_DEFFNM="${NVT_DEFFNM:-${ROOT_DIR}/equilibration/nvt_full}"
NPT_DEFFNM="${NPT_DEFFNM:-${ROOT_DIR}/equilibration/npt_full}"

# Full EM for stable starting coordinates.
"$GMX" grompp \
  -f "${EM_MDP}" \
  -c "${CONF_GRO}" \
  -r "${CONF_GRO}" \
  -p "${TOP_FILE}" \
  -n "${INDEX_FILE}" \
  -o "${EM_DEFFNM}.tpr"

mpirun -np 1 "$GMX" mdrun \
  -deffnm "${EM_DEFFNM}" \
  -ntomp 8 \
  -nb cpu \
  -pme cpu \
  -bonded cpu \
  -update cpu

# Rebuild NVT input from fully minimized structure.
"$GMX" grompp \
  -f "${NVT_MDP}" \
  -c "${EM_DEFFNM}.gro" \
  -r "${CONF_GRO}" \
  -p "${TOP_FILE}" \
  -n "${INDEX_FILE}" \
  -o "${NVT_DEFFNM}.tpr"

# NVT equilibration (50,000 MD steps at 2 fs = 100 ps).
mpirun -np 1 "$GMX" mdrun \
  -deffnm "${NVT_DEFFNM}" \
  -ntomp 8 \
  -nb gpu \
  -pme gpu \
  -bonded gpu \
  -update cpu

# Build NPT input from NVT state.
"$GMX" grompp \
  -f "${NPT_MDP}" \
  -c "${NVT_DEFFNM}.gro" \
  -t "${NVT_DEFFNM}.cpt" \
  -r "${CONF_GRO}" \
  -p "${TOP_FILE}" \
  -n "${INDEX_FILE}" \
  -o "${NPT_DEFFNM}.tpr"

# NPT equilibration (500,000 MD steps at 2 fs = 1 ns).
mpirun -np 1 "$GMX" mdrun \
  -deffnm "${NPT_DEFFNM}" \
  -ntomp 8 \
  -nb gpu \
  -pme gpu \
  -bonded gpu \
  -update cpu
