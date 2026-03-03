#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"

CONF_DIR="${CONF_DIR:-${ROOT_DIR}/100-conf}"
TOP_FILE="${TOP_FILE:-${ROOT_DIR}/top/topol.top}"
MDP_DIR="${MDP_DIR:-${ROOT_DIR}/mdp}"
MDP_ESP="${MDP_ESP:-${MDP_DIR}/esp.mdp}"
MDP_PME="${MDP_PME:-${MDP_DIR}/pme.mdp}"
MDP_PME_REF="${MDP_PME_REF:-${MDP_DIR}/pme_ref.mdp}"

GMX_ESP="${GMX_ESP:-/mnt/home/jliang/local/gromacs_esp_gpu/build_cpu_double/bin/gmx_mpi_d}"
GMX_PME="${GMX_PME:-/mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi_d}"

OPENMPI_LIB="/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/lib"
export LD_LIBRARY_PATH="${OPENMPI_LIB}:${LD_LIBRARY_PATH:-}"
export GMX_NBNXN_EWALD_ANALYTICAL=ON

NTOMP="${NTOMP:-16}"
MAXWARN="${MAXWARN:-1}"
MDRUN_NSTEPS="${MDRUN_NSTEPS:-0}"
PIN_MODE="${PIN_MODE:-off}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${ROOT_DIR}/force_eval_first100_esp_pme_pmeref_${RUN_TAG}"
CASES_DIR="${OUT_ROOT}/cases"
mkdir -p "${CASES_DIR}"

for f in "${CONF_DIR}" "${TOP_FILE}" "${MDP_ESP}" "${MDP_PME}" "${MDP_PME_REF}" "${GMX_ESP}" "${GMX_PME}"; do
  if [[ ! -e "${f}" ]]; then
    echo "Error: required file not found: ${f}" >&2
    exit 1
  fi
done

maybe_load_modules() {
  if [[ "${SKIP_MODULES:-0}" == "1" ]]; then
    return 0
  fi
  if type module >/dev/null 2>&1; then
    module load openmpi gcc fftw openblas cuda
  else
    echo "[info] module command not found; assuming runtime dependencies are already available" >&2
  fi
}

maybe_load_modules

run_mode() {
  local case_dir="$1"
  local mode="$2"
  local mdp_file="$3"
  local gmx_bin="$4"
  local force_prefix="$5"

  local mode_dir="${case_dir}/${mode}"
  mkdir -p "${mode_dir}"

  "${gmx_bin}" grompp \
    -f "${mdp_file}" \
    -c "${case_dir}/input.gro" \
    -r "${case_dir}/input.gro" \
    -p "${TOP_FILE}" \
    -o "${mode_dir}/${mode}.tpr" \
    -maxwarn "${MAXWARN}" \
    > "${mode_dir}/grompp.log" 2>&1

  rm -f "${mode_dir}/force_longrange_${force_prefix}.txt" "${mode_dir}/force_shortrange_${force_prefix}.txt"

  local ok=0
  local rc_try=999
  local nt_used="NA"
  for nt in "${NTOMP}" 8 4 1; do
    set +e
    (
      cd "${mode_dir}" && \
      "${gmx_bin}" mdrun \
        -s "${mode}.tpr" \
        -ntomp "${nt}" \
        -pin "${PIN_MODE}" \
        -nsteps "${MDRUN_NSTEPS}" \
        -noconfout \
        -cpt -1 \
        -deffnm "${mode}" \
        > "mdrun_nt${nt}.log" 2>&1
    )
    rc_try=$?
    set -e

    if [[ -f "${mode_dir}/force_longrange_${force_prefix}.txt" && -f "${mode_dir}/force_shortrange_${force_prefix}.txt" ]]; then
      ok=1
      nt_used="${nt}"
      break
    fi
  done

  if [[ "${ok}" -ne 1 ]]; then
    echo "Error: missing force files in ${mode_dir}" >&2
    exit 1
  fi

  echo "mode=${mode} ntomp=${nt_used} rc=${rc_try}"
}

for i in $(seq 1 100); do
  frame_name="$(printf "frame_%03d" "${i}")"
  frame_gro="${CONF_DIR}/${frame_name}.gro"
  if [[ ! -f "${frame_gro}" ]]; then
    echo "Error: missing input frame: ${frame_gro}" >&2
    exit 1
  fi

  case_dir="${CASES_DIR}/${frame_name}"
  mkdir -p "${case_dir}"
  cp -f "${frame_gro}" "${case_dir}/input.gro"

  echo "=== ${frame_name} ==="
  run_mode "${case_dir}" "PME_REF" "${MDP_PME_REF}" "${GMX_PME}" "pme"
  run_mode "${case_dir}" "PME"     "${MDP_PME}"     "${GMX_PME}" "pme"
  run_mode "${case_dir}" "ESP"     "${MDP_ESP}"     "${GMX_ESP}" "esp"
done

python3 - "${OUT_ROOT}" <<PY
import math
import sys
from pathlib import Path
import numpy as np

out_root = Path(sys.argv[1])
cases_dir = out_root / "cases"
rows = []
num_pme = 0.0
den_pme = 0.0
num_esp = 0.0
den_esp = 0.0

for i in range(1, 101):
    frame = f"frame_{i:03d}"
    ref_dir = cases_dir / frame / "PME_REF"
    pme_dir = cases_dir / frame / "PME"
    esp_dir = cases_dir / frame / "ESP"

    ref = np.loadtxt(ref_dir / "force_longrange_pme.txt") + np.loadtxt(ref_dir / "force_shortrange_pme.txt")
    pme = np.loadtxt(pme_dir / "force_longrange_pme.txt") + np.loadtxt(pme_dir / "force_shortrange_pme.txt")
    esp = np.loadtxt(esp_dir / "force_longrange_esp.txt") + np.loadtxt(esp_dir / "force_shortrange_esp.txt")

    ref = ref.reshape(-1, 3)
    pme = pme.reshape(-1, 3)
    esp = esp.reshape(-1, 3)

    if ref.shape != pme.shape or ref.shape != esp.shape:
      raise ValueError(f"shape mismatch in {frame}: ref={ref.shape}, pme={pme.shape}, esp={esp.shape}")

    d_pme = pme - ref
    d_esp = esp - ref

    n_pme = float(np.sum(d_pme * d_pme))
    n_esp = float(np.sum(d_esp * d_esp))
    d_ref = float(np.sum(ref * ref))

    delta_pme = math.sqrt(n_pme / d_ref) if d_ref > 0 else float("nan")
    delta_esp = math.sqrt(n_esp / d_ref) if d_ref > 0 else float("nan")

    num_pme += n_pme
    den_pme += d_ref
    num_esp += n_esp
    den_esp += d_ref

    rows.append((frame, ref.shape[0], delta_pme, delta_esp))

summary = out_root / "force_error_summary.tsv"
with summary.open("w", encoding="utf-8") as f:
    f.write("case\tN_atoms\tdelta_PME_vs_PME_REF\tdelta_ESP_vs_PME_REF\n")
    for frame, n_atoms, dp, de in rows:
        f.write(f"{frame}\t{n_atoms}\t{dp:.16e}\t{de:.16e}\n")

global_pme = math.sqrt(num_pme / den_pme) if den_pme > 0 else float("nan")
global_esp = math.sqrt(num_esp / den_esp) if den_esp > 0 else float("nan")

global_txt = out_root / "force_error_global.txt"
with global_txt.open("w", encoding="utf-8") as f:
    f.write(f"valid_cases\t{len(rows)}\n")
    f.write(f"global_delta_PME_vs_PME_REF\t{global_pme:.16e}\n")
    f.write(f"global_delta_ESP_vs_PME_REF\t{global_esp:.16e}\n")

print(f"wrote: {summary}")
print(f"wrote: {global_txt}")
print(f"global_delta_PME_vs_PME_REF={global_pme:.16e}")
print(f"global_delta_ESP_vs_PME_REF={global_esp:.16e}")
PY

echo "Done. Output: ${OUT_ROOT}"
