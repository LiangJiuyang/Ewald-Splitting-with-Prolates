#!/usr/bin/env bash
set -u -o pipefail

BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONF_DIR="${CONF_DIR:-${BASE_DIR}/100-conf}"
OUT_ROOT="${OUT_ROOT:-${BASE_DIR}/force_eval_100conf}"
CASES_DIR="${CASES_DIR:-${OUT_ROOT}/cases}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/tmp/force_eval_100conf_scratch}"

GMX_ESP="${GMX_ESP:-/mnt/home/jliang/local/gromacs_esp_gpu/build_cpu_double/bin/gmx_mpi_d}"
GMX_ESP_REF="${GMX_ESP_REF:-${GMX_ESP}}"
GMX_PME="${GMX_PME:-/mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi_d}"

MDP_ESP="${MDP_ESP:-${BASE_DIR}/mdp/esp.mdp}"
MDP_ESP_REF="${MDP_ESP_REF:-${BASE_DIR}/mdp/esp_ref.mdp}"
MDP_PME="${MDP_PME:-${BASE_DIR}/mdp/pme.mdp}"
MDP_PME_REF="${MDP_PME_REF:-${BASE_DIR}/mdp/pme_ref.mdp}"
TOP="${TOP:-${BASE_DIR}/system/complex.top}"

START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-100}"
NTOMP="${NTOMP:-48}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
MAXWARN="${MAXWARN:-1}"
MDRUN_NSTEPS="${MDRUN_NSTEPS:-0}"
PIN_MODE="${PIN_MODE:-on}"
ESP_LONG_FORCE_FILE="${ESP_LONG_FORCE_FILE:-force_longrange_esp.txt}"
ESP_SHORT_FORCE_FILE="${ESP_SHORT_FORCE_FILE:-force_shortrange_esp.txt}"
ESP_REF_LONG_FORCE_FILE="${ESP_REF_LONG_FORCE_FILE:-${ESP_LONG_FORCE_FILE}}"
ESP_REF_SHORT_FORCE_FILE="${ESP_REF_SHORT_FORCE_FILE:-${ESP_SHORT_FORCE_FILE}}"
PME_LONG_FORCE_FILE="${PME_LONG_FORCE_FILE:-force_longrange_pme.txt}"
PME_SHORT_FORCE_FILE="${PME_SHORT_FORCE_FILE:-force_shortrange_pme.txt}"
RUN_ESP_REF="${RUN_ESP_REF:-no}"

OPENMPI_LIB="/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/lib"
export LD_LIBRARY_PATH="${OPENMPI_LIB}:${LD_LIBRARY_PATH:-}"
export GMX_NBNXN_EWALD_ANALYTICAL=ON
export OMP_NUM_THREADS="${NTOMP}"

mkdir -p "${CASES_DIR}"
mkdir -p "${SCRATCH_ROOT}"
FAIL_LOG="${OUT_ROOT}/run_failures.log"
touch "${FAIL_LOG}"
FAIL_COUNT=0

record_failure() {
    local case_name="$1"
    local mode="$2"
    local step="$3"
    local rc="$4"
    printf "%s\t%s\t%s\t%s\texit=%s\n" \
        "$(date +'%F %T')" "${case_name}" "${mode}" "${step}" "${rc}" >> "${FAIL_LOG}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[WARN] ${case_name} ${mode} ${step} failed (exit=${rc})"
}

run_mode() {
    local case_name="$1"
    local case_dir="$2"
    local mode="$3"
    local gmx_bin="$4"
    local mdp_file="$5"
    local long_file="$6"
    local short_file="$7"

    local mode_dir="${case_dir}/${mode}"
    local mode_scratch="${SCRATCH_ROOT}/${case_name}/${mode}"
    local tpr_file="${case_dir}/${mode}.tpr"
    local tpr_scratch="${mode_scratch}/${mode}.tpr"
    rm -rf "${mode_scratch}"
    mkdir -p "${mode_scratch}"
    mkdir -p "${mode_dir}"
    cp -f "${case_dir}/input.gro" "${mode_scratch}/input.gro"

    "${gmx_bin}" grompp \
        -f "${mdp_file}" \
        -c "${mode_scratch}/input.gro" \
        -p "${TOP}" \
        -o "${tpr_scratch}" \
        -maxwarn "${MAXWARN}" \
        > "${mode_scratch}/grompp.log" 2>&1
    local rc=$?
    cp -f "${mode_scratch}/grompp.log" "${mode_dir}/grompp.log" 2>/dev/null || true
    if (( rc != 0 )); then
        record_failure "${case_name}" "${mode}" "grompp" "${rc}"
        return 1
    fi
    cp -f "${tpr_scratch}" "${tpr_file}"

    rm -f "${mode_dir}"/force_*.txt
    (
        cd "${mode_scratch}" && \
        timeout "${TIMEOUT_SEC}s" "${gmx_bin}" mdrun \
            -s "${mode}.tpr" \
            -ntomp "${NTOMP}" \
            -pin "${PIN_MODE}" \
            -nsteps "${MDRUN_NSTEPS}" \
            -noconfout \
            -cpt -1 \
            -deffnm "${mode}" \
            > mdrun.log 2>&1
    )
    rc=$?
    cp -f "${mode_scratch}/mdrun.log" "${mode_dir}/mdrun.log" 2>/dev/null || true
    if (( rc != 0 )); then
        record_failure "${case_name}" "${mode}" "mdrun" "${rc}"
        return 1
    fi

    if [[ ! -f "${mode_scratch}/${long_file}" || ! -f "${mode_scratch}/${short_file}" ]]; then
        record_failure "${case_name}" "${mode}" "missing_force_file" 1
        return 1
    fi
    cp -f "${mode_scratch}/${long_file}" "${mode_dir}/${long_file}"
    cp -f "${mode_scratch}/${short_file}" "${mode_dir}/${short_file}"
    rm -rf "${mode_scratch}"

    return 0
}

for ((i = START_INDEX; i <= END_INDEX; i++)); do
    frame_name="$(printf 'frame_%03d' "${i}")"
    frame_gro="${CONF_DIR}/${frame_name}.gro"
    case_dir="${CASES_DIR}/${frame_name}"

    if [[ ! -f "${frame_gro}" ]]; then
        record_failure "${frame_name}" "ALL" "missing_input" 1
        continue
    fi

    mkdir -p "${case_dir}"
    cp -f "${frame_gro}" "${case_dir}/input.gro"

    echo "=== ${frame_name} ==="
    run_mode "${frame_name}" "${case_dir}" "ESP" "${GMX_ESP}" "${MDP_ESP}" \
        "${ESP_LONG_FORCE_FILE}" "${ESP_SHORT_FORCE_FILE}" || true
    if [[ "${RUN_ESP_REF}" == "yes" ]]; then
        run_mode "${frame_name}" "${case_dir}" "ESP_REF" "${GMX_ESP_REF}" "${MDP_ESP_REF}" \
            "${ESP_REF_LONG_FORCE_FILE}" "${ESP_REF_SHORT_FORCE_FILE}" || true
    fi
    run_mode "${frame_name}" "${case_dir}" "PME" "${GMX_PME}" "${MDP_PME}" \
        "${PME_LONG_FORCE_FILE}" "${PME_SHORT_FORCE_FILE}" || true
    run_mode "${frame_name}" "${case_dir}" "PME_REF" "${GMX_PME}" "${MDP_PME_REF}" \
        "${PME_LONG_FORCE_FILE}" "${PME_SHORT_FORCE_FILE}" || true
done

echo "Run completed. failures=${FAIL_COUNT}"
echo "Failure log: ${FAIL_LOG}"
