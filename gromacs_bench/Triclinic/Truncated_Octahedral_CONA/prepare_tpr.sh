#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pick_gromacs_bin() {
  local candidate=""

  for candidate in \
    "${GROMACS_BIN:-${GMX_BIN:-}}" \
    /mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi_d \
    /mnt/home/jliang/local/gromacs_esp_gpu/build_cpu_double/bin/gmx_mpi_d \
    /mnt/home/jliang/local/gromacs_esp_gpu/build_gpu/bin/gmx_mpi \
    /Users/jliang/Documents/GitHub/Ewald-Splitting-with-Prolates-GROMACS/build/bin/gmx
  do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  for candidate in gmx_mpi_d gmx_mpi gmx; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done

  echo "Error: could not find a GROMACS executable. Set GROMACS_BIN or GMX_BIN." >&2
  exit 1
}

GROMACS_BIN="$(pick_gromacs_bin)"

cd "${SCRIPT_DIR}"

"${GROMACS_BIN}" grompp \
  -f mdp/em.mdp \
  -c system/complex-solv-ions.gro \
  -p system/complex.top \
  -o cona_em.tpr

echo "Done: cona_em.tpr"
