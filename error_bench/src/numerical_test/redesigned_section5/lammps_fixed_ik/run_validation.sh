#!/bin/sh
set -eu

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
project_root="$(CDPATH= cd -- "$script_dir/../../../.." && pwd)"
lmp_executable="${LMP_FIXED_IK:-$script_dir/../pppm_symmetric_scan/lmp.pppm_symmetric_scan}"
work_dir="$script_dir/work"

mkdir -p "$work_dir"

cd "$project_root"
"$lmp_executable" \
  -in src/numerical_test/redesigned_section5/lammps_fixed_ik/in.random_fixed_ik \
  -log src/numerical_test/redesigned_section5/lammps_fixed_ik/work/log.esp_fixed_test
python3 src/numerical_test/redesigned_section5/lammps_fixed_ik/validate_lammps_fixed_ik.py
