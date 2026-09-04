#!/usr/bin/env python3
"""Production ESP-AD and Ewald helpers for the Figure 5 validation stage.

This module contains no pilot parameter selection and no finite-band direct
reference.  Callers must verify their frozen prediction artifact before
calling these routines, which run production ESP-AD and tight Ewald on the
validation trajectory.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ad_validation_common import (
    HERE,
    PROJECT,
    ADCase,
    ad_kspace_input,
    parse_log_metadata,
    project_relative,
    run_lammps,
)
import fixed_ik_reference as ikref


WORK = HERE / "runs_water"
WATER_ROOT = PROJECT / "numerical_examples/water_trajectory_benchmark"
WATER_DATA = WATER_ROOT / "water.data"
TRAJECTORY = WATER_ROOT / "water_short_traj.lammpstrj"
OLD_EWALD = WATER_ROOT / "forces.ref_ewald.dump"

PILOT_COUNT = 25
TOTAL_COUNT = 51

MOLECULAR_PREAMBLE = """bond_style harmonic
angle_style harmonic
bond_coeff 1 0.0 1.000
angle_coeff 1 0.0 109.47
special_bonds lj/coul 0 0 0.5

"""


def water_input(
    case: ADCase, dump_path: Path, input_path: Path, scale_zero: bool
) -> None:
    input_path.write_text(
        ad_kspace_input(
            case,
            WATER_DATA,
            TRAJECTORY,
            dump_path,
            atom_style="full",
            molecular_preamble=MOLECULAR_PREAMBLE,
            kspace_scale_zero=scale_zero,
        )
    )


def run_water_case(case: ADCase, rerun: bool):
    """Run or read a full-trajectory production ESP-AD force calculation."""

    case_dir = WORK / case.case_id / "water"
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for label, scale_zero in (("full", False), ("scale0", True)):
        dump = case_dir / f"forces.{label}.dump"
        inp = case_dir / f"in.{label}"
        log = case_dir / f"log.{label}.lammps"
        screen = case_dir / f"screen.{label}.txt"
        water_input(case, dump, inp, scale_zero)
        if rerun or not dump.is_file():
            run_lammps(inp, log, screen)
        metadata = parse_log_metadata(log)
        expected_grid = (case.mesh,) * 3
        if tuple(metadata["actual_grid"]) != expected_grid:
            raise RuntimeError(
                f"stale AD water grid for {case.case_id}: "
                f"{metadata['actual_grid']} != {expected_grid}"
            )
        if abs(float(metadata["actual_csplit"]) - case.csplit) > 5.0e-4:
            raise RuntimeError(f"stale AD water csplit for {case.case_id}")
        if abs(float(metadata["actual_cspread"]) - case.cspread) > 5.0e-4:
            raise RuntimeError(f"stale AD water cspread for {case.case_id}")
        if not bool(metadata["diff_ad"]):
            raise RuntimeError(f"non-AD water output for {case.case_id}")
        paths[label] = {"dump": dump, "input": inp, "log": log, "screen": screen}
    full = ikref.parse_force_dump(paths["full"]["dump"])
    zero = ikref.parse_force_dump(paths["scale0"]["dump"])
    if len(full) != TOTAL_COUNT or len(zero) != TOTAL_COUNT:
        raise RuntimeError(f"water frame count mismatch for {case.case_id}")
    result = []
    for (time_full, force_full), (time_zero, force_zero) in zip(full, zero):
        if time_full != time_zero:
            raise RuntimeError("full/scale-zero timestep mismatch")
        result.append((time_full, force_full, force_zero, force_full - force_zero))
    return result, paths


def ewald_input(dump_path: Path) -> str:
    return f"""newton on
units real
atom_style full
read_data {project_relative(WATER_DATA)}
reset_timestep 0

{MOLECULAR_PREAMBLE}pair_style coul/long 9.0
kspace_style ewald 1.0e-12
pair_coeff * *

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes one 4000
thermo_style custom step atoms ecoul elong
thermo 100
dump force all custom 1 {project_relative(dump_path)} id fx fy fz
dump_modify force sort id format line \"%d %.17g %.17g %.17g\"
rerun {project_relative(TRAJECTORY)} dump x y z ix iy iz box yes format native
"""


def refresh_ewald_reference(rerun: bool):
    """Run or read the current Ewald reference and audit the archived dump."""

    work = WORK / "ewald_reference"
    work.mkdir(parents=True, exist_ok=True)
    dump = work / "forces.ewald_current.dump"
    inp = work / "in.ewald_current"
    log = work / "log.ewald_current.lammps"
    screen = work / "screen.ewald_current.txt"
    inp.write_text(ewald_input(dump))
    if rerun or not dump.is_file():
        run_lammps(inp, log, screen)
    current = ikref.parse_force_dump(dump)
    previous = ikref.parse_force_dump(OLD_EWALD)
    if len(current) != TOTAL_COUNT or len(previous) != TOTAL_COUNT:
        raise RuntimeError("Ewald reference frame count mismatch")
    rows = []
    for frame_index, ((time_new, force_new), (time_old, force_old)) in enumerate(
        zip(current, previous)
    ):
        if time_new != time_old:
            raise RuntimeError("old/new Ewald timestep mismatch")
        difference = force_new - force_old
        rows.append(
            {
                "frame_zero_based": frame_index,
                "timestep": time_new,
                "rms_absolute_difference": math.sqrt(
                    float(np.mean(np.sum(difference**2, axis=1)))
                ),
                "rms_relative_difference": math.sqrt(
                    float(np.sum(difference**2) / np.sum(force_new**2))
                ),
                "maximum_component_difference": float(np.max(np.abs(difference))),
            }
        )
    return current, rows, {"dump": dump, "input": inp, "log": log, "screen": screen}


def sem(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def partition_summary(case: ADCase, rows: list[dict], partition: str) -> dict:
    """Summarize production total-force errors using balanced five-frame blocks."""

    selected = rows if partition == "all" else [row for row in rows if row["partition"] == partition]
    diff2 = sum(float(row["sum_total_difference_squared"]) for row in selected)
    ref2 = sum(float(row["sum_reference_squared"]) for row in selected)
    mesh2 = np.asarray([float(row["mesh_total_rms_absolute"]) ** 2 for row in selected])
    pair2 = np.asarray([float(row["mesh_pair_rms_absolute"]) ** 2 for row in selected])
    self2 = np.asarray([float(row["residual_self_rms_absolute"]) ** 2 for row in selected])
    cross = np.asarray([float(row["pair_self_dot_mean"]) for row in selected])
    relative_frames = np.asarray([float(row["total_relative_error"]) for row in selected])
    n_blocks = max(1, len(selected) // 5)
    block_relative = []
    start = 0
    for block_index in range(n_blocks):
        stop = start + 5 if block_index < n_blocks - 1 else len(selected)
        block = selected[start:stop]
        block_diff2 = sum(float(row["sum_total_difference_squared"]) for row in block)
        block_ref2 = sum(float(row["sum_reference_squared"]) for row in block)
        block_relative.append(math.sqrt(block_diff2 / block_ref2))
        start = stop
    return {
        "case_id": case.case_id,
        "partition": partition,
        "n_frames": len(selected),
        "target_relative_error": case.target_relative_error,
        "mesh": case.mesh,
        "order": case.order,
        "csplit": case.csplit,
        "cspread": case.cspread,
        "pooled_total_relative_error": math.sqrt(diff2 / ref2),
        "mean_frame_total_relative_error": float(relative_frames.mean()),
        "block5_sem_total_relative_error": sem(
            np.asarray(block_relative, dtype=np.float64)
        ),
        "pooled_mesh_total_absolute": math.sqrt(float(mesh2.mean())),
        "pooled_mesh_pair_absolute": math.sqrt(float(pair2.mean())),
        "pooled_residual_self_absolute": math.sqrt(float(self2.mean())),
        "pooled_pair_self_correlation": float(cross.mean())
        / max(math.sqrt(float(pair2.mean() * self2.mean())), 1.0e-300),
        "maximum_lammps_numpy_kspace_relative": max(
            float(row["lammps_numpy_kspace_relative_rms"]) for row in selected
        ),
        "target_satisfied": math.sqrt(diff2 / ref2) <= case.target_relative_error,
    }
