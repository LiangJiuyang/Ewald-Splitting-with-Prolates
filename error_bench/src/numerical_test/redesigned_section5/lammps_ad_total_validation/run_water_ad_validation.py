#!/usr/bin/env python3
"""Pilot-freeze and 51-frame production-LAMMPS AD validation for SPC/E water.

The script enforces the following order:

1. read only frames 0--24 and no Ewald total-force reference;
2. use a matched finite-band direct reference to build homogeneous and
   pilot-conditioned AD predictions, then freeze them;
3. only then run/read production LAMMPS and the independent Ewald reference;
4. report pilot, holdout, and all-frame errors without changing parameters.

The pilot-conditioned result is an implementation-specific operator
calibration, not the homogeneous analytical AD estimator of the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ad_validation_common import (
    HERE,
    LMP,
    PROJECT,
    RCUT,
    ADCase,
    ad_kspace_input,
    case_dict,
    coefficients,
    correction_force,
    fit_self_correction,
    lammps_version,
    operator,
    parse_log_metadata,
    project_relative,
    reference_operator_dependencies,
    residual_self_cell_rms,
    run_lammps,
    sha256,
    water_cases,
    write_csv,
    write_json,
)

import fixed_ad_reference as adref
import fixed_ik_reference as ikref


WORK = HERE / "runs_water"
WATER_ROOT = PROJECT / "numerical_examples/water_trajectory_benchmark"
WATER_DATA = WATER_ROOT / "water.data"
TRAJECTORY = WATER_ROOT / "water_short_traj.lammpstrj"
OLD_EWALD = WATER_ROOT / "forces.ref_ewald.dump"
PILOT_INPUTS = HERE / "water_ad_pilot_inputs.json"

PILOT_COUNT = 25
TOTAL_COUNT = 51

SELF_CSV = HERE / "water_ad_self_probe.csv"
PILOT_CSV = HERE / "water_ad_pilot_prediction_by_frame.csv"
FROZEN_JSON = HERE / "water_ad_pilot_frozen.json"
DETAIL_CSV = HERE / "water_ad_total_by_frame.csv"
SUMMARY_CSV = HERE / "water_ad_partition_summary.csv"
COMPONENT_CSV = HERE / "water_ad_estimator_components.csv"
REFERENCE_CSV = HERE / "water_ewald_reference_crosscheck.csv"
MANIFEST = HERE / "water_ad_manifest.json"
RAW_MANIFEST = HERE / "water_raw_artifact_manifest.json"
SOURCE_SNAPSHOT_MANIFEST = HERE / "lammps_source_snapshot/source_snapshot_manifest.json"

MOLECULAR_PREAMBLE = """bond_style harmonic
angle_style harmonic
bond_coeff 1 0.0 1.000
angle_coeff 1 0.0 109.47
special_bonds lj/coul 0 0 0.5

"""


def parse_trajectory_prefix(path: Path, count: int):
    frames = []
    with path.open() as handle:
        while len(frames) < count:
            header = handle.readline()
            if not header:
                break
            if not header.startswith("ITEM: TIMESTEP"):
                raise RuntimeError(f"malformed trajectory header in {path}")
            timestep = int(handle.readline())
            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("missing atom count")
            natoms = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("missing box bounds")
            bounds = [tuple(map(float, handle.readline().split()[:2])) for _ in range(3)]
            columns = handle.readline().split()[2:]
            col = {name: index for index, name in enumerate(columns)}
            ids = np.empty(natoms, dtype=np.int64)
            q = np.empty(natoms, dtype=np.float64)
            xyz = np.empty((natoms, 3), dtype=np.float64)
            for row in range(natoms):
                fields = handle.readline().split()
                ids[row] = int(fields[col["id"]])
                q[row] = float(fields[col["q"]])
                xyz[row] = [float(fields[col[name]]) for name in ("x", "y", "z")]
            order = np.argsort(ids)
            lo = np.asarray([value[0] for value in bounds])
            lengths = np.asarray([value[1] - value[0] for value in bounds])
            if not np.allclose(lengths, lengths[0]):
                raise RuntimeError("water validation requires a cubic box")
            frames.append((timestep, q[order], xyz[order] - lo, float(lengths[0])))
    if len(frames) != count:
        raise RuntimeError(f"expected {count} trajectory frames, found {len(frames)}")
    return frames


def load_pilot_inputs() -> dict[float, dict[str, float]]:
    payload = json.loads(PILOT_INPUTS.read_text())
    source_record = payload["source_artifact"]
    source_path = PROJECT / source_record["path"]
    if not source_path.is_file() or sha256(source_path) != source_record["sha256"]:
        raise RuntimeError("Figure-5 pilot source artifact is missing or has changed")
    result = {
        float(row["target_relative_error"]): row
        for row in payload["pilot_only_scalars"]
    }
    for case in water_cases():
        row = result.get(case.target_relative_error)
        if row is None:
            raise RuntimeError(f"missing Figure-5 pilot input for {case.case_id}")
        expected = {
            "mesh": case.mesh,
            "order": case.order,
            "split_input_tolerance": case.split_input_tolerance,
            "spread_input_tolerance": case.spread_input_tolerance,
            "csplit": case.csplit,
            "cspread": case.cspread,
        }
        for key, value in expected.items():
            observed = float(row[key])
            if not math.isclose(observed, float(value), rel_tol=0.0, abs_tol=1.0e-12):
                raise RuntimeError(
                    f"stale Figure-5 pilot tuple for {case.case_id}: "
                    f"{key}={observed}, expected {value}"
                )
        if not math.isclose(
            float(row["split_input_tolerance"]), case.target_relative_error,
            rel_tol=0.0, abs_tol=1.0e-15,
        ):
            raise RuntimeError(f"split tolerance is not target-aligned: {case.case_id}")
    return result


def pilot_frame_data_sha256(frames) -> str:
    """Hash only parsed pilot values, without reading beyond frame 24."""
    digest = hashlib.sha256()
    for timestep, q, xyz, box in frames:
        digest.update(np.asarray([timestep], dtype="<i8").tobytes())
        digest.update(np.asarray([box], dtype="<f8").tobytes())
        digest.update(np.asarray(q, dtype="<f8").tobytes())
        digest.update(np.asarray(xyz, dtype="<f8").tobytes())
    return digest.hexdigest()


def sem(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def pilot_block_relative_values(
    mesh_errors: list[np.ndarray], fourier_absolute: float, force_scale: float
) -> np.ndarray:
    values = []
    for start in range(0, len(mesh_errors), 5):
        block = mesh_errors[start : start + 5]
        mesh_absolute = rms_vectors(block)
        values.append(math.hypot(mesh_absolute, fourier_absolute) / force_scale)
    return np.asarray(values, dtype=np.float64)


def rms_vectors(values: list[np.ndarray]) -> float:
    return math.sqrt(float(np.mean([np.mean(np.sum(value**2, axis=1)) for value in values])))


def pilot_prediction(
    case: ADCase,
    frames,
    ab: np.ndarray,
    self_cell_rms: float,
    fig5: dict[str, float],
) -> tuple[list[dict], dict]:
    coeff = coefficients(case)
    op = operator(case, frames[0][3])
    mesh_errors: list[np.ndarray] = []
    rows: list[dict] = []
    for frame_index, (timestep, q, xyz, box) in enumerate(frames):
        raw = adref.fixed_ad_mesh_force(q, xyz, op, coeff.real)
        corrected = raw - correction_force(q, xyz, box, case.mesh, ab)
        direct = ikref.direct_truncated_force(q, xyz, box, op.kernel)
        error = corrected - direct
        mesh_errors.append(error)
        rows.append(
            {
                "case_id": case.case_id,
                "target_relative_error": case.target_relative_error,
                "frame_zero_based": frame_index,
                "timestep": timestep,
                "mesh": case.mesh,
                "order": case.order,
                "csplit": case.csplit,
                "cspread": case.cspread,
                "pilot_mesh_rms_absolute": math.sqrt(
                    float(np.mean(np.sum(error**2, axis=1)))
                ),
                "ewald_total_force_accessed": "false",
                "matched_finite_band_direct_reference_used": "true",
                "holdout_coordinate_accessed": "false",
            }
        )
    mesh_absolute = rms_vectors(mesh_errors)
    fourier_absolute = fig5["pilot_fourier_absolute"]
    total_absolute = math.hypot(mesh_absolute, fourier_absolute)
    force_scale = fig5["pilot_force_scale"]
    relative = total_absolute / force_scale
    block_relative = pilot_block_relative_values(
        mesh_errors, fourier_absolute, force_scale
    )
    relative_sem = sem(block_relative)
    # One-sided 95% Student-t quantile for the five prespecified blocks (df=4).
    one_sided_t95_df4 = 2.13184678632665
    upper95 = relative + one_sided_t95_df4 * relative_sem

    q = frames[0][1]
    pair_homogeneous, pair_meta = adref.fixed_ad_pair_estimate_cell_moments(
        q,
        frames[0][3],
        case.mesh,
        case.order,
        RCUT,
        case.csplit,
        case.cspread,
        coeff,
        quadrature_order_per_half=32,
    )
    q4_mean = float(np.mean(q**4))
    self_prediction = math.sqrt(q4_mean) * self_cell_rms
    homogeneous_total_absolute = math.sqrt(
        pair_homogeneous**2 + self_prediction**2 + fourier_absolute**2
    )
    return rows, {
        "case_id": case.case_id,
        "target_relative_error": case.target_relative_error,
        "mesh": case.mesh,
        "order": case.order,
        "csplit": case.csplit,
        "cspread": case.cspread,
        "pilot_frames": len(frames),
        "pilot_force_scale": force_scale,
        "pilot_fourier_absolute": fourier_absolute,
        "homogeneous_pair_absolute": pair_homogeneous,
        "residual_self_absolute": self_prediction,
        "homogeneous_total_absolute": homogeneous_total_absolute,
        "homogeneous_total_relative": homogeneous_total_absolute / force_scale,
        "homogeneous_a_priori_feasible": homogeneous_total_absolute / force_scale
        <= case.target_relative_error,
        "pilot_conditioned_mesh_absolute": mesh_absolute,
        "pilot_conditioned_total_absolute": total_absolute,
        "pilot_conditioned_total_relative": relative,
        "pilot_conditioned_block5_sem_relative": relative_sem,
        "pilot_conditioned_one_sided_95_upper_relative": upper95,
        "pilot_conditioned_feasible": upper95 <= case.target_relative_error,
        "minimum_abs_deconvolution_product": pair_meta[
            "minimum_abs_deconvolution_product"
        ],
        "evaluation_role": "prespecified-case, pilot-conditioned matched-operator calibration; finite-band direct reference used; no Ewald total-force reference or holdout coordinates",
    }


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


def partition_summary(case: ADCase, rows: list[dict], partition: str) -> dict:
    selected = rows if partition == "all" else [row for row in rows if row["partition"] == partition]
    diff2 = sum(float(row["sum_total_difference_squared"]) for row in selected)
    ref2 = sum(float(row["sum_reference_squared"]) for row in selected)
    mesh2 = np.asarray([float(row["mesh_total_rms_absolute"]) ** 2 for row in selected])
    pair2 = np.asarray([float(row["mesh_pair_rms_absolute"]) ** 2 for row in selected])
    self2 = np.asarray([float(row["residual_self_rms_absolute"]) ** 2 for row in selected])
    cross = np.asarray([float(row["pair_self_dot_mean"]) for row in selected])
    relative_frames = np.asarray([float(row["total_relative_error"]) for row in selected])
    # Use contiguous, near-five-frame blocks without creating a singleton
    # remainder.  Thus 25, 26, and 51 frames are partitioned as
    # 5+5+5+5+5, 5+5+5+5+6, and 5+...+5+6, respectively.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-lammps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rerun = not args.reuse_lammps
    started = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    pilot_inputs = load_pilot_inputs()

    # Deliberately stop reading at frame 24 for prediction and freezing.
    pilot_frames = parse_trajectory_prefix(TRAJECTORY, PILOT_COUNT)
    q0 = pilot_frames[0][1]
    box = pilot_frames[0][3]

    self_rows: list[dict] = []
    pilot_rows: list[dict] = []
    component_rows: list[dict] = []
    effective_cases: list[ADCase] = []
    correction_by_id: dict[str, np.ndarray] = {}
    self_cell_by_id: dict[str, float] = {}

    for case in water_cases():
        ab, rows, audit = fit_self_correction(case, box, WORK, rerun=rerun)
        if audit["holdout_max_abs_component"] > 5.0e-10:
            raise RuntimeError(f"water self-correction audit failed: {case.case_id}")
        actual_case = replace(case, mesh=int(audit["actual_mesh"]))
        if actual_case.mesh != case.mesh:
            raise RuntimeError(
                f"water requested grid changed unexpectedly: {case.mesh} -> {actual_case.mesh}"
            )
        self8 = residual_self_cell_rms(actual_case, box, ab, 8)
        self12 = residual_self_cell_rms(actual_case, box, ab, 12)
        if abs(self12 - self8) / max(self12, 1.0e-300) > 1.0e-7:
            raise RuntimeError(f"water self quadrature not converged: {case.case_id}")
        rows = [
            {
                **row,
                "self_cell_rms_n8_per_half": self8,
                "self_cell_rms_n12_per_half": self12,
            }
            for row in rows
        ]
        self_rows.extend(rows)
        frame_rows, component = pilot_prediction(
            actual_case,
            pilot_frames,
            ab,
            self12,
            pilot_inputs[case.target_relative_error],
        )
        pilot_rows.extend(frame_rows)
        component.update(
            correction_holdout_max_abs=audit["holdout_max_abs_component"],
            self_cell_n8_to_n12_relative=abs(self12 - self8) / max(self12, 1.0e-300),
        )
        component_rows.append(component)
        effective_cases.append(actual_case)
        correction_by_id[case.case_id] = ab
        self_cell_by_id[case.case_id] = self12

    write_csv(SELF_CSV, self_rows)
    write_csv(PILOT_CSV, pilot_rows)
    write_csv(COMPONENT_CSV, component_rows)
    write_json(
        FROZEN_JSON,
        {
            "created_before_ewald_or_holdout_coordinate_access": True,
            "pilot_frame_indices": list(range(PILOT_COUNT)),
            "holdout_coordinate_values_used_for_calibration": False,
            "ewald_total_force_accessed_before_freeze": False,
            "matched_finite_band_direct_reference_used": True,
            "acceptance_rule": "prespecified case is pilot-feasible when the pilot-conditioned total relative point estimate plus the one-sided 95% Student-t multiplier (df=4) times the five-block SEM is <= target",
            "prespecified_cases": component_rows,
            "pilot_frame_data_sha256": pilot_frame_data_sha256(pilot_frames),
            "pilot_input_artifact": {
                "path": str(PILOT_INPUTS.relative_to(PROJECT)),
                "sha256": sha256(PILOT_INPUTS),
            },
        },
    )

    # Only after the frozen artifact exists may the holdout/reference be read.
    full_frames = ikref.parse_charge_trajectory(TRAJECTORY)
    if len(full_frames) != TOTAL_COUNT:
        raise RuntimeError("water trajectory no longer has 51 frames")
    reference, reference_rows, reference_paths = refresh_ewald_reference(rerun)
    write_csv(REFERENCE_CSV, reference_rows)

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    run_paths: list[Path] = []
    for case in effective_cases:
        print(f"production water: {case.case_id}", flush=True)
        observed, paths = run_water_case(case, rerun)
        for values in paths.values():
            run_paths.extend(values.values())
        coeff = coefficients(case)
        op = operator(case, box)
        ab = correction_by_id[case.case_id]
        case_rows: list[dict] = []
        for frame_index, (
            (timestep, q, xyz, frame_box),
            (time_lmp, full_force, scale0_force, kspace_force),
            (time_ref, ref_force),
        ) in enumerate(zip(full_frames, observed, reference)):
            if timestep != time_lmp or timestep != time_ref:
                raise RuntimeError("water timestep mismatch")
            raw = adref.fixed_ad_mesh_force(q, xyz, op, coeff.real)
            correction = correction_force(q, xyz, frame_box, case.mesh, ab)
            numpy_corrected = raw - correction
            fractions = np.mod(xyz / (frame_box / case.mesh), 1.0)
            raw_self = (q * q)[:, None] * adref.ad_self_response_at_fractions(
                fractions, op, coeff.real
            )
            residual_self = raw_self - correction
            direct = ikref.direct_truncated_force(q, xyz, frame_box, op.kernel)
            mesh_total_error = kspace_force - direct
            mesh_pair_error = kspace_force - residual_self - direct
            difference = full_force - ref_force
            sum_diff2 = float(np.sum(difference**2))
            sum_ref2 = float(np.sum(ref_force**2))
            pair_rms = math.sqrt(float(np.mean(np.sum(mesh_pair_error**2, axis=1))))
            self_rms = math.sqrt(float(np.mean(np.sum(residual_self**2, axis=1))))
            cross = float(np.mean(np.sum(mesh_pair_error * residual_self, axis=1)))
            row = {
                "case_id": case.case_id,
                "target_relative_error": case.target_relative_error,
                "frame_zero_based": frame_index,
                "timestep": timestep,
                "partition": "pilot" if frame_index < PILOT_COUNT else "holdout",
                "mesh": case.mesh,
                "order": case.order,
                "csplit": case.csplit,
                "cspread": case.cspread,
                "sum_total_difference_squared": sum_diff2,
                "sum_reference_squared": sum_ref2,
                "total_rms_absolute_error": math.sqrt(sum_diff2 / len(q)),
                "total_relative_error": math.sqrt(sum_diff2 / sum_ref2),
                "mesh_total_rms_absolute": math.sqrt(
                    float(np.mean(np.sum(mesh_total_error**2, axis=1)))
                ),
                "mesh_pair_rms_absolute": pair_rms,
                "residual_self_rms_absolute": self_rms,
                "pair_self_dot_mean": cross,
                "pair_self_correlation": cross / max(pair_rms * self_rms, 1.0e-300),
                "lammps_numpy_kspace_relative_rms": ikref.rms_vector_error(
                    kspace_force, numpy_corrected
                )
                / max(
                    math.sqrt(float(np.mean(np.sum(numpy_corrected**2, axis=1)))),
                    1.0e-300,
                ),
                "lammps_numpy_kspace_max_component": float(
                    np.max(np.abs(kspace_force - numpy_corrected))
                ),
                "scale0_force_rms": math.sqrt(
                    float(np.mean(np.sum(scale0_force**2, axis=1)))
                ),
            }
            case_rows.append(row)
            detail_rows.append(row)
        for partition in ("pilot", "holdout", "all"):
            summary_rows.append(partition_summary(case, case_rows, partition))

    write_csv(DETAIL_CSV, detail_rows)
    write_csv(SUMMARY_CSV, summary_rows)

    raw_files = sorted(
        path
        for path in WORK.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and not path.name.startswith("._")
    )
    write_json(
        RAW_MANIFEST,
        {
            "purpose": "complete hash index of generated water/self-probe LAMMPS inputs, logs, screens, and force dumps",
            "artifacts": [
                {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
                for path in raw_files
            ],
        },
    )

    source_files = [
        Path(__file__),
        HERE / "ad_validation_common.py",
        *reference_operator_dependencies(),
    ]
    output_files = [
        SELF_CSV,
        PILOT_CSV,
        FROZEN_JSON,
        DETAIL_CSV,
        SUMMARY_CSV,
        COMPONENT_CSV,
        REFERENCE_CSV,
        RAW_MANIFEST,
    ]
    run_paths.extend(reference_paths.values())
    write_json(
        MANIFEST,
        {
            "purpose": "pilot-frozen evaluation of three prespecified cases on a 51-frame production-LAMMPS ESP AD total-force validation",
            "logical_order": [
                "single-charge correction audit",
                "pilot-only operator prediction",
                "freeze artifact",
                "holdout and Ewald access",
            ],
            "pilot_conditioned_scope": "prespecified-case, implementation-specific matched-operator calibration; not a molecular closed-form AD estimator or an AD candidate search",
            "force_units": "kcal mol^-1 A^-1 (LAMMPS units real)",
            "lammps_executable": str(LMP.relative_to(PROJECT)),
            "lammps_executable_sha256": sha256(LMP),
            "lammps_version": lammps_version(),
            "lammps_source_provenance": {
                "path": str(SOURCE_SNAPSHOT_MANIFEST.relative_to(PROJECT)),
                "sha256": sha256(SOURCE_SNAPSHOT_MANIFEST),
            },
            "trajectory": {
                "path": str(TRAJECTORY.relative_to(PROJECT)),
                "sha256": sha256(TRAJECTORY),
            },
            "pilot_input_artifact": {
                "path": str(PILOT_INPUTS.relative_to(PROJECT)),
                "sha256": sha256(PILOT_INPUTS),
            },
            "water_data": {
                "path": str(WATER_DATA.relative_to(PROJECT)),
                "sha256": sha256(WATER_DATA),
            },
            "old_ewald_reference": {
                "path": str(OLD_EWALD.relative_to(PROJECT)),
                "sha256": sha256(OLD_EWALD),
            },
            "cases": [case_dict(case) for case in effective_cases],
            "raw_artifact_index": {
                "path": str(RAW_MANIFEST.relative_to(PROJECT)),
                "sha256": sha256(RAW_MANIFEST),
                "n_files": len(raw_files),
            },
            "sources": [
                {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
                for path in source_files
            ],
            "outputs": [
                {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
                for path in output_files
            ],
            "run_artifacts": [
                {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
                for path in sorted(set(run_paths))
                if path.is_file()
            ],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
