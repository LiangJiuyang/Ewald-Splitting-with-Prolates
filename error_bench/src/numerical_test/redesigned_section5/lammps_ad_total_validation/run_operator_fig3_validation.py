#!/usr/bin/env python3
"""Run direct development-LAMMPS AD operator and Figure-3 validations.

This runner supplies the missing production-code closure:

* unit-charge probes identify and hold out the two-harmonic self correction;
* neutral two-charge probes test pair/self/total decomposition directly; and
* all valid Figure-3 sweep points are run through development LAMMPS on the
  ten random-charge configurations.

LAMMPS real-space pair forces are reconstructed from compute pair/local and
subtracted from the dumped total force.  The resulting reciprocal force can
therefore be compared directly with the independent NumPy AD operator.
"""

from __future__ import annotations

import argparse
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
    extract_kspace_frames,
    figure3_cases,
    fit_self_correction,
    lammps_version,
    operator,
    pooled_rms,
    pooled_rms_jackknife_sem,
    residual_self_cell_rms,
    reference_operator_dependencies,
    run_lammps,
    sha256,
    trajectory_text,
    write_csv,
    write_json,
    charge_data_text,
)

import fixed_ad_reference as adref
import fixed_ik_reference as ikref


WORK = HERE / "runs_operator_fig3"
SELF_CSV = HERE / "operator_self_probe.csv"
TWO_CHARGE_CSV = HERE / "operator_two_charge_audit.csv"
DETAIL_CSV = HERE / "fig3_lammps_ad_by_config.csv"
SUMMARY_CSV = HERE / "fig3_lammps_ad_summary.csv"
MANIFEST = HERE / "operator_fig3_manifest.json"
RAW_MANIFEST = HERE / "operator_raw_artifact_manifest.json"
SOURCE_SNAPSHOT_MANIFEST = HERE / "lammps_source_snapshot/source_snapshot_manifest.json"

BOX = 48.0
RANDOM_ROOT = PROJECT / "numerical_examples/random_charges"
TWO_CHARGE_FRACTIONS = np.asarray(
    [
        (0.071, 0.193, 0.317),
        (0.149, 0.283, 0.431),
        (0.237, 0.379, 0.557),
        (0.331, 0.487, 0.683),
        (0.419, 0.593, 0.809),
        (0.517, 0.701, 0.923),
        (0.613, 0.817, 0.109),
        (0.739, 0.907, 0.241),
    ],
    dtype=np.float64,
)


def relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    denominator = math.sqrt(float(np.mean(np.sum(b * b, axis=1))))
    return ikref.rms_vector_error(a, b) / denominator


def unique_valid_cases() -> tuple[list[ADCase], list[dict], dict[tuple, str]]:
    valid: list[ADCase] = []
    invalid: list[dict] = []
    key_to_id: dict[tuple, str] = {}
    for case in figure3_cases():
        try:
            operator(case, BOX)
        except FloatingPointError as error:
            invalid.append(
                {
                    **case_dict(case),
                    "status": "invalid_zero_deconvolution",
                    "reason": str(error),
                }
            )
            continue
        if case.tuple_key not in key_to_id:
            key_to_id[case.tuple_key] = case.case_id
            valid.append(case)
    return valid, invalid, key_to_id


def ordered_random_systems() -> tuple[np.ndarray, list[np.ndarray], list[Path]]:
    paths = [
        RANDOM_ROOT / f"config_{index:02d}/random_charges.data"
        for index in range(1, 11)
    ]
    systems = [ikref.parse_charge_data(path) for path in paths]
    base_q = systems[0][0]
    if any(len(q) != 512 or not math.isclose(box, BOX) for q, _, box in systems):
        raise RuntimeError("random benchmark no longer has N=512 and L=48")
    frames: list[np.ndarray] = []
    for q, xyz, _ in systems:
        reordered = np.empty_like(xyz)
        for sign in (-1.0, 1.0):
            target = np.flatnonzero(base_q == sign)
            source = np.flatnonzero(q == sign)
            if len(target) != len(source):
                raise RuntimeError("random configurations have incompatible charge counts")
            reordered[target] = xyz[source]
        frames.append(reordered)
    return base_q, frames, paths


def run_two_charge_case(
    case: ADCase, ab: np.ndarray, rerun: bool
) -> list[dict]:
    case_dir = WORK / case.case_id / "two_charge"
    case_dir.mkdir(parents=True, exist_ok=True)
    h = BOX / case.mesh
    cell = case.mesh // 2
    frames: list[np.ndarray] = []
    for fraction in TWO_CHARGE_FRACTIONS:
        first = h * (cell + fraction)
        second = np.mod(first + np.asarray([12.0, 0.0, 0.0]), BOX)
        delta = second - first
        delta -= BOX * np.rint(delta / BOX)
        if np.linalg.norm(delta) <= RCUT:
            raise RuntimeError("two-charge audit accidentally entered the real cutoff")
        frames.append(np.vstack((first, second)))
    q = np.asarray([1.0, -1.0])
    data_path = case_dir / "two_charge.data"
    trajectory_path = case_dir / "two_charge.lammpstrj"
    total_dump = case_dir / "forces.two_charge.dump"
    pair_dump = case_dir / "pairs.two_charge.dump"
    input_path = case_dir / "in.two_charge"
    log_path = case_dir / "log.two_charge.lammps"
    screen_path = case_dir / "screen.two_charge.txt"
    data_path.write_text(charge_data_text(q, frames[0], BOX))
    trajectory_path.write_text(trajectory_text(frames, BOX))
    input_path.write_text(
        ad_kspace_input(
            case,
            data_path,
            trajectory_path,
            total_dump,
            pair_dump_path=pair_dump,
        )
    )
    if rerun or not total_dump.is_file() or not pair_dump.is_file():
        run_lammps(input_path, log_path, screen_path)
    observed = extract_kspace_frames(total_dump, pair_dump, len(q))
    if len(observed) != len(frames):
        raise RuntimeError(f"two-charge frame mismatch for {case.case_id}")
    coeff = coefficients(case)
    op = operator(case, BOX)
    rows: list[dict] = []
    for frame_index, (xyz, (_, total, real_pair, kspace)) in enumerate(
        zip(frames, observed)
    ):
        numpy_raw = adref.fixed_ad_mesh_force(q, xyz, op, coeff.real)
        fraction = np.mod(xyz / h, 1.0)
        numpy_self_raw = (q * q)[:, None] * adref.ad_self_response_at_fractions(
            fraction, op, coeff.real
        )
        correction = correction_force(q, xyz, BOX, case.mesh, ab)
        numpy_corrected = numpy_raw - correction
        residual_self = numpy_self_raw - correction
        lammps_pair_only = kspace - residual_self
        numpy_pair_only = numpy_raw - numpy_self_raw
        rows.append(
            {
                "case_id": case.case_id,
                "frame": frame_index,
                "separation_A": 12.0,
                "real_pair_rms": math.sqrt(float(np.mean(np.sum(real_pair**2, axis=1)))),
                "lammps_kspace_rms": math.sqrt(float(np.mean(np.sum(kspace**2, axis=1)))),
                "numpy_corrected_rms": math.sqrt(
                    float(np.mean(np.sum(numpy_corrected**2, axis=1)))
                ),
                "total_operator_relative_rms_difference": relative_rms(
                    kspace, numpy_corrected
                ),
                "total_operator_max_component_difference": float(
                    np.max(np.abs(kspace - numpy_corrected))
                ),
                "pair_operator_relative_rms_difference": relative_rms(
                    lammps_pair_only, numpy_pair_only
                ),
                "pair_operator_max_component_difference": float(
                    np.max(np.abs(lammps_pair_only - numpy_pair_only))
                ),
                "pair_force_sum_norm": float(
                    np.linalg.norm(lammps_pair_only.sum(axis=0))
                ),
            }
        )
    return rows


def run_random_lammps_case(
    case: ADCase,
    q: np.ndarray,
    xyz_frames: list[np.ndarray],
    rerun: bool,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    case_dir = WORK / case.case_id / "random"
    case_dir.mkdir(parents=True, exist_ok=True)
    data_path = case_dir / "random_base.data"
    trajectory_path = case_dir / "random_10.lammpstrj"
    total_dump = case_dir / "forces.random.dump"
    pair_dump = case_dir / "pairs.random.dump"
    input_path = case_dir / "in.random"
    log_path = case_dir / "log.random.lammps"
    screen_path = case_dir / "screen.random.txt"
    data_path.write_text(charge_data_text(q, xyz_frames[0], BOX))
    trajectory_path.write_text(trajectory_text(xyz_frames, BOX))
    input_path.write_text(
        ad_kspace_input(
            case,
            data_path,
            trajectory_path,
            total_dump,
            pair_dump_path=pair_dump,
        )
    )
    if rerun or not total_dump.is_file() or not pair_dump.is_file():
        run_lammps(input_path, log_path, screen_path)
    return extract_kspace_frames(total_dump, pair_dump, len(q))


def analyze_random_case(
    case: ADCase,
    q: np.ndarray,
    xyz_frames: list[np.ndarray],
    observed,
    ab: np.ndarray,
) -> tuple[list[dict], dict]:
    coeff = coefficients(case)
    op = operator(case, BOX)
    if len(observed) != len(xyz_frames):
        raise RuntimeError(f"random frame mismatch for {case.case_id}")
    detail: list[dict] = []
    for config_index, (xyz, (_, total, real_pair, kspace)) in enumerate(
        zip(xyz_frames, observed), start=1
    ):
        numpy_raw = adref.fixed_ad_mesh_force(q, xyz, op, coeff.real)
        fractions = np.mod(xyz / (BOX / case.mesh), 1.0)
        raw_self = (q * q)[:, None] * adref.ad_self_response_at_fractions(
            fractions, op, coeff.real
        )
        correction = correction_force(q, xyz, BOX, case.mesh, ab)
        numpy_corrected = numpy_raw - correction
        residual_self = raw_self - correction
        lammps_pair_only = kspace - residual_self
        direct = ikref.direct_truncated_force(q, xyz, BOX, op.kernel)
        pair_error_vector = lammps_pair_only - direct
        total_error_vector = kspace - direct
        pair_error = math.sqrt(float(np.mean(np.sum(pair_error_vector**2, axis=1))))
        total_error = math.sqrt(float(np.mean(np.sum(total_error_vector**2, axis=1))))
        self_error = math.sqrt(float(np.mean(np.sum(residual_self**2, axis=1))))
        cross = float(np.mean(np.sum(pair_error_vector * residual_self, axis=1)))
        decomposition_residual = total_error_vector - pair_error_vector - residual_self
        detail.append(
            {
                "case_id": case.case_id,
                "config": config_index,
                "panel": case.panel,
                "x": case.x,
                "mesh": case.mesh,
                "order": case.order,
                "csplit": case.csplit,
                "cspread": case.cspread,
                "lammps_pair_only_rms_error": pair_error,
                "lammps_corrected_total_rms_error": total_error,
                "residual_self_rms": self_error,
                "pair_error_squared_mean": pair_error * pair_error,
                "self_error_squared_mean": self_error * self_error,
                "pair_self_dot_mean": cross,
                "pair_self_correlation": cross / max(pair_error * self_error, 1.0e-300),
                "decomposition_max_component_residual": float(
                    np.max(np.abs(decomposition_residual))
                ),
                "lammps_vs_numpy_corrected_relative_rms": relative_rms(
                    kspace, numpy_corrected
                ),
                "lammps_vs_numpy_corrected_max_component": float(
                    np.max(np.abs(kspace - numpy_corrected))
                ),
                "reconstructed_real_pair_rms": math.sqrt(
                    float(np.mean(np.sum(real_pair**2, axis=1)))
                ),
                "dumped_total_rms": math.sqrt(float(np.mean(np.sum(total**2, axis=1)))),
            }
        )

    pair_prediction20, _ = adref.fixed_ad_pair_estimate_cell_moments(
        q,
        BOX,
        case.mesh,
        case.order,
        RCUT,
        case.csplit,
        case.cspread,
        coeff,
        quadrature_order_per_half=20,
    )
    pair_prediction32, meta = adref.fixed_ad_pair_estimate_cell_moments(
        q,
        BOX,
        case.mesh,
        case.order,
        RCUT,
        case.csplit,
        case.cspread,
        coeff,
        quadrature_order_per_half=32,
    )
    self8 = residual_self_cell_rms(case, BOX, ab, 8)
    self12 = residual_self_cell_rms(case, BOX, ab, 12)
    q4_mean = float(np.mean(q**4))
    self_prediction = math.sqrt(q4_mean) * self12
    total_prediction = math.hypot(pair_prediction32, self_prediction)
    measured_pair = [row["lammps_pair_only_rms_error"] for row in detail]
    measured_total = [row["lammps_corrected_total_rms_error"] for row in detail]
    measured_self = [row["residual_self_rms"] for row in detail]
    pooled_cross = float(np.mean([row["pair_self_dot_mean"] for row in detail]))
    pooled_pair2 = float(np.mean([row["pair_error_squared_mean"] for row in detail]))
    pooled_self2 = float(np.mean([row["self_error_squared_mean"] for row in detail]))
    summary = {
        "case_id": case.case_id,
        "status": "valid",
        "panel": case.panel,
        "x": case.x,
        "mesh": case.mesh,
        "order": case.order,
        "csplit": case.csplit,
        "cspread": case.cspread,
        "n_config": len(detail),
        "predicted_pair_cell_moment": pair_prediction32,
        "predicted_pair_quadrature20": pair_prediction20,
        "pair_quadrature20_to32_relative": abs(pair_prediction32 - pair_prediction20)
        / pair_prediction32,
        "measured_pair_pooled_rms": pooled_rms(measured_pair),
        "measured_pair_jackknife_sem": pooled_rms_jackknife_sem(measured_pair),
        "measured_pair_over_predicted": pooled_rms(measured_pair) / pair_prediction32,
        "predicted_residual_self": self_prediction,
        "measured_residual_self_pooled_rms": pooled_rms(measured_self),
        "measured_self_over_predicted": pooled_rms(measured_self)
        / max(self_prediction, 1.0e-300),
        "pooled_pair_self_correlation": pooled_cross
        / max(math.sqrt(pooled_pair2 * pooled_self2), 1.0e-300),
        "maximum_decomposition_component_residual": max(
            row["decomposition_max_component_residual"] for row in detail
        ),
        "self_cell_rms_n8_per_half": self8,
        "self_cell_rms_n12_per_half": self12,
        "self_n8_to_n12_relative": abs(self12 - self8) / max(self12, 1.0e-300),
        "predicted_total_quadrature": total_prediction,
        "measured_total_pooled_rms": pooled_rms(measured_total),
        "measured_total_jackknife_sem": pooled_rms_jackknife_sem(measured_total),
        "measured_total_over_predicted": pooled_rms(measured_total) / total_prediction,
        "maximum_lammps_numpy_relative_rms": max(
            row["lammps_vs_numpy_corrected_relative_rms"] for row in detail
        ),
        "maximum_lammps_numpy_component_difference": max(
            row["lammps_vs_numpy_corrected_max_component"] for row in detail
        ),
        "minimum_abs_deconvolution_product": meta[
            "minimum_abs_deconvolution_product"
        ],
        "active_mode_count": meta["active_mode_count"],
        "zeroed_active_mode_count": meta["zeroed_active_mode_count"],
        "zeroed_active_mode_fraction": meta["zeroed_active_mode_fraction"],
        "zero_mode_missing_chi2": meta["zero_mode_missing_chi2"],
        "zero_mode_missing_pair_force": meta["zero_mode_missing_pair_force"],
        "zero_deconvolution_policy": meta["zero_deconvolution_policy"],
        "correction_sin1": ab[0],
        "correction_sin2": ab[1],
    }
    return detail, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-lammps",
        action="store_true",
        help="reuse existing force/pair dumps while recomputing all estimators",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    rerun = not args.reuse_lammps
    valid_cases, invalid_cases, key_to_id = unique_valid_cases()
    q, random_frames, input_paths = ordered_random_systems()
    WORK.mkdir(parents=True, exist_ok=True)

    correction_by_key: dict[tuple, np.ndarray] = {}
    correction_audit_by_key: dict[tuple, dict] = {}
    self_rows: list[dict] = []
    two_charge_rows: list[dict] = []
    detail_rows: list[dict] = []
    unique_summaries: dict[tuple, dict] = {}

    for index, case in enumerate(valid_cases, start=1):
        print(f"[{index}/{len(valid_cases)}] {case.case_id}", flush=True)
        ab, rows, audit = fit_self_correction(case, BOX, WORK, rerun=rerun)
        if audit["holdout_max_abs_component"] > 5.0e-10:
            raise RuntimeError(
                f"self-correction holdout failed for {case.case_id}: {audit}"
            )
        actual_mesh = int(audit["actual_mesh"])
        actual_x = (
            math.pi * RCUT * actual_mesh / (case.csplit * BOX)
            if case.panel == "sigma_up"
            else case.x
        )
        actual_case = replace(case, mesh=actual_mesh, x=actual_x)
        correction_by_key[case.tuple_key] = ab
        correction_audit_by_key[case.tuple_key] = audit
        self_rows.extend(rows)

        two_rows = run_two_charge_case(actual_case, ab, rerun=rerun)
        if max(row["total_operator_max_component_difference"] for row in two_rows) > 5.0e-10:
            raise RuntimeError(f"two-charge total operator mismatch for {case.case_id}")
        if max(row["pair_operator_max_component_difference"] for row in two_rows) > 5.0e-10:
            raise RuntimeError(f"two-charge pair operator mismatch for {case.case_id}")
        two_charge_rows.extend(two_rows)

        observed = run_random_lammps_case(actual_case, q, random_frames, rerun=rerun)
        detail, summary = analyze_random_case(
            actual_case, q, random_frames, observed, ab
        )
        summary["requested_mesh"] = case.mesh
        summary["actual_mesh"] = actual_mesh
        summary["requested_x"] = case.x
        if summary["maximum_lammps_numpy_component_difference"] > 5.0e-10:
            raise RuntimeError(f"random LAMMPS/NumPy mismatch for {case.case_id}")
        detail_rows.extend(detail)
        unique_summaries[case.tuple_key] = summary

    plotted_summary: list[dict] = []
    for case in figure3_cases():
        if case.tuple_key in unique_summaries:
            row = dict(unique_summaries[case.tuple_key])
            plotted_x = row["x"] if case.panel == "sigma_up" else case.x
            row.update(
                case_id=case.case_id,
                unique_case_id=key_to_id[case.tuple_key],
                panel=case.panel,
                x=plotted_x,
                requested_x=case.x,
            )
            audit = correction_audit_by_key[case.tuple_key]
            row.update(
                correction_fit_max_abs=audit["fit_max_abs_component"],
                correction_holdout_max_abs=audit["holdout_max_abs_component"],
            )
            plotted_summary.append(row)
        else:
            invalid = next(
                item
                for item in invalid_cases
                if item["case_id"] == case.case_id
            )
            plotted_summary.append(
                {
                    "case_id": case.case_id,
                    "unique_case_id": case.case_id,
                    "status": invalid["status"],
                    "panel": case.panel,
                    "x": case.x,
                    "mesh": case.mesh,
                    "order": case.order,
                    "csplit": case.csplit,
                    "cspread": case.cspread,
                    "requested_mesh": case.mesh,
                    "actual_mesh": "",
                    "requested_x": case.x,
                    "reason": invalid["reason"],
                }
            )

    write_csv(SELF_CSV, self_rows)
    write_csv(TWO_CHARGE_CSV, two_charge_rows)
    write_csv(DETAIL_CSV, detail_rows)
    write_csv(SUMMARY_CSV, plotted_summary)

    # Index only the raw artifacts consumed by the current set of active
    # unique cases.  Historical case directories may remain under WORK for
    # auditability, but they must not silently enter the formal provenance
    # manifest after the production case list changes.
    raw_files = sorted(
        path
        for case in valid_cases
        for path in (WORK / case.case_id).rglob("*")
        if path.is_file() and path.name != ".DS_Store" and not path.name.startswith("._")
    )
    write_json(
        RAW_MANIFEST,
        {
            "purpose": "complete hash index of generated LAMMPS inputs, trajectories, logs, screens, total-force dumps, and pair/local dumps for the active unique Figure-3 cases",
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
        TWO_CHARGE_CSV,
        DETAIL_CSV,
        SUMMARY_CSV,
        RAW_MANIFEST,
    ]
    write_json(
        MANIFEST,
        {
            "purpose": "direct production-LAMMPS AD single/two-charge and Figure-3 total-force validation",
            "lammps_executable": str(LMP.relative_to(PROJECT)),
            "lammps_executable_sha256": sha256(LMP),
            "lammps_version": lammps_version(),
            "lammps_source_provenance": {
                "path": str(SOURCE_SNAPSHOT_MANIFEST.relative_to(PROJECT)),
                "sha256": sha256(SOURCE_SNAPSHOT_MANIFEST),
            },
            "operator": "development ESP AD with Fourier-polynomial deconvolution, classical derivative gather, and two-harmonic self correction",
            "force_units": "kcal mol^-1 A^-1 (LAMMPS units real)",
            "kspace_isolation": "dumped total force minus compute pair/local real-space pair force reconstructed by atom ID",
            "self_identification": "four unit-charge fit positions; four held-out positions; no multiparticle reference force",
            "two_charge_test": "neutral charges separated by 12 A > rcut; pair-only force obtained after residual-self subtraction",
            "pair_force_sum_note": "pair_force_sum_norm is a translational/momentum-conservation diagnostic for the discrete AD mesh operator, not a LAMMPS-NumPy mismatch metric",
            "random_test": "ten neutral N=512 configurations; actual LAMMPS k-space forces for every valid Figure-3 tuple",
            "raw_artifact_index": {
                "path": str(RAW_MANIFEST.relative_to(PROJECT)),
                "sha256": sha256(RAW_MANIFEST),
                "n_files": len(raw_files),
            },
            "valid_unique_cases": [case_dict(case) for case in valid_cases],
            "invalid_cases": invalid_cases,
            "thresholds": {
                "self_correction_holdout_max_abs_component": 5.0e-10,
                "two_charge_max_abs_component": 5.0e-10,
                "random_lammps_numpy_max_abs_component": 5.0e-10,
                "absolute_component_units": "kcal mol^-1 A^-1",
            },
            "inputs": [
                {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
                for path in input_paths
            ],
            "sources": [
                {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
                for path in source_files
            ],
            "outputs": [
                {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
                for path in output_files
            ],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
