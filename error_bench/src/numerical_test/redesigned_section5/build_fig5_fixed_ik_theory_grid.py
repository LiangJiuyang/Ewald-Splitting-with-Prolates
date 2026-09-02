#!/usr/bin/env python3
"""Build the theory-only screening record for the revised main-text Figure 5.

The prediction stage deliberately reads only (i) the first 25 SPC/E coordinate
frames and (ii) a coarse PPPM force evaluation used solely to normalize a
relative tolerance.  It does *not* read Ewald reference forces, LAMMPS ESP
force differences, or frames 26--50.  For each fixed-influence ``i k`` ESP
candidate it evaluates the quadrature of the closed Fourier estimate (Eq. 55)
and the measured-``S_q`` all-alias mesh estimate (Eq. 90), using the LAMMPS
Fourier-polynomial deconvolution convention.

After the prediction table and its frozen selection record have been written,
the optional validation stage joins the already archived nonoverlapping
frames-26--50 Ewald-force measurements.  The joined table is for plotting
only; its measured values never enter the frozen theoretical selections.

This file intentionally covers only fixed-influence ``i k``.  The present
manuscript establishes no molecular-AD acceptance estimator, so it must not
be used to draw an AD ``theoretical prediction`` curve for Figure 5.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
WATER_ROOT = PROJECT / "numerical_examples" / "water_trajectory_benchmark"
TRAJECTORY = WATER_ROOT / "water_short_traj.lammpstrj"
# This is a deliberately inexpensive full-force scale, not an Ewald reference.
PILOT_FORCE_DUMP = WATER_ROOT / "forces.pppm_mesh20.dump"
ORDER_SCAN_SUMMARY = (
    HERE / "fig5_ik_ad_order_scan" / "fig5_ik_ad_order_scan_summary.csv"
)

PREDICTION_CSV = HERE / "fig5_fixed_ik_theory_grid_prediction.csv"
PARTIAL_PREDICTION_CSV = HERE / "fig5_fixed_ik_theory_grid_prediction.partial.csv"
FROZEN_JSON = HERE / "fig5_fixed_ik_theory_grid_frozen.json"
FROZEN_SHA256 = HERE / "fig5_fixed_ik_theory_grid_frozen.json.sha256"
PLOT_SOURCE_CSV = HERE / "fig5_fixed_ik_theory_grid_source.csv"
MANIFEST = HERE / "fig5_fixed_ik_theory_grid_manifest.json"

sys.path.insert(0, str(HERE))
import fixed_ik_reference as ref  # noqa: E402
import sq_alias_tools as sqtools  # noqa: E402


RCUT = 9.0
PILOT_N = 25
TOTAL_N = 50
ORDERS = tuple(range(5, 10))
MAX_ALIAS_SHELL = 12
OUTER_SAMPLES = 4096
SEED = 20260809


@dataclass(frozen=True)
class Target:
    value: float
    epsilon_split: float
    epsilon_spread: float
    csplit: float
    cspread: float
    meshes: tuple[int, ...]


TARGETS = (
    Target(
        value=1.0e-4,
        epsilon_split=1.0e-4,
        epsilon_spread=1.0e-4,
        csplit=12.024,
        cspread=12.024,
        meshes=(12, 15, 16, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80),
    ),
    Target(
        value=1.0e-5,
        epsilon_split=1.0e-5,
        epsilon_spread=1.0e-5,
        csplit=14.471,
        cspread=14.471,
        meshes=(12, 16, 18, 20, 24, 32, 36, 40, 48, 64, 80),
    ),
)


_PILOT_FRAMES: list[tuple[int, np.ndarray, np.ndarray, float]] | None = None
_FORCE_SCALE: float | None = None
_FORCE_SCALE_BY_FRAME: np.ndarray | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def prediction_key(row: dict[str, object]) -> tuple[float, int, int]:
    return (
        float(row["target_relative_rms"]),
        int(row["order"]),
        int(row["actual_nx"]),
    )


def sort_prediction_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["target_relative_rms"]),
            int(row["order"]),
            int(row["actual_nx"]),
        ),
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def target_from_value(value: float) -> Target:
    for target in TARGETS:
        if math.isclose(value, target.value, rel_tol=0.0, abs_tol=1.0e-15):
            return target
    raise KeyError(value)


def force_scale() -> tuple[float, np.ndarray]:
    frames = ref.parse_force_dump(PILOT_FORCE_DUMP)
    if len(frames) != TOTAL_N:
        raise RuntimeError(f"expected {TOTAL_N} scale-force frames, found {len(frames)}")
    per_frame = np.asarray(
        [
            math.sqrt(float(np.mean(np.sum(force * force, axis=1))))
            for _, force in frames[:PILOT_N]
        ],
        dtype=np.float64,
    )
    return float(math.sqrt(float(np.mean(per_frame * per_frame)))), per_frame


def initialize_worker() -> None:
    global _PILOT_FRAMES, _FORCE_SCALE, _FORCE_SCALE_BY_FRAME
    frames = ref.parse_charge_trajectory(TRAJECTORY)
    if len(frames) != TOTAL_N:
        raise RuntimeError(f"expected {TOTAL_N} coordinate frames, found {len(frames)}")
    _PILOT_FRAMES = frames[:PILOT_N]
    _FORCE_SCALE, _FORCE_SCALE_BY_FRAME = force_scale()


def prediction_specifications() -> list[tuple[float, int, int]]:
    return [
        (target.value, order, mesh)
        for target in TARGETS
        for order in ORDERS
        for mesh in target.meshes
    ]


def predict_case(spec: tuple[float, int, int]) -> dict[str, object]:
    """Evaluate one Eq. (55)+Eq. (90) candidate from pilot-only data."""

    if _PILOT_FRAMES is None or _FORCE_SCALE is None or _FORCE_SCALE_BY_FRAME is None:
        initialize_worker()
    target_value, order, mesh = spec
    target = target_from_value(target_value)
    pilot = _PILOT_FRAMES
    force_rms = float(_FORCE_SCALE)
    force_by_frame = np.asarray(_FORCE_SCALE_BY_FRAME, dtype=np.float64)
    _, q0, _, box = pilot[0]
    if any(not math.isclose(frame[3], box, rel_tol=0.0, abs_tol=1.0e-12) for frame in pilot):
        raise RuntimeError("Figure 5 theory prediction requires a fixed cubic cell")

    coeff = ref.load_coefficients(
        0.1 * target.epsilon_split,
        0.1 * target.epsilon_spread,
        target.csplit,
        target.cspread,
        order,
    )
    # Retain the established Figure-5 seed rule so the overlapping 10^-5
    # cases reproduce the previously archived theory-only source exactly.
    actual_seed = SEED + mesh * 100 + order
    population = sqtools.prepare_alias_population(
        mesh=mesh,
        order=order,
        box_length=box,
        rcut=RCUT,
        csplit=target.csplit,
        cspread=target.cspread,
        coeff=coeff,
        max_shell=MAX_ALIAS_SHELL,
        outer_samples=OUTER_SAMPLES,
        seed=actual_seed,
        base_deconvolution="lammps-fourier-polynomial",
    )
    modes, mappings = sqtools.population_mode_union([population])
    mapping = mappings[0]
    sq_sum = np.zeros(len(modes), dtype=np.float64)
    chi_by_frame: list[float] = []
    for _, q, xyz, frame_box in pilot:
        sq = sqtools.evaluate_sq_modes(q, xyz, frame_box, modes)
        sq_sum += sq
        chi, _ = sqtools.corrected_chi2(population, mapping, sq)
        chi_by_frame.append(chi)
    pilot_sq_mean = sq_sum / PILOT_N
    chi_mean, _, sampling_variances = sqtools.corrected_chi2_with_sampling(
        population, mapping, pilot_sq_mean
    )
    chi = np.asarray(chi_by_frame, dtype=np.float64)
    if not math.isclose(chi_mean, float(np.mean(chi)), rel_tol=2.0e-13, abs_tol=1.0e-30):
        raise RuntimeError("mean pilot S_q and mean frame-level chi-squared disagree")

    qsum = float(np.sum(q0 * q0))
    alias_scale = ref.COULOMB_REAL * qsum / math.sqrt(len(q0))
    mesh_abs = alias_scale * math.sqrt(chi_mean)
    fourier_abs = ref.closed_fourier_estimate(
        q0, box, RCUT, target.csplit, coeff, kmax=target.csplit / RCUT
    )
    total_abs = math.hypot(mesh_abs, fourier_abs)
    block_relative = np.asarray(
        [
            math.sqrt(
                alias_scale * alias_scale * float(np.mean(chi[5 * index : 5 * (index + 1)]))
                + fourier_abs * fourier_abs
            )
            / math.sqrt(float(np.mean(force_by_frame[5 * index : 5 * (index + 1)] ** 2)))
            for index in range(PILOT_N // 5)
        ],
        dtype=np.float64,
    )
    temporal_sem = statistics.stdev(block_relative) / math.sqrt(len(block_relative))
    sampling_chi2_sem = math.sqrt(sum(sampling_variances.values()))
    sampling_mesh_sem = alias_scale * sampling_chi2_sem / (2.0 * math.sqrt(chi_mean))
    sampling_total_sem = mesh_abs / total_abs * sampling_mesh_sem
    sigma_up = math.pi * RCUT * mesh / (target.csplit * box)

    return {
        "method": "ESP fixed-influence ik",
        "candidate_id": f"fig5_theory_ik_{target.value:.0e}_p{order}_m{mesh}",
        "target_relative_rms": target.value,
        "order": order,
        "actual_nx": mesh,
        "actual_grid_points": mesh**3,
        "sigma_up": sigma_up,
        "resolved_band": sigma_up >= 1.0,
        "epsilon_split": target.epsilon_split,
        "epsilon_spread": target.epsilon_spread,
        "csplit": target.csplit,
        "cspread": target.cspread,
        "pilot_frames": PILOT_N,
        "holdout_frames": TOTAL_N - PILOT_N,
        "pilot_force_scale": force_rms,
        "pilot_force_scale_source": "coarse PPPM force evaluation; no Ewald reference",
        "predicted_mesh_absolute_rms": mesh_abs,
        "predicted_fourier_absolute_rms": fourier_abs,
        "predicted_total_absolute_rms": total_abs,
        "predicted_total_relative_rms": total_abs / force_rms,
        "predicted_total_relative_block5_sem": temporal_sem,
        "predicted_total_relative_importance_sampling_sem": sampling_total_sem / force_rms,
        "prediction_passes_target": total_abs / force_rms <= target.value,
        "alias_shell": MAX_ALIAS_SHELL,
        "outer_samples_per_shell": OUTER_SAMPLES,
        "actual_seed": actual_seed,
        "pilot_mode_count": len(modes),
        "zeroed_active_mode_count": population.zeroed_active_mode_count,
        "prediction_operator": (
            "quadrature of Eq. (55) closed Fourier estimate and Eq. (90) "
            "measured-S_q all-alias fixed-influence ik mesh estimate; "
            "LAMMPS Fourier-polynomial deconvolution"
        ),
        "ewald_reference_force_accessed": False,
        "holdout_coordinates_accessed": False,
    }


def freeze_prediction(rows: list[dict[str, object]]) -> None:
    selections: list[dict[str, object]] = []
    for target in TARGETS:
        for order in ORDERS:
            candidates = sorted(
                (
                    row
                    for row in rows
                    if math.isclose(float(row["target_relative_rms"]), target.value, abs_tol=1.0e-15)
                    and int(row["order"]) == order
                    and as_bool(row["resolved_band"])
                    and as_bool(row["prediction_passes_target"])
                ),
                key=lambda row: int(row["actual_nx"]),
            )
            selections.append(
                {
                    "target_relative_rms": target.value,
                    "order": order,
                    "first_theory_feasible_mesh": (
                        int(candidates[0]["actual_nx"]) if candidates else None
                    ),
                    "selection_criterion": "resolved-band theoretical prediction <= target",
                }
            )
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Figure 5 fixed-influence ik theory-only prediction and candidate selection",
        "inputs": {
            "trajectory": record(TRAJECTORY),
            "pilot_force_scale": record(PILOT_FORCE_DUMP),
            "pilot_frame_indices_zero_based": list(range(PILOT_N)),
            "holdout_frame_indices_zero_based": list(range(PILOT_N, TOTAL_N)),
        },
        "operator": (
            "fixed-influence ik; LAMMPS Fourier-polynomial deconvolution; "
            "radial PSWF reciprocal band"
        ),
        "prediction": {
            "fourier": "Eq. (55) closed estimate",
            "mesh": "Eq. (90) measured-S_q all-alias estimate",
            "error_combination": "quadrature; Fourier/mesh covariance not modeled",
            "alias_shell": MAX_ALIAS_SHELL,
            "outer_samples_per_shell": OUTER_SAMPLES,
        },
        "candidate_space": {
            f"{target.value:.0e}": {
                "epsilon_split": target.epsilon_split,
                "epsilon_spread": target.epsilon_spread,
                "c_split": target.csplit,
                "c_spread": target.cspread,
                "meshes": list(target.meshes),
                "orders": list(ORDERS),
            }
            for target in TARGETS
        },
        "reference_force_differences_used": False,
        "holdout_coordinates_used": False,
        "selections": selections,
        "prediction_csv": record(PREDICTION_CSV),
    }
    FROZEN_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    FROZEN_SHA256.write_text(
        f"{sha256(FROZEN_JSON)}  {FROZEN_JSON.name}\n", encoding="utf-8"
    )


def run_prediction(*, restart: bool) -> None:
    specs = prediction_specifications()
    if restart and PARTIAL_PREDICTION_CSV.exists():
        PARTIAL_PREDICTION_CSV.unlink()
    rows: list[dict[str, object]] = (
        list(read_rows(PARTIAL_PREDICTION_CSV))
        if PARTIAL_PREDICTION_CSV.is_file()
        else []
    )
    completed = {prediction_key(row) for row in rows}
    initialize_worker()
    for spec in specs:
        if spec in completed:
            print(
                f"target={spec[0]:.0e} P={spec[1]} M={spec[2]} reused checkpoint",
                flush=True,
            )
            continue
        row = predict_case(spec)
        rows.append(row)
        completed.add(spec)
        # Persist every completed theoretical candidate.  This makes a long
        # all-alias sweep recoverable without ever mixing in validation data.
        write_rows(PARTIAL_PREDICTION_CSV, sort_prediction_rows(rows))
        print(
            f"target={spec[0]:.0e} P={spec[1]} M={spec[2]} "
            f"prediction={float(row['predicted_total_relative_rms']):.8e} "
            f"zeroed={int(row['zeroed_active_mode_count'])}",
            flush=True,
        )
    rows = sort_prediction_rows(rows)
    expected = sum(len(target.meshes) for target in TARGETS) * len(ORDERS)
    if len(rows) != expected:
        raise RuntimeError(f"prediction matrix incomplete: {len(rows)} != {expected}")
    write_rows(PREDICTION_CSV, rows)
    freeze_prediction(rows)
    print(f"wrote {PREDICTION_CSV} ({len(rows)} theory-only rows)")


def load_frozen_prediction() -> list[dict[str, str]]:
    if not FROZEN_JSON.is_file() or not FROZEN_SHA256.is_file() or not PREDICTION_CSV.is_file():
        raise FileNotFoundError("run --stage prediction before --stage validation")
    expected = FROZEN_SHA256.read_text(encoding="utf-8").split()[0]
    if expected != sha256(FROZEN_JSON):
        raise RuntimeError("Figure 5 theory freeze SHA256 mismatch")
    frozen = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))
    if frozen.get("reference_force_differences_used") is not False:
        raise RuntimeError("the frozen prediction record accessed reference-force differences")
    if frozen.get("holdout_coordinates_used") is not False:
        raise RuntimeError("the frozen prediction record accessed holdout coordinates")
    if frozen["prediction_csv"]["sha256"] != sha256(PREDICTION_CSV):
        raise RuntimeError("the theory prediction CSV changed after freezing")
    return read_rows(PREDICTION_CSV)


def attach_validation() -> None:
    predictions = load_frozen_prediction()
    measurements = {
        (float(row["target_relative_rms"]), int(row["order"]), int(row["actual_nx"])): row
        for row in read_rows(ORDER_SCAN_SUMMARY)
        if row["method"] == "ik"
    }
    output: list[dict[str, object]] = []
    for prediction in predictions:
        key = (
            float(prediction["target_relative_rms"]),
            int(prediction["order"]),
            int(prediction["actual_nx"]),
        )
        measured = measurements.get(key)
        if measured is None:
            raise RuntimeError(f"missing fixed-ik holdout measurement for {key}")
        if not math.isclose(float(measured["csplit"]), float(prediction["csplit"]), abs_tol=5.0e-4):
            raise RuntimeError(f"mismatched c_split for {key}")
        if not math.isclose(float(measured["cspread"]), float(prediction["cspread"]), abs_tol=5.0e-4):
            raise RuntimeError(f"mismatched c_spread for {key}")
        output.append(
            {
                **prediction,
                "validation_relative_rms": float(measured["holdout_relative_rms"]),
                "validation_relative_rms_balanced_block5_sem": float(
                    measured["holdout_balanced_block5_sem"]
                ),
                "validation_frame_first": 26,
                "validation_frame_last": 50,
                "validation_frame_count": int(measured["holdout_frames"]),
                "validation_operator": measured["operator"],
                "validation_reference": "Ewald forces; excluded from theory prediction and selection",
                "validation_passes_target": str(measured["holdout_passes_target"]).lower() == "true",
            }
        )
    write_rows(PLOT_SOURCE_CSV, output)
    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "Revised Figure 5 theory/validation source",
                "frozen_theory_prediction": record(FROZEN_JSON),
                "theory_prediction_csv": record(PREDICTION_CSV),
                "validation_measurements": record(ORDER_SCAN_SUMMARY),
                "plot_source": record(PLOT_SOURCE_CSV),
                "prediction_selection_uses_ewald": False,
                "validation_is_nonoverlapping": True,
                "ad_panels_included": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {PLOT_SOURCE_CSV} ({len(output)} theory/validation rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prediction", "validation", "all"), default="all")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard only the recoverable theory-only checkpoint before rebuilding it",
    )
    args = parser.parse_args()
    if args.stage in ("prediction", "all"):
        run_prediction(restart=args.restart)
    if args.stage in ("validation", "all"):
        attach_validation()


if __name__ == "__main__":
    main()
