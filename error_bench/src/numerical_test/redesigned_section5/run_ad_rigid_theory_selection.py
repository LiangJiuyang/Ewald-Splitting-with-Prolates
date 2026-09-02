#!/usr/bin/env python3
r"""Generate supporting finite-band AD component tables for Figure 5.

The current lower AD row uses finite-band theoretical analysis with a 25-frame
pilot correction. This legacy-named script retains the trajectory-free
rigid-SPC/E calculation as a diagnostic and writes the frozen self/Fourier
component records required by the current workflow. It does not generate the
AD curves displayed in Figure 5.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
AD_VALIDATION = HERE / "lammps_ad_total_validation"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(AD_VALIDATION) not in sys.path:
    sys.path.insert(0, str(AD_VALIDATION))

import fixed_ad_reference as adref  # noqa: E402
import fixed_ik_reference as ikref  # noqa: E402
import build_fig5_ad_rigid_sq_theory as rigid_theory  # noqa: E402
import ad_validation_common as adcommon  # noqa: E402
from ad_validation_common import (  # noqa: E402
    coefficients,
    correction_force,
    fit_self_correction,
    operator,
)
from run_water_ad_validation import (  # noqa: E402
    PILOT_COUNT,
    TOTAL_COUNT,
    TRAJECTORY,
    partition_summary,
    refresh_ewald_reference,
    run_water_case,
)


TARGET = 1.0e-5
CSPLIT = 14.471
MESHES = (16, 18, 20, 24)
ORDERS = tuple(range(5, 10))
# The first branch is target-aligned c_spread; the additional branches make
# spreading stricter while retaining exactly the same PSWF split.  Each is a
# declared finite candidate set, not a post hoc fit.
SPREAD_BRANCHES = ((14.471, 1.0e-5), (16.894, 1.0e-6))
ONE_SIDED_95_Z = 1.6448536269514722

OUTDIR = HERE / "fig5_ad_rigid_theory_selection"
PREDICTION_CSV = OUTDIR / "prediction_before_validation.csv"
FROZEN_JSON = OUTDIR / "frozen_selection.json"
DETAIL_CSV = OUTDIR / "holdout_validation_by_frame.csv"
SUMMARY_CSV = OUTDIR / "holdout_validation_summary.csv"
MANIFEST = OUTDIR / "manifest.json"


def configure_target(target: float) -> None:
    """Select one prespecified target-specific rigid-screen candidate set."""

    global TARGET, CSPLIT, MESHES, ORDERS, SPREAD_BRANCHES
    global OUTDIR, PREDICTION_CSV, FROZEN_JSON, DETAIL_CSV, SUMMARY_CSV, MANIFEST

    if math.isclose(target, 1.0e-5, rel_tol=0.0, abs_tol=1.0e-16):
        TARGET = 1.0e-5
        CSPLIT = 14.471
        MESHES = (16, 18, 20, 24)
        ORDERS = tuple(range(5, 10))
        SPREAD_BRANCHES = ((14.471, 1.0e-5), (16.894, 1.0e-6))
        OUTDIR = HERE / "fig5_ad_rigid_theory_selection"
    elif math.isclose(target, 1.0e-4, rel_tol=0.0, abs_tol=1.0e-15):
        TARGET = 1.0e-4
        CSPLIT = 12.024
        MESHES = (15, 16, 18, 20)
        ORDERS = tuple(range(5, 10))
        SPREAD_BRANCHES = ((12.024, 1.0e-4), (13.251, 3.0e-5), (14.471, 1.0e-5))
        OUTDIR = HERE / "fig5_ad_rigid_theory_selection_1e-4"
    else:
        raise ValueError("only the prespecified 1e-4 and 1e-5 AD screens are available")
    PREDICTION_CSV = OUTDIR / "prediction_before_validation.csv"
    FROZEN_JSON = OUTDIR / "frozen_selection.json"
    DETAIL_CSV = OUTDIR / "holdout_validation_by_frame.csv"
    SUMMARY_CSV = OUTDIR / "holdout_validation_summary.csv"
    MANIFEST = OUTDIR / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(PROJECT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def candidate_specs() -> list[tuple[rigid_theory.Target, int, int]]:
    return [
        (
            rigid_theory.Target(
                value=TARGET,
                epsilon_split=TARGET,
                epsilon_spread=epsilon_spread,
                csplit=CSPLIT,
                cspread=cspread,
                meshes=(mesh,),
            ),
            order,
            mesh,
        )
        for mesh in MESHES
        for cspread, epsilon_spread in SPREAD_BRANCHES
        for order in ORDERS
    ]


def select_candidate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Choose the first a-priori-feasible resolved-band candidate.

    We intentionally use a transparent resolution-first criterion rather
    than infer a hardware cost model.  Candidates are ordered by actual grid
    size, then local spreading order, then c_spread.  Therefore this script
    tests a reproducible feasibility rule, not a claim of universal minimum
    wall time.
    """

    feasible = [row for row in rows if bool(row["selection_passes_target"])]
    if not feasible:
        raise RuntimeError("no prespecified candidate passed the rigid a-priori screen")
    return min(
        feasible,
        key=lambda row: (
            int(row["actual_grid_points"]),
            int(row["order"]),
            float(row["cspread"]),
        ),
    )


def theory_screen() -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Run the complete no-reference-force stage and select one candidate."""

    charges, box, molecule_count = rigid_theory.parse_spce_topology(rigid_theory.WATER_DATA)
    force_scale = rigid_theory.coarse_force_scale()
    rows: list[dict[str, object]] = []
    for target, order, mesh in candidate_specs():
        prediction, _, _ = rigid_theory.predict_case(
            target,
            order,
            mesh,
            charges,
            box,
            molecule_count,
            force_scale,
            alias_shell=4,
            samples_per_shell=8192,
        )
        sem = float(prediction["predicted_total_relative_sampling_sem"])
        upper = float(prediction["predicted_total_relative_rms"]) + ONE_SIDED_95_Z * sem
        prediction.update(
            prediction_one_sided_95_upper_relative=upper,
            selection_passes_target=(
                bool(prediction["resolved_band"]) and upper <= TARGET
            ),
            selection_rule=(
                "resolved band and a-priori relative prediction plus one-sided 95% "
                "alias-sampling uncertainty <= target"
            ),
        )
        rows.append(prediction)
        print(
            json.dumps(
                {
                    "stage": "a_priori_rigid_AD_screen",
                    "M": mesh,
                    "P": order,
                    "csplit": target.csplit,
                    "cspread": target.cspread,
                    "upper95": upper,
                    "passes": prediction["selection_passes_target"],
                }
            ),
            flush=True,
        )
    rows.sort(key=lambda row: (int(row["actual_grid_points"]), int(row["order"]), float(row["cspread"])))
    selected = select_candidate(rows)
    selected_target = next(
        target
        for target, order, mesh in candidate_specs()
        if (
            int(order) == int(selected["order"])
            and int(mesh) == int(selected["actual_nx"])
            and math.isclose(target.cspread, float(selected["cspread"]), abs_tol=1.0e-12)
        )
    )
    return rows, selected, {
        "box_length": box,
        "molecule_count": molecule_count,
        "force_scale": force_scale,
        "target": selected_target,
    }


def validate_selected(selected: dict[str, object], *, rerun_lammps: bool) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Read no reference data until after the frozen choice has been saved."""

    target = rigid_theory.Target(
        value=TARGET,
        epsilon_split=TARGET,
        epsilon_spread=float(selected["epsilon_spread"]),
        csplit=float(selected["csplit"]),
        cspread=float(selected["cspread"]),
        meshes=(int(selected["actual_nx"]),),
    )
    case = rigid_theory.case_for(target, int(selected["order"]), int(selected["actual_nx"]))
    # Unit-charge self probes are implementation-specific but independent of
    # water coordinates and force references.  The probe has already passed
    # during the theory stage; this repeat obtains its fitted coefficients for
    # the matching production-force decomposition.
    correction, _, audit = fit_self_correction(
        case, float(selected["box_length"]), rigid_theory.SELF_WORK, rerun=False
    )
    if float(audit["holdout_max_abs_component"]) > rigid_theory.SELF_AUDIT_MAX:
        raise RuntimeError("selected candidate failed the independent unit-charge self audit")

    # From this point on, force and coordinate data are validation only.
    observed, run_paths = run_water_case(case, rerun=rerun_lammps)
    all_frames = ikref.parse_charge_trajectory(TRAJECTORY)
    reference, reference_rows, reference_paths = refresh_ewald_reference(rerun=False)
    if len(all_frames) != TOTAL_COUNT or len(observed) != TOTAL_COUNT or len(reference) != TOTAL_COUNT:
        raise RuntimeError("selected-candidate validation frame count mismatch")

    coeff = coefficients(case)
    op = operator(case, float(selected["box_length"]))
    details: list[dict[str, object]] = []
    for index in range(PILOT_COUNT, TOTAL_COUNT):
        timestep, q, xyz, box = all_frames[index]
        time_lmp, full_force, _, kspace_force = observed[index]
        time_ref, reference_force = reference[index]
        if timestep != time_lmp or timestep != time_ref:
            raise RuntimeError("selected-candidate validation timestep mismatch")
        raw = adref.fixed_ad_mesh_force(q, xyz, op, coeff.real)
        correction_force_value = correction_force(q, xyz, box, case.mesh, correction)
        numpy_corrected = raw - correction_force_value
        fractions = np.mod(xyz / (box / case.mesh), 1.0)
        residual_self = (q * q)[:, None] * adref.ad_self_response_at_fractions(
            fractions, op, coeff.real
        ) - correction_force_value
        direct = ikref.direct_truncated_force(q, xyz, box, op.kernel)
        mesh_total = kspace_force - direct
        mesh_pair = mesh_total - residual_self
        difference = full_force - reference_force
        pair_rms = math.sqrt(float(np.mean(np.sum(mesh_pair * mesh_pair, axis=1))))
        self_rms = math.sqrt(float(np.mean(np.sum(residual_self * residual_self, axis=1))))
        cross = float(np.mean(np.sum(mesh_pair * residual_self, axis=1)))
        details.append(
            {
                "case_id": case.case_id,
                "target_relative_error": TARGET,
                "frame_zero_based": index,
                "timestep": timestep,
                "partition": "independent_holdout",
                "mesh": case.mesh,
                "order": case.order,
                "csplit": case.csplit,
                "cspread": case.cspread,
                "sum_total_difference_squared": float(np.sum(difference * difference)),
                "sum_reference_squared": float(np.sum(reference_force * reference_force)),
                "total_rms_absolute_error": math.sqrt(float(np.mean(np.sum(difference * difference, axis=1)))),
                "total_relative_error": math.sqrt(float(np.sum(difference * difference) / np.sum(reference_force * reference_force))),
                "mesh_total_rms_absolute": math.sqrt(float(np.mean(np.sum(mesh_total * mesh_total, axis=1)))),
                "mesh_pair_rms_absolute": pair_rms,
                "residual_self_rms_absolute": self_rms,
                "pair_self_dot_mean": cross,
                "pair_self_correlation": cross / max(pair_rms * self_rms, 1.0e-300),
                "lammps_numpy_kspace_relative_rms": ikref.rms_vector_error(kspace_force, numpy_corrected)
                / max(math.sqrt(float(np.mean(np.sum(numpy_corrected * numpy_corrected, axis=1)))), 1.0e-300),
                "lammps_numpy_kspace_max_component": float(np.max(np.abs(kspace_force - numpy_corrected))),
            }
        )
    summary = partition_summary(case, details, "all")
    summary.update(
        partition="independent_holdout",
        prediction_relative_rms=float(selected["predicted_total_relative_rms"]),
        prediction_upper95_relative=float(selected["prediction_one_sided_95_upper_relative"]),
        prediction_to_measurement_ratio=float(selected["predicted_total_relative_rms"])
        / float(summary["pooled_total_relative_error"]),
        selection_used_holdout=False,
        validation_frames="26--51",
    )
    paths = {
        "selected_ad_runs": run_paths,
        "ewald_reference": reference_paths,
        "ewald_crosscheck_rows": reference_rows,
    }
    return details, summary, paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=float,
        default=1.0e-5,
        help="prespecified relative target to screen (1e-4 or 1e-5)",
    )
    parser.add_argument(
        "--reuse-lammps",
        action="store_true",
        help="reuse a matching selected-candidate LAMMPS force dump if it exists",
    )
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help=(
            "write the frozen a-priori component table without opening a "
            "molecular trajectory or Ewald-force holdout"
        ),
    )
    parser.add_argument(
        "--lmp",
        type=Path,
        default=None,
        help="ESP-LAMMPS executable (defaults to ESP_LAMMPS_BIN or the in-tree build)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adcommon.configure_lmp(args.lmp)
    configure_target(args.target)
    started = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: no molecular coordinates or reference force may be opened.
    prediction_rows, selected, screen_metadata = theory_screen()
    for row in prediction_rows:
        row["box_length"] = screen_metadata["box_length"]
    write_csv(PREDICTION_CSV, prediction_rows)
    frozen = {
        "schema_version": 1,
        "purpose": "supporting finite-band AD component selection and legacy rigid-SPC/E diagnostic",
        "logical_order": [
            "rigid topology, unit-charge operator probes, and closed Fourier estimate",
            "write complete theoretical candidate table",
            "freeze resolution-first selected candidate",
            "only then access molecular trajectory and Ewald holdout",
        ],
        "candidate_set": {
            "target_relative_rms": TARGET,
            "csplit": CSPLIT,
            "meshes": list(MESHES),
            "orders": list(ORDERS),
            "spread_branches": [
                {"cspread": cspread, "epsilon_spread": epsilon}
                for cspread, epsilon in SPREAD_BRANCHES
            ],
        },
        "selection_rule": selected["selection_rule"],
        "selection_order": "minimum M^3, then minimum P, then minimum c_spread",
        "selected": selected,
        "prediction_reference_force_accessed": False,
        "prediction_molecular_coordinates_accessed": False,
        "force_scale": screen_metadata["force_scale"],
        "force_scale_role": "coarse-PPPM relative normalization only; no Ewald reference",
        "prediction_table_sha256": sha256(PREDICTION_CSV),
        "created_unix_time": time.time(),
    }
    FROZEN_JSON.write_text(json.dumps(jsonable(frozen), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "selection_frozen", "selected": selected}, default=jsonable), flush=True)

    if args.prediction_only:
        return

    # Stage 2: independent holdout validation, after the immutable freeze.
    selected_with_box = {**selected, "box_length": screen_metadata["box_length"]}
    details, summary, paths = validate_selected(
        selected_with_box, rerun_lammps=not args.reuse_lammps
    )
    write_csv(DETAIL_CSV, details)
    write_csv(SUMMARY_CSV, [summary])
    manifest = {
        "schema_version": 1,
        "purpose": "legacy rigid-SPC/E AD diagnostic with independent validation",
        "frozen_selection": file_record(FROZEN_JSON),
        "prediction": file_record(PREDICTION_CSV),
        "validation": {
            "detail": file_record(DETAIL_CSV),
            "summary": file_record(SUMMARY_CSV),
            "used_for_selection": False,
            "frames": "26--51",
            "operator": "production LAMMPS ESP analytical differentiation with matching unit-charge residual-self correction",
            "reference": "tight Ewald total force",
        },
        "selected": selected,
        "validation_summary": summary,
        "raw_paths": paths,
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    MANIFEST.write_text(json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "validation_complete", "summary": summary}, default=jsonable), flush=True)


if __name__ == "__main__":
    main()
