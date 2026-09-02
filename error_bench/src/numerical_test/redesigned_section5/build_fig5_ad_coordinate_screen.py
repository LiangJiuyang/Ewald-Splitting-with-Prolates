#!/usr/bin/env python3
r"""Generate the AD data used in Figure 5.

The lower AD row combines finite-band theoretical analysis with a 25-frame
pilot correction. Dashed curves use SPC/E configurations from frames 1--25;
filled markers report independent validation on frames 26--50. The calculation
uses the production AD operator, a finite PSWF Fourier reference, and the
closed Fourier term. No Ewald-force difference is used to form a prediction.

``--baseline`` writes the curves, ``--joint-target`` freezes a declared
candidate set, and the corresponding validation commands append the later
measurements. The result is a short-trajectory, implementation-specific
screen for SPC/E water.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
AD_VALIDATION = HERE / "lammps_ad_total_validation"
sys.path[:0] = [str(HERE), str(AD_VALIDATION)]

import fixed_ad_reference as adref  # noqa: E402
import fixed_ik_reference as ikref  # noqa: E402
import build_fig5_ad_rigid_sq_theory as baseline  # noqa: E402
import ad_validation_common as adcommon  # noqa: E402
from ad_validation_common import coefficients, correction_force, operator  # noqa: E402


TRAJECTORY = baseline.WATER_ROOT / "water_short_traj.lammpstrj"
SCAN_SUMMARY = HERE / "fig5_ik_ad_order_scan" / "fig5_ik_ad_order_scan_summary.csv"
BASE_COMPONENTS = HERE / "fig5_ad_rigid_sq_theory_prediction.csv"
JOINT_COMPONENTS = {
    1.0e-4: HERE / "fig5_ad_rigid_theory_selection_1e-4" / "prediction_before_validation.csv",
    1.0e-5: HERE / "fig5_ad_rigid_theory_selection" / "prediction_before_validation.csv",
}

OUTDIR = HERE / "fig5_ad_coordinate_screen"
BASELINE_PREDICTION = OUTDIR / "baseline_prediction.csv"
BASELINE_BY_FRAME = OUTDIR / "baseline_prediction_by_frame.csv"
BASELINE_SOURCE = OUTDIR / "baseline_source.csv"
BASELINE_MANIFEST = OUTDIR / "baseline_manifest.json"
DIRECT_CACHE = OUTDIR / "direct_target_cache"

PILOT_N = 25
ONE_SIDED_T95_DF4 = 2.13184678632665
DIRECT_BLOCK_SIZE = 96


@dataclass(frozen=True)
class Candidate:
    target: baseline.Target
    order: int
    mesh: int
    scope: str

    @property
    def case(self):
        return baseline.case_for(self.target, self.order, self.mesh)

    @property
    def key(self) -> tuple[float, int, int, float]:
        return (
            float(self.target.value),
            int(self.order),
            int(self.mesh),
            round(float(self.target.cspread), 6),
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def json_safe_paths(value: object) -> object:
    """Convert nested generated-artifact paths to JSON-safe strings."""

    if isinstance(value, Path):
        try:
            return str(value.relative_to(PROJECT))
        except ValueError:
            return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_paths(item) for item in value]
    return value


def value(row: dict[str, str], field: str) -> float:
    return float(row[field])


def component_key(row: dict[str, str]) -> tuple[float, int, int, float]:
    return (
        float(row["target_relative_rms"]),
        int(row["order"]),
        int(row["actual_nx"]),
        round(float(row["cspread"]), 6),
    )


def load_components() -> dict[tuple[float, int, int, float], dict[str, str]]:
    """Load only the audited self/Fourier components, never rigid pair data."""

    tables = [BASE_COMPONENTS, *JOINT_COMPONENTS.values()]
    result: dict[tuple[float, int, int, float], dict[str, str]] = {}
    for path in tables:
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_csv(path):
            key = component_key(row)
            existing = result.get(key)
            if existing is not None:
                for field in (
                    "residual_self_absolute_rms",
                    "self_correction_sin1",
                    "self_correction_sin2",
                    "fourier_absolute_rms",
                ):
                    if not math.isclose(value(existing, field), value(row, field), rel_tol=0.0, abs_tol=2.0e-11):
                        raise RuntimeError(f"inconsistent operator component {field} for {key}")
            result[key] = row
    return result


def trajectory_digest(frames: Iterable[tuple[int, np.ndarray, np.ndarray, float]]) -> str:
    digest = hashlib.sha256()
    for timestep, q, xyz, box in frames:
        digest.update(np.asarray([timestep], dtype="<i8").tobytes())
        digest.update(np.asarray([box], dtype="<f8").tobytes())
        digest.update(np.asarray(q, dtype="<f8").tobytes())
        digest.update(np.asarray(xyz, dtype="<f8").tobytes())
    return digest.hexdigest()


def load_pilot_frames() -> list[tuple[int, np.ndarray, np.ndarray, float]]:
    frames = ikref.parse_charge_trajectory(TRAJECTORY)
    if len(frames) < PILOT_N:
        raise RuntimeError("SPC/E trajectory has fewer than 25 screening frames")
    pilot = frames[:PILOT_N]
    q0 = pilot[0][1]
    box0 = pilot[0][3]
    if not all(np.array_equal(q, q0) and math.isclose(box, box0) for _, q, _, box in pilot):
        raise RuntimeError("the AD pilot correction requires a constant-charge, cubic SPC/E trajectory")
    return pilot


def baseline_candidates() -> list[Candidate]:
    return [Candidate(target, order, mesh, "fixed_band") for target, order, mesh in baseline.candidates()]


def joint_candidates(target_value: float) -> list[Candidate]:
    if math.isclose(target_value, 1.0e-4, rel_tol=0.0, abs_tol=1.0e-15):
        csplit = 12.024
        meshes = (15, 16, 18, 20)
        branches = ((12.024, 1.0e-4), (13.251, 3.0e-5), (14.471, 1.0e-5))
    elif math.isclose(target_value, 1.0e-5, rel_tol=0.0, abs_tol=1.0e-16):
        csplit = 14.471
        meshes = (16, 18, 20, 24)
        branches = ((14.471, 1.0e-5), (16.894, 1.0e-6))
    else:
        raise ValueError("joint AD pilot corrections are defined only at 1e-4 and 1e-5")
    result: list[Candidate] = []
    for mesh in meshes:
        for cspread, epsilon_spread in branches:
            for order in range(5, 10):
                target = baseline.Target(
                    value=target_value,
                    epsilon_split=target_value,
                    epsilon_spread=epsilon_spread,
                    csplit=csplit,
                    cspread=cspread,
                    meshes=(mesh,),
                )
                result.append(Candidate(target, order, mesh, "joint_window"))
    return result


def canonical_modes(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Return sorted signed modes, coefficients, and a physical-mode hash."""

    mesh = kernel.shape[0]
    if kernel.shape != (mesh, mesh, mesh):
        raise ValueError("direct kernel must be cubic")
    indices = np.nonzero(kernel)
    axis = np.rint(np.fft.fftfreq(mesh) * mesh).astype(np.int64)
    modes = np.column_stack((axis[indices[0]], axis[indices[1]], axis[indices[2]])).astype(np.int64)
    coefficients = kernel[indices].astype(np.float64, copy=False)
    ordering = np.lexsort((modes[:, 2], modes[:, 1], modes[:, 0]))
    modes = modes[ordering]
    coefficients = coefficients[ordering]
    digest = hashlib.sha256()
    digest.update(np.asarray(modes, dtype="<i8").tobytes())
    digest.update(np.asarray(coefficients, dtype="<f8").tobytes())
    return modes, coefficients, digest.hexdigest()


def active_mode_force(
    q: np.ndarray,
    xyz: np.ndarray,
    box_length: float,
    modes: np.ndarray,
    coefficients: np.ndarray,
    *,
    block_size: int,
) -> np.ndarray:
    """Direct finite-band force evaluated in blocks of physical Fourier modes."""

    wavevectors = (2.0 * math.pi / box_length) * np.asarray(modes, dtype=np.float64)
    force = np.zeros((len(q), 3), dtype=np.float64)
    prefactor = ikref.COULOMB_REAL / box_length**3
    for start in range(0, len(coefficients), block_size):
        stop = min(start + block_size, len(coefficients))
        kblock = wavevectors[start:stop]
        phase_minus = np.exp(-1j * xyz @ kblock.T)
        rho = q @ phase_minus
        mode_force = (-1j * coefficients[start:stop] * rho)[:, None] * kblock
        force += q[:, None] * (phase_minus.conj() @ mode_force).real
    return prefactor * force


def direct_cache_path(target: baseline.Target, signature: str) -> Path:
    """Return a cache path for one *actual* finite Fourier target.

    Resolved meshes at fixed ``c_split`` normally share this signature.  A
    below-band mesh does not: its Nyquist cube removes physical PSWF modes and
    must therefore receive a different direct target rather than silently
    borrowing the resolved-band one.
    """

    tag = f"cs{target.csplit:.3f}".replace(".", "p")
    return DIRECT_CACHE / f"direct_target_{tag}_{signature[:16]}.npz"


def direct_for_candidate(
    candidate: Candidate,
    frames: list[tuple[int, np.ndarray, np.ndarray, float]],
    *,
    force_rebuild: bool,
    memory: dict[str, tuple[np.ndarray, str, int, float, Path]],
) -> tuple[np.ndarray, str, int, float]:
    """Load or create the direct force for this candidate's physical mode set."""

    case = candidate.case
    coeff = coefficients(case)
    op = operator(case, frames[0][3])
    modes, kernel_values, signature = canonical_modes(op.kernel)
    if signature in memory:
        forces, _, mode_count, regression, _ = memory[signature]
        return forces, signature, mode_count, regression

    cache_path = direct_cache_path(candidate.target, signature)
    digest = trajectory_digest(frames)
    if cache_path.is_file() and not force_rebuild:
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_signature = str(archive["signature"].item())
            cached_digest = str(archive["trajectory_digest"].item())
            cached_modes = archive["modes"]
            cached_values = archive["kernel_values"]
            forces = archive["forces"]
        if (
            cached_signature == signature
            and cached_digest == digest
            and np.array_equal(cached_modes, modes)
            and np.array_equal(cached_values, kernel_values)
            and forces.shape == (len(frames), len(frames[0][1]), 3)
        ):
            payload = (forces, signature, len(modes), 0.0, cache_path)
            memory[signature] = payload
            return forces, signature, len(modes), 0.0

    DIRECT_CACHE.mkdir(parents=True, exist_ok=True)
    forces = np.empty((len(frames), len(frames[0][1]), 3), dtype=np.float64)
    for frame_index, (_, q, xyz, box) in enumerate(frames):
        forces[frame_index] = active_mode_force(
            q, xyz, box, modes, kernel_values, block_size=DIRECT_BLOCK_SIZE
        )
    # Regression against the pre-existing finite-cube evaluator is deliberately
    # performed only on the smallest resolved grid, where its tensor form is
    # inexpensive.  This is an internal numerical check, not a manuscript claim.
    direct_reference = ikref.direct_truncated_force(
        frames[0][1], frames[0][2], frames[0][3], op.kernel
    )
    regression = float(np.max(np.abs(forces[0] - direct_reference)))
    if regression > 3.0e-11:
        raise RuntimeError(f"finite-band direct-force regression failed: {regression:.3e}")
    np.savez_compressed(
        cache_path,
        signature=np.asarray(signature),
        trajectory_digest=np.asarray(digest),
        modes=modes,
        kernel_values=kernel_values,
        forces=forces,
    )
    payload = (forces, signature, len(modes), regression, cache_path)
    memory[signature] = payload
    return forces, signature, len(modes), regression


def mean_square(vector: np.ndarray) -> float:
    return float(np.mean(np.sum(vector * vector, axis=1)))


def mean_dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.sum(left * right, axis=1)))


def block_sem(values: list[float]) -> float:
    if len(values) != PILOT_N:
        raise ValueError("screening uncertainty requires exactly 25 frames")
    blocks = [
        math.sqrt(float(np.mean(np.asarray(values[start : start + 5], dtype=np.float64) ** 2)))
        for start in range(0, PILOT_N, 5)
    ]
    return float(np.std(blocks, ddof=1) / math.sqrt(len(blocks)))


def assert_matching_direct_target(
    candidate: Candidate,
    op: adref.ADOperator,
    expected_signature: str,
) -> None:
    modes, _, signature = canonical_modes(op.kernel)
    if signature != expected_signature:
        raise RuntimeError(
            "pilot-corrected AD operator and finite-band target do not "
            f"match: {candidate.case.case_id} ({len(modes)} modes)"
        )


def evaluate_candidate(
    candidate: Candidate,
    component: dict[str, str],
    frames: list[tuple[int, np.ndarray, np.ndarray, float]],
    direct_forces: np.ndarray,
    direct_signature: str,
    direct_mode_count: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate finite-band AD theory plus the pilot correction for one candidate."""

    case = candidate.case
    box = frames[0][3]
    coeff = coefficients(case)
    op = operator(case, box)
    sigma_up = math.pi * baseline.RCUT * candidate.mesh / (candidate.target.csplit * box)
    assert_matching_direct_target(candidate, op, direct_signature)
    correction = np.asarray(
        [value(component, "self_correction_sin1"), value(component, "self_correction_sin2")],
        dtype=np.float64,
    )
    fourier2 = value(component, "fourier_absolute_rms") ** 2
    force_scale = baseline.coarse_force_scale()

    frame_rows: list[dict[str, object]] = []
    frame_relative: list[float] = []
    for frame_index, (timestep, q, xyz, frame_box) in enumerate(frames):
        raw_mesh = adref.fixed_ad_mesh_force(q, xyz, op, coeff.real)
        correction_vector = correction_force(q, xyz, frame_box, candidate.mesh, correction)
        corrected_mesh = raw_mesh - correction_vector
        # This exact finite-band difference includes all particle-pair and
        # residual-self terms, including their molecular covariance, under the
        # matched AD implementation.  No Ewald/reference-force difference is
        # taken at this stage.
        mesh2 = mean_square(corrected_mesh - direct_forces[frame_index])
        total_relative = math.sqrt(mesh2 + fourier2) / force_scale
        frame_relative.append(total_relative)
        frame_rows.append(
            {
                "scope": candidate.scope,
                "candidate_id": case.case_id,
                "target_relative_rms": candidate.target.value,
                "order": candidate.order,
                "actual_nx": candidate.mesh,
                "actual_grid_points": candidate.mesh**3,
                "csplit": candidate.target.csplit,
                "cspread": candidate.target.cspread,
                "screening_frame": frame_index + 1,
                "screening_frame_zero_based": frame_index,
                "timestep": timestep,
                "mesh_mean_square_direct_coordinate": mesh2,
                "fourier_mean_square_closed": fourier2,
                "predicted_total_relative_rms": total_relative,
            }
        )

    mesh2 = float(
        np.mean([float(row["mesh_mean_square_direct_coordinate"]) for row in frame_rows])
    )
    total_relative = math.sqrt(mesh2 + fourier2) / force_scale
    relative_sem = block_sem(frame_relative)
    upper95 = total_relative + ONE_SIDED_T95_DF4 * relative_sem

    row = {
        "method": "ESP production AD",
        "scope": candidate.scope,
        "candidate_id": case.case_id,
        "target_relative_rms": candidate.target.value,
        "order": candidate.order,
        "actual_nx": candidate.mesh,
        "actual_grid_points": candidate.mesh**3,
        "sigma_up": sigma_up,
        "resolved_band": sigma_up >= 1.0,
        "epsilon_split": candidate.target.epsilon_split,
        "epsilon_spread": candidate.target.epsilon_spread,
        "csplit": candidate.target.csplit,
        "cspread": candidate.target.cspread,
        "screening_frames": PILOT_N,
        "screening_force_scale": force_scale,
        "screening_force_scale_source": "coarse PPPM force evaluation on frames 1--25; no Ewald reference",
        "ad_estimator": "finite-band theoretical analysis + 25-frame pilot correction",
        "coordinate_mesh_absolute_rms": math.sqrt(mesh2),
        "coordinate_mesh_direct_absolute_rms": math.sqrt(mesh2),
        "coordinate_mesh_formula_to_direct_ratio": 1.0,
        "closed_fourier_absolute_rms": math.sqrt(fourier2),
        "predicted_total_absolute_rms": math.sqrt(mesh2 + fourier2),
        "predicted_total_relative_rms": total_relative,
        "predicted_total_relative_block5_sem": relative_sem,
        "predicted_total_relative_one_sided_95_upper": upper95,
        "prediction_passes_target": total_relative <= candidate.target.value,
        "selection_passes_target": sigma_up >= 1.0 and upper95 <= candidate.target.value,
        "prediction_reference_force_accessed": False,
        "prediction_molecular_coordinates_accessed": True,
        "prediction_structure_input": "SPC/E pilot configurations, frames 1--25",
        "direct_reference": "finite-band PSWF Fourier reference",
        "closed_terms": "Eq. (56) closed Fourier contribution",
        "selection_scope": "finite-band theoretical analysis with a 25-frame pilot correction; not a universal molecular AD estimator",
        "direct_mode_signature": direct_signature,
        "direct_active_mode_count": direct_mode_count,
        "zeroed_active_mode_count": op.zeroed_active_mode_count,
    }
    return row, frame_rows


def evaluate_screen(
    candidates: list[Candidate], *, force_rebuild_direct: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if not candidates:
        raise ValueError("candidate list is empty")
    components = load_components()
    frames = load_pilot_frames()
    grouped: dict[float, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(float(candidate.target.value), []).append(candidate)
    predictions: list[dict[str, object]] = []
    by_frame: list[dict[str, object]] = []
    direct_records: dict[str, object] = {}
    direct_memory: dict[str, tuple[np.ndarray, str, int, float, Path]] = {}
    started = time.time()
    for target_value, group in sorted(grouped.items(), reverse=True):
        for local_index, candidate in enumerate(group, start=1):
            direct_forces, signature, mode_count, direct_regression = direct_for_candidate(
                candidate,
                frames,
                force_rebuild=force_rebuild_direct,
                memory=direct_memory,
            )
            record_key = f"{target_value:.0e}:{signature[:16]}"
            if record_key not in direct_records:
                direct_records[record_key] = {
                    "signature": signature,
                    "active_mode_count": mode_count,
                    "regression_max_component": direct_regression,
                    "cache": str(
                        direct_cache_path(candidate.target, signature).relative_to(PROJECT)
                    ),
                }
            component = components.get(candidate.key)
            if component is None:
                raise KeyError(f"missing audited self/Fourier components for {candidate.key}")
            row, detail = evaluate_candidate(
                candidate, component, frames, direct_forces, signature, mode_count
            )
            predictions.append(row)
            by_frame.extend(detail)
            print(
                json.dumps(
                    {
                        "stage": "coordinate_AD_screen",
                        "scope": candidate.scope,
                        "target": target_value,
                        "candidate": local_index,
                        "target_candidates": len(group),
                        "P": candidate.order,
                        "M": candidate.mesh,
                        "cspread": candidate.target.cspread,
                        "prediction": row["predicted_total_relative_rms"],
                        "upper95": row["predicted_total_relative_one_sided_95_upper"],
                        "elapsed_s": round(time.time() - started, 2),
                    }
                ),
                flush=True,
            )
    predictions.sort(
        key=lambda row: (
            -float(row["target_relative_rms"]),
            int(row["actual_grid_points"]),
            int(row["order"]),
            float(row["cspread"]),
        )
    )
    by_frame.sort(
        key=lambda row: (
            -float(row["target_relative_rms"]),
            int(row["actual_grid_points"]),
            int(row["order"]),
            float(row["cspread"]),
            int(row["screening_frame_zero_based"]),
        )
    )
    return predictions, by_frame, {
        "elapsed_seconds": time.time() - started,
        "direct_targets": direct_records,
        "screening_trajectory_sha256": trajectory_digest(frames),
    }


def write_baseline(force_rebuild_direct: bool) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    predictions, by_frame, runtime = evaluate_screen(
        baseline_candidates(), force_rebuild_direct=force_rebuild_direct
    )
    expected = len(baseline.candidates())
    if len(predictions) != expected:
        raise RuntimeError("baseline pilot-corrected AD matrix is incomplete")
    write_csv(BASELINE_PREDICTION, predictions)
    write_csv(BASELINE_BY_FRAME, by_frame)
    payload = {
        "schema_version": 1,
        "purpose": "Figure 5 AD theoretical analysis with a 25-frame pilot correction",
        "logical_order": [
            "evaluate the finite-band AD term on frames 1--25",
            "apply the pilot correction and write the prediction table",
            "append independent validation only after prediction is frozen",
        ],
        "prediction": {
            "reference_force_accessed": False,
            "holdout_coordinates_accessed": False,
            "molecular_coordinates_accessed": True,
            "frames": "1--25",
            "estimator": "finite-band theoretical analysis + 25-frame pilot correction",
            "selection_limit": "short-trajectory SPC/E-water screen, not a universal molecular AD estimator",
        },
        "candidate_count": len(predictions),
        "inputs": [
            file_record(path)
            for path in (TRAJECTORY, BASE_COMPONENTS, *JOINT_COMPONENTS.values(), Path(__file__))
        ],
        "outputs": [file_record(path) for path in (BASELINE_PREDICTION, BASELINE_BY_FRAME)],
        "runtime": runtime,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    BASELINE_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(BASELINE_PREDICTION)


def join_baseline_validation() -> None:
    """Join holdout errors after the baseline prediction matrix is frozen."""

    if not BASELINE_PREDICTION.is_file() or not BASELINE_MANIFEST.is_file():
        raise FileNotFoundError("write the frozen baseline AD pilot correction before validation join")
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if bool(manifest["prediction"]["reference_force_accessed"]):
        raise RuntimeError("baseline prediction manifest indicates reference-force leakage")
    predictions = read_csv(BASELINE_PREDICTION)
    validation = {
        (float(row["target_relative_rms"]), int(row["order"]), int(row["actual_nx"])): row
        for row in read_csv(SCAN_SUMMARY)
        if row["method"] == "ad"
    }
    joined: list[dict[str, object]] = []
    for row in predictions:
        key = (float(row["target_relative_rms"]), int(row["order"]), int(row["actual_nx"]))
        held = validation.get(key)
        if held is None:
            raise KeyError(f"missing AD Ewald holdout result for {key}")
        joined.append(
            {
                **row,
                "validation_relative_rms": value(held, "holdout_relative_rms"),
                "validation_relative_rms_balanced_block5_sem": value(
                    held, "holdout_balanced_block5_sem"
                ),
                "validation_passes_target": value(held, "holdout_relative_rms")
                <= value(row, "target_relative_rms"),
                "prediction_to_validation_ratio": value(row, "predicted_total_relative_rms")
                / value(held, "holdout_relative_rms"),
                "validation_frame_first": 26,
                "validation_frame_last": 50,
                "validation_frame_count": 25,
                "validation_operator": held["operator"],
                "validation_reference": "pre-existing tight-Ewald total-force error",
                "validation_used_for_prediction": False,
                "validation_used_for_selection": False,
            }
        )
    write_csv(BASELINE_SOURCE, joined)
    manifest["validation"] = {
        "joined_after_prediction_freeze": True,
        "source": "archived frames-26--50 production-AD/Ewald total-force errors",
        "used_for_prediction_or_selection": False,
        "output": file_record(BASELINE_SOURCE),
    }
    BASELINE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(BASELINE_SOURCE)


def joint_paths(target: float) -> dict[str, Path]:
    tag = f"{target:.0e}".replace("e-0", "e-")
    directory = OUTDIR / f"joint_{tag}"
    return {
        "directory": directory,
        "prediction": directory / "prediction_before_validation.csv",
        "by_frame": directory / "prediction_by_frame.csv",
        "frozen": directory / "frozen_selection.json",
        "detail": directory / "holdout_validation_by_frame.csv",
        "summary": directory / "holdout_validation_summary.csv",
        "manifest": directory / "manifest.json",
    }


def select_joint(rows: list[dict[str, object]]) -> dict[str, object]:
    passing = [row for row in rows if bool(row["selection_passes_target"])]
    if not passing:
        raise RuntimeError("no declared joint pilot-corrected AD candidate satisfies the one-sided target")
    return min(
        passing,
        key=lambda row: (
            int(row["actual_grid_points"]),
            int(row["order"]),
            float(row["cspread"]),
        ),
    )


def write_joint(target: float, force_rebuild_direct: bool) -> None:
    paths = joint_paths(target)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    predictions, by_frame, runtime = evaluate_screen(
        joint_candidates(target), force_rebuild_direct=force_rebuild_direct
    )
    selected = select_joint(predictions)
    write_csv(paths["prediction"], predictions)
    write_csv(paths["by_frame"], by_frame)
    frozen = {
        "schema_version": 1,
        "purpose": "joint AD pilot-correction selection frozen before validation",
        "logical_order": [
            "use only frames 1--25 coordinates and a coarse PPPM normalization",
            "evaluate the complete declared (M, P, c_spread) candidate set",
            "freeze the resolution-first selection",
            "only then permit Ewald-reference validation",
        ],
        "candidate_set": {
            "target_relative_rms": target,
            "c_split": float(predictions[0]["csplit"]),
            "meshes": sorted({int(row["actual_nx"]) for row in predictions}),
            "orders": sorted({int(row["order"]) for row in predictions}),
            "c_spread": sorted({float(row["cspread"]) for row in predictions}),
        },
        "selection_rule": "resolved band and pilot-corrected AD prediction plus one-sided 95% five-block uncertainty <= target; minimum M^3, then P, then c_spread",
        "selected": selected,
        "prediction_reference_force_accessed": False,
        "prediction_molecular_coordinates_accessed": True,
        "prediction_table_sha256": sha256(paths["prediction"]),
        "runtime": runtime,
    }
    paths["frozen"].write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "purpose": "joint AD theoretical analysis with a pilot correction and deferred validation",
        "frozen_selection": file_record(paths["frozen"]),
        "prediction": file_record(paths["prediction"]),
        "screening": {
            "frames": "1--25",
            "reference_force_accessed": False,
            "coordinate_accessed": True,
            "estimator": "finite-band theoretical analysis + 25-frame pilot correction",
        },
        "validation": {"performed": False, "used_for_selection": False},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "joint_coordinate_selection_frozen", "target": target, "selected": selected}))


def validate_joint(target: float, *, rerun_lammps: bool) -> None:
    """Run/read Ewald validation only after reading the frozen selection."""

    paths = joint_paths(target)
    if not paths["frozen"].is_file() or not paths["manifest"].is_file():
        raise FileNotFoundError("freeze a joint coordinate selection before validation")
    frozen = json.loads(paths["frozen"].read_text(encoding="utf-8"))
    if bool(frozen["prediction_reference_force_accessed"]):
        raise RuntimeError("joint selection is contaminated by a reference-force prediction")
    selected = frozen["selected"]
    case_target = baseline.Target(
        value=target,
        epsilon_split=float(selected["epsilon_split"]),
        epsilon_spread=float(selected["epsilon_spread"]),
        csplit=float(selected["csplit"]),
        cspread=float(selected["cspread"]),
        meshes=(int(selected["actual_nx"]),),
    )
    case = baseline.case_for(case_target, int(selected["order"]), int(selected["actual_nx"]))
    from run_water_ad_validation import (  # noqa: E402
        PILOT_COUNT,
        TOTAL_COUNT,
        TRAJECTORY as VALIDATION_TRAJECTORY,
        refresh_ewald_reference,
        run_water_case,
    )

    observed, run_paths = run_water_case(case, rerun=rerun_lammps)
    reference, _, reference_paths = refresh_ewald_reference(rerun=False)
    frames = ikref.parse_charge_trajectory(VALIDATION_TRAJECTORY)
    if len(observed) != TOTAL_COUNT or len(reference) != TOTAL_COUNT or len(frames) != TOTAL_COUNT:
        raise RuntimeError("joint pilot-corrected AD validation has incomplete force records")
    details: list[dict[str, object]] = []
    for frame_index in range(PILOT_COUNT, TOTAL_COUNT):
        timestep, _, _, _ = frames[frame_index]
        observed_time, full, _, _ = observed[frame_index]
        reference_time, reference_force = reference[frame_index]
        if timestep != observed_time or timestep != reference_time:
            raise RuntimeError("joint pilot-corrected AD validation timestep mismatch")
        difference = full - reference_force
        details.append(
            {
                "candidate_id": case.case_id,
                "target_relative_rms": target,
                "frame_zero_based": frame_index,
                "timestep": timestep,
                "sum_total_difference_squared": float(np.sum(difference * difference)),
                "sum_reference_squared": float(np.sum(reference_force * reference_force)),
                "total_relative_error": math.sqrt(
                    float(np.sum(difference * difference) / np.sum(reference_force * reference_force))
                ),
            }
        )
    diff2 = float(sum(float(row["sum_total_difference_squared"]) for row in details))
    ref2 = float(sum(float(row["sum_reference_squared"]) for row in details))
    block_values: list[float] = []
    for start in range(0, len(details), 5):
        block = details[start : start + 5]
        block_values.append(
            math.sqrt(
                sum(float(row["sum_total_difference_squared"]) for row in block)
                / sum(float(row["sum_reference_squared"]) for row in block)
            )
        )
    holdout = math.sqrt(diff2 / ref2)
    summary = {
        "candidate_id": case.case_id,
        "target_relative_rms": target,
        "actual_nx": case.mesh,
        "actual_grid_points": case.mesh**3,
        "order": case.order,
        "csplit": case.csplit,
        "cspread": case.cspread,
        "validation_frames": "26--50",
        "validation_frame_count": len(details),
        "validation_relative_rms": holdout,
        "validation_relative_rms_block5_sem": float(np.std(block_values, ddof=1) / math.sqrt(len(block_values))),
        "prediction_relative_rms": float(selected["predicted_total_relative_rms"]),
        "prediction_upper95_relative": float(selected["predicted_total_relative_one_sided_95_upper"]),
        "prediction_to_validation_ratio": float(selected["predicted_total_relative_rms"]) / holdout,
        "validation_passes_target": holdout <= target,
        "selection_used_holdout": False,
        "validation_operator": "production LAMMPS ESP AD with residual-self correction",
        "validation_reference": "tight Ewald total force",
    }
    write_csv(paths["detail"], details)
    write_csv(paths["summary"], [summary])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["validation"] = {
        "performed": True,
        "used_for_selection": False,
        "frames": "26--50",
        "detail": file_record(paths["detail"]),
        "summary": file_record(paths["summary"]),
        "raw_paths": json_safe_paths(
            {"selected_ad": run_paths, "ewald_reference": reference_paths}
        ),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "joint_coordinate_validation_complete", "target": target, "summary": summary}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--baseline",
        action="store_true",
        help="write the fixed-band AD theoretical-analysis and pilot-correction table",
    )
    action.add_argument(
        "--join-baseline-validation",
        action="store_true",
        help="append frames-26--50 Ewald values after the baseline prediction is frozen",
    )
    action.add_argument(
        "--joint-target",
        type=float,
        metavar="TARGET",
        help="write and freeze one declared joint (M,P,c_spread) pilot-corrected AD screen",
    )
    action.add_argument(
        "--validate-joint",
        type=float,
        metavar="TARGET",
        help="validate an already frozen joint pilot-corrected AD screen against Ewald",
    )
    parser.add_argument(
        "--rebuild-direct-cache",
        action="store_true",
        help="recompute cached finite-band direct forces from frames 1--25",
    )
    parser.add_argument(
        "--rerun-lammps",
        action="store_true",
        help="rerun selected-candidate LAMMPS validation rather than reuse a matching archive",
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
    if args.baseline:
        write_baseline(args.rebuild_direct_cache)
    elif args.join_baseline_validation:
        join_baseline_validation()
    elif args.joint_target is not None:
        write_joint(args.joint_target, args.rebuild_direct_cache)
    elif args.validate_joint is not None:
        validate_joint(args.validate_joint, rerun_lammps=args.rerun_lammps)
    else:
        raise AssertionError("one action must be selected")


if __name__ == "__main__":
    main()
