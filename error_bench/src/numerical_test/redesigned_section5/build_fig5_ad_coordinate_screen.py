#!/usr/bin/env python3
r"""Generate and validate the vector-complete AD theory data for Figure 5.

Prediction is deliberately isolated from validation. It reads only frames
1--25 and evaluates both the target-conditioned diagonal source spectrum
``S_tag(q)`` and a phase-resolved full-source/target-cell descriptor.  The
main AD prediction contracts the full-source pair alias and production
residual-self vectors first and forms the particle RMS only afterwards.
``S_tag`` and the exact homogeneous cell-moment estimator remain explicit
diagonal-spectrum diagnostics.  The closed Fourier tail is then combined
with the joint in-band pair/self RMS in quadrature.

The main prediction accumulates one analytical discrete-minus-continuum AD
error operator.  It never reads LAMMPS forces, an Ewald force dump, holdout
coordinates, or a precomputed finite-band force-difference table.

The separate ``--diagnostic-direct-check`` action may evaluate that direct
finite-band force difference *after* a selection is frozen. It is an
implementation diagnostic only and never participates in selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
AD_VALIDATION = HERE / "lammps_ad_total_validation"
sys.path[:0] = [str(HERE), str(AD_VALIDATION)]

import ad_sq_descriptor as adsq  # noqa: E402
import ad_joint_quadratic as adjoint  # noqa: E402
import fixed_ad_reference as adref  # noqa: E402
import fixed_ik_reference as ikref  # noqa: E402
# Shared targets, topology readers, and cell-quadrature helpers retain their
# historical module name. Its rigid-molecule predictor remains diagnostic;
# the active Figure-5 path uses the pilot-coordinate joint vector, while
# measured pilot-frame S_tag remains a diagonal diagnostic.
import fig5_ad_theory_common as baseline  # noqa: E402
import ad_validation_common as adcommon  # noqa: E402
from generated_output import section_output_root  # noqa: E402
from ad_validation_common import (  # noqa: E402
    coefficients,
    correction_force,
    fit_self_correction,
    operator,
)


TRAJECTORY = baseline.WATER_ROOT / "water_short_traj.lammpstrj"
OUTPUT_ROOT = section_output_root()
SCAN_SUMMARY = OUTPUT_ROOT / "fig5_ik_ad_order_scan" / "fig5_ik_ad_order_scan_summary.csv"
OUTDIR = OUTPUT_ROOT / "fig5_ad_coordinate_screen"
RUNTIME = OUTDIR / "runtime"
SELF_WORK = RUNTIME / "self_probes"

BASELINE_PREDICTION = OUTDIR / "baseline_prediction.csv"
BASELINE_BLOCKS = OUTDIR / "baseline_prediction_by_frame.csv"
BASELINE_ALIASES = OUTDIR / "baseline_alias_shell.csv"
BASELINE_SPECTRA = OUTDIR / "baseline_structure_spectra.npz"
BASELINE_SOURCE = OUTDIR / "baseline_source.csv"
BASELINE_MANIFEST = OUTDIR / "baseline_manifest.json"
UNIT_TESTS = OUTDIR / "theory_unit_tests.json"

PILOT_N = 25
PILOT_BLOCKS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25))
HOLDOUT_BLOCKS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 26))
ONE_SIDED_T95_DF4 = 2.13184678632665
SELF_AUDIT_MAX = baseline.SELF_AUDIT_MAX


@dataclass(frozen=True)
class Candidate:
    target: baseline.Target
    order: int
    mesh: int
    scope: str

    @property
    def case(self):
        return baseline.case_for(self.target, self.order, self.mesh)


@dataclass
class SpectralADTheory:
    candidate: Candidate
    population: adsq.ADSourceSpectrumPopulation
    correction: np.ndarray
    residual_self: float
    fourier: float
    self_metadata: dict[str, object]
    operator: adref.ADOperator
    coeff: ikref.PSWFCoefficients


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": relative_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def lammps_record() -> dict[str, object]:
    """Record the configured executable without embedding a host path."""

    record = file_record(adcommon.LMP)
    record["path"] = "$LMP"
    return record


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def target_tag(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-")


def load_pilot_frames() -> tuple[list[tuple[int, np.ndarray, np.ndarray, float]], str]:
    frames, prefix_digest = ikref.parse_charge_trajectory_prefix(
        TRAJECTORY, PILOT_N, return_sha256=True
    )
    q0 = frames[0][1]
    box0 = frames[0][3]
    if not all(
        np.array_equal(q, q0) and math.isclose(box, box0)
        for _, q, _, box in frames
    ):
        raise RuntimeError("Figure-5 AD prediction requires fixed charges and a cubic cell")
    return frames, prefix_digest


def baseline_candidates() -> list[Candidate]:
    return [
        Candidate(target, order, mesh, "fixed_band")
        for target, order, mesh in baseline.candidates()
    ]


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
        raise ValueError("joint AD screens are defined only for 1e-4 and 1e-5")

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


def candidate_seed(candidate: Candidate, alias_shell: int) -> int:
    payload = (
        f"{candidate.target.value:.17g}|{candidate.mesh}|{candidate.order}|"
        f"{candidate.target.csplit:.17g}|{candidate.target.cspread:.17g}|"
        f"{alias_shell}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def prepare_spectral_theory_candidate(
    candidate: Candidate,
    charges: np.ndarray,
    box: float,
    *,
    alias_shell: int,
    samples_per_shell: int,
    rerun_self_probes: bool,
) -> SpectralADTheory:
    case = candidate.case
    coeff = coefficients(case)
    population = adsq.prepare_ad_source_spectrum_population(
        q=charges,
        mesh=candidate.mesh,
        order=candidate.order,
        box_length=box,
        rcut=baseline.RCUT,
        csplit=candidate.target.csplit,
        cspread=candidate.target.cspread,
        coeff=coeff,
        max_shell=alias_shell,
        samples_per_shell=samples_per_shell,
        seed=candidate_seed(candidate, alias_shell),
    )
    correction, _, audit = fit_self_correction(
        case, box, SELF_WORK, rerun=rerun_self_probes
    )
    if int(audit["actual_mesh"]) != candidate.mesh:
        raise RuntimeError(
            f"{case.case_id}: LAMMPS rounded M={candidate.mesh} "
            f"to M={audit['actual_mesh']}"
        )
    if float(audit["holdout_max_abs_component"]) > SELF_AUDIT_MAX:
        raise RuntimeError(f"{case.case_id}: unit-charge self correction failed")
    (
        self_cell,
        self8,
        self12,
        quadrature_order,
        n8_to_n12,
        final_refinement,
    ) = baseline.converged_residual_self_cell_rms(case, box, correction)
    residual_self = math.sqrt(float(np.mean(charges**4))) * self_cell
    fourier = ikref.closed_fourier_estimate(
        charges,
        box,
        baseline.RCUT,
        candidate.target.csplit,
        coeff,
        kmax=candidate.target.csplit / baseline.RCUT,
    )
    metadata = {
        "self_correction_sin1": float(correction[0]),
        "self_correction_sin2": float(correction[1]),
        "self_probe_fit_max_abs_component": float(audit["fit_max_abs_component"]),
        "self_probe_holdout_max_abs_component": float(
            audit["holdout_max_abs_component"]
        ),
        "residual_self_cell_rms": self_cell,
        "residual_self_cell_rms_n8": self8,
        "residual_self_cell_rms_n12": self12,
        "residual_self_quadrature_order_per_half": quadrature_order,
        "residual_self_n8_to_n12_relative": n8_to_n12,
        "residual_self_final_refinement_relative": final_refinement,
    }
    return SpectralADTheory(
        candidate=candidate,
        population=population,
        correction=correction,
        residual_self=residual_self,
        fourier=fourier,
        self_metadata=metadata,
        operator=operator(case, box),
        coeff=coeff,
    )


def evaluate_spectral_theory_spectra(
    frames: list[tuple[int, np.ndarray, np.ndarray, float]],
    modes: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tagged_sum = np.zeros(len(modes), dtype=np.float64)
    charge_sum = np.zeros(len(modes), dtype=np.float64)
    tagged_blocks = np.zeros((len(PILOT_BLOCKS), len(modes)), dtype=np.float64)
    charge_blocks = np.zeros_like(tagged_blocks)
    for frame_index, (_, q, xyz, box) in enumerate(frames):
        tagged, ordinary = adsq.evaluate_tagged_pair_spectrum(
            q,
            xyz,
            box,
            modes,
            chunk_size=chunk_size,
            return_charge_spectrum=True,
        )
        tagged_sum += tagged
        charge_sum += ordinary
        tagged_blocks[frame_index // 5] += tagged
        charge_blocks[frame_index // 5] += ordinary
        print(
            json.dumps(
                {
                    "stage": "pilot_structure_factor",
                    "frame": frame_index + 1,
                    "frames": PILOT_N,
                    "mode_count": len(modes),
                }
            ),
            flush=True,
        )
    return (
        tagged_sum / PILOT_N,
        charge_sum / PILOT_N,
        tagged_blocks / 5.0,
        charge_blocks / 5.0,
    )


def pair_rms(charges: np.ndarray, chi2: float) -> float:
    q2 = charges * charges
    pair_factor = float((np.sum(q2) ** 2 - np.sum(q2 * q2)) / len(charges))
    return ikref.COULOMB_REAL * math.sqrt(max(pair_factor * chi2, 0.0))


def pooled_rms(mean_squares: list[float]) -> float:
    if not mean_squares:
        raise ValueError("cannot pool an empty set of frame mean squares")
    return math.sqrt(math.fsum(mean_squares) / len(mean_squares))


def alias_relative_sem(
    charges: np.ndarray,
    chi2: float,
    pair: float,
    total: float,
    force_scale: float,
    shell_variances: dict[int, float],
) -> float:
    if chi2 <= 0.0 or total <= 0.0:
        return 0.0
    chi_sem = math.sqrt(math.fsum(shell_variances.values()))
    q2 = charges * charges
    pair_factor = float((np.sum(q2) ** 2 - np.sum(q2 * q2)) / len(charges))
    pair_sem = (
        ikref.COULOMB_REAL
        * math.sqrt(pair_factor)
        * chi_sem
        / (2.0 * math.sqrt(chi2))
    )
    return (pair / total) * pair_sem / force_scale


def evaluate_spectral_theory(
    theory: SpectralADTheory,
    mapping: dict[str, object],
    charges: np.ndarray,
    frames: list[tuple[int, np.ndarray, np.ndarray, float]],
    tagged_mean: np.ndarray,
    charge_mean: np.ndarray,
    tagged_blocks: np.ndarray,
    force_scale: float,
    pilot_prefix_sha256: str,
    box: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate the Figure-5 vector-complete pair/self error formula.

    ``S_tag`` reweights the exact homogeneous cell-moment estimator as a
    diagonal-spectrum diagnostic.  The main prediction instead evaluates the
    phase-resolved full-source AD error amplitude on every pilot frame, adds
    the production self-correction vector, and only then forms the RMS.  This
    retains pair/self and cross-mode covariance by construction.
    """
    candidate = theory.candidate
    population = theory.population
    corrected_chi2, shell_values, shell_variances = (
        adsq.corrected_chi2_with_sampling(population, mapping, tagged_mean)
    )
    ordinary_chi2, _, _ = adsq.corrected_chi2_with_sampling(
        population, mapping, charge_mean
    )
    homogeneous_pair = pair_rms(charges, population.homogeneous_chi2)
    corrected_pair = pair_rms(charges, corrected_chi2)
    ordinary_pair = pair_rms(charges, ordinary_chi2)

    raw_alias2: list[float] = []
    correction2: list[float] = []
    alias_correction_cross: list[float] = []
    distinct_pair2: list[float] = []
    residual_self2: list[float] = []
    pair_residual_self_cross: list[float] = []
    joint_pair_self2: list[float] = []
    vector_identity_residuals: list[float] = []
    component_identity_residuals: list[float] = []
    for _, frame_charges, xyz, frame_box in frames:
        if not np.array_equal(frame_charges, charges):
            raise RuntimeError("pilot-frame charges changed within the AD screen")
        if not math.isclose(frame_box, theory.operator.box_length):
            raise RuntimeError("pilot-frame cell changed within the AD screen")
        moments = adjoint.evaluate_joint_pair_self_quadratic(
            frame_charges,
            xyz,
            theory.operator,
            theory.coeff.real,
            theory.correction,
        )
        raw_alias2.append(moments.full_source_alias_mean_square)
        correction2.append(moments.self_correction_mean_square)
        alias_correction_cross.append(moments.alias_minus_correction_dot_mean)
        distinct_pair2.append(moments.distinct_pair_mean_square)
        residual_self2.append(moments.residual_self_mean_square)
        pair_residual_self_cross.append(
            moments.pair_residual_self_dot_mean
        )
        joint_pair_self2.append(moments.joint_pair_self_mean_square)
        vector_identity_residuals.append(
            moments.vector_identity_absolute_residual
        )
        component_identity_residuals.append(moments.component_identity_max_abs)

    raw_alias_rms = pooled_rms(raw_alias2)
    self_correction_rms = pooled_rms(correction2)
    pair_self_cross = math.fsum(alias_correction_cross) / len(
        alias_correction_cross
    )
    phase_pair_rms = pooled_rms(distinct_pair2)
    phase_residual_self_rms = pooled_rms(residual_self2)
    phase_pair_self_cross = math.fsum(pair_residual_self_cross) / len(
        pair_residual_self_cross
    )
    joint_pair_self_rms = pooled_rms(joint_pair_self2)
    raw_correction_expanded_joint2 = (
        raw_alias_rms**2 + self_correction_rms**2 + 2.0 * pair_self_cross
    )
    expanded_joint2 = (
        phase_pair_rms**2
        + phase_residual_self_rms**2
        + 2.0 * phase_pair_self_cross
    )
    if not math.isclose(
        joint_pair_self_rms**2,
        expanded_joint2,
        rel_tol=2.0e-12,
        abs_tol=2.0e-18,
    ):
        raise AssertionError("pooled joint pair/self vector identity failed")
    total = math.hypot(joint_pair_self_rms, theory.fourier)
    relative = total / force_scale

    block_rows: list[dict[str, object]] = []
    block_relative: list[float] = []
    for block_index, (start, stop) in enumerate(PILOT_BLOCKS):
        block_chi2, _, _ = adsq.corrected_chi2_with_sampling(
            population, mapping, tagged_blocks[block_index]
        )
        block_pair = pair_rms(charges, block_chi2)
        block_phase_pair = pooled_rms(distinct_pair2[start:stop])
        block_phase_self = pooled_rms(residual_self2[start:stop])
        block_phase_cross = math.fsum(
            pair_residual_self_cross[start:stop]
        ) / (stop - start)
        block_joint = pooled_rms(joint_pair_self2[start:stop])
        block_total = math.hypot(block_joint, theory.fourier)
        block_relative.append(block_total / force_scale)
        block_rows.append(
            {
                "candidate_id": candidate.case.case_id,
                "target": candidate.target.value,
                "block": block_index + 1,
                "frame_first": start + 1,
                "frame_last": stop,
                "measured_stag_corrected_pair_rms": block_pair,
                "phase_resolved_distinct_pair_absolute_rms": block_phase_pair,
                "phase_resolved_residual_self_absolute_rms": block_phase_self,
                "pair_residual_self_dot_mean": block_phase_cross,
                "joint_pair_self_absolute_rms": block_joint,
                "total_predicted_relative_rms": block_total / force_scale,
            }
        )
    frame_sem = statistics.stdev(block_relative) / math.sqrt(len(block_relative))
    diagonal_sampling_sem = alias_relative_sem(
        charges,
        corrected_chi2,
        corrected_pair,
        total,
        force_scale,
        shell_variances,
    )
    # The main joint operator includes every discrete grid alias directly, so
    # it has no importance-sampling uncertainty.  The sampled S_tag quantity
    # remains a diagnostic and its SEM is reported under a distinct field.
    sampling_sem = 0.0
    combined_sem = frame_sem
    upper95 = relative + ONE_SIDED_T95_DF4 * combined_sem
    sigma_up = (
        math.pi
        * baseline.RCUT
        * candidate.mesh
        / (candidate.target.csplit * box)
    )

    row: dict[str, object] = {
        "method": "ESP production AD",
        "scope": candidate.scope,
        "candidate_id": candidate.case.case_id,
        "target": candidate.target.value,
        "target_relative_rms": candidate.target.value,
        "M": candidate.mesh,
        "actual_nx": candidate.mesh,
        "actual_grid_points": candidate.mesh**3,
        "P": candidate.order,
        "order": candidate.order,
        "c_split": candidate.target.csplit,
        "csplit": candidate.target.csplit,
        "c_spread": candidate.target.cspread,
        "cspread": candidate.target.cspread,
        "epsilon_split": candidate.target.epsilon_split,
        "epsilon_spread": candidate.target.epsilon_spread,
        "sigma_up": sigma_up,
        "resolved_band": sigma_up >= 1.0,
        "homogeneous_pair_chi2": population.homogeneous_chi2,
        "measured_stag_corrected_pair_chi2": corrected_chi2,
        "homogeneous_pair_rms": homogeneous_pair,
        "homogeneous_pair_absolute_rms": homogeneous_pair,
        "measured_stag_corrected_pair_rms": corrected_pair,
        "measured_stag_corrected_pair_absolute_rms": corrected_pair,
        "ordinary_sq_corrected_pair_rms_diagnostic": ordinary_pair,
        "residual_self_rms": phase_residual_self_rms,
        "residual_self_absolute_rms": phase_residual_self_rms,
        "residual_self_cell_quadrature_rms_diagnostic": theory.residual_self,
        "phase_resolved_distinct_pair_absolute_rms": phase_pair_rms,
        "phase_resolved_residual_self_absolute_rms": phase_residual_self_rms,
        "pair_residual_self_dot_mean": phase_pair_self_cross,
        "full_source_alias_absolute_rms": raw_alias_rms,
        "production_self_correction_absolute_rms": self_correction_rms,
        "full_source_alias_self_correction_dot_mean": pair_self_cross,
        "joint_pair_self_absolute_rms": joint_pair_self_rms,
        "joint_pair_self_mean_square": joint_pair_self_rms**2,
        "joint_pair_self_expanded_mean_square": expanded_joint2,
        "joint_raw_alias_correction_expanded_mean_square": (
            raw_correction_expanded_joint2
        ),
        "joint_pair_self_vector_identity_max_abs": max(
            vector_identity_residuals
        ),
        "joint_pair_self_component_identity_max_abs": max(
            component_identity_residuals
        ),
        "fourier_rms": theory.fourier,
        "fourier_absolute_rms": theory.fourier,
        "total_predicted_absolute_rms": total,
        "predicted_total_absolute_rms": total,
        "total_predicted_relative_rms": relative,
        "predicted_total_relative_rms": relative,
        "five_block_sem_relative": frame_sem,
        "predicted_total_relative_five_block_sem": frame_sem,
        "predicted_total_relative_block5_sem": frame_sem,
        "alias_sampling_sem_relative": sampling_sem,
        "predicted_total_relative_alias_sampling_sem": sampling_sem,
        "diagonal_stag_alias_sampling_sem_relative_diagnostic": (
            diagonal_sampling_sem
        ),
        "combined_sem_relative": combined_sem,
        "predicted_total_relative_combined_sem": combined_sem,
        "one_sided_95_upper_relative": upper95,
        "predicted_total_relative_one_sided_95_upper": upper95,
        "captured_homogeneous_alias_fraction": (
            population.captured_homogeneous_chi2 / population.homogeneous_chi2
        ),
        "captured_homogeneous_chi2_fraction": (
            population.captured_homogeneous_chi2 / population.homogeneous_chi2
        ),
        "unresolved_homogeneous_alias_fraction": (
            population.unresolved_homogeneous_chi2 / population.homogeneous_chi2
        ),
        "unresolved_homogeneous_chi2_fraction": (
            population.unresolved_homogeneous_chi2 / population.homogeneous_chi2
        ),
        "alias_shell": population.alias_shell,
        "samples_per_alias_shell": population.samples_per_shell,
        "base_mode_count": population.base_mode_count,
        "zeroed_active_mode_count": population.zeroed_active_mode_count,
        "prediction_passes_target": relative <= candidate.target.value,
        "selection_passes_target": (
            sigma_up >= 1.0 and upper95 <= candidate.target.value
        ),
        "selection_result": (
            "pass"
            if sigma_up >= 1.0 and upper95 <= candidate.target.value
            else "fail"
        ),
        "pilot_frames": "1--25",
        "pilot_frame_count": PILOT_N,
        "pilot_coordinate_prefix_sha256": pilot_prefix_sha256,
        "screening_force_scale": force_scale,
        "screening_force_scale_source": (
            "coarse PPPM force evaluation on frames 1--25; no Ewald reference"
        ),
        "prediction_reference_force_accessed": False,
        "prediction_holdout_coordinates_accessed": False,
        "prediction_molecular_coordinates_accessed": True,
        "prediction_structure_input": (
            "phase-resolved full-source and target-cell descriptor from "
            "frames 1--25; measured S_tag retained as a diagonal diagnostic"
        ),
        "ad_estimator": (
            "vector-complete full-source AD pair/self quadratic form from "
            "pilot coordinates; measured S_tag and exact homogeneous cell "
            "moments retained as diagnostics; closed Fourier error in quadrature"
        ),
        "pair_source_scope": "all source particles including j=i",
        "self_source_scope": (
            "production self correction applied to the same all-source vector"
        ),
        "uncertainty_combination": (
            "five contiguous pilot-frame block SEM; the main joint operator "
            "has no alias Monte Carlo sampling"
        ),
        "covariance_approximation": (
            "pair/self and in-band cross-mode covariance retained; covariance "
            "between the joint in-band error and closed Fourier tail neglected"
        ),
        **theory.self_metadata,
    }
    alias_rows = [
        {
            "candidate_id": candidate.case.case_id,
            "target": candidate.target.value,
            "M": candidate.mesh,
            "P": candidate.order,
            "c_spread": candidate.target.cspread,
            "alias_shell": shell,
            "homogeneous_shell_chi2": population.shell_weight_sums[shell],
            "measured_stag_shell_correction_chi2": shell_values[shell],
            "importance_sampling_chi2_sem": math.sqrt(shell_variances[shell]),
            "samples": population.samples_per_shell,
        }
        for shell in range(1, population.alias_shell + 1)
    ]
    return row, block_rows, alias_rows


def save_spectral_theory_spectra(
    path: Path,
    modes: np.ndarray,
    tagged_mean: np.ndarray,
    charge_mean: np.ndarray,
    tagged_blocks: np.ndarray,
    charge_blocks: np.ndarray,
    pilot_prefix_sha256: str,
) -> None:
    np.savez_compressed(
        path,
        modes=np.asarray(modes, dtype=np.int64),
        mean_target_conditioned_s_tag=tagged_mean,
        mean_charge_s_q=charge_mean,
        block_mean_target_conditioned_s_tag=tagged_blocks,
        block_mean_charge_s_q=charge_blocks,
        block_frame_bounds=np.asarray(
            ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25)),
            dtype=np.int64,
        ),
        pilot_coordinate_prefix_sha256=np.asarray(pilot_prefix_sha256),
    )


def evaluate_spectral_theory_screen(
    candidates: list[Candidate],
    *,
    alias_shell: int,
    samples_per_shell: int,
    chunk_size: int,
    rerun_self_probes: bool,
    spectra_path: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    if not candidates:
        raise ValueError("candidate list is empty")
    frames, prefix_digest = load_pilot_frames()
    charges = frames[0][1]
    box = frames[0][3]
    force_scale = baseline.coarse_force_scale()
    started = time.time()
    prepared: list[SpectralADTheory] = []
    for index, candidate in enumerate(candidates, start=1):
        prepared.append(
            prepare_spectral_theory_candidate(
                candidate,
                charges,
                box,
                alias_shell=alias_shell,
                samples_per_shell=samples_per_shell,
                rerun_self_probes=rerun_self_probes,
            )
        )
        print(
            json.dumps(
                {
                    "stage": "prepare_ad_population",
                    "candidate": index,
                    "candidate_count": len(candidates),
                    "id": candidate.case.case_id,
                }
            ),
            flush=True,
        )
    modes, mappings = adsq.population_mode_union(
        [item.population for item in prepared]
    )
    tagged_mean, charge_mean, tagged_blocks, charge_blocks = (
        evaluate_spectral_theory_spectra(frames, modes, chunk_size=chunk_size)
    )
    spectra_path.parent.mkdir(parents=True, exist_ok=True)
    save_spectral_theory_spectra(
        spectra_path,
        modes,
        tagged_mean,
        charge_mean,
        tagged_blocks,
        charge_blocks,
        prefix_digest,
    )

    predictions: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    alias_rows: list[dict[str, object]] = []
    for theory, mapping in zip(prepared, mappings):
        row, local_blocks, local_aliases = evaluate_spectral_theory(
            theory,
            mapping,
            charges,
            frames,
            tagged_mean,
            charge_mean,
            tagged_blocks,
            force_scale,
            prefix_digest,
            box,
        )
        predictions.append(row)
        block_rows.extend(local_blocks)
        alias_rows.extend(local_aliases)
    predictions.sort(
        key=lambda row: (
            -float(row["target_relative_rms"]),
            int(row["actual_grid_points"]),
            int(row["order"]),
            float(row["cspread"]),
        )
    )
    return predictions, block_rows, alias_rows, {
        "elapsed_seconds": time.time() - started,
        "pilot_coordinate_prefix_sha256": prefix_digest,
        "pilot_coordinate_frames_read": PILOT_N,
        "holdout_coordinate_frames_read": 0,
        "structure_mode_count": len(modes),
        "alias_shell": alias_shell,
        "samples_per_shell": samples_per_shell,
        "covariance_policy": (
            "pair/self and in-band cross-mode covariance are retained in one "
            "vector quadratic form; only in-band/Fourier-tail covariance is ignored"
        ),
        "uncertainty_policy": (
            "five contiguous pilot-frame block SEM for the joint operator; "
            "the S_tag alias-sampling SEM is diagnostic only"
        ),
    }


def prediction_manifest(
    purpose: str,
    predictions: Path,
    blocks: Path,
    aliases: Path,
    runtime: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 6,
        "purpose": purpose,
        "logical_order": [
            "read only coordinate frames 1--25 with the prefix reader",
            "evaluate measured target-conditioned S_tag(q) and ordinary S_q(q)",
            "retain S_tag reweighting of the exact AD cell-moment estimator as a diagonal diagnostic",
            "contract the full-source pair alias and production self-correction vectors before squaring",
            "combine the resulting joint in-band RMS with the closed Fourier tail in quadrature",
            "estimate uncertainty from five contiguous 5-frame pilot blocks",
            "freeze the complete candidate table and its SHA-256",
            "permit holdout coordinate/reference access only in a later validation action",
        ],
        "prediction": {
            "reference_force_accessed": False,
            "holdout_coordinates_accessed": False,
            "coordinate_frames": "1--25",
            "coordinate_prefix_sha256": runtime[
                "pilot_coordinate_prefix_sha256"
            ],
            "structure_input": (
                "phase-resolved full-source and target-cell descriptor from frames "
                "1--25; measured S_tag retained as a diagonal diagnostic"
            ),
            "alias_formula": (
                "one analytical discrete-source/gather minus continuum-band "
                "AD error amplitude with all source particles, including j=i"
            ),
            "total_formula": (
                "sqrt(<|e_full-source-alias-c_self|^2> + Delta_F_Fourier^2)"
            ),
            "covariance_approximation": (
                "pair/self and in-band cross-mode covariance retained; covariance "
                "between the joint in-band error and closed Fourier tail neglected"
            ),
            "frame_uncertainty": (
                "SEM over five contiguous blocks of five pilot frames"
            ),
            "alias_uncertainty": (
                "zero for the main all-alias joint operator; importance-sampling "
                "SEM is retained only for the diagonal S_tag diagnostic"
            ),
            "combined_uncertainty": "five-block frame SEM for the joint operator",
            "upper_rule": (
                "prediction + t_0.95,4 * combined_SEM; t_0.95,4="
                f"{ONE_SIDED_T95_DF4:.14g}"
            ),
        },
        "candidate_count": len(read_csv(predictions)),
        "inputs": {
            "trajectory_path": relative_path(TRAJECTORY),
            "trajectory_prefix_frames": "1--25",
            "trajectory_prefix_sha256": runtime[
                "pilot_coordinate_prefix_sha256"
            ],
            "runner": file_record(Path(__file__)),
            "spectral_descriptor": file_record(Path(adsq.__file__)),
            "joint_pair_self_quadratic": file_record(Path(adjoint.__file__)),
            "production_validation_helper": file_record(
                AD_VALIDATION / "water_ad_production.py"
            ),
            "lammps_executable": lammps_record(),
        },
        "outputs": {
            path.name: file_record(path) for path in (predictions, blocks, aliases)
        },
        "runtime": runtime,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "validation": {"performed": False, "used_for_selection": False},
    }


def write_baseline(args: argparse.Namespace) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    predictions, blocks, aliases, runtime = evaluate_spectral_theory_screen(
        baseline_candidates(),
        alias_shell=args.alias_shell,
        samples_per_shell=args.samples_per_shell,
        chunk_size=args.chunk_size,
        rerun_self_probes=args.rerun_self_probes,
        spectra_path=BASELINE_SPECTRA,
    )
    if len(predictions) != len(baseline.candidates()):
        raise RuntimeError("baseline spectral AD matrix is incomplete")
    write_csv(BASELINE_PREDICTION, predictions)
    write_csv(BASELINE_BLOCKS, blocks)
    write_csv(BASELINE_ALIASES, aliases)
    manifest = prediction_manifest(
        "Figure 5 AD curves from the vector-complete pilot-coordinate pair/self theory",
        BASELINE_PREDICTION,
        BASELINE_BLOCKS,
        BASELINE_ALIASES,
        runtime,
    )
    manifest["prediction_table_sha256"] = sha256(BASELINE_PREDICTION)
    BASELINE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(BASELINE_PREDICTION)


def joint_paths(target: float) -> dict[str, Path]:
    directory = OUTDIR / f"joint_{target_tag(target)}"
    return {
        "directory": directory,
        "prediction": directory / "prediction_before_validation.csv",
        "blocks": directory / "prediction_blocks.csv",
        "aliases": directory / "prediction_alias_shell.csv",
        "spectra": directory / "pilot_structure_spectra.npz",
        "frozen": directory / "frozen_selection.json",
        "detail": directory / "holdout_validation_by_frame.csv",
        "summary": directory / "holdout_validation_summary.csv",
        "manifest": directory / "manifest.json",
        "diagnostic": directory / "diagnostic_direct_check_summary.csv",
        "diagnostic_frames": directory / "diagnostic_direct_check_by_frame.csv",
        "convergence": directory / "alias_shell_convergence.csv",
    }


def select_joint(rows: list[dict[str, object]]) -> dict[str, object]:
    passing = [row for row in rows if as_bool(row["selection_passes_target"])]
    if not passing:
        raise RuntimeError(
            "no declared vector-complete AD candidate satisfies the "
            "one-sided prediction target"
        )
    return min(
        passing,
        key=lambda row: (
            int(row["actual_grid_points"]),
            int(row["order"]),
            float(row["cspread"]),
        ),
    )


def write_joint(target: float, args: argparse.Namespace) -> None:
    paths = joint_paths(target)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    predictions, blocks, aliases, runtime = evaluate_spectral_theory_screen(
        joint_candidates(target),
        alias_shell=args.alias_shell,
        samples_per_shell=args.samples_per_shell,
        chunk_size=args.chunk_size,
        rerun_self_probes=args.rerun_self_probes,
        spectra_path=paths["spectra"],
    )
    write_csv(paths["prediction"], predictions)
    write_csv(paths["blocks"], blocks)
    write_csv(paths["aliases"], aliases)
    selected = select_joint(predictions)
    prediction_sha = sha256(paths["prediction"])
    frozen = {
        "schema_version": 3,
        "purpose": (
            "vector-complete AD pair/self parameter selection frozen before "
            "holdout coordinate or Ewald-force access"
        ),
        "candidate_set": {
            "target": target,
            "c_split": float(predictions[0]["csplit"]),
            "meshes": sorted({int(row["actual_nx"]) for row in predictions}),
            "orders": sorted({int(row["order"]) for row in predictions}),
            "c_spread": sorted({float(row["cspread"]) for row in predictions}),
            "candidate_count": len(predictions),
        },
        "selection_rule": (
            "sigma_up >= 1 and prediction+t_0.95,4*combined_SEM <= target; "
            "then minimum M^3, P, c_spread"
        ),
        "prediction_table_path": relative_path(paths["prediction"]),
        "prediction_table_sha256": prediction_sha,
        "pilot_coordinate_prefix_sha256": runtime[
            "pilot_coordinate_prefix_sha256"
        ],
        "prediction_reference_force_accessed": False,
        "prediction_holdout_coordinates_accessed": False,
        "prediction_structure_input": (
            "phase-resolved full-source and target-cell descriptor from frames "
            "1--25; measured S_tag retained as a diagonal diagnostic"
        ),
        "selected": selected,
    }
    paths["frozen"].write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = prediction_manifest(
        "joint vector-complete AD pair/self screen with deferred holdout validation",
        paths["prediction"],
        paths["blocks"],
        paths["aliases"],
        runtime,
    )
    manifest["frozen_selection"] = file_record(paths["frozen"])
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage": "selection_frozen",
                "target": target,
                "selected": selected,
                "frozen_sha256": sha256(paths["frozen"]),
            }
        )
    )


def verify_frozen(target: float) -> tuple[dict[str, object], dict[str, Path]]:
    paths = joint_paths(target)
    for key in ("prediction", "frozen", "manifest"):
        if not paths[key].is_file():
            raise FileNotFoundError(
                f"freeze target {target:.0e} before validation: {paths[key]}"
            )
    frozen = json.loads(paths["frozen"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    frozen_record = manifest.get("frozen_selection")
    if not isinstance(frozen_record, dict):
        raise RuntimeError("the freeze manifest does not record frozen_selection")
    expected_frozen_record = file_record(paths["frozen"])
    if frozen_record != expected_frozen_record:
        raise RuntimeError("the frozen-selection JSON disagrees with its manifest SHA256")
    if frozen["prediction_table_sha256"] != sha256(paths["prediction"]):
        raise RuntimeError("the frozen candidate table changed after selection")
    if bool(frozen["prediction_reference_force_accessed"]):
        raise RuntimeError("the frozen prediction accessed reference forces")
    if bool(frozen["prediction_holdout_coordinates_accessed"]):
        raise RuntimeError("the frozen prediction accessed holdout coordinates")
    selected = frozen["selected"]
    matches = [
        row
        for row in read_csv(paths["prediction"])
        if row["candidate_id"] == selected["candidate_id"]
    ]
    if len(matches) != 1:
        raise RuntimeError("frozen selection is not a unique candidate-table row")
    candidate = matches[0]
    if set(selected) != set(candidate):
        raise RuntimeError("frozen selection fields do not match the candidate-table schema")
    for key, selected_value in selected.items():
        candidate_value = candidate[key]
        if isinstance(selected_value, bool):
            matches_field = as_bool(candidate_value) is selected_value
        elif isinstance(selected_value, int):
            try:
                matches_field = int(candidate_value) == selected_value
            except ValueError:
                matches_field = False
        elif isinstance(selected_value, float):
            try:
                matches_field = float(candidate_value) == selected_value
            except ValueError:
                matches_field = False
        else:
            matches_field = candidate_value == selected_value
        if not matches_field:
            raise RuntimeError(
                f"frozen selection field differs from candidate table: {key}"
            )
    return frozen, paths


def validate_joint(target: float, *, rerun_lammps: bool) -> None:
    frozen, paths = verify_frozen(target)
    selected = frozen["selected"]
    case_target = baseline.Target(
        value=target,
        epsilon_split=float(selected["epsilon_split"]),
        epsilon_spread=float(selected["epsilon_spread"]),
        csplit=float(selected["csplit"]),
        cspread=float(selected["cspread"]),
        meshes=(int(selected["actual_nx"]),),
    )
    case = baseline.case_for(
        case_target, int(selected["order"]), int(selected["actual_nx"])
    )

    # Importing the production validation module is intentionally delayed
    # until the frozen artifact and candidate-table hash have been verified.
    from water_ad_production import (  # noqa: E402
        PILOT_COUNT,
        TOTAL_COUNT,
        refresh_ewald_reference,
        run_water_case,
    )

    observed, run_paths = run_water_case(case, rerun=rerun_lammps)
    reference, _, reference_paths = refresh_ewald_reference(rerun=rerun_lammps)
    if len(observed) != TOTAL_COUNT or len(reference) != TOTAL_COUNT:
        raise RuntimeError("production AD/Ewald validation has incomplete force records")
    details: list[dict[str, object]] = []
    for frame_index in range(PILOT_COUNT, TOTAL_COUNT):
        observed_time, full, _, _ = observed[frame_index]
        reference_time, reference_force = reference[frame_index]
        if observed_time != reference_time:
            raise RuntimeError("production AD/Ewald validation timestep mismatch")
        difference = full - reference_force
        details.append(
            {
                "candidate_id": case.case_id,
                "target": target,
                "frame": frame_index + 1,
                "frame_zero_based": frame_index,
                "timestep": observed_time,
                "sum_total_difference_squared": float(np.sum(difference**2)),
                "sum_reference_squared": float(np.sum(reference_force**2)),
                "frame_relative_rms": math.sqrt(
                    float(np.sum(difference**2) / np.sum(reference_force**2))
                ),
            }
        )
    if len(details) != 26:
        raise RuntimeError("holdout must contain frames 26--51")
    diff2 = math.fsum(float(row["sum_total_difference_squared"]) for row in details)
    ref2 = math.fsum(float(row["sum_reference_squared"]) for row in details)
    block_values = []
    block_sizes = []
    for start, stop in HOLDOUT_BLOCKS:
        block = details[start:stop]
        block_sizes.append(len(block))
        block_values.append(
            math.sqrt(
                math.fsum(
                    float(row["sum_total_difference_squared"]) for row in block
                )
                / math.fsum(float(row["sum_reference_squared"]) for row in block)
            )
        )
    if block_sizes != [5, 5, 5, 5, 6]:
        raise AssertionError(f"unexpected holdout blocks: {block_sizes}")
    holdout = math.sqrt(diff2 / ref2)
    validation_sem = statistics.stdev(block_values) / math.sqrt(len(block_values))
    prediction = float(selected["predicted_total_relative_rms"])
    summary = {
        "candidate_id": case.case_id,
        "target_relative_rms": target,
        "actual_nx": case.mesh,
        "actual_grid_points": case.mesh**3,
        "order": case.order,
        "csplit": case.csplit,
        "cspread": case.cspread,
        "validation_frames": "26--51",
        "validation_frame_count": len(details),
        "validation_block_sizes": "5+5+5+5+6",
        "validation_relative_rms": holdout,
        "validation_relative_rms_block5_sem": validation_sem,
        "prediction_relative_rms": prediction,
        "prediction_upper95_relative": float(
            selected["predicted_total_relative_one_sided_95_upper"]
        ),
        "prediction_to_validation_ratio": prediction / holdout,
        "validation_passes_target": holdout <= target,
        "selection_used_holdout": False,
        "validation_operator": (
            "production LAMMPS ESP AD with unit-charge residual-self correction"
        ),
        "validation_reference": "tight Ewald total force, input tolerance 1e-12",
    }
    write_csv(paths["detail"], details)
    write_csv(paths["summary"], [summary])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["validation"] = {
        "performed": True,
        "performed_after_frozen_sha256_verification": True,
        "used_for_prediction": False,
        "used_for_selection": False,
        "frames": "26--51",
        "block_sizes": [5, 5, 5, 5, 6],
        "detail": file_record(paths["detail"]),
        "summary": file_record(paths["summary"]),
        "production_ad_artifacts": {
            key: {
                label: relative_path(path)
                for label, path in records.items()
            }
            for key, records in run_paths.items()
        },
        "ewald_artifacts": {
            key: relative_path(path) for key, path in reference_paths.items()
        },
        "lammps_executable": lammps_record(),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def join_baseline_validation() -> None:
    if not BASELINE_PREDICTION.is_file() or not BASELINE_MANIFEST.is_file():
        raise FileNotFoundError("write the baseline prediction before validation")
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if sha256(BASELINE_PREDICTION) != manifest["prediction_table_sha256"]:
        raise RuntimeError("baseline prediction changed before validation")
    if not SCAN_SUMMARY.is_file():
        raise FileNotFoundError(
            "run fig5_ik_ad_order_scan/run_fig5_ik_ad_order_scan.py first"
        )
    validation = {
        (float(row["target_relative_rms"]), int(row["order"]), int(row["actual_nx"])): row
        for row in read_csv(SCAN_SUMMARY)
        if row["method"] == "ad"
    }
    joined = []
    for row in read_csv(BASELINE_PREDICTION):
        key = (
            float(row["target_relative_rms"]),
            int(row["order"]),
            int(row["actual_nx"]),
        )
        held = validation.get(key)
        if held is None:
            raise KeyError(f"missing independent AD holdout result for {key}")
        predicted = float(row["predicted_total_relative_rms"])
        measured = float(held["holdout_relative_rms"])
        joined.append(
            {
                **row,
                "validation_relative_rms": measured,
                "validation_relative_rms_balanced_block5_sem": float(
                    held["holdout_balanced_block5_sem"]
                ),
                "validation_passes_target": measured
                <= float(row["target_relative_rms"]),
                "prediction_to_validation_ratio": predicted / measured,
                "validation_frame_first": 26,
                "validation_frame_last": 51,
                "validation_frame_count": 26,
                "validation_block_sizes": "5+5+5+5+6",
                "validation_operator": held["operator"],
                "validation_reference": "tight Ewald total force",
                "validation_used_for_prediction": False,
                "validation_used_for_selection": False,
            }
        )
    write_csv(BASELINE_SOURCE, joined)
    manifest["validation"] = {
        "performed": True,
        "joined_after_prediction_table_sha256_verification": True,
        "used_for_prediction_or_selection": False,
        "frames": "26--51",
        "block_sizes": [5, 5, 5, 5, 6],
        "source": file_record(SCAN_SUMMARY),
        "output": file_record(BASELINE_SOURCE),
    }
    BASELINE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(BASELINE_SOURCE)


def canonical_modes(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh = kernel.shape[0]
    indices = np.nonzero(kernel)
    axis = np.rint(np.fft.fftfreq(mesh) * mesh).astype(np.int64)
    modes = np.column_stack(
        (axis[indices[0]], axis[indices[1]], axis[indices[2]])
    ).astype(np.int64)
    values = kernel[indices].astype(np.float64, copy=False)
    ordering = np.lexsort((modes[:, 2], modes[:, 1], modes[:, 0]))
    return modes[ordering], values[ordering]


def active_mode_force(
    q: np.ndarray,
    xyz: np.ndarray,
    box: float,
    modes: np.ndarray,
    kernel: np.ndarray,
    *,
    block_size: int = 96,
) -> np.ndarray:
    wavevectors = (2.0 * math.pi / box) * modes
    force = np.zeros((len(q), 3), dtype=np.float64)
    prefactor = ikref.COULOMB_REAL / box**3
    for start in range(0, len(kernel), block_size):
        stop = min(start + block_size, len(kernel))
        kblock = wavevectors[start:stop]
        phase_minus = np.exp(-1j * xyz @ kblock.T)
        rho = q @ phase_minus
        mode_force = (-1j * kernel[start:stop] * rho)[:, None] * kblock
        force += q[:, None] * (phase_minus.conj() @ mode_force).real
    return prefactor * force


def diagnostic_direct_check(target: float) -> None:
    frozen, paths = verify_frozen(target)
    selected = frozen["selected"]
    frames, prefix_digest = load_pilot_frames()
    target_spec = baseline.Target(
        value=target,
        epsilon_split=float(selected["epsilon_split"]),
        epsilon_spread=float(selected["epsilon_spread"]),
        csplit=float(selected["csplit"]),
        cspread=float(selected["cspread"]),
        meshes=(int(selected["actual_nx"]),),
    )
    candidate = Candidate(
        target_spec,
        int(selected["order"]),
        int(selected["actual_nx"]),
        "diagnostic_only",
    )
    case = candidate.case
    coeff = coefficients(case)
    op = operator(case, frames[0][3])
    modes, kernel = canonical_modes(op.kernel)
    correction = np.asarray(
        [
            float(selected["self_correction_sin1"]),
            float(selected["self_correction_sin2"]),
        ]
    )
    rows = []
    for frame_index, (timestep, q, xyz, box) in enumerate(frames):
        mesh_force = adref.fixed_ad_mesh_force(q, xyz, op, coeff.real)
        mesh_force -= correction_force(
            q, xyz, box, candidate.mesh, correction
        )
        direct = active_mode_force(q, xyz, box, modes, kernel)
        difference = mesh_force - direct
        rows.append(
            {
                "candidate_id": case.case_id,
                "pilot_frame": frame_index + 1,
                "timestep": timestep,
                "mesh_minus_direct_mean_square": float(
                    np.mean(np.sum(difference**2, axis=1))
                ),
                "pilot_coordinate_prefix_sha256": prefix_digest,
                "used_for_selection": False,
            }
        )
    direct_rms = math.sqrt(
        float(np.mean([row["mesh_minus_direct_mean_square"] for row in rows]))
    )
    diagonal_pair_rms = float(selected["measured_stag_corrected_pair_rms"])
    pair_rms_value = float(
        selected["phase_resolved_distinct_pair_absolute_rms"]
    )
    self_rms_value = float(selected["residual_self_rms"])
    theory_band_rms = float(selected["joint_pair_self_absolute_rms"])
    force_scale = float(selected["screening_force_scale"])
    summary = {
        "candidate_id": case.case_id,
        "target": target,
        "pilot_frames": "1--25",
        "diagonal_stag_pair_rms_diagnostic": diagonal_pair_rms,
        "phase_resolved_distinct_pair_rms": pair_rms_value,
        "phase_resolved_residual_self_rms": self_rms_value,
        "pair_residual_self_dot_mean": float(
            selected["pair_residual_self_dot_mean"]
        ),
        "pair_self_scalar_quadrature_rms_diagnostic": math.hypot(
            pair_rms_value, self_rms_value
        ),
        "vector_complete_pair_self_rms": theory_band_rms,
        "direct_finite_band_mesh_rms": direct_rms,
        "vector_complete_to_direct_ratio": theory_band_rms / direct_rms,
        "direct_finite_band_relative_rms": direct_rms / force_scale,
        "spectral_total_relative_rms": float(
            selected["predicted_total_relative_rms"]
        ),
        "used_for_prediction": False,
        "used_for_selection": False,
        "diagnostic_definition": (
            "post-freeze comparison of direct finite-band force difference with "
            "the frozen vector-complete pair/self quadratic form"
        ),
    }
    write_csv(paths["diagnostic_frames"], rows)
    write_csv(paths["diagnostic"], [summary])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["direct_finite_band_diagnostic"] = {
        "performed_after_selection_freeze": True,
        "used_for_prediction_or_selection": False,
        "summary": file_record(paths["diagnostic"]),
        "by_frame": file_record(paths["diagnostic_frames"]),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def spectral_theory_audit(target: float, args: argparse.Namespace) -> None:
    """Audit shell convergence of the diagonal S_tag diagnostic estimator."""
    frozen, paths = verify_frozen(target)
    selected = frozen["selected"]
    frames, _ = load_pilot_frames()
    charges = frames[0][1]
    box = frames[0][3]
    target_spec = baseline.Target(
        value=target,
        epsilon_split=float(selected["epsilon_split"]),
        epsilon_spread=float(selected["epsilon_spread"]),
        csplit=float(selected["csplit"]),
        cspread=float(selected["cspread"]),
        meshes=(int(selected["actual_nx"]),),
    )
    candidate = Candidate(
        target_spec,
        int(selected["order"]),
        int(selected["actual_nx"]),
        "alias_convergence",
    )
    populations = []
    for shell in range(1, 6):
        populations.append(
            adsq.prepare_ad_source_spectrum_population(
                q=charges,
                mesh=candidate.mesh,
                order=candidate.order,
                box_length=box,
                rcut=baseline.RCUT,
                csplit=target_spec.csplit,
                cspread=target_spec.cspread,
                coeff=coefficients(candidate.case),
                max_shell=shell,
                samples_per_shell=args.samples_per_shell,
                # Common random numbers isolate alias-shell truncation from
                # importance-sampling noise in the shell 1--5 comparison.
                seed=candidate_seed(candidate, 5),
            )
        )
    modes, mappings = adsq.population_mode_union(populations)
    tagged, _, _, _ = evaluate_spectral_theory_spectra(
        frames, modes, chunk_size=args.chunk_size
    )
    force_scale = float(selected["screening_force_scale"])
    rows = []
    for shell, population, mapping in zip(range(1, 6), populations, mappings):
        homogeneous, corrections, homogeneous_variance = (
            adsq.corrected_chi2_with_sampling(
                population, mapping, np.ones(len(modes))
            )
        )
        if homogeneous != population.homogeneous_chi2:
            raise AssertionError("S_tag=1 did not exactly recover homogeneous chi2")
        if any(value != 0.0 for value in corrections.values()):
            raise AssertionError("S_tag=1 has a nonzero structure correction")
        if any(value != 0.0 for value in homogeneous_variance.values()):
            raise AssertionError("S_tag=1 has nonzero alias sampling variance")
        chi2, _, variances = adsq.corrected_chi2_with_sampling(
            population, mapping, tagged
        )
        pair = pair_rms(charges, chi2)
        rows.append(
            {
                "candidate_id": candidate.case.case_id,
                "target": target,
                "alias_shell": shell,
                "captured_homogeneous_alias_fraction": (
                    population.captured_homogeneous_chi2
                    / population.homogeneous_chi2
                ),
                "spectral_pair_relative_rms": pair / force_scale,
                "alias_sampling_chi2_sem": math.sqrt(
                    math.fsum(variances.values())
                ),
                "homogeneous_recovery_exact": True,
            }
        )
    shell5 = float(rows[-1]["spectral_pair_relative_rms"])
    for row in rows:
        row["relative_difference_from_shell5"] = (
            float(row["spectral_pair_relative_rms"]) / shell5 - 1.0
        )
    write_csv(paths["convergence"], rows)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["alias_shell_convergence"] = {
        "performed_after_selection_freeze": True,
        "used_for_prediction_or_selection": False,
        "scope": "Figure-5 diagonal S_tag diagnostic shell convergence",
        "shells": [1, 2, 3, 4, 5],
        "homogeneous_recovery_exact_for_every_shell": True,
        "output": file_record(paths["convergence"]),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


def direct_tagged_definition(
    q: np.ndarray, xyz: np.ndarray, box: float, modes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    wave = 2.0 * math.pi * np.asarray(modes, dtype=np.float64) / box
    phase_minus = np.exp(-1j * xyz @ wave.T)
    rho = q @ phase_minus
    denominator = float(np.sum(q * q) ** 2 - np.sum(q**4))
    tagged = np.empty(len(modes))
    for mode_index in range(len(modes)):
        removed = phase_minus[:, mode_index].conj() * rho[mode_index] - q
        tagged[mode_index] = (
            float(np.sum(q * q * np.abs(removed) ** 2)) / denominator
        )
    ordinary = np.abs(rho) ** 2 / np.sum(q * q)
    return tagged, ordinary


def run_unit_tests() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    q = np.asarray((1.0, -0.7, 0.2, -0.5))
    xyz = np.asarray(
        (
            (0.3, 1.7, 2.1),
            (1.1, 2.6, 3.2),
            (2.8, 0.4, 1.9),
            (3.4, 3.1, 0.7),
        )
    )
    modes = np.asarray(((1, 0, 0), (0, 1, -1), (2, -1, 1), (-2, 1, 0)))
    fast = adsq.evaluate_tagged_pair_spectrum(
        q, xyz, 4.0, modes, return_charge_spectrum=True
    )
    expected_tagged, expected_ordinary = direct_tagged_definition(
        q, xyz, 4.0, modes
    )
    tagged_error = float(np.max(np.abs(fast[0] - expected_tagged)))
    ordinary_error = float(np.max(np.abs(fast[1] - expected_ordinary)))
    if tagged_error > 2.0e-14 or ordinary_error > 2.0e-14:
        raise AssertionError("target-conditioned spectrum direct-sum test failed")

    target = baseline.TARGETS[0]
    case = baseline.case_for(target, 5, 12)
    population = adsq.prepare_ad_source_spectrum_population(
        q=q,
        mesh=12,
        order=5,
        box_length=4.0,
        rcut=1.0,
        csplit=target.csplit,
        cspread=target.cspread,
        coeff=coefficients(case),
        max_shell=1,
        samples_per_shell=64,
        seed=7,
    )
    union, mappings = adsq.population_mode_union([population])
    recovered, corrections, variances = adsq.corrected_chi2_with_sampling(
        population, mappings[0], np.ones(len(union))
    )
    exact_homogeneous = recovered == population.homogeneous_chi2
    if (
        not exact_homogeneous
        or any(value != 0.0 for value in corrections.values())
        or any(value != 0.0 for value in variances.values())
    ):
        raise AssertionError("S_tag=1 homogeneous recovery test failed")

    joint_operator = adref.build_ad_operator(
        12,
        4.0,
        5,
        1.0,
        target.csplit,
        target.cspread,
        coefficients(case),
    )
    joint_correction = np.asarray((1.25e-4, -2.5e-5), dtype=np.float64)
    joint = adjoint.evaluate_joint_pair_self_quadratic(
        q,
        xyz,
        joint_operator,
        coefficients(case).real,
        joint_correction,
        mode_block_size=17,
    )
    reference_mesh = adref.fixed_ad_mesh_force(
        q, xyz, joint_operator, coefficients(case).real
    )
    reference_band = ikref.direct_truncated_force(
        q, xyz, 4.0, joint_operator.kernel
    )
    reference_joint = (
        reference_mesh
        - reference_band
        - correction_force(q, xyz, 4.0, 12, joint_correction)
    )
    reference_joint2 = float(
        np.mean(np.einsum("ij,ij->i", reference_joint, reference_joint))
    )
    joint_operator_error = abs(
        joint.joint_pair_self_mean_square - reference_joint2
    )
    if joint_operator_error > 3.0e-15 * max(reference_joint2, 1.0):
        raise AssertionError("joint pair/self analytical operator test failed")
    pilot, prefix_digest = load_pilot_frames()
    payload = {
        "schema_version": 2,
        "evaluate_tagged_pair_spectrum_direct_sum_max_abs": tagged_error,
        "ordinary_sq_direct_sum_max_abs": ordinary_error,
        "homogeneous_recovery_bitwise_exact": exact_homogeneous,
        "homogeneous_chi2": population.homogeneous_chi2,
        "joint_pair_self_vector_identity_absolute_residual": (
            joint.vector_identity_absolute_residual
        ),
        "joint_pair_plus_residual_self_component_max_abs": (
            joint.component_identity_max_abs
        ),
        "joint_pair_self_reference_mean_square_max_abs": joint_operator_error,
        "prefix_reader_frame_count": len(pilot),
        "prefix_reader_first_timestep": pilot[0][0],
        "prefix_reader_last_timestep": pilot[-1][0],
        "prefix_reader_sha256": prefix_digest,
        "passed": True,
    }
    UNIT_TESTS.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--baseline", action="store_true")
    action.add_argument("--join-baseline-validation", action="store_true")
    action.add_argument("--joint-target", type=float, metavar="TARGET")
    action.add_argument("--validate-joint", type=float, metavar="TARGET")
    action.add_argument("--diagnostic-direct-check", type=float, metavar="TARGET")
    action.add_argument(
        "--spectral-theory-audit",
        type=float,
        metavar="TARGET",
        help="post-freeze shell-convergence audit for the diagonal S_tag diagnostic",
    )
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--alias-shell", type=int, default=5)
    parser.add_argument("--samples-per-shell", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--rerun-self-probes", action="store_true")
    parser.add_argument("--rerun-lammps", action="store_true")
    parser.add_argument(
        "--lmp",
        type=Path,
        default=None,
        help="ESP-LAMMPS executable; defaults to ESP_LAMMPS_BIN or the local build",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adcommon.configure_lmp(args.lmp)
    if args.alias_shell < 1:
        raise ValueError("--alias-shell must be positive")
    if args.samples_per_shell < 64:
        raise ValueError("--samples-per-shell must be at least 64")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if args.baseline:
        write_baseline(args)
    elif args.join_baseline_validation:
        join_baseline_validation()
    elif args.joint_target is not None:
        write_joint(args.joint_target, args)
    elif args.validate_joint is not None:
        validate_joint(args.validate_joint, rerun_lammps=args.rerun_lammps)
    elif args.diagnostic_direct_check is not None:
        diagnostic_direct_check(args.diagnostic_direct_check)
    elif args.spectral_theory_audit is not None:
        spectral_theory_audit(args.spectral_theory_audit, args)
    elif args.self_test:
        run_unit_tests()
    else:
        raise AssertionError("one action is required")


if __name__ == "__main__":
    main()
