#!/usr/bin/env python3
r"""Generate and validate the structure-conditioned AD theory for Figure 5.

The workflow has a strict data boundary. ``--freeze-pilot-spectrum`` is the
only prediction-side action that reads molecular coordinates: it reads the
first 25 frames with a prefix reader and freezes ``S_tag(q)``, ordinary
``S_q(q)``, and the charge-class-conditioned pair amplitude ``mu_a(q)`` with
their input-prefix hashes. All candidate prediction and parameter-selection
actions subsequently read only those frozen structural spectra, algorithm
parameters, charge moments, and the frozen force normalization.

The zero-mean pair fluctuation reweights the exact homogeneous AD cell-moment
source weights. The coherent all-source pair field, including ``i=j``, and
the production self-correction field are added as vectors on one mesh cell
before squaring. This is checked against the equivalent distinct-pair plus
residual-self decomposition. The closed Fourier tail is then added in
quadrature. This is a pure spectral/cell-moment contraction and includes
pair/self covariance without evaluating particlewise force differences.

The phase-resolved joint operator and direct finite-band force difference are
available only through ``--diagnostic-direct-check`` after a selection has
been frozen.  They diagnose the closure error and never participate in
prediction or selection.
"""

from __future__ import annotations

import argparse
import csv
import gc
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
import ad_pair_self_theory as adps  # noqa: E402
import ad_joint_quadratic as adjoint  # noqa: E402
import fixed_ad_reference as adref  # noqa: E402
import fixed_ik_reference as ikref  # noqa: E402
# Shared targets, normalization, and cell-quadrature helpers.
import fig5_ad_theory_common as baseline  # noqa: E402
import ad_validation_common as adcommon  # noqa: E402
from generated_output import section_output_root  # noqa: E402
from ad_validation_common import (  # noqa: E402
    coefficients,
    correction_force,
    operator,
    production_self_correction_coefficients,
)


TRAJECTORY = baseline.WATER_ROOT / "water_short_traj.lammpstrj"
OUTPUT_ROOT = section_output_root()
SCAN_SUMMARY = OUTPUT_ROOT / "fig5_ik_ad_order_scan" / "fig5_ik_ad_order_scan_summary.csv"
OUTDIR = OUTPUT_ROOT / "fig5_ad_stag_screen"

PILOT_SPECTRUM = OUTDIR / "pilot_stag_spectrum.npz"
PILOT_SPECTRUM_FROZEN = OUTDIR / "pilot_stag_spectrum_frozen.json"
PILOT_SPECTRUM_MANIFEST = OUTDIR / "pilot_stag_spectrum_manifest.json"

BASELINE_PREDICTION = OUTDIR / "baseline_prediction.csv"
BASELINE_BLOCKS = OUTDIR / "baseline_prediction_by_frame.csv"
BASELINE_ALIASES = OUTDIR / "baseline_alias_shell.csv"
BASELINE_SOURCE = OUTDIR / "baseline_source.csv"
BASELINE_MANIFEST = OUTDIR / "baseline_manifest.json"
UNIT_TESTS = OUTDIR / "theory_unit_tests.json"

PILOT_N = 25
PILOT_BLOCKS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25))
HOLDOUT_BLOCKS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 26))
ONE_SIDED_T95_DF4 = 2.13184678632665


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
    ad_operator: adref.ADOperator
    real_coeff: np.ndarray
    self_quadrature_order_per_half: int
    self_metadata: dict[str, object]


@dataclass(frozen=True)
class FrozenPilotSpectrum:
    modes: np.ndarray
    tagged_mean: np.ndarray
    charge_mean: np.ndarray
    tagged_blocks: np.ndarray
    charge_blocks: np.ndarray
    conditional_pair_mean: np.ndarray
    conditional_pair_blocks: np.ndarray
    charge_classes: np.ndarray
    class_counts: np.ndarray
    charges: np.ndarray
    box_length: float
    force_scale: float
    coordinate_prefix_sha256: str
    force_prefix_sha256: str
    alias_shell: int
    samples_per_shell: int
    candidate_catalog_sha256: str
    artifact_sha256: str


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
    """Return a bundle-relative or configured-output-relative path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT).as_posix()
    except ValueError:
        try:
            suffix = resolved.relative_to(OUTPUT_ROOT)
        except ValueError:
            return resolved.as_posix()
        return f"$OUTPUT_ROOT/{suffix.as_posix()}"


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


def stag_candidates(target_value: float) -> list[Candidate]:
    if math.isclose(target_value, 1.0e-4, rel_tol=0.0, abs_tol=1.0e-15):
        csplit = 12.024
        meshes = (15, 16, 18, 20)
        branches = ((12.024, 1.0e-4), (13.251, 3.0e-5), (14.471, 1.0e-5))
    elif math.isclose(target_value, 1.0e-5, rel_tol=0.0, abs_tol=1.0e-16):
        csplit = 14.471
        meshes = (16, 18, 20, 24)
        branches = ((14.471, 1.0e-5), (16.894, 1.0e-6))
    else:
        raise ValueError("S_tag AD screens are defined only for 1e-4 and 1e-5")

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
                result.append(Candidate(target, order, mesh, "stag_selection"))
    return result


def all_spectrum_candidates() -> list[Candidate]:
    """Return the deterministic union covered by the frozen pilot spectrum."""

    by_id: dict[str, Candidate] = {}
    for candidate in (
        baseline_candidates() + stag_candidates(1.0e-4) + stag_candidates(1.0e-5)
    ):
        case_id = candidate.case.case_id
        previous = by_id.get(case_id)
        if previous is not None and (
            previous.mesh != candidate.mesh
            or previous.order != candidate.order
            or previous.target.csplit != candidate.target.csplit
            or previous.target.cspread != candidate.target.cspread
        ):
            raise RuntimeError(f"conflicting Figure-5 candidate ID: {case_id}")
        by_id[case_id] = candidate
    return [by_id[key] for key in sorted(by_id)]


def candidate_catalog_sha256(
    candidates: list[Candidate], alias_shell: int, samples_per_shell: int
) -> str:
    payload = {
        "alias_shell": alias_shell,
        "samples_per_shell": samples_per_shell,
        "coherent_pair_self_closure": "charge-class conditional mean",
        "coherent_source_aliases": adps.FACE_ALIASES.tolist(),
        "candidates": [
            {
                "candidate_id": candidate.case.case_id,
                "target": candidate.target.value,
                "M": candidate.mesh,
                "P": candidate.order,
                "c_split": candidate.target.csplit,
                "c_spread": candidate.target.cspread,
            }
            for candidate in candidates
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def frozen_force_scale() -> tuple[float, str]:
    """Read only the 25-frame normalization prefix during spectrum freezing."""

    frames, prefix_digest = ikref.parse_force_dump_prefix(
        baseline.PILOT_FORCE_DUMP, PILOT_N, return_sha256=True
    )
    per_frame = np.asarray(
        [
            math.sqrt(float(np.mean(np.sum(force * force, axis=1))))
            for _, force in frames
        ],
        dtype=np.float64,
    )
    scale = math.sqrt(float(np.mean(per_frame * per_frame)))
    if not math.isclose(scale, 27.379457967539718, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"unexpected Figure-5 force scale: {scale:.16g}")
    return scale, prefix_digest


def candidate_seed(candidate: Candidate, alias_shell: int) -> int:
    payload = (
        f"{candidate.target.value:.17g}|{candidate.mesh}|{candidate.order}|"
        f"{candidate.target.csplit:.17g}|{candidate.target.cspread:.17g}|"
        f"{alias_shell}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def prepare_population(
    candidate: Candidate,
    charges: np.ndarray,
    box: float,
    *,
    alias_shell: int,
    samples_per_shell: int,
    coeff: ikref.PSWFCoefficients | None = None,
) -> adsq.ADSourceSpectrumPopulation:
    return adsq.prepare_ad_source_spectrum_population(
        q=charges,
        mesh=candidate.mesh,
        order=candidate.order,
        box_length=box,
        rcut=baseline.RCUT,
        csplit=candidate.target.csplit,
        cspread=candidate.target.cspread,
        coeff=coefficients(candidate.case) if coeff is None else coeff,
        max_shell=alias_shell,
        samples_per_shell=samples_per_shell,
        seed=candidate_seed(candidate, alias_shell),
    )


def prepare_spectral_theory_candidate(
    candidate: Candidate,
    charges: np.ndarray,
    box: float,
    *,
    alias_shell: int,
    samples_per_shell: int,
) -> SpectralADTheory:
    case = candidate.case
    coeff = coefficients(case)
    population = prepare_population(
        candidate,
        charges,
        box,
        alias_shell=alias_shell,
        samples_per_shell=samples_per_shell,
        coeff=coeff,
    )
    ad_operator = operator(case, box)
    correction, correction_metadata = production_self_correction_coefficients(
        case, box, coeff=coeff, ad_operator=ad_operator
    )
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
        **correction_metadata,
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
        ad_operator=ad_operator,
        real_coeff=np.asarray(coeff.real, dtype=np.float64),
        self_quadrature_order_per_half=quadrature_order,
        self_metadata=metadata,
    )


def evaluate_spectral_theory_spectra(
    frames: list[tuple[int, np.ndarray, np.ndarray, float]],
    modes: np.ndarray,
    *,
    chunk_size: int,
    backend: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    tagged_sum = np.zeros(len(modes), dtype=np.float64)
    charge_sum = np.zeros(len(modes), dtype=np.float64)
    tagged_blocks = np.zeros((len(PILOT_BLOCKS), len(modes)), dtype=np.float64)
    charge_blocks = np.zeros_like(tagged_blocks)
    conditional_sum: np.ndarray | None = None
    conditional_blocks: np.ndarray | None = None
    charge_classes: np.ndarray | None = None
    class_counts: np.ndarray | None = None
    for frame_index, (_, q, xyz, box) in enumerate(frames):
        spectra = adps.evaluate_structure_spectra(
            q,
            xyz,
            box,
            modes,
            backend=backend,
            chunk_size=chunk_size,
        )
        tagged = spectra.tagged_pair
        ordinary = spectra.ordinary_charge
        if conditional_sum is None:
            charge_classes = spectra.charge_classes.copy()
            class_counts = spectra.class_counts.copy()
            conditional_sum = np.zeros_like(spectra.conditional_pair_mean)
            conditional_blocks = np.zeros(
                (len(PILOT_BLOCKS), *spectra.conditional_pair_mean.shape),
                dtype=np.complex128,
            )
        elif not (
            np.array_equal(charge_classes, spectra.charge_classes)
            and np.array_equal(class_counts, spectra.class_counts)
        ):
            raise RuntimeError("target charge classes changed across pilot frames")
        tagged_sum += tagged
        charge_sum += ordinary
        conditional_sum += spectra.conditional_pair_mean
        tagged_blocks[frame_index // 5] += tagged
        charge_blocks[frame_index // 5] += ordinary
        conditional_blocks[frame_index // 5] += spectra.conditional_pair_mean
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
    if (
        conditional_sum is None
        or conditional_blocks is None
        or charge_classes is None
        or class_counts is None
    ):
        raise RuntimeError("no pilot structure spectra were evaluated")
    return (
        tagged_sum / PILOT_N,
        charge_sum / PILOT_N,
        tagged_blocks / 5.0,
        charge_blocks / 5.0,
        conditional_sum / PILOT_N,
        conditional_blocks / 5.0,
        charge_classes,
        class_counts,
    )


def pair_rms(charges: np.ndarray, chi2: float) -> float:
    return math.sqrt(pair_mean_square(charges, chi2))


def pair_mean_square(charges: np.ndarray, chi2: float) -> float:
    q2 = charges * charges
    pair_factor = float((np.sum(q2) ** 2 - np.sum(q2 * q2)) / len(charges))
    return ikref.COULOMB_REAL**2 * max(pair_factor * chi2, 0.0)


def fluctuation_s_tag(
    tagged: np.ndarray, coherent: np.ndarray
) -> tuple[np.ndarray, float]:
    """Subtract the conditional mean while enforcing variance positivity."""

    fluctuation = np.asarray(tagged, dtype=np.float64) - np.asarray(
        coherent, dtype=np.float64
    )
    minimum = float(np.min(fluctuation, initial=0.0))
    tolerance = 512.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(tagged), initial=0.0)), 1.0
    )
    if minimum < -tolerance:
        raise FloatingPointError(
            "conditional-mean subtraction produced a negative pair variance: "
            f"{minimum:.6e}"
        )
    return np.maximum(fluctuation, 0.0), minimum


def pooled_rms(mean_squares: list[float]) -> float:
    if not mean_squares:
        raise ValueError("cannot pool an empty set of frame mean squares")
    return math.sqrt(math.fsum(mean_squares) / len(mean_squares))


def alias_relative_sem(
    charges: np.ndarray,
    total: float,
    force_scale: float,
    shell_variances: dict[int, float],
) -> float:
    if total <= 0.0:
        return 0.0
    chi_sem = math.sqrt(math.fsum(shell_variances.values()))
    q2 = charges * charges
    pair_factor = float((np.sum(q2) ** 2 - np.sum(q2 * q2)) / len(charges))
    total_sem = (
        ikref.COULOMB_REAL**2
        * pair_factor
        * chi_sem
        / (2.0 * total)
    )
    return total_sem / force_scale


def evaluate_spectral_theory(
    theory: SpectralADTheory,
    mapping: dict[str, object],
    charges: np.ndarray,
    modes: np.ndarray,
    tagged_mean: np.ndarray,
    charge_mean: np.ndarray,
    tagged_blocks: np.ndarray,
    conditional_pair_mean: np.ndarray,
    conditional_pair_blocks: np.ndarray,
    charge_classes: np.ndarray,
    class_counts: np.ndarray,
    force_scale: float,
    pilot_prefix_sha256: str,
    spectrum_artifact_sha256: str,
    box: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate the Figure-5 AD theory from frozen structural spectra.

    ``S_tag`` is decomposed into a charge-class conditional mean and a
    zero-mean fluctuation. The latter enters the diagonal cell-moment theory.
    The coherent all-source pair field (including ``i=j``) and the production
    self-correction field are added on one mesh cell before squaring. This is
    checked against the equivalent distinct-pair plus residual-self
    decomposition. No coordinates or force arrays enter this function.
    """
    candidate = theory.candidate
    population = theory.population
    conditional_batch = np.concatenate(
        (conditional_pair_mean[None, ...], conditional_pair_blocks), axis=0
    )
    coherent_batch = adps.coherent_s_tag(
        charges, charge_classes, class_counts, conditional_batch
    )
    fluctuation_batch = np.empty_like(coherent_batch)
    fluctuation_minima = []
    all_tagged = np.concatenate((tagged_mean[None, :], tagged_blocks), axis=0)
    for index in range(len(all_tagged)):
        fluctuation_batch[index], minimum = fluctuation_s_tag(
            all_tagged[index], coherent_batch[index]
        )
        fluctuation_minima.append(minimum)

    fluctuation_chi2, shell_values, shell_variances = (
        adsq.corrected_chi2_with_sampling(
            population, mapping, fluctuation_batch[0]
        )
    )
    diagonal_stag_chi2, _, _ = adsq.corrected_chi2_with_sampling(
        population, mapping, tagged_mean
    )
    ordinary_chi2, _, _ = adsq.corrected_chi2_with_sampling(
        population, mapping, charge_mean
    )
    joint = adps.coherent_pair_self_cell_moments(
        q=charges,
        frozen_modes=modes,
        conditional_pair_mean=conditional_batch,
        charge_classes=charge_classes,
        class_counts=class_counts,
        operator=theory.ad_operator,
        real_coeff=theory.real_coeff,
        self_correction=theory.correction,
        quadrature_order_per_half=theory.self_quadrature_order_per_half,
    )
    coherent_pair2 = np.asarray(joint.coherent_pair_mean_square, dtype=np.float64)
    all_source_pair2 = np.asarray(
        joint.all_source_pair_mean_square, dtype=np.float64
    )
    coherent_joint2 = np.asarray(joint.coherent_joint_mean_square, dtype=np.float64)
    self2 = np.asarray(joint.residual_self_mean_square, dtype=np.float64)
    self_correction2 = np.asarray(
        joint.self_correction_mean_square, dtype=np.float64
    )
    distinct_residual_cross = np.asarray(
        joint.pair_self_dot_mean, dtype=np.float64
    )
    all_source_correction_cross = np.asarray(
        joint.all_source_pair_self_correction_dot_mean, dtype=np.float64
    )
    expected_self2 = theory.residual_self * theory.residual_self
    self_consistency = float(np.max(np.abs(self2 - expected_self2)))
    if self_consistency > 2.0e-10 * max(expected_self2, 1.0e-300):
        raise RuntimeError(
            "joint-theory self moment differs from residual-self quadrature"
        )

    homogeneous_pair = pair_rms(charges, population.homogeneous_chi2)
    fluctuation_pair2 = pair_mean_square(charges, fluctuation_chi2)
    fluctuation_pair = math.sqrt(fluctuation_pair2)
    corrected_pair2 = fluctuation_pair2 + float(coherent_pair2[0])
    corrected_pair = math.sqrt(max(corrected_pair2, 0.0))
    diagonal_stag_pair = pair_rms(charges, diagonal_stag_chi2)
    ordinary_pair = pair_rms(charges, ordinary_chi2)
    mesh_joint2 = fluctuation_pair2 + float(coherent_joint2[0])
    if mesh_joint2 < 0.0:
        raise FloatingPointError("joint AD mesh variance became negative")
    mesh_joint = math.sqrt(mesh_joint2)
    total = math.hypot(mesh_joint, theory.fourier)
    relative = total / force_scale

    block_rows: list[dict[str, object]] = []
    block_relative: list[float] = []
    for block_index, (start, stop) in enumerate(PILOT_BLOCKS):
        block_chi2, _, _ = adsq.corrected_chi2_with_sampling(
            population, mapping, fluctuation_batch[block_index + 1]
        )
        block_fluctuation_pair2 = pair_mean_square(charges, block_chi2)
        block_pair2 = (
            block_fluctuation_pair2 + float(coherent_pair2[block_index + 1])
        )
        block_all_source_pair2 = (
            block_fluctuation_pair2
            + float(all_source_pair2[block_index + 1])
        )
        block_mesh2 = (
            block_fluctuation_pair2 + float(coherent_joint2[block_index + 1])
        )
        if block_pair2 < 0.0 or block_mesh2 < 0.0:
            raise FloatingPointError("block joint AD variance became negative")
        block_pair = math.sqrt(block_pair2)
        block_mesh = math.sqrt(block_mesh2)
        block_total = math.hypot(block_mesh, theory.fourier)
        block_relative.append(block_total / force_scale)
        block_rows.append(
            {
                "candidate_id": candidate.case.case_id,
                "target": candidate.target.value,
                "block": block_index + 1,
                "frame_first": start + 1,
                "frame_last": stop,
                "fluctuation_pair_rms": math.sqrt(block_fluctuation_pair2),
                "coherent_pair_rms": math.sqrt(
                    max(float(coherent_pair2[block_index + 1]), 0.0)
                ),
                "measured_stag_corrected_pair_rms": block_pair,
                "all_source_pair_rms": math.sqrt(block_all_source_pair2),
                "residual_self_absolute_rms": theory.residual_self,
                "self_correction_rms": math.sqrt(
                    float(self_correction2[block_index + 1])
                ),
                "pair_self_dot_mean": float(
                    all_source_correction_cross[block_index + 1]
                ),
                "distinct_pair_residual_self_dot_mean": float(
                    distinct_residual_cross[block_index + 1]
                ),
                "joint_mesh_absolute_rms": block_mesh,
                "fourier_absolute_rms": theory.fourier,
                "total_predicted_relative_rms": block_total / force_scale,
            }
        )
    frame_sem = statistics.stdev(block_relative) / math.sqrt(len(block_relative))
    diagonal_sampling_sem = alias_relative_sem(
        charges,
        total,
        force_scale,
        shell_variances,
    )
    sampling_sem = diagonal_sampling_sem
    combined_sem = math.hypot(frame_sem, sampling_sem)
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
        "diagonal_stag_pair_chi2_diagnostic": diagonal_stag_chi2,
        "fluctuation_pair_chi2": fluctuation_chi2,
        "homogeneous_pair_rms": homogeneous_pair,
        "homogeneous_pair_absolute_rms": homogeneous_pair,
        "diagonal_stag_pair_rms_diagnostic": diagonal_stag_pair,
        "fluctuation_pair_rms": fluctuation_pair,
        "coherent_pair_rms": math.sqrt(max(float(coherent_pair2[0]), 0.0)),
        "measured_stag_corrected_pair_rms": corrected_pair,
        "measured_stag_corrected_pair_absolute_rms": corrected_pair,
        "ordinary_sq_corrected_pair_rms_diagnostic": ordinary_pair,
        "residual_self_rms": theory.residual_self,
        "residual_self_absolute_rms": theory.residual_self,
        "residual_self_cell_quadrature_rms_diagnostic": theory.residual_self,
        "all_source_pair_rms": math.sqrt(
            max(fluctuation_pair2 + float(all_source_pair2[0]), 0.0)
        ),
        "self_correction_rms": math.sqrt(
            max(float(self_correction2[0]), 0.0)
        ),
        "pair_self_dot_mean": float(all_source_correction_cross[0]),
        "pair_self_cross_twice": 2.0 * float(
            all_source_correction_cross[0]
        ),
        "distinct_pair_residual_self_dot_mean": float(
            distinct_residual_cross[0]
        ),
        "distinct_pair_residual_self_cross_twice": 2.0 * float(
            distinct_residual_cross[0]
        ),
        "joint_mesh_rms": mesh_joint,
        "joint_mesh_absolute_rms": mesh_joint,
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
        "frozen_pilot_spectrum_sha256": spectrum_artifact_sha256,
        "screening_force_scale": force_scale,
        "screening_force_scale_source": (
            "coarse PPPM force evaluation on frames 1--25; no Ewald reference"
        ),
        "prediction_reference_force_accessed": False,
        "prediction_holdout_coordinates_accessed": False,
        "prediction_molecular_coordinates_accessed": False,
        "prediction_unit_charge_force_accessed": False,
        "prediction_particlewise_force_difference_evaluated": False,
        "prediction_structure_input": (
            "measured target-conditioned S_tag from frames 1--25"
        ),
        "prediction_coherent_structure_input": (
            "charge-class conditional pair amplitudes from frames 1--25"
        ),
        "ad_estimator": (
            "conditional-mean AD pair/self theory: zero-mean S_tag fluctuation "
            "plus coherent all-source pair and self correction added before "
            "cell squaring; "
            "closed Fourier error in quadrature"
        ),
        "pair_source_scope": (
            "all source particles including i=j; the deterministic raw mesh "
            "self field is added to the S_tag distinct-source pair field"
        ),
        "self_source_scope": (
            "negative production self-correction field combined with the "
            "all-source pair field"
        ),
        "uncertainty_combination": (
            "quadrature of five-block spectrum SEM and alias-importance-sampling SEM"
        ),
        "covariance_approximation": (
            "zero-mean pair fluctuations retain a diagonal physical-mode "
            "closure; coherent source aliases beyond six faces and "
            "in-band/Fourier-tail covariance are neglected"
        ),
        "pair_self_covariance_included": True,
        "pair_self_covariance_scope": (
            "charge-class conditional coherent all-source pair field and "
            "production self-correction field are added before cell squaring; "
            "equivalent to distinct pair plus residual self"
        ),
        "equivalent_pair_self_decompositions": (
            "F_pair,all + F_self,corr = F_pair,j!=i + F_residual-self"
        ),
        "equivalent_decomposition_component_max_abs": (
            joint.equivalent_decomposition_component_max_abs
        ),
        "coherent_source_aliases": "zero plus six nearest faces",
        "coherent_source_alias_count": len(adps.FACE_ALIASES),
        "conditional_fluctuation_minimum": min(fluctuation_minima),
        "joint_self_quadrature_consistency_absolute": self_consistency,
        "joint_quadratic_identity_max_abs": joint.quadratic_identity_max_abs,
        "joint_maximum_imaginary_component": joint.maximum_imaginary_component,
        "joint_production_real_projection_applied": (
            joint.production_real_projection_applied
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
            "fluctuation_stag_shell_correction_chi2": shell_values[shell],
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
    conditional_pair_mean: np.ndarray,
    conditional_pair_blocks: np.ndarray,
    charge_classes: np.ndarray,
    class_counts: np.ndarray,
    charges: np.ndarray,
    box_length: float,
    force_scale: float,
    pilot_prefix_sha256: str,
    force_prefix_sha256: str,
    alias_shell: int,
    samples_per_shell: int,
    catalog_sha256: str,
) -> None:
    np.savez_compressed(
        path,
        modes=np.asarray(modes, dtype=np.int64),
        mean_target_conditioned_s_tag=tagged_mean,
        mean_charge_s_q=charge_mean,
        block_mean_target_conditioned_s_tag=tagged_blocks,
        block_mean_charge_s_q=charge_blocks,
        mean_charge_class_conditional_pair_amplitude=np.asarray(
            conditional_pair_mean, dtype=np.complex128
        ),
        block_mean_charge_class_conditional_pair_amplitude=np.asarray(
            conditional_pair_blocks, dtype=np.complex128
        ),
        target_charge_classes=np.asarray(charge_classes, dtype=np.float64),
        target_charge_class_counts=np.asarray(class_counts, dtype=np.int64),
        block_frame_bounds=np.asarray(
            ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25)),
            dtype=np.int64,
        ),
        charges=np.asarray(charges, dtype=np.float64),
        box_length=np.asarray(box_length, dtype=np.float64),
        force_scale=np.asarray(force_scale, dtype=np.float64),
        pilot_coordinate_prefix_sha256=np.asarray(pilot_prefix_sha256),
        pilot_force_prefix_sha256=np.asarray(force_prefix_sha256),
        alias_shell=np.asarray(alias_shell, dtype=np.int64),
        samples_per_shell=np.asarray(samples_per_shell, dtype=np.int64),
        candidate_catalog_sha256=np.asarray(catalog_sha256),
    )


def freeze_pilot_spectrum(args: argparse.Namespace) -> None:
    """Read frames 1--25 once and freeze all structural prediction inputs."""

    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames, coordinate_prefix = load_pilot_frames()
    charges = frames[0][1]
    box = frames[0][3]
    force_scale, force_prefix = frozen_force_scale()
    candidates = all_spectrum_candidates()
    catalog_sha = candidate_catalog_sha256(
        candidates, args.alias_shell, args.samples_per_shell
    )
    populations = []
    for index, candidate in enumerate(candidates, start=1):
        populations.append(
            prepare_population(
                candidate,
                charges,
                box,
                alias_shell=args.alias_shell,
                samples_per_shell=args.samples_per_shell,
            )
        )
        print(
            json.dumps(
                {
                    "stage": "prepare_pilot_spectrum_modes",
                    "candidate": index,
                    "candidate_count": len(candidates),
                    "id": candidate.case.case_id,
                }
            ),
            flush=True,
        )
    diagonal_modes, _ = adsq.population_mode_union(populations)
    coherent_mode_blocks = []
    for candidate, population in zip(candidates, populations):
        signed_base = np.concatenate(
            (population.base_modes, -population.base_modes), axis=0
        )
        coherent_mode_blocks.append(
            (
                signed_base[:, None, :]
                + candidate.mesh * adps.FACE_ALIASES[None, :, :]
            ).reshape(-1, 3)
        )
    modes = adps.canonical_mode_union(diagonal_modes, *coherent_mode_blocks)
    print(
        json.dumps(
            {
                "stage": "pilot_spectrum_mode_union",
                "mode_count": len(modes),
            }
        ),
        flush=True,
    )
    del populations, diagonal_modes, coherent_mode_blocks
    gc.collect()
    (
        tagged_mean,
        charge_mean,
        tagged_blocks,
        charge_blocks,
        conditional_mean,
        conditional_blocks,
        charge_classes,
        class_counts,
    ) = evaluate_spectral_theory_spectra(
        frames,
        modes,
        chunk_size=args.chunk_size,
        backend=args.structure_backend,
    )
    save_spectral_theory_spectra(
        PILOT_SPECTRUM,
        modes,
        tagged_mean,
        charge_mean,
        tagged_blocks,
        charge_blocks,
        conditional_mean,
        conditional_blocks,
        charge_classes,
        class_counts,
        charges,
        box,
        force_scale,
        coordinate_prefix,
        force_prefix,
        args.alias_shell,
        args.samples_per_shell,
        catalog_sha,
    )
    frozen = {
        "schema_version": 2,
        "purpose": "freeze Figure-5 structural inputs before AD prediction",
        "structure_definition": (
            "measured target-conditioned S_tag(q) and charge-class "
            "conditional pair amplitude mu_a(q)"
        ),
        "ordinary_s_q_saved_as_diagnostic": True,
        "coordinate_frames": "1--25",
        "coordinate_frame_count": PILOT_N,
        "coordinate_prefix_sha256": coordinate_prefix,
        "force_normalization_frames": "1--25",
        "force_prefix_sha256": force_prefix,
        "force_scale": force_scale,
        "candidate_catalog_sha256": catalog_sha,
        "candidate_count": len(candidates),
        "alias_shell": args.alias_shell,
        "samples_per_shell": args.samples_per_shell,
        "mode_count": len(modes),
        "target_charge_classes": charge_classes.tolist(),
        "target_charge_class_counts": class_counts.tolist(),
        "structure_transform_backend": args.structure_backend,
        "structure_transform_frequency_tile_width": (
            adps.FINUFFT_TILE_WIDTH if args.structure_backend == "finufft" else None
        ),
        "structure_transform_tiling_is_exact": True,
        "coherent_source_aliases": adps.FACE_ALIASES.tolist(),
        "spectrum_artifact": file_record(PILOT_SPECTRUM),
        "reference_force_accessed": False,
        "holdout_coordinates_accessed": False,
    }
    PILOT_SPECTRUM_FROZEN.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "logical_order": [
            "read only coordinate frames 1--25 with a prefix reader",
            "evaluate S_tag, ordinary S_q, and charge-class conditional pair amplitudes per frame",
            "average frames 1--25 and five contiguous five-frame blocks",
            "freeze spectrum, normalization, candidate catalog, and SHA-256",
            "allow later prediction to read only this frozen artifact",
        ],
        "inputs": {
            "trajectory": {
                "path": relative_path(TRAJECTORY),
                "frames": "1--25",
                "prefix_sha256": coordinate_prefix,
            },
            "force_normalization": {
                "path": relative_path(baseline.PILOT_FORCE_DUMP),
                "frames": "1--25",
                "prefix_sha256": force_prefix,
            },
            "runner": file_record(Path(__file__)),
            "spectral_descriptor": file_record(Path(adsq.__file__)),
            "pair_self_theory": file_record(Path(adps.__file__)),
            "ad_cell_moment_operator": file_record(Path(adref.__file__)),
            "ad_self_correction": file_record(Path(adcommon.__file__)),
            "theory_common": file_record(Path(baseline.__file__)),
        },
        "frozen_spectrum": file_record(PILOT_SPECTRUM_FROZEN),
        "spectrum_artifact": file_record(PILOT_SPECTRUM),
        "reference_force_accessed": False,
        "holdout_coordinates_accessed": False,
    }
    PILOT_SPECTRUM_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(frozen, indent=2))


def load_frozen_pilot_spectrum(
    *, alias_shell: int, samples_per_shell: int
) -> FrozenPilotSpectrum:
    """Verify and load the frozen spectrum without opening trajectory files."""

    for path in (
        PILOT_SPECTRUM,
        PILOT_SPECTRUM_FROZEN,
        PILOT_SPECTRUM_MANIFEST,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"run --freeze-pilot-spectrum before prediction: {path}"
            )
    frozen = json.loads(PILOT_SPECTRUM_FROZEN.read_text(encoding="utf-8"))
    manifest = json.loads(PILOT_SPECTRUM_MANIFEST.read_text(encoding="utf-8"))
    if int(frozen.get("schema_version", 0)) != 2:
        raise RuntimeError("pilot spectrum uses an obsolete schema; refreeze it")
    if manifest.get("frozen_spectrum") != file_record(PILOT_SPECTRUM_FROZEN):
        raise RuntimeError("pilot-spectrum frozen JSON fails manifest verification")
    if frozen.get("spectrum_artifact") != file_record(PILOT_SPECTRUM):
        raise RuntimeError("pilot-spectrum artifact fails frozen SHA-256 verification")
    if frozen.get("reference_force_accessed") is not False:
        raise RuntimeError("pilot spectrum accessed a reference-force result")
    if frozen.get("holdout_coordinates_accessed") is not False:
        raise RuntimeError("pilot spectrum accessed holdout coordinates")
    if int(frozen["alias_shell"]) != alias_shell:
        raise RuntimeError("prediction alias shell differs from frozen spectrum")
    if int(frozen["samples_per_shell"]) != samples_per_shell:
        raise RuntimeError("prediction alias sample count differs from frozen spectrum")
    expected_catalog = candidate_catalog_sha256(
        all_spectrum_candidates(), alias_shell, samples_per_shell
    )
    if frozen["candidate_catalog_sha256"] != expected_catalog:
        raise RuntimeError("candidate catalog differs from frozen pilot spectrum")

    with np.load(PILOT_SPECTRUM, allow_pickle=False) as data:
        result = FrozenPilotSpectrum(
            modes=np.asarray(data["modes"], dtype=np.int64),
            tagged_mean=np.asarray(
                data["mean_target_conditioned_s_tag"], dtype=np.float64
            ),
            charge_mean=np.asarray(data["mean_charge_s_q"], dtype=np.float64),
            tagged_blocks=np.asarray(
                data["block_mean_target_conditioned_s_tag"], dtype=np.float64
            ),
            charge_blocks=np.asarray(
                data["block_mean_charge_s_q"], dtype=np.float64
            ),
            conditional_pair_mean=np.asarray(
                data["mean_charge_class_conditional_pair_amplitude"],
                dtype=np.complex128,
            ),
            conditional_pair_blocks=np.asarray(
                data["block_mean_charge_class_conditional_pair_amplitude"],
                dtype=np.complex128,
            ),
            charge_classes=np.asarray(data["target_charge_classes"], dtype=np.float64),
            class_counts=np.asarray(
                data["target_charge_class_counts"], dtype=np.int64
            ),
            charges=np.asarray(data["charges"], dtype=np.float64),
            box_length=float(data["box_length"]),
            force_scale=float(data["force_scale"]),
            coordinate_prefix_sha256=str(data["pilot_coordinate_prefix_sha256"]),
            force_prefix_sha256=str(data["pilot_force_prefix_sha256"]),
            alias_shell=int(data["alias_shell"]),
            samples_per_shell=int(data["samples_per_shell"]),
            candidate_catalog_sha256=str(data["candidate_catalog_sha256"]),
            artifact_sha256=sha256(PILOT_SPECTRUM),
        )
    mode_count = len(result.modes)
    if result.tagged_mean.shape != (mode_count,):
        raise RuntimeError("frozen S_tag mean has an invalid shape")
    if result.charge_mean.shape != (mode_count,):
        raise RuntimeError("frozen S_q mean has an invalid shape")
    if result.tagged_blocks.shape != (len(PILOT_BLOCKS), mode_count):
        raise RuntimeError("frozen S_tag blocks have an invalid shape")
    if result.charge_blocks.shape != (len(PILOT_BLOCKS), mode_count):
        raise RuntimeError("frozen S_q blocks have an invalid shape")
    class_count = len(result.charge_classes)
    if result.class_counts.shape != (class_count,):
        raise RuntimeError("frozen target-class counts have an invalid shape")
    if result.conditional_pair_mean.shape != (class_count, mode_count):
        raise RuntimeError("frozen conditional pair mean has an invalid shape")
    if result.conditional_pair_blocks.shape != (
        len(PILOT_BLOCKS),
        class_count,
        mode_count,
    ):
        raise RuntimeError("frozen conditional pair blocks have an invalid shape")
    if not np.array_equal(
        np.asarray(
            [np.count_nonzero(result.charges == value) for value in result.charge_classes]
        ),
        result.class_counts,
    ):
        raise RuntimeError("frozen target charge classes disagree with charges")
    if result.coordinate_prefix_sha256 != frozen["coordinate_prefix_sha256"]:
        raise RuntimeError("coordinate-prefix hash differs inside spectrum artifact")
    if result.force_prefix_sha256 != frozen["force_prefix_sha256"]:
        raise RuntimeError("force-prefix hash differs inside spectrum artifact")
    return result


def mappings_for_frozen_modes(
    populations: list[adsq.ADSourceSpectrumPopulation], modes: np.ndarray
) -> list[dict[str, object]]:
    lookup = {tuple(mode): index for index, mode in enumerate(modes.tolist())}
    if len(lookup) != len(modes):
        raise RuntimeError("frozen pilot spectrum contains duplicate modes")
    mappings: list[dict[str, object]] = []
    for population in populations:
        try:
            base = np.asarray(
                [lookup[tuple(mode)] for mode in population.base_modes],
                dtype=np.int64,
            )
            sampled = {
                shell: np.asarray(
                    [lookup[tuple(mode)] for mode in shell_modes],
                    dtype=np.int64,
                )
                for shell, shell_modes in population.sampled_modes.items()
            }
        except KeyError as error:
            raise RuntimeError(
                "candidate requires a mode absent from the frozen pilot spectrum"
            ) from error
        mappings.append({"base": base, "sampled": sampled})
    return mappings


def evaluate_spectral_theory_screen(
    candidates: list[Candidate],
    *,
    alias_shell: int,
    samples_per_shell: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    if not candidates:
        raise ValueError("candidate list is empty")
    spectrum = load_frozen_pilot_spectrum(
        alias_shell=alias_shell, samples_per_shell=samples_per_shell
    )
    charges = spectrum.charges
    box = spectrum.box_length
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
    mappings = mappings_for_frozen_modes(
        [item.population for item in prepared], spectrum.modes
    )

    predictions: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    alias_rows: list[dict[str, object]] = []
    for index, (theory, mapping) in enumerate(zip(prepared, mappings), start=1):
        row, local_blocks, local_aliases = evaluate_spectral_theory(
            theory,
            mapping,
            charges,
            spectrum.modes,
            spectrum.tagged_mean,
            spectrum.charge_mean,
            spectrum.tagged_blocks,
            spectrum.conditional_pair_mean,
            spectrum.conditional_pair_blocks,
            spectrum.charge_classes,
            spectrum.class_counts,
            spectrum.force_scale,
            spectrum.coordinate_prefix_sha256,
            spectrum.artifact_sha256,
            box,
        )
        predictions.append(row)
        block_rows.extend(local_blocks)
        alias_rows.extend(local_aliases)
        print(
            json.dumps(
                {
                    "stage": "contract_ad_pair_self_theory",
                    "candidate": index,
                    "candidate_count": len(prepared),
                    "id": theory.candidate.case.case_id,
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
    return predictions, block_rows, alias_rows, {
        "elapsed_seconds": time.time() - started,
        "pilot_coordinate_prefix_sha256": spectrum.coordinate_prefix_sha256,
        "pilot_force_prefix_sha256": spectrum.force_prefix_sha256,
        "frozen_pilot_spectrum_sha256": spectrum.artifact_sha256,
        "pilot_coordinate_frames_read": 0,
        "pilot_force_frames_read": 0,
        "holdout_coordinate_frames_read": 0,
        "particlewise_force_difference_evaluations": 0,
        "structure_mode_count": len(spectrum.modes),
        "alias_shell": alias_shell,
        "samples_per_shell": samples_per_shell,
        "covariance_policy": (
            "charge-class coherent all-source pair/self-correction covariance "
            "retained before cell squaring; zero-mean pair fluctuations remain "
            "diagonal in physical mode; in-band/Fourier-tail covariance is ignored"
        ),
        "uncertainty_policy": (
            "quadrature of five contiguous pilot-spectrum block SEM and "
            "alias-importance-sampling SEM"
        ),
        "prediction_structural_file_access": "frozen pilot spectrum only",
    }


def prediction_manifest(
    purpose: str,
    predictions: Path,
    blocks: Path,
    aliases: Path,
    runtime: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 10,
        "purpose": purpose,
        "logical_order": [
            "verify the separately frozen pilot-spectrum artifact and SHA-256",
            "read no molecular coordinates or force arrays during prediction",
            "split frozen S_tag into charge-class coherent and zero-mean fluctuation parts",
            "contract the zero-mean fluctuation with exact homogeneous AD cell-moment weights",
            "form the complete coherent all-source pair field including i=j",
            "add the production self-correction field before cell squaring",
            "add the closed Fourier theory in quadrature with the joint in-band result",
            "combine five-block spectrum SEM and alias-sampling SEM in quadrature",
            "freeze the complete candidate table and its SHA-256",
            "permit holdout coordinate/reference access only in a later validation action",
        ],
        "prediction": {
            "reference_force_accessed": False,
            "holdout_coordinates_accessed": False,
            "molecular_coordinates_accessed": False,
            "unit_charge_force_accessed": False,
            "particlewise_force_difference_evaluated": False,
            "structural_file_access": "frozen pilot spectrum only",
            "structure_input": (
                "measured target-conditioned S_tag from frames 1--25"
            ),
            "coherent_structure_input": (
                "charge-class conditional pair amplitudes from frames 1--25"
            ),
            "alias_formula": (
                "exact homogeneous AD cell-moment weights reweighted by the "
                "zero-mean S_tag fluctuation; coherent mean uses zero plus "
                "six nearest source face aliases"
            ),
            "total_formula": (
                "sqrt(Delta_F_pair,fluctuation^2 + "
                "mean_cell(|F_pair,all,coherent + F_self,correction|^2) + "
                "Delta_F_Fourier^2)"
            ),
            "covariance_approximation": (
                "all-source pair/self-correction covariance retained for the "
                "charge-class conditional coherent field (equivalently, "
                "distinct-pair/residual-self covariance); zero-mean pair "
                "fluctuations remain diagonal in physical mode; coherent aliases "
                "beyond six faces and in-band/Fourier-tail covariance neglected"
            ),
            "frame_uncertainty": (
                "SEM from five frozen contiguous five-frame blocks of S_tag "
                "and conditional pair spectra"
            ),
            "alias_uncertainty": (
                "importance-sampling SEM of the finite-shell S_tag correction"
            ),
            "combined_uncertainty": (
                "quadrature of frame-block and alias-sampling SEM; their covariance is ignored"
            ),
            "upper_rule": (
                "prediction + t_0.95,4 * combined_SEM; t_0.95,4="
                f"{ONE_SIDED_T95_DF4:.14g}"
            ),
        },
        "candidate_count": len(read_csv(predictions)),
        "inputs": {
            "frozen_pilot_spectrum": file_record(PILOT_SPECTRUM),
            "frozen_pilot_spectrum_record": file_record(
                PILOT_SPECTRUM_FROZEN
            ),
            "frozen_pilot_spectrum_manifest": file_record(
                PILOT_SPECTRUM_MANIFEST
            ),
            "pilot_coordinate_prefix_sha256_provenance": runtime[
                "pilot_coordinate_prefix_sha256"
            ],
            "pilot_force_prefix_sha256_provenance": runtime[
                "pilot_force_prefix_sha256"
            ],
            "runner": file_record(Path(__file__)),
            "spectral_descriptor": file_record(Path(adsq.__file__)),
            "pair_self_theory": file_record(Path(adps.__file__)),
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
    )
    if len(predictions) != len(baseline.candidates()):
        raise RuntimeError("baseline spectral AD matrix is incomplete")
    write_csv(BASELINE_PREDICTION, predictions)
    write_csv(BASELINE_BLOCKS, blocks)
    write_csv(BASELINE_ALIASES, aliases)
    manifest = prediction_manifest(
        "Figure 5 AD curves from frozen conditional pair/self theory",
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


def stag_paths(target: float) -> dict[str, Path]:
    directory = OUTDIR / f"stag_{target_tag(target)}"
    return {
        "directory": directory,
        "prediction": directory / "prediction_before_validation.csv",
        "blocks": directory / "prediction_blocks.csv",
        "aliases": directory / "prediction_alias_shell.csv",
        "frozen": directory / "frozen_selection.json",
        "detail": directory / "holdout_validation_by_frame.csv",
        "summary": directory / "holdout_validation_summary.csv",
        "manifest": directory / "manifest.json",
        "diagnostic": directory / "diagnostic_direct_check_summary.csv",
        "diagnostic_frames": directory / "diagnostic_direct_check_by_frame.csv",
        "convergence": directory / "alias_shell_convergence.csv",
    }


def select_stag(rows: list[dict[str, object]]) -> dict[str, object]:
    passing = [row for row in rows if as_bool(row["selection_passes_target"])]
    if not passing:
        raise RuntimeError(
            "no declared conditional pair/self AD candidate satisfies the "
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


def write_stag_selection(target: float, args: argparse.Namespace) -> None:
    paths = stag_paths(target)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    predictions, blocks, aliases, runtime = evaluate_spectral_theory_screen(
        stag_candidates(target),
        alias_shell=args.alias_shell,
        samples_per_shell=args.samples_per_shell,
    )
    write_csv(paths["prediction"], predictions)
    write_csv(paths["blocks"], blocks)
    write_csv(paths["aliases"], aliases)
    selected = select_stag(predictions)
    prediction_sha = sha256(paths["prediction"])
    frozen = {
        "schema_version": 7,
        "purpose": (
            "conditional pair/self AD theory selection frozen before "
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
        "frozen_pilot_spectrum_sha256": runtime[
            "frozen_pilot_spectrum_sha256"
        ],
        "prediction_reference_force_accessed": False,
        "prediction_holdout_coordinates_accessed": False,
        "prediction_molecular_coordinates_accessed": False,
        "prediction_unit_charge_force_accessed": False,
        "prediction_particlewise_force_difference_evaluated": False,
        "prediction_structure_input": (
            "measured target-conditioned S_tag from frames 1--25"
        ),
        "prediction_coherent_structure_input": (
            "charge-class conditional pair amplitudes from frames 1--25"
        ),
        "selected": selected,
    }
    paths["frozen"].write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = prediction_manifest(
        "conditional pair/self AD theory screen with deferred holdout validation",
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
    paths = stag_paths(target)
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
    prediction_contract = manifest.get("prediction")
    if not isinstance(prediction_contract, dict):
        raise RuntimeError("the selection manifest has no prediction contract")
    for key in (
        "reference_force_accessed",
        "holdout_coordinates_accessed",
        "molecular_coordinates_accessed",
        "unit_charge_force_accessed",
        "particlewise_force_difference_evaluated",
    ):
        if prediction_contract.get(key) is not False:
            raise RuntimeError(f"the selection manifest violates its data boundary: {key}")
    if frozen["prediction_table_sha256"] != sha256(paths["prediction"]):
        raise RuntimeError("the frozen candidate table changed after selection")
    if bool(frozen["prediction_reference_force_accessed"]):
        raise RuntimeError("the frozen prediction accessed reference forces")
    if bool(frozen["prediction_holdout_coordinates_accessed"]):
        raise RuntimeError("the frozen prediction accessed holdout coordinates")
    if bool(frozen["prediction_molecular_coordinates_accessed"]):
        raise RuntimeError("the prediction stage reopened molecular coordinates")
    if bool(frozen["prediction_unit_charge_force_accessed"]):
        raise RuntimeError("the prediction stage read a unit-charge force result")
    if bool(frozen["prediction_particlewise_force_difference_evaluated"]):
        raise RuntimeError("the prediction stage evaluated a particlewise force difference")
    if frozen["frozen_pilot_spectrum_sha256"] != sha256(PILOT_SPECTRUM):
        raise RuntimeError("the frozen pilot spectrum changed after selection")
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


def validate_stag_selection(target: float, *, rerun_lammps: bool) -> None:
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
    """Compare the frozen S_tag theory with post-selection operator diagnostics."""

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
    joint_pair2 = []
    joint_self2 = []
    joint_cross = []
    joint_total2 = []
    for frame_index, (timestep, q, xyz, box) in enumerate(frames):
        moments = adjoint.evaluate_joint_pair_self_quadratic(
            q, xyz, op, coeff.real, correction
        )
        mesh_force = adref.fixed_ad_mesh_force(q, xyz, op, coeff.real)
        mesh_force -= correction_force(
            q, xyz, box, candidate.mesh, correction
        )
        direct = active_mode_force(q, xyz, box, modes, kernel)
        difference = mesh_force - direct
        direct2 = float(np.mean(np.sum(difference**2, axis=1)))
        joint_pair2.append(moments.distinct_pair_mean_square)
        joint_self2.append(moments.residual_self_mean_square)
        joint_cross.append(moments.pair_residual_self_dot_mean)
        joint_total2.append(moments.joint_pair_self_mean_square)
        rows.append(
            {
                "candidate_id": case.case_id,
                "pilot_frame": frame_index + 1,
                "timestep": timestep,
                "phase_resolved_distinct_pair_mean_square": (
                    moments.distinct_pair_mean_square
                ),
                "phase_resolved_residual_self_mean_square": (
                    moments.residual_self_mean_square
                ),
                "phase_resolved_pair_self_dot_mean": (
                    moments.pair_residual_self_dot_mean
                ),
                "phase_resolved_joint_mean_square": (
                    moments.joint_pair_self_mean_square
                ),
                "mesh_minus_direct_mean_square": direct2,
                "joint_to_direct_absolute_residual": abs(
                    moments.joint_pair_self_mean_square - direct2
                ),
                "pilot_coordinate_prefix_sha256": prefix_digest,
                "used_for_prediction": False,
                "used_for_selection": False,
            }
        )
    direct_rms = math.sqrt(
        float(np.mean([row["mesh_minus_direct_mean_square"] for row in rows]))
    )
    theory_pair_rms = float(selected["measured_stag_corrected_pair_rms"])
    theory_all_source_pair_rms = float(selected["all_source_pair_rms"])
    theory_self_rms = float(selected["residual_self_rms"])
    theory_self_correction_rms = float(selected["self_correction_rms"])
    theory_band_rms = float(selected["joint_mesh_rms"])
    phase_pair_rms = pooled_rms(joint_pair2)
    phase_self_rms = pooled_rms(joint_self2)
    phase_cross = math.fsum(joint_cross) / len(joint_cross)
    phase_joint_rms = pooled_rms(joint_total2)
    force_scale = float(selected["screening_force_scale"])
    summary = {
        "candidate_id": case.case_id,
        "target": target,
        "pilot_frames": "1--25",
        "stag_theory_pair_rms": theory_pair_rms,
        "stag_theory_all_source_pair_rms": theory_all_source_pair_rms,
        "stag_theory_residual_self_rms": theory_self_rms,
        "stag_theory_self_correction_rms": theory_self_correction_rms,
        "stag_theory_pair_self_joint_rms": theory_band_rms,
        "stag_theory_pair_self_dot_mean": float(selected["pair_self_dot_mean"]),
        "stag_theory_distinct_pair_residual_self_dot_mean": float(
            selected["distinct_pair_residual_self_dot_mean"]
        ),
        "phase_resolved_distinct_pair_rms": phase_pair_rms,
        "phase_resolved_residual_self_rms": phase_self_rms,
        "phase_resolved_pair_residual_self_dot_mean": phase_cross,
        "phase_resolved_joint_pair_self_rms": phase_joint_rms,
        "direct_finite_band_mesh_rms": direct_rms,
        "phase_resolved_joint_to_direct_ratio": phase_joint_rms / direct_rms,
        "stag_theory_to_phase_resolved_joint_ratio": (
            theory_band_rms / phase_joint_rms
        ),
        "direct_finite_band_relative_rms": direct_rms / force_scale,
        "stag_theory_total_relative_rms": float(
            selected["predicted_total_relative_rms"]
        ),
        "used_for_prediction": False,
        "used_for_selection": False,
        "diagnostic_definition": (
            "post-freeze comparison of conditional all-source pair plus "
            "self-correction spectral theory, its equivalent distinct-pair plus "
            "residual-self form, the phase-resolved joint operator, and the "
            "direct finite-band force difference"
        ),
    }
    write_csv(paths["diagnostic_frames"], rows)
    write_csv(paths["diagnostic"], [summary])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["joint_operator_and_direct_finite_band_diagnostic"] = {
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
    """Audit shell convergence of the frozen-S_tag main pair estimator."""
    frozen, paths = verify_frozen(target)
    selected = frozen["selected"]
    spectrum = load_frozen_pilot_spectrum(
        alias_shell=args.alias_shell,
        samples_per_shell=args.samples_per_shell,
    )
    charges = spectrum.charges
    box = spectrum.box_length
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
    mappings = mappings_for_frozen_modes(populations, spectrum.modes)
    force_scale = spectrum.force_scale
    coherent = adps.coherent_s_tag(
        charges,
        spectrum.charge_classes,
        spectrum.class_counts,
        spectrum.conditional_pair_mean,
    )
    fluctuation, _ = fluctuation_s_tag(spectrum.tagged_mean, coherent)
    rows = []
    for shell, population, mapping in zip(range(1, 6), populations, mappings):
        homogeneous, corrections, homogeneous_variance = (
            adsq.corrected_chi2_with_sampling(
                population, mapping, np.ones(len(spectrum.modes))
            )
        )
        if homogeneous != population.homogeneous_chi2:
            raise AssertionError("S_tag=1 did not exactly recover homogeneous chi2")
        if any(value != 0.0 for value in corrections.values()):
            raise AssertionError("S_tag=1 has a nonzero structure correction")
        if any(value != 0.0 for value in homogeneous_variance.values()):
            raise AssertionError("S_tag=1 has nonzero alias sampling variance")
        chi2, _, variances = adsq.corrected_chi2_with_sampling(
            population, mapping, fluctuation
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
                "spectral_fluctuation_pair_relative_rms": pair / force_scale,
                "alias_sampling_chi2_sem": math.sqrt(
                    math.fsum(variances.values())
                ),
                "homogeneous_recovery_exact": True,
            }
        )
    shell5 = float(rows[-1]["spectral_fluctuation_pair_relative_rms"])
    for row in rows:
        row["relative_difference_from_shell5"] = (
            float(row["spectral_fluctuation_pair_relative_rms"]) / shell5 - 1.0
        )
    write_csv(paths["convergence"], rows)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["alias_shell_convergence"] = {
        "performed_after_selection_freeze": True,
        "used_for_prediction_or_selection": False,
        "scope": "Figure-5 zero-mean S_tag fluctuation shell convergence",
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
    joint_structure = adps.evaluate_structure_spectra(
        q, xyz, 4.0, modes, backend="direct", chunk_size=2
    )
    joint_tagged_error = float(
        np.max(np.abs(joint_structure.tagged_pair - expected_tagged))
    )
    joint_ordinary_error = float(
        np.max(np.abs(joint_structure.ordinary_charge - expected_ordinary))
    )
    if joint_tagged_error > 2.0e-14 or joint_ordinary_error > 2.0e-14:
        raise AssertionError("joint-theory structure-spectrum test failed")
    far_modes = np.asarray(
        (
            (1, 0, 0),
            (127, -4, 3),
            (129, 2, -1),
            (-257, 145, 9),
            (399, -401, 255),
            (-511, -382, 290),
        ),
        dtype=np.int64,
    )
    tiled_direct = adps.evaluate_structure_spectra(
        q, xyz, 4.0, far_modes, backend="direct", chunk_size=2
    )
    tiled_finufft = adps.evaluate_structure_spectra(
        q, xyz, 4.0, far_modes, backend="finufft"
    )
    tiled_errors = {
        name: float(
            np.max(
                np.abs(
                    getattr(tiled_finufft, name) - getattr(tiled_direct, name)
                )
            )
        )
        for name in (
            "tagged_pair",
            "ordinary_charge",
            "conditional_pair_mean",
        )
    }
    if max(tiled_errors.values()) > 5.0e-11:
        raise AssertionError("frequency-tiled FINUFFT direct-sum test failed")
    coherent_part = adps.coherent_s_tag(
        q,
        joint_structure.charge_classes,
        joint_structure.class_counts,
        joint_structure.conditional_pair_mean,
    )
    structure_fluctuation, structure_minimum = fluctuation_s_tag(
        joint_structure.tagged_pair, coherent_part
    )
    if float(np.max(np.abs(structure_fluctuation))) > 3.0e-14:
        raise AssertionError(
            "one-target charge classes did not exhaust the tagged pair spectrum"
        )

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
    coherent_modes = adps.canonical_mode_union(
        union, adps.face_alias_modes(joint_operator)
    )
    zero_conditional = np.zeros(
        (len(joint_structure.charge_classes), len(coherent_modes)),
        dtype=np.complex128,
    )
    homogeneous_joint = adps.coherent_pair_self_cell_moments(
        q=q,
        frozen_modes=coherent_modes,
        conditional_pair_mean=zero_conditional,
        charge_classes=joint_structure.charge_classes,
        class_counts=joint_structure.class_counts,
        operator=joint_operator,
        real_coeff=coefficients(case).real,
        self_correction=joint_correction,
        quadrature_order_per_half=12,
    )
    if (
        float(homogeneous_joint.coherent_pair_mean_square) != 0.0
        or float(homogeneous_joint.pair_self_dot_mean) != 0.0
        or float(homogeneous_joint.coherent_joint_mean_square)
        != float(homogeneous_joint.residual_self_mean_square)
    ):
        raise AssertionError(
            "zero conditional mean did not recover homogeneous pair/self closure"
        )
    nonzero_structure = adps.evaluate_structure_spectra(
        q, xyz, 4.0, coherent_modes, backend="direct", chunk_size=17
    )
    nonzero_joint = adps.coherent_pair_self_cell_moments(
        q=q,
        frozen_modes=coherent_modes,
        conditional_pair_mean=nonzero_structure.conditional_pair_mean,
        charge_classes=nonzero_structure.charge_classes,
        class_counts=nonzero_structure.class_counts,
        operator=joint_operator,
        real_coeff=coefficients(case).real,
        self_correction=joint_correction,
        quadrature_order_per_half=12,
    )
    nonzero_identity = abs(
        float(nonzero_joint.coherent_joint_mean_square)
        - (
            float(nonzero_joint.coherent_pair_mean_square)
            + float(nonzero_joint.residual_self_mean_square)
            + 2.0 * float(nonzero_joint.pair_self_dot_mean)
        )
    )
    if nonzero_identity > 2.0e-12:
        raise AssertionError("nonzero pair/self vector-sum identity failed")
    nonzero_all_source_identity = abs(
        float(nonzero_joint.coherent_joint_mean_square)
        - (
            float(nonzero_joint.all_source_pair_mean_square)
            + float(nonzero_joint.self_correction_mean_square)
            + 2.0
            * float(nonzero_joint.all_source_pair_self_correction_dot_mean)
        )
    )
    if nonzero_all_source_identity > 2.0e-12:
        raise AssertionError("nonzero all-source pair/self identity failed")
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

    # Exercise the formal predictor with every coordinate/force and
    # force-difference entry point replaced by a failing sentinel.  This is
    # the executable data-boundary contract: only the frozen spectrum may be
    # opened after the freeze stage.
    this_module = sys.modules[__name__]
    guarded_entries = (
        (this_module, "load_pilot_frames"),
        (this_module, "frozen_force_scale"),
        (this_module, "active_mode_force"),
        (adref, "fixed_ad_mesh_force"),
        (ikref, "direct_truncated_force"),
        (ikref, "parse_charge_trajectory_prefix"),
        (ikref, "parse_force_dump_prefix"),
        (adjoint, "evaluate_joint_pair_self_quadratic"),
        (adps, "evaluate_structure_spectra"),
        (adcommon, "fit_self_correction"),
    )
    original_entries = [getattr(module, name) for module, name in guarded_entries]

    def forbidden_prediction_access(*_args, **_kwargs):
        raise AssertionError(
            "formal Figure-5 AD prediction crossed its frozen-spectrum boundary"
        )

    try:
        for module, name in guarded_entries:
            setattr(module, name, forbidden_prediction_access)
        guarded_rows, _, _, guarded_runtime = evaluate_spectral_theory_screen(
            [Candidate(target, 5, 12, "data_boundary_test")],
            alias_shell=5,
            samples_per_shell=2048,
        )
    finally:
        for (module, name), original in zip(guarded_entries, original_entries):
            setattr(module, name, original)
    if len(guarded_rows) != 1:
        raise AssertionError("frozen-spectrum boundary test returned the wrong row count")
    guarded_row = guarded_rows[0]
    if (
        guarded_runtime["pilot_coordinate_frames_read"] != 0
        or guarded_runtime["pilot_force_frames_read"] != 0
        or guarded_runtime["holdout_coordinate_frames_read"] != 0
        or guarded_runtime["particlewise_force_difference_evaluations"] != 0
        or bool(guarded_row["prediction_reference_force_accessed"])
        or bool(guarded_row["prediction_holdout_coordinates_accessed"])
        or bool(guarded_row["prediction_molecular_coordinates_accessed"])
        or bool(guarded_row["prediction_unit_charge_force_accessed"])
        or bool(guarded_row["prediction_particlewise_force_difference_evaluated"])
    ):
        raise AssertionError("formal Figure-5 AD prediction violated its data boundary")

    payload = {
        "schema_version": 5,
        "evaluate_tagged_pair_spectrum_direct_sum_max_abs": tagged_error,
        "ordinary_sq_direct_sum_max_abs": ordinary_error,
        "joint_structure_tagged_direct_sum_max_abs": joint_tagged_error,
        "joint_structure_ordinary_direct_sum_max_abs": joint_ordinary_error,
        "frequency_tiled_finufft_direct_sum_max_abs": tiled_errors,
        "frequency_tiled_finufft_tile_width": adps.FINUFFT_TILE_WIDTH,
        "conditional_decomposition_fluctuation_max_abs": float(
            np.max(np.abs(structure_fluctuation))
        ),
        "conditional_decomposition_minimum_before_clip": structure_minimum,
        "homogeneous_recovery_bitwise_exact": exact_homogeneous,
        "homogeneous_chi2": population.homogeneous_chi2,
        "homogeneous_zero_conditional_pair_mean_square": float(
            homogeneous_joint.coherent_pair_mean_square
        ),
        "homogeneous_zero_conditional_pair_self_dot_mean": float(
            homogeneous_joint.pair_self_dot_mean
        ),
        "homogeneous_joint_equals_residual_self_bitwise": (
            float(homogeneous_joint.coherent_joint_mean_square)
            == float(homogeneous_joint.residual_self_mean_square)
        ),
        "joint_production_real_projection_applied": (
            homogeneous_joint.production_real_projection_applied
        ),
        "nonzero_pair_self_dot_mean": float(
            nonzero_joint.pair_self_dot_mean
        ),
        "nonzero_pair_self_vector_sum_identity_max_abs": nonzero_identity,
        "nonzero_all_source_pair_self_vector_sum_identity_max_abs": (
            nonzero_all_source_identity
        ),
        "nonzero_equivalent_decomposition_component_max_abs": (
            nonzero_joint.equivalent_decomposition_component_max_abs
        ),
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
        "formal_prediction_frozen_spectrum_boundary_enforced": True,
        "formal_prediction_particlewise_force_difference_evaluations": 0,
        "passed": True,
    }
    UNIT_TESTS.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-pilot-spectrum", action="store_true")
    action.add_argument("--baseline", action="store_true")
    action.add_argument("--join-baseline-validation", action="store_true")
    action.add_argument("--select-target", type=float, metavar="TARGET")
    action.add_argument("--validate-selection", type=float, metavar="TARGET")
    action.add_argument("--diagnostic-direct-check", type=float, metavar="TARGET")
    action.add_argument(
        "--spectral-theory-audit",
        type=float,
        metavar="TARGET",
        help="post-freeze shell-convergence audit for the S_tag main estimator",
    )
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--alias-shell", type=int, default=5)
    parser.add_argument("--samples-per-shell", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--structure-backend",
        choices=("finufft", "direct"),
        default="finufft",
        help="pilot structural-transform backend; FINUFFT is used for production",
    )
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
    if args.freeze_pilot_spectrum:
        freeze_pilot_spectrum(args)
    elif args.baseline:
        write_baseline(args)
    elif args.join_baseline_validation:
        join_baseline_validation()
    elif args.select_target is not None:
        write_stag_selection(args.select_target, args)
    elif args.validate_selection is not None:
        validate_stag_selection(
            args.validate_selection, rerun_lammps=args.rerun_lammps
        )
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
