#!/usr/bin/env python3
"""Data-disjoint structure-aware SPC/E-water Fourier-tail diagnostic.

The measured quantity is the direct PSWF smooth-force cube truncation, with no
particle mesh.  Twenty-five pilot frames provide vector-resolved S_q(m) on
the low omitted modes; an orientation-averaged intramolecular SPC/E form
factor closes the higher modes.  Ten later trajectory frames are used only
for the force measurement.  Their infinite all-charge smooth force is the
direct conducting-periodic Coulomb Ewald force minus the exact compact PSWF
near force, so the production special-bonds convention never enters.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

import fixed_ik_reference as ref
from fig2_fourier_reference import (
    ExactPSWFContinuation,
    direct_force_from_kernel,
    direct_periodic_coulomb_ewald_force,
    discrete_eq46_sum,
    eq46_force_error,
    lattice_shell_counts,
    pooled_rms_jackknife_sem,
    rms_vector_error,
    symmetric_kernel_grid,
)
import sq_alias_tools as sqtools


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
TRAJECTORY = (
    PROJECT_ROOT
    / "numerical_examples"
    / "water_trajectory_benchmark"
    / "water_short_traj.lammpstrj"
)
RCUT = 9.0
RECIPROCAL_MESH = 21
CSPLITS = (12.024, 14.471, 16.894)
QUADRATURE_ORDER = 768
PILOT_INDICES = tuple(range(25))
HOLDOUT_INDICES = tuple(range(25, 35))
SQ_EXPLICIT_RADIUS = 20
SQ_CONVERGENCE_RADII = (14, 16, 18, 20)
TAIL_RADIUS = 320
TAIL_CONVERGENCE_RADII = (240, 320)
EWALD_PRODUCTION = dict(alpha=0.35, real_cutoff=18.0, reciprocal_mesh=41)
EWALD_CROSSCHECKS = (
    dict(alpha=0.30, real_cutoff=20.0, reciprocal_mesh=37),
    dict(alpha=0.40, real_cutoff=16.0, reciprocal_mesh=47),
)

SUMMARY = HERE / "fig2_water_fourier_tail_summary.csv"
BY_FRAME = HERE / "fig2_water_fourier_tail_by_frame.csv"
PILOT_MODES = HERE / "fig2_water_fourier_tail_pilot_modes.csv"
SQ_CONVERGENCE = HERE / "fig2_water_fourier_tail_sq_convergence.csv"
TAIL_CONVERGENCE = HERE / "fig2_water_fourier_tail_radial_convergence.csv"
REFERENCE_CHECKS = HERE / "fig2_water_fourier_tail_reference_checks.csv"
REFERENCE_NPZ = HERE / "fig2_water_all_charge_coulomb_reference.npz"
MANIFEST = HERE / "fig2_water_fourier_tail_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sinc(values: np.ndarray) -> np.ndarray:
    """Return sin(x)/x with the continuous value at zero."""

    return np.sinc(np.asarray(values, dtype=np.float64) / math.pi)


def minimum_image(displacement: np.ndarray, box_length: float) -> np.ndarray:
    return displacement - box_length * np.rint(displacement / box_length)


def molecular_geometry(
    frames: list[tuple[int, np.ndarray, np.ndarray, float]],
) -> dict[str, object]:
    distances = {"OH1": [], "OH2": [], "HH": []}
    for _, q, xyz, box_length in frames:
        if len(q) % 3 or not np.allclose(q.reshape(-1, 3)[:, 0], -0.8476):
            raise ValueError("water atoms are not ordered O,H,H by molecule")
        molecules = xyz.reshape(-1, 3, 3)
        vectors = {
            "OH1": minimum_image(molecules[:, 1] - molecules[:, 0], box_length),
            "OH2": minimum_image(molecules[:, 2] - molecules[:, 0], box_length),
            "HH": minimum_image(molecules[:, 2] - molecules[:, 1], box_length),
        }
        for key, vector in vectors.items():
            distances[key].extend(np.linalg.norm(vector, axis=1).tolist())
    result: dict[str, object] = {}
    for key, values in distances.items():
        array = np.asarray(values)
        result[key] = {
            "mean_A": float(np.mean(array)),
            "sample_sd_A": float(np.std(array, ddof=1)),
            "minimum_A": float(np.min(array)),
            "maximum_A": float(np.max(array)),
        }
    return result


def intramolecular_sq(k: np.ndarray, geometry: dict[str, object]) -> np.ndarray:
    """Orientation-averaged rigid-molecule charge form factor / Q_mol."""

    q_o = -0.8476
    q_h = 0.4238
    q_mol = q_o * q_o + 2.0 * q_h * q_h
    r_oh1 = float(geometry["OH1"]["mean_A"])  # type: ignore[index]
    r_oh2 = float(geometry["OH2"]["mean_A"])  # type: ignore[index]
    r_hh = float(geometry["HH"]["mean_A"])  # type: ignore[index]
    values = 1.0 + 2.0 * (
        q_o * q_h * sinc(k * r_oh1)
        + q_o * q_h * sinc(k * r_oh2)
        + q_h * q_h * sinc(k * r_hh)
    ) / q_mol
    if np.any(values < -1.0e-12):
        raise FloatingPointError("negative orientation-averaged molecular spectrum")
    return np.maximum(values, 0.0)


def canonical_omitted_modes(radius: int, half_width: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(-radius, radius + 1, dtype=np.int64)
    nx, ny, nz = np.meshgrid(axis, axis, axis, indexing="ij")
    modes = np.column_stack([nx.ravel(), ny.ravel(), nz.ravel()])
    squared = np.einsum("ij,ij->i", modes, modes)
    keep = (
        (squared > 0)
        & (squared <= radius * radius)
        & (np.max(np.abs(modes), axis=1) > half_width)
    )
    full_count = int(np.count_nonzero(keep))
    canonical = np.unique(sqtools.canonical_modes(modes[keep]), axis=0)
    if 2 * len(canonical) != full_count:
        raise RuntimeError("omitted reciprocal population is not inversion paired")
    squared_canonical = np.einsum("ij,ij->i", canonical, canonical)
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0], squared_canonical))
    return canonical[order], squared_canonical[order]


def mode_weights(
    pswf: ExactPSWFContinuation,
    squared_modes: np.ndarray,
    box_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    k = 2.0 * math.pi / box_length * np.sqrt(squared_modes.astype(np.float64))
    attenuation = pswf.attenuation(RCUT * k / pswf.c)
    weights = 16.0 * math.pi**2 * attenuation**2 / k**2
    return k, weights


def inside_cube_counts(half_width: int, maximum_squared: int) -> np.ndarray:
    axis = np.arange(-half_width, half_width + 1, dtype=np.int64)
    nx, ny, nz = np.meshgrid(axis, axis, axis, indexing="ij")
    squared = (nx * nx + ny * ny + nz * nz).ravel()
    return np.bincount(squared, minlength=maximum_squared + 1)[: maximum_squared + 1]


def radial_mode_sums(
    pswf: ExactPSWFContinuation,
    box_length: float,
    geometry: dict[str, object],
    shell_counts: np.ndarray,
    tail_radius: int,
) -> tuple[float, float, dict[str, object]]:
    homogeneous, metadata = discrete_eq46_sum(
        pswf,
        box_length,
        RCUT,
        RECIPROCAL_MESH,
        shell_counts,
        tail_radius,
    )
    s = np.arange(1, tail_radius * tail_radius + 1, dtype=np.int64)
    k = 2.0 * math.pi / box_length * np.sqrt(s.astype(np.float64))
    attenuation = pswf.attenuation(RCUT * k / pswf.c)
    weights = 16.0 * math.pi**2 * attenuation**2 / k**2
    half_width = RECIPROCAL_MESH // 2
    retained = inside_cube_counts(half_width, tail_radius * tail_radius)
    omitted = shell_counts[: tail_radius * tail_radius + 1] - retained
    if np.any(omitted < 0):
        raise RuntimeError("negative omitted radial degeneracy")
    explicit_intramolecular = float(
        np.dot(omitted[1:], weights * intramolecular_sq(k, geometry))
    )
    # At k >= 2*pi*(R+1/2)/L the molecular sinc terms are oscillatory and
    # vanish as 1/k.  The existing continuum closure therefore uses S=1;
    # R=240/320 convergence quantifies the residual closure sensitivity.
    intramolecular = explicit_intramolecular + float(
        metadata["asymptotic_continuum_tail"]
    )
    return homogeneous, intramolecular, {
        **metadata,
        "explicit_intramolecular_sum": explicit_intramolecular,
        "far_tail_structure_factor": "unity high-k limit",
    }


def pilot_prediction_sem(frame_mode_sums: np.ndarray, scale: float) -> float:
    if len(frame_mode_sums) != 25:
        raise ValueError("pilot block SEM is defined for 25 frames")
    blocks = frame_mode_sums.reshape(5, 5).mean(axis=1)
    predictions = scale * np.sqrt(blocks)
    return float(np.std(predictions, ddof=1) / math.sqrt(len(predictions)))


def measure_holdout_frame(
    payload: tuple[int, int, np.ndarray, np.ndarray, float]
) -> dict[str, object]:
    frame_index, timestep, q, xyz, box_length = payload
    coulomb, ewald_metadata = direct_periodic_coulomb_ewald_force(
        q, xyz, box_length, **EWALD_PRODUCTION
    )
    cases: list[dict[str, object]] = []
    for csplit in CSPLITS:
        pswf = ExactPSWFContinuation(csplit, QUADRATURE_ORDER)
        near = pswf.compact_near_force(q, xyz, box_length, RCUT)
        infinite = coulomb - near
        kernel = symmetric_kernel_grid(
            pswf, RECIPROCAL_MESH, box_length, RCUT
        )
        truncated = direct_force_from_kernel(q, xyz, box_length, kernel)
        difference = infinite - truncated
        component_rms = np.sqrt(np.mean(difference * difference, axis=0))
        cases.append(
            {
                "c_split": csplit,
                "measured_abs_rms_kcal_per_mol_A": float(
                    np.sqrt(np.sum(component_rms * component_rms))
                ),
                "measured_x_rms_kcal_per_mol_A": float(component_rms[0]),
                "measured_y_rms_kcal_per_mol_A": float(component_rms[1]),
                "measured_z_rms_kcal_per_mol_A": float(component_rms[2]),
                "smooth_infinite_rms_kcal_per_mol_A": float(
                    np.sqrt(np.mean(np.sum(infinite * infinite, axis=1)))
                ),
                "smooth_truncated_rms_kcal_per_mol_A": float(
                    np.sqrt(np.mean(np.sum(truncated * truncated, axis=1)))
                ),
            }
        )
    return {
        "frame_index": frame_index,
        "timestep": timestep,
        "coulomb": coulomb,
        "ewald_metadata": ewald_metadata,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
        help="Parallel backend for independent holdout frames.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    started = time.time()
    frames = ref.parse_charge_trajectory(TRAJECTORY)
    if len(frames) != 50:
        raise RuntimeError(f"expected 50 water frames, found {len(frames)}")
    q0 = frames[0][1]
    box_length = frames[0][3]
    if (
        len(q0) != 2703
        or not math.isclose(box_length, 30.0)
        or not math.isclose(float(np.sum(q0)), 0.0, abs_tol=1.0e-12)
    ):
        raise RuntimeError("unexpected SPC/E trajectory metadata")
    for _, q, _, frame_box in frames:
        if not np.array_equal(q, q0) or frame_box != box_length:
            raise RuntimeError("water charges or box change across the trajectory")
    natoms = len(q0)
    qsum = float(np.sum(q0 * q0))
    volume = box_length**3
    scale = ref.COULOMB_REAL * qsum / (math.sqrt(natoms) * volume)
    geometry = molecular_geometry([frames[index] for index in PILOT_INDICES])

    half_width = RECIPROCAL_MESH // 2
    modes, squared_modes = canonical_omitted_modes(
        SQ_EXPLICIT_RADIUS, half_width
    )
    pilot_sq = np.empty((len(PILOT_INDICES), len(modes)), dtype=np.float64)
    for local_index, frame_index in enumerate(PILOT_INDICES):
        _, q, xyz, frame_box = frames[frame_index]
        pilot_sq[local_index] = sqtools.evaluate_sq_modes(
            q, xyz, frame_box, modes
        )
        print(
            json.dumps(
                {
                    "stage": "pilot_sq",
                    "frame": frame_index,
                    "modes": len(modes),
                    "elapsed_s": time.time() - started,
                }
            ),
            flush=True,
        )
    pilot_sq_mean = np.mean(pilot_sq, axis=0)
    pilot_block_means = pilot_sq.reshape(5, 5, len(modes)).mean(axis=1)
    pilot_sq_block_sem = np.std(pilot_block_means, axis=0, ddof=1) / math.sqrt(5)

    shell_counts = lattice_shell_counts(TAIL_RADIUS)
    pswfs = {
        csplit: ExactPSWFContinuation(csplit, QUADRATURE_ORDER)
        for csplit in CSPLITS
    }
    prediction_records: dict[float, dict[str, object]] = {}
    sq_convergence_rows: list[dict[str, object]] = []
    tail_convergence_rows: list[dict[str, object]] = []
    for csplit, pswf in pswfs.items():
        k_modes, weights = mode_weights(pswf, squared_modes, box_length)
        molecular_modes = intramolecular_sq(k_modes, geometry)
        homogeneous, intramolecular, tail_metadata = radial_mode_sums(
            pswf, box_length, geometry, shell_counts, TAIL_RADIUS
        )
        mode_difference = pilot_sq - molecular_modes[None, :]
        frame_mode_sums = intramolecular + 2.0 * (mode_difference @ weights)
        if np.any(frame_mode_sums <= 0.0):
            raise FloatingPointError("non-positive pilot structure-aware mode sum")
        corrected_mode_sum = float(np.mean(frame_mode_sums))
        corrected_prediction = scale * math.sqrt(corrected_mode_sum)
        prediction_records[csplit] = {
            "homogeneous_mode_sum": homogeneous,
            "intramolecular_mode_sum": intramolecular,
            "pilot_corrected_mode_sum": corrected_mode_sum,
            "homogeneous_prediction": scale * math.sqrt(homogeneous),
            "intramolecular_prediction": scale * math.sqrt(intramolecular),
            "pilot_corrected_prediction": corrected_prediction,
            "pilot_corrected_block5_sem": pilot_prediction_sem(
                frame_mode_sums, scale
            ),
            "tail_metadata": tail_metadata,
        }
        for radius in SQ_CONVERGENCE_RADII:
            mask = squared_modes <= radius * radius
            corrected_at_radius = intramolecular + 2.0 * float(
                np.dot(weights[mask], pilot_sq_mean[mask] - molecular_modes[mask])
            )
            sq_convergence_rows.append(
                {
                    "c_split": csplit,
                    "pilot_sq_explicit_radius": radius,
                    "canonical_mode_count": int(np.count_nonzero(mask)),
                    "prediction_kcal_per_mol_A": scale
                    * math.sqrt(corrected_at_radius),
                    "relative_to_radius20": math.sqrt(
                        corrected_at_radius / corrected_mode_sum
                    ),
                    "outer_closure": "orientation-averaged intramolecular SPC/E form factor",
                }
            )
        for radius in TAIL_CONVERGENCE_RADII:
            homogeneous_r, intra_r, metadata_r = radial_mode_sums(
                pswf, box_length, geometry, shell_counts, radius
            )
            corrected_r = intra_r + 2.0 * float(
                np.dot(weights, pilot_sq_mean - molecular_modes)
            )
            tail_convergence_rows.append(
                {
                    "c_split": csplit,
                    "radial_tail_radius": radius,
                    "homogeneous_prediction_kcal_per_mol_A": scale
                    * math.sqrt(homogeneous_r),
                    "intramolecular_prediction_kcal_per_mol_A": scale
                    * math.sqrt(intra_r),
                    "pilot_corrected_prediction_kcal_per_mol_A": scale
                    * math.sqrt(corrected_r),
                    "continuum_tail_fraction_homogeneous": metadata_r[
                        "tail_fraction_of_omitted_sum"
                    ],
                    "far_tail_structure_factor": "unity high-k limit",
                }
            )

    mode_rows: list[dict[str, object]] = []
    k_all = 2.0 * math.pi / box_length * np.sqrt(squared_modes)
    molecular_all = intramolecular_sq(k_all, geometry)
    for mode, squared, k, mean_sq, sem_sq, molecular_sq in zip(
        modes,
        squared_modes,
        k_all,
        pilot_sq_mean,
        pilot_sq_block_sem,
        molecular_all,
    ):
        mode_rows.append(
            {
                "mx": int(mode[0]),
                "my": int(mode[1]),
                "mz": int(mode[2]),
                "integer_radius_squared": int(squared),
                "k_inverse_A": float(k),
                "pilot_sq_mean": float(mean_sq),
                "pilot_sq_block5_sem": float(sem_sq),
                "intramolecular_sq_closure": float(molecular_sq),
                "normalization": "S_q=<|rho(k)|^2>/Q",
                "inversion_multiplicity": 2,
            }
        )

    payloads = [
        (index, frames[index][0], frames[index][1], frames[index][2], frames[index][3])
        for index in HOLDOUT_INDICES
    ]
    if args.workers == 1:
        holdout_results = [measure_holdout_frame(payload) for payload in payloads]
    else:
        executor_class = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
        with executor_class(max_workers=args.workers) as executor:
            holdout_results = list(executor.map(measure_holdout_frame, payloads))
    holdout_results.sort(key=lambda result: int(result["frame_index"]))
    print(
        json.dumps(
            {
                "stage": "holdout_complete",
                "frames": len(holdout_results),
                "elapsed_s": time.time() - started,
            }
        ),
        flush=True,
    )

    # Independent all-charge Ewald decomposition checks on the first holdout.
    first_index = HOLDOUT_INDICES[0]
    first_timestep, first_q, first_xyz, first_box = frames[first_index]
    production_reference = holdout_results[0]["coulomb"]
    reference_rows: list[dict[str, object]] = []
    for parameters in (EWALD_PRODUCTION,) + EWALD_CROSSCHECKS:
        if parameters == EWALD_PRODUCTION:
            force = production_reference
            metadata = holdout_results[0]["ewald_metadata"]
        else:
            force, metadata = direct_periodic_coulomb_ewald_force(
                first_q, first_xyz, first_box, **parameters
            )
        reference_rows.append(
            {
                "frame_index": first_index,
                "timestep": first_timestep,
                **parameters,
                "rms_difference_from_production": rms_vector_error(
                    np.asarray(force), np.asarray(production_reference)
                ),
                "real_screen_at_cutoff": metadata["real_screen_at_cutoff"],
                "reciprocal_screen_at_face": metadata[
                    "reciprocal_screen_at_face"
                ],
                "operator": metadata["operator"],
                "charge_convention": "all Coulomb pairs; no special-bonds scaling",
            }
        )

    coulomb_array = np.stack(
        [np.asarray(result["coulomb"]) for result in holdout_results]
    )
    np.savez_compressed(
        REFERENCE_NPZ,
        frame_indices=np.asarray(HOLDOUT_INDICES, dtype=np.int64),
        timesteps=np.asarray(
            [int(result["timestep"]) for result in holdout_results],
            dtype=np.int64,
        ),
        q=q0,
        coulomb_forces=coulomb_array,
        operator=np.asarray(
            "direct symmetric conducting-boundary Ewald sums; all charge pairs"
        ),
    )

    by_frame_rows: list[dict[str, object]] = []
    errors_by_c: dict[float, list[float]] = {csplit: [] for csplit in CSPLITS}
    for result in holdout_results:
        for case in result["cases"]:  # type: ignore[assignment]
            csplit = float(case["c_split"])
            errors_by_c[csplit].append(
                float(case["measured_abs_rms_kcal_per_mol_A"])
            )
            by_frame_rows.append(
                {
                    "frame_index": result["frame_index"],
                    "timestep": result["timestep"],
                    "partition": "holdout",
                    **case,
                    "reciprocal_mesh": RECIPROCAL_MESH,
                    "finite_operator": "direct exact-PSWF Fourier sum on symmetric I_M",
                    "infinite_operator": "direct all-charge conducting-periodic Ewald minus exact compact PSWF near force",
                    "special_bonds_present": False,
                    "particle_mesh_present": False,
                }
            )

    summary_rows: list[dict[str, object]] = []
    for csplit in CSPLITS:
        errors = np.asarray(errors_by_c[csplit])
        measured = float(np.sqrt(np.mean(errors * errors)))
        prediction = prediction_records[csplit]
        summary_rows.append(
            {
                "system": "SPC/E water, all-charge Fourier-tail diagnostic",
                "c_split": csplit,
                "r_c_A": RCUT,
                "box_length_A": box_length,
                "reciprocal_mesh": RECIPROCAL_MESH,
                "reciprocal_half_width": half_width,
                "Kmax_inverse_A": 2.0 * math.pi * half_width / box_length,
                "n_atoms": natoms,
                "Q_sum_q_squared": qsum,
                "pilot_frames": len(PILOT_INDICES),
                "holdout_frames": len(HOLDOUT_INDICES),
                "measured_holdout_pooled_abs_rms_kcal_per_mol_A": measured,
                "measured_holdout_jackknife_sem_kcal_per_mol_A": pooled_rms_jackknife_sem(
                    errors
                ),
                "homogeneous_prediction_kcal_per_mol_A": prediction[
                    "homogeneous_prediction"
                ],
                "intramolecular_prediction_kcal_per_mol_A": prediction[
                    "intramolecular_prediction"
                ],
                "pilot_sq_corrected_prediction_kcal_per_mol_A": prediction[
                    "pilot_corrected_prediction"
                ],
                "pilot_sq_corrected_block5_sem_kcal_per_mol_A": prediction[
                    "pilot_corrected_block5_sem"
                ],
                "homogeneous_over_measured": float(
                    prediction["homogeneous_prediction"]
                )
                / measured,
                "intramolecular_over_measured": float(
                    prediction["intramolecular_prediction"]
                )
                / measured,
                "pilot_sq_corrected_over_measured": float(
                    prediction["pilot_corrected_prediction"]
                )
                / measured,
                "pilot_sq_explicit_radius": SQ_EXPLICIT_RADIUS,
                "radial_tail_radius": TAIL_RADIUS,
                "data_disjoint": True,
                "structure_model": "vector-resolved pilot S_q through integer radius 20; orientation-averaged intramolecular SPC/E form factor outside; unity continuum high-k limit",
                "estimator_scope": "diagonal-mode structure-aware diagnostic; cross-mode and target-conditioned correlations omitted",
            }
        )

    write_csv(SUMMARY, summary_rows)
    write_csv(BY_FRAME, by_frame_rows)
    write_csv(PILOT_MODES, mode_rows)
    write_csv(SQ_CONVERGENCE, sq_convergence_rows)
    write_csv(TAIL_CONVERGENCE, tail_convergence_rows)
    write_csv(REFERENCE_CHECKS, reference_rows)
    outputs = (
        SUMMARY,
        BY_FRAME,
        PILOT_MODES,
        SQ_CONVERGENCE,
        TAIL_CONVERGENCE,
        REFERENCE_CHECKS,
        REFERENCE_NPZ,
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": "Does a pilot-only vector-resolved S_q correction predict direct SPC/E-water Fourier cube truncation on data-disjoint frames?",
        "operator_chain": {
            "finite": "direct exact-PSWF particle-mode sum on the M=21 symmetric cube",
            "infinite": "direct all-charge conducting-periodic Ewald minus exact compact PSWF near force",
            "particle_mesh": False,
            "special_bonds": False,
            "reason_new_reference_is_required": "the existing production Ewald dump uses special_bonds lj/coul 0 0 0.5 and is not the all-charge Coulomb operator in the splitting identity",
        },
        "trajectory": file_record(TRAJECTORY),
        "pilot_indices_zero_based": list(PILOT_INDICES),
        "holdout_indices_zero_based": list(HOLDOUT_INDICES),
        "unused_indices_zero_based": [
            index
            for index in range(len(frames))
            if index not in PILOT_INDICES and index not in HOLDOUT_INDICES
        ],
        "data_disjoint": set(PILOT_INDICES).isdisjoint(HOLDOUT_INDICES),
        "system": {
            "n_atoms": natoms,
            "n_molecules": natoms // 3,
            "box_length_A": box_length,
            "Q_sum_q_squared": qsum,
            "molecular_geometry_from_pilot": geometry,
        },
        "parameters": {
            "r_c_A": RCUT,
            "c_split": list(CSPLITS),
            "reciprocal_mesh": RECIPROCAL_MESH,
            "pilot_sq_explicit_radius": SQ_EXPLICIT_RADIUS,
            "pilot_sq_convergence_radii": list(SQ_CONVERGENCE_RADII),
            "radial_tail_radius": TAIL_RADIUS,
            "radial_tail_convergence_radii": list(TAIL_CONVERGENCE_RADII),
            "ewald_production": EWALD_PRODUCTION,
            "ewald_crosschecks": list(EWALD_CROSSCHECKS),
            "workers": args.workers,
            "executor": args.executor,
        },
        "structure_aware_estimator": {
            "normalization": "S_q(k)=<|rho(k)|^2>/Q",
            "low_omitted_modes": "vector-resolved mean over 25 pilot frames",
            "outer_modes": "orientation-averaged SPC/E intramolecular form factor",
            "continuum_tail": "S_q -> 1 high-k limit",
            "limitations": "diagonal modal power only; cross-mode covariance and target-conditioned correlations are not restored",
        },
        "reference_crosscheck_max_rms_kcal_per_mol_A": max(
            float(row["rms_difference_from_production"])
            for row in reference_rows
        ),
        "outputs": [file_record(path) for path in outputs],
        "code": [
            file_record(Path(__file__)),
            file_record(HERE / "fig2_fourier_reference.py"),
            file_record(HERE / "sq_alias_tools.py"),
            file_record(HERE / "eval_pswf_profile.cpp"),
        ],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_seconds": time.time() - started,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("c_split measured homogeneous intramolecular pilot-Sq corrected/measured")
    for row in summary_rows:
        print(
            f"{float(row['c_split']):8.4f} "
            f"{float(row['measured_holdout_pooled_abs_rms_kcal_per_mol_A']):.9e} "
            f"{float(row['homogeneous_prediction_kcal_per_mol_A']):.9e} "
            f"{float(row['intramolecular_prediction_kcal_per_mol_A']):.9e} "
            f"{float(row['pilot_sq_corrected_over_measured']):.6f}"
        )


if __name__ == "__main__":
    main()
