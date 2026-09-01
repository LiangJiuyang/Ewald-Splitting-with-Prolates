#!/usr/bin/env python3
"""Generate charge-spectrum and operator-matched water data for Fig. 4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

import fixed_ik_reference as ref
import sq_alias_tools as sqtools


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
TRAJECTORY = PROJECT / "numerical_examples" / "water_trajectory_benchmark" / "water_short_traj.lammpstrj"
RANDOM_ROOT = PROJECT / "numerical_examples" / "random_charges"

CORRECTION_OUT = HERE / "fig4_sq_correction_source.csv"
BY_FRAME_OUT = HERE / "fig4_sq_correction_by_frame.csv"
SHELL_OUT = HERE / "fig4_sq_alias_shell_source.csv"
MANIFEST = HERE / "fig4_sq_correction_manifest.json"

RCUT = 9.0
CSPLIT = 16.894
CSPREAD = 13.251
MESH = 18
SPLIT_INPUT_TOL = 1.0e-6
SPREAD_INPUT_TOL = 3.0e-5
ORDERS = (4, 5, 6, 7, 8)
PILOT_N = 25
MAX_ALIAS_SHELL = 12
TAIL_CHECK_SHELL = 20
OUTER_SAMPLES = 4096
SEED = 20260808
Q_O = -0.8476
Q_H = 0.4238
D_OH = 1.0
HOH_ANGLE_DEG = 109.47
D_HH = math.sqrt(2.0 - 2.0 * math.cos(math.radians(HOH_ANGLE_DEG)))
Q_MOL = Q_O * Q_O + 2.0 * Q_H * Q_H
INTRA_OH_COEFFICIENT = 4.0 * Q_O * Q_H / Q_MOL
INTRA_HH_COEFFICIENT = 2.0 * Q_H * Q_H / Q_MOL


def rigid_spce_sq(k: np.ndarray) -> np.ndarray:
    """Orientationally averaged rigid-SPC/E intramolecular weight."""
    values = (
        1.0
        + INTRA_OH_COEFFICIENT * np.sinc(k * D_OH / math.pi)
        + INTRA_HH_COEFFICIENT * np.sinc(k * D_HH / math.pi)
    )
    if np.any(values < -1.0e-12):
        raise FloatingPointError("negative rigid-SPC/E correlation weight")
    return np.maximum(values, 0.0)


def block_sem(values: np.ndarray, block_size: int = 5) -> float:
    count = len(values) // block_size
    if count < 2:
        return float("nan")
    means = np.asarray([values[i * block_size : (i + 1) * block_size].mean() for i in range(count)])
    return float(means.std(ddof=1) / math.sqrt(count))


def pooled_rms_block_sem(values: np.ndarray, block_size: int = 5) -> float:
    """SEM of nonoverlapping block-level pooled RMS values."""
    count = len(values) // block_size
    if count < 2:
        return float("nan")
    block_rms = np.asarray(
        [
            math.sqrt(float(np.mean(values[i * block_size : (i + 1) * block_size] ** 2)))
            for i in range(count)
        ]
    )
    return float(block_rms.std(ddof=1) / math.sqrt(count))


def chi_rms_block_sem(chi: np.ndarray, scale: float, block_size: int = 5) -> float:
    """SEM of block predictions scale*sqrt(mean(chi within block))."""
    count = len(chi) // block_size
    if count < 2:
        return float("nan")
    block_predictions = np.asarray(
        [
            scale
            * math.sqrt(float(np.mean(chi[i * block_size : (i + 1) * block_size])))
            for i in range(count)
        ]
    )
    return float(block_predictions.std(ddof=1) / math.sqrt(count))


def file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(PROJECT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main():
    started = time.time()
    frames = ref.parse_charge_trajectory(TRAJECTORY)
    if len(frames) != 51:
        raise RuntimeError(f"expected 51 water frames, found {len(frames)}")
    pilot = frames[:PILOT_N]
    holdout = frames[PILOT_N:]
    q_water = frames[0][1]
    box = frames[0][3]

    populations = []
    coefficients = []
    for order in ORDERS:
        coeff = ref.load_coefficients(
            0.1 * SPLIT_INPUT_TOL, 0.1 * SPREAD_INPUT_TOL,
            CSPLIT, CSPREAD, order,
        )
        coefficients.append(coeff)
        populations.append(
            sqtools.prepare_alias_population(
                mesh=MESH, order=order, box_length=box, rcut=RCUT,
                csplit=CSPLIT, cspread=CSPREAD, coeff=coeff,
                max_shell=MAX_ALIAS_SHELL, outer_samples=OUTER_SAMPLES,
                seed=SEED + order,
            )
        )
    union, mappings = sqtools.population_mode_union(populations)
    rigid_sq = rigid_spce_sq(
        2.0 * math.pi / box * np.linalg.norm(union, axis=1)
    )
    rigid_chi_by_order: dict[int, float] = {}
    rigid_sampling_variance_by_order: dict[int, float] = {}
    for order, population, mapping in zip(ORDERS, populations, mappings):
        rigid_chi, _, rigid_shell_sampling_variances = (
            sqtools.corrected_chi2_with_sampling(population, mapping, rigid_sq)
        )
        rigid_chi_by_order[order] = rigid_chi
        rigid_sampling_variance_by_order[order] = sum(
            rigid_shell_sampling_variances.values()
        )
    chi_by_order = {order: [] for order in ORDERS}
    pilot_sq_sum = np.zeros(len(union), dtype=np.float64)
    for frame_index, (_, q, xyz, frame_box) in enumerate(pilot):
        sq = sqtools.evaluate_sq_modes(q, xyz, frame_box, union)
        pilot_sq_sum += sq
        for order, population, mapping in zip(ORDERS, populations, mappings):
            chi, _ = sqtools.corrected_chi2(population, mapping, sq)
            chi_by_order[order].append(chi)
        print(json.dumps(dict(stage="pilot_sq", frame=frame_index, modes=len(union), elapsed=time.time()-started)))
    pilot_sq_mean = pilot_sq_sum / len(pilot)

    split = ref.split_continuation(coefficients[0], CSPLIT)
    kernel = ref.split_kernel_grid(MESH, box, RCUT, CSPLIT, split, band_limited=True)
    frame_rows = []
    measured_by_order = {order: [] for order in ORDERS}
    for local_index, (timestep, q, xyz, frame_box) in enumerate(holdout):
        direct = ref.direct_truncated_force(q, xyz, frame_box, kernel)
        for order, coeff in zip(ORDERS, coefficients):
            mesh_force, _ = ref.fixed_ik_mesh_force(
                q, xyz, frame_box, MESH, order, RCUT, CSPLIT, CSPREAD, coeff
            )
            error = ref.rms_vector_error(mesh_force, direct)
            measured_by_order[order].append(error)
            frame_rows.append(
                dict(frame=PILOT_N + local_index, timestep=timestep, order=order,
                     absolute_rms_mesh_error=error, partition="holdout",
                     operator="fixed-influence ik; exact polynomial-window transform",
                     reference="direct truncated PSWF Fourier force on identical radial band and I_M")
            )
        print(json.dumps(dict(stage="holdout_force", frame=local_index, elapsed=time.time()-started)))

    qsum = float(np.sum(q_water * q_water))
    scale = ref.COULOMB_REAL * qsum / math.sqrt(len(q_water))
    summary = []
    shell_rows = []
    for order, coeff, population, mapping in zip(
        ORDERS, coefficients, populations, mappings
    ):
        chi = np.asarray(chi_by_order[order])
        measured = np.asarray(measured_by_order[order])
        chi_mean, shell_values, shell_sampling_variances = (
            sqtools.corrected_chi2_with_sampling(population, mapping, pilot_sq_mean)
        )
        if not math.isclose(chi_mean, float(chi.mean()), rel_tol=2.0e-13, abs_tol=1.0e-30):
            raise RuntimeError("frame-averaged S_q and mean per-frame chi^2 disagree")
        sampling_chi2_sem = math.sqrt(sum(shell_sampling_variances.values()))
        corrected = scale * math.sqrt(chi_mean)
        sampling_prediction_sem = scale * sampling_chi2_sem / (2.0 * math.sqrt(chi_mean))
        homogeneous = scale * math.sqrt(population.homogeneous_chi2)
        rigid_chi = rigid_chi_by_order[order]
        rigid = scale * math.sqrt(rigid_chi)
        rigid_sampling_prediction_sem = (
            scale
            * math.sqrt(rigid_sampling_variance_by_order[order])
            / (2.0 * math.sqrt(rigid_chi))
        )
        homogeneous_convergence = sqtools.homogeneous_chi2_convergence(
            mesh=MESH, order=order, box_length=box, rcut=RCUT,
            csplit=CSPLIT, cspread=CSPREAD, coeff=coeff,
            max_shell=TAIL_CHECK_SHELL,
        )
        if not math.isclose(
            homogeneous_convergence[MAX_ALIAS_SHELL],
            population.homogeneous_chi2,
            rel_tol=2.0e-12,
            abs_tol=1.0e-30,
        ):
            raise RuntimeError("sampled-population homogeneous chi^2 disagrees with exact shell sum")
        cumulative_chi2 = 0.0
        cumulative_sampling_variance = 0.0
        previous_prediction = 0.0
        for shell in sorted(shell_values):
            cumulative_chi2 += shell_values[shell]
            cumulative_sampling_variance += shell_sampling_variances[shell]
            cumulative_prediction = scale * math.sqrt(cumulative_chi2)
            cumulative_mc_sem = (
                scale * math.sqrt(cumulative_sampling_variance)
                / (2.0 * math.sqrt(cumulative_chi2))
                if cumulative_chi2 > 0.0
                else 0.0
            )
            shell_rows.append(
                dict(
                    order=order, shell=shell, actual_seed=SEED + order,
                    shell_chi2_contribution=shell_values[shell],
                    cumulative_chi2=cumulative_chi2,
                    cumulative_absolute_rms_prediction=cumulative_prediction,
                    shell_importance_sampling_variance=shell_sampling_variances[shell],
                    cumulative_importance_sampling_sem=cumulative_mc_sem,
                    incremental_force_fraction=(
                        (cumulative_prediction - previous_prediction) / cumulative_prediction
                        if cumulative_prediction > 0.0 else 0.0
                    ),
                )
            )
            previous_prediction = cumulative_prediction
        previous_shell_prediction = scale * math.sqrt(
            chi_mean - shell_values[MAX_ALIAS_SHELL]
        )
        summary.append(
            dict(
                order=order, mesh_actual=MESH,
                sigma_up=math.pi * RCUT * MESH / (CSPLIT * box),
                csplit=CSPLIT, cspread=CSPREAD,
                measured_holdout_pooled_rms=float(np.sqrt(np.mean(measured * measured))),
                measured_holdout_mean_frame_rms=float(measured.mean()),
                measured_holdout_block5_sem=pooled_rms_block_sem(measured),
                homogeneous_prediction=homogeneous,
                rigid_molecule_prediction=rigid,
                rigid_molecule_importance_sampling_sem=rigid_sampling_prediction_sem,
                pilot_sq_corrected_prediction=corrected,
                pilot_sq_corrected_block5_sem=chi_rms_block_sem(chi, scale),
                pilot_sq_importance_sampling_sem=sampling_prediction_sem,
                pilot_sq_importance_sampling_chi2_sem=sampling_chi2_sem,
                homogeneous_over_measured=homogeneous / float(np.sqrt(np.mean(measured * measured))),
                rigid_molecule_over_measured=rigid / float(np.sqrt(np.mean(measured * measured))),
                corrected_over_measured=corrected / float(np.sqrt(np.mean(measured * measured))),
                pilot_frames=len(pilot), holdout_frames=len(holdout),
                alias_shell=MAX_ALIAS_SHELL, outer_samples_per_shell=OUTER_SAMPLES,
                actual_seed=SEED + order,
                last_shell_force_increment=(corrected - previous_shell_prediction) / corrected,
                homogeneous_lmax_over_l20_force=math.sqrt(
                    homogeneous_convergence[MAX_ALIAS_SHELL]
                    / homogeneous_convergence[TAIL_CHECK_SHELL]
                ),
                alias_contributions="source S_q(k+G_l) plus gather S_q(k)",
                operator="fixed-influence ik; exact polynomial-window transform",
                uncertainty=(
                    "pooled-RMS five-frame block SEM for holdout; block prediction SEM "
                    "for pilot; alias importance-sampling SEM reported separately"
                ),
            )
        )

    with CORRECTION_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    with BY_FRAME_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader(); writer.writerows(frame_rows)
    with SHELL_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shell_rows[0]))
        writer.writeheader(); writer.writerows(shell_rows)
    MANIFEST.write_text(
        json.dumps(
            dict(
                path_basis="bundle_root",
                purpose="Main Figure 4 molecular charge-correlation correction",
                generator=file_record(Path(__file__)),
                utilities=[file_record(HERE / "fixed_ik_reference.py"), file_record(HERE / "sq_alias_tools.py")],
                trajectory=file_record(TRAJECTORY), pilot_indices=list(range(PILOT_N)),
                holdout_indices=list(range(PILOT_N, len(frames))),
                operator="fixed-influence ik; exact transform of the same spreading polynomial",
                correction=(
                    "generalized corrected Eq. (90), exhaustive shell 1 and "
                    f"importance-sampled shells 2-{MAX_ALIAS_SHELL}"
                ),
                rigid_molecule_model={
                    "definition": "orientationally averaged rigid SPC/E intramolecular form factor",
                    "q_O_e": Q_O,
                    "q_H_e": Q_H,
                    "d_OH_Angstrom": D_OH,
                    "angle_HOH_degree": HOH_ANGLE_DEG,
                    "d_HH_Angstrom": D_HH,
                    "weight": "1 + (4 qO qH/Qmol) sinc(k dOH) + (2 qH^2/Qmol) sinc(k dHH)",
                },
                alias_contributions="source S_q(k+G_l) plus gather S_q(k)",
                base_seed=SEED,
                actual_seed_by_order={str(order): SEED + order for order in ORDERS},
                outer_samples_per_shell=OUTER_SAMPLES,
                alias_importance_sampling_uncertainty=(
                    "paired source+gather with-replacement variance evaluated on the "
                    "25-frame mean S_q and delta-propagated to RMS"
                ),
                homogeneous_tail_check=f"exact matched-operator shells through {TAIL_CHECK_SHELL}",
                union_alias_modes=len(union),
                spectrum=(
                    "The Supporting Information volume-normalized spectrum is "
                    "generated independently by run_fig4_charge_spectrum.py; "
                    "the present script retains <|rho|^2>/Q only as the "
                    "dimensionless estimator weight."
                ),
                outputs=[
                    file_record(CORRECTION_OUT), file_record(BY_FRAME_OUT),
                    file_record(SHELL_OUT),
                ],
                python=platform.python_version(), numpy=np.__version__,
                elapsed_seconds=time.time()-started,
            ), indent=2,
        )
    )
    print(CORRECTION_OUT)


if __name__ == "__main__":
    main()
