#!/usr/bin/env python3
"""Resolve the structure-aware Figure 4 mesh variance by physical |k|.

The abscissa is the magnitude of the reciprocal vector at which the
dimensionless correlation weight S_q(k)=<|rho(k)|^2>/Q is evaluated.
Consequently, source terms are binned at |k+G_l| and gather terms at |k|.
Summing every bin recovers the generalized fixed-influence estimator exactly
(up to floating-point roundoff).
"""

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
TRAJECTORY = (
    PROJECT
    / "numerical_examples"
    / "water_trajectory_benchmark"
    / "water_short_traj.lammpstrj"
)
SUMMARY_SOURCE = HERE / "fig4_sq_correction_source.csv"
OUT = HERE / "fig4_k_resolved_variance_source.csv"
MANIFEST = HERE / "fig4_k_resolved_variance_manifest.json"

RCUT = 9.0
CSPLIT = 16.894
CSPREAD = 13.251
MESH = 18
ORDER = 5
SPLIT_INPUT_TOL = 1.0e-6
SPREAD_INPUT_TOL = 3.0e-5
PILOT_N = 25
MAX_ALIAS_SHELL = 12
OUTER_SAMPLES = 4096
SEED = 20260808 + ORDER
BIN_WIDTH = 0.2
K_MAX = 85.0
DISPLAY_K_MAX = 4.0


def file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(PROJECT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def bin_indices(modes: np.ndarray, box_length: float, n_bins: int) -> np.ndarray:
    physical_k = 2.0 * math.pi / box_length * np.linalg.norm(modes, axis=1)
    indices = np.floor(physical_k / BIN_WIDTH).astype(np.int64)
    if len(indices) and (indices.min() < 0 or indices.max() >= n_bins):
        raise RuntimeError(
            f"physical-k population exceeds [0,{K_MAX}) Angstrom^-1: "
            f"maximum is {physical_k.max():.9g}"
        )
    return indices


def weighted_bins(indices: np.ndarray, weights: np.ndarray, n_bins: int) -> np.ndarray:
    return np.bincount(indices, weights=weights, minlength=n_bins).astype(np.float64)


def block_sem(frame_values: np.ndarray, block_size: int = 5) -> np.ndarray:
    n_blocks = len(frame_values) // block_size
    if n_blocks < 2 or n_blocks * block_size != len(frame_values):
        raise ValueError("frame count must form at least two complete blocks")
    blocks = np.asarray(
        [
            frame_values[start : start + block_size].mean(axis=0)
            for start in range(0, len(frame_values), block_size)
        ]
    )
    return blocks.std(axis=0, ddof=1) / math.sqrt(n_blocks)


def read_reference_row() -> dict[str, str]:
    with SUMMARY_SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["order"]) == ORDER]
    if len(selected) != 1:
        raise RuntimeError(f"expected one P={ORDER} Figure 4 summary row")
    return selected[0]


def main() -> None:
    started = time.time()
    frames = ref.parse_charge_trajectory(TRAJECTORY)
    if len(frames) != 51:
        raise RuntimeError(f"expected 51 water frames, found {len(frames)}")
    pilot = frames[:PILOT_N]
    box = float(pilot[0][3])
    charges = pilot[0][1]

    coeff = ref.load_coefficients(
        0.1 * SPLIT_INPUT_TOL,
        0.1 * SPREAD_INPUT_TOL,
        CSPLIT,
        CSPREAD,
        ORDER,
    )
    population = sqtools.prepare_alias_population(
        mesh=MESH,
        order=ORDER,
        box_length=box,
        rcut=RCUT,
        csplit=CSPLIT,
        cspread=CSPREAD,
        coeff=coeff,
        max_shell=MAX_ALIAS_SHELL,
        outer_samples=OUTER_SAMPLES,
        seed=SEED,
    )
    union, mappings = sqtools.population_mode_union([population])
    mapping = mappings[0]

    edges = np.arange(0.0, K_MAX + BIN_WIDTH * 0.5, BIN_WIDTH)
    n_bins = len(edges) - 1
    source_exact_bins = bin_indices(population.exact_modes, box, n_bins)
    gather_exact_bins = bin_indices(population.exact_gather_modes, box, n_bins)
    zero_bins = bin_indices(population.zero_modes, box, n_bins)
    sampled_source_bins = {
        shell: bin_indices(modes, box, n_bins)
        for shell, modes in population.sampled_modes.items()
    }
    sampled_gather_bins = {
        shell: bin_indices(modes, box, n_bins)
        for shell, modes in population.sampled_gather_modes.items()
    }

    homogeneous_source = weighted_bins(
        source_exact_bins, population.exact_weights, n_bins
    )
    homogeneous_gather = weighted_bins(
        gather_exact_bins, population.exact_gather_weights, n_bins
    )
    homogeneous_zero = weighted_bins(zero_bins, population.zero_weights, n_bins)
    for shell, indices in sampled_source_bins.items():
        per_draw = population.shell_weight_sums[shell] / len(indices)
        homogeneous_source += weighted_bins(
            indices, np.full(len(indices), per_draw), n_bins
        )
        gather_indices = sampled_gather_bins[shell]
        homogeneous_gather += weighted_bins(
            gather_indices, np.full(len(gather_indices), per_draw), n_bins
        )
    homogeneous_total = homogeneous_source + homogeneous_gather + homogeneous_zero

    corrected_source_frames = []
    corrected_gather_frames = []
    corrected_zero_frames = []
    direct_chi2 = []
    sq_sum = np.zeros(len(union), dtype=np.float64)
    for frame_index, (_, q, xyz, frame_box) in enumerate(pilot, start=1):
        if not np.array_equal(q, charges) or not math.isclose(frame_box, box):
            raise RuntimeError("charge vector or box changed within the water trajectory")
        sq = sqtools.evaluate_sq_modes(q, xyz, frame_box, union)
        sq_sum += sq
        source = weighted_bins(
            source_exact_bins,
            population.exact_weights * sq[mapping["exact"]],
            n_bins,
        )
        gather = weighted_bins(
            gather_exact_bins,
            population.exact_gather_weights * sq[mapping["exact_gather"]],
            n_bins,
        )
        zero = weighted_bins(
            zero_bins,
            population.zero_weights * sq[mapping["zero"]],
            n_bins,
        )
        for shell, indices in sampled_source_bins.items():
            per_draw = population.shell_weight_sums[shell] / len(indices)
            source += weighted_bins(
                indices,
                per_draw * sq[mapping["sampled"][shell]],
                n_bins,
            )
            gather_indices = sampled_gather_bins[shell]
            gather += weighted_bins(
                gather_indices,
                per_draw * sq[mapping["sampled_gather"][shell]],
                n_bins,
            )
        total = source + gather + zero
        chi2, _ = sqtools.corrected_chi2(population, mapping, sq)
        if not math.isclose(total.sum(), chi2, rel_tol=2.0e-13, abs_tol=1.0e-30):
            raise RuntimeError("k-resolved bins do not recover the per-frame estimator")
        corrected_source_frames.append(source)
        corrected_gather_frames.append(gather)
        corrected_zero_frames.append(zero)
        direct_chi2.append(chi2)
        print(json.dumps({"frame": frame_index, "modes": len(union)}))

    corrected_source_frames = np.asarray(corrected_source_frames)
    corrected_gather_frames = np.asarray(corrected_gather_frames)
    corrected_zero_frames = np.asarray(corrected_zero_frames)
    corrected_total_frames = (
        corrected_source_frames + corrected_gather_frames + corrected_zero_frames
    )
    corrected_source = corrected_source_frames.mean(axis=0)
    corrected_gather = corrected_gather_frames.mean(axis=0)
    corrected_zero = corrected_zero_frames.mean(axis=0)
    corrected_total = corrected_total_frames.mean(axis=0)
    corrected_sem = block_sem(corrected_total_frames)
    sq_mean = sq_sum / PILOT_N
    sampling_variance_by_bin = np.zeros(n_bins, dtype=np.float64)
    for shell, source_indices in mapping["sampled"].items():
        source_bins = sampled_source_bins[shell]
        gather_bins = sampled_gather_bins[shell]
        source_sq = sq_mean[source_indices]
        gather_sq = sq_mean[mapping["sampled_gather"][shell]]
        shell_weight = population.shell_weight_sums[shell]
        for selected_bin in np.unique(np.concatenate([source_bins, gather_bins])):
            paired_bin_value = np.zeros(len(source_bins), dtype=np.float64)
            source_mask = source_bins == selected_bin
            gather_mask = gather_bins == selected_bin
            paired_bin_value[source_mask] += source_sq[source_mask]
            paired_bin_value[gather_mask] += gather_sq[gather_mask]
            sampling_variance_by_bin[selected_bin] += (
                shell_weight**2
                * float(np.var(paired_bin_value, ddof=1))
                / len(paired_bin_value)
            )
    sampling_sem = np.sqrt(sampling_variance_by_bin)

    if not math.isclose(
        homogeneous_total.sum(),
        population.homogeneous_chi2,
        rel_tol=2.0e-14,
        abs_tol=1.0e-30,
    ):
        raise RuntimeError("homogeneous k bins do not recover the full estimator")
    if not math.isclose(
        corrected_total.sum(),
        float(np.mean(direct_chi2)),
        rel_tol=2.0e-13,
        abs_tol=1.0e-30,
    ):
        raise RuntimeError("mean k bins do not recover the mean corrected estimator")

    reference = read_reference_row()
    qsum = float(np.sum(charges * charges))
    scale = ref.COULOMB_REAL * qsum / math.sqrt(len(charges))
    reference_corrected_chi2 = (
        float(reference["pilot_sq_corrected_prediction"]) / scale
    ) ** 2
    if not math.isclose(
        corrected_total.sum(),
        reference_corrected_chi2,
        rel_tol=2.0e-12,
        abs_tol=1.0e-30,
    ):
        raise RuntimeError("new decomposition disagrees with archived Figure 4 summary")

    scale2 = scale * scale
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        rows.append(
            {
                "k_lower_Angstrom-1": lower,
                "k_upper_Angstrom-1": upper,
                "k_center_Angstrom-1": 0.5 * (lower + upper),
                "order": ORDER,
                "mesh": MESH,
                "c_split": CSPLIT,
                "c_spread": CSPREAD,
                "homogeneous_source_chi2": homogeneous_source[index],
                "homogeneous_gather_chi2": homogeneous_gather[index],
                "homogeneous_zero_chi2": homogeneous_zero[index],
                "homogeneous_total_chi2": homogeneous_total[index],
                "corrected_source_chi2": corrected_source[index],
                "corrected_gather_chi2": corrected_gather[index],
                "corrected_zero_chi2": corrected_zero[index],
                "corrected_total_chi2": corrected_total[index],
                "corrected_total_block5_sem_chi2": corrected_sem[index],
                "corrected_total_importance_sampling_sem_chi2": sampling_sem[index],
                "homogeneous_force_variance": scale2 * homogeneous_total[index],
                "corrected_force_variance": scale2 * corrected_total[index],
                "corrected_force_variance_block5_sem": scale2 * corrected_sem[index],
                "corrected_force_variance_importance_sampling_sem": (
                    scale2 * sampling_sem[index]
                ),
                "normalization_chi2": population.homogeneous_chi2,
                "binning_definition": (
                    "magnitude of S_q argument: |k+G_l| for source, "
                    "|k| for gather and zero-alias residual"
                ),
                "estimator_weight_normalization": "S_q(k)=<|rho(k)|^2>/Q",
            }
        )
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    display = edges[1:] <= DISPLAY_K_MAX + 1.0e-12
    shell_values = sqtools.corrected_chi2_with_sampling(
        population,
        mapping,
        sq_mean,
    )[1]
    corrected_sum = float(corrected_total.sum())
    homogeneous_sum = float(homogeneous_total.sum())
    MANIFEST.write_text(
        json.dumps(
            {
                "path_basis": "bundle_root",
                "purpose": "Main Figure 4 k-resolved mesh-error variance mechanism",
                "generator": file_record(Path(__file__)),
                "utilities": [
                    file_record(HERE / "fixed_ik_reference.py"),
                    file_record(HERE / "sq_alias_tools.py"),
                ],
                "trajectory": file_record(TRAJECTORY),
                "summary_crosscheck": file_record(SUMMARY_SOURCE),
                "pilot_indices_zero_based": list(range(PILOT_N)),
                "parameters": {
                    "r_c_Angstrom": RCUT,
                    "c_split": CSPLIT,
                    "c_spread": CSPREAD,
                    "mesh": MESH,
                    "order": ORDER,
                    "max_alias_shell": MAX_ALIAS_SHELL,
                    "outer_samples_per_shell": OUTER_SAMPLES,
                    "seed": SEED,
                },
                "binning": {
                    "width_Angstrom-1": BIN_WIDTH,
                    "computed_range_Angstrom-1": [0.0, K_MAX],
                    "main_display_range_Angstrom-1": [0.0, DISPLAY_K_MAX],
                    "definition": (
                        "magnitude of the reciprocal vector at which S_q is "
                        "evaluated; |k+G_l| for source and |k| for gather/residual"
                    ),
                },
                "estimator_weight_normalization": "S_q(k)=<|rho(k)|^2>/Q",
                "conservation": {
                    "homogeneous_sum_over_bins": homogeneous_sum,
                    "homogeneous_estimator_chi2": population.homogeneous_chi2,
                    "corrected_sum_over_bins": corrected_sum,
                    "corrected_mean_direct_chi2": float(np.mean(direct_chi2)),
                    "corrected_archived_chi2": reference_corrected_chi2,
                },
                "mechanism_summary": {
                    "corrected_over_homogeneous_variance": corrected_sum / homogeneous_sum,
                    "corrected_over_homogeneous_rms": math.sqrt(corrected_sum / homogeneous_sum),
                    "fraction_homogeneous_variance_at_k_le_10": (
                        float(homogeneous_total[display].sum()) / homogeneous_sum
                    ),
                    "fraction_corrected_variance_at_k_le_10": (
                        float(corrected_total[display].sum()) / corrected_sum
                    ),
                    "fraction_corrected_chi2_from_exact_shell_1": (
                        shell_values[1] / sum(shell_values.values())
                    ),
                },
                "uncertainty": (
                    "five nonoverlapping five-frame block SEM and a separate paired "
                    "with-replacement importance-sampling SEM for each corrected bin; "
                    "the latter retains source-gather covariance within each draw"
                ),
                "output": file_record(OUT),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )
    print(OUT)


if __name__ == "__main__":
    main()
