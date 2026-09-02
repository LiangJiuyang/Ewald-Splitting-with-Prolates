#!/usr/bin/env python3
"""Generate the all-charge SPC/E-water Fourier-truncation reference.

This runner deliberately avoids the historical LAMMPS Ewald dump.  That dump
uses the production ``special_bonds`` convention, whereas the exact splitting
identity required here is an *all-charge* identity.  It also avoids the
polynomial erfc approximation in ``pair_style coul/long``.

For every one of the 50 archived water frames, the script evaluates

    F_smooth,infinity = F_Coulomb,all-charge^direct-Ewald
                        - F_near,PSWF^all-charge,

and compares it with the exact PSWF particle--mode sum on the symmetric
``M=21`` reciprocal cube.  There is no spreading, FFT, differentiation
choice, influence function, or special-bond correction in this measurement.

The production direct Ewald decomposition uses exact scipy.special.erfc,
alpha=0.6 A^-1, a 9 A real-space cutoff, and a symmetric M=65 reciprocal
cube.  Frames 0, 25, and 50 are independently recomputed with alpha=0.5,
a 12 A cutoff, and M=55.  Compact PSWF near forces use a periodic pair list
and a 4097-node cubic spline built from the exact project-local MathPSWF
profile; pointwise spline errors are checked against fresh exact evaluations.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from scipy.special import erfc

from fixed_ik_reference import COULOMB_REAL, PSWF_SOURCE_DIR, parse_charge_trajectory
from fig2_fourier_reference import (
    ExactPSWFContinuation,
    direct_force_from_kernel,
    discrete_eq46_sum,
    eq46_force_error,
    eq56_closed_force_error,
    exact_inside_profile,
    lattice_shell_counts,
    symmetric_kernel_grid,
    symmetric_reciprocal_arrays,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
WATER_ROOT = PROJECT_ROOT / "numerical_examples" / "water_trajectory_benchmark"
WATER_DATA = WATER_ROOT / "water.data"
TRAJECTORY = WATER_ROOT / "water_short_traj.lammpstrj"

RCUT = 9.0
RECIPROCAL_MESH = 21
PSWF_QUADRATURE_ORDER = 768
PSWF_SPLINE_NODES = 4097
SPLINE_TEST_MIDPOINTS = 4096
SPLINE_TEST_RANDOM = 1024
SPLINE_TEST_SEED = 20260809
EQ46_TAIL_RADIUS = 320

# The tolerance is table provenance only.  The independent variable in the
# Fourier-tail experiment is c_split.
SPLIT_CASES = (
    (4.0e-3, 8.0189),
    (1.0e-3, 9.5392),
    (5.0e-4, 10.29),
    (1.0e-4, 12.024),
    (5.0e-5, 12.762),
    (1.0e-5, 14.471),
    (1.0e-6, 16.894),
)

PILOT_INDICES = tuple(range(25))
HOLDOUT_INDICES = tuple(range(25, 50))
CROSSCHECK_INDICES = (0, 25, 49)

EWALD_PRODUCTION = {
    "alpha_inverse_A": 0.6,
    "real_cutoff_A": 9.0,
    "reciprocal_mesh": 65,
}
EWALD_CROSSCHECK = {
    "alpha_inverse_A": 0.5,
    "real_cutoff_A": 12.0,
    "reciprocal_mesh": 55,
}

BY_FRAME = HERE / "fig2_water_fourier_reference_by_frame.csv"
SUMMARY = HERE / "fig2_water_fourier_reference_summary.csv"
BLOCKS = HERE / "fig2_water_fourier_reference_blocks.csv"
EWALD_CHECKS = HERE / "fig2_water_direct_ewald_crosscheck.csv"
SPLINE_CHECKS = HERE / "fig2_water_pswf_spline_accuracy.csv"
EWALD_NPZ = HERE / "fig2_water_allcharge_direct_ewald_reference.npz"
MANIFEST = HERE / "fig2_water_fourier_reference_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    try:
        display_path = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms_vectors(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def pair_geometry(
    xyz: np.ndarray, box_length: float, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return periodic unordered pairs, minimum-image vectors, and radii."""

    wrapped = np.mod(np.asarray(xyz, dtype=np.float64), box_length)
    pairs = cKDTree(wrapped, boxsize=box_length).query_pairs(
        cutoff, output_type="ndarray"
    )
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise RuntimeError("periodic pair-list construction failed")
    displacement = wrapped[pairs[:, 0]] - wrapped[pairs[:, 1]]
    displacement -= box_length * np.rint(displacement / box_length)
    radius = np.linalg.norm(displacement, axis=1)
    if np.any(radius <= 0.0) or np.any(radius >= cutoff):
        raise RuntimeError("invalid radius in periodic pair list")
    return pairs, displacement, radius


def accumulate_pair_force(
    natoms: int,
    pairs: np.ndarray,
    pair_force: np.ndarray,
) -> np.ndarray:
    force = np.zeros((natoms, 3), dtype=np.float64)
    np.add.at(force, pairs[:, 0], pair_force)
    np.add.at(force, pairs[:, 1], -pair_force)
    return force


def ewald_kernel(
    alpha: float, reciprocal_mesh: int, box_length: float
) -> np.ndarray:
    _, k2 = symmetric_reciprocal_arrays(reciprocal_mesh, box_length)
    kernel = np.zeros_like(k2)
    nonzero = k2 > 0.0
    kernel[nonzero] = (
        4.0
        * math.pi
        * np.exp(-k2[nonzero] / (4.0 * alpha * alpha))
        / k2[nonzero]
    )
    return kernel


def direct_all_charge_ewald(
    q: np.ndarray,
    xyz: np.ndarray,
    box_length: float,
    settings: dict[str, float | int],
    kernel: np.ndarray,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Exact-erfc real sum plus direct symmetric reciprocal Ewald sum."""

    alpha = float(settings["alpha_inverse_A"])
    cutoff = float(settings["real_cutoff_A"])
    reciprocal_mesh = int(settings["reciprocal_mesh"])
    if reciprocal_mesh % 2 != 1:
        raise ValueError("direct Ewald reciprocal mesh must be odd")
    if cutoff >= 0.5 * box_length:
        raise ValueError("real cutoff must be below half the cubic box length")
    if not math.isclose(float(np.sum(q)), 0.0, abs_tol=1.0e-12):
        raise ValueError("direct periodic Ewald requires a neutral system")
    if geometry is None:
        geometry = pair_geometry(xyz, box_length, cutoff)
    pairs, displacement, radius = geometry
    screen = erfc(alpha * radius) + (
        2.0
        * alpha
        / math.sqrt(math.pi)
        * radius
        * np.exp(-(alpha * radius) ** 2)
    )
    pair_force = (
        COULOMB_REAL
        * (
            q[pairs[:, 0]]
            * q[pairs[:, 1]]
            * screen
            / radius**3
        )[:, None]
        * displacement
    )
    real_force = accumulate_pair_force(len(q), pairs, pair_force)
    reciprocal_force = direct_force_from_kernel(
        q, np.mod(xyz, box_length), box_length, kernel
    )
    force = real_force + reciprocal_force
    half_width = reciprocal_mesh // 2
    kmax = 2.0 * math.pi * half_width / box_length
    metadata = {
        **settings,
        "real_pair_count": len(pairs),
        "reciprocal_half_width": half_width,
        "reciprocal_Kmax_inverse_A": kmax,
        "real_gaussian_screen_at_cutoff": math.exp(
            -(alpha * cutoff) ** 2
        ),
        "real_force_screen_at_cutoff": float(
            erfc(alpha * cutoff)
            + 2.0
            * alpha
            / math.sqrt(math.pi)
            * cutoff
            * math.exp(-(alpha * cutoff) ** 2)
        ),
        "reciprocal_gaussian_screen_at_face": math.exp(
            -(kmax * kmax) / (4.0 * alpha * alpha)
        ),
        "real_operator": "periodic cKDTree pair list; scipy.special.erfc; full Coulomb weights",
        "reciprocal_operator": "direct particle-mode symmetric Gaussian Ewald cube",
        "boundary": "three-dimensional periodic conducting convention",
        "net_force_norm_kcal_per_mol_A": float(np.linalg.norm(np.sum(force, axis=0))),
    }
    return force, metadata


class PSWFSpline:
    """Exact PSWF continuation plus a checked spline for compact near forces."""

    def __init__(self, csplit: float) -> None:
        self.csplit = float(csplit)
        self.pswf = ExactPSWFContinuation(
            self.csplit, PSWF_QUADRATURE_ORDER
        )
        self.nodes = np.linspace(0.0, 1.0, PSWF_SPLINE_NODES)
        constants, psi, integral = exact_inside_profile(
            self.csplit, self.nodes
        )
        if not math.isclose(constants.c0, self.pswf.constants.c0, rel_tol=1.0e-14):
            raise RuntimeError("inconsistent exact PSWF constants")
        self.psi = CubicSpline(self.nodes, psi)
        self.integral = CubicSpline(self.nodes, integral)

    def near_force(
        self,
        q: np.ndarray,
        geometry: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        pairs, displacement, radius = geometry
        scaled = radius / RCUT
        factor = (
            1.0
            - self.integral(scaled) / self.pswf.constants.c0
            + scaled * self.psi(scaled) / self.pswf.constants.c0
        )
        pair_force = (
            COULOMB_REAL
            * (
                q[pairs[:, 0]]
                * q[pairs[:, 1]]
                * factor
                / radius**3
            )[:, None]
            * displacement
        )
        return accumulate_pair_force(len(q), pairs, pair_force)

    def accuracy_row(self) -> dict[str, object]:
        midpoint = (
            np.arange(SPLINE_TEST_MIDPOINTS, dtype=np.float64) + 0.5
        ) / SPLINE_TEST_MIDPOINTS
        random = np.random.default_rng(SPLINE_TEST_SEED).random(
            SPLINE_TEST_RANDOM
        )
        points = np.concatenate((midpoint, random))
        constants, psi_exact, integral_exact = exact_inside_profile(
            self.csplit, points
        )
        psi_error = np.asarray(self.psi(points)) - psi_exact
        integral_error = np.asarray(self.integral(points)) - integral_exact
        exact_factor = (
            1.0
            - integral_exact / constants.c0
            + points * psi_exact / constants.c0
        )
        spline_factor = (
            1.0
            - self.integral(points) / constants.c0
            + points * self.psi(points) / constants.c0
        )
        factor_error = np.asarray(spline_factor) - exact_factor
        return {
            "c_split": self.csplit,
            "spline_nodes": PSWF_SPLINE_NODES,
            "test_midpoints": SPLINE_TEST_MIDPOINTS,
            "test_random_points": SPLINE_TEST_RANDOM,
            "random_seed": SPLINE_TEST_SEED,
            "max_abs_psi_error": float(np.max(np.abs(psi_error))),
            "rms_psi_error": float(np.sqrt(np.mean(psi_error**2))),
            "max_abs_integral_error": float(
                np.max(np.abs(integral_error))
            ),
            "rms_integral_error": float(
                np.sqrt(np.mean(integral_error**2))
            ),
            "max_abs_near_force_factor_error": float(
                np.max(np.abs(factor_error))
            ),
            "rms_near_force_factor_error": float(
                np.sqrt(np.mean(factor_error**2))
            ),
            "interpolant": "scipy.interpolate.CubicSpline, not-a-knot",
            "exact_source": "project-local MathPSWF via eval_pswf_profile.cpp",
        }


def pooled_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def block_statistics(
    values: np.ndarray,
    frame_indices: np.ndarray,
    partition: str,
    csplit: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return fixed-five block SEM and balanced delete-one-block jackknife."""

    values = np.asarray(values, dtype=np.float64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if len(values) != len(frame_indices) or len(values) < 10:
        raise ValueError("invalid input to block statistics")
    rows: list[dict[str, object]] = []

    n_fixed = len(values) // 5
    fixed_rms = []
    for block in range(n_fixed):
        selection = np.arange(block * 5, (block + 1) * 5)
        estimate = pooled_rms(values[selection])
        fixed_rms.append(estimate)
        rows.append(
            {
                "c_split": csplit,
                "partition": partition,
                "statistic_family": "nonoverlapping_contiguous_block5",
                "block_index": block,
                "frame_indices_zero_based": ";".join(
                    str(item) for item in frame_indices[selection]
                ),
                "block_frame_count": len(selection),
                "block_pooled_rms_kcal_per_mol_A": estimate,
                "delete_one_block_estimate_kcal_per_mol_A": "",
            }
        )
    fixed_rms_array = np.asarray(fixed_rms)
    fixed_sem = float(
        np.std(fixed_rms_array, ddof=1) / math.sqrt(len(fixed_rms_array))
    )

    balanced_count = 10 if partition == "all" else 5
    balanced = [
        np.asarray(item, dtype=np.int64)
        for item in np.array_split(np.arange(len(values)), balanced_count)
    ]
    deleted = []
    all_local = np.arange(len(values))
    for block, selection in enumerate(balanced):
        keep = np.setdiff1d(all_local, selection, assume_unique=True)
        estimate = pooled_rms(values[keep])
        deleted.append(estimate)
        rows.append(
            {
                "c_split": csplit,
                "partition": partition,
                "statistic_family": "balanced_contiguous_delete_one_block",
                "block_index": block,
                "frame_indices_zero_based": ";".join(
                    str(item) for item in frame_indices[selection]
                ),
                "block_frame_count": len(selection),
                "block_pooled_rms_kcal_per_mol_A": pooled_rms(
                    values[selection]
                ),
                "delete_one_block_estimate_kcal_per_mol_A": estimate,
            }
        )
    deleted_array = np.asarray(deleted)
    deleted_center = float(np.mean(deleted_array))
    delete_one_sem = float(
        np.sqrt(
            (len(deleted_array) - 1)
            / len(deleted_array)
            * np.sum((deleted_array - deleted_center) ** 2)
        )
    )
    summary = {
        "block5_size": 5,
        "block5_complete_blocks": n_fixed,
        "block5_frames_used": n_fixed * 5,
        "block5_remainder_frames": len(values) - n_fixed * 5,
        "contiguous_block5_pooled_rms_sem_kcal_per_mol_A": fixed_sem,
        "delete_one_balanced_block_count": len(balanced),
        "delete_one_balanced_block_sizes": ";".join(
            str(len(item)) for item in balanced
        ),
        "delete_one_balanced_block_jackknife_sem_kcal_per_mol_A": delete_one_sem,
    }
    return summary, rows


def main() -> None:
    started = time.perf_counter()
    frames = parse_charge_trajectory(TRAJECTORY)
    if len(frames) != 50:
        raise RuntimeError(f"expected 50 water frames, found {len(frames)}")
    first_q = frames[0][1]
    box_length = frames[0][3]
    qsum = float(np.sum(first_q * first_q))
    if (
        len(first_q) != 2703
        or not math.isclose(box_length, 30.0)
        or not math.isclose(float(np.sum(first_q)), 0.0, abs_tol=1.0e-12)
        or not math.isclose(qsum, 970.95241464, rel_tol=1.0e-13)
    ):
        raise RuntimeError("unexpected SPC/E water metadata")
    for timestep, q, _, frame_box in frames:
        if not np.array_equal(q, first_q) or not math.isclose(
            frame_box, box_length
        ):
            raise RuntimeError(
                f"charges or box changed at trajectory timestep {timestep}"
            )

    splines = {csplit: PSWFSpline(csplit) for _, csplit in SPLIT_CASES}
    spline_rows = [splines[csplit].accuracy_row() for _, csplit in SPLIT_CASES]
    maximum_spline_factor_error = max(
        float(row["max_abs_near_force_factor_error"])
        for row in spline_rows
    )
    if maximum_spline_factor_error >= 1.0e-10:
        raise RuntimeError(
            "PSWF spline accuracy is insufficient: "
            f"max factor error {maximum_spline_factor_error:.3e}"
        )

    finite_kernels = {
        csplit: symmetric_kernel_grid(
            spline.pswf, RECIPROCAL_MESH, box_length, RCUT
        )
        for csplit, spline in splines.items()
    }
    production_kernel = ewald_kernel(
        float(EWALD_PRODUCTION["alpha_inverse_A"]),
        int(EWALD_PRODUCTION["reciprocal_mesh"]),
        box_length,
    )
    crosscheck_kernel = ewald_kernel(
        float(EWALD_CROSSCHECK["alpha_inverse_A"]),
        int(EWALD_CROSSCHECK["reciprocal_mesh"]),
        box_length,
    )

    frame_rows: list[dict[str, object]] = []
    ewald_rows: list[dict[str, object]] = []
    ewald_force_frames: list[np.ndarray] = []
    errors_by_c = {
        csplit: np.empty(len(frames), dtype=np.float64)
        for _, csplit in SPLIT_CASES
    }
    timesteps = np.empty(len(frames), dtype=np.int64)
    production_metadata: dict[str, object] | None = None

    for frame_index, (timestep, q, xyz, frame_box) in enumerate(frames):
        timesteps[frame_index] = timestep
        geometry9 = pair_geometry(xyz, frame_box, RCUT)
        coulomb, metadata = direct_all_charge_ewald(
            q,
            xyz,
            frame_box,
            EWALD_PRODUCTION,
            production_kernel,
            geometry9,
        )
        production_metadata = metadata
        ewald_force_frames.append(coulomb)
        if float(metadata["net_force_norm_kcal_per_mol_A"]) >= 1.0e-8:
            raise RuntimeError("direct Ewald violates net-force cancellation")

        if frame_index in CROSSCHECK_INDICES:
            geometry12 = pair_geometry(
                xyz, frame_box, float(EWALD_CROSSCHECK["real_cutoff_A"])
            )
            alternative, alternative_metadata = direct_all_charge_ewald(
                q,
                xyz,
                frame_box,
                EWALD_CROSSCHECK,
                crosscheck_kernel,
                geometry12,
            )
            difference = alternative - coulomb
            ewald_rows.append(
                {
                    "frame_index": frame_index,
                    "timestep": timestep,
                    "production_alpha_inverse_A": EWALD_PRODUCTION[
                        "alpha_inverse_A"
                    ],
                    "production_real_cutoff_A": EWALD_PRODUCTION[
                        "real_cutoff_A"
                    ],
                    "production_reciprocal_mesh": EWALD_PRODUCTION[
                        "reciprocal_mesh"
                    ],
                    "production_real_pair_count": metadata["real_pair_count"],
                    "production_real_force_screen_at_cutoff": metadata[
                        "real_force_screen_at_cutoff"
                    ],
                    "production_reciprocal_screen_at_face": metadata[
                        "reciprocal_gaussian_screen_at_face"
                    ],
                    "crosscheck_alpha_inverse_A": EWALD_CROSSCHECK[
                        "alpha_inverse_A"
                    ],
                    "crosscheck_real_cutoff_A": EWALD_CROSSCHECK[
                        "real_cutoff_A"
                    ],
                    "crosscheck_reciprocal_mesh": EWALD_CROSSCHECK[
                        "reciprocal_mesh"
                    ],
                    "crosscheck_real_pair_count": alternative_metadata[
                        "real_pair_count"
                    ],
                    "crosscheck_real_force_screen_at_cutoff": alternative_metadata[
                        "real_force_screen_at_cutoff"
                    ],
                    "crosscheck_reciprocal_screen_at_face": alternative_metadata[
                        "reciprocal_gaussian_screen_at_face"
                    ],
                    "rms_force_difference_kcal_per_mol_A": rms_vectors(
                        difference
                    ),
                    "max_abs_component_difference_kcal_per_mol_A": float(
                        np.max(np.abs(difference))
                    ),
                    "operator": "independent exact-erfc direct Ewald decompositions; all charge pairs",
                }
            )

        wrapped = np.mod(xyz, frame_box)
        for input_tolerance, csplit in SPLIT_CASES:
            spline = splines[csplit]
            near = spline.near_force(q, geometry9)
            infinite = coulomb - near
            truncated = direct_force_from_kernel(
                q, wrapped, frame_box, finite_kernels[csplit]
            )
            difference = infinite - truncated
            component = np.sqrt(np.mean(difference * difference, axis=0))
            measured = float(np.sqrt(np.sum(component * component)))
            errors_by_c[csplit][frame_index] = measured
            partition = "pilot" if frame_index in PILOT_INDICES else "holdout"
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "timestep": timestep,
                    "partition": partition,
                    "c_split": csplit,
                    "split_input_tolerance_table_provenance": input_tolerance,
                    "r_c_A": RCUT,
                    "box_length_A": frame_box,
                    "n_atoms": len(q),
                    "Q_sum_q_squared": qsum,
                    "reciprocal_mesh": RECIPROCAL_MESH,
                    "reciprocal_half_width": RECIPROCAL_MESH // 2,
                    "Kmax_inverse_A": 2.0
                    * math.pi
                    * (RECIPROCAL_MESH // 2)
                    / frame_box,
                    "resolved_factor_Kmax_rc_over_csplit": 2.0
                    * math.pi
                    * (RECIPROCAL_MESH // 2)
                    / frame_box
                    * RCUT
                    / csplit,
                    "measured_abs_rms_force_error_kcal_per_mol_A": measured,
                    "measured_x_rms_kcal_per_mol_A": float(component[0]),
                    "measured_y_rms_kcal_per_mol_A": float(component[1]),
                    "measured_z_rms_kcal_per_mol_A": float(component[2]),
                    "all_charge_coulomb_rms_kcal_per_mol_A": rms_vectors(
                        coulomb
                    ),
                    "compact_near_rms_kcal_per_mol_A": rms_vectors(near),
                    "smooth_infinite_rms_kcal_per_mol_A": rms_vectors(infinite),
                    "smooth_truncated_rms_kcal_per_mol_A": rms_vectors(
                        truncated
                    ),
                    "finite_operator": "direct exact-PSWF particle-mode sum on symmetric M=21 cube",
                    "infinite_operator": "exact-erfc direct all-charge Ewald minus all-charge compact PSWF near force",
                    "special_bonds_present": False,
                    "particle_mesh_present": False,
                }
            )
        if (frame_index + 1) % 5 == 0 or frame_index + 1 == len(frames):
            print(
                json.dumps(
                    {
                        "stage": "water_reference",
                        "frames_complete": frame_index + 1,
                        "frames_total": len(frames),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    maximum_ewald_crosscheck = max(
        float(row["rms_force_difference_kcal_per_mol_A"])
        for row in ewald_rows
    )
    if len(ewald_rows) != len(CROSSCHECK_INDICES):
        raise RuntimeError("not all requested Ewald crosschecks were evaluated")
    if maximum_ewald_crosscheck >= 1.0e-9:
        raise RuntimeError(
            "direct Ewald decomposition crosscheck failed: "
            f"{maximum_ewald_crosscheck:.3e}"
        )

    partitions = {
        "all": np.arange(50, dtype=np.int64),
        "pilot": np.asarray(PILOT_INDICES, dtype=np.int64),
        "holdout": np.asarray(HOLDOUT_INDICES, dtype=np.int64),
    }
    shell_counts = lattice_shell_counts(EQ46_TAIL_RADIUS)
    summary_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    for input_tolerance, csplit in SPLIT_CASES:
        pswf = splines[csplit].pswf
        mode_sum, tail_metadata = discrete_eq46_sum(
            pswf,
            box_length,
            RCUT,
            RECIPROCAL_MESH,
            shell_counts,
            EQ46_TAIL_RADIUS,
        )
        homogeneous_eq46 = eq46_force_error(
            mode_sum, qsum, len(first_q), box_length**3
        )
        closed_eq56 = eq56_closed_force_error(
            pswf,
            qsum,
            len(first_q),
            box_length**3,
            2.0 * math.pi * (RECIPROCAL_MESH // 2) / box_length,
            RCUT,
        )
        for partition, indices in partitions.items():
            selected = errors_by_c[csplit][indices]
            block_summary, rows = block_statistics(
                selected, indices, partition, csplit
            )
            block_rows.extend(rows)
            measured = pooled_rms(selected)
            summary_rows.append(
                {
                    "system": "SPC/E water, all-charge Fourier-tail reference",
                    "partition": partition,
                    "frame_index_start": int(indices[0]),
                    "frame_index_end": int(indices[-1]),
                    "n_frames": len(indices),
                    "c_split": csplit,
                    "split_input_tolerance_table_provenance": input_tolerance,
                    "r_c_A": RCUT,
                    "box_length_A": box_length,
                    "reciprocal_mesh": RECIPROCAL_MESH,
                    "reciprocal_half_width": RECIPROCAL_MESH // 2,
                    "Kmax_inverse_A": 2.0
                    * math.pi
                    * (RECIPROCAL_MESH // 2)
                    / box_length,
                    "n_atoms": len(first_q),
                    "Q_sum_q_squared": qsum,
                    "measured_pooled_abs_rms_kcal_per_mol_A": measured,
                    "frame_rms_mean_kcal_per_mol_A": float(
                        np.mean(selected)
                    ),
                    "frame_rms_sample_sd_kcal_per_mol_A": float(
                        np.std(selected, ddof=1)
                    ),
                    "frame_rms_sem_kcal_per_mol_A": float(
                        np.std(selected, ddof=1) / math.sqrt(len(selected))
                    ),
                    **block_summary,
                    "homogeneous_eq46_discrete_kcal_per_mol_A": homogeneous_eq46,
                    "closed_eq56_kcal_per_mol_A": closed_eq56,
                    "measured_over_homogeneous_eq46": measured
                    / homogeneous_eq46,
                    "eq46_tail_radius_index": EQ46_TAIL_RADIUS,
                    "eq46_continuum_tail_fraction": tail_metadata[
                        "tail_fraction_of_omitted_sum"
                    ],
                    "measurement_operator": "all-charge direct Ewald minus exact compact near, compared with direct M=21 PSWF cube",
                }
            )

    if len(frame_rows) != 50 * len(SPLIT_CASES):
        raise RuntimeError("wrong number of per-frame rows")
    if len(summary_rows) != 3 * len(SPLIT_CASES):
        raise RuntimeError("wrong number of summary rows")
    if min(float(row["resolved_factor_Kmax_rc_over_csplit"]) for row in frame_rows) < 1.0:
        raise RuntimeError("the M=21 cube does not resolve every tested band")

    write_csv(BY_FRAME, frame_rows)
    write_csv(SUMMARY, summary_rows)
    write_csv(BLOCKS, block_rows)
    write_csv(EWALD_CHECKS, ewald_rows)
    write_csv(SPLINE_CHECKS, spline_rows)
    ewald_array = np.stack(ewald_force_frames)
    np.savez_compressed(
        EWALD_NPZ,
        frame_indices=np.arange(50, dtype=np.int64),
        timesteps=timesteps,
        charges=first_q,
        all_charge_coulomb_forces=ewald_array,
        alpha_inverse_A=np.asarray(EWALD_PRODUCTION["alpha_inverse_A"]),
        real_cutoff_A=np.asarray(EWALD_PRODUCTION["real_cutoff_A"]),
        reciprocal_mesh=np.asarray(EWALD_PRODUCTION["reciprocal_mesh"]),
        operator=np.asarray(
            "exact-erfc real sum plus direct symmetric reciprocal Ewald cube; all charge pairs"
        ),
    )

    output_paths = (
        BY_FRAME,
        SUMMARY,
        BLOCKS,
        EWALD_CHECKS,
        SPLINE_CHECKS,
        EWALD_NPZ,
    )
    code_paths = (
        Path(__file__),
        HERE / "fig2_fourier_reference.py",
        HERE / "fixed_ik_reference.py",
        HERE / "eval_pswf_profile.cpp",
        PSWF_SOURCE_DIR / "math_pswf.cpp",
        PSWF_SOURCE_DIR / "math_pswf.h",
        PSWF_SOURCE_DIR / "math_const.h",
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "What is the isolated exact-PSWF Fourier cube-truncation error "
            "for all 50 archived SPC/E water frames?"
        ),
        "operator_chain": {
            "infinite_smooth_force": (
                "exact-erfc direct all-charge conducting-periodic Ewald "
                "minus all-charge exact compact PSWF near force"
            ),
            "finite_smooth_force": (
                "direct exact-PSWF particle-mode sum on the symmetric M=21 cube"
            ),
            "particle_mesh": False,
            "special_bonds": False,
            "historical_lammps_reference_used": False,
        },
        "inputs": [file_record(WATER_DATA), file_record(TRAJECTORY)],
        "trajectory": {
            "frames": len(frames),
            "timesteps": timesteps.tolist(),
            "pilot_indices_zero_based": list(PILOT_INDICES),
            "holdout_indices_zero_based": list(HOLDOUT_INDICES),
            "partitions_are_disjoint": set(PILOT_INDICES).isdisjoint(
                HOLDOUT_INDICES
            ),
        },
        "system": {
            "n_atoms": len(first_q),
            "n_molecules": len(first_q) // 3,
            "box_length_A": box_length,
            "sum_q": float(np.sum(first_q)),
            "Q_sum_q_squared": qsum,
        },
        "parameters": {
            "r_c_A": RCUT,
            "c_split": [csplit for _, csplit in SPLIT_CASES],
            "split_input_tolerance_table_provenance": [
                tolerance for tolerance, _ in SPLIT_CASES
            ],
            "finite_reciprocal_mesh": RECIPROCAL_MESH,
            "finite_reciprocal_half_width": RECIPROCAL_MESH // 2,
            "finite_Kmax_inverse_A": 2.0
            * math.pi
            * (RECIPROCAL_MESH // 2)
            / box_length,
            "pswf_quadrature_order": PSWF_QUADRATURE_ORDER,
            "pswf_spline_nodes": PSWF_SPLINE_NODES,
            "eq46_tail_radius": EQ46_TAIL_RADIUS,
        },
        "direct_ewald": {
            "production": EWALD_PRODUCTION,
            "crosscheck": EWALD_CROSSCHECK,
            "crosscheck_frame_indices_zero_based": list(CROSSCHECK_INDICES),
            "maximum_crosscheck_rms_kcal_per_mol_A": maximum_ewald_crosscheck,
            "maximum_allowed_crosscheck_rms_kcal_per_mol_A": 1.0e-9,
            "production_metadata_last_frame": production_metadata,
        },
        "spline_validation": {
            "test_midpoints": SPLINE_TEST_MIDPOINTS,
            "test_random_points": SPLINE_TEST_RANDOM,
            "random_seed": SPLINE_TEST_SEED,
            "maximum_near_force_factor_error": maximum_spline_factor_error,
            "maximum_allowed_near_force_factor_error": 1.0e-10,
        },
        "uncertainty": {
            "center": "pooled RMS = sqrt(mean of per-frame RMS squared)",
            "block5_sem": (
                "SEM over nonoverlapping contiguous five-frame pooled-RMS "
                "blocks; an incomplete final block is omitted only from this SEM"
            ),
            "delete_one_block": (
                "delete-one-balanced-contiguous-block jackknife over all frames; "
                "five blocks for pilot/holdout and ten blocks for all frames"
            ),
            "autocorrelation_corrected": False,
        },
        "self_checks": {
            "per_frame_rows": len(frame_rows),
            "summary_rows": len(summary_rows),
            "ewald_crosscheck_rows": len(ewald_rows),
            "spline_check_rows": len(spline_rows),
            "minimum_resolved_factor": min(
                float(row["resolved_factor_Kmax_rc_over_csplit"])
                for row in frame_rows
            ),
            "maximum_production_net_force_norm_kcal_per_mol_A": max(
                float(np.linalg.norm(np.sum(force, axis=0)))
                for force in ewald_force_frames
            ),
        },
        "outputs": [file_record(path) for path in output_paths],
        "code": [file_record(path) for path in code_paths],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("c_split all pilot holdout measured/Eq46(holdout)")
    for _, csplit in SPLIT_CASES:
        selected = [
            row
            for row in summary_rows
            if math.isclose(float(row["c_split"]), csplit)
        ]
        indexed = {str(row["partition"]): row for row in selected}
        print(
            f"{csplit:8.4f} "
            f"{float(indexed['all']['measured_pooled_abs_rms_kcal_per_mol_A']):.9e} "
            f"{float(indexed['pilot']['measured_pooled_abs_rms_kcal_per_mol_A']):.9e} "
            f"{float(indexed['holdout']['measured_pooled_abs_rms_kcal_per_mol_A']):.9e} "
            f"{float(indexed['holdout']['measured_over_homogeneous_eq46']):.6f}"
        )
    print(
        f"maximum Ewald crosscheck RMS: {maximum_ewald_crosscheck:.3e}; "
        f"maximum spline factor error: {maximum_spline_factor_error:.3e}; "
        f"elapsed: {time.perf_counter() - started:.1f} s"
    )


if __name__ == "__main__":
    main()
