#!/usr/bin/env python3
"""Freeze the pilot-only SPC/E Fourier-tail prediction for Figure 2.

This generator deliberately stops after trajectory frames 0--24.  It never
parses a holdout coordinate or a reference-force file.  A FINUFFT type-1
transform evaluates the vector-resolved pilot charge spectrum on every mode
inside a radius-80 integer sphere.  Squared-radius accumulation is exact
because the Fourier-tail multiplier is radial; no angular averaging is
applied to the measured modes.

Beyond the measured sphere, the calculation uses the rigid SPC/E
intramolecular form factor on exact integer-lattice shells through radius
320.  The remaining continuum is closed with the leading exact-PSWF sinc
tail, including the two intramolecular sinc terms analytically.

Run with the pinned local environment::

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
      bin/python \
      run_fig2_water_fourier_prediction.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import finufft
import numpy as np
import scipy
from scipy.special import sici

import fixed_ik_reference as ref
from generated_output import manifest_path, section_output_root
from fig2_fourier_reference import (
    ExactPSWFContinuation,
    _sine_square_tail,
    discrete_eq46_sum,
    lattice_shell_counts,
)
from sq_alias_tools import evaluate_sq_modes


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
OUTPUT_ROOT = section_output_root(create=True)
TRAJECTORY = (
    PROJECT_ROOT
    / "numerical_examples"
    / "water_trajectory_benchmark"
    / "water_short_traj.lammpstrj"
)

BLOCK_OUT = OUTPUT_ROOT / "fig2_water_fourier_prediction_by_block.csv"
SUMMARY_OUT = OUTPUT_ROOT / "fig2_water_fourier_prediction_summary.csv"
CONVERGENCE_OUT = OUTPUT_ROOT / "fig2_water_fourier_prediction_convergence.csv"
ACCURACY_OUT = OUTPUT_ROOT / "fig2_water_finufft_accuracy.csv"
MANIFEST_OUT = OUTPUT_ROOT / "fig2_water_fourier_prediction_manifest.json"

RCUT = 9.0
RECIPROCAL_MESH = 21
RECIPROCAL_HALF_WIDTH = RECIPROCAL_MESH // 2
PILOT_N = 25
BLOCK_SIZE = 5
MEASURED_RADII = (32, 40, 48, 64, 80)
MAX_MEASURED_RADIUS = max(MEASURED_RADII)
TAIL_RADII = (160, 240, 320)
MAX_TAIL_RADIUS = max(TAIL_RADII)
QUADRATURE_ORDER = 768
FINUFFT_EPS = 1.0e-12
FINUFFT_NTHREADS = 1
FINUFFT_MODEORD = 0
FINUFFT_REQUIRED_VERSION = "2.5.0"
ACCURACY_SEED = 20260809
ACCURACY_MODE_COUNT = 1000

# Table provenance is retained separately from the physical bandlimit.
SPLIT_CASES = (
    (4.0e-3, 8.0189),
    (1.0e-3, 9.5392),
    (5.0e-4, 10.29),
    (1.0e-4, 12.024),
    (5.0e-5, 12.762),
    (1.0e-5, 14.471),
    (1.0e-6, 16.894),
)

Q_O = -0.8476
Q_H = 0.4238
D_OH = 1.0
HOH_ANGLE_DEG = 109.47
D_HH = math.sqrt(2.0 - 2.0 * math.cos(math.radians(HOH_ANGLE_DEG)))
Q_MOL = Q_O * Q_O + 2.0 * Q_H * Q_H
INTRA_OH_COEFFICIENT = 4.0 * Q_O * Q_H / Q_MOL
INTRA_HH_COEFFICIENT = 2.0 * Q_H * Q_H / Q_MOL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": manifest_path(path, PROJECT_ROOT),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def environment_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path(sys.prefix).resolve()))
    except ValueError:
        for marker in ("site-packages", "dist-packages"):
            if marker in resolved.parts:
                index = resolved.parts.index(marker)
                return Path(*resolved.parts[index:]).as_posix()
        return resolved.name


def environment_file_record(path: Path) -> dict[str, object]:
    return {
        "path": environment_relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def pilot_prefix_record(path: Path, prefix_bytes: int) -> dict[str, object]:
    """Hash only the byte prefix consumed by the unbuffered pilot parser."""

    digest = hashlib.sha256()
    remaining = prefix_bytes
    with path.open("rb", buffering=0) as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("trajectory ended while hashing the pilot prefix")
            digest.update(chunk)
            remaining -= len(chunk)
    return {
        "path": str(path.resolve().relative_to(PROJECT_ROOT)),
        "total_bytes_from_stat_only": path.stat().st_size,
        "pilot_prefix_bytes": prefix_bytes,
        "pilot_prefix_sha256": digest.hexdigest(),
        "full_file_sha256": "not evaluated; holdout bytes are not read",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_pilot_frames(path: Path, limit: int) -> tuple[list[tuple], int]:
    """Parse exactly ``limit`` frames without buffered reads past the pilot."""

    frames = []
    # Raw, unbuffered binary I/O makes ``tell()`` an exact byte boundary and
    # prevents TextIOWrapper read-ahead from touching the first holdout frame.
    with path.open("rb", buffering=0) as handle:
        while len(frames) < limit:
            header = handle.readline()
            if not header:
                raise RuntimeError(f"trajectory ended before {limit} pilot frames")
            if not header.startswith(b"ITEM: TIMESTEP"):
                raise RuntimeError(f"malformed trajectory header in {path}")
            timestep = int(handle.readline())
            if not handle.readline().startswith(b"ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("missing atom count")
            natoms = int(handle.readline())
            if not handle.readline().startswith(b"ITEM: BOX BOUNDS"):
                raise RuntimeError("missing box bounds")
            bounds = [tuple(map(float, handle.readline().split()[:2])) for _ in range(3)]
            columns = [item.decode("ascii") for item in handle.readline().split()[2:]]
            col = {name: i for i, name in enumerate(columns)}
            required = {"id", "q", "x", "y", "z"}
            if not required.issubset(col):
                raise RuntimeError(f"trajectory is missing columns {sorted(required - set(col))}")
            ids = np.empty(natoms, dtype=np.int64)
            q = np.empty(natoms, dtype=np.float64)
            xyz = np.empty((natoms, 3), dtype=np.float64)
            for row in range(natoms):
                fields = handle.readline().split()
                ids[row] = int(fields[col["id"]])
                q[row] = float(fields[col["q"]])
                xyz[row] = [float(fields[col[name]]) for name in ("x", "y", "z")]
            order = np.argsort(ids)
            lo = np.asarray([item[0] for item in bounds], dtype=np.float64)
            lengths = np.asarray([item[1] - item[0] for item in bounds], dtype=np.float64)
            if not np.allclose(lengths, lengths[0], rtol=0.0, atol=1.0e-12):
                raise ValueError("pilot prediction requires a cubic cell")
            frames.append((timestep, q[order], xyz[order] - lo, float(lengths[0])))
        bytes_consumed = handle.tell()
    return frames, bytes_consumed


def intramolecular_sq(k: np.ndarray) -> np.ndarray:
    """Rigid, orientationally averaged SPC/E intramolecular form factor."""

    values = np.asarray(k, dtype=np.float64)
    return (
        1.0
        + INTRA_OH_COEFFICIENT * np.sinc(values * D_OH / math.pi)
        + INTRA_HH_COEFFICIENT * np.sinc(values * D_HH / math.pi)
    )


def sine_over_k3_tail(b: float, lower_k: float) -> float:
    """Return integral_K^infinity sin(b*k)/k^3 dk analytically."""

    if b == 0.0:
        return 0.0
    si = float(sici(b * lower_k)[0])
    sine_tail = math.copysign(math.pi / 2.0, b) - si
    return (
        0.5
        * b
        * (math.cos(b * lower_k) / lower_k - b * sine_tail)
        + math.sin(b * lower_k) / (2.0 * lower_k * lower_k)
    )


def sinc_weighted_sine_square_tail(
    lower_k: float, rcut: float, distance: float
) -> float:
    """Return integral sin(rc*k)^2*sinc(d*k)/k^2 from K to infinity."""

    return (
        0.5 * sine_over_k3_tail(distance, lower_k)
        - 0.25 * sine_over_k3_tail(distance + 2.0 * rcut, lower_k)
        - 0.25 * sine_over_k3_tail(distance - 2.0 * rcut, lower_k)
    ) / distance


def intramolecular_continuum_integral(lower_k: float, rcut: float) -> tuple[float, dict]:
    homogeneous = _sine_square_tail(lower_k, rcut)
    oh = sinc_weighted_sine_square_tail(lower_k, rcut, D_OH)
    hh = sinc_weighted_sine_square_tail(lower_k, rcut, D_HH)
    corrected = (
        homogeneous
        + INTRA_OH_COEFFICIENT * oh
        + INTRA_HH_COEFFICIENT * hh
    )
    if corrected <= 0.0:
        raise FloatingPointError("non-positive intramolecular continuum integral")
    return corrected, {
        "homogeneous_integral": homogeneous,
        "oh_sinc_integral": oh,
        "hh_sinc_integral": hh,
        "corrected_over_homogeneous": corrected / homogeneous,
    }


def block_prediction_sem(chi: np.ndarray, scale: float) -> tuple[list[float], float]:
    if len(chi) != PILOT_N or PILOT_N % BLOCK_SIZE:
        raise ValueError("pilot frames do not form five equal blocks")
    predictions = [
        scale * math.sqrt(float(np.mean(chi[start : start + BLOCK_SIZE])))
        for start in range(0, PILOT_N, BLOCK_SIZE)
    ]
    sem = float(np.std(predictions, ddof=1) / math.sqrt(len(predictions)))
    return predictions, sem


def finufft_environment() -> dict[str, object]:
    if finufft.__version__ != FINUFFT_REQUIRED_VERSION:
        raise RuntimeError(
            f"requires FINUFFT {FINUFFT_REQUIRED_VERSION}, found {finufft.__version__}; "
            "run with a Python environment that provides the required version"
        )
    package = Path(finufft.__file__).resolve().parent
    library = package / "libfinufft.dylib"
    libomp = package / ".dylibs" / "libomp.dylib"
    if not library.is_file() or not libomp.is_file():
        raise FileNotFoundError("pinned FINUFFT shared libraries are missing")
    return {
        "path_basis": "active Python environment prefix",
        "python_executable": environment_relative(Path(sys.executable)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "finufft_version": finufft.__version__,
        "finufft_module": environment_relative(Path(finufft.__file__)),
        "libfinufft": environment_file_record(library),
        "libomp": environment_file_record(libomp),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "requested_finufft_nthreads": FINUFFT_NTHREADS,
    }


def main() -> None:
    started = time.perf_counter()
    environment = finufft_environment()
    frames, bytes_consumed = parse_pilot_frames(TRAJECTORY, PILOT_N)
    if len(frames) != PILOT_N:
        raise RuntimeError("pilot parser did not return exactly 25 frames")

    timesteps = [int(frame[0]) for frame in frames]
    reference_q = frames[0][1]
    natoms = len(reference_q)
    box_length = frames[0][3]
    qsum = float(np.sum(reference_q * reference_q))
    expected_qsum = 901.0 * Q_MOL
    if natoms != 2703:
        raise RuntimeError(f"expected 2703 SPC/E charge sites, found {natoms}")
    if not math.isclose(box_length, 30.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"expected a 30-Angstrom box, found {box_length}")
    if not math.isclose(qsum, expected_qsum, rel_tol=1.0e-13, abs_tol=1.0e-13):
        raise RuntimeError(f"unexpected SPC/E Q: {qsum} != {expected_qsum}")
    for _, q, _, frame_box in frames:
        if not np.array_equal(q, reference_q):
            raise RuntimeError("charges changed inside the pilot partition")
        if not math.isclose(frame_box, box_length, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError("box length changed inside the pilot partition")
    if bytes_consumed >= TRAJECTORY.stat().st_size:
        raise RuntimeError("pilot-only parser unexpectedly consumed the complete trajectory")

    radius = MAX_MEASURED_RADIUS
    grid_size = 2 * radius + 1
    integer_axis = np.arange(-radius, radius + 1, dtype=np.int32)
    squared_radius = (
        integer_axis[:, None, None] ** 2
        + integer_axis[None, :, None] ** 2
        + integer_axis[None, None, :] ** 2
    ).astype(np.int32)
    max_component_inside = (
        (np.abs(integer_axis)[:, None, None] <= RECIPROCAL_HALF_WIDTH)
        & (np.abs(integer_axis)[None, :, None] <= RECIPROCAL_HALF_WIDTH)
        & (np.abs(integer_axis)[None, None, :] <= RECIPROCAL_HALF_WIDTH)
    )
    measured_mask = (
        (squared_radius > 0)
        & (squared_radius <= radius * radius)
        & (~max_component_inside)
    )
    measured_squared_radius = squared_radius[measured_mask]

    shell_counts = lattice_shell_counts(MAX_TAIL_RADIUS)
    cube_axis = np.arange(
        -RECIPROCAL_HALF_WIDTH, RECIPROCAL_HALF_WIDTH + 1, dtype=np.int32
    )
    cube_x, cube_y, cube_z = np.meshgrid(
        cube_axis, cube_axis, cube_axis, indexing="ij"
    )
    cube_squared = (cube_x * cube_x + cube_y * cube_y + cube_z * cube_z).ravel()
    inside_counts = np.bincount(
        cube_squared, minlength=MAX_TAIL_RADIUS * MAX_TAIL_RADIUS + 1
    )[: MAX_TAIL_RADIUS * MAX_TAIL_RADIUS + 1]
    omitted_counts = shell_counts - inside_counts
    if np.any(omitted_counts < 0) or omitted_counts[0] != 0:
        raise RuntimeError("invalid cube-complement lattice degeneracies")
    measured_counts = np.bincount(
        measured_squared_radius, minlength=radius * radius + 1
    )
    np.testing.assert_array_equal(
        measured_counts, omitted_counts[: radius * radius + 1]
    )

    rng = np.random.default_rng(ACCURACY_SEED)
    measured_flat = np.flatnonzero(measured_mask.ravel())
    selected_flat = rng.choice(
        measured_flat, size=ACCURACY_MODE_COUNT, replace=False
    )
    selected_grid_indices = np.column_stack(
        np.unravel_index(selected_flat, measured_mask.shape)
    )
    selected_modes = selected_grid_indices.astype(np.int64) - radius

    per_frame_shell_sq = []
    accuracy_rows: list[dict[str, object]] = []
    inversion_sq_max_abs = 0.0
    for frame_index, (timestep, q, xyz, frame_box) in enumerate(frames):
        angles = [
            2.0 * math.pi * (np.mod(xyz[:, dim], frame_box) / frame_box - 0.5)
            for dim in range(3)
        ]
        rho = finufft.nufft3d1(
            *angles,
            q.astype(np.complex128),
            (grid_size, grid_size, grid_size),
            eps=FINUFFT_EPS,
            isign=-1,
            modeord=FINUFFT_MODEORD,
            nthreads=FINUFFT_NTHREADS,
        )
        sq_grid = (rho.real * rho.real + rho.imag * rho.imag) / qsum
        per_frame_shell_sq.append(
            np.bincount(
                measured_squared_radius,
                weights=sq_grid[measured_mask],
                minlength=radius * radius + 1,
            )
        )
        if frame_index == 0:
            selected_finufft = sq_grid[tuple(selected_grid_indices.T)]
            selected_direct = evaluate_sq_modes(
                q, xyz, frame_box, selected_modes
            )
            for mode, direct_value, finufft_value in zip(
                selected_modes, selected_direct, selected_finufft
            ):
                absolute = abs(float(finufft_value - direct_value))
                accuracy_rows.append(
                    {
                        "pilot_frame_index_zero_based": 0,
                        "timestep": timestep,
                        "mode_x": int(mode[0]),
                        "mode_y": int(mode[1]),
                        "mode_z": int(mode[2]),
                        "sq_direct": float(direct_value),
                        "sq_finufft": float(finufft_value),
                        "absolute_difference": absolute,
                        "relative_difference": absolute
                        / max(abs(float(direct_value)), 1.0e-300),
                        "selection_seed": ACCURACY_SEED,
                        "selection_population": (
                            "m outside I_21 with 0<|m|_2<=80"
                        ),
                    }
                )
            inversion_sq_max_abs = float(
                np.max(np.abs(sq_grid - sq_grid[::-1, ::-1, ::-1]))
            )
        del rho, sq_grid
        gc.collect()
        print(
            json.dumps(
                {
                    "stage": "pilot_FINUFFT",
                    "frame_index_zero_based": frame_index,
                    "timestep": timestep,
                    "elapsed_s": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    shell_sq = np.asarray(per_frame_shell_sq, dtype=np.float64)
    if shell_sq.shape != (PILOT_N, radius * radius + 1):
        raise RuntimeError(f"unexpected shell-Sq array shape: {shell_sq.shape}")

    accuracy_direct = np.asarray([row["sq_direct"] for row in accuracy_rows])
    accuracy_finufft = np.asarray([row["sq_finufft"] for row in accuracy_rows])
    accuracy_difference = accuracy_finufft - accuracy_direct
    accuracy_max_abs = float(np.max(np.abs(accuracy_difference)))
    accuracy_max_rel = float(
        np.max(np.abs(accuracy_difference) / np.maximum(np.abs(accuracy_direct), 1.0e-300))
    )
    accuracy_relative_rms = float(
        np.sqrt(np.mean(accuracy_difference * accuracy_difference))
        / np.sqrt(np.mean(accuracy_direct * accuracy_direct))
    )
    if accuracy_max_abs >= 1.0e-9 or accuracy_relative_rms >= 1.0e-10:
        raise RuntimeError(
            "FINUFFT/direct structure-factor crosscheck failed: "
            f"max_abs={accuracy_max_abs}, relative_rms={accuracy_relative_rms}"
        )

    beta = 2.0 * math.pi / box_length
    squared_shell = np.arange(1, MAX_TAIL_RADIUS * MAX_TAIL_RADIUS + 1)
    shell_k = beta * np.sqrt(squared_shell.astype(np.float64))
    intra_sq = intramolecular_sq(shell_k)
    if np.any(~np.isfinite(intra_sq)) or np.min(intra_sq) < 0.0:
        raise FloatingPointError("invalid rigid-SPC/E intramolecular form factor")

    scale = ref.COULOMB_REAL * qsum / (math.sqrt(natoms) * box_length**3)
    block_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for split_input_tolerance, csplit in SPLIT_CASES:
        pswf = ExactPSWFContinuation(csplit, QUADRATURE_ORDER)
        attenuation = pswf.attenuation(RCUT * shell_k / csplit)
        term = 16.0 * math.pi**2 * attenuation * attenuation / (shell_k * shell_k)
        amplitude = 2.0 * pswf.constants.psi1 / (
            pswf.constants.eigenvalue * pswf.constants.psi0 * RCUT
        )

        tail_data: dict[int, dict[str, float]] = {}
        for tail_radius in TAIL_RADII:
            upper = tail_radius * tail_radius
            lower_k = beta * (tail_radius + 0.5)
            corrected_integral, integral_metadata = intramolecular_continuum_integral(
                lower_k, RCUT
            )
            continuum = (
                8.0
                * box_length**3
                * amplitude
                * amplitude
                * corrected_integral
            )
            homogeneous_continuum = (
                8.0
                * box_length**3
                * amplitude
                * amplitude
                * integral_metadata["homogeneous_integral"]
            )
            baseline = float(
                np.dot(omitted_counts[1 : upper + 1], term[:upper] * intra_sq[:upper])
                + continuum
            )
            homogeneous = float(
                np.dot(omitted_counts[1 : upper + 1], term[:upper])
                + homogeneous_continuum
            )
            if baseline <= 0.0 or homogeneous <= 0.0:
                raise FloatingPointError("non-positive Fourier-tail chi-squared factor")
            tail_data[tail_radius] = {
                "baseline": baseline,
                "homogeneous": homogeneous,
                "continuum": continuum,
                "homogeneous_continuum": homogeneous_continuum,
                "continuum_ratio": integral_metadata[
                    "corrected_over_homogeneous"
                ],
            }

        # This independently checks the homogeneous lattice and continuum
        # normalization against the established Eq. (46) implementation.
        independent_homogeneous, _ = discrete_eq46_sum(
            pswf,
            box_length,
            RCUT,
            RECIPROCAL_MESH,
            shell_counts,
            MAX_TAIL_RADIUS,
        )
        homogeneous_relative_difference = abs(
            tail_data[MAX_TAIL_RADIUS]["homogeneous"] - independent_homogeneous
        ) / independent_homogeneous
        if homogeneous_relative_difference >= 2.0e-12:
            raise RuntimeError(
                "homogeneous tail normalization check failed: "
                f"{homogeneous_relative_difference}"
            )

        prediction_cache: dict[tuple[int, int], dict[str, object]] = {}
        for measured_radius in MEASURED_RADII:
            explicit_upper = measured_radius * measured_radius
            model_shell_sum = (
                omitted_counts[: explicit_upper + 1]
                * np.concatenate(([1.0], intra_sq[:explicit_upper]))
            )
            correction = (
                shell_sq[:, : explicit_upper + 1] - model_shell_sum[None, :]
            ) @ np.concatenate(([0.0], term[:explicit_upper]))

            for tail_radius in TAIL_RADII:
                chi_by_frame = tail_data[tail_radius]["baseline"] + correction
                if np.any(~np.isfinite(chi_by_frame)) or np.min(chi_by_frame) <= 0.0:
                    raise FloatingPointError("invalid frame-level corrected Fourier chi squared")
                prediction = scale * math.sqrt(float(np.mean(chi_by_frame)))
                block_predictions, block_sem = block_prediction_sem(chi_by_frame, scale)
                if not math.isclose(
                    prediction,
                    math.sqrt(float(np.mean(np.asarray(block_predictions) ** 2))),
                    rel_tol=2.0e-14,
                    abs_tol=0.0,
                ):
                    raise RuntimeError("block and full-pilot pooled predictions disagree")
                prediction_cache[(measured_radius, tail_radius)] = {
                    "prediction": prediction,
                    "block_predictions": block_predictions,
                    "block_sem": block_sem,
                    "chi_by_frame": chi_by_frame,
                }
                convergence_rows.append(
                    {
                        "c_split": csplit,
                        "split_input_tolerance_table_provenance": split_input_tolerance,
                        "measured_lattice_radius": measured_radius,
                        "intramolecular_lattice_tail_radius": tail_radius,
                        "pilot_frames": PILOT_N,
                        "pilot_block_count": PILOT_N // BLOCK_SIZE,
                        "absolute_rms_prediction_kcal_per_mol_A": prediction,
                        "block5_sem_kcal_per_mol_A": block_sem,
                        "intramolecular_baseline_chi2": tail_data[tail_radius][
                            "baseline"
                        ],
                        "analytic_continuum_chi2": tail_data[tail_radius][
                            "continuum"
                        ],
                        "analytic_continuum_fraction_of_corrected_chi2": (
                            tail_data[tail_radius]["continuum"]
                            / float(np.mean(chi_by_frame))
                        ),
                        "analytic_intramolecular_over_homogeneous_continuum": (
                            tail_data[tail_radius]["continuum_ratio"]
                        ),
                        "mode_set": (
                            "all vector-resolved modes outside I_21 within measured sphere; "
                            "exact radial lattice degeneracies beyond"
                        ),
                    }
                )

        canonical = prediction_cache[(MAX_MEASURED_RADIUS, MAX_TAIL_RADIUS)]
        canonical_chi = np.asarray(canonical["chi_by_frame"])
        for block_index, block_prediction in enumerate(canonical["block_predictions"]):
            start = block_index * BLOCK_SIZE
            stop = start + BLOCK_SIZE
            block_rows.append(
                {
                    "c_split": csplit,
                    "split_input_tolerance_table_provenance": split_input_tolerance,
                    "block_index_zero_based": block_index,
                    "pilot_frame_start_zero_based": start,
                    "pilot_frame_stop_zero_based_inclusive": stop - 1,
                    "timestep_start": timesteps[start],
                    "timestep_stop": timesteps[stop - 1],
                    "frames_in_block": BLOCK_SIZE,
                    "measured_lattice_radius": MAX_MEASURED_RADIUS,
                    "intramolecular_lattice_tail_radius": MAX_TAIL_RADIUS,
                    "block_mean_chi2": float(np.mean(canonical_chi[start:stop])),
                    "block_absolute_rms_prediction_kcal_per_mol_A": block_prediction,
                }
            )

        prediction = float(canonical["prediction"])
        r64_prediction = float(
            prediction_cache[(64, MAX_TAIL_RADIUS)]["prediction"]
        )
        tail240_prediction = float(
            prediction_cache[(MAX_MEASURED_RADIUS, 240)]["prediction"]
        )
        mean_chi = float(np.mean(canonical_chi))
        explicit_upper = MAX_MEASURED_RADIUS * MAX_MEASURED_RADIUS
        model_tail_beyond_measured = float(
            np.dot(
                omitted_counts[explicit_upper + 1 : MAX_TAIL_RADIUS**2 + 1],
                term[explicit_upper: MAX_TAIL_RADIUS**2]
                * intra_sq[explicit_upper: MAX_TAIL_RADIUS**2],
            )
            + tail_data[MAX_TAIL_RADIUS]["continuum"]
        )
        summary_rows.append(
            {
                "c_split": csplit,
                "split_input_tolerance_table_provenance": split_input_tolerance,
                "r_c_A": RCUT,
                "reciprocal_mesh": RECIPROCAL_MESH,
                "reciprocal_half_width": RECIPROCAL_HALF_WIDTH,
                "Kmax_inverse_A": beta * RECIPROCAL_HALF_WIDTH,
                "resolved_factor_Kmax_rc_over_csplit": (
                    beta * RECIPROCAL_HALF_WIDTH * RCUT / csplit
                ),
                "pilot_frames": PILOT_N,
                "measured_lattice_radius": MAX_MEASURED_RADIUS,
                "intramolecular_lattice_tail_radius": MAX_TAIL_RADIUS,
                "homogeneous_prediction_kcal_per_mol_A": (
                    scale
                    * math.sqrt(tail_data[MAX_TAIL_RADIUS]["homogeneous"])
                ),
                "intramolecular_only_prediction_kcal_per_mol_A": (
                    scale * math.sqrt(tail_data[MAX_TAIL_RADIUS]["baseline"])
                ),
                "pilot_sq_prediction_kcal_per_mol_A": prediction,
                "pilot_sq_block5_sem_kcal_per_mol_A": canonical["block_sem"],
                "pilot_sq_over_homogeneous": prediction
                / (
                    scale
                    * math.sqrt(tail_data[MAX_TAIL_RADIUS]["homogeneous"])
                ),
                "relative_prediction_change_R64_to_R80": abs(
                    prediction - r64_prediction
                )
                / prediction,
                "relative_prediction_change_tail_R240_to_R320": abs(
                    prediction - tail240_prediction
                )
                / prediction,
                "model_tail_beyond_R80_fraction_of_corrected_chi2": (
                    model_tail_beyond_measured / mean_chi
                ),
                "analytic_continuum_fraction_of_corrected_chi2": (
                    tail_data[MAX_TAIL_RADIUS]["continuum"] / mean_chi
                ),
                "analytic_intramolecular_over_homogeneous_continuum": (
                    tail_data[MAX_TAIL_RADIUS]["continuum_ratio"]
                ),
                "homogeneous_independent_check_relative_difference": (
                    homogeneous_relative_difference
                ),
                "prediction_status": (
                    "frozen pilot-only prediction; no holdout coordinates or forces accessed"
                ),
            }
        )

    max_measured_radius_change = max(
        float(row["relative_prediction_change_R64_to_R80"])
        for row in summary_rows
    )
    max_tail_radius_change = max(
        float(row["relative_prediction_change_tail_R240_to_R320"])
        for row in summary_rows
    )
    if max_measured_radius_change >= 5.0e-4:
        raise RuntimeError(
            f"measured-Sq radius is not converged: {max_measured_radius_change}"
        )
    if max_tail_radius_change >= 5.0e-4:
        raise RuntimeError(
            f"intramolecular lattice closure is not converged: {max_tail_radius_change}"
        )

    write_csv(BLOCK_OUT, block_rows)
    write_csv(SUMMARY_OUT, summary_rows)
    write_csv(CONVERGENCE_OUT, convergence_rows)
    write_csv(ACCURACY_OUT, accuracy_rows)

    manifest = {
        "path_basis": (
            "bundle_root for distributed inputs; "
            "$ESP_ERROR_BENCH_OUTPUT_DIR/redesigned_section5 for outputs"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_role": (
            "frozen pilot-only SPC/E structure-factor prediction for the "
            "cube-complement Fourier-truncation diagnostic"
        ),
        "holdout_accessed": False,
        "data_partition": {
            "pilot_frame_indices_zero_based": list(range(PILOT_N)),
            "pilot_timesteps": timesteps,
            "parser_stops_after_frame_index_zero_based": PILOT_N - 1,
            "trajectory_bytes_consumed_through_pilot": bytes_consumed,
            "trajectory_total_bytes": TRAJECTORY.stat().st_size,
            "holdout_frame_indices_zero_based": "not parsed or accessed",
            "reference_force_files": "not opened",
        },
        "system": {
            "model": "SPC/E water",
            "n_molecules": 901,
            "n_charge_sites": natoms,
            "box_length_A": box_length,
            "Q_sum_q_squared": qsum,
            "q_O": Q_O,
            "q_H": Q_H,
            "d_OH_A": D_OH,
            "d_HH_A": D_HH,
            "HOH_angle_deg": HOH_ANGLE_DEG,
            "trajectory_pilot_prefix": pilot_prefix_record(
                TRAJECTORY, bytes_consumed
            ),
        },
        "parameters": {
            "r_c_A": RCUT,
            "reciprocal_mesh": RECIPROCAL_MESH,
            "reciprocal_half_width": RECIPROCAL_HALF_WIDTH,
            "Kmax_inverse_A": beta * RECIPROCAL_HALF_WIDTH,
            "c_split": [item[1] for item in SPLIT_CASES],
            "split_input_tolerance_table_provenance": [
                item[0] for item in SPLIT_CASES
            ],
            "measured_lattice_radii": list(MEASURED_RADII),
            "intramolecular_lattice_tail_radii": list(TAIL_RADII),
            "pswf_quadrature_order": QUADRATURE_ORDER,
        },
        "structure_factor": {
            "normalization": "S_q(m)=|rho_m|^2/Q per frame",
            "finufft_type": 1,
            "isign": -1,
            "modeord": FINUFFT_MODEORD,
            "eps": FINUFFT_EPS,
            "nthreads": FINUFFT_NTHREADS,
            "coordinate_map": "x_finufft=2*pi*(mod(r,L)/L-1/2)",
            "measured_mode_accumulation": (
                "all individual vector modes outside I_21 and inside each integer "
                "sphere are evaluated; squared-radius bincount is exact because the "
                "tail multiplier is radial"
            ),
        },
        "high_k_closure": {
            "lattice_model": (
                "rigid orientationally averaged SPC/E intramolecular form factor"
            ),
            "exact_lattice_shells_through": MAX_TAIL_RADIUS,
            "continuum": (
                "leading exact-PSWF sinc continuation with analytic homogeneous, "
                "O-H sinc, and H-H sinc integrals"
            ),
            "I3_formula": (
                "I3(b,K)=b/2*[cos(bK)/K-b*(sgn(b)*pi/2-Si(bK))]"
                "+sin(bK)/(2K^2)"
            ),
            "J_formula": (
                "J(d)=[I3(d)/2-I3(d+2rc)/4-I3(d-2rc)/4]/d"
            ),
        },
        "estimator": {
            "chi_squared": (
                "sum_{m outside I_21} 16*pi^2*a_c(rc|k_m|)^2/|k_m|^2*S_q(k_m)"
            ),
            "force_scale": "k_e*Q/(sqrt(N)*V)",
            "caveat": (
                "diagonal-mode structure-factor estimator; cross-mode and "
                "target-conditioned covariances remain neglected"
            ),
        },
        "statistics": {
            "main_prediction": "scale*sqrt(mean of 25 frame-level chi-squared values)",
            "blocks": "five contiguous nonoverlapping blocks of five pilot frames",
            "sem": "sample standard deviation of five block predictions divided by sqrt(5)",
        },
        "accuracy_crosscheck": {
            "frame_index_zero_based": 0,
            "seed": ACCURACY_SEED,
            "sampled_modes": ACCURACY_MODE_COUNT,
            "population": "m outside I_21 with 0<|m|_2<=80",
            "max_absolute_Sq_difference": accuracy_max_abs,
            "max_relative_Sq_difference": accuracy_max_rel,
            "relative_rms_Sq_difference": accuracy_relative_rms,
            "max_inversion_symmetry_Sq_difference": inversion_sq_max_abs,
        },
        "convergence_checks": {
            "maximum_relative_prediction_change_R64_to_R80": (
                max_measured_radius_change
            ),
            "maximum_relative_prediction_change_tail_R240_to_R320": (
                max_tail_radius_change
            ),
            "acceptance_threshold_for_each": 5.0e-4,
        },
        "environment": environment,
        "code": [
            file_record(Path(__file__).resolve()),
            file_record(HERE / "fig2_fourier_reference.py"),
            file_record(HERE / "fixed_ik_reference.py"),
            file_record(HERE / "sq_alias_tools.py"),
        ],
        "outputs": [
            file_record(BLOCK_OUT),
            file_record(SUMMARY_OUT),
            file_record(CONVERGENCE_OUT),
            file_record(ACCURACY_OUT),
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Read-back guards catch malformed headers or accidental row-count changes.
    expected_rows = {
        BLOCK_OUT: len(SPLIT_CASES) * (PILOT_N // BLOCK_SIZE),
        SUMMARY_OUT: len(SPLIT_CASES),
        CONVERGENCE_OUT: len(SPLIT_CASES) * len(MEASURED_RADII) * len(TAIL_RADII),
        ACCURACY_OUT: ACCURACY_MODE_COUNT,
    }
    for path, expected in expected_rows.items():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected:
            raise RuntimeError(f"{path.name}: expected {expected} rows, found {len(rows)}")

    print(
        json.dumps(
            {
                "stage": "complete",
                "holdout_accessed": False,
                "summary_rows": len(summary_rows),
                "convergence_rows": len(convergence_rows),
                "accuracy_rows": len(accuracy_rows),
                "max_R64_to_R80": max_measured_radius_change,
                "max_tail_R240_to_R320": max_tail_radius_change,
                "finufft_relative_rms": accuracy_relative_rms,
                "elapsed_s": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
