#!/usr/bin/env python3
"""Generate the volume-normalized physical charge spectrum reported in the SI.

Every reciprocal-lattice mode in each 0.1-Angstrom^-1 spherical bin is
included through |k| = 10 Angstrom^-1.  The calculation uses the same pinned
FINUFFT type-1 path that was independently checked against direct mode sums
for Figure 2.  This script owns only the panel-a spectrum source and cannot
overwrite any of the force-error data used in Figure 4b.

Run with an environment containing the pinned dependencies::

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
      python3 \
      run_fig4_charge_spectrum.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
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

import fixed_ik_reference as ref
import sq_alias_tools as sqtools


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
TRAJECTORY = (
    PROJECT_ROOT
    / "numerical_examples"
    / "water_trajectory_benchmark"
    / "water_short_traj.lammpstrj"
)
RANDOM_ROOT = PROJECT_ROOT / "numerical_examples" / "random_charges"

SPECTRUM_OUT = HERE / "fig4_charge_spectrum_source.csv"
MANIFEST_OUT = HERE / "fig4_charge_spectrum_manifest.json"

PILOT_N = 25
RANDOM_N = 10
BLOCK_SIZE = 5
K_MAX = 10.0
BIN_WIDTH = 0.1
FINUFFT_EPS = 1.0e-12
FINUFFT_NTHREADS = 1
FINUFFT_MODEORD = 0
FINUFFT_REQUIRED_VERSION = "2.5.0"
ACCURACY_MODE_COUNT = 256
ACCURACY_SEED = 20260819


def file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
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


def block_sem(values: np.ndarray, block_size: int = BLOCK_SIZE) -> float:
    count = len(values) // block_size
    if count < 2:
        return float("nan")
    means = np.asarray(
        [values[i * block_size : (i + 1) * block_size].mean() for i in range(count)]
    )
    return float(means.std(ddof=1) / math.sqrt(count))


def finufft_environment() -> dict[str, object]:
    if finufft.__version__ != FINUFFT_REQUIRED_VERSION:
        raise RuntimeError(
            f"requires FINUFFT {FINUFFT_REQUIRED_VERSION}, found {finufft.__version__}; "
            "run with a Python environment that provides the required version"
        )
    return {
        "path_basis": "active Python environment prefix",
        "python_executable": environment_relative(Path(sys.executable)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "finufft_version": finufft.__version__,
        "finufft_module": environment_relative(Path(finufft.__file__)),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "requested_finufft_nthreads": FINUFFT_NTHREADS,
    }


def reciprocal_bin_geometry(
    box_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return the centered grid mask and exact physical-|k| bin mapping."""

    radius = int(math.ceil(K_MAX * box_length / (2.0 * math.pi)))
    grid_size = 2 * radius + 1
    integer_axis = np.arange(-radius, radius + 1, dtype=np.int32)
    squared_radius = (
        integer_axis[:, None, None] ** 2
        + integer_axis[None, :, None] ** 2
        + integer_axis[None, None, :] ** 2
    ).astype(np.int32)
    kmag = (2.0 * math.pi / box_length) * np.sqrt(squared_radius)
    mask = (squared_radius > 0) & (kmag <= K_MAX + 1.0e-13)
    edges = np.linspace(0.0, K_MAX, int(round(K_MAX / BIN_WIDTH)) + 1)
    selected_k = kmag[mask]
    bin_index = np.searchsorted(edges, selected_k, side="right") - 1
    bin_index = np.minimum(bin_index, len(edges) - 2)
    counts = np.bincount(bin_index, minlength=len(edges) - 1)
    return mask, bin_index, counts, edges, squared_radius, grid_size


def one_configuration_spectrum(
    q: np.ndarray,
    xyz: np.ndarray,
    box_length: float,
    mask: np.ndarray,
    bin_index: np.ndarray,
    counts: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    angles = [
        2.0 * math.pi * (np.mod(xyz[:, dim], box_length) / box_length - 0.5)
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
    density_grid = (rho.real * rho.real + rho.imag * rho.imag) / box_length**3
    sums = np.bincount(
        bin_index,
        weights=density_grid[mask],
        minlength=len(counts),
    )
    binned = np.full(len(counts), np.nan, dtype=np.float64)
    populated = counts > 0
    binned[populated] = sums[populated] / counts[populated]
    return binned, density_grid


def direct_crosscheck(
    q: np.ndarray,
    xyz: np.ndarray,
    box_length: float,
    mask: np.ndarray,
    density_grid: np.ndarray,
    radius: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    flat_population = np.flatnonzero(mask.ravel())
    selected_flat = rng.choice(
        flat_population,
        size=min(ACCURACY_MODE_COUNT, len(flat_population)),
        replace=False,
    )
    selected_indices = np.column_stack(np.unravel_index(selected_flat, mask.shape))
    selected_modes = selected_indices.astype(np.int64) - radius
    direct = (
        sqtools.evaluate_sq_modes(q, xyz, box_length, selected_modes)
        * float(np.sum(q * q))
        / box_length**3
    )
    transformed = density_grid[tuple(selected_indices.T)]
    difference = transformed - direct
    return {
        "n_modes": len(selected_modes),
        "max_absolute_difference_e2_Angstrom-3": float(np.max(np.abs(difference))),
        "relative_rms_difference": float(
            np.sqrt(np.mean(difference * difference))
            / np.sqrt(np.mean(direct * direct))
        ),
    }


def evaluate_system(
    systems: list[tuple[np.ndarray, np.ndarray]],
    system_name: str,
    box_length: float,
    uncertainty_mode: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    mask, bin_index, counts, edges, squared_radius, grid_size = reciprocal_bin_geometry(
        box_length
    )
    radius = (grid_size - 1) // 2
    per_configuration = []
    crosscheck: dict[str, float | int] | None = None
    q_reference = systems[0][0]
    qsum = float(np.sum(q_reference * q_reference))
    charge_squared_density = qsum / box_length**3

    started = time.perf_counter()
    for index, (q, xyz) in enumerate(systems):
        if not math.isclose(
            float(np.sum(q * q)), qsum, rel_tol=1.0e-13, abs_tol=1.0e-13
        ):
            raise RuntimeError(f"Q changed within {system_name}")
        binned, density_grid = one_configuration_spectrum(
            q, xyz, box_length, mask, bin_index, counts, grid_size
        )
        per_configuration.append(binned)
        if index == 0:
            crosscheck = direct_crosscheck(
                q, xyz, box_length, mask, density_grid, radius, rng
            )
        print(
            json.dumps(
                {
                    "stage": "volume_normalized_spectrum",
                    "system": system_name,
                    "configuration_zero_based": index,
                    "grid_size": grid_size,
                    "elapsed_s": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    values = np.asarray(per_configuration)
    rows: list[dict[str, object]] = []
    for bin_number, count in enumerate(counts):
        if count == 0:
            continue
        sample = values[:, bin_number]
        if uncertainty_mode == "configuration-sem":
            sem = float(sample.std(ddof=1) / math.sqrt(len(sample)))
            uncertainty = "SEM over independently generated configurations"
            n_blocks: int | str = ""
        elif uncertainty_mode == "block5-sem":
            sem = block_sem(sample)
            uncertainty = (
                "SEM over nonoverlapping five-frame block means; "
                "not autocorrelation-corrected"
            )
            n_blocks = len(sample) // BLOCK_SIZE
        else:
            raise ValueError(f"unknown uncertainty mode: {uncertainty_mode}")
        rows.append(
            {
                "system": system_name,
                "k_center": 0.5 * (edges[bin_number] + edges[bin_number + 1]),
                "k_lower": edges[bin_number],
                "k_upper": edges[bin_number + 1],
                "cq_volume_mean": float(sample.mean()),
                "cq_volume_sem": sem,
                "charge_squared_density": charge_squared_density,
                "n_reciprocal_modes": int(count),
                "n_samples": len(sample),
                "n_blocks": n_blocks,
                "uncertainty": uncertainty,
                "normalization": "C_q(k)=<|rho(k)|^2>/V",
                "units": "e^2 Angstrom^-3",
            }
        )
    assert crosscheck is not None
    if (
        crosscheck["max_absolute_difference_e2_Angstrom-3"] >= 1.0e-9
        or crosscheck["relative_rms_difference"] >= 1.0e-10
    ):
        raise RuntimeError(f"FINUFFT/direct crosscheck failed for {system_name}: {crosscheck}")
    metadata: dict[str, object] = {
        "system": system_name,
        "box_length_Angstrom": box_length,
        "Q_e2": qsum,
        "volume_Angstrom3": box_length**3,
        "Q_over_V_e2_Angstrom-3": charge_squared_density,
        "grid_size": grid_size,
        "integer_radius": radius,
        "sphere_mode_count": int(np.count_nonzero(mask)),
        "squared_radius_grid_shape": list(squared_radius.shape),
        "direct_crosscheck": crosscheck,
    }
    return rows, metadata


def main() -> None:
    started = time.perf_counter()
    environment = finufft_environment()
    frames = ref.parse_charge_trajectory(TRAJECTORY)
    if len(frames) != 50:
        raise RuntimeError(f"expected 50 water frames, found {len(frames)}")
    pilot = frames[:PILOT_N]
    water_box = pilot[0][3]
    water_systems = [(q, xyz) for _, q, xyz, box in pilot if box == water_box]
    if len(water_systems) != PILOT_N:
        raise RuntimeError("water box changed within frames 1--25")

    random_systems = []
    random_box = None
    for index in range(1, RANDOM_N + 1):
        q, xyz, box = ref.parse_charge_data(
            RANDOM_ROOT / f"config_{index:02d}" / "random_charges.data"
        )
        if random_box is None:
            random_box = box
        elif not math.isclose(box, random_box, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError("random-charge box changed across configurations")
        random_systems.append((q, xyz))
    assert random_box is not None

    rng = np.random.default_rng(ACCURACY_SEED)
    random_rows, random_metadata = evaluate_system(
        random_systems,
        "random charges",
        random_box,
        "configuration-sem",
        rng,
    )
    water_rows, water_metadata = evaluate_system(
        water_systems,
        "SPC/E water (frames 1--25)",
        water_box,
        "block5-sem",
        rng,
    )
    rows = random_rows + water_rows
    with SPECTRUM_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    MANIFEST_OUT.write_text(
        json.dumps(
            {
                "path_basis": "bundle_root",
                "purpose": "Supporting Information volume-normalized physical charge-spectrum diagnostic",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "generator": file_record(Path(__file__)),
                "utilities": [
                    file_record(HERE / "fixed_ik_reference.py"),
                    file_record(HERE / "sq_alias_tools.py"),
                ],
                "trajectory": file_record(TRAJECTORY),
                "water_frame_indices_zero_based": list(range(PILOT_N)),
                "random_configuration_indices_one_based": list(
                    range(1, RANDOM_N + 1)
                ),
                "normalization": "C_q(k)=<|rho(k)|^2>/V",
                "units": "e^2 Angstrom^-3",
                "k_max_Angstrom-1": K_MAX,
                "bin_width_Angstrom-1": BIN_WIDTH,
                "mode_population": (
                    "all nonzero reciprocal-lattice modes with |k|<=10 Angstrom^-1; "
                    "no angular subsampling"
                ),
                "finufft_transform": {
                    "type": 1,
                    "eps": FINUFFT_EPS,
                    "isign": -1,
                    "modeord": FINUFFT_MODEORD,
                    "coordinate_map": "x=2*pi*(mod(r,L)/L-1/2)",
                },
                "random_system": random_metadata,
                "water_system": water_metadata,
                "environment": environment,
                "output": file_record(SPECTRUM_OUT),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )
    print(SPECTRUM_OUT)


if __name__ == "__main__":
    main()
