#!/usr/bin/env python3
"""Compare the validation-only LAMMPS fixed-influence ik path with NumPy.

The standalone calculation intentionally reproduces the implementation
convention in ``ESP::compute_gf_fixed_ik``: project-local Fourier polynomials
inside their declared radial support and zero outside.  It is therefore a
code-path/sign/normalization test, not a test of the exact PSWF outside
continuation used by the manuscript's analytical alias estimator.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REDESIGN = HERE.parent
PROJECT = HERE.parents[3]
sys.path.insert(0, str(REDESIGN))

import fixed_ik_reference as ref  # noqa: E402


MESH = 24
ORDER = 6
BOX = 48.0
RCUT = 9.0
CSPLIT = 12.024
CSPREAD = 12.024
SPLIT_TOL = 1.0e-4
SPREAD_TOL = 1.0e-4


def parse_dump_force(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    marker = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    columns = lines[marker].split()[2:]
    col = {name: i for i, name in enumerate(columns)}
    rows = []
    for line in lines[marker + 1 :]:
        fields = line.split()
        if not fields or fields[0] == "ITEM:":
            break
        rows.append((int(fields[col["id"]]), *(float(fields[col[key]]) for key in ("fx", "fy", "fz"))))
    rows.sort(key=lambda row: row[0])
    return np.asarray([row[1:] for row in rows], dtype=np.float64)


def lammps_polynomial_fixed_force(
    q: np.ndarray, xyz: np.ndarray, coeff: ref.PSWFCoefficients
) -> np.ndarray:
    _, k1, kx, ky, kz, k2 = ref.reciprocal_arrays(MESH, BOX)

    kernel = np.zeros_like(k2)
    active = (k2 > 0.0) & (k2 <= (CSPLIT / RCUT) ** 2)
    u = RCUT * np.sqrt(k2[active]) / CSPLIT
    kernel[active] = 2.0 * math.pi * ref.horner_array(2.0 * u - 1.0, coeff.split) / k2[active]

    h = BOX / MESH
    t = 0.5 * ORDER * h * np.abs(k1) / CSPREAD
    w1 = np.zeros_like(t)
    inside = t <= 1.0
    w1[inside] = 0.5 * ORDER * ref.horner_array(2.0 * t[inside] - 1.0, coeff.spread)
    window = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]

    green = np.zeros_like(kernel)
    stable = active & (window != 0.0)
    green[stable] = kernel[stable] / window[stable] ** 2

    stencil = ref.particle_stencil(xyz, MESH, BOX, ORDER, coeff.real)
    density = ref.spread_density(q, stencil, MESH, BOX)
    rho_hat = np.fft.fftn(density)
    field = np.empty((MESH, MESH, MESH, 3), dtype=np.float64)
    for dim, kd in enumerate((kx, ky, kz)):
        field[..., dim] = np.fft.ifftn((-1j * kd) * green * rho_hat).real
    return ref.COULOMB_REAL * ref.gather_vector_field(q, stencil, field)


def main() -> None:
    data = PROJECT / "numerical_examples/random_charges/config_01/random_charges.data"
    dump = HERE / "work" / "forces.esp_fixed_test.dump"
    if not dump.is_file():
        raise FileNotFoundError(f"run the patched LAMMPS smoke input first: {dump}")

    q, xyz, box = ref.parse_charge_data(data)
    if not math.isclose(box, BOX):
        raise RuntimeError(f"unexpected box length: {box}")
    coeff = ref.load_coefficients(
        0.1 * SPLIT_TOL, 0.1 * SPREAD_TOL, CSPLIT, CSPREAD, ORDER
    )
    expected = lammps_polynomial_fixed_force(q, xyz, coeff)
    continuation_force, _ = ref.fixed_ik_mesh_force(
        q, xyz, box, MESH, ORDER, RCUT, CSPLIT, CSPREAD, coeff
    )
    observed = parse_dump_force(dump)
    if expected.shape != observed.shape:
        raise RuntimeError(f"shape mismatch: {expected.shape} != {observed.shape}")

    difference = observed - expected
    reversed_difference = observed + expected
    scale = float(np.sqrt(np.mean(np.sum(expected * expected, axis=1))))
    continuation_delta = ref.rms_vector_error(expected, continuation_force)
    result = {
        "system": "random_charges/config_01, N=512, L=48 A",
        "mesh": MESH,
        "order": ORDER,
        "csplit": CSPLIT,
        "cspread": CSPREAD,
        "comparison_convention": "inside-support MathPSWF polynomials; zero outside",
        "reference_force_rms": scale,
        "difference_rms": ref.rms_vector_error(observed, expected),
        "relative_rms_difference": ref.rms_vector_error(observed, expected) / scale,
        "maximum_absolute_component_difference": float(np.max(np.abs(difference))),
        "sign_reversed_rms_difference": float(
            np.sqrt(np.mean(np.sum(reversed_difference * reversed_difference, axis=1)))
        ),
        "polynomial_green_vs_exact_real_window_transform_rms": continuation_delta,
        "polynomial_green_vs_exact_real_window_transform_relative_rms": continuation_delta
        / scale,
        "pass_threshold": "relative_rms_difference < 5e-11",
    }
    result["passed"] = bool(result["relative_rms_difference"] < 5.0e-11)
    output = HERE / "validation_result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
