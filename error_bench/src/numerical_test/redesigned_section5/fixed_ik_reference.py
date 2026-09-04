#!/usr/bin/env python3
"""Reference fixed-influence ik operator for the redesigned Section 5.

The implementation deliberately mirrors the mathematical operator rather
than the Hockney--Eastwood influence function in the development LAMMPS ESP
solver.  Both the fixed mesh force and the direct truncated force use the same
split multiplier and the same canonical FFT reciprocal set.
"""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


COULOMB_REAL = 332.06371
WINDOW_FOURIER_QUADRATURE_POINTS_PER_PIECE = 64
MAX_VERIFIED_AXIAL_ALIAS_INDEX = 12
CONTINUOUS_BASE_WINDOW_SCAN_POINTS = 2049
CONTINUOUS_BASE_WINDOW_RELATIVE_ZERO_TOL = 64.0 * np.finfo(np.float64).eps
HERE = Path(__file__).resolve().parent
NUMERICAL_ROOT = HERE.parent
COEFF_SOURCE = NUMERICAL_ROOT / "dump_pswf_coeff.cpp"
PSWF_SOURCE_DIR = NUMERICAL_ROOT / "lammps_math_pswf"
COEFF_EXE = HERE / "dump_pswf_coeff"


def horner_scalar(x: float, coeff: np.ndarray) -> float:
    value = float(coeff[-1])
    for item in coeff[-2::-1]:
        value = value * x + float(item)
    return value


def horner_array(x: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    value = np.full_like(np.asarray(x, dtype=np.float64), float(coeff[-1]))
    for item in coeff[-2::-1]:
        value = value * x + float(item)
    return value


def ensure_coeff_executable() -> Path:
    required = (
        COEFF_SOURCE,
        PSWF_SOURCE_DIR / "math_pswf.cpp",
        PSWF_SOURCE_DIR / "math_pswf.h",
        PSWF_SOURCE_DIR / "math_const.h",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing MathPSWF inputs: {missing}")
    newest = max(path.stat().st_mtime for path in required)
    if COEFF_EXE.is_file() and COEFF_EXE.stat().st_mtime >= newest:
        return COEFF_EXE
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            f"-I{PSWF_SOURCE_DIR}",
            str(COEFF_SOURCE),
            str(PSWF_SOURCE_DIR / "math_pswf.cpp"),
            "-o",
            str(COEFF_EXE),
        ],
        check=True,
        cwd=HERE,
    )
    return COEFF_EXE


@dataclass(frozen=True)
class PSWFCoefficients:
    split: np.ndarray
    split_lambda: float
    spread: np.ndarray
    spread_lambda: float
    real: np.ndarray


def load_coefficients(
    split_tol: float,
    spread_tol: float,
    csplit: float,
    cspread: float,
    order: int,
) -> PSWFCoefficients:
    exe = ensure_coeff_executable()
    out = subprocess.run(
        [
            str(exe),
            f"{split_tol:.17g}",
            f"{spread_tol:.17g}",
            f"{csplit:.17g}",
            f"{cspread:.17g}",
            str(order),
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    values: dict[str, list[float]] = {"split": [], "spread": [], "real": []}
    lambdas: dict[str, float] = {}
    for kind, index, value in csv.reader(out.splitlines()):
        if index == "lambda":
            lambdas[kind] = float(value)
        else:
            values[kind].append(float(value))
    if not all(values.values()) or "split" not in lambdas or "spread" not in lambdas:
        raise RuntimeError("incomplete output from dump_pswf_coeff")
    real = np.asarray(values["real"], dtype=np.float64).reshape((-1, order))
    return PSWFCoefficients(
        split=np.asarray(values["split"], dtype=np.float64),
        split_lambda=lambdas["split"],
        spread=np.asarray(values["spread"], dtype=np.float64),
        spread_lambda=lambdas["spread"],
        real=real,
    )


class EvenPSWFContinuation:
    """Finite-Fourier continuation generated from one inside representation."""

    def __init__(
        self,
        c: float,
        inside,
        eigenvalue: float,
        quadrature_order: int = 384,
    ) -> None:
        self.c = float(c)
        self.inside = inside
        self.eigenvalue = float(eigenvalue)
        self.xq, self.wq = np.polynomial.legendre.leggauss(quadrature_order)
        self.inside_q = np.asarray(self.inside(np.abs(self.xq)), dtype=np.float64)

    def value(self, u) -> np.ndarray:
        u_arr = np.abs(np.asarray(u, dtype=np.float64))
        flat = u_arr.ravel()
        result = np.empty_like(flat)
        mask = flat <= 1.0
        if np.any(mask):
            result[mask] = self.inside(flat[mask])
        outside = np.flatnonzero(~mask)
        # Chunk the dense quadrature matrix so c_spread sweeps remain modest in
        # memory.  The continuation is real because the ground-state PSWF is
        # even.
        for start in range(0, len(outside), 4096):
            idx = outside[start : start + 4096]
            phase = np.cos(self.c * np.outer(flat[idx], self.xq))
            result[idx] = phase @ (self.wq * self.inside_q) / self.eigenvalue
        return result.reshape(u_arr.shape)


def split_continuation(coeff: PSWFCoefficients, csplit: float) -> EvenPSWFContinuation:
    # MathPSWF::fourier_poly represents lambda*psi(u)/C0 = 2*a_c(c*u)
    # on 0 <= u <= 1, using the mapped monomial coordinate 2*u-1.
    def inside(u):
        return horner_array(2.0 * np.asarray(u) - 1.0, coeff.split)

    return EvenPSWFContinuation(csplit, inside, coeff.split_lambda)


def piecewise_real_window(coeff: PSWFCoefficients, order: int):
    """Evaluate the same piecewise polynomial used for particle spreading."""

    def inside(u):
        values = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        s = (values + 1.0) * order / 2.0
        lane = np.clip(np.floor(s).astype(np.int64), 0, order - 1)
        dx = 0.5 * order * values + 0.5 * order - lane - 0.5
        result = np.empty_like(values)
        for j in range(order):
            mask = lane == j
            if np.any(mask):
                result[mask] = horner_array(dx[mask], coeff.real[:, j])
        return result

    return inside


def window_transform_continuation(
    coeff: PSWFCoefficients,
    cspread: float,
    order: int,
    *,
    quadrature_points_per_piece: int = WINDOW_FOURIER_QUADRATURE_POINTS_PER_PIECE,
) -> EvenPSWFContinuation:
    # The integral of the real spreading polynomial is used at all frequencies
    # so that the fixed multiplier and the actual spreading kernel share one
    # representation.  For an exact PSWF this integral is lambda*psi(u).
    if (
        isinstance(quadrature_points_per_piece, bool)
        or not isinstance(quadrature_points_per_piece, (int, np.integer))
        or quadrature_points_per_piece < 1
    ):
        raise ValueError("quadrature_points_per_piece must be a positive integer")
    real_inside = piecewise_real_window(coeff, order)

    # Integrate each polynomial lane separately.  A single global Gauss rule
    # converges slowly once high alias orders make the cosine strongly
    # oscillatory, because derivatives jump at the lane boundaries.  The
    # piecewise rule resolves those boundaries exactly and is stable through
    # the verified axial-image cutoff below.
    local_x, local_w = np.polynomial.legendre.leggauss(
        quadrature_points_per_piece
    )
    boundaries = [0.0]
    boundaries.extend(
        boundary
        for lane in range(1, order)
        if 0.0 < (boundary := 2.0 * lane / order - 1.0) < 1.0
    )
    boundaries.append(1.0)
    positive_x = []
    positive_w = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        positive_x.append(
            0.5 * (upper - lower) * local_x + 0.5 * (upper + lower)
        )
        positive_w.append(0.5 * (upper - lower) * local_w)
    positive_x = np.concatenate(positive_x)
    positive_w = np.concatenate(positive_w)
    xq = np.concatenate((-positive_x[::-1], positive_x))
    wq = np.concatenate((positive_w[::-1], positive_w))
    pq = real_inside(np.abs(xq))
    integral0 = float(np.dot(wq, pq))
    # Reuse EvenPSWFContinuation by passing the normalized eigenfunction-like
    # inside value I(u); its outside rule divides by eigenvalue, so lambda=1.
    class WindowTransform(EvenPSWFContinuation):
        def __init__(self):
            self.c = float(cspread)
            self.xq = xq
            self.wq = wq
            self.inside_q = pq
            self.eigenvalue = 1.0
            self.inside = self._integral
            self.integral0 = integral0

        def _integral(self, u):
            arr = np.asarray(u, dtype=np.float64)
            shape = arr.shape
            flat = arr.ravel()
            out = np.empty_like(flat)
            for start in range(0, len(flat), 4096):
                sl = slice(start, start + 4096)
                out[sl] = np.cos(self.c * np.outer(flat[sl], self.xq)) @ (
                    self.wq * self.inside_q
                )
            return out.reshape(shape)

        def value(self, u):
            return self._integral(np.abs(np.asarray(u, dtype=np.float64)))

    return WindowTransform()


def fft_modes(mesh: int, box_length: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.rint(np.fft.fftfreq(mesh) * mesh).astype(np.int64)
    return indices, (2.0 * math.pi / box_length) * indices.astype(np.float64)


def reciprocal_arrays(mesh: int, box_length: float):
    indices, k1 = fft_modes(mesh, box_length)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    return indices, k1, kx, ky, kz, k2


def split_kernel_grid(
    mesh: int,
    box_length: float,
    rcut: float,
    csplit: float,
    continuation: EvenPSWFContinuation,
    band_limited: bool = False,
) -> np.ndarray:
    _, _, _, _, _, k2 = reciprocal_arrays(mesh, box_length)
    kernel = np.zeros_like(k2)
    u_all = np.zeros_like(k2)
    u_all[k2 > 0.0] = rcut * np.sqrt(k2[k2 > 0.0]) / csplit
    nonzero = k2 > 0.0
    if band_limited:
        # The practical truncated smooth sum and Eqs. (60)/(67) retain the
        # nominal PSWF band only.  The omitted outside continuation is the
        # Fourier-truncation term validated separately in Fig. 2.
        nonzero &= u_all <= 1.0
    u = u_all[nonzero]
    # continuation.value = lambda*psi/C0 = 2*a_c; therefore 2*pi*value/k^2
    # is exactly 4*pi*a_c/k^2, as in the manuscript.
    kernel[nonzero] = 2.0 * math.pi * continuation.value(u) / k2[nonzero]
    return kernel


def parse_charge_data(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    lines = path.read_text().splitlines()
    bounds = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[-2:] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
            bounds.append((float(fields[0]), float(fields[1])))
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("Atoms")) + 1
    rows = []
    for line in lines[start:]:
        fields = line.split()
        if not fields:
            continue
        if not fields[0].lstrip("+-").isdigit():
            if rows:
                break
            continue
        if len(fields) < 6:
            continue
        rows.append((int(fields[0]), float(fields[2]), *(float(x) for x in fields[3:6])))
    rows.sort(key=lambda row: row[0])
    if len(bounds) != 3 or not rows:
        raise RuntimeError(f"could not parse charge data: {path}")
    lengths = np.array([hi - lo for lo, hi in bounds])
    if not np.allclose(lengths, lengths[0]):
        raise ValueError("reference implementation currently requires a cubic cell")
    q = np.asarray([row[1] for row in rows], dtype=np.float64)
    xyz = np.asarray([row[2:] for row in rows], dtype=np.float64)
    xyz -= np.asarray([lo for lo, _ in bounds])
    return q, xyz, float(lengths[0])


def parse_charge_trajectory(path: Path):
    frames = []
    with path.open() as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            if not header.startswith("ITEM: TIMESTEP"):
                raise RuntimeError(f"malformed trajectory header in {path}")
            timestep = int(handle.readline())
            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("missing atom count")
            natoms = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("missing bounds")
            bounds = [tuple(map(float, handle.readline().split()[:2])) for _ in range(3)]
            columns = handle.readline().split()[2:]
            col = {name: i for i, name in enumerate(columns)}
            q = np.empty(natoms, dtype=np.float64)
            xyz = np.empty((natoms, 3), dtype=np.float64)
            ids = np.empty(natoms, dtype=np.int64)
            for i in range(natoms):
                fields = handle.readline().split()
                ids[i] = int(fields[col["id"]])
                q[i] = float(fields[col["q"]])
                xyz[i] = [float(fields[col[name]]) for name in ("x", "y", "z")]
            order = np.argsort(ids)
            lo = np.asarray([item[0] for item in bounds])
            lengths = np.asarray([item[1] - item[0] for item in bounds])
            if not np.allclose(lengths, lengths[0]):
                raise ValueError("reference implementation currently requires a cubic cell")
            frames.append((timestep, q[order], xyz[order] - lo, float(lengths[0])))
    return frames


def parse_charge_trajectory_prefix(
    path: Path, count: int, *, return_sha256: bool = False
):
    """Read exactly the first ``count`` trajectory frames and then stop.

    The file is opened without userspace buffering so the prediction stage
    does not read ahead into a holdout frame.  When requested, the digest is
    over the exact byte prefix ending at the final atom record of frame
    ``count``.
    """

    if count < 1:
        raise ValueError("count must be positive")
    import hashlib

    digest = hashlib.sha256()
    frames = []
    with path.open("rb", buffering=0) as handle:
        def line() -> bytes:
            raw = handle.readline()
            digest.update(raw)
            return raw

        while len(frames) < count:
            header = line()
            if not header:
                raise RuntimeError(
                    f"trajectory {path} contains fewer than {count} frames"
                )
            if not header.startswith(b"ITEM: TIMESTEP"):
                raise RuntimeError(f"malformed trajectory header in {path}")
            timestep = int(line())
            if not line().startswith(b"ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("missing atom count")
            natoms = int(line())
            if not line().startswith(b"ITEM: BOX BOUNDS"):
                raise RuntimeError("missing bounds")
            bounds = [tuple(map(float, line().split()[:2])) for _ in range(3)]
            columns = line().split()[2:]
            col = {name.decode("ascii"): i for i, name in enumerate(columns)}
            q = np.empty(natoms, dtype=np.float64)
            xyz = np.empty((natoms, 3), dtype=np.float64)
            ids = np.empty(natoms, dtype=np.int64)
            for index in range(natoms):
                fields = line().split()
                ids[index] = int(fields[col["id"]])
                q[index] = float(fields[col["q"]])
                xyz[index] = [float(fields[col[name]]) for name in ("x", "y", "z")]
            order = np.argsort(ids)
            lo = np.asarray([item[0] for item in bounds])
            lengths = np.asarray([item[1] - item[0] for item in bounds])
            if not np.allclose(lengths, lengths[0]):
                raise ValueError("reference implementation currently requires a cubic cell")
            frames.append((timestep, q[order], xyz[order] - lo, float(lengths[0])))
    if return_sha256:
        return frames, digest.hexdigest()
    return frames


def parse_force_dump(path: Path):
    frames = []
    with path.open() as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                raise RuntimeError(f"malformed force dump: {path}")
            timestep = int(handle.readline())
            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("missing force-dump atom count")
            natoms = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("missing force-dump bounds")
            for _ in range(3):
                handle.readline()
            columns = handle.readline().split()[2:]
            col = {name: i for i, name in enumerate(columns)}
            ids = np.empty(natoms, dtype=np.int64)
            force = np.empty((natoms, 3), dtype=np.float64)
            for index in range(natoms):
                fields = handle.readline().split()
                ids[index] = int(fields[col["id"]])
                force[index] = [float(fields[col[name]]) for name in ("fx", "fy", "fz")]
            order = np.argsort(ids)
            frames.append((timestep, force[order]))
    return frames


def parse_force_dump_prefix(
    path: Path, count: int, *, return_sha256: bool = False
):
    """Read exactly the first ``count`` force-dump frames and then stop.

    The optional digest covers the precise byte prefix ending at the final
    atom record of frame ``count``.  As for the coordinate prefix reader,
    unbuffered binary input prevents a prediction-stage read-ahead into the
    holdout records.
    """

    if count < 1:
        raise ValueError("count must be positive")
    import hashlib

    digest = hashlib.sha256()
    frames = []
    with path.open("rb", buffering=0) as handle:
        def line() -> bytes:
            raw = handle.readline()
            digest.update(raw)
            return raw

        while len(frames) < count:
            header = line()
            if not header:
                raise RuntimeError(f"force dump {path} contains fewer than {count} frames")
            if not header.startswith(b"ITEM: TIMESTEP"):
                raise RuntimeError(f"malformed force dump: {path}")
            timestep = int(line())
            if not line().startswith(b"ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("missing force-dump atom count")
            natoms = int(line())
            if not line().startswith(b"ITEM: BOX BOUNDS"):
                raise RuntimeError("missing force-dump bounds")
            for _ in range(3):
                line()
            columns = line().split()[2:]
            col = {name.decode("ascii"): index for index, name in enumerate(columns)}
            ids = np.empty(natoms, dtype=np.int64)
            force = np.empty((natoms, 3), dtype=np.float64)
            for index in range(natoms):
                fields = line().split()
                ids[index] = int(fields[col["id"]])
                force[index] = [float(fields[col[name]]) for name in ("fx", "fy", "fz")]
            order = np.argsort(ids)
            frames.append((timestep, force[order]))
    if return_sha256:
        return frames, digest.hexdigest()
    return frames


def stencil_1d(
    coordinate: np.ndarray,
    mesh: int,
    box_length: float,
    order: int,
    real_coeff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaled = np.mod(coordinate, box_length) * mesh / box_length
    if order % 2:
        center = np.floor(scaled + 0.5).astype(np.int64)
        shiftone = 0.0
    else:
        center = np.floor(scaled).astype(np.int64)
        shiftone = 0.5
    dx = center + shiftone - scaled
    # Parentheses matter for even P: C++ uses -(P-1)/2 with integer division,
    # e.g. -1..2 for P=4, not Python's floor-divided -2..2.
    offsets = np.arange(-((order - 1) // 2), order // 2 + 1, dtype=np.int64)
    indices = (center[:, None] + offsets[None, :]) % mesh
    weights = np.empty((len(coordinate), order), dtype=np.float64)
    for lane in range(order):
        weights[:, lane] = horner_array(dx, real_coeff[:, lane])
    return indices, weights


def particle_stencil(
    xyz: np.ndarray,
    mesh: int,
    box_length: float,
    order: int,
    real_coeff: np.ndarray,
):
    return tuple(
        stencil_1d(xyz[:, dim], mesh, box_length, order, real_coeff) for dim in range(3)
    )


def spread_density(
    q: np.ndarray,
    stencil,
    mesh: int,
    box_length: float,
) -> np.ndarray:
    (ix, wx), (iy, wy), (iz, wz) = stencil
    grid = np.zeros((mesh, mesh, mesh), dtype=np.float64)
    p = wx.shape[1]
    linear = (
        (ix[:, :, None, None] * mesh + iy[:, None, :, None]) * mesh
        + iz[:, None, None, :]
    ).reshape(-1)
    weights = (
        q[:, None, None, None]
        * wx[:, :, None, None]
        * wy[:, None, :, None]
        * wz[:, None, None, :]
    ).reshape(-1)
    h = box_length / mesh
    np.add.at(grid.ravel(), linear, weights / (h * h * h))
    return grid


def gather_vector_field(q: np.ndarray, stencil, field: np.ndarray) -> np.ndarray:
    (ix, wx), (iy, wy), (iz, wz) = stencil
    n = len(q)
    p = wx.shape[1]
    result = np.zeros((n, 3), dtype=np.float64)
    for a in range(p):
        for b in range(p):
            wab = wx[:, a] * wy[:, b]
            for c in range(p):
                result += (wab * wz[:, c])[:, None] * field[ix[:, a], iy[:, b], iz[:, c]]
    return q[:, None] * result


def fixed_ik_mesh_force(
    q: np.ndarray,
    xyz: np.ndarray,
    box_length: float,
    mesh: int,
    order: int,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: PSWFCoefficients,
    base_deconvolution: str = "exact-real-transform",
) -> tuple[np.ndarray, dict]:
    split = split_continuation(coeff, csplit)
    window = window_transform_continuation(coeff, cspread, order)
    kernel = split_kernel_grid(
        mesh, box_length, rcut, csplit, split, band_limited=True
    )
    _, k1, kx, ky, kz, _ = reciprocal_arrays(mesh, box_length)
    h = box_length / mesh
    t1 = 0.5 * order * h * np.abs(k1) / cspread
    if base_deconvolution == "exact-real-transform":
        what1 = 0.5 * order * window.value(t1)
        operator_name = "fixed-influence ik; exact polynomial-window transform"
    elif base_deconvolution == "lammps-fourier-polynomial":
        fitted = np.zeros_like(t1)
        inside = t1 <= 1.0
        fitted[inside] = horner_array(2.0 * t1[inside] - 1.0, coeff.spread)
        what1 = 0.5 * order * fitted
        operator_name = "fixed-influence ik; LAMMPS Fourier-polynomial deconvolution"
    else:
        raise ValueError(f"unknown base deconvolution: {base_deconvolution}")
    what = (
        what1[:, None, None] * what1[None, :, None] * what1[None, None, :]
    )
    nonzero = kernel != 0.0
    zeroed = nonzero & (what == 0.0)
    retained = nonzero & ~zeroed
    if not np.any(retained):
        raise FloatingPointError("all active fixed-influence modes have zero deconvolution")
    stability = float(np.min(np.abs(what[retained])))
    if stability <= 1.0e-14:
        raise FloatingPointError(f"unstable fixed deconvolution: min |W|={stability:.3e}")
    green = np.zeros_like(kernel)
    # LAMMPS leaves a zero deconvolution mode inactive.  The associated direct
    # Fourier-mode mismatch is accounted for by the discrete estimator rather
    # than by dividing through a vanishing polynomial.
    green[retained] = kernel[retained] / (what[retained] * what[retained])
    stencil = particle_stencil(xyz, mesh, box_length, order, coeff.real)
    density = spread_density(q, stencil, mesh, box_length)
    rho_hat = np.fft.fftn(density)
    field = np.empty((mesh, mesh, mesh, 3), dtype=np.float64)
    for dim, kd in enumerate((kx, ky, kz)):
        field[..., dim] = np.fft.ifftn((-1j * kd) * green * rho_hat).real
    force = COULOMB_REAL * gather_vector_field(q, stencil, field)
    return force, {
        "operator": operator_name,
        "minimum_abs_dimensionless_window_product": stability,
        "zeroed_active_mode_count": int(np.count_nonzero(zeroed)),
        "split_kernel_normalization": "2*pi*(lambda*psi/C0)/k^2 = 4*pi*a_c/k^2",
    }


def direct_truncated_force(
    q: np.ndarray,
    xyz: np.ndarray,
    box_length: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """Direct nonuniform evaluation on the kernel's canonical FFT cube.

    The tensor contractions are algebraically identical to the explicit
    particle--mode sums but avoid constructing an N by M^3 phase matrix.
    """

    mesh = kernel.shape[0]
    if kernel.shape != (mesh, mesh, mesh):
        raise ValueError("kernel must be cubic")
    _, k1 = fft_modes(mesh, box_length)
    e_minus = [np.exp(-1j * np.outer(xyz[:, d], k1)) for d in range(3)]
    ex, ey, ez = e_minus
    rho = np.empty((mesh, mesh, mesh), dtype=np.complex128)
    for iz in range(mesh):
        rho[:, :, iz] = ex.T @ ((q * ez[:, iz])[:, None] * ey)
    e_plus = [value.conj() for value in e_minus]
    result = np.empty((len(q), 3), dtype=np.float64)
    for dim in range(3):
        kd_shape = [1, 1, 1]
        kd_shape[dim] = mesh
        kd = k1.reshape(kd_shape)
        coefficient = (-1j * kd) * kernel * rho
        values = np.zeros(len(q), dtype=np.complex128)
        for iz in range(mesh):
            partial = e_plus[0] @ coefficient[:, :, iz]
            values += e_plus[2][:, iz] * np.sum(partial * e_plus[1], axis=1)
        result[:, dim] = (q / box_length**3) * values.real
    return COULOMB_REAL * result


def direct_truncated_energy(
    q: np.ndarray,
    xyz: np.ndarray,
    box_length: float,
    kernel: np.ndarray,
) -> float:
    """Direct reciprocal energy on the same canonical cube (self term kept).

    The position-independent reciprocal self contribution is intentionally
    retained because it cancels in the finite-difference force test.
    """

    mesh = kernel.shape[0]
    _, k1 = fft_modes(mesh, box_length)
    ex = np.exp(-1j * np.outer(xyz[:, 0], k1))
    ey = np.exp(-1j * np.outer(xyz[:, 1], k1))
    ez = np.exp(-1j * np.outer(xyz[:, 2], k1))
    rho = np.empty((mesh, mesh, mesh), dtype=np.complex128)
    for iz in range(mesh):
        rho[:, :, iz] = ex.T @ ((q * ez[:, iz])[:, None] * ey)
    return float(
        COULOMB_REAL
        * 0.5
        / box_length**3
        * np.sum(kernel * (rho.real * rho.real + rho.imag * rho.imag))
    )


def rms_vector_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _validate_axial_alias_index(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= value <= MAX_VERIFIED_AXIAL_ALIAS_INDEX
    ):
        raise ValueError(
            "max_abs_axial_alias_index must be an integer in "
            f"[1, {MAX_VERIFIED_AXIAL_ALIAS_INDEX}]"
        )
    return int(value)


def _stable_all_alias_ratio(alias_sum: np.ndarray) -> np.ndarray:
    """Expand the separable three-dimensional alias ratio into positive terms.

    If ``alias_sum`` is the nonzero one-dimensional image sum ``a``, the
    mathematical ratio is ``(1+a_x)(1+a_y)(1+a_z)-1``.  Forming that product
    and then subtracting one loses the leading alias terms when ``a`` is small.
    The explicit expansion below is algebraically identical and contains no
    cancellation.
    """

    ax = alias_sum[:, None, None]
    ay = alias_sum[None, :, None]
    az = alias_sum[None, None, :]
    return (
        ax
        + ay
        + az
        + ax * ay
        + ax * az
        + ay * az
        + ax * ay * az
    )


def _continuous_base_window_safety_check(
    window: EvenPSWFContinuation,
    eta: float,
    sigma_up: float,
) -> dict:
    """Reject zero or uncertified continuum deconvolution denominators.

    The one-dimensional estimator divides by the squared continuous base
    transform on ``0 <= y <= 1``.  A sign change or numerical zero makes its
    improper integral divergent.  The ground-state PSWF concentration
    interval ``|u| <= 1`` supplies the certified zero-free domain used by the
    derivation; values outside it are rejected even if a finite scan happens
    not to land on a zero.
    """

    maximum_argument = eta / sigma_up
    y_scan = np.linspace(0.0, 1.0, CONTINUOUS_BASE_WINDOW_SCAN_POINTS)
    values = np.asarray(window.value(maximum_argument * y_scan), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("continuous base-window transform is nonfinite")
    scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    relative_minimum = float(np.min(np.abs(values)) / scale)
    sign_change = bool(np.any(np.signbit(values[:-1]) != np.signbit(values[1:])))
    sampled_zero = relative_minimum <= CONTINUOUS_BASE_WINDOW_RELATIVE_ZERO_TOL
    if sign_change or sampled_zero:
        raise ValueError(
            "continuous base-window transform crosses zero on 0 <= y <= 1; "
            "the one-dimensional deconvolution integral is divergent"
        )
    safe_argument_tolerance = 64.0 * np.finfo(np.float64).eps
    if maximum_argument > 1.0 + safe_argument_tolerance:
        raise ValueError(
            "continuous base-window argument leaves the certified |u| <= 1 "
            f"safety interval (eta_P/sigma_up={maximum_argument:.16g})"
        )
    return {
        "maximum_base_window_argument": maximum_argument,
        "minimum_relative_base_window_value": relative_minimum,
        "scan_points": CONTINUOUS_BASE_WINDOW_SCAN_POINTS,
    }


def fixed_alias_estimate_homogeneous(
    q: np.ndarray,
    box_length: float,
    mesh: int,
    order: int,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: PSWFCoefficients,
    alias_shell: int = 6,
) -> tuple[float, dict]:
    """Discrete Eq. (60), with a stable separable all-alias shell sum."""

    if (
        isinstance(alias_shell, bool)
        or not isinstance(alias_shell, (int, np.integer))
        or alias_shell < 0
    ):
        raise ValueError("alias_shell must be a nonnegative integer")
    alias_shell = int(alias_shell)

    split = split_continuation(coeff, csplit)
    window = window_transform_continuation(coeff, cspread, order)
    kernel = split_kernel_grid(
        mesh, box_length, rcut, csplit, split, band_limited=True
    )
    _, k1, _, _, _, k2 = reciprocal_arrays(mesh, box_length)
    h = box_length / mesh
    tbase = 0.5 * order * h * np.abs(k1) / cspread
    wbase = window.value(tbase)
    # Accumulate only nonzero one-dimensional images.  Keeping the unit
    # zero-image contribution separate permits the stable positive expansion
    # in _stable_all_alias_ratio.
    alias_sum = np.zeros(mesh, dtype=np.float64)
    shell_values = [0.0]
    for shell in range(1, alias_shell + 1):
        for ell in (-shell, shell):
            q1 = k1 + (2.0 * math.pi / h) * ell
            ta = 0.5 * order * h * np.abs(q1) / cspread
            alias_sum += (window.value(ta) / wbase) ** 2
        ratio3 = _stable_all_alias_ratio(alias_sum)
        # There are two equal leading contributions: an alias in spreading
        # (source image) and an alias in gathering (target image).  The prose
        # preceding Eq. (60) retains both symmetric terms; the original printed
        # equation inadvertently contained only one.  The factor two here is
        # therefore structural, not a fitted prefactor.
        chi2 = float(2.0 * np.sum(k2 * kernel * kernel * ratio3) / box_length**6)
        shell_values.append(chi2)
    qsum = float(np.sum(q * q))
    estimate = (
        COULOMB_REAL
        * qsum
        / math.sqrt(len(q))
        * math.sqrt(max(shell_values[-1], 0.0))
    )
    return estimate, {
        "operator": "corrected Eq. (60), fixed-influence ik, homogeneous S_q=1",
        "symmetric_source_gather_factor": 2,
        "stable_positive_expansion": True,
        "alias_shell": alias_shell,
        "chi2_by_shell": shell_values,
        "last_shell_relative_change": (
            abs(shell_values[-1] - shell_values[-2]) / shell_values[-1]
            if len(shell_values) > 1 and shell_values[-1] > 0
            else 0.0
        ),
    }


def fixed_alias_estimate_discrete_axial(
    q: np.ndarray,
    box_length: float,
    mesh: int,
    order: int,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: PSWFCoefficients,
    *,
    max_abs_axial_alias_index: int = 1,
) -> tuple[float, dict]:
    """Discrete axial multi-image estimate on the Eq. (60) grid/operator.

    Only face images ``ell=+-n e_d`` with ``1 <= n <= K`` are retained.
    Unlike the continuum reduction, this estimator evaluates the canonical FFT
    modes directly.  Its positive sum is therefore well defined even when a
    continuum base-window transform would cross zero between grid modes.
    """

    depth = _validate_axial_alias_index(max_abs_axial_alias_index)
    split = split_continuation(coeff, csplit)
    window = window_transform_continuation(coeff, cspread, order)
    kernel = split_kernel_grid(
        mesh, box_length, rcut, csplit, split, band_limited=True
    )
    _, k1, _, _, _, k2 = reciprocal_arrays(mesh, box_length)
    h = box_length / mesh
    tbase = 0.5 * order * h * np.abs(k1) / cspread
    wbase = window.value(tbase)
    alias_sum = np.zeros(mesh, dtype=np.float64)
    chi2_by_depth = [0.0]
    for alias_index in range(1, depth + 1):
        for ell in (-alias_index, alias_index):
            q1 = k1 + (2.0 * math.pi / h) * ell
            ta = 0.5 * order * h * np.abs(q1) / cspread
            alias_sum += (window.value(ta) / wbase) ** 2
        ax = alias_sum[:, None, None]
        ratio_axial = ax + alias_sum[None, :, None] + alias_sum[None, None, :]
        chi2_by_depth.append(
            float(
                2.0
                * np.sum(k2 * kernel * kernel * ratio_axial)
                / box_length**6
            )
        )
    qsum = float(np.sum(q * q))
    force_scale = COULOMB_REAL * qsum / math.sqrt(len(q))
    estimates = [force_scale * math.sqrt(max(value, 0.0)) for value in chi2_by_depth]
    return estimates[-1], {
        "operator": "discrete axial multi-image, fixed-influence ik, homogeneous S_q=1",
        "symmetric_source_gather_factor": 2,
        "max_abs_axial_alias_index": depth,
        "total_axial_face_images": 6 * depth,
        "chi2_by_axial_depth": chi2_by_depth,
        "estimate_by_axial_depth": estimates,
        "last_axial_layer_relative_change": (
            abs(estimates[-1] - estimates[-2]) / estimates[-1]
            if estimates[-1] > 0.0
            else 0.0
        ),
        "stable_positive_sum": True,
    }


def fixed_alias_estimate_one_dimensional(
    q: np.ndarray,
    box_length: float,
    order: int,
    rcut: float,
    csplit: float,
    cspread: float,
    sigma_up: float,
    coeff: PSWFCoefficients,
    quadrature_order: int = 320,
    *,
    max_abs_axial_alias_index: int = 1,
    allow_extrapolation: bool = False,
) -> float:
    """One-dimensional axial-image approximation in Eq. (67).

    The unknown absolute normalization of psi cancels by writing H with
    psi_split(t)/psi_split(0).  Likewise, lambda_spread cancels in the window
    ratio.  This keeps the numerical expression faithful to Eq. (67) while
    using the exact continuation of the piecewise-polynomial window.

    ``max_abs_axial_alias_index=1`` recovers the nearest-face expression.
    Larger values add the axial images ``ell=+-n e_d`` for
    ``n=1,...,max_abs_axial_alias_index``.  Mixed-axis edge and corner images
    are not part of this one-dimensional reduction; they remain in the
    discrete all-alias estimator.

    The derivation assumes ``sigma_up >= 1``.  By default the implementation
    enforces that domain.  ``allow_extrapolation=True`` is reserved for an
    explicitly labelled boundary diagnostic and must not be interpreted as
    extending the derivation.  This opt-in never bypasses the independent
    continuous base-window safety check: a zero denominator makes the
    improper integral divergent rather than merely inaccurate.
    """

    if not math.isfinite(sigma_up) or sigma_up <= 0.0:
        raise ValueError("sigma_up must be finite and positive")
    if sigma_up < 1.0 and not allow_extrapolation:
        raise ValueError("Eq. (67) is evaluated only in the resolved-band domain")
    depth = _validate_axial_alias_index(max_abs_axial_alias_index)
    if (
        isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, (int, np.integer))
        or quadrature_order < 1
    ):
        raise ValueError("quadrature_order must be a positive integer")
    split = split_continuation(coeff, csplit)
    window = window_transform_continuation(coeff, cspread, order)
    eta = math.pi * order / (2.0 * cspread)
    _continuous_base_window_safety_check(window, eta, sigma_up)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    y = 0.5 * (nodes + 1.0)
    wy = 0.5 * weights

    # H_norm(y) = int_y^1 |psi(t)/psi(0)|^2 dt/t.  A separate Gauss rule on
    # every [y,1] avoids a grid-dependent cumulative integration near y=0.
    t = y[:, None] + 0.5 * (1.0 - y[:, None]) * (nodes[None, :] + 1.0)
    wt = 0.5 * (1.0 - y[:, None]) * weights[None, :]
    split0 = float(split.value(np.array([0.0]))[0])
    psi_ratio2 = (split.value(t) / split0) ** 2
    h_norm = np.sum(wt * psi_ratio2 / t, axis=1)

    base = window.value(eta * y / sigma_up)
    numerator = np.zeros_like(y)
    for alias_index in range(1, depth + 1):
        plus = window.value(
            eta * (2.0 * alias_index + y / sigma_up)
        )
        minus = window.value(
            eta * (2.0 * alias_index - y / sigma_up)
        )
        numerator += plus * plus + minus * minus
    ratio = numerator / (base * base)
    integral = float(np.dot(wy, h_norm * ratio))
    qsum = float(np.sum(q * q))
    # The same source/gather symmetry factor that corrects Eq. (60) changes
    # the continuum prefactor from 24 to 48.
    variance = 48.0 * qsum * qsum * csplit * integral / (
        len(q) * box_length**3 * rcut
    )
    return COULOMB_REAL * math.sqrt(max(variance, 0.0))


def closed_fourier_estimate(
    q: np.ndarray,
    box_length: float,
    rcut: float,
    csplit: float,
    coeff: PSWFCoefficients,
    kmax: float | None = None,
) -> float:
    """Evaluate Eq. (56) without the undocumented 5.46 asymptotic proxy."""

    if kmax is None:
        kmax = csplit / rcut
    if kmax * rcut < csplit:
        raise ValueError("Eq. (56) outside-tail form requires Kmax*rc >= csplit")
    split = split_continuation(coeff, csplit)
    values = split.value(np.asarray([0.0, 1.0]))
    psi1_over_psi0 = abs(float(values[1] / values[0]))
    qsum = float(np.sum(q * q))
    prefactor = qsum / math.sqrt(len(q) * box_length**3)
    return (
        COULOMB_REAL
        * prefactor
        * 4.0
        * psi1_over_psi0
        / (coeff.split_lambda * math.sqrt(kmax) * rcut)
    )


if __name__ == "__main__":
    print("fixed_ik_reference.py is a library; run run_section5_calculations.py", file=sys.stderr)
