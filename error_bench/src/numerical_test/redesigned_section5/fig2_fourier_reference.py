#!/usr/bin/env python3
"""Exact-reference helpers for the redesigned Fourier-truncation test.

The finite force is a direct nonuniform evaluation of the PSWF smooth
Fourier sum.  The infinite smooth force is evaluated through the exact
periodic identity

    F_smooth^(infinity) = F_Coulomb^periodic - F_near^compact,

where the first term is converged with LAMMPS Ewald and the second is
evaluated from the same exact PSWF.  This avoids pretending that a finite
reciprocal cube is an infinity reference for the algebraically decaying PSWF
tail.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fixed_ik_reference import COULOMB_REAL, PSWF_SOURCE_DIR, reciprocal_arrays


HERE = Path(__file__).resolve().parent
PROFILE_SOURCE = HERE / "eval_pswf_profile.cpp"
PROFILE_EXE = HERE / "eval_pswf_profile"
DEFAULT_LAMMPS = HERE / "pppm_symmetric_scan" / "lmp.pppm_symmetric_scan"


def ensure_profile_executable() -> Path:
    required = (
        PROFILE_SOURCE,
        PSWF_SOURCE_DIR / "math_pswf.cpp",
        PSWF_SOURCE_DIR / "math_pswf.h",
        PSWF_SOURCE_DIR / "math_const.h",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing exact MathPSWF inputs: {missing}")
    newest = max(path.stat().st_mtime for path in required)
    if PROFILE_EXE.is_file() and PROFILE_EXE.stat().st_mtime >= newest:
        return PROFILE_EXE
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            f"-I{PSWF_SOURCE_DIR}",
            str(PROFILE_SOURCE),
            str(PSWF_SOURCE_DIR / "math_pswf.cpp"),
            "-o",
            str(PROFILE_EXE),
        ],
        cwd=HERE,
        check=True,
    )
    return PROFILE_EXE


@dataclass(frozen=True)
class PSWFConstants:
    psi0: float
    psi1: float
    c0: float
    eigenvalue: float


def exact_inside_profile(c: float, x) -> tuple[PSWFConstants, np.ndarray, np.ndarray]:
    """Return exact psi(x), integral_0^x psi, and normalization constants."""

    values = np.asarray(x, dtype=np.float64)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("exact_inside_profile accepts only 0 <= x <= 1")
    flat = values.ravel()
    input_text = "".join(f"{item:.17g}\n" for item in flat)
    output = subprocess.run(
        [str(ensure_profile_executable()), f"{float(c):.17g}"],
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
        cwd=HERE,
    ).stdout.splitlines()
    fields = output[0].split()
    if len(fields) != 5 or fields[0] != "constants":
        raise RuntimeError("malformed exact PSWF bridge output")
    constants = PSWFConstants(*(float(item) for item in fields[1:]))
    if len(output) != len(flat) + 1:
        raise RuntimeError("exact PSWF bridge returned the wrong number of rows")
    psi = np.empty(len(flat), dtype=np.float64)
    integral = np.empty(len(flat), dtype=np.float64)
    for i, line in enumerate(output[1:]):
        row = line.split()
        psi[i] = float(row[1])
        integral[i] = float(row[2])
    return constants, psi.reshape(values.shape), integral.reshape(values.shape)


class ExactPSWFContinuation:
    """Exact inside PSWF plus its finite-Fourier outside continuation."""

    def __init__(self, c: float, quadrature_order: int = 768) -> None:
        self.c = float(c)
        self.xq, self.wq = np.polynomial.legendre.leggauss(quadrature_order)
        constants, psi_q, _ = exact_inside_profile(self.c, np.abs(self.xq))
        self.constants = constants
        self.psi_q = psi_q

    def attenuation(self, u) -> np.ndarray:
        """Evaluate a_c(c*u)=psi_0^c(u)/psi_0^c(0)."""

        values = np.abs(np.asarray(u, dtype=np.float64))
        flat = values.ravel()
        unique, inverse = np.unique(flat, return_inverse=True)
        result = np.empty_like(unique)
        inside = unique <= 1.0
        if np.any(inside):
            _, psi, _ = exact_inside_profile(self.c, unique[inside])
            result[inside] = psi / self.constants.psi0
        outside_indices = np.flatnonzero(~inside)
        weighted = self.wq * self.psi_q
        denominator = self.constants.eigenvalue * self.constants.psi0
        for start in range(0, len(outside_indices), 2048):
            idx = outside_indices[start : start + 2048]
            phase = np.cos(self.c * np.outer(unique[idx], self.xq))
            result[idx] = phase @ weighted / denominator
        return result[inverse].reshape(values.shape)

    def kernel_grid(self, mesh: int, box_length: float, rcut: float) -> np.ndarray:
        _, _, _, _, _, k2 = reciprocal_arrays(mesh, box_length)
        kernel = np.zeros_like(k2)
        nonzero = k2 > 0.0
        u = rcut * np.sqrt(k2[nonzero]) / self.c
        kernel[nonzero] = 4.0 * math.pi * self.attenuation(u) / k2[nonzero]
        return kernel

    def compact_near_force(
        self,
        q: np.ndarray,
        xyz: np.ndarray,
        box_length: float,
        rcut: float,
    ) -> np.ndarray:
        """Exact minimum-image force from (1-phi(r))/r for r < rcut."""

        n = len(q)
        pairs_i = []
        pairs_j = []
        displacements = []
        radii = []
        for i in range(n - 1):
            delta = xyz[i] - xyz[i + 1 :]
            delta -= box_length * np.rint(delta / box_length)
            radius = np.linalg.norm(delta, axis=1)
            keep = np.flatnonzero(radius < rcut)
            if len(keep):
                pairs_i.extend([i] * len(keep))
                pairs_j.extend((i + 1 + keep).tolist())
                displacements.append(delta[keep])
                radii.append(radius[keep])
        force = np.zeros((n, 3), dtype=np.float64)
        if not radii:
            return force
        ii = np.asarray(pairs_i, dtype=np.int64)
        jj = np.asarray(pairs_j, dtype=np.int64)
        delta = np.concatenate(displacements, axis=0)
        radius = np.concatenate(radii)
        scaled = radius / rcut
        _, psi, integral = exact_inside_profile(self.c, scaled)
        c0 = self.constants.c0
        # -d[(1-phi)/r]/dr, expressed as a multiplier of r_vec/r^3.
        factor = 1.0 - integral / c0 + scaled * psi / c0
        pair_force = (
            COULOMB_REAL
            * (q[ii] * q[jj] * factor / radius**3)[:, None]
            * delta
        )
        np.add.at(force, ii, pair_force)
        np.add.at(force, jj, -pair_force)
        return force


def parse_force_dump(path: Path) -> np.ndarray:
    rows = []
    in_atoms = False
    for line in path.read_text().splitlines():
        if line.startswith("ITEM: ATOMS"):
            in_atoms = True
            continue
        if not in_atoms:
            continue
        fields = line.split()
        if len(fields) != 4:
            continue
        rows.append((int(fields[0]), *(float(item) for item in fields[1:])))
    rows.sort(key=lambda row: row[0])
    if not rows:
        raise RuntimeError(f"no forces parsed from {path}")
    return np.asarray([row[1:] for row in rows], dtype=np.float64)


def tight_periodic_coulomb_force(
    data_path: Path,
    requested_accuracy: float = 1.0e-14,
    lammps_binary: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Run a table-free, tight Ewald force-only calculation on fixed coordinates."""

    binary = lammps_binary or Path(os.environ.get("ESP_LAMMPS_BIN", DEFAULT_LAMMPS))
    if not binary.is_file():
        raise FileNotFoundError(
            f"LAMMPS executable not found: {binary}; set ESP_LAMMPS_BIN to override"
        )
    with tempfile.TemporaryDirectory(prefix="esp-fig2-ewald-") as tmp:
        dump = Path(tmp) / "forces.dump"
        input_text = f"""units real
atom_style charge
read_data {data_path.resolve()}
pair_style coul/long 9.0
pair_coeff * *
pair_modify table 0
kspace_style ewald {requested_accuracy:.17g}
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
dump f all custom 1 {dump} id fx fy fz
dump_modify f sort id format line \"%d %.17g %.17g %.17g\"
run 0
"""
        completed = subprocess.run(
            [str(binary), "-log", "none", "-in", "/dev/stdin"],
            input=input_text,
            text=True,
            capture_output=True,
            check=True,
        )
        force = parse_force_dump(dump)
    accuracy_match = re.search(
        r"estimated absolute RMS force accuracy\s*=\s*([0-9.eE+-]+)",
        completed.stdout,
    )
    vector_match = re.search(
        r"KSpace vectors: actual max1d max3d\s*=\s*(\d+)\s+(\d+)\s+(\d+)",
        completed.stdout,
    )
    return force, {
        "requested_accuracy": requested_accuracy,
        "estimated_absolute_rms_force_accuracy": (
            float(accuracy_match.group(1)) if accuracy_match else math.nan
        ),
        "kspace_vectors": (int(vector_match.group(1)) if vector_match else -1),
        "max_1d_index": (int(vector_match.group(2)) if vector_match else -1),
        "max_3d_allocation": (int(vector_match.group(3)) if vector_match else -1),
        "pair_table": "disabled (pair_modify table 0)",
        "boundary": "three-dimensional periodic, conducting Ewald convention",
    }


def direct_periodic_coulomb_ewald_force(
    q: np.ndarray,
    xyz: np.ndarray,
    box_length: float,
    alpha: float = 0.25,
    real_cutoff: float = 36.0,
    reciprocal_mesh: int = 41,
) -> tuple[np.ndarray, dict]:
    """Explicit real- and reciprocal-sum Ewald reference.

    This implementation is intentionally independent of LAMMPS's automatic
    Ewald parameter selection.  Both sums are direct and symmetric; changing
    ``alpha`` while enlarging the two cutoffs provides a stringent convergence
    test for the periodic Coulomb force used in the infinity identity.
    """

    if reciprocal_mesh % 2 != 1:
        raise ValueError("the direct Ewald reciprocal set must be odd and symmetric")
    if abs(float(np.sum(q))) > 1.0e-12:
        raise ValueError("the direct periodic Ewald reference requires charge neutrality")
    # Since each unshifted component lies in (-L,L), images beyond
    # floor(R/L)+1 cannot enter the radius-R sphere.
    nmax = int(math.floor(real_cutoff / box_length)) + 1
    real = np.zeros((len(q), 3), dtype=np.float64)
    prefactor = 2.0 * alpha / math.sqrt(math.pi)
    try:
        from scipy.special import erfc
    except ImportError as exc:  # pragma: no cover - SciPy is in the project env
        raise RuntimeError("SciPy is required for the direct Ewald reference") from exc
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                shift = box_length * np.asarray([nx, ny, nz], dtype=np.float64)
                for i in range(len(q)):
                    delta = xyz[i] - xyz + shift
                    r2 = np.einsum("ij,ij->i", delta, delta)
                    keep = (r2 > 0.0) & (r2 < real_cutoff * real_cutoff)
                    if not np.any(keep):
                        continue
                    radius = np.sqrt(r2[keep])
                    screening = erfc(alpha * radius) + (
                        prefactor * radius * np.exp(-(alpha * radius) ** 2)
                    )
                    real[i] += (
                        COULOMB_REAL
                        * q[i]
                        * np.sum(
                            (
                                q[keep]
                                * screening
                                / radius**3
                            )[:, None]
                            * delta[keep],
                            axis=0,
                        )
                    )

    _, k2 = symmetric_reciprocal_arrays(reciprocal_mesh, box_length)
    kernel = np.zeros_like(k2)
    nonzero = k2 > 0.0
    kernel[nonzero] = (
        4.0
        * math.pi
        * np.exp(-k2[nonzero] / (4.0 * alpha * alpha))
        / k2[nonzero]
    )
    reciprocal = direct_force_from_kernel(q, xyz, box_length, kernel)
    force = real + reciprocal
    kmax = 2.0 * math.pi * (reciprocal_mesh // 2) / box_length
    return force, {
        "alpha": alpha,
        "real_cutoff": real_cutoff,
        "real_image_index_limit": nmax,
        "reciprocal_mesh": reciprocal_mesh,
        "reciprocal_kmax": kmax,
        "real_screen_at_cutoff": float(erfc(alpha * real_cutoff)),
        "reciprocal_screen_at_face": math.exp(-kmax * kmax / (4.0 * alpha * alpha)),
        "operator": "direct symmetric conducting-boundary Ewald sums",
    }


def direct_force_from_kernel(
    q: np.ndarray,
    xyz: np.ndarray,
    box_length: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """Direct particle--mode evaluation on an odd, inversion-symmetric cube."""

    mesh = kernel.shape[0]
    if mesh % 2 != 1 or kernel.shape != (mesh, mesh, mesh):
        raise ValueError("Fig. 2 uses an odd cubic grid to avoid a Nyquist convention")
    indices = np.arange(-(mesh // 2), mesh // 2 + 1, dtype=np.int64)
    k1 = 2.0 * math.pi * indices / box_length
    e_minus = [np.exp(-1j * np.outer(xyz[:, dim], k1)) for dim in range(3)]
    ex, ey, ez = e_minus
    rho = np.empty((mesh, mesh, mesh), dtype=np.complex128)
    for iz in range(mesh):
        rho[:, :, iz] = ex.T @ ((q * ez[:, iz])[:, None] * ey)
    e_plus = [item.conj() for item in e_minus]
    result = np.empty((len(q), 3), dtype=np.float64)
    for dim in range(3):
        shape = [1, 1, 1]
        shape[dim] = mesh
        kd = k1.reshape(shape)
        coefficient = (-1j * kd) * kernel * rho
        values = np.zeros(len(q), dtype=np.complex128)
        for iz in range(mesh):
            partial = e_plus[0] @ coefficient[:, :, iz]
            values += e_plus[2][:, iz] * np.sum(partial * e_plus[1], axis=1)
        result[:, dim] = q * values.real / box_length**3
    return COULOMB_REAL * result


def symmetric_reciprocal_arrays(mesh: int, box_length: float):
    if mesh % 2 != 1:
        raise ValueError("the symmetric Fig. 2 reciprocal cube requires odd mesh")
    indices = np.arange(-(mesh // 2), mesh // 2 + 1, dtype=np.int64)
    k1 = 2.0 * math.pi * indices / box_length
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    return k1, k2


def symmetric_kernel_grid(
    pswf: ExactPSWFContinuation,
    mesh: int,
    box_length: float,
    rcut: float,
) -> np.ndarray:
    _, k2 = symmetric_reciprocal_arrays(mesh, box_length)
    kernel = np.zeros_like(k2)
    mask = k2 > 0.0
    kernel[mask] = (
        4.0
        * math.pi
        * pswf.attenuation(rcut * np.sqrt(k2[mask]) / pswf.c)
        / k2[mask]
    )
    return kernel


def rms_vector_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def pooled_rms_jackknife_sem(values) -> float:
    """Delete-one jackknife SEM for ``sqrt(mean(values**2))``."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) < 2 or np.any(~np.isfinite(data)):
        raise ValueError("jackknife input must be a finite vector of length >= 2")
    squared_sum = float(np.sum(data * data))
    estimates = np.sqrt((squared_sum - data * data) / (len(data) - 1))
    center = float(np.mean(estimates))
    return float(
        np.sqrt((len(data) - 1) / len(data) * np.sum((estimates - center) ** 2))
    )


def lattice_shell_counts(radius: int) -> np.ndarray:
    """Exact number of integer triples for n_x^2+n_y^2+n_z^2=s, s<=R^2."""

    polynomial = np.zeros(radius * radius + 1, dtype=np.float64)
    polynomial[0] = 1.0
    n = np.arange(1, radius + 1, dtype=np.int64)
    polynomial[n * n] = 2.0
    convolution_length = 3 * radius * radius + 1
    nfft = 1 << (convolution_length - 1).bit_length()
    transformed = np.fft.rfft(polynomial, n=nfft)
    counts = np.fft.irfft(transformed**3, n=nfft)[: radius * radius + 1]
    return np.rint(counts).astype(np.int64)


def _sine_square_tail(lower_k: float, rcut: float) -> float:
    """Integral_K^infinity sin(rc*k)^2/k^2 dk."""

    try:
        from scipy.special import sici
    except ImportError as exc:  # pragma: no cover - SciPy is in the project env
        raise RuntimeError("SciPy is required for the analytic asymptotic tail") from exc
    x = 2.0 * rcut * lower_k
    si = float(sici(x)[0])
    return math.sin(rcut * lower_k) ** 2 / lower_k + rcut * (math.pi / 2.0 - si)


def discrete_eq46_sum(
    pswf: ExactPSWFContinuation,
    box_length: float,
    rcut: float,
    mesh: int,
    shell_counts: np.ndarray,
    tail_radius: int,
) -> tuple[float, dict]:
    """Eq. (46) cube-complement sum plus a documented far-tail closure."""

    if mesh % 2 != 1:
        raise ValueError("Eq. (46) test uses an odd symmetric reciprocal cube")
    if tail_radius * tail_radius >= len(shell_counts):
        raise ValueError("shell_counts does not cover tail_radius")
    beta = 2.0 * math.pi / box_length
    s = np.arange(1, tail_radius * tail_radius + 1, dtype=np.float64)
    k = beta * np.sqrt(s)
    attenuation = pswf.attenuation(rcut * k / pswf.c)
    term = 16.0 * math.pi**2 * attenuation**2 / k**2
    half = mesh // 2
    index = np.arange(-half, half + 1, dtype=np.int64)
    nx, ny, nz = np.meshgrid(index, index, index, indexing="ij")
    squared = nx * nx + ny * ny + nz * nz
    inside_counts = np.bincount(
        squared.ravel(), minlength=tail_radius * tail_radius + 1
    )[: tail_radius * tail_radius + 1]
    radial_counts = shell_counts[: tail_radius * tail_radius + 1]
    omitted_counts = radial_counts - inside_counts
    if np.any(omitted_counts < 0):
        raise RuntimeError("inside-cube degeneracy exceeds full radial degeneracy")
    # Sum the omitted degeneracies directly.  Subtracting two O(10^5)
    # retained-band sums loses digits for the tightest c_split, whose omitted
    # sum is O(10^-8).
    explicit_omitted_sum = float(np.dot(omitted_counts[1:], term))
    radial_sum = float(np.dot(radial_counts[1:], term))
    inside_sum = float(np.dot(inside_counts[1:], term))

    # Leading PSWF continuation: a(k) ~ B sin(rc*k)/k.  Only the far
    # sphere beyond tail_radius is closed this way; the plotted Eq. (46)
    # value is dominated by the explicitly summed discrete modes.
    constants = pswf.constants
    amplitude = 2.0 * constants.psi1 / (
        constants.eigenvalue * constants.psi0 * rcut
    )
    lower_k = beta * (tail_radius + 0.5)
    integral = amplitude**2 * _sine_square_tail(lower_k, rcut)
    continuum_tail = 8.0 * box_length**3 * integral
    omitted_sum = explicit_omitted_sum + continuum_tail
    if omitted_sum <= 0.0:
        raise FloatingPointError("non-positive Eq. (46) mode sum")
    return omitted_sum, {
        "tail_radius_index": tail_radius,
        "explicit_radial_sum": radial_sum,
        "retained_cube_sum": inside_sum,
        "explicit_omitted_sum": explicit_omitted_sum,
        "asymptotic_continuum_tail": continuum_tail,
        "tail_fraction_of_omitted_sum": continuum_tail / omitted_sum,
        "tail_lower_k": lower_k,
        "tail_closure": "leading exact-PSWF sinc tail plus continuum density of states",
    }


def eq46_force_error(mode_sum: float, qsum: float, natoms: int, volume: float) -> float:
    return COULOMB_REAL * qsum / (math.sqrt(natoms) * volume) * math.sqrt(mode_sum)


def eq55_closed_force_error(
    pswf: ExactPSWFContinuation,
    qsum: float,
    natoms: int,
    volume: float,
    kmax: float,
    rcut: float,
) -> float:
    constants = pswf.constants
    return (
        COULOMB_REAL
        * qsum
        / math.sqrt(natoms * volume)
        * 4.0
        / (constants.eigenvalue * math.sqrt(kmax) * rcut)
        * abs(constants.psi1 / constants.psi0)
    )
