#!/usr/bin/env python3
"""Operator-matched analytical-differentiation (AD) reference utilities.

This module is deliberately separate from the main Figure 3 implementation.
It audits the development LAMMPS ESP AD path, whose particle gather uses the
classical derivative of a compact, piecewise-polynomial assignment window.

The important distinction is that this implemented derivative is *not* the
distributional derivative of the compact extension when the assignment
window is nonzero (or has small polynomial mismatches) at a piece boundary.
Consequently its Fourier transform must be evaluated directly.  Replacing it
by ``i q W_hat(q)`` drops boundary/jump terms and gives the spurious
high-alias behaviour discussed in the audit report.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REDESIGNED_ROOT = HERE.parent
if str(REDESIGNED_ROOT) not in sys.path:
    sys.path.insert(0, str(REDESIGNED_ROOT))

import fixed_ik_reference as ikref


@dataclass(frozen=True)
class WindowTransform1D:
    """Transforms of the actual piecewise-polynomial particle-grid window.

    ``source`` is the transform used by spreading, with the ``exp(-i q x)``
    convention.  ``gather`` is its conjugate-phase counterpart.  ``gradient``
    is the transform of the classical polynomial derivative used by the AD
    gather, including the physical factor ``1/h``.
    """

    source: np.ndarray
    gather: np.ndarray
    gradient: np.ndarray


def _poly_moments(z: float, degree: int) -> np.ndarray:
    """Return int_{-1/2}^{1/2} x^n exp(i*z*x) dx, n=0..degree.

    Low frequencies use Gauss--Legendre quadrature to avoid cancellation in
    the integration-by-parts recurrence.  The recurrence is stable for the
    large alias frequencies and avoids under-resolving oscillatory tails.
    """

    if abs(z) < 12.0:
        nodes, weights = np.polynomial.legendre.leggauss(max(48, degree + 8))
        x = 0.5 * nodes
        phase = np.exp(1j * z * x)
        result = np.empty(degree + 1, dtype=np.complex128)
        power = np.ones_like(x)
        for n in range(degree + 1):
            result[n] = 0.5 * np.dot(weights, power * phase)
            power *= x
        return result

    result = np.empty(degree + 1, dtype=np.complex128)
    iz = 1j * z
    upper_phase = np.exp(0.5j * z)
    lower_phase = np.exp(-0.5j * z)
    result[0] = (upper_phase - lower_phase) / iz
    upper_power = 1.0
    lower_power = 1.0
    for n in range(1, degree + 1):
        upper_power *= 0.5
        lower_power *= -0.5
        boundary = upper_power * upper_phase - lower_power * lower_phase
        result[n] = boundary / iz - (n / iz) * result[n - 1]
    return result


def _poly_integral(coeff: np.ndarray, z: float) -> complex:
    moments = _poly_moments(z, len(coeff) - 1)
    return complex(np.dot(np.asarray(coeff, dtype=np.float64), moments))


def actual_window_transforms_1d(
    q: np.ndarray,
    mesh_spacing: float,
    order: int,
    real_coeff: np.ndarray,
) -> WindowTransform1D:
    """Evaluate transforms of exactly the polynomials used by LAMMPS.

    The stencil coefficients are polynomials in the common local coordinate
    ``dx`` on ``[-1/2,1/2]``.  The lane at integer offset ``m`` is located at
    ``y = dx + m - shiftone``.  This is the same even/odd indexing convention
    implemented in ``ESP::particle_map`` and ``ESP::compute_rho1d*``.
    """

    q = np.asarray(q, dtype=np.float64)
    flat = q.ravel()
    coeff = np.asarray(real_coeff, dtype=np.float64)
    if coeff.ndim != 2 or coeff.shape[1] != order:
        raise ValueError("real_coeff must have shape (polynomial_degree+1, order)")
    offsets = np.arange(-((order - 1) // 2), order // 2 + 1, dtype=np.int64)
    if len(offsets) != order:
        raise AssertionError("stencil offset convention is inconsistent")
    shiftone = 0.5 if order % 2 == 0 else 0.0

    source = np.empty(flat.shape, dtype=np.complex128)
    gather = np.empty(flat.shape, dtype=np.complex128)
    gradient = np.empty(flat.shape, dtype=np.complex128)
    derivative_coeff = np.zeros_like(coeff)
    if len(coeff) > 1:
        derivative_coeff[:-1] = (
            np.arange(1, len(coeff), dtype=np.float64)[:, None] * coeff[1:]
        )

    for index, qvalue in enumerate(flat):
        z = float(qvalue * mesh_spacing)
        source_value = 0.0j
        gather_value = 0.0j
        gradient_value = 0.0j
        for lane, offset in enumerate(offsets):
            center = float(offset) - shiftone
            lane_coeff = coeff[:, lane]
            lane_derivative = derivative_coeff[:-1, lane]
            minus = np.exp(-1j * z * center)
            plus = np.exp(1j * z * center)
            source_value += minus * _poly_integral(lane_coeff, -z)
            gather_value += plus * _poly_integral(lane_coeff, z)
            gradient_value += plus * _poly_integral(lane_derivative, z)
        source[index] = source_value
        gather[index] = gather_value
        gradient[index] = gradient_value / mesh_spacing

    shape = q.shape
    return WindowTransform1D(
        source=source.reshape(shape),
        gather=gather.reshape(shape),
        gradient=gradient.reshape(shape),
    )


def lammps_deconvolution_1d(
    q: np.ndarray,
    mesh_spacing: float,
    order: int,
    cspread: float,
    fourier_coeff: np.ndarray,
) -> np.ndarray:
    """One-dimensional transform used in ``ESP::compute_gf_ad``."""

    q = np.asarray(q, dtype=np.float64)
    t = 0.5 * order * mesh_spacing * np.abs(q) / cspread
    result = np.zeros_like(t)
    inside = t <= 1.0
    result[inside] = 0.5 * order * ikref.horner_array(
        2.0 * t[inside] - 1.0, np.asarray(fourier_coeff, dtype=np.float64)
    )
    return result


def derivative_stencil_1d(
    coordinate: np.ndarray,
    mesh: int,
    box_length: float,
    order: int,
    real_coeff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices, weights, and d(weight)/d(dx) for one coordinate."""

    coordinate = np.asarray(coordinate, dtype=np.float64)
    scaled = np.mod(coordinate, box_length) * mesh / box_length
    if order % 2:
        center = np.floor(scaled + 0.5).astype(np.int64)
        shiftone = 0.0
    else:
        center = np.floor(scaled).astype(np.int64)
        shiftone = 0.5
    dx = center + shiftone - scaled
    offsets = np.arange(-((order - 1) // 2), order // 2 + 1, dtype=np.int64)
    indices = (center[:, None] + offsets[None, :]) % mesh
    coeff = np.asarray(real_coeff, dtype=np.float64)
    weights = np.empty((len(coordinate), order), dtype=np.float64)
    derivatives = np.empty_like(weights)
    for lane in range(order):
        weights[:, lane] = ikref.horner_array(dx, coeff[:, lane])
        if len(coeff) == 1:
            derivatives[:, lane] = 0.0
        else:
            dcoeff = np.arange(1, len(coeff), dtype=np.float64) * coeff[1:, lane]
            derivatives[:, lane] = ikref.horner_array(dx, dcoeff)
    return indices, weights, derivatives


def particle_stencil_with_derivative(
    xyz: np.ndarray,
    mesh: int,
    box_length: float,
    order: int,
    real_coeff: np.ndarray,
):
    return tuple(
        derivative_stencil_1d(xyz[:, dim], mesh, box_length, order, real_coeff)
        for dim in range(3)
    )


def _plain_stencil(stencil):
    return tuple((indices, weights) for indices, weights, _ in stencil)


def gather_ad_gradient(stencil, potential: np.ndarray, mesh_spacing: float) -> np.ndarray:
    """Gather the classical polynomial derivative exactly as ``fieldforce_ad``."""

    (ix, wx, dx), (iy, wy, dy), (iz, wz, dz) = stencil
    n = len(ix)
    order = wx.shape[1]
    result = np.zeros((n, 3), dtype=np.float64)
    for a in range(order):
        for b in range(order):
            for c in range(order):
                value = potential[ix[:, a], iy[:, b], iz[:, c]] / mesh_spacing
                result[:, 0] += dx[:, a] * wy[:, b] * wz[:, c] * value
                result[:, 1] += wx[:, a] * dy[:, b] * wz[:, c] * value
                result[:, 2] += wx[:, a] * wy[:, b] * dz[:, c] * value
    return result


@dataclass(frozen=True)
class ADOperator:
    mesh: int
    box_length: float
    order: int
    kernel: np.ndarray
    green: np.ndarray
    minimum_abs_deconvolution_product: float
    active_mode_count: int
    zeroed_active_mode_count: int


@dataclass(frozen=True)
class CellResponseMoments1D:
    mean_source: np.ndarray
    second_source: np.ndarray
    mean_gather: np.ndarray
    second_gather: np.ndarray
    mean_gradient: np.ndarray
    second_gradient: np.ndarray


@dataclass(frozen=True)
class CellResponses1D:
    source: np.ndarray
    gather: np.ndarray
    gradient: np.ndarray


def cell_responses_1d(
    fractions: np.ndarray,
    mesh: int,
    box_length: float,
    order: int,
    real_coeff: np.ndarray,
) -> CellResponses1D:
    """Discrete phase-factored stencil responses at chosen cell fractions."""

    s = np.asarray(fractions, dtype=np.float64)
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("cell fractions must lie in [0,1]")
    h = box_length / mesh
    coordinate = h * s
    indices, assignment, derivative = derivative_stencil_1d(
        coordinate, mesh, box_length, order, real_coeff
    )
    _, k1 = ikref.fft_modes(mesh, box_length)
    phase_argument = k1[:, None, None] * h * indices[None, :, :]
    phase_minus = np.exp(-1j * phase_argument)
    phase_plus = np.conjugate(phase_minus)
    particle_minus = np.exp(-1j * k1[:, None] * h * s[None, :])
    particle_plus = np.conjugate(particle_minus)
    return CellResponses1D(
        source=particle_plus
        * np.sum(phase_minus * assignment[None, :, :], axis=2),
        gather=particle_minus
        * np.sum(phase_plus * assignment[None, :, :], axis=2),
        gradient=particle_minus
        * np.sum(phase_plus * derivative[None, :, :], axis=2)
        / h,
    )


def cell_response_moments_1d(
    mesh: int,
    box_length: float,
    order: int,
    real_coeff: np.ndarray,
    quadrature_order_per_half: int = 32,
) -> CellResponseMoments1D:
    """Cell-average exact discrete source/gather responses for each FFT mode.

    Factoring out the exact particle phase makes every response periodic in
    the fractional cell coordinate ``s``.  Composite Gauss--Legendre
    quadrature on ``[0,1/2]`` and ``[1/2,1]`` resolves the odd-order stencil
    recentering at ``s=1/2`` without sampling the convention-dependent point.
    This moment representation is an exact-all-alias alternative to slowly
    convergent Fourier-image sums.
    """

    if quadrature_order_per_half < 8:
        raise ValueError("quadrature_order_per_half must be at least 8")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order_per_half)
    s_parts = []
    w_parts = []
    for lo, hi in ((0.0, 0.5), (0.5, 1.0)):
        s_parts.append(0.5 * (hi - lo) * nodes + 0.5 * (hi + lo))
        w_parts.append(0.5 * (hi - lo) * weights)
    s = np.concatenate(s_parts)
    weights_cell = np.concatenate(w_parts)
    response = cell_responses_1d(s, mesh, box_length, order, real_coeff)
    source = response.source
    gather = response.gather
    gradient = response.gradient

    def mean(values: np.ndarray) -> np.ndarray:
        return np.sum(values * weights_cell[None, :], axis=1)

    def second(values: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(values) ** 2 * weights_cell[None, :], axis=1)

    return CellResponseMoments1D(
        mean_source=mean(source),
        second_source=second(source),
        mean_gather=mean(gather),
        second_gather=second(gather),
        mean_gradient=mean(gradient),
        second_gradient=second(gradient),
    )


def ad_self_response_cell_grid(
    fractions: np.ndarray,
    operator: ADOperator,
    real_coeff: np.ndarray,
) -> np.ndarray:
    """Unit-charge uncorrected self force on a tensor cell grid.

    All reciprocal modes are summed coherently before the cell RMS is taken.
    This ordering is essential: a modewise variance sum is not a self-force
    estimator because the same particle phase couples all modes.
    """

    response = cell_responses_1d(
        fractions,
        operator.mesh,
        operator.box_length,
        operator.order,
        real_coeff,
    )
    ordinary = response.gather * response.source
    derivative = response.gradient * response.source
    green = operator.green.astype(np.complex128, copy=False)
    fx = np.einsum(
        "ijk,ia,jb,kc->abc",
        green,
        derivative,
        ordinary,
        ordinary,
        optimize=True,
    )
    fy = np.einsum(
        "ijk,ia,jb,kc->abc",
        green,
        ordinary,
        derivative,
        ordinary,
        optimize=True,
    )
    fz = np.einsum(
        "ijk,ia,jb,kc->abc",
        green,
        ordinary,
        ordinary,
        derivative,
        optimize=True,
    )
    scale = ikref.COULOMB_REAL / operator.box_length**3
    result = scale * np.stack((fx, fy, fz), axis=-1)
    if float(np.max(np.abs(result.imag))) > 2.0e-10:
        raise FloatingPointError("self response has an unexpected imaginary component")
    return result.real


def ad_self_response_at_fractions(
    fractions: np.ndarray,
    operator: ADOperator,
    real_coeff: np.ndarray,
) -> np.ndarray:
    """Unit-charge coherent self response at arbitrary 3D cell fractions."""

    fractions = np.asarray(fractions, dtype=np.float64)
    if fractions.ndim != 2 or fractions.shape[1] != 3:
        raise ValueError("fractions must have shape (n,3)")
    response = [
        cell_responses_1d(
            fractions[:, dim],
            operator.mesh,
            operator.box_length,
            operator.order,
            real_coeff,
        )
        for dim in range(3)
    ]
    ordinary = [item.gather * item.source for item in response]
    derivative = [item.gradient * item.source for item in response]
    green = operator.green.astype(np.complex128, copy=False)
    fx = np.einsum(
        "abc,ai,bi,ci->i",
        green,
        derivative[0],
        ordinary[1],
        ordinary[2],
        optimize=True,
    )
    fy = np.einsum(
        "abc,ai,bi,ci->i",
        green,
        ordinary[0],
        derivative[1],
        ordinary[2],
        optimize=True,
    )
    fz = np.einsum(
        "abc,ai,bi,ci->i",
        green,
        ordinary[0],
        ordinary[1],
        derivative[2],
        optimize=True,
    )
    scale = ikref.COULOMB_REAL / operator.box_length**3
    result = scale * np.column_stack((fx, fy, fz))
    if float(np.max(np.abs(result.imag))) > 2.0e-10:
        raise FloatingPointError("particle self response has an unexpected imaginary component")
    return result.real


def build_ad_operator(
    mesh: int,
    box_length: float,
    order: int,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: ikref.PSWFCoefficients,
) -> ADOperator:
    """Build the scalar fixed diagonal used by the development AD solver."""

    split = ikref.split_continuation(coeff, csplit)
    kernel = ikref.split_kernel_grid(
        mesh, box_length, rcut, csplit, split, band_limited=True
    )
    _, k1 = ikref.fft_modes(mesh, box_length)
    h = box_length / mesh
    d1 = lammps_deconvolution_1d(k1, h, order, cspread, coeff.spread)
    d3 = d1[:, None, None] * d1[None, :, None] * d1[None, None, :]
    active = kernel != 0.0
    active_count = int(np.count_nonzero(active))
    zeroed = active & (d3 == 0.0)
    retained = active & ~zeroed
    stability = float(np.min(np.abs(d3[active])))
    green = np.zeros_like(kernel)
    # Match ESP::compute_gf_ad exactly: g1/g2 are initialized to zero and a
    # mode whose Fourier-polynomial denominator is exactly zero is skipped.
    # Such a mode is therefore an omitted exact Fourier contribution, not an
    # invalid operator.  Near-zero but nonzero denominators are retained, as
    # they are in the production code.
    green[retained] = kernel[retained] / (d3[retained] * d3[retained])
    return ADOperator(
        mesh,
        box_length,
        order,
        kernel,
        green,
        stability,
        active_count,
        int(np.count_nonzero(zeroed)),
    )


def fixed_ad_mesh_force(
    q: np.ndarray,
    xyz: np.ndarray,
    operator: ADOperator,
    real_coeff: np.ndarray,
) -> np.ndarray:
    """Uncorrected AD mesh force with the development-solver convention."""

    stencil = particle_stencil_with_derivative(
        xyz, operator.mesh, operator.box_length, operator.order, real_coeff
    )
    density = ikref.spread_density(
        q, _plain_stencil(stencil), operator.mesh, operator.box_length
    )
    potential = np.fft.ifftn(operator.green * np.fft.fftn(density)).real
    gradient = gather_ad_gradient(stencil, potential, operator.box_length / operator.mesh)
    return ikref.COULOMB_REAL * q[:, None] * gradient


def exact_self_force(
    q: np.ndarray,
    xyz: np.ndarray,
    operator: ADOperator,
    real_coeff: np.ndarray,
) -> np.ndarray:
    """Compute the uncorrected one-body AD response at every particle.

    This is intentionally an exact numerical isolation of the operator's self
    term, not the two-harmonic self correction currently implemented in
    LAMMPS.  It lets the pair-alias estimator be tested independently before
    the implementation-specific residual self floor is reintroduced.
    """

    result = np.empty((len(q), 3), dtype=np.float64)
    for index in range(len(q)):
        result[index] = fixed_ad_mesh_force(
            q[index : index + 1],
            xyz[index : index + 1],
            operator,
            real_coeff,
        )[0]
    return result


def fixed_ad_pair_estimate_homogeneous(
    q: np.ndarray,
    box_length: float,
    mesh: int,
    order: int,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: ikref.PSWFCoefficients,
    alias_shell: int,
) -> tuple[float, dict[str, float | int | str]]:
    """Exact double-alias weak-correlation variance for the AD pair error.

    Both source and gather aliases are retained.  The gather factor is the
    directly transformed classical derivative of the implemented polynomial
    stencil.  The zero-alias mismatch relative to ``-i*k`` is retained, and
    the finite cubic alias set is ``[-L,L]^3`` in each of the two alias sums.
    Because the two sums factor by dimension, evaluating the full double sum
    is inexpensive.
    """

    if alias_shell < 0:
        raise ValueError("alias_shell must be nonnegative")
    operator = build_ad_operator(
        mesh, box_length, order, rcut, csplit, cspread, coeff
    )
    _, k1, kx, ky, kz, _ = ikref.reciprocal_arrays(mesh, box_length)
    h = box_length / mesh
    aliases = np.arange(-alias_shell, alias_shell + 1, dtype=np.int64)
    q_alias = k1[:, None] + (2.0 * math.pi / h) * aliases[None, :]
    transforms = actual_window_transforms_1d(q_alias, h, order, coeff.real)

    source_sum = np.sum(np.abs(transforms.source) ** 2, axis=1)
    gather_sum = np.sum(np.abs(transforms.gather) ** 2, axis=1)
    # The two should agree to roundoff for a real assignment window.  Keep the
    # source and gather names explicit so transform conventions remain audited.
    transform_symmetry = float(
        np.max(np.abs(source_sum - gather_sum)) / max(float(np.max(source_sum)), 1.0)
    )
    gradient_sum = np.sum(np.abs(transforms.gradient) ** 2, axis=1)
    zero_index = alias_shell
    source0 = transforms.source[:, zero_index]
    gather0 = transforms.gather[:, zero_index]
    gradient0 = transforms.gradient[:, zero_index]

    total_source = (
        source_sum[:, None, None]
        * source_sum[None, :, None]
        * source_sum[None, None, :]
    )
    total_gradient = (
        gradient_sum[:, None, None]
        * gather_sum[None, :, None]
        * gather_sum[None, None, :]
        + gather_sum[:, None, None]
        * gradient_sum[None, :, None]
        * gather_sum[None, None, :]
        + gather_sum[:, None, None]
        * gather_sum[None, :, None]
        * gradient_sum[None, None, :]
    )

    source_zero_3 = (
        source0[:, None, None] * source0[None, :, None] * source0[None, None, :]
    )
    gather_zero_x = (
        gradient0[:, None, None]
        * gather0[None, :, None]
        * gather0[None, None, :]
    )
    gather_zero_y = (
        gather0[:, None, None]
        * gradient0[None, :, None]
        * gather0[None, None, :]
    )
    gather_zero_z = (
        gather0[:, None, None]
        * gather0[None, :, None]
        * gradient0[None, None, :]
    )
    gather_zero_norm2 = (
        np.abs(gather_zero_x) ** 2
        + np.abs(gather_zero_y) ** 2
        + np.abs(gather_zero_z) ** 2
    )

    d1 = lammps_deconvolution_1d(k1, h, order, cspread, coeff.spread)
    d3 = d1[:, None, None] * d1[None, :, None] * d1[None, None, :]
    active = operator.kernel != 0.0
    zeroed = active & (d3 == 0.0)
    retained = active & ~zeroed
    inv_d2 = np.zeros_like(d3)
    inv_d2[retained] = 1.0 / (d3[retained] * d3[retained])

    mesh0_x = gather_zero_x * source_zero_3 * inv_d2
    mesh0_y = gather_zero_y * source_zero_3 * inv_d2
    mesh0_z = gather_zero_z * source_zero_3 * inv_d2
    zero_variance = (
        np.abs(mesh0_x + 1j * kx) ** 2
        + np.abs(mesh0_y + 1j * ky) ** 2
        + np.abs(mesh0_z + 1j * kz) ** 2
    )

    nonzero_alias_variance = np.zeros_like(operator.kernel)
    nonzero_alias_variance[retained] = (
        (
            total_gradient[retained] * total_source[retained]
            - gather_zero_norm2[retained] * np.abs(source_zero_3[retained]) ** 2
        )
        * inv_d2[retained]
        * inv_d2[retained]
    )
    # Numerical cancellation can create tiny negative values when L=0.
    nonzero_alias_variance = np.maximum(nonzero_alias_variance, 0.0)
    mode_variance = operator.kernel * operator.kernel * (
        zero_variance + nonzero_alias_variance
    )
    chi2 = float(np.sum(mode_variance) / box_length**6)

    q = np.asarray(q, dtype=np.float64)
    q2 = q * q
    charge_factor_pair = float(
        ((np.sum(q2) ** 2) - np.sum(q2 * q2)) / len(q)
    )
    estimate = ikref.COULOMB_REAL * math.sqrt(max(charge_factor_pair * chi2, 0.0))
    zero_chi2 = float(
        np.sum(operator.kernel * operator.kernel * zero_variance) / box_length**6
    )
    alias_chi2 = float(
        np.sum(operator.kernel * operator.kernel * nonzero_alias_variance)
        / box_length**6
    )
    zero_mode_missing_chi2 = float(
        np.sum(
            operator.kernel[zeroed]
            * operator.kernel[zeroed]
            * (kx[zeroed] * kx[zeroed] + ky[zeroed] * ky[zeroed] + kz[zeroed] * kz[zeroed])
        )
        / box_length**6
    )
    return estimate, {
        "operator": "fixed AD; LAMMPS Fourier-polynomial deconvolution; actual classical derivative transform",
        "alias_shell": alias_shell,
        "charge_factor_pair": charge_factor_pair,
        "chi2_pair": chi2,
        "chi2_zero_alias_mismatch": zero_chi2,
        "chi2_nonzero_double_alias": alias_chi2,
        "transform_source_gather_symmetry_relative": transform_symmetry,
        "minimum_abs_deconvolution_product": operator.minimum_abs_deconvolution_product,
        "active_mode_count": operator.active_mode_count,
        "zeroed_active_mode_count": operator.zeroed_active_mode_count,
        "zeroed_active_mode_fraction": operator.zeroed_active_mode_count
        / operator.active_mode_count,
        "zero_mode_missing_chi2": zero_mode_missing_chi2,
        "zero_mode_missing_pair_force": ikref.COULOMB_REAL
        * math.sqrt(max(charge_factor_pair * zero_mode_missing_chi2, 0.0)),
        "zero_deconvolution_policy": "green=0 when D=0 (matched LAMMPS compute_gf_ad)",
    }


def fixed_ad_pair_estimate_cell_moments(
    q: np.ndarray,
    box_length: float,
    mesh: int,
    order: int,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: ikref.PSWFCoefficients,
    quadrature_order_per_half: int = 32,
) -> tuple[float, dict[str, float | int | str]]:
    """Exact-all-alias weak-correlation AD pair estimator.

    The calculation averages the *implemented discrete stencil responses*
    over one mesh cell before using reciprocal-mode orthogonality.  It thus
    includes source aliases, gather aliases, their double-alias products,
    the zero-alias mismatch, polynomial boundary terms, and all cross-alias
    covariances that collapse into the same cell response.  No alias-shell
    cutoff or ``i q W_hat`` substitution is used.
    """

    operator = build_ad_operator(
        mesh, box_length, order, rcut, csplit, cspread, coeff
    )
    moments = cell_response_moments_1d(
        mesh,
        box_length,
        order,
        coeff.real,
        quadrature_order_per_half=quadrature_order_per_half,
    )
    _, k1, kx, ky, kz, _ = ikref.reciprocal_arrays(mesh, box_length)
    h = box_length / mesh
    d1 = lammps_deconvolution_1d(k1, h, order, cspread, coeff.spread)
    d3 = d1[:, None, None] * d1[None, :, None] * d1[None, None, :]
    active = operator.kernel != 0.0
    zeroed = active & (d3 == 0.0)
    retained = active & ~zeroed
    inv_d2 = np.zeros_like(d3)
    inv_d2[retained] = 1.0 / (d3[retained] * d3[retained])

    mean_source_3 = (
        moments.mean_source[:, None, None]
        * moments.mean_source[None, :, None]
        * moments.mean_source[None, None, :]
    )
    second_source_3 = (
        moments.second_source[:, None, None]
        * moments.second_source[None, :, None]
        * moments.second_source[None, None, :]
    )
    mean_bx = (
        moments.mean_gradient[:, None, None]
        * moments.mean_gather[None, :, None]
        * moments.mean_gather[None, None, :]
    )
    mean_by = (
        moments.mean_gather[:, None, None]
        * moments.mean_gradient[None, :, None]
        * moments.mean_gather[None, None, :]
    )
    mean_bz = (
        moments.mean_gather[:, None, None]
        * moments.mean_gather[None, :, None]
        * moments.mean_gradient[None, None, :]
    )
    second_b = (
        moments.second_gradient[:, None, None]
        * moments.second_gather[None, :, None]
        * moments.second_gather[None, None, :]
        + moments.second_gather[:, None, None]
        * moments.second_gradient[None, :, None]
        * moments.second_gather[None, None, :]
        + moments.second_gather[:, None, None]
        * moments.second_gather[None, :, None]
        * moments.second_gradient[None, None, :]
    )

    mean_mesh_x = mean_bx * mean_source_3 * inv_d2
    mean_mesh_y = mean_by * mean_source_3 * inv_d2
    mean_mesh_z = mean_bz * mean_source_3 * inv_d2
    second_mesh = np.zeros_like(operator.kernel)
    second_mesh[retained] = (
        second_b[retained]
        * second_source_3[retained]
        * inv_d2[retained] ** 2
    )
    mean_mesh_norm2 = (
        np.abs(mean_mesh_x) ** 2
        + np.abs(mean_mesh_y) ** 2
        + np.abs(mean_mesh_z) ** 2
    )
    alias_fluctuation = np.maximum(second_mesh - mean_mesh_norm2, 0.0)
    zero_mismatch = (
        np.abs(mean_mesh_x + 1j * kx) ** 2
        + np.abs(mean_mesh_y + 1j * ky) ** 2
        + np.abs(mean_mesh_z + 1j * kz) ** 2
    )
    mode_variance = operator.kernel * operator.kernel * (
        alias_fluctuation + zero_mismatch
    )
    chi2 = float(np.sum(mode_variance) / box_length**6)
    zero_chi2 = float(
        np.sum(operator.kernel * operator.kernel * zero_mismatch) / box_length**6
    )
    alias_chi2 = float(
        np.sum(operator.kernel * operator.kernel * alias_fluctuation)
        / box_length**6
    )
    zero_mode_missing_chi2 = float(
        np.sum(
            operator.kernel[zeroed]
            * operator.kernel[zeroed]
            * (kx[zeroed] * kx[zeroed] + ky[zeroed] * ky[zeroed] + kz[zeroed] * kz[zeroed])
        )
        / box_length**6
    )

    q = np.asarray(q, dtype=np.float64)
    q2 = q * q
    charge_factor_pair = float(
        ((np.sum(q2) ** 2) - np.sum(q2 * q2)) / len(q)
    )
    estimate = ikref.COULOMB_REAL * math.sqrt(max(charge_factor_pair * chi2, 0.0))
    source_gather_mean_symmetry = float(
        np.max(np.abs(moments.mean_gather - np.conjugate(moments.mean_source)))
    )
    return estimate, {
        "operator": "fixed AD; LAMMPS Fourier-polynomial deconvolution; exact implemented cell moments",
        "quadrature_order_per_half": quadrature_order_per_half,
        "charge_factor_pair": charge_factor_pair,
        "chi2_pair": chi2,
        "chi2_zero_alias_mismatch": zero_chi2,
        "chi2_all_alias_fluctuation": alias_chi2,
        "source_gather_mean_symmetry_absolute": source_gather_mean_symmetry,
        "minimum_abs_deconvolution_product": operator.minimum_abs_deconvolution_product,
        "active_mode_count": operator.active_mode_count,
        "zeroed_active_mode_count": operator.zeroed_active_mode_count,
        "zeroed_active_mode_fraction": operator.zeroed_active_mode_count
        / operator.active_mode_count,
        "zero_mode_missing_chi2": zero_mode_missing_chi2,
        "zero_mode_missing_pair_force": ikref.COULOMB_REAL
        * math.sqrt(max(charge_factor_pair * zero_mode_missing_chi2, 0.0)),
        "zero_deconvolution_policy": "green=0 when D=0 (matched LAMMPS compute_gf_ad)",
    }


def formal_iq_window_single_alias_estimate(
    q: np.ndarray,
    box_length: float,
    mesh: int,
    order: int,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: ikref.PSWFCoefficients,
    alias_shell: int,
) -> float:
    """Audit the manuscript's formal ``|k+G|^2 |W_G/W_0|^2`` sum.

    This intentionally applies the distributional ``i q W_hat`` identity to
    the compact polynomial window and retains only the printed gather-image
    term.  It is provided solely to demonstrate why that expression is not a
    converged estimator for the implemented classical derivative.
    """

    operator = build_ad_operator(
        mesh, box_length, order, rcut, csplit, cspread, coeff
    )
    _, k1 = ikref.fft_modes(mesh, box_length)
    h = box_length / mesh
    aliases = np.arange(-alias_shell, alias_shell + 1, dtype=np.int64)
    q_alias = k1[:, None] + (2.0 * math.pi / h) * aliases[None, :]
    transforms = actual_window_transforms_1d(q_alias, h, order, coeff.real)
    w2 = np.abs(transforms.source) ** 2
    sum_w2 = np.sum(w2, axis=1)
    sum_q2w2 = np.sum(q_alias * q_alias * w2, axis=1)
    zero = alias_shell
    w0 = transforms.source[:, zero]
    w03 = w0[:, None, None] * w0[None, :, None] * w0[None, None, :]
    _, _, kx, ky, kz, _ = ikref.reciprocal_arrays(mesh, box_length)
    weighted = (
        sum_q2w2[:, None, None] * sum_w2[None, :, None] * sum_w2[None, None, :]
        + sum_w2[:, None, None] * sum_q2w2[None, :, None] * sum_w2[None, None, :]
        + sum_w2[:, None, None] * sum_w2[None, :, None] * sum_q2w2[None, None, :]
        - (kx * kx + ky * ky + kz * kz) * np.abs(w03) ** 2
    )
    denominator = np.abs(w03) ** 2
    active = operator.kernel != 0.0
    ratio = np.zeros_like(operator.kernel)
    ratio[active] = weighted[active] / denominator[active]
    chi2 = float(
        np.sum(operator.kernel * operator.kernel * ratio) / box_length**6
    )
    q = np.asarray(q, dtype=np.float64)
    q2 = q * q
    charge_factor_pair = float(
        ((np.sum(q2) ** 2) - np.sum(q2 * q2)) / len(q)
    )
    return ikref.COULOMB_REAL * math.sqrt(max(charge_factor_pair * chi2, 0.0))
