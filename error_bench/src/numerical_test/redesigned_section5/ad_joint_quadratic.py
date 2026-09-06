#!/usr/bin/env python3
r"""Post-selection phase-resolved AD pair/self diagnostic for Figure 5.

The production AD reciprocal error over its represented Fourier band is
written as one particlewise amplitude,

.. math::

   \mathbf e_i^{\rm band}
   = \mathbf e_{i,\rm full\ source}^{\rm alias}
     - \mathbf c_i^{\rm self}
   = \mathbf e_{i,\rm distinct\ pair}^{\rm alias}
     + \mathbf e_{i,\rm self}^{\rm residual}.

The full-source term contains every source charge, including ``j=i``. Its raw
one-particle response is isolated analytically, leaving the distinct-source
pair vector; subtracting the two-harmonic correction from that raw self
response gives the residual-self vector. These two vectors are added before
any norm or particle average is taken. Expanding the implemented calculation
gives

.. math::

   \langle|\mathbf e^{\rm band}|^2\rangle
   = \langle|\mathbf e_{\rm pair}|^2\rangle
     + \langle|\mathbf e_{\rm self}^{\rm residual}|^2\rangle
     + 2\langle\mathbf e_{\rm pair}\!\cdot
       \mathbf e_{\rm self}^{\rm residual}\rangle.

Thus all pair/self and cross-mode coherence present in the 25 pilot
configurations is retained.  This module evaluates the analytical discrete
source/gather transfer minus its continuum in-band transfer in one
accumulator.  It does not read LAMMPS forces, Ewald forces, holdout
coordinates, or a finite-band force-difference table.

This object is not the Figure-5 prediction or selection statistic.  The main
curve uses frozen ``S_tag`` for the zero-mean diagonal term and a frozen
charge-class conditional amplitude for the coherent pair/self term.  This
diagnostic is run only after selection to quantify cross-mode and residual
pair/self covariance omitted by that conditional-mean closure.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
AD_AUDIT = HERE / "ad_operator_audit"
if str(AD_AUDIT) not in sys.path:
    sys.path.insert(0, str(AD_AUDIT))

import ad_operator_reference as adref
import fixed_ik_reference as ikref


@dataclass(frozen=True)
class JointADBandQuadratic:
    """Particle-averaged moments of the vector-complete in-band error."""

    full_source_alias_mean_square: float
    self_correction_mean_square: float
    alias_minus_correction_dot_mean: float
    distinct_pair_mean_square: float
    residual_self_mean_square: float
    pair_residual_self_dot_mean: float
    joint_pair_self_mean_square: float
    vector_identity_absolute_residual: float
    component_identity_max_abs: float


def _mean_square(vector: np.ndarray) -> float:
    return float(np.mean(np.einsum("ij,ij->i", vector, vector)))


def _unit_self_response_at_fractions(
    fractions: np.ndarray,
    operator: adref.ADOperator,
    real_coeff: np.ndarray,
    *,
    mode_block_size: int,
) -> np.ndarray:
    """Evaluate the uncorrected unit-charge AD self vector sparsely."""

    response = [
        adref.cell_responses_1d(
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
    active = np.argwhere(operator.green != 0.0)
    result = np.zeros((len(fractions), 3), dtype=np.complex128)
    for start in range(0, len(active), mode_block_size):
        block = active[start : start + mode_block_size]
        ix, iy, iz = block.T
        green = operator.green[ix, iy, iz]
        result[:, 0] += np.einsum(
            "m,mi,mi,mi->i",
            green,
            derivative[0][ix],
            ordinary[1][iy],
            ordinary[2][iz],
            optimize=True,
        )
        result[:, 1] += np.einsum(
            "m,mi,mi,mi->i",
            green,
            ordinary[0][ix],
            derivative[1][iy],
            ordinary[2][iz],
            optimize=True,
        )
        result[:, 2] += np.einsum(
            "m,mi,mi,mi->i",
            green,
            ordinary[0][ix],
            ordinary[1][iy],
            derivative[2][iz],
            optimize=True,
        )
    result *= ikref.COULOMB_REAL / operator.box_length**3
    imaginary_max = float(np.max(np.abs(result.imag), initial=0.0))
    if imaginary_max > 2.0e-10:
        raise FloatingPointError(
            "particle self response has an unexpected imaginary component"
        )
    return result.real


def evaluate_joint_pair_self_quadratic(
    q: np.ndarray,
    xyz: np.ndarray,
    operator: adref.ADOperator,
    real_coeff: np.ndarray,
    self_correction: np.ndarray,
    *,
    mode_block_size: int = 256,
) -> JointADBandQuadratic:
    """Contract the full-source AD pair and self vectors before squaring.

    The first contribution is the implemented source/spread, Green-function,
    and AD-gather map.  The continuum zero-alias contribution is accumulated
    immediately with the opposite sign into the same array.  The production
    self correction is then subtracted, after which the RMS moment is formed.
    No separately materialized mesh-force or direct-band-force array enters
    the calculation.
    """

    charges = np.asarray(q, dtype=np.float64)
    coordinates = np.asarray(xyz, dtype=np.float64)
    correction_coeff = np.asarray(self_correction, dtype=np.float64)
    if coordinates.shape != (len(charges), 3):
        raise ValueError("xyz must have shape (len(q), 3)")
    if correction_coeff.shape != (2,):
        raise ValueError(
            "self_correction must contain the sin(2*pi*s) and "
            "sin(4*pi*s) coefficients"
        )
    if mode_block_size < 1:
        raise ValueError("mode_block_size must be positive")

    stencil = adref.particle_stencil_with_derivative(
        coordinates,
        operator.mesh,
        operator.box_length,
        operator.order,
        np.asarray(real_coeff, dtype=np.float64),
    )
    density = ikref.spread_density(
        charges,
        tuple((indices, weights) for indices, weights, _ in stencil),
        operator.mesh,
        operator.box_length,
    )
    potential = np.fft.ifftn(operator.green * np.fft.fftn(density)).real
    gradient = adref.gather_ad_gradient(
        stencil, potential, operator.box_length / operator.mesh
    )
    # This is one analytical error amplitude.  The continuum term below is
    # added with the cancellation sign instead of being stored as a second
    # force array and subtracted after two independent RMS calculations.
    band_error = ikref.COULOMB_REAL * charges[:, None] * gradient

    active = np.argwhere(operator.kernel != 0.0)
    if len(active):
        axis = np.rint(np.fft.fftfreq(operator.mesh) * operator.mesh).astype(
            np.int64
        )
        modes = np.column_stack(
            (axis[active[:, 0]], axis[active[:, 1]], axis[active[:, 2]])
        )
        kernel = operator.kernel[active[:, 0], active[:, 1], active[:, 2]]
        wavevectors = 2.0 * math.pi / operator.box_length * modes
        prefactor = ikref.COULOMB_REAL / operator.box_length**3
        for start in range(0, len(kernel), mode_block_size):
            stop = min(start + mode_block_size, len(kernel))
            kblock = wavevectors[start:stop]
            phase_minus = np.exp(-1j * coordinates @ kblock.T)
            rho = charges @ phase_minus
            # The continuum in-band field uses -i*k*K(k).  Its contribution
            # to mesh-minus-continuum is therefore +i*k*K(k).
            cancellation = (1j * kernel[start:stop] * rho)[:, None] * kblock
            band_error += (
                prefactor
                * charges[:, None]
                * (phase_minus.conj() @ cancellation).real
            )

    full_source_alias = band_error.copy()
    mesh_spacing = operator.box_length / operator.mesh
    fractions = np.mod(coordinates / mesh_spacing, 1.0)
    correction_vector = charges[:, None] ** 2 * (
        correction_coeff[0] * np.sin(2.0 * math.pi * fractions)
        + correction_coeff[1] * np.sin(4.0 * math.pi * fractions)
    )
    band_error -= correction_vector

    unit_self = _unit_self_response_at_fractions(
        fractions,
        operator,
        np.asarray(real_coeff, dtype=np.float64),
        mode_block_size=mode_block_size,
    )
    raw_self = charges[:, None] ** 2 * unit_self
    distinct_pair = full_source_alias - raw_self
    residual_self = raw_self - correction_vector
    component_sum = distinct_pair + residual_self
    component_identity = float(np.max(np.abs(component_sum - band_error)))
    component_tolerance = 8.0e-13 * max(
        float(np.max(np.abs(band_error), initial=0.0)), 1.0e-300
    )
    if component_identity > component_tolerance:
        raise FloatingPointError(
            "joint AD error differs from pair plus residual-self components"
        )

    raw2 = _mean_square(full_source_alias)
    correction2 = _mean_square(correction_vector)
    cross = float(
        np.mean(
            np.einsum(
                "ij,ij->i", full_source_alias, -correction_vector
            )
        )
    )
    pair2 = _mean_square(distinct_pair)
    residual_self2 = _mean_square(residual_self)
    pair_self_cross = float(
        np.mean(np.einsum("ij,ij->i", distinct_pair, residual_self))
    )
    joint2 = _mean_square(band_error)
    expanded = raw2 + correction2 + 2.0 * cross
    residual = abs(joint2 - expanded)
    tolerance = 2.0e-12 * max(joint2, raw2, correction2, 1.0e-300)
    if residual > tolerance:
        raise FloatingPointError(
            "joint AD pair/self vector identity failed: "
            f"{joint2:.16e} != {expanded:.16e}"
        )
    if joint2 < 0.0:
        raise FloatingPointError("joint AD pair/self mean square became negative")
    component_expanded = pair2 + residual_self2 + 2.0 * pair_self_cross
    if not math.isclose(
        joint2,
        component_expanded,
        rel_tol=2.0e-12,
        abs_tol=2.0e-18,
    ):
        raise FloatingPointError(
            "pair plus residual-self quadratic expansion is inconsistent"
        )
    return JointADBandQuadratic(
        full_source_alias_mean_square=raw2,
        self_correction_mean_square=correction2,
        alias_minus_correction_dot_mean=cross,
        distinct_pair_mean_square=pair2,
        residual_self_mean_square=residual_self2,
        pair_residual_self_dot_mean=pair_self_cross,
        joint_pair_self_mean_square=joint2,
        vector_identity_absolute_residual=residual,
        component_identity_max_abs=component_identity,
    )
