#!/usr/bin/env python3
r"""Configuration-conditioned, all-source AD aliasing quadratic form.

The Figure-5 AD prediction is evaluated from the same discrete operator used
by ESP/AD, but it never invokes LAMMPS or an Ewald-force dump.  For each pilot
configuration the source density is the full

``rho(q) = sum_j q_j exp(-i q.r_j)``.

Consequently the field on a target particle contains its ``j=i`` source
contribution.  The production two-harmonic self correction is subsequently
applied to that *same vector field*, before the force RMS is taken.  This
retains every alias/self cross term.  The old off-diagonal ``rho-q_i``
descriptor is deliberately not used here.

The implementation writes the discrete source and AD gather operators out
explicitly, rather than invoking the legacy Figure-5 direct-force diagnostic
or reading a force dump. The continuum zero-alias response is the reference
term in the analytical operator difference; no force from an Ewald
calculation enters this module. Since the complete configuration-conditioned
quadratic form is evaluated without a statistical closure, it is algebraically
identical to the squared discrete-mesh-minus-zero-alias-band field evaluated
on the same coordinates. The post-freeze direct-band diagnostic tests that
identity independently; it does not calibrate or alter the selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ad_operator_reference as adref
import fixed_ik_reference as ikref


@dataclass(frozen=True)
class AllSourceADField:
    """Per-frame components of the full-source AD quadratic form in force units."""

    raw_mesh: np.ndarray
    continuum_band: np.ndarray
    raw_aliasing: np.ndarray
    self_correction: np.ndarray
    corrected_aliasing: np.ndarray


def _implemented_mesh_field(
    q: np.ndarray,
    xyz: np.ndarray,
    operator: adref.ADOperator,
    real_coeff: np.ndarray,
) -> np.ndarray:
    """Evaluate the all-source discrete AD field from its source/gather map."""

    stencil = adref.particle_stencil_with_derivative(
        xyz, operator.mesh, operator.box_length, operator.order, real_coeff
    )
    density = ikref.spread_density(
        q,
        tuple((indices, weights) for indices, weights, _ in stencil),
        operator.mesh,
        operator.box_length,
    )
    potential = np.fft.ifftn(operator.green * np.fft.fftn(density)).real
    gradient = adref.gather_ad_gradient(
        stencil, potential, operator.box_length / operator.mesh
    )
    return ikref.COULOMB_REAL * np.asarray(q, dtype=np.float64)[:, None] * gradient


def _continuum_band_field(
    q: np.ndarray,
    xyz: np.ndarray,
    operator: adref.ADOperator,
) -> np.ndarray:
    """Evaluate the zero-alias continuum response on the AD reciprocal band.

    This is the reference term of the analytical operator difference.  Its
    density is the physical full-source density, not a pair-only density.
    """

    mesh = operator.mesh
    active = np.argwhere(operator.kernel != 0.0)
    if not len(active):
        return np.zeros((len(q), 3), dtype=np.float64)
    axis = np.rint(np.fft.fftfreq(mesh) * mesh).astype(np.int64)
    modes = np.column_stack(
        (axis[active[:, 0]], axis[active[:, 1]], axis[active[:, 2]])
    )
    kernel = operator.kernel[active[:, 0], active[:, 1], active[:, 2]]
    wavevectors = (2.0 * np.pi / operator.box_length) * modes
    charges = np.asarray(q, dtype=np.float64)
    coordinates = np.asarray(xyz, dtype=np.float64)
    field = np.zeros((len(charges), 3), dtype=np.float64)
    # The split kernel is compact in the reciprocal band.  Iterating only its
    # active modes avoids an O(N M^3) dense-cube evaluation on fine grids.
    for start in range(0, len(kernel), 256):
        stop = min(start + 256, len(kernel))
        kblock = wavevectors[start:stop]
        phase_minus = np.exp(-1j * coordinates @ kblock.T)
        rho = charges @ phase_minus
        mode_field = (-1j * kernel[start:stop] * rho)[:, None] * kblock
        field += charges[:, None] * (phase_minus.conj() @ mode_field).real
    return ikref.COULOMB_REAL * field / operator.box_length**3


def evaluate_all_source_ad_aliasing(
    q: np.ndarray,
    xyz: np.ndarray,
    operator: adref.ADOperator,
    real_coeff: np.ndarray,
    self_correction: np.ndarray,
) -> AllSourceADField:
    """Return the corrected all-source AD aliasing field for one frame.

    ``self_correction`` contains the two production LAMMPS coefficients.  It
    is applied particlewise after the raw full-density AD operator is formed;
    it is therefore not independently RMS-averaged.
    """

    charges = np.asarray(q, dtype=np.float64)
    coordinates = np.asarray(xyz, dtype=np.float64)
    if coordinates.shape != (len(charges), 3):
        raise ValueError("xyz must have shape (len(q), 3)")
    raw_mesh = _implemented_mesh_field(charges, coordinates, operator, real_coeff)
    continuum_band = _continuum_band_field(charges, coordinates, operator)
    raw_aliasing = raw_mesh - continuum_band
    fraction = np.mod(coordinates / (operator.box_length / operator.mesh), 1.0)
    correction = (charges * charges)[:, None] * (
        self_correction[0] * np.sin(2.0 * np.pi * fraction)
        + self_correction[1] * np.sin(4.0 * np.pi * fraction)
    )
    return AllSourceADField(
        raw_mesh=raw_mesh,
        continuum_band=continuum_band,
        raw_aliasing=raw_aliasing,
        self_correction=correction,
        corrected_aliasing=raw_aliasing - correction,
    )


def mean_square(vector: np.ndarray) -> float:
    """Return the particle-averaged squared vector norm."""

    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("vector must have shape (n, 3)")
    return float(np.mean(np.einsum("ij,ij->i", values, values)))
