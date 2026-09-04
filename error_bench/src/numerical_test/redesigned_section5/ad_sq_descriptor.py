#!/usr/bin/env python3
r"""Structure-aware descriptor for the implemented AD mesh-pair estimator.

This is retained for Figure 4 and for explicit legacy Figure-5 diagnostics.
It is not the current Figure-5 AD estimator: the latter uses the full-source,
target-cell-phase-resolved quadratic form in ad_all_source_theory.

The production AD path uses the classical derivative of the tabulated
piecewise-polynomial spreading stencil.  Its homogeneous pair estimator is
therefore evaluated with the exact one-cell response moments in
``ad_operator_reference``.  This module adds a deliberately narrow
structure-aware extension to that *same* operator.

For each resolved mesh mode, the exact homogeneous cell-moment variance can
be decomposed into positive contributions from the physical source wave
vectors ``k + 2*pi*l/h``.  We replace the homogeneous source variance by the
measured diagonal charge spectrum,

    <rho(q) rho(q')*> ~= Q S_q(q) delta_{q,q'},

while retaining the actual AD gather response and the Fourier-polynomial
deconvolution.  The resulting descriptor recovers the exact cell-moment
homogeneous result identically when ``S_q = 1``.  It is nevertheless a
*diagonal source-spectrum closure*, not a full molecular AD theorem: it does
not restore off-diagonal source-alias covariance or target-conditioned
particle--grid correlations.  The residual AD self term remains a separate
one-body contribution and is not structure-factor weighted.

The source-alias correction is evaluated through a stated finite alias shell.
The uncorrected exact cell-moment value supplies the full all-alias baseline;
outside the sampled shell, the correction assumes the high-wave-vector limit
``S_q = 1``.  This makes the approximation and its shell coverage explicit.

In addition to the ordinary diagonal charge spectrum, this module provides a
``tagged_pair_spectrum``.  For a physical Fourier vector ``q`` it is

.. math::

   S_{\mathrm{tag}}(q) =
   \frac{\sum_i q_i^2\left|e^{i q\cdot r_i}\rho(q)-q_i\right|^2}
        {Q_2^2-Q_4}.

The source charge of the target particle is removed before squaring.  Thus the
numerator is the exact diagonal-in-physical-mode contribution
``sum_i q_i^2 sum_{j,l != i} q_j q_l exp[-i q.(r_j-r_l)]``.  Its random,
uncorrelated limit is one, so it can replace the scalar ``S_q`` in the existing
pair estimator without changing the homogeneous reference.  Unlike ordinary
``S_q``, it retains the lowest-order target--source correlation required by a
force RMS.  It still does *not* contain cross physical-mode or cross gather-
alias coherences; it is deliberately a low-rank diagnostic closure.
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

import ad_operator_reference as adref  # noqa: E402
import fixed_ik_reference as ikref  # noqa: E402
import sq_alias_tools as sqtools  # noqa: E402


def _canonical_modes(modes: np.ndarray) -> np.ndarray:
    """Map inversion-related nonzero reciprocal modes to one representative."""

    values = np.asarray(modes, dtype=np.int64).copy()
    nonzero = values != 0
    if np.any(~np.any(nonzero, axis=1)):
        raise ValueError("the zero mode cannot enter a structure-factor descriptor")
    first = np.argmax(nonzero, axis=1)
    sign = np.sign(values[np.arange(len(values)), first])
    values *= sign[:, None]
    return values


def _aggregate_modes(modes: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    canonical = _canonical_modes(modes)
    unique, inverse = np.unique(canonical, axis=0, return_inverse=True)
    return unique, np.bincount(inverse, weights=np.asarray(weights), minlength=len(unique))


def _shell_aliases(shell: int) -> np.ndarray:
    return np.asarray(
        [
            (a, b, c)
            for a in range(-shell, shell + 1)
            for b in range(-shell, shell + 1)
            for c in range(-shell, shell + 1)
            if max(abs(a), abs(b), abs(c)) == shell
        ],
        dtype=np.int64,
    )


def evaluate_tagged_pair_spectrum(
    q: np.ndarray,
    xyz: np.ndarray,
    box_length: float,
    modes: np.ndarray,
    *,
    chunk_size: int = 512,
    return_charge_spectrum: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate the target-conditioned diagonal pair spectrum.

    The direct definition would require one source sum for every target
    particle and Fourier mode.  Algebra reduces it to two ordinary Fourier
    charge sums:

    ``Q2*|rho_1|^2 + Q4 - 2 Re[rho_1 rho_3*]``,

    where ``rho_p(q)=sum_i q_i^p exp(-i q.r_i)``.  This is exact for the
    diagonal physical-mode contribution and costs only one additional charge
    transform relative to ``S_q``.
    """

    modes = np.asarray(modes, dtype=np.int64)
    charges = np.asarray(q, dtype=np.float64)
    coordinates = np.asarray(xyz, dtype=np.float64)
    if modes.ndim != 2 or modes.shape[1] != 3:
        raise ValueError("modes must have shape (n, 3)")
    if coordinates.shape != (len(charges), 3):
        raise ValueError("xyz must have shape (len(q), 3)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if np.any(~np.any(modes != 0, axis=1)):
        raise ValueError("the tagged pair spectrum is undefined at the zero mode")

    q2 = float(np.dot(charges, charges))
    q4 = float(np.dot(charges * charges, charges * charges))
    denominator = q2 * q2 - q4
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("at least two nonzero charges are required for a pair spectrum")

    frac = np.mod(coordinates, box_length) / box_length
    unique_x, inverse_x = np.unique(modes[:, 0], return_inverse=True)
    unique_y, inverse_y = np.unique(modes[:, 1], return_inverse=True)
    unique_z, inverse_z = np.unique(modes[:, 2], return_inverse=True)
    ex = np.exp(-2j * math.pi * np.outer(unique_x, frac[:, 0]))
    ey = np.exp(-2j * math.pi * np.outer(unique_y, frac[:, 1]))
    ez = np.exp(-2j * math.pi * np.outer(unique_z, frac[:, 2]))
    q3 = charges * charges * charges
    result = np.empty(len(modes), dtype=np.float64)
    charge_spectrum = (
        np.empty(len(modes), dtype=np.float64) if return_charge_spectrum else None
    )
    for start in range(0, len(modes), chunk_size):
        stop = min(start + chunk_size, len(modes))
        phase = ex[inverse_x[start:stop]] * ey[inverse_y[start:stop]]
        phase *= ez[inverse_z[start:stop]]
        rho1 = phase @ charges
        rho3 = phase @ q3
        if charge_spectrum is not None:
            charge_spectrum[start:stop] = (
                rho1.real * rho1.real + rho1.imag * rho1.imag
            ) / q2
        numerator = q2 * (rho1.real * rho1.real + rho1.imag * rho1.imag) + q4
        numerator -= 2.0 * np.real(rho1 * np.conjugate(rho3))
        # The expression is a sum of squared magnitudes.  Permit only the
        # roundoff-scale negative values produced by its subtraction form.
        scale = max(float(np.max(np.abs(numerator))), denominator, 1.0)
        if float(np.min(numerator)) < -128.0 * np.finfo(np.float64).eps * scale:
            raise FloatingPointError("tagged pair spectrum became materially negative")
        result[start:stop] = np.maximum(numerator, 0.0) / denominator
    if charge_spectrum is not None:
        return result, charge_spectrum
    return result


def rigid_molecule_tagged_pair_spectrum(
    modes: np.ndarray,
    box_length: float,
    molecule_count: int,
    site_charges: np.ndarray,
    site_distances: np.ndarray,
) -> np.ndarray:
    r"""Return the orientationally averaged rigid-molecule tagged spectrum.

    This is the trajectory-free counterpart of
    :func:`evaluate_tagged_pair_spectrum` for a neutral, rigid molecule whose
    centres and orientations are independently and uniformly distributed.
    Let ``b`` and ``c`` label source sites within one molecule and let
    ``Q2 = N_mol sum_b q_b^2`` be the total charge square.  Averaging the
    target-conditioned diagonal pair spectrum gives

    .. math::

       S_{\rm tag}^{\rm rigid}(k) =
       \frac{N_{\rm mol}}{Q_2^2-Q_4}
       \sum_{b,c} q_b q_c\,[Q_2-e_{bc}]\,
       \operatorname{sinc}(k r_{bc}),

    where ``e_bc=q_b^2`` for ``b=c`` and
    ``e_bc=q_b^2+q_c^2`` otherwise.  The exclusion factor is important: it
    removes the target particle from the source sum before squaring and is
    what makes this quantity appropriate for a force-RMS pair estimator.

    The formula contains the exact finite-system charge factors of the
    homogeneous AD pair estimate.  It approaches one at high wave number,
    recovers ``Q4/(Q2^2-Q4)`` at ``k=0`` for a neutral molecule, and can be
    inserted directly into :func:`corrected_chi2_with_sampling`.  It is a
    rigid intramolecular *diagonal-source-spectrum* closure only; it does not
    model intermolecular correlations or off-diagonal physical-mode aliases.
    """

    indices = np.asarray(modes, dtype=np.int64)
    charges = np.asarray(site_charges, dtype=np.float64)
    distances = np.asarray(site_distances, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("modes must have shape (n, 3)")
    if not math.isfinite(box_length) or box_length <= 0.0:
        raise ValueError("box_length must be finite and positive")
    if isinstance(molecule_count, bool) or not isinstance(molecule_count, (int, np.integer)):
        raise ValueError("molecule_count must be a positive integer")
    if molecule_count < 1:
        raise ValueError("molecule_count must be positive")
    if charges.ndim != 1 or len(charges) < 2:
        raise ValueError("site_charges must contain at least two sites")
    if distances.shape != (len(charges), len(charges)):
        raise ValueError("site_distances must have shape (nsite, nsite)")
    if not np.all(np.isfinite(charges)) or not np.all(np.isfinite(distances)):
        raise ValueError("site charges and distances must be finite")
    if np.any(distances < 0.0) or not np.allclose(distances, distances.T, atol=1.0e-13):
        raise ValueError("site_distances must be a symmetric nonnegative matrix")
    if not np.allclose(np.diag(distances), 0.0, atol=1.0e-13):
        raise ValueError("site_distances must have a zero diagonal")
    if not math.isclose(float(np.sum(charges)), 0.0, abs_tol=1.0e-12):
        raise ValueError("the rigid tagged spectrum requires a neutral molecule")

    n_molecules = int(molecule_count)
    q2_molecule = float(np.dot(charges, charges))
    q4_molecule = float(np.dot(charges * charges, charges * charges))
    q2_total = n_molecules * q2_molecule
    q4_total = n_molecules * q4_molecule
    denominator = q2_total * q2_total - q4_total
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("rigid tagged spectrum requires a nonzero pair charge factor")

    wave_number = 2.0 * math.pi / box_length * np.linalg.norm(indices, axis=1)
    numerator_per_molecule = np.zeros(len(indices), dtype=np.float64)
    for source_b in range(len(charges)):
        for source_c in range(len(charges)):
            exclusion = (
                charges[source_b] * charges[source_b]
                if source_b == source_c
                else charges[source_b] * charges[source_b]
                + charges[source_c] * charges[source_c]
            )
            orientation_average = (
                1.0
                if source_b == source_c
                else np.sinc(wave_number * distances[source_b, source_c] / math.pi)
            )
            numerator_per_molecule += (
                charges[source_b]
                * charges[source_c]
                * (q2_total - exclusion)
                * orientation_average
            )
    spectrum = n_molecules * numerator_per_molecule / denominator
    scale = max(float(np.max(np.abs(spectrum), initial=0.0)), 1.0)
    if float(np.min(spectrum, initial=0.0)) < -128.0 * np.finfo(np.float64).eps * scale:
        raise FloatingPointError("rigid tagged pair spectrum became materially negative")
    return np.maximum(spectrum, 0.0)


def rigid_spce_tagged_pair_spectrum(
    modes: np.ndarray,
    box_length: float,
    molecule_count: int,
) -> np.ndarray:
    r"""Return the rigid-SPC/E version of the tagged AD pair spectrum.

    The SPC/E geometry is ``r_OH=1.0`` \AA{} and
    ``angle_HOH=109.47`` degrees, with charges ``(-0.8476, 0.4238, 0.4238)``
    in electron-charge units.  This helper keeps the model used for the AD
    molecular estimator explicit and avoids any trajectory-derived
    structure-factor input.
    """

    q_oxygen = -0.8476
    q_hydrogen = 0.4238
    d_oh = 1.0
    angle_hoh = math.radians(109.47)
    d_hh = math.sqrt(2.0 * d_oh * d_oh * (1.0 - math.cos(angle_hoh)))
    return rigid_molecule_tagged_pair_spectrum(
        modes,
        box_length,
        molecule_count,
        np.asarray((q_oxygen, q_hydrogen, q_hydrogen), dtype=np.float64),
        np.asarray(
            ((0.0, d_oh, d_oh), (d_oh, 0.0, d_hh), (d_oh, d_hh, 0.0)),
            dtype=np.float64,
        ),
    )


@dataclass(frozen=True)
class ADSourceSpectrumPopulation:
    """Exact base-mode and sampled source-alias weights for one AD setting."""

    base_modes: np.ndarray
    base_weights: np.ndarray
    sampled_modes: dict[int, np.ndarray]
    shell_weight_sums: dict[int, float]
    homogeneous_chi2: float
    captured_homogeneous_chi2: float
    unresolved_homogeneous_chi2: float
    base_mode_count: int
    alias_shell: int
    samples_per_shell: int
    zeroed_active_mode_count: int


def _source_weight_arrays(
    *,
    mesh: int,
    order: int,
    box_length: float,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: ikref.PSWFCoefficients,
) -> tuple[
    adref.ADOperator,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return the exact cell-moment source decomposition ingredients.

    ``base_weight`` includes all terms whose physical source mode is the
    resolved mode itself.  ``alias_prefactor`` multiplies
    ``|W_source(k+G)|^2`` for every nonzero source alias.  Their sum over all
    aliases is the homogeneous cell-moment ``chi2``.
    """

    operator = adref.build_ad_operator(
        mesh, box_length, order, rcut, csplit, cspread, coeff
    )
    moments = adref.cell_response_moments_1d(mesh, box_length, order, coeff.real)
    indices, k1, kx, ky, kz, _ = ikref.reciprocal_arrays(mesh, box_length)
    h = box_length / mesh
    d1 = adref.lammps_deconvolution_1d(k1, h, order, cspread, coeff.spread)
    d3 = d1[:, None, None] * d1[None, :, None] * d1[None, None, :]
    active = operator.kernel != 0.0
    zeroed = active & (d3 == 0.0)
    retained = active & ~zeroed
    inv_d2 = np.zeros_like(d3)
    inv_d2[retained] = 1.0 / (d3[retained] * d3[retained])
    inv_d4 = inv_d2 * inv_d2

    mean_source = (
        moments.mean_source[:, None, None]
        * moments.mean_source[None, :, None]
        * moments.mean_source[None, None, :]
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
    gather_zero_norm2 = (
        np.abs(mean_bx) ** 2 + np.abs(mean_by) ** 2 + np.abs(mean_bz) ** 2
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
    mesh0_x = mean_bx * mean_source * inv_d2
    mesh0_y = mean_by * mean_source * inv_d2
    mesh0_z = mean_bz * mean_source * inv_d2
    zero_mismatch = (
        np.abs(mesh0_x + 1j * kx) ** 2
        + np.abs(mesh0_y + 1j * ky) ** 2
        + np.abs(mesh0_z + 1j * kz) ** 2
    )
    scale = operator.kernel * operator.kernel / box_length**6
    # The l=0 source contains the reference mismatch and all nonzero gather
    # aliases.  The latter coefficient is nonnegative by Parseval; clip only
    # roundoff-scale negative values.
    base_weight = scale * (
        zero_mismatch
        + np.maximum(second_b - gather_zero_norm2, 0.0)
        * np.abs(mean_source) ** 2
        * inv_d4
    )
    # If LAMMPS zeroes a Fourier-polynomial denominator, the entire exact
    # resolved source contribution is omitted.  It must remain in the
    # source-spectrum decomposition even though the present LiTFSI cases do
    # not encounter this branch.
    base_weight[zeroed] = scale[zeroed] * (
        kx[zeroed] * kx[zeroed]
        + ky[zeroed] * ky[zeroed]
        + kz[zeroed] * kz[zeroed]
    )
    alias_prefactor = scale * second_b * inv_d4
    return (
        operator,
        indices,
        k1,
        retained,
        base_weight,
        alias_prefactor,
        zeroed,
        np.asarray(coeff.real, dtype=np.float64),
        np.asarray((h,), dtype=np.float64),
    )


def prepare_ad_source_spectrum_population(
    *,
    q: np.ndarray,
    mesh: int,
    order: int,
    box_length: float,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: ikref.PSWFCoefficients,
    max_shell: int = 4,
    samples_per_shell: int = 8192,
    seed: int = 20260823,
    homogeneous_chi2_override: float | None = None,
) -> ADSourceSpectrumPopulation:
    """Prepare a finite-shell diagonal-``S_q`` correction for one AD setting.

    The all-alias homogeneous baseline comes from the exact cell-moment
    estimator.  Shell samples are importance sampled according to their
    positive homogeneous contribution, so the finite-shell correction is an
    unbiased estimate under the stated diagonal spectral closure.
    """

    if max_shell < 0:
        raise ValueError("max_shell must be nonnegative")
    if samples_per_shell < 64:
        raise ValueError("samples_per_shell must be at least 64")
    (
        operator,
        indices,
        k1,
        retained,
        base_weight,
        alias_prefactor,
        zeroed,
        real_coeff,
        h_array,
    ) = _source_weight_arrays(
        mesh=mesh,
        order=order,
        box_length=box_length,
        rcut=rcut,
        csplit=csplit,
        cspread=cspread,
        coeff=coeff,
    )
    h = float(h_array[0])
    active_or_zeroed = retained | zeroed
    array_indices = np.argwhere(active_or_zeroed)
    base_modes_unaggregated = np.column_stack(
        [indices[array_indices[:, dim]] for dim in range(3)]
    )
    base_modes, base_weights = _aggregate_modes(
        base_modes_unaggregated, base_weight[active_or_zeroed]
    )

    # The exact full all-alias cell-moment value is deliberately retained as
    # the S_q=1 reference rather than approximated by a finite source shell.
    # A caller that has already audited this operator-specific number may pass
    # it back explicitly; this avoids recomputing the deterministic 32-point
    # cell quadrature during a coordinate-only structure-factor sweep.
    if homogeneous_chi2_override is None:
        _, exact_meta = adref.fixed_ad_pair_estimate_cell_moments(
            np.asarray(q, dtype=np.float64),
            box_length,
            mesh,
            order,
            rcut,
            csplit,
            cspread,
            coeff,
            quadrature_order_per_half=32,
        )
        homogeneous_chi2 = float(exact_meta["chi2_pair"])
    else:
        homogeneous_chi2 = float(homogeneous_chi2_override)
        if not math.isfinite(homogeneous_chi2) or homogeneous_chi2 <= 0.0:
            raise ValueError("homogeneous_chi2_override must be finite and positive")

    rng = np.random.default_rng(seed)
    sampled_modes: dict[int, np.ndarray] = {}
    shell_weight_sums: dict[int, float] = {}
    captured = float(np.sum(base_weights))
    if max_shell:
        aliases = np.arange(-max_shell, max_shell + 1, dtype=np.int64)
        q_alias = k1[:, None] + (2.0 * math.pi / h) * aliases[None, :]
        transforms = adref.actual_window_transforms_1d(
            q_alias, h, order, real_coeff
        ).source
        source_norm2 = np.abs(transforms) ** 2
        retained_indices = np.argwhere(retained)
        retained_modes = np.column_stack(
            [indices[retained_indices[:, dim]] for dim in range(3)]
        )
        retained_prefactor = alias_prefactor[retained]
        for shell in range(1, max_shell + 1):
            alias_vectors = _shell_aliases(shell)
            per_alias_weights: list[np.ndarray] = []
            alias_totals = np.empty(len(alias_vectors), dtype=np.float64)
            for number, alias in enumerate(alias_vectors):
                source_weight = (
                    source_norm2[retained_indices[:, 0], int(alias[0]) + max_shell]
                    * source_norm2[retained_indices[:, 1], int(alias[1]) + max_shell]
                    * source_norm2[retained_indices[:, 2], int(alias[2]) + max_shell]
                )
                values = retained_prefactor * source_weight
                per_alias_weights.append(values)
                alias_totals[number] = float(np.sum(values))
            total = float(np.sum(alias_totals))
            if not math.isfinite(total) or total <= 0.0:
                raise FloatingPointError(f"nonpositive AD source-alias shell {shell}")
            shell_weight_sums[shell] = total
            captured += total
            selected_aliases = rng.choice(
                len(alias_vectors), size=samples_per_shell, replace=True, p=alias_totals / total
            )
            sampled = np.empty((samples_per_shell, 3), dtype=np.int64)
            for alias_index in np.unique(selected_aliases):
                chosen = np.flatnonzero(selected_aliases == alias_index)
                local = per_alias_weights[int(alias_index)]
                if len(chosen):
                    base_choice = rng.choice(
                        len(retained_modes),
                        size=len(chosen),
                        replace=True,
                        p=local / alias_totals[int(alias_index)],
                    )
                    sampled[chosen] = (
                        retained_modes[base_choice] + mesh * alias_vectors[int(alias_index)]
                    )
            sampled_modes[shell] = _canonical_modes(sampled)

    unresolved = homogeneous_chi2 - captured
    tolerance = 3.0e-11 * max(homogeneous_chi2, 1.0e-300)
    if unresolved < -tolerance:
        raise RuntimeError(
            "AD source decomposition exceeds its exact cell-moment baseline: "
            f"{captured:.16e} > {homogeneous_chi2:.16e}"
        )
    unresolved = max(unresolved, 0.0)
    return ADSourceSpectrumPopulation(
        base_modes=base_modes,
        base_weights=base_weights,
        sampled_modes=sampled_modes,
        shell_weight_sums=shell_weight_sums,
        homogeneous_chi2=homogeneous_chi2,
        captured_homogeneous_chi2=captured,
        unresolved_homogeneous_chi2=unresolved,
        base_mode_count=len(base_modes),
        alias_shell=max_shell,
        samples_per_shell=samples_per_shell,
        zeroed_active_mode_count=int(np.count_nonzero(zeroed)),
    )


def population_mode_union(
    populations: list[ADSourceSpectrumPopulation],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return one canonical mode table and lookup maps for several settings."""

    blocks: list[np.ndarray] = []
    for population in populations:
        blocks.append(population.base_modes)
        blocks.extend(population.sampled_modes.values())
    union = np.unique(np.concatenate(blocks), axis=0)
    lookup = {tuple(mode): number for number, mode in enumerate(union.tolist())}
    mappings: list[dict[str, object]] = []
    for population in populations:
        mappings.append(
            {
                "base": np.asarray(
                    [lookup[tuple(mode)] for mode in population.base_modes], dtype=np.int64
                ),
                "sampled": {
                    shell: np.asarray(
                        [lookup[tuple(mode)] for mode in modes], dtype=np.int64
                    )
                    for shell, modes in population.sampled_modes.items()
                },
            }
        )
    return union, mappings


def corrected_chi2_with_sampling(
    population: ADSourceSpectrumPopulation,
    mapping: dict[str, object],
    structure_factor: np.ndarray,
) -> tuple[float, dict[int | str, float], dict[int, float]]:
    """Apply measured ``S_q`` and return correction values and MC variances."""

    sq = np.asarray(structure_factor, dtype=np.float64)
    base_indices = np.asarray(mapping["base"], dtype=np.int64)
    base_correction = float(np.dot(population.base_weights, sq[base_indices] - 1.0))
    value = population.homogeneous_chi2 + base_correction
    shell_values: dict[int | str, float] = {"base": base_correction}
    shell_variances: dict[int, float] = {}
    sampled_mapping = mapping["sampled"]
    if not isinstance(sampled_mapping, dict):
        raise TypeError("malformed AD source-spectrum mapping")
    for shell, indices in sampled_mapping.items():
        values = sq[np.asarray(indices, dtype=np.int64)] - 1.0
        weight = population.shell_weight_sums[int(shell)]
        correction = weight * float(np.mean(values))
        value += correction
        shell_values[int(shell)] = correction
        shell_variances[int(shell)] = (
            weight * weight * float(np.var(values, ddof=1)) / len(values)
            if len(values) > 1
            else 0.0
        )
    if value < 0.0:
        raise FloatingPointError("the diagonal-S_q AD variance became negative")
    return value, shell_values, shell_variances


def corrected_chi2(
    population: ADSourceSpectrumPopulation,
    mapping: dict[str, object],
    structure_factor: np.ndarray,
) -> tuple[float, dict[int | str, float]]:
    value, shell_values, _ = corrected_chi2_with_sampling(
        population, mapping, structure_factor
    )
    return value, shell_values
