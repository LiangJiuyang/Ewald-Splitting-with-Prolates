#!/usr/bin/env python3
"""Measured-Sq utilities for the corrected fixed-influence Eq. (90)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import fixed_ik_reference as ref


def canonical_modes(modes: np.ndarray) -> np.ndarray:
    values = np.asarray(modes, dtype=np.int64).copy()
    nonzero = values != 0
    if np.any(~np.any(nonzero, axis=1)):
        raise ValueError("zero mode cannot be canonicalized")
    first = np.argmax(nonzero, axis=1)
    sign = np.sign(values[np.arange(len(values)), first])
    values *= sign[:, None]
    return values


def shell_aliases(shell: int) -> np.ndarray:
    return np.asarray(
        [
            (i, j, k)
            for i in range(-shell, shell + 1)
            for j in range(-shell, shell + 1)
            for k in range(-shell, shell + 1)
            if max(abs(i), abs(j), abs(k)) == shell
        ],
        dtype=np.int64,
    )


@dataclass
class AliasPopulation:
    exact_modes: np.ndarray
    exact_weights: np.ndarray
    exact_gather_modes: np.ndarray
    exact_gather_weights: np.ndarray
    sampled_modes: dict[int, np.ndarray]
    sampled_gather_modes: dict[int, np.ndarray]
    zero_modes: np.ndarray
    zero_weights: np.ndarray
    shell_weight_sums: dict[int, float]
    homogeneous_chi2: float
    base_mode_count: int
    zeroed_active_mode_count: int


def _aggregate_modes(modes: np.ndarray, weights: np.ndarray):
    canonical = canonical_modes(modes)
    unique, inverse = np.unique(canonical, axis=0, return_inverse=True)
    return unique, np.bincount(inverse, weights=weights, minlength=len(unique))


def prepare_alias_population(
    *,
    mesh: int,
    order: int,
    box_length: float,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: ref.PSWFCoefficients,
    max_shell: int = 4,
    outer_samples: int = 4096,
    seed: int = 20260808,
    base_deconvolution: str = "exact-real-transform",
) -> AliasPopulation:
    """Prepare exhaustive shell 1 and importance-sampled outer shells.

    Source and gather aliases are retained separately.  The former is weighted
    by S_q(k+G_l), whereas the latter is weighted by S_q(k).  For homogeneous
    charges both spectra equal one and the two terms reduce to the factor two
    required by the corrected Eq. (60).
    """

    split = ref.split_continuation(coeff, csplit)
    window = ref.window_transform_continuation(coeff, cspread, order)
    kernel = ref.split_kernel_grid(
        mesh, box_length, rcut, csplit, split, band_limited=True
    )
    indices, k1, _, _, _, k2 = ref.reciprocal_arrays(mesh, box_length)
    active = kernel != 0.0
    base_array_indices = np.argwhere(active)
    base_modes = indices[base_array_indices]
    # The advanced indexing above returns three columns through broadcasting
    # only on recent NumPy; construct explicitly for portability.
    base_modes = np.column_stack([indices[base_array_indices[:, d]] for d in range(3)])
    base_k2 = k2[active]
    base_kernel = kernel[active]
    h = box_length / mesh
    unitk = 2.0 * math.pi / box_length

    max_integer = (max_shell + 1) * mesh
    integer_axis = np.arange(-max_integer, max_integer + 1, dtype=np.int64)
    t_axis = 0.5 * order * h * unitk * np.abs(integer_axis) / cspread
    w_axis = window.value(t_axis)
    w_lookup = {int(mode): float(value) for mode, value in zip(integer_axis, w_axis)}
    base_w2 = np.ones(len(base_modes), dtype=np.float64)
    for dim in range(3):
        base_w2 *= np.asarray([w_lookup[int(value)] ** 2 for value in base_modes[:, dim]])
    if base_deconvolution == "exact-real-transform":
        denominator_w2 = base_w2.copy()
    elif base_deconvolution == "lammps-fourier-polynomial":
        denominator_w2 = np.ones(len(base_modes), dtype=np.float64)
        for dim in range(3):
            t = 0.5 * order * h * unitk * np.abs(base_modes[:, dim]) / cspread
            fitted = np.zeros_like(t)
            inside = t <= 1.0
            fitted[inside] = ref.horner_array(2.0 * t[inside] - 1.0, coeff.spread)
            denominator_w2 *= fitted * fitted
    else:
        raise ValueError(f"unknown base deconvolution: {base_deconvolution}")

    # The LAMMPS implementation sets the Green multiplier to zero if its
    # Fourier-polynomial deconvolution denominator vanishes.  Such a resolved
    # mode is not an undefined estimator contribution: its mesh response is
    # exactly zero, so its complete direct-Fourier mismatch is retained as a
    # base (l=0) residual.  It must *not* generate source or gather aliases,
    # because the zeroed multiplier suppresses the entire mesh response.
    zeroed = denominator_w2 == 0.0
    retained = ~zeroed
    full_base_modes = base_modes
    full_base_k2 = base_k2
    full_base_kernel = base_kernel
    full_base_w2 = base_w2
    base_modes = full_base_modes[retained]
    base_k2 = full_base_k2[retained]
    base_kernel = full_base_kernel[retained]
    base_w2 = full_base_w2[retained]
    denominator_w2 = denominator_w2[retained]
    # Each single source/gather image contains W_actual(k) W_actual(k+G)
    # divided by W_deconv(k)^2.  Squaring leaves the prefactor below, followed
    # by the aliased W_actual(k+G)^2 in the shell loop.
    prefactor = (
        base_k2 * base_kernel * base_kernel / box_length**6
        * base_w2 / (denominator_w2 * denominator_w2)
    )
    retained_zero_weights_raw = (
        base_k2 * base_kernel * base_kernel / box_length**6
        * (base_w2 / denominator_w2 - 1.0) ** 2
    )
    zero_mode_blocks = []
    zero_weight_blocks = []
    zero_positive = retained_zero_weights_raw > 0.0
    if np.any(zero_positive):
        zero_mode_blocks.append(base_modes[zero_positive])
        zero_weight_blocks.append(retained_zero_weights_raw[zero_positive])
    if np.any(zeroed):
        # With Green(k)=0, F_mesh(k)=0 while the direct reference retains
        # i k K(k) rho(k); therefore the squared mismatch is k^2 K(k)^2.
        zero_mode_blocks.append(full_base_modes[zeroed])
        zero_weight_blocks.append(
            full_base_k2[zeroed] * full_base_kernel[zeroed] ** 2 / box_length**6
        )
    if zero_mode_blocks:
        zero_modes, zero_weights = _aggregate_modes(
            np.concatenate(zero_mode_blocks), np.concatenate(zero_weight_blocks)
        )
    else:
        zero_modes = np.empty((0, 3), dtype=np.int64)
        zero_weights = np.empty(0, dtype=np.float64)

    rng = np.random.default_rng(seed)
    exact_modes = None
    exact_weights = None
    exact_gather_modes = None
    exact_gather_weights = None
    sampled_modes: dict[int, np.ndarray] = {}
    sampled_gather_modes: dict[int, np.ndarray] = {}
    shell_sums: dict[int, float] = {}
    for shell in range(1, max_shell + 1):
        aliases = shell_aliases(shell)
        mode_blocks = []
        base_mode_blocks = []
        weight_blocks = []
        for alias in aliases:
            full = base_modes + mesh * alias[None, :]
            wa2 = np.ones(len(full), dtype=np.float64)
            for dim in range(3):
                wa2 *= np.asarray([w_lookup[int(value)] ** 2 for value in full[:, dim]])
            weight = prefactor * wa2
            positive = np.isfinite(weight) & (weight > 0.0)
            mode_blocks.append(full[positive])
            base_mode_blocks.append(base_modes[positive])
            weight_blocks.append(weight[positive])
        modes = np.concatenate(mode_blocks)
        gather_modes = np.concatenate(base_mode_blocks)
        weights = np.concatenate(weight_blocks)
        total = float(weights.sum())
        shell_sums[shell] = total
        if shell == 1:
            exact_modes, exact_weights = _aggregate_modes(modes, weights)
            exact_gather_modes, exact_gather_weights = _aggregate_modes(gather_modes, weights)
        else:
            selected = rng.choice(len(weights), size=outer_samples, replace=True, p=weights / total)
            sampled_modes[shell] = canonical_modes(modes[selected])
            sampled_gather_modes[shell] = canonical_modes(gather_modes[selected])
    assert exact_modes is not None and exact_weights is not None
    assert exact_gather_modes is not None and exact_gather_weights is not None
    return AliasPopulation(
        exact_modes=exact_modes,
        exact_weights=exact_weights,
        exact_gather_modes=exact_gather_modes,
        exact_gather_weights=exact_gather_weights,
        sampled_modes=sampled_modes,
        sampled_gather_modes=sampled_gather_modes,
        zero_modes=zero_modes,
        zero_weights=zero_weights,
        shell_weight_sums=shell_sums,
        homogeneous_chi2=2.0 * sum(shell_sums.values()) + float(zero_weights.sum()),
        base_mode_count=len(full_base_modes),
        zeroed_active_mode_count=int(np.count_nonzero(zeroed)),
    )


def evaluate_sq_modes(q: np.ndarray, xyz: np.ndarray, box_length: float, modes: np.ndarray) -> np.ndarray:
    modes = np.asarray(modes, dtype=np.int64)
    unique_x, inv_x = np.unique(modes[:, 0], return_inverse=True)
    unique_y, inv_y = np.unique(modes[:, 1], return_inverse=True)
    unique_z, inv_z = np.unique(modes[:, 2], return_inverse=True)
    frac = np.mod(xyz, box_length) / box_length
    ex = np.exp(-2j * math.pi * np.outer(unique_x, frac[:, 0]))
    ey = np.exp(-2j * math.pi * np.outer(unique_y, frac[:, 1]))
    ez = np.exp(-2j * math.pi * np.outer(unique_z, frac[:, 2]))
    result = np.empty(len(modes), dtype=np.float64)
    qsum = float(np.sum(q * q))
    for start in range(0, len(modes), 512):
        stop = min(start + 512, len(modes))
        phase = ex[inv_x[start:stop]] * ey[inv_y[start:stop]]
        phase *= ez[inv_z[start:stop]]
        rho = phase @ q
        result[start:stop] = (rho.real * rho.real + rho.imag * rho.imag) / qsum
    return result


def population_mode_union(populations: list[AliasPopulation]):
    blocks = []
    for population in populations:
        blocks.append(population.exact_modes)
        blocks.append(population.exact_gather_modes)
        if len(population.zero_modes):
            blocks.append(population.zero_modes)
        blocks.extend(population.sampled_modes.values())
        blocks.extend(population.sampled_gather_modes.values())
    union = np.unique(np.concatenate(blocks), axis=0)
    lookup = {tuple(mode): index for index, mode in enumerate(union.tolist())}
    mappings = []
    for population in populations:
        mappings.append(
            dict(
                exact=np.asarray([lookup[tuple(mode)] for mode in population.exact_modes], dtype=np.int64),
                exact_gather=np.asarray(
                    [lookup[tuple(mode)] for mode in population.exact_gather_modes], dtype=np.int64
                ),
                zero=np.asarray(
                    [lookup[tuple(mode)] for mode in population.zero_modes], dtype=np.int64
                ),
                sampled={
                    shell: np.asarray([lookup[tuple(mode)] for mode in modes], dtype=np.int64)
                    for shell, modes in population.sampled_modes.items()
                },
                sampled_gather={
                    shell: np.asarray([lookup[tuple(mode)] for mode in modes], dtype=np.int64)
                    for shell, modes in population.sampled_gather_modes.items()
                },
            )
        )
    return union, mappings


def corrected_chi2_with_sampling(
    population: AliasPopulation, mapping: dict, sq: np.ndarray
):
    """Evaluate the finite-shell corrected chi^2 and its sampling variance.

    Shell 1 and the zero-alias deconvolution residual are summed exactly.
    For every sampled outer shell, the source and gather values originate
    from the same importance draw and therefore form a paired variate
    ``S_q(k+G_l) + S_q(k)``.  The returned shell variance is the usual
    with-replacement variance of that shell's importance-sampling mean.
    It quantifies Monte Carlo sampling only; trajectory uncertainty is kept
    separate by the calling scripts.
    """
    source = float(np.dot(population.exact_weights, sq[mapping["exact"]]))
    gather = float(
        np.dot(population.exact_gather_weights, sq[mapping["exact_gather"]])
    )
    zero = (
        float(np.dot(population.zero_weights, sq[mapping["zero"]]))
        if len(population.zero_weights)
        else 0.0
    )
    value = source + gather + zero
    shell_values = {1: value}
    shell_sampling_variances = {1: 0.0}
    for shell, indices in mapping["sampled"].items():
        paired_sq = sq[indices] + sq[mapping["sampled_gather"][shell]]
        shell_weight = population.shell_weight_sums[shell]
        contribution = shell_weight * float(np.mean(paired_sq))
        value += contribution
        shell_values[shell] = contribution
        shell_sampling_variances[shell] = (
            shell_weight * shell_weight * float(np.var(paired_sq, ddof=1)) / len(paired_sq)
            if len(paired_sq) > 1
            else 0.0
        )
    return value, shell_values, shell_sampling_variances


def corrected_chi2(population: AliasPopulation, mapping: dict, sq: np.ndarray):
    value, shell_values, _ = corrected_chi2_with_sampling(population, mapping, sq)
    return value, shell_values


def homogeneous_chi2_convergence(
    *,
    mesh: int,
    order: int,
    box_length: float,
    rcut: float,
    csplit: float,
    cspread: float,
    coeff: ref.PSWFCoefficients,
    max_shell: int,
    base_deconvolution: str = "exact-real-transform",
) -> np.ndarray:
    """Return exact homogeneous chi^2 values for cubic shells 0..max_shell.

    The alias cube factorizes into three one-dimensional sums, so this check
    does not construct the full outer-shell mode population.  The formula
    uses the same radial active set, actual window transform, and optional
    LAMMPS Fourier-polynomial deconvolution as ``prepare_alias_population``.
    """

    split = ref.split_continuation(coeff, csplit)
    window = ref.window_transform_continuation(coeff, cspread, order)
    kernel = ref.split_kernel_grid(
        mesh, box_length, rcut, csplit, split, band_limited=True
    )
    indices, _, _, _, _, k2 = ref.reciprocal_arrays(mesh, box_length)
    active = kernel != 0.0
    base_array_indices = np.argwhere(active)
    base_modes = np.column_stack(
        [indices[base_array_indices[:, dim]] for dim in range(3)]
    )
    base_k2 = k2[active]
    base_kernel = kernel[active]
    h = box_length / mesh
    unitk = 2.0 * math.pi / box_length

    max_integer = (max_shell + 1) * mesh
    integer_axis = np.arange(-max_integer, max_integer + 1, dtype=np.int64)
    t_axis = 0.5 * order * h * unitk * np.abs(integer_axis) / cspread
    w_axis = window.value(t_axis)
    w_lookup = {int(mode): float(value) for mode, value in zip(integer_axis, w_axis)}

    base_w2 = np.ones(len(base_modes), dtype=np.float64)
    for dim in range(3):
        base_w2 *= np.asarray(
            [w_lookup[int(value)] ** 2 for value in base_modes[:, dim]]
        )
    if base_deconvolution == "exact-real-transform":
        denominator_w2 = base_w2.copy()
    elif base_deconvolution == "lammps-fourier-polynomial":
        denominator_w2 = np.ones(len(base_modes), dtype=np.float64)
        for dim in range(3):
            t = 0.5 * order * h * unitk * np.abs(base_modes[:, dim]) / cspread
            fitted = np.zeros_like(t)
            inside = t <= 1.0
            fitted[inside] = ref.horner_array(2.0 * t[inside] - 1.0, coeff.spread)
            denominator_w2 *= fitted * fitted
    else:
        raise ValueError(f"unknown base deconvolution: {base_deconvolution}")

    # Mirror the LAMMPS zero-denominator convention used above: a zeroed
    # multiplier contributes the direct-mode mismatch k^2 K(k)^2 and no mesh
    # aliases.  Retained modes use the ordinary all-alias expression.
    zeroed = denominator_w2 == 0.0
    retained = ~zeroed
    zeroed_chi2 = float(
        np.sum(
            base_k2[zeroed] * base_kernel[zeroed] * base_kernel[zeroed] / box_length**6
        )
    )
    base_modes = base_modes[retained]
    base_k2 = base_k2[retained]
    base_kernel = base_kernel[retained]
    base_w2 = base_w2[retained]
    denominator_w2 = denominator_w2[retained]

    prefactor = (
        base_k2 * base_kernel * base_kernel / box_length**6
        * base_w2 / (denominator_w2 * denominator_w2)
    )
    zero = zeroed_chi2 + float(
        np.sum(
            base_k2 * base_kernel * base_kernel / box_length**6
            * (base_w2 / denominator_w2 - 1.0) ** 2
        )
    )
    base_axis = [
        np.asarray([w_lookup[int(value)] ** 2 for value in base_modes[:, dim]])
        for dim in range(3)
    ]
    alias_axis = [np.zeros(len(base_modes), dtype=np.float64) for _ in range(3)]
    values = []
    for shell in range(max_shell + 1):
        if shell > 0:
            for dim in range(3):
                coordinate = base_modes[:, dim]
                alias_axis[dim] += np.asarray(
                    [w_lookup[int(value + shell * mesh)] ** 2 for value in coordinate]
                )
                alias_axis[dim] += np.asarray(
                    [w_lookup[int(value - shell * mesh)] ** 2 for value in coordinate]
                )
        # Expand prod_d(base_d + alias_d) - prod_d(base_d) into seven
        # nonnegative terms.  Direct subtraction loses many digits for high
        # orders, where the true outer-shell sum can be 1e-15 or smaller.
        b0, b1, b2 = base_axis
        a0, a1, a2 = alias_axis
        alias_window_sum = (
            a0 * b1 * b2
            + b0 * a1 * b2
            + b0 * b1 * a2
            + a0 * a1 * b2
            + a0 * b1 * a2
            + b0 * a1 * a2
            + a0 * a1 * a2
        )
        values.append(zero + 2.0 * float(np.sum(prefactor * alias_window_sum)))
    return np.asarray(values, dtype=np.float64)
