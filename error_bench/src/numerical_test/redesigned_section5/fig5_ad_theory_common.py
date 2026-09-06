#!/usr/bin/env python3
r"""Shared definitions for the pure-theory Figure 5 AD workflow.

The Figure 5 driver uses these fixed candidate sets, the pilot-only coarse
force normalization, and the production residual-self cell quadrature.  The
active estimator combines the measured-``S_tag`` diagonal fluctuation in
``ad_sq_descriptor.py`` with the conditional pair/self closure in
``ad_pair_self_theory.py``.  ``ad_joint_quadratic.py`` is restricted to a
post-selection closure diagnostic.  This module deliberately contains no
force-difference estimator or selector.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
WATER_ROOT = PROJECT / "numerical_examples" / "water_trajectory_benchmark"
PILOT_FORCE_DUMP = WATER_ROOT / "forces.pppm_mesh20.dump"

sys.path.insert(0, str(HERE / "lammps_ad_total_validation"))
from ad_validation_common import ADCase, RCUT, residual_self_cell_rms  # noqa: E402


ORDERS = tuple(range(5, 10))
SELF_QUADRATURE_RELATIVE_MAX = 1.0e-7


@dataclass(frozen=True)
class Target:
    value: float
    epsilon_split: float
    epsilon_spread: float
    csplit: float
    cspread: float
    meshes: tuple[int, ...]


TARGETS = (
    Target(
        value=1.0e-4,
        epsilon_split=1.0e-4,
        epsilon_spread=1.0e-4,
        csplit=12.024,
        cspread=12.024,
        meshes=(12, 15, 16, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80),
    ),
    Target(
        value=1.0e-5,
        epsilon_split=1.0e-5,
        epsilon_spread=1.0e-5,
        csplit=14.471,
        cspread=14.471,
        meshes=(12, 16, 18, 20, 24, 32, 36, 40, 48, 64, 80),
    ),
)


def target_tag(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-")


def parameter_tag(value: float) -> str:
    """Return a stable path-safe tag for a continuous PSWF parameter."""

    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def candidates() -> list[tuple[Target, int, int]]:
    return [
        (target, order, mesh)
        for target in TARGETS
        for order in ORDERS
        for mesh in target.meshes
    ]


def case_for(target: Target, order: int, mesh: int) -> ADCase:
    return ADCase(
        case_id=(
            f"fig5_ad_{target_tag(target.value)}_p{order}_m{mesh}_"
            f"cs{parameter_tag(target.csplit)}_cw{parameter_tag(target.cspread)}"
        ),
        mesh=mesh,
        order=order,
        csplit=target.csplit,
        cspread=target.cspread,
        split_input_tolerance=target.epsilon_split,
        spread_input_tolerance=target.epsilon_spread,
        target_relative_error=target.value,
    )


def converged_residual_self_cell_rms(
    case: ADCase, box: float, correction: np.ndarray
) -> tuple[float, float, float, int, float, float]:
    """Converge the production residual-self cell quadrature."""

    self8 = residual_self_cell_rms(case, box, correction, 8)
    self12 = residual_self_cell_rms(case, box, correction, 12)
    n8_to_n12 = abs(self12 - self8) / max(self12, 1.0e-300)
    previous = self12
    previous_order = 12
    last_refinement = n8_to_n12
    if last_refinement <= SELF_QUADRATURE_RELATIVE_MAX:
        return self12, self8, self12, previous_order, n8_to_n12, last_refinement
    for order in (16, 20, 24, 32):
        current = residual_self_cell_rms(case, box, correction, order)
        last_refinement = abs(current - previous) / max(current, 1.0e-300)
        if last_refinement <= SELF_QUADRATURE_RELATIVE_MAX:
            return current, self8, self12, order, n8_to_n12, last_refinement
        previous = current
        previous_order = order
    raise RuntimeError(
        "AD residual-self cell quadrature did not converge through "
        f"n={previous_order}: {case.case_id}"
    )
