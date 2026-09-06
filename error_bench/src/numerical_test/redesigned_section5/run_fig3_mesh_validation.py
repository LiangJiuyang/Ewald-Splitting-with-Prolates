#!/usr/bin/env python3
"""Generate operator-matched random-charge source data for main Fig. 3."""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

import fixed_ik_reference as ref
from generated_output import section_output_root


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
OUTPUT_ROOT = section_output_root(create=True)
RANDOM_ROOT = PROJECT / "numerical_examples" / "random_charges"
OUTPUT = OUTPUT_ROOT / "fig3_mesh_validation_source.csv"
BY_CONFIG = OUTPUT_ROOT / "fig3_mesh_validation_by_config.csv"
MANIFEST = OUTPUT_ROOT / "fig3_mesh_validation_manifest.json"

RCUT = 9.0
LBOX = 48.0
ALIAS_SHELL = 12
MAX_ABS_AXIAL_ALIAS_INDEX = 12


SPREAD_TABLES = {
    9.5392: 1.0e-3,
    10.29: 5.0e-4,
    12.024: 1.0e-4,
    12.762: 5.0e-5,
    13.251: 3.0e-5,
    14.471: 1.0e-5,
    16.894: 1.0e-6,
}

STRICT_CSPREAD_ORDERS = {
    9.5392: 6,
    10.29: 6,
    12.024: 8,
    12.762: 8,
    13.251: 8,
    14.471: 8,
    16.894: 8,
}


def cases():
    result = []
    # The original 10^-4 order sweep is retained.  A second, physically
    # resolved sweep uses the c_split=c_spread=16.894 bandlimits associated
    # with the 10^-6 parameter record.  On the L=48 random-charge box, M=24
    # would give sigma_up=0.837; M=30 is therefore used for the stringent
    # sweep so that the complete radial split band remains resolved.
    for order in range(4, 11):
        result.append(
            dict(panel="P", x=float(order), mesh=24, order=order,
                 csplit=12.024, split_input_tol=1.0e-4, cspread=12.024,
                 spread_input_tol=SPREAD_TABLES[12.024])
        )
    for order in range(4, 10):
        result.append(
            dict(panel="P", x=float(order), mesh=30, order=order,
                 csplit=16.894, split_input_tol=1.0e-6, cspread=16.894,
                 spread_input_tol=SPREAD_TABLES[16.894])
        )
    # This combination deliberately spans the transition from a marginally
    # separated alias band to a well-separated band; it makes the sigma_up
    # dependence visible without changing P or c_spread along the sweep.
    # Use the actual FFT grids realized in the production AD sweep.  M=20
    # gives sigma_up=0.980 and is retained only as a near-critical diagnostic;
    # requesting M=21 is rounded to M=24 and would duplicate an existing
    # point.  The previously requested M=42 point is realized as M=45.
    for order in (5, 8):
        for mesh in (20, 24, 27, 30, 32, 36, 40, 45, 48):
            result.append(
                dict(
                    panel="sigma_up",
                    x=math.pi * RCUT * mesh / (12.024 * LBOX),
                    mesh=mesh,
                    order=order,
                    csplit=12.024,
                    split_input_tol=1.0e-4,
                    cspread=13.251,
                    spread_input_tol=SPREAD_TABLES[13.251],
                )
            )
    # Compare the spreading-bandlimit dependence for the 10^-4 and 10^-6
    # split records.  The baseline curve fixes P=6.  The stringent curve pairs
    # P=6 with the two smallest spreading bandlimits and P=8 with the remaining
    # five values so that every continuous denominator stays in its zero-free
    # interval.  M=30 keeps sigma_up >= 1 on the L=48 random-charge box.
    for cspread, tolerance in SPREAD_TABLES.items():
        result.append(
            dict(
                panel="c_spread",
                x=cspread,
                mesh=24,
                order=6,
                csplit=12.024,
                split_input_tol=1.0e-4,
                cspread=cspread,
                spread_input_tol=tolerance,
            )
        )
        result.append(
            dict(
                panel="c_spread",
                x=cspread,
                mesh=30,
                order=STRICT_CSPREAD_ORDERS[cspread],
                csplit=16.894,
                split_input_tol=1.0e-6,
                cspread=cspread,
                spread_input_tol=tolerance,
            )
        )
    return result


def pooled_rms_jackknife_sem(values: np.ndarray) -> float:
    """Delete-one jackknife SEM for sqrt(mean(values**2))."""
    if len(values) < 3:
        return float("nan")
    leave_one_out = np.asarray(
        [math.sqrt(float(np.mean(np.delete(values, index) ** 2))) for index in range(len(values))]
    )
    return float(
        math.sqrt(
            (len(values) - 1.0) / len(values)
            * np.sum((leave_one_out - leave_one_out.mean()) ** 2)
        )
    )


def main():
    started = time.time()
    config_paths = [
        RANDOM_ROOT / f"config_{index:02d}" / "random_charges.data" for index in range(1, 11)
    ]
    missing = [str(path) for path in config_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    systems = [ref.parse_charge_data(path) for path in config_paths]
    if any(len(q) != 512 or not math.isclose(box, LBOX) for q, _, box in systems):
        raise RuntimeError("random-charge benchmark no longer matches N=512, L=48")

    direct_cache: dict[tuple[int, int, float], np.ndarray] = {}
    split_continuations = {}
    for csplit, split_tol in sorted(
        {(item["csplit"], item["split_input_tol"]) for item in cases()}
    ):
        split_coeff = ref.load_coefficients(
            0.1 * split_tol,
            0.1 * split_tol,
            csplit,
            csplit,
            6,
        )
        split_continuations[csplit] = ref.split_continuation(split_coeff, csplit)
    for config_index, (q, xyz, box) in enumerate(systems, start=1):
        direct_keys = sorted({(item["mesh"], item["csplit"]) for item in cases()})
        for mesh, csplit in direct_keys:
            kernel = ref.split_kernel_grid(
                mesh,
                box,
                RCUT,
                csplit,
                split_continuations[csplit],
                band_limited=True,
            )
            direct_cache[(config_index, mesh, csplit)] = ref.direct_truncated_force(
                q, xyz, box, kernel
            )
        print(json.dumps(dict(stage="direct", config=config_index, elapsed=time.time() - started)))

    detail_rows = []
    summary_rows = []
    for case_index, case in enumerate(cases(), start=1):
        coeff = ref.load_coefficients(
            0.1 * case["split_input_tol"],
            0.1 * case["spread_input_tol"],
            case["csplit"],
            case["cspread"],
            case["order"],
        )
        measured = []
        stability = []
        for config_index, (q, xyz, box) in enumerate(systems, start=1):
            mesh_force, metadata = ref.fixed_ik_mesh_force(
                q, xyz, box, case["mesh"], case["order"], RCUT,
                case["csplit"], case["cspread"], coeff,
            )
            error = ref.rms_vector_error(
                mesh_force,
                direct_cache[(config_index, case["mesh"], case["csplit"])],
            )
            measured.append(error)
            stability.append(metadata["minimum_abs_dimensionless_window_product"])
            detail_rows.append(
                dict(
                    panel=case["panel"], x=case["x"], config=config_index,
                    absolute_rms_mesh_error=error, mesh_actual=case["mesh"],
                    sigma_up=(
                        math.pi * RCUT * case["mesh"]
                        / (case["csplit"] * LBOX)
                    ),
                    order=case["order"], csplit=case["csplit"],
                    cspread=case["cspread"],
                    split_input_tolerance=case["split_input_tol"],
                    spread_input_tolerance=case["spread_input_tol"],
                    operator=(
                        "fixed-influence ik; matched piecewise-polynomial "
                        "window transform"
                    ),
                    direct_reference="direct truncated PSWF Fourier sum on identical I_M",
                )
            )
        eq60, eq60_meta = ref.fixed_alias_estimate_homogeneous(
            systems[0][0], LBOX, case["mesh"], case["order"], RCUT,
            case["csplit"], case["cspread"], coeff, alias_shell=ALIAS_SHELL,
        )
        axial, axial_meta = ref.fixed_alias_estimate_discrete_axial(
            systems[0][0], LBOX, case["mesh"], case["order"], RCUT,
            case["csplit"], case["cspread"], coeff,
            max_abs_axial_alias_index=MAX_ABS_AXIAL_ALIAS_INDEX,
        )
        sigma = math.pi * RCUT * case["mesh"] / (case["csplit"] * LBOX)
        # The continuum reduction is plotted as a convention-defined proxy.
        # Values below sigma_up=1 are explicitly extrapolated.  If the
        # continuous denominator crosses zero, the plotted/source value is set
        # to zero while the status fields preserve that this is a plotting
        # convention rather than a converged improper integral.
        continuum_eq67_zeroed = False
        continuum_eq67_extrapolated = sigma < 1.0
        try:
            continuum_eq67 = ref.fixed_alias_estimate_one_dimensional(
                systems[0][0], LBOX, case["order"], RCUT, case["csplit"],
                case["cspread"], sigma, coeff,
                max_abs_axial_alias_index=MAX_ABS_AXIAL_ALIAS_INDEX,
                allow_extrapolation=continuum_eq67_extrapolated,
            )
            continuum_eq67_status = (
                "extrapolated below resolved-band domain (sigma_up < 1)"
                if continuum_eq67_extrapolated
                else "valid zero-free continuous denominator"
            )
        except ValueError as error:
            if "integral is divergent" not in str(error):
                raise
            continuum_eq67 = 0.0
            continuum_eq67_zeroed = True
            continuum_eq67_status = (
                "zeroed by plotting convention after continuous "
                f"base-window denominator crossing: {error}"
            )
        values = np.asarray(measured)
        measured_pooled = float(np.sqrt(np.mean(values * values)))
        guard_lo = math.pi * case["order"] / (2.0 * sigma)
        guard_hi = 0.5 * math.pi * case["order"] * (2.0 - 1.0 / sigma)
        summary_rows.append(
            dict(
                panel=case["panel"], x=case["x"], n_config=len(values),
                measured_mean=float(values.mean()), measured_sd=float(values.std(ddof=1)),
                measured_sem=float(values.std(ddof=1) / math.sqrt(len(values))),
                measured_pooled_rms=measured_pooled,
                measured_pooled_jackknife_sem=pooled_rms_jackknife_sem(values),
                eq60_discrete=eq60,
                axial_discrete=axial,
                axial_max_abs_alias_index=MAX_ABS_AXIAL_ALIAS_INDEX,
                axial_total_face_images=6 * MAX_ABS_AXIAL_ALIAS_INDEX,
                axial_last_layer_relative_change=(
                    axial_meta["last_axial_layer_relative_change"]
                ),
                axial_over_eq60=axial / eq60,
                continuum_eq67=continuum_eq67,
                continuum_eq67_status=continuum_eq67_status,
                continuum_eq67_zeroed=continuum_eq67_zeroed,
                continuum_eq67_extrapolated=continuum_eq67_extrapolated,
                measured_over_eq60=measured_pooled / eq60,
                measured_over_axial=measured_pooled / axial,
                mesh_actual=case["mesh"], sigma_up=sigma, order=case["order"],
                csplit=case["csplit"], cspread=case["cspread"],
                guard_band_lower=guard_lo, guard_band_upper=guard_hi,
                in_guard_band=guard_lo <= case["cspread"] <= guard_hi,
                minimum_abs_dimensionless_window_product=min(stability),
                alias_shell=ALIAS_SHELL,
                eq60_last_shell_relative_change=eq60_meta["last_shell_relative_change"],
                split_input_tolerance=case["split_input_tol"],
                spread_input_tolerance=case["spread_input_tol"],
                uncertainty=(
                    "center is pooled RMS over ten independent configurations; "
                    "delete-one jackknife SEM estimates uncertainty of that same statistic"
                ),
                operator=(
                    "fixed-influence ik; matched piecewise-polynomial "
                    "window transform"
                ),
            )
        )
        print(json.dumps(dict(stage="case", case=case_index, **case, elapsed=time.time() - started)))

    with BY_CONFIG.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader(); writer.writerows(detail_rows)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    MANIFEST.write_text(
        json.dumps(
            dict(
                path_basis=(
                    "bundle_root for distributed inputs; "
                    "$ESP_ERROR_BENCH_OUTPUT_DIR/redesigned_section5 for outputs"
                ),
                purpose="Main Figure 3 operator-matched pure mesh validation",
                source_configurations=[
                    str(path.resolve().relative_to(PROJECT)) for path in config_paths
                ],
                reciprocal_set="numpy/LAMMPS canonical FFT cube: even M includes -M/2 and excludes +M/2",
                operator=(
                    "fixed-influence ik with matched piecewise-polynomial "
                    "window transform"
                ),
                comparison="direct truncated PSWF Fourier force on identical I_M",
                figure_3c_cspread_sweeps={
                    "baseline": {
                        "mesh": 24,
                        "order": 6,
                        "csplit": 12.024,
                        "split_input_tolerance": 1.0e-4,
                    },
                    "strict": {
                        "mesh": 30,
                        "orders_by_cspread": STRICT_CSPREAD_ORDERS,
                        "csplit": 16.894,
                        "split_input_tolerance": 1.0e-6,
                        "basis": (
                            "P is paired with cspread to keep the continuous "
                            "base-window denominator zero-free"
                        ),
                    },
                },
                equation_60_alias_shell=ALIAS_SHELL,
                equation_60_stable_positive_expansion=True,
                discrete_axial_alias_model=(
                    "axial face-image pairs ell=+-1,...,+-K along each "
                    "coordinate direction; mixed-axis edge/corner images omitted"
                ),
                discrete_axial_max_abs_alias_index=MAX_ABS_AXIAL_ALIAS_INDEX,
                discrete_axial_total_face_images=6 * MAX_ABS_AXIAL_ALIAS_INDEX,
                discrete_axial_role=(
                    "reduced estimator plotted in Figure 3; evaluated on the "
                    "same canonical finite FFT grid as the all-alias estimator"
                ),
                continuum_equation_67_role=(
                    "continuum axial proxy plotted in Figure 3; never used for "
                    "final parameter screening"
                ),
                continuum_equation_67_outer_quadrature_order=320,
                continuum_equation_67_zero_policy=(
                    "store zero by explicit plotting convention if the "
                    "continuous base-window denominator crosses zero; retain "
                    "a status flag because this is not a converged improper integral"
                ),
                window_transform_quadrature_points_per_piece=(
                    ref.WINDOW_FOURIER_QUADRATURE_POINTS_PER_PIECE
                ),
                continuum_equation_67_base_window_scan_points=(
                    ref.CONTINUOUS_BASE_WINDOW_SCAN_POINTS
                ),
                continuum_equation_67_domain=(
                    "native derivation uses sigma_up >= 1 and a zero-free "
                    "continuous deconvolution denominator on 0 <= y <= 1; "
                    "Figure 3 also displays explicitly flagged extrapolations"
                ),
                M20_boundary_display=(
                    "M=20, sigma_up=0.980 is retained as a discrete boundary "
                    "test; the continuum proxy is explicitly extrapolated"
                ),
                coulomb_units="LAMMPS real, qqrd2e=332.06371",
                python=platform.python_version(), numpy=np.__version__,
                elapsed_seconds=time.time() - started,
            ),
            indent=2,
        )
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
