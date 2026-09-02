#!/usr/bin/env python3
"""Generate source data for redesigned main-text Fig. 2.

No particle mesh appears in this calculation.  For each of the ten existing
random-charge configurations, the script compares the exact PSWF smooth force
on a fixed symmetric reciprocal cube against the infinite smooth force.  The
latter is obtained from a converged direct periodic Ewald sum minus the exact
compact PSWF near force, an identity that avoids an unconverged finite-cube
surrogate for the algebraic PSWF Fourier tail.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fixed_ik_reference import parse_charge_data
from fig2_fourier_reference import (
    ExactPSWFContinuation,
    direct_force_from_kernel,
    direct_periodic_coulomb_ewald_force,
    discrete_eq46_sum,
    eq46_force_error,
    eq56_closed_force_error,
    lattice_shell_counts,
    pooled_rms_jackknife_sem,
    rms_vector_error,
    symmetric_kernel_grid,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RANDOM_ROOT = PROJECT_ROOT / "numerical_examples" / "random_charges"
RCUT = 9.0
RECIPROCAL_MESH = 33
TAIL_RADIUS = 320
TAIL_RADII = (80, 120, 160, 240, 320)
QUADRATURE_ORDER = 768

# These are entries of the existing MathPSWF c table.  The software
# tolerances are recorded only as table provenance; the independent variable
# in the experiment is the physical bandlimit c_split.
SPLIT_CASES = (
    (4.0e-3, 8.0189),
    (1.0e-3, 9.5392),
    (5.0e-4, 10.29),
    (1.0e-4, 12.024),
    (5.0e-5, 12.762),
    (1.0e-5, 14.471),
    (1.0e-6, 16.894),
)

EWALD_PRODUCTION = dict(alpha=0.25, real_cutoff=36.0, reciprocal_mesh=41)
EWALD_CROSSCHECKS = (
    dict(alpha=0.20, real_cutoff=40.0, reciprocal_mesh=39),
    dict(alpha=0.25, real_cutoff=36.0, reciprocal_mesh=49),
    dict(alpha=0.30, real_cutoff=32.0, reciprocal_mesh=57),
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config_paths = sorted(RANDOM_ROOT.glob("config_*/random_charges.data"))
    if len(config_paths) != 10:
        raise RuntimeError(f"expected ten random-charge configurations, found {len(config_paths)}")
    configurations = []
    for path in config_paths:
        q, xyz, box_length = parse_charge_data(path)
        configurations.append((path.parent.name, path, q, xyz, box_length))
    natoms = len(configurations[0][2])
    box_length = configurations[0][4]
    qsum = float(np.sum(configurations[0][2] ** 2))
    if natoms != 512 or not math.isclose(box_length, 48.0) or not math.isclose(qsum, 512.0):
        raise RuntimeError("the controlled random-charge benchmark no longer matches N=Q=512, L=48")

    # A direct, alpha-independent Ewald reference is computed for every fixed
    # coordinate set.  No trajectory is regenerated.
    coulomb_forces = {}
    ewald_metadata = {}
    for name, _, q, xyz, length in configurations:
        force, metadata = direct_periodic_coulomb_ewald_force(
            q, xyz, length, **EWALD_PRODUCTION
        )
        coulomb_forces[name] = force
        ewald_metadata[name] = metadata

    # Cross-check the direct Ewald decomposition on configuration 1 by
    # changing alpha and both cutoffs.  This is the numerical residual of the
    # infinity reference, not an estimator printed by an external program.
    reference_rows = []
    first_name, _, first_q, first_xyz, first_length = configurations[0]
    production_force = coulomb_forces[first_name]
    for parameters in (EWALD_PRODUCTION,) + EWALD_CROSSCHECKS:
        force, metadata = direct_periodic_coulomb_ewald_force(
            first_q, first_xyz, first_length, **parameters
        )
        reference_rows.append(
            {
                "check": "direct_Ewald_alpha_and_cutoff_invariance",
                **parameters,
                "rms_difference_from_production": rms_vector_error(force, production_force),
                "real_screen_at_cutoff": metadata["real_screen_at_cutoff"],
                "reciprocal_screen_at_face": metadata["reciprocal_screen_at_face"],
                "operator": metadata["operator"],
            }
        )
    reference_residual = max(float(row["rms_difference_from_production"]) for row in reference_rows)

    shell_counts = lattice_shell_counts(TAIL_RADIUS)
    by_config_rows = []
    summary_rows = []
    tail_rows = []
    cached_infinite = {}
    cached_pswf = {}
    volume = box_length**3
    half_width = RECIPROCAL_MESH // 2
    kmax = 2.0 * math.pi * half_width / box_length

    for input_tolerance, csplit in SPLIT_CASES:
        pswf = ExactPSWFContinuation(csplit, QUADRATURE_ORDER)
        cached_pswf[csplit] = pswf
        kernel = symmetric_kernel_grid(pswf, RECIPROCAL_MESH, box_length, RCUT)
        config_errors = []
        for name, _, q, xyz, length in configurations:
            near = pswf.compact_near_force(q, xyz, length, RCUT)
            infinite = coulomb_forces[name] - near
            truncated = direct_force_from_kernel(q, xyz, length, kernel)
            error = rms_vector_error(infinite, truncated)
            config_errors.append(error)
            cached_infinite[(name, csplit)] = infinite
            by_config_rows.append(
                {
                    "configuration": name,
                    "c_split": csplit,
                    "split_input_tolerance_table_provenance": input_tolerance,
                    "r_c_A": RCUT,
                    "box_length_A": length,
                    "n_atoms": len(q),
                    "Q_sum_q_squared": float(np.sum(q * q)),
                    "reciprocal_mesh": RECIPROCAL_MESH,
                    "reciprocal_half_width": half_width,
                    "Kmax_inverse_A": kmax,
                    "resolved_factor_Kmax_rc_over_csplit": kmax * RCUT / csplit,
                    "measured_abs_rms_force_error_kcal_per_mol_A": error,
                    "smooth_infinite_force_rms_kcal_per_mol_A": float(
                        np.sqrt(np.mean(np.sum(infinite * infinite, axis=1)))
                    ),
                    "smooth_truncated_force_rms_kcal_per_mol_A": float(
                        np.sqrt(np.mean(np.sum(truncated * truncated, axis=1)))
                    ),
                    "finite_operator": "direct exact-PSWF Fourier sum on symmetric I_M",
                    "infinite_operator": "direct periodic Ewald sum minus exact compact PSWF near force",
                }
            )

        mode_sum = None
        mode_metadata = None
        for radius in TAIL_RADII:
            candidate_sum, candidate_metadata = discrete_eq46_sum(
                pswf,
                box_length,
                RCUT,
                RECIPROCAL_MESH,
                shell_counts,
                radius,
            )
            candidate_error = eq46_force_error(
                candidate_sum, qsum, natoms, volume
            )
            tail_rows.append(
                {
                    "c_split": csplit,
                    "tail_radius_index": radius,
                    "eq46_abs_rms_force_error_kcal_per_mol_A": candidate_error,
                    "explicit_radial_sum": candidate_metadata["explicit_radial_sum"],
                    "retained_cube_sum": candidate_metadata["retained_cube_sum"],
                    "explicit_omitted_sum": candidate_metadata["explicit_omitted_sum"],
                    "asymptotic_continuum_tail": candidate_metadata[
                        "asymptotic_continuum_tail"
                    ],
                    "tail_fraction_of_omitted_sum": candidate_metadata[
                        "tail_fraction_of_omitted_sum"
                    ],
                    "tail_lower_k_inverse_A": candidate_metadata["tail_lower_k"],
                    "tail_closure": candidate_metadata["tail_closure"],
                }
            )
            if radius == TAIL_RADIUS:
                mode_sum = candidate_sum
                mode_metadata = candidate_metadata
        assert mode_sum is not None and mode_metadata is not None
        eq46 = eq46_force_error(mode_sum, qsum, natoms, volume)
        eq56 = eq56_closed_force_error(pswf, qsum, natoms, volume, kmax, RCUT)
        config_errors_array = np.asarray(config_errors)
        pooled = float(np.sqrt(np.mean(config_errors_array**2)))
        summary_rows.append(
            {
                "c_split": csplit,
                "split_input_tolerance_table_provenance": input_tolerance,
                "r_c_A": RCUT,
                "reciprocal_mesh": RECIPROCAL_MESH,
                "reciprocal_half_width": half_width,
                "Kmax_inverse_A": kmax,
                "resolved_factor_Kmax_rc_over_csplit": kmax * RCUT / csplit,
                "n_config": len(config_errors),
                "measured_pooled_abs_rms_kcal_per_mol_A": pooled,
                "measured_config_mean_abs_rms_kcal_per_mol_A": float(
                    np.mean(config_errors_array)
                ),
                "measured_config_sample_sd_kcal_per_mol_A": float(
                    np.std(config_errors_array, ddof=1)
                ),
                "measured_config_se_kcal_per_mol_A": float(
                    np.std(config_errors_array, ddof=1) / math.sqrt(len(config_errors))
                ),
                "measured_pooled_jackknife_se_kcal_per_mol_A": (
                    pooled_rms_jackknife_sem(config_errors_array)
                ),
                "eq46_discrete_abs_rms_kcal_per_mol_A": eq46,
                "eq56_closed_abs_rms_kcal_per_mol_A": eq56,
                "measured_over_eq46": pooled / eq46,
                "eq56_over_eq46": eq56 / eq46,
                "eq46_tail_radius_index": TAIL_RADIUS,
                "eq46_tail_fraction_of_omitted_sum": mode_metadata[
                    "tail_fraction_of_omitted_sum"
                ],
                "pswf_quadrature_order": QUADRATURE_ORDER,
            }
        )

    # A representative finite-cube sequence makes explicit why M=33 is the
    # test cutoff, not the infinity reference.
    direct_convergence_rows = []
    representative_c = 12.024
    pswf = cached_pswf[representative_c]
    infinite = cached_infinite[(first_name, representative_c)]
    for mesh in (17, 25, 33, 41, 49, 65):
        kernel = symmetric_kernel_grid(pswf, mesh, first_length, RCUT)
        truncated = direct_force_from_kernel(first_q, first_xyz, first_length, kernel)
        direct_convergence_rows.append(
            {
                "configuration": first_name,
                "c_split": representative_c,
                "reciprocal_mesh": mesh,
                "reciprocal_half_width": mesh // 2,
                "Kmax_inverse_A": 2.0 * math.pi * (mesh // 2) / first_length,
                "rms_difference_from_infinite_smooth_kcal_per_mol_A": rms_vector_error(
                    truncated, infinite
                ),
                "reference": "direct periodic Ewald minus exact compact PSWF near force",
            }
        )

    write_csv(
        HERE / "fig2_fourier_truncation_by_config.csv",
        list(by_config_rows[0]),
        by_config_rows,
    )
    write_csv(
        HERE / "fig2_fourier_truncation_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    write_csv(
        HERE / "fig2_eq46_tail_convergence.csv",
        list(tail_rows[0]),
        tail_rows,
    )
    write_csv(
        HERE / "fig2_infinity_reference_checks.csv",
        list(reference_rows[0]),
        reference_rows,
    )
    write_csv(
        HERE / "fig2_direct_cube_convergence.csv",
        list(direct_convergence_rows[0]),
        direct_convergence_rows,
    )

    tail_convergence = {}
    for _, csplit in SPLIT_CASES:
        rows = [row for row in tail_rows if row["c_split"] == csplit]
        tail_convergence[str(csplit)] = abs(
            float(rows[-1]["eq46_abs_rms_force_error_kcal_per_mol_A"])
            - float(rows[-2]["eq46_abs_rms_force_error_kcal_per_mol_A"])
        ) / float(rows[-1]["eq46_abs_rms_force_error_kcal_per_mol_A"])

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Does discrete Eq. (46) predict the measured exact-PSWF Fourier-truncation "
            "error, and how accurate is closed Eq. (56)?"
        ),
        "system": {
            "source": str(RANDOM_ROOT.relative_to(PROJECT_ROOT)),
            "configurations": [item[0] for item in configurations],
            "n_config": len(configurations),
            "n_atoms": natoms,
            "box_length_A": box_length,
            "Q_sum_q_squared": qsum,
            "charges": "neutral random +/-1",
        },
        "parameters": {
            "r_c_A": RCUT,
            "c_split": [item[1] for item in SPLIT_CASES],
            "reciprocal_set": (
                f"I_M = {{m in Z^3: -{half_width} <= m_d <= {half_width}}}; "
                f"M={RECIPROCAL_MESH}"
            ),
            "Kmax_inverse_A": kmax,
            "pswf_source": str(
                (PROJECT_ROOT / "src/numerical_test/lammps_math_pswf/math_pswf.cpp")
                .relative_to(PROJECT_ROOT)
            ),
            "outside_continuation": (
                "finite-Fourier PSWF identity; Gauss-Legendre order "
                f"{QUADRATURE_ORDER}; never extended by zero"
            ),
        },
        "measured_error": {
            "definition": "sqrt(N^-1 sum_i |F_smooth,infinity-F_smooth,I_M|^2)",
            "finite_force": "direct particle-mode PSWF Fourier sum; no spreading or FFT",
            "infinite_force": (
                "mathematically equivalent direct conducting-periodic Ewald force minus "
                "the exact compact PSWF near force"
            ),
            "important_disclosure": (
                "The infinity force is not represented by a finite Fourier cube. This is "
                "an exact splitting identity, used because the nonzero PSWF tail makes a "
                "finite high-M cube converge only algebraically."
            ),
            "ewald_production": EWALD_PRODUCTION,
            "ewald_crosschecks": list(EWALD_CROSSCHECKS),
            "maximum_crosscheck_rms_difference_kcal_per_mol_A": reference_residual,
        },
        "eq46": {
            "explicit_integer_sphere_radius": TAIL_RADIUS,
            "convergence_radii": list(TAIL_RADII),
            "far_tail": (
                "leading PSWF sinc continuation integrated with the continuum density "
                "of states only beyond the explicit integer sphere"
            ),
            "relative_change_R240_to_R320": tail_convergence,
        },
        "averaging": {
            "main_measured_value": (
                "pooled RMS = sqrt(mean over configurations of per-configuration RMS^2)"
            ),
            "uncertainty_columns": (
                "sample dispersion of the ten per-configuration RMS values and a "
                "delete-one jackknife SEM matched to the pooled-RMS center"
            ),
        },
        "units": "force values are kcal mol^-1 Angstrom^-1 (LAMMPS real units)",
        "source_data": [
            "fig2_fourier_truncation_by_config.csv",
            "fig2_fourier_truncation_summary.csv",
            "fig2_eq46_tail_convergence.csv",
            "fig2_infinity_reference_checks.csv",
            "fig2_direct_cube_convergence.csv",
        ],
        "code": [
            "run_fig2_fourier_validation.py",
            "fig2_fourier_reference.py",
            "eval_pswf_profile.cpp",
            "fixed_ik_reference.py",
        ],
    }
    (HERE / "fig2_fourier_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("c_split measured_pooled Eq46 Eq56 measured/Eq46 Eq56/Eq46")
    for row in summary_rows:
        print(
            f"{row['c_split']:8g} "
            f"{row['measured_pooled_abs_rms_kcal_per_mol_A']:.8e} "
            f"{row['eq46_discrete_abs_rms_kcal_per_mol_A']:.8e} "
            f"{row['eq56_closed_abs_rms_kcal_per_mol_A']:.8e} "
            f"{row['measured_over_eq46']:.5f} "
            f"{row['eq56_over_eq46']:.5f}"
        )
    print(f"maximum direct-Ewald crosscheck residual: {reference_residual:.3e}")


if __name__ == "__main__":
    main()
