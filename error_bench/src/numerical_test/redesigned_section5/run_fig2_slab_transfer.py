#!/usr/bin/env python3
"""Build the slab-like Fourier-truncation transfer diagnostic for Figure 2.

This calculation contains no particle mesh.  It applies the exact same finite
PSWF Fourier cube and infinite-smooth-force identity as the random-charge
validation to ten pre-existing charge-separated configurations.  Equations
(46) and (55) are retained as *homogeneous* baselines; the purpose is to test
their transfer to a deliberately anisotropic system, not to refit them.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from fixed_ik_reference import PSWF_SOURCE_DIR, parse_charge_data
from fig2_fourier_reference import (
    ExactPSWFContinuation,
    direct_force_from_kernel,
    direct_periodic_coulomb_ewald_force,
    pooled_rms_jackknife_sem,
    rms_vector_error,
    symmetric_kernel_grid,
)
from run_fig2_fourier_validation import (
    EWALD_CROSSCHECKS,
    EWALD_PRODUCTION,
    QUADRATURE_ORDER,
    RCUT,
    RECIPROCAL_MESH,
    SPLIT_CASES,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
SLAB_ROOT = PROJECT_ROOT / "numerical_examples" / "inhomogeneous_charges"
RANDOM_SUMMARY = HERE / "fig2_fourier_truncation_summary.csv"
BY_CONFIG = HERE / "fig2_slab_fourier_truncation_by_config.csv"
SUMMARY = HERE / "fig2_slab_fourier_truncation_summary.csv"
REFERENCE_CHECKS = HERE / "fig2_slab_infinity_reference_checks.csv"
MANIFEST = HERE / "fig2_slab_fourier_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_homogeneous_predictions() -> dict[float, dict[str, str]]:
    with RANDOM_SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {float(row["c_split"]): row for row in rows}
    expected = {float(csplit) for _, csplit in SPLIT_CASES}
    if result.keys() != expected:
        raise ValueError("random-charge Figure-2 table and slab c_split sets differ")
    return result


def main() -> None:
    paths = sorted(SLAB_ROOT.glob("config_*/inhomogeneous_charges.data"))
    if len(paths) != 10:
        raise RuntimeError(f"expected ten slab-like configurations, found {len(paths)}")
    configurations: list[tuple[str, Path, np.ndarray, np.ndarray, float]] = []
    coordinate_records: list[dict[str, object]] = []
    for path in paths:
        q, xyz, box_length = parse_charge_data(path)
        if (
            len(q) != 512
            or not math.isclose(box_length, 48.0)
            or not math.isclose(float(np.sum(q)), 0.0, abs_tol=1.0e-12)
            or not math.isclose(float(np.sum(q * q)), 512.0)
        ):
            raise RuntimeError(f"unexpected slab benchmark metadata in {path}")
        positive_center = np.mean(xyz[q > 0.0], axis=0)
        negative_center = np.mean(xyz[q < 0.0], axis=0)
        configurations.append((path.parent.name, path, q, xyz, box_length))
        coordinate_records.append(
            {
                "configuration": path.parent.name,
                "data_path": str(path.relative_to(PROJECT_ROOT)),
                "data_sha256": sha256(path),
                "positive_charge_center_A": positive_center.tolist(),
                "negative_charge_center_A": negative_center.tolist(),
                "charge_center_separation_x_A": float(
                    negative_center[0] - positive_center[0]
                ),
                "dipole_eA": np.sum(q[:, None] * xyz, axis=0).tolist(),
            }
        )

    homogeneous = load_homogeneous_predictions()
    coulomb_forces: dict[str, np.ndarray] = {}
    ewald_metadata: dict[str, dict[str, object]] = {}
    for name, _, q, xyz, box_length in configurations:
        force, metadata = direct_periodic_coulomb_ewald_force(
            q, xyz, box_length, **EWALD_PRODUCTION
        )
        coulomb_forces[name] = force
        ewald_metadata[name] = metadata

    first_name, _, first_q, first_xyz, first_length = configurations[0]
    production_force = coulomb_forces[first_name]
    reference_rows: list[dict[str, object]] = []
    for parameters in (EWALD_PRODUCTION,) + EWALD_CROSSCHECKS:
        force, metadata = direct_periodic_coulomb_ewald_force(
            first_q, first_xyz, first_length, **parameters
        )
        reference_rows.append(
            {
                "configuration": first_name,
                "check": "direct_Ewald_alpha_and_cutoff_invariance",
                **parameters,
                "rms_difference_from_production": rms_vector_error(
                    force, production_force
                ),
                "real_screen_at_cutoff": metadata["real_screen_at_cutoff"],
                "reciprocal_screen_at_face": metadata[
                    "reciprocal_screen_at_face"
                ],
                "operator": metadata["operator"],
            }
        )

    by_config_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    half_width = RECIPROCAL_MESH // 2
    kmax = 2.0 * math.pi * half_width / configurations[0][4]
    for input_tolerance, csplit in SPLIT_CASES:
        pswf = ExactPSWFContinuation(csplit, QUADRATURE_ORDER)
        kernel = symmetric_kernel_grid(
            pswf, RECIPROCAL_MESH, configurations[0][4], RCUT
        )
        total_errors: list[float] = []
        component_errors: list[np.ndarray] = []
        for name, _, q, xyz, box_length in configurations:
            near = pswf.compact_near_force(q, xyz, box_length, RCUT)
            infinite = coulomb_forces[name] - near
            truncated = direct_force_from_kernel(q, xyz, box_length, kernel)
            difference = infinite - truncated
            component = np.sqrt(np.mean(difference * difference, axis=0))
            total = float(np.sqrt(np.sum(component * component)))
            if not math.isclose(
                total, rms_vector_error(infinite, truncated), rel_tol=1.0e-13
            ):
                raise RuntimeError("component and vector RMS definitions disagree")
            total_errors.append(total)
            component_errors.append(component)
            by_config_rows.append(
                {
                    "system": "charge-separated slab-like random charges",
                    "configuration": name,
                    "c_split": csplit,
                    "split_input_tolerance_table_provenance": input_tolerance,
                    "r_c_A": RCUT,
                    "box_length_A": box_length,
                    "n_atoms": len(q),
                    "Q_sum_q_squared": float(np.sum(q * q)),
                    "reciprocal_mesh": RECIPROCAL_MESH,
                    "Kmax_inverse_A": kmax,
                    "measured_abs_rms_force_error_kcal_per_mol_A": total,
                    "measured_x_rms_kcal_per_mol_A": float(component[0]),
                    "measured_y_rms_kcal_per_mol_A": float(component[1]),
                    "measured_z_rms_kcal_per_mol_A": float(component[2]),
                    "finite_operator": "direct exact-PSWF Fourier sum on symmetric I_M",
                    "infinite_operator": "direct periodic Ewald sum minus exact compact PSWF near force",
                    "particle_mesh_present": False,
                }
            )

        total_array = np.asarray(total_errors)
        component_array = np.asarray(component_errors)
        pooled_components = np.sqrt(np.mean(component_array * component_array, axis=0))
        pooled_total = float(np.sqrt(np.mean(total_array * total_array)))
        if not math.isclose(
            pooled_total,
            float(np.sqrt(np.sum(pooled_components * pooled_components))),
            rel_tol=1.0e-13,
        ):
            raise RuntimeError("pooled component and total RMS values disagree")
        baseline = homogeneous[float(csplit)]
        eq46 = float(baseline["eq46_discrete_abs_rms_kcal_per_mol_A"])
        eq55 = float(baseline["eq55_closed_abs_rms_kcal_per_mol_A"])
        transverse = math.sqrt(
            0.5 * (pooled_components[1] ** 2 + pooled_components[2] ** 2)
        )
        summary_rows.append(
            {
                "system": "charge-separated slab-like random charges",
                "diagnostic_role": "anisotropic transfer stress test of homogeneous Fourier-tail estimates",
                "c_split": csplit,
                "split_input_tolerance_table_provenance": input_tolerance,
                "r_c_A": RCUT,
                "reciprocal_mesh": RECIPROCAL_MESH,
                "Kmax_inverse_A": kmax,
                "n_config": len(total_errors),
                "measured_pooled_abs_rms_kcal_per_mol_A": pooled_total,
                "measured_config_mean_abs_rms_kcal_per_mol_A": float(
                    np.mean(total_array)
                ),
                "measured_config_sample_sd_kcal_per_mol_A": float(
                    np.std(total_array, ddof=1)
                ),
                "measured_config_se_kcal_per_mol_A": float(
                    np.std(total_array, ddof=1) / math.sqrt(len(total_array))
                ),
                "measured_pooled_jackknife_se_kcal_per_mol_A": (
                    pooled_rms_jackknife_sem(total_array)
                ),
                "measured_pooled_x_rms_kcal_per_mol_A": float(
                    pooled_components[0]
                ),
                "measured_pooled_y_rms_kcal_per_mol_A": float(
                    pooled_components[1]
                ),
                "measured_pooled_z_rms_kcal_per_mol_A": float(
                    pooled_components[2]
                ),
                "normal_to_transverse_rms_ratio": float(
                    pooled_components[0] / transverse
                ),
                "eq46_homogeneous_abs_rms_kcal_per_mol_A": eq46,
                "eq55_homogeneous_abs_rms_kcal_per_mol_A": eq55,
                "measured_over_eq46_homogeneous": pooled_total / eq46,
                "measured_over_eq55_homogeneous": pooled_total / eq55,
                "homogeneous_baseline_source": RANDOM_SUMMARY.name,
            }
        )

    write_csv(BY_CONFIG, by_config_rows)
    write_csv(SUMMARY, summary_rows)
    write_csv(REFERENCE_CHECKS, reference_rows)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Do the homogeneous Fourier-tail estimates transfer to a strongly "
            "charge-separated slab-like ensemble when the measured quantity is "
            "isolated direct Fourier truncation with no particle mesh?"
        ),
        "interpretation": (
            "anisotropic transfer stress test only; Eqs. (46) and (55) are not "
            "claimed to include the anisotropic charge correlations"
        ),
        "system": {
            "source": str(SLAB_ROOT.relative_to(PROJECT_ROOT)),
            "n_config": len(configurations),
            "n_atoms": 512,
            "box_length_A": 48.0,
            "Q_sum_q_squared": 512.0,
            "charges": "neutral +/-1, separated predominantly along x",
            "coordinate_records": coordinate_records,
        },
        "parameters": {
            "r_c_A": RCUT,
            "c_split": [item[1] for item in SPLIT_CASES],
            "reciprocal_mesh": RECIPROCAL_MESH,
            "Kmax_inverse_A": kmax,
            "pswf_quadrature_order": QUADRATURE_ORDER,
        },
        "measured_error": {
            "definition": "sqrt(N^-1 sum_i |F_smooth,infinity-F_smooth,I_M|^2)",
            "finite_force": "direct particle-mode PSWF Fourier sum; no spreading or FFT",
            "infinite_force": "direct conducting-periodic Ewald force minus exact compact PSWF near force",
            "component_axis": "x is normal to the charge-separated slabs",
            "ewald_production": EWALD_PRODUCTION,
            "ewald_crosschecks": list(EWALD_CROSSCHECKS),
            "maximum_crosscheck_rms_difference_kcal_per_mol_A": max(
                float(row["rms_difference_from_production"])
                for row in reference_rows
            ),
        },
        "homogeneous_baseline": {
            "source": RANDOM_SUMMARY.name,
            "source_sha256": sha256(RANDOM_SUMMARY),
            "reason": "N, Q, V, r_c, I_M, and c_split are identical",
        },
        "source_data": [BY_CONFIG.name, SUMMARY.name, REFERENCE_CHECKS.name],
        "code": [
            Path(__file__).name,
            "run_fig2_fourier_validation.py",
            "fig2_fourier_reference.py",
            "eval_pswf_profile.cpp",
        ],
        "code_records": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
            }
            for path in (
                Path(__file__),
                HERE / "run_fig2_fourier_validation.py",
                HERE / "fig2_fourier_reference.py",
                HERE / "eval_pswf_profile.cpp",
                PSWF_SOURCE_DIR / "math_pswf.cpp",
                PSWF_SOURCE_DIR / "math_pswf.h",
            )
        ],
        "runner_sha256": sha256(Path(__file__)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("c_split measured slab/Eq46 normal/transverse")
    for row in summary_rows:
        print(
            f"{float(row['c_split']):8.4f} "
            f"{float(row['measured_pooled_abs_rms_kcal_per_mol_A']):.9e} "
            f"{float(row['measured_over_eq46_homogeneous']):.5f} "
            f"{float(row['normal_to_transverse_rms_ratio']):.5f}"
        )


if __name__ == "__main__":
    main()
