#!/usr/bin/env python3
"""Run an operator-matched AD audit on the ten random-charge systems.

The benchmark isolates the particle--mesh error against the direct truncated
PSWF Fourier force on the identical active reciprocal set.  The development
LAMMPS AD self-correction is identified from independent one-charge probes and
verified on held-out cell positions; no multi-particle force error enters that
identification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import ad_operator_reference as adref
import fixed_ik_reference as ikref


HERE = Path(__file__).resolve().parent
REDESIGNED_ROOT = HERE.parent
PROJECT = HERE.parents[3]
RANDOM_ROOT = PROJECT / "numerical_examples" / "random_charges"
DEFAULT_LMP = REDESIGNED_ROOT / "pppm_symmetric_scan" / "lmp.pppm_symmetric_scan"
LMP = Path(os.environ.get("ESP_LAMMPS_BIN", DEFAULT_LMP)).expanduser().resolve()

RCUT = 9.0
CSPLIT = 12.024
CSPREAD = 12.024
SPLIT_INPUT_TOL = 1.0e-4
SPREAD_INPUT_TOL = 1.0e-4
MESH = 24
ORDERS = (4, 5, 6, 7, 8)
SHELLS = (1, 2, 4, 8, 16, 32, 64, 128)

DETAIL_CSV = HERE / "ad_operator_audit_by_config.csv"
SUMMARY_CSV = HERE / "ad_operator_audit_summary.csv"
SELF_CSV = HERE / "ad_self_correction_probe.csv"
SHELL_CSV = HERE / "ad_alias_convergence.csv"
MANIFEST = HERE / "ad_operator_audit_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pooled_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def pooled_rms_jackknife_sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    leave = np.asarray(
        [pooled_rms(np.delete(values, index)) for index in range(len(values))]
    )
    return float(
        math.sqrt(
            (len(values) - 1.0)
            / len(values)
            * np.sum((leave - leave.mean()) ** 2)
        )
    )


def coefficients(order: int) -> ikref.PSWFCoefficients:
    return ikref.load_coefficients(
        0.1 * SPLIT_INPUT_TOL,
        0.1 * SPREAD_INPUT_TOL,
        CSPLIT,
        CSPREAD,
        order,
    )


def one_charge_data(box: float, xyz: np.ndarray) -> str:
    return "\n".join(
        [
            "LAMMPS one-charge AD operator probe",
            "",
            "1 atoms",
            "1 atom types",
            "",
            f"0 {box:.17g} xlo xhi",
            f"0 {box:.17g} ylo yhi",
            f"0 {box:.17g} zlo zhi",
            "",
            "Masses",
            "",
            "1 1.0",
            "",
            "Atoms # charge",
            "",
            f"1 1 1.0 {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g}",
            "",
        ]
    )


def one_charge_input(order: int) -> str:
    return f"""units real
atom_style charge
boundary p p p
read_data probe.data
pair_style coul/esp {RCUT:.17g}
pair_coeff * *
kspace_style esp {SPLIT_INPUT_TOL:.17g} {SPREAD_INPUT_TOL:.17g}
kspace_modify order {order} mesh {MESH} {MESH} {MESH} diff ad cspread {CSPREAD:.17g}
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
dump audit all custom 1 force.dump id x y z fx fy fz
dump_modify audit sort id format line "%d %.17g %.17g %.17g %.17g %.17g %.17g"
run 0 post no
"""


def run_one_charge_lammps(order: int, xyz: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix=f"esp_ad_p{order}_") as tmp:
        work = Path(tmp)
        (work / "probe.data").write_text(one_charge_data(48.0, xyz))
        (work / "in.probe").write_text(one_charge_input(order))
        subprocess.run(
            [str(LMP), "-in", "in.probe", "-log", "log.lammps", "-screen", "none"],
            cwd=work,
            check=True,
        )
        frames = ikref.parse_force_dump(work / "force.dump")
        if len(frames) != 1 or frames[0][1].shape != (1, 3):
            raise RuntimeError("malformed one-charge LAMMPS probe")
        return frames[0][1][0]


TRAIN_FRACTIONS = np.asarray(
    [
        (0.113, 0.271, 0.389),
        (0.227, 0.413, 0.671),
        (0.349, 0.587, 0.829),
        (0.461, 0.719, 0.943),
    ],
    dtype=np.float64,
)
HOLDOUT_FRACTIONS = np.asarray(
    [
        (0.077, 0.333, 0.777),
        (0.159, 0.531, 0.887),
        (0.293, 0.647, 0.913),
    ],
    dtype=np.float64,
)


def correction_from_fraction(fraction: np.ndarray, coefficients_ab: np.ndarray) -> np.ndarray:
    theta = 2.0 * math.pi * np.asarray(fraction, dtype=np.float64)
    return coefficients_ab[0] * np.sin(theta) + coefficients_ab[1] * np.sin(2.0 * theta)


def fit_lammps_self_correction(order: int) -> tuple[np.ndarray, list[dict], float]:
    coeff = coefficients(order)
    operator = adref.build_ad_operator(
        MESH, 48.0, order, RCUT, CSPLIT, CSPREAD, coeff
    )
    h = 48.0 / MESH
    cell = MESH // 2
    matrix = []
    rhs = []
    records = []
    for split, fractions in (("fit", TRAIN_FRACTIONS), ("holdout", HOLDOUT_FRACTIONS)):
        for point_index, fraction in enumerate(fractions, start=1):
            xyz = h * (cell + fraction)
            raw = adref.fixed_ad_mesh_force(
                np.asarray([1.0]), xyz[None, :], operator, coeff.real
            )[0]
            implemented = run_one_charge_lammps(order, xyz)
            observed_correction = raw - implemented
            if split == "fit":
                for component in range(3):
                    theta = 2.0 * math.pi * fraction[component]
                    matrix.append((math.sin(theta), math.sin(2.0 * theta)))
                    rhs.append(observed_correction[component])
            records.append(
                dict(
                    order=order,
                    split=split,
                    point=point_index,
                    sx=fraction[0],
                    sy=fraction[1],
                    sz=fraction[2],
                    raw_fx=raw[0],
                    raw_fy=raw[1],
                    raw_fz=raw[2],
                    lammps_corrected_fx=implemented[0],
                    lammps_corrected_fy=implemented[1],
                    lammps_corrected_fz=implemented[2],
                    observed_correction_fx=observed_correction[0],
                    observed_correction_fy=observed_correction[1],
                    observed_correction_fz=observed_correction[2],
                )
            )
    ab, _, _, _ = np.linalg.lstsq(
        np.asarray(matrix, dtype=np.float64), np.asarray(rhs, dtype=np.float64), rcond=None
    )
    max_holdout = 0.0
    for row in records:
        fraction = np.asarray([row["sx"], row["sy"], row["sz"]])
        predicted = correction_from_fraction(fraction, ab)
        observed = np.asarray(
            [
                row["observed_correction_fx"],
                row["observed_correction_fy"],
                row["observed_correction_fz"],
            ]
        )
        residual = observed - predicted
        row.update(
            correction_sin_1=ab[0],
            correction_sin_2=ab[1],
            fitted_correction_fx=predicted[0],
            fitted_correction_fy=predicted[1],
            fitted_correction_fz=predicted[2],
            correction_fit_residual_norm=float(np.linalg.norm(residual)),
        )
        if row["split"] == "holdout":
            max_holdout = max(max_holdout, float(np.max(np.abs(residual))))
    return ab, records, max_holdout


def correction_force(q: np.ndarray, xyz: np.ndarray, ab: np.ndarray) -> np.ndarray:
    h = 48.0 / MESH
    fraction = np.mod(xyz / h, 1.0)
    return (q * q)[:, None] * correction_from_fraction(fraction, ab)


def self_floor_quadrature(
    order: int, ab: np.ndarray, order_per_half: int
) -> tuple[float, float]:
    nodes, weights = np.polynomial.legendre.leggauss(order_per_half)
    fraction_parts = []
    weight_parts = []
    for lo, hi in ((0.0, 0.5), (0.5, 1.0)):
        fraction_parts.append(0.5 * (hi - lo) * nodes + 0.5 * (hi + lo))
        weight_parts.append(0.5 * (hi - lo) * weights)
    fraction1 = np.concatenate(fraction_parts)
    weight1 = np.concatenate(weight_parts)
    sx, sy, sz = np.meshgrid(fraction1, fraction1, fraction1, indexing="ij")
    wx, wy, wz = np.meshgrid(weight1, weight1, weight1, indexing="ij")
    fractions = np.column_stack((sx.ravel(), sy.ravel(), sz.ravel()))
    weights3 = (wx * wy * wz).ravel()
    coeff = coefficients(order)
    operator = adref.build_ad_operator(
        MESH, 48.0, order, RCUT, CSPLIT, CSPREAD, coeff
    )
    raw_grid = adref.ad_self_response_cell_grid(fraction1, operator, coeff.real)
    correction_grid = np.empty_like(raw_grid)
    correction_grid[..., 0] = correction_from_fraction(fraction1, ab)[:, None, None]
    correction_grid[..., 1] = correction_from_fraction(fraction1, ab)[None, :, None]
    correction_grid[..., 2] = correction_from_fraction(fraction1, ab)[None, None, :]
    residual_grid = raw_grid - correction_grid
    raw = raw_grid.reshape((-1, 3))
    residual = residual_grid.reshape((-1, 3))
    raw_chi = math.sqrt(float(np.sum(weights3 * np.sum(raw * raw, axis=1))))
    residual_chi = math.sqrt(
        float(np.sum(weights3 * np.sum(residual * residual, axis=1)))
    )
    return raw_chi, residual_chi


def measure_one(order: int, config_index: int, path_string: str, ab_values: tuple[float, float]):
    path = Path(path_string)
    q, xyz, box = ikref.parse_charge_data(path)
    coeff = coefficients(order)
    operator = adref.build_ad_operator(
        MESH, box, order, RCUT, CSPLIT, CSPREAD, coeff
    )
    mesh_uncorrected = adref.fixed_ad_mesh_force(q, xyz, operator, coeff.real)
    fractions = np.mod(xyz / (box / MESH), 1.0)
    self_uncorrected = (q * q)[:, None] * adref.ad_self_response_at_fractions(
        fractions, operator, coeff.real
    )
    correction = correction_force(q, xyz, np.asarray(ab_values))
    mesh_implemented = mesh_uncorrected - correction
    self_residual = self_uncorrected - correction
    direct = ikref.direct_truncated_force(q, xyz, box, operator.kernel)
    pair_only = mesh_uncorrected - self_uncorrected
    return dict(
        order=order,
        config=config_index,
        n_particles=len(q),
        mesh=MESH,
        csplit=CSPLIT,
        cspread=CSPREAD,
        measured_pair_only_rms=ikref.rms_vector_error(pair_only, direct),
        measured_implemented_total_rms=ikref.rms_vector_error(mesh_implemented, direct),
        measured_uncorrected_total_rms=ikref.rms_vector_error(mesh_uncorrected, direct),
        uncorrected_self_rms=float(np.sqrt(np.mean(np.sum(self_uncorrected**2, axis=1)))),
        residual_self_rms=float(np.sqrt(np.mean(np.sum(self_residual**2, axis=1)))),
        minimum_abs_deconvolution_product=operator.minimum_abs_deconvolution_product,
        source_file=str(path),
    )


def main() -> None:
    global LMP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmp",
        type=Path,
        default=Path(os.environ.get("ESP_LAMMPS_BIN", DEFAULT_LMP)),
        help="ESP-LAMMPS executable; defaults to ESP_LAMMPS_BIN",
    )
    args = parser.parse_args()
    LMP = args.lmp.expanduser().resolve()
    started = time.time()
    if not LMP.is_file():
        raise FileNotFoundError(f"archived validation executable is missing: {LMP}")
    config_paths = [
        RANDOM_ROOT / f"config_{index:02d}" / "random_charges.data"
        for index in range(1, 11)
    ]
    missing = [str(path) for path in config_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    correction_by_order = {}
    correction_rows = []
    self_quadrature = {}
    for order in ORDERS:
        ab, rows, holdout_max = fit_lammps_self_correction(order)
        if holdout_max > 5.0e-10:
            raise RuntimeError(
                f"P={order} Python/LAMMPS self-correction holdout mismatch {holdout_max:.3e}"
            )
        raw8, residual8 = self_floor_quadrature(order, ab, 8)
        raw12, residual12 = self_floor_quadrature(order, ab, 12)
        correction_by_order[order] = ab
        correction_rows.extend(rows)
        self_quadrature[order] = dict(
            raw_chi_n8_per_half=raw8,
            residual_chi_n8_per_half=residual8,
            raw_chi_n12_per_half=raw12,
            residual_chi_n12_per_half=residual12,
            residual_n8_to_n12_relative=abs(residual12 - residual8) / residual12,
            correction_holdout_max_abs=holdout_max,
        )
        print(json.dumps(dict(stage="self", order=order, elapsed=time.time() - started)))

    tasks = []
    # Threads avoid platform semaphore limits in restricted reproducibility
    # environments; NumPy's FFT and array kernels release the GIL.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for order in ORDERS:
            for config_index, path in enumerate(config_paths, start=1):
                tasks.append(
                    pool.submit(
                        measure_one,
                        order,
                        config_index,
                        str(path),
                        tuple(float(x) for x in correction_by_order[order]),
                    )
                )
        detail_rows = []
        for future in as_completed(tasks):
            row = future.result()
            detail_rows.append(row)
            print(
                json.dumps(
                    dict(
                        stage="configuration",
                        order=row["order"],
                        config=row["config"],
                        elapsed=time.time() - started,
                    )
                )
            )
    detail_rows.sort(key=lambda row: (row["order"], row["config"]))

    summary_rows = []
    q0, _, box0 = ikref.parse_charge_data(config_paths[0])
    q4_mean = float(np.mean(q0**4))
    for order in ORDERS:
        rows = [row for row in detail_rows if row["order"] == order]
        pair = np.asarray([row["measured_pair_only_rms"] for row in rows])
        implemented = np.asarray([row["measured_implemented_total_rms"] for row in rows])
        uncorrected = np.asarray([row["measured_uncorrected_total_rms"] for row in rows])
        residual_self = np.asarray([row["residual_self_rms"] for row in rows])
        coeff = coefficients(order)
        prediction20, meta20 = adref.fixed_ad_pair_estimate_cell_moments(
            q0,
            box0,
            MESH,
            order,
            RCUT,
            CSPLIT,
            CSPREAD,
            coeff,
            quadrature_order_per_half=20,
        )
        prediction32, meta32 = adref.fixed_ad_pair_estimate_cell_moments(
            q0,
            box0,
            MESH,
            order,
            RCUT,
            CSPLIT,
            CSPREAD,
            coeff,
            quadrature_order_per_half=32,
        )
        self_floor = (
            math.sqrt(q4_mean)
            * self_quadrature[order]["residual_chi_n12_per_half"]
        )
        total_prediction = math.sqrt(prediction32**2 + self_floor**2)
        summary_rows.append(
            dict(
                order=order,
                n_config=len(rows),
                mesh=MESH,
                sigma_up=math.pi * RCUT * MESH / (CSPLIT * box0),
                csplit=CSPLIT,
                cspread=CSPREAD,
                measured_pair_pooled_rms=pooled_rms(pair),
                measured_pair_jackknife_sem=pooled_rms_jackknife_sem(pair),
                predicted_pair_cell_moment=prediction32,
                predicted_pair_quadrature20=prediction20,
                prediction_quadrature20_to32_relative=abs(prediction32 - prediction20) / prediction32,
                measured_pair_over_predicted=pooled_rms(pair) / prediction32,
                measured_implemented_total_pooled_rms=pooled_rms(implemented),
                measured_implemented_total_jackknife_sem=pooled_rms_jackknife_sem(implemented),
                predicted_residual_self_floor=self_floor,
                predicted_total_quadrature=total_prediction,
                measured_total_over_predicted=pooled_rms(implemented) / total_prediction,
                measured_uncorrected_total_pooled_rms=pooled_rms(uncorrected),
                measured_residual_self_pooled_rms=pooled_rms(residual_self),
                self_residual_cell_quadrature_n8_per_half=self_quadrature[order]["residual_chi_n8_per_half"],
                self_residual_cell_quadrature_n12_per_half=self_quadrature[order]["residual_chi_n12_per_half"],
                self_residual_n8_to_n12_relative=self_quadrature[order]["residual_n8_to_n12_relative"],
                lammps_self_correction_sin1=correction_by_order[order][0],
                lammps_self_correction_sin2=correction_by_order[order][1],
                correction_holdout_max_abs=self_quadrature[order]["correction_holdout_max_abs"],
                chi2_zero_alias_mismatch=meta32["chi2_zero_alias_mismatch"],
                chi2_all_alias_fluctuation=meta32["chi2_all_alias_fluctuation"],
                minimum_abs_deconvolution_product=meta32["minimum_abs_deconvolution_product"],
                operator=meta32["operator"],
            )
        )

    representative_order = 6
    coeff6 = coefficients(representative_order)
    exact6 = next(
        row["predicted_pair_cell_moment"]
        for row in summary_rows
        if row["order"] == representative_order
    )
    shell_rows = []
    for shell in SHELLS:
        actual_derivative, meta = adref.fixed_ad_pair_estimate_homogeneous(
            q0,
            box0,
            MESH,
            representative_order,
            RCUT,
            CSPLIT,
            CSPREAD,
            coeff6,
            alias_shell=shell,
        )
        formal_iq = adref.formal_iq_window_single_alias_estimate(
            q0,
            box0,
            MESH,
            representative_order,
            RCUT,
            CSPLIT,
            CSPREAD,
            coeff6,
            alias_shell=shell,
        )
        shell_rows.append(
            dict(
                order=representative_order,
                alias_shell=shell,
                actual_derivative_double_alias=actual_derivative,
                exact_all_alias_cell_moment=exact6,
                actual_derivative_over_exact=actual_derivative / exact6,
                formal_iqW_gather_only=formal_iq,
                formal_iqW_over_exact=formal_iq / exact6,
                chi2_zero_alias_mismatch=meta["chi2_zero_alias_mismatch"],
                chi2_nonzero_double_alias=meta["chi2_nonzero_double_alias"],
            )
        )
        print(json.dumps(dict(stage="alias_shell", shell=shell, elapsed=time.time() - started)))

    for path, rows in (
        (DETAIL_CSV, detail_rows),
        (SUMMARY_CSV, summary_rows),
        (SELF_CSV, correction_rows),
        (SHELL_CSV, shell_rows),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    help_text = subprocess.run(
        [str(LMP), "-help"], check=True, text=True, capture_output=True
    ).stdout.splitlines()
    simulator = next(
        (line.strip() for line in help_text if line.startswith("Large-scale Atomic")),
        "unknown",
    )
    git_info = next((line.strip() for line in help_text if line.startswith("Git info")), "")
    lammps_version = f"{simulator}; {git_info}".rstrip("; ")
    output_files = [DETAIL_CSV, SUMMARY_CSV, SELF_CSV, SHELL_CSV]
    MANIFEST.write_text(
        json.dumps(
            dict(
                purpose="operator-matched AD estimator and residual-self audit",
                conclusion=(
                    "The implemented classical derivative requires exact cell moments; "
                    "the formal iqW alias formula is not the implemented operator."
                ),
                operator=(
                    "development ESP AD: piecewise-polynomial spread/classical derivative gather, "
                    "Fourier-polynomial fixed deconvolution, implemented two-harmonic self correction"
                ),
                comparison="direct truncated PSWF Fourier force on identical radial active set",
                parameters=dict(
                    rcut=RCUT,
                    csplit=CSPLIT,
                    cspread=CSPREAD,
                    mesh=MESH,
                    orders=ORDERS,
                    split_input_tolerance=SPLIT_INPUT_TOL,
                    spread_input_tolerance=SPREAD_INPUT_TOL,
                ),
                statistics=(
                    "pooled RMS over ten independent random-charge configurations; "
                    "delete-one jackknife SEM for the same statistic"
                ),
                self_correction=(
                    "two Fourier harmonics identified from four independent one-charge cell points "
                    "and verified on three held-out positions; residual floor from tensor Gauss cell quadrature"
                ),
                executable="$LMP",
                executable_sha256=sha256(LMP),
                lammps_version=lammps_version,
                inputs=[
                    dict(path=str(path.relative_to(PROJECT)), sha256=sha256(path))
                    for path in config_paths
                ],
                outputs=[
                    dict(path=str(path.relative_to(PROJECT)), sha256=sha256(path))
                    for path in output_files
                ],
                source_files=[
                    str(Path(__file__).relative_to(PROJECT)),
                    str((HERE / "ad_operator_reference.py").relative_to(PROJECT)),
                ],
                python=platform.python_version(),
                numpy=np.__version__,
                elapsed_seconds=time.time() - started,
            ),
            indent=2,
        )
        + "\n"
    )
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
