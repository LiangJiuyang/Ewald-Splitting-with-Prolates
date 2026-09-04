#!/usr/bin/env python3
r"""Legacy rigid S_tag AD diagnostic.

The current Figure-5 AD curve is generated exclusively by
build_fig5_ad_coordinate_screen.py, which evaluates the
configuration-conditioned full-source quadratic form. This module retains the
older trajectory-free rigid-SPC/E, pair-only diagonal-spectrum closure for
diagnostic comparisons. It omits source/target self correlations, target-cell
phases, physical-mode coherences, and alias/self cross terms, so it cannot be
used for Figure-5 selection or plotting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from generated_output import section_output_root  # noqa: E402

OUTPUT_ROOT = section_output_root()
WATER_ROOT = PROJECT / "numerical_examples" / "water_trajectory_benchmark"
WATER_DATA = WATER_ROOT / "water.data"
# This deliberately inexpensive full-force calculation defines only the
# relative-error normalization; it is not an Ewald reference force.
PILOT_FORCE_DUMP = WATER_ROOT / "forces.pppm_mesh20.dump"
SCAN_SUMMARY = OUTPUT_ROOT / "fig5_ik_ad_order_scan" / "fig5_ik_ad_order_scan_summary.csv"

PREDICTION_CSV = OUTPUT_ROOT / "legacy_fig5_ad_rigid_sq_prediction.csv"
VALIDATION_CSV = OUTPUT_ROOT / "legacy_fig5_ad_rigid_sq_source.csv"
SELF_CSV = OUTPUT_ROOT / "legacy_fig5_ad_rigid_sq_self_probe.csv"
SHELL_CSV = OUTPUT_ROOT / "legacy_fig5_ad_rigid_sq_alias_shell.csv"
MANIFEST = OUTPUT_ROOT / "legacy_fig5_ad_rigid_sq_manifest.json"
SELF_WORK = OUTPUT_ROOT / "legacy_fig5_ad_rigid_sq_self_probes"

sys.path.insert(0, str(HERE / "lammps_ad_total_validation"))
import ad_sq_descriptor as adsq  # noqa: E402
import fixed_ik_reference as ikref  # noqa: E402
import ad_validation_common as adcommon  # noqa: E402
from ad_validation_common import (  # noqa: E402
    ADCase,
    RCUT,
    coefficients,
    fit_self_correction,
    residual_self_cell_rms,
)


PILOT_N = 25
TOTAL_N = 51
ORDERS = tuple(range(5, 10))
SELF_AUDIT_MAX = 5.0e-10
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(PROJECT))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def target_tag(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-")


def parameter_tag(value: float) -> str:
    """Return a stable path-safe tag for a continuous PSWF parameter."""

    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_spce_topology(path: Path) -> tuple[np.ndarray, float, int]:
    """Read only charges, box, and molecule identities from a LAMMPS data file."""

    lines = path.read_text(encoding="utf-8").splitlines()
    bounds: dict[str, tuple[float, float]] = {}
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[-2:] in (
            ["xlo", "xhi"],
            ["ylo", "yhi"],
            ["zlo", "zhi"],
        ):
            bounds[fields[-2][0]] = (float(fields[0]), float(fields[1]))
    if set(bounds) != {"x", "y", "z"}:
        raise RuntimeError("SPC/E data file does not define three periodic bounds")
    lengths = [bounds[axis][1] - bounds[axis][0] for axis in "xyz"]
    if not np.allclose(lengths, lengths[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("rigid-SPC/E AD estimator currently requires a cubic cell")

    try:
        start = next(index for index, line in enumerate(lines) if line.startswith("Atoms"))
    except StopIteration as error:
        raise RuntimeError("SPC/E data file lacks an Atoms section") from error
    molecule_charges: dict[int, list[float]] = {}
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0].isalpha():
            break
        fields = stripped.split()
        # atom_style full: id molecule-ID type q x y z [image flags]
        if len(fields) < 7:
            raise RuntimeError(f"malformed SPC/E atom record: {line}")
        molecule_charges.setdefault(int(fields[1]), []).append(float(fields[3]))
    if not molecule_charges:
        raise RuntimeError("SPC/E data file has no atom charges")
    expected = np.sort(np.asarray((-0.8476, 0.4238, 0.4238), dtype=np.float64))
    for molecule, charges in molecule_charges.items():
        if len(charges) != 3 or not np.allclose(np.sort(charges), expected, atol=1.0e-12):
            raise RuntimeError(f"molecule {molecule} is not rigid SPC/E topology")
    charges = np.asarray(
        [charge for values in molecule_charges.values() for charge in values], dtype=np.float64
    )
    if not math.isclose(float(np.sum(charges)), 0.0, abs_tol=1.0e-12):
        raise RuntimeError("SPC/E input is not charge neutral")
    return charges, float(lengths[0]), len(molecule_charges)


def parse_force_prefix(path: Path, count: int) -> list[np.ndarray]:
    """Read the first ``count`` force frames without opening later records."""

    frames: list[np.ndarray] = []
    with path.open(encoding="utf-8") as handle:
        while len(frames) < count:
            if handle.readline().strip() != "ITEM: TIMESTEP":
                raise RuntimeError("malformed coarse-PPPM force dump")
            handle.readline()  # timestep
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise RuntimeError("force dump misses atom count")
            atom_count = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("force dump misses box bounds")
            for _ in range(3):
                handle.readline()
            header = handle.readline().split()[2:]
            positions = {name: index for index, name in enumerate(header)}
            if not {"id", "fx", "fy", "fz"}.issubset(positions):
                raise RuntimeError("force dump lacks id/fx/fy/fz columns")
            values = np.empty((atom_count, 3), dtype=np.float64)
            seen = np.zeros(atom_count, dtype=bool)
            for _ in range(atom_count):
                fields = handle.readline().split()
                atom_id = int(fields[positions["id"]]) - 1
                if not 0 <= atom_id < atom_count or seen[atom_id]:
                    raise RuntimeError("coarse-PPPM force dump has invalid atom IDs")
                seen[atom_id] = True
                values[atom_id] = [float(fields[positions[key]]) for key in ("fx", "fy", "fz")]
            if not np.all(seen):
                raise RuntimeError("coarse-PPPM force dump has missing atom IDs")
            frames.append(values)
    return frames


def coarse_force_scale() -> float:
    frames = parse_force_prefix(PILOT_FORCE_DUMP, PILOT_N)
    per_frame = np.asarray(
        [math.sqrt(float(np.mean(np.sum(force * force, axis=1)))) for force in frames],
        dtype=np.float64,
    )
    scale = math.sqrt(float(np.mean(per_frame * per_frame)))
    # Regression guard: this is the scale already used in the fixed-influence
    # Figure-5 prediction, computed here without reading its validation table.
    if not math.isclose(scale, 27.379457967539718, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"unexpected Figure-5 coarse PPPM force scale: {scale:.16g}")
    return scale


def candidates() -> list[tuple[Target, int, int]]:
    return [
        (target, order, mesh)
        for target in TARGETS
        for order in ORDERS
        for mesh in target.meshes
    ]


def case_for(target: Target, order: int, mesh: int) -> ADCase:
    return ADCase(
        # Include both continuous bandlimits.  Unit-charge self probes are
        # operator-specific, so reusing a cache entry from another c_spread
        # would silently invalidate the self-correction audit in a joint
        # (M, P, c_spread) sweep.
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
    """Converge the cell quadrature without rejecting a low-grid diagnostic.

    The standard 8/12 comparison is sufficient for almost all candidates.
    Some deliberately retained, below-band diagnostics converge only after a
    modest refinement because their cell response varies more sharply.  The
    returned value is always the first member of the refinement sequence for
    which the immediately preceding comparison meets the common tolerance.
    """

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


def predict_case(
    target: Target,
    order: int,
    mesh: int,
    charges: np.ndarray,
    box: float,
    molecule_count: int,
    force_scale: float,
    *,
    alias_shell: int,
    samples_per_shell: int,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Evaluate one fully a-priori rigid-SPC/E AD candidate."""

    case = case_for(target, order, mesh)
    coeff = coefficients(case)
    seed = 20260902 + 1000 * mesh + order
    population = adsq.prepare_ad_source_spectrum_population(
        q=charges,
        mesh=mesh,
        order=order,
        box_length=box,
        rcut=RCUT,
        csplit=target.csplit,
        cspread=target.cspread,
        coeff=coeff,
        max_shell=alias_shell,
        samples_per_shell=samples_per_shell,
        seed=seed,
    )
    modes, mappings = adsq.population_mode_union([population])
    rigid_s_tag = adsq.rigid_spce_tagged_pair_spectrum(modes, box, molecule_count)
    rigid_chi2, shell_values, shell_variances = adsq.corrected_chi2_with_sampling(
        population, mappings[0], rigid_s_tag
    )

    q2 = charges * charges
    pair_factor = float((np.sum(q2) ** 2 - np.sum(q2 * q2)) / len(charges))
    homogeneous_pair = ikref.COULOMB_REAL * math.sqrt(
        max(pair_factor * population.homogeneous_chi2, 0.0)
    )
    rigid_pair = ikref.COULOMB_REAL * math.sqrt(max(pair_factor * rigid_chi2, 0.0))
    sampled_chi_sem = math.sqrt(sum(shell_variances.values()))
    rigid_pair_sampling_sem = (
        ikref.COULOMB_REAL
        * math.sqrt(pair_factor)
        * sampled_chi_sem
        / (2.0 * math.sqrt(rigid_chi2))
        if rigid_chi2 > 0.0
        else 0.0
    )

    ab, probe_rows, self_audit = fit_self_correction(case, box, SELF_WORK, rerun=False)
    if int(self_audit["actual_mesh"]) != mesh:
        raise RuntimeError(f"AD unit probe rounded M={mesh} to {self_audit['actual_mesh']}")
    if float(self_audit["holdout_max_abs_component"]) > SELF_AUDIT_MAX:
        raise RuntimeError(f"AD unit-self correction audit failed: {case.case_id}")
    (
        self_converged,
        self8,
        self12,
        self_quadrature_order,
        self_n8_to_n12,
        self_last_refinement,
    ) = converged_residual_self_cell_rms(case, box, ab)
    residual_self = math.sqrt(float(np.mean(charges**4))) * self_converged
    fourier = ikref.closed_fourier_estimate(
        charges, box, RCUT, target.csplit, coeff, kmax=target.csplit / RCUT
    )
    total = math.sqrt(rigid_pair * rigid_pair + residual_self * residual_self + fourier * fourier)
    total_sampling_sem = total and rigid_pair / total * rigid_pair_sampling_sem
    relative = total / force_scale
    sigma_up = math.pi * RCUT * mesh / (target.csplit * box)

    row = {
        "method": "ESP production AD",
        "candidate_id": case.case_id,
        "target_relative_rms": target.value,
        "order": order,
        "actual_nx": mesh,
        "actual_grid_points": mesh**3,
        "sigma_up": sigma_up,
        "resolved_band": sigma_up >= 1.0,
        "epsilon_split": target.epsilon_split,
        "epsilon_spread": target.epsilon_spread,
        "csplit": target.csplit,
        "cspread": target.cspread,
        "molecule_count": molecule_count,
        "pilot_frames": PILOT_N,
        "pilot_force_scale": force_scale,
        "pilot_force_scale_source": "coarse PPPM force evaluation on frames 1--25; no Ewald reference",
        "rigid_descriptor": "analytical target-conditioned rigid-SPC/E diagonal source spectrum",
        "ad_pair_estimator": "exact cell-moment AD pair baseline reweighted by rigid S_tag",
        "alias_shell": alias_shell,
        "samples_per_alias_shell": samples_per_shell,
        "actual_seed": seed,
        "base_mode_count": population.base_mode_count,
        "captured_homogeneous_chi2_fraction": population.captured_homogeneous_chi2
        / population.homogeneous_chi2,
        "unresolved_homogeneous_chi2_fraction": population.unresolved_homogeneous_chi2
        / population.homogeneous_chi2,
        "zeroed_active_mode_count": population.zeroed_active_mode_count,
        "homogeneous_pair_chi2": population.homogeneous_chi2,
        "rigid_tagged_pair_chi2": rigid_chi2,
        "rigid_to_homogeneous_pair_chi2_ratio": rigid_chi2 / population.homogeneous_chi2,
        "homogeneous_pair_absolute_rms": homogeneous_pair,
        "rigid_tagged_pair_absolute_rms": rigid_pair,
        "rigid_tagged_pair_sampling_sem": rigid_pair_sampling_sem,
        "residual_self_absolute_rms": residual_self,
        "residual_self_cell_rms": self_converged,
        "residual_self_cell_quadrature_order_per_half": self_quadrature_order,
        "residual_self_n8_to_n12_relative": self_n8_to_n12,
        "residual_self_last_refinement_relative": self_last_refinement,
        "self_correction_sin1": float(ab[0]),
        "self_correction_sin2": float(ab[1]),
        "self_probe_holdout_max_abs_component": self_audit["holdout_max_abs_component"],
        "fourier_absolute_rms": fourier,
        "predicted_total_absolute_rms": total,
        "predicted_total_relative_rms": relative,
        "predicted_total_relative_sampling_sem": total_sampling_sem / force_scale,
        "prediction_passes_target": relative <= target.value,
        "prediction_reference_force_accessed": False,
        "prediction_molecular_coordinates_accessed": False,
        "prediction_structure_input": "rigid SPC/E charges and geometry only",
        "selection_scope": (
            "a-priori rigid-molecule diagonal-source-spectrum screen; no force calibration; "
            "not a complete molecular-AD theorem"
        ),
        "closure_limit": (
            "does not include intermolecular correlations, off-diagonal physical-mode aliases, "
            "or pair--residual-self cross terms"
        ),
    }
    shell_rows = [
        {
            "candidate_id": case.case_id,
            "target_relative_rms": target.value,
            "order": order,
            "actual_nx": mesh,
            "alias_shell": shell,
            "homogeneous_source_chi2": population.shell_weight_sums[shell],
            "rigid_tagged_correction_chi2": shell_values[shell],
            "rigid_tagged_correction_chi2_fraction": shell_values[shell]
            / population.homogeneous_chi2,
            "importance_sampling_chi2_sem": math.sqrt(shell_variances[shell]),
            "samples": samples_per_shell,
        }
        for shell in range(1, alias_shell + 1)
    ]
    self_row = {
        "candidate_id": case.case_id,
        "target_relative_rms": target.value,
        "order": order,
        "actual_nx": mesh,
        "correction_sin1": float(ab[0]),
        "correction_sin2": float(ab[1]),
        "fit_max_abs_component": self_audit["fit_max_abs_component"],
        "holdout_max_abs_component": self_audit["holdout_max_abs_component"],
        "self_cell_rms_n8_per_half": self8,
        "self_cell_rms_n12_per_half": self12,
        "self_cell_rms_converged": self_converged,
        "self_cell_quadrature_order_per_half": self_quadrature_order,
        "self_cell_n8_to_n12_relative": self_n8_to_n12,
        "self_cell_last_refinement_relative": self_last_refinement,
        "unit_charge_probe_only": True,
    }
    return row, shell_rows, self_row


def validate_after_prediction(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    """Join archived AD/Ewald holdout values only after predictions are frozen."""

    scan_rows = [row for row in read_csv(SCAN_SUMMARY) if row["method"] == "ad"]
    expected = {
        (float(row["target_relative_rms"]), int(row["order"]), int(row["actual_nx"]))
        for row in predictions
    }
    actual = {
        (float(row["target_relative_rms"]), int(row["order"]), int(row["actual_nx"])): row
        for row in scan_rows
    }
    if set(actual) != expected:
        raise RuntimeError("AD holdout archive does not match the theory candidate matrix")
    joined: list[dict[str, object]] = []
    for prediction in predictions:
        key = (
            float(prediction["target_relative_rms"]),
            int(prediction["order"]),
            int(prediction["actual_nx"]),
        )
        validation = actual[key]
        predicted = float(prediction["predicted_total_relative_rms"])
        held_out = float(validation["holdout_relative_rms"])
        joined.append(
            {
                **prediction,
                "validation_relative_rms": held_out,
                "validation_relative_rms_balanced_block5_sem": float(
                    validation["holdout_balanced_block5_sem"]
                ),
                "validation_passes_target": held_out <= float(prediction["target_relative_rms"]),
                "prediction_to_validation_ratio": predicted / held_out,
                "validation_frame_first": 26,
                "validation_frame_last": 51,
                "validation_frame_count": 26,
                "validation_operator": validation["operator"],
                "validation_reference": "pre-existing tight-Ewald total-force error",
                "validation_used_for_prediction": False,
                "validation_used_for_selection": False,
            }
        )
    return sorted(
        joined,
        key=lambda row: (
            -float(row["target_relative_rms"]),
            int(row["order"]),
            int(row["actual_nx"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-rigid-diagnostic",
        action="store_true",
        help="required acknowledgement that this pair-only closure is not the Figure 5 AD workflow",
    )
    parser.add_argument("--alias-shell", type=int, default=4)
    parser.add_argument("--samples-per-shell", type=int, default=8192)
    parser.add_argument(
        "--lmp",
        type=Path,
        default=None,
        help="ESP-LAMMPS executable (defaults to ESP_LAMMPS_BIN or the in-tree build)",
    )
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="write the a-priori table but do not open the AD/Ewald holdout archive",
    )
    args = parser.parse_args()
    if not args.legacy_rigid_diagnostic:
        parser.error(
            "pass --legacy-rigid-diagnostic to run the retained pair-only closure; "
            "use build_fig5_ad_coordinate_screen.py for Figure 5"
        )
    adcommon.configure_lmp(args.lmp)
    if args.alias_shell < 1:
        raise ValueError("--alias-shell must be at least one")
    if args.samples_per_shell < 64:
        raise ValueError("--samples-per-shell must be at least 64")
    for path in (WATER_DATA, PILOT_FORCE_DUMP):
        if not path.is_file():
            raise FileNotFoundError(path)

    started = time.time()
    charges, box, molecule_count = parse_spce_topology(WATER_DATA)
    force_scale = coarse_force_scale()
    predictions: list[dict[str, object]] = []
    shell_rows: list[dict[str, object]] = []
    self_rows: list[dict[str, object]] = []
    specs = candidates()
    for number, (target, order, mesh) in enumerate(specs, start=1):
        row, local_shells, self_row = predict_case(
            target,
            order,
            mesh,
            charges,
            box,
            molecule_count,
            force_scale,
            alias_shell=args.alias_shell,
            samples_per_shell=args.samples_per_shell,
        )
        predictions.append(row)
        shell_rows.extend(local_shells)
        self_rows.append(self_row)
        print(
            json.dumps(
                {
                    "stage": "legacy_rigid_stag_diagnostic",
                    "candidate": number,
                    "total": len(specs),
                    "target": target.value,
                    "P": order,
                    "M": mesh,
                    "relative_prediction": row["predicted_total_relative_rms"],
                    "elapsed_s": time.time() - started,
                }
            ),
            flush=True,
        )
    predictions.sort(
        key=lambda row: (
            -float(row["target_relative_rms"]),
            int(row["order"]),
            int(row["actual_nx"]),
        )
    )
    write_csv(PREDICTION_CSV, predictions)
    write_csv(SHELL_CSV, shell_rows)
    write_csv(SELF_CSV, self_rows)

    validation_rows: list[dict[str, object]] = []
    if not args.prediction_only:
        if not SCAN_SUMMARY.is_file():
            raise FileNotFoundError(SCAN_SUMMARY)
        validation_rows = validate_after_prediction(predictions)
        write_csv(VALIDATION_CSV, validation_rows)

    records = [file_record(path) for path in (WATER_DATA, PILOT_FORCE_DUMP, Path(__file__), HERE / "ad_sq_descriptor.py")]
    if validation_rows:
        records.append(file_record(SCAN_SUMMARY))
    manifest = {
        "schema_version": 1,
        "purpose": "legacy pair-only rigid-SPC/E S_tag diagnostic; not Figure 5 AD",
        "logical_order": [
            "explicit legacy-diagnostic acknowledgement",
            "rigid SPC/E topology/charges plus coarse PPPM normalization",
            "unit-charge AD self probes and pair-only rigid S_tag predictions",
            "write diagnostic table",
            "optionally join independent frames-26--51 AD/Ewald validation",
        ],
        "prediction": {
            "reference_force_accessed": False,
            "molecular_coordinates_accessed": False,
            "force_scale": force_scale,
            "force_scale_frames": list(range(1, PILOT_N + 1)),
            "force_scale_role": "relative normalization only; coarse PPPM, not Ewald",
            "pair_estimator": "exact AD cell-moment baseline plus rigid target-conditioned diagonal source-spectrum correction",
            "rigid_model": "orientationally averaged SPC/E intramolecular geometry",
            "residual_self": "unit-charge-probed two-harmonic LAMMPS correction plus converged cell quadrature",
            "total_error": "rigid pair, residual self, and Eq. (56) Fourier terms combined in quadrature",
            "closure_limit": (
                "pair-only closure: no source/target self correlations, target-cell phases, "
                "off-diagonal physical-mode aliases, or alias/self cross terms"
            ),
            "used_for_figure5_selection_or_plot": False,
            "alias_shell": args.alias_shell,
            "samples_per_shell": args.samples_per_shell,
        },
        "validation": {
            "performed": bool(validation_rows),
            "source": "pre-existing frames-26--51 AD/Ewald total-force archive" if validation_rows else "not opened",
            "used_for_prediction_or_selection": False,
        },
        "candidate_count": len(predictions),
        "molecule_count": molecule_count,
        "inputs": records,
        "outputs": [
            file_record(path)
            for path in (PREDICTION_CSV, SHELL_CSV, SELF_CSV)
            + ((VALIDATION_CSV,) if validation_rows else ())
        ],
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(PREDICTION_CSV)
    if validation_rows:
        ratios = np.asarray([float(row["prediction_to_validation_ratio"]) for row in validation_rows])
        print(
            json.dumps(
                {
                    "validation_ratio_min": float(np.min(ratios)),
                    "validation_ratio_median": float(np.median(ratios)),
                    "validation_ratio_max": float(np.max(ratios)),
                }
            )
        )


if __name__ == "__main__":
    main()
