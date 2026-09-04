#!/usr/bin/env python3
"""Shared utilities for production-LAMMPS ESP analytical-differentiation tests.

The routines in this module deliberately keep three objects separate:

1. the uncorrected reciprocal AD operator reproduced by NumPy;
2. the two-harmonic one-body correction used by development LAMMPS; and
3. the residual self response left after that correction.

The LAMMPS correction coefficients are identified only from unit-charge
operator probes.  No random-system or water reference force enters that
identification.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REDESIGNED = HERE.parent
PROJECT = HERE.parents[3]
if str(REDESIGNED) not in sys.path:
    sys.path.insert(0, str(REDESIGNED))

import fixed_ad_reference as adref  # noqa: E402
import fixed_ik_reference as ikref  # noqa: E402


DEFAULT_LMP = REDESIGNED / "pppm_symmetric_scan/lmp.pppm_symmetric_scan"
LMP = Path(os.environ.get("ESP_LAMMPS_BIN", DEFAULT_LMP)).expanduser().resolve()
RCUT = 9.0
COULOMB_REAL = ikref.COULOMB_REAL


def configure_lmp(value: str | Path | None = None) -> Path:
    """Configure the ESP-LAMMPS executable for an AD calculation.

    Entry-point scripts may pass ``--lmp`` explicitly; otherwise the shared
    ``ESP_LAMMPS_BIN`` environment variable is honored.  The fallback retains
    the historical in-tree location for existing local workflows.
    """

    global LMP
    candidate = (
        value
        if value is not None
        else os.environ.get("ESP_LAMMPS_BIN", DEFAULT_LMP)
    )
    LMP = Path(candidate).expanduser().resolve()
    return LMP


def project_relative(path: Path) -> str:
    """Return a portable LAMMPS path for in-tree inputs and external outputs.

    Reproducibility inputs live below ``PROJECT`` and remain relative to the
    bundle working directory.  Generated probes and force dumps are directed
    outside the Git worktree, so they cannot be relativized to that directory;
    in that case LAMMPS receives their absolute path.
    """

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT).as_posix()
    except ValueError:
        return resolved.as_posix()


def reference_operator_dependencies() -> list[Path]:
    """Return every local code/binary dependency of the NumPy reference path."""
    numerical_root = REDESIGNED.parent
    return [
        REDESIGNED / "fixed_ad_reference.py",
        REDESIGNED / "ad_operator_audit/ad_operator_reference.py",
        REDESIGNED / "fixed_ik_reference.py",
        REDESIGNED / "dump_pswf_coeff",
        numerical_root / "dump_pswf_coeff.cpp",
        numerical_root / "lammps_math_pswf/math_pswf.cpp",
        numerical_root / "lammps_math_pswf/math_pswf.h",
        numerical_root / "lammps_math_pswf/math_const.h",
    ]


@dataclass(frozen=True)
class ADCase:
    case_id: str
    mesh: int
    order: int
    csplit: float
    cspread: float
    split_input_tolerance: float
    spread_input_tolerance: float
    panel: str = ""
    x: float = math.nan
    target_relative_error: float = math.nan

    @property
    def tuple_key(self) -> tuple[int, int, float, float, float, float]:
        return (
            self.mesh,
            self.order,
            self.csplit,
            self.cspread,
            self.split_input_tolerance,
            self.spread_input_tolerance,
        )


SPREAD_TABLES = {
    9.5392: 1.0e-3,
    10.29: 5.0e-4,
    12.024: 1.0e-4,
    12.762: 5.0e-5,
    13.251: 3.0e-5,
    14.471: 1.0e-5,
    16.894: 1.0e-6,
}


def figure3_cases() -> list[ADCase]:
    cases: list[ADCase] = []
    # Original 10^-4 bandlimit setting on the 24^3 random-charge grid.
    for order in range(4, 11):
        cases.append(
            ADCase(
                case_id=f"fig3_P_P{order}_M24_c12p024",
                panel="P",
                x=float(order),
                mesh=24,
                order=order,
                csplit=12.024,
                cspread=12.024,
                split_input_tolerance=1.0e-4,
                spread_input_tolerance=1.0e-4,
            )
        )
    # Stringent 10^-6 bandlimits.  M=30 is the smallest production-friendly
    # grid used here that resolves c_split=16.894 on the L=48 box
    # (sigma_up=1.046); retaining M=24 would leave the split band unresolved.
    for order in range(4, 10):
        cases.append(
            ADCase(
                case_id=f"fig3_P_P{order}_M30_c16p894",
                panel="P",
                x=float(order),
                mesh=30,
                order=order,
                csplit=16.894,
                cspread=16.894,
                split_input_tolerance=1.0e-6,
                spread_input_tolerance=1.0e-6,
            )
        )
    # M=20 gives sigma_up=0.980 and brackets the unupsampled limit as a
    # near-critical diagnostic.  Requesting M=21 is rounded to M=24 by the
    # production FFT path and therefore would not add an independent point.
    # Repeat the same grid sweep at P=5 and P=8 to expose the coupled
    # order--upsampling behavior of the AD operator.
    for order in (5, 8):
        for mesh in (20, 24, 27, 30, 32, 36, 40, 42, 48):
            cases.append(
                ADCase(
                    case_id=f"fig3_sigma_P{order}_M{mesh}_c13p251",
                    panel="sigma_up",
                    x=math.pi * RCUT * mesh / (12.024 * 48.0),
                    mesh=mesh,
                    order=order,
                    csplit=12.024,
                    cspread=13.251,
                    split_input_tolerance=1.0e-4,
                    spread_input_tolerance=3.0e-5,
                )
            )
    # Match the fixed-ik spreading-bandlimit comparison in panel (c): the
    # 10^-4 parameter record uses P=6 on M=24, whereas the resolved 10^-6
    # sensitivity slice uses P=10 on M=30.  The baseline cspread=csplit
    # member duplicates its order-sweep tuple; the strict P=10 member is a
    # separate production case because the displayed strict order sweep ends
    # at P=9.
    for csplit, split_tolerance, mesh, order in (
        (12.024, 1.0e-4, 24, 6),
        (16.894, 1.0e-6, 30, 10),
    ):
        split_tag = str(csplit).replace(".", "p")
        for cspread, tolerance in SPREAD_TABLES.items():
            spread_tag = str(cspread).replace(".", "p")
            case_id = (
                f"fig3_cspread_P6_M24_c{spread_tag}"
                if math.isclose(csplit, 12.024)
                else (
                    f"fig3_cspread_P{order}_M{mesh}_csplit{split_tag}_"
                    f"cspread{spread_tag}"
                )
            )
            cases.append(
                ADCase(
                    case_id=case_id,
                    panel="c_spread",
                    x=cspread,
                    mesh=mesh,
                    order=order,
                    csplit=csplit,
                    cspread=cspread,
                    split_input_tolerance=split_tolerance,
                    spread_input_tolerance=tolerance,
                )
            )
    return cases


def water_cases() -> list[ADCase]:
    return [
        ADCase(
            case_id="water_ad_target_1e-3",
            mesh=12,
            order=4,
            csplit=9.5392,
            cspread=9.5392,
            split_input_tolerance=1.0e-3,
            spread_input_tolerance=1.0e-3,
            target_relative_error=1.0e-3,
        ),
        ADCase(
            case_id="water_ad_target_1e-4",
            mesh=15,
            order=6,
            csplit=12.024,
            cspread=13.251,
            split_input_tolerance=1.0e-4,
            spread_input_tolerance=3.0e-5,
            target_relative_error=1.0e-4,
        ),
        ADCase(
            case_id="water_ad_target_1e-5",
            mesh=16,
            order=7,
            csplit=14.471,
            cspread=14.471,
            split_input_tolerance=1.0e-5,
            spread_input_tolerance=1.0e-5,
            target_relative_error=1.0e-5,
        ),
    ]


def coefficients(case: ADCase) -> ikref.PSWFCoefficients:
    return ikref.load_coefficients(
        0.1 * case.split_input_tolerance,
        0.1 * case.spread_input_tolerance,
        case.csplit,
        case.cspread,
        case.order,
    )


def operator(case: ADCase, box_length: float) -> adref.ADOperator:
    return adref.build_ad_operator(
        case.mesh,
        box_length,
        case.order,
        RCUT,
        case.csplit,
        case.cspread,
        coefficients(case),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    def standard_json(item):
        if isinstance(item, dict):
            return {key: standard_json(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [standard_json(child) for child in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(standard_json(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def charge_data_text(q: np.ndarray, xyz: np.ndarray, box_length: float) -> str:
    q = np.asarray(q, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape != (len(q), 3):
        raise ValueError("charge-data coordinate shape mismatch")
    rows = [
        "LAMMPS charge data for production ESP AD validation",
        "",
        f"{len(q)} atoms",
        "1 atom types",
        "",
        f"0 {box_length:.17g} xlo xhi",
        f"0 {box_length:.17g} ylo yhi",
        f"0 {box_length:.17g} zlo zhi",
        "",
        "Masses",
        "",
        "1 1.0",
        "",
        "Atoms # charge",
        "",
    ]
    for index, (charge, position) in enumerate(zip(q, xyz), start=1):
        rows.append(
            f"{index} 1 {charge:.17g} "
            f"{position[0]:.17g} {position[1]:.17g} {position[2]:.17g}"
        )
    rows.append("")
    return "\n".join(rows)


def trajectory_text(
    frames: Sequence[np.ndarray], box_length: float, timesteps: Sequence[int] | None = None
) -> str:
    if not frames:
        raise ValueError("trajectory requires at least one frame")
    natoms = len(frames[0])
    if timesteps is None:
        timesteps = list(range(len(frames)))
    if len(timesteps) != len(frames):
        raise ValueError("trajectory timestep count mismatch")
    lines: list[str] = []
    for timestep, xyz in zip(timesteps, frames):
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.shape != (natoms, 3):
            raise ValueError("trajectory frame shape mismatch")
        lines.extend(
            [
                "ITEM: TIMESTEP",
                str(int(timestep)),
                "ITEM: NUMBER OF ATOMS",
                str(natoms),
                "ITEM: BOX BOUNDS pp pp pp",
                f"0 {box_length:.17g}",
                f"0 {box_length:.17g}",
                f"0 {box_length:.17g}",
                "ITEM: ATOMS id x y z ix iy iz",
            ]
        )
        for atom_id, position in enumerate(xyz, start=1):
            wrapped = np.mod(position, box_length)
            lines.append(
                f"{atom_id} {wrapped[0]:.17g} {wrapped[1]:.17g} "
                f"{wrapped[2]:.17g} 0 0 0"
            )
    lines.append("")
    return "\n".join(lines)


def ad_kspace_input(
    case: ADCase,
    data_path: Path,
    trajectory_path: Path,
    dump_path: Path,
    exclude_pair: bool = False,
    atom_style: str = "charge",
    molecular_preamble: str = "",
    pair_dump_path: Path | None = None,
    kspace_scale_zero: bool = False,
) -> str:
    exclusion = "neigh_modify exclude group all all\n" if exclude_pair else ""
    pair_diagnostics = ""
    if pair_dump_path is not None:
        pair_diagnostics = f"""
compute adpid all property/local patom1 patom2
compute adpf all pair/local fx fy fz
dump adpair all local 1 {project_relative(pair_dump_path)} index c_adpid[1] c_adpid[2] c_adpf[1] c_adpf[2] c_adpf[3]
dump_modify adpair format line \"%d %.0f %.0f %.17g %.17g %.17g\"
"""
    scale_fix = ""
    if kspace_scale_zero:
        scale_fix = """
variable adzero equal 0.0
fix adkzero all adapt 0 kspace v_adzero scale no reset no
"""
    return f"""newton on
units real
atom_style {atom_style}
read_data {project_relative(data_path)}
reset_timestep 0

{molecular_preamble}pair_style coul/esp {RCUT:.17g}
kspace_style esp {case.split_input_tolerance:.17g} {case.spread_input_tolerance:.17g}
kspace_modify order {case.order} mesh {case.mesh} {case.mesh} {case.mesh} diff ad cspread {case.cspread:.17g}
pair_coeff * *
{scale_fix}

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes one 4000
{exclusion}thermo_style custom step atoms ecoul elong
thermo_modify flush yes
thermo 100

dump force all custom 1 {project_relative(dump_path)} id fx fy fz
dump_modify force sort id format line \"%d %.17g %.17g %.17g\"
{pair_diagnostics}
rerun {project_relative(trajectory_path)} dump x y z ix iy iz box yes format native
"""


def run_lammps(input_path: Path, log_path: Path, screen_path: Path) -> None:
    if not LMP.is_file():
        raise FileNotFoundError(f"development LAMMPS executable is missing: {LMP}")
    if not os.access(LMP, os.X_OK):
        raise PermissionError(f"LAMMPS executable is not executable: {LMP}")
    subprocess.run(
        [
            str(LMP),
            "-in",
            str(input_path),
            "-log",
            str(log_path),
            "-screen",
            str(screen_path),
        ],
        cwd=PROJECT,
        check=True,
    )


def parse_log_metadata(path: Path) -> dict[str, object]:
    text = path.read_text()
    grid_matches = re.findall(r"\n\s*grid\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text)
    split_matches = re.findall(r"Splitting parameter c\s*=\s*([0-9.eE+-]+)", text)
    spread_matches = re.findall(r"Spreading parameter c\s*=\s*([0-9.eE+-]+)", text)
    if not grid_matches or not split_matches or not spread_matches:
        raise RuntimeError(f"missing ESP metadata in {path}")
    return {
        "actual_grid": tuple(int(item) for item in grid_matches[-1]),
        "actual_csplit": float(split_matches[-1]),
        "actual_cspread": float(spread_matches[-1]),
        "diff_ad": "differentiation = ad" in text.lower() or "diff ad" in text.lower(),
    }


def parse_pair_local_dump(path: Path, natoms: int) -> list[np.ndarray]:
    """Reconstruct per-atom real-space pair forces from a LAMMPS local dump."""

    frames: list[np.ndarray] = []
    with path.open() as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                raise RuntimeError(f"malformed local pair dump: {path}")
            handle.readline()
            if not handle.readline().startswith("ITEM: NUMBER OF ENTRIES"):
                raise RuntimeError(f"missing local-entry count in {path}")
            nentries = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(f"missing local-dump bounds in {path}")
            for _ in range(3):
                handle.readline()
            header = handle.readline()
            if not header.startswith("ITEM: ENTRIES"):
                raise RuntimeError(f"missing local-entry header in {path}")
            force = np.zeros((natoms, 3), dtype=np.float64)
            for _ in range(nentries):
                fields = handle.readline().split()
                atom_i = int(float(fields[1])) - 1
                atom_j = int(float(fields[2])) - 1
                if not (0 <= atom_i < natoms and 0 <= atom_j < natoms):
                    raise RuntimeError(f"out-of-range pair IDs in {path}: {fields[:3]}")
                pair_force = np.asarray([float(value) for value in fields[3:6]])
                force[atom_i] += pair_force
                force[atom_j] -= pair_force
            frames.append(force)
    if not frames:
        raise RuntimeError(f"no pair-local frames parsed from {path}")
    return frames


def extract_kspace_frames(
    total_dump_path: Path, pair_dump_path: Path, natoms: int
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Return timestep, total, reconstructed pair, and corrected k-space force."""

    total = ikref.parse_force_dump(total_dump_path)
    pair = parse_pair_local_dump(pair_dump_path, natoms)
    if len(total) != len(pair):
        raise RuntimeError(
            f"total/pair frame mismatch: {len(total)} versus {len(pair)}"
        )
    return [
        (timestep, total_force, pair_force, total_force - pair_force)
        for (timestep, total_force), pair_force in zip(total, pair)
    ]


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
        (0.381, 0.743, 0.057),
    ],
    dtype=np.float64,
)


def correction_from_fraction(fraction: np.ndarray, ab: np.ndarray) -> np.ndarray:
    theta = 2.0 * math.pi * np.asarray(fraction, dtype=np.float64)
    return ab[0] * np.sin(theta) + ab[1] * np.sin(2.0 * theta)


def correction_force(
    q: np.ndarray, xyz: np.ndarray, box_length: float, mesh: int, ab: np.ndarray
) -> np.ndarray:
    fraction = np.mod(np.asarray(xyz) / (box_length / mesh), 1.0)
    return (np.asarray(q) ** 2)[:, None] * correction_from_fraction(fraction, ab)


def fit_self_correction(
    case: ADCase, box_length: float, work_dir: Path, rerun: bool = True
) -> tuple[np.ndarray, list[dict], dict[str, float]]:
    """Identify and hold out the exact two-harmonic LAMMPS AD correction."""

    case_dir = work_dir / case.case_id / "one_charge"
    case_dir.mkdir(parents=True, exist_ok=True)
    h = box_length / case.mesh
    cell = case.mesh // 2
    all_fractions = np.vstack((TRAIN_FRACTIONS, HOLDOUT_FRACTIONS))
    xyz_frames = [h * (cell + fraction)[None, :] for fraction in all_fractions]
    data_path = case_dir / "single_charge.data"
    trajectory_path = case_dir / "single_charge.lammpstrj"
    dump_path = case_dir / "forces.single_charge.dump"
    input_path = case_dir / "in.single_charge"
    log_path = case_dir / "log.single_charge.lammps"
    screen_path = case_dir / "screen.single_charge.txt"
    data_path.write_text(charge_data_text(np.asarray([1.0]), xyz_frames[0], box_length))
    trajectory_path.write_text(trajectory_text(xyz_frames, box_length))
    input_path.write_text(
        ad_kspace_input(case, data_path, trajectory_path, dump_path, exclude_pair=False)
    )
    if rerun or not dump_path.is_file():
        run_lammps(input_path, log_path, screen_path)
    frames = ikref.parse_force_dump(dump_path)
    if len(frames) != len(all_fractions):
        raise RuntimeError(f"single-charge frame mismatch for {case.case_id}")

    log_meta = parse_log_metadata(log_path)
    actual_grid = tuple(log_meta["actual_grid"])
    if len(set(actual_grid)) != 1:
        raise RuntimeError(f"noncubic actual grid for {case.case_id}: {actual_grid}")
    actual_case = replace(case, mesh=actual_grid[0])
    coeff = coefficients(actual_case)
    op = operator(actual_case, box_length)
    matrix: list[tuple[float, float]] = []
    rhs: list[float] = []
    rows: list[dict] = []
    for index, (nominal_fraction, xyz, (_, implemented)) in enumerate(
        zip(all_fractions, xyz_frames, frames)
    ):
        fraction = np.mod(xyz[0] / (box_length / actual_case.mesh), 1.0)
        raw = adref.ad_self_response_at_fractions(fraction[None, :], op, coeff.real)[0]
        observed_correction = raw - implemented[0]
        split = "fit" if index < len(TRAIN_FRACTIONS) else "holdout"
        if split == "fit":
            for component in range(3):
                theta = 2.0 * math.pi * fraction[component]
                matrix.append((math.sin(theta), math.sin(2.0 * theta)))
                rhs.append(observed_correction[component])
        rows.append(
            {
                "case_id": case.case_id,
                "requested_mesh": case.mesh,
                "actual_mesh": actual_case.mesh,
                "split": split,
                "point": index + 1,
                "nominal_sx": nominal_fraction[0],
                "nominal_sy": nominal_fraction[1],
                "nominal_sz": nominal_fraction[2],
                "sx": fraction[0],
                "sy": fraction[1],
                "sz": fraction[2],
                "raw_fx": raw[0],
                "raw_fy": raw[1],
                "raw_fz": raw[2],
                "lammps_residual_fx": implemented[0, 0],
                "lammps_residual_fy": implemented[0, 1],
                "lammps_residual_fz": implemented[0, 2],
                "observed_correction_fx": observed_correction[0],
                "observed_correction_fy": observed_correction[1],
                "observed_correction_fz": observed_correction[2],
            }
        )

    ab, _, _, _ = np.linalg.lstsq(
        np.asarray(matrix, dtype=np.float64), np.asarray(rhs, dtype=np.float64), rcond=None
    )
    fit_max = 0.0
    holdout_max = 0.0
    for row in rows:
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
        maximum = float(np.max(np.abs(residual)))
        row.update(
            correction_sin1=ab[0],
            correction_sin2=ab[1],
            predicted_correction_fx=predicted[0],
            predicted_correction_fy=predicted[1],
            predicted_correction_fz=predicted[2],
            correction_residual_max_component=maximum,
        )
        if row["split"] == "fit":
            fit_max = max(fit_max, maximum)
        else:
            holdout_max = max(holdout_max, maximum)
    return ab, rows, {
        "fit_max_abs_component": fit_max,
        "holdout_max_abs_component": holdout_max,
        "requested_mesh": case.mesh,
        "actual_mesh": actual_case.mesh,
        "actual_csplit": float(log_meta["actual_csplit"]),
        "actual_cspread": float(log_meta["actual_cspread"]),
        "minimum_abs_deconvolution_product": op.minimum_abs_deconvolution_product,
        "active_mode_count": op.active_mode_count,
        "zeroed_active_mode_count": op.zeroed_active_mode_count,
        "zeroed_active_mode_fraction": op.zeroed_active_mode_count
        / op.active_mode_count,
        "zero_deconvolution_policy": "green=0 when D=0 (matched LAMMPS compute_gf_ad)",
    }


def residual_self_cell_rms(
    case: ADCase,
    box_length: float,
    ab: np.ndarray,
    quadrature_order_per_half: int,
) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order_per_half)
    fraction_parts = []
    weight_parts = []
    for lo, hi in ((0.0, 0.5), (0.5, 1.0)):
        fraction_parts.append(0.5 * (hi - lo) * nodes + 0.5 * (hi + lo))
        weight_parts.append(0.5 * (hi - lo) * weights)
    fraction1 = np.concatenate(fraction_parts)
    weight1 = np.concatenate(weight_parts)
    coeff = coefficients(case)
    op = operator(case, box_length)
    raw = adref.ad_self_response_cell_grid(fraction1, op, coeff.real)
    correction = np.empty_like(raw)
    one = correction_from_fraction(fraction1, ab)
    correction[..., 0] = one[:, None, None]
    correction[..., 1] = one[None, :, None]
    correction[..., 2] = one[None, None, :]
    residual = raw - correction
    wx, wy, wz = np.meshgrid(weight1, weight1, weight1, indexing="ij")
    weight3 = wx * wy * wz
    return math.sqrt(float(np.sum(weight3 * np.sum(residual * residual, axis=-1))))


def pooled_rms(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def pooled_rms_jackknife_sem(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    leave = np.asarray(
        [pooled_rms(np.delete(array, index)) for index in range(len(array))]
    )
    return float(
        math.sqrt(
            (len(array) - 1.0) / len(array) * np.sum((leave - leave.mean()) ** 2)
        )
    )


def block_sem(values: Iterable[float], block_size: int = 5) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    nblock = len(array) // block_size
    if nblock < 2:
        return 0.0
    means = np.asarray(
        [array[i * block_size : (i + 1) * block_size].mean() for i in range(nblock)]
    )
    return float(means.std(ddof=1) / math.sqrt(nblock))


def lammps_version() -> str:
    lines = subprocess.run(
        [str(LMP), "-help"], check=True, text=True, capture_output=True
    ).stdout.splitlines()
    simulator = next(
        (line.strip() for line in lines if line.startswith("Large-scale Atomic")),
        "unknown",
    )
    git_info = next((line.strip() for line in lines if line.startswith("Git info")), "")
    return f"{simulator}; {git_info}".rstrip("; ")


def case_dict(case: ADCase) -> dict:
    return asdict(case)
