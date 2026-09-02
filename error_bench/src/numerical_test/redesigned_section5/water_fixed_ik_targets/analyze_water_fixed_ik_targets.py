#!/usr/bin/env python3
"""Analyze the three frozen water fixed-influence ik candidates.

Frames 0--24 are the pilot partition and frames 25--50 are the independent
holdout partition.  This script only evaluates frozen candidates; it contains
no parameter-search or acceptance logic.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
REFERENCE = PROJECT / "numerical_examples/water_trajectory_benchmark/forces.ref_ewald.dump"
TRAJECTORY = PROJECT / "numerical_examples/water_trajectory_benchmark/water_short_traj.lammpstrj"
WATER_DATA = PROJECT / "numerical_examples/water_trajectory_benchmark/water.data"
REFERENCE_LOG = PROJECT / "numerical_examples/water_trajectory_benchmark/log.force_ref_ewald.lammps"
LMP_USED = HERE.parent / "pppm_symmetric_scan" / "lmp.pppm_symmetric_scan"
FIXED_IK_PATCH = HERE.parent / "lammps_fixed_ik/esp_fixed_ik_validation.patch"
BUILD_METADATA = HERE.parent / "lammps_fixed_ik/build_metadata.json"
PILOT_STOP = 25
RCUT = 9.0
BOX_LENGTH = 30.0

CASES = (
    {
        "case_id": "fixed_ik_target_1e-3",
        "target_relative_error": 1.0e-3,
        "requested_grid": "12 12 12",
        "order": 4,
        "epsilon_split": 1.0e-3,
        "epsilon_spread": 1.0e-3,
        "csplit": 9.5392,
        "cspread": 9.5392,
    },
    {
        "case_id": "fixed_ik_target_1e-4",
        "target_relative_error": 1.0e-4,
        "requested_grid": "15 15 15",
        "order": 6,
        "epsilon_split": 1.0e-4,
        "epsilon_spread": 3.0e-5,
        "csplit": 12.024,
        "cspread": 13.251,
    },
    {
        "case_id": "fixed_ik_target_1e-5",
        "target_relative_error": 1.0e-5,
        "requested_grid": "16 16 16",
        "order": 7,
        "epsilon_split": 1.0e-5,
        "epsilon_spread": 1.0e-5,
        "csplit": 14.471,
        "cspread": 14.471,
    },
)

CANONICAL_SPLIT_BANDLIMIT = {
    1.0e-3: 9.5392,
    1.0e-4: 12.024,
    1.0e-5: 14.471,
}


@dataclass
class ForceFrame:
    timestep: int
    ids: np.ndarray
    force: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_force_dump(path: Path) -> list[ForceFrame]:
    frames: list[ForceFrame] = []
    with path.open() as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                raise RuntimeError(f"malformed dump header in {path}: {line.rstrip()}")
            timestep = int(handle.readline())
            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError(f"missing atom-count header in {path}")
            n_atoms = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(f"missing box header in {path}")
            for _ in range(3):
                handle.readline()
            atom_header = handle.readline()
            if not atom_header.startswith("ITEM: ATOMS"):
                raise RuntimeError(f"missing atom header in {path}")
            columns = atom_header.split()[2:]
            col = {name: index for index, name in enumerate(columns)}
            required = ("id", "fx", "fy", "fz")
            if any(name not in col for name in required):
                raise RuntimeError(f"force columns missing from {path}: {columns}")
            ids = np.empty(n_atoms, dtype=np.int64)
            force = np.empty((n_atoms, 3), dtype=np.float64)
            for row in range(n_atoms):
                fields = handle.readline().split()
                ids[row] = int(fields[col["id"]])
                force[row] = [float(fields[col[name]]) for name in ("fx", "fy", "fz")]
            order = np.argsort(ids)
            frames.append(ForceFrame(timestep, ids[order], force[order]))
    if not frames:
        raise RuntimeError(f"no frames parsed from {path}")
    return frames


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text()
    grid_matches = re.findall(r"\n\s*grid\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text)
    split_matches = re.findall(r"Splitting parameter c\s*=\s*([0-9.eE+-]+)", text)
    spread_matches = re.findall(r"Spreading parameter c\s*=\s*([0-9.eE+-]+)", text)
    influence_matches = re.findall(r"ik influence\s*=\s*(.+)", text)
    if not grid_matches or not split_matches or not spread_matches or not influence_matches:
        raise RuntimeError(f"missing ESP initialization metadata in {path}")
    grid = tuple(int(value) for value in grid_matches[-1])
    return {
        "actual_grid_tuple": grid,
        "actual_grid": " ".join(str(value) for value in grid),
        "actual_grid_points": math.prod(grid),
        "actual_csplit": float(split_matches[-1]),
        "actual_cspread": float(spread_matches[-1]),
        "influence": influence_matches[-1].strip(),
        "force_only_warning_present": "use force output only" in text,
    }


def sample_sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def block_sem(values: np.ndarray, block_size: int = 5) -> float:
    n_blocks = len(values) // block_size
    if n_blocks < 2:
        return 0.0
    means = np.asarray(
        [values[index * block_size : (index + 1) * block_size].mean() for index in range(n_blocks)]
    )
    return sample_sd(means) / math.sqrt(n_blocks)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(case: dict[str, object], log_meta: dict[str, object], rows, partition: str):
    selected = [row for row in rows if row["partition"] == partition] if partition != "all" else rows
    relative = np.asarray([row["rms_relative_force_error"] for row in selected], dtype=np.float64)
    absolute = np.asarray([row["rms_absolute_force_error"] for row in selected], dtype=np.float64)
    sum_diff2 = sum(float(row["sum_squared_force_difference"]) for row in selected)
    sum_ref2 = sum(float(row["sum_squared_reference_force"]) for row in selected)
    pooled = math.sqrt(sum_diff2 / sum_ref2)
    actual_grid = tuple(log_meta["actual_grid_tuple"])
    sigma_up = math.pi * RCUT * actual_grid[0] / (float(case["csplit"]) * BOX_LENGTH)
    return {
        "case_id": case["case_id"],
        "partition": partition,
        "n_frames": len(selected),
        "frame_zero_based_first": min(int(row["frame_zero_based"]) for row in selected),
        "frame_zero_based_last": max(int(row["frame_zero_based"]) for row in selected),
        "target_relative_error": case["target_relative_error"],
        "requested_grid": case["requested_grid"],
        "actual_grid": log_meta["actual_grid"],
        "grid_was_rounded": str(log_meta["actual_grid"] != case["requested_grid"]).lower(),
        "actual_grid_points": log_meta["actual_grid_points"],
        "order_P": case["order"],
        "epsilon_split": case["epsilon_split"],
        "epsilon_spread": case["epsilon_spread"],
        "csplit": case["csplit"],
        "cspread": case["cspread"],
        "sigma_up": f"{sigma_up:.12g}",
        "mean_frame_rms_relative_error": f"{relative.mean():.12g}",
        "sample_sd_frame_rms_relative_error": f"{sample_sd(relative):.12g}",
        "sem_frame_rms_relative_error": f"{sample_sd(relative) / math.sqrt(len(relative)):.12g}",
        "block5_sem_frame_rms_relative_error": f"{block_sem(relative):.12g}",
        "pooled_rms_relative_error": f"{pooled:.12g}",
        "mean_absolute_rms_force_error": f"{absolute.mean():.12g}",
        "pooled_target_satisfied": str(pooled <= float(case["target_relative_error"])).lower(),
        "operator": "LAMMPS fixed-influence ik; inside-support MathPSWF Fourier polynomial",
        "selection_role": "frozen by pilot-only screen; this partition was not used to modify parameters",
    }


def main() -> None:
    for case in CASES:
        target = float(case["target_relative_error"])
        if not math.isclose(float(case["epsilon_split"]), target):
            raise RuntimeError(f"split tolerance is not target-aligned: {case['case_id']}")
        if not math.isclose(
            float(case["csplit"]), CANONICAL_SPLIT_BANDLIMIT[target]
        ):
            raise RuntimeError(f"noncanonical csplit: {case['case_id']}")

    required = [REFERENCE, REFERENCE_LOG, TRAJECTORY, WATER_DATA, FIXED_IK_PATCH, BUILD_METADATA]
    for case in CASES:
        required.extend(
            [
                HERE / f"forces.{case['case_id']}.dump",
                HERE / f"log.force_{case['case_id']}.lammps",
                HERE / f"in.force_{case['case_id']}",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs: " + ", ".join(missing))

    reference = parse_force_dump(REFERENCE)
    if len(reference) != 50:
        raise RuntimeError(f"expected 50 Ewald frames, found {len(reference)}")

    per_frame_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    manifest_cases = []
    for case in CASES:
        case_id = str(case["case_id"])
        dump_path = HERE / f"forces.{case_id}.dump"
        log_path = HERE / f"log.force_{case_id}.lammps"
        test = parse_force_dump(dump_path)
        if len(test) != len(reference):
            raise RuntimeError(f"frame mismatch for {case_id}: {len(test)} != {len(reference)}")
        log_meta = parse_log(log_path)
        if abs(float(log_meta["actual_csplit"]) - float(case["csplit"])) > 5.0e-4:
            raise RuntimeError(f"unexpected csplit for {case_id}: {log_meta['actual_csplit']}")
        if abs(float(log_meta["actual_cspread"]) - float(case["cspread"])) > 5.0e-4:
            raise RuntimeError(f"unexpected cspread for {case_id}: {log_meta['actual_cspread']}")
        if not str(log_meta["influence"]).startswith("fixed"):
            raise RuntimeError(f"wrong influence convention for {case_id}: {log_meta['influence']}")

        case_rows = []
        for frame_index, (ref_frame, test_frame) in enumerate(zip(reference, test)):
            if ref_frame.timestep != test_frame.timestep:
                raise RuntimeError(f"timestep mismatch for {case_id} frame {frame_index}")
            if not np.array_equal(ref_frame.ids, test_frame.ids):
                raise RuntimeError(f"atom-id mismatch for {case_id} frame {frame_index}")
            difference = test_frame.force - ref_frame.force
            atom_diff2 = np.sum(difference * difference, axis=1)
            atom_ref2 = np.sum(ref_frame.force * ref_frame.force, axis=1)
            sum_diff2 = float(atom_diff2.sum())
            sum_ref2 = float(atom_ref2.sum())
            n_atoms = len(ref_frame.ids)
            relative = math.sqrt(sum_diff2 / sum_ref2)
            absolute = math.sqrt(sum_diff2 / n_atoms)
            reference_rms = math.sqrt(sum_ref2 / n_atoms)
            nonzero = atom_ref2 > 0.0
            max_atom_relative = float(np.sqrt(np.max(atom_diff2[nonzero] / atom_ref2[nonzero])))
            row = {
                "case_id": case_id,
                "target_relative_error": case["target_relative_error"],
                "frame_zero_based": frame_index,
                "timestep": ref_frame.timestep,
                "partition": "pilot" if frame_index < PILOT_STOP else "holdout",
                "n_atoms": n_atoms,
                "rms_relative_force_error": f"{relative:.15g}",
                "rms_absolute_force_error": f"{absolute:.15g}",
                "rms_reference_force": f"{reference_rms:.15g}",
                "max_atom_relative_force_error": f"{max_atom_relative:.15g}",
                "sum_squared_force_difference": f"{sum_diff2:.17g}",
                "sum_squared_reference_force": f"{sum_ref2:.17g}",
                "requested_grid": case["requested_grid"],
                "actual_grid": log_meta["actual_grid"],
                "order_P": case["order"],
                "csplit": case["csplit"],
                "cspread": case["cspread"],
                "operator": "LAMMPS fixed-influence ik; inside-support MathPSWF Fourier polynomial",
            }
            case_rows.append(row)
            per_frame_rows.append(row)

        for partition in ("pilot", "holdout", "all"):
            summary_rows.append(summarize(case, log_meta, case_rows, partition))
        input_path = HERE / f"in.force_{case_id}"
        screen_path = HERE / f"screen.force_{case_id}.txt"
        manifest_cases.append(
            {
                **case,
                **{key: value for key, value in log_meta.items() if key != "actual_grid_tuple"},
                "input": str(input_path.relative_to(PROJECT)),
                "input_sha256": sha256(input_path),
                "log": str(log_path.relative_to(PROJECT)),
                "log_sha256": sha256(log_path),
                "screen": str(screen_path.relative_to(PROJECT)),
                "screen_sha256": sha256(screen_path),
                "dump": str(dump_path.relative_to(PROJECT)),
                "dump_sha256": sha256(dump_path),
            }
        )

    write_csv(HERE / "water_fixed_ik_force_error_by_frame.csv", per_frame_rows)
    write_csv(HERE / "water_fixed_ik_partition_summary.csv", summary_rows)
    holdout = [row for row in summary_rows if row["partition"] == "holdout"]
    write_csv(HERE / "water_fixed_ik_holdout_summary.csv", holdout)

    manifest = {
        "purpose": "50-frame total-force reevaluation of three frozen water candidates",
        "selection_rule": (
            "Candidates were frozen from the pilot-only screen before this calculation. "
            "Frames 25--50 are validation-only and never alter M, P, csplit, or cspread."
        ),
        "frame_partition": {
            "pilot_zero_based": list(range(0, PILOT_STOP)),
            "holdout_zero_based": list(range(PILOT_STOP, len(reference))),
        },
        "main_report_partition": "holdout",
        "reference": {
            "method": "Ewald 1e-12, existing common reference dump",
            "path": str(REFERENCE.relative_to(PROJECT)),
            "sha256": sha256(REFERENCE),
            "log": str(REFERENCE_LOG.relative_to(PROJECT)),
            "log_sha256": sha256(REFERENCE_LOG),
            "n_frames": len(reference),
        },
        "trajectory": {
            "path": str(TRAJECTORY.relative_to(PROJECT)),
            "sha256": sha256(TRAJECTORY),
        },
        "water_data": {
            "path": str(WATER_DATA.relative_to(PROJECT)),
            "sha256": sha256(WATER_DATA),
        },
        "force_definition": (
            "Per-frame relative RMS = sqrt(sum_i |F_fixed-ik_i-F_Ewald_i|^2 / "
            "sum_i |F_Ewald_i|^2); absolute RMS = sqrt(sum_i |Delta F_i|^2/N)."
        ),
        "operator_limitation": (
            "Validation-only LAMMPS fixed-influence ik uses its inside-support Fourier "
            "polynomial/zero-tail convention; energy and virial are not used."
        ),
        "implementation": {
            "lammps_executable_used": str(LMP_USED.relative_to(PROJECT)),
            "lammps_executable_sha256": sha256(LMP_USED) if LMP_USED.is_file() else "unavailable",
            "fixed_ik_patch": str(FIXED_IK_PATCH.relative_to(PROJECT)),
            "fixed_ik_patch_sha256": sha256(FIXED_IK_PATCH),
            "build_metadata": str(BUILD_METADATA.relative_to(PROJECT)),
            "build_metadata_sha256": sha256(BUILD_METADATA),
        },
        "cases": manifest_cases,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for row in holdout:
        print(
            f"{row['case_id']}: actual grid {row['actual_grid']}, "
            f"holdout mean={row['mean_frame_rms_relative_error']}, "
            f"pooled={row['pooled_rms_relative_error']}"
        )


if __name__ == "__main__":
    main()
