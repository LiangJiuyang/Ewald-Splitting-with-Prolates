#!/usr/bin/env python3
"""Run the matched IK/AD order scans for the four-panel main Figure 5.

The two columns hold the ESP split and spreading bandlimits fixed at either
12.024 (1e-4) or 14.471 (1e-5).  The two rows use fixed-influence IK and
production AD differentiation, respectively.  Only the stencil order and FFT
grid vary within a panel.  All reported errors use frames 26--51 so the scan
can share the existing fixed-G PPPM holdout baseline without data leakage.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REDESIGNED = HERE.parent
PROJECT = HERE.parents[3]
SOURCE = PROJECT / "numerical_examples" / "water_trajectory_benchmark"
DATA = SOURCE / "water.data"
TRAJECTORY = SOURCE / "water_short_traj.lammpstrj"
REFERENCE = SOURCE / "forces.ref_ewald.dump"
LMP = REDESIGNED / "pppm_symmetric_scan" / "lmp.pppm_symmetric_scan"
EXPECTED_LMP_SHA256 = (
    "34332fa52c4e2ba72b9561cffbc841c9b4fdbf5809eb745b1c1656e4ac960d6a"
)

RCUT = 9.0
BOX_LENGTH = 30.0
TOTAL_FRAMES = 51
PILOT_FRAMES = 25
ORDERS = tuple(range(5, 10))
TARGET_SETTINGS = {
    1.0e-4: {
        "epsilon_split": 1.0e-4,
        "epsilon_spread": 1.0e-4,
        "csplit": 12.024,
        "cspread": 12.024,
        "meshes": (12, 15, 16, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80),
    },
    1.0e-5: {
        "epsilon_split": 1.0e-5,
        "epsilon_spread": 1.0e-5,
        "csplit": 14.471,
        "cspread": 14.471,
        "meshes": (12, 16, 18, 20, 24, 32, 36, 40, 48, 64, 80),
    },
}

RAW = HERE / "raw"
SUMMARY = HERE / "fig5_ik_ad_order_scan_summary.csv"
BY_FRAME = HERE / "fig5_ik_ad_order_scan_by_frame.csv"
MANIFEST = HERE / "fig5_ik_ad_order_scan_manifest.json"

IK_DIR = REDESIGNED / "water_fixed_ik_p_scan"
AD_DIR = REDESIGNED / "lammps_ad_total_validation"
sys.path.insert(0, str(IK_DIR))
sys.path.insert(0, str(AD_DIR))
import run_water_fixed_ik_p_scan as ikscan  # noqa: E402
import ad_validation_common as adcommon  # noqa: E402
import run_water_ad_validation as adproduction  # noqa: E402


@dataclass(frozen=True)
class Case:
    method: str
    target: float
    order: int
    mesh: int
    epsilon_split: float
    epsilon_spread: float
    csplit: float
    cspread: float

    @property
    def target_tag(self) -> str:
        return f"{self.target:.0e}".replace("e-0", "e-")

    @property
    def case_id(self) -> str:
        return f"fig5_{self.method}_{self.target_tag}_p{self.order}_m{self.mesh}"

    @property
    def sigma_up(self) -> float:
        return math.pi * RCUT * self.mesh / (self.csplit * BOX_LENGTH)


def build_cases() -> tuple[Case, ...]:
    cases: list[Case] = []
    for method in ("ik", "ad"):
        for target, settings in TARGET_SETTINGS.items():
            for order in ORDERS:
                for mesh in settings["meshes"]:
                    cases.append(
                        Case(
                            method=method,
                            target=target,
                            order=order,
                            mesh=mesh,
                            epsilon_split=float(settings["epsilon_split"]),
                            epsilon_spread=float(settings["epsilon_spread"]),
                            csplit=float(settings["csplit"]),
                            cspread=float(settings["cspread"]),
                        )
                    )
    return tuple(cases)


CASES = build_cases()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT))


def reused_paths(case: Case) -> dict[str, Path] | None:
    if case.target != 1.0e-5:
        return None
    if (
        case.method == "ik"
        and case.order in (5, 6, 7, 8)
        and case.mesh in TARGET_SETTINGS[1.0e-5]["meshes"][1:]
    ):
        old_id = f"fixed_ik_target_1e-5_p{case.order}_m{case.mesh}"
        return {
            "input": IK_DIR / f"in.{old_id}",
            "dump": IK_DIR / f"forces.{old_id}.dump",
            "log": IK_DIR / f"log.{old_id}.lammps",
        }
    if case.method == "ad" and case.order == 8 and case.mesh == 16:
        base = AD_DIR / "runs_water_grid_reduction" / "water_ad_grid_1e-5_P8_M16"
        return {
            "input": base / "in.full",
            "dump": base / "forces.full.dump",
            "log": base / "log.full.lammps",
        }
    return None


def paths_for(case: Case) -> tuple[dict[str, Path], str]:
    reused = reused_paths(case)
    if reused is not None:
        return reused, "reused audited Figure-5 force record"
    base = RAW / case.case_id
    return (
        {
            "input": base / "in.lammps",
            "dump": base / "forces.dump",
            "log": base / "log.lammps",
        },
        "dedicated four-panel Figure-5 order scan",
    )


def input_text(case: Case, dump: Path) -> str:
    if case.method == "ik":
        return f"""# Matched fixed-band ESP-IK order scan for Figure 5.
newton on
units real
atom_style full
read_data {relpath(DATA)}
reset_timestep 0

{adproduction.MOLECULAR_PREAMBLE}pair_style coul/esp {RCUT:.1f}
kspace_style esp {case.epsilon_split:.1e} {case.epsilon_spread:.1e}
kspace_modify order {case.order} diff ik mesh {case.mesh} {case.mesh} {case.mesh} cspread {case.cspread:.12g} influence fixed
pair_coeff * *

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes one 4000
thermo_style custom step atoms
thermo 100

dump force all custom 1 {relpath(dump)} id fx fy fz
dump_modify force sort id format line "%d %.17g %.17g %.17g"
rerun {relpath(TRAJECTORY)} dump x y z ix iy iz box yes format native
"""
    ad_case = adcommon.ADCase(
        case_id=case.case_id,
        mesh=case.mesh,
        order=case.order,
        csplit=case.csplit,
        cspread=case.cspread,
        split_input_tolerance=case.epsilon_split,
        spread_input_tolerance=case.epsilon_spread,
        target_relative_error=case.target,
    )
    return adcommon.ad_kspace_input(
        ad_case,
        DATA,
        TRAJECTORY,
        dump,
        atom_style="full",
        molecular_preamble=adproduction.MOLECULAR_PREAMBLE,
    )


def parse_metadata(case: Case, log: Path) -> dict[str, object]:
    text = log.read_text(encoding="utf-8", errors="replace")
    grids = re.findall(r"\n\s*grid\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text)
    splits = re.findall(r"Splitting parameter c\s*=\s*([0-9.eE+-]+)", text)
    spreads = re.findall(r"Spreading parameter c\s*=\s*([0-9.eE+-]+)", text)
    orders = re.findall(r"^\s*stencil order\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    loops = re.findall(
        r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps",
        text,
    )
    versions = re.findall(r"^LAMMPS \((.+)\)\s*$", text, re.MULTILINE)
    if not all((grids, splits, spreads, orders, loops, versions)) or "ERROR:" in text:
        raise RuntimeError(f"incomplete LAMMPS log: {log}")
    metadata = {
        "grid": tuple(int(item) for item in grids[-1]),
        "csplit": float(splits[-1]),
        "cspread": float(spreads[-1]),
        "order": int(orders[-1]),
        "loop_time_s": float(loops[-1][0]),
        "mpi_ranks": int(loops[-1][1]),
        "loop_steps": int(loops[-1][2]),
        "version": versions[-1],
        "diff_ad": "differentiation = ad" in text.lower() or "diff ad" in text.lower(),
        "influence_fixed": "influence function = fixed" in text.lower(),
    }
    if metadata["grid"] != (case.mesh,) * 3:
        raise RuntimeError(f"{case.case_id}: actual grid mismatch")
    if metadata["order"] != case.order:
        raise RuntimeError(f"{case.case_id}: actual order mismatch")
    if abs(float(metadata["csplit"]) - case.csplit) > 5.0e-4:
        raise RuntimeError(f"{case.case_id}: actual csplit mismatch")
    if abs(float(metadata["cspread"]) - case.cspread) > 5.0e-4:
        raise RuntimeError(f"{case.case_id}: actual cspread mismatch")
    if case.method == "ad" and not metadata["diff_ad"]:
        raise RuntimeError(f"{case.case_id}: AD differentiation was not used")
    if case.method == "ik" and metadata["diff_ad"]:
        raise RuntimeError(f"{case.case_id}: IK differentiation was not used")
    if metadata["mpi_ranks"] != 1 or metadata["loop_steps"] != TOTAL_FRAMES:
        raise RuntimeError(f"{case.case_id}: wrong rank or frame count")
    return metadata


def complete(case: Case, paths: dict[str, Path]) -> bool:
    if any(not paths[key].is_file() for key in ("input", "dump", "log")):
        return False
    try:
        parse_metadata(case, paths["log"])
        frames = ikscan.core.parse_force_dump(paths["dump"])
    except (OSError, RuntimeError, ValueError):
        return False
    return len(frames) == TOTAL_FRAMES


def run_case(case: Case, force: bool) -> str:
    paths, role = paths_for(case)
    if role.startswith("reused"):
        if not complete(case, paths):
            raise RuntimeError(f"{case.case_id}: reused source is incomplete")
        return f"{case.case_id}: reused"

    paths["input"].parent.mkdir(parents=True, exist_ok=True)
    paths["input"].write_text(input_text(case, paths["dump"]), encoding="utf-8")
    if complete(case, paths) and not force:
        return f"{case.case_id}: cached"
    existing = [path for key, path in paths.items() if key != "input" and path.exists()]
    if existing and not force:
        raise RuntimeError(f"{case.case_id}: incomplete output exists; use --force")
    if force:
        for key, path in paths.items():
            if key != "input" and path.is_file():
                path.unlink()

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    result = subprocess.run(
        [
            str(LMP),
            "-in",
            str(paths["input"]),
            "-log",
            str(paths["log"]),
            "-screen",
            "none",
        ],
        cwd=PROJECT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not complete(case, paths):
        raise RuntimeError(
            f"{case.case_id}: LAMMPS run failed:\n{(result.stdout or '')[-4000:]}"
        )
    return f"{case.case_id}: complete"


def pooled(diff2: list[float], ref2: list[float]) -> float:
    return math.sqrt(math.fsum(diff2) / math.fsum(ref2))


def balanced_holdout_sem(diff2: list[float], ref2: list[float]) -> float:
    if len(diff2) != TOTAL_FRAMES - PILOT_FRAMES or len(ref2) != len(diff2):
        raise ValueError("holdout must contain frames 26--51")
    bounds = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 26))
    values = [pooled(diff2[start:stop], ref2[start:stop]) for start, stop in bounds]
    return statistics.stdev(values) / math.sqrt(len(values))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze() -> None:
    reference = ikscan.core.parse_force_dump(REFERENCE)
    if len(reference) != TOTAL_FRAMES:
        raise RuntimeError("tight Ewald reference does not contain 51 frames")

    summaries: list[dict[str, object]] = []
    by_frame: list[dict[str, object]] = []
    raw_artifacts: list[dict[str, object]] = []
    for case in CASES:
        paths, role = paths_for(case)
        if not complete(case, paths):
            raise RuntimeError(f"{case.case_id}: output is incomplete")
        metadata = parse_metadata(case, paths["log"])
        forces = ikscan.core.parse_force_dump(paths["dump"])
        all_diff2: list[float] = []
        all_ref2: list[float] = []
        for frame, (test_frame, ref_frame) in enumerate(zip(forces, reference)):
            if test_frame.timestep != ref_frame.timestep:
                raise RuntimeError(f"{case.case_id}: reference frame mismatch")
            if not (test_frame.ids == ref_frame.ids).all():
                raise RuntimeError(f"{case.case_id}: reference atom-id mismatch")
            test = test_frame.force
            ref = ref_frame.force
            if test.shape != ref.shape:
                raise RuntimeError(f"{case.case_id}: reference force-shape mismatch")
            delta = test - ref
            diff2 = float((delta * delta).sum())
            ref2 = float((ref * ref).sum())
            all_diff2.append(diff2)
            all_ref2.append(ref2)
            by_frame.append(
                {
                    "case_id": case.case_id,
                    "method": case.method,
                    "target_relative_rms": case.target,
                    "order": case.order,
                    "actual_nx": case.mesh,
                    "frame_zero_based": frame,
                    "partition": "pilot" if frame < PILOT_FRAMES else "holdout",
                    "timestep": ref_frame.timestep,
                    "sum_squared_force_difference": diff2,
                    "sum_squared_reference_force": ref2,
                    "frame_relative_rms": math.sqrt(diff2 / ref2),
                }
            )
        holdout_diff2 = all_diff2[PILOT_FRAMES:]
        holdout_ref2 = all_ref2[PILOT_FRAMES:]
        holdout_error = pooled(holdout_diff2, holdout_ref2)
        summaries.append(
            {
                "case_id": case.case_id,
                "method": case.method,
                "differentiation": "fixed-influence ik" if case.method == "ik" else "production ad",
                "target_relative_rms": case.target,
                "order": case.order,
                "requested_mesh": case.mesh,
                "actual_nx": case.mesh,
                "actual_ny": case.mesh,
                "actual_nz": case.mesh,
                "actual_grid_points": case.mesh**3,
                "sigma_up": case.sigma_up,
                "epsilon_split": case.epsilon_split,
                "epsilon_spread": case.epsilon_spread,
                "csplit": case.csplit,
                "cspread": case.cspread,
                "total_frames": TOTAL_FRAMES,
                "holdout_frames": TOTAL_FRAMES - PILOT_FRAMES,
                "all_frame_relative_rms": pooled(all_diff2, all_ref2),
                "holdout_relative_rms": holdout_error,
                "holdout_balanced_block5_sem": balanced_holdout_sem(
                    holdout_diff2, holdout_ref2
                ),
                "holdout_passes_target": holdout_error <= case.target,
                "loop_time_s_single_rank": metadata["loop_time_s"],
                "force_dump_sha256": sha256(paths["dump"]),
                "input_sha256": sha256(paths["input"]),
                "log_sha256": sha256(paths["log"]),
                "source_role": role,
                "operator": (
                    "LAMMPS ESP fixed-influence IK"
                    if case.method == "ik"
                    else "LAMMPS ESP production AD with residual-self correction"
                ),
            }
        )
        for kind in ("input", "dump", "log"):
            path = paths[kind]
            raw_artifacts.append(
                {
                    "case_id": case.case_id,
                    "kind": kind,
                    "path": relpath(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "source_role": role,
                }
            )

    summaries.sort(
        key=lambda row: (
            str(row["method"]),
            -float(row["target_relative_rms"]),
            int(row["order"]),
            int(row["actual_nx"]),
        )
    )
    by_frame.sort(
        key=lambda row: (
            str(row["method"]),
            -float(row["target_relative_rms"]),
            int(row["order"]),
            int(row["actual_nx"]),
            int(row["frame_zero_based"]),
        )
    )
    write_csv(SUMMARY, summaries)
    write_csv(BY_FRAME, by_frame)

    expected = sum(len(settings["meshes"]) for settings in TARGET_SETTINGS.values())
    expected *= len(ORDERS) * 2
    if len(summaries) != expected:
        raise RuntimeError("four-panel scan has the wrong number of cases")
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Main Figure 5 matched fixed-band IK/AD order scan",
        "figure_contract": (
            "Panels a/c use csplit=cspread=12.024 at target 1e-4; panels b/d "
            "use csplit=cspread=14.471 at target 1e-5. All panels scan P=5--9, "
            "vary only M along a curve, include the common under-resolved "
            "M=12 diagnostic, and report frames 26--51."
        ),
        "orders": list(ORDERS),
        "target_settings": {
            f"{target:.0e}": {
                key: list(value) if key == "meshes" else value
                for key, value in settings.items()
            }
            for target, settings in TARGET_SETTINGS.items()
        },
        "lammps_executable": relpath(LMP),
        "lammps_executable_sha256": sha256(LMP),
        "expected_lammps_executable_sha256": EXPECTED_LMP_SHA256,
        "single_rank_per_case": True,
        "omp_num_threads": 1,
        "trajectory": {"path": relpath(TRAJECTORY), "sha256": sha256(TRAJECTORY)},
        "water_data": {"path": relpath(DATA), "sha256": sha256(DATA)},
        "reference": {"path": relpath(REFERENCE), "sha256": sha256(REFERENCE)},
        "error_definition": (
            "Pooled relative RMS on frames 26--51 against the common tight "
            "Ewald force dump; uncertainty is the SEM over balanced holdout "
            "blocks of sizes 5,5,5,5,6."
        ),
        "runner": {"path": relpath(Path(__file__)), "sha256": sha256(Path(__file__))},
        "raw_artifacts": raw_artifacts,
        "outputs": {
            path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in (SUMMARY, BY_FRAME)
        },
    }
    if manifest["lammps_executable_sha256"] != EXPECTED_LMP_SHA256:
        raise RuntimeError("LAMMPS executable hash does not match the manuscript build")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Analyzed {len(summaries)} cases: "
        f"{sum(row['source_role'].startswith('reused') for row in summaries)} reused, "
        f"{sum(not row['source_role'].startswith('reused') for row in summaries)} dedicated"
    )


def select_cases(patterns: list[str]) -> tuple[Case, ...]:
    if not patterns:
        return CASES
    selected = tuple(
        case for case in CASES if any(pattern in case.case_id for pattern in patterns)
    )
    if not selected:
        raise ValueError("--case filters did not match any case")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.analyze_only and (args.case or args.skip_analyze or args.force):
        raise ValueError("--analyze-only cannot be combined with run controls")
    for path in (LMP, DATA, TRAJECTORY, REFERENCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(LMP) != EXPECTED_LMP_SHA256:
        raise RuntimeError("LAMMPS executable hash does not match the manuscript build")

    if not args.analyze_only:
        selected = select_cases(args.case)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_case, case, args.force): case for case in selected}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    print(future.result(), flush=True)
                except Exception as error:
                    raise RuntimeError(f"scan failed for {case.case_id}") from error
        if args.case and not args.skip_analyze:
            raise ValueError("filtered runs require --skip-analyze until the full matrix exists")
    if not args.skip_analyze:
        analyze()


if __name__ == "__main__":
    main()
