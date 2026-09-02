#!/usr/bin/env python3
"""Run the SPC/E fixed-influence ik grid scans used in main Figure 5.

Panel (a) fixes the tightest compared ESP split/spread tuple,
``(epsilon_split, epsilon_spread)=(1e-5,1e-5)`` and
``(c_split,c_spread)=(14.471,14.471)``, then scans orders P=4--8 and
FFT-friendly grids.  Panel (b) compares fixed-P=5 PPPM with target-specific
ESP orders P=5, 6, 8, and 10 at targets 1e-3, 1e-4, 1e-5, and 1e-6,
respectively.  The ESP side applies the target-specific split/spread tuples
and explicitly measures every smaller declared grid needed to establish its
first qualifying grid.  The PPPM source scan is maintained separately.

All errors are measured on the same 50 SPC/E configurations against the same
tight Ewald reference.  This is a retrospective matched-accuracy benchmark,
not an automatic parameter-selection or held-out validation calculation.
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
PARENT = HERE.parent
PROJECT = HERE.parents[3]
SOURCE_DIR = PROJECT / "numerical_examples" / "water_trajectory_benchmark"
DATA = SOURCE_DIR / "water.data"
TRAJECTORY = SOURCE_DIR / "water_short_traj.lammpstrj"
REFERENCE = SOURCE_DIR / "forces.ref_ewald.dump"
DEFAULT_LMP = PARENT / "pppm_symmetric_scan" / "lmp.pppm_symmetric_scan"
EXPECTED_LMP_SHA256 = (
    "34332fa52c4e2ba72b9561cffbc841c9b4fdbf5809eb745b1c1656e4ac960d6a"
)
EXPECTED_FRAMES = 50
RCUT = 9.0
BOX_LENGTH = 30.0

# M=16 is included in panel (a) because it is the smallest FFT-friendly grid
# that resolves the c_split=14.471 band.
PANEL_A_GRIDS = (16, 18, 20, 24, 32, 36, 40, 48, 64, 80)
PANEL_A_TARGET = 1.0e-5
PANEL_A_ORDERS = tuple(range(4, 9))

# Figure 5(b) contrasts a conventional fixed-P=5 PPPM scan with the requested
# target-dependent ESP orders.  Each target-specific list contains every
# declared grid below the first anticipated qualifying grid.  M=18 is the
# first grid that resolves the c_split=16.894 band for the 1e-6 case.
PANEL_B_ESP_ORDER_BY_TARGET = {
    1.0e-3: 5,
    1.0e-4: 6,
    1.0e-5: 8,
    1.0e-6: 10,
}
PANEL_B_ESP_GRIDS_BY_TARGET = {
    1.0e-3: (12,),
    1.0e-4: (12, 15),
    1.0e-5: (12, 15),
    1.0e-6: (12, 15, 16, 18),
}

TARGET_TUPLES = {
    1.0e-3: {
        "epsilon_split": 1.0e-3,
        "epsilon_spread": 1.0e-3,
        "csplit": 9.5392,
        "cspread": 9.5392,
    },
    1.0e-4: {
        "epsilon_split": 1.0e-4,
        "epsilon_spread": 3.0e-5,
        "csplit": 12.024,
        "cspread": 13.251,
    },
    1.0e-5: {
        "epsilon_split": 1.0e-5,
        "epsilon_spread": 1.0e-5,
        "csplit": 14.471,
        "cspread": 14.471,
    },
    1.0e-6: {
        "epsilon_split": 1.0e-6,
        "epsilon_spread": 1.0e-6,
        "csplit": 16.894,
        "cspread": 16.894,
    },
}

SUMMARY_CSV = HERE / "fixed_ik_p_scan_summary.csv"
BY_FRAME_CSV = HERE / "fixed_ik_p_scan_by_frame.csv"
SELECTION_CSV = HERE / "fixed_ik_p_scan_selection.csv"
MANIFEST = HERE / "manifest.json"

sys.path.insert(0, str(PARENT / "water_fixed_ik_targets"))
import analyze_water_fixed_ik_targets as core  # noqa: E402


@dataclass(frozen=True)
class Case:
    target: float
    order: int
    mesh: int
    epsilon_split: float
    epsilon_spread: float
    csplit: float
    cspread: float
    scopes: tuple[str, ...]

    @property
    def target_tag(self) -> str:
        return f"{self.target:.0e}".replace("e-0", "e-")

    @property
    def case_id(self) -> str:
        return f"fixed_ik_target_{self.target_tag}_p{self.order}_m{self.mesh}"

    @property
    def sigma_up(self) -> float:
        return math.pi * RCUT * self.mesh / (self.csplit * BOX_LENGTH)


def build_cases() -> tuple[Case, ...]:
    records: dict[tuple[float, int, int], Case] = {}

    def add(
        target: float,
        order: int,
        mesh: int,
        scope: str,
        *,
        allow_underresolved: bool = False,
    ) -> None:
        settings = TARGET_TUPLES[target]
        case = Case(
            target=target,
            order=order,
            mesh=mesh,
            epsilon_split=float(settings["epsilon_split"]),
            epsilon_spread=float(settings["epsilon_spread"]),
            csplit=float(settings["csplit"]),
            cspread=float(settings["cspread"]),
            scopes=(scope,),
        )
        if case.sigma_up < 1.0 - 1.0e-12 and not allow_underresolved:
            raise ValueError(f"undersampled requested case {case.case_id}: sigma={case.sigma_up}")
        key = (target, order, mesh)
        prior = records.get(key)
        if prior is None:
            records[key] = case
        else:
            records[key] = Case(
                target=prior.target,
                order=prior.order,
                mesh=prior.mesh,
                epsilon_split=prior.epsilon_split,
                epsilon_spread=prior.epsilon_spread,
                csplit=prior.csplit,
                cspread=prior.cspread,
                scopes=tuple(sorted(set(prior.scopes + (scope,)))),
            )

    for order in PANEL_A_ORDERS:
        for mesh in PANEL_A_GRIDS:
            add(PANEL_A_TARGET, order, mesh, "panel_a_fixed_band")

    for target, order in PANEL_B_ESP_ORDER_BY_TARGET.items():
        for mesh in PANEL_B_ESP_GRIDS_BY_TARGET[target]:
            # The small-grid measurements are a retrospective numerical
            # comparison.  Under-resolved candidates are retained to prove
            # that they fail rather than being discarded a priori.
            add(
                target,
                order,
                mesh,
                "panel_b_mixed_order",
                allow_underresolved=True,
            )

    return tuple(
        records[key]
        for key in sorted(records, key=lambda item: (-item[0], item[1], item[2]))
    )


CASES = build_cases()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_for(case: Case) -> dict[str, Path]:
    return {
        "input": HERE / f"in.{case.case_id}",
        "dump": HERE / f"forces.{case.case_id}.dump",
        "log": HERE / f"log.{case.case_id}.lammps",
        "screen": HERE / f"screen.{case.case_id}.txt",
        "launcher": HERE / f"launcher.{case.case_id}.txt",
    }


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT).as_posix()


def input_text(case: Case, dump: Path) -> str:
    return f"""# SPC/E fixed-P grid scan for Figure 5; force output only.
newton on
units real
atom_style full
read_data {project_relative(DATA)}
reset_timestep 0

bond_style harmonic
angle_style harmonic
bond_coeff 1 0.0 1.000
angle_coeff 1 0.0 109.47
special_bonds lj/coul 0 0 0.5

pair_style coul/esp {RCUT:.1f}
kspace_style esp {case.epsilon_split:.1e} {case.epsilon_spread:.1e}
kspace_modify order {case.order} diff ik mesh {case.mesh} {case.mesh} {case.mesh} cspread {case.cspread:.12g} influence fixed
pair_coeff * *

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes one 2000
thermo_style custom step atoms
thermo 100

dump f all custom 1 {project_relative(dump)} id fx fy fz
dump_modify f sort id format line "%d %.17g %.17g %.17g"
rerun {project_relative(TRAJECTORY)} dump x y z ix iy iz box yes format native
"""


def parse_log(path: Path) -> dict[str, object]:
    metadata = core.parse_log(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    orders = re.findall(r"^\s*stencil order\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    loops = re.findall(
        r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps",
        text,
    )
    versions = re.findall(r"^LAMMPS \((.+)\)\s*$", text, re.MULTILINE)
    if not orders or not loops or not versions or "ERROR:" in text:
        raise RuntimeError(f"incomplete or failed LAMMPS log: {path}")
    metadata.update(
        {
            "actual_order": int(orders[-1]),
            "loop_time_s": float(loops[-1][0]),
            "mpi_ranks": int(loops[-1][1]),
            "loop_steps": int(loops[-1][2]),
            "lammps_version": versions[-1],
        }
    )
    return metadata


def complete(case: Case, paths: dict[str, Path]) -> bool:
    if not paths["dump"].is_file() or not paths["log"].is_file():
        return False
    try:
        metadata = parse_log(paths["log"])
        frames = core.parse_force_dump(paths["dump"])
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        metadata["actual_order"] == case.order
        and tuple(metadata["actual_grid_tuple"]) == (case.mesh,) * 3
        and abs(float(metadata["actual_csplit"]) - case.csplit) <= 5.0e-4
        and abs(float(metadata["actual_cspread"]) - case.cspread) <= 5.0e-4
        and str(metadata["influence"]).startswith("fixed")
        and metadata["loop_steps"] == EXPECTED_FRAMES
        and len(frames) == EXPECTED_FRAMES
    )


def run_case(lmp: Path, case: Case, force: bool) -> str:
    paths = paths_for(case)
    paths["input"].write_text(input_text(case, paths["dump"]), encoding="utf-8")
    if complete(case, paths) and not force:
        return f"{case.case_id}: cached"
    existing = [
        path for key, path in paths.items() if key != "input" and path.exists()
    ]
    if existing and not force:
        raise RuntimeError(f"{case.case_id}: incomplete output exists; use --force")
    if force:
        for key, path in paths.items():
            if key != "input" and path.is_file():
                path.unlink()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            str(lmp),
            "-in",
            str(paths["input"]),
            "-log",
            str(paths["log"]),
            "-screen",
            str(paths["screen"]),
        ],
        cwd=PROJECT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    paths["launcher"].write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0 or not complete(case, paths):
        raise RuntimeError(f"{case.case_id}: LAMMPS run or output audit failed")
    return f"{case.case_id}: complete"


def block_rms_sem(values: list[float], block_size: int = 5) -> float:
    count = len(values) // block_size
    block_values = [
        math.sqrt(
            math.fsum(
                value * value
                for value in values[index * block_size : (index + 1) * block_size]
            )
            / block_size
        )
        for index in range(count)
    ]
    return statistics.stdev(block_values) / math.sqrt(count)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(lmp: Path) -> None:
    reference = core.parse_force_dump(REFERENCE)
    if len(reference) != EXPECTED_FRAMES:
        raise RuntimeError(f"expected {EXPECTED_FRAMES} reference frames")

    summaries: list[dict[str, object]] = []
    by_frame: list[dict[str, object]] = []
    manifest_cases: list[dict[str, object]] = []
    for case in CASES:
        paths = paths_for(case)
        test = core.parse_force_dump(paths["dump"])
        metadata = parse_log(paths["log"])
        if len(test) != len(reference):
            raise RuntimeError(f"{case.case_id}: incomplete force dump")
        if metadata["actual_order"] != case.order:
            raise RuntimeError(f"{case.case_id}: actual order mismatch")
        if tuple(metadata["actual_grid_tuple"]) != (case.mesh,) * 3:
            raise RuntimeError(f"{case.case_id}: actual grid mismatch")
        if abs(float(metadata["actual_csplit"]) - case.csplit) > 5.0e-4:
            raise RuntimeError(f"{case.case_id}: csplit mismatch")
        if abs(float(metadata["actual_cspread"]) - case.cspread) > 5.0e-4:
            raise RuntimeError(f"{case.case_id}: cspread mismatch")
        if not str(metadata["influence"]).startswith("fixed"):
            raise RuntimeError(f"{case.case_id}: influence mismatch")

        relative_values: list[float] = []
        total_diff2 = 0.0
        total_ref2 = 0.0
        for frame_index, (ref_frame, test_frame) in enumerate(zip(reference, test)):
            if ref_frame.timestep != test_frame.timestep:
                raise RuntimeError(f"{case.case_id}: timestep mismatch")
            if not (ref_frame.ids == test_frame.ids).all():
                raise RuntimeError(f"{case.case_id}: atom-id mismatch")
            difference = test_frame.force - ref_frame.force
            diff2 = float((difference * difference).sum())
            ref2 = float((ref_frame.force * ref_frame.force).sum())
            relative = math.sqrt(diff2 / ref2)
            relative_values.append(relative)
            total_diff2 += diff2
            total_ref2 += ref2
            by_frame.append(
                {
                    "case_id": case.case_id,
                    "target_relative_rms": case.target,
                    "order": case.order,
                    "requested_mesh": case.mesh,
                    "frame_zero_based": frame_index,
                    "timestep": ref_frame.timestep,
                    "relative_rms_force_error": relative,
                    "sum_squared_force_difference": diff2,
                    "sum_squared_reference_force": ref2,
                }
            )

        pooled = math.sqrt(total_diff2 / total_ref2)
        actual_grid = tuple(metadata["actual_grid_tuple"])
        summaries.append(
            {
                "case_id": case.case_id,
                "target_relative_rms": case.target,
                "scopes": ";".join(case.scopes),
                "n_frames": len(relative_values),
                "order": case.order,
                "requested_mesh": case.mesh,
                "actual_nx": actual_grid[0],
                "actual_ny": actual_grid[1],
                "actual_nz": actual_grid[2],
                "actual_grid_points": math.prod(actual_grid),
                "sigma_up": case.sigma_up,
                "epsilon_split": case.epsilon_split,
                "epsilon_spread": case.epsilon_spread,
                "csplit": case.csplit,
                "cspread": case.cspread,
                "pooled_rms_relative_error": pooled,
                "block5_rms_sem": block_rms_sem(relative_values),
                "passes_target": pooled <= case.target,
                "loop_time_s_single_rank": metadata["loop_time_s"],
                "force_dump_sha256": sha256(paths["dump"]),
                "operator": "LAMMPS fixed-influence ik; inside-support Fourier polynomial",
            }
        )
        manifest_cases.append(
            {
                "case_id": case.case_id,
                "target_relative_rms": case.target,
                "scopes": list(case.scopes),
                "requested_mesh": case.mesh,
                "actual_grid": list(actual_grid),
                "order": case.order,
                "epsilon_split": case.epsilon_split,
                "epsilon_spread": case.epsilon_spread,
                "expected_csplit": case.csplit,
                "actual_csplit": metadata["actual_csplit"],
                "expected_cspread": case.cspread,
                "actual_cspread": metadata["actual_cspread"],
                "sigma_up": case.sigma_up,
                "input_sha256": sha256(paths["input"]),
                "dump_sha256": sha256(paths["dump"]),
                "log_sha256": sha256(paths["log"]),
                "screen_sha256": sha256(paths["screen"]),
            }
        )

    summaries.sort(
        key=lambda row: (
            -float(row["target_relative_rms"]),
            int(row["order"]),
            int(row["requested_mesh"]),
        )
    )
    by_frame.sort(
        key=lambda row: (
            -float(row["target_relative_rms"]),
            int(row["order"]),
            int(row["requested_mesh"]),
            int(row["frame_zero_based"]),
        )
    )

    selection_rows: list[dict[str, object]] = []
    # Panel (a): report the first passing grid when the fixed-order curve reaches
    # the target.  If it does not, retain the lowest measured point in the
    # predeclared scan and mark it explicitly as non-passing.  This avoids
    # silently dropping a scientifically important failed curve (P=4 here).
    for order in PANEL_A_ORDERS:
        scanned = [
            row
            for row in summaries
            if float(row["target_relative_rms"]) == PANEL_A_TARGET
            and int(row["order"]) == order
            and "panel_a_fixed_band" in str(row["scopes"])
        ]
        if not scanned:
            raise RuntimeError(f"P={order} is absent from the panel-a scan")
        passing = [row for row in scanned if bool(row["passes_target"])]
        if passing:
            selected = min(passing, key=lambda row: int(row["actual_grid_points"]))
            selection_scope = "smallest panel-a grid meeting 1e-5 at fixed bandlimit"
            selection_status = "target_met"
        else:
            selected = min(
                scanned,
                key=lambda row: (
                    float(row["pooled_rms_relative_error"]),
                    int(row["actual_grid_points"]),
                ),
            )
            selection_scope = "lowest measured panel-a error; target not met in scanned grids"
            selection_status = "target_not_met"
        selection_rows.append(
            {
                "panel": "a",
                "selection_scope": selection_scope,
                "selection_status": selection_status,
                **selected,
            }
        )

    # Panel (b): target-dependent ESP orders.  Each ordered small-grid search
    # explicitly includes every candidate below the selected grid, so the
    # reported point is the first measured feasible grid within the declared
    # candidate set.  PPPM is selected separately at fixed P=5 in the plotting
    # script from its complete measured scan.
    for target in sorted(TARGET_TUPLES, reverse=True):
        order = PANEL_B_ESP_ORDER_BY_TARGET[target]
        scanned = [
            row
            for row in summaries
            if float(row["target_relative_rms"]) == target
            and int(row["order"]) == order
            and int(row["requested_mesh"]) in PANEL_B_ESP_GRIDS_BY_TARGET[target]
            and "panel_b_mixed_order" in str(row["scopes"])
        ]
        expected_meshes = PANEL_B_ESP_GRIDS_BY_TARGET[target]
        if tuple(sorted(int(row["requested_mesh"]) for row in scanned)) != expected_meshes:
            raise RuntimeError(
                f"ESP target {target:.0e}, P={order} does not cover the "
                f"declared grids {expected_meshes}"
            )
        candidates = [row for row in scanned if bool(row["passes_target"])]
        if not candidates:
            raise RuntimeError(
                f"ESP P={order} does not meet target {target:.0e} "
                "in the declared small-grid search"
            )
        selected = min(candidates, key=lambda row: int(row["actual_grid_points"]))
        selection_rows.append(
            {
                "panel": "b",
                "selection_scope": (
                    "first qualifying target-specific-order ESP grid in the "
                    "declared target-specific ordered grid search"
                ),
                "selection_status": "target_met",
                **selected,
            }
        )

    write_csv(SUMMARY_CSV, summaries)
    write_csv(BY_FRAME_CSV, by_frame)
    write_csv(SELECTION_CSV, selection_rows)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Main Figure 5 SPC/E fixed-band and target-specific-order "
            "fixed-influence ik grid scans"
        ),
        "figure_contract": {
            "panel_a": (
                "Measured grid convergence at fixed csplit=cspread=14.471 for "
                "P=4--8; no split or spread retuning within a curve.  A curve "
                "that does not reach 1e-5 in the predeclared grid set is "
                "retained and marked as non-passing."
            ),
            "panel_b": (
                "First qualifying ESP grid at target-specific P=5,6,8,10 for "
                "targets 1e-3,1e-4,1e-5,1e-6, respectively, from declared "
                "target-specific ordered grid searches; PPPM remains fixed at "
                "P=5 and is selected from its independently measured scan.  "
                "Under-resolved ESP points are measured diagnostics outside the "
                "analytical screening domain."
            ),
        },
        "panel_a_grid_set": list(PANEL_A_GRIDS),
        "panel_b_esp_order_by_target": {
            f"{target:.0e}": order
            for target, order in PANEL_B_ESP_ORDER_BY_TARGET.items()
        },
        "panel_b_esp_grid_search_by_target": {
            f"{target:.0e}": list(meshes)
            for target, meshes in PANEL_B_ESP_GRIDS_BY_TARGET.items()
        },
        "lammps_executable": (
            str(lmp.relative_to(PROJECT)) if lmp.is_relative_to(PROJECT) else str(lmp)
        ),
        "lammps_executable_sha256": sha256(lmp),
        "archived_lammps_executable_sha256": EXPECTED_LMP_SHA256,
        "reference": str(REFERENCE.relative_to(PROJECT)),
        "reference_sha256": sha256(REFERENCE),
        "trajectory": str(TRAJECTORY.relative_to(PROJECT)),
        "trajectory_sha256": sha256(TRAJECTORY),
        "water_data": str(DATA.relative_to(PROJECT)),
        "water_data_sha256": sha256(DATA),
        "runner": str(Path(__file__).resolve().relative_to(PROJECT)),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "output_tables": {
            str(SUMMARY_CSV.name): sha256(SUMMARY_CSV),
            str(BY_FRAME_CSV.name): sha256(BY_FRAME_CSV),
            str(SELECTION_CSV.name): sha256(SELECTION_CSV),
        },
        "single_rank_per_case": True,
        "omp_num_threads": 1,
        "n_cases": len(CASES),
        "cases": manifest_cases,
        "error_definition": (
            "50-frame pooled relative RMS against one common Ewald reference; "
            "uncertainty is the SEM of ten nonoverlapping five-frame RMS blocks."
        ),
        "selection_boundary": (
            "Measured errors define this retrospective matched-accuracy benchmark. "
            "They do not enter the estimator-based Table-1 screening workflow."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for row in selection_rows:
        print(
            f"panel {row['panel']} target={float(row['target_relative_rms']):.0e} "
            f"P={row['order']} M={row['requested_mesh']} "
            f"error={float(row['pooled_rms_relative_error']):.9e} "
            f"status={row['selection_status']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, default=DEFAULT_LMP)
    parser.add_argument("--require-lmp-sha256")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    lmp = args.lmp.resolve()
    for path in (lmp, DATA, TRAJECTORY, REFERENCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not os.access(lmp, os.X_OK):
        raise PermissionError(f"LAMMPS executable is not executable: {lmp}")
    digest = sha256(lmp)
    if args.require_lmp_sha256 and digest != args.require_lmp_sha256.lower():
        raise RuntimeError(
            f"LAMMPS SHA-256 mismatch: expected {args.require_lmp_sha256.lower()}, "
            f"found {digest}"
        )

    if not args.analyze_only:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_case, lmp, case, args.force): case for case in CASES
            }
            for future in as_completed(futures):
                case = futures[future]
                try:
                    print(future.result(), flush=True)
                except Exception as error:
                    raise RuntimeError(f"scan failed for {case.case_id}") from error
    analyze(lmp)


if __name__ == "__main__":
    main()
