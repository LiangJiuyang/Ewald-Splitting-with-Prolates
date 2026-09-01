#!/usr/bin/env python3
"""Build operator-matched fixed-G PPPM baselines for main Figure 5.

Each panel freezes the G_ewald reported by native PPPM at the panel's
differentiation mode and calibration tolerance.  The left column is calibrated
at 1e-5 and the right column at 1e-6.
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
DEFAULT_LMP = REDESIGNED / "pppm_symmetric_scan" / "lmp.pppm_symmetric_scan"
LMP = DEFAULT_LMP
EXPECTED_LMP_SHA256 = (
    "34332fa52c4e2ba72b9561cffbc841c9b4fdbf5809eb745b1c1656e4ac960d6a"
)

ORDER = 5
RCUT = 9.0
TOTAL_FRAMES = 51
HOLDOUT_START = 25
MESHES = (12, 15, 18, 20, 24, 32, 36, 40, 48, 64, 80)
PANEL_MESHES = {
    "a": (12, 15, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80),
    "b": MESHES,
    "c": (12, 15, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80),
    "d": MESHES,
}
RAW = HERE / "raw"
SUMMARY = HERE / "fig5_pppm_ik_ad_fixed_g_summary.csv"
BY_FRAME = HERE / "fig5_pppm_ik_ad_fixed_g_by_frame.csv"
MANIFEST = HERE / "fig5_pppm_ik_ad_fixed_g_manifest.json"

sys.path.insert(0, str(REDESIGNED))
import build_fig6_pppm_order_scan as core  # noqa: E402


@dataclass(frozen=True)
class Branch:
    panel: str
    differentiation: str
    target: float
    calibration_tolerance: float
    expected_g: float
    expected_auto_grid: int
    @property
    def tag(self) -> str:
        return f"{self.differentiation}_{self.target:.0e}".replace("e-0", "e-")


BRANCHES = (
    Branch("a", "ik", 1.0e-4, 1.0e-5, 0.34166005, 30),
    Branch("b", "ik", 1.0e-5, 1.0e-6, 0.37759658, 48),
    Branch("c", "ad", 1.0e-4, 1.0e-5, 0.33593464, 36),
    Branch("d", "ad", 1.0e-5, 1.0e-6, 0.37738337, 72),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relpath(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT).as_posix()
    except ValueError:
        return resolved.as_posix()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": relpath(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def case_id(branch: Branch, mesh: int) -> str:
    return f"fig5_pppm_{branch.tag}_p{ORDER}_m{mesh}"


def case_paths(branch: Branch, mesh: int) -> tuple[dict[str, Path], str]:
    directory = RAW / case_id(branch, mesh)
    return (
        {
            "input": directory / "in.lammps",
            "dump": directory / "forces.dump",
            "log": directory / "log.lammps",
        },
        "dedicated operator-matched Figure-5 PPPM record",
    )


def calibration_paths(branch: Branch) -> dict[str, Path]:
    directory = RAW / f"calibration_{branch.tag}"
    return {
        "input": directory / "in.lammps",
        "log": directory / "log.lammps",
    }


def molecular_preamble() -> str:
    return """bond_style harmonic
angle_style harmonic
bond_coeff 1 0.0 1.000
angle_coeff 1 0.0 109.47
special_bonds lj/coul 0 0 0.5

"""


def diff_command(branch: Branch) -> str:
    return f"kspace_modify diff {branch.differentiation}\n"


def calibration_input(branch: Branch) -> str:
    return f"""# Native PPPM-{branch.differentiation.upper()} calibration for Figure 5.
newton on
units real
atom_style full
read_data {relpath(DATA)}
reset_timestep 0

{molecular_preamble()}neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes one 2000
thermo_style custom step pe ecoul elong press
thermo_modify flush yes
thermo 1
pair_style coul/long {RCUT:.1f}
kspace_style pppm {branch.calibration_tolerance:.1e}
kspace_modify order {ORDER}
{diff_command(branch)}pair_coeff * *

run 0
"""


def case_input(branch: Branch, mesh: int, dump: Path) -> str:
    return f"""# Fixed-G PPPM-{branch.differentiation.upper()} scan for Figure 5.
newton on
units real
atom_style full
read_data {relpath(DATA)}
reset_timestep 0

{molecular_preamble()}neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes one 2000
thermo_style custom step pe ecoul elong press
thermo_modify flush yes
thermo 100
pair_style coul/long {RCUT:.1f}
kspace_style pppm {branch.calibration_tolerance:.1e}
kspace_modify order {ORDER} mesh {mesh} {mesh} {mesh}
{diff_command(branch)}kspace_modify gewald {branch.expected_g:.8g}
pair_coeff * *

dump force all custom 1 {relpath(dump)} id fx fy fz
dump_modify force sort id format line "%d %.17g %.17g %.17g"
rerun {relpath(TRAJECTORY)} dump x y z ix iy iz box yes format native
"""


def run_lammps(input_path: Path, log_path: Path) -> None:
    command = [
        str(LMP),
        "-in",
        str(input_path),
        "-log",
        str(log_path),
        "-screen",
        "none",
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command,
        cwd=PROJECT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"LAMMPS exited with {completed.returncode} for {input_path}:\n"
            f"{completed.stdout}"
        )


def calibration_metadata(branch: Branch, log: Path) -> dict[str, object]:
    text = log.read_text(encoding="utf-8", errors="replace")
    grids = re.findall(r"^\s*grid\s*=\s*(\d+)\s+(\d+)\s+(\d+)\s*$", text, re.MULTILINE)
    orders = re.findall(r"^\s*stencil order\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    g_values = re.findall(r"G vector \(1/distance\)\s*=\s*([0-9.eE+-]+)", text)
    estimates = re.findall(
        r"estimated relative force accuracy\s*=\s*([0-9.eE+-]+)", text
    )
    if "ERROR:" in text or not all((grids, orders, g_values, estimates)):
        raise RuntimeError(f"incomplete calibration log: {log}")
    metadata = {
        "actual_grid": tuple(int(value) for value in grids[-1]),
        "actual_order": int(orders[-1]),
        "g_ewald_inverse_A": float(g_values[-1]),
        "estimated_relative_force_accuracy": float(estimates[-1]),
    }
    if metadata["actual_grid"] != (branch.expected_auto_grid,) * 3:
        raise RuntimeError(f"{branch.panel}: unexpected calibration grid")
    if metadata["actual_order"] != ORDER:
        raise RuntimeError(f"{branch.panel}: unexpected calibration order")
    if not math.isclose(
        float(metadata["g_ewald_inverse_A"]),
        branch.expected_g,
        rel_tol=0.0,
        abs_tol=5.0e-9,
    ):
        raise RuntimeError(f"{branch.panel}: unexpected calibrated G_ewald")
    return metadata


def run_calibration(branch: Branch, force: bool) -> dict[str, object]:
    paths = calibration_paths(branch)
    paths["input"].parent.mkdir(parents=True, exist_ok=True)
    paths["input"].write_text(calibration_input(branch), encoding="utf-8")
    if force or not paths["log"].is_file():
        if force and paths["log"].is_file():
            paths["log"].unlink()
        run_lammps(paths["input"], paths["log"])
    return calibration_metadata(branch, paths["log"])


def case_complete(branch: Branch, mesh: int, paths: dict[str, Path]) -> bool:
    if any(not paths[key].is_file() for key in ("input", "dump", "log")):
        return False
    try:
        parsed = core.parse_log(paths["log"])
        input_text = paths["input"].read_text(encoding="utf-8")
        frames = core.read_force_dump(paths["dump"])
    except (OSError, TypeError, ValueError):
        return False
    uses_ad = bool(re.search(r"^\s*kspace_modify\s+diff\s+ad\s*$", input_text, re.MULTILINE))
    return (
        parsed["actual_grid"] == (mesh,) * 3
        and parsed["actual_order"] == ORDER
        and parsed["loop_steps"] == TOTAL_FRAMES
        and math.isclose(
            float(parsed["g_ewald_inverse_A"]),
            branch.expected_g,
            rel_tol=0.0,
            abs_tol=5.0e-9,
        )
        and uses_ad == (branch.differentiation == "ad")
        and len(frames) == TOTAL_FRAMES
    )


def run_case(branch: Branch, mesh: int, force: bool) -> str:
    paths, role = case_paths(branch, mesh)
    paths["input"].parent.mkdir(parents=True, exist_ok=True)
    paths["input"].write_text(case_input(branch, mesh, paths["dump"]), encoding="utf-8")
    if case_complete(branch, mesh, paths) and not force:
        return f"{case_id(branch, mesh)}: cached"
    existing = [path for key, path in paths.items() if key != "input" and path.exists()]
    if existing and not force:
        raise RuntimeError(f"{case_id(branch, mesh)}: incomplete output; use --force")
    if force:
        for path in existing:
            path.unlink()
    run_lammps(paths["input"], paths["log"])
    if not case_complete(branch, mesh, paths):
        raise RuntimeError(f"{case_id(branch, mesh)}: output audit failed")
    return f"{case_id(branch, mesh)}: completed ({role})"


def pooled_error(rows: list[dict[str, object]]) -> float:
    return math.sqrt(
        math.fsum(float(row["sum_diff_squared"]) for row in rows)
        / math.fsum(float(row["sum_reference_squared"]) for row in rows)
    )


def balanced_block_sem(rows: list[dict[str, object]]) -> float:
    blocks = (rows[0:5], rows[5:10], rows[10:15], rows[15:20], rows[20:26])
    values = [pooled_error(block) for block in blocks]
    return statistics.stdev(values) / math.sqrt(len(values))


def measure_case(
    branch: Branch,
    mesh: int,
    reference: list[tuple[int, dict[int, tuple[float, float, float]]]],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    paths, role = case_paths(branch, mesh)
    test = core.read_force_dump(paths["dump"])
    parsed = core.parse_log(paths["log"])
    frame_rows: list[dict[str, object]] = []
    for frame_index, ((ref_step, ref_atoms), (test_step, test_atoms)) in enumerate(
        zip(reference, test)
    ):
        if ref_step != test_step or ref_atoms.keys() != test_atoms.keys():
            raise RuntimeError(f"{case_id(branch, mesh)}: reference alignment failed")
        diff2 = 0.0
        ref2 = 0.0
        for atom_id, ref_force in ref_atoms.items():
            test_force = test_atoms[atom_id]
            for ref_component, test_component in zip(ref_force, test_force):
                delta = test_component - ref_component
                diff2 += delta * delta
                ref2 += ref_component * ref_component
        if frame_index >= HOLDOUT_START:
            frame_rows.append(
                {
                    "case_id": case_id(branch, mesh),
                    "panel": branch.panel,
                    "differentiation": branch.differentiation,
                    "target_relative_rms": branch.target,
                    "order": ORDER,
                    "actual_nx": mesh,
                    "frame_index_zero_based": frame_index,
                    "timestep": ref_step,
                    "natoms": len(ref_atoms),
                    "relative_rms": math.sqrt(diff2 / ref2),
                    "sum_diff_squared": diff2,
                    "sum_reference_squared": ref2,
                }
            )
    if len(frame_rows) != 26:
        raise RuntimeError(f"{case_id(branch, mesh)}: expected 26 holdout frames")
    holdout_error = pooled_error(frame_rows)
    summary = {
        "case_id": case_id(branch, mesh),
        "panel": branch.panel,
        "method": "PPPM",
        "differentiation": branch.differentiation,
        "operator": (
            "Hockney-Eastwood optimal-influence IK"
            if branch.differentiation == "ik"
            else "LAMMPS PPPM analytical differentiation"
        ),
        "target_relative_rms": branch.target,
        "order": ORDER,
        "requested_mesh": mesh,
        "actual_nx": mesh,
        "actual_ny": mesh,
        "actual_nz": mesh,
        "actual_grid_points": mesh**3,
        "calibration_input_tolerance": branch.calibration_tolerance,
        "calibration_actual_nx": branch.expected_auto_grid,
        "fixed_gewald_inverse_A": branch.expected_g,
        "total_frames": TOTAL_FRAMES,
        "holdout_frames": len(frame_rows),
        "holdout_relative_rms": holdout_error,
        "holdout_balanced_block5_sem": balanced_block_sem(frame_rows),
        "holdout_passes_target": holdout_error <= branch.target,
        "loop_time_s_single_rank": parsed["loop_time_s"],
        "force_dump_sha256": sha256(paths["dump"]),
        "input_sha256": sha256(paths["input"]),
        "log_sha256": sha256(paths["log"]),
        "source_role": role,
    }
    audit = {
        "case_id": case_id(branch, mesh),
        "status": "accepted",
        "actual_grid": [mesh, mesh, mesh],
        "actual_order": parsed["actual_order"],
        "g_ewald_inverse_A": parsed["g_ewald_inverse_A"],
        "input": file_record(paths["input"]),
        "force_dump": file_record(paths["dump"]),
        "log": file_record(paths["log"]),
    }
    return summary, frame_rows, audit


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def analyze(calibrations: dict[str, dict[str, object]]) -> None:
    reference = core.read_force_dump(REFERENCE)
    if len(reference) != TOTAL_FRAMES:
        raise RuntimeError("Ewald reference does not contain 51 frames")
    summaries: list[dict[str, object]] = []
    by_frame: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for branch in BRANCHES:
        for mesh in PANEL_MESHES[branch.panel]:
            summary, frame_rows, audit = measure_case(branch, mesh, reference)
            summaries.append(summary)
            by_frame.extend(frame_rows)
            audits.append(audit)
    summaries.sort(key=lambda row: (str(row["panel"]), int(row["actual_nx"])))
    by_frame.sort(
        key=lambda row: (
            str(row["panel"]),
            int(row["actual_nx"]),
            int(row["frame_index_zero_based"]),
        )
    )
    write_csv(SUMMARY, summaries)
    write_csv(BY_FRAME, by_frame)

    calibration_records = {}
    for branch in BRANCHES:
        paths = calibration_paths(branch)
        calibration_records[branch.panel] = {
            "differentiation": branch.differentiation,
            "input_tolerance": branch.calibration_tolerance,
            "expected_auto_grid": [branch.expected_auto_grid] * 3,
            "fixed_gewald_inverse_A": branch.expected_g,
            "metadata": calibrations[branch.panel],
            "input": file_record(paths["input"]),
            "log": file_record(paths["log"]),
        }
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runner": file_record(Path(__file__)),
        "design": (
            "operator-matched PPPM P=5 baselines; panels a/c calibrate at "
            "1e-5 and panels b/d at 1e-6, then freeze the reported G_ewald"
        ),
        "meshes_by_panel": {
            panel: list(meshes) for panel, meshes in PANEL_MESHES.items()
        },
        "holdout_frames_zero_based": [25, 50],
        "single_rank": True,
        "omp_num_threads": 1,
        "lammps_executable": file_record(LMP),
        "archived_lammps_executable_sha256": EXPECTED_LMP_SHA256,
        "water_data": file_record(DATA),
        "trajectory": file_record(TRAJECTORY),
        "ewald_reference": file_record(REFERENCE),
        "calibrations": calibration_records,
        "case_audits": audits,
        "outputs": {
            "summary": file_record(SUMMARY),
            "by_frame": file_record(BY_FRAME),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY.name}, {BY_FRAME.name}, and {MANIFEST.name}")


def main() -> None:
    global LMP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmp",
        type=Path,
        default=Path(os.environ.get("ESP_LAMMPS_BIN", DEFAULT_LMP)),
    )
    parser.add_argument("--require-lmp-sha256")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    LMP = args.lmp.resolve()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for path in (DATA, TRAJECTORY, REFERENCE, LMP):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not os.access(LMP, os.X_OK):
        raise PermissionError(f"LAMMPS executable is not executable: {LMP}")
    digest = sha256(LMP)
    if args.require_lmp_sha256 and digest != args.require_lmp_sha256.lower():
        raise RuntimeError(
            f"LAMMPS SHA-256 mismatch: expected {args.require_lmp_sha256.lower()}, "
            f"found {digest}"
        )

    calibrations = {
        branch.panel: run_calibration(branch, args.force) for branch in BRANCHES
    }
    if not args.analyze_only:
        work = [
            (branch, mesh)
            for branch in BRANCHES
            for mesh in PANEL_MESHES[branch.panel]
        ]
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_case, branch, mesh, args.force): (branch, mesh)
                for branch, mesh in work
            }
            for future in as_completed(futures):
                print(future.result(), flush=True)
    analyze(calibrations)


if __name__ == "__main__":
    main()
