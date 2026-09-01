#!/usr/bin/env python3
"""Run the predeclared same-binary PPPM order/grid scan used in Figure 6.

The scan brackets each of the three manuscript targets for every native LAMMPS
PPPM order P=4--7.  All candidates use one hash-pinned executable, the same
51-frame SPC/E trajectory, and the same tight Ewald reference.  This script
only generates and audits raw runs; the Figure-6 builder independently
recomputes every force error from the raw dumps.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE_DIR = REPO / "numerical_examples" / "water_trajectory_benchmark"
DATA = SOURCE_DIR / "water.data"
TRAJECTORY = SOURCE_DIR / "water_short_traj.lammpstrj"
REFERENCE = SOURCE_DIR / "forces.ref_ewald.dump"
DEFAULT_LMP = HERE / "lmp.pppm_symmetric_scan"
EXPECTED_LMP_SHA256 = "34332fa52c4e2ba72b9561cffbc841c9b4fdbf5809eb745b1c1656e4ac960d6a"
EXPECTED_FRAMES = 51

# Predeclared Cartesian design.  The common requested-grid set covers the
# low-grid P=7 regime and brackets all three targets even for P=4.  The actual
# grids are audited after execution rather than assumed to equal these values.
COMMON_REQUESTED_MESHES = (12, 15, 18, 20, 24, 32, 36, 40, 48, 64, 80)
DEFAULT_CASES: dict[int, tuple[int, ...]] = {
    order: COMMON_REQUESTED_MESHES for order in range(4, 8)
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lammps_quote(path: Path) -> str:
    relative = path.resolve().relative_to(REPO).as_posix()
    return '"' + relative.replace('"', '\\"') + '"'


def paths_for(order: int, mesh: int) -> dict[str, Path]:
    stem = f"pppm_p{order}_mesh{mesh}"
    return {
        "input": HERE / f"in.{stem}",
        "dump": HERE / f"forces.{stem}.dump",
        "log": HERE / f"log.{stem}.lammps",
        "screen": HERE / f"screen.{stem}.txt",
        "launcher": HERE / f"launcher.{stem}.txt",
    }


def input_text(order: int, mesh: int, dump_path: Path) -> str:
    return f"""# Same-binary SPC/E-water PPPM scan: P={order}, requested M={mesh}.
# Predeclared target-bracketing Figure-6 design; 51 shared trajectory frames.
newton on
units real
atom_style full
read_data {lammps_quote(DATA)}
reset_timestep 0

bond_style harmonic
angle_style harmonic
bond_coeff 1 0.0 1.000
angle_coeff 1 0.0 109.47
special_bonds lj/coul 0 0 0.5

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes one 2000
thermo_style custom step pe ecoul elong press
thermo_modify flush yes
thermo 100
pair_style coul/long 9.0
kspace_style pppm 1.0e-4
kspace_modify order {order} mesh {mesh} {mesh} {mesh}
pair_coeff * *

dump f all custom 1 {lammps_quote(dump_path)} id fx fy fz
dump_modify f sort id format line "%d %.17g %.17g %.17g"
rerun {lammps_quote(TRAJECTORY)} dump x y z ix iy iz box yes format native
"""


def count_dump_frames(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.rstrip("\n") == "ITEM: TIMESTEP":
                count += 1
    return count


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "ERROR:" in text:
        raise RuntimeError(f"LAMMPS reported an error in {path.name}")
    versions = re.findall(r"^LAMMPS \((.+)\)\s*$", text, re.MULTILINE)
    grids = re.findall(
        r"^\s*grid\s*=\s*(\d+)\s+(\d+)\s+(\d+)\s*$", text, re.MULTILINE
    )
    orders = re.findall(r"^\s*stencil order\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    loops = re.findall(
        r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps",
        text,
    )
    fft = re.findall(r"^\s*using double precision (.+?)\s*$", text, re.MULTILINE)
    if not versions or not grids or not orders or not loops or not fft:
        raise ValueError(f"incomplete LAMMPS metadata in {path.name}")
    loop_time, ranks, steps = loops[-1]
    return {
        "lammps_version": versions[-1],
        "actual_grid": [int(value) for value in grids[-1]],
        "actual_order": int(orders[-1]),
        "loop_time_s": float(loop_time),
        "mpi_ranks": int(ranks),
        "steps": int(steps),
        "fft_backend": fft[-1].strip(),
    }


def complete(order: int, paths: dict[str, Path]) -> bool:
    if not paths["dump"].is_file() or not paths["log"].is_file():
        return False
    try:
        metadata = parse_log(paths["log"])
        return (
            metadata["actual_order"] == order
            and metadata["steps"] == EXPECTED_FRAMES
            and count_dump_frames(paths["dump"]) == EXPECTED_FRAMES
        )
    except (OSError, ValueError, RuntimeError):
        return False


def run_case(lmp: Path, order: int, mesh: int, force: bool) -> None:
    paths = paths_for(order, mesh)
    paths["input"].write_text(
        input_text(order, mesh, paths["dump"]), encoding="utf-8"
    )
    if complete(order, paths) and not force:
        print(f"P={order}, M={mesh}: complete output exists; keeping it", flush=True)
        return
    existing = [
        path for key, path in paths.items() if key != "input" and path.exists()
    ]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise RuntimeError(
            f"P={order}, M={mesh}: incomplete outputs ({names}); use --force"
        )
    if force:
        for key, path in paths.items():
            if key != "input" and path.is_file():
                path.unlink()
    command = [
        str(lmp),
        "-in",
        str(paths["input"]),
        "-log",
        str(paths["log"]),
        "-screen",
        str(paths["screen"]),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    print(f"P={order}, M={mesh}: running", flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    paths["launcher"].write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"P={order}, M={mesh}: LAMMPS exited with {completed.returncode}"
        )
    if not complete(order, paths):
        raise RuntimeError(f"P={order}, M={mesh}: output audit failed")


def parse_case(text: str) -> tuple[int, tuple[int, ...]]:
    try:
        order_text, mesh_text = text.split(":", 1)
        order = int(order_text)
        meshes = tuple(int(value) for value in mesh_text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must look like 7:12,18,24") from error
    if order not in range(4, 8) or not meshes or any(mesh <= 0 for mesh in meshes):
        raise argparse.ArgumentTypeError("orders must be 4--7 and meshes positive")
    if len(set(meshes)) != len(meshes):
        raise argparse.ArgumentTypeError("meshes within a case must be distinct")
    return order, meshes


def write_manifest(lmp: Path, cases: dict[int, tuple[int, ...]]) -> None:
    runs: list[dict[str, object]] = []
    versions: set[str] = set()
    fft_backends: set[str] = set()
    for order in sorted(cases):
        for mesh in cases[order]:
            paths = paths_for(order, mesh)
            if not complete(order, paths):
                raise RuntimeError(f"P={order}, M={mesh}: incomplete during manifest audit")
            metadata = parse_log(paths["log"])
            versions.add(str(metadata["lammps_version"]))
            fft_backends.add(str(metadata["fft_backend"]))
            runs.append(
                {
                    "order": order,
                    "requested_mesh": mesh,
                    **metadata,
                    "input": paths["input"].name,
                    "input_sha256": sha256(paths["input"]),
                    "force_dump": paths["dump"].name,
                    "force_dump_sha256": sha256(paths["dump"]),
                    "log": paths["log"].name,
                    "log_sha256": sha256(paths["log"]),
                    "screen": paths["screen"].name,
                    "screen_sha256": sha256(paths["screen"]),
                    "force_dump_frames": count_dump_frames(paths["dump"]),
                }
            )
    if len(versions) != 1 or len(fft_backends) != 1:
        raise RuntimeError(
            f"scan is not implementation-uniform: versions={versions}, FFT={fft_backends}"
        )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runner": Path(__file__).name,
        "runner_sha256": sha256(Path(__file__)),
        "design": "predeclared Cartesian scan over P=4--7 and one common requested-grid set",
        "common_requested_meshes": list(COMMON_REQUESTED_MESHES),
        "default_case_matrix": {
            str(order): list(meshes) for order, meshes in DEFAULT_CASES.items()
        },
        "executed_case_matrix": {
            str(order): list(meshes) for order, meshes in cases.items()
        },
        "lammps_executable": (
            str(lmp.relative_to(REPO)) if lmp.is_relative_to(REPO) else str(lmp)
        ),
        "lammps_executable_sha256": sha256(lmp),
        "expected_lammps_executable_sha256": EXPECTED_LMP_SHA256,
        "lammps_versions": sorted(versions),
        "fft_backends": sorted(fft_backends),
        "single_rank": True,
        "omp_num_threads": 1,
        "water_data": str(DATA.relative_to(REPO)),
        "water_data_sha256": sha256(DATA),
        "trajectory": str(TRAJECTORY.relative_to(REPO)),
        "trajectory_sha256": sha256(TRAJECTORY),
        "reference": str(REFERENCE.relative_to(REPO)),
        "reference_sha256": sha256(REFERENCE),
        "reference_not_used_for_run_acceptance": True,
        "runs": runs,
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, default=DEFAULT_LMP)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="order:comma-separated-meshes; repeat for multiple orders",
    )
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    lmp = args.lmp.resolve()
    for path in (DATA, TRAJECTORY, REFERENCE, lmp):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not os.access(lmp, os.X_OK):
        raise PermissionError(f"LAMMPS executable is not executable: {lmp}")
    digest = sha256(lmp)
    if digest != EXPECTED_LMP_SHA256:
        raise RuntimeError(
            f"LAMMPS SHA-256 mismatch: expected {EXPECTED_LMP_SHA256}, found {digest}"
        )

    if args.case:
        cases: dict[int, tuple[int, ...]] = {}
        for order, meshes in args.case:
            if order in cases:
                raise ValueError(f"order {order} specified more than once")
            cases[order] = meshes
    else:
        cases = dict(DEFAULT_CASES)

    if not args.analyze_only:
        for order in sorted(cases):
            for mesh in cases[order]:
                run_case(lmp, order, mesh, args.force)
    write_manifest(lmp, cases)
    print(
        f"audited {sum(len(meshes) for meshes in cases.values())} runs; "
        f"orders={sorted(cases)}; manifest.json written",
        flush=True,
    )


if __name__ == "__main__":
    main()
