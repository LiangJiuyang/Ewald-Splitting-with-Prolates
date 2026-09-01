#!/usr/bin/env python3
"""Extend the AD PPPM fixed-order convergence scan above M=96.

This deliberately narrow, reproducible follow-up to the Figure 6(d) scan
retains the same 21,624-atom SPC/E water configurations, Ewald reference,
cutoff, 1e-5 split tolerance, AD differentiation, and four-frame acceptance
rule.  The default reproduces the P=5 extension; --order can test a lower
stencil order before a fixed-order point is promoted to the minimum-order
Figure 6 source table.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import run_large_water_window_scan as scan


HERE = Path(__file__).resolve().parent
TARGET = scan.Target(
    label="1e-5",
    force_target=1.0e-5,
    split_tolerance=1.0e-5,
    csplit=14.471,
    meshes=(),
    orders=(5,),
)
# These have only small prime factors and bracket the P=5 threshold suggested
# by the directly measured M=96 value.  All requested sizes are retained in
# the output together with the actual LAMMPS grid.
DEFAULT_MESHES = (108, 112, 120, 128, 135, 144)
FRAMES = scan.VALIDATION_FRAMES


def key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["requested_mesh"]), int(row["frame"])


def read_rows(output: Path) -> list[dict[str, str]]:
    if not output.exists():
        return []
    with output.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(output: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "stage", "differentiation", "target", "target_force_error",
        "split_tolerance", "window", "frame", "requested_mesh",
        "actual_nx", "actual_ny", "actual_nz", "actual_grid_volume",
        "order", "c_split", "c_spread", "gamma", "sigma_up", "natoms",
        "rms_relative_force_error", "rms_abs_force_error", "rms_ref_force",
        "max_atom_relative_force_error",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--meshes", type=int, nargs="+", default=DEFAULT_MESHES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.order < 2 or args.order > 7:
        raise ValueError("native LAMMPS PPPM supports B-spline orders 2--7")
    meshes = tuple(args.meshes)
    target = scan.Target(
        label=TARGET.label,
        force_target=TARGET.force_target,
        split_tolerance=TARGET.split_tolerance,
        csplit=TARGET.csplit,
        meshes=(),
        orders=(args.order,),
    )
    output = HERE / "ad" / f"pppm_p{args.order}_extension.csv"

    scan.DIFFERENTIATION = "ad"
    scan.configure_output_root(HERE / "ad")
    scan.ensure_inputs()
    references = scan.ensure_references(args.lmp, force=False)
    existing = {key(row): dict(row) for row in read_rows(output)}
    for mesh in meshes:
        for frame in FRAMES:
            candidate_key = (mesh, frame)
            if candidate_key in existing and not args.force:
                continue
            row = scan.evaluate_candidate(
                lmp=args.lmp,
                target=target,
                window="pppm",
                mesh=mesh,
                order=args.order,
                gamma=None,
                frame=frame,
                stage=f"pppm_p{args.order}_extension",
                reference=references[frame],
                force=args.force,
            )
            existing[candidate_key] = row
            write_rows(output, [existing[item] for item in sorted(existing)])

    rows = [existing[item] for item in sorted(existing)]
    for mesh in meshes:
        errors = [
            float(row["rms_relative_force_error"])
            for row in rows
            if int(row["requested_mesh"]) == mesh
        ]
        if len(errors) != len(FRAMES):
            raise RuntimeError(f"M={mesh} has {len(errors)} validation frames")
        status = "feasible" if max(errors) <= target.force_target else "above target"
        actual = next(
            int(row["actual_nx"])
            for row in rows
            if int(row["requested_mesh"]) == mesh
        )
        print(
            f"requested M={mesh}, actual M={actual}, P={args.order}, "
            f"max relative RMS error={max(errors):.6e}: {status}"
        )


if __name__ == "__main__":
    main()
