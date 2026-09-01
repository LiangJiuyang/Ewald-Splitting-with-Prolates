#!/usr/bin/env python3
"""All-atom SPC/E water grid--window scan used to replace Figure 6.

The calculation intentionally separates a small *selection* step from the
reported validation.  Frame 1000 selects ``c_spread`` for every PSWF
``(target, M, P)`` candidate.  Frames 1250--2000 are then evaluated without
retuning.  A candidate is considered feasible only when every validation
frame satisfies the requested total relative RMS force-error target.

Within a sweep, all ESP variants retain the same PSWF real-/reciprocal-space
split, cutoff, coordinates, and selected differentiation convention.  The
default ``ik`` sweep uses the Hockney--Eastwood optimal influence function;
the ``ad`` sweep uses its matched AD Green function and self-force correction.
``esp/bspline`` differs from ``esp`` only in its spreading/interpolation
window.  ``pppm`` is a separately labelled Gaussian-split B-spline reference.

The script requires a LAMMPS executable containing the benchmark-only
``esp/bspline`` implementation.  It never treats a requested mesh as an
observed result: the actual mesh emitted by LAMMPS and its volume are recorded
in every CSV row.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BASE_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = BASE_ROOT.parents[1]
ROOT = BASE_ROOT
SOURCE_WATER = BASE_ROOT.parent / "water_trajectory_benchmark" / "water.data"
TRAJECTORY = BASE_ROOT / "large_water_short_traj.lammpstrj"
RUNS = ROOT / "runs"
REFERENCE = ROOT / "reference"
SELECTION_CSV = ROOT / "selection_scan.csv"
VALIDATION_CSV = ROOT / "validation_by_frame.csv"
SUMMARY_CSV = ROOT / "validation_summary.csv"

CUTOFF = 9.0
BOX = 60.0
SELECTION_FRAME = 1000
VALIDATION_FRAMES = (1250, 1500, 1750, 2000)
POLY_TOLERANCE = 1.0e-10
# Kept as a module-level setting so an independently namespaced AD scan can
# reuse the exact same target, window, grid, and frozen-frame protocol.
DIFFERENTIATION = "ik"


def bundle_relative(path: Path) -> str:
    """Use a bundle-relative path when possible, otherwise an absolute path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(BUNDLE_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def configure_output_root(output_root: Path | None) -> None:
    """Namespace a sweep without changing the shared input trajectory.

    The AD scan must not reuse an IK CSV or candidate cache: the two force
    operators have different error values and separate selection records.  A
    relative output name is deliberately resolved below this calculation's
    directory, so ``--output-root ad`` yields a self-contained sibling data
    set while still using the same coordinate and Ewald-reference files.
    """

    global ROOT, RUNS, REFERENCE, SELECTION_CSV, VALIDATION_CSV, SUMMARY_CSV
    ROOT = BASE_ROOT if output_root is None else (
        output_root.resolve() if output_root.is_absolute() else BASE_ROOT / output_root
    )
    RUNS = ROOT / "runs"
    REFERENCE = BASE_ROOT / "reference"
    SELECTION_CSV = ROOT / "selection_scan.csv"
    VALIDATION_CSV = ROOT / "validation_by_frame.csv"
    SUMMARY_CSV = ROOT / "validation_summary.csv"


@dataclass(frozen=True)
class Target:
    label: str
    force_target: float
    split_tolerance: float
    csplit: float
    meshes: tuple[int, ...]
    orders: tuple[int, ...]


TARGETS = (
    Target(
        label="1e-4",
        force_target=1.0e-4,
        split_tolerance=1.0e-4,
        csplit=12.024,
        meshes=(27, 30, 32, 36, 40, 45, 48, 54, 60, 64, 72, 80),
        orders=(3, 4, 5, 6, 7, 8),
    ),
    Target(
        label="1e-5",
        force_target=1.0e-5,
        split_tolerance=1.0e-5,
        csplit=14.471,
        meshes=(27, 30, 32, 36, 40, 45, 48, 54, 60, 64, 72, 80, 96),
        orders=(4, 5, 6, 7, 8, 9, 10),
    ),
)
WINDOWS = ("pswf", "bspline", "pppm")


def window_meshes(target: Target, window: str) -> tuple[int, ...]:
    """Return the target/window-specific FFT-friendly grid scan."""

    if window == "pppm" and target.label == "1e-4":
        # The two rightmost Fig. 6(a) grids are now scanned for every branch.
        return (27, 30, 36, 40, 45, 48, 54, 60, 64, 72, 80)
    if window == "pppm" and target.label == "1e-5":
        # The initial 32--54 scan establishes that the conventional Gaussian
        # reference remains above 1e-5 at M=54 and P=7.  The PPPM-only
        # 60, 64, and 72 grids locate its feasible onset; the two rightmost
        # Fig. 6(b) grids (80 and 96) are now scanned for every branch.
        return (27, 32, 36, 40, 45, 48, 54, 60, 64, 72, 80, 96)
    return target.meshes


def window_orders(target: Target, window: str) -> tuple[int, ...]:
    """Return only stencil orders implemented by the selected solver path."""

    if window == "bspline" and target.label == "1e-5":
        # M=32 remains just above the target at P=10.  Extend the ESP B-spline
        # sweep only within the already implemented ESP order range before
        # deciding that the coarsest resolved grid is infeasible.
        return (*target.orders, 11, 12)
    if window != "pppm":
        return target.orders
    # Native LAMMPS PPPM explicitly supports B-spline orders 2--7.  ESP has a
    # wider implementation range, so applying its P=8--10 candidates to the
    # Gaussian PPPM reference would be an invalid comparison rather than a
    # failed physical parameter choice.
    return tuple(order for order in target.orders if 2 <= order <= 7)


def gamma_candidates(target: Target, mesh: int, order: int) -> tuple[float, ...]:
    """A deliberately compact, symmetric PSWF-window selection sweep.

    ``gamma = 2 c_spread/(pi P)``.  The common values cover the transition
    between compact real-space stencils and Fourier concentration; the
    target-specific default generated by ``c_spread=c_split`` is included as
    a control.  The selected value is frozen before validation.
    """

    sigma_up = math.pi * CUTOFF * mesh / (target.csplit * BOX)
    baseline = 2.0 * target.csplit / (math.pi * order)
    adaptive = (0.90, 1.10, 1.30, 1.50, 1.70)
    # Near the resolved-band edge, include a guard-band value implied by the
    # actual sigma_up.  It is a candidate, not an assumed optimum.
    edge = max(0.80, 1.0 / sigma_up)
    return tuple(sorted({round(value, 8) for value in (*adaptive, edge, baseline)}))


def ensure_inputs() -> None:
    missing = [str(path) for path in (SOURCE_WATER, TRAJECTORY) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing large-water input(s): " + ", ".join(missing) + ". "
            "Run in.generate_large_water_trajectory first."
        )
    RUNS.mkdir(parents=True, exist_ok=True)
    REFERENCE.mkdir(parents=True, exist_ok=True)


def read_dump(path: Path) -> dict[int, tuple[float, float, float]]:
    """Read a single-frame LAMMPS custom force dump."""

    rows: dict[int, tuple[float, float, float]] = {}
    in_atoms = False
    for line in path.read_text().splitlines():
        if line.startswith("ITEM: ATOMS"):
            in_atoms = True
            continue
        if line.startswith("ITEM:"):
            in_atoms = False
            continue
        if in_atoms:
            parts = line.split()
            rows[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
    if not rows:
        raise RuntimeError(f"No force rows were read from {path}")
    return rows


def has_readable_force_dump(path: Path) -> bool:
    """Return whether a cached force dump is complete enough to reuse.

    A cancelled LAMMPS run can leave an empty dump beside a log file.  Treat
    that state as a cache miss so a restart reruns the calculation instead of
    attempting to parse incomplete force data.
    """

    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return bool(read_dump(path))
    except RuntimeError:
        return False


def compare_forces(
    reference: dict[int, tuple[float, float, float]],
    candidate: dict[int, tuple[float, float, float]],
) -> dict[str, float]:
    ids = sorted(reference)
    if ids != sorted(candidate):
        raise RuntimeError("Reference and candidate force dumps have different atom IDs")
    sum_diff2 = 0.0
    sum_ref2 = 0.0
    max_atom_relative = 0.0
    for atom_id in ids:
        ref = reference[atom_id]
        test = candidate[atom_id]
        diff2 = sum((test[index] - ref[index]) ** 2 for index in range(3))
        ref2 = sum(value * value for value in ref)
        sum_diff2 += diff2
        sum_ref2 += ref2
        if ref2 > 0.0:
            max_atom_relative = max(max_atom_relative, math.sqrt(diff2 / ref2))
    return {
        "natoms": float(len(ids)),
        "rms_relative_force_error": math.sqrt(sum_diff2 / sum_ref2),
        "rms_abs_force_error": math.sqrt(sum_diff2 / len(ids)),
        "rms_ref_force": math.sqrt(sum_ref2 / len(ids)),
        "max_atom_relative_force_error": max_atom_relative,
    }


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    patterns = {
        "grid": r"\bgrid\s*=\s*(\d+)\s+(\d+)\s+(\d+)",
        "order": r"\bstencil order\s*=\s*(\d+)",
        "csplit": r"\bSplitting parameter c\s*=\s*([0-9.eE+-]+)",
        "cspread": r"\bSpreading parameter c\s*=\s*([0-9.eE+-]+)",
    }
    parsed: dict[str, Any] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if name in {"csplit", "cspread"} and not match:
            parsed[name] = ""
            continue
        if not match:
            raise RuntimeError(f"Missing {name} in {path}")
        if name == "grid":
            parsed[name] = tuple(int(value) for value in match.groups())
        elif name == "order":
            parsed[name] = int(match.group(1))
        else:
            parsed[name] = float(match.group(1))
    return parsed


def pair_block(kind: str, target: Target | None = None) -> str:
    if kind == "ewald":
        return """\
pair_style lj/cut/coul/long 9.0
pair_coeff 1 1 0.15535 3.166
pair_coeff 1 2 0.0 1.0
pair_coeff 2 2 0.0 1.0
pair_modify table 0
kspace_style ewald 1.0e-12
"""
    assert target is not None
    if kind == "pppm":
        return f"""\
pair_style lj/cut/coul/long 9.0
pair_coeff 1 1 0.15535 3.166
pair_coeff 1 2 0.0 1.0
pair_coeff 2 2 0.0 1.0
pair_modify table 0
kspace_style pppm {target.split_tolerance:.1e}
"""
    return f"""\
pair_style lj/cut/coul/esp 9.0
pair_coeff 1 1 0.15535 3.166
pair_coeff 1 2 0.0 1.0
pair_coeff 2 2 0.0 1.0
pair_modify table 0
kspace_style {'esp' if kind == 'pswf' else 'esp/bspline'} {target.split_tolerance:.1e} {POLY_TOLERANCE:.1e}
"""


def lammps_input(
    *,
    kind: str,
    target: Target | None,
    mesh: int | None,
    order: int | None,
    cspread: float | None,
    frame: int,
    dump_path: Path,
) -> str:
    modification = ""
    if kind != "ewald":
        assert mesh is not None and order is not None
        cspread_option = f"cspread {cspread:.12g} " if kind == "pswf" and cspread else ""
        modification = (
            f"kspace_modify {cspread_option}order {order} mesh {mesh} {mesh} {mesh} "
            f"diff {DIFFERENTIATION}"
        )
    return textwrap.dedent(
        f"""\
        # Automatically generated force calculation for the large all-atom water scan.
        newton on
        units real
        atom_style full
        read_data {bundle_relative(SOURCE_WATER)}
        replicate 2 2 2
        reset_timestep 0

        bond_style harmonic
        angle_style harmonic
        bond_coeff 1 0.0 1.000
        angle_coeff 1 0.0 109.47
        special_bonds lj/coul 0 0 0.5

        {pair_block(kind, target).strip()}
        {modification}

        neighbor 1.0 bin
        neigh_modify every 1 delay 0 check yes one 2000
        thermo_style custom step pe ecoul elong press
        thermo 1
        dump f all custom 1 {bundle_relative(dump_path)} id fx fy fz
        dump_modify f sort id format line "%d %.17g %.17g %.17g"
        rerun {bundle_relative(TRAJECTORY)} first {frame} last {frame} every 1 dump x y z ix iy iz box yes format native
        """
    )


def run_lammps(lmp: Path, input_path: Path, log_path: Path, screen_path: Path) -> None:
    with screen_path.open("w") as screen:
        subprocess.run(
            [str(lmp), "-in", str(input_path), "-log", str(log_path)],
            cwd=BUNDLE_ROOT,
            stdout=screen,
            stderr=subprocess.STDOUT,
            check=True,
        )


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    items = list(rows)
    if not items:
        return
    # Targeted extensions may add provenance columns to older source tables.
    # Preserve every existing column and append any new fields deterministically
    # instead of assuming that the first historical row has the latest schema.
    fields = list(
        dict.fromkeys(field for item in items for field in item.keys())
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


def numeric(value: str | float | int | None) -> float:
    return float(value) if value not in (None, "") else math.nan


def row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row.get(name, ""))
        for name in ("stage", "target", "window", "frame", "requested_mesh", "order", "gamma")
    )


def candidate_tag(target: Target, window: str, mesh: int, order: int, gamma: float | None, frame: int) -> str:
    suffix = "" if gamma is None else f"_g{gamma:.8f}".replace(".", "p")
    return f"{target.label}_{window}_m{mesh}_p{order}{suffix}_frame{frame}"


def evaluate_candidate(
    *,
    lmp: Path,
    target: Target,
    window: str,
    mesh: int,
    order: int,
    gamma: float | None,
    frame: int,
    stage: str,
    reference: dict[int, tuple[float, float, float]],
    force: bool,
) -> dict[str, Any]:
    """Compute one candidate force error and discard its transient dump."""

    cspread = None if gamma is None else 0.5 * math.pi * order * gamma
    run_dir = RUNS / stage / candidate_tag(target, window, mesh, order, gamma, frame)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump = run_dir / "forces.dump"
    input_path = run_dir / "in.force"
    log_path = run_dir / "log.lammps"
    screen_path = run_dir / "screen.txt"
    input_path.write_text(
        lammps_input(
            kind=window,
            target=target,
            mesh=mesh,
            order=order,
            cspread=cspread,
            frame=frame,
            dump_path=dump,
        )
    )
    if force or not (has_readable_force_dump(dump) and log_path.exists()):
        run_lammps(lmp, input_path, log_path, screen_path)
    force_error = compare_forces(reference, read_dump(dump))
    metadata = parse_log(log_path)
    actual_grid = tuple(int(value) for value in metadata["grid"])
    actual_csplit = numeric(metadata["csplit"])
    if window != "pppm" and not math.isclose(actual_csplit, target.csplit, rel_tol=0.0, abs_tol=5.0e-4):
        raise RuntimeError(
            f"{window} used c_split={actual_csplit}, expected {target.csplit} for {target.label}"
        )
    actual_cspread = numeric(metadata["cspread"]) if window == "pswf" else math.nan
    row: dict[str, Any] = {
        "stage": stage,
        "differentiation": DIFFERENTIATION,
        "target": target.label,
        "target_force_error": target.force_target,
        "split_tolerance": target.split_tolerance,
        "window": window,
        "frame": frame,
        "requested_mesh": mesh,
        "actual_nx": actual_grid[0],
        "actual_ny": actual_grid[1],
        "actual_nz": actual_grid[2],
        "actual_grid_volume": math.prod(actual_grid),
        "order": int(metadata["order"]),
        "c_split": actual_csplit if window != "pppm" else "",
        "c_spread": actual_cspread if window == "pswf" else "",
        "gamma": gamma if gamma is not None else "",
        "sigma_up": (
            math.pi * CUTOFF * actual_grid[0] / (actual_csplit * BOX)
            if window != "pppm"
            else ""
        ),
        **force_error,
    }
    # These are temporary evaluation dumps.  The Ewald reference dumps and
    # all derived source-data CSVs are retained; hundreds of redundant 21k-
    # atom candidate dumps are deliberately not accumulated.
    dump.unlink(missing_ok=True)
    return row


def reference_path(frame: int) -> Path:
    return REFERENCE / f"forces.ewald_frame{frame}.dump"


def ensure_references(lmp: Path, force: bool) -> dict[int, dict[int, tuple[float, float, float]]]:
    references: dict[int, dict[int, tuple[float, float, float]]] = {}
    for frame in (SELECTION_FRAME, *VALIDATION_FRAMES):
        dump = reference_path(frame)
        run_dir = REFERENCE / f"frame{frame}"
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / "in.ewald"
        log_path = run_dir / "log.ewald.lammps"
        screen_path = run_dir / "screen.ewald.txt"
        input_path.write_text(
            lammps_input(
                kind="ewald",
                target=None,
                mesh=None,
                order=None,
                cspread=None,
                frame=frame,
                dump_path=dump,
            )
        )
        if force or not dump.exists():
            run_lammps(lmp, input_path, log_path, screen_path)
        references[frame] = read_dump(dump)
        if len(references[frame]) != 21624:
            raise RuntimeError(f"Frame {frame} reference contains {len(references[frame])} atoms, not 21624")
    return references


def target_by_label(label: str) -> Target:
    for target in TARGETS:
        if target.label == label:
            return target
    raise KeyError(label)


def selected_targets(labels: tuple[str, ...]) -> tuple[Target, ...]:
    if not labels:
        return TARGETS
    return tuple(target_by_label(label) for label in labels)


def existing_index(rows: Iterable[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    return {row_key(row): row for row in rows}


def meshes_to_scan(
    target: Target, window: str, requested_meshes: tuple[int, ...] | None
) -> tuple[int, ...]:
    """Return either the published mesh sweep or an explicit extension list.

    The latter is used for isolated follow-up points without rewriting the
    complete target-level validation summary.  The LAMMPS-reported actual grid
    remains the recorded observable in either case.
    """

    return requested_meshes if requested_meshes else window_meshes(target, window)


def selection_stage(
    lmp: Path,
    targets: tuple[Target, ...],
    references: dict[int, dict[int, tuple[float, float, float]]],
    force: bool,
    *,
    windows: tuple[str, ...] = WINDOWS,
    requested_meshes: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [dict(row) for row in csv_rows(SELECTION_CSV)]
    index = existing_index(rows)
    for target in targets:
        for window in windows:
            for mesh in meshes_to_scan(target, window, requested_meshes):
                for order in window_orders(target, window):
                    gammas: tuple[float | None, ...]
                    if window == "pswf":
                        gammas = gamma_candidates(target, mesh, order)
                    else:
                        gammas = (None,)
                    for gamma in gammas:
                        prototype = {
                            "stage": "selection",
                            "target": target.label,
                            "window": window,
                            "frame": SELECTION_FRAME,
                            "requested_mesh": mesh,
                            "order": order,
                            "gamma": "" if gamma is None else gamma,
                        }
                        key = row_key(prototype)
                        if key in index and not force:
                            continue
                        row = evaluate_candidate(
                            lmp=lmp,
                            target=target,
                            window=window,
                            mesh=mesh,
                            order=order,
                            gamma=gamma,
                            frame=SELECTION_FRAME,
                            stage="selection",
                            reference=references[SELECTION_FRAME],
                            force=force,
                        )
                        if key in index:
                            rows = [item for item in rows if row_key(item) != key]
                        rows.append(row)
                        index[key] = row
                        write_rows(SELECTION_CSV, rows)
    return rows


def best_selection_rows(rows: Iterable[dict[str, Any]], target: Target, window: str, mesh: int) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if row["target"] == target.label
        and row["window"] == window
        and int(row["requested_mesh"]) == mesh
    ]
    best: list[dict[str, Any]] = []
    for order in window_orders(target, window):
        candidates = [row for row in filtered if int(row["order"]) == order]
        if candidates:
            best.append(min(candidates, key=lambda row: numeric(row["rms_relative_force_error"])))
    return best


def validation_stage(
    lmp: Path,
    targets: tuple[Target, ...],
    references: dict[int, dict[int, tuple[float, float, float]]],
    force: bool,
    *,
    windows: tuple[str, ...] = WINDOWS,
    requested_meshes: tuple[int, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection_rows: list[dict[str, Any]] = [dict(row) for row in csv_rows(SELECTION_CSV)]
    if not selection_rows:
        raise RuntimeError("No selection data available; run --stage selection first")
    rows: list[dict[str, Any]] = [dict(row) for row in csv_rows(VALIDATION_CSV)]
    old_summaries: list[dict[str, Any]] = [dict(row) for row in csv_rows(SUMMARY_CSV)]
    index = existing_index(rows)
    summaries: list[dict[str, Any]] = []
    for target in targets:
        for window in windows:
            for mesh in meshes_to_scan(target, window, requested_meshes):
                selection_by_order = best_selection_rows(selection_rows, target, window, mesh)
                if not selection_by_order:
                    raise RuntimeError(f"Missing selection rows for {target.label}, {window}, M={mesh}")
                candidate_reports: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
                for selected in selection_by_order:
                    order = int(selected["order"])
                    gamma_raw = selected.get("gamma", "")
                    gamma = numeric(gamma_raw) if gamma_raw not in ("", None) else None
                    frame_rows: list[dict[str, Any]] = []
                    for frame in VALIDATION_FRAMES:
                        prototype = {
                            "stage": "validation",
                            "target": target.label,
                            "window": window,
                            "frame": frame,
                            "requested_mesh": mesh,
                            "order": order,
                            "gamma": "" if gamma is None else gamma,
                        }
                        key = row_key(prototype)
                        if key in index and not force:
                            row = index[key]
                        else:
                            row = evaluate_candidate(
                                lmp=lmp,
                                target=target,
                                window=window,
                                mesh=mesh,
                                order=order,
                                gamma=gamma,
                                frame=frame,
                                stage="validation",
                                reference=references[frame],
                                force=force,
                            )
                            if key in index:
                                rows = [item for item in rows if row_key(item) != key]
                            rows.append(row)
                            index[key] = row
                            write_rows(VALIDATION_CSV, rows)
                        frame_rows.append(row)
                    candidate_reports.append((selected, frame_rows))
                    if all(numeric(row["rms_relative_force_error"]) <= target.force_target for row in frame_rows):
                        break

                selected, frame_rows = candidate_reports[-1]
                errors = [numeric(row["rms_relative_force_error"]) for row in frame_rows]
                feasible = all(error <= target.force_target for error in errors)
                summaries.append(
                    {
                        "differentiation": DIFFERENTIATION,
                        "target": target.label,
                        "target_force_error": target.force_target,
                        "window": window,
                        "requested_mesh": mesh,
                        "actual_nx": selected["actual_nx"],
                        "actual_ny": selected["actual_ny"],
                        "actual_nz": selected["actual_nz"],
                        "actual_grid_volume": selected["actual_grid_volume"],
                        "order": selected["order"],
                        "c_split": selected.get("c_split", ""),
                        "c_spread": selected.get("c_spread", ""),
                        "gamma": selected.get("gamma", ""),
                        "sigma_up": selected.get("sigma_up", ""),
                        "selection_rms_relative_force_error": selected["rms_relative_force_error"],
                        "validation_rms_relative_force_error_mean": statistics.fmean(errors),
                        "validation_rms_relative_force_error_std": statistics.stdev(errors) if len(errors) > 1 else 0.0,
                        "validation_rms_relative_force_error_max": max(errors),
                        "validation_rms_relative_force_error_min": min(errors),
                        "all_validation_frames_feasible": int(feasible),
                        "n_validation_frames": len(frame_rows),
                    }
                )
    # A targeted extension must not erase the rest of a target's published
    # grid sweep.  Replace only the (target, window, requested-mesh) summaries
    # recomputed above, while retaining all other source records.
    refreshed_keys = {
        (str(row["target"]), str(row["window"]), str(row["requested_mesh"]))
        for row in summaries
    }
    all_summaries = [
        row
        for row in old_summaries
        if (str(row.get("target", "")), str(row.get("window", "")), str(row.get("requested_mesh", "")))
        not in refreshed_keys
    ] + summaries
    all_summaries.sort(key=lambda row: (row["target"], row["window"], int(row["requested_mesh"])))
    write_rows(SUMMARY_CSV, all_summaries)
    return rows, summaries


def print_protocol(
    targets: tuple[Target, ...],
    windows: tuple[str, ...],
    requested_meshes: tuple[int, ...] | None,
) -> None:
    print("Large all-atom SPC/E water scan")
    print(f"  coordinates: {TRAJECTORY}")
    print("  system: 7,208 waters / 21,624 atoms / 60 A cubic cell")
    print(f"  differentiation: {DIFFERENTIATION}")
    print(f"  selection frame: {SELECTION_FRAME}; validation frames: {VALIDATION_FRAMES}")
    for target in targets:
        meshes = requested_meshes if requested_meshes else target.meshes
        minimum_sigma = min(math.pi * CUTOFF * mesh / (target.csplit * BOX) for mesh in meshes)
        print(
            f"  {target.label}: c_split={target.csplit:.3f}, ESP P={target.orders}, "
            f"B-spline P={window_orders(target, 'bspline')}, "
            f"PPPM P={window_orders(target, 'pppm')}, ESP meshes={target.meshes}, "
            f"PPPM meshes={window_meshes(target, 'pppm')}, selected windows={windows}, "
            f"selected meshes={meshes}, minimum sigma_up={minimum_sigma:.3f}"
        )


def main() -> None:
    global DIFFERENTIATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True, help="LAMMPS executable with esp/bspline")
    parser.add_argument(
        "--differentiation",
        choices=("ik", "ad"),
        default="ik",
        help="force-differentiation operator for this namespaced sweep",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="output directory, relative to this calculation directory when not absolute",
    )
    parser.add_argument(
        "--stage",
        choices=("references", "selection", "validation", "all"),
        default="all",
        help="workflow stage to execute",
    )
    parser.add_argument("--target", action="append", choices=("1e-4", "1e-5"), default=[])
    parser.add_argument(
        "--window",
        action="append",
        choices=WINDOWS,
        default=[],
        help="restrict an extension to one or more spreading windows",
    )
    parser.add_argument(
        "--mesh",
        type=int,
        action="append",
        default=[],
        help="restrict an extension to one or more requested cubic FFT grids",
    )
    parser.add_argument("--force", action="store_true", help="rerun existing cached calculations")
    args = parser.parse_args()
    DIFFERENTIATION = args.differentiation
    configure_output_root(args.output_root)
    ensure_inputs()
    if not args.lmp.exists():
        raise FileNotFoundError(args.lmp)
    targets = selected_targets(tuple(args.target))
    windows = tuple(dict.fromkeys(args.window)) if args.window else WINDOWS
    requested_meshes = tuple(dict.fromkeys(args.mesh)) if args.mesh else None
    if requested_meshes and any(mesh <= 0 for mesh in requested_meshes):
        parser.error("--mesh values must be positive")
    print_protocol(targets, windows, requested_meshes)
    if args.stage == "references":
        ensure_references(args.lmp, args.force)
        return
    references = ensure_references(args.lmp, args.force)
    if args.stage in {"selection", "all"}:
        selection_stage(
            args.lmp,
            targets,
            references,
            args.force,
            windows=windows,
            requested_meshes=requested_meshes,
        )
    if args.stage in {"validation", "all"}:
        _, summary = validation_stage(
            args.lmp,
            targets,
            references,
            args.force,
            windows=windows,
            requested_meshes=requested_meshes,
        )
        for row in summary:
            print(
                f"{row['target']:>4} {row['window']:>7} M={row['actual_nx']} "
                f"P={row['order']} max validation error="
                f"{numeric(row['validation_rms_relative_force_error_max']):.3e} "
                f"feasible={row['all_validation_frames_feasible']}"
            )


if __name__ == "__main__":
    main()
