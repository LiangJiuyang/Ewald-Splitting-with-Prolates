#!/usr/bin/env python3
"""Build the Figure-6 PPPM order-scan source tables from raw force dumps.

Only complete 50-frame water force scans with a parseable LAMMPS version,
actual FFT grid, and actual stencil order are accepted.  All accepted dumps
are re-evaluated against one common Ewald force dump by this script.  The
matched-target derivative table jointly selects the measured PPPM order and
actual grid that minimize the FFT-grid volume at each target.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WATER_DIR = REPO / "numerical_examples" / "water_trajectory_benchmark"
SYMMETRIC_DIR = HERE / "pppm_symmetric_scan"
SYMMETRIC_MANIFEST = SYMMETRIC_DIR / "manifest.json"
REFERENCE = WATER_DIR / "forces.ref_ewald.dump"
REFERENCE_INPUT = WATER_DIR / "in.force_ref_ewald"
REFERENCE_LOG = WATER_DIR / "log.force_ref_ewald.lammps"
TRAJECTORY = WATER_DIR / "water_short_traj.lammpstrj"
WATER_DATA = WATER_DIR / "water.data"
EXPECTED_FRAMES = 50
TARGETS = (1.0e-3, 1.0e-4, 1.0e-5)

SOURCE_CSV = HERE / "fig6_pppm_order_scan_source.csv"
BY_FRAME_CSV = HERE / "fig6_pppm_order_scan_by_frame.csv"
DEDUPLICATED_CSV = HERE / "fig6_pppm_order_scan_deduplicated.csv"
DEDUP_MAP_CSV = HERE / "fig6_pppm_order_scan_dedup_map.csv"
BEST_CSV = HERE / "best_pppm_by_target.csv"
MANIFEST = HERE / "fig6_pppm_order_scan_manifest.json"


@dataclass
class Candidate:
    order: int
    requested_mesh: int
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        return f"pppm_p{self.order}_requested_mesh{self.requested_mesh}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def load_symmetric_manifest() -> dict[str, object]:
    if not SYMMETRIC_MANIFEST.is_file():
        raise FileNotFoundError(SYMMETRIC_MANIFEST)
    manifest = json.loads(SYMMETRIC_MANIFEST.read_text(encoding="utf-8"))
    required = (
        "lammps_executable_sha256",
        "lammps_versions",
        "fft_backends",
        "executed_case_matrix",
        "runs",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"symmetric scan manifest lacks {missing}")
    if len(manifest["lammps_versions"]) != 1 or len(manifest["fft_backends"]) != 1:
        raise ValueError("symmetric scan manifest is not implementation-uniform")
    digest = str(manifest["lammps_executable_sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("symmetric scan manifest has an invalid executable SHA-256")
    return manifest


def discover_candidates() -> list[Candidate]:
    families = tuple(
        (
            order,
            SYMMETRIC_DIR,
            {
                "input": re.compile(rf"^in\.pppm_p{order}_mesh([0-9]+)$"),
                "dump": re.compile(rf"^forces\.pppm_p{order}_mesh([0-9]+)\.dump$"),
                "log": re.compile(rf"^log\.pppm_p{order}_mesh([0-9]+)\.lammps$"),
                "screen": re.compile(rf"^screen\.pppm_p{order}_mesh([0-9]+)\.txt$"),
            },
        )
        for order in range(4, 8)
    )
    found: dict[tuple[int, int], Candidate] = {}
    for order, directory, patterns in families:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file():
                continue
            for kind, pattern in patterns.items():
                match = pattern.fullmatch(path.name)
                if match:
                    requested_mesh = int(match.group(1))
                    candidate = found.setdefault(
                        (order, requested_mesh), Candidate(order, requested_mesh)
                    )
                    candidate.files[kind] = path
                    break
    return [found[key] for key in sorted(found)]


def read_force_dump(
    path: Path,
) -> list[tuple[int, dict[int, tuple[float, float, float]]]]:
    frames: list[tuple[int, dict[int, tuple[float, float, float]]]] = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            marker = handle.readline()
            if not marker:
                break
            if marker.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"expected ITEM: TIMESTEP, found {marker.strip()!r}")
            timestep = int(handle.readline().strip())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"missing NUMBER OF ATOMS at timestep {timestep}")
            natoms = int(handle.readline().strip())
            if not handle.readline().strip().startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"missing BOX BOUNDS at timestep {timestep}")
            for _ in range(3):
                if not handle.readline():
                    raise ValueError(f"truncated box at timestep {timestep}")
            header = handle.readline().strip().split()
            if header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError(f"missing ATOMS header at timestep {timestep}")
            columns = header[2:]
            required = ("id", "fx", "fy", "fz")
            if any(column not in columns for column in required):
                raise ValueError(f"force columns {columns} do not contain {required}")
            indices = {column: columns.index(column) for column in required}
            atoms: dict[int, tuple[float, float, float]] = {}
            for _ in range(natoms):
                fields = handle.readline().split()
                if len(fields) != len(columns):
                    raise ValueError(f"truncated atom data at timestep {timestep}")
                atom_id = int(fields[indices["id"]])
                atoms[atom_id] = (
                    float(fields[indices["fx"]]),
                    float(fields[indices["fy"]]),
                    float(fields[indices["fz"]]),
                )
            if len(atoms) != natoms:
                raise ValueError(f"duplicate or missing atom ids at timestep {timestep}")
            frames.append((timestep, atoms))
    if not frames:
        raise ValueError("no force frames")
    return frames


def parse_requested_settings(text: str) -> tuple[int, tuple[int, int, int]]:
    matches = re.findall(
        r"^\s*kspace_modify\s+order\s+([0-9]+)\s+mesh\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not matches:
        raise ValueError("missing unambiguous kspace_modify order/mesh command")
    order, nx, ny, nz = matches[-1]
    return int(order), (int(nx), int(ny), int(nz))


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    version = re.findall(r"^LAMMPS \((.+)\)\s*$", text, re.MULTILINE)
    actual_grids = re.findall(
        r"^\s*grid\s*=\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*$",
        text,
        re.MULTILINE,
    )
    actual_orders = re.findall(
        r"^\s*stencil order\s*=\s*([0-9]+)\s*$", text, re.MULTILINE
    )
    loops = re.findall(
        r"Loop time of\s+([0-9.eE+-]+)\s+on\s+([0-9]+)\s+procs for\s+([0-9]+)\s+steps",
        text,
    )
    internal_estimates = re.findall(
        r"estimated relative force accuracy\s*=\s*([0-9.eE+-]+)", text
    )
    g_vectors = re.findall(r"G vector \(1/distance\)\s*=\s*([0-9.eE+-]+)", text)
    fft_lines = re.findall(r"^\s*using double precision (.+?)\s*$", text, re.MULTILINE)
    return {
        "text": text,
        "version": version[-1] if version else None,
        "actual_grid": (
            tuple(int(value) for value in actual_grids[-1]) if actual_grids else None
        ),
        "actual_order": int(actual_orders[-1]) if actual_orders else None,
        "loop_time_s": float(loops[-1][0]) if loops else None,
        "mpi_ranks": int(loops[-1][1]) if loops else None,
        "loop_steps": int(loops[-1][2]) if loops else None,
        "internal_relative_accuracy": (
            float(internal_estimates[-1]) if internal_estimates else None
        ),
        "g_ewald_inverse_A": float(g_vectors[-1]) if g_vectors else None,
        "fft_backend": fft_lines[-1].strip() if fft_lines else None,
        "error_count": text.count("ERROR:"),
        "requested_settings": parse_requested_settings(text),
    }


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return math.fsum(data) / len(data)


def sem(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def block_pooled_rms_sem(
    values: list[float], block_size: int = 5
) -> tuple[int, float]:
    """SEM of nonoverlapping block-level RMS values.

    Each block statistic is ``sqrt(mean(e_f**2))``, where ``e_f`` is the
    per-frame relative RMS force error.  The incomplete final block is not
    used in this uncertainty estimate.
    """
    nblocks = len(values) // block_size
    if nblocks < 2:
        return nblocks, 0.0
    block_rms_values = [
        math.sqrt(
            mean(
                value * value
                for value in values[index * block_size : (index + 1) * block_size]
            )
        )
        for index in range(nblocks)
    ]
    return nblocks, sem(block_rms_values)


def existing_file_record(path: Path) -> dict[str, object]:
    return {"path": relpath(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def audit_and_measure(
    candidate: Candidate,
    reference: list[tuple[int, dict[int, tuple[float, float, float]]]],
    executable_sha256: str,
) -> tuple[dict[str, object], dict[str, object] | None, list[dict[str, object]]]:
    audit: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "nominal_order_from_family": candidate.order,
        "requested_mesh_from_filename": candidate.requested_mesh,
        "status": "excluded",
        "reasons": [],
        "files": {
            key: existing_file_record(path) for key, path in sorted(candidate.files.items())
        },
    }
    reasons: list[str] = audit["reasons"]  # type: ignore[assignment]
    for required in ("input", "dump", "log"):
        if required not in candidate.files:
            reasons.append(f"missing_{required}")
    if reasons:
        return audit, None, []

    input_path = candidate.files["input"]
    dump_path = candidate.files["dump"]
    log_path = candidate.files["log"]
    input_text = input_path.read_text(encoding="utf-8", errors="replace")
    try:
        input_order, input_mesh = parse_requested_settings(input_text)
    except ValueError as error:
        reasons.append(f"input_parse_error:{error}")
        return audit, None, []
    try:
        log = parse_log(log_path)
    except ValueError as error:
        reasons.append(f"log_parse_error:{error}")
        return audit, None, []

    log_order, log_requested_mesh = log["requested_settings"]  # type: ignore[misc]
    audit.update(
        {
            "input_requested_order": input_order,
            "input_requested_grid": list(input_mesh),
            "log_requested_order": log_order,
            "log_requested_grid": list(log_requested_mesh),
            "actual_order": log["actual_order"],
            "actual_grid": list(log["actual_grid"]) if log["actual_grid"] else None,
            "lammps_version": log["version"],
            "loop_steps": log["loop_steps"],
            "error_count": log["error_count"],
        }
    )
    expected_requested = (candidate.requested_mesh,) * 3
    if input_order != candidate.order or input_mesh != expected_requested:
        reasons.append("input_order_or_requested_grid_disagrees_with_filename_family")
    if log_order != input_order or log_requested_mesh != input_mesh:
        reasons.append("log_echo_disagrees_with_input_order_or_requested_grid")
    if log["version"] is None:
        reasons.append("unknown_lammps_version")
    if log["actual_grid"] is None:
        reasons.append("missing_actual_grid_in_log")
    if log["actual_order"] is None:
        reasons.append("missing_actual_stencil_order_in_log")
    elif log["actual_order"] != candidate.order:
        reasons.append("actual_stencil_order_disagrees_with_candidate_family")
    if log["loop_steps"] != EXPECTED_FRAMES:
        reasons.append(f"log_loop_steps_is_{log['loop_steps']}_not_{EXPECTED_FRAMES}")
    if log["error_count"]:
        reasons.append(f"lammps_log_contains_{log['error_count']}_errors")
    if "kspace_style pppm" not in input_text:
        reasons.append("input_is_not_pppm")
    if "water_short_traj.lammpstrj" not in input_text:
        reasons.append("input_does_not_name_common_water_trajectory")
    if reasons:
        return audit, None, []

    try:
        test = read_force_dump(dump_path)
    except (OSError, ValueError) as error:
        reasons.append(f"force_dump_parse_error:{error}")
        return audit, None, []
    audit["dump_frame_count"] = len(test)
    if len(test) != EXPECTED_FRAMES:
        reasons.append(f"force_dump_has_{len(test)}_frames_not_{EXPECTED_FRAMES}")
        return audit, None, []
    if len(test) != len(reference):
        reasons.append(f"force_dump_has_{len(test)}_frames_but_reference_has_{len(reference)}")
        return audit, None, []

    actual_grid = log["actual_grid"]
    assert isinstance(actual_grid, tuple)
    relative_values: list[float] = []
    absolute_values: list[float] = []
    reference_values: list[float] = []
    frame_rows: list[dict[str, object]] = []
    total_diff2 = 0.0
    total_ref2 = 0.0
    for frame_index, ((ref_step, ref_atoms), (test_step, test_atoms)) in enumerate(
        zip(reference, test)
    ):
        if ref_step != test_step:
            reasons.append(
                f"frame_{frame_index}_timestep_{test_step}_does_not_match_reference_{ref_step}"
            )
            return audit, None, []
        if ref_atoms.keys() != test_atoms.keys():
            reasons.append(f"frame_{frame_index}_atom_ids_do_not_match_reference")
            return audit, None, []
        diff2 = 0.0
        ref2 = 0.0
        for atom_id, ref_force in ref_atoms.items():
            test_force = test_atoms[atom_id]
            for ref_component, test_component in zip(ref_force, test_force):
                delta = test_component - ref_component
                diff2 += delta * delta
                ref2 += ref_component * ref_component
        natoms = len(ref_atoms)
        relative = math.sqrt(diff2 / ref2)
        absolute = math.sqrt(diff2 / natoms)
        reference_rms = math.sqrt(ref2 / natoms)
        relative_values.append(relative)
        absolute_values.append(absolute)
        reference_values.append(reference_rms)
        total_diff2 += diff2
        total_ref2 += ref2
        frame_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "order": candidate.order,
                "requested_mesh": candidate.requested_mesh,
                "actual_nx": actual_grid[0],
                "actual_ny": actual_grid[1],
                "actual_nz": actual_grid[2],
                "actual_mesh": actual_grid[0] if len(set(actual_grid)) == 1 else "",
                "actual_grid_points": math.prod(actual_grid),
                "frame_index_zero_based": frame_index,
                "timestep": ref_step,
                "natoms": natoms,
                "relative_rms": relative,
                "absolute_rms_kcal_per_mol_A": absolute,
                "reference_rms_kcal_per_mol_A": reference_rms,
                "sum_diff_squared": diff2,
                "sum_reference_squared": ref2,
            }
        )

    nblocks, block5 = block_pooled_rms_sem(relative_values)
    summary = {
        "candidate_id": candidate.candidate_id,
        "method": "PPPM",
        "differentiation": "ik",
        "influence_function": "Hockney-Eastwood optimal",
        "order": candidate.order,
        "requested_mesh": candidate.requested_mesh,
        "requested_grid_points": candidate.requested_mesh**3,
        "actual_nx": actual_grid[0],
        "actual_ny": actual_grid[1],
        "actual_nz": actual_grid[2],
        "actual_mesh": actual_grid[0] if len(set(actual_grid)) == 1 else "",
        "actual_grid_points": math.prod(actual_grid),
        "actual_to_requested_grid_volume_ratio": math.prod(actual_grid)
        / candidate.requested_mesh**3,
        "nframes": len(relative_values),
        "relative_rms_mean": mean(relative_values),
        "relative_rms_sd_across_frames": statistics.stdev(relative_values),
        "relative_rms_frame_sem": sem(relative_values),
        "relative_rms_block5_sem": block5,
        "n_nonoverlapping_blocks_of_5": nblocks,
        "relative_rms_pooled": math.sqrt(total_diff2 / total_ref2),
        "absolute_rms_mean_kcal_per_mol_A": mean(absolute_values),
        "reference_rms_mean_kcal_per_mol_A": mean(reference_values),
        "lammps_internal_estimated_relative_accuracy": log[
            "internal_relative_accuracy"
        ],
        "g_ewald_inverse_A": log["g_ewald_inverse_A"],
        "loop_time_s_single_rank": log["loop_time_s"],
        "mpi_ranks": log["mpi_ranks"],
        "fft_backend": log["fft_backend"],
        "lammps_version": log["version"],
        "lammps_executable_sha256": executable_sha256,
        "input_path": relpath(input_path),
        "force_dump_path": relpath(dump_path),
        "force_dump_sha256": sha256(dump_path),
        "log_path": relpath(log_path),
    }
    audit["status"] = "accepted"
    audit["reasons"] = []
    audit["dump_frame_count"] = len(test)
    audit["summary_relative_rms_mean"] = summary["relative_rms_mean"]
    return audit, summary, frame_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


DEDUPLICATION_CONSISTENCY_FIELDS = (
    "actual_nx",
    "actual_ny",
    "actual_nz",
    "actual_mesh",
    "actual_grid_points",
    "nframes",
    "relative_rms_mean",
    "relative_rms_sd_across_frames",
    "relative_rms_frame_sem",
    "relative_rms_block5_sem",
    "n_nonoverlapping_blocks_of_5",
    "relative_rms_pooled",
    "absolute_rms_mean_kcal_per_mol_A",
    "reference_rms_mean_kcal_per_mol_A",
)


def deduplicate_summaries(
    summaries: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collapse physically identical PPPM runs before target selection.

    The identity key is the interpolation order together with the SHA-256 of
    the raw force dump.  Repeated requested meshes are accepted as duplicates
    only if their actual grid and every recorded 50-frame error statistic are
    exactly identical.  Any disagreement is treated as a provenance failure,
    rather than silently choosing one row.
    """
    groups: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in summaries:
        order = int(row["order"])
        digest = str(row["force_dump_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                f"candidate {row['candidate_id']} has invalid force-dump SHA-256 {digest!r}"
            )
        if int(row["nframes"]) != EXPECTED_FRAMES:
            raise ValueError(
                f"candidate {row['candidate_id']} has {row['nframes']} frames, "
                f"expected {EXPECTED_FRAMES}"
            )
        groups.setdefault((order, digest), []).append(row)

    deduplicated: list[dict[str, object]] = []
    mapping: list[dict[str, object]] = []
    for (order, digest), members in sorted(groups.items()):
        members.sort(key=lambda row: (int(row["requested_mesh"]), str(row["candidate_id"])))
        representative = members[0]
        for member in members[1:]:
            disagreements = [
                field
                for field in DEDUPLICATION_CONSISTENCY_FIELDS
                if member[field] != representative[field]
            ]
            if disagreements:
                raise ValueError(
                    "force-hash duplicate group has inconsistent actual-grid or "
                    "50-frame statistics: "
                    f"{representative['candidate_id']} vs {member['candidate_id']}; "
                    f"fields={','.join(disagreements)}"
                )

        member_ids = ";".join(str(row["candidate_id"]) for row in members)
        unique_row = dict(representative)
        unique_row.update(
            {
                "dedup_group_size": len(members),
                "dedup_member_candidate_ids": member_ids,
            }
        )
        deduplicated.append(unique_row)
        for member in members:
            mapping.append(
                {
                    "candidate_id": member["candidate_id"],
                    "representative_candidate_id": representative["candidate_id"],
                    "order": order,
                    "requested_mesh": member["requested_mesh"],
                    "actual_nx": member["actual_nx"],
                    "actual_ny": member["actual_ny"],
                    "actual_nz": member["actual_nz"],
                    "actual_grid_points": member["actual_grid_points"],
                    "force_dump_sha256": digest,
                    "dedup_group_size": len(members),
                    "is_representative": member is representative,
                    "validation_status": "actual grid and all recorded 50-frame statistics agree",
                }
            )

    deduplicated.sort(
        key=lambda row: (
            int(row["order"]),
            int(row["actual_grid_points"]),
            int(row["requested_mesh"]),
        )
    )
    mapping.sort(key=lambda row: (int(row["order"]), int(row["requested_mesh"])))
    return deduplicated, mapping


def reference_metadata(
    reference: list[tuple[int, dict[int, tuple[float, float, float]]]],
) -> dict[str, object]:
    log_text = REFERENCE_LOG.read_text(encoding="utf-8", errors="replace")
    versions = re.findall(r"^LAMMPS \((.+)\)\s*$", log_text, re.MULTILINE)
    loops = re.findall(
        r"Loop time of\s+([0-9.eE+-]+)\s+on\s+([0-9]+)\s+procs for\s+([0-9]+)\s+steps",
        log_text,
    )
    estimates = re.findall(
        r"estimated relative force accuracy\s*=\s*([0-9.eE+-]+)", log_text
    )
    return {
        "force_dump": existing_file_record(REFERENCE),
        "input": existing_file_record(REFERENCE_INPUT),
        "log": existing_file_record(REFERENCE_LOG),
        "lammps_version": versions[-1] if versions else None,
        "frame_count": len(reference),
        "natoms_each_frame": sorted({len(atoms) for _, atoms in reference}),
        "timesteps": [step for step, _ in reference],
        "loop_time_s": float(loops[-1][0]) if loops else None,
        "mpi_ranks": int(loops[-1][1]) if loops else None,
        "loop_steps": int(loops[-1][2]) if loops else None,
        "lammps_internal_estimated_relative_accuracy": (
            float(estimates[-1]) if estimates else None
        ),
    }


def validate_cartesian_scan(
    scan_manifest: dict[str, object],
    candidates: list[Candidate],
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    """Require one complete common-grid scan and verify raw-file provenance."""
    manifest_runs = scan_manifest["runs"]
    if not isinstance(manifest_runs, list):
        raise ValueError("symmetric scan manifest runs must be a list")
    run_by_key: dict[tuple[int, int], dict[str, object]] = {}
    for run in manifest_runs:
        if not isinstance(run, dict):
            raise ValueError("symmetric scan manifest contains a non-object run")
        key = (int(run["order"]), int(run["requested_mesh"]))
        if key in run_by_key:
            raise ValueError(f"duplicate manifest run {key}")
        run_by_key[key] = run

    candidate_by_key = {
        (candidate.order, candidate.requested_mesh): candidate
        for candidate in candidates
    }
    if candidate_by_key.keys() != run_by_key.keys():
        missing = sorted(run_by_key.keys() - candidate_by_key.keys())
        extra = sorted(candidate_by_key.keys() - run_by_key.keys())
        raise ValueError(f"manifest/discovery mismatch: missing={missing}, extra={extra}")
    for key, run in run_by_key.items():
        candidate = candidate_by_key[key]
        for kind, manifest_key in (
            ("input", "input_sha256"),
            ("dump", "force_dump_sha256"),
            ("log", "log_sha256"),
        ):
            if kind not in candidate.files:
                raise ValueError(f"{key} is missing {kind}")
            found = sha256(candidate.files[kind])
            expected = str(run[manifest_key])
            if found != expected:
                raise ValueError(
                    f"{key} {kind} SHA-256 differs from runner manifest: "
                    f"{found} != {expected}"
                )

    expected_orders = (4, 5, 6, 7)
    requested_sets = {
        order: tuple(
            sorted(
                int(row["requested_mesh"])
                for row in summaries
                if int(row["order"]) == order
            )
        )
        for order in expected_orders
    }
    if any(not requested_sets[order] for order in expected_orders):
        raise ValueError(f"missing one or more PPPM orders: {requested_sets}")
    common_requested = requested_sets[expected_orders[0]]
    if any(requested_sets[order] != common_requested for order in expected_orders[1:]):
        raise ValueError(f"requested-grid sets differ by order: {requested_sets}")

    actual_sets = {
        order: tuple(
            sorted(
                (
                    int(row["actual_nx"]),
                    int(row["actual_ny"]),
                    int(row["actual_nz"]),
                )
                for row in summaries
                if int(row["order"]) == order
            )
        )
        for order in expected_orders
    }
    common_actual = actual_sets[expected_orders[0]]
    if any(actual_sets[order] != common_actual for order in expected_orders[1:]):
        raise ValueError(f"actual-grid sets differ by order: {actual_sets}")
    if any(nx != ny or ny != nz for nx, ny, nz in common_actual):
        raise ValueError(f"Figure 6 requires cubic actual grids: {common_actual}")

    versions = {str(row["lammps_version"]) for row in summaries}
    backends = {str(row["fft_backend"]) for row in summaries}
    executable_hashes = {str(row["lammps_executable_sha256"]) for row in summaries}
    if len(versions) != 1 or len(backends) != 1 or len(executable_hashes) != 1:
        raise ValueError(
            "Figure-6 scan is not implementation-uniform: "
            f"versions={versions}, backends={backends}, hashes={executable_hashes}"
        )
    expected_hash = str(scan_manifest["lammps_executable_sha256"])
    if executable_hashes != {expected_hash}:
        raise ValueError("summary executable hash differs from runner manifest")

    for order in expected_orders:
        errors = [
            float(row["relative_rms_pooled"])
            for row in summaries
            if int(row["order"]) == order
        ]
        for target in TARGETS:
            if not any(error > target for error in errors) or not any(
                error <= target for error in errors
            ):
                raise ValueError(
                    f"P={order} does not bracket target {target:.0e}: "
                    f"range={min(errors):.3e}--{max(errors):.3e}"
                )
    return {
        "orders": list(expected_orders),
        "common_requested_meshes": list(common_requested),
        "common_actual_grids": [list(grid) for grid in common_actual],
        "candidates_per_order": len(common_requested),
        "total_candidates": len(summaries),
        "lammps_version": next(iter(versions)),
        "fft_backend": next(iter(backends)),
        "lammps_executable_sha256": expected_hash,
        "all_orders_bracket_all_targets": True,
    }


def main() -> None:
    required_sources = (
        REFERENCE,
        REFERENCE_INPUT,
        REFERENCE_LOG,
        TRAJECTORY,
        WATER_DATA,
        SYMMETRIC_MANIFEST,
    )
    for path in required_sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = read_force_dump(REFERENCE)
    if len(reference) != EXPECTED_FRAMES:
        raise ValueError(
            f"common Ewald reference has {len(reference)} frames, expected {EXPECTED_FRAMES}"
        )

    scan_manifest = load_symmetric_manifest()
    executable_sha256 = str(scan_manifest["lammps_executable_sha256"])
    candidates = discover_candidates()
    audits: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for candidate in candidates:
        audit, summary, candidate_frames = audit_and_measure(
            candidate, reference, executable_sha256
        )
        audits.append(audit)
        if summary is not None:
            summaries.append(summary)
            frame_rows.extend(candidate_frames)

    summaries.sort(
        key=lambda row: (
            int(row["order"]),
            int(row["actual_grid_points"]),
            int(row["requested_mesh"]),
        )
    )
    frame_rows.sort(
        key=lambda row: (
            int(row["order"]),
            int(row["actual_grid_points"]),
            int(row["requested_mesh"]),
            int(row["frame_index_zero_based"]),
        )
    )

    cartesian_audit = validate_cartesian_scan(scan_manifest, candidates, summaries)
    deduplicated, dedup_map = deduplicate_summaries(summaries)

    best_rows: list[dict[str, object]] = []
    for target in TARGETS:
        feasible = [
            row for row in deduplicated
            if float(row["relative_rms_pooled"]) <= target
        ]
        if not feasible:
            best_rows.append(
                {
                    "target_relative_rms": target,
                    "selection_metric": "50-frame pooled relative RMS",
                    "status": "no_feasible_candidate",
                    "n_unique_feasible_candidates": 0,
                    "candidate_id": "",
                    "force_dump_sha256": "",
                    "order": "",
                    "requested_mesh": "",
                    "actual_nx": "",
                    "actual_ny": "",
                    "actual_nz": "",
                    "actual_mesh": "",
                    "actual_grid_points": "",
                    "relative_rms_pooled": "",
                    "relative_rms_block5_sem": "",
                    "selection_rule": "among measured PPPM P=4--7 candidates: minimum actual grid points; then lower interpolation order and requested mesh",
                }
            )
            continue
        selected = min(
            feasible,
            key=lambda row: (
                int(row["actual_grid_points"]),
                int(row["order"]),
                int(row["requested_mesh"]),
                str(row["candidate_id"]),
            ),
        )
        best_rows.append(
            {
                "target_relative_rms": target,
                "selection_metric": "50-frame pooled relative RMS",
                "status": "selected",
                "n_unique_feasible_candidates": len(feasible),
                "candidate_id": selected["candidate_id"],
                "force_dump_sha256": selected["force_dump_sha256"],
                "order": selected["order"],
                "requested_mesh": selected["requested_mesh"],
                "actual_nx": selected["actual_nx"],
                "actual_ny": selected["actual_ny"],
                "actual_nz": selected["actual_nz"],
                "actual_mesh": selected["actual_mesh"],
                "actual_grid_points": selected["actual_grid_points"],
                "relative_rms_pooled": selected["relative_rms_pooled"],
                "relative_rms_block5_sem": selected["relative_rms_block5_sem"],
                "selection_rule": "among force-hash-unique measured PPPM P=4--7 candidates: minimum actual grid points; then lower interpolation order and requested mesh",
            }
        )

    write_csv(SOURCE_CSV, summaries)
    write_csv(BY_FRAME_CSV, frame_rows)
    write_csv(DEDUPLICATED_CSV, deduplicated)
    write_csv(DEDUP_MAP_CSV, dedup_map)
    write_csv(BEST_CSV, best_rows)

    accepted = [audit for audit in audits if audit["status"] == "accepted"]
    excluded = [audit for audit in audits if audit["status"] == "excluded"]
    version_counts = Counter(str(row["lammps_version"]) for row in summaries)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": existing_file_record(Path(__file__)),
        "purpose": "PPPM-only source data for the water order/grid scan; no ESP values are read or written",
        "common_sources": {
            "water_data": existing_file_record(WATER_DATA),
            "trajectory": existing_file_record(TRAJECTORY),
            "ewald_reference": reference_metadata(reference),
            "symmetric_scan_runner_manifest": existing_file_record(
                SYMMETRIC_MANIFEST
            ),
        },
        "cartesian_scan_audit": cartesian_audit,
        "acceptance_requirements": [
            "input, raw force dump, and LAMMPS log all exist",
            "input and echoed log command agree on requested order and requested grid",
            "LAMMPS version, actual FFT grid, and actual stencil order parse from the log",
            "actual stencil order agrees with the P=4, P=5, P=6, or P=7 filename family",
            "LAMMPS log has no ERROR and reports 50 rerun steps",
            "raw force dump parses completely into 50 frames",
            "each timestep and atom-id set matches the common Ewald reference",
        ],
        "error_definition": "For each frame, sqrt(sum_i |F_PPPM-F_Ewald|^2 / sum_i |F_Ewald|^2); source summary records both the arithmetic mean and pooled RMS over 50 frames.",
        "uncertainty_definition": "Sample SEM across 50 per-frame relative RMS values and SEM across 10 non-overlapping five-frame block RMS values, where each block value is sqrt(mean(e_f^2)); all 50 frames enter the ten non-overlapping five-frame blocks.",
        "deduplication_rule": "Before target selection, group by (interpolation order, raw force-dump SHA-256), require exact agreement of actual grid and every recorded 50-frame statistic, and retain the lowest-requested-mesh row as the representative.",
        "target_selection_rule": "Among deduplicated measured PPPM P=4--7 candidates whose 50-frame pooled relative RMS is <= target, minimize actual nx*ny*nz and break ties by lower interpolation order and lower requested mesh.",
        "candidate_discovery_scope": {
            "P4_to_P7": "only redesigned_section5/pppm_symmetric_scan files named *pppm_pP_meshN* and listed in that directory's hash-pinned manifest",
            "excluded_by_design": "all historical asymmetric P=4--6 force scans and timing-only PPPM runs",
        },
        "candidate_counts": {
            "discovered": len(audits),
            "accepted": len(accepted),
            "excluded": len(excluded),
            "unique_after_order_force_hash_deduplication": len(deduplicated),
            "accepted_by_order": dict(Counter(str(row["order"]) for row in summaries)),
            "unique_by_order": dict(
                Counter(str(row["order"]) for row in deduplicated)
            ),
            "accepted_by_lammps_version": dict(sorted(version_counts.items())),
        },
        "candidate_audit": audits,
        "excluded_candidates": excluded,
        "output_tables": {
            "summary": existing_file_record(SOURCE_CSV),
            "by_frame": existing_file_record(BY_FRAME_CSV),
            "deduplicated_summary": existing_file_record(DEDUPLICATED_CSV),
            "deduplication_map": existing_file_record(DEDUP_MAP_CSV),
            "best_by_target": existing_file_record(BEST_CSV),
        },
        "notes": [
            "Requested and actual grids are retained separately; FFT-friendly rounding is never inferred from the filename.",
            "Target feasibility counts and best candidates are computed only after order-plus-force-hash deduplication across the measured P=4--7 scan.",
            "All accepted rows were generated by one SHA-256-pinned LAMMPS executable and one FFT backend.",
            "Every scanned order P=4--7 uses the identical requested and actual cubic-grid sets, and every order brackets every target.",
            "The joint order--grid selection ranks actual grid volume only and is not a wall-time optimization.",
            "Loop times are single-rank provenance fields, not a timing comparison or optimization result.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"accepted {len(accepted)} candidates ({len(deduplicated)} unique force runs); "
        f"excluded {len(excluded)}; wrote {SOURCE_CSV.name}, {BY_FRAME_CSV.name}, "
        f"{DEDUPLICATED_CSV.name}, {DEDUP_MAP_CSV.name}, {BEST_CSV.name}, and {MANIFEST.name}"
    )
    for row in best_rows:
        print(
            f"target {float(row['target_relative_rms']):.0e}: {row['candidate_id']} "
            f"actual={row['actual_nx']}x{row['actual_ny']}x{row['actual_nz']} "
            f"pooled={float(row['relative_rms_pooled']):.9e}"
        )


if __name__ == "__main__":
    main()
