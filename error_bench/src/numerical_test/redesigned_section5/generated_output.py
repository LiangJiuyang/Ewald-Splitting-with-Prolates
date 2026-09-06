"""Locations for generated Figure 5 artifacts.

Numerical tables, manifests, diagnostics, and rendered figures are runtime
products. They must stay outside the Git working tree.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def section_output_root(
    section: str = "redesigned_section5", *, create: bool = False
) -> Path:
    """Return the external root used for regenerable manuscript artifacts."""

    configured = os.environ.get("ESP_ERROR_BENCH_OUTPUT_DIR")
    base = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir()) / "ewald-splitting-with-prolates-results"
    )
    root = base.resolve() / section
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def manifest_path(path: Path, project_root: Path) -> str:
    """Return a portable path for a distributed input or generated artifact."""

    resolved = path.resolve()
    project_root = project_root.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        output_root = section_output_root()
        try:
            relative = resolved.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                f"path is outside both the bundle and generated-output roots: {resolved}"
            ) from error
        return (
            Path("$ESP_ERROR_BENCH_OUTPUT_DIR")
            / output_root.name
            / relative
        ).as_posix()
