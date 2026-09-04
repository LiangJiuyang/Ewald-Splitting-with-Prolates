"""Locations for generated Figure 5 artifacts.

Numerical tables, manifests, diagnostics, and rendered figures are runtime
products. They must stay outside the Git working tree.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def section_output_root(section: str = "redesigned_section5") -> Path:
    """Return the external root used for regenerable Figure 5 artifacts."""

    configured = os.environ.get("ESP_ERROR_BENCH_OUTPUT_DIR")
    base = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir()) / "ewald-splitting-with-prolates-results"
    )
    return base.resolve() / section
