#!/usr/bin/env python3
"""Plot the redesigned main-text Figures 2--6.

Figure contract
---------------
Backend
    Python/matplotlib only.
Archetype
    Quantitative validation grid.
Core evidence chain
    Fig. 2 validates Fourier truncation; Fig. 3 validates matched fixed-
    influence ik and analytical-differentiation mesh estimators; Fig. 4 tests
    the molecular structure-factor correction; Fig. 5 compares fixed-influence
    ik theory and conditional pair/self AD theory frozen from 25 pilot
    frames, each with independent validation, and shows a fixed-G,
    fixed-P=5 PPPM measurement baseline;
    Fig. 6
    tests the grid--window trade-off under a common PSWF split.
Export contract
    7.1-inch double-column figures; editable PDF/SVG text; 300 dpi PNG
    previews and 600 dpi LZW-compressed TIFF files; sans-serif typography;
    all plotted values and rendered files are read from or written to the
    external result directory selected by ESP_ERROR_BENCH_OUTPUT_DIR.

The retired ``fig5_target_screening`` calculation remains callable through
``figure5()`` for archival/SI use, but it is not generated, inventoried, or
listed as a current main-text figure.  The current Figure-5 output uses the
matching ``fig5_pppm_efficiency`` basename.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generated_output import section_output_root  # noqa: E402

OUTPUT_ROOT = section_output_root()
REPO = HERE.parents[2]
WINDOW_BENCHMARK = REPO / "numerical_examples" / "window_upsampling_optimized_benchmark"
WINDOW_M16 = REPO / "numerical_examples" / "window_upsampling_m16_extension"
WINDOW_TIMING = REPO / "numerical_examples" / "window_upsampling_timing_benchmark"
GAUSSIAN_PPPM_CONTROL = REPO / "numerical_examples" / "gaussian_pppm_window_control"
LARGE_WATER_WINDOW = REPO / "numerical_examples" / "large_water_window_upsampling"
LARGE_WATER_AD_P4_EXTENSION = LARGE_WATER_WINDOW / "ad" / "pppm_p4_extension.csv"
LARGE_WATER_AD_P5_EXTENSION = LARGE_WATER_WINDOW / "ad" / "pppm_p5_extension.csv"

# Colorblind-safe Okabe--Ito family, with a neutral measurement color.
COLORS = {
    "black": "#242424",
    "gray": "#6C757D",
    "light_gray": "#D9DEE3",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
FIGURE3_MAX_DISPLAYED_ORDER = 9
FIGURE3_DIRECT_LABEL_FONTSIZE = 7.4
FIGURE3_IK_YLIMS = {
    "P": (5.0e-6, 2.0),
    "sigma_up": (1.5e-4, 8.0e-2),
    "c_spread": (7.0e-6, 8.0e-2),
}
# Retained as source metadata for archived dense-tick plots; the refined
# continuous axes below no longer rotate every sampled abscissa.
FIGURE3_DENSE_TICK_ROTATION = 55
FIGURE2_SIZE = (7.1, 5.0)
FIGURE2_XLIM = (7.6, 17.3)
FIGURE2_MAJOR_XTICKS = (8.0, 10.0, 12.0, 14.0, 16.0)
FIGURE2_MINOR_XTICKS = (9.0, 11.0, 13.0, 15.0, 17.0)
FIGURE2_ABSOLUTE_YLIM = (3.5e-6, 5.0e-2)
FIGURE4_SIZE = (7.1, 3.0)
FIGURE4_MECHANISM_ORDER = 5
FIGURE4_MECHANISM_XLIM = (0.0, 4.0)
FIGURE_SI_SPECTRUM_SIZE = (7.1, 3.0)
FIGURE6_ESP_ORDERS = (4, 5, 6, 7, 8)
FIGURE6_PPPM_ORDER = 5
FIGURE6_PPPM_PANEL_A_INPUT_TOLERANCE = 9.0e-7
FIGURE6_PPPM_PANEL_A_FIXED_GEWALD = 0.37735282
FIGURE6_PPPM_PANEL_B_INPUT_TOLERANCE = 1.0e-4
FIGURE6_TARGETS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
FIGURE6_PPPM_MESHES = (12, 15, 18, 20, 24, 32, 36, 40, 48, 64, 80, 96)
FIGURE6_ESP_PANEL_A_MESHES = (16, 18, 20, 24, 32, 36, 40, 48, 64, 80)
FIGURE6_ESP_PANEL_B_ORDERS = {
    1.0e-3: 5,
    1.0e-4: 6,
    1.0e-5: 8,
    1.0e-6: 10,
}
FIGURE6_ESP_PANEL_B_MESHES = {
    1.0e-3: (12,),
    1.0e-4: (12, 15),
    1.0e-5: (12, 15),
    1.0e-6: (12, 15, 16, 18),
}
FIGURE6_FIXED_BAND_TARGET = 1.0e-5
# Direct, mesh-free PSWF Fourier truncation at c_split=14.471 on the same
# 51 SPC/E frames, normalized by the common Ewald-reference force norm.
FIGURE6_FOURIER_TAIL_REFERENCE_RELATIVE = 4.211764706776317e-6
FIGURE7_TARGET = 1.0e-4
FIGURE7_TIGHT_TARGET = 1.0e-5
FIGURE7_CALIBRATION_MESHES = (16, 20)
FIGURE7_RESOLVED_MESHES = (24, 30, 36, 40)
FIGURE7_GAUSSIAN_MESHES = (16, 20, 24, 30, 36, 40, 48)
FIGURE7_NOMINAL_BAND_EDGE_M = 12.024 * 48.0 / (math.pi * 9.0)
FIGURE7_TIMING_ORDER = (
    "pswf_m24_p5_c12.024",
    "bspline_m24_p6",
    "gaussian_pppm_m30_p6",
    "pswf_m30_p5_c12.024",
    "bspline_m30_p5",
    "bspline_m36_p4",
)
FIGURE7_TIMING_ARCHIVED_ONLY: tuple[str, ...] = ()
FIGURE7_SIZE = (7.1, 3.25)
LARGE_WATER_WINDOW_FIGURE_SIZE = (7.1, 6.30)
FIGURE_SI_WINDOW_SIZE = (7.1, 3.15)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.4,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.4,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def read_rows(filename: str) -> list[dict[str, str]]:
    """Read a source-data CSV without changing its text fields."""
    path = OUTPUT_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_path_rows(path: Path) -> list[dict[str, str]]:
    """Read an explicitly identified source table outside the figure directory."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_fixed_order_pppm_extension(
    path: Path,
    *,
    order: int,
    actual_mesh: int,
) -> dict[str, str | float]:
    """Reduce four AD PPPM extension frames to a Figure-6 source record.

    The main scan records one selected order per grid.  A fixed-order
    extension is included only after the next lower order has been measured
    on the same four frozen validation configurations.
    """

    raw = [
        row
        for row in read_path_rows(path)
        if int(row["actual_nx"]) == actual_mesh
        and int(row["actual_ny"]) == actual_mesh
        and int(row["actual_nz"]) == actual_mesh
        and int(row["order"]) == order
    ]
    if len(raw) != 4:
        raise ValueError(
            f"expected four fixed-order extension frames for P={order}, "
            f"M={actual_mesh}; found {len(raw)} in {path}"
        )
    frames = sorted(int(row["frame"]) for row in raw)
    if frames != [1250, 1500, 1750, 2000]:
        raise ValueError(
            f"unexpected extension validation frames for P={order}, "
            f"M={actual_mesh}: {frames}"
        )
    if not all(
        row["differentiation"] == "ad"
        and row["target"] == "1e-5"
        and row["window"] == "pppm"
        for row in raw
    ):
        raise ValueError(f"extension convention mismatch in {path}")
    errors = [value(row, "rms_relative_force_error") for row in raw]
    first = raw[0]
    target = value(first, "target_force_error")
    return {
        "differentiation": "ad",
        "target": "1e-5",
        "target_force_error": first["target_force_error"],
        "window": "pppm",
        "requested_mesh": first["requested_mesh"],
        "actual_nx": first["actual_nx"],
        "actual_ny": first["actual_ny"],
        "actual_nz": first["actual_nz"],
        "actual_grid_volume": first["actual_grid_volume"],
        "order": str(order),
        "c_split": "",
        "c_spread": "",
        "gamma": "",
        "sigma_up": "",
        "selection_rms_relative_force_error": "",
        "validation_rms_relative_force_error_mean": statistics.fmean(errors),
        "validation_rms_relative_force_error_std": statistics.stdev(errors),
        "validation_rms_relative_force_error_max": max(errors),
        "validation_rms_relative_force_error_min": min(errors),
        "all_validation_frames_feasible": "1" if max(errors) <= target else "0",
        "n_validation_frames": "4",
        "_source_record": (
            f"fixed-order AD PPPM extension, P={order}, "
            "four frozen validation configurations"
        ),
    }


def value(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        result = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing/non-numeric {key!r}: {raw!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {key!r}: {raw!r}")
    return result


def optional_value(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or raw.strip() == "":
        return None
    try:
        result = float(raw)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def set_log_y(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())


def set_log_xy(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    set_log_y(ax)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())


def light_y_grid(ax: plt.Axes) -> None:
    ax.grid(axis="y", which="major", color="#E8EBEE", linewidth=0.55, zorder=0)


def panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.115,
    y: float = 1.08,
    fontsize: float = 9.2,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=COLORS["black"],
    )


def save_figure(fig: plt.Figure, stem: str, title: str) -> None:
    """Write vector originals plus journal- and screen-resolution rasters."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": title, "Creator": "plot_redesigned_main_figures.py"}
    fig.savefig(OUTPUT_ROOT / f"{stem}.pdf", dpi=600, metadata=metadata)
    fig.savefig(OUTPUT_ROOT / f"{stem}.svg", metadata={"Title": title})
    fig.savefig(OUTPUT_ROOT / f"{stem}.png", dpi=300, metadata={"Title": title})
    fig.savefig(
        OUTPUT_ROOT / f"{stem}.tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def errorbar(
    ax: plt.Axes,
    x: Iterable[float],
    y: Iterable[float],
    yerr: Iterable[float],
    *,
    color: str,
    marker: str,
    label: str,
    zorder: int = 4,
    filled: bool = True,
    marker_size: float = 4.8,
    linestyle: str | tuple = "-",
    alpha: float = 1.0,
) -> None:
    ax.errorbar(
        list(x),
        list(y),
        yerr=list(yerr),
        fmt=marker,
        linestyle=linestyle,
        color=color,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        markeredgewidth=0.8,
        markersize=marker_size,
        linewidth=1.3,
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.75,
        label=label,
        zorder=zorder,
        alpha=alpha,
    )


def uncertainty_band(
    ax: plt.Axes,
    x: Iterable[float],
    y: Iterable[float],
    yerr: Iterable[float],
    *,
    color: str,
    marker: str,
    label: str,
    zorder: int = 4,
    filled: bool = True,
    marker_size: float = 4.8,
    linestyle: str | tuple = "-",
    alpha: float = 1.0,
    band_alpha: float = 0.11,
) -> None:
    """Plot a central curve with a shaded plus/minus-one-SEM region."""
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    sem_values = np.asarray(list(yerr), dtype=float)
    lower = y_values - sem_values
    if ax.get_yscale() == "log":
        lower = np.maximum(lower, np.finfo(float).tiny)
    ax.fill_between(
        x_values,
        lower,
        y_values + sem_values,
        color=color,
        alpha=band_alpha * alpha,
        linewidth=0,
        zorder=max(0, zorder - 2),
    )
    ax.plot(
        x_values,
        y_values,
        marker=marker,
        linestyle=linestyle,
        color=color,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        markeredgewidth=0.8,
        markersize=marker_size,
        linewidth=1.3,
        label=label,
        zorder=zorder,
        alpha=alpha,
    )


def uncertainty_legend_proxy(
    ax: plt.Axes,
    *,
    color: str,
    marker: str,
    label: str,
    filled: bool,
    marker_size: float,
    linestyle: str = "none",
) -> None:
    """Add a capped-errorbar legend key without drawing data on the axes."""
    ax.errorbar(
        [np.nan],
        [np.nan],
        yerr=np.array([[1.0], [1.0]]),
        fmt=marker,
        linestyle=linestyle,
        color=color,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        markeredgewidth=0.85,
        markersize=marker_size,
        linewidth=1.35,
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.75,
        label=label,
    )


def figure2() -> None:
    """Test homogeneous and structure-aware Fourier-tail estimates."""
    # Reuse the manuscript-wide colorblind-safe palette while keeping measured
    # references and predictions distinct through both color and marker shape.
    figure_colors = {
        "ink": COLORS["black"],
        "blue": COLORS["blue"],
        "ochre": COLORS["orange"],
        "slab": COLORS["green"],
        "teal": COLORS["green"],
        "homogeneous": COLORS["blue"],
        "grid": "#E8EAEC",
    }
    random_rows = sorted(
        read_rows("fig2_fourier_truncation_summary.csv"),
        key=lambda row: value(row, "c_split"),
    )
    slab_rows = sorted(
        read_rows("fig2_slab_fourier_truncation_summary.csv"),
        key=lambda row: value(row, "c_split"),
    )
    x = np.array([value(row, "c_split") for row in random_rows])
    if not np.array_equal(
        x, np.array([value(row, "c_split") for row in slab_rows])
    ):
        raise ValueError("Figure 2 random and slab c_split grids differ")
    eq46 = np.array(
        [value(row, "eq46_discrete_abs_rms_kcal_per_mol_A") for row in random_rows]
    )
    eq56 = np.array(
        [value(row, "eq56_closed_abs_rms_kcal_per_mol_A") for row in random_rows]
    )

    random_measured = np.array(
        [value(row, "measured_pooled_abs_rms_kcal_per_mol_A") for row in random_rows]
    )
    slab_measured = np.array(
        [value(row, "measured_pooled_abs_rms_kcal_per_mol_A") for row in slab_rows]
    )
    random_sem = np.array(
        [value(row, "measured_pooled_jackknife_se_kcal_per_mol_A") for row in random_rows]
    )
    slab_sem = np.array(
        [value(row, "measured_pooled_jackknife_se_kcal_per_mol_A") for row in slab_rows]
    )
    random_ratio = eq46 / random_measured
    slab_ratio = eq46 / slab_measured
    random_ratio_sem = random_ratio * random_sem / random_measured
    slab_ratio_sem = slab_ratio * slab_sem / slab_measured

    water_prediction_rows = sorted(
        read_rows("fig2_water_fourier_prediction_summary.csv"),
        key=lambda row: value(row, "c_split"),
    )
    water_holdout_rows = sorted(
        (
            row
            for row in read_rows("fig2_water_fourier_reference_summary.csv")
            if row["partition"] == "holdout"
        ),
        key=lambda row: value(row, "c_split"),
    )
    water_x = np.array([value(row, "c_split") for row in water_prediction_rows])
    if not np.array_equal(water_x, x) or not np.array_equal(
        water_x, np.array([value(row, "c_split") for row in water_holdout_rows])
    ):
        raise ValueError("Figure 2 water prediction/reference c_split grids differ")
    water_measured = np.array(
        [
            value(row, "measured_pooled_abs_rms_kcal_per_mol_A")
            for row in water_holdout_rows
        ]
    )
    water_measured_sem = np.array(
        [
            value(row, "delete_one_balanced_block_jackknife_sem_kcal_per_mol_A")
            for row in water_holdout_rows
        ]
    )
    water_homogeneous = np.array(
        [
            value(row, "homogeneous_prediction_kcal_per_mol_A")
            for row in water_prediction_rows
        ]
    )
    water_intramolecular = np.array(
        [
            value(row, "intramolecular_only_prediction_kcal_per_mol_A")
            for row in water_prediction_rows
        ]
    )
    water_pilot = np.array(
        [value(row, "pilot_sq_prediction_kcal_per_mol_A") for row in water_prediction_rows]
    )
    water_pilot_sem = np.array(
        [
            value(row, "pilot_sq_block5_sem_kcal_per_mol_A")
            for row in water_prediction_rows
        ]
    )
    water_homogeneous_ratio = water_homogeneous / water_measured
    water_intramolecular_ratio = water_intramolecular / water_measured
    water_pilot_ratio = water_pilot / water_measured
    water_homogeneous_ratio_sem = (
        water_homogeneous_ratio * water_measured_sem / water_measured
    )
    water_intramolecular_ratio_sem = (
        water_intramolecular_ratio * water_measured_sem / water_measured
    )
    water_pilot_ratio_sem = water_pilot_ratio * np.sqrt(
        (water_pilot_sem / water_pilot) ** 2
        + (water_measured_sem / water_measured) ** 2
    )

    fig, axes = plt.subplots(2, 2, figsize=FIGURE2_SIZE, sharex=True)
    fig.subplots_adjust(
        left=0.092,
        right=0.995,
        bottom=0.105,
        top=0.965,
        wspace=0.27,
        hspace=0.27,
    )
    ax = axes[0, 0]
    set_log_y(ax)
    ax.grid(axis="y", which="major", color=figure_colors["grid"], linewidth=0.45)
    ax.plot(
        x, eq46, "s-", color=figure_colors["blue"],
        markerfacecolor=figure_colors["blue"], markeredgewidth=0.75,
        markersize=3.7, linewidth=1.4,
        label="Discrete sum", zorder=3,
    )
    ax.plot(
        x, eq56, "^--", color=figure_colors["ochre"], markerfacecolor="white",
        markeredgewidth=0.85, markersize=3.9, linewidth=1.35,
        label="Closed form", zorder=2,
    )
    ax.fill_between(
        x,
        random_measured - random_sem,
        random_measured + random_sem,
        color=figure_colors["ink"],
        alpha=0.07,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        x, random_measured, "o", color=figure_colors["ink"],
        markerfacecolor="white", markeredgewidth=0.85, markersize=4.4,
        label="_nolegend_", zorder=5,
    )
    ax.fill_between(
        x,
        slab_measured - slab_sem,
        slab_measured + slab_sem,
        color=figure_colors["slab"],
        alpha=0.08,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        x, slab_measured, "D", color=figure_colors["slab"],
        markerfacecolor="white", markeredgewidth=0.85, markersize=4.2,
        label="_nolegend_", zorder=4,
    )
    uncertainty_legend_proxy(
        ax,
        color=figure_colors["ink"],
        marker="o",
        label="Random charges",
        filled=False,
        marker_size=4.4,
    )
    uncertainty_legend_proxy(
        ax,
        color=figure_colors["slab"],
        marker="D",
        label="Slab-like systems",
        filled=False,
        marker_size=4.2,
    )
    ax.set_ylim(*FIGURE2_ABSOLUTE_YLIM)
    ax.set_ylabel(
        r"Absolute force error (kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$)"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    label_order = [
        "Random charges",
        "Discrete sum",
        "Slab-like systems",
        "Closed form",
    ]
    ax.legend(
        [handles_by_label[label] for label in label_order],
        label_order,
        loc="upper right", bbox_to_anchor=(0.995, 1.015), borderaxespad=0.4,
        ncol=2, fontsize=7.1, handlelength=1.25,
        handletextpad=0.38, columnspacing=0.65, labelspacing=0.50,
    )
    panel_label(ax, "a", x=-0.13, y=1.075, fontsize=10.5)

    ax = axes[0, 1]
    ax.grid(axis="y", which="major", color=figure_colors["grid"], linewidth=0.45)
    ax.axhline(1.0, color="#808080", linestyle=":", linewidth=0.7, zorder=1)
    ax.fill_between(
        x,
        random_ratio - random_ratio_sem,
        random_ratio + random_ratio_sem,
        color=figure_colors["ink"],
        alpha=0.10,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        x, random_ratio, "o-",
        color=figure_colors["ink"], markerfacecolor="white", markeredgewidth=0.85,
        markersize=4.3, linewidth=1.4,
        label="_nolegend_", zorder=4,
    )
    ax.fill_between(
        x,
        slab_ratio - slab_ratio_sem,
        slab_ratio + slab_ratio_sem,
        color=figure_colors["slab"],
        alpha=0.12,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        x, slab_ratio, "D-",
        color=figure_colors["slab"], markerfacecolor="white", markeredgewidth=0.85,
        markersize=4.1, linewidth=1.4,
        label="_nolegend_", zorder=5,
    )
    uncertainty_legend_proxy(
        ax,
        color=figure_colors["ink"],
        marker="o",
        label="Random charges",
        filled=False,
        marker_size=4.3,
        linestyle="-",
    )
    uncertainty_legend_proxy(
        ax,
        color=figure_colors["slab"],
        marker="D",
        label="Slab-like systems",
        filled=False,
        marker_size=4.1,
        linestyle="-",
    )
    ax.set_ylim(0.94, 1.04)
    ax.set_ylabel("Prediction / measurement")
    ax.legend(
        loc="upper right", bbox_to_anchor=(0.995, 1.015), borderaxespad=0.4,
        fontsize=7.1, handlelength=1.3, handletextpad=0.4,
        labelspacing=0.50,
    )
    panel_label(ax, "b", x=-0.13, y=1.075, fontsize=10.5)

    ax = axes[1, 0]
    set_log_y(ax)
    ax.grid(axis="y", which="major", color=figure_colors["grid"], linewidth=0.45)
    ax.plot(
        water_x,
        water_homogeneous,
        "s--",
        color=figure_colors["homogeneous"],
        markerfacecolor="white",
        markeredgewidth=0.85,
        markersize=3.7,
        linewidth=1.35,
        label=r"$S_q=Q/V$",
        zorder=2,
    )
    ax.plot(
        water_x,
        water_intramolecular,
        "^--",
        color=figure_colors["ochre"],
        markerfacecolor="white",
        markeredgewidth=0.85,
        markersize=3.9,
        linewidth=1.35,
        label=r"Rigid $S_q$",
        zorder=3,
    )
    ax.plot(
        water_x,
        water_pilot,
        "o-",
        color=figure_colors["teal"],
        markerfacecolor=figure_colors["teal"],
        markeredgewidth=0.75,
        markersize=3.6,
        linewidth=1.4,
        label=r"Measured $S_q$",
        zorder=4,
    )
    ax.fill_between(
        water_x,
        water_measured - water_measured_sem,
        water_measured + water_measured_sem,
        color=figure_colors["ink"],
        alpha=0.07,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        water_x,
        water_measured,
        "o",
        color=figure_colors["ink"],
        markerfacecolor="white",
        markeredgewidth=0.85,
        markersize=4.6,
        label="_nolegend_",
        zorder=5,
    )
    uncertainty_legend_proxy(
        ax,
        color=figure_colors["ink"],
        marker="o",
        label="SPC/E water",
        filled=False,
        marker_size=4.6,
    )
    ax.set_ylim(*FIGURE2_ABSOLUTE_YLIM)
    ax.set_ylabel(
        r"Absolute force error (kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$)"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    label_order = [
        "SPC/E water",
        r"Rigid $S_q$",
        r"$S_q=Q/V$",
        r"Measured $S_q$",
    ]
    ax.legend(
        [handles_by_label[label] for label in label_order],
        label_order,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.015),
        borderaxespad=0.4,
        ncol=2,
        fontsize=7.1,
        handlelength=1.25,
        handletextpad=0.38,
        columnspacing=0.75,
        labelspacing=0.50,
    )
    panel_label(ax, "c", x=-0.13, y=1.075, fontsize=10.5)

    ax = axes[1, 1]
    ax.grid(axis="y", which="major", color=figure_colors["grid"], linewidth=0.45)
    ax.axhline(1.0, color="#808080", linestyle=":", linewidth=0.7, zorder=1)
    ax.fill_between(
        water_x,
        water_homogeneous_ratio - water_homogeneous_ratio_sem,
        water_homogeneous_ratio + water_homogeneous_ratio_sem,
        color=figure_colors["homogeneous"],
        alpha=0.10,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        water_x,
        water_homogeneous_ratio,
        "s--",
        color=figure_colors["homogeneous"],
        markerfacecolor="white",
        markeredgewidth=0.85,
        markersize=3.7,
        linewidth=1.35,
        label="_nolegend_",
        zorder=3,
    )
    ax.fill_between(
        water_x,
        water_intramolecular_ratio - water_intramolecular_ratio_sem,
        water_intramolecular_ratio + water_intramolecular_ratio_sem,
        color=figure_colors["ochre"],
        alpha=0.11,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        water_x,
        water_intramolecular_ratio,
        "^--",
        color=figure_colors["ochre"],
        markerfacecolor="white",
        markeredgewidth=0.85,
        markersize=3.9,
        linewidth=1.35,
        label="_nolegend_",
        zorder=4,
    )
    ax.fill_between(
        water_x,
        water_pilot_ratio - water_pilot_ratio_sem,
        water_pilot_ratio + water_pilot_ratio_sem,
        color=figure_colors["teal"],
        alpha=0.12,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        water_x,
        water_pilot_ratio,
        "o-",
        color=figure_colors["teal"],
        markerfacecolor=figure_colors["teal"],
        markeredgewidth=0.75,
        markersize=3.6,
        linewidth=1.4,
        label="_nolegend_",
        zorder=5,
    )
    uncertainty_legend_proxy(
        ax,
        color=figure_colors["homogeneous"],
        marker="s",
        label=r"$S_q=Q/V$",
        filled=False,
        marker_size=3.7,
        linestyle="--",
    )
    uncertainty_legend_proxy(
        ax,
        color=figure_colors["ochre"],
        marker="^",
        label=r"Rigid $S_q$",
        filled=False,
        marker_size=3.9,
        linestyle="--",
    )
    uncertainty_legend_proxy(
        ax,
        color=figure_colors["teal"],
        marker="o",
        label=r"Measured $S_q$",
        filled=True,
        marker_size=3.6,
        linestyle="-",
    )
    ax.set_ylim(0.94, 1.04)
    ax.set_ylabel("Prediction / measurement")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 1.015),
        borderaxespad=0.4,
        fontsize=7.1,
        handlelength=1.3,
        handletextpad=0.4,
        labelspacing=0.50,
    )
    panel_label(ax, "d", x=-0.13, y=1.075, fontsize=10.5)

    for ax in axes.flat:
        # c_split is a continuous physical bandlimit.  Keep every datum at its
        # true (nonuniform) abscissa, but use uniformly spaced axis ticks so
        # nearby table entries do not produce overlapping labels.  Equal-width
        # panels make the physical horizontal scale identical in (a)--(d).
        ax.set_xlim(*FIGURE2_XLIM)
        ax.set_xticks(FIGURE2_MAJOR_XTICKS)
        ax.set_xticks(FIGURE2_MINOR_XTICKS, minor=True)
        ax.tick_params(axis="x", labelrotation=0)
    for ax in axes[0, :]:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    for ax in axes[1, :]:
        ax.set_xlabel(r"Splitting bandlimit, $c_{\mathrm{split}}$")
        ax.tick_params(axis="x", which="both", labelbottom=True)
    save_figure(fig, "fig2_fourier_validation", "Fourier-truncation validation")


def figure3() -> None:
    """Compare fixed-ik and total-AD mesh-error estimates with measurements."""
    ik_rows = read_rows("fig3_mesh_validation_source.csv")
    ad_rows = read_rows(
        "lammps_ad_total_validation/fig3_lammps_ad_summary.csv"
    )
    panels = [
        ("P", "Spreading order", "Order sweep"),
        ("sigma_up", r"Upsampling factor, $\sigma_{\mathrm{up}}$", "Upsampling sweep"),
        (
            "c_spread",
            r"Spreading bandlimit, $c_{\mathrm{spread}}$",
            "Spreading bandlimit sweep",
        ),
    ]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.1, 5.0),
        sharex="col",
        sharey=False,
    )
    fig.subplots_adjust(
        left=0.092,
        right=0.995,
        bottom=0.115,
        top=0.84,
        wspace=0.25,
        hspace=0.56,
    )
    # Use matched axes-fraction coordinates for corresponding direct labels in
    # the upper and lower rows.  This keeps panels (a,d) and (c,f) vertically
    # consistent even though their logarithmic y ranges differ.
    direct_label_positions = {
        "P": ((0.78, 0.57), (0.62, 0.23)),
        "c_spread": ((0.62, 0.78), (0.50, 0.20)),
    }
    # Panel (a) needs a small outward horizontal adjustment relative to panel
    # (d): move the upper label right and the lower label left while retaining
    # the matched vertical coordinates of the two order-sweep panels.
    ik_order_label_positions = ((0.82, 0.57), (0.50, 0.23))
    ad_order_label_positions = ((0.78, 0.57), (0.68, 0.23))

    for index, (panel_name, xlabel, title) in enumerate(panels):
        ik_ax = axes[0, index]
        ad_ax = axes[1, index]
        ik_subset = sorted(
            (row for row in ik_rows if row["panel"] == panel_name),
            key=lambda row: value(row, "x"),
        )
        ad_subset = sorted(
            (row for row in ad_rows if row["panel"] == panel_name),
            key=lambda row: value(row, "x"),
        )
        if not ik_subset or not ad_subset:
            raise ValueError(f"Figure 3 panel {panel_name!r} has an empty source sweep")
        if panel_name == "P":
            ik_subset = [
                row for row in ik_subset
                if value(row, "x") <= FIGURE3_MAX_DISPLAYED_ORDER
            ]
            # Keep the archived baseline AD P=10 case as a zeroed-mode
            # diagnostic, but display the same P=4--9 interval in panels
            # (a) and (d) so the two order sweeps remain visually comparable.
            ad_subset = [
                row for row in ad_subset
                if value(row, "x") <= FIGURE3_MAX_DISPLAYED_ORDER
            ]

        set_log_y(ik_ax)
        light_y_grid(ik_ax)
        if panel_name == "sigma_up":
            ik_group_key = lambda row: int(value(row, "order"))
        elif panel_name == "P":
            ik_group_key = lambda row: (
                value(row, "csplit"),
                value(row, "cspread"),
            )
        elif panel_name == "c_spread":
            ik_group_key = lambda row: value(row, "csplit")
        else:  # pragma: no cover - the panel list above is exhaustive
            ik_group_key = lambda row: 0
        ik_group_values = sorted({ik_group_key(row) for row in ik_subset})
        expected_ik_groups = 2
        if len(ik_group_values) != expected_ik_groups:
            raise ValueError(
                f"Figure 3 ik {panel_name} sweep requires two curve groups, "
                f"got {ik_group_values}"
            )
        ik_endpoints: list[tuple[float, float, str]] = []
        for group_index, group_value in enumerate(ik_group_values):
            group = sorted(
                (row for row in ik_subset if ik_group_key(row) == group_value),
                key=lambda row: value(row, "x"),
            )
            ik_x = np.array([value(row, "x") for row in group])
            ik_measured = np.array(
                [value(row, "measured_pooled_rms") for row in group]
            )
            ik_sem = np.array(
                [value(row, "measured_pooled_jackknife_sem") for row in group]
            )
            axial_discrete = np.array(
                [value(row, "axial_discrete") for row in group]
            )
            continuum_axial = np.array(
                [value(row, "continuum_eq67") for row in group]
            )
            continuum_zeroed = np.array(
                [
                    row["continuum_eq67_zeroed"].strip().lower() == "true"
                    for row in group
                ],
                dtype=bool,
            )
            continuum_extrapolated = np.array(
                [
                    row["continuum_eq67_extrapolated"].strip().lower() == "true"
                    for row in group
                ],
                dtype=bool,
            )
            continuum_display = continuum_axial.copy()
            continuum_display[continuum_zeroed] = (
                1.12 * FIGURE3_IK_YLIMS[panel_name][0]
            )
            line_style = "-" if group_index == 0 else (0, (3.2, 1.8))
            line_alpha = 1.0 if group_index == 0 else 0.84
            uncertainty_band(
                ik_ax,
                ik_x,
                ik_measured,
                ik_sem,
                color=COLORS["black"],
                marker="o",
                label="_nolegend_",
                filled=False,
                linestyle=line_style,
                alpha=line_alpha,
                band_alpha=0.10,
                zorder=5 if group_index == 0 else 7,
            )
            if group_index == 0:
                uncertainty_legend_proxy(
                    ik_ax,
                    color=COLORS["black"],
                    marker="o",
                    label="Actual error",
                    filled=False,
                    marker_size=4.8,
                    linestyle="-",
                )
            ik_ax.plot(
                ik_x,
                axial_discrete,
                marker="s",
                linestyle=line_style,
                color=COLORS["blue"],
                markerfacecolor=COLORS["blue"],
                label=(
                    "Discrete estimator" if group_index == 0 else "_nolegend_"
                ),
                zorder=4 if group_index == 0 else 6,
                alpha=line_alpha,
            )
            ik_ax.plot(
                ik_x,
                continuum_display,
                marker="D",
                linestyle=line_style,
                color=COLORS["green"],
                markersize=4.6,
                label=(
                    "Continuous estimator" if group_index == 0 else "_nolegend_"
                ),
                zorder=3 if group_index == 0 else 4,
                alpha=line_alpha,
            )
            if np.any(continuum_zeroed):
                ik_ax.scatter(
                    ik_x[continuum_zeroed],
                    continuum_display[continuum_zeroed],
                    marker="v",
                    s=28,
                    facecolors=COLORS["green"],
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=8,
                )
            if np.any(continuum_extrapolated):
                ik_ax.scatter(
                    ik_x[continuum_extrapolated],
                    continuum_display[continuum_extrapolated],
                    marker="D",
                    s=25,
                    facecolors="white",
                    edgecolors=COLORS["green"],
                    linewidths=1.0,
                    zorder=9,
                )
            if panel_name == "sigma_up":
                group_label = rf"$P={int(group_value)}$"
            elif panel_name == "P":
                group_label = rf"$c={group_value[0]:.3f}$"
            else:
                group_label = rf"$c_{{\rm split}}={group_value:.3f}$"
            ik_endpoints.append(
                (
                    float(ik_x[-1]),
                    float(math.sqrt(ik_measured[-1] * axial_discrete[-1])),
                    group_label,
                )
            )
        ik_ax.set_title(title, pad=4.0)
        ik_ax.set_ylim(FIGURE3_IK_YLIMS[panel_name])
        panel_label(
            ik_ax, chr(ord("a") + index), x=-0.15, y=1.08, fontsize=10.5
        )
        ik_ax.text(
            0.96,
            0.95,
            r"$\mathrm{i}\mathbf{k}$",
            transform=ik_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.6,
            fontweight="bold",
            color=COLORS["gray"],
            zorder=10,
        )

        for group_index, (x_end, y_end, group_label) in enumerate(ik_endpoints):
            label_style = dict(
                fontsize=FIGURE3_DIRECT_LABEL_FONTSIZE,
                fontweight="bold",
                color=COLORS["gray"],
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.88,
                    pad=0.15,
                ),
                zorder=10,
            )
            if panel_name in direct_label_positions:
                if panel_name == "P":
                    label_x, label_y = ik_order_label_positions[group_index]
                else:
                    label_x, label_y = direct_label_positions[panel_name][group_index]
                ik_ax.text(
                    label_x,
                    label_y,
                    group_label,
                    transform=ik_ax.transAxes,
                    ha="center",
                    va="center",
                    **label_style,
                )
            else:
                ik_ax.annotate(
                    group_label,
                    xy=(x_end, y_end),
                    xytext=(7, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    annotation_clip=False,
                    **label_style,
                )

        invalid_ad = [row for row in ad_subset if row.get("status") != "valid"]
        valid_ad = [row for row in ad_subset if row.get("status") == "valid"]
        if invalid_ad:
            raise ValueError(
                "Figure 3 AD source contains unevaluated cases even though the "
                "production zero-deconvolution convention is now estimator-matched: "
                f"{[row.get('case_id') for row in invalid_ad]}"
            )
        if not valid_ad:
            raise ValueError(f"Figure 3 AD panel {panel_name!r} has no valid cases")
        set_log_y(ad_ax)
        light_y_grid(ad_ax)
        if panel_name in {"P", "sigma_up"}:
            if panel_name == "sigma_up":
                ad_group_key = lambda row: int(value(row, "order"))
                expected_ad_groups = [5, 8]
            else:
                ad_group_key = lambda row: (
                    value(row, "csplit"),
                    value(row, "cspread"),
                )
                expected_ad_groups = [(12.024, 12.024), (16.894, 16.894)]
            ad_group_values = sorted({ad_group_key(row) for row in valid_ad})
            if ad_group_values != expected_ad_groups:
                raise ValueError(
                    f"Figure 3 AD {panel_name} sweep requires groups "
                    f"{expected_ad_groups}, got {ad_group_values}"
                )
            ad_endpoints: list[tuple[float, float, str]] = []
            for group_index, group_value in enumerate(ad_group_values):
                group = sorted(
                    (row for row in valid_ad if ad_group_key(row) == group_value),
                    key=lambda row: value(row, "x"),
                )
                ad_x = np.array([value(row, "x") for row in group])
                ad_measured = np.array(
                    [value(row, "measured_total_pooled_rms") for row in group]
                )
                ad_sem = np.array(
                    [value(row, "measured_total_jackknife_sem") for row in group]
                )
                ad_prediction = np.array(
                    [value(row, "predicted_total_quadrature") for row in group]
                )
                line_style = "-" if group_index == 0 else (0, (3.2, 1.8))
                line_alpha = 1.0 if group_index == 0 else 0.84
                uncertainty_band(
                    ad_ax,
                    ad_x,
                    ad_measured,
                    ad_sem,
                    color=COLORS["black"],
                    marker="o",
                    label="_nolegend_",
                    filled=False,
                    linestyle=line_style,
                    alpha=line_alpha,
                    band_alpha=0.10,
                    zorder=4 if group_index == 0 else 6,
                )
                if group_index == 0:
                    uncertainty_legend_proxy(
                        ad_ax,
                        color=COLORS["black"],
                        marker="o",
                        label="Actual error",
                        filled=False,
                        marker_size=4.8,
                        linestyle="-",
                    )
                ad_ax.plot(
                    ad_x,
                    ad_prediction,
                    marker="s",
                    linestyle=line_style,
                    color=COLORS["blue"],
                    markerfacecolor=COLORS["blue"],
                    markeredgewidth=0.8,
                    label=(
                        "Discrete estimator" if group_index == 0 else "_nolegend_"
                    ),
                    zorder=3 if group_index == 0 else 5,
                    alpha=line_alpha,
                )
                group_label = (
                    rf"$P={int(group_value)}$"
                    if panel_name == "sigma_up"
                    else (
                        rf"$c={group_value[0]:.3f}$"
                    )
                )
                ad_endpoints.append(
                    (
                        float(ad_x[-1]),
                        float(math.sqrt(ad_measured[-1] * ad_prediction[-1])),
                        group_label,
                    )
                )
            for group_index, (x_end, y_end, group_label) in enumerate(ad_endpoints):
                label_style = dict(
                    fontsize=FIGURE3_DIRECT_LABEL_FONTSIZE,
                    fontweight="bold",
                    color=COLORS["gray"],
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.88,
                        pad=0.15,
                    ),
                    zorder=10,
                )
                if panel_name == "P":
                    label_x, label_y = ad_order_label_positions[group_index]
                    ad_ax.text(
                        label_x,
                        label_y,
                        group_label,
                        transform=ad_ax.transAxes,
                        ha="center",
                        va="center",
                        **label_style,
                    )
                else:
                    ad_ax.annotate(
                        group_label,
                        xy=(x_end, y_end),
                        xytext=(7, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        annotation_clip=False,
                        **label_style,
                    )
        else:
            ad_group_values = sorted(
                {value(row, "csplit") for row in valid_ad}
            )
            if ad_group_values != [12.024, 16.894]:
                raise ValueError(
                    "Figure 3 AD c_spread sweep requires c_split groups "
                    f"[12.024, 16.894], got {ad_group_values}"
                )
            ad_endpoints = []
            for group_index, group_value in enumerate(ad_group_values):
                group = sorted(
                    (
                        row
                        for row in valid_ad
                        if math.isclose(value(row, "csplit"), group_value)
                    ),
                    key=lambda row: value(row, "x"),
                )
                ad_x = np.array([value(row, "x") for row in group])
                ad_measured = np.array(
                    [value(row, "measured_total_pooled_rms") for row in group]
                )
                ad_sem = np.array(
                    [value(row, "measured_total_jackknife_sem") for row in group]
                )
                ad_prediction = np.array(
                    [value(row, "predicted_total_quadrature") for row in group]
                )
                group_orders = sorted({int(value(row, "order")) for row in group})
                if len(group_orders) != 1:
                    raise ValueError(
                        "Figure 3 AD c_spread curve must use one fixed P, "
                        f"got {group_orders} for c_split={group_value}"
                    )
                line_style = "-" if group_index == 0 else (0, (3.2, 1.8))
                line_alpha = 1.0 if group_index == 0 else 0.84
                uncertainty_band(
                    ad_ax,
                    ad_x,
                    ad_measured,
                    ad_sem,
                    color=COLORS["black"],
                    marker="o",
                    label="_nolegend_",
                    filled=False,
                    linestyle=line_style,
                    alpha=line_alpha,
                    band_alpha=0.10,
                    zorder=4 if group_index == 0 else 6,
                )
                if group_index == 0:
                    uncertainty_legend_proxy(
                        ad_ax,
                        color=COLORS["black"],
                        marker="o",
                        label="Actual error",
                        filled=False,
                        marker_size=4.8,
                        linestyle="-",
                    )
                ad_ax.plot(
                    ad_x,
                    ad_prediction,
                    marker="s",
                    linestyle=line_style,
                    color=COLORS["blue"],
                    markerfacecolor=COLORS["blue"],
                    markeredgewidth=0.8,
                    label=(
                        "Discrete estimator"
                        if group_index == 0
                        else "_nolegend_"
                    ),
                    zorder=3 if group_index == 0 else 5,
                    alpha=line_alpha,
                )
                ad_endpoints.append(
                    (
                        float(ad_x[-1]),
                        float(math.sqrt(ad_measured[-1] * ad_prediction[-1])),
                        (
                            rf"$c_{{\rm split}}={group_value:.3f}$"
                        ),
                    )
                )
            for group_index, (x_end, y_end, group_label) in enumerate(ad_endpoints):
                label_x, label_y = direct_label_positions["c_spread"][group_index]
                ad_ax.text(
                    label_x,
                    label_y,
                    group_label,
                    transform=ad_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=FIGURE3_DIRECT_LABEL_FONTSIZE,
                    fontweight="bold",
                    color=COLORS["gray"],
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.88,
                        pad=0.15,
                    ),
                    zorder=10,
                )
        ad_ax.set_ylim(
            {
                "P": (1.5e-4, 1.5e1),
                "sigma_up": (3.0e-3, 8.0e-2),
                "c_spread": (2.0e-4, 3.5e-1),
            }[panel_name]
        )
        ad_ax.set_xlabel(xlabel)
        panel_label(
            ad_ax, chr(ord("d") + index), x=-0.15, y=1.08, fontsize=10.5
        )
        ad_ax.text(
            0.96,
            0.95,
            "AD",
            transform=ad_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.6,
            fontweight="bold",
            color=COLORS["gray"],
            zorder=10,
        )

        if panel_name == "P":
            all_x = sorted(
                {value(row, "x") for row in ik_subset + valid_ad}
            )
            ad_ax.set_xticks(all_x)
            ad_ax.set_xlim(min(all_x) - 0.35, max(all_x) + 0.35)
            ik_ax.set_ylabel(
                r"Absolute mesh-aliasing error" "\n"
                r"(kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$)"
            )
            ad_ax.set_ylabel(
                r"Absolute mesh-aliasing error" "\n"
                r"(kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$)"
            )
        elif panel_name == "sigma_up":
            ad_ax.set_xlim(0.9, 2.55)
            ad_ax.set_xticks([1.0, 1.3, 1.6, 1.9, 2.2, 2.5])
            ad_ax.set_xticks([1.15, 1.45, 1.75, 2.05, 2.35], minor=True)
        else:
            ad_ax.set_xlim(9.0, 17.3)
            ad_ax.set_xticks([10.0, 12.0, 14.0, 16.0])
            ad_ax.set_xticks([9.0, 11.0, 13.0, 15.0, 17.0], minor=True)

    ik_handles, ik_labels = axes[0, 0].get_legend_handles_labels()
    ik_by_label = dict(zip(ik_labels, ik_handles))
    ik_labels = [
        "Actual error",
        "Discrete estimator",
        "Continuous estimator",
    ]
    ik_handles = [ik_by_label[label] for label in ik_labels]
    fig.legend(
        ik_handles,
        ik_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.954),
        ncol=3,
        fontsize=8.2,
        handlelength=1.55,
        columnspacing=0.85,
    )
    ad_handles, ad_labels = axes[1, 0].get_legend_handles_labels()
    ad_by_label = dict(zip(ad_labels, ad_handles))
    ad_labels = [
        "Actual error",
        "Discrete estimator",
    ]
    ad_handles = [ad_by_label[label] for label in ad_labels]
    fig.legend(
        ad_handles,
        ad_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.478),
        ncol=2,
        fontsize=8.2,
        handlelength=1.7,
        columnspacing=1.0,
    )
    save_figure(fig, "fig3_mesh_validation", "Particle-mesh error validation")


def figure_si_charge_spectrum() -> None:
    """Archive physical-spectrum and estimator-mechanism diagnostics in the SI."""
    spectrum = read_rows("fig4_charge_spectrum_source.csv")
    mechanism = sorted(
        (
            row
            for row in read_rows("fig4_k_resolved_variance_source.csv")
            if int(round(value(row, "order"))) == FIGURE4_MECHANISM_ORDER
            and value(row, "k_upper_Angstrom-1")
            <= FIGURE4_MECHANISM_XLIM[1] + 1.0e-12
        ),
        key=lambda row: value(row, "k_center_Angstrom-1"),
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SI_SPECTRUM_SIZE)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.92, wspace=0.34)
    ax = axes[0]
    set_log_y(ax)
    light_y_grid(ax)
    spectrum_styles = {
        "random charges": (COLORS["gray"], "o", "Random charges"),
        "SPC/E water (frames 1--25)": (
            COLORS["blue"], "s", "SPC/E water"
        ),
    }
    for system in ("random charges", "SPC/E water (frames 1--25)"):
        subset = sorted(
            (row for row in spectrum if row["system"] == system),
            key=lambda row: value(row, "k_center"),
        )
        color, marker, label = spectrum_styles[system]
        x = np.asarray([value(row, "k_center") for row in subset])
        y = np.asarray([value(row, "cq_volume_mean") for row in subset])
        sem = np.asarray([value(row, "cq_volume_sem") for row in subset])
        lower = np.maximum(y - sem, np.finfo(float).tiny)
        upper = y + sem
        ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            label=label,
            linewidth=1.05,
            markersize=3.2,
            markerfacecolor=(color if system.startswith("SPC") else "white"),
            markeredgecolor=color,
            markeredgewidth=0.65,
            markevery=5,
            zorder=4 if system.startswith("SPC") else 3,
        )
        baseline = value(subset[0], "charge_squared_density")
        ax.axhline(
            baseline, color=color, linewidth=0.72, linestyle=":", alpha=0.62, zorder=1
        )
    ax.set_xlim(0.0, 10.05)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_ylim(5.0e-6, 7.0e-2)
    ax.set_xlabel(r"Physical wave number, $|\mathbf{k}|$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(
        r"$C_q(k)=\langle|\rho(\mathbf{k})|^2\rangle/V$"
        "\n"
        r"($e^2\,\mathrm{\AA}^{-3}$)"
    )
    ax.legend(
        loc="lower right", fontsize=6.9, handlelength=1.5, labelspacing=0.35
    )
    panel_label(ax, "a", y=1.07)

    ax = axes[1]
    light_y_grid(ax)
    k = np.asarray([value(row, "k_center_Angstrom-1") for row in mechanism])
    normalization = value(mechanism[0], "normalization_chi2")
    homogeneous_contribution = 100.0 * np.asarray(
        [value(row, "homogeneous_total_chi2") for row in mechanism]
    ) / normalization
    corrected_contribution = 100.0 * np.asarray(
        [value(row, "corrected_total_chi2") for row in mechanism]
    ) / normalization
    corrected_block_sem = 100.0 * np.asarray(
        [value(row, "corrected_total_block5_sem_chi2") for row in mechanism]
    ) / normalization
    corrected_sampling_sem = 100.0 * np.asarray(
        [
            value(row, "corrected_total_importance_sampling_sem_chi2")
            for row in mechanism
        ]
    ) / normalization
    corrected_display_sem = np.sqrt(
        corrected_block_sem**2 + corrected_sampling_sem**2
    )
    ax.fill_between(
        k,
        np.maximum(corrected_contribution - corrected_display_sem, 0.0),
        corrected_contribution + corrected_display_sem,
        step="mid",
        color=COLORS["blue"],
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    ax.step(
        k,
        homogeneous_contribution,
        where="mid",
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.35,
        label=r"$S_q=Q/V$",
        zorder=3,
    )
    ax.step(
        k,
        corrected_contribution,
        where="mid",
        color=COLORS["blue"],
        linewidth=1.45,
        label=r"Measured $S_q$",
        zorder=4,
    )
    ax.set_xlim(*FIGURE4_MECHANISM_XLIM)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_ylim(0.0, 22.0)
    ax.set_xlabel(
        r"Wave number entering $S_q$, $k_{\rho}$ "
        r"($\mathrm{\AA}^{-1}$)"
    )
    ax.set_ylabel(
        "Estimator-variance contribution\n"
        r"(\% of $S_q=Q/V$ total per $0.2$ $\mathrm{\AA}^{-1}$)"
    )
    ax.legend(
        loc="upper right", fontsize=6.7, handlelength=1.6, labelspacing=0.35
    )
    panel_label(ax, "b", y=1.07)
    save_figure(fig, "figS_charge_spectrum", "Charge-correlation diagnostics")


def figure4() -> None:
    """Validate homogeneous, rigid-molecule, and measured-Sq estimates."""
    correction = sorted(
        read_rows("fig4_sq_correction_source.csv"),
        key=lambda row: value(row, "order"),
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGURE4_SIZE)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.19, top=0.92, wspace=0.34)
    ax = axes[0]
    set_log_y(ax)
    light_y_grid(ax)
    order = np.array([int(round(value(row, "order"))) for row in correction])
    measured = np.array([value(row, "measured_holdout_pooled_rms") for row in correction])
    measured_sem = np.array([value(row, "measured_holdout_block5_sem") for row in correction])
    homogeneous = np.array([value(row, "homogeneous_prediction") for row in correction])
    rigid = np.array([value(row, "rigid_molecule_prediction") for row in correction])
    rigid_sampling_sem = np.array(
        [value(row, "rigid_molecule_importance_sampling_sem") for row in correction]
    )
    corrected = np.array([value(row, "pilot_sq_corrected_prediction") for row in correction])
    corrected_sem = np.array([value(row, "pilot_sq_corrected_block5_sem") for row in correction])

    uncertainty_band(
        ax,
        order,
        measured,
        measured_sem,
        color=COLORS["black"],
        marker="o",
        label="_nolegend_",
        filled=False,
        band_alpha=0.10,
        zorder=5,
    )
    uncertainty_legend_proxy(
        ax,
        color=COLORS["black"],
        marker="o",
        label="Actual error",
        filled=False,
        marker_size=4.8,
        linestyle="-",
    )
    ax.plot(
        order,
        homogeneous,
        "^--",
        color=COLORS["orange"],
        markerfacecolor="white",
        markeredgewidth=0.8,
        label=r"$S_q=Q/V$",
        zorder=2,
    )
    ax.plot(
        order,
        rigid,
        marker="D",
        linestyle="-.",
        color=COLORS["green"],
        markerfacecolor="white",
        markeredgecolor=COLORS["green"],
        markeredgewidth=0.8,
        markersize=4.8,
        linewidth=1.3,
        label=r"Rigid $S_q$",
        zorder=3,
    )
    ax.plot(
        order,
        corrected,
        marker="s",
        linestyle="-",
        color=COLORS["blue"],
        markerfacecolor=COLORS["blue"],
        markeredgecolor=COLORS["blue"],
        markeredgewidth=0.8,
        markersize=4.8,
        linewidth=1.3,
        label=r"Measured $S_q$",
        zorder=4,
    )
    ax.set_xticks(order)
    ax.set_ylim(2.5e-4, 6.0e-1)
    ax.set_xlabel(r"Spreading order, $P$")
    ax.set_ylabel(
        "Absolute mesh-aliasing error\n"
        r"(kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$)"
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = [
        "Actual error",
        r"$S_q=Q/V$",
        r"Rigid $S_q$",
        r"Measured $S_q$",
    ]
    ax.legend(
        [by_label[label] for label in ordered_labels], ordered_labels,
        loc="upper right", fontsize=7.6, handlelength=1.5, labelspacing=0.35,
    )
    panel_label(ax, "a", y=1.07, fontsize=10.5)

    ax = axes[1]
    light_y_grid(ax)
    sampling_sem = np.array(
        [value(row, "pilot_sq_importance_sampling_sem") for row in correction]
    )
    homogeneous_ratio = homogeneous / measured
    rigid_ratio = rigid / measured
    corrected_ratio = corrected / measured
    homogeneous_ratio_sem = homogeneous_ratio * measured_sem / measured
    rigid_ratio_sem = rigid_ratio * np.sqrt(
        (rigid_sampling_sem / rigid) ** 2 + (measured_sem / measured) ** 2
    )
    corrected_prediction_sem = np.sqrt(corrected_sem**2 + sampling_sem**2)
    corrected_ratio_sem = corrected_ratio * np.sqrt(
        (corrected_prediction_sem / corrected) ** 2
        + (measured_sem / measured) ** 2
    )
    ax.axhline(1.0, color=COLORS["gray"], linewidth=0.8, linestyle=":", zorder=1)
    uncertainty_band(
        ax,
        order,
        homogeneous_ratio,
        homogeneous_ratio_sem,
        color=COLORS["orange"],
        marker="^",
        label="_nolegend_",
        filled=False,
        linestyle="--",
        band_alpha=0.10,
        zorder=3,
    )
    uncertainty_legend_proxy(
        ax,
        color=COLORS["orange"],
        marker="^",
        label=r"$S_q=Q/V$",
        filled=False,
        marker_size=4.8,
        linestyle="--",
    )
    uncertainty_band(
        ax,
        order,
        rigid_ratio,
        rigid_ratio_sem,
        color=COLORS["green"],
        marker="D",
        label="_nolegend_",
        filled=False,
        linestyle="-.",
        band_alpha=0.11,
        zorder=3,
    )
    uncertainty_legend_proxy(
        ax,
        color=COLORS["green"],
        marker="D",
        label=r"Rigid $S_q$",
        filled=False,
        marker_size=4.8,
        linestyle="-.",
    )
    uncertainty_band(
        ax,
        order,
        corrected_ratio,
        corrected_ratio_sem,
        color=COLORS["blue"],
        marker="s",
        label="_nolegend_",
        band_alpha=0.11,
        zorder=4,
    )
    uncertainty_legend_proxy(
        ax,
        color=COLORS["blue"],
        marker="s",
        label=r"Measured $S_q$",
        filled=True,
        marker_size=4.8,
        linestyle="-",
    )
    ax.set_xlim(3.7, 8.3)
    ax.set_xticks(order)
    ax.set_ylim(0.8, 2.0)
    ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_xlabel(r"Spreading order, $P$")
    ax.set_ylabel("Estimator / actual error")
    ax.legend(
        loc="upper right", fontsize=7.6, handlelength=1.5, labelspacing=0.35
    )
    panel_label(ax, "b", y=1.07, fontsize=10.5)
    save_figure(fig, "fig4_sq_correction", "Charge-structure-factor correction")


def load_figure5_validation(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Prefer final LAMMPS full-force holdout, otherwise use matched mesh."""
    def balanced_holdout_sem() -> np.ndarray:
        frame_rows = read_rows(
            "water_fixed_ik_targets/water_fixed_ik_force_error_by_frame.csv"
        )
        by_target: dict[float, list[dict[str, str]]] = {}
        for item in frame_rows:
            if item.get("partition") != "holdout":
                continue
            by_target.setdefault(value(item, "target_relative_error"), []).append(item)
        result = []
        for selected in rows:
            members = sorted(
                by_target[value(selected, "target")],
                key=lambda item: int(round(value(item, "frame_zero_based"))),
            )
            if len(members) != 26:
                raise ValueError("Figure 5 fixed-ik holdout must contain 26 frames")
            blocks = (
                members[0:5], members[5:10], members[10:15],
                members[15:20], members[20:26],
            )
            block_pooled = [
                math.sqrt(
                    math.fsum(value(item, "sum_squared_force_difference") for item in block)
                    / math.fsum(value(item, "sum_squared_reference_force") for item in block)
                )
                for block in blocks
            ]
            result.append(statistics.stdev(block_pooled) / math.sqrt(len(block_pooled)))
        return np.asarray(result, dtype=float)

    full_y = [optional_value(row, "holdout_full_lammps_relative") for row in rows]
    if all(item is not None for item in full_y):
        return (
            np.asarray(full_y, dtype=float),
            balanced_holdout_sem(),
            r"Held-out LAMMPS full force, fixed-$ik$ (pooled RMS; block-pooled SEM)",
        )

    summary_path = HERE / "water_fixed_ik_targets" / "water_fixed_ik_holdout_summary.csv"
    if summary_path.is_file():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            summary = list(csv.DictReader(handle))
        by_target = {value(row, "target_relative_error"): row for row in summary}
        try:
            matched = [by_target[value(row, "target")] for row in rows]
        except KeyError:
            matched = []
        if len(matched) == len(rows):
            return (
                np.array([value(row, "pooled_rms_relative_error") for row in matched]),
                balanced_holdout_sem(),
                r"Held-out LAMMPS full force, fixed-$ik$ (pooled RMS; block-pooled SEM)",
            )

    return (
        np.array([value(row, "holdout_matched_mesh_relative") for row in rows]),
        np.array([value(row, "holdout_matched_mesh_block5_sem") for row in rows]),
        r"Held-out matched pure mesh (pooled RMS; block-RMS SEM)",
    )


def load_figure5_fixed_ik_extension() -> tuple[dict[str, str], dict[str, str]]:
    """Load the independently frozen fixed-ik 1e-6 screen and holdout."""
    root = HERE / "fixed_ik_1e6_extension"
    frozen_path = root / "fixed_ik_1e6_pilot_frozen.json"
    companion_path = root / "fixed_ik_1e6_pilot_frozen.json.sha256"
    if not frozen_path.is_file() or not companion_path.is_file():
        raise FileNotFoundError("fixed-ik 1e-6 frozen selection is missing")
    expected = companion_path.read_text().split()[0]
    import hashlib

    actual = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("fixed-ik 1e-6 frozen selection SHA256 mismatch")
    frozen = json.loads(frozen_path.read_text())
    selected = frozen.get("selected_rows", [])
    if len(selected) != 1:
        raise ValueError("fixed-ik 1e-6 freeze must contain one selected row")
    row = {key: str(item) for key, item in selected[0].items()}
    if str(row.get("holdout_used_in_selection", "")).lower() != "false":
        raise ValueError("fixed-ik 1e-6 holdout entered parameter selection")
    if not math.isclose(value(row, "target"), 1.0e-6):
        raise ValueError("fixed-ik extension target is not 1e-6")
    if (int(round(value(row, "mesh_actual"))), int(round(value(row, "order")))) != (18, 9):
        raise ValueError("fixed-ik 1e-6 frozen tuple changed")

    summary = read_path_rows(root / "fixed_ik_1e6_holdout_summary.csv")
    if len(summary) != 1 or summary[0].get("partition") != "holdout":
        raise ValueError("fixed-ik 1e-6 extension must have one holdout summary")
    holdout = summary[0]
    if int(round(value(holdout, "n_frames"))) != 26:
        raise ValueError("fixed-ik 1e-6 holdout must contain 26 frames")
    if holdout.get("pooled_target_satisfied", "").lower() != "true":
        raise ValueError("fixed-ik 1e-6 holdout does not satisfy its target")
    if holdout.get("frozen_selection_sha256") != actual:
        raise ValueError("fixed-ik 1e-6 holdout is not linked to the frozen tuple")
    return row, holdout


def load_figure5_ad_transfer() -> list[dict[str, float | str]]:
    """Join selected, adjacent-robustness, and frozen 1e-6 AD cases."""
    root = HERE / "lammps_ad_total_validation"
    components = read_path_rows(root / "water_ad_estimator_components.csv")
    partitions = read_path_rows(root / "water_ad_partition_summary.csv")
    holdout = {
        row["case_id"]: row for row in partitions if row.get("partition") == "holdout"
    }
    joined: list[dict[str, float | str]] = []
    for row in components:
        case_id = row["case_id"]
        if case_id not in holdout:
            raise ValueError(f"missing AD holdout row for {case_id}")
        validation = holdout[case_id]
        joined.append(
            {
                "case_id": case_id,
                "target": value(row, "target_relative_error"),
                "mesh": value(row, "mesh"),
                "order": value(row, "order"),
                "homogeneous": value(row, "homogeneous_total_relative"),
                "pilot": value(row, "pilot_conditioned_total_relative"),
                "pilot_sem": value(row, "pilot_conditioned_block5_sem_relative"),
                "holdout": value(validation, "pooled_total_relative_error"),
                "holdout_sem": value(validation, "block5_sem_total_relative_error"),
                "homogeneous_feasible": row.get("homogeneous_a_priori_feasible", ""),
                "pilot_feasible": row.get("pilot_conditioned_feasible", ""),
                "selection_role": "selected_target_tuple",
            }
        )
    robustness_components = read_path_rows(
        root / "water_ad_robustness_components.csv"
    )
    robustness_partitions = read_path_rows(
        root / "water_ad_robustness_partition_summary.csv"
    )
    robustness_holdout = {
        row["case_id"]: row
        for row in robustness_partitions
        if row.get("partition") == "holdout"
    }
    if len(robustness_components) != 5 or len(robustness_holdout) != 5:
        raise ValueError("Figure 5 expects all five adjacent AD robustness tuples")
    for row in robustness_components:
        case_id = row["case_id"]
        if case_id not in robustness_holdout:
            raise ValueError(f"missing AD robustness holdout row for {case_id}")
        validation = robustness_holdout[case_id]
        joined.append(
            {
                "case_id": case_id,
                "target": value(row, "target_relative_error"),
                "mesh": value(row, "mesh"),
                "order": value(row, "order"),
                "homogeneous": value(row, "homogeneous_total_relative"),
                "pilot": value(row, "pilot_conditioned_total_relative"),
                "pilot_sem": value(row, "pilot_conditioned_block5_sem_relative"),
                "holdout": value(validation, "pooled_total_relative_error"),
                "holdout_sem": value(
                    validation, "block5_sem_total_relative_error"
                ),
                "homogeneous_feasible": row.get(
                    "homogeneous_a_priori_feasible", ""
                ),
                "pilot_feasible": row.get("pilot_conditioned_feasible", ""),
                "selection_role": "adjacent_candidate_robustness",
            }
        )
    extension_components = read_path_rows(
        root / "water_ad_1e6_estimator_component.csv"
    )
    extension_partitions = read_path_rows(
        root / "water_ad_1e6_partition_summary.csv"
    )
    if len(extension_components) != 1 or len(extension_partitions) != 1:
        raise ValueError("Figure 5 expects one independently frozen AD 1e-6 extension")
    extension = extension_components[0]
    extension_holdout = extension_partitions[0]
    if extension_holdout.get("partition") != "holdout":
        raise ValueError("AD 1e-6 extension must report the holdout partition")
    joined.append(
        {
            "case_id": extension["case_id"],
            "target": value(extension, "target_relative_error"),
            "mesh": value(extension, "mesh"),
            "order": value(extension, "order"),
            "homogeneous": value(extension, "homogeneous_total_relative"),
            "pilot": value(extension, "pilot_conditioned_total_relative"),
            "pilot_sem": value(
                extension, "pilot_conditioned_block5_sem_relative"
            ),
            "holdout": value(extension_holdout, "pooled_total_relative_error"),
            "holdout_sem": value(
                extension_holdout, "block5_sem_total_relative_error"
            ),
            "homogeneous_feasible": extension.get(
                "homogeneous_a_priori_feasible", ""
            ),
            "pilot_feasible": extension.get("pilot_conditioned_feasible", ""),
            "scope": "independently pilot-frozen AD-only calibration extension",
            "selection_role": "independently_frozen_target_1e6",
        }
    )
    joined.sort(key=lambda item: float(item["target"]))
    expected_ids = {
        "water_ad_target_1e-3",
        "water_ad_target_1e-4",
        "water_ad_target_1e-5",
        "water_ad_target_1e-6_M18_P9",
        "water_ad_robust_1e-3_M12_P5",
        "water_ad_robust_1e-4_M15_P4",
        "water_ad_robust_1e-4_M15_P5",
        "water_ad_robust_1e-5_M16_P5",
        "water_ad_robust_1e-5_M16_P6",
    }
    if {str(row["case_id"]) for row in joined} != expected_ids:
        raise ValueError("Figure 5 AD case set differs from the audited nine-case set")
    if any(str(row["homogeneous_feasible"]).lower() != "false" for row in joined):
        raise ValueError("Figure 5 expects all homogeneous molecular-AD screens to fail")
    return joined


def figure5() -> str:
    """Contrast fixed-ik screening with prespecified-tuple AD calibration."""
    rows = sorted(
        read_rows("fig5_target_screening_source.csv"),
        key=lambda row: value(row, "target"),
    )
    target = np.array([value(row, "target") for row in rows])
    pilot = np.array([value(row, "pilot_total_relative") for row in rows])
    # The stored total-prediction block SEM is absolute; the pilot force scale
    # converts it to the same relative-error normalization used by the center.
    pilot_sem = np.array(
        [
            value(
                row,
                "pilot_total_block5_sem"
                if optional_value(row, "pilot_total_block5_sem") is not None
                else "pilot_mesh_block5_sem",
            )
            / value(row, "pilot_force_scale")
            for row in rows
        ]
    )
    holdout, holdout_sem, holdout_label = load_figure5_validation(rows)
    extension, extension_holdout = load_figure5_fixed_ik_extension()
    target = np.append(target, value(extension, "target"))
    pilot = np.append(pilot, value(extension, "pilot_total_relative"))
    pilot_sem = np.append(
        pilot_sem, value(extension, "pilot_total_relative_block5_sem")
    )
    holdout = np.append(
        holdout, value(extension_holdout, "pooled_rms_relative_error")
    )
    holdout_sem = np.append(
        holdout_sem,
        value(extension_holdout, "balanced_five_block_pooled_rms_sem"),
    )
    fixed_ik_rows = rows + [extension]
    ordering = np.argsort(target)
    target = target[ordering]
    pilot = pilot[ordering]
    pilot_sem = pilot_sem[ordering]
    holdout = holdout[ordering]
    holdout_sem = holdout_sem[ordering]
    fixed_ik_rows = [fixed_ik_rows[index] for index in ordering]

    ad_rows = load_figure5_ad_transfer()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.25),
        gridspec_kw={"width_ratios": [1.08, 0.92]},
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.89, wspace=0.31)
    ax = axes[0]
    set_log_xy(ax)
    light_y_grid(ax)

    lower, upper = 4.5e-7, 1.8e-3
    diagonal = np.logspace(np.log10(5.5e-7), np.log10(upper), 300)
    ax.fill_between(diagonal, lower, diagonal, color=COLORS["green"], alpha=0.055, zorder=0)
    ax.plot(
        diagonal, diagonal, color=COLORS["gray"], linestyle=":",
        linewidth=0.9, label=r"Boundary, $y=x$", zorder=1,
    )

    ax.errorbar(
        target,
        pilot,
        yerr=pilot_sem,
        fmt="s--",
        color=COLORS["blue"],
        markerfacecolor=COLORS["blue"],
        markeredgewidth=0.8,
        elinewidth=0.75,
        capsize=2.1,
        label=r"$S_q$ estimate (frames 1–25)",
        zorder=4,
    )
    ax.errorbar(
        target,
        holdout,
        yerr=holdout_sem,
        fmt="o-",
        color=COLORS["black"],
        markerfacecolor="white",
        markeredgecolor=COLORS["black"],
        markeredgewidth=0.9,
        elinewidth=0.75,
        capsize=2.1,
        label="Direct calculation (frames 26–51)",
        zorder=5,
    )

    for row, x_point, y_point in zip(fixed_ik_rows, target, holdout):
        order = int(round(value(row, "order")))
        mesh = int(round(value(row, "mesh_actual")))
        target_decade = int(round(math.log10(x_point)))
        placements = {
            -6: ((12, 10), "left", "bottom"),
            -5: ((10, -16), "left", "top"),
            -4: ((8, -15), "left", "top"),
            -3: ((-9, -15), "right", "top"),
        }
        offset, horizontal, vertical = placements[target_decade]
        ax.annotate(
            rf"$P={order}$, $M={mesh}$",
            xy=(x_point, y_point),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal,
            va=vertical,
            fontsize=6.5,
            color=COLORS["gray"],
            arrowprops={
                "arrowstyle": "-", "color": COLORS["gray"],
                "linewidth": 0.45, "shrinkA": 1.5, "shrinkB": 3.0,
            },
        )

    ax.set_xlim(5.5e-7, 1.8e-3)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Target relative RMS force error")
    ax.set_ylabel("Estimated or calculated relative RMS error")
    ax.set_title(r"Fixed-influence $\mathrm{i}\mathbf{k}$ screening", pad=4)
    ax.legend(
        loc="upper left", fontsize=6.5, handlelength=1.55,
        labelspacing=0.32,
    )
    ax.text(
        0.985,
        0.055,
        "Parameters fixed before validation",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=COLORS["gray"],
    )
    panel_label(ax, "a", x=-0.15, y=1.06)

    ax = axes[1]
    set_log_xy(ax)
    light_y_grid(ax)
    lower_ad, upper_ad = 2.5e-7, 2.5e-2
    diagonal_ad = np.logspace(np.log10(lower_ad), np.log10(upper_ad), 300)
    ax.plot(
        diagonal_ad,
        diagonal_ad,
        color=COLORS["gray"],
        linestyle=":",
        linewidth=0.9,
        label=r"Agreement, $y=x$",
        zorder=1,
    )
    homogeneous = np.asarray([float(row["homogeneous"]) for row in ad_rows])
    pilot_ad = np.asarray([float(row["pilot"]) for row in ad_rows])
    pilot_ad_sem = np.asarray([float(row["pilot_sem"]) for row in ad_rows])
    holdout_ad = np.asarray([float(row["holdout"]) for row in ad_rows])
    holdout_ad_sem = np.asarray([float(row["holdout_sem"]) for row in ad_rows])
    robustness_mask = np.asarray(
        [row["selection_role"] == "adjacent_candidate_robustness" for row in ad_rows]
    )
    highlighted_mask = ~robustness_mask
    ax.errorbar(
        homogeneous[robustness_mask],
        holdout_ad[robustness_mask],
        yerr=holdout_ad_sem[robustness_mask],
        fmt="^",
        linestyle="none",
        color=COLORS["orange"],
        markerfacecolor="white",
        markeredgecolor=COLORS["orange"],
        markeredgewidth=0.75,
        markersize=4.2,
        elinewidth=0.55,
        capsize=1.6,
        alpha=0.42,
        zorder=2,
    )
    ax.errorbar(
        pilot_ad[robustness_mask],
        holdout_ad[robustness_mask],
        xerr=pilot_ad_sem[robustness_mask],
        yerr=holdout_ad_sem[robustness_mask],
        fmt="s",
        linestyle="none",
        color=COLORS["blue"],
        markerfacecolor=COLORS["sky"],
        markeredgecolor=COLORS["blue"],
        markeredgewidth=0.55,
        markersize=4.0,
        elinewidth=0.55,
        capsize=1.6,
        alpha=0.48,
        zorder=3,
    )
    ax.errorbar(
        homogeneous[highlighted_mask],
        holdout_ad[highlighted_mask],
        yerr=holdout_ad_sem[highlighted_mask],
        fmt="^",
        linestyle="none",
        color=COLORS["orange"],
        markerfacecolor="white",
        markeredgecolor=COLORS["orange"],
        markeredgewidth=0.9,
        elinewidth=0.7,
        capsize=2.0,
        label="Homogeneous estimate",
        zorder=3,
    )
    ax.errorbar(
        pilot_ad[highlighted_mask],
        holdout_ad[highlighted_mask],
        xerr=pilot_ad_sem[highlighted_mask],
        yerr=holdout_ad_sem[highlighted_mask],
        fmt="s",
        linestyle="none",
        color=COLORS["blue"],
        markerfacecolor=COLORS["blue"],
        markeredgecolor=COLORS["black"],
        markeredgewidth=0.45,
        elinewidth=0.7,
        capsize=2.0,
        label="Finite-band calibration",
        zorder=5,
    )
    for row in ad_rows:
        if row["selection_role"] == "adjacent_candidate_robustness":
            continue
        ax.annotate(
            rf"$10^{{{int(round(math.log10(float(row['target']))))}}}$",
            xy=(float(row["pilot"]), float(row["holdout"])),
            xytext=(4, -8),
            textcoords="offset points",
            fontsize=6.4,
            color=COLORS["gray"],
        )
    ax.set_xlim(lower_ad, 8.0e-2)
    ax.set_ylim(lower_ad, upper_ad)
    ax.set_xlabel("AD estimate or calibration value")
    ax.set_ylabel("Calculated AD relative RMS error")
    ax.set_title("AD calibration and validation", pad=4)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0], [0], linestyle="none", marker="o", markersize=4.0,
            markerfacecolor=COLORS["light_gray"], markeredgecolor=COLORS["gray"],
            alpha=0.58,
        )
    )
    labels.append("Adjacent candidates (lighter)")
    ax.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.82),
        fontsize=6.2, handlelength=1.40, labelspacing=0.27,
    )
    ax.text(
        0.04,
        0.96,
        "Calibration, not parameter selection",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=COLORS["gray"],
    )
    panel_label(ax, "b", x=-0.17, y=1.06)
    save_figure(fig, "fig5_target_screening", "Prediction and validation at target accuracy")
    return "full LAMMPS" if "full force" in holdout_label else "matched pure mesh fallback"


def load_esp_all_frames() -> tuple[list[dict[str, str]], str]:
    manifest_path = HERE / "water_fixed_ik_targets" / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    force_hash_by_case = {
        str(case["case_id"]): str(case["dump_sha256"])
        for case in manifest["cases"]
    }

    def attach_force_hashes(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        for row in rows:
            case_id = row.get("case_id", "")
            digest = force_hash_by_case.get(case_id, "")
            if len(digest) != 64:
                raise ValueError(f"missing force-dump SHA-256 for ESP case {case_id!r}")
            row["force_dump_sha256"] = digest
        return rows

    def attach_common_block_sem(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        frame_file = HERE / "water_fixed_ik_targets" / "water_fixed_ik_force_error_by_frame.csv"
        with frame_file.open(newline="", encoding="utf-8") as handle:
            frame_rows = list(csv.DictReader(handle))
        by_case: dict[str, list[dict[str, str]]] = {}
        for frame in frame_rows:
            by_case.setdefault(frame["case_id"], []).append(frame)
        for row in rows:
            expected_frames = int(row["n_frames"])
            frames = sorted(
                (
                    frame
                    for frame in by_case.get(row["case_id"], [])
                    if expected_frames == 51 or frame.get("partition") == "holdout"
                ),
                key=lambda frame: int(frame["frame_zero_based"]),
            )
            if len(frames) != expected_frames:
                raise ValueError(
                    f"ESP case {row['case_id']} has {len(frames)} frame rows, "
                    f"expected {row['n_frames']}"
                )
            errors = [value(frame, "rms_relative_force_error") for frame in frames]
            nblocks = len(errors) // 5
            block_rms = [
                math.sqrt(
                    math.fsum(error * error for error in errors[5 * index : 5 * index + 5])
                    / 5.0
                )
                for index in range(nblocks)
            ]
            if nblocks < 2:
                raise ValueError(f"too few five-frame blocks for {row['case_id']}")
            row["common_block5_rms_sem"] = str(
                statistics.stdev(block_rms) / math.sqrt(nblocks)
            )
            row["common_block5_rms_count"] = str(nblocks)
        return rows

    partition_file = HERE / "water_fixed_ik_targets" / "water_fixed_ik_partition_summary.csv"
    if partition_file.is_file():
        with partition_file.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("partition") == "all"]
        if rows:
            return attach_common_block_sem(attach_force_hashes(rows)), "51 frames"
    holdout_file = HERE / "water_fixed_ik_targets" / "water_fixed_ik_holdout_summary.csv"
    with holdout_file.open(newline="", encoding="utf-8") as handle:
        return attach_common_block_sem(
            attach_force_hashes(list(csv.DictReader(handle)))
        ), "26-frame holdout"


def parse_grid(text: str) -> tuple[int, int, int]:
    fields = text.replace("x", " ").split()
    if len(fields) == 1:
        fields *= 3
    if len(fields) != 3:
        raise ValueError(f"grid must contain one or three dimensions: {text!r}")
    grid = tuple(int(field) for field in fields)
    if any(dimension <= 0 for dimension in grid):
        raise ValueError(f"grid dimensions must be positive: {text!r}")
    return grid  # type: ignore[return-value]


def write_figure6_plot_source(
    esp_panel_a: list[dict[str, str]],
    esp_panel_a_selection: dict[int, dict[str, str]],
    pppm_panel_a: list[dict[str, str]],
    pppm_by_target: dict[float, dict[str, str]],
    esp_ik_by_target: dict[float, dict[str, str]],
    esp_ad_by_target: dict[float, dict[str, str]],
    targets: list[float],
    ik_volume_ratios: list[float],
    ad_volume_ratios: list[float],
) -> None:
    """Write the prediction/validation rows plotted in current Figure 5.

    The output basename retains its historical ``fig6`` prefix so existing
    manuscript links are not broken.
    """
    fieldnames = [
        "panel",
        "series",
        "method",
        "differentiation",
        "influence_function",
        "pppm_input_tolerance",
        "pppm_gewald_inverse_A",
        "pppm_gewald_mode",
        "candidate_id",
        "selection_status",
        "selection_scope",
        "order",
        "target_relative_rms",
        "epsilon_split",
        "epsilon_spread",
        "csplit",
        "cspread",
        "sigma_up",
        "requested_nx",
        "requested_ny",
        "requested_nz",
        "actual_nx",
        "actual_ny",
        "actual_nz",
        "actual_grid_points",
        "relative_rms",
        "relative_rms_block5_sem",
        "predicted_total_relative_rms",
        "predicted_total_relative_block5_sem",
        "predicted_total_relative_importance_sampling_sem",
        "validation_relative_rms",
        "validation_relative_rms_balanced_block5_sem",
        "pilot_frames",
        "holdout_frames",
        "evidence_role",
        "force_dump_sha256",
        "passes_target",
        "fourier_tail_reference_relative_rms",
        "pppm_candidate_id",
        "esp_candidate_id",
        "pppm_force_dump_sha256",
        "esp_force_dump_sha256",
        "pppm_order",
        "esp_order",
        "pppm_relative_rms",
        "esp_relative_rms",
        "grid_point_ratio_pppm_over_esp",
        "pppm_actual_nx",
        "pppm_actual_ny",
        "pppm_actual_nz",
        "pppm_actual_grid_points",
        "esp_actual_nx",
        "esp_actual_ny",
        "esp_actual_nz",
        "esp_actual_grid_points",
        "data_scope",
        "selection_rule",
    ]
    output: list[dict[str, object]] = []
    selected_case_status = {
        row.get("case_id", row.get("candidate_id", "")): (
            row["selection_status"], row["selection_scope"]
        )
        for row in esp_panel_a_selection.values()
    }
    for row in sorted(
        esp_panel_a,
        key=lambda item: (int(item["order"]), int(item["actual_grid_points"])),
    ):
        case_id = row.get("case_id", row.get("candidate_id", ""))
        status, scope = selected_case_status.get(case_id, ("", ""))
        output.append(
            {
                "panel": "a",
                "series": f"ESP P={int(round(value(row, 'order')))}",
                "method": "ESP fixed-influence ik",
                "differentiation": "ik",
                "influence_function": "fixed Fourier multiplier",
                "candidate_id": case_id,
                "selection_status": status,
                "selection_scope": scope,
                "order": int(round(value(row, "order"))),
                "target_relative_rms": value(row, "target_relative_rms"),
                "epsilon_split": value(row, "epsilon_split"),
                "epsilon_spread": value(row, "epsilon_spread"),
                "csplit": value(row, "csplit"),
                "cspread": value(row, "cspread"),
                "sigma_up": value(row, "sigma_up"),
                "requested_nx": int(round(value(row, "actual_nx"))),
                "requested_ny": int(round(value(row, "actual_nx"))),
                "requested_nz": int(round(value(row, "actual_nx"))),
                "actual_nx": int(round(value(row, "actual_nx"))),
                "actual_ny": int(round(value(row, "actual_nx"))),
                "actual_nz": int(round(value(row, "actual_nx"))),
                "actual_grid_points": int(round(value(row, "actual_grid_points"))),
                "relative_rms": value(row, "validation_relative_rms"),
                "relative_rms_block5_sem": value(
                    row, "validation_relative_rms_balanced_block5_sem"
                ),
                "predicted_total_relative_rms": value(
                    row, "predicted_total_relative_rms"
                ),
                "predicted_total_relative_block5_sem": value(
                    row, "predicted_total_relative_block5_sem"
                ),
                "predicted_total_relative_importance_sampling_sem": value(
                    row, "predicted_total_relative_importance_sampling_sem"
                ),
                "validation_relative_rms": value(row, "validation_relative_rms"),
                "validation_relative_rms_balanced_block5_sem": value(
                    row, "validation_relative_rms_balanced_block5_sem"
                ),
                "pilot_frames": int(round(value(row, "pilot_frames"))),
                "holdout_frames": int(round(value(row, "holdout_frames"))),
                "evidence_role": (
                    "pilot prediction line plus independent holdout marker"
                ),
                "force_dump_sha256": row.get("force_dump_sha256", ""),
                "passes_target": row["validation_passes_target"],
                "fourier_tail_reference_relative_rms": FIGURE6_FOURIER_TAIL_REFERENCE_RELATIVE,
                "pppm_candidate_id": "",
                "esp_candidate_id": case_id,
                "pppm_force_dump_sha256": "",
                "esp_force_dump_sha256": row.get("force_dump_sha256", ""),
                "pppm_order": "",
                "esp_order": int(round(value(row, "order"))),
                "pppm_relative_rms": "",
                "esp_relative_rms": value(row, "validation_relative_rms"),
                "grid_point_ratio_pppm_over_esp": "",
                "pppm_actual_nx": "",
                "pppm_actual_ny": "",
                "pppm_actual_nz": "",
                "pppm_actual_grid_points": "",
                "esp_actual_nx": "",
                "esp_actual_ny": "",
                "esp_actual_nz": "",
                "esp_actual_grid_points": "",
                "data_scope": (
                    "prediction: frames 1-25; validation: nonoverlapping "
                    "frames 26-51"
                ),
                "selection_rule": (
                    "P=5--8: smallest scanned actual grid whose pilot-only "
                    "predicted total error meets 1e-5, then checked on holdout; "
                    "P=4: predicted target not met, so the marked source row "
                    "is only the lowest predicted error in the scanned set"
                ),
            }
        )

    pppm_target_1e5 = min(
        (
            row for row in pppm_panel_a
            if value(row, "validation_relative_rms") <= FIGURE6_FIXED_BAND_TARGET
        ),
        key=lambda row: int(row["actual_grid_points"]),
    )["candidate_id"]
    for row in sorted(pppm_panel_a, key=lambda item: int(item["actual_nx"])):
        selected = row["candidate_id"] == pppm_target_1e5
        output.append(
            {
                "panel": "a",
                "series": "PPPM fixed-G P=5",
                "method": "PPPM fixed-G optimal-influence ik",
                "differentiation": "ik",
                "influence_function": "Hockney--Eastwood optimal influence",
                "pppm_input_tolerance": value(row, "pppm_input_tolerance"),
                "pppm_gewald_inverse_A": value(
                    row, "pppm_fixed_gewald_inverse_A"
                ),
                "pppm_gewald_mode": row["pppm_gewald_selection"],
                "candidate_id": row["candidate_id"],
                "selection_status": "target_met" if selected else "",
                "selection_scope": (
                    "smallest fixed-G, fixed-P=5 PPPM grid meeting 1e-5"
                    if selected else ""
                ),
                "order": FIGURE6_PPPM_ORDER,
                "target_relative_rms": FIGURE6_FIXED_BAND_TARGET,
                "epsilon_split": "",
                "epsilon_spread": "",
                "csplit": "",
                "cspread": "",
                "sigma_up": "",
                "requested_nx": int(round(value(row, "actual_nx"))),
                "requested_ny": int(round(value(row, "actual_nx"))),
                "requested_nz": int(round(value(row, "actual_nx"))),
                "actual_nx": int(round(value(row, "actual_nx"))),
                "actual_ny": int(round(value(row, "actual_nx"))),
                "actual_nz": int(round(value(row, "actual_nx"))),
                "actual_grid_points": int(round(value(row, "actual_grid_points"))),
                "relative_rms": value(row, "validation_relative_rms"),
                "relative_rms_block5_sem": value(
                    row, "validation_relative_rms_balanced_block5_sem"
                ),
                "predicted_total_relative_rms": "",
                "predicted_total_relative_block5_sem": "",
                "predicted_total_relative_importance_sampling_sem": "",
                "validation_relative_rms": value(row, "validation_relative_rms"),
                "validation_relative_rms_balanced_block5_sem": value(
                    row, "validation_relative_rms_balanced_block5_sem"
                ),
                "pilot_frames": "",
                "holdout_frames": int(round(value(row, "holdout_frames"))),
                "evidence_role": "independent holdout baseline",
                "force_dump_sha256": row["force_dump_sha256"],
                "passes_target": value(row, "validation_relative_rms") <= FIGURE6_FIXED_BAND_TARGET,
                "fourier_tail_reference_relative_rms": "",
                "pppm_candidate_id": row["candidate_id"],
                "esp_candidate_id": "",
                "pppm_force_dump_sha256": row["force_dump_sha256"],
                "esp_force_dump_sha256": "",
                "pppm_order": FIGURE6_PPPM_ORDER,
                "esp_order": "",
                "pppm_relative_rms": value(row, "validation_relative_rms"),
                "esp_relative_rms": "",
                "grid_point_ratio_pppm_over_esp": "",
                "pppm_actual_nx": int(round(value(row, "actual_nx"))),
                "pppm_actual_ny": int(round(value(row, "actual_nx"))),
                "pppm_actual_nz": int(round(value(row, "actual_nx"))),
                "pppm_actual_grid_points": int(round(value(row, "actual_grid_points"))),
                "esp_actual_nx": "",
                "esp_actual_ny": "",
                "esp_actual_nz": "",
                "esp_actual_grid_points": "",
                "data_scope": "nonoverlapping validation frames 26-51",
                "selection_rule": (
                    "smallest measured holdout actual grid volume meeting 1e-5 "
                    "at fixed G_ewald"
                ),
            }
        )

    for differentiation, selections, ratios in (
        ("ik", esp_ik_by_target, ik_volume_ratios),
        ("ad", esp_ad_by_target, ad_volume_ratios),
    ):
        for target, volume_ratio in zip(targets, ratios):
            pppm_row = pppm_by_target[target]
            esp_row = selections[target]
            requested = int(round(value(pppm_row, "requested_mesh")))
            esp_case_id = esp_row.get("case_id", esp_row.get("candidate_id", ""))
            esp_hash = esp_row["force_dump_sha256"]
            esp_error_key = "pooled_rms_relative_error"
            output.append(
                {
                    "panel": "b",
                    "series": f"ESP {differentiation.upper()} grid reduction",
                    "method": "fixed-P=5 PPPM / target-specific-order ESP",
                    "differentiation": differentiation,
                    "influence_function": (
                        "fixed Fourier multiplier"
                        if differentiation == "ik"
                        else "production AD Fourier/spreading multiplier"
                    ),
                    "pppm_input_tolerance": FIGURE6_PPPM_PANEL_B_INPUT_TOLERANCE,
                    "pppm_gewald_inverse_A": value(pppm_row, "g_ewald_inverse_A"),
                    "pppm_gewald_mode": "auto-balanced separately for each mesh",
                    "candidate_id": esp_case_id,
                    "selection_status": "target_met",
                    "selection_scope": "smallest declared measured ESP grid and fixed-P=5 PPPM grid meeting target",
                    "order": "",
                    "target_relative_rms": target,
                    "epsilon_split": value(esp_row, "epsilon_split"),
                    "epsilon_spread": value(esp_row, "epsilon_spread"),
                    "csplit": value(esp_row, "csplit"),
                    "cspread": value(esp_row, "cspread"),
                    "sigma_up": value(esp_row, "sigma_up"),
                    "requested_nx": requested,
                    "requested_ny": requested,
                    "requested_nz": requested,
                    "actual_nx": "",
                    "actual_ny": "",
                    "actual_nz": "",
                    "actual_grid_points": "",
                    "relative_rms": "",
                    "relative_rms_block5_sem": "",
                    "force_dump_sha256": esp_hash,
                    "passes_target": "True",
                    "fourier_tail_reference_relative_rms": "",
                    "pppm_candidate_id": pppm_row["candidate_id"],
                    "esp_candidate_id": esp_case_id,
                    "pppm_force_dump_sha256": pppm_row["force_dump_sha256"],
                    "esp_force_dump_sha256": esp_hash,
                    "pppm_order": FIGURE6_PPPM_ORDER,
                    "esp_order": int(round(value(esp_row, "order"))),
                    "pppm_relative_rms": value(pppm_row, "relative_rms_pooled"),
                    "esp_relative_rms": value(esp_row, esp_error_key),
                    "grid_point_ratio_pppm_over_esp": volume_ratio,
                    "pppm_actual_nx": int(round(value(pppm_row, "actual_nx"))),
                    "pppm_actual_ny": int(round(value(pppm_row, "actual_ny"))),
                    "pppm_actual_nz": int(round(value(pppm_row, "actual_nz"))),
                    "pppm_actual_grid_points": int(round(value(pppm_row, "actual_grid_points"))),
                    "esp_actual_nx": int(round(value(esp_row, "actual_nx"))),
                    "esp_actual_ny": int(round(value(esp_row, "actual_ny"))),
                    "esp_actual_nz": int(round(value(esp_row, "actual_nz"))),
                    "esp_actual_grid_points": int(round(value(esp_row, "actual_grid_points"))),
                    "data_scope": "same 51 frames and tight Ewald reference for both methods",
                    "selection_rule": "PPPM fixed P=5 with G_ewald auto-balanced separately for each mesh; ESP fixed target-specific P; smallest declared measured actual grid volume meeting target",
                }
            )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "fig5_pppm_efficiency_plot_source.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


def figure6_archived_two_panel() -> tuple[int, int, list[float], list[float], int]:
    """Generate the archived two-panel Figure 5 comparison."""
    prediction_validation = read_rows(
        "fig5_fixed_band_prediction_validation.csv"
    )
    pppm_all = read_rows("fig6_pppm_order_scan_source.csv")
    pppm_core = read_rows("fig6_pppm_order_scan_deduplicated.csv")
    pppm_tail = read_path_rows(
        HERE / "pppm_extended_tail" / "fig6_pppm_extended_tail_source.csv"
    )
    esp_scan = read_rows(
        "water_fixed_ik_p_scan/fixed_ik_p_scan_summary.csv"
    )
    esp_selection_source = read_rows(
        "water_fixed_ik_p_scan/fixed_ik_p_scan_selection.csv"
    )
    esp_ad_scan = read_path_rows(
        HERE
        / "lammps_ad_total_validation"
        / "water_ad_grid_reduction_summary.csv"
    )
    esp_ad_selection_source = read_path_rows(
        HERE
        / "lammps_ad_total_validation"
        / "water_ad_grid_reduction_selection.csv"
    )

    unique_keys = {
        (int(round(value(row, "order"))), row["force_dump_sha256"])
        for row in pppm_core
    }
    if len(unique_keys) != len(pppm_core):
        raise ValueError("builder-produced PPPM table is not force-hash unique")
    if any(len(row["force_dump_sha256"]) != 64 for row in pppm_core):
        raise ValueError("builder-produced PPPM table contains an invalid force hash")
    pppm_p5 = [
        row
        for row in pppm_core + pppm_tail
        if int(round(value(row, "order"))) == FIGURE6_PPPM_ORDER
        and int(round(value(row, "actual_nx"))) <= max(FIGURE6_PPPM_MESHES)
    ]
    if tuple(sorted(int(round(value(row, "actual_nx"))) for row in pppm_p5)) != FIGURE6_PPPM_MESHES:
        raise ValueError("fixed-P=5 PPPM scan does not cover the declared grid set")
    if len({row["force_dump_sha256"] for row in pppm_p5}) != len(pppm_p5):
        raise ValueError("fixed-P=5 PPPM scan contains duplicate force dumps")

    panel_a = [
        row for row in prediction_validation
        if row["method"] == "ESP fixed-influence ik"
    ]
    for row in panel_a:
        row["case_id"] = row["candidate_id"]
    if len(panel_a) != len(FIGURE6_ESP_ORDERS) * len(FIGURE6_ESP_PANEL_A_MESHES):
        raise ValueError("fixed-band ESP panel-a scan is incomplete")
    if any(
        int(row["pilot_frames"]) != 25
        or int(row["holdout_frames"]) != 26
        or not math.isclose(value(row, "csplit"), 14.471, abs_tol=1.0e-12)
        or not math.isclose(value(row, "cspread"), 14.471, abs_tol=1.0e-12)
        or not math.isclose(
            value(row, "target_relative_rms"),
            FIGURE6_FIXED_BAND_TARGET,
            abs_tol=1.0e-15,
        )
        for row in panel_a
    ):
        raise ValueError("panel-a ESP rows violate the fixed-band split-sample contract")

    panel_a_selections: dict[int, dict[str, str]] = {}
    for order in FIGURE6_ESP_ORDERS:
        subset = [row for row in panel_a if int(row["order"]) == order]
        if tuple(sorted(int(row["actual_nx"]) for row in subset)) != FIGURE6_ESP_PANEL_A_MESHES:
            raise ValueError(f"P={order} fixed-band grid sequence is incomplete")
        feasible = [
            row for row in subset
            if value(row, "predicted_total_relative_rms") <= FIGURE6_FIXED_BAND_TARGET
        ]
        if order == 4:
            if feasible:
                raise ValueError("P=4 must be retained as a target-not-met scan")
            expected = min(
                subset,
                key=lambda row: (
                    value(row, "predicted_total_relative_rms"),
                    int(row["actual_grid_points"]),
                ),
            )
            expected["selection_status"] = "target_not_met"
            expected["selection_scope"] = "lowest predicted error; target not met"
        else:
            if not feasible:
                raise ValueError(f"P={order} lacks a qualifying panel-a selection")
            expected = min(
                feasible,
                key=lambda row: (
                    int(row["actual_grid_points"]),
                    int(row["actual_nx"]),
                ),
            )
            expected["selection_status"] = "target_met"
            expected["selection_scope"] = (
                "smallest pilot-predicted grid meeting 1e-5; holdout validated"
            )
            if value(expected, "validation_relative_rms") > FIGURE6_FIXED_BAND_TARGET:
                raise ValueError(f"P={order} prediction-selected grid fails holdout")
        panel_a_selections[order] = expected

    legacy_panel_a_selections = {
        int(row["order"]): row
        for row in esp_selection_source if row["panel"] == "a"
    }
    if {
        order: row.get("case_id", row.get("candidate_id", ""))
        for order, row in panel_a_selections.items()
    } != {
        order: row["case_id"] for order, row in legacy_panel_a_selections.items()
    }:
        raise ValueError("pilot-predicted and archived 51-frame panel-a selections differ")

    pppm_panel_a = sorted(
        (
            row for row in prediction_validation
            if row["method"] == "PPPM fixed-G optimal-influence ik"
        ),
        key=lambda row: int(row["actual_nx"]),
    )
    if tuple(int(row["actual_nx"]) for row in pppm_panel_a) != FIGURE6_PPPM_MESHES:
        raise ValueError("holdout PPPM P=5 scan does not cover the declared grid set")
    if any(int(row["holdout_frames"]) != 26 for row in pppm_panel_a):
        raise ValueError("holdout PPPM P=5 scan must use frames 26-51")
    if any(
        not math.isclose(
            value(row, "pppm_input_tolerance"),
            FIGURE6_PPPM_PANEL_A_INPUT_TOLERANCE,
            rel_tol=0.0,
            abs_tol=1.0e-18,
        )
        or not math.isclose(
            value(row, "pppm_fixed_gewald_inverse_A"),
            FIGURE6_PPPM_PANEL_A_FIXED_GEWALD,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for row in pppm_panel_a
    ):
        raise ValueError("panel-a PPPM rows violate the fixed-G calibration contract")
    pppm_panel_a_display = [
        row for row in pppm_panel_a if int(row["actual_nx"]) <= 80
    ]

    targets = list(FIGURE6_TARGETS)
    esp_ik_by_target = {
        value(row, "target_relative_rms"): row
        for row in esp_selection_source if row["panel"] == "b"
    }
    if set(esp_ik_by_target) != set(targets):
        raise ValueError("panel-b ESP-IK selection table has the wrong target set")
    for target in targets:
        expected_order = FIGURE6_ESP_PANEL_B_ORDERS[target]
        expected_meshes = FIGURE6_ESP_PANEL_B_MESHES[target]
        candidates = [
            row for row in esp_scan
            if "panel_b_mixed_order" in row.get("scopes", "").split(";")
            and int(row["order"]) == expected_order
            and math.isclose(value(row, "target_relative_rms"), target)
            and int(row["requested_mesh"]) in expected_meshes
        ]
        if tuple(sorted(int(row["requested_mesh"]) for row in candidates)) != expected_meshes:
            raise ValueError(
                f"ESP target {target:.0e}, P={expected_order} does not cover "
                f"the declared grids {expected_meshes}"
            )
        feasible = [
            row for row in candidates
            if value(row, "pooled_rms_relative_error") <= target
        ]
        expected = min(
            feasible,
            key=lambda row: (
                int(row["actual_grid_points"]),
                int(row["requested_mesh"]),
            ),
        )
        selected = esp_ik_by_target[target]
        if (
            selected["selection_status"] != "target_met"
            or int(selected["order"]) != expected_order
            or selected["case_id"] != expected["case_id"]
            or selected["force_dump_sha256"] != expected["force_dump_sha256"]
        ):
            raise ValueError(f"stale panel-b ESP-IK selection for target {target:.0e}")

    esp_ad_by_target = {
        value(row, "target_relative_rms"): row for row in esp_ad_selection_source
    }
    if set(esp_ad_by_target) != set(targets):
        raise ValueError("panel-b ESP-AD selection table has the wrong target set")
    for target in targets:
        expected_order = FIGURE6_ESP_PANEL_B_ORDERS[target]
        expected_meshes = FIGURE6_ESP_PANEL_B_MESHES[target]
        candidates = [
            row
            for row in esp_ad_scan
            if int(row["order"]) == expected_order
            and math.isclose(value(row, "target_relative_rms"), target)
            and int(row["requested_mesh"]) in expected_meshes
        ]
        if tuple(int(row["requested_mesh"]) for row in candidates) != expected_meshes:
            raise ValueError(
                f"ESP-AD target {target:.0e}, P={expected_order} does not cover "
                f"the declared grids {expected_meshes}"
            )
        feasible = [
            row for row in candidates if value(row, "pooled_rms_relative_error") <= target
        ]
        expected = min(feasible, key=lambda row: int(row["actual_grid_points"]))
        selected = esp_ad_by_target[target]
        if (
            selected["selection_status"] != "target_met"
            or int(selected["order"]) != expected_order
            or selected["case_id"] != expected["case_id"]
            or selected["force_dump_sha256"] != expected["force_dump_sha256"]
        ):
            raise ValueError(f"stale panel-b ESP-AD selection for target {target:.0e}")

    pppm_by_target: dict[float, dict[str, str]] = {}
    for target in targets:
        feasible = [row for row in pppm_p5 if value(row, "relative_rms_pooled") <= target]
        if not feasible:
            raise ValueError(f"fixed-P=5 PPPM scan does not meet target {target:.0e}")
        pppm_by_target[target] = min(
            feasible,
            key=lambda row: (
                int(round(value(row, "actual_grid_points"))),
                int(round(value(row, "requested_mesh"))),
            ),
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.15),
        gridspec_kw={"width_ratios": [1.55, 1.08]},
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.19, top=0.90, wspace=0.31)

    ax = axes[0]
    set_log_y(ax)
    light_y_grid(ax)
    order_style = {
        4: (COLORS["blue"], "o"),
        5: (COLORS["orange"], "s"),
        6: (COLORS["green"], "^"),
        7: (COLORS["vermillion"], "v"),
        8: (COLORS["purple"], "D"),
    }
    for order in FIGURE6_ESP_ORDERS:
        subset = sorted(
            (row for row in panel_a if int(row["order"]) == order),
            key=lambda row: int(row["actual_nx"]),
        )
        color, marker = order_style[order]
        ax.plot(
            [value(row, "actual_nx") for row in subset],
            [value(row, "predicted_total_relative_rms") for row in subset],
            color=color,
            linewidth=1.3,
            label=rf"$P={order}$",
            zorder=3,
        )
        errorbar(
            ax,
            [value(row, "actual_nx") for row in subset],
            [value(row, "validation_relative_rms") for row in subset],
            [
                value(row, "validation_relative_rms_balanced_block5_sem")
                for row in subset
            ],
            color=color,
            marker=marker,
            label="_nolegend_",
            filled=False,
            linestyle="none",
            marker_size=4.3,
            zorder=5,
        )
    errorbar(
        ax,
        [value(row, "actual_nx") for row in pppm_panel_a_display],
        [value(row, "validation_relative_rms") for row in pppm_panel_a_display],
        [
            value(row, "validation_relative_rms_balanced_block5_sem")
            for row in pppm_panel_a_display
        ],
        color=COLORS["black"],
        marker="X",
        label=r"PPPM fixed-$G$ holdout, $P=5$",
        filled=False,
        linestyle="--",
        zorder=4,
    )
    qualifying_rows = [
        row for order, row in panel_a_selections.items()
        if order != 4 and row["selection_status"] == "target_met"
    ]
    ax.scatter(
        [value(row, "actual_nx") for row in qualifying_rows],
        [value(row, "predicted_total_relative_rms") for row in qualifying_rows],
        s=54,
        facecolors="none",
        edgecolors=COLORS["black"],
        linewidths=0.9,
        zorder=7,
    )
    pppm_panel_a_selected = min(
        (
            row for row in pppm_panel_a
            if value(row, "validation_relative_rms") <= FIGURE6_FIXED_BAND_TARGET
        ),
        key=lambda row: int(row["actual_grid_points"]),
    )
    ax.scatter(
        [value(pppm_panel_a_selected, "actual_nx")],
        [value(pppm_panel_a_selected, "validation_relative_rms")],
        s=54,
        facecolors="none",
        edgecolors=COLORS["black"],
        linewidths=0.9,
        zorder=7,
    )
    ax.axhline(
        FIGURE6_FIXED_BAND_TARGET,
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.0,
        label=r"Target, $10^{-5}$",
        zorder=1,
    )
    ax.axhline(
        FIGURE6_FOURIER_TAIL_REFERENCE_RELATIVE,
        color=COLORS["black"],
        linestyle=(0, (4, 2)),
        linewidth=1.0,
        label="Fourier-tail prediction",
        zorder=1,
    )
    ax.text(
        0.98,
        0.95,
        r"$P=4$: predicted target not reached",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["blue"],
        fontsize=6.6,
    )
    ax.set_xlim(10.5, 84.0)
    ax.set_ylim(1.0e-6, 3.0e-2)
    ax.set_xticks([12, 16, 20, 24, 32, 48, 64, 80])
    ax.set_xlabel(r"Actual cubic FFT grid size, $M$")
    ax.set_ylabel("Relative RMS force error")
    ax.set_title(r"ESP fixed band; PPPM fixed $G$, $P=5$", pad=4)
    handles, labels = ax.get_legend_handles_labels()
    validation_handle = Line2D(
        [], [], marker="o", linestyle="none", markerfacecolor="white",
        markeredgecolor=COLORS["black"], markeredgewidth=0.8, markersize=4.8,
        label="ESP holdout symbols",
    )
    selection_handle = Line2D(
        [], [], marker="o", linestyle="none", markerfacecolor="none",
        markeredgecolor=COLORS["black"], markeredgewidth=0.9, markersize=6.2,
        label=r"Selected by prediction; holdout pass",
    )
    ax.legend(
        handles + [validation_handle, selection_handle],
        labels + [validation_handle.get_label(), selection_handle.get_label()],
        loc="upper right", bbox_to_anchor=(1.0, 0.80),
        fontsize=5.8, ncol=2, handlelength=1.5, frameon=False,
        columnspacing=0.8, labelspacing=0.25,
    )
    panel_label(ax, "a")

    ik_volume_ratios: list[float] = []
    ad_volume_ratios: list[float] = []
    pppm_meshes: list[int] = []
    ik_meshes: list[int] = []
    ad_meshes: list[int] = []
    for target in targets:
        pppm_row = pppm_by_target[target]
        ik_row = esp_ik_by_target[target]
        ad_row = esp_ad_by_target[target]
        ik_volume_ratios.append(
            value(pppm_row, "actual_grid_points")
            / value(ik_row, "actual_grid_points")
        )
        ad_volume_ratios.append(
            value(pppm_row, "actual_grid_points")
            / value(ad_row, "actual_grid_points")
        )
        pppm_meshes.append(int(round(value(pppm_row, "actual_nx"))))
        ik_meshes.append(int(round(value(ik_row, "actual_nx"))))
        ad_meshes.append(int(round(value(ad_row, "actual_nx"))))

    write_figure6_plot_source(
        panel_a,
        panel_a_selections,
        pppm_panel_a,
        pppm_by_target,
        esp_ik_by_target,
        esp_ad_by_target,
        targets,
        ik_volume_ratios,
        ad_volume_ratios,
    )

    ax = axes[1]
    positions = np.arange(len(targets))
    offset = 0.045
    ax.set_yscale("log")
    ik_line = ax.plot(
        positions - offset,
        ik_volume_ratios,
        color=COLORS["blue"],
        marker="o",
        markerfacecolor="#C5E3F1",
        markeredgecolor=COLORS["blue"],
        markeredgewidth=0.8,
        linewidth=1.2,
        label=r"ESP IK",
        zorder=3,
    )[0]
    ad_line = ax.plot(
        positions + offset,
        ad_volume_ratios,
        color=COLORS["orange"],
        marker="s",
        markerfacecolor="#F4D69B",
        markeredgecolor=COLORS["orange"],
        markeredgewidth=0.8,
        linestyle="--",
        linewidth=1.2,
        label=r"ESP AD",
        zorder=3,
    )[0]
    ax.axhline(1.0, color=COLORS["gray"], linestyle=":", linewidth=0.8, zorder=1)
    for position, ik_ratio, ad_ratio in zip(
        positions, ik_volume_ratios, ad_volume_ratios
    ):
        if math.isclose(ik_ratio, ad_ratio, rel_tol=0.0, abs_tol=1.0e-12):
            ax.text(
                position,
                ik_ratio * 1.12,
                rf"${ik_ratio:.2f}$",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
        else:
            for x_value, ratio, color in (
                (position - offset, ik_ratio, COLORS["blue"]),
                (position + offset, ad_ratio, COLORS["orange"]),
            ):
                ax.text(
                    x_value,
                    ratio * 1.10,
                    rf"${ratio:.2f}$",
                    ha="center",
                    va="bottom",
                    color=color,
                    fontsize=6.2,
                )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            target_label
            + "\n"
            + rf"$P_{{\rm E}}={esp_order}$"
            + "\n"
            + (rf"$M={pppm_mesh}/{ik_mesh}$" if ik_mesh == ad_mesh
               else rf"$M={pppm_mesh}/{ik_mesh}/{ad_mesh}$")
            for target_label, esp_order, pppm_mesh, ik_mesh, ad_mesh in zip(
                (r"$10^{-3}$", r"$10^{-4}$", r"$10^{-5}$", r"$10^{-6}$"),
                (5, 6, 8, 10),
                pppm_meshes,
                ik_meshes,
                ad_meshes,
            )
        ]
    )
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(0.82, 230.0)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_xlabel("Target relative RMS force error")
    ax.set_ylabel(r"Grid-point reduction, $M_{\rm PPPM}^3/M_{\rm ESP}^3$")
    ax.set_title(r"PPPM auto-$G$: $P=5$; ESP: $P=5,6,8,10$", pad=4)
    ax.grid(axis="y", which="major", color="#E8EBEE", linewidth=0.55, zorder=0)
    ax.legend(
        [ik_line, ad_line],
        [r"ESP IK", r"ESP AD"],
        loc="upper left",
        fontsize=6.5,
        handlelength=1.4,
    )
    panel_label(ax, "b", x=-0.14)

    save_figure(
        fig,
        "fig5_pppm_efficiency",
        "Pilot-predicted fixed-band ESP convergence with fixed-G PPPM holdout validation and a separate auto-G PPPM/ESP grid comparison",
    )
    return (
        len(pppm_all),
        len(pppm_p5),
        ik_volume_ratios,
        ad_volume_ratios,
        len(panel_a),
    )


FIGURE5_DISPLAY_MAX_MESH = {"a": 40, "b": 80, "c": 40, "d": 80}
FIGURE5_PILOT_FRAMES = 25
FIGURE5_PILOT_BLOCK_SIZE = 5


def figure5_pilot_force_calibrations(
    frame_rows: list[dict[str, str]],
) -> dict[tuple[str, float, int, int], dict[str, float | int]]:
    """Pool frames 1--25 into an implementation-specific force calibration.

    This is deliberately not a theoretical error estimator.  It is retained
    solely for the historical archival diagnostic; the current main-text
    Figure 5 does not call it.
    """
    groups: dict[tuple[str, float, int, int], list[dict[str, str]]] = {}
    for row in frame_rows:
        key = (
            row["method"],
            value(row, "target_relative_rms"),
            int(row["order"]),
            int(row["actual_nx"]),
        )
        groups.setdefault(key, []).append(row)

    def pooled_relative_rms(rows: list[dict[str, str]]) -> float:
        numerator = math.fsum(
            value(row, "sum_squared_force_difference") for row in rows
        )
        denominator = math.fsum(
            value(row, "sum_squared_reference_force") for row in rows
        )
        if numerator < 0.0 or denominator <= 0.0:
            raise ValueError("Figure 5 calibration has an invalid force norm")
        return math.sqrt(numerator / denominator)

    calibrations: dict[
        tuple[str, float, int, int], dict[str, float | int]
    ] = {}
    for key, members in groups.items():
        ordered = sorted(members, key=lambda row: int(row["frame_zero_based"]))
        frame_indices = [int(row["frame_zero_based"]) for row in ordered]
        if len(ordered) != 51 or frame_indices != list(range(51)):
            raise ValueError(
                "Figure 5 calibration requires one complete 51-frame record "
                f"per case; invalid case {key}"
            )
        pilot = ordered[:FIGURE5_PILOT_FRAMES]
        holdout = ordered[FIGURE5_PILOT_FRAMES:]
        if (
            any(row["partition"] != "pilot" for row in pilot)
            or any(row["partition"] != "holdout" for row in holdout)
            or len(holdout) != 26
        ):
            raise ValueError(
                "Figure 5 calibration/holdout frame partitions are invalid "
                f"for case {key}"
            )
        blocks = [
            pilot[start : start + FIGURE5_PILOT_BLOCK_SIZE]
            for start in range(
                0, FIGURE5_PILOT_FRAMES, FIGURE5_PILOT_BLOCK_SIZE
            )
        ]
        block_values = [pooled_relative_rms(block) for block in blocks]
        calibration = pooled_relative_rms(pilot)
        sem = statistics.stdev(block_values) / math.sqrt(len(block_values))
        if not math.isfinite(calibration) or not math.isfinite(sem):
            raise ValueError(f"Figure 5 calibration is not finite for case {key}")
        calibrations[key] = {
            "calibrated_relative_rms": calibration,
            "calibrated_balanced_block5_sem": sem,
            "calibration_frame_first": 1,
            "calibration_frame_last": FIGURE5_PILOT_FRAMES,
            "calibration_frame_count": FIGURE5_PILOT_FRAMES,
            "calibration_block_count": len(block_values),
        }
    return calibrations


def write_archived_figure5_pilot_force_source(
    scan_rows: list[dict[str, str]],
    pppm_rows: list[dict[str, str]],
    calibrations: dict[tuple[str, float, int, int], dict[str, float | int]],
) -> None:
    panel_for = {
        ("ik", 1.0e-4): "a",
        ("ik", 1.0e-5): "b",
        ("ad", 1.0e-4): "c",
        ("ad", 1.0e-5): "d",
    }
    output: list[dict[str, object]] = []
    for row in scan_rows:
        method = row["method"]
        target = value(row, "target_relative_rms")
        order = int(row["order"])
        panel = panel_for[(method, target)]
        if int(row["actual_nx"]) > FIGURE5_DISPLAY_MAX_MESH[panel]:
            continue
        common = {
            "panel": panel,
            "method": f"ESP {method.upper()}",
            "target_relative_rms": target,
            "order": order,
            "actual_nx": int(row["actual_nx"]),
            "actual_grid_points": int(row["actual_grid_points"]),
            "sigma_up": value(row, "sigma_up"),
            "csplit": value(row, "csplit"),
            "cspread": value(row, "cspread"),
            "resolved_band": value(row, "sigma_up") >= 1.0,
            "pppm_fixed_gewald_inverse_A": "",
            "case_id": row["case_id"],
            "force_dump_sha256": row["force_dump_sha256"],
        }
        output.append(
            {
                **common,
                "record_type": "ESP measured holdout",
                "relative_rms": value(row, "holdout_relative_rms"),
                "balanced_block5_sem": value(
                    row, "holdout_balanced_block5_sem"
                ),
                "holdout_relative_rms": value(row, "holdout_relative_rms"),
                "holdout_balanced_block5_sem": value(
                    row, "holdout_balanced_block5_sem"
                ),
                "calibrated_relative_rms": "",
                "calibrated_balanced_block5_sem": "",
                "calibration_frame_first": "",
                "calibration_frame_last": "",
                "calibration_frame_count": "",
                "calibration_block_count": "",
                "calibration_partition": "",
                "holdout_used_for_calibration": "",
                "passes_target": row["holdout_passes_target"],
                "source_csv": (
                    "fig5_ik_ad_order_scan/fig5_ik_ad_order_scan_summary.csv"
                ),
            }
        )
        calibration = calibrations[
            (method, target, order, int(row["actual_nx"]))
        ]
        output.append(
            {
                **common,
                "record_type": "ESP 25-frame water calibration",
                "relative_rms": calibration["calibrated_relative_rms"],
                "balanced_block5_sem": calibration[
                    "calibrated_balanced_block5_sem"
                ],
                "holdout_relative_rms": "",
                "holdout_balanced_block5_sem": "",
                "calibrated_relative_rms": calibration[
                    "calibrated_relative_rms"
                ],
                "calibrated_balanced_block5_sem": calibration[
                    "calibrated_balanced_block5_sem"
                ],
                "calibration_frame_first": calibration[
                    "calibration_frame_first"
                ],
                "calibration_frame_last": calibration[
                    "calibration_frame_last"
                ],
                "calibration_frame_count": calibration[
                    "calibration_frame_count"
                ],
                "calibration_block_count": calibration[
                    "calibration_block_count"
                ],
                "calibration_partition": "pilot",
                "holdout_used_for_calibration": False,
                "passes_target": (
                    calibration["calibrated_relative_rms"] <= target
                ),
                "source_csv": (
                    "fig5_ik_ad_order_scan/fig5_ik_ad_order_scan_by_frame.csv"
                ),
            }
        )
    for row in pppm_rows:
        panel = row["panel"]
        if int(row["actual_nx"]) > FIGURE5_DISPLAY_MAX_MESH[panel]:
            continue
        output.append(
            {
                "panel": panel,
                "record_type": "PPPM fixed-G measured holdout",
                "method": f"PPPM {row['differentiation'].upper()}",
                "target_relative_rms": value(row, "target_relative_rms"),
                "order": int(row["order"]),
                "actual_nx": int(row["actual_nx"]),
                "actual_grid_points": int(row["actual_grid_points"]),
                "sigma_up": "",
                "csplit": "",
                "cspread": "",
                "relative_rms": value(row, "holdout_relative_rms"),
                "balanced_block5_sem": value(
                    row, "holdout_balanced_block5_sem"
                ),
                "holdout_relative_rms": value(row, "holdout_relative_rms"),
                "holdout_balanced_block5_sem": value(
                    row, "holdout_balanced_block5_sem"
                ),
                "calibrated_relative_rms": "",
                "calibrated_balanced_block5_sem": "",
                "calibration_frame_first": "",
                "calibration_frame_last": "",
                "calibration_frame_count": "",
                "calibration_block_count": "",
                "calibration_partition": "",
                "holdout_used_for_calibration": "",
                "passes_target": row["holdout_passes_target"],
                "resolved_band": "",
                "pppm_fixed_gewald_inverse_A": value(
                    row, "fixed_gewald_inverse_A"
                ),
                "case_id": row["case_id"],
                "force_dump_sha256": row["force_dump_sha256"],
                "source_csv": (
                    "fig5_pppm_ik_ad_fixed_g_scan/"
                    "fig5_pppm_ik_ad_fixed_g_summary.csv"
                ),
            }
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "figS_figure5_pilot_force_calibration_source.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output)


def archived_figure5_pilot_force_calibration_figure() -> dict[str, object]:
    """Generate the retired four-panel pilot-force calibration diagnostic.

    This is an archival/SI routine, not the current main-text Figure 5.
    """
    scan_rows = read_path_rows(
        HERE
        / "fig5_ik_ad_order_scan"
        / "fig5_ik_ad_order_scan_summary.csv"
    )
    expected_orders = tuple(range(5, 10))
    expected_meshes = {
        1.0e-4: (12, 15, 16, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80),
        1.0e-5: (12, 16, 18, 20, 24, 32, 36, 40, 48, 64, 80),
    }
    expected_band = {1.0e-4: 12.024, 1.0e-5: 14.471}
    expected_keys = {
        (method, target, order, mesh)
        for method in ("ik", "ad")
        for target, meshes in expected_meshes.items()
        for order in expected_orders
        for mesh in meshes
    }
    actual_keys = {
        (
            row["method"],
            value(row, "target_relative_rms"),
            int(row["order"]),
            int(row["actual_nx"]),
        )
        for row in scan_rows
    }
    if actual_keys != expected_keys or len(scan_rows) != len(expected_keys):
        raise ValueError("Figure 5 IK/AD order scan is incomplete or duplicated")
    if any(
        int(row["total_frames"]) != 51
        or int(row["holdout_frames"]) != 26
        or not math.isclose(
            value(row, "csplit"),
            expected_band[value(row, "target_relative_rms")],
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            value(row, "cspread"),
            expected_band[value(row, "target_relative_rms")],
            abs_tol=1.0e-12,
        )
        or len(row["force_dump_sha256"]) != 64
        for row in scan_rows
    ):
        raise ValueError("Figure 5 scan violates the fixed-band holdout contract")

    scan_frame_rows = read_path_rows(
        HERE
        / "fig5_ik_ad_order_scan"
        / "fig5_ik_ad_order_scan_by_frame.csv"
    )
    calibrations = figure5_pilot_force_calibrations(scan_frame_rows)
    if (
        len(scan_frame_rows) != len(expected_keys) * 51
        or set(calibrations) != actual_keys
    ):
        raise ValueError("Figure 5 water calibration records are incomplete")

    pppm_rows = read_path_rows(
        HERE
        / "fig5_pppm_ik_ad_fixed_g_scan"
        / "fig5_pppm_ik_ad_fixed_g_summary.csv"
    )
    expected_pppm_meshes = {
        "a": (12, 15, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80),
        "b": (12, 15, 18, 20, 24, 32, 36, 40, 48, 64, 80),
        "c": (12, 15, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80),
        "d": (12, 15, 18, 20, 24, 32, 36, 40, 48, 64, 80),
    }
    expected_pppm = {
        "a": ("ik", 1.0e-4, 1.0e-5, 0.34166005, 30),
        "b": ("ik", 1.0e-5, 1.0e-6, 0.37759658, 48),
        "c": ("ad", 1.0e-4, 1.0e-5, 0.33593464, 36),
        "d": ("ad", 1.0e-5, 1.0e-6, 0.37738337, 72),
    }
    pppm_keys = {
        (row["panel"], row["differentiation"], int(row["actual_nx"]))
        for row in pppm_rows
    }
    expected_pppm_keys = {
        (panel, settings[0], mesh)
        for panel, settings in expected_pppm.items()
        for mesh in expected_pppm_meshes[panel]
    }
    if pppm_keys != expected_pppm_keys or len(pppm_rows) != sum(
        len(meshes) for meshes in expected_pppm_meshes.values()
    ):
        raise ValueError("Figure 5 fixed-G PPPM display scan is incomplete")
    if any(
        int(row["order"]) != 5
        or int(row["holdout_frames"]) != 26
        or not math.isclose(
            value(row, "target_relative_rms"),
            expected_pppm[row["panel"]][1],
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            value(row, "calibration_input_tolerance"),
            expected_pppm[row["panel"]][2],
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            value(row, "fixed_gewald_inverse_A"),
            expected_pppm[row["panel"]][3],
            abs_tol=1.0e-12,
        )
        or int(row["calibration_actual_nx"])
        != expected_pppm[row["panel"]][4]
        or len(row["force_dump_sha256"]) != 64
        for row in pppm_rows
    ):
        raise ValueError("Figure 5 PPPM rows violate the fixed-G contract")

    esp_selected: dict[tuple[str, float, int], dict[str, str] | None] = {}
    calibration_selected: dict[tuple[str, float, int], dict[str, str] | None] = {}
    for method in ("ik", "ad"):
        for target in expected_meshes:
            for order in expected_orders:
                subset = [
                    row
                    for row in scan_rows
                    if row["method"] == method
                    and math.isclose(value(row, "target_relative_rms"), target)
                    and int(row["order"]) == order
                ]
                feasible = [
                    row
                    for row in subset
                    if value(row, "sigma_up") >= 1.0
                    and value(row, "holdout_relative_rms") <= target
                ]
                esp_selected[(method, target, order)] = (
                    min(feasible, key=lambda row: int(row["actual_grid_points"]))
                    if feasible
                    else None
                )
                calibration_feasible = [
                    row
                    for row in subset
                    if value(row, "sigma_up") >= 1.0
                    and calibrations[
                        (method, target, order, int(row["actual_nx"]))
                    ]["calibrated_relative_rms"]
                    <= target
                ]
                calibration_selected[(method, target, order)] = (
                    min(
                        calibration_feasible,
                        key=lambda row: int(row["actual_grid_points"]),
                    )
                    if calibration_feasible
                    else None
                )
    pppm_selected = {
        panel: min(
            (
                row
                for row in pppm_rows
                if row["panel"] == panel
                and value(row, "holdout_relative_rms") <= settings[1]
            ),
            key=lambda row: int(row["actual_grid_points"]),
        )
        for panel, settings in expected_pppm.items()
    }
    write_archived_figure5_pilot_force_source(scan_rows, pppm_rows, calibrations)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 5.0),
        sharex="col",
        sharey=False,
    )
    fig.subplots_adjust(
        left=0.095, right=0.99, bottom=0.115, top=0.82, wspace=0.17, hspace=0.36
    )
    order_style = {
        5: (COLORS["blue"], "o"),
        6: (COLORS["orange"], "s"),
        7: (COLORS["green"], "^"),
        8: (COLORS["vermillion"], "D"),
        9: (COLORS["purple"], "v"),
    }
    panel_specs = (
        (axes[0, 0], "ik", 1.0e-4, "a"),
        (axes[0, 1], "ik", 1.0e-5, "b"),
        (axes[1, 0], "ad", 1.0e-4, "c"),
        (axes[1, 1], "ad", 1.0e-5, "d"),
    )
    for ax, method, target, label in panel_specs:
        set_log_y(ax)
        light_y_grid(ax)
        for order in expected_orders:
            subset = sorted(
                (
                    row
                    for row in scan_rows
                    if row["method"] == method
                    and math.isclose(value(row, "target_relative_rms"), target)
                    and int(row["order"]) == order
                    and int(row["actual_nx"])
                    <= FIGURE5_DISPLAY_MAX_MESH[label]
                ),
                key=lambda row: int(row["actual_nx"]),
            )
            color, marker = order_style[order]
            ax.plot(
                [value(row, "actual_nx") for row in subset],
                [
                    calibrations[
                        (method, target, order, int(row["actual_nx"]))
                    ]["calibrated_relative_rms"]
                    for row in subset
                ],
                color=color,
                linewidth=1.3,
                linestyle="--",
                label="_nolegend_",
                zorder=4,
            )
            errorbar(
                ax,
                [value(row, "actual_nx") for row in subset],
                [value(row, "holdout_relative_rms") for row in subset],
                [value(row, "holdout_balanced_block5_sem") for row in subset],
                color=color,
                marker=marker,
                label="_nolegend_",
                filled=True,
                marker_size=4.0,
                linestyle="none",
                zorder=5,
            )
        pppm_panel_rows = sorted(
            (
                row
                for row in pppm_rows
                if row["panel"] == label
                and int(row["actual_nx"])
                <= FIGURE5_DISPLAY_MAX_MESH[label]
            ),
            key=lambda row: int(row["actual_nx"]),
        )
        errorbar(
            ax,
            [value(row, "actual_nx") for row in pppm_panel_rows],
            [value(row, "holdout_relative_rms") for row in pppm_panel_rows],
            [
                value(row, "holdout_balanced_block5_sem")
                for row in pppm_panel_rows
            ],
            color=COLORS["black"],
            marker="X",
            label="_nolegend_",
            filled=False,
            marker_size=4.0,
            linestyle="none",
            zorder=3,
        )
        ax.axhline(
            target,
            color=COLORS["gray"],
            linestyle=":",
            linewidth=0.9,
            label="_nolegend_",
            zorder=1,
        )
        if FIGURE5_DISPLAY_MAX_MESH[label] == 40:
            ax.set_xlim(10.5, 41.0)
            mesh_ticks = [12, 16, 20, 24, 28, 32, 36, 40]
        else:
            ax.set_xlim(10.5, 84.0)
            mesh_ticks = [12, 18, 24, 32, 48, 64, 80]
        ax.set_xticks(mesh_ticks)
        ax.set_xticklabels([f"{mesh}\N{SUPERSCRIPT THREE}" for mesh in mesh_ticks])
        ax.tick_params(axis="x", labelsize=6.8)
        ax.set_ylim(1.0e-5 if target == 1.0e-4 else 1.0e-6, 3.0e-2)
        differentiation_label = (
            r"$\mathrm{i}\boldsymbol{k}$" if method == "ik" else "AD"
        )
        ax.set_title(
            differentiation_label
            + rf"; $\varepsilon=10^{{{int(round(math.log10(target)))}}}$",
            pad=4,
        )
        panel_label(ax, label, x=-0.13, y=1.09)
    axes[0, 0].set_ylabel("Relative RMS force error")
    axes[1, 0].set_ylabel("Relative RMS force error")
    axes[0, 0].tick_params(axis="x", labelbottom=True)
    axes[0, 1].tick_params(axis="x", labelbottom=True)
    axes[1, 0].set_xlabel("Actual FFT grid size")
    axes[1, 1].set_xlabel("Actual FFT grid size")

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=5.6,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            label=rf"ESP $P={order}$",
        )
        for order, (color, marker) in order_style.items()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=COLORS["black"],
            marker="X",
            linestyle="none",
            markersize=5.6,
            markerfacecolor="white",
            markeredgecolor=COLORS["black"],
            markeredgewidth=0.8,
            label="PPPM",
        )
    )
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.96),
        ncol=6,
        fontsize=9.4,
        handlelength=1.0,
        handletextpad=0.95,
        columnspacing=2.25,
        labelspacing=0.9,
        frameon=False,
    )
    save_figure(
        fig,
        "figS_figure5_pilot_force_calibration",
        "Archived pilot-force calibration diagnostic; not a theoretical ESP prediction",
    )
    first_crossings = {
        method: {
            f"{target:.0e}": {
                str(order): (
                    int(esp_selected[(method, target, order)]["actual_nx"])
                    if esp_selected[(method, target, order)] is not None
                    else None
                )
                for order in expected_orders
            }
            for target in expected_meshes
        }
        for method in ("ik", "ad")
    }
    calibration_first_crossings = {
        method: {
            f"{target:.0e}": {
                str(order): (
                    int(calibration_selected[(method, target, order)]["actual_nx"])
                    if calibration_selected[(method, target, order)] is not None
                    else None
                )
                for order in expected_orders
            }
            for target in expected_meshes
        }
        for method in ("ik", "ad")
    }
    calibration_quality: dict[str, dict[str, float | int]] = {}
    for _, method, target, panel in panel_specs:
        displayed = [
            row
            for row in scan_rows
            if row["method"] == method
            and math.isclose(value(row, "target_relative_rms"), target)
            and int(row["actual_nx"]) <= FIGURE5_DISPLAY_MAX_MESH[panel]
        ]
        fractional_differences = [
            calibrations[
                (method, target, int(row["order"]), int(row["actual_nx"]))
            ]["calibrated_relative_rms"]
            / value(row, "holdout_relative_rms")
            - 1.0
            for row in displayed
        ]
        calibration_quality[panel] = {
            "points": len(fractional_differences),
            "mean_absolute_fractional_difference": statistics.fmean(
                abs(item) for item in fractional_differences
            ),
            "median_absolute_fractional_difference": statistics.median(
                abs(item) for item in fractional_differences
            ),
            "maximum_absolute_fractional_difference": max(
                abs(item) for item in fractional_differences
            ),
            "mean_signed_fractional_difference": statistics.fmean(
                fractional_differences
            ),
        }
    esp_displayed_rows = sum(
        int(row["actual_nx"])
        <= FIGURE5_DISPLAY_MAX_MESH[
            {
                ("ik", 1.0e-4): "a",
                ("ik", 1.0e-5): "b",
                ("ad", 1.0e-4): "c",
                ("ad", 1.0e-5): "d",
            }[(row["method"], value(row, "target_relative_rms"))]
        ]
        for row in scan_rows
    )
    return {
        "esp_rows": len(scan_rows),
        "esp_calibration_rows": len(calibrations),
        "pppm_rows": len(pppm_rows),
        "pppm_rows_by_panel": {
            panel: len(meshes) for panel, meshes in expected_pppm_meshes.items()
        },
        "esp_displayed_rows": esp_displayed_rows,
        "esp_calibration_displayed_rows": esp_displayed_rows,
        "pppm_displayed_rows": sum(
            int(row["actual_nx"])
            <= FIGURE5_DISPLAY_MAX_MESH[row["panel"]]
            for row in pppm_rows
        ),
        "display_max_mesh_by_panel": dict(FIGURE5_DISPLAY_MAX_MESH),
        "orders": list(expected_orders),
        "first_resolved_crossings": first_crossings,
        "calibration_first_resolved_crossings": calibration_first_crossings,
        "calibration_crossings_match_holdout": (
            calibration_first_crossings == first_crossings
        ),
        "calibration_quality_by_panel": calibration_quality,
        "calibration_frames": [1, FIGURE5_PILOT_FRAMES],
        "calibration_block_count": 5,
        "calibration_uses_holdout": False,
        "calibration_marker": None,
        "calibration_curve_drawn": True,
        "calibration_curve_style": "dashed line without markers",
        "holdout_connecting_lines_drawn": False,
        "holdout_marker_fill": "solid",
        "pppm_connecting_lines_drawn": False,
        "pppm_marker_fill": "open",
        "legend_style": "marker-only, matching numerical symbols",
        "legend_fontsize_pt": 9.4,
        "legend_marker_size_pt": 5.6,
        "legend_handletextpad": 0.95,
        "legend_columnspacing": 2.25,
        "pppm_first_crossings": {
            panel: int(row["actual_nx"])
            for panel, row in pppm_selected.items()
        },
        "pppm_calibrations": {
            panel: {
                "differentiation": settings[0],
                "input_tolerance": settings[2],
                "fixed_gewald_inverse_A": settings[3],
                "auto_grid": settings[4],
            }
            for panel, settings in expected_pppm.items()
        },
    }


def figure5_theoretical_fixed_ik() -> dict[str, object]:
    """Draw the theory/holdout Figure 5 from a frozen fixed-IK screen.

    The data source is generated in two stages.  The prediction table contains
    Eq. (56)+Eq. (90) values based only on frames 1--25 and a coarse PPPM force
    scale; it is hash-frozen before the source table is joined to Ewald-force
    measurements on frames 26--51.  This deliberately replaces the former
    Figure-5 pilot-force-calibration curves.
    """

    theory_rows = read_rows("fig5_fixed_ik_theory_grid_source.csv")
    pppm_rows = read_path_rows(
        OUTPUT_ROOT
        / "fig5_pppm_ik_ad_fixed_g_scan"
        / "fig5_pppm_ik_ad_fixed_g_summary.csv"
    )
    expected_meshes = {
        1.0e-4: (12, 15, 16, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80),
        1.0e-5: (12, 16, 18, 20, 24, 32, 36, 40, 48, 64, 80),
    }
    expected_orders = tuple(range(5, 10))
    expected_count = sum(len(meshes) for meshes in expected_meshes.values()) * len(expected_orders)
    if len(theory_rows) != expected_count:
        raise ValueError("Figure 5 fixed-IK theory matrix is incomplete")
    keys = {
        (value(row, "target_relative_rms"), int(row["order"]), int(row["actual_nx"]))
        for row in theory_rows
    }
    expected_keys = {
        (target, order, mesh)
        for target, meshes in expected_meshes.items()
        for order in expected_orders
        for mesh in meshes
    }
    if keys != expected_keys:
        raise ValueError("Figure 5 fixed-IK theory keys are incomplete or duplicated")
    if any(
        str(row.get("ewald_reference_force_accessed", "")).lower() == "true"
        or str(row.get("holdout_coordinates_accessed", "")).lower() == "true"
        or int(row["validation_frame_first"]) != 26
        or int(row["validation_frame_last"]) != 51
        or int(row["validation_frame_count"]) != 26
        for row in theory_rows
    ):
        raise ValueError("Figure 5 theory/holdout partition is invalid")
    if any(
        "Eq. (56)" not in row["prediction_operator"]
        or "Eq. (90)" not in row["prediction_operator"]
        for row in theory_rows
    ):
        raise ValueError("Figure 5 theory source does not state the required estimators")

    pppm_by_target = {
        1.0e-4: sorted(
            [row for row in pppm_rows if row["panel"] == "a"],
            key=lambda row: int(row["actual_nx"]),
        ),
        1.0e-5: sorted(
            [row for row in pppm_rows if row["panel"] == "b"],
            key=lambda row: int(row["actual_nx"]),
        ),
    }
    for target, rows in pppm_by_target.items():
        if not rows or any(
            row["differentiation"] != "ik"
            or int(row["order"]) != 5
            or not math.isclose(value(row, "target_relative_rms"), target, abs_tol=1.0e-15)
            or int(row["holdout_frames"]) != 26
            for row in rows
        ):
            raise ValueError(f"Figure 5 PPPM fixed-IK baseline is invalid for {target:.0e}")

    order_style = {
        5: (COLORS["blue"], "o"),
        6: (COLORS["orange"], "s"),
        7: (COLORS["green"], "^"),
        8: (COLORS["vermillion"], "D"),
        9: (COLORS["purple"], "v"),
    }
    panel_specs = (
        (1.0e-4, "a", (3.5e-5, 1.7e-3), [15, 20, 24, 32, 40, 64, 80]),
        (1.0e-5, "b", (2.0e-6, 2.5e-3), [16, 20, 24, 32, 40, 64, 80]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.28), sharey=False)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.19, top=0.72, wspace=0.22)

    first_theory: dict[str, dict[str, int | None]] = {}
    first_validation: dict[str, dict[str, int | None]] = {}
    plot_source: list[dict[str, object]] = []
    for ax, (target, panel, ylim, ticks) in zip(axes, panel_specs):
        set_log_y(ax)
        light_y_grid(ax)
        per_target = [
            row for row in theory_rows
            if math.isclose(value(row, "target_relative_rms"), target, abs_tol=1.0e-15)
        ]
        first_theory[f"{target:.0e}"] = {}
        first_validation[f"{target:.0e}"] = {}
        for order in expected_orders:
            color, marker = order_style[order]
            all_rows = sorted(
                [row for row in per_target if int(row["order"]) == order],
                key=lambda row: int(row["actual_nx"]),
            )
            resolved = [
                row for row in all_rows
                if str(row["resolved_band"]).lower() == "true"
            ]
            if not resolved:
                raise ValueError(f"Figure 5 has no resolved rows for {target:.0e}, P={order}")
            ax.plot(
                [value(row, "actual_grid_points") for row in resolved],
                [value(row, "predicted_total_relative_rms") for row in resolved],
                color=color,
                linewidth=1.25,
                linestyle="--",
                zorder=3,
                label="_nolegend_",
            )
            errorbar(
                ax,
                [value(row, "actual_grid_points") for row in resolved],
                [value(row, "validation_relative_rms") for row in resolved],
                [value(row, "validation_relative_rms_balanced_block5_sem") for row in resolved],
                color=color,
                marker=marker,
                label="_nolegend_",
                filled=True,
                marker_size=4.25,
                linestyle="none",
                zorder=5,
            )
            predicted_crossings = [
                row for row in resolved
                if str(row["prediction_passes_target"]).lower() == "true"
            ]
            validation_crossings = [
                row for row in resolved
                if str(row["validation_passes_target"]).lower() == "true"
            ]
            first_theory[f"{target:.0e}"][str(order)] = (
                int(predicted_crossings[0]["actual_nx"]) if predicted_crossings else None
            )
            first_validation[f"{target:.0e}"][str(order)] = (
                int(validation_crossings[0]["actual_nx"]) if validation_crossings else None
            )
            for row in all_rows:
                plot_source.append(
                    {
                        "panel": panel,
                        "record_type": "ESP theoretical prediction",
                        "series": f"ESP P={order}",
                        "target_relative_rms": target,
                        "order": order,
                        "actual_nx": row["actual_nx"],
                        "actual_grid_points": row["actual_grid_points"],
                        "sigma_up": row["sigma_up"],
                        "resolved_band": row["resolved_band"],
                        "relative_rms": row["predicted_total_relative_rms"],
                        "relative_rms_sem": row["predicted_total_relative_block5_sem"],
                        "source_partition": "frames 1-25; no Ewald force difference",
                        "estimator": row["prediction_operator"],
                        "used_for_selection": True,
                    }
                )
                plot_source.append(
                    {
                        "panel": panel,
                        "record_type": "ESP Ewald holdout validation",
                        "series": f"ESP P={order}",
                        "target_relative_rms": target,
                        "order": order,
                        "actual_nx": row["actual_nx"],
                        "actual_grid_points": row["actual_grid_points"],
                        "sigma_up": row["sigma_up"],
                        "resolved_band": row["resolved_band"],
                        "relative_rms": row["validation_relative_rms"],
                        "relative_rms_sem": row["validation_relative_rms_balanced_block5_sem"],
                        "source_partition": "frames 26-51; Ewald validation only",
                        "estimator": row["validation_operator"],
                        "used_for_selection": False,
                    }
                )
        pppm = pppm_by_target[target]
        errorbar(
            ax,
            [value(row, "actual_grid_points") for row in pppm],
            [value(row, "holdout_relative_rms") for row in pppm],
            [value(row, "holdout_balanced_block5_sem") for row in pppm],
            color=COLORS["black"],
            marker="X",
            label="_nolegend_",
            filled=False,
            marker_size=4.1,
            linestyle="none",
            zorder=4,
        )
        for row in pppm:
            plot_source.append(
                {
                    "panel": panel,
                    "record_type": "PPPM Ewald holdout validation",
                    "series": "Gaussian-split PPPM P=5",
                    "target_relative_rms": target,
                    "order": 5,
                    "actual_nx": row["actual_nx"],
                    "actual_grid_points": row["actual_grid_points"],
                    "sigma_up": "",
                    "resolved_band": "",
                    "relative_rms": row["holdout_relative_rms"],
                    "relative_rms_sem": row["holdout_balanced_block5_sem"],
                    "source_partition": "frames 26-51; Ewald validation",
                    "estimator": "not plotted as an ESP theoretical estimate",
                    "used_for_selection": False,
                }
            )
        ax.axhline(target, color=COLORS["gray"], linestyle=":", linewidth=0.9, zorder=1)
        volume_ticks = [tick**3 for tick in ticks]
        ax.set_xscale("log")
        ax.set_xlim(min(volume_ticks) / 1.18, max(volume_ticks) * 1.18)
        ax.set_xticks(volume_ticks)
        ax.set_xticklabels([rf"${tick}^3$" for tick in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", labelsize=7.1)
        ax.text(
            0.975,
            0.955,
            r"fixed-influence $\mathrm{i}\mathbf{k}$"
            + "\n"
            + rf"$\varepsilon = 10^{{{int(round(math.log10(target)))}}}$"
            + "\n"
            + rf"$c_{{\rm split}}=c_{{\rm spread}}={per_target[0]['csplit']}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.15,
            linespacing=1.35,
            color=COLORS["black"],
        )
        panel_label(ax, panel, x=-0.125, y=1.08, fontsize=10.0)

    axes[0].set_ylabel("Relative RMS force error")
    for ax in axes:
        ax.set_xlabel(r"Actual FFT grid size, $M^3$")

    fig.text(
        0.50,
        0.975,
        r"ESP: dashed lines, Eq. (56)+Eq. (90) theory; filled symbols, Ewald validation (frames 26--51)",
        ha="center",
        va="top",
        fontsize=8.0,
    )
    handles = [
        Line2D(
            [0], [0], color=color, marker=marker, linestyle="--", linewidth=1.25,
            markersize=5.2, markerfacecolor=color, markeredgecolor=color,
            label=rf"ESP $P={order}$",
        )
        for order, (color, marker) in order_style.items()
    ]
    handles.append(
        Line2D(
            [0], [0], color=COLORS["black"], marker="X", linestyle="none",
            markersize=5.2, markerfacecolor="white", markeredgecolor=COLORS["black"],
            label=r"Gaussian-split PPPM $P=5$",
        )
    )
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.895),
        ncol=6,
        fontsize=7.4,
        handlelength=1.65,
        handletextpad=0.55,
        columnspacing=1.05,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "fig5_pppm_efficiency_plot_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(plot_source[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(plot_source)
    save_figure(
        fig,
        "fig5_pppm_efficiency",
        "Theory-only fixed-influence ik ESP grid/order predictions and nonoverlapping Ewald validation",
    )
    return {
        "esp_rows": len(theory_rows),
        "esp_theory_rows": len(theory_rows),
        "pppm_rows": sum(len(rows) for rows in pppm_by_target.values()),
        "pppm_rows_by_panel": {f"{target:.0e}": len(rows) for target, rows in pppm_by_target.items()},
        "esp_displayed_rows": sum(
            str(row["resolved_band"]).lower() == "true" for row in theory_rows
        ),
        "pppm_displayed_rows": sum(len(rows) for rows in pppm_by_target.values()),
        "orders": list(expected_orders),
        "first_theory_resolved_crossings": first_theory,
        "first_validation_resolved_crossings": first_validation,
        "theory_frames": [1, 25],
        "theory_uses_reference_force_differences": False,
        "theory_curve_style": "dashed line without markers",
        "validation_curve_style": "filled markers with balanced-block SEM; no connecting line",
        "pppm_curve_style": "open X markers with balanced-block SEM; no connecting line",
        "ad_panels_included": False,
        "below_band_points_omitted_from_main_plot": True,
    }


def figure5_theory_and_stag_ad() -> dict[str, object]:
    """Draw Figure 5 with fixed-IK and conditional pair/self AD theory.

    Panels (a,b) use the frozen fixed-influence IK theoretical analysis.
    Panels (c,d) use structural spectra frozen from pilot frames 1--25. The
    coherent all-source AD pair and self-correction fields are added before
    cell squaring; the zero-mean pair fluctuation and closed Fourier tail
    complete the theory.
    Dashed curves are predictions; filled markers are independent Ewald
    validation on frames 26--51.  Frozen AD selections and their validation
    are retained in the source data but are not plotted separately.
    """

    theory_rows = read_rows("fig5_fixed_ik_theory_grid_source.csv")
    stag_ad_rows = read_path_rows(
        OUTPUT_ROOT / "fig5_ad_stag_screen" / "baseline_source.csv"
    )
    pppm_rows = read_path_rows(
        OUTPUT_ROOT
        / "fig5_pppm_ik_ad_fixed_g_scan"
        / "fig5_pppm_ik_ad_fixed_g_summary.csv"
    )
    expected_meshes = {
        1.0e-4: (12, 15, 16, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80),
        1.0e-5: (12, 16, 18, 20, 24, 32, 36, 40, 48, 64, 80),
    }
    expected_orders = tuple(range(5, 10))
    expected_count = sum(len(meshes) for meshes in expected_meshes.values()) * len(
        expected_orders
    )
    expected_keys = {
        (target, order, mesh)
        for target, meshes in expected_meshes.items()
        for order in expected_orders
        for mesh in meshes
    }

    theory_keys = {
        (value(row, "target_relative_rms"), int(row["order"]), int(row["actual_nx"]))
        for row in theory_rows
    }
    if len(theory_rows) != expected_count or theory_keys != expected_keys:
        raise ValueError("Figure 5 fixed-IK theory matrix is incomplete")
    if any(
        str(row.get("ewald_reference_force_accessed", "")).lower() == "true"
        or str(row.get("holdout_coordinates_accessed", "")).lower() == "true"
        or int(row["validation_frame_first"]) != 26
        or int(row["validation_frame_last"]) != 51
        or int(row["validation_frame_count"]) != 26
        for row in theory_rows
    ):
        raise ValueError("Figure 5 theory/holdout partition is invalid")
    if any(
        "Eq. (56)" not in row["prediction_operator"]
        or "Eq. (90)" not in row["prediction_operator"]
        for row in theory_rows
    ):
        raise ValueError("Figure 5 theory source does not state the required estimators")

    stag_ad_keys = {
        (value(row, "target_relative_rms"), int(row["order"]), int(row["actual_nx"]))
        for row in stag_ad_rows
    }
    if len(stag_ad_rows) != expected_count or stag_ad_keys != expected_keys:
        raise ValueError("Figure 5 spectral AD matrix is incomplete")
    if any(
        str(row["prediction_reference_force_accessed"]).lower() == "true"
        or str(row["prediction_holdout_coordinates_accessed"]).lower() == "true"
        or str(row["prediction_molecular_coordinates_accessed"]).lower() != "false"
        or str(row["prediction_unit_charge_force_accessed"]).lower() != "false"
        or str(row["prediction_particlewise_force_difference_evaluated"]).lower()
        != "false"
        or row["prediction_structure_input"]
        != "measured target-conditioned S_tag from frames 1--25"
        or row["prediction_coherent_structure_input"]
        != "charge-class conditional pair amplitudes from frames 1--25"
        or "conditional-mean AD pair/self theory" not in row["ad_estimator"]
        or str(row["pair_self_covariance_included"]).lower() != "true"
        or int(row["validation_frame_first"]) != 26
        or int(row["validation_frame_last"]) != 51
        or int(row["validation_frame_count"]) != 26
        or not math.isclose(
            value(row, "csplit"),
            12.024 if math.isclose(value(row, "target_relative_rms"), 1.0e-4) else 14.471,
            abs_tol=5.0e-4,
        )
        or not math.isclose(value(row, "csplit"), value(row, "cspread"), abs_tol=5.0e-4)
        for row in stag_ad_rows
    ):
        raise ValueError(
            "Figure 5 spectral AD source violates the fixed-band contract"
        )

    # The two strict-window selections are frozen from spectral AD theory
    # before their Ewald holdout checks are opened.
    stag_screen_dirs = {
        1.0e-4: OUTPUT_ROOT / "fig5_ad_stag_screen" / "stag_1e-4",
        1.0e-5: OUTPUT_ROOT / "fig5_ad_stag_screen" / "stag_1e-5",
    }
    stag_screens: dict[float, dict[str, object]] = {}
    for target, directory in stag_screen_dirs.items():
        frozen = json.loads((directory / "frozen_selection.json").read_text())
        summary_rows = read_path_rows(directory / "holdout_validation_summary.csv")
        manifest = json.loads((directory / "manifest.json").read_text())
        if len(summary_rows) != 1:
            raise ValueError(
                f"Figure 5 spectral AD has invalid summary: {directory}"
            )
        selected = dict(frozen["selected"])
        summary = summary_rows[0]
        validation = manifest.get("validation", {})
        if (
            not math.isclose(float(selected["target_relative_rms"]), target, abs_tol=1.0e-15)
            or not bool(selected["selection_passes_target"])
            or bool(selected["prediction_reference_force_accessed"])
            or bool(selected["prediction_holdout_coordinates_accessed"])
            or bool(selected["prediction_molecular_coordinates_accessed"])
            or bool(selected["prediction_unit_charge_force_accessed"])
            or bool(selected["prediction_particlewise_force_difference_evaluated"])
            or selected["prediction_structure_input"]
            != "measured target-conditioned S_tag from frames 1--25"
            or selected["prediction_coherent_structure_input"]
            != "charge-class conditional pair amplitudes from frames 1--25"
            or int(summary["validation_frame_count"]) != 26
            or summary["validation_frames"] != "26--51"
            or summary["selection_used_holdout"].strip().lower() != "false"
            or int(summary["actual_nx"]) != int(selected["actual_nx"])
            or int(summary["order"]) != int(selected["order"])
            or not math.isclose(float(summary["csplit"]), float(selected["csplit"]), abs_tol=5.0e-12)
            or not math.isclose(float(summary["cspread"]), float(selected["cspread"]), abs_tol=5.0e-12)
            or not bool(validation.get("performed"))
            or bool(validation.get("used_for_selection"))
        ):
            raise ValueError(
                "Figure 5 spectral AD violates the frozen-selection "
                f"contract: {directory}"
            )
        stag_screens[target] = {"selected": selected, "summary": summary}

    panel_by_operator_target = {
        ("ik", 1.0e-4): "a",
        ("ik", 1.0e-5): "b",
        ("ad", 1.0e-4): "c",
        ("ad", 1.0e-5): "d",
    }
    pppm_by_operator_target: dict[tuple[str, float], list[dict[str, str]]] = {}
    for key, panel in panel_by_operator_target.items():
        method, target = key
        rows = sorted(
            [row for row in pppm_rows if row["panel"] == panel],
            key=lambda row: int(row["actual_nx"]),
        )
        if not rows or any(
            row["differentiation"] != method
            or int(row["order"]) != 5
            or not math.isclose(value(row, "target_relative_rms"), target, abs_tol=1.0e-15)
            or int(row["holdout_frames"]) != 26
            for row in rows
        ):
            raise ValueError(
                f"Figure 5 PPPM {method} baseline is invalid for {target:.0e}"
            )
        pppm_by_operator_target[key] = rows

    order_style = {
        5: (COLORS["blue"], "o"),
        6: (COLORS["orange"], "s"),
        7: (COLORS["green"], "^"),
        8: (COLORS["vermillion"], "D"),
        9: (COLORS["purple"], "v"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), sharex="col")
    panel_specs = (
        (axes[0, 0], "ik", 1.0e-4, "a"),
        (axes[0, 1], "ik", 1.0e-5, "b"),
        (axes[1, 0], "ad", 1.0e-4, "c"),
        (axes[1, 1], "ad", 1.0e-5, "d"),
    )
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.105, top=0.83, hspace=0.41, wspace=0.22)

    ticks_by_target = {
        1.0e-4: [15, 20, 24, 32, 40, 64, 80],
        1.0e-5: [16, 20, 24, 32, 40, 64, 80],
    }
    ylims_by_target = {1.0e-4: (1.0e-5, 3.0e-2), 1.0e-5: (1.0e-6, 3.0e-2)}
    first_theory: dict[str, dict[str, int | None]] = {}
    first_ik_validation: dict[str, dict[str, int | None]] = {}
    first_ad_screen: dict[str, dict[str, int | None]] = {}
    first_ad_validation: dict[str, dict[str, int | None]] = {}
    plot_source: list[dict[str, object]] = []

    def append_source(
        *,
        panel: str,
        record_type: str,
        series: str,
        differentiation: str,
        target: float,
        order: int,
        row: dict[str, str],
        relative_rms: float,
        relative_rms_sem: float,
        source_partition: str,
        estimator: str,
        used_for_selection: bool,
        plotted: bool,
    ) -> None:
        if row.get("resolved_band", "") != "":
            resolved_band: str | bool = row["resolved_band"]
        elif row.get("sigma_up", "") != "":
            resolved_band = value(row, "sigma_up") >= 1.0
        else:
            resolved_band = ""
        plot_source.append(
            {
                "panel": panel,
                "record_type": record_type,
                "series": series,
                "differentiation": differentiation,
                "target_relative_rms": target,
                "order": order,
                "actual_nx": row["actual_nx"],
                "actual_grid_points": row["actual_grid_points"],
                "sigma_up": row.get("sigma_up", ""),
                "resolved_band": resolved_band,
                "relative_rms": relative_rms,
                "relative_rms_sem": relative_rms_sem,
                "source_partition": source_partition,
                "estimator": estimator,
                "used_for_selection": used_for_selection,
                "plotted": plotted,
            }
        )

    for ax, method, target, panel in panel_specs:
        set_log_y(ax)
        light_y_grid(ax)
        ticks = ticks_by_target[target]
        if method == "ik":
            per_target = [
                row
                for row in theory_rows
                if math.isclose(value(row, "target_relative_rms"), target, abs_tol=1.0e-15)
            ]
            first_theory[f"{target:.0e}"] = {}
            first_ik_validation[f"{target:.0e}"] = {}
        else:
            per_target = [
                row
                for row in stag_ad_rows
                if math.isclose(value(row, "target_relative_rms"), target, abs_tol=1.0e-15)
            ]
            first_ad_screen[f"{target:.0e}"] = {}
            first_ad_validation[f"{target:.0e}"] = {}

        for order in expected_orders:
            color, marker = order_style[order]
            all_rows = sorted(
                [row for row in per_target if int(row["order"]) == order],
                key=lambda row: int(row["actual_nx"]),
            )
            resolved = [row for row in all_rows if value(row, "sigma_up") >= 1.0]
            if not resolved:
                raise ValueError(f"Figure 5 has no resolved {method} rows for {target:.0e}, P={order}")

            if method == "ik":
                estimated = [value(row, "predicted_total_relative_rms") for row in resolved]
                estimated_sem = [
                    value(row, "predicted_total_relative_block5_sem") for row in resolved
                ]
                measured = [value(row, "validation_relative_rms") for row in resolved]
                measured_sem = [
                    value(row, "validation_relative_rms_balanced_block5_sem")
                    for row in resolved
                ]
                estimated_crossings = [
                    row for row in resolved
                    if str(row["prediction_passes_target"]).lower() == "true"
                ]
                measured_crossings = [
                    row for row in resolved
                    if str(row["validation_passes_target"]).lower() == "true"
                ]
                first_theory[f"{target:.0e}"][str(order)] = (
                    int(estimated_crossings[0]["actual_nx"]) if estimated_crossings else None
                )
                first_ik_validation[f"{target:.0e}"][str(order)] = (
                    int(measured_crossings[0]["actual_nx"]) if measured_crossings else None
                )
                for row in all_rows:
                    append_source(
                        panel=panel,
                        record_type="ESP theoretical prediction",
                        series=f"ESP P={order}",
                        differentiation="fixed-influence ik",
                        target=target,
                        order=order,
                        row=row,
                        relative_rms=value(row, "predicted_total_relative_rms"),
                        relative_rms_sem=value(row, "predicted_total_relative_block5_sem"),
                        source_partition="frames 1-25; no Ewald force difference",
                        estimator=row["prediction_operator"],
                        used_for_selection=True,
                        plotted=value(row, "sigma_up") >= 1.0,
                    )
                    append_source(
                        panel=panel,
                        record_type="ESP Ewald holdout validation",
                        series=f"ESP P={order}",
                        differentiation="fixed-influence ik",
                        target=target,
                        order=order,
                        row=row,
                        relative_rms=value(row, "validation_relative_rms"),
                        relative_rms_sem=value(
                            row, "validation_relative_rms_balanced_block5_sem"
                        ),
                        source_partition="frames 26-51; Ewald validation only",
                        estimator=row["validation_operator"],
                        used_for_selection=False,
                        plotted=value(row, "sigma_up") >= 1.0,
                    )
            else:
                estimated = [value(row, "predicted_total_relative_rms") for row in resolved]
                estimated_sem = [
                    value(row, "predicted_total_relative_combined_sem")
                    for row in resolved
                ]
                measured = [value(row, "validation_relative_rms") for row in resolved]
                measured_sem = [
                    value(row, "validation_relative_rms_balanced_block5_sem")
                    for row in resolved
                ]
                estimated_crossings = [
                    row for row in resolved
                    if str(row["selection_passes_target"]).lower() == "true"
                ]
                measured_crossings = [
                    row
                    for row in resolved
                    if str(row["validation_passes_target"]).lower() == "true"
                ]
                first_ad_screen[f"{target:.0e}"][str(order)] = (
                    int(estimated_crossings[0]["actual_nx"]) if estimated_crossings else None
                )
                first_ad_validation[f"{target:.0e}"][str(order)] = (
                    int(measured_crossings[0]["actual_nx"]) if measured_crossings else None
                )
                for row in all_rows:
                    append_source(
                        panel=panel,
                        record_type="AD conditional pair/self theoretical prediction",
                        series=f"ESP P={order}",
                        differentiation="AD",
                        target=target,
                        order=order,
                        row=row,
                        relative_rms=value(row, "predicted_total_relative_rms"),
                        relative_rms_sem=value(
                            row, "predicted_total_relative_combined_sem"
                        ),
                        source_partition=(
                            "frozen S_tag and conditional pair spectra from frames 1-25; "
                            "prediction reopens no coordinates or forces"
                        ),
                        estimator=row["ad_estimator"],
                        used_for_selection=True,
                        plotted=value(row, "sigma_up") >= 1.0,
                    )
                    append_source(
                        panel=panel,
                        record_type="AD Ewald holdout validation",
                        series=f"ESP P={order}",
                        differentiation="AD",
                        target=target,
                        order=order,
                        row=row,
                        relative_rms=value(row, "validation_relative_rms"),
                        relative_rms_sem=value(
                            row, "validation_relative_rms_balanced_block5_sem"
                        ),
                        source_partition="frames 26-51; Ewald validation only",
                        estimator=row["validation_operator"],
                        used_for_selection=False,
                        plotted=value(row, "sigma_up") >= 1.0,
                    )

            ax.plot(
                [value(row, "actual_grid_points") for row in resolved],
                estimated,
                color=color,
                linewidth=1.25,
                linestyle="--",
                zorder=3,
                label="_nolegend_",
            )
            errorbar(
                ax,
                [value(row, "actual_grid_points") for row in resolved],
                measured,
                measured_sem,
                color=color,
                marker=marker,
                label="_nolegend_",
                filled=True,
                marker_size=4.25,
                linestyle="none",
                zorder=5,
            )

        pppm = pppm_by_operator_target[(method, target)]
        errorbar(
            ax,
            [value(row, "actual_grid_points") for row in pppm],
            [value(row, "holdout_relative_rms") for row in pppm],
            [value(row, "holdout_balanced_block5_sem") for row in pppm],
            color=COLORS["black"],
            marker="X",
            label="_nolegend_",
            filled=False,
            marker_size=4.1,
            linestyle="none",
            zorder=4,
        )
        for row in pppm:
            append_source(
                panel=panel,
                record_type="PPPM Ewald holdout validation",
                series=f"Gaussian-split PPPM P=5 ({method})",
                differentiation=method,
                target=target,
                order=5,
                row=row,
                relative_rms=value(row, "holdout_relative_rms"),
                relative_rms_sem=value(row, "holdout_balanced_block5_sem"),
                source_partition="frames 26-51; Ewald validation",
                estimator="not plotted as an ESP theoretical estimate",
                used_for_selection=False,
                plotted=True,
            )

        if method == "ad":
            stag_screen = stag_screens[target]
            selected = stag_screen["selected"]
            summary = stag_screen["summary"]
            assert isinstance(selected, dict)
            assert isinstance(summary, dict)
            predicted = float(selected["predicted_total_relative_rms"])
            predicted_sem = float(selected["predicted_total_relative_combined_sem"])
            measured = float(summary["validation_relative_rms"])
            measured_sem = float(summary["validation_relative_rms_block5_sem"])
            order = int(selected["order"])
            append_source(
                panel=panel,
                record_type="AD frozen conditional pair/self theoretical selection",
                series=f"selected AD candidate P={order}",
                differentiation="AD",
                target=target,
                order=order,
                row=selected,
                relative_rms=predicted,
                relative_rms_sem=predicted_sem,
                source_partition=(
                    "frozen S_tag and conditional pair spectra from frames 1-25; "
                    "prediction reopens no coordinates or forces"
                ),
                estimator=selected["ad_estimator"],
                used_for_selection=True,
                plotted=False,
            )
            append_source(
                panel=panel,
                record_type="AD frozen-selection Ewald holdout validation",
                series=f"selected AD candidate P={order}",
                differentiation="AD",
                target=target,
                order=order,
                row=selected,
                relative_rms=measured,
                relative_rms_sem=measured_sem,
                source_partition="frames 26-51; Ewald validation only",
                estimator="matched production AD mesh force minus Ewald reference",
                used_for_selection=False,
                plotted=False,
            )

        csplit = value(per_target[0], "csplit")
        ax.axhline(target, color=COLORS["gray"], linestyle=":", linewidth=0.9, zorder=1)
        volume_ticks = [tick**3 for tick in ticks]
        ax.set_xscale("log")
        ax.set_xlim(min(volume_ticks) / 1.18, max(volume_ticks) * 1.18)
        ax.set_xticks(volume_ticks)
        ax.set_xticklabels([rf"${tick}^3$" for tick in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(*ylims_by_target[target])
        ax.tick_params(axis="x", labelsize=7.1)
        ax.text(
            0.975,
            0.955,
            (r"fixed-influence $\mathrm{i}\mathbf{k}$" if method == "ik" else "AD")
            + "\n"
            + rf"$\varepsilon = 10^{{{int(round(math.log10(target)))}}}$"
            + "\n"
            + (
                rf"$c_{{\rm split}}=c_{{\rm spread}}={csplit:.3f}$"
                if method == "ik"
                else rf"$c_{{\rm split}}=c_{{\rm spread}}={csplit:.3f}$"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.15,
            linespacing=1.35,
            color=COLORS["black"],
        )
        panel_label(ax, panel, x=-0.125, y=1.08, fontsize=10.0)

    axes[0, 0].set_ylabel("Relative RMS force error")
    axes[1, 0].set_ylabel("Relative RMS force error")
    axes[0, 0].tick_params(axis="x", labelbottom=False)
    axes[0, 1].tick_params(axis="x", labelbottom=False)
    for ax in axes[1, :]:
        ax.set_xlabel(r"Actual FFT grid size, $M^3$")

    handles = [
        Line2D(
            [0], [0], color=color, marker=marker, linestyle="--", linewidth=1.25,
            markersize=5.2, markerfacecolor=color, markeredgecolor=color,
            label=rf"ESP $P={order}$",
        )
        for order, (color, marker) in order_style.items()
    ]
    handles.append(
        Line2D(
            [0], [0], color=COLORS["black"], marker="X", linestyle="none",
            markersize=5.2, markerfacecolor="white", markeredgecolor=COLORS["black"],
            label=r"Gaussian-split PPPM $P=5$",
        )
    )
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.957),
        ncol=6,
        fontsize=7.4,
        handlelength=1.65,
        handletextpad=0.55,
        columnspacing=1.05,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "fig5_pppm_efficiency_plot_source.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(plot_source[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(plot_source)
    save_figure(
        fig,
        "fig5_pppm_efficiency",
        "Fixed-influence IK and conditional pair/self AD theory with independent Ewald validation",
    )
    return {
        "layout": "2 x 2",
        "esp_rows": len(theory_rows),
        "esp_theory_rows": len(theory_rows),
        "ad_stag_screen_rows": len(stag_ad_rows),
        "pppm_rows": sum(len(rows) for rows in pppm_by_operator_target.values()),
        "pppm_rows_by_panel": {
            panel_by_operator_target[key]: len(rows)
            for key, rows in pppm_by_operator_target.items()
        },
        "esp_displayed_rows": sum(
            value(row, "sigma_up") >= 1.0 for row in theory_rows
        ),
        "ad_stag_screen_displayed_rows": sum(
            value(row, "sigma_up") >= 1.0 for row in stag_ad_rows
        ),
        "pppm_displayed_rows": sum(len(rows) for rows in pppm_by_operator_target.values()),
        "orders": list(expected_orders),
        "first_theory_resolved_crossings": first_theory,
        "first_validation_resolved_crossings": first_ik_validation,
        "first_ad_stag_screen_resolved_crossings": first_ad_screen,
        "first_ad_validation_resolved_crossings": first_ad_validation,
        "theory_frames": [1, 25],
        "theory_uses_reference_force_differences": False,
        "theory_curve_style": "dashed line without markers",
        "ad_stag_screen_uses_reference_force_differences": False,
        "ad_stag_screen_curve_style": "dashed line without markers",
        "ad_stag_screen_selection_scope": (
            "frozen conditional pair/self AD theory; selected candidates "
            "and independent holdout values are retained in source data only"
        ),
        "ad_stag_selection_rows": 2,
        "ad_stag_selection": {
            f"{target:.0e}": {
                "actual_nx": int(screen["selected"]["actual_nx"]),
                "actual_grid_points": int(screen["selected"]["actual_grid_points"]),
                "order": int(screen["selected"]["order"]),
                "csplit": float(screen["selected"]["csplit"]),
                "cspread": float(screen["selected"]["cspread"]),
                "prediction_relative_rms": float(
                    screen["selected"]["predicted_total_relative_rms"]
                ),
                "prediction_combined_sem_relative": float(
                    screen["selected"]["predicted_total_relative_combined_sem"]
                ),
                "holdout_relative_rms": float(screen["summary"]["validation_relative_rms"]),
                "holdout_block5_sem": float(
                    screen["summary"]["validation_relative_rms_block5_sem"]
                ),
                "prediction_to_measurement_ratio": (
                    float(screen["selected"]["predicted_total_relative_rms"])
                    / float(screen["summary"]["validation_relative_rms"])
                ),
                "selection_used_holdout": False,
            }
            for target, screen in stag_screens.items()
        },
        "ad_stag_selection_uses_reference_force_differences": False,
        "ad_stag_selection_holdout_frames": [26, 51],
        "ad_stag_selection_curve_style": (
            "frozen conditional pair/self AD theory selection and independent validation "
            "retained in source data but not plotted separately"
        ),
        "validation_curve_style": "filled markers with balanced-block SEM; no connecting line",
        "pppm_curve_style": "open X markers with balanced-block SEM; no connecting line",
        "ad_panels_included": True,
        "below_band_points_omitted_from_main_plot": True,
    }


def figure7_archived_random_charge() -> float:
    """Plot the common-split trade-off plus a conventional-PPPM baseline."""
    summary = read_path_rows(
        WINDOW_BENCHMARK / "window_upsampling_validation_summary.csv"
    )
    m16_summary = read_path_rows(
        WINDOW_M16 / "window_upsampling_validation_summary_m16.csv"
    )
    gaussian_summary = read_path_rows(
        GAUSSIAN_PPPM_CONTROL / "gaussian_pppm_validation_summary.csv"
    )
    all_accuracy_rows = m16_summary + summary
    timing = read_path_rows(WINDOW_TIMING / "window_upsampling_timing_summary.csv")
    method_style = {
        "pswf": (COLORS["orange"], "PSWF-split/PSWF-spread"),
        "bspline": (COLORS["blue"], "PSWF-split/B-spline-spread"),
        "gaussian_pppm": (COLORS["gray"], "Gaussian-split/B-spline PPPM"),
    }

    def candidate(window: str, mesh: int, order: int) -> dict[str, str]:
        matches = [
            row
            for row in summary
            if row["window"] == window
            and int(round(value(row, "requested_mesh"))) == mesh
            and int(round(value(row, "order"))) == order
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one {window} M={mesh}, P={order} row")
        return matches[0]

    # Select the minimum *measured* feasible order on the resolved solid-curve
    # domain; requested below-band calibrations are handled separately below.
    minimum_rows: list[dict[str, str | float | int]] = []
    for window in ("pswf", "bspline"):
        for mesh in FIGURE7_RESOLVED_MESHES:
            rows_at_mesh = sorted(
                (
                    row
                    for row in summary
                    if row["window"] == window
                    and parse_grid(row["actual_grid"])[0] == mesh
                ),
                key=lambda row: int(round(value(row, "order"))),
            )
            if len(rows_at_mesh) != 6:
                raise ValueError(f"incomplete {window} M={mesh} order scan")
            if value(rows_at_mesh[0], "actual_sigma_up") <= 1.0:
                raise ValueError(f"main Figure 6 contains unresolved grid M={mesh}")
            feasible = [
                row
                for row in rows_at_mesh
                if value(row, "rms_relative_force_error_mean") <= FIGURE7_TARGET
            ]
            if not feasible:
                raise ValueError(f"no {window} M={mesh} candidate meets 1e-4")
            selected = min(feasible, key=lambda row: int(round(value(row, "order"))))
            order = int(round(value(selected, "order")))
            previous = next(
                (row for row in rows_at_mesh if int(round(value(row, "order"))) == order - 1),
                None,
            )
            if previous is not None and value(previous, "rms_relative_force_error_mean") <= FIGURE7_TARGET:
                raise ValueError(f"{window} M={mesh}, P={order} is not the minimum")
            minimum_rows.append(
                {
                    "panel": "a",
                    "window": window,
                    "branch": method_style[window][1],
                    "actual_mesh": mesh,
                    "actual_grid_points": mesh**3,
                    "actual_sigma_up": value(selected, "actual_sigma_up"),
                    "minimum_feasible_order": order,
                    "selected_relative_error_mean": value(
                        selected, "rms_relative_force_error_mean"
                    ),
                    "selected_relative_error_std": value(
                        selected, "rms_relative_force_error_std"
                    ),
                    "previous_order": "" if previous is None else order - 1,
                    "previous_relative_error_mean": (
                        "" if previous is None else value(previous, "rms_relative_force_error_mean")
                    ),
                    "target": FIGURE7_TARGET,
                    "band_status": "resolved",
                    "display_order": order,
                    "highest_tested_order": 8,
                    "selection": "minimum measured P with mean relative error <= target",
                }
            )

    # Conventional PPPM is a baseline, not a third fixed-PSWF-split window.
    # Native LAMMPS PPPM selects the Gaussian split while optimizing its own
    # requested accuracy.  Only the measured target-feasible points are drawn;
    # the raw scan also contains the M=16, 20, and 24 non-feasible records.
    gaussian_minimum_rows: list[dict[str, str | float | int]] = []
    for mesh in FIGURE7_GAUSSIAN_MESHES:
        rows_at_mesh = sorted(
            (
                row
                for row in gaussian_summary
                if int(round(value(row, "requested_mesh"))) == mesh
            ),
            key=lambda row: int(round(value(row, "order"))),
        )
        if len(rows_at_mesh) != 5:
            raise ValueError(f"incomplete Gaussian PPPM M={mesh} order scan")
        feasible = [
            row
            for row in rows_at_mesh
            if value(row, "rms_relative_force_error_mean") <= FIGURE7_TARGET
        ]
        if not feasible:
            gaussian_minimum_rows.append(
                {
                    "panel": "a",
                    "window": "gaussian_pppm",
                    "branch": method_style["gaussian_pppm"][1],
                    "actual_mesh": mesh,
                    "actual_grid_points": mesh**3,
                    "actual_sigma_up": "",
                    "minimum_feasible_order": "",
                    "selected_relative_error_mean": "",
                    "selected_relative_error_std": "",
                    "previous_order": "",
                    "previous_relative_error_mean": "",
                    "target": FIGURE7_TARGET,
                    "band_status": "conventional_control_not_feasible",
                    "display_order": 7,
                    "highest_tested_order": 7,
                    "selection": "target not reached through native PPPM P=7",
                }
            )
            continue
        selected = min(feasible, key=lambda row: int(round(value(row, "order"))))
        order = int(round(value(selected, "order")))
        previous = next(
            (row for row in rows_at_mesh if int(round(value(row, "order"))) == order - 1),
            None,
        )
        if previous is not None and value(previous, "rms_relative_force_error_mean") <= FIGURE7_TARGET:
            raise ValueError(f"Gaussian PPPM M={mesh}, P={order} is not the minimum")
        gaussian_minimum_rows.append(
            {
                "panel": "a",
                "window": "gaussian_pppm",
                "branch": method_style["gaussian_pppm"][1],
                "actual_mesh": mesh,
                "actual_grid_points": mesh**3,
                "actual_sigma_up": "",
                "minimum_feasible_order": order,
                "selected_relative_error_mean": value(
                    selected, "rms_relative_force_error_mean"
                ),
                "selected_relative_error_std": value(
                    selected, "rms_relative_force_error_std"
                ),
                "previous_order": "" if previous is None else order - 1,
                "previous_relative_error_mean": (
                    "" if previous is None else value(previous, "rms_relative_force_error_mean")
                ),
                "target": FIGURE7_TARGET,
                "band_status": "conventional_control",
                "display_order": order,
                "highest_tested_order": 7,
                "selection": "minimum measured native-PPPM P with mean relative error <= target",
            }
        )

    # M=16 and M=20 are available below-band calibration data.  Only M=20 is
    # shown in the main panel; the more strongly underresolved M=16 results
    # remain in the SI diagnostic sweep.
    calibration_rows: list[dict[str, str | float | int]] = []
    for window in ("pswf", "bspline"):
        for mesh in FIGURE7_CALIBRATION_MESHES:
            source = m16_summary if mesh == 16 else summary
            rows_at_mesh = sorted(
                (
                    row for row in source
                    if row["window"] == window
                    and parse_grid(row["actual_grid"])[0] == mesh
                ),
                key=lambda row: int(round(value(row, "order"))),
            )
            if len(rows_at_mesh) != 6:
                raise ValueError(f"incomplete calibration scan for {window} M={mesh}")
            sigma_up = value(rows_at_mesh[0], "actual_sigma_up")
            if sigma_up >= 1.0:
                raise ValueError(f"calibration grid M={mesh} unexpectedly resolves the band")
            feasible = [
                row for row in rows_at_mesh
                if value(row, "rms_relative_force_error_mean") <= FIGURE7_TARGET
            ]
            if feasible:
                selected = min(feasible, key=lambda row: int(round(value(row, "order"))))
                order = int(round(value(selected, "order")))
                selection = "below-band calibration; minimum measured feasible P"
                minimum_order: int | str = order
            else:
                selected = min(
                    rows_at_mesh,
                    key=lambda row: value(row, "rms_relative_force_error_mean"),
                )
                order = int(round(value(selected, "order")))
                if mesh != 16 or order != 8:
                    raise ValueError(
                        f"unexpected non-feasible calibration endpoint {window} M={mesh}, P={order}"
                    )
                selection = "below-band calibration; target not reached for P<=8"
                minimum_order = ""
            calibration_rows.append(
                {
                    "panel": "a",
                    "window": window,
                    "branch": method_style[window][1],
                    "actual_mesh": mesh,
                    "actual_grid_points": mesh**3,
                    "actual_sigma_up": sigma_up,
                    "minimum_feasible_order": minimum_order,
                    "selected_relative_error_mean": value(
                        selected, "rms_relative_force_error_mean"
                    ),
                    "selected_relative_error_std": value(
                        selected, "rms_relative_force_error_std"
                    ),
                    "target": FIGURE7_TARGET,
                    "band_status": "below_band_calibration",
                    "display_order": order,
                    "highest_tested_order": 8,
                    "selection": selection,
                }
            )

    timing_by_id = {row["case_id"]: row for row in timing}
    classified_timing_ids = set(FIGURE7_TIMING_ORDER) | set(FIGURE7_TIMING_ARCHIVED_ONLY)
    if set(timing_by_id) != classified_timing_ids:
        raise ValueError("Figure-6 timing source does not match the classified cases")
    timing_rows = [timing_by_id[case_id] for case_id in FIGURE7_TIMING_ORDER]
    for row in timing_rows:
        if row["window"] != "gaussian_pppm" and value(row, "actual_sigma_up") <= 1.0:
            raise ValueError(f"unresolved timing candidate {row['case_id']}")
        if value(row, "measured_relative_error_mean") > FIGURE7_TARGET:
            raise ValueError(f"infeasible timing candidate {row['case_id']}")

    # Clean, machine-readable source table for the two main-text panels.
    main_source_fields = [
        "panel", "case_id", "window", "branch", "actual_mesh",
        "actual_grid_points", "actual_sigma_up", "order", "minimum_feasible_order",
        "display_order", "highest_tested_order", "band_status",
        "measured_relative_error_mean", "measured_relative_error_std",
        "wall_time_per_step_mean_s", "wall_time_per_step_std_s", "n_repeats",
        "timed_steps_per_repeat", "target", "selection",
    ]
    main_source: list[dict[str, object]] = []
    for row in minimum_rows:
        main_source.append(
            {
                "panel": "a",
                "case_id": "",
                "window": row["window"],
                "branch": row["branch"],
                "actual_mesh": row["actual_mesh"],
                "actual_grid_points": row["actual_grid_points"],
                "actual_sigma_up": row["actual_sigma_up"],
                "order": "",
                "minimum_feasible_order": row["minimum_feasible_order"],
                "display_order": row["display_order"],
                "highest_tested_order": row["highest_tested_order"],
                "band_status": row["band_status"],
                "measured_relative_error_mean": row["selected_relative_error_mean"],
                "measured_relative_error_std": row["selected_relative_error_std"],
                "wall_time_per_step_mean_s": "",
                "wall_time_per_step_std_s": "",
                "n_repeats": "",
                "timed_steps_per_repeat": "",
                "target": FIGURE7_TARGET,
                "selection": row["selection"],
            }
        )
    for row in calibration_rows:
        if int(row["actual_mesh"]) == 16:
            continue
        main_source.append(
            {
                "panel": "a",
                "case_id": "",
                "window": row["window"],
                "branch": row["branch"],
                "actual_mesh": row["actual_mesh"],
                "actual_grid_points": row["actual_grid_points"],
                "actual_sigma_up": row["actual_sigma_up"],
                "order": "",
                "minimum_feasible_order": row["minimum_feasible_order"],
                "display_order": row["display_order"],
                "highest_tested_order": row["highest_tested_order"],
                "band_status": row["band_status"],
                "measured_relative_error_mean": row["selected_relative_error_mean"],
                "measured_relative_error_std": row["selected_relative_error_std"],
                "wall_time_per_step_mean_s": "",
                "wall_time_per_step_std_s": "",
                "n_repeats": "",
                "timed_steps_per_repeat": "",
                "target": FIGURE7_TARGET,
                "selection": row["selection"],
            }
        )
    for row in gaussian_minimum_rows:
        # M=16 is retained solely in the SI diagnostic sweep and is omitted
        # from the main-text source table together with the plotted panel.
        if int(row["actual_mesh"]) == 16:
            continue
        main_source.append(
            {
                "panel": "a",
                "case_id": "",
                "window": row["window"],
                "branch": row["branch"],
                "actual_mesh": row["actual_mesh"],
                "actual_grid_points": row["actual_grid_points"],
                "actual_sigma_up": row["actual_sigma_up"],
                "order": "",
                "minimum_feasible_order": row["minimum_feasible_order"],
                "display_order": row["display_order"],
                "highest_tested_order": row["highest_tested_order"],
                "band_status": row["band_status"],
                "measured_relative_error_mean": row["selected_relative_error_mean"],
                "measured_relative_error_std": row["selected_relative_error_std"],
                "wall_time_per_step_mean_s": "",
                "wall_time_per_step_std_s": "",
                "n_repeats": "",
                "timed_steps_per_repeat": "",
                "target": FIGURE7_TARGET,
                "selection": row["selection"],
            }
        )
    for row in timing_rows:
        mesh = parse_grid(row["actual_grid"])[0]
        main_source.append(
            {
                "panel": "b",
                "case_id": row["case_id"],
                "window": row["window"],
                "branch": method_style[row["window"]][1],
                "actual_mesh": mesh,
                "actual_grid_points": mesh**3,
                "actual_sigma_up": row["actual_sigma_up"],
                "order": row["order"],
                "minimum_feasible_order": "",
                "display_order": "",
                "highest_tested_order": "",
                "band_status": "resolved_timing",
                "measured_relative_error_mean": row["measured_relative_error_mean"],
                "measured_relative_error_std": row["measured_relative_error_std"],
                "wall_time_per_step_mean_s": row["wall_time_per_step_mean_s"],
                "wall_time_per_step_std_s": row["wall_time_per_step_std_s"],
                "n_repeats": row["n_repeats"],
                "timed_steps_per_repeat": row["timed_steps_per_repeat"],
                "target": FIGURE7_TARGET,
                "selection": "prespecified accuracy-feasible timing candidate",
            }
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "fig7_window_upsampling_plot_source.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=main_source_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(main_source)

    # Archived diagnostic rendering retained only to preserve the independently
    # reproducible timing record.  It is overwritten below: the manuscript
    # Figure 6 contains the two target-specific grid--stencil sweeps instead.
    fig, axes = plt.subplots(
        1, 2, figsize=FIGURE7_SIZE,
        gridspec_kw={"width_ratios": [1.08, 1.22]},
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.91, wspace=0.33)

    ax = axes[0]
    for window in ("pswf", "bspline"):
        color, label = method_style[window]
        rows_for_window = sorted(
            (row for row in minimum_rows if row["window"] == window),
            key=lambda row: int(row["actual_mesh"]),
        )
        # Retain the measured M=20 diagnostic as the first point of the
        # displayed trend.  It lies just below the nominal band edge, but the
        # solid connector makes the grid--stencil progression visually
        # continuous; its status remains explicit in the source table and
        # caption without adding a visual region marker.
        m20 = next(
            row for row in calibration_rows
            if row["window"] == window and int(row["actual_mesh"]) == 20
        )
        displayed_rows = [m20, *rows_for_window]
        ax.plot(
            [int(row["actual_mesh"]) for row in displayed_rows],
            [int(row["minimum_feasible_order"]) for row in displayed_rows],
            color=color,
            marker="s" if window == "pswf" else "o",
            markerfacecolor=color if window == "pswf" else "white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            linewidth=1.35,
            label=label,
            zorder=3,
        )
    gaussian_visible = [
        row
        for row in gaussian_minimum_rows
        if row["minimum_feasible_order"] != "" and int(row["actual_mesh"]) <= 40
    ]
    ax.plot(
        [int(row["actual_mesh"]) for row in gaussian_visible],
        [int(row["minimum_feasible_order"]) for row in gaussian_visible],
        color=COLORS["gray"],
        marker="D",
        markerfacecolor="white",
        markeredgecolor=COLORS["gray"],
        markeredgewidth=0.9,
        linestyle=(0, (3.0, 1.5)),
        linewidth=1.2,
        label=method_style["gaussian_pppm"][1],
        zorder=3,
    )
    mesh_ticks = [20, *FIGURE7_RESOLVED_MESHES]
    ax.set_xticks(mesh_ticks)
    ax.set_xticklabels([rf"${mesh}^3$" for mesh in mesh_ticks])
    ax.set_xlim(18.5, 41.0)
    ax.set_ylim(3.55, 8.55)
    ax.set_yticks([4, 5, 6, 7, 8])
    ax.set_xlabel(r"Actual FFT grid, $M^3$")
    ax.set_ylabel(r"Minimum feasible spreading order, $P$")
    ax.set_title(r"Grid--stencil trade-off at $10^{-4}$", pad=4)
    ax.grid(axis="y", color="#E8EBEE", linewidth=0.55, zorder=0)
    ax.legend(loc="upper right", fontsize=5.0, handlelength=1.3, labelspacing=0.25)
    panel_label(ax, "a", x=-0.15)

    ax = axes[1]
    positions = np.arange(len(timing_rows))
    means_ms = np.array(
        [1000.0 * value(row, "wall_time_per_step_mean_s") for row in timing_rows]
    )
    std_ms = np.array(
        [1000.0 * value(row, "wall_time_per_step_std_s") for row in timing_rows]
    )
    face_colors = [
        (
            "#F4D69B"
            if row["window"] == "pswf"
            else "#B9DCF0"
            if row["window"] == "bspline"
            else COLORS["light_gray"]
        )
        for row in timing_rows
    ]
    edge_colors = [method_style[row["window"]][0] for row in timing_rows]
    bars = ax.bar(
        positions, means_ms, yerr=std_ms, width=0.66,
        color=face_colors, edgecolor=edge_colors, linewidth=0.85,
        error_kw={"elinewidth": 0.75, "capsize": 2.0, "capthick": 0.7},
        zorder=3,
    )
    annotation_offsets = (0.15, 0.075, 0.075, 0.15, 0.075, 0.075)
    for bar, row, mean_ms, std_value, offset in zip(
        bars, timing_rows, means_ms, std_ms, annotation_offsets
    ):
        ax.text(
            bar.get_x() + 0.5 * bar.get_width(),
            mean_ms + std_value + offset,
            rf"${mean_ms:.3f}$",
            ha="center", va="bottom", fontsize=5.6,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            rf"${parse_grid(row['actual_grid'])[0]}^3$"
            + "\n"
            + rf"$P={int(round(value(row, 'order')))}$"
            for row in timing_rows
        ],
        fontsize=6.0,
    )
    ax.set_ylim(0.0, 4.8)
    ax.set_ylabel("Wall time (ms/step)")
    ax.set_title("Archived timing diagnostic", pad=4)
    ax.grid(axis="y", color="#E8EBEE", linewidth=0.55, zorder=0)
    ax.text(
        0.03, 0.96, "5 repeats; 1 rank, 1 thread",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=6.0, color=COLORS["gray"],
    )
    ax.text(
        0.03, 0.90, "Colours identify the branches in panel a",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=5.35, color=COLORS["gray"],
    )
    panel_label(ax, "b", x=-0.11)

    save_figure(
        fig,
        "fig7_window_upsampling",
        "PSWF grid-window trade-off with a conventional PPPM control",
    )

    # Replace the temporary timing rendering above with the two accuracy
    # panels used in the manuscript: the same measured grid--stencil sweep at
    # 1e-4 and 1e-5.  Timing records remain archived separately and are not a
    # main-text performance claim.
    def target_common_rows(
        target: float, panel: str,
    ) -> tuple[list[dict[str, str | float | int]], list[dict[str, str | float | int]]]:
        resolved_rows: list[dict[str, str | float | int]] = []
        m20_rows: list[dict[str, str | float | int]] = []
        for window in ("pswf", "bspline"):
            for mesh in FIGURE7_RESOLVED_MESHES:
                rows_at_mesh = sorted(
                    (
                        row
                        for row in summary
                        if row["window"] == window
                        and parse_grid(row["actual_grid"])[0] == mesh
                    ),
                    key=lambda row: int(round(value(row, "order"))),
                )
                feasible = [
                    row for row in rows_at_mesh
                    if value(row, "rms_relative_force_error_mean") <= target
                ]
                if feasible:
                    selected = min(
                        feasible, key=lambda row: int(round(value(row, "order")))
                    )
                    minimum_order: int | str = int(round(value(selected, "order")))
                    status = "resolved"
                    selection = "minimum measured P with mean relative error <= target"
                else:
                    selected = min(
                        rows_at_mesh,
                        key=lambda row: value(row, "rms_relative_force_error_mean"),
                    )
                    minimum_order = ""
                    status = "resolved_target_not_reached"
                    selection = "target not reached through P<=8"
                resolved_rows.append(
                    {
                        "panel": panel,
                        "window": window,
                        "branch": method_style[window][1],
                        "actual_mesh": mesh,
                        "actual_grid_points": mesh**3,
                        "actual_sigma_up": value(selected, "actual_sigma_up"),
                        "minimum_feasible_order": minimum_order,
                        "selected_relative_error_mean": value(
                            selected, "rms_relative_force_error_mean"
                        ),
                        "selected_relative_error_std": value(
                            selected, "rms_relative_force_error_std"
                        ),
                        "target": target,
                        "band_status": status,
                        "display_order": int(round(value(selected, "order"))),
                        "highest_tested_order": 8,
                        "selection": selection,
                    }
                )

            m20_scan = sorted(
                (
                    row
                    for row in summary
                    if row["window"] == window
                    and parse_grid(row["actual_grid"])[0] == 20
                ),
                key=lambda row: int(round(value(row, "order"))),
            )
            feasible = [
                row for row in m20_scan
                if value(row, "rms_relative_force_error_mean") <= target
            ]
            if feasible:
                selected = min(
                    feasible, key=lambda row: int(round(value(row, "order")))
                )
                minimum_order = int(round(value(selected, "order")))
                status = "near_band_edge_calibration"
                selection = "near-band-edge calibration; minimum measured feasible P"
            else:
                selected = min(
                    m20_scan,
                    key=lambda row: value(row, "rms_relative_force_error_mean"),
                )
                minimum_order = ""
                status = "near_band_edge_target_not_reached"
                selection = "near-band-edge calibration; target not reached through P<=8"
            m20_rows.append(
                {
                    "panel": panel,
                    "window": window,
                    "branch": method_style[window][1],
                    "actual_mesh": 20,
                    "actual_grid_points": 20**3,
                    "actual_sigma_up": value(selected, "actual_sigma_up"),
                    "minimum_feasible_order": minimum_order,
                    "selected_relative_error_mean": value(
                        selected, "rms_relative_force_error_mean"
                    ),
                    "selected_relative_error_std": value(
                        selected, "rms_relative_force_error_std"
                    ),
                    "target": target,
                    "band_status": status,
                    "display_order": int(round(value(selected, "order"))),
                    "highest_tested_order": 8,
                    "selection": selection,
                }
            )
        return resolved_rows, m20_rows

    def target_gaussian_rows(
        target: float, panel: str,
    ) -> list[dict[str, str | float | int]]:
        selected_rows: list[dict[str, str | float | int]] = []
        for mesh in (20, 24, 30, 36, 40, 48):
            rows_at_mesh = sorted(
                (
                    row for row in gaussian_summary
                    if int(round(value(row, "requested_mesh"))) == mesh
                ),
                key=lambda row: int(round(value(row, "order"))),
            )
            feasible = [
                row for row in rows_at_mesh
                if value(row, "rms_relative_force_error_mean") <= target
            ]
            if feasible:
                selected = min(
                    feasible, key=lambda row: int(round(value(row, "order")))
                )
                minimum_order: int | str = int(round(value(selected, "order")))
                status = "conventional_control"
                selection = "minimum measured native-PPPM P with mean relative error <= target"
            else:
                selected = min(
                    rows_at_mesh,
                    key=lambda row: value(row, "rms_relative_force_error_mean"),
                )
                minimum_order = ""
                status = "conventional_control_target_not_reached"
                selection = "target not reached through native PPPM P=7"
            selected_rows.append(
                {
                    "panel": panel,
                    "window": "gaussian_pppm",
                    "branch": method_style["gaussian_pppm"][1],
                    "actual_mesh": mesh,
                    "actual_grid_points": mesh**3,
                    "actual_sigma_up": "",
                    "minimum_feasible_order": minimum_order,
                    "selected_relative_error_mean": value(
                        selected, "rms_relative_force_error_mean"
                    ),
                    "selected_relative_error_std": value(
                        selected, "rms_relative_force_error_std"
                    ),
                    "target": target,
                    "band_status": status,
                    "display_order": int(round(value(selected, "order"))),
                    "highest_tested_order": 7,
                    "selection": selection,
                }
            )
        return selected_rows

    accuracy_panel_a, calibration_panel_a = target_common_rows(FIGURE7_TARGET, "a")
    gaussian_panel_a = target_gaussian_rows(FIGURE7_TARGET, "a")
    accuracy_panel_b, calibration_panel_b = target_common_rows(FIGURE7_TIGHT_TARGET, "b")
    gaussian_panel_b = target_gaussian_rows(FIGURE7_TIGHT_TARGET, "b")

    # Replace the timing-oriented source table with the two measured accuracy
    # sweeps that are actually shown in the main-text Figure 6.
    main_source = []
    for panel_rows, calibration, gaussian in (
        (accuracy_panel_a, calibration_panel_a, gaussian_panel_a),
        (accuracy_panel_b, calibration_panel_b, gaussian_panel_b),
    ):
        for row in [*panel_rows, *calibration, *gaussian]:
            main_source.append(
                {
                    "panel": row["panel"],
                    "case_id": "",
                    "window": row["window"],
                    "branch": row["branch"],
                    "actual_mesh": row["actual_mesh"],
                    "actual_grid_points": row["actual_grid_points"],
                    "actual_sigma_up": row["actual_sigma_up"],
                    "order": "",
                    "minimum_feasible_order": row["minimum_feasible_order"],
                    "display_order": row["display_order"],
                    "highest_tested_order": row["highest_tested_order"],
                    "band_status": row["band_status"],
                    "measured_relative_error_mean": row["selected_relative_error_mean"],
                    "measured_relative_error_std": row["selected_relative_error_std"],
                    "wall_time_per_step_mean_s": "",
                    "wall_time_per_step_std_s": "",
                    "n_repeats": "",
                    "timed_steps_per_repeat": "",
                    "target": row["target"],
                    "selection": row["selection"],
                }
            )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "fig7_window_upsampling_plot_source.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=main_source_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(main_source)

    def plot_accuracy_tradeoff(
        axis: plt.Axes,
        target: float,
        resolved: list[dict[str, str | float | int]],
        m20: list[dict[str, str | float | int]],
        gaussian: list[dict[str, str | float | int]],
        panel: str,
    ) -> None:
        for window in ("pswf", "bspline"):
            color, label = method_style[window]
            rows_for_window = [
                row for row in m20
                if row["window"] == window and row["minimum_feasible_order"] != ""
            ] + [
                row for row in resolved
                if row["window"] == window and row["minimum_feasible_order"] != ""
            ]
            rows_for_window.sort(key=lambda row: int(row["actual_mesh"]))
            if not rows_for_window:
                continue
            axis.plot(
                [int(row["actual_mesh"]) for row in rows_for_window],
                [int(row["minimum_feasible_order"]) for row in rows_for_window],
                color=color,
                marker="s" if window == "pswf" else "o",
                markerfacecolor=color if window == "pswf" else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                linewidth=1.35,
                label=label,
                zorder=3,
            )
        conventional = [
            row for row in gaussian if row["minimum_feasible_order"] != ""
        ]
        if conventional:
            axis.plot(
                [int(row["actual_mesh"]) for row in conventional],
                [int(row["minimum_feasible_order"]) for row in conventional],
                color=COLORS["gray"],
                marker="D",
                markerfacecolor="white",
                markeredgecolor=COLORS["gray"],
                markeredgewidth=0.9,
                linestyle=(0, (3.0, 1.5)),
                linewidth=1.2,
                label=method_style["gaussian_pppm"][1],
                zorder=3,
            )
        mesh_ticks = [20, 24, 30, 36, 40, 48]
        axis.set_xticks(mesh_ticks)
        axis.set_xticklabels([rf"${mesh}^3$" for mesh in mesh_ticks])
        axis.set_xlim(18.5, 49.5)
        if math.isclose(target, FIGURE7_TARGET):
            axis.set_ylim(3.55, 8.55)
            axis.set_yticks([4, 5, 6, 7, 8])
        else:
            axis.set_ylim(5.55, 8.45)
            axis.set_yticks([6, 7, 8])
        axis.set_xlabel(r"Actual FFT grid, $M^3$")
        axis.set_ylabel(r"Minimum feasible spreading order, $P$")
        exponent = int(round(math.log10(target)))
        axis.set_title(rf"Grid--stencil trade-off at $10^{{{exponent}}}$", pad=4)
        axis.grid(axis="y", color="#E8EBEE", linewidth=0.55, zorder=0)
        if panel == "a":
            # The figure is reduced slightly by the JCTC manuscript layout;
            # keep the branch labels readable at final PDF scale.
            axis.legend(loc="upper right", fontsize=6.0, handlelength=1.25, labelspacing=0.22)

    fig, axes = plt.subplots(1, 2, figsize=FIGURE7_SIZE)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.91, wspace=0.36)
    plot_accuracy_tradeoff(
        axes[0], FIGURE7_TARGET, accuracy_panel_a, calibration_panel_a,
        gaussian_panel_a, "a",
    )
    plot_accuracy_tradeoff(
        axes[1], FIGURE7_TIGHT_TARGET, accuracy_panel_b, calibration_panel_b,
        gaussian_panel_b, "b",
    )
    panel_label(axes[0], "a", x=-0.15)
    panel_label(axes[1], "b", x=-0.15)
    save_figure(
        fig,
        "fig7_window_upsampling",
        "PSWF grid--window trade-off at 1e-4 and 1e-5 with a conventional PPPM control",
    )

    # SI panel (a): the full measured force-error sweep, including the M=16
    # and M=20 below-band diagnostics.  SI panel (b): the operation-count proxy.
    bspline_cost = candidate("bspline", 36, 4)
    pswf_cost = candidate("pswf", 24, 5)
    base_pg = 2.0 * 512 * 4**3
    base_fft = 4.0 * 20**3 * math.log2(20**3)
    b_pg = value(bspline_cost, "particle_grid_proxy") / base_pg
    b_fft = value(bspline_cost, "fft_proxy_ik") / base_fft
    p_pg = value(pswf_cost, "particle_grid_proxy") / base_pg
    p_fft = value(pswf_cost, "fft_proxy_ik") / base_fft
    crossover = (p_pg - b_pg) / (b_fft - p_fft)
    if not math.isclose(crossover, 0.18532045101039687, rel_tol=2.0e-12):
        raise ValueError(f"unexpected SI cost-proxy crossover {crossover}")

    weights = np.logspace(-2, 1, 300)
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SI_WINDOW_SIZE)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.91, wspace=0.33)
    ax = axes[0]
    set_log_y(ax)
    light_y_grid(ax)
    order_style = {3: ("v", "-"), 4: ("o", "-"), 5: ("s", "--"), 6: ("^", ":"), 7: ("D", "-."), 8: ("P", (0, (1, 1)))}
    all_meshes = sorted({parse_grid(row["actual_grid"])[0] for row in all_accuracy_rows})
    for window in ("pswf", "bspline"):
        color, _ = method_style[window]
        for order in range(3, 9):
            marker, linestyle = order_style[order]
            subset = sorted(
                (
                    row for row in all_accuracy_rows
                    if row["window"] == window
                    and int(round(value(row, "order"))) == order
                ),
                key=lambda row: parse_grid(row["actual_grid"])[0],
            )
            ax.errorbar(
                [parse_grid(row["actual_grid"])[0] for row in subset],
                [value(row, "rms_relative_force_error_mean") for row in subset],
                yerr=[value(row, "rms_relative_force_error_std") for row in subset],
                color=color, marker=marker, linestyle=linestyle,
                markerfacecolor=color if window == "pswf" else "white",
                markeredgecolor=color, markeredgewidth=0.7,
                markersize=3.5, linewidth=0.9, elinewidth=0.5, capsize=1.3,
                alpha=0.88, zorder=3,
            )
    ax.axhline(FIGURE7_TARGET, color=COLORS["gray"], linestyle=":", linewidth=0.8)
    ax.axvspan(14.8, FIGURE7_NOMINAL_BAND_EDGE_M,
               color=COLORS["light_gray"], alpha=0.38, zorder=0)
    ax.axvline(FIGURE7_NOMINAL_BAND_EDGE_M, color=COLORS["gray"],
               linestyle=":", linewidth=0.75, zorder=1)
    ax.set_xticks(all_meshes)
    ax.set_xticklabels([rf"${mesh}^3$" for mesh in all_meshes], fontsize=6.2)
    ax.set_ylim(4.5e-6, 8.0e-3)
    ax.set_xlabel(r"Actual FFT grid, $M^3$")
    ax.set_ylabel("Mean relative RMS force error")
    ax.set_title("Full measured window/order sweep", pad=4)
    window_handles = [
        Line2D([0], [0], color=method_style[window][0], marker="o",
               markerfacecolor=method_style[window][0] if window == "pswf" else "white",
               label=method_style[window][1])
        for window in ("pswf", "bspline")
    ]
    order_handles = [
        Line2D([0], [0], color=COLORS["black"], marker=order_style[order][0],
               linestyle=order_style[order][1], markerfacecolor="white", label=rf"$P={order}$")
        for order in range(3, 9)
    ]
    legend1 = ax.legend(handles=window_handles, loc="upper right", fontsize=5.2, labelspacing=0.25)
    ax.add_artist(legend1)
    ax.legend(handles=order_handles, loc="lower left", fontsize=5.0, ncol=3,
              handlelength=1.3, columnspacing=0.55, labelspacing=0.2)
    panel_label(ax, "a", x=-0.15)

    ax = axes[1]
    b_total = b_pg + weights * b_fft
    p_total = p_pg + weights * p_fft
    ax.loglog(weights, b_total, color=COLORS["blue"],
              label=r"PSWF-split/B-spline-spread, $36^3$, $P=4$")
    ax.loglog(weights, p_total, color=COLORS["orange"],
              label=r"PSWF-split/PSWF-spread, $24^3$, $P=5$")
    ax.axvline(crossover, color=COLORS["gray"], linestyle=":", linewidth=0.85)
    cross_cost = b_pg + crossover * b_fft
    ax.scatter([crossover], [cross_cost], color=COLORS["black"], s=13, zorder=5)
    ax.annotate(
        rf"$w_{{\rm FFT}}={crossover:.3f}$",
        xy=(crossover, cross_cost), xytext=(14, 12), textcoords="offset points",
        fontsize=6.0, color=COLORS["gray"],
        arrowprops={"arrowstyle": "-", "color": COLORS["gray"], "linewidth": 0.45},
    )
    ax.set_xlim(1.0e-2, 1.0e1)
    ax.set_xlabel(r"Illustrative FFT weight, $w_{\rm FFT}$")
    ax.set_ylabel("Normalized operation-count proxy")
    ax.set_title("Operation-count illustration", pad=4)
    ax.legend(loc="upper left", fontsize=5.4, labelspacing=0.3)
    ax.grid(which="major", color="#E8EBEE", linewidth=0.55, zorder=0)
    ax.text(
        0.98, 0.04,
        "Not a processor-count or universal crossover",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=5.45, color=COLORS["gray"],
    )
    panel_label(ax, "b", x=-0.12)

    save_figure(
        fig,
        "figS_window_upsampling_diagnostics",
        "Window-upsampling accuracy sweep and operation-count diagnostic",
    )

    si_fields = [
        "record_type", "window", "actual_mesh", "actual_sigma_up", "order",
        "relative_error_mean", "relative_error_std", "particle_grid_coefficient",
        "fft_coefficient", "fft_weight_crossover", "scope",
    ]
    si_rows: list[dict[str, object]] = []
    for row in all_accuracy_rows:
        mesh = parse_grid(row["actual_grid"])[0]
        si_rows.append(
            {
                "record_type": "accuracy_sweep",
                "window": row["window"],
                "actual_mesh": mesh,
                "actual_sigma_up": row["actual_sigma_up"],
                "order": row["order"],
                "relative_error_mean": row["rms_relative_force_error_mean"],
                "relative_error_std": row["rms_relative_force_error_std"],
                "particle_grid_coefficient": "",
                "fft_coefficient": "",
                "fft_weight_crossover": "",
                "scope": "M=16 and M=20 are below-band diagnostics; M>=24 is resolved",
            }
        )
    for row in gaussian_summary:
        mesh = parse_grid(row["actual_grid"])[0]
        si_rows.append(
            {
                "record_type": "conventional_pppm_accuracy_sweep",
                "window": "Gaussian-split/B-spline PPPM",
                "actual_mesh": mesh,
                "actual_sigma_up": "",
                "order": row["order"],
                "relative_error_mean": row["rms_relative_force_error_mean"],
                "relative_error_std": row["rms_relative_force_error_std"],
                "particle_grid_coefficient": "",
                "fft_coefficient": "",
                "fft_weight_crossover": "",
                "scope": "native PPPM control; Gaussian split selected internally from the requested 1e-4 accuracy",
            }
        )
    for window, pg, fft in (("bspline", b_pg, b_fft), ("pswf", p_pg, p_fft)):
        si_rows.append(
            {
                "record_type": "operation_count_proxy",
                "window": window,
                "actual_mesh": 36 if window == "bspline" else 24,
                "actual_sigma_up": "",
                "order": 4 if window == "bspline" else 5,
                "relative_error_mean": "",
                "relative_error_std": "",
                "particle_grid_coefficient": pg,
                "fft_coefficient": fft,
                "fft_weight_crossover": crossover,
                "scope": "illustrative operation counts only; no processor mapping",
            }
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "figS_window_upsampling_diagnostics_plot_source.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=si_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(si_rows)

    return crossover


def figure7() -> float:
    """Plot the large-SPC/E-water grid--window trade-off for Figure 6.

    The two ESP branches share a target-matched PSWF splitting kernel at each
    target and differ only in their spreading window.  Top and bottom rows
    use the same parameter-selection and four-frame validation protocol, but
    are independently measured with i*k and AD differentiation, respectively.
    Gaussian-split PPPM remains a separately labelled conventional reference.
    """

    summary_paths = {
        "ik": LARGE_WATER_WINDOW / "validation_summary.csv",
        "ad": LARGE_WATER_WINDOW / "ad" / "validation_summary.csv",
    }
    summary_by_differentiation: dict[str, list[dict[str, str]]] = {}
    for differentiation, summary_path in summary_paths.items():
        summary = read_path_rows(summary_path)
        if not summary:
            raise ValueError(f"empty Figure 6 {differentiation} source: {summary_path}")
        for row in summary:
            reported = row.get("differentiation", "")
            if reported and reported != differentiation:
                raise ValueError(
                    f"Figure 6 {differentiation} source contains {reported} data"
                )
        summary_by_differentiation[differentiation] = summary

    expected = {
        "1e-4": {"target": 1.0e-4, "csplit": 12.024},
        "1e-5": {"target": 1.0e-5, "csplit": 14.471},
    }
    # The native-PPPM follow-up brackets the M=128 P=5 point with the same
    # four frozen water configurations: P=4 fails strongly, whereas P=5 is
    # only 7.2% above the nominal 1e-5 target.  The latter is retained as a
    # clearly documented 10%-near-target comparison in panel (d), rather
    # than being silently relabelled as an exact-target result.
    ad_p4_m128 = summarize_fixed_order_pppm_extension(
        LARGE_WATER_AD_P4_EXTENSION, order=4, actual_mesh=128
    )
    ad_p5_m128 = summarize_fixed_order_pppm_extension(
        LARGE_WATER_AD_P5_EXTENSION, order=5, actual_mesh=128
    )
    p4_max = value(ad_p4_m128, "validation_rms_relative_force_error_max")
    p5_max = value(ad_p5_m128, "validation_rms_relative_force_error_max")
    if (
        ad_p4_m128["all_validation_frames_feasible"] != "0"
        or ad_p5_m128["all_validation_frames_feasible"] != "0"
        or not p4_max > 1.0e-5
        or not 1.0e-5 < p5_max <= 1.10e-5
    ):
        raise ValueError(
            "AD M=128 PPPM extension must bracket the stated 10%-near-target "
            "P=5 comparison with a failed P=4 record"
        )
    summary_by_differentiation["ad"].extend((ad_p4_m128, ad_p5_m128))
    styles = {
        "pswf": {
            "color": COLORS["orange"],
            "marker": "s",
            "filled": True,
            "linestyle": "-",
            "label": "PSWF split + PSWF spread",
        },
        "bspline": {
            "color": COLORS["blue"],
            "marker": "o",
            "filled": False,
            "linestyle": "-",
            "label": "PSWF split + B-spline spread",
        },
        "pppm": {
            "color": COLORS["gray"],
            "marker": "D",
            "filled": True,
            "linestyle": "--",
            "label": "Gaussian split + B-spline spread (PPPM)",
        },
    }
    branch_names = {key: item["label"] for key, item in styles.items()}
    # The more strongly underresolved M=27 tight-target scan remains raw/SI
    # data.  Every panel instead begins with the near-band M=30 comparison.
    # M=48 is omitted from all panels; M=64 is also omitted in the 1e-4
    # panels, but retained in the two tight-target panels for their extended
    # grid comparison.  Every omitted measurement remains in source data.
    hidden_meshes_by_panel = {
        ("ik", "1e-4"): {48, 64},
        ("ik", "1e-5"): {48},
        ("ad", "1e-4"): {48, 64},
        ("ad", "1e-5"): {48},
    }
    display_meshes = {
        "1e-4": {27, 30, 32, 36, 40, 45, 54, 60, 72, 80},
        "1e-5": {30, 32, 36, 40, 45, 54, 60, 64, 72, 80, 96, 128},
    }

    def mesh_of(row: dict[str, str | float]) -> int:
        return int(str(row["actual_nx"]))

    def is_figure_accepted(row: dict[str, str | float], differentiation: str) -> bool:
        """Return the exact or explicitly stated near-target display status."""

        if int(float(row["all_validation_frames_feasible"])) == 1:
            return True
        return (
            differentiation == "ad"
            and row["target"] == "1e-5"
            and row["window"] == "pppm"
            and mesh_of(row) == 128
            and int(str(row["order"])) == 5
            and value(row, "validation_rms_relative_force_error_max")
            <= 1.10 * expected["1e-5"]["target"]
        )

    def figure_acceptance_label(
        row: dict[str, str | float], differentiation: str
    ) -> str:
        if int(float(row["all_validation_frames_feasible"])) == 1:
            return "exact target on all four validation configurations"
        if is_figure_accepted(row, differentiation):
            return "within stated 10% near-target margin on all four validation configurations"
        return "does not meet the figure acceptance criterion"

    def is_displayed_grid(
        row: dict[str, str | float], differentiation: str
    ) -> bool:
        mesh = mesh_of(row)
        return mesh not in hidden_meshes_by_panel[(differentiation, str(row["target"]))] and (
            row["window"] == "pppm" or mesh in display_meshes[str(row["target"])]
        )

    source_fields = [
        "differentiation",
        "target",
        "target_force_error",
        "window",
        "branch",
        "requested_mesh",
        "actual_nx",
        "actual_ny",
        "actual_nz",
        "actual_grid_volume",
        "order",
        "c_split",
        "c_spread",
        "gamma",
        "sigma_up",
        "selection_rms_relative_force_error",
        "validation_rms_relative_force_error_mean",
        "validation_rms_relative_force_error_std",
        "validation_rms_relative_force_error_max",
        "validation_rms_relative_force_error_min",
        "all_validation_frames_feasible",
        "n_validation_frames",
        "figure_acceptance",
        "source_record",
        "plot_status",
        "selection_protocol",
    ]
    source_rows: list[dict[str, object]] = []
    for differentiation, summary in summary_by_differentiation.items():
        for row in summary:
            target = row["target"]
            if target not in expected:
                raise ValueError(f"unexpected large-water target {target}")
            window = row["window"]
            if window not in styles:
                raise ValueError(f"unexpected large-water window {window}")
            grid = (
                int(row["actual_nx"]), int(row["actual_ny"]), int(row["actual_nz"])
            )
            if grid[0] != grid[1] or grid[1] != grid[2]:
                raise ValueError(f"Figure 6 expects a cubic actual FFT grid, got {grid}")
            if not math.isclose(value(row, "target_force_error"), expected[target]["target"]):
                raise ValueError(f"target field disagrees with {target}")
            if window != "pppm" and not math.isclose(
                value(row, "c_split"), expected[target]["csplit"], abs_tol=5.0e-4
            ):
                raise ValueError(f"{window} row does not use target-matched c_split")
            mesh = grid[0]
            accepted = is_figure_accepted(row, differentiation)
            if mesh in hidden_meshes_by_panel[(differentiation, target)]:
                plot_status = "not shown (omitted from figure for readability)"
            elif accepted and is_displayed_grid(row, differentiation):
                plot_status = "plotted"
            else:
                plot_status = "not shown (raw scan)"
            source_rows.append(
                {
                    **{field: row.get(field, "") for field in source_fields},
                    "differentiation": differentiation,
                    "branch": branch_names[window],
                    "figure_acceptance": figure_acceptance_label(row, differentiation),
                    "source_record": row.get(
                        "_source_record",
                        "grid--window selection and four-frame validation summary",
                    ),
                    "plot_status": plot_status,
                    "selection_protocol": (
                        (
                            "fixed-order native-PPPM extension on four frozen "
                            "SPC/E validation configurations"
                        )
                        if "_source_record" in row
                        else (
                            "one configuration selected c_spread for PSWF; four "
                            "later SPC/E configurations were evaluated with "
                            "c_spread frozen"
                        )
                    ),
                }
            )
    source_rows.sort(
        key=lambda row: (
            {"ik": 0, "ad": 1}[str(row["differentiation"])],
            str(row["target"]),
            str(row["window"]),
            int(str(row["actual_grid_volume"])),
        )
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "fig7_window_upsampling_plot_source.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=source_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(source_rows)

    # Keep each panel compact, while adding one visible P unit above the data
    # in panels (a), (b), and (d) as requested for the revised layout.
    y_layout: dict[tuple[str, str], tuple[tuple[float, float], range]] = {}
    for differentiation, summary in summary_by_differentiation.items():
        for target in expected:
            feasible_orders = [
                int(str(row["order"]))
                for row in summary
                if row["target"] == target
                and is_figure_accepted(row, differentiation)
                and is_displayed_grid(row, differentiation)
            ]
            if not feasible_orders:
                raise ValueError(f"no Figure 6 displayed rows for {differentiation}, {target}")
            lower, upper = min(feasible_orders), max(feasible_orders)
            extra_upper = 1 if (differentiation, target) in {
                ("ik", "1e-4"),
                ("ik", "1e-5"),
                ("ad", "1e-5"),
            } else 0
            y_layout[(differentiation, target)] = (
                (lower - 0.65, upper + 0.65 + extra_upper),
                range(lower, upper + 1 + extra_upper),
            )

    fig, axes = plt.subplots(
        2, 2, figsize=LARGE_WATER_WINDOW_FIGURE_SIZE, constrained_layout=False
    )
    panel_layout = (
        ("ik", "1e-4", r"10^{-4}", "a"),
        ("ik", "1e-5", r"10^{-5}", "b"),
        ("ad", "1e-4", r"10^{-4}", "c"),
        ("ad", "1e-5", r"10^{-5}", "d"),
    )
    for panel_index, (differentiation, target, target_label, label) in enumerate(panel_layout):
        row_index, column_index = divmod(panel_index, 2)
        ax = axes[row_index, column_index]
        rows_for_target = [
            row
            for row in summary_by_differentiation[differentiation]
            if row["target"] == target
        ]
        plotted_x: list[int] = []
        for window in ("pswf", "bspline", "pppm"):
            feasible_rows = sorted(
                (
                    row
                    for row in rows_for_target
                    if row["window"] == window
                    and is_displayed_grid(row, differentiation)
                    and is_figure_accepted(row, differentiation)
                ),
                key=lambda row: int(row["actual_grid_volume"]),
            )
            if not feasible_rows:
                continue
            style = styles[window]
            x = [int(row["actual_grid_volume"]) for row in feasible_rows]
            y = [int(row["order"]) for row in feasible_rows]
            plotted_x.extend(x)
            ax.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                markerfacecolor=style["color"] if style["filled"] else "white",
                markeredgecolor=style["color"],
                markeredgewidth=0.85,
                markersize=5.4,
                linestyle=style["linestyle"],
                linewidth=1.35,
                label=style["label"],
                zorder=4 if window != "pppm" else 3,
            )
        if not plotted_x:
            raise ValueError(f"no feasible Figure 6 rows for {differentiation}, {target}")
        ticks = sorted(set(plotted_x))
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{round(tick ** (1.0 / 3.0))}³" for tick in ticks], fontsize=6.0
        )
        # The neighbouring FFT-friendly grids 60^3 and 64^3 are both shown
        # in the tight-target panels.  Anchor their labels outward so their
        # close logarithmic positions remain legible without moving data.
        for tick, tick_label in zip(ticks, ax.get_xticklabels()):
            mesh = round(tick ** (1.0 / 3.0))
            if mesh == 60:
                tick_label.set_ha("right")
            elif mesh == 64:
                tick_label.set_ha("left")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="major", length=3.1, width=0.72, pad=3.1)
        ax.set_xlim(min(ticks) * 0.82, max(ticks) * 1.20)
        ax.set_ylim(*y_layout[(differentiation, target)][0])
        ax.set_yticks(list(y_layout[(differentiation, target)][1]))
        ax.set_xlabel(r"Actual FFT grid size, $M^3$")
        if column_index == 0:
            ax.set_ylabel(r"Minimum feasible spreading order, $P$")
        light_y_grid(ax)

        csplit = expected[target]["csplit"]
        annotation_top = 0.96
        annotation_line_spacing = 1.5 * 0.07
        annotation_y = (annotation_top, annotation_top - annotation_line_spacing)
        for y, annotation_label, value_text in (
            (annotation_y[0], r"$\epsilon$", "$" + target_label + "$"),
            (annotation_y[1], r"$c_{{\rm split}}$", "$" + f"{csplit:.3f}" + "$"),
        ):
            ax.text(
                0.82, y, annotation_label, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.25, color=COLORS["black"],
            )
            ax.text(
                0.84, y, r"$=$", transform=ax.transAxes, ha="center", va="top",
                fontsize=8.25, color=COLORS["black"],
            )
            ax.text(
                0.86, y, value_text, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.25, color=COLORS["black"],
            )
        ax.text(
            0.04, 0.96,
            r"$\mathrm{i}\mathbf{k}$" if differentiation == "ik" else "AD",
            transform=ax.transAxes, ha="left", va="top", fontsize=11.0,
            color=COLORS["black"], fontweight="bold",
        )
        panel_label(ax, label, x=-0.14, y=1.11, fontsize=12.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        fontsize=9.25,
        handlelength=1.8,
        columnspacing=1.15,
        handletextpad=0.40,
    )
    fig.subplots_adjust(
        left=0.082, right=0.993, bottom=0.105, top=0.865, hspace=0.47, wspace=0.29
    )
    save_figure(
        fig,
        "fig7_window_upsampling",
        "Large all-atom SPC/E-water grid-window trade-off under target-matched PSWF splitting",
    )
    # The prior operation-count crossover remains an archival SI illustration.
    # It is intentionally not used by the main-text water figure.
    return 0.0


def write_source_inventory() -> None:
    """Record the figure-to-source mapping in a compact tabular inventory."""
    rows = [
        {
            "figure": "2",
            "panel": "a-d",
            "source_csv": (
                "fig2_fourier_truncation_summary.csv; "
                "fig2_slab_fourier_truncation_summary.csv; "
                "fig2_water_fourier_prediction_summary.csv; "
                "fig2_water_fourier_reference_summary.csv"
            ),
            "role": (
                "random-charge validation, slab-like transfer diagnostic, and "
                "data-disjoint SPC/E structure-aware cube-tail validation"
            ),
            "uncertainty": (
                "random/slab: delete-one-configuration jackknife SEM of pooled RMS; "
                "water absolute panel: balanced delete-one-block holdout "
                "jackknife SEM shown for the validation reference, while the "
                "pilot measured-S_q sampling SEM is retained in the source data "
                "but not rendered; "
                "water ratio panel: pilot and holdout uncertainties combined "
                "in quadrature; rendered uncertainties use shaded "
                "plus/minus-one-SEM bands"
            ),
            "operator_or_grid_convention": (
                "all panels use direct finite PSWF Fourier sums and exact "
                "all-charge Ewald-minus-near infinite-sum identities; random/slab "
                "use M=33,L=48 and water uses M=21,L=30, giving the same "
                "cube-face Kmax; no particle mesh"
            ),
        },
        {
            "figure": "3",
            "panel": "a-f",
            "source_csv": (
                "fig3_mesh_validation_source.csv; "
                "lammps_ad_total_validation/fig3_lammps_ad_summary.csv"
            ),
            "role": (
                "top row: fixed-ik P, sigma_up, and c_spread sweeps; bottom "
                "row: total-AD pair/residual-self quadrature sweeps; panels "
                "a/d compare resolved 1e-4 and 1e-6 bandlimit settings, "
                "panels b/e compare P=5 and P=8; panel c compares the baseline "
                "(c_split,P)=(12.024,6) slice with a zero-free strict-band path "
                "at c_split=16.894 using P=(6,6,8,8,8,8,8), while panel f "
                "retains the fixed-P=10 AD diagnostic"
            ),
            "uncertainty": (
                "both rows: delete-one-configuration jackknife SEM of the "
                "pooled RMS over ten independent configurations, rendered as "
                "shaded plus/minus-one-SEM regions around actual-error curves"
            ),
            "operator_or_grid_convention": (
                "top: fixed-influence ik with the matched piecewise-polynomial "
                "window transform; the reduced estimator is a discrete axial "
                "multi-image sum through |ell_d|=12 on the same finite FFT "
                "grid (72 face images in total; mixed-axis edge/corner images "
                "omitted from the reduced estimate but retained by the "
                "source-table all-alias estimator, which is not plotted); the "
                "continuum axial proxy uses "
                "the same 12 axial layers and explicitly extrapolates the two "
                "sigma_up=0.980 positions; the displayed panel-c path is "
                "zero-free; bottom: analytical "
                "differentiation with the "
                "cell-moment pair estimate and residual-self budget combined in "
                "quadrature; when the implemented Fourier-polynomial denominator "
                "is zero, the production AD Green multiplier is set to zero and "
                "the estimator retains the complete missing-mode mismatch; all "
                "45 requested AD positions are evaluated and 44 are plotted; "
                "the baseline P=10 zeroed-mode case is retained source-only; all "
                "abscissae use actual grids; panels a/d use "
                "(M,c_split,c_spread)=(24,12.024,12.024) and "
                "(30,16.894,16.894); panel c uses (M,P,c_split)="
                "(24,6,12.024) and a strict M=30,c_split=16.894 path with "
                "P=(6,6,8,8,8,8,8), whereas panel f retains the fixed "
                "(30,10,16.894) AD diagnostic; the strict AD slice contains all seven c_spread "
                "points, including six for "
                "which one or more active reciprocal multipliers are zeroed; "
                "M=20,24,27,30,32,36,40,45,48 in panels b/e; M=20 has "
                "sigma_up=0.980 and is retained as a discrete boundary test, "
                "with explicitly flagged continuum extrapolation; it is excluded from "
                "resolved-range summaries"
            ),
        },
        {
            "figure": "4",
            "panel": "a-b",
            "source_csv": "fig4_sq_correction_source.csv",
            "role": "absolute and prediction/actual views of S_q=Q/V, rigid-SPC/E, and measured-S_q estimates versus validation errors",
            "uncertainty": "panel a renders only the validation block-RMS SEM around the actual-error curve; estimator sampling SEMs remain in the source data but are not displayed there; panel b uses shaded descriptive quadrature propagation because cross-block covariance is unavailable and includes alias-MC SEM",
            "operator_or_grid_convention": "fixed-influence ik; dimensionless S_q=<|rho|^2>/Q from frames 1--25 and nonoverlapping validation measurement on frames 26--51",
        },
        {
            "figure": "SI charge-correlation diagnostics",
            "panel": "a-b",
            "source_csv": "fig4_charge_spectrum_source.csv; fig4_k_resolved_variance_source.csv",
            "role": "volume-normalized physical-k spectrum and wave-number-resolved Eq. (90) estimator variance",
            "uncertainty": "panel a: configuration/five-block SEM; panel b: five-block trajectory and paired alias-importance-sampling SEMs",
            "operator_or_grid_convention": "panel a uses C_q=<|rho|^2>/V through 10 Angstrom^-1; panel b uses dimensionless S_q=<|rho|^2>/Q and bins fixed-ik source aliases at |k+G_l| and resolved gather terms at |k| for P=5",
        },
        {
            "figure": "5",
            "panel": "a-d",
            "source_csv": (
                "fig5_pppm_efficiency_plot_source.csv; "
                "fig5_fixed_ik_theory_grid_prediction.csv; "
                "fig5_fixed_ik_theory_grid_frozen.json; "
                "fig5_fixed_ik_theory_grid_source.csv; "
                "fig5_fixed_ik_theory_grid_manifest.json; "
                "build_fig5_fixed_ik_theory_grid.py; "
                "fig5_ad_stag_screen/pilot_stag_spectrum.npz; "
                "fig5_ad_stag_screen/pilot_stag_spectrum_frozen.json; "
                "fig5_ad_stag_screen/pilot_stag_spectrum_manifest.json; "
                "fig5_ad_stag_screen/baseline_prediction.csv; "
                "fig5_ad_stag_screen/baseline_prediction_by_frame.csv; "
                "fig5_ad_stag_screen/baseline_alias_shell.csv; "
                "fig5_ad_stag_screen/baseline_source.csv; "
                "fig5_ad_stag_screen/baseline_manifest.json; "
                "fig5_ad_stag_screen/stag_1e-4/"
                "prediction_before_validation.csv; "
                "fig5_ad_stag_screen/stag_1e-4/frozen_selection.json; "
                "fig5_ad_stag_screen/stag_1e-4/"
                "holdout_validation_summary.csv; "
                "fig5_ad_stag_screen/stag_1e-5/"
                "prediction_before_validation.csv; "
                "fig5_ad_stag_screen/stag_1e-5/frozen_selection.json; "
                "fig5_ad_stag_screen/stag_1e-5/"
                "holdout_validation_summary.csv; "
                "build_fig5_ad_coordinate_screen.py; "
                "fig5_pppm_ik_ad_fixed_g_scan/"
                "fig5_pppm_ik_ad_fixed_g_summary.csv; "
                "fig5_pppm_ik_ad_fixed_g_scan/fig5_pppm_ik_ad_fixed_g_manifest.json"
            ),
            "role": (
                "panels a/b: fixed-influence ik theoretical analysis and independent "
                "validation for P=5--9; panels c/d: conditional pair/self AD "
                "cell-moment theory frozen from 25 pilot frames, with independent validation. "
                "Frozen AD candidates at targets of 1e-4 and 1e-5, with their holdout "
                "values, remain in the source data but are not plotted separately. The "
                "PPPM baseline uses fixed "
                "G and P=5 under the corresponding differentiation convention"
            ),
            "uncertainty": (
                "top-row dashed curves are theoretical predictions from frames 1--25. "
                "Bottom-row dashed curves combine zero-mean S_tag fluctuations with "
                "coherent all-source pair and self-correction fields before cell squaring, then add "
                "closed Fourier theory in quadrature. Filled ESP and AD markers "
                "and open PPPM symbols use frames 26--51. All plotted validation symbols have "
                "five-block pooled-RMS SEMs and no connecting lines"
            ),
            "operator_or_grid_convention": (
                "all colored curves fix c_split=c_spread at 12.024 or 14.471. Panels "
                "a/b use fixed-influence ik. Panels c/d use measured S_tag and "
                "charge-class conditional pair spectra with exact AD cell-moment source "
                "weights; the coherent all-source pair and production self-correction "
                "fields are added before cell squaring, with the equivalent distinct-pair "
                "plus residual-self form checked explicitly. Frozen selections and their "
                "validation are source-data only. "
                "M=12 remains in the source table but is "
                "below the resolved-band domain and omitted from the main plot. "
                "PPPM uses P=5 and its separately fixed Gaussian Ewald parameter "
                "under the differentiation convention of the row."
            ),
        },
        {
            "figure": "6",
            "panel": "a",
            "source_csv": (
                "fig7_window_upsampling_plot_source.csv; "
                "numerical_examples/large_water_window_upsampling/"
                "selection_scan.csv; "
                "numerical_examples/large_water_window_upsampling/"
                "validation_by_frame.csv; "
                "numerical_examples/large_water_window_upsampling/"
                "validation_summary.csv"
            ),
            "role": "21,624-atom SPC/E-water i*k minimum-order grid--window trade-off at 1e-4, plus a separate conventional Gaussian-split/B-spline PPPM reference",
            "uncertainty": "one water configuration selects the PSWF c_spread; every selected point then satisfies the target on each of four later frozen-parameter configurations. The source table retains their mean, standard deviation, and maximum; these short-spaced configurations are not treated as independent samples.",
            "operator_or_grid_convention": "orange/blue branches have c_split=12.024, r_c=9 Angstrom, the same all-atom water configurations and Ewald reference, Hockney--Eastwood optimal influence, and ik differentiation; only spreading changes. Requested and actual grids agree, and all feasible colored points resolve the nominal band (minimum physical sigma_up=1.058). Gray diamonds are a separate native Gaussian-split/B-spline PPPM reference.",
        },
        {
            "figure": "6",
            "panel": "b",
            "source_csv": "fig7_window_upsampling_plot_source.csv; numerical_examples/large_water_window_upsampling/selection_scan.csv; numerical_examples/large_water_window_upsampling/validation_by_frame.csv; numerical_examples/large_water_window_upsampling/validation_summary.csv",
            "role": "21,624-atom SPC/E-water i*k minimum-order grid--window trade-off at 1e-5, plus a separate conventional Gaussian-split/B-spline PPPM reference",
            "uncertainty": "one water configuration selects the PSWF c_spread; every selected point then satisfies the target on each of four later frozen-parameter configurations. The source table retains their mean, standard deviation, and maximum; these short-spaced configurations are not treated as independent samples.",
            "operator_or_grid_convention": "orange/blue branches have target-matched c_split=14.471, r_c=9 Angstrom, the same all-atom water configurations and Ewald reference, Hockney--Eastwood optimal influence, and ik differentiation; only spreading changes. Requested and actual grids agree. The displayed near-band M=30 point has sigma_up=0.977: PSWF spreading reaches the target at P=8, whereas B-spline spreading fails through P=12. Fully resolved colored candidates begin at M=32 (sigma_up=1.042); the more strongly underresolved M=27 records are SI-only. Gray diamonds are a separate native Gaussian-split/B-spline PPPM reference; native PPPM first reaches the target at 64 cubed within P<=7.",
        },
        {
            "figure": "6",
            "panel": "c",
            "source_csv": "fig7_window_upsampling_plot_source.csv; numerical_examples/large_water_window_upsampling/ad/selection_scan.csv; numerical_examples/large_water_window_upsampling/ad/validation_by_frame.csv; numerical_examples/large_water_window_upsampling/ad/validation_summary.csv",
            "role": "21,624-atom SPC/E-water AD minimum-order grid--window trade-off at 1e-4, plus a separate conventional Gaussian-split/B-spline PPPM reference",
            "uncertainty": "one water configuration selects the PSWF c_spread; every selected point then satisfies the target on each of four later frozen-parameter configurations. The source table retains their mean, standard deviation, and maximum; these short-spaced configurations are not treated as independent samples.",
            "operator_or_grid_convention": "orange/blue branches have c_split=12.024, r_c=9 Angstrom, the same all-atom water configurations and Ewald reference, and the same AD Green function/self-force convention within the row; only spreading changes. Requested and actual grids agree. The PSWF-spread branch is feasible from M=27, while the B-spline branch is first feasible at M=30. Gray diamonds are a separate native Gaussian-split/B-spline PPPM reference under AD differentiation.",
        },
        {
            "figure": "6",
            "panel": "d",
            "source_csv": "fig7_window_upsampling_plot_source.csv; numerical_examples/large_water_window_upsampling/ad/selection_scan.csv; numerical_examples/large_water_window_upsampling/ad/validation_by_frame.csv; numerical_examples/large_water_window_upsampling/ad/validation_summary.csv; numerical_examples/large_water_window_upsampling/ad/pppm_p4_extension.csv; numerical_examples/large_water_window_upsampling/ad/pppm_p5_extension.csv",
            "role": "21,624-atom SPC/E-water AD minimum-order grid--window trade-off at 1e-5, plus a separate conventional Gaussian-split/B-spline PPPM reference",
            "uncertainty": "one water configuration selects the PSWF c_spread; every selected point then satisfies the target on each of four later frozen-parameter configurations. The source table retains their mean, standard deviation, and maximum; these short-spaced configurations are not treated as independent samples.",
            "operator_or_grid_convention": "orange/blue branches have c_split=14.471, r_c=9 Angstrom, the same all-atom water configurations and Ewald reference, and the same AD Green function/self-force convention within the row; only spreading changes. The near-band M=30 PSWF-spread point is feasible at P=8, whereas B-spline spreading remains above target through P=12 and begins at the resolved M=32 point. Gray diamonds are a separate native Gaussian-split/B-spline PPPM reference under AD differentiation. The displayed M=128,P=5 point has maximum four-frame error 1.072e-5, within the stated 10% near-target margin; the matched M=128,P=4 record fails strongly. The M=48 and 64 records are source-only display omissions.",
        },
        {
            "figure": "SI window--upsampling diagnostics",
            "panel": "a-b",
            "source_csv": "figS_window_upsampling_diagnostics_plot_source.csv; numerical_examples/window_upsampling_optimized_benchmark/window_upsampling_validation_summary.csv; numerical_examples/window_upsampling_m16_extension/window_upsampling_validation_summary_m16.csv; numerical_examples/gaussian_pppm_window_control/gaussian_pppm_validation_summary.csv; numerical_examples/window_upsampling_optimized_benchmark/window_upsampling_cost_envelope.csv",
            "role": "archived small-random-charge PSWF-split window/order diagnostic and w_FFT operation-count illustration; it is distinct from the all-atom-water main Figure 6",
            "uncertainty": "panel a: standard deviation across nine validation configurations; panel b is a deterministic operation-count proxy",
            "operator_or_grid_convention": "archived common-PSWF-split optimal-influence ik random-charge benchmark; M=16 and M=20 are diagnostic only. The operation-count proxy has no processor-count mapping and omits implementation, cache, and communication costs.",
        },
    ]
    fieldnames = list(rows[0])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "figure_source_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_plot_manifest(
    figure5_summary: dict[str, object],
    window_cost_crossover: float,
) -> None:
    manifest = {
        "purpose": "Reproducible plotting manifest for redesigned main-text Figures 2-6",
        "backend": "Python/matplotlib only",
        "journal_contract": {
            "width_inches": 7.1,
            "minimum_font_points": 7.0,
            "vector_exports": ["PDF", "SVG"],
            "raster_preview": "PNG at 300 dpi",
            "font": "Arial/Helvetica with DejaVu Sans fallback",
            "palette": "colorblind-safe Okabe-Ito family plus neutral black/gray",
        },
        "source_inventory": "figure_source_inventory.csv",
        "figure2": {
            "layout": "2 x 2",
            "panels": {
                "a": "random/slab absolute Fourier-truncation errors",
                "b": "random/slab homogeneous-estimate transfer ratios",
                "c": "SPC/E absolute Fourier-truncation errors",
                "d": "SPC/E structure-aware prediction ratios"
            },
            "x_axis": "common linear physical c_split axis in all four panels",
            "data_abscissae": [8.0189, 9.5392, 10.29, 12.024, 12.762, 14.471, 16.894],
            "data_spacing": "true nonuniform MathPSWF-table spacing; not categorical",
            "major_ticks": list(FIGURE2_MAJOR_XTICKS),
            "minor_ticks": list(FIGURE2_MINOR_XTICKS),
            "x_limits": list(FIGURE2_XLIM),
            "absolute_y_limits": list(FIGURE2_ABSOLUTE_YLIM),
            "panel_widths": "equal",
            "uncertainty_rendering": (
                "rendered uncertainties use shaded plus/minus-one-SEM bands; "
                "the panel-c measured-S_q sampling SEM is intentionally omitted"
            ),
        },
        "figure3": {
            "ik_source": "fig3_mesh_validation_source.csv",
            "ad_source": "lammps_ad_total_validation/fig3_lammps_ad_summary.csv",
            "ik_operator": "fixed-influence ik",
            "ik_reduced_estimator": (
                "discrete axial multi-image estimate on the actual FFT grid "
                "with max_abs_alias_index=12 (72 face images); mixed-axis "
                "edge/corner images omitted"
            ),
            "ik_displayed_estimators": (
                "discrete axial multi-image estimate (blue filled squares) "
                "and continuum reduction (green diamonds); the source-table "
                "all-alias estimate is not plotted"
            ),
            "ik_continuum_estimator": (
                "continuum axial proxy plotted with 12 axial layers; the two "
                "sigma_up=0.980 values are explicitly extrapolated, and the "
                "displayed panel-c parameter path is zero-free; it is not used "
                "for screening"
            ),
            "ik_window_transform_quadrature": (
                "64-point Gauss-Legendre rule on each polynomial segment"
            ),
            "ad_quantity": "total AD mesh error",
            "ad_estimator": "cell-moment pair estimate plus residual-self budget in quadrature",
            "ad_zero_deconvolution_convention": "when the Fourier-polynomial denominator is exactly zero on an active mode, the matched production and estimator paths set that reciprocal Green multiplier to zero and include the resulting full missing-mode mismatch; six displayed positions contain at least one such mode",
            "actual_grid_note": "panels a/d display P=4--9 for (M,c_split,c_spread)=(24,12.024,12.024) and resolved (30,16.894,16.894) order sweeps; the archived baseline fixed-ik and AD P=10 cases are source-only and are not plotted; panel c compares the baseline (M,P,c_split)=(24,6,12.024) slice with an M=30,c_split=16.894 path using P=(6,6,8,8,8,8,8), while panel f retains the fixed (30,10,16.894) AD diagnostic; panels b/e compare P=5 and P=8 on actual M=20,24,27,30,32,36,40,45,48; requested M=42 is realized and plotted at M=45; M=20 has sigma_up=0.980 and is retained as a discrete boundary test with explicitly flagged continuum extrapolation; the displayed upsampling axis begins at 0.9",
            "uncertainty": "delete-one-configuration jackknife SEM of pooled RMS over ten independent configurations",
            "uncertainty_rendering": "shaded plus/minus-one-SEM regions around actual-error curves; estimator curves have no uncertainty regions",
        },
        "figure4": {
            "layout": "1 x 2 absolute-error and estimator/actual panels",
            "panel_a_b_source": "fig4_sq_correction_source.csv",
            "series": ["actual error", "S_q=Q/V", "rigid SPC/E model", "measured S_q"],
            "estimator_weight": "S_q=<|rho|^2>/Q",
            "panel_b_y_limits": [0.8, 2.0],
            "operator": "fixed-influence ik with nonoverlapping force validation on frames 26--51",
            "panel_b_uncertainty": "descriptive prediction/actual quadrature propagation includes the appropriate numerator and validation-denominator SEMs plus alias-MC uncertainty; contiguous-block covariance is unavailable",
            "uncertainty_rendering": "panel a shows a shaded validation SEM region only for actual error; panel b shows shaded propagated uncertainty regions for all ratios",
            "supporting_information_diagnostics": {
                "sources": ["fig4_charge_spectrum_source.csv", "fig4_k_resolved_variance_source.csv"],
                "manifests": ["fig4_charge_spectrum_manifest.json", "fig4_k_resolved_variance_manifest.json"],
                "panels": ["volume-normalized physical spectrum", "P=5 k_rho-resolved estimator variance"],
                "output": "figS_charge_spectrum.pdf",
            },
        },
        "figure5": {
            "output_basename": "fig5_pppm_efficiency",
            "layout": figure5_summary["layout"],
            "plot_source": "fig5_pppm_efficiency_plot_source.csv",
            "theory_prediction_source": "fig5_fixed_ik_theory_grid_prediction.csv",
            "theory_freeze": "fig5_fixed_ik_theory_grid_frozen.json",
            "theory_runner": "build_fig5_fixed_ik_theory_grid.py",
            "theory_validation_join": "fig5_fixed_ik_theory_grid_source.csv",
            "ad_stag_screen_source": "fig5_ad_stag_screen/baseline_source.csv",
            "ad_stag_screen_manifest": "fig5_ad_stag_screen/baseline_manifest.json",
            "ad_stag_screen_runner": "build_fig5_ad_coordinate_screen.py",
            "ad_stag_screen_sources": {
                "1e-04": (
                    "fig5_ad_stag_screen/stag_1e-4/"
                    "prediction_before_validation.csv; "
                    "fig5_ad_stag_screen/stag_1e-4/frozen_selection.json; "
                    "fig5_ad_stag_screen/stag_1e-4/"
                    "holdout_validation_summary.csv"
                ),
                "1e-05": (
                    "fig5_ad_stag_screen/stag_1e-5/"
                    "prediction_before_validation.csv; "
                    "fig5_ad_stag_screen/stag_1e-5/frozen_selection.json; "
                    "fig5_ad_stag_screen/stag_1e-5/"
                    "holdout_validation_summary.csv"
                ),
            },
            "pppm_summary_source": (
                "fig5_pppm_ik_ad_fixed_g_scan/"
                "fig5_pppm_ik_ad_fixed_g_summary.csv"
            ),
            "panels": {
                "a": "fixed-influence ik, epsilon=1e-4, c_split=c_spread=12.024",
                "b": "fixed-influence ik, epsilon=1e-5, c_split=c_spread=14.471",
                "c": "conditional pair/self AD theory at c_split=c_spread=12.024",
                "d": "conditional pair/self AD theory at c_split=c_spread=14.471",
            },
            "orders": figure5_summary["orders"],
            "fixed_bandlimits_by_target": {
                "1e-04": {"c_split": 12.024, "c_spread": 12.024},
                "1e-05": {"c_split": 14.471, "c_spread": 14.471},
            },
            "esp_actual_meshes_by_target": {
                "1e-04": [12, 15, 16, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80],
                "1e-05": [12, 16, 18, 20, 24, 32, 36, 40, 48, 64, 80],
            },
            "displayed_resolved_meshes_by_panel": {
                "a": [15, 16, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80],
                "b": [16, 18, 20, 24, 32, 36, 40, 48, 64, 80],
                "c": [15, 16, 18, 20, 24, 27, 32, 36, 40, 48, 64, 80],
                "d": [16, 18, 20, 24, 32, 36, 40, 48, 64, 80],
            },
            "holdout_frames": [26, 51],
            "theory_frames": figure5_summary["theory_frames"],
            "theory_uses_reference_force_differences": figure5_summary[
                "theory_uses_reference_force_differences"
            ],
            "theory_curve_style": figure5_summary["theory_curve_style"],
            "ad_stag_screen_uses_reference_force_differences": figure5_summary[
                "ad_stag_screen_uses_reference_force_differences"
            ],
            "ad_stag_screen_curve_style": figure5_summary[
                "ad_stag_screen_curve_style"
            ],
            "ad_stag_screen_selection_scope": figure5_summary[
                "ad_stag_screen_selection_scope"
            ],
            "ad_stag_selection": figure5_summary["ad_stag_selection"],
            "ad_stag_selection_rows": figure5_summary[
                "ad_stag_selection_rows"
            ],
            "ad_stag_selection_uses_reference_force_differences": figure5_summary[
                "ad_stag_selection_uses_reference_force_differences"
            ],
            "ad_stag_selection_holdout_frames": figure5_summary[
                "ad_stag_selection_holdout_frames"
            ],
            "ad_stag_selection_curve_style": figure5_summary[
                "ad_stag_selection_curve_style"
            ],
            "validation_curve_style": figure5_summary["validation_curve_style"],
            "pppm_curve_style": figure5_summary["pppm_curve_style"],
            "esp_rows": figure5_summary["esp_rows"],
            "esp_theory_rows": figure5_summary["esp_theory_rows"],
            "ad_stag_screen_rows": figure5_summary["ad_stag_screen_rows"],
            "pppm_rows": figure5_summary["pppm_rows"],
            "pppm_rows_by_panel": figure5_summary["pppm_rows_by_panel"],
            "esp_displayed_rows": figure5_summary["esp_displayed_rows"],
            "ad_stag_screen_displayed_rows": figure5_summary[
                "ad_stag_screen_displayed_rows"
            ],
            "pppm_displayed_rows": figure5_summary["pppm_displayed_rows"],
            "first_theory_resolved_crossings": figure5_summary[
                "first_theory_resolved_crossings"
            ],
            "first_validation_resolved_crossings": figure5_summary[
                "first_validation_resolved_crossings"
            ],
            "first_ad_stag_screen_resolved_crossings": figure5_summary[
                "first_ad_stag_screen_resolved_crossings"
            ],
            "first_ad_validation_resolved_crossings": figure5_summary[
                "first_ad_validation_resolved_crossings"
            ],
            "below_band_points_omitted_from_main_plot": figure5_summary[
                "below_band_points_omitted_from_main_plot"
            ],
            "uncertainty": (
                "top-row predictions use frames 1--25. Colored bottom-row AD curves "
                "combine five-block pilot SEMs and alias-sampling SEMs in quadrature. All "
                "plotted holdout bars use five balanced pooled-RMS blocks from frames "
                "26--51; frozen-selection holdout values remain in the source data only"
            ),
            "operator_scope": (
                "panels a/b use fixed-influence ik and Eqs. (55) and (90). Panels c/d "
                "use measured S_tag for zero-mean pair fluctuations and charge-class "
                "conditional spectra for a coherent all-source pair field that is added to "
                "the production self-correction field before cell squaring; the equivalent "
                "distinct-pair plus residual-self form is checked; the closed Fourier "
                "term is added in quadrature. Frozen selected candidates and associated "
                "holdout values are source-data only. PPPM "
                "uses P=5 and a separately frozen Gaussian Ewald parameter under "
                "the differentiation convention of the row."
            ),
            "selection_scope": (
                "The frozen top-row theoretical screen accepts resolved-band candidates "
                "whose prediction is at or below the target. The lower AD rows use a "
                "short SPC/E-water configuration-conditioned estimate, not a universal molecular "
                "acceptance theorem. The M=12 diagnostics are source-only. "
                "First crossings are neither timing optima nor global order "
                "selections."
            ),
            "m12_upsampling_ratio": {
                "1e-04": 0.9405966028711956,
                "1e-05": 0.7815447137670689,
            },
            "ad_panels_included": figure5_summary["ad_panels_included"],
        },
        "archived_small_random_charge_window_diagnostic": {
            "legacy_output_basename": "fig7_window_upsampling",
            "layout": "1 x 2",
            "plot_source": "fig7_window_upsampling_plot_source.csv",
            "accuracy_source": "numerical_examples/window_upsampling_optimized_benchmark/window_upsampling_validation_summary.csv",
            "m16_extension_source": "numerical_examples/window_upsampling_m16_extension/window_upsampling_validation_summary_m16.csv",
            "m16_extension_manifest": "numerical_examples/window_upsampling_m16_extension/manifest.json",
            "gaussian_pppm_control_source": "numerical_examples/gaussian_pppm_window_control/gaussian_pppm_validation_summary.csv",
            "gaussian_pppm_control_manifest": "numerical_examples/gaussian_pppm_window_control/manifest.json",
            "panel_a": "fixed-PSWF-split minimum measured feasible P versus actual M^3 at target 1e-4, including the near-band-edge M=20 observation and a separate conventional-PPPM baseline",
            "panel_b": "fixed-PSWF-split minimum measured feasible P versus actual M^3 at target 1e-5, with a separate conventional-PPPM baseline",
            "common_split": "PSWF splitting with the same r_c and c_split in both branches",
            "branch_difference": "spreading window only: PSWF or B-spline",
            "conventional_control": "native Gaussian-split/B-spline-spread PPPM, with the same configurations, reference, cutoff, ik differentiation, and measured target but a target-selected Gaussian split",
            "nominal_band_edge_mesh": FIGURE7_NOMINAL_BAND_EDGE_M,
            "below_band_main_meshes": [20],
            "resolved_main_meshes": list(FIGURE7_RESOLVED_MESHES),
            "targets_for_order_panels": [FIGURE7_TARGET, FIGURE7_TIGHT_TARGET],
            "resolved_band_based_actual_sigma_up": [
                1.1757457535889946,
                1.469682191986243,
                1.7636186303834918,
                1.9595762559816576,
            ],
            "minimum_feasible_orders": {
                "PSWF-split/PSWF-spread": [5, 5, 5, 4],
                "PSWF-split/B-spline-spread": [6, 5, 4, 4],
            },
            "conventional_pppm_minimum_feasible_orders": {
                "M16": "not reached through P=7",
                "M20": "not reached through P=7",
                "M24": "not reached through P=7",
                "M30": 6,
                "M36": 5,
                "M40": 4,
                "M48": 4,
            },
            "below_band_calibration": {
                "M16": {
                    "sigma_up": 0.783830502392663,
                    "PSWF-split/PSWF-spread": "target not reached for P<=8; P=8 error 1.066512739278812e-3",
                    "PSWF-split/B-spline-spread": "target not reached for P<=8; P=8 error 1.3353324586046409e-3",
                },
                "M20": {
                    "sigma_up": 0.9797881279908288,
                    "PSWF-split/PSWF-spread_minimum_P": 6,
                    "PSWF-split/B-spline-spread_minimum_P": 8,
                },
            },
            "claim_limit": "controlled random-charge optimal-influence ik benchmark; accuracy feasibility alone does not establish a universal or hardware-specific timing crossover",
            "supporting_information_diagnostic": {
                "basename": "figS_window_upsampling_diagnostics",
                "plot_source": "figS_window_upsampling_diagnostics_plot_source.csv",
                "accuracy_scope": "full P=3--8 sweeps on M=16,20,24,30,36,40; M=16 and M=20 have sigma_up=0.784 and 0.980 and are below-band diagnostics",
                "cost_envelope_source": "numerical_examples/window_upsampling_optimized_benchmark/window_upsampling_cost_envelope.csv",
                "fft_weight_crossover": window_cost_crossover,
                "proxy_limit": "simple operation-count illustration only; no processor-count mapping; omits implementation arithmetic, cache effects, and communication",
            },
        },
        "exports": {
            str(number): [
                f"{stem}.pdf",
                f"{stem}.svg",
                f"{stem}.png",
                f"{stem}.tiff",
            ]
            for number, stem in [
                (2, "fig2_fourier_validation"),
                (3, "fig3_mesh_validation"),
                (4, "fig4_sq_correction"),
                (5, "fig5_pppm_efficiency"),
                (6, "fig7_window_upsampling"),
            ]
        },
        "supporting_information_exports": [
            "figS_charge_spectrum.pdf",
            "figS_charge_spectrum.svg",
            "figS_charge_spectrum.png",
            "figS_charge_spectrum.tiff",
            "figS_window_upsampling_diagnostics.pdf",
            "figS_window_upsampling_diagnostics.svg",
            "figS_window_upsampling_diagnostics.png",
            "figS_window_upsampling_diagnostics.tiff",
        ],
    }
    # The archived small-random-charge calculation above remains explicit in
    # the manifest as an SI diagnostic.  The main Figure 6 provenance below
    # is solely the large all-atom SPC/E-water benchmark.
    manifest["figure6"] = {
        "legacy_output_basename": "fig7_window_upsampling",
        "layout": "2 x 2",
        "plot_source": "fig7_window_upsampling_plot_source.csv",
        "selection_source": "numerical_examples/large_water_window_upsampling/selection_scan.csv",
        "validation_by_frame_source": "numerical_examples/large_water_window_upsampling/validation_by_frame.csv",
        "validation_summary_source": "numerical_examples/large_water_window_upsampling/validation_summary.csv",
        "ad_selection_source": "numerical_examples/large_water_window_upsampling/ad/selection_scan.csv",
        "ad_validation_by_frame_source": "numerical_examples/large_water_window_upsampling/ad/validation_by_frame.csv",
        "ad_validation_summary_source": "numerical_examples/large_water_window_upsampling/ad/validation_summary.csv",
        "system": {
            "model": "all-atom SPC/E water",
            "molecules": 7208,
            "atoms": 21624,
            "cell": "60 Angstrom cubic",
            "preparation": "2x2x2 replication of the equilibrated 30 Angstrom water box followed by short 300 K NVT relaxation",
        },
        "common_ESP_convention": {
            "real_space_cutoff_Angstrom": 9.0,
            "reference": "high-accuracy Ewald",
            "influence": {
                "ik": "Hockney-Eastwood optimal influence",
                "ad": "operator-matched AD Green function with self-force correction",
            },
            "differentiation": ["ik", "ad"],
            "branch_difference": "PSWF spreading versus cardinal B-spline spreading only",
            "selection": "one configuration chooses c_spread for each PSWF candidate; c_spread is then frozen",
            "validation": "each selected point must meet its force-error target on each of four later short-spaced configurations; the configurations are not treated as statistically independent",
        },
        "panels": {
            "a": {
                "differentiation": "ik",
                "target": 1.0e-4,
                "c_split": 12.024,
                "actual_cubic_grids": [27, 30, 32, 36, 40, 45, 48, 54, 60, 64, 72, 80],
                "omitted_from_display_for_readability": [48, 64],
                "minimum_feasible_physical_sigma_up": 1.058171178230095,
                "minimum_feasible_order": {
                    "PSWF-split/PSWF-spread": [6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                    "PSWF-split/B-spline-spread": [8, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4],
                    "Gaussian split + B-spline spread (PPPM)": {
                        "45": 7, "48": 6, "54": 6, "60": 5, "64": 5,
                        "72": 5, "80": 4
                    },
                },
            },
            "b": {
                "differentiation": "ik",
                "target": 1.0e-5,
                "c_split": 14.471,
                "displayed_actual_cubic_grids": [30, 32, 36, 40, 45, 54, 60, 72, 80, 96],
                "omitted_from_display_for_readability": [48, 64],
                "near_band_displayed_diagnostic": {
                    "grid": 30,
                    "sigma_up": 0.9769308922088362,
                    "PSWF-split/PSWF-spread": {
                        "minimum_feasible_order": 8,
                        "maximum_validation_relative_error": 8.030602607086028e-6,
                    },
                    "PSWF-split/B-spline-spread": {
                        "largest_tested_order": 12,
                        "maximum_validation_relative_error": 1.411533007378793e-5,
                    },
                },
                "resolved_colored_candidates_begin_at": {
                    "grid": 32,
                    "sigma_up": 1.042059618356092,
                },
                "SI_only_more_underresolved_diagnostic": {
                    "grid": 27,
                    "sigma_up": 0.8792378029879526,
                    "PSWF-split/PSWF-spread": {
                        "largest_tested_order": 10,
                        "maximum_validation_relative_error": 2.1837066e-5,
                    },
                    "PSWF-split/B-spline-spread": {
                        "largest_tested_order": 12,
                        "maximum_validation_relative_error": 6.6833677e-5,
                    },
                },
                "minimum_feasible_order": {
                    "PSWF-split/PSWF-spread": [8, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 5],
                    "PSWF-split/B-spline-spread": [11, 9, 8, 7, 7, 6, 6, 6, 5, 5, 5],
                    "Gaussian split + B-spline spread (PPPM)": {
                        "64": 7, "72": 6, "80": 6, "96": 5
                    },
                },
            },
            "c": {
                "differentiation": "ad",
                "target": 1.0e-4,
                "c_split": 12.024,
                "actual_cubic_grids": [27, 30, 32, 36, 40, 45, 48, 54, 60, 64, 72, 80],
                "omitted_from_display_for_readability": [48, 64],
                "minimum_feasible_order": {
                    "PSWF-split/PSWF-spread": [6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                    "PSWF-split/B-spline-spread": {
                        "30": 7, "32": 7, "36": 6, "40": 6, "45": 5,
                        "48": 5, "54": 5, "60": 5, "64": 5, "72": 5, "80": 4
                    },
                    "Gaussian split + B-spline spread (PPPM)": {
                        "45": 7, "48": 7, "54": 6, "60": 6, "64": 6,
                        "72": 5, "80": 5
                    },
                },
                "unplotted_raw_diagnostic": {
                    "PSWF-split/B-spline-spread": {
                        "grid": 27, "largest_tested_order": 8,
                        "maximum_validation_relative_error": 1.0120784976064058e-4,
                    },
                },
            },
            "d": {
                "differentiation": "ad",
                "target": 1.0e-5,
                "c_split": 14.471,
                "displayed_actual_cubic_grids": [30, 32, 36, 40, 45, 54, 60, 72, 80, 96, 128],
                "omitted_from_display_for_readability": [48, 64],
                "minimum_feasible_order": {
                    "PSWF-split/PSWF-spread": [8, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                    "PSWF-split/B-spline-spread": {
                        "32": 12, "36": 10, "40": 8, "45": 7, "48": 7,
                        "54": 7, "60": 6, "64": 6, "72": 6, "80": 5, "96": 5,
                        "128": 5
                    },
                    "Gaussian split + B-spline spread (PPPM)": {
                        "72": 7, "80": 7, "96": 6, "128": 5
                    },
                },
                "near_band_displayed_diagnostic": {
                    "grid": 30,
                    "sigma_up": 0.9769308922088362,
                    "PSWF-split/PSWF-spread": {
                        "minimum_feasible_order": 8,
                        "maximum_validation_relative_error": 8.500313610787785e-6,
                    },
                    "PSWF-split/B-spline-spread": {
                        "largest_tested_order": 12,
                        "maximum_validation_relative_error": 2.0599876327545754e-5,
                    },
                },
                "near_target_conventional_PPPM_display": {
                    "grid": 128,
                    "order": 5,
                    "maximum_validation_relative_error": 1.072171630751798e-5,
                    "acceptance_margin": 0.10,
                    "matched_lower_order": 4,
                    "matched_lower_order_maximum_validation_relative_error": 1.1082297271505583e-4,
                },
            },
        },
        "conventional_control": "native Gaussian-split/B-spline PPPM, shown only as a separate conventional reference and not as a third branch of the common-PSWF-split comparison",
        "ad_fixed_P5_pppm_extension": {
            "source": "numerical_examples/large_water_window_upsampling/ad/pppm_p5_extension.csv",
            "scope": "four-frame AD follow-up at fixed native PPPM P=5; M=128 is plotted under the stated 10% near-target margin after a matched P=4 extension fails strongly, while M=135 and 144 remain source-only",
            "actual_grid_maximum_relative_errors": {
                "128": 1.072171630751798e-5,
                "135": 8.96774554082871e-6,
                "144": 7.250039259158671e-6,
            },
        },
        "ad_fixed_P4_pppm_extension": {
            "source": "numerical_examples/large_water_window_upsampling/ad/pppm_p4_extension.csv",
            "scope": "four-frame AD native-PPPM P=4 tests at M=128 and 135; the M=128 failure brackets the displayed P=5 near-target comparison",
            "actual_grid_maximum_relative_errors": {
                "128": 1.1082297271505583e-4,
                "135": 9.741870776841563e-5,
            },
        },
        "claim_limit": "accuracy feasibility maps the grid--stencil exchange but does not establish a universal timing crossover or universal optimal spreading window",
        "supporting_information_diagnostic": {
            "basename": "figS_window_upsampling_diagnostics",
            "scope": "archived small-random-charge window/order and operation-count diagnostic, distinct from the large-water main Figure 6",
            "fft_weight_crossover": "SI-only illustrative operation-count quantity; not used for the large-water main-text result",
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "plot_redesigned_main_figures_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def verify_outputs(stems: Iterable[str] | None = None) -> None:
    """Verify vector outputs and the exact raster size and resolution."""
    from PIL import Image

    def verify_dpi(image: Image.Image, target: float, path: Path) -> None:
        dpi = image.info.get("dpi")
        if dpi is None or len(dpi) != 2:
            raise RuntimeError(f"{path.name} has no two-axis DPI metadata")
        if not all(math.isclose(value, target, abs_tol=0.02) for value in dpi):
            raise RuntimeError(f"{path.name} has DPI {dpi}, expected {target}")

    sizes = {
        "fig2_fourier_validation": FIGURE2_SIZE,
        "fig3_mesh_validation": (7.1, 5.0),
        "fig4_sq_correction": FIGURE4_SIZE,
        "figS_charge_spectrum": FIGURE_SI_SPECTRUM_SIZE,
        "fig5_pppm_efficiency": (7.1, 5.0),
        "fig7_window_upsampling": LARGE_WATER_WINDOW_FIGURE_SIZE,
        "figS_window_upsampling_diagnostics": FIGURE_SI_WINDOW_SIZE,
    }
    selected = sizes if stems is None else {stem: sizes[stem] for stem in stems}
    for stem, (width, height) in selected.items():
        for suffix in (".pdf", ".svg", ".png", ".tiff"):
            path = OUTPUT_ROOT / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"missing/empty output: {path}")
        png_path = OUTPUT_ROOT / f"{stem}.png"
        with Image.open(png_path) as image:
            expected = (round(width * 300), round(height * 300))
            if any(abs(observed - wanted) > 1 for observed, wanted in zip(image.size, expected)):
                raise RuntimeError(f"{stem}.png is {image.size}, expected {expected}")
            verify_dpi(image, 300.0, png_path)
        tiff_path = OUTPUT_ROOT / f"{stem}.tiff"
        with Image.open(tiff_path) as image:
            expected = (round(width * 600), round(height * 600))
            if any(abs(observed - wanted) > 1 for observed, wanted in zip(image.size, expected)):
                raise RuntimeError(f"{stem}.tiff is {image.size}, expected {expected}")
            verify_dpi(image, 600.0, tiff_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("all", "5"),
        default="all",
        help="render all redesigned main-text figures or only the Figure 5 panel",
    )
    args = parser.parse_args()
    if args.figure == "5":
        figure5_summary = figure5_theory_and_stag_ad()
        verify_outputs(("fig5_pppm_efficiency",))
        print(
            "Created Figure 5 as PDF/SVG, 300 dpi PNG, and 600 dpi TIFF: "
            f"{figure5_summary['esp_theory_rows']} fixed-IK rows, "
            f"{figure5_summary['ad_stag_screen_rows']} conditional pair/self AD rows, "
            f"and {figure5_summary['pppm_rows']} fixed-G PPPM rows."
        )
        return
    figure2()
    figure3()
    figure_si_charge_spectrum()
    figure4()
    figure5_summary = figure5_theory_and_stag_ad()
    window_cost_crossover = figure7()
    write_source_inventory()
    write_plot_manifest(figure5_summary, window_cost_crossover)
    verify_outputs()
    print("Created Figures 2--6 as PDF/SVG, 300 dpi PNG, and 600 dpi TIFF.")
    print(
        "Figure 5 fixed-influence IK and conditional pair/self AD theory/validation, "
        "with frozen AD selections: "
        f"{figure5_summary['esp_theory_rows']} ESP theory rows and "
        f"{figure5_summary['ad_stag_screen_rows']} conditional pair/self AD rows and "
        f"{figure5_summary['ad_stag_selection_rows']} frozen AD selections and "
        f"{figure5_summary['pppm_rows']} fixed-G PPPM validation rows."
    )
    print(
        "Figure 6 uses 21,624-atom all-atom SPC/E-water grid--stencil "
        "selections at 1e-4 and 1e-5, with target-matched PSWF splitting."
    )


if __name__ == "__main__":
    main()
