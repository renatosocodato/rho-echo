"""Shared publication-grade figure styling primitives for the active figure package."""

from __future__ import annotations

import shutil
import textwrap
import csv
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import matplotlib.collections as mcoll
import matplotlib.patches as mpatches
import matplotlib.text as mtext
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw

FONT_STACK = ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"]
ELIFE_FONT_STACK = FONT_STACK
FRAME_COLOR = "#4a4a4a"
TEXT_COLOR = "#222222"
LIGHT_TEXT_COLOR = "#666666"
PANEL_TITLE_COLOR = "#111111"
BOX_FACE = "#fbfbfa"
BOX_EDGE = "#d0d0ca"

PUBLICATION_PALETTE = {
    "navy": "#1a3a5a",
    "gold": "#f0a500",
    "red": "#E24B4A",
    "amber": "#BA7517",
    "plum": "#993556",
    "purple": "#534AB7",
    "rock_blue": "#378ADD",
    "green": "#639922",
    "warm_gray": "#888780",
    "stone": "#B4B2A9",
    "charcoal": "#2C2C2A",
    "grid": "#E8E8E6",
    "tick": "#555555",
    "restrained": "#E6F1FB",
    "reactive": "#EAF3DE",
    "dysregulated": "#FCEBEB",
    "panel_background": "#FFFFFF",
}
ELIFE_PALETTE = PUBLICATION_PALETTE

ELIFE_TEXT_SIZES = {
    "panel_letter": 8.6,
    "panel_title": 7.0,
    "axis_label": 6.8,
    "tick": 6.0,
    "legend": 6.0,
    "annotation": 6.0,
    "stat": 6.5,
    "inset_title": 6.0,
    "colorbar_title": 5.8,
    "colorbar_tick": 5.2,
    "fragility": 6.0,
    "table": 6.0,
}

ELIFE_LINE_WIDTHS = {
    "axis": 0.78,
    "tick": 0.72,
    "primary": 1.55,
    "mean": 1.55,
    "secondary": 1.05,
    "thin": 0.85,
    "hairline": 0.55,
    "grid": 0.3,
    "image_border": 0.5,
    "panel_frame": 0.75,
}

TEXT_SIZES = {
    "base": 6.8,
    "tick": 6.0,
    "axis_label": 6.8,
    "legend": 6.0,
    "annotation": 6.0,
    "stat_box": 6.7,
    "status_chip": 5.3,
    "provenance": 6.2,
    "panel_title": 8.0,
    "panel_letter": 10.2,
}

TYPOGRAPHY_PRESETS = {
    "main": {
        "panel_letter": TEXT_SIZES["panel_letter"],
        "panel_title": TEXT_SIZES["panel_title"],
        "axis_label": TEXT_SIZES["axis_label"],
        "tick": TEXT_SIZES["tick"],
        "legend": TEXT_SIZES["legend"],
        "annotation": TEXT_SIZES["annotation"],
        "direct_label": TEXT_SIZES["annotation"],
        "callout": TEXT_SIZES["annotation"],
        "stat_box": TEXT_SIZES["stat_box"],
        "stat_bracket": TEXT_SIZES["stat_box"],
        "status_chip": TEXT_SIZES["status_chip"],
        "provenance": TEXT_SIZES["provenance"],
        "source-limited": TEXT_SIZES["annotation"],
        "table": TEXT_SIZES["annotation"],
        "inset_title": 6.0,
        "colorbar_title": 6.2,
        "colorbar_tick": 5.2,
    },
    "supplementary": {
        "panel_letter": ELIFE_TEXT_SIZES["panel_letter"],
        "panel_title": ELIFE_TEXT_SIZES["panel_title"],
        "axis_label": ELIFE_TEXT_SIZES["axis_label"],
        "tick": ELIFE_TEXT_SIZES["tick"],
        "legend": ELIFE_TEXT_SIZES["legend"],
        "annotation": ELIFE_TEXT_SIZES["annotation"],
        "direct_label": ELIFE_TEXT_SIZES["annotation"],
        "callout": ELIFE_TEXT_SIZES["annotation"],
        "stat_box": ELIFE_TEXT_SIZES["stat"],
        "stat_bracket": ELIFE_TEXT_SIZES["stat"],
        "status_chip": ELIFE_TEXT_SIZES["fragility"],
        "provenance": ELIFE_TEXT_SIZES["annotation"],
        "source-limited": ELIFE_TEXT_SIZES["annotation"],
        "table": ELIFE_TEXT_SIZES["table"],
        "inset_title": ELIFE_TEXT_SIZES["inset_title"],
        "colorbar_title": ELIFE_TEXT_SIZES["colorbar_title"],
        "colorbar_tick": ELIFE_TEXT_SIZES["colorbar_tick"],
    },
}

LINE_WIDTHS = {
    "axes": 0.78,
    "ticks": 0.72,
    "primary": 1.55,
    "secondary": 1.05,
    "tertiary": 0.85,
    "frame": 0.75,
    "image_tile": 0.5,
    "bbox": 0.55,
}

MODALITY_TEXT_SIZES = {
    "main_plot": {"title": 8.0, "axis": 6.8, "tick": 6.0, "annotation": 6.0, "min": 4.0},
    "dense_supplement": {"title": 7.0, "axis": 6.8, "tick": 6.0, "annotation": 6.0, "min": 5.0},
    "image_montage": {"title": 6.8, "axis": 6.0, "tick": 5.5, "annotation": 6.0, "min": 5.0},
    "schematic": {"title": 8.0, "axis": 6.8, "tick": 6.0, "annotation": 6.0, "min": 4.5},
    "table": {"title": 7.0, "axis": 6.0, "tick": 6.0, "annotation": 6.0, "min": 5.5},
    "inset": {"title": 6.0, "axis": 5.5, "tick": 5.0, "annotation": 5.5, "min": 4.0},
    "colorbar": {"title": 6.2, "axis": 5.8, "tick": 5.2, "annotation": 5.5, "min": 4.0},
}

ROLE_MIN_FONT_SIZES = {
    "panel_letter": 7.0,
    "panel_title": 5.6,
    "callout": 5.6,
    "stat_box": 5.6,
    "status_chip": 4.0,
    "provenance": 5.0,
    "source-limited": 6.0,
    "stat_bracket": 5.6,
}


def typography_preset_name(figure_id: str | None = None, qa_config: FigureQAConfig | None = None) -> str:
    """Choose the active typography ladder for a saved figure."""
    if figure_id and str(figure_id).lower().startswith("supplementary"):
        return "supplementary"
    if qa_config is not None and qa_config.min_font_size >= 5.0:
        return "supplementary"
    return "main"


def font_size(role: str, *, preset: str = "main") -> float:
    """Return a role-based font size from the active publication hierarchy."""
    ladder = TYPOGRAPHY_PRESETS.get(preset, TYPOGRAPHY_PRESETS["main"])
    return float(ladder.get(role, ladder["annotation"]))


def _font_role_for_text(text: mtext.Text) -> str:
    role = str(getattr(text, "_windowed_annotation_role", ""))
    if role:
        return role
    ax = text.axes
    if ax is not None:
        if text is ax.xaxis.label or text is ax.yaxis.label:
            return "axis_label"
        if text in set(ax.get_xticklabels() + ax.get_yticklabels()):
            return "tick"
    return "text"

QA_REPORT_FIELDS = [
    "pass_label",
    "figure_id",
    "panel_id",
    "severity",
    "element_type",
    "message",
    "suggested_fix",
]


@dataclass(frozen=True)
class FigureQAConfig:
    """Visual integrity policy for publication figures."""

    min_font_size: float = 4.0
    critical_min_font_size: float = 3.0
    allowed_font_families: tuple[str, ...] = ("Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif")
    max_linewidth: float = 2.35
    required_panel_letters: bool = True
    check_collisions: bool = True
    check_empty_axes: bool = True
    expected_dpi: int = 600
    critical_on_missing_panel_letter: bool = True
    qa_pass_label: str = ""
    title_min_font_size: float = 5.6
    max_title_lines: int = 1
    edge_margin_px: float = 2.0
    panel_crop_dir: str | None = None
    check_scale_bars: bool = True
    check_axis_aspect: bool = True
    safe_zone_px: float = 18.0
    title_safe_zone_px: float = 24.0
    crop_panel_axes_only: bool = True
    visual_audit_report_path: str | None = None


@dataclass(frozen=True)
class FigureQAIssue:
    """Structured figure-integrity finding emitted during rendering."""

    figure_id: str
    panel_id: str
    severity: str
    element_type: str
    message: str
    suggested_fix: str = ""
    pass_label: str = ""

    def as_row(self) -> dict[str, str]:
        return {
            "pass_label": self.pass_label,
            "figure_id": self.figure_id,
            "panel_id": self.panel_id,
            "severity": self.severity,
            "element_type": self.element_type,
            "message": self.message,
            "suggested_fix": self.suggested_fix,
        }


@dataclass(frozen=True)
class AnnotationSpec:
    """Deterministic biological-feature annotation placement request."""

    text: str
    target_xy: tuple[float, float]
    preferred_anchor: str = "upper right"
    priority: int = 0
    role: str = "callout"
    color: str = TEXT_COLOR
    arrow: bool = False
    fontsize: float | None = None
    avoid_regions: tuple[Bbox, ...] = field(default_factory=tuple)
    edge_margin: float = 0.04
    allow_gutter: bool = False


@dataclass(frozen=True)
class DataFrameStatus:
    """Small typed result for source-limited dataframe validation."""

    ok: bool
    empty: bool
    missing_columns: tuple[str, ...]
    context: str
    message: str
    dataframe: Any = None


@dataclass(frozen=True)
class AggressiveIMDConfig:
    """Display-only IMD contrast policy for ROCK/Src representative images."""

    ratio_vmin: float = 0.85
    ratio_vmax: float = 1.15
    cmap_name: str = "jet"
    donor_percentiles: tuple[float, float] = (0.15, 99.85)
    brightness_gamma: float = 0.62
    brightness_floor: float = 0.18
    edge_dilate_px: int = 2
    signal_percentile: float = 0.2
    signal_support_dilate_px: int = 2
    clahe_enabled: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: tuple[int, int] = (50, 50)
    clahe_luminance_blend: float = 0.82
    clahe_background_floor: float = 0.015


S4_AGGRESSIVE_IMD_CONFIG = AggressiveIMDConfig()


MAIN_QA_CONFIG = FigureQAConfig(min_font_size=4.0, critical_min_font_size=3.0, max_linewidth=2.35)
SUPPLEMENTARY_QA_CONFIG = FigureQAConfig(min_font_size=5.0, critical_min_font_size=3.0, max_linewidth=2.35)
_FIGURE_QA_ISSUES: list[FigureQAIssue] = []


def reset_figure_qa_issues() -> None:
    _FIGURE_QA_ISSUES.clear()


def get_figure_qa_issues() -> list[FigureQAIssue]:
    return list(_FIGURE_QA_ISSUES)


def qa_config_for_pass(config: FigureQAConfig, pass_label: str, panel_crop_dir: str | Path | None = None) -> FigureQAConfig:
    return replace(config, qa_pass_label=pass_label, panel_crop_dir=str(panel_crop_dir) if panel_crop_dir else None)


def mark_panel_axis(ax, panel_id: str | None = None):
    """Register a true figure panel for validation and crop/contact-sheet output."""
    if panel_id is not None:
        ax._windowed_panel_id = panel_id
    ax._windowed_panel_axis = True
    ax._windowed_non_panel_axis = False
    return ax


def mark_non_panel_axis(ax):
    """Register helper axes such as colorbars/insets so they do not masquerade as panels."""
    ax._windowed_non_panel_axis = True
    ax._windowed_panel_axis = False
    return ax


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": TEXT_SIZES["base"],
            "font.weight": "normal",
            "axes.labelsize": TEXT_SIZES["axis_label"],
            "axes.labelweight": "normal",
            "axes.titlesize": TEXT_SIZES["panel_title"],
            "axes.titleweight": "normal",
            "axes.linewidth": LINE_WIDTHS["axes"],
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": FRAME_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.major.width": LINE_WIDTHS["ticks"],
            "ytick.major.width": LINE_WIDTHS["ticks"],
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Helvetica",
            "mathtext.it": "Helvetica:italic",
            "mathtext.bf": "Helvetica",
        }
    )


def apply_elife_supplementary_style() -> None:
    """Apply the supplementary-density preset while preserving the main V44 visual language."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ELIFE_FONT_STACK,
            "font.size": ELIFE_TEXT_SIZES["annotation"],
            "font.weight": "normal",
            "axes.labelsize": ELIFE_TEXT_SIZES["axis_label"],
            "axes.labelweight": "normal",
            "axes.titlesize": ELIFE_TEXT_SIZES["panel_title"],
            "axes.titleweight": "normal",
            "axes.linewidth": ELIFE_LINE_WIDTHS["axis"],
            "axes.labelcolor": ELIFE_PALETTE["charcoal"],
            "axes.edgecolor": ELIFE_PALETTE["charcoal"],
            "xtick.color": ELIFE_PALETTE["tick"],
            "ytick.color": ELIFE_PALETTE["tick"],
            "xtick.major.width": ELIFE_LINE_WIDTHS["tick"],
            "ytick.major.width": ELIFE_LINE_WIDTHS["tick"],
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": ELIFE_TEXT_SIZES["tick"],
            "figure.facecolor": "white",
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Helvetica",
            "mathtext.it": "Helvetica:italic",
            "mathtext.bf": "Helvetica",
        }
    )


def _box_style(face=BOX_FACE, edge=BOX_EDGE, alpha=0.94, linewidth=LINE_WIDTHS["bbox"], pad=0.16):
    return {
        "boxstyle": f"round,pad={pad}",
        "fc": face,
        "ec": edge,
        "lw": linewidth,
        "alpha": alpha,
    }


def _single_line(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def fit_single_line_text(
    fig,
    artist: mtext.Text,
    *,
    max_width_px: float,
    min_font_size: float = 5.6,
    ellipsis: str = "…",
) -> mtext.Text:
    """Keep a text artist on one line by scaling, then truncating with an ellipsis."""
    original = _single_line(artist.get_text())
    artist.set_text(original)
    artist.set_linespacing(1.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    while artist.get_fontsize() > min_font_size:
        bbox = artist.get_window_extent(renderer)
        if bbox.width <= max_width_px:
            return artist
        artist.set_fontsize(max(artist.get_fontsize() - 0.25, min_font_size))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    if artist.get_window_extent(renderer).width <= max_width_px:
        return artist
    base = original
    while len(base) > 4:
        base = base[:-1].rstrip()
        artist.set_text(f"{base}{ellipsis}")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if artist.get_window_extent(renderer).width <= max_width_px:
            return artist
    artist.set_text(original[:3] + ellipsis if len(original) > 3 else original)
    return artist


def place_panel_title(ax, artist: mtext.Text, *, min_font_size: float = 5.6) -> mtext.Text:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)
    x_axes, _ = artist.get_position()
    max_width_px = max(ax_bbox.width * max(0.20, 1.0 - x_axes - 0.03), 24.0)
    artist._windowed_annotation_role = "panel_title"
    artist._windowed_single_line_title = True
    return fit_single_line_text(fig, artist, max_width_px=max_width_px, min_font_size=min_font_size)


def add_panel_letter(ax, letter: str, x: float = -0.055, y: float = 1.082):
    text_kwargs = {
        "fontsize": TEXT_SIZES["panel_letter"],
        "fontweight": "normal",
        "fontname": "Helvetica",
        "color": PANEL_TITLE_COLOR,
        "va": "top",
        "ha": "left",
    }
    if getattr(ax, "name", "") == "3d":
        text = ax.text2D(x, y, letter, transform=ax.transAxes, **text_kwargs)
        text._windowed_annotation_role = "panel_letter"
        return text
    text = ax.text(x, y, letter, transform=ax.transAxes, clip_on=False, **text_kwargs)
    text._windowed_annotation_role = "panel_letter"
    return text


def add_panel_title(ax, title: str, x: float = 0.11, y: float = 1.082, width: int = 24):
    clean_title = _single_line(title)
    text_kwargs = {
        "fontsize": TEXT_SIZES["panel_title"],
        "fontweight": "normal",
        "fontname": "Helvetica",
        "color": PANEL_TITLE_COLOR,
        "va": "top",
        "ha": "left",
    }
    if getattr(ax, "name", "") == "3d":
        ax.set_title("")
        text = ax.text2D(x, y, clean_title, transform=ax.transAxes, **text_kwargs)
        return place_panel_title(ax, text, min_font_size=MAIN_QA_CONFIG.title_min_font_size)
    ax.set_title("")
    text = ax.text(x, y, clean_title, transform=ax.transAxes, clip_on=False, **text_kwargs)
    return place_panel_title(ax, text, min_font_size=MAIN_QA_CONFIG.title_min_font_size)


def style_plot_panel(ax) -> None:
    if getattr(ax, "name", "") == "3d":
        ax.tick_params(labelsize=TEXT_SIZES["tick"], pad=1.2, width=LINE_WIDTHS["ticks"], length=2.8)
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(ax, axis_name, None)
            pane = getattr(axis, "pane", None)
            if pane is not None:
                try:
                    pane.set_edgecolor(FRAME_COLOR)
                except Exception:
                    pass
        return
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LINE_WIDTHS["axes"])
        spine.set_edgecolor(FRAME_COLOR)
    ax.tick_params(labelsize=TEXT_SIZES["tick"], colors=TEXT_COLOR, width=LINE_WIDTHS["ticks"], length=2.8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.grid(False)


def style_elife_plot_panel(ax, *, grid: bool = False) -> None:
    """Supplementary plot chrome aligned to the main V44 boxed-axis language."""
    if getattr(ax, "name", "") == "3d":
        ax.tick_params(
            labelsize=ELIFE_TEXT_SIZES["tick"],
            pad=1.2,
            width=ELIFE_LINE_WIDTHS["tick"],
            length=2.6,
            direction="out",
            colors=ELIFE_PALETTE["tick"],
        )
        return
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(ELIFE_LINE_WIDTHS["axis"])
        spine.set_edgecolor(ELIFE_PALETTE["charcoal"])
    ax.tick_params(
        labelsize=ELIFE_TEXT_SIZES["tick"],
        colors=ELIFE_PALETTE["tick"],
        width=ELIFE_LINE_WIDTHS["tick"],
        length=2.6,
        direction="out",
        top=False,
        right=False,
    )
    ax.xaxis.label.set_size(ELIFE_TEXT_SIZES["axis_label"])
    ax.yaxis.label.set_size(ELIFE_TEXT_SIZES["axis_label"])
    ax.xaxis.label.set_color(ELIFE_PALETTE["charcoal"])
    ax.yaxis.label.set_color(ELIFE_PALETTE["charcoal"])
    if grid:
        ax.grid(True, axis="y", color=ELIFE_PALETTE["grid"], linewidth=ELIFE_LINE_WIDTHS["grid"])
    else:
        ax.grid(False)


def style_elife_image_panel(ax, *, border_color: str | None = None, linewidth: float | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if border_color is not None:
        frame = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            linewidth=linewidth if linewidth is not None else ELIFE_LINE_WIDTHS["image_border"],
            edgecolor=border_color,
            zorder=20,
            clip_on=False,
        )
        ax.add_patch(frame)


def add_elife_panel_letter(ax, letter: str, x: float = -0.045, y: float = 1.035):
    text = ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=ELIFE_TEXT_SIZES["panel_letter"],
        fontweight="normal",
        fontfamily=ELIFE_FONT_STACK,
        color=ELIFE_PALETTE["charcoal"],
        va="top",
        ha="left",
        clip_on=False,
    )
    text._windowed_annotation_role = "panel_letter"
    return text


def add_elife_panel_title(ax, title: str, x: float = 0.08, y: float = 1.035, width: int = 44):
    ax.set_title("")
    text = ax.text(
        x,
        y,
        _single_line(title),
        transform=ax.transAxes,
        fontsize=ELIFE_TEXT_SIZES["panel_title"],
        fontweight="normal",
        fontfamily=ELIFE_FONT_STACK,
        color=ELIFE_PALETTE["charcoal"],
        va="top",
        ha="left",
        clip_on=False,
        linespacing=1.03,
    )
    return place_panel_title(ax, text, min_font_size=SUPPLEMENTARY_QA_CONFIG.title_min_font_size)


def apply_elife_panel_header(ax, panel_letter: str, title: str, *, kind: str = "plot", width: int = 44, grid: bool = False) -> None:
    if not getattr(ax, "_windowed_panel_id", None):
        mark_panel_axis(ax, panel_letter)
    else:
        mark_panel_axis(ax, str(getattr(ax, "_windowed_panel_id")))
    if kind == "plot":
        style_elife_plot_panel(ax, grid=grid)
    else:
        style_elife_image_panel(ax)
        ax._windowed_container_axis = True
    add_elife_panel_letter(ax, panel_letter)
    add_elife_panel_title(ax, title, width=width)


def add_elife_stat_box(ax, text: str, xy=(0.97, 0.95), *, fontsize: float | None = None, color: str | None = None, ha: str = "right", va: str = "top"):
    artist = ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize or ELIFE_TEXT_SIZES["stat"],
        fontweight="normal",
        fontfamily=ELIFE_FONT_STACK,
        color=color or ELIFE_PALETTE["charcoal"],
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": ELIFE_PALETTE["grid"],
            "lw": ELIFE_LINE_WIDTHS["hairline"],
            "alpha": 0.92,
        },
    )
    artist._windowed_annotation_role = "stat_box"
    return artist


def add_elife_callout(ax, text: str, xy=(0.03, 0.05), *, fontsize: float | None = None, color: str | None = None, weight: str = "normal", ha: str = "left", va: str = "bottom"):
    artist = ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize or ELIFE_TEXT_SIZES["annotation"],
        fontweight=weight,
        fontfamily=ELIFE_FONT_STACK,
        color=color or ELIFE_PALETTE["charcoal"],
        bbox={
            "boxstyle": "round,pad=0.14",
            "fc": "white",
            "ec": ELIFE_PALETTE["grid"],
            "lw": ELIFE_LINE_WIDTHS["hairline"],
            "alpha": 0.90,
        },
    )
    artist._windowed_annotation_role = "callout"
    return artist


def sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    *,
    color: str | None = None,
    label_position: str = "above",
):
    bracket_color = color or ELIFE_PALETTE["charcoal"]
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=ELIFE_LINE_WIDTHS["thin"], c=bracket_color)
    if label_position == "below":
        text_y = y + h * 0.68
        text_va = "top"
    else:
        text_y = y + h * 1.1
        text_va = "bottom"
    artist = ax.text(
        (x1 + x2) / 2,
        text_y,
        text,
        ha="center",
        va=text_va,
        color=bracket_color,
        fontsize=ELIFE_TEXT_SIZES["stat"],
        fontweight="normal",
        fontfamily=ELIFE_FONT_STACK,
    )
    artist._windowed_annotation_role = "stat_bracket"
    return artist


def add_scale_bar(ax, *, label: str = "10 µm", x0: float = 0.72, y0: float = 0.08, length: float = 0.20, color: str = "white"):
    ax._windowed_scale_bar = True
    ax.plot([x0, x0 + length], [y0, y0], transform=ax.transAxes, color="black", lw=2.1, solid_capstyle="butt", zorder=30)
    ax.plot([x0, x0 + length], [y0, y0], transform=ax.transAxes, color=color, lw=1.25, solid_capstyle="butt", zorder=31)
    ax.text(
        x0 + length / 2,
        y0 - 0.035,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        fontfamily=ELIFE_FONT_STACK,
        color=color,
        bbox={"boxstyle": "round,pad=0.08", "fc": "black", "ec": "none", "alpha": 0.45},
        zorder=32,
    )


def add_safe_scale_bar(
    ax,
    *,
    label: str | None = "10 µm",
    x0: float = 0.76,
    y0: float = 0.20,
    length: float = 0.18,
    color: str = "white",
    label_position: str = "above",
):
    """Add a scale bar with the label kept inside an image-panel safe zone."""
    ax._windowed_scale_bar = True
    bar_lw_outer = 1.9
    bar_lw_inner = 1.05
    ax.plot([x0, x0 + length], [y0, y0], transform=ax.transAxes, color="black", lw=bar_lw_outer, solid_capstyle="butt", zorder=30)
    ax.plot([x0, x0 + length], [y0, y0], transform=ax.transAxes, color=color, lw=bar_lw_inner, solid_capstyle="butt", zorder=31)
    if not label:
        return None
    label_offset = 0.040 if label_position == "above" else -0.040
    label_va = "bottom" if label_position == "above" else "top"
    artist = ax.text(
        x0 + length / 2,
        y0 + label_offset,
        label,
        transform=ax.transAxes,
        ha="center",
        va=label_va,
        fontsize=6.0,
        fontfamily=ELIFE_FONT_STACK,
        color=color,
        bbox={"boxstyle": "round,pad=0.08", "fc": "black", "ec": "none", "alpha": 0.55},
        zorder=32,
    )
    artist._windowed_annotation_role = "direct_label"
    return artist


def add_calibrated_scale_bar(
    ax,
    *,
    image_shape: tuple[int, int] | tuple[int, int, int],
    micrometers: float = 10.0,
    pixels_per_micrometer: float,
    x1: float = 0.965,
    y0: float = 0.105,
    min_x0: float = 0.030,
    label: str | None = "10 µm",
    color: str = "white",
    label_position: str = "above",
):
    """Add a physically calibrated scale bar for an imshow image axis.

    The helper converts a requested real-world length into an axes-fraction
    length using the rendered image width, keeping calibration consistent
    across differently sized composites and single-tile insets.
    """
    width_px = float(max(int(image_shape[1]), 1))
    length = float(micrometers) * float(pixels_per_micrometer) / width_px
    length = float(np.clip(length, 0.025, 0.88))
    x0 = max(float(min_x0), min(float(x1) - length, 0.975 - length))
    return add_safe_scale_bar(
        ax,
        label=label,
        x0=x0,
        y0=y0,
        length=length,
        color=color,
        label_position=label_position,
    )


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Small dependency-free 8-connected binary dilation for display masks."""
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(int(iterations), 0)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                expanded |= padded[1 + dy:1 + dy + result.shape[0], 1 + dx:1 + dx + result.shape[1]]
        result = expanded
    return result


def expand_display_mask(
    mask: np.ndarray | None,
    donor: np.ndarray | None = None,
    acceptor: np.ndarray | None = None,
    *,
    dilation_px: int = 1,
    signal_percentile: float = 8.0,
    signal_support_dilate_px: int = 1,
) -> np.ndarray | None:
    """Conservatively expand image-display masks only where channel signal supports an edge."""
    base = None if mask is None else np.asarray(mask) > 0
    signal_support = None
    channels = [np.asarray(channel, dtype=float) for channel in (donor, acceptor) if channel is not None]
    if channels:
        signal = np.zeros_like(channels[0], dtype=float)
        for channel in channels:
            signal = signal + np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
        positive = signal[np.isfinite(signal) & (signal > 0)]
        if positive.size:
            threshold = float(np.nanpercentile(positive, signal_percentile))
            signal_support = signal >= max(threshold, 1e-8)
    if base is None:
        return signal_support
    if not np.any(base):
        return signal_support if signal_support is not None else base
    expanded = _binary_dilate(base, dilation_px)
    if signal_support is not None:
        return base | (expanded & _binary_dilate(signal_support, signal_support_dilate_px))
    return expanded


def _normalize_channel(values: np.ndarray, percentiles: tuple[float, float] = (2.0, 99.2)) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = np.nan_to_num(finite, nan=0.0, posinf=0.0, neginf=0.0)
    positive = finite[finite > 0]
    if positive.size:
        lo, hi = np.nanpercentile(positive, percentiles)
    else:
        lo, hi = np.nanpercentile(finite, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(finite, dtype=float)
    return np.clip((finite - lo) / (hi - lo), 0.0, 1.0)


def _effective_clahe_tile_grid(shape: tuple[int, int], requested: tuple[int, int]) -> tuple[int, int]:
    """Clamp CLAHE grid counts so small microscopy tiles do not over-fragment."""
    height, width = max(int(shape[0]), 1), max(int(shape[1]), 1)
    req_x, req_y = max(int(requested[0]), 1), max(int(requested[1]), 1)
    max_x = max(2, min(width // 4, 50))
    max_y = max(2, min(height // 4, 50))
    return (max(1, min(req_x, max_x)), max(1, min(req_y, max_y)))


def apply_clahe_intensity(
    channel: np.ndarray,
    config: AggressiveIMDConfig = S4_AGGRESSIVE_IMD_CONFIG,
    *,
    percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply display-only CLAHE to an intensity channel for IMD brightness."""
    normalized = _normalize_channel(channel, percentiles or config.donor_percentiles)
    if not config.clahe_enabled:
        return normalized
    try:
        import cv2  # type: ignore
    except Exception:
        return normalized
    uint = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    if uint.ndim != 2 or min(uint.shape) < 2:
        return normalized
    tile_grid = _effective_clahe_tile_grid(uint.shape, config.clahe_tile_grid_size)
    clahe = cv2.createCLAHE(clipLimit=float(config.clahe_clip_limit), tileGridSize=tile_grid)
    enhanced = clahe.apply(uint).astype(float) / 255.0
    dark = normalized <= float(config.clahe_background_floor)
    enhanced = np.where(dark, normalized, enhanced)
    return np.clip(enhanced, 0.0, 1.0)


def enhance_archival_imd_luminance(
    rgb: np.ndarray | None,
    config: AggressiveIMDConfig = S4_AGGRESSIVE_IMD_CONFIG,
) -> np.ndarray | None:
    """Sharpen archived RGB IMD displays by applying CLAHE to luminance only."""
    if rgb is None:
        return None
    arr = np.asarray(rgb, dtype=float)
    arr = arr / 255.0 if arr.max(initial=0) > 1.5 else np.clip(arr, 0.0, 1.0)
    if not config.clahe_enabled or arr.ndim != 3 or arr.shape[-1] != 3 or min(arr.shape[:2]) < 2:
        return arr
    try:
        import cv2  # type: ignore
    except Exception:
        return arr
    uint = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(uint, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    tile_grid = _effective_clahe_tile_grid(value.shape, config.clahe_tile_grid_size)
    clahe = cv2.createCLAHE(clipLimit=float(config.clahe_clip_limit), tileGridSize=tile_grid)
    enhanced_value = clahe.apply(value)
    blend = float(np.clip(config.clahe_luminance_blend, 0.0, 1.0))
    mixed_value = ((1.0 - blend) * value.astype(float) + blend * enhanced_value.astype(float)).astype(np.uint8)
    background = value <= int(round(config.clahe_background_floor * 255))
    mixed_value = np.where(background, value, mixed_value).astype(np.uint8)
    hsv[..., 2] = mixed_value
    enhanced_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(float) / 255.0
    return np.clip(enhanced_rgb, 0.0, 1.0)


def render_imd_fret_tile(
    donor: np.ndarray | None,
    fret: np.ndarray | None,
    mask: np.ndarray | None = None,
    *,
    ratio_vmin: float = 0.85,
    ratio_vmax: float = 1.15,
    cmap_name: str = "viridis",
    donor_percentiles: tuple[float, float] = (2.0, 99.2),
    edge_dilate_px: int = 1,
    fallback_rgb: np.ndarray | None = None,
    imd_config: AggressiveIMDConfig | None = None,
) -> np.ndarray:
    """Render an intensity-modulated FRET tile: ratio controls hue, donor controls brightness."""
    if imd_config is not None:
        ratio_vmin = imd_config.ratio_vmin
        ratio_vmax = imd_config.ratio_vmax
        cmap_name = imd_config.cmap_name
        donor_percentiles = imd_config.donor_percentiles
        edge_dilate_px = imd_config.edge_dilate_px
    if donor is None or fret is None:
        if fallback_rgb is None:
            return np.zeros((32, 32, 3), dtype=float)
        fallback = np.asarray(fallback_rgb, dtype=float)
        return fallback / 255.0 if fallback.max(initial=0) > 1.5 else np.clip(fallback, 0.0, 1.0)
    donor_arr = np.asarray(donor, dtype=float)
    fret_arr = np.asarray(fret, dtype=float)
    if donor_arr.ndim != 2 or fret_arr.ndim != 2:
        if fallback_rgb is not None:
            fallback = np.asarray(fallback_rgb, dtype=float)
            return fallback / 255.0 if fallback.max(initial=0) > 1.5 else np.clip(fallback, 0.0, 1.0)
        donor_arr = np.squeeze(donor_arr)
        fret_arr = np.squeeze(fret_arr)
    ratio = donor_arr / np.maximum(fret_arr, 1.0)
    ratio_norm = np.clip((ratio - ratio_vmin) / max(ratio_vmax - ratio_vmin, 1e-8), 0.0, 1.0)
    hue_rgb = plt.get_cmap(cmap_name)(ratio_norm)[..., :3]
    if imd_config is not None and imd_config.clahe_enabled:
        brightness = apply_clahe_intensity(donor_arr, imd_config, percentiles=donor_percentiles)
    else:
        brightness = _normalize_channel(donor_arr, donor_percentiles)
    brightness_gamma = imd_config.brightness_gamma if imd_config is not None else 1.0
    brightness_floor = imd_config.brightness_floor if imd_config is not None else 0.16
    brightness = np.power(np.clip(brightness, 0.0, 1.0), max(brightness_gamma, 1e-6))
    display_mask = expand_display_mask(
        mask,
        donor_arr,
        fret_arr,
        dilation_px=edge_dilate_px,
        signal_percentile=imd_config.signal_percentile if imd_config is not None else 8.0,
        signal_support_dilate_px=imd_config.signal_support_dilate_px if imd_config is not None else 1,
    )
    rgb = hue_rgb * (brightness_floor + (1.0 - brightness_floor) * brightness[..., None])
    if display_mask is not None:
        rgb = np.where(display_mask[..., None], rgb, 0.0)
    return np.clip(rgb, 0.0, 1.0)


def resolve_archival_imd_frames(root: Path, condition: str, sheet_name: str, replicate_id: str | None = None) -> list[Path]:
    """Return governed archived IMD frames for a ROCK/Src representative when available."""
    condition_dir = "shROCK2" if str(condition).lower().replace("_", "").startswith("shrock") else "pLKO"
    archive_root = Path(root) / "data" / "raw" / "fret" / "rock_src" / "lif_archival" / condition_dir
    sheet = str(sheet_name)
    replicate_stub = str(replicate_id or "")
    if "_" in replicate_stub:
        replicate_stub = replicate_stub.split("_", 1)[1]
    candidates = []
    for name in (sheet, replicate_stub):
        if name:
            candidates.append(archive_root / name / "imd_fret")
    if "-" in sheet:
        candidates.append(archive_root / sheet / "imd_fret")
    for candidate in candidates:
        frames = sorted(candidate.glob("imd_fret_t*.png"))
        if frames:
            return frames
    return []


def load_archival_imd_frame(frame_paths: Sequence[Path], frame_idx: int) -> np.ndarray | None:
    """Load an archived IMD frame as a float RGB tile without modifying source data."""
    if not frame_paths:
        return None
    idx = min(max(int(frame_idx), 0), len(frame_paths) - 1)
    image = Image.open(frame_paths[idx]).convert("RGB")
    return np.asarray(image, dtype=float) / 255.0


def add_image_index_label(ax, text: str, *, xy: tuple[float, float] = (0.045, 0.955)):
    """Place a compact image index in the top-left safe corner."""
    artist = ax.text(
        xy[0],
        xy[1],
        str(text),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        fontfamily=ELIFE_FONT_STACK,
        fontweight="normal",
        color="white",
        bbox={"boxstyle": "round,pad=0.18", "fc": "black", "ec": "none", "alpha": 0.68},
        zorder=40,
    )
    artist._windowed_annotation_role = "direct_label"
    return artist


def style_elife_legend(legend) -> None:
    if legend is None:
        return
    legend.set_frame_on(False)
    for text in legend.get_texts():
        text.set_fontfamily(ELIFE_FONT_STACK)
        text.set_fontsize(ELIFE_TEXT_SIZES["tick"])
        text.set_color(ELIFE_PALETTE["charcoal"])


def style_schematic_panel(ax) -> None:
    if getattr(ax, "name", "") == "3d":
        style_plot_panel(ax)
        return
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax._windowed_container_axis = True
    frame = Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        linewidth=LINE_WIDTHS["frame"],
        edgecolor=FRAME_COLOR,
        zorder=20,
        clip_on=False,
    )
    ax.add_patch(frame)


def style_image_tile(ax) -> None:
    frame = Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        linewidth=LINE_WIDTHS["image_tile"],
        edgecolor="#c0c0c0",
        zorder=20,
        clip_on=False,
    )
    ax.add_patch(frame)


def apply_panel_header(ax, panel_id: str, title: str, *, kind: str = "plot") -> None:
    mark_panel_axis(ax, panel_id)
    if kind == "plot":
        style_plot_panel(ax)
    else:
        style_schematic_panel(ax)
    add_panel_letter(ax, panel_id[-1])
    add_panel_title(ax, title)


def add_stat_box(ax, text: str, xy=(0.97, 0.95), fontsize: float = 6.7, ha: str = "right", va: str = "top"):
    artist = ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=TEXT_COLOR,
        fontname="Helvetica",
        fontweight="normal",
        bbox=_box_style(alpha=0.9),
    )
    artist._windowed_annotation_role = "stat_box"
    return artist


def add_callout(ax, text: str, xy=(0.03, 0.06), fontsize: float = 6.9, ha: str = "left", va: str = "bottom", color: str = TEXT_COLOR):
    artist = ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontname="Helvetica",
        fontweight="normal",
        color=color,
        bbox=_box_style(alpha=0.88),
    )
    artist._windowed_annotation_role = "callout"
    return artist


def add_provenance_box(ax, text: str, xy=(0.02, 0.02), fontsize: float = 6.2, ha: str = "left", va: str = "bottom"):
    artist = ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontname="Helvetica",
        fontweight="normal",
        color=LIGHT_TEXT_COLOR,
        bbox=_box_style(alpha=0.82, linewidth=0.7, pad=0.18),
    )
    artist._windowed_annotation_role = "provenance"
    return artist


def add_status_chip(ax, text: str, xy=(0.03, 0.04), fontsize: float = 5.3, ha: str = "left", va: str = "bottom", color: str = TEXT_COLOR):
    artist = ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontname="Helvetica",
        fontweight="normal",
        color=color,
        linespacing=1.08,
        bbox=_box_style(alpha=0.84, linewidth=0.7, pad=0.16),
    )
    artist._windowed_annotation_role = "status_chip"
    return artist


def add_footer_note(ax, text: str, y: float = -0.15, fontsize: float = 4.8):
    artist = ax.text(
        0.50,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        color=LIGHT_TEXT_COLOR,
        fontname="Helvetica",
        clip_on=False,
    )
    artist._windowed_annotation_role = "footer_note"
    return artist


def add_halo_label(ax, text: str, x: float, y: float, fontsize: float = 6.0, color: str = TEXT_COLOR, ha: str = "left", va: str = "center"):
    artist = ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        fontname="Helvetica",
        fontweight="normal",
        color=color,
        ha=ha,
        va=va,
        zorder=12,
    )
    artist._windowed_annotation_role = "direct_label"
    return artist


INTERIOR_SAFE_TEXT_ROLES = {
    "callout",
    "stat_box",
    "status_chip",
    "direct_label",
    "provenance",
    "source-limited",
    "stat_bracket",
}


def _axis_safe_bbox(ax, renderer, margin_px: float) -> Bbox:
    bbox = ax.get_window_extent(renderer)
    if bbox.width <= 2 * margin_px or bbox.height <= 2 * margin_px:
        return bbox
    return Bbox.from_extents(
        bbox.x0 + margin_px,
        bbox.y0 + margin_px,
        bbox.x1 - margin_px,
        bbox.y1 - margin_px,
    )


def clamp_artist_to_axes(artist: mtext.Text, *, margin_px: float = 18.0) -> mtext.Text:
    """Nudge an axes-fraction text artist inside the panel's interior safe zone."""
    ax = artist.axes
    if ax is None:
        return artist
    # Only mutate axes-fraction text. Data-coordinate labels should be moved by recipe logic.
    if artist.get_transform() != ax.transAxes:
        return artist
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    try:
        bbox = artist.get_window_extent(renderer)
        ax_bbox = ax.get_window_extent(renderer)
    except Exception:
        return artist
    safe = _axis_safe_bbox(ax, renderer, margin_px)
    if safe.contains(bbox.x0, bbox.y0) and safe.contains(bbox.x1, bbox.y1):
        return artist
    dx_px = 0.0
    dy_px = 0.0
    if bbox.x0 < safe.x0:
        dx_px += safe.x0 - bbox.x0
    if bbox.x1 > safe.x1:
        dx_px -= bbox.x1 - safe.x1
    if bbox.y0 < safe.y0:
        dy_px += safe.y0 - bbox.y0
    if bbox.y1 > safe.y1:
        dy_px -= bbox.y1 - safe.y1
    x, y = artist.get_position()
    artist.set_position((x + dx_px / max(ax_bbox.width, 1.0), y + dy_px / max(ax_bbox.height, 1.0)))
    return artist


def place_safe_text(
    ax,
    text: str,
    *,
    xy: tuple[float, float] = (0.05, 0.05),
    fontsize: float | None = None,
    role: str = "callout",
    color: str | None = None,
    weight: str = "normal",
    ha: str = "left",
    va: str = "bottom",
    bbox: bool = True,
    candidates: Sequence[tuple[float, float, str, str]] | None = None,
    margin_px: float = 18.0,
    zorder: float = 20,
):
    """Place text inside the panel safe zone using deterministic candidates."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidates = list(candidates or [(xy[0], xy[1], ha, va)])
    avoid = _annotation_avoid_bboxes(ax, renderer)
    scored: list[tuple[float, mtext.Text, Bbox]] = []
    for x, y, cand_ha, cand_va in candidates:
        artist = ax.text(
            x,
            y,
            _single_line(text) if role == "source-limited" else text,
            transform=ax.transAxes,
            ha=cand_ha,
            va=cand_va,
            fontsize=fontsize or TEXT_SIZES["annotation"],
            fontfamily=FONT_STACK,
            fontweight=weight,
            color=color or TEXT_COLOR,
            bbox=_box_style(alpha=0.88, linewidth=LINE_WIDTHS["bbox"], pad=0.14) if bbox else None,
            zorder=zorder,
        )
        artist._windowed_annotation_role = role
        artist = clamp_artist_to_axes(artist, margin_px=margin_px)
        fig.canvas.draw()
        try:
            text_bbox = artist.get_window_extent(fig.canvas.get_renderer()).expanded(1.02, 1.05)
        except Exception:
            return artist
        safe = _axis_safe_bbox(ax, fig.canvas.get_renderer(), margin_px)
        edge_penalty = 0.0
        edge_penalty += max(safe.x0 - text_bbox.x0, 0)
        edge_penalty += max(safe.y0 - text_bbox.y0, 0)
        edge_penalty += max(text_bbox.x1 - safe.x1, 0)
        edge_penalty += max(text_bbox.y1 - safe.y1, 0)
        overlap = sum(_bbox_overlap_area(text_bbox, region) for region in avoid)
        score = overlap * 8.0 + edge_penalty * 4.0
        if overlap <= 2.0 and edge_penalty <= 1.0:
            for _, other, _ in scored:
                other.remove()
            return artist
        scored.append((score, artist, text_bbox))
    scored.sort(key=lambda item: item[0])
    best = scored[0][1]
    for _, artist, _ in scored[1:]:
        artist.remove()
    return best


def resolve_text_collisions(ax, *, margin_px: float = 18.0) -> None:
    """Best-effort late nudge for safe-zone annotations that still collide."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    movable = [
        text for text in ax.texts
        if getattr(text, "_windowed_annotation_role", "") in INTERIOR_SAFE_TEXT_ROLES
        and text.get_transform() == ax.transAxes
    ]
    ax_bbox = ax.get_window_extent(renderer)
    for _ in range(3):
        moved = False
        boxes: list[tuple[mtext.Text, Bbox]] = []
        for text in movable:
            clamp_artist_to_axes(text, margin_px=margin_px)
            fig.canvas.draw()
            boxes.append((text, text.get_window_extent(fig.canvas.get_renderer()).expanded(1.02, 1.05)))
        for idx, (left_text, left_bbox) in enumerate(boxes):
            for right_text, right_bbox in boxes[idx + 1:]:
                overlap = _bbox_overlap_area(left_bbox, right_bbox)
                if overlap <= 8.0:
                    continue
                x, y = right_text.get_position()
                right_text.set_position((x, min(max(y - 18.0 / max(ax_bbox.height, 1.0), 0.04), 0.96)))
                moved = True
        if not moved:
            break


def render_empty_panel(ax, message: str, panel_id: str | None = None, severity: str = "source-limited"):
    """Render an intentional empty/source-limited state without leaving a misleading blank panel."""
    if panel_id is not None:
        mark_panel_axis(ax, panel_id)
    ax._windowed_empty_state = True
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LINE_WIDTHS["axes"])
        spine.set_edgecolor(BOX_EDGE)
    artist = ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=max(TEXT_SIZES["annotation"], 6.0),
        fontfamily=FONT_STACK,
        fontweight="normal",
        color=LIGHT_TEXT_COLOR,
        linespacing=1.15,
        bbox=_box_style(face="white", edge=BOX_EDGE, alpha=0.90, linewidth=LINE_WIDTHS["bbox"], pad=0.20),
    )
    artist._windowed_annotation_role = severity
    return artist


def safe_dataframe(df: Any, required_columns: Sequence[str], context: str) -> DataFrameStatus:
    """Validate dataframe-like inputs before figure recipes render them."""
    if df is None:
        return DataFrameStatus(False, True, tuple(required_columns), context, f"{context}: dataframe is unavailable", df)
    columns = set(getattr(df, "columns", []))
    missing = tuple(col for col in required_columns if col not in columns)
    empty = bool(getattr(df, "empty", False))
    if missing:
        return DataFrameStatus(False, empty, missing, context, f"{context}: missing columns {', '.join(missing)}", df)
    if empty:
        return DataFrameStatus(False, True, tuple(), context, f"{context}: dataframe is empty", df)
    return DataFrameStatus(True, False, tuple(), context, f"{context}: ok", df)


def _candidate_offsets(anchor: str) -> list[tuple[float, float, str, str]]:
    candidates = {
        "upper right": [(0.03, 0.03, "left", "bottom"), (-0.03, 0.03, "right", "bottom"), (0.03, -0.03, "left", "top"), (-0.03, -0.03, "right", "top")],
        "upper left": [(-0.03, 0.03, "right", "bottom"), (0.03, 0.03, "left", "bottom"), (-0.03, -0.03, "right", "top"), (0.03, -0.03, "left", "top")],
        "lower right": [(0.03, -0.03, "left", "top"), (-0.03, -0.03, "right", "top"), (0.03, 0.03, "left", "bottom"), (-0.03, 0.03, "right", "bottom")],
        "lower left": [(-0.03, -0.03, "right", "top"), (0.03, -0.03, "left", "top"), (-0.03, 0.03, "right", "bottom"), (0.03, 0.03, "left", "bottom")],
    }
    return candidates.get(anchor, candidates["upper right"])


def _annotation_avoid_bboxes(ax, renderer, extra_regions: Iterable[Bbox] = ()) -> list[Bbox]:
    bboxes: list[Bbox] = list(extra_regions)
    legend = ax.get_legend()
    if legend is not None:
        try:
            bboxes.append(legend.get_window_extent(renderer))
        except Exception:
            pass
    for text in ax.texts:
        if text.get_visible() and text.get_text().strip():
            try:
                bboxes.append(text.get_window_extent(renderer).expanded(1.04, 1.12))
            except Exception:
                pass
    return bboxes


def _bbox_collides(candidate: Bbox, avoid_regions: Sequence[Bbox], *, min_overlap: float = 2.0) -> bool:
    for region in avoid_regions:
        if candidate.overlaps(region):
            x0 = max(candidate.x0, region.x0)
            y0 = max(candidate.y0, region.y0)
            x1 = min(candidate.x1, region.x1)
            y1 = min(candidate.y1, region.y1)
            if (x1 - x0) * (y1 - y0) >= min_overlap:
                return True
    return False


def _bbox_overlap_area(left: Bbox, right: Bbox) -> float:
    if not left.overlaps(right):
        return 0.0
    x0 = max(left.x0, right.x0)
    y0 = max(left.y0, right.y0)
    x1 = min(left.x1, right.x1)
    y1 = min(left.y1, right.y1)
    return max(x1 - x0, 0) * max(y1 - y0, 0)


def _axis_edge_penalty(ax, renderer, bbox: Bbox, *, margin_px: float = 5.0) -> float:
    try:
        ax_bbox = ax.get_window_extent(renderer)
    except Exception:
        return 0.0
    penalty = 0.0
    penalty += max(ax_bbox.x0 + margin_px - bbox.x0, 0)
    penalty += max(ax_bbox.y0 + margin_px - bbox.y0, 0)
    penalty += max(bbox.x1 - (ax_bbox.x1 - margin_px), 0)
    penalty += max(bbox.y1 - (ax_bbox.y1 - margin_px), 0)
    return penalty


def place_annotation(ax, spec: AnnotationSpec, candidates: Sequence[tuple[float, float, str, str]] | None = None):
    """Place an annotation by scoring deterministic nearby positions before falling back quietly."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    avoid_regions = _annotation_avoid_bboxes(ax, renderer, spec.avoid_regions)
    candidate_offsets = list(candidates or _candidate_offsets(spec.preferred_anchor))
    fontsize = spec.fontsize or TEXT_SIZES["annotation"]
    scored_artists = []
    for dx, dy, ha, va in candidate_offsets:
        margin = spec.edge_margin
        x = min(max(spec.target_xy[0] + dx, margin), 1.0 - margin)
        y = min(max(spec.target_xy[1] + dy, margin), 1.0 - margin)
        if spec.arrow:
            artist = ax.annotate(
                spec.text,
                xy=spec.target_xy,
                xycoords=ax.transAxes,
                xytext=(x, y),
                textcoords=ax.transAxes,
                ha=ha,
                va=va,
                fontsize=fontsize,
                fontfamily=FONT_STACK,
                fontweight="normal",
                color=spec.color,
                arrowprops={"arrowstyle": "->", "lw": LINE_WIDTHS["tertiary"], "color": spec.color},
                bbox=_box_style(alpha=0.88, linewidth=LINE_WIDTHS["bbox"], pad=0.14),
            )
        else:
            artist = ax.text(
                x,
                y,
                spec.text,
                transform=ax.transAxes,
                ha=ha,
                va=va,
                fontsize=fontsize,
                fontfamily=FONT_STACK,
                fontweight="normal",
                color=spec.color,
                bbox=_box_style(alpha=0.88, linewidth=LINE_WIDTHS["bbox"], pad=0.14),
            )
        artist._windowed_annotation_role = spec.role
        fig.canvas.draw()
        try:
            bbox = artist.get_window_extent(fig.canvas.get_renderer()).expanded(1.02, 1.08)
        except Exception:
            return artist
        overlap = sum(_bbox_overlap_area(bbox, region) for region in avoid_regions)
        edge_penalty = 0 if spec.allow_gutter else _axis_edge_penalty(ax, renderer, bbox, margin_px=fig.dpi * spec.edge_margin / 2.0)
        distance_penalty = abs(dx) + abs(dy)
        score = overlap * 8.0 + edge_penalty * 2.0 + distance_penalty * 20.0
        if overlap <= 2.0 and edge_penalty <= 2.0:
            return artist
        scored_artists.append((score, artist))

    if scored_artists:
        scored_artists.sort(key=lambda item: item[0])
        best_artist = scored_artists[0][1]
        for _, artist in scored_artists[1:]:
            artist.remove()
        if scored_artists[0][0] < 100.0:
            return best_artist
        best_artist.remove()

    fallback = ax.text(
        min(max(spec.target_xy[0], 0.04), 0.96),
        min(max(spec.target_xy[1], 0.04), 0.96),
        spec.text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=max(fontsize - 0.5, 5.5),
        fontfamily=FONT_STACK,
        fontweight="normal",
        color=spec.color,
        bbox=_box_style(face="white", edge=BOX_EDGE, alpha=0.82, linewidth=LINE_WIDTHS["bbox"], pad=0.12),
    )
    fallback._windowed_annotation_role = spec.role
    issues = getattr(ax, "_windowed_qa_issues", [])
    issues.append(
        FigureQAIssue(
            figure_id="pending",
            panel_id=str(getattr(ax, "_windowed_panel_id", "")),
            severity="warning",
            element_type="annotation",
            message=f"Annotation '{spec.text}' required fallback placement.",
            suggested_fix="Review annotation landing or add panel-specific whitespace.",
        )
    )
    ax._windowed_qa_issues = issues
    return fallback


def place_annotations(ax, specs: Sequence[AnnotationSpec]):
    artists = []
    for spec in sorted(specs, key=lambda item: item.priority, reverse=True):
        artists.append(place_annotation(ax, spec))
    return artists


def thin_linewidth(linewidth):
    if linewidth is None:
        return linewidth
    if linewidth >= 2.6:
        return LINE_WIDTHS["primary"]
    if linewidth >= 2.0:
        return 1.35
    if linewidth >= 1.4:
        return LINE_WIDTHS["secondary"]
    if linewidth >= 1.0:
        return LINE_WIDTHS["tertiary"]
    if linewidth >= 0.7:
        return LINE_WIDTHS["ticks"]
    return linewidth


def polish_figure(fig, *, typography: str = "main") -> None:
    fig.canvas.draw()
    preset = typography if typography in TYPOGRAPHY_PRESETS else "main"
    ladder = TYPOGRAPHY_PRESETS[preset]
    preset_min_font = 5.0 if preset == "supplementary" else 4.0
    axes_lw = ELIFE_LINE_WIDTHS["axis"] if preset == "supplementary" else LINE_WIDTHS["axes"]
    tick_lw = ELIFE_LINE_WIDTHS["tick"] if preset == "supplementary" else LINE_WIDTHS["ticks"]
    tick_len = 2.6 if preset == "supplementary" else 2.8
    for ax in fig.axes:
        if getattr(ax, "name", "") != "3d":
            if getattr(ax, "_windowed_open_axes", False):
                for name, spine in ax.spines.items():
                    spine.set_visible(name in {"left", "bottom"})
                    spine.set_linewidth(axes_lw)
                    spine.set_edgecolor(FRAME_COLOR)
                ax.tick_params(
                    width=tick_lw,
                    length=tick_len,
                    labelsize=ladder["tick"],
                    top=False,
                    right=False,
                    direction="out",
                )
            elif getattr(ax, "_windowed_no_polish_spines", False):
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_frame_on(False)
            else:
                for spine in ax.spines.values():
                    spine.set_linewidth(axes_lw)
                    spine.set_edgecolor(FRAME_COLOR)
                ax.tick_params(width=tick_lw, length=tick_len, labelsize=ladder["tick"], direction="out")
            ax.xaxis.label.set_size(ladder["axis_label"])
            ax.yaxis.label.set_size(ladder["axis_label"])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(ladder["tick"])
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontfamily(FONT_STACK)
                text.set_fontweight("normal")
                text.set_fontsize(ladder["legend"])
                text.set_path_effects([])
            for handle in getattr(legend, "legend_handles", []):
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(thin_linewidth(handle.get_linewidth()))

    for text in fig.findobj(mtext.Text):
        text.set_fontfamily(FONT_STACK)
        text.set_text(_single_line(text.get_text()) if getattr(text, "_windowed_single_line_title", False) else text.get_text())
        current_weight = str(text.get_fontweight()).lower()
        if current_weight not in {"bold", "semibold", "heavy", "demibold"}:
            text.set_fontweight("normal")
        role = str(getattr(text, "_windowed_annotation_role", ""))
        if role in ladder:
            text.set_fontsize(ladder[role])
        elif text.get_fontsize() < preset_min_font:
            text.set_fontsize(preset_min_font)
        min_size = ROLE_MIN_FONT_SIZES.get(role)
        if min_size is not None and text.get_fontsize() < min_size:
            text.set_fontsize(min_size)
        if not getattr(text, "_windowed_allow_path_effects", False):
            text.set_path_effects([])
        bbox_patch = text.get_bbox_patch()
        if bbox_patch is not None:
            bbox_patch.set_linewidth(min(bbox_patch.get_linewidth(), 0.6))
            bbox_patch.set_edgecolor(BOX_EDGE)
            bbox_patch.set_alpha(max(min(bbox_patch.get_alpha() or 1.0, 0.94), 0.82))
        if getattr(text, "_windowed_single_line_title", False) and text.axes is not None:
            place_panel_title(text.axes, text, min_font_size=ROLE_MIN_FONT_SIZES["panel_title"])
        if role in INTERIOR_SAFE_TEXT_ROLES:
            clamp_artist_to_axes(text, margin_px=SUPPLEMENTARY_QA_CONFIG.safe_zone_px)

    for ax in fig.axes:
        resolve_text_collisions(ax, margin_px=SUPPLEMENTARY_QA_CONFIG.safe_zone_px)

    for line in fig.findobj(matplotlib.lines.Line2D):
        line.set_linewidth(thin_linewidth(line.get_linewidth()))
        if hasattr(line, "set_solid_capstyle"):
            line.set_solid_capstyle("round")
        if hasattr(line, "set_solid_joinstyle"):
            line.set_solid_joinstyle("round")

    for patch in fig.findobj(mpatches.Patch):
        if hasattr(patch, "get_linewidth") and hasattr(patch, "set_linewidth"):
            patch.set_linewidth(thin_linewidth(patch.get_linewidth()))

    for collection in fig.findobj(mcoll.Collection):
        if hasattr(collection, "get_linewidths") and hasattr(collection, "set_linewidths"):
            widths = collection.get_linewidths()
            if widths is not None and len(widths):
                collection.set_linewidths([thin_linewidth(float(w)) for w in widths])


def _text_font_allowed(text: mtext.Text, config: FigureQAConfig) -> bool:
    families = text.get_fontfamily()
    if isinstance(families, str):
        families = [families]
    normalized = {str(family).lower() for family in families}
    allowed = {family.lower() for family in config.allowed_font_families}
    return bool(normalized & allowed)


def _text_bbox(fig, text: mtext.Text):
    if not text.get_visible() or not text.get_text().strip():
        return None
    try:
        return text.get_window_extent(fig.canvas.get_renderer())
    except Exception:
        return None


def _figure_texts(fig) -> list[mtext.Text]:
    return [text for text in fig.findobj(mtext.Text) if text.get_visible() and text.get_text().strip()]


def _is_tick_or_axis_text(text: mtext.Text) -> bool:
    ax = text.axes
    if ax is None:
        return False
    tick_texts = set(ax.get_xticklabels() + ax.get_yticklabels())
    if text in tick_texts:
        return True
    if text is ax.xaxis.label or text is ax.yaxis.label:
        return True
    return False


def _is_layout_annotation(text: mtext.Text) -> bool:
    role = str(getattr(text, "_windowed_annotation_role", ""))
    return role in {
        "panel_letter",
        "panel_title",
        "callout",
        "stat_box",
        "status_chip",
        "direct_label",
        "provenance",
        "source-limited",
        "stat_bracket",
    }


def _issue(config: FigureQAConfig, figure_id: str, panel_id: str, severity: str, element_type: str, message: str, suggested_fix: str = "") -> FigureQAIssue:
    return FigureQAIssue(
        figure_id=figure_id,
        panel_id=panel_id,
        severity=severity,
        element_type=element_type,
        message=message,
        suggested_fix=suggested_fix,
        pass_label=config.qa_pass_label,
    )


def _has_data_content(ax) -> bool:
    if getattr(ax, "_windowed_empty_state", False):
        return True
    if getattr(ax, "tables", None):
        return True
    if ax.images or ax.lines or ax.collections:
        return True
    meaningful_patches = [patch for patch in ax.patches if not isinstance(patch, matplotlib.spines.Spine)]
    if meaningful_patches:
        return True
    data_texts = [
        text for text in ax.texts
        if text.get_text().strip() and getattr(text, "_windowed_annotation_role", "") not in {"panel_letter", "panel_title"}
    ]
    return bool(data_texts)


def validate_figure_integrity(fig, figure_id: str, config: FigureQAConfig | None = None) -> list[FigureQAIssue]:
    """Best-effort visual integrity checks for rendered publication figures."""
    config = config or FigureQAConfig()
    fig.canvas.draw()
    issues: list[FigureQAIssue] = []
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox

    for ax in fig.axes:
        for local_issue in getattr(ax, "_windowed_qa_issues", []):
            issues.append(
                _issue(config, figure_id, local_issue.panel_id, local_issue.severity, local_issue.element_type, local_issue.message, local_issue.suggested_fix)
            )

    texts = _figure_texts(fig)
    for text in texts:
        panel_id = str(getattr(text.axes, "_windowed_panel_id", "")) if text.axes is not None else ""
        role = str(getattr(text, "_windowed_annotation_role", "text"))
        if text.get_path_effects() and not getattr(text, "_windowed_allow_path_effects", False):
            issues.append(_issue(config, figure_id, panel_id, "critical", role, f"Text '{text.get_text()[:32]}' has path effects.", "Remove halo/stroke effects; solve readability through placement."))
        if not _text_font_allowed(text, config):
            issues.append(_issue(config, figure_id, panel_id, "warning", role, f"Text '{text.get_text()[:32]}' does not request the approved font stack.", "Use the shared Helvetica-first helpers."))
        fontsize = float(text.get_fontsize())
        role_min = ROLE_MIN_FONT_SIZES.get(role, config.min_font_size)
        if fontsize < config.critical_min_font_size:
            issues.append(_issue(config, figure_id, panel_id, "critical", role, f"Text '{text.get_text()[:32]}' is below critical size ({fontsize:.2f} pt).", "Increase text size or simplify the annotation."))
        elif fontsize < config.min_font_size and not getattr(text.axes, "_windowed_non_panel_axis", False):
            issues.append(_issue(config, figure_id, panel_id, "warning", role, f"Text '{text.get_text()[:32]}' is below the figure minimum ({fontsize:.2f} pt).", "Use the shared typography preset or mark helper axes as non-panel."))
        elif not _is_tick_or_axis_text(text) and _is_layout_annotation(text) and fontsize < role_min:
            issues.append(_issue(config, figure_id, panel_id, "warning", role, f"Text '{text.get_text()[:32]}' is below target size ({fontsize:.2f} pt).", "Use the modality text-size preset."))
        if role == "panel_title":
            if "\n" in text.get_text():
                issues.append(_issue(config, figure_id, panel_id, "critical", role, f"Panel title '{text.get_text()[:32]}' is multi-line.", "Use fit_single_line_text or shorten the title."))
        bbox = _text_bbox(fig, text)
        if bbox is not None and _is_layout_annotation(text) and not Bbox.intersection(fig_bbox.expanded(1.002, 1.002), bbox):
            issues.append(_issue(config, figure_id, panel_id, "warning", role, f"Text '{text.get_text()[:32]}' is outside the figure canvas.", "Move annotation inside the panel or increase margins."))
        if (
            bbox is not None
            and text.axes is not None
            and role in INTERIOR_SAFE_TEXT_ROLES
            and text.get_transform() == text.axes.transAxes
        ):
            try:
                safe_bbox = _axis_safe_bbox(text.axes, renderer, config.safe_zone_px)
                if (
                    bbox.x0 < safe_bbox.x0
                    or bbox.y0 < safe_bbox.y0
                    or bbox.x1 > safe_bbox.x1
                    or bbox.y1 > safe_bbox.y1
                ):
                    issues.append(_issue(config, figure_id, panel_id, "warning", "safe_zone", f"Annotation '{text.get_text()[:32]}' breaches the panel safe zone.", "Use place_safe_text or reserve local annotation whitespace."))
            except Exception:
                pass
        if bbox is not None and text.axes is not None and role in {"status_chip", "provenance", "source-limited"}:
            ax_bbox = text.axes.get_window_extent(renderer).expanded(1.08, 1.12)
            tolerance = max(config.edge_margin_px, 30.0)
            if (
                bbox.x0 < ax_bbox.x0 - tolerance
                or bbox.y0 < ax_bbox.y0 - tolerance
                or bbox.x1 > ax_bbox.x1 + tolerance
                or bbox.y1 > ax_bbox.y1 + tolerance
            ):
                issues.append(_issue(config, figure_id, panel_id, "warning", "edge_clutter", f"Annotation '{text.get_text()[:32]}' is close to or outside the panel edge.", "Use place_annotation with edge-safe candidates."))

    if config.required_panel_letters:
        for ax in fig.axes:
            panel_id = str(getattr(ax, "_windowed_panel_id", ""))
            if not panel_id:
                continue
            expected = panel_id[-1]
            found = any(text.get_text().strip() == expected for text in ax.texts)
            if not found:
                severity = "critical" if config.critical_on_missing_panel_letter else "warning"
                issues.append(_issue(config, figure_id, panel_id, severity, "panel_letter", f"Panel {panel_id} is missing its panel letter.", "Use apply_panel_header/apply_elife_panel_header."))

    if config.check_empty_axes:
        for ax in fig.axes:
            panel_id = str(getattr(ax, "_windowed_panel_id", ""))
            if panel_id and not getattr(ax, "_windowed_container_axis", False) and not _has_data_content(ax):
                issues.append(_issue(config, figure_id, panel_id, "critical", "empty_panel", "Panel axis has no visible data or explicit source-limited placeholder.", "Render a source-limited placeholder or remove the unused panel axis."))
            if config.check_scale_bars and getattr(ax, "_windowed_requires_scale_bar", False) and not getattr(ax, "_windowed_scale_bar", False):
                issues.append(_issue(config, figure_id, panel_id, "critical", "scale_bar", "Image panel requires a scale bar but none was registered.", "Call add_scale_bar for calibrated image panels."))

    for line in fig.findobj(matplotlib.lines.Line2D):
        try:
            linewidth = float(line.get_linewidth())
        except Exception:
            continue
        if linewidth > config.max_linewidth:
            panel_id = str(getattr(line.axes, "_windowed_panel_id", "")) if line.axes is not None else ""
            issues.append(_issue(config, figure_id, panel_id, "warning", "line", f"Line width {linewidth:.2f} exceeds the shared target.", "Use the shared line-width ladder."))

    if config.check_collisions:
        collision_texts = [
            text for text in texts
            if getattr(text, "_windowed_annotation_role", "") in {"panel_letter", "panel_title", "callout", "stat_box", "status_chip", "direct_label", "provenance", "source-limited"}
        ]
        bboxes: list[tuple[mtext.Text, Bbox]] = []
        for text in collision_texts:
            bbox = _text_bbox(fig, text)
            if bbox is not None:
                bboxes.append((text, bbox.expanded(1.02, 1.05)))
        for idx, (left_text, left_bbox) in enumerate(bboxes):
            for right_text, right_bbox in bboxes[idx + 1:]:
                if left_text.axes is not right_text.axes:
                    continue
                if left_bbox.overlaps(right_bbox):
                    x0 = max(left_bbox.x0, right_bbox.x0)
                    y0 = max(left_bbox.y0, right_bbox.y0)
                    x1 = min(left_bbox.x1, right_bbox.x1)
                    y1 = min(left_bbox.y1, right_bbox.y1)
                    overlap = max(x1 - x0, 0) * max(y1 - y0, 0)
                    left_role = str(getattr(left_text, "_windowed_annotation_role", ""))
                    right_role = str(getattr(right_text, "_windowed_annotation_role", ""))
                    if {left_role, right_role} == {"panel_letter", "panel_title"} and overlap <= 28.0:
                        continue
                    if {"panel_letter", "panel_title"} & {left_role, right_role} and "direct_label" in {left_role, right_role}:
                        continue
                    if overlap > 12.0:
                        panel_id = str(getattr(left_text.axes, "_windowed_panel_id", "")) if left_text.axes is not None else ""
                        issues.append(_issue(config, figure_id, panel_id, "warning", "collision", f"Annotation collision between '{left_text.get_text()[:20]}' and '{right_text.get_text()[:20]}'.", "Use place_annotation or adjust panel-specific landing."))
                        break

    return issues


def write_figure_qa_report(issues: Sequence[FigureQAIssue], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QA_REPORT_FIELDS, delimiter="\t")
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.as_row())


def summarize_figure_qa_issues(issues: Sequence[FigureQAIssue]) -> dict[str, Any]:
    severity_counts: dict[str, int] = {}
    for issue in issues:
        severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
    return {
        "total_issues": len(issues),
        "severity_counts": severity_counts,
        "critical_issues": severity_counts.get("critical", 0),
        "warning_issues": severity_counts.get("warning", 0),
    }


def write_figure_qa_summary(path, *, main_issues: Sequence[FigureQAIssue] | None = None, supplementary_issues: Sequence[FigureQAIssue] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}
    summary["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if main_issues is not None:
        summary["main"] = summarize_figure_qa_issues(main_issues)
    if supplementary_issues is not None:
        summary["supplementary"] = summarize_figure_qa_issues(supplementary_issues)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def assert_no_critical_figure_issues(issues: Sequence[FigureQAIssue]) -> None:
    critical = [issue for issue in issues if issue.severity == "critical"]
    if critical:
        preview = "; ".join(f"{issue.figure_id}:{issue.panel_id}:{issue.message}" for issue in critical[:5])
        raise RuntimeError(f"Critical figure QA issues detected: {preview}")


def figure_issue_fingerprint(issue: FigureQAIssue) -> tuple[str, str, str, str, str]:
    return (
        issue.figure_id,
        issue.panel_id,
        issue.severity,
        issue.element_type,
        issue.message,
    )


def persistent_figure_warnings(pass_one: Sequence[FigureQAIssue], pass_two: Sequence[FigureQAIssue]) -> list[FigureQAIssue]:
    first_warnings = {figure_issue_fingerprint(issue) for issue in pass_one if issue.severity == "warning"}
    return [issue for issue in pass_two if issue.severity == "warning" and figure_issue_fingerprint(issue) in first_warnings]


def assert_no_persistent_figure_warnings(pass_one: Sequence[FigureQAIssue], pass_two: Sequence[FigureQAIssue]) -> None:
    persistent = persistent_figure_warnings(pass_one, pass_two)
    if persistent:
        preview = "; ".join(f"{issue.figure_id}:{issue.panel_id}:{issue.message}" for issue in persistent[:5])
        raise RuntimeError(f"Persistent figure QA warnings detected after two-pass render: {preview}")


def write_targeted_patch_notes(pass_one: Sequence[FigureQAIssue], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    warnings = [issue for issue in pass_one if issue.severity == "warning"]
    if not warnings:
        body = "# V44 targeted patch notes\n\nPass 1 produced no warnings requiring targeted layout patches.\n"
    else:
        lines = ["# V44 targeted patch notes", "", "Pass 1 warnings to review before pass 2:"]
        for issue in warnings:
            lines.append(f"- {issue.figure_id} {issue.panel_id or '(figure-level)'}: {issue.element_type} - {issue.message}")
        body = "\n".join(lines) + "\n"
    path.write_text(body, encoding="utf-8")


def write_panel_crop_contact_sheet(fig, figure_id: str, output_dir: str | Path, *, panel_axes_only: bool = True) -> Path | None:
    """Write deterministic per-panel crops and a compact contact sheet for visual QA."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    try:
        rgba = np.asarray(fig.canvas.buffer_rgba())
    except Exception:
        return None
    image = Image.fromarray(rgba).convert("RGB")
    fig_height = image.height
    crops: list[tuple[str, Image.Image]] = []
    renderer = fig.canvas.get_renderer()
    for idx, ax in enumerate(fig.axes, start=1):
        panel_id = str(getattr(ax, "_windowed_panel_id", ""))
        if panel_axes_only and (not panel_id or getattr(ax, "_windowed_non_panel_axis", False)):
            continue
        panel_id = panel_id or f"panel{idx:02d}"
        if panel_id.startswith("<"):
            panel_id = f"panel{idx:02d}"
        try:
            bbox = ax.get_window_extent(renderer).expanded(1.10, 1.18)
        except Exception:
            continue
        left = int(max(0, bbox.x0))
        right = int(min(image.width, bbox.x1))
        upper = int(max(0, fig_height - bbox.y1))
        lower = int(min(fig_height, fig_height - bbox.y0))
        if right <= left or lower <= upper:
            continue
        crop = image.crop((left, upper, right, lower))
        crop_path = output_dir / f"{figure_id}_{panel_id}.png"
        crop.save(crop_path)
        crops.append((panel_id, crop))
    if not crops:
        return None
    thumb_w = 320
    label_h = 24
    thumbs: list[Image.Image] = []
    for panel_id, crop in crops:
        scale = thumb_w / max(crop.width, 1)
        thumb = crop.resize((thumb_w, max(1, int(crop.height * scale))))
        canvas = Image.new("RGB", (thumb_w, thumb.height + label_h), "white")
        canvas.paste(thumb, (0, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), panel_id, fill=(44, 44, 42))
        thumbs.append(canvas)
    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * thumb_w, (idx // cols) * cell_h))
    sheet_path = output_dir / f"{figure_id}_panel_contact_sheet.png"
    sheet.save(sheet_path)
    return sheet_path


def write_image_contact_sheet(image_paths: Sequence[str | Path], output_path: str | Path, *, thumb_width: int = 420) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cards: list[Image.Image] = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        with Image.open(path) as source:
            image = source.convert("RGB")
        scale = thumb_width / max(image.width, 1)
        thumb = image.resize((thumb_width, max(1, int(image.height * scale))))
        label_h = 28
        card = Image.new("RGB", (thumb.width, thumb.height + label_h), "white")
        card.paste(thumb, (0, label_h))
        draw = ImageDraw.Draw(card)
        draw.text((6, 7), path.name, fill=(44, 44, 42))
        cards.append(card)
    if not cards:
        empty = Image.new("RGB", (thumb_width, 80), "white")
        ImageDraw.Draw(empty).text((8, 30), "No figures found", fill=(44, 44, 42))
        empty.save(output_path)
        return output_path
    cols = min(3, len(cards))
    rows = int(np.ceil(len(cards) / cols))
    cell_h = max(card.height for card in cards)
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), "white")
    for idx, card in enumerate(cards):
        sheet.paste(card, ((idx % cols) * thumb_width, (idx // cols) * cell_h))
    sheet.save(output_path)
    return output_path


def _validate_and_record(fig, figure_id: str | None, qa_config: FigureQAConfig | None, fail_on_critical: bool) -> list[FigureQAIssue]:
    if qa_config is None:
        return []
    issues = validate_figure_integrity(fig, figure_id or "unknown_figure", qa_config)
    _FIGURE_QA_ISSUES.extend(issues)
    if qa_config.panel_crop_dir:
        write_panel_crop_contact_sheet(
            fig,
            figure_id or "unknown_figure",
            qa_config.panel_crop_dir,
            panel_axes_only=qa_config.crop_panel_axes_only,
        )
    if fail_on_critical:
        assert_no_critical_figure_issues(issues)
    return issues


def save_publication_figure(fig, output_path, display_path=None, dpi: int = 600, close: bool = True, *, qa_config: FigureQAConfig | None = None, figure_id: str | None = None, fail_on_critical: bool = True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preset = typography_preset_name(figure_id or output_path.stem, qa_config)
    polish_figure(fig, typography=preset)
    issues = _validate_and_record(fig, figure_id or output_path.stem, qa_config, fail_on_critical)
    fig.savefig(output_path, dpi=dpi)
    if display_path is not None:
        display_path = Path(display_path)
        display_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, display_path)
    if close:
        plt.close(fig)
    return issues


def save_publication_bundle(fig, output_path, display_path=None, dpi: int = 600, close: bool = True, *, qa_config: FigureQAConfig | None = None, figure_id: str | None = None, fail_on_critical: bool = True):
    output_path = Path(output_path)
    output_pdf = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preset = typography_preset_name(figure_id or output_path.stem, qa_config)
    polish_figure(fig, typography=preset)
    issues = _validate_and_record(fig, figure_id or output_path.stem, qa_config, fail_on_critical)
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_pdf)
    if display_path is not None:
        display_path = Path(display_path)
        display_pdf = display_path.with_suffix(".pdf")
        display_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, display_path)
        shutil.copy2(output_pdf, display_pdf)
    if close:
        plt.close(fig)
    return issues
