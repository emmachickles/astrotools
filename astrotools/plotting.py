"""Plotting utilities for publication-ready figures."""

from __future__ import annotations

from typing import Tuple

import matplotlib as mpl


def apply_publication_style(base_fontsize: float = 8.0, font_family: str = "serif") -> None:
    """Apply a compact, serif plotting style suitable for publication figures."""

    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": base_fontsize,
            "axes.labelsize": base_fontsize,
            "axes.titlesize": base_fontsize + 1,
            "xtick.labelsize": base_fontsize - 1,
            "ytick.labelsize": base_fontsize - 1,
            "legend.fontsize": base_fontsize - 1,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "lines.linewidth": 0.8,
            "savefig.dpi": 300,
        }
    )


def half_column_figsize(width_in: float = 3.5, aspect: float = 0.55) -> Tuple[float, float]:
    """Return a half-column figure size in inches."""

    return (width_in, width_in * aspect)
