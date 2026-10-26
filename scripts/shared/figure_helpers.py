"""Reusable plotting helpers. Never recompute scientific distributions here.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from __future__ import annotations

from figure_labels import MM_PER_CM
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from figure_settings import DPI, SHOW_FIGURES


def save_figure(fig, path, *, dpi=None, bbox_inches="tight", show=False,
                close=False, **savefig_kwargs) -> Path:
    """Save a PNG figure consistently; optionally display then close it.

    Defaults match the original manuscript PNG calls: configured DPI, tight
    bbox, and no automatic closing. Use bbox_inches=None to preserve an
    existing non-tight figure. show=True uses the active matplotlib backend;
    batch runs leave show=False. Return the saved file path.
    """
    target = Path(path).expanduser()
    if target.suffix.lower() != ".png":
        raise ValueError(f"Expected a PNG output path, got: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=DPI if dpi is None else dpi,
                bbox_inches=bbox_inches, **savefig_kwargs)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return target


def maybe_show() -> None:
    """Avoid FigureCanvasAgg non-interactive warnings for batch runs."""
    if SHOW_FIGURES:
        plt.show()


def b25_samples_cm(dataframe, ice_sheet: str, year: int, warming: float) -> np.ndarray:
    """Retrieve original B25 draws and convert mm to cm (never sum quantiles)."""
    samples = dataframe.loc[
        (dataframe['ice_sheet'] == ice_sheet)
        & (dataframe['year'] == year)
        & (dataframe['warming'] == warming), 'value'
    ].dropna()
    if samples.empty:
        raise ValueError(f'No B25 samples: {ice_sheet}, {year}, {warming:g}°C')
    return samples.to_numpy() / MM_PER_CM
