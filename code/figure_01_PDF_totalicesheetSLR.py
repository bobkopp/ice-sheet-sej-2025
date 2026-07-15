#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:50:47 2026

Figure 1. Probability density function of total ice-sheet contributions to
sea-level rise in 2100 and 2300 under low- and high-warming scenarios.

@author: elizabeth
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from B25data_processing import (
    load_and_prepare_data,
    project_dir
)


# ============================================================
# 1. SETTINGS
# ============================================================

output_dir = os.path.join(
    project_dir,
    "output",
    "figures"
)

os.makedirs(output_dir, exist_ok=True)

figure_png = os.path.join(
    output_dir,
    "Figure01_total_ice_sheet_SLR_PDFs.png"
)

figure_pdf = os.path.join(
    output_dir,
    "Figure01_total_ice_sheet_SLR_PDFs.pdf"
)


years = [2100, 2300]

warming_scenarios = {
    "Low warming (2.0°C)": 2.0,
    "High warming (5.0°C)": 5.0
}

# Colorblind-friendly palette
scenario_colors = {
    "Low warming (2.0°C)": "#0072B2",
    "High warming (5.0°C)": "#D55E00"
}

panel_labels = {
    2100: "A",
    2300: "B"
}

percentile_levels = [
    5,
    17,
    50,
    83,
    95
]


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

data = load_and_prepare_data()

mc_total_ext = data["mc_total_ext"]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_total_slr_samples(dataframe, year, warming):
    """
    Extract total ice-sheet SLR samples and convert mm to cm.
    """

    samples = dataframe.loc[
        (
            dataframe["ice_sheet"]
            == "Total ice-sheet SLR"
        )
        & (
            dataframe["year"]
            == year
        )
        & (
            dataframe["warming"]
            == warming
        ),
        "value"
    ].dropna()

    if samples.empty:
        raise ValueError(
            "No total ice-sheet SLR samples found for "
            f"year={year} and warming={warming:.1f}°C."
        )

    # Convert millimeters to centimeters
    return samples.to_numpy() / 10.0


def calculate_kde(samples, x_min, x_max, n_points=1000):
    """
    Calculate a Gaussian kernel density estimate over a
    specified x-axis range.
    """

    x_values = np.linspace(
        x_min,
        x_max,
        n_points
    )

    kde = gaussian_kde(samples)

    density = kde(x_values)

    return x_values, density


def fmt(value):
    """
    Format a rounded value using a Unicode minus sign.
    """

    return f"{value:.0f}".replace("-", "−")


def add_percentile_bar(
    ax,
    samples,
    y_position,
    color,
    annotation_offset,
    x_range,
    min_spacing_fraction=0.06
):
    """
    Draw percentile ranges and annotate the 5th, 17th,
    50th, 83rd, and 95th percentiles.

    Labels are shifted horizontally when needed to maintain
    a minimum separation based on the panel's x-axis range.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the percentile bar.

    samples : array-like
        Monte Carlo samples.

    y_position : float
        Vertical location of the percentile bar.

    color : str
        Line and annotation color.

    annotation_offset : float
        Vertical distance between the bar and its labels.

    x_range : float
        Total x-axis range for the panel.

    min_spacing_fraction : float
        Minimum horizontal spacing between adjacent labels,
        expressed as a fraction of the x-axis range.
    """

    q05, q17, q50, q83, q95 = np.percentile(
        samples,
        percentile_levels
    )

    # --------------------------------------------------------
    # PERCENTILE BAR
    # --------------------------------------------------------

    # Thin 5th–95th percentile range
    ax.hlines(
        y=y_position,
        xmin=q05,
        xmax=q95,
        color=color,
        linewidth=1.2,
        zorder=4
    )

    # Thick 17th–83rd percentile range
    ax.hlines(
        y=y_position,
        xmin=q17,
        xmax=q83,
        color=color,
        linewidth=3.0,
        zorder=5
    )

    # Median marker
    ax.plot(
        q50,
        y_position,
        marker="o",
        markersize=4,
        color=color,
        zorder=6
    )

    tick_height = annotation_offset * 0.25

    for percentile in [q05, q17, q50, q83, q95]:
        ax.vlines(
            percentile,
            y_position - tick_height,
            y_position + tick_height,
            color=color,
            linewidth=0.8,
            zorder=5
        )

    # --------------------------------------------------------
    # DYNAMIC LABEL POSITIONS
    # --------------------------------------------------------

    minimum_spacing = min_spacing_fraction * x_range

    # Labels shown above the bar
    above_values = np.array(
        [q05, q50, q95],
        dtype=float
    )

    # Labels shown below the bar
    below_values = np.array(
        [q17, q83],
        dtype=float
    )

    def spread_positions(values, minimum_spacing):
        """
        Shift labels horizontally so adjacent positions are
        separated by at least minimum_spacing.
        """

        positions = values.copy()

        # Forward pass
        for i in range(1, len(positions)):
            if positions[i] - positions[i - 1] < minimum_spacing:
                positions[i] = positions[i - 1] + minimum_spacing

        # Recenter the adjusted labels around their original mean
        positions -= positions.mean() - values.mean()

        return positions

    above_positions = spread_positions(
        above_values,
        minimum_spacing
    )

    below_positions = spread_positions(
        below_values,
        minimum_spacing
    )

    # --------------------------------------------------------
    # LABELS ABOVE BAR: 5th, 50th, 95th
    # --------------------------------------------------------

    for value, x_position, is_median in zip(
        above_values,
        above_positions,
        [False, True, False]
    ):
        ax.text(
            x_position,
            y_position + annotation_offset,
            fmt(value),
            color=color,
            fontsize=8,
            fontweight="bold" if is_median else "normal",
            ha="center",
            va="bottom",
            clip_on=False
        )

    # --------------------------------------------------------
    # LABELS BELOW BAR: 17th, 83rd
    # --------------------------------------------------------

    for value, x_position in zip(
        below_values,
        below_positions
    ):
        ax.text(
            x_position,
            y_position - annotation_offset,
            fmt(value),
            color=color,
            fontsize=8,
            ha="center",
            va="top",
            clip_on=False
        )


# ============================================================
# 4. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(11, 4.4)
)


for ax, year in zip(axes, years):

    distributions = {}

    # Extract both warming distributions first so the KDEs use
    # the same x-axis support within each panel.
    for scenario_name, warming in warming_scenarios.items():

        distributions[scenario_name] = (
            get_total_slr_samples(
                dataframe=mc_total_ext,
                year=year,
                warming=warming
            )
        )

    combined_samples = np.concatenate(
        list(distributions.values())
    )

    sample_range = (
        combined_samples.max()
        - combined_samples.min()
    )

    x_padding = 0.08 * sample_range

    x_min = combined_samples.min() - x_padding
    x_max = combined_samples.max() + x_padding

    densities = {}

    for scenario_name, samples in distributions.items():

        x_values, density = calculate_kde(
            samples=samples,
            x_min=x_min,
            x_max=x_max
        )

        densities[scenario_name] = density

        ax.plot(
            x_values,
            density,
            color=scenario_colors[scenario_name],
            linewidth=1.8,
            label=scenario_name
        )

    max_density = max(
        density.max()
        for density in densities.values()
    )

    # Place percentile bars above the KDE curves
    bar_positions = {
        "Low warming (2.0°C)": max_density * 1.38,
        "High warming (5.0°C)": max_density * 1.15
    }

    annotation_offset = max_density * 0.035

    for scenario_name, samples in distributions.items():

        add_percentile_bar(
            ax=ax,
            samples=samples,
            y_position=bar_positions[scenario_name],
            color=scenario_colors[scenario_name],
            annotation_offset=annotation_offset,
            x_range = x_max - x_min
        )

    ax.set_xlim(
        x_min,
        x_max
    )

    ax.set_ylim(
        0,
        max_density * 1.56
    )

    ax.set_xlabel(
        "Total ice-sheet contribution to SLR (cm)",
        fontsize=11,
        fontweight = 'bold'
    )

    ax.set_ylabel(
        "Probability density, p[SLR]",
        fontsize=11,
        fontweight = 'bold'
    )

    # Panel letter
    ax.text(
        0.01,
        0.98,
        panel_labels[year],
        transform=ax.transAxes,
        fontsize=13,
        ha="left",
        va="top"
    )

    # Year
    ax.text(
        0.98,
        0.98,
        str(year),
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="top"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        direction="out",
        labelsize=10
    )


# Only keep the y-axis label on the left panel
axes[1].set_ylabel("")


# ============================================================
# 5. LEGEND AND LAYOUT
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=2,
    frameon=False,
    fontsize=10
)

fig.subplots_adjust(
    left=0.09,
    right=0.98,
    bottom=0.20,
    top=0.95,
    wspace=0.22
)


# ============================================================
# 6. SAVE FIGURE
# ============================================================

fig.savefig(
    figure_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    figure_pdf,
    bbox_inches="tight"
)

plt.show()

print("\nFigure saved to:")
print(figure_png)
print(figure_pdf)