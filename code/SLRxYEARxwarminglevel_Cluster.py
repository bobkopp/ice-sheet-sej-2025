#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:33:36 2026

Ice-sheet contributions to sea-level rise by projection year
and warming level.

Creates a 2 x 2 figure with panels for:
    A. GrIS
    B. EAIS
    C. WAIS
    D. Total ice sheets

The x-axis shows projection year:
    2100
    2200
    2300

Within each year, clustered probability bars represent:
    2.0°C warming
    3.5°C warming
    5.0°C warming

Each probability bar shows:
    Thin vertical line: 5th–95th percentile range
    Thick vertical line: 17th–83rd percentile range
    Circle: 50th percentile (median)

@author: elizabeth
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

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

os.makedirs(
    output_dir,
    exist_ok=True
)


figure_png = os.path.join(
    output_dir,
    "Figure05_clustered_SLR_by_year.png"
)

figure_pdf = os.path.join(
    output_dir,
    "Figure05_clustered_SLR_by_year.pdf"
)


years = [
    2100,
    2200,
    2300
]


warming_levels = [
    2.0,
    3.5,
    5.0
]


percentile_levels = [
    5,
    17,
    50,
    83,
    95
]


# Display name mapped to the exact name used in mc_total_ext.
#
# "Total ice-sheet SLR" includes:
#     EAIS + WAIS + GrIS
#
# It is therefore different from "Total AIS," which only
# includes the Antarctic contribution.
panel_ice_sheets = {
    "GrIS": "GrIS",
    "EAIS": "EAIS",
    "WAIS": "WAIS",
    "Total ice sheets": "Total ice-sheet SLR"
}


panel_labels = {
    "GrIS": "A",
    "EAIS": "B",
    "WAIS": "C",
    "Total ice sheets": "D"
}


warming_labels = {
    2.0: "2.0°C",
    3.5: "3.5°C",
    5.0: "5.0°C"
}


# Colorblind-friendly colors for warming levels.
warming_colors = {
    2.0: "#0072B2",
    3.5: "#6A3D9A",
    5.0: "#D55E00"
}


# Use evenly spaced categorical positions rather than the
# numerical year values themselves.
year_positions = {
    2100: 1.0,
    2200: 2.0,
    2300: 3.0
}


# Horizontal offset of each warming level within a year cluster.
warming_offsets = {
    2.0: -0.20,
    3.5: 0.00,
    5.0: 0.20
}


# Half-width of the horizontal percentile ticks.
tick_half_width = 0.045


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

data = load_and_prepare_data()

mc_total_ext = data["mc_total_ext"]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_ice_sheet_samples(
    dataframe,
    ice_sheet,
    year,
    warming
):
    """
    Extract Monte Carlo samples for one ice-sheet component.

    Values are converted from millimeters to centimeters.
    """

    samples = dataframe.loc[
        (
            dataframe["ice_sheet"] == ice_sheet
        )
        & (
            dataframe["year"] == year
        )
        & (
            dataframe["warming"] == warming
        ),
        "value"
    ].dropna()

    if samples.empty:
        raise ValueError(
            "No samples found for "
            f"{ice_sheet}, {year}, {warming:.1f}°C."
        )

    return samples.to_numpy() / 10.0


def add_vertical_probability_bar(
    ax,
    samples,
    x_position,
    color
):
    """
    Add one vertical probability bar.

    Thin line:
        5th–95th percentile range

    Thick line:
        17th–83rd percentile range

    Circle:
        50th percentile
    """

    q05, q17, q50, q83, q95 = np.percentile(
        samples,
        percentile_levels
    )

    # --------------------------------------------------------
    # 5TH–95TH PERCENTILE RANGE
    # --------------------------------------------------------

    ax.vlines(
        x=x_position,
        ymin=q05,
        ymax=q95,
        color=color,
        linewidth=1.5,
        zorder=3
    )

    # --------------------------------------------------------
    # 17TH–83RD PERCENTILE RANGE
    # --------------------------------------------------------

    ax.vlines(
        x=x_position,
        ymin=q17,
        ymax=q83,
        color=color,
        linewidth=5.0,
        zorder=4
    )

    # --------------------------------------------------------
    # MEDIAN
    # --------------------------------------------------------

    ax.plot(
        x_position,
        q50,
        marker="o",
        markersize=6,
        color=color,
        markeredgecolor=color,
        zorder=5
    )

    # --------------------------------------------------------
    # PERCENTILE TICKS
    # --------------------------------------------------------

    for quantile in [
        q05,
        q17,
        q50,
        q83,
        q95
    ]:

        ax.hlines(
            y=quantile,
            xmin=x_position - tick_half_width,
            xmax=x_position + tick_half_width,
            color=color,
            linewidth=0.9,
            zorder=5
        )


def calculate_panel_y_limits(
    distributions,
    padding_fraction=0.10
):
    """
    Calculate panel y-axis limits from the displayed
    5th–95th percentile ranges.
    """

    percentile_ranges = np.array([
        np.percentile(
            samples,
            [5, 95]
        )
        for samples in distributions.values()
    ])

    y_min_data = percentile_ranges[:, 0].min()
    y_max_data = percentile_ranges[:, 1].max()

    display_range = (
        y_max_data
        - y_min_data
    )

    if display_range == 0:
        display_range = 1.0

    padding = (
        padding_fraction
        * display_range
    )

    return (
        y_min_data - padding,
        y_max_data + padding
    )


# ============================================================
# 4. PREPARE ALL DISTRIBUTIONS
# ============================================================

all_distributions = {}

for panel_name, data_name in panel_ice_sheets.items():

    all_distributions[panel_name] = {}

    for year in years:

        for warming in warming_levels:

            all_distributions[panel_name][
                (year, warming)
            ] = get_ice_sheet_samples(
                dataframe=mc_total_ext,
                ice_sheet=data_name,
                year=year,
                warming=warming
            )


# ============================================================
# 5. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(11.5, 8.5),
    sharex=True
)

axes = axes.flatten()


for ax, panel_name in zip(
    axes,
    panel_ice_sheets.keys()
):

    distributions = all_distributions[
        panel_name
    ]

    # --------------------------------------------------------
    # DRAW CLUSTERED PROBABILITY BARS
    # --------------------------------------------------------

    for year in years:

        for warming in warming_levels:

            x_position = (
                year_positions[year]
                + warming_offsets[warming]
            )

            samples = distributions[
                (year, warming)
            ]

            add_vertical_probability_bar(
                ax=ax,
                samples=samples,
                x_position=x_position,
                color=warming_colors[warming]
            )

    # --------------------------------------------------------
    # Y-AXIS LIMITS
    # --------------------------------------------------------

    y_min, y_max = calculate_panel_y_limits(
        distributions=distributions,
        padding_fraction=0.10
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    # Add one minor interval between each pair of major
    # y-axis ticks.
    ax.yaxis.set_minor_locator(
        AutoMinorLocator(2)
    )

    # --------------------------------------------------------
    # X AXIS
    # --------------------------------------------------------

    ax.set_xlim(
        0.55,
        3.45
    )

    ax.set_xticks([
        year_positions[year]
        for year in years
    ])

    ax.set_xticklabels([
        str(year)
        for year in years
    ])

    # Do not add minor x ticks because the bars themselves
    # occupy the positions between the year centers.
    ax.tick_params(
        axis="x",
        which="minor",
        bottom=False
    )

    # --------------------------------------------------------
    # Y AXIS
    # --------------------------------------------------------

    ax.set_ylabel(
        "Contribution to SLR (cm)",
        fontsize=11,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # PANEL TITLE AND LABEL
    # --------------------------------------------------------

    ax.set_title(
        panel_name,
        fontsize=12,
        fontweight="bold",
        pad=8
    )

    ax.text(
        0.02,
        0.98,
        panel_labels[panel_name],
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top"
    )

    # --------------------------------------------------------
    # MAJOR AND MINOR GRID LINES
    # --------------------------------------------------------

    # Major horizontal and vertical grid lines.
    ax.grid(
        axis="both",
        which="major",
        linestyle="--",
        linewidth=0.65,
        color="0.82",
        alpha=0.9,
        zorder=0
    )

    # Minor horizontal grid lines.
    ax.grid(
        axis="y",
        which="minor",
        linestyle=":",
        linewidth=0.50,
        color="0.91",
        alpha=0.9,
        zorder=0
    )

    # --------------------------------------------------------
    # GENERAL FORMATTING
    # --------------------------------------------------------

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=10,
        length=4
    )

    ax.tick_params(
        axis="y",
        which="minor",
        direction="out",
        length=2.5
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )


# ============================================================
# 6. SHARED X-AXIS LABELS
# ============================================================

axes[2].set_xlabel(
    "Projection year",
    fontsize=11,
    fontweight="bold"
)

axes[3].set_xlabel(
    "Projection year",
    fontsize=11,
    fontweight="bold"
)


# ============================================================
# 7. LEGEND
# ============================================================

legend_handles = [
    Line2D(
        [0],
        [0],
        color=warming_colors[warming],
        linewidth=4,
        marker="o",
        markersize=6,
        label=warming_labels[warming]
    )
    for warming in warming_levels
]


fig.legend(
    handles=legend_handles,
    title="Warming level",
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    frameon=False,
    fontsize=10,
    title_fontsize=10
)


# ============================================================
# 8. LAYOUT AND SAVE
# ============================================================

fig.subplots_adjust(
    left=0.09,
    right=0.98,
    bottom=0.14,
    top=0.94,
    wspace=0.22,
    hspace=0.28
)


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
plt.close(fig)


print(
    "\nSaved clustered year figure:"
)

print(
    figure_png
)

print(
    figure_pdf
)