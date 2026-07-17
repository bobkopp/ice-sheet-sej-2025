#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:23:06 2026

Figure: Ice-sheet contribution to sea-level rise as a function
of warming level.

Creates a 2 x 2 figure with panels for:
    A. GrIS
    B. EAIS
    C. WAIS
    D. Total ice-sheet SLR

Within each warming-level cluster, probability bars represent:
    2100
    2200
    2300

Each probability bar shows:
    Thin vertical line: 5th–95th percentile range
    Thick vertical line: 17th–83rd percentile range
    Circle: 50th percentile (median)

@author: elizabeth
"""

import os
import matplotlib.pyplot as plt
import numpy as np
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

os.makedirs(output_dir, exist_ok=True)

figure_png = os.path.join(
    output_dir,
    "Figure05_clustered_SLR_by_warming.png"
)

figure_pdf = os.path.join(
    output_dir,
    "Figure05_clustered_SLR_by_warming.pdf"
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


# Display name mapped to the name used in mc_total_ext
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


# Distinguish years by color
year_colors = {
    2100: "#0072B2",
    2200: "#CC79A7",
    2300: "#D55E00"
}


# Horizontal offsets around each warming level
year_offsets = {
    2100: -0.18,
    2200: 0.00,
    2300: 0.18
}


# Width of percentile ticks, measured in warming-level units
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
    Extract samples for one ice-sheet component and convert
    millimeters to centimeters.
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
        median
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
    Calculate y-axis limits from the displayed 5th–95th
    percentile ranges.
    """

    percentile_ranges = np.array([
        np.percentile(samples, [5, 95])
        for samples in distributions.values()
    ])

    y_min_data = percentile_ranges[:, 0].min()
    y_max_data = percentile_ranges[:, 1].max()

    display_range = y_max_data - y_min_data

    if display_range == 0:
        display_range = 1.0

    padding = padding_fraction * display_range

    return (
        y_min_data - padding,
        y_max_data + padding
    )


# ============================================================
# 4. PREPARE DISTRIBUTIONS
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

    distributions = all_distributions[panel_name]

    # --------------------------------------------------------
    # DRAW CLUSTERED PROBABILITY BARS
    # --------------------------------------------------------

    for warming in warming_levels:

        for year in years:

            x_position = (
                warming
                + year_offsets[year]
            )

            samples = distributions[
                (year, warming)
            ]

            add_vertical_probability_bar(
                ax=ax,
                samples=samples,
                x_position=x_position,
                color=year_colors[year]
            )

    # --------------------------------------------------------
    # PANEL-SPECIFIC Y LIMITS
    # --------------------------------------------------------

    y_min, y_max = calculate_panel_y_limits(
        distributions=distributions,
        padding_fraction=0.10
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    # --------------------------------------------------------
    # X AXIS
    # --------------------------------------------------------

    ax.set_xlim(
        1.55,
        5.45
    )

    ax.set_xticks(
        warming_levels
    )

    ax.set_xticklabels([
        "2.0°C",
        "3.5°C",
        "5.0°C"
    ])

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
    # FORMATTING
    # --------------------------------------------------------

    ax.tick_params(
        axis="both",
        direction="out",
        labelsize=10,
        length=4
    )

    # Major ticks
    ax.grid(
        which="major",
        color="0.85",
        linestyle="--",
        linewidth=0.6,
        zorder=0
    )
    
    # Minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(
        which="minor",
        color="0.93",
        linestyle="--",
        linewidth=0.5,
        zorder=0
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Shared x-axis labels on bottom row
axes[2].set_xlabel(
    "Warming level",
    fontsize=11,
    fontweight="bold"
)

axes[3].set_xlabel(
    "Warming level",
    fontsize=11,
    fontweight="bold"
)


# ============================================================
# 6. LEGEND
# ============================================================

legend_handles = []

for year in years:

    handle = plt.Line2D(
        [0],
        [0],
        color=year_colors[year],
        linewidth=4,
        marker="o",
        markersize=6,
        label=str(year)
    )

    legend_handles.append(handle)


fig.legend(
    handles=legend_handles,
    title="Projection year",
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    frameon=False,
    fontsize=10,
    title_fontsize=10
)


# ============================================================
# 7. LAYOUT AND SAVE
# ============================================================

fig.subplots_adjust(
    left=0.09,
    right=0.98,
    bottom=0.13,
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

print("\nSaved clustered warming-level figure:")
print(figure_png)
print(figure_pdf)