#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:10:45 2026

Individual ice-sheet contributions to sea-level rise as a function of warming
level.

Creates four figures:
    EAIS
    WAIS
    Total AIS
    GrIS
    
Each figure contains three vertically stacked subplots:
    2100
    2200
    2300
    
Each probability bar shows:
    Thin line: 5th-95th percentile range
    Thick line: 17th-83rd percentile range
    Circle: 50th percentile (median)

@author: elizabeth
"""

import os
import matplotlib.pyplot as plt
import numpy as np
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


ice_sheets = [
    "EAIS",
    "WAIS",
    "Total AIS",
    "GrIS"
]

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


warming_labels = {
    2.0: "Low warming (2.0°C)",
    3.5: "Medium warming (3.5°C)",
    5.0: "High warming (5.0°C)"
}


ice_sheet_colors = {
    "EAIS": "#0072B2",      # blue
    "WAIS": "#E69F00",      # orange
    "Total AIS": "#CC79A7", # pink
    "GrIS": "#009E73"       # green
}


file_labels = {
    "EAIS": "EAIS",
    "WAIS": "WAIS",
    "Total AIS": "Total_AIS",
    "GrIS": "GrIS"
}

panel_labels = {
    2100: "A",
    2200: "B",
    2300: "C"
}


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

data = load_and_prepare_data()

mc_total_ext = data["mc_total_ext"]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def fmt(value):
    """Format a rounded value using a Unicode minus sign."""

    if np.isnan(value):
        return ""

    return f"{value:.0f}".replace("-", "−")


def get_ice_sheet_samples(
    dataframe,
    ice_sheet,
    year,
    warming
):
    """
    Extract samples for one ice sheet and convert mm to cm.
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


def spread_positions(
    values,
    minimum_spacing
):
    """
    Shift label positions so adjacent labels have at least
    the specified horizontal separation.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    positions = values.copy()

    for index in range(1, len(positions)):

        if (
            positions[index]
            - positions[index - 1]
            < minimum_spacing
        ):
            positions[index] = (
                positions[index - 1]
                + minimum_spacing
            )

    positions -= (
        positions.mean()
        - values.mean()
    )

    return positions


def add_probability_bar(
    ax,
    samples,
    y_position,
    color,
    x_range,
    vertical_offset=0.14,
    minimum_spacing_fraction=0.045
):
    """
    Add a labeled probability bar for one warming level.
    """

    q05, q17, q50, q83, q95 = np.percentile(
        samples,
        percentile_levels
    )

    # 5th–95th percentile range
    ax.hlines(
        y=y_position,
        xmin=q05,
        xmax=q95,
        color=color,
        linewidth=1.4,
        zorder=3
    )

    # 17th–83rd percentile range
    ax.hlines(
        y=y_position,
        xmin=q17,
        xmax=q83,
        color=color,
        linewidth=4.0,
        zorder=4
    )

    # Median
    ax.plot(
        q50,
        y_position,
        marker="o",
        markersize=6,
        color=color,
        zorder=5
    )

    # Percentile ticks
    tick_height = 0.055

    for quantile in [
        q05,
        q17,
        q50,
        q83,
        q95
    ]:
        ax.vlines(
            quantile,
            y_position - tick_height,
            y_position + tick_height,
            color=color,
            linewidth=0.9,
            zorder=4
        )

    minimum_spacing = (
        minimum_spacing_fraction
        * x_range
    )

    # 5th, 50th, and 95th above
    upper_values = np.array([
        q05,
        q50,
        q95
    ])

    upper_positions = spread_positions(
        upper_values,
        minimum_spacing
    )

    # 17th and 83rd below
    lower_values = np.array([
        q17,
        q83
    ])

    lower_positions = spread_positions(
        lower_values,
        minimum_spacing
    )

    for value, x_position, is_median in zip(
        upper_values,
        upper_positions,
        [False, True, False]
    ):
        ax.text(
            x_position,
            y_position + vertical_offset,
            fmt(value),
            color=color,
            fontsize=10,
            fontweight=(
                "bold"
                if is_median
                else "normal"
            ),
            ha="center",
            va="bottom",
            clip_on=False,
            zorder=10
        )

    for value, x_position in zip(
        lower_values,
        lower_positions
    ):
        ax.text(
            x_position,
            y_position - vertical_offset,
            fmt(value),
            color=color,
            fontsize=10,
            ha="center",
            va="top",
            clip_on=False,
            zorder=10
        )


def calculate_axis_limits(
    distributions,
    padding_fraction=0.12
):
    """
    Calculate x-axis limits from the displayed 5th–95th
    percentile ranges.
    """

    percentile_ranges = np.array([
        np.percentile(samples, [5, 95])
        for samples in distributions.values()
    ])

    x_min_data = percentile_ranges[:, 0].min()
    x_max_data = percentile_ranges[:, 1].max()

    display_range = (
        x_max_data
        - x_min_data
    )

    if display_range == 0:
        display_range = 1.0

    padding = (
        padding_fraction
        * display_range
    )

    return (
        x_min_data - padding,
        x_max_data + padding
    )


# ============================================================
# 4. CREATE ONE FIGURE FOR EACH ICE SHEET
# ============================================================

for ice_sheet in ice_sheets:

    figure_png = os.path.join(
        output_dir,
        (
            "Figure04_SLR_by_warming_"
            f"{file_labels[ice_sheet]}.png"
        )
    )

    figure_pdf = os.path.join(
        output_dir,
        (
            "Figure04_SLR_by_warming_"
            f"{file_labels[ice_sheet]}.pdf"
        )
    )

    # --------------------------------------------------------
    # PREPARE ALL DISTRIBUTIONS FOR THIS ICE SHEET
    # --------------------------------------------------------

    distributions_by_year = {
        year: {
            warming: get_ice_sheet_samples(
                dataframe=mc_total_ext,
                ice_sheet=ice_sheet,
                year=year,
                warming=warming
            )
            for warming in warming_levels
        }
        for year in years
    }

    # Calculate one common x-axis range using the 5th–95th
    # percentile bounds across every year and warming level
    # for this ice sheet.
    all_percentile_ranges = np.array([
        np.percentile(samples, [5, 95])
        for year_distributions in distributions_by_year.values()
        for samples in year_distributions.values()
    ])

    x_min_data = all_percentile_ranges[:, 0].min()
    x_max_data = all_percentile_ranges[:, 1].max()

    display_range = x_max_data - x_min_data

    if display_range == 0:
        display_range = 1.0

    padding = 0.12 * display_range

    x_min = x_min_data - padding
    x_max = x_max_data + padding
    x_range = x_max - x_min

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7.5, 10.0),
        sharex=True,
        sharey=True
    )

    for ax, year in zip(
        axes,
        years
    ):

        distributions = distributions_by_year[year]

        for warming in warming_levels:

            scenario_name = warming_labels[warming]

            add_probability_bar(
                ax=ax,
                samples=distributions[warming],
                y_position=warming,
                color=ice_sheet_colors[ice_sheet],
                x_range=x_range,
                vertical_offset=0.14,
                minimum_spacing_fraction=0.045
            )

        # ----------------------------------------------------
        # AXES
        # ----------------------------------------------------

        ax.set_xlim(
            x_min,
            x_max
        )

        ax.set_ylim(
            1.45,
            5.55
        )

        ax.set_yticks(
            warming_levels
        )

        ax.set_yticklabels([
            "2.0°C",
            "3.5°C",
            "5.0°C"
        ])

        ax.set_ylabel(
            "Warming level",
            fontsize=11,
            fontweight="bold"
        )

        ax.tick_params(
            axis="both",
            direction="out",
            labelsize=10,
            length=4
        )

        ax.grid(
            axis="both",
            which="major",
            linestyle="--",
            linewidth=0.5,
            color="0.88",
            alpha=0.8,
            zorder=0
        )

        # Year above the upper-right corner
        ax.text(
            0.98,
            1.02,
            str(year),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="bottom",
            clip_on=False
        )
        
        # Panel label
        ax.text(
            0.01,
            0.98,
            panel_labels[year],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # One shared x-axis label is sufficient because the
    # subplots use the same x-axis scale.
    axes[-1].set_xlabel(
        f"{ice_sheet} contribution to SLR (cm)",
        fontsize=11,
        fontweight="bold"
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.08,
        top=0.97,
        hspace=0.35
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

    print(f"\nSaved {ice_sheet} figure:")
    print(figure_png)
    print(figure_pdf)