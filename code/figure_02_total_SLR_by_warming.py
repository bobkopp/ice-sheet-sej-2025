#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:17:44 2026

Figure 3. Total ice-sheet contribution to sea-level rise as a function of
warming level in 2100, 2200, and 2300.

Each probability bar shows:
    Thin line: 5-95th percentile range
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

figure_png = os.path.join(
    output_dir,
    "Figure03_total_SLR_by_warming.png"
)

figure_pdf = os.path.join(
    output_dir,
    "Figure03_total_SLR_by_warming.pdf"
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

panel_labels = {
    2100: "A",
    2200: "B",
    2300: "C"
}

percentile_levels = [
    5,
    17,
    50,
    83,
    95
]


# Colorblind-friendly colors, distinct from the palettes
# used for the individual ice sheets.
warming_colors = {
    2.0: "#332288",  # indigo
    3.5: "#6A3D9A",  # purple
    5.0: "#CC6677"   # muted red
}
scenario_colors = {
    "Low warming (2.0°C)": "#0072B2",
    "Medium warming (3.5°C)": "#6A3D9A",
    "High warming (5.0°C)": "#D55E00"
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
    """
    Format a rounded value using a Unicode minus sign.
    """

    if np.isnan(value):
        return ""

    return f"{value:.0f}".replace("-", "−")


def get_total_slr_samples(
    dataframe,
    year,
    warming
):
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
            f"{year}, {warming:.1f}°C."
        )

    return samples.to_numpy() / 10.0


def spread_positions(
    values,
    minimum_spacing
):
    """
    Shift annotation positions so adjacent labels are separated
    by at least a specified horizontal distance.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    positions = values.copy()

    for index in range(1, len(positions)):

        separation = (
            positions[index]
            - positions[index - 1]
        )

        if separation < minimum_spacing:
            positions[index] = (
                positions[index - 1]
                + minimum_spacing
            )

    # Recenter the adjusted positions around the original values
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
    vertical_offset=0.16,
    minimum_spacing_fraction=0.045
):
    """
    Add a labeled percentile bar at one warming level.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which the bar is drawn.

    samples : array-like
        Monte Carlo SLR samples in centimeters.

    y_position : float
        Warming level used as the vertical bar position.

    color : str
        Bar and annotation color.

    x_range : float
        Width of the panel's x-axis.

    vertical_offset : float
        Vertical distance between the bar and annotations,
        expressed in degrees Celsius on the y-axis.

    minimum_spacing_fraction : float
        Minimum annotation spacing as a fraction of the
        panel's x-axis range.
    """

    q05, q17, q50, q83, q95 = np.percentile(
        samples,
        percentile_levels
    )

    # --------------------------------------------------------
    # PERCENTILE BAR
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DYNAMIC LABEL POSITIONS
    # --------------------------------------------------------

    minimum_spacing = (
        minimum_spacing_fraction
        * x_range
    )

    # Put 5th, 50th, and 95th above the bar
    upper_values = np.array([
        q05,
        q50,
        q95
    ])

    upper_positions = spread_positions(
        upper_values,
        minimum_spacing
    )

    # Put 17th and 83rd below the bar
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

# ============================================================
# GLOBAL X-AXIS LIMITS
# ============================================================

all_percentiles = []

for year in years:
    for warming in warming_levels:

        samples = get_total_slr_samples(
            dataframe=mc_total_ext,
            year=year,
            warming=warming
        )

        q05, q95 = np.percentile(samples, [5, 95])
        all_percentiles.extend([q05, q95])

x_min = min(all_percentiles)
x_max = max(all_percentiles)

display_range = x_max - x_min

padding = 0.10 * display_range

x_min -= padding
x_max += padding

x_range = x_max - x_min
# ============================================================
# 4. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(10.5, 11.5),
    sharey=True
)

warming_labels = {
    2.0: "Low warming (2.0°C)",
    3.5: "Medium warming (3.5°C)",
    5.0: "High warming (5.0°C)"
}

for ax, year in zip(
    axes,
    years
):

    distributions = {
        warming: get_total_slr_samples(
            dataframe=mc_total_ext,
            year=year,
            warming=warming
        )
        for warming in warming_levels
    }

    

    for warming in warming_levels:

        scenario_name = warming_labels[warming]

        add_probability_bar(
            ax=ax,
            samples=distributions[warming],
            y_position=warming,
            color=scenario_colors[scenario_name],
            x_range=x_range,
            vertical_offset=0.16,
            minimum_spacing_fraction=0.045
        )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

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

    ax.set_xlabel(
        "Total ice-sheet contribution to SLR (cm)",
        fontsize=11,
        fontweight="bold"
    )

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
        linewidth=0.6,
        color="lightgray",
        alpha=0.8,
        zorder=0
    )

    # Panel label
    ax.text(
        0.01,
        0.98,
        panel_labels[year],
        transform=ax.transAxes,
        fontsize=14,
        ha="left",
        va="top"
    )

    # Year in upper-right corner
    ax.text(
        0.98,
        1.02,
        str(year),
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="top"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 5. LAYOUT AND SAVE
# ============================================================

fig.subplots_adjust(
    left=0.12,
    right=0.97,
    bottom=0.07,
    top=0.98,
    hspace=0.38
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



## Now each plot individually
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

    # --------------------------------------------------------
    # PERCENTILE BAR
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DYNAMIC LABEL POSITIONS
    # --------------------------------------------------------

    minimum_spacing = (
        minimum_spacing_fraction
        * x_range
    )

    # 5th, 50th, and 95th percentiles above the bar
    upper_values = np.array([
        q05,
        q50,
        q95
    ])

    upper_positions = spread_positions(
        upper_values,
        minimum_spacing
    )

    # 17th and 83rd percentiles below the bar
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
            fontsize=9,
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
            fontsize=9,
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
    percentile ranges rather than extreme Monte Carlo values.
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

    x_min = x_min_data - padding
    x_max = x_max_data + padding

    return x_min, x_max


# ============================================================
# 4. CREATE ONE FIGURE FOR EACH YEAR
# ============================================================

for year in years:

    figure_png = os.path.join(
        output_dir,
        f"Figure03_total_SLR_by_warming_{year}.png"
    )

    figure_pdf = os.path.join(
        output_dir,
        f"Figure03_total_SLR_by_warming_{year}.pdf"
    )

    # --------------------------------------------------------
    # EXTRACT DISTRIBUTIONS
    # --------------------------------------------------------

    distributions = {
        warming: get_total_slr_samples(
            dataframe=mc_total_ext,
            year=year,
            warming=warming
        )
        for warming in warming_levels
    }

    # Each year receives its own optimized x-axis.
    x_min, x_max = calculate_axis_limits(
        distributions=distributions,
        padding_fraction=0.12
    )

    x_range = x_max - x_min

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 3.8)
    )

    for warming in warming_levels:

        scenario_name = warming_labels[warming]

        add_probability_bar(
            ax=ax,
            samples=distributions[warming],
            y_position=warming,
            color=scenario_colors[scenario_name],
            x_range=x_range,
            vertical_offset=0.14,
            minimum_spacing_fraction=0.045
        )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

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

    ax.set_xlabel(
        "Total ice-sheet contribution to SLR (cm)",
        fontsize=11,
        fontweight="bold"
    )

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

    # Light dashed grid lines
    ax.grid(
        axis="both",
        which="major",
        linestyle="--",
        linewidth=0.5,
        color="0.88",
        alpha=0.8,
        zorder=0
    )

    # Year slightly above the upper-right corner
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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # LAYOUT AND SAVE
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.19,
        top=0.91
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

    print(f"\nSaved {year} figure:")
    print(figure_png)
    print(figure_pdf)
