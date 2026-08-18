#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:14:57 2026

Indicative probability distributions of process-level
ice-sheet contributions to sea-level rise.

Creates one figure for each year:
    2100
    2200
    2300

Within each process and ice sheet:
    Left half  = low warming (2.0°C)
    Right half = high warming (5.0°C)

Summary markers:
    Black horizontal line = median
    Red squares = 5th and 95th percentiles

@author: elizabeth
"""

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
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


years = [
    2100,
    2200,
    2300
]

ice_sheets = [
    "EAIS",
    "WAIS",
    "GrIS",
    "Total AIS"
]

processes = [
    "Discharge",
    "Accumulation",
    "Runoff",
    "Runoff minus accumulation"
]

process_labels = {
    "Discharge": "Discharge",
    "Accumulation": "Accumulation",
    "Runoff": "Runoff",
    "Runoff minus accumulation": "Runoff − Accumulation"
}

ice_sheet_labels = {
    "EAIS": "EAIS",
    "WAIS": "WAIS",
    "GrIS": "GrIS",
    "Total AIS": "AIS"
}


warming_scenarios = {
    "Low warming (2.0°C)": 2.0,
    "High warming (5.0°C)": 5.0
}


scenario_colors = {
    "Low warming (2.0°C)": "#0072B2",
    "High warming (5.0°C)": "#D55E00"
}


# Violin appearance
violin_half_width = 0.38
violin_alpha = 0.45
violin_edge_width = 0.9

# Horizontal spacing
within_group_spacing = 1.0
between_group_spacing = 1.0

# KDE settings
kde_points = 500
kde_bandwidth_multiplier = 1.0

# Summary-marker settings
median_half_width = 0.13
summary_marker_size = 3.5

# Limit violin tails
violin_lower_percentile = 0.5
violin_upper_percentile = 99.5


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

data = load_and_prepare_data()

process_ext = data["process_ext"]


# ============================================================
# 3. ADD TOTAL AIS PROCESS CONTRIBUTIONS
# ============================================================

def add_total_ais_processes(process_data):
    """
    Add Total AIS process contributions as the draw-by-draw
    sum of EAIS and WAIS.
    """

    ais_processes = (
        process_data
        .loc[
            process_data["ice_sheet"].isin([
                "EAIS",
                "WAIS"
            ])
        ]
        .pivot(
            index=[
                "sample_id",
                "process",
                "process_code",
                "warming",
                "warming_code",
                "year",
                "scenario_name"
            ],
            columns="ice_sheet",
            values="value"
        )
        .dropna(subset=["EAIS", "WAIS"])
        .reset_index()
    )

    ais_processes["value"] = (
        ais_processes["EAIS"]
        + ais_processes["WAIS"]
    )

    ais_processes["variable"] = pd.NA
    ais_processes["ice_sheet_code"] = "AIS"
    ais_processes["ice_sheet"] = "Total AIS"

    ais_processes = ais_processes[
        process_data.columns
    ]

    return pd.concat(
        [
            process_data,
            ais_processes
        ],
        ignore_index=True
    )


process_ext = add_total_ais_processes(
    process_ext
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_process_samples(
    dataframe,
    year,
    warming,
    process,
    ice_sheet
):
    """
    Extract process-level samples and convert mm to cm.
    """

    samples = dataframe.loc[
        (
            dataframe["year"] == year
        )
        & (
            dataframe["warming"] == warming
        )
        & (
            dataframe["process"] == process
        )
        & (
            dataframe["ice_sheet"] == ice_sheet
        ),
        "value"
    ].dropna()

    if samples.empty:
        raise ValueError(
            "No samples found for "
            f"{ice_sheet}, {process}, {year}, "
            f"{warming:.1f}°C."
        )

    return samples.to_numpy() / 10.0


def calculate_violin_density(
    samples,
    n_points=500,
    bandwidth_multiplier=1.0
):
    """
    Calculate KDE for a vertical half violin.
    """

    y_min, y_max = np.percentile(
        samples,
        [
            violin_lower_percentile,
            violin_upper_percentile
        ]
    )

    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    y_values = np.linspace(
        y_min,
        y_max,
        n_points
    )

    kde = gaussian_kde(samples)

    kde.set_bandwidth(
        bw_method=(
            kde.factor
            * bandwidth_multiplier
        )
    )

    density = kde(y_values)

    if density.max() > 0:
        density = density / density.max()

    return y_values, density


def draw_half_violin(
    ax,
    samples,
    center,
    side,
    color
):
    """
    Draw one half of a vertical violin.
    """

    y_values, density = calculate_violin_density(
        samples=samples,
        n_points=kde_points,
        bandwidth_multiplier=kde_bandwidth_multiplier
    )

    widths = (
        density
        * violin_half_width
    )

    if side == "left":

        x_outer = center - widths

        ax.fill_betweenx(
            y_values,
            center,
            x_outer,
            facecolor=color,
            edgecolor=color,
            linewidth=violin_edge_width,
            alpha=violin_alpha,
            zorder=2
        )

    elif side == "right":

        x_outer = center + widths

        ax.fill_betweenx(
            y_values,
            center,
            x_outer,
            facecolor=color,
            edgecolor=color,
            linewidth=violin_edge_width,
            alpha=violin_alpha,
            zorder=2
        )

    else:
        raise ValueError(
            "side must be either 'left' or 'right'."
        )

    ax.vlines(
        center,
        y_values.min(),
        y_values.max(),
        color="0.45",
        linewidth=0.5,
        alpha=0.6,
        zorder=1
    )


def add_summary_markers(
    ax,
    samples,
    center,
    side
):
    """
    Add median and 5th/95th percentile markers.
    """

    q05, q50, q95 = np.percentile(
        samples,
        [5, 50, 95]
    )

    if side == "left":

        median_xmin = (
            center
            - median_half_width
        )

        median_xmax = center

        marker_x = (
            center
            - median_half_width * 0.55
        )

    elif side == "right":

        median_xmin = center

        median_xmax = (
            center
            + median_half_width
        )

        marker_x = (
            center
            + median_half_width * 0.55
        )

    else:
        raise ValueError(
            "side must be either 'left' or 'right'."
        )

    # Median
    ax.hlines(
        y=q50,
        xmin=median_xmin,
        xmax=median_xmax,
        color="black",
        linewidth=2.0,
        zorder=6
    )

    # 5th and 95th percentiles
    ax.plot(
        [marker_x, marker_x],
        [q05, q95],
        linestyle="none",
        marker="s",
        markersize=summary_marker_size,
        markerfacecolor="#E41A1C",
        markeredgecolor="#E41A1C",
        zorder=7
    )


def create_x_positions():
    """
    Create x positions for each process/ice-sheet combination.
    """

    positions = {}
    group_centers = {}

    current_position = 0.0

    for process in processes:

        group_positions = []

        for ice_sheet in ice_sheets:

            positions[
                (
                    process,
                    ice_sheet
                )
            ] = current_position

            group_positions.append(
                current_position
            )

            current_position += (
                within_group_spacing
            )

        group_centers[process] = np.mean(
            group_positions
        )

        current_position += (
            between_group_spacing
        )

    return positions, group_centers


def calculate_year_y_limits(
    dataframe,
    year,
    padding_fraction=0.08
):
    """
    Calculate one shared y-axis range for all distributions
    shown in a given year figure.
    """

    percentile_bounds = []

    for process in processes:
        for ice_sheet in ice_sheets:
            for warming in warming_scenarios.values():

                samples = get_process_samples(
                    dataframe=dataframe,
                    year=year,
                    warming=warming,
                    process=process,
                    ice_sheet=ice_sheet
                )

                percentile_bounds.append(
                    np.percentile(
                        samples,
                        [0.5, 99.5]
                    )
                )

    percentile_bounds = np.asarray(
        percentile_bounds
    )

    y_min_data = (
        percentile_bounds[:, 0].min()
    )

    y_max_data = (
        percentile_bounds[:, 1].max()
    )

    y_range = (
        y_max_data
        - y_min_data
    )

    if y_range == 0:
        y_range = 1.0

    padding = (
        padding_fraction
        * y_range
    )

    return (
        y_min_data - padding,
        y_max_data + padding
    )


# ============================================================
# 5. CREATE FIGURES
# ============================================================

x_positions, group_centers = (
    create_x_positions()
)


for year in years:

    figure_png = os.path.join(
        output_dir,
        f"Figure05_process_distributions_{year}.png"
    )


    fig, ax = plt.subplots(
        figsize=(14.0, 6.5)
    )

    # --------------------------------------------------------
    # DRAW DISTRIBUTIONS
    # --------------------------------------------------------

    for process in processes:

        for ice_sheet in ice_sheets:

            center = x_positions[
                (
                    process,
                    ice_sheet
                )
            ]

            low_samples = get_process_samples(
                dataframe=process_ext,
                year=year,
                warming=2.0,
                process=process,
                ice_sheet=ice_sheet
            )

            high_samples = get_process_samples(
                dataframe=process_ext,
                year=year,
                warming=5.0,
                process=process,
                ice_sheet=ice_sheet
            )

            # Low warming: left half
            draw_half_violin(
                ax=ax,
                samples=low_samples,
                center=center,
                side="left",
                color=scenario_colors[
                    "Low warming (2.0°C)"
                ]
            )

            add_summary_markers(
                ax=ax,
                samples=low_samples,
                center=center,
                side="left"
            )

            # High warming: right half
            draw_half_violin(
                ax=ax,
                samples=high_samples,
                center=center,
                side="right",
                color=scenario_colors[
                    "High warming (5.0°C)"
                ]
            )

            add_summary_markers(
                ax=ax,
                samples=high_samples,
                center=center,
                side="right"
            )

    # --------------------------------------------------------
    # ICE-SHEET LABELS
    # --------------------------------------------------------

    tick_positions = []
    tick_labels = []

    for process in processes:
        for ice_sheet in ice_sheets:

            tick_positions.append(
                x_positions[
                    (
                        process,
                        ice_sheet
                    )
                ]
            )

            tick_labels.append(
                ice_sheet_labels[
                    ice_sheet
                ]
            )

    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        tick_labels,
        rotation=90,
        fontsize=9
    )

    # --------------------------------------------------------
    # PROCESS GROUP LABELS
    # --------------------------------------------------------

    for process in processes:

        ax.text(
            group_centers[process],
            -0.16,
            process_labels[process],
            transform=ax.get_xaxis_transform(),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
            clip_on=False
        )

    # --------------------------------------------------------
    # PROCESS GROUP SEPARATORS
    # --------------------------------------------------------

    for process_index in range(
        len(processes) - 1
    ):

        left_process = processes[
            process_index
        ]

        right_process = processes[
            process_index + 1
        ]

        separator = (
            x_positions[
                (
                    left_process,
                    ice_sheets[-1]
                )
            ]
            + x_positions[
                (
                    right_process,
                    ice_sheets[0]
                )
            ]
        ) / 2

        ax.axvline(
            separator,
            color="0.80",
            linewidth=0.8,
            linestyle="-",
            zorder=0
        )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    y_min, y_max = calculate_year_y_limits(
        dataframe=process_ext,
        year=year,
        padding_fraction=0.08
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    final_position = max(
        x_positions.values()
    )

    ax.set_xlim(
        -0.75,
        final_position + 0.75
    )

    ax.set_ylabel(
        "SLR contribution (cm)",
        fontsize=12,
        fontweight="bold"
    )

    ax.tick_params(
        axis="y",
        direction="out",
        labelsize=10,
        length=4
    )

    ax.tick_params(
        axis="x",
        length=0,
        pad=4
    )

    ax.grid(
        axis="y",
        which="major",
        linestyle="--",
        linewidth=0.5,
        color="0.85",
        alpha=0.8,
        zorder=0
    )

    ax.axhline(
        0,
        color="0.35",
        linewidth=0.8,
        zorder=1
    )

    # Year label
    ax.text(
        0.02,
        0.97,
        str(year),
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend_handles = [
        Patch(
            facecolor=scenario_colors[
                "Low warming (2.0°C)"
            ],
            edgecolor=scenario_colors[
                "Low warming (2.0°C)"
            ],
            alpha=violin_alpha,
            label="Low warming (2.0°C)"
        ),
        Patch(
            facecolor=scenario_colors[
                "High warming (5.0°C)"
            ],
            edgecolor=scenario_colors[
                "High warming (5.0°C)"
            ],
            alpha=violin_alpha,
            label="High warming (5.0°C)"
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            label="Median"
        ),
        Line2D(
            [0],
            [0],
            color="#E41A1C",
            marker="s",
            linestyle="none",
            markersize=5,
            label="5th and 95th percentiles"
        )
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=9,
        ncol=2
    )

    # --------------------------------------------------------
    # LAYOUT AND SAVE
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.24,
        top=0.96
    )

    fig.savefig(
        figure_png,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()
    plt.close(fig)

    print(f"\nSaved {year} process figure:")
    print(figure_png)