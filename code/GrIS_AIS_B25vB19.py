#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of B19 and B25 individual ice-sheet contributions
to sea-level rise in 2100.

The figure contains two vertically stacked panels:
    A. Low warming (2.0°C)
    B. High warming (5.0°C)

Each panel shows B25 probability density functions for:
    GrIS
    Total AIS

Above the B25 probability density functions, percentile bars
compare B19 and B25 projections.

Each percentile bar shows:
    Thin line: 5th–95th percentile range
    Thick line: 17th–83rd percentile range
    Circle: 50th percentile (median)

B25:
    Solid percentile bars

B19:
    Dashed percentile bars

Because the underlying B19 Monte Carlo samples are unavailable,
only B19 percentile bars are shown; no B19 probability density
functions are plotted.

@author: elizabeth
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
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

os.makedirs(
    output_dir,
    exist_ok=True
)


figure_png = os.path.join(
    output_dir,
    "Figure02_B19_B25_GrIS_AIS_PDFs_2100.png"
)



year = 2100


warming_scenarios = {
    "Low warming (2.0°C)": 2.0,
    "High warming (5.0°C)": 5.0
}


# The dictionary keys are the names displayed in the figure.
# The values are the exact ice-sheet names in mc_total_ext.
ice_sheets = {
    "GrIS": "GrIS",
    "AIS": "Total AIS"
}


# Colorblind-friendly colors matching the earlier figure.
ice_sheet_colors = {
    "GrIS": "#009E73",
    "AIS": "#CC79A7"
}


panel_labels = {
    "Low warming (2.0°C)": "A",
    "High warming (5.0°C)": "B"
}


percentile_levels = [
    5,
    17,
    50,
    83,
    95
]


# B19 percentile values in centimeters:
# [5th, 17th, 50th, 83rd, 95th]
b19_percentiles = {

    "Low warming (2.0°C)": {

        "GrIS": [
            2,
            7,
            13,
            31,
            57
        ],

        "AIS": [
            -8,
            -1,
            9,
            26,
            53
        ]
    },

    "High warming (5.0°C)": {

        "GrIS": [
            2,
            10,
            23,
            60,
            99
        ],

        "AIS": [
            -11,
            2,
            21,
            57,
            132
        ]
    }
}


# Multiplier applied to scipy's default Scott bandwidth.
# Increase for smoother KDE curves; decrease for more detail.
kde_bandwidth = 1.0


# Fixed 2100 x-axis limits.
x_min = -100
x_max = 175

x_range = (
    x_max
    - x_min
)

x_values = np.linspace(
    x_min,
    x_max,
    1200
)


# Set to False to suppress the numerical percentile labels.
annotate_percentiles = True


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
    Extract B25 samples for one ice sheet and convert values
    from millimeters to centimeters.
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

    return (
        samples.to_numpy()
        / 10.0
    )


def calculate_kde(
    samples,
    x_values,
    bandwidth_multiplier=1.0
):
    """
    Calculate a Gaussian kernel density estimate.
    """

    kde = gaussian_kde(
        samples
    )

    kde.set_bandwidth(
        bw_method=(
            kde.factor
            * bandwidth_multiplier
        )
    )

    return kde(
        x_values
    )


def fmt(value):
    """
    Format percentile values as whole numbers and use a
    typographic minus sign.
    """

    return f"{value:.0f}".replace(
        "-",
        "−"
    )


def spread_positions(
    values,
    minimum_spacing
):
    """
    Shift annotation positions horizontally so adjacent
    labels maintain a minimum separation.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    positions = values.copy()

    # Forward pass.
    for index in range(
        1,
        len(positions)
    ):

        if (
            positions[index]
            - positions[index - 1]
            < minimum_spacing
        ):

            positions[index] = (
                positions[index - 1]
                + minimum_spacing
            )

    # Recenter the adjusted labels around their original mean.
    positions -= (
        positions.mean()
        - values.mean()
    )

    return positions


def add_percentile_annotations(
    ax,
    percentiles,
    y_position,
    color,
    annotation_offset,
    x_range
):
    """
    Annotate the five percentile values.

    The 5th, 50th, and 95th percentiles are shown above
    the bar. The 17th and 83rd percentiles are shown below.
    """

    q05, q17, q50, q83, q95 = percentiles

    minimum_spacing = (
        0.045
        * x_range
    )

    above_values = np.array(
        [
            q05,
            q50,
            q95
        ],
        dtype=float
    )

    below_values = np.array(
        [
            q17,
            q83
        ],
        dtype=float
    )

    above_positions = spread_positions(
        values=above_values,
        minimum_spacing=minimum_spacing
    )

    below_positions = spread_positions(
        values=below_values,
        minimum_spacing=minimum_spacing
    )

    # --------------------------------------------------------
    # LABELS ABOVE: 5TH, 50TH, AND 95TH
    # --------------------------------------------------------

    for value, x_position, is_median in zip(
        above_values,
        above_positions,
        [
            False,
            True,
            False
        ]
    ):

        ax.text(
            x_position,
            y_position + annotation_offset,
            fmt(value),
            color=color,
            fontsize=7.5,
            fontweight=(
                "bold"
                if is_median
                else "normal"
            ),
            ha="center",
            va="bottom",
            clip_on=False,
            zorder=20
        )

    # --------------------------------------------------------
    # LABELS BELOW: 17TH AND 83RD
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
            fontsize=7.5,
            ha="center",
            va="top",
            clip_on=False,
            zorder=20
        )


def add_percentile_bar(
    ax,
    percentiles,
    y_position,
    color,
    study,
    annotation_offset,
    x_range
):
    """
    Draw one B19 or B25 percentile bar.

    B19:
        Dashed ranges and open median marker.

    B25:
        Solid ranges and filled median marker.
    """

    q05, q17, q50, q83, q95 = np.asarray(
        percentiles,
        dtype=float
    )

    if study == "B19":

        linestyle = "--"
        median_facecolor = "white"
        thin_linewidth = 1.2
        thick_linewidth = 3.0

    elif study == "B25":

        linestyle = "-"
        median_facecolor = color
        thin_linewidth = 1.3
        thick_linewidth = 3.5

    else:

        raise ValueError(
            "study must be either 'B19' or 'B25'."
        )

    # --------------------------------------------------------
    # THIN 5TH–95TH PERCENTILE RANGE
    # --------------------------------------------------------

    ax.hlines(
        y=y_position,
        xmin=q05,
        xmax=q95,
        color=color,
        linewidth=thin_linewidth,
        linestyle=linestyle,
        zorder=4
    )

    # --------------------------------------------------------
    # THICK 17TH–83RD PERCENTILE RANGE
    # --------------------------------------------------------

    ax.hlines(
        y=y_position,
        xmin=q17,
        xmax=q83,
        color=color,
        linewidth=thick_linewidth,
        linestyle=linestyle,
        zorder=5
    )

    # --------------------------------------------------------
    # MEDIAN MARKER
    # --------------------------------------------------------

    ax.plot(
        q50,
        y_position,
        marker="o",
        markersize=4.5,
        markerfacecolor=median_facecolor,
        markeredgecolor=color,
        markeredgewidth=1.0,
        linestyle="none",
        zorder=7
    )

    # --------------------------------------------------------
    # PERCENTILE TICKS
    # --------------------------------------------------------

    tick_height = (
        annotation_offset
        * 0.24
    )

    for percentile in [
        q05,
        q17,
        q50,
        q83,
        q95
    ]:

        ax.vlines(
            x=percentile,
            ymin=(
                y_position
                - tick_height
            ),
            ymax=(
                y_position
                + tick_height
            ),
            color=color,
            linewidth=0.8,
            linestyle=linestyle,
            zorder=6
        )

    # --------------------------------------------------------
    # OPTIONAL NUMERICAL ANNOTATIONS
    # --------------------------------------------------------

    if annotate_percentiles:

        add_percentile_annotations(
            ax=ax,
            percentiles=[
                q05,
                q17,
                q50,
                q83,
                q95
            ],
            y_position=y_position,
            color=color,
            annotation_offset=annotation_offset,
            x_range=x_range
        )


# ============================================================
# 4. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(7.0, 9.0),
    sharex=True
)

axes[0].tick_params(labelbottom=True)

for ax, (
    scenario_name,
    warming
) in zip(
    axes,
    warming_scenarios.items()
):

    # --------------------------------------------------------
    # EXTRACT B25 DISTRIBUTIONS AND PLOT KDE CURVES
    # --------------------------------------------------------

    distributions = {}
    densities = {}

    for display_name, data_name in ice_sheets.items():

        samples = get_ice_sheet_samples(
            dataframe=mc_total_ext,
            ice_sheet=data_name,
            year=year,
            warming=warming
        )

        density = calculate_kde(
            samples=samples,
            x_values=x_values,
            bandwidth_multiplier=kde_bandwidth
        )

        distributions[
            display_name
        ] = samples

        densities[
            display_name
        ] = density

        ax.plot(
            x_values,
            density,
            color=ice_sheet_colors[
                display_name
            ],
            linewidth=1.5,
            label=display_name,
            zorder=3
        )

    max_density = max(
        density.max()
        for density in densities.values()
    )

    # --------------------------------------------------------
    # PERCENTILE-BAR POSITIONS
    # --------------------------------------------------------

    # Leave a clear gap between the KDE curves and the lowest
    # percentile bar.
    curve_gap = (
        max_density
        * 0.18
    )

    # Separation between each B19/B25 percentile bar.
    bar_gap = (
        max_density
        * 0.19
    )

    lowest_bar = (
        max_density
        + curve_gap
    )

    # Bars are ordered from bottom to top:
    #
    # AIS B19
    # AIS B25
    # GrIS B19
    # GrIS B25
    #
    # This keeps the two studies paired within each ice sheet.
    bar_positions = {

        "AIS": {
            "B19": lowest_bar,
            "B25": lowest_bar + bar_gap
        },

        "GrIS": {
            "B19": lowest_bar + 2 * bar_gap,
            "B25": lowest_bar + 3 * bar_gap
        }
    }


    annotation_offset = (
        max_density
        * 0.025
    )

    # --------------------------------------------------------
    # DRAW B19 AND B25 PERCENTILE BARS
    # --------------------------------------------------------

    for display_name in [
        "AIS",
        "GrIS"
    ]:

        color = ice_sheet_colors[
            display_name
        ]

        # B19 values are supplied directly as percentiles.
        add_percentile_bar(
            ax=ax,
            percentiles=b19_percentiles[
                scenario_name
            ][
                display_name
            ],
            y_position=bar_positions[
                display_name
            ][
                "B19"
            ],
            color=color,
            study="B19",
            annotation_offset=annotation_offset,
            x_range=x_range
        )

        # B25 percentiles are calculated from the underlying
        # Monte Carlo distribution.
        b25_values = np.percentile(
            distributions[
                display_name
            ],
            percentile_levels
        )

        add_percentile_bar(
            ax=ax,
            percentiles=b25_values,
            y_position=bar_positions[
                display_name
            ][
                "B25"
            ],
            color=color,
            study="B25",
            annotation_offset=annotation_offset,
            x_range=x_range
        )

    # --------------------------------------------------------
    # ICE-SHEET AND STUDY LABELS
    # --------------------------------------------------------

    label_x = (
        x_min
        + 0.02 * x_range
    )

    for display_name in [
        "AIS",
        "GrIS"
    ]:

        color = ice_sheet_colors[
            display_name
        ]

        for study in [
            "B19",
            "B25"
        ]:

            ax.text(
                label_x,
                bar_positions[
                    display_name
                ][
                    study
                ],
                f"{display_name} {study}",
                color=color,
                fontsize=8.5,
                fontweight=(
                    "bold"
                    if study == "B25"
                    else "normal"
                ),
                ha="left",
                va="center",
                clip_on=True,
                zorder=20
            )

    # --------------------------------------------------------
    # AXES AND LABELS
    # --------------------------------------------------------

    ax.set_xlim(
        x_min,
        x_max
    )

    highest_bar = bar_positions[
        "GrIS"
    ][
        "B25"
    ]

    ax.set_ylim(
        0,
        highest_bar
        + max_density * 0.18
    )

    ax.set_ylabel(
        "Probability density, p[SLR]",
        fontsize=11,
        fontweight="bold"
    )

    # Panel label.
    ax.text(
        0.01,
        1.0,
        panel_labels[
            scenario_name
        ],
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top"
    )

    # Year
    ax.text(
        0.98,
        0.985,
        "2100",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="top"
    )
    
    # Warming scenario
    warming_label = f"{warming:g}°C"

    ax.text(
        0.98,
        0.93,
        warming_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="top"
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.tick_params(
        axis="both",
        direction="out",
        labelsize=10,
        length=4
    )
    
    ax.tick_params(labelsize=12)
    
    ax.grid(
        axis="x",
        color="0.93",
        linewidth=0.5,
        linestyle="-",
        zorder=0
    )

    ax.set_ylabel(
        "Probability density, p[SLR]",
        fontsize=12,
        fontweight="bold"
    )
    
    axes[1].set_xlabel(
        "Ice-sheet contribution to SLR (cm)",
        fontsize=12,
        fontweight="bold"
    )


# ============================================================
# 5. SHARED X-AXIS LABEL
# ============================================================

axes[0].set_xlabel(
    ""
)

axes[1].set_xlabel(
    "Ice-sheet contribution to SLR (cm)",
    fontsize=11,
    fontweight="bold"
)


# ============================================================
# 8. LAYOUT AND SAVE
# ============================================================

fig.subplots_adjust(
    left=0.13,
    right=0.98,
    bottom=0.13,
    top=0.96,
    hspace=0.16
)


fig.savefig(
    figure_png,
    dpi=300,
    bbox_inches="tight"
)


plt.show()
plt.close(
    fig
)


print(
    "\nSaved 2100 B19–B25 comparison figure:"
)

print(
    figure_png
)
