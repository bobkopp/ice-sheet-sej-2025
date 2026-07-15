#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:10:53 2026

Figure 2. Probability density functions of individual ice-sheet contributions
to sea-level rise in 2100 and 2300.

The current version shows B25 Monte Carlo projections for:
    EAIS
    WAIS
    Total AIS
    GrIS
    
B19, AR6, and ISMIP6 comparisons to be added later once data become available.

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


years = [
    2100,
    2200,
    2300
]

warming_scenarios = {
    "Low warming (2.0°C)": 2.0,
    "High warming (5.0°C)": 5.0
}

ice_sheets = [
    "EAIS",
    "WAIS",
    "Total AIS",
    "GrIS"
]


# Colorblind-friendly palette that kind of matches B19
ice_sheet_colors = {
    "EAIS": "#0072B2",      # blue
    "WAIS": "#E69F00",      # orange
    "Total AIS": "#CC79A7", # reddish purple / pink
    "GrIS": "#009E73"       # bluish green
}

panel_labels = [
    "A",
    "B"
]


# Multiplier applied to scipy's default Scott bandwidth.
# Increase for smoother curves; decrease for more detail.
kde_bandwidth = 1.0

percentile_levels = [5, 17, 50, 83, 95]


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


def calculate_kde(
    samples,
    x_values,
    bandwidth_multiplier=1.0
):
    """
    Calculate a Gaussian kernel density estimate.
    """

    kde = gaussian_kde(samples)

    kde.set_bandwidth(
        bw_method=kde.factor * bandwidth_multiplier
    )

    return kde(x_values)


def calculate_figure_limits(
    dataframe,
    year,
    warming_values,
    ice_sheet_names,
    padding_fraction=0.08
):
    """
    Calculate one shared x-axis range for both panels
    within a figure.
    """

    distributions = []

    for warming in warming_values:
        for ice_sheet in ice_sheet_names:

            samples = get_ice_sheet_samples(
                dataframe=dataframe,
                ice_sheet=ice_sheet,
                year=year,
                warming=warming
            )

            distributions.append(samples)

    combined_samples = np.concatenate(distributions)

    x_min = combined_samples.min()
    x_max = combined_samples.max()

    data_range = x_max - x_min

    if data_range == 0:
        data_range = 1.0

    padding = padding_fraction * data_range

    return (
        x_min - padding,
        x_max + padding
    )

def fmt(value):
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
            clip_on=False,
            zorder=20
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
            clip_on=False,
            zorder = 20
        )

# ============================================================
# 4. CREATE ONE FIGURE FOR EACH YEAR
# ============================================================

for year in years:

    figure_png = os.path.join(
        output_dir,
        f"Figure02_individual_ice_sheet_PDFs_{year}.png"
    )

    figure_pdf = os.path.join(
        output_dir,
        f"Figure02_individual_ice_sheet_PDFs_{year}.pdf"
    )

    # Use the same x-axis range for the low- and high-warming
    # panels within each year.
    x_min, x_max = calculate_figure_limits(
        dataframe=mc_total_ext,
        year=year,
        warming_values=warming_scenarios.values(),
        ice_sheet_names=ice_sheets
    )

    x_range = x_max - x_min

    x_values = np.linspace(
        x_min,
        x_max,
        1200
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.5, 5.3),
        sharex=True
    )

    for ax, panel_label, (
        scenario_name,
        warming
    ) in zip(
        axes,
        panel_labels,
        warming_scenarios.items()
    ):

        # ----------------------------------------------------
        # EXTRACT AND PLOT DISTRIBUTIONS
        # ----------------------------------------------------

        distributions = {}
        densities = {}

        for ice_sheet in ice_sheets:

            samples = get_ice_sheet_samples(
                dataframe=mc_total_ext,
                ice_sheet=ice_sheet,
                year=year,
                warming=warming
            )

            density = calculate_kde(
                samples=samples,
                x_values=x_values,
                bandwidth_multiplier=kde_bandwidth
            )

            distributions[ice_sheet] = samples
            densities[ice_sheet] = density

            ax.plot(
                x_values,
                density,
                color=ice_sheet_colors[ice_sheet],
                linewidth=1.5,
                label=ice_sheet
            )

        max_density = max(
            density.max()
            for density in densities.values()
        )

        # ----------------------------------------------------
        # PERCENTILE BARS
        # ----------------------------------------------------

        # Highest bar corresponds to the first ice sheet in
        # ice_sheets; bars descend evenly toward the curves.
        bar_positions = np.linspace(
            max_density * 1.70,
            max_density * 1.12,
            len(ice_sheets)
        )

        # Keep percentile annotations close to each bar.
        annotation_offset = max_density * 0.025

        for ice_sheet, y_position in zip(
            ice_sheets,
            bar_positions
        ):

            samples = distributions[ice_sheet]

            add_percentile_bar(
                ax=ax,
                samples=samples,
                y_position=y_position,
                color=ice_sheet_colors[ice_sheet],
                annotation_offset=annotation_offset,
                x_range=x_range,
                min_spacing_fraction=0.045
            )

            # Use the 5th percentile to position the ice-sheet
            # label immediately to the left of its bar.
            q05 = np.percentile(
                samples,
                5
            )

            label_x = max(
                q05 - 0.025 * x_range,
                x_min + 0.01 * x_range
            )

            ax.text(
                label_x,
                y_position,
                ice_sheet,
                color=ice_sheet_colors[ice_sheet],
                fontsize=8.5,
                fontweight="bold",
                ha="right",
                va="center",
                clip_on=False
            )

        # ----------------------------------------------------
        # AXES AND LABELS
        # ----------------------------------------------------

        ax.set_xlim(
            x_min,
            x_max
        )

        # Provide enough space for the top percentile bar and
        # its annotation labels.
        ax.set_ylim(
            0,
            max_density * 1.9
            )

        ax.set_xlabel(
            "Ice-sheet contribution to SLR (cm)",
            fontsize=11,
            fontweight="bold"
        )

        ax.set_ylabel(
            "Probability density, p[SLR]",
            fontsize=11,
            fontweight="bold"
        )

        # Panel label
        ax.text(
            0.01,
            0.98,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            ha="left",
            va="top"
        )

        # Warming scenario
        ax.text(
            0.98,
            0.98,
            scenario_name,
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
            labelsize=10,
            length=4
        )

    # Only show the y-axis title on the left panel
    axes[1].set_ylabel("")

    # Year as a figure-level title
    fig.suptitle(
        str(year),
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    # --------------------------------------------------------
    # LEGEND AND LAYOUT
    # --------------------------------------------------------

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        fontsize=10,
        handlelength=3
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.18,
        top=0.88,
        wspace=0.22
    )

    # --------------------------------------------------------
    # SAVE FIGURE
    # --------------------------------------------------------

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