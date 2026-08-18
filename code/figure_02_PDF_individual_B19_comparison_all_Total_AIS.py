#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2. Probability density functions of individual ice-sheet contributions
for 2100, 2200, and 2300, with comparison between B25 and B19.

B25 Monte Carlo distributions are shown for:
    EAIS
    WAIS
    Total AIS
    GrIS

B19 percentile comparisons are shown for:
    EAIS
    WAIS
    Total AIS
    GrIS

All B19 comparisons, including Total AIS for 2100, 2200, and 2300, are read
directly from B19_data.xlsx.

@author: elizabeth
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

b19_file = os.path.join(
    project_dir,
    "data",
    "B19_data.xlsx"
)

years = [
    2100,
    2200,
    2300
]

warming_scenarios = {
    "2.0°C": {
        "warming": 2.0,
        "b19_scenario": "low"
    },
    "5.0°C": {
        "warming": 5.0,
        "b19_scenario": "high"
    }
}

ice_sheets = [
    "EAIS",
    "WAIS",
    "Total AIS",
    "GrIS"
]

# Individual ice sheets plotted after the Total AIS comparison.
b19_comparison_sheets = [
    "EAIS",
    "WAIS",
    "GrIS"
]

# Colorblind-friendly palette that approximately matches B19.
ice_sheet_colors = {
    "EAIS": "#0072B2",
    "WAIS": "#E69F00",
    "Total AIS": "#CC79A7",
    "GrIS": "#009E73"
}

panel_labels = [
    "A",
    "B"
]

percentile_levels = [
    5,
    17,
    50,
    83,
    95
]

# Multiplier applied to scipy's default Scott bandwidth.
# Increase for smoother curves; decrease for more detail.
kde_bandwidth = 1.0

# Set to False to suppress numerical percentile labels.
annotate_percentiles = True

# Year-specific x-axis limits in centimeters.
x_limits = {
    2100: (-100, 175),
    2200: (-200, 625),
    2300: (-250, 1000)
}


# ============================================================
# 2. LOAD B25 AND B19 DATA
# ============================================================

data = load_and_prepare_data()
mc_total_ext = data["mc_total_ext"]

if not os.path.exists(b19_file):
    raise FileNotFoundError(
        "B19 spreadsheet not found:\n"
        f"{b19_file}"
    )

b19_data = pd.read_excel(
    b19_file
)


# ============================================================
# 3. CLEAN AND VALIDATE B19 DATA
# ============================================================

def clean_excel_column_name(column):
    """
    Convert numeric Excel percentage headers such as 0.05 to
    readable names such as '5%'.
    """

    percentage_headers = {
        0.05: "5%",
        0.17: "17%",
        0.50: "Median (50%)",
        0.83: "83%",
        0.95: "95%"
    }

    if isinstance(
        column,
        (int, float, np.integer, np.floating)
    ):
        for numeric_value, label in percentage_headers.items():
            if np.isclose(
                float(column),
                numeric_value
            ):
                return label

    text = str(column).strip()

    text_map = {
        "0.05": "5%",
        "0.17": "17%",
        "0.5": "Median (50%)",
        "0.50": "Median (50%)",
        "0.83": "83%",
        "0.95": "95%"
    }

    return text_map.get(
        text,
        text
    )


b19_data.columns = [
    clean_excel_column_name(column)
    for column in b19_data.columns
]

required_b19_columns = {
    "Year",
    "Ice Sheet",
    "Scenario",
    "Median (50%)",
    "5%",
    "17%",
    "83%",
    "95%"
}

missing_b19_columns = (
    required_b19_columns
    - set(b19_data.columns)
)

if missing_b19_columns:
    raise ValueError(
        "The B19 spreadsheet is missing these columns: "
        + ", ".join(
            sorted(missing_b19_columns)
        )
        + "\nColumns read by pandas: "
        + str(b19_data.columns.tolist())
    )

b19_data["Year"] = pd.to_numeric(
    b19_data["Year"],
    errors="raise"
).astype(int)

b19_data["Ice Sheet"] = (
    b19_data["Ice Sheet"]
    .astype(str)
    .str.strip()
)

b19_data["Scenario"] = (
    b19_data["Scenario"]
    .astype(str)
    .str.strip()
    .str.lower()
)

for column in [
    "5%",
    "17%",
    "Median (50%)",
    "83%",
    "95%"
]:
    b19_data[column] = pd.to_numeric(
        b19_data[column],
        errors="raise"
    )


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_ice_sheet_samples(
    dataframe,
    ice_sheet,
    year,
    warming
):
    """
    Extract B25 samples for one ice sheet and convert mm to cm.
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
            "No B25 samples found for "
            f"{ice_sheet}, {year}, {warming:.1f}°C."
        )

    return samples.to_numpy() / 10.0


def get_b19_percentiles(
    dataframe,
    year,
    ice_sheet,
    scenario
):
    """
    Extract B19 percentiles in plotting order:
    5th, 17th, 50th, 83rd, and 95th.
    """

    selection = dataframe.loc[
        (
            dataframe["Year"] == year
        )
        & (
            dataframe["Ice Sheet"] == ice_sheet
        )
        & (
            dataframe["Scenario"] == scenario.lower()
        )
    ]

    if selection.empty:
        available_rows = (
            dataframe.loc[
                dataframe["Year"] == year,
                [
                    "Ice Sheet",
                    "Scenario"
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "Ice Sheet",
                    "Scenario"
                ]
            )
            .to_string(index=False)
        )

        raise ValueError(
            f"No B19 row found for {year}, "
            f"{ice_sheet}, {scenario}.\n\n"
            f"Available rows for {year}:\n"
            f"{available_rows}"
        )

    if len(selection) > 1:
        raise ValueError(
            f"Multiple B19 rows found for {year}, "
            f"{ice_sheet}, {scenario}."
        )

    row = selection.iloc[0]

    return np.array(
        [
            row["5%"],
            row["17%"],
            row["Median (50%)"],
            row["83%"],
            row["95%"]
        ],
        dtype=float
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
    Shift label positions horizontally so adjacent labels maintain
    a minimum separation.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    positions = values.copy()

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
    Annotate the 5th, 17th, 50th, 83rd, and 95th percentiles.
    """

    q05, q17, q50, q83, q95 = percentiles

    minimum_spacing = 0.045 * x_range

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
        Dashed percentile ranges and open median marker.

    B25:
        Solid percentile ranges and filled median marker.
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

    ax.hlines(
        y=y_position,
        xmin=q05,
        xmax=q95,
        color=color,
        linewidth=thin_linewidth,
        linestyle=linestyle,
        zorder=4
    )

    ax.hlines(
        y=y_position,
        xmin=q17,
        xmax=q83,
        color=color,
        linewidth=thick_linewidth,
        linestyle=linestyle,
        zorder=5
    )

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

    tick_height = annotation_offset * 0.24

    for percentile in [
        q05,
        q17,
        q50,
        q83,
        q95
    ]:
        ax.vlines(
            x=percentile,
            ymin=y_position - tick_height,
            ymax=y_position + tick_height,
            color=color,
            linewidth=0.8,
            linestyle=linestyle,
            zorder=6
        )

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
# 5. CREATE ONE FIGURE FOR EACH YEAR
# ============================================================

for year in years:

    if year not in x_limits:
        raise ValueError(
            f"No x-axis limits specified for {year}."
        )

    #x_min, x_max = x_limits[year]
    if year == 2100:
        x_min, x_max = -125, 175
    
    elif year == 2200:
        x_min, x_max = -225, 625

    elif year == 2300:
        x_min, x_max = -350, 1000
    
    x_range = x_max - x_min

    x_values = np.linspace(
        x_min,
        x_max,
        1200
    )

    figure_png = os.path.join(
        output_dir,
        f"Figure02_individual_ice_sheet_B19_B25_{year}_UPDATED.png"
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7.0, 10.0),
        sharex=True
    )

    for ax, panel_label, (
        scenario_name,
        scenario_settings
    ) in zip(
        axes,
        panel_labels,
        warming_scenarios.items()
    ):

        warming = scenario_settings["warming"]
        b19_scenario = scenario_settings["b19_scenario"]

        # ----------------------------------------------------
        # EXTRACT AND PLOT B25 DISTRIBUTIONS
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
                label=ice_sheet,
                zorder=3
            )

        max_density = max(
            density.max()
            for density in densities.values()
        )

        # ----------------------------------------------------
        # PERCENTILE-BAR POSITIONS
        # ----------------------------------------------------

        curve_gap = max_density * 0.16

        # Separation between the B19 and B25 bars within a pair.
        # This is intentionally larger than before to prevent their
        # percentile annotations from overlapping.
        comparison_gap = max_density * 0.19
        group_gap = max_density * 0.085

        current_y = max_density + curve_gap
        bar_positions = {}

        # From bottom to top:
        # Paired B19/B25 bars for Total AIS, EAIS, WAIS, and GrIS.
        bar_positions["Total AIS"] = {
            "B19": current_y,
            "B25": current_y + comparison_gap
        }
        current_y += 2 * comparison_gap + group_gap

        for ice_sheet in [
            "EAIS",
            "WAIS",
            "GrIS"
        ]:
            bar_positions[ice_sheet] = {
                "B19": current_y,
                "B25": current_y + comparison_gap
            }
            current_y += 2 * comparison_gap + group_gap

        annotation_offset = max_density * 0.022

        # ----------------------------------------------------
        # DRAW TOTAL AIS BAR(S)
        # ----------------------------------------------------

        total_ais_b19 = get_b19_percentiles(
            dataframe=b19_data,
            year=year,
            ice_sheet="Total AIS",
            scenario=b19_scenario
        )

        total_ais_b25 = np.round(
            np.percentile(
                distributions["Total AIS"],
                percentile_levels
            )
        )

        add_percentile_bar(
            ax=ax,
            percentiles=total_ais_b19,
            y_position=bar_positions["Total AIS"]["B19"],
            color=ice_sheet_colors["Total AIS"],
            study="B19",
            annotation_offset=annotation_offset,
            x_range=x_range
        )

        add_percentile_bar(
            ax=ax,
            percentiles=total_ais_b25,
            y_position=bar_positions["Total AIS"]["B25"],
            color=ice_sheet_colors["Total AIS"],
            study="B25",
            annotation_offset=annotation_offset,
            x_range=x_range
        )

        # ----------------------------------------------------
        # DRAW PAIRED B19 AND B25 BARS
        # ----------------------------------------------------

        for ice_sheet in b19_comparison_sheets:

            color = ice_sheet_colors[ice_sheet]

            b19_values = get_b19_percentiles(
                dataframe=b19_data,
                year=year,
                ice_sheet=ice_sheet,
                scenario=b19_scenario
            )

            b25_values = np.round(
                np.percentile(
                    distributions[ice_sheet],
                    percentile_levels
                )
            )

            add_percentile_bar(
                ax=ax,
                percentiles=b19_values,
                y_position=bar_positions[ice_sheet]["B19"],
                color=color,
                study="B19",
                annotation_offset=annotation_offset,
                x_range=x_range
            )

            add_percentile_bar(
                ax=ax,
                percentiles=b25_values,
                y_position=bar_positions[ice_sheet]["B25"],
                color=color,
                study="B25",
                annotation_offset=annotation_offset,
                x_range=x_range
            )

        # ----------------------------------------------------
        # ICE-SHEET AND STUDY LABELS
        # ----------------------------------------------------

        label_x = x_min + 0.02 * x_range

        total_ais_studies = ["B19", "B25"]

        for study in total_ais_studies:
            ax.text(
                label_x,
                bar_positions["Total AIS"][study],
                f"Total AIS {study}",
                color=ice_sheet_colors["Total AIS"],
                fontsize=8.2,
                fontweight="bold" if study == "B25" else "normal",
                ha="left",
                va="center",
                clip_on=True,
                zorder=20
            )

        for ice_sheet in b19_comparison_sheets:
            color = ice_sheet_colors[ice_sheet]

            for study in [
                "B19",
                "B25"
            ]:
                ax.text(
                    label_x,
                    bar_positions[ice_sheet][study],
                    f"{ice_sheet} {study}",
                    color=color,
                    fontsize=8.2,
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

        # ----------------------------------------------------
        # AXES AND LABELS
        # ----------------------------------------------------

        ax.set_xlim(
            x_min,
            x_max
        )

        highest_bar = bar_positions["GrIS"]["B25"]

        ax.set_ylim(
            0,
            highest_bar + max_density * 0.15
        )

        ax.set_ylabel(
            "Probability density, p[SLR]",
            fontsize=11,
            fontweight="bold"
        )

        ax.text(
            0.01,
            1.0,
            panel_label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top"
        )

        ax.text(
            0.98,
            0.98,
            str(year),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="top"
        )

        ax.text(
            0.98,
            0.92,
            scenario_name,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="top"
        )

        ax.grid(
            axis="x",
            color="0.93",
            linewidth=0.5,
            zorder=0
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            axis="both",
            direction="out",
            labelsize=10,
            length=4,
            labelbottom=True
        )

    # ========================================================
    # SHARED LABELS AND TITLE
    # ========================================================

    axes[0].set_xlabel("")

    axes[1].set_xlabel(
        "Ice-sheet contribution to SLR (cm)",
        fontsize=11,
        fontweight="bold"
    )

    # ========================================================
    # LAYOUT AND SAVE
    # ========================================================

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.08,
        top=0.98,
        hspace=0.14
    )

    fig.savefig(
        figure_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(
        f"\nSaved {year} B19-B25 comparison figure:"
    )
    print(figure_png)
