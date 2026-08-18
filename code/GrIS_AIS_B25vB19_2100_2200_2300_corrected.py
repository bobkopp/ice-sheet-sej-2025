#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of B19 and B25 Greenland Ice Sheet (GrIS) and total
Antarctic Ice Sheet (AIS) contributions to sea-level rise.

Three figures are produced:
    1. 2100: low warming (2°C) and high warming (5°C)
    2. 2200: low warming (2°C) and high warming (5°C)
    3. 2300: low warming (2°C) and high warming (5°C)

Each panel contains:
    - B25 KDE curves for GrIS and total AIS
    - B19 and B25 percentile bars for GrIS
    - B19 and B25 percentile bars for total AIS

B19 is shown with dashed bars and open median markers.
B25 is shown with solid bars and filled median markers.

The corrected total-AIS B19 percentile values supplied by the user are
hard-coded below. Greenland B19 percentiles for 2200 and 2300 are read
from data/B19_data.xlsx because those values were not supplied here.
The 2100 Greenland values are retained from the original script.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from B25data_processing import load_and_prepare_data, project_dir


# ============================================================
# 1. SETTINGS
# ============================================================

output_dir = os.path.join(project_dir, "output", "figures")
os.makedirs(output_dir, exist_ok=True)

b19_file = os.path.join(project_dir, "data", "B19_data.xlsx")

years = [2100, 2200, 2300]

warming_scenarios = {
    "Low warming": 2.0,
    "High warming": 5.0,
}

panel_labels = {
    "Low warming": "A",
    "High warming": "B",
}

ice_sheets = {
    "GrIS": "GrIS",
    "AIS": "Total AIS",
}

ice_sheet_colors = {
    "GrIS": "#009E73",
    "AIS": "#CC79A7",
}

# Corrected B19 total-AIS percentiles in centimeters.
# Each entry contains the percentile levels and corresponding values.
b19_ais = {
    2100: {
        "Low warming": {
            "levels": [5, 17, 50, 83, 95],
            "values": [-8.0, -1.0, 9.0, 26.0, 53.0],
        },
        "High warming": {
            "levels": [5, 17, 50, 83, 95],
            "values": [-11.0, 2.0, 21.0, 57.0, 132.0],
        },
    },
    2200: {
        "Low warming": {
            "levels": [5, 17, 50, 83, 95],
            "values": [-32, -8, 25, 80, 143],
        },
        "High warming": {
            "levels": [5, 17, 50, 83, 95],
            "values": [-45, -3, 53, 165, 641],
        },
    },
    2300: {
        "Low warming": {
            "levels": [5, 17, 50, 83, 95],
            "values": [-62, -16, 43, 153, 292],
        },
        "High warming": {
            "levels": [5, 17, 50, 83, 95],
            "values": [-65, -4, 90, 295, 782],
        },
    },
}

# Original 2100 B19 Greenland values, in centimeters.
b19_gris_2100 = {
    "Low warming": {
        "levels": [5, 17, 50, 83, 95],
        "values": [2.0, 7.0, 13.0, 31.0, 57.0],
    },
    "High warming": {
        "levels": [5, 17, 50, 83, 95],
        "values": [2.0, 10.0, 23.0, 60.0, 99.0],
    },
}

# Year-specific x-axis limits in centimeters.
x_limits = {
    2100: (-100, 175),
    2200: (-200, 700),
    2300: (-225, 850),
}

kde_bandwidth = 1.0
annotate_percentiles = True


# ============================================================
# 2. LOAD DATA
# ============================================================

data = load_and_prepare_data()
mc_total_ext = data["mc_total_ext"]

# Greenland B19 values for 2200 and 2300 are expected in this file.
if not os.path.exists(b19_file):
    raise FileNotFoundError(
        "Greenland B19 values for 2200 and 2300 were not supplied, and "
        f"the expected spreadsheet was not found: {b19_file}"
    )

b19_spreadsheet = pd.read_excel(b19_file)

def normalize_b19_column(column):
    """Normalize Excel percentage headers and text headers."""

    # Excel may store percentage headers as decimal numbers.
    if isinstance(column, (int, float, np.integer, np.floating)):
        percentage_headers = {
            0.05: "5%",
            0.17: "17%",
            0.50: "Median (50%)",
            0.83: "83%",
            0.95: "95%",
        }

        for numeric_value, normalized_name in percentage_headers.items():
            if np.isclose(float(column), numeric_value):
                return normalized_name

    # Normalize headers that pandas reads as text.
    column_text = str(column).strip()

    text_replacements = {
        "0.05": "5%",
        "0.17": "17%",
        "0.5": "Median (50%)",
        "0.50": "Median (50%)",
        "0.83": "83%",
        "0.95": "95%",
        "5.0%": "5%",
        "17.0%": "17%",
        "50.0%": "Median (50%)",
        "83.0%": "83%",
        "95.0%": "95%",
        "50%": "Median (50%)",
        "Median": "Median (50%)",
    }

    return text_replacements.get(column_text, column_text)


b19_spreadsheet.columns = [
    normalize_b19_column(column)
    for column in b19_spreadsheet.columns
]

print(
    "B19 spreadsheet columns:",
    b19_spreadsheet.columns.tolist()
)

required_columns = {
    "Year",
    "Ice Sheet",
    "Scenario",
    "5%",
    "17%",
    "Median (50%)",
    "83%",
    "95%",
}

missing_columns = required_columns - set(b19_spreadsheet.columns)
if missing_columns:
    raise ValueError(
        "B19_data.xlsx is missing these required columns: "
        + ", ".join(sorted(missing_columns))
    )

b19_spreadsheet["Year"] = pd.to_numeric(
    b19_spreadsheet["Year"], errors="raise"
).astype(int)

b19_spreadsheet["Ice Sheet"] = (
    b19_spreadsheet["Ice Sheet"].astype(str).str.strip()
)

b19_spreadsheet["Scenario"] = (
    b19_spreadsheet["Scenario"].astype(str).str.strip().str.lower()
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_ice_sheet_samples(dataframe, ice_sheet, year, warming):
    """Extract B25 samples for one ice sheet and convert mm to cm."""

    samples = dataframe.loc[
        (dataframe["ice_sheet"] == ice_sheet)
        & (dataframe["year"] == year)
        & (dataframe["warming"] == warming),
        "value",
    ].dropna()

    if samples.empty:
        raise ValueError(
            f"No B25 samples found for {ice_sheet}, {year}, "
            f"{warming:.1f}°C."
        )

    return samples.to_numpy() / 10.0


def get_b19_gris(year, scenario_name):
    """Return B19 Greenland percentiles for one year/scenario."""

    if year == 2100:
        return b19_gris_2100[scenario_name]

    scenario_lookup = {
        "Low warming": "low",
        "High warming": "high",
    }

    selection = b19_spreadsheet.loc[
        (b19_spreadsheet["Year"] == year)
        & (
            b19_spreadsheet["Ice Sheet"].str.lower()
            .isin(["gris", "greenland", "greenland ice sheet"])
        )
        & (
            b19_spreadsheet["Scenario"]
            == scenario_lookup[scenario_name]
        )
    ]

    if selection.empty:
        raise ValueError(
            "No B19 Greenland row found in B19_data.xlsx for "
            f"{year}, {scenario_name}."
        )

    if len(selection) > 1:
        raise ValueError(
            "Multiple B19 Greenland rows found in B19_data.xlsx for "
            f"{year}, {scenario_name}."
        )

    row = selection.iloc[0]

    return {
        "levels": [5, 17, 50, 83, 95],
        "values": [
            float(row["5%"]),
            float(row["17%"]),
            float(row["Median (50%)"]),
            float(row["83%"]),
            float(row["95%"]),
        ],
    }


def calculate_kde(samples, x_values, bandwidth_multiplier=1.0):
    """Calculate a Gaussian kernel density estimate."""

    kde = gaussian_kde(samples)
    kde.set_bandwidth(bw_method=kde.factor * bandwidth_multiplier)
    return kde(x_values)


def fmt(value):
    """Format values to one decimal only when needed."""

    if np.isclose(value, round(value)):
        text = f"{value:.0f}"
    else:
        text = f"{value:.1f}"

    return text.replace("-", "−")


def spread_positions(values, minimum_spacing):
    """Shift annotation positions horizontally to reduce overlap."""

    values = np.asarray(values, dtype=float)
    positions = values.copy()

    for index in range(1, len(positions)):
        if positions[index] - positions[index - 1] < minimum_spacing:
            positions[index] = positions[index - 1] + minimum_spacing

    positions -= positions.mean() - values.mean()
    return positions


def add_percentile_annotations(
    ax,
    values,
    y_position,
    color,
    annotation_offset,
    x_range,
):
    """Annotate the five percentile values around one bar."""

    q1, q2, q3, q4, q5 = values
    minimum_spacing = 0.045 * x_range

    above_values = np.array([q1, q3, q5], dtype=float)
    below_values = np.array([q2, q4], dtype=float)

    above_positions = spread_positions(above_values, minimum_spacing)
    below_positions = spread_positions(below_values, minimum_spacing)

    for value, x_position, is_median in zip(
        above_values,
        above_positions,
        [False, True, False],
    ):
        ax.text(
            x_position,
            y_position + annotation_offset,
            fmt(value),
            color=color,
            fontsize=7.5,
            fontweight="bold" if is_median else "normal",
            ha="center",
            va="bottom",
            clip_on=False,
            zorder=20,
        )

    for value, x_position in zip(below_values, below_positions):
        ax.text(
            x_position,
            y_position - annotation_offset,
            fmt(value),
            color=color,
            fontsize=7.5,
            ha="center",
            va="top",
            clip_on=False,
            zorder=20,
        )


def add_percentile_bar(
    ax,
    values,
    y_position,
    color,
    study,
    annotation_offset,
    x_range,
):
    """Draw a B19 or B25 percentile bar."""

    q1, q2, q3, q4, q5 = np.asarray(values, dtype=float)

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
        raise ValueError("study must be either 'B19' or 'B25'.")

    ax.hlines(
        y=y_position,
        xmin=q1,
        xmax=q5,
        color=color,
        linewidth=thin_linewidth,
        linestyle=linestyle,
        zorder=4,
    )

    ax.hlines(
        y=y_position,
        xmin=q2,
        xmax=q4,
        color=color,
        linewidth=thick_linewidth,
        linestyle=linestyle,
        zorder=5,
    )

    ax.plot(
        q3,
        y_position,
        marker="o",
        markersize=4.5,
        markerfacecolor=median_facecolor,
        markeredgecolor=color,
        markeredgewidth=1.0,
        linestyle="none",
        zorder=7,
    )

    tick_height = annotation_offset * 0.24

    for percentile_value in [q1, q2, q3, q4, q5]:
        ax.vlines(
            x=percentile_value,
            ymin=y_position - tick_height,
            ymax=y_position + tick_height,
            color=color,
            linewidth=0.8,
            linestyle=linestyle,
            zorder=6,
        )

    if annotate_percentiles:
        add_percentile_annotations(
            ax=ax,
            values=[q1, q2, q3, q4, q5],
            y_position=y_position,
            color=color,
            annotation_offset=annotation_offset,
            x_range=x_range,
        )


# ============================================================
# 4. CREATE ONE TWO-PANEL FIGURE PER YEAR
# ============================================================

for year in years:

    x_min, x_max = x_limits[year]
    x_range = x_max - x_min
    x_values = np.linspace(x_min, x_max, 1200)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7.0, 9.0),
        sharex=True,
    )

    axes[0].tick_params(labelbottom=True)

    for ax, (scenario_name, warming) in zip(
        axes,
        warming_scenarios.items(),
    ):

        distributions = {}
        densities = {}

        for display_name, data_name in ice_sheets.items():
            samples = get_ice_sheet_samples(
                dataframe=mc_total_ext,
                ice_sheet=data_name,
                year=year,
                warming=warming,
            )

            density = calculate_kde(
                samples=samples,
                x_values=x_values,
                bandwidth_multiplier=kde_bandwidth,
            )

            distributions[display_name] = samples
            densities[display_name] = density

            ax.plot(
                x_values,
                density,
                color=ice_sheet_colors[display_name],
                linewidth=1.5,
                zorder=3,
            )

        max_density = max(
            density.max() for density in densities.values()
        )

        curve_gap = max_density * 0.18
        bar_gap = max_density * 0.19
        lowest_bar = max_density + curve_gap

        # Bottom to top: AIS B19, AIS B25, GrIS B19, GrIS B25.
        bar_positions = {
            "AIS": {
                "B19": lowest_bar,
                "B25": lowest_bar + bar_gap,
            },
            "GrIS": {
                "B19": lowest_bar + 2 * bar_gap,
                "B25": lowest_bar + 3 * bar_gap,
            },
        }

        annotation_offset = max_density * 0.025

        scenario_data = {
            "AIS": b19_ais[year][scenario_name],
            "GrIS": get_b19_gris(year, scenario_name),
        }

        for display_name in ["AIS", "GrIS"]:
            color = ice_sheet_colors[display_name]
            levels = scenario_data[display_name]["levels"]
            b19_values = scenario_data[display_name]["values"]
            b25_values = np.percentile(
                distributions[display_name],
                levels,
            )
            
            b25_values = np.round(b25_values, 0)

            add_percentile_bar(
                ax=ax,
                values=b19_values,
                y_position=bar_positions[display_name]["B19"],
                color=color,
                study="B19",
                annotation_offset=annotation_offset,
                x_range=x_range,
            )

            add_percentile_bar(
                ax=ax,
                values=b25_values,
                y_position=bar_positions[display_name]["B25"],
                color=color,
                study="B25",
                annotation_offset=annotation_offset,
                x_range=x_range,
            )

        label_x = x_min + 0.02 * x_range

        for display_name in ["AIS", "GrIS"]:
            color = ice_sheet_colors[display_name]

            for study in ["B19", "B25"]:
                ax.text(
                    label_x,
                    bar_positions[display_name][study],
                    f"{display_name} {study}",
                    color=color,
                    fontsize=8.5,
                    fontweight="bold" if study == "B25" else "normal",
                    ha="left",
                    va="center",
                    clip_on=True,
                    zorder=20,
                )

        highest_bar = bar_positions["GrIS"]["B25"]

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, highest_bar + max_density * 0.18)

        ax.set_ylabel(
            "Probability density, p[SLR]",
            fontsize=12,
            fontweight="bold",
        )

        ax.text(
            0.01,
            1.0,
            panel_labels[scenario_name],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )

        ax.text(
            0.98,
            0.985,
            str(year),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="top",
        )

        ax.text(
            0.98,
            0.93,
            f"{warming:g}°C",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="top",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            axis="both",
            direction="out",
            labelsize=12,
            length=4,
        )

        ax.grid(
            axis="x",
            color="0.93",
            linewidth=0.5,
            linestyle="-",
            zorder=0,
        )

    axes[0].set_xlabel("")
    axes[1].set_xlabel(
        "Ice-sheet contribution to SLR (cm)",
        fontsize=12,
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.09,
        top=0.98,
        hspace=0.16,
    )

    figure_png = os.path.join(
        output_dir,
        f"Figure_B19_B25_GrIS_AIS_PDFs_{year}_UPDATED.png",
    )


    fig.savefig(
        figure_png,
        dpi=300,
        bbox_inches="tight",
    )


    plt.show()
    plt.close(fig)

    print(f"\nSaved {year} B19–B25 GrIS/AIS comparison figure:")
    print(figure_png)

