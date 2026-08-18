#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B19–B25 split-violin comparison of process-level SLR contributions.

Creates one figure per year (2100, 2200, 2300), each with two stacked
panels: low warming (2.0°C) and high warming (5.0°C).

Within each process/ice-sheet category:
    left half  = B19 Monte Carlo samples
    right half = B25 Monte Carlo samples

B19 is read from data/B19_processdata_MC.xlsx. Expected names include
PW01EA_2100L through PW01WR_2300H. B19 values are assumed to be cm;
B25 values are converted from mm to cm.
"""

import os
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from B25data_processing import load_and_prepare_data, project_dir


# ============================================================
# 1. SETTINGS
# ============================================================

output_dir = os.path.join(project_dir, "output", "figures")
os.makedirs(output_dir, exist_ok=True)

b19_file = os.path.join(project_dir, "data", "B19_processdata_MC.xlsx")

years = [2100, 2200, 2300]

scenario_settings = {
    "Low warming (2.0°C)": {"warming": 2.0, "suffix": "L", "panel": "A"},
    "High warming (5.0°C)": {"warming": 5.0, "suffix": "H", "panel": "B"},
}

ice_sheets = ["EAIS", "WAIS", "Total AIS", "GrIS"]

# processes = [
#     "Discharge",
#     "Accumulation",
#     "Runoff",
#     "Runoff minus accumulation",
# ]

processes = [
    "Discharge",
    "Runoff minus accumulation",
]

ice_sheet_labels = {
    "EAIS": "EAIS",
    "WAIS": "WAIS",
    "Total AIS": "AIS",
    "GrIS": "GrIS",
}

# process_labels = {
#     "Discharge": "Discharge",
#     "Accumulation": "Accumulation",
#     "Runoff": "Runoff",
#     "Runoff minus accumulation": "Runoff − Accumulation",
# }

process_labels = {
    "Discharge": "Discharge",
    "Runoff minus accumulation": "Runoff − Accumulation",
}

# Same colors as the PDF-comparison figures.
ice_sheet_colors = {
    "EAIS": "#0072B2",
    "WAIS": "#E69F00",
    "Total AIS": "#CC79A7",
    "GrIS": "#009E73",
}

violin_half_width = 0.38
b19_alpha = 0.30
b25_alpha = 0.65
violin_edge_width = 1.0
kde_points = 500
kde_bandwidth_multiplier = 1.0
violin_lower_percentile = 0.5
violin_upper_percentile = 99.5
median_half_width = 0.14
percentile_tick_half_width = 0.055
within_group_spacing = 1.0
between_group_spacing = 1.0


# ============================================================
# 2. LOAD AND PREPARE B25 DATA
# ============================================================

data = load_and_prepare_data()
process_ext = data["process_ext"].copy()


def add_total_ais_processes(process_data):
    """Add Total AIS as the draw-by-draw sum of EAIS and WAIS."""
    ais = (
        process_data.loc[process_data["ice_sheet"].isin(["EAIS", "WAIS"])]
        .pivot(
            index=[
                "sample_id", "process", "process_code", "warming",
                "warming_code", "year", "scenario_name",
            ],
            columns="ice_sheet",
            values="value",
        )
        .dropna(subset=["EAIS", "WAIS"])
        .reset_index()
    )

    ais["value"] = ais["EAIS"] + ais["WAIS"]
    ais["variable"] = pd.NA
    ais["ice_sheet_code"] = "AIS"
    ais["ice_sheet"] = "Total AIS"
    ais = ais[process_data.columns]

    return pd.concat([process_data, ais], ignore_index=True)


process_ext = add_total_ais_processes(process_ext)


# ============================================================
# 3. LOAD AND VALIDATE B19 DATA
# ============================================================

if not os.path.exists(b19_file):
    raise FileNotFoundError(f"B19 process workbook not found:\n{b19_file}")

b19_data = pd.read_excel(b19_file)
b19_data.columns = [str(c).strip() for c in b19_data.columns]

base_codes = ["EA", "ED", "GA", "GD", "GR", "WA", "WD", "WR"]
required_columns = {
    f"PW01{code}_{year}{suffix}"
    for year in years
    for suffix in ["L", "H"]
    for code in base_codes
}
missing = required_columns - set(b19_data.columns)

if missing:
    raise ValueError(
        "B19_processdata_MC.xlsx is missing these columns:\n"
        + "\n".join(sorted(missing))
        + "\n\nColumns read by pandas:\n"
        + str(b19_data.columns.tolist())
    )

for column in required_columns:
    b19_data[column] = pd.to_numeric(b19_data[column], errors="coerce")


# ============================================================
# 4. SAMPLE EXTRACTION
# ============================================================


def clean_samples(values, label):
    samples = (
        pd.Series(values)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    if samples.size == 0:
        raise ValueError(f"No finite samples found for {label}.")
    return samples


def get_b25_samples(year, warming, process, ice_sheet):
    selection = process_ext.loc[
        (process_ext["year"] == year)
        & np.isclose(process_ext["warming"], warming)
        & (process_ext["process"] == process)
        & (process_ext["ice_sheet"] == ice_sheet),
        "value",
    ]
    # B25 is stored in mm; plot in cm.
    return clean_samples(
        selection,
        f"B25 {ice_sheet}, {process}, {year}, {warming:.1f}°C",
    ) / 10.0


def b19_column(year, suffix, code):
    name = f"PW01{code}_{year}{suffix}"
    return clean_samples(b19_data[name], name)


def aligned_b19(year, suffix, codes):
    names = [f"PW01{code}_{year}{suffix}" for code in codes]
    aligned = (
        b19_data[names]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    if aligned.empty:
        raise ValueError("No aligned B19 samples for: " + ", ".join(names))
    return {
        code: aligned[name].to_numpy()
        for code, name in zip(codes, names)
    }


def get_b19_samples(year, suffix, process, ice_sheet):
    """Return B19 samples converted from mm to cm.

    None means no raw B19 quantity exists.
    """
    direct = {
        ("EAIS", "Accumulation"): "EA",
        ("EAIS", "Discharge"): "ED",
        ("WAIS", "Accumulation"): "WA",
        ("WAIS", "Discharge"): "WD",
        ("WAIS", "Runoff"): "WR",
        ("GrIS", "Accumulation"): "GA",
        ("GrIS", "Discharge"): "GD",
        ("GrIS", "Runoff"): "GR",
    }

    code = direct.get((ice_sheet, process))

    if code is not None:
        samples_mm = b19_column(year, suffix, code)

    # No PW01ER column exists; do not invent EAIS runoff.
    elif ice_sheet == "EAIS" and process == "Runoff":
        return None

    elif ice_sheet == "Total AIS" and process == "Accumulation":
        d = aligned_b19(year, suffix, ["EA", "WA"])
        samples_mm = d["EA"] + d["WA"]

    elif ice_sheet == "Total AIS" and process == "Discharge":
        d = aligned_b19(year, suffix, ["ED", "WD"])
        samples_mm = d["ED"] + d["WD"]

    elif ice_sheet == "Total AIS" and process == "Runoff":
        # With no EAIS runoff term, Antarctic runoff is WAIS runoff.
        samples_mm = b19_column(year, suffix, "WR")

    elif ice_sheet == "EAIS" and process == "Runoff minus accumulation":
        samples_mm = -b19_column(year, suffix, "EA")

    elif ice_sheet == "WAIS" and process == "Runoff minus accumulation":
        d = aligned_b19(year, suffix, ["WR", "WA"])
        samples_mm = d["WR"] - d["WA"]

    elif ice_sheet == "GrIS" and process == "Runoff minus accumulation":
        d = aligned_b19(year, suffix, ["GR", "GA"])
        samples_mm = d["GR"] - d["GA"]

    elif ice_sheet == "Total AIS" and process == "Runoff minus accumulation":
        d = aligned_b19(year, suffix, ["WR", "EA", "WA"])
        samples_mm = d["WR"] - d["EA"] - d["WA"]

    else:
        raise ValueError(
            f"Unsupported B19 combination: {ice_sheet}, {process}"
        )

    # Convert millimeters to centimeters.
    return samples_mm / 10.0


# ============================================================
# 5. VIOLIN FUNCTIONS
# ============================================================


def calculate_density(samples):
    samples = np.asarray(samples, dtype=float)
    y_min, y_max = np.percentile(
        samples, [violin_lower_percentile, violin_upper_percentile]
    )

    if np.unique(samples).size < 2 or np.isclose(np.ptp(samples), 0.0):
        center = float(np.median(samples))
        y = np.linspace(center - 0.25, center + 0.25, kde_points)
        density = np.exp(-0.5 * ((y - center) / 0.08) ** 2)
        return y, density / density.max()

    if np.isclose(y_min, y_max):
        padding = max(abs(y_min) * 0.01, 0.25)
        y_min -= padding
        y_max += padding

    y = np.linspace(y_min, y_max, kde_points)

    try:
        kde = gaussian_kde(samples)
        kde.set_bandwidth(kde.factor * kde_bandwidth_multiplier)
        density = kde(y)
    except np.linalg.LinAlgError:
        warnings.warn("Singular KDE; using a normal approximation.")
        center = float(np.median(samples))
        scale = max(float(np.std(samples)), 0.08)
        density = np.exp(-0.5 * ((y - center) / scale) ** 2)

    if density.max() > 0:
        density /= density.max()
    return y, density


def draw_half_violin(ax, samples, center, side, color, study):
    y, density = calculate_density(samples)
    widths = density * violin_half_width
    outer = center - widths if side == "left" else center + widths

    alpha = b19_alpha if study == "B19" else b25_alpha
    linestyle = "--" if study == "B19" else "-"

    collection = ax.fill_betweenx(
        y,
        center,
        outer,
        facecolor=color,
        edgecolor=color,
        linewidth=violin_edge_width,
        alpha=alpha,
        zorder=2,
    )
    collection.set_linestyle(linestyle)


def add_summary_markers(ax, samples, center, side, study):
    q05, q50, q95 = np.percentile(samples, [5, 50, 95])

    if side == "left":
        median_xmin, median_xmax = center - median_half_width, center
        marker_x = center - median_half_width * 0.62
    else:
        median_xmin, median_xmax = center, center + median_half_width
        marker_x = center + median_half_width * 0.62

    linestyle = "--" if study == "B19" else "-"
    facecolor = "white" if study == "B19" else "black"

    ax.hlines(
        q50,
        median_xmin,
        median_xmax,
        color="black",
        linewidth=2.0,
        linestyle=linestyle,
        zorder=6,
    )
    ax.hlines(
        [q05, q95],
        marker_x - percentile_tick_half_width,
        marker_x + percentile_tick_half_width,
        color="black",
        linewidth=1.1,
        linestyle=linestyle,
        zorder=7,
    )
    ax.plot(
        [marker_x, marker_x],
        [q05, q95],
        linestyle="none",
        marker="o",
        markersize=3.4,
        markerfacecolor=facecolor,
        markeredgecolor="black",
        markeredgewidth=0.9,
        zorder=8,
    )


# ============================================================
# 6. POSITIONS AND Y LIMITS
# ============================================================


def create_x_positions():
    positions = {}
    group_centers = {}
    x = 0.0

    for process in processes:
        group = []
        for ice_sheet in ice_sheets:
            positions[(process, ice_sheet)] = x
            group.append(x)
            x += within_group_spacing
        group_centers[process] = np.mean(group)
        x += between_group_spacing

    return positions, group_centers


x_positions, group_centers = create_x_positions()


# def year_y_limits(year, padding_fraction=0.08):
#     bounds = []

#     for scenario in scenario_settings.values():
#         for process in processes:
#             for ice_sheet in ice_sheets:
#                 b25 = get_b25_samples(
#                     year, scenario["warming"], process, ice_sheet
#                 )
#                 bounds.append(
#                     np.percentile(
#                         b25,
#                         [violin_lower_percentile, violin_upper_percentile],
#                     )
#                 )

#                 b19 = get_b19_samples(
#                     year, scenario["suffix"], process, ice_sheet
#                 )
#                 if b19 is not None:
#                     bounds.append(
#                         np.percentile(
#                             b19,
#                             [violin_lower_percentile, violin_upper_percentile],
#                         )
#                     )

#     bounds = np.asarray(bounds)
#     low, high = bounds[:, 0].min(), bounds[:, 1].max()
#     span = high - low if not np.isclose(high, low) else 1.0
#     padding = padding_fraction * span
#     return low - padding, high + padding

def panel_y_limits(
    year,
    warming,
    suffix,
    padding_fraction=0.08,
):
    """Calculate y-limits for one year and one warming panel."""
    bounds = []

    for process in processes:
        for ice_sheet in ice_sheets:

            b25 = get_b25_samples(
                year,
                warming,
                process,
                ice_sheet,
            )

            bounds.append(
                np.percentile(
                    b25,
                    [
                        violin_lower_percentile,
                        violin_upper_percentile,
                    ],
                )
            )

            b19 = get_b19_samples(
                year,
                suffix,
                process,
                ice_sheet,
            )

            if b19 is not None:
                bounds.append(
                    np.percentile(
                        b19,
                        [
                            violin_lower_percentile,
                            violin_upper_percentile,
                        ],
                    )
                )

    bounds = np.asarray(bounds)

    low = bounds[:, 0].min()
    high = bounds[:, 1].max()

    span = high - low

    if np.isclose(span, 0.0):
        span = 1.0

    padding = padding_fraction * span

    return (
        low - padding,
        high + padding,
    )


# ============================================================
# 7. CREATE FIGURES
# ============================================================


for year in years:
    # png_file = os.path.join(
    #     output_dir,
    #     f"Figure_process_B19_B25_split_violins_{year}.png",
    # )
    
    png_file = os.path.join(
        output_dir,
        f"Figure_process_B19_B25_split_violins_discharge_runoffminusacc_{year}.png",
    )

    fig, axes = plt.subplots(
        2,
        1,
        #figsize=(14.0, 11.0),
        figsize=(8.0, 11.0),
        sharex=True,
        sharey=False,
    )

    for ax, (scenario_name, scenario) in zip(
        axes,
        scenario_settings.items(),
    ):
        # Calculate limits separately for each warming panel.
        y_min, y_max = panel_y_limits(
            year=year,
            warming=scenario["warming"],
            suffix=scenario["suffix"],
        )

        for process in processes:
            for ice_sheet in ice_sheets:
                center = x_positions[(process, ice_sheet)]
                color = ice_sheet_colors[ice_sheet]

                b19 = get_b19_samples(
                    year,
                    scenario["suffix"],
                    process,
                    ice_sheet,
                )

                b25 = get_b25_samples(
                    year,
                    scenario["warming"],
                    process,
                    ice_sheet,
                )

                if b19 is not None:
                    draw_half_violin(
                        ax,
                        b19,
                        center,
                        "left",
                        color,
                        "B19",
                    )

                    add_summary_markers(
                        ax,
                        b19,
                        center,
                        "left",
                        "B19",
                    )

                draw_half_violin(
                    ax,
                    b25,
                    center,
                    "right",
                    color,
                    "B25",
                )

                add_summary_markers(
                    ax,
                    b25,
                    center,
                    "right",
                    "B25",
                )

        # Separators between process groups.
        for i in range(len(processes) - 1):
            left = x_positions[
                (processes[i], ice_sheets[-1])
            ]

            right = x_positions[
                (processes[i + 1], ice_sheets[0])
            ]

            ax.axvline(
                (left + right) / 2,
                color="0.78",
                linewidth=0.9,
                zorder=0,
            )

        # Each panel receives its own y-axis range.
        ax.set_ylim(y_min, y_max)

        ax.set_xlim(
            -0.75,
            max(x_positions.values()) + 0.75,
        )

        ax.set_ylabel(
            "SLR contribution (cm)",
            fontsize=12,
            fontweight="bold",
        )

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.5,
            color="0.85",
            alpha=0.8,
            zorder=0,
        )

        ax.axhline(
            0,
            color="0.35",
            linewidth=0.8,
            zorder=1,
        )

        ax.text(
            0.01,
            0.97,
            scenario["panel"],
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
        )

        ax.text(
            0.99,
            0.97,
            f"{year} — {scenario_name}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="top",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            axis="y",
            labelsize=10,
            direction="out",
            length=4,
        )

    # --------------------------------------------------------
    # X-axis labels
    # --------------------------------------------------------

    tick_positions = []
    tick_labels = []

    for process in processes:
        for ice_sheet in ice_sheets:
            tick_positions.append(
                x_positions[(process, ice_sheet)]
            )

            tick_labels.append(
                ice_sheet_labels[ice_sheet]
            )

    axes[-1].set_xticks(tick_positions)

    axes[-1].set_xticklabels(
        tick_labels,
        rotation=90,
        fontsize=10.5,
    )

    axes[-1].tick_params(
        axis="x",
        length=0,
        pad=4,
    )

    # Process-group labels beneath the x-axis.
    for process in processes:
        axes[-1].text(
            group_centers[process],
            -0.18,
            process_labels[process],
            transform=axes[-1].get_xaxis_transform(),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
            clip_on=False,
        )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ice_handles = [
        Patch(
            facecolor=ice_sheet_colors[sheet],
            edgecolor=ice_sheet_colors[sheet],
            alpha=0.65,
            label=ice_sheet_labels[sheet],
        )
        for sheet in ice_sheets
    ]

    study_handles = [
        Patch(
            facecolor="0.65",
            edgecolor="0.25",
            linestyle="--",
            alpha=b19_alpha,
            label="B19 — left half",
        ),
        Patch(
            facecolor="0.35",
            edgecolor="0.20",
            alpha=b25_alpha,
            label="B25 — right half",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="none",
            markersize=4,
            label="5th and 95th percentiles",
        ),
    ]

    axes[0].legend(
        handles=ice_handles + study_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
        fontsize=9,
        ncol=4,
    )

    # --------------------------------------------------------
    # Layout and save
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.17,
        top=0.91,
        hspace=0.12,
    )

    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(
        f"\nSaved {year} B19–B25 process comparison:"
    )
    print(png_file)