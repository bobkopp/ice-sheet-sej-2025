#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:03:19 2026

B19-B25 split-violin comparison of process-level SLR contributions.

Creates one figure per year (2100, 2200, 2300), each with two stacked
panels: low warming (2.0°C) and high warming (5.0°C).

Within each process/ice-sheet category:
    left half  = baseline-adjusted B19 Monte Carlo samples
    right half = B25 Monte Carlo samples

B19 process-level violin distributions are read from:
    B19_process_and_total_MC_adjusted_to_B25_baseline.xlsx
    sheet: B19_adjusted_MC_mm

For the B19 total ice-sheet category:
    - the violin shape is drawn from the adjusted B19 total MC samples;
    - the 5th percentile, median, and 95th percentile markers are read
      directly from data/B19_data.xlsx.

Both B19 and B25 Monte Carlo source values are in mm and are converted
to cm for plotting. Values in B19_data.xlsx are already in cm.

Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""

from figure_labels import ICE_SHEET_NAMES, PROJECTION_YEARS, VIOLIN_ICE_SHEET_LABELS, VIOLIN_PROCESS_LABELS

from figure_settings import PROJECT_ROOT, DATA_ROOT, UPSTREAM_ROOT, FIGURE_ROOT, output_for, DPI, SHOW_FIGURES, B25_MC_FILE, B19_ADJUSTED_FILE, CMIP6_ANNUAL_DIR, CMIP7_ANNUAL_DIR, CMIP6_SELECTION_FILE, CMIP7_SELECTION_FILE
from figure_settings import B19_SUMMARY_FILE, EXPERT_QUESTIONS_FILE, EXPERT_WORKBOOK_FILE, PERFORMANCE_FILE, AR6_WORKBOOK_FILE, COUPLED_SEJ_FILE
from figure_helpers import save_figure, maybe_show
from figure_style import ICE_SHEET_COLORS

import os
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from B25data_processing import (
    load_and_prepare_data,
    project_dir,
)


# ============================================================
# 1. SETTINGS
# ============================================================

output_dir = str(output_for("005_endpoint_distributions", "process_violin"))

os.makedirs(
    output_dir,
    exist_ok=True,
)


# Baseline-adjusted B19 process and total Monte Carlo samples.
b19_file = str(B19_ADJUSTED_FILE)

b19_sheet_name = "B19_adjusted_MC_mm"


# Published B19 total summary statistics.
# These values are already in cm.
b19_total_summary_file = str(B19_SUMMARY_FILE)


years = list(PROJECTION_YEARS)


scenario_settings = {
    "Low warming (2.0°C)": {
        "warming": 2.0,
        "suffix": "L",
        "panel": "A",
    },
    "High warming (5.0°C)": {
        "warming": 5.0,
        "suffix": "H",
        "panel": "B",
    },
}


ice_sheets = list(ICE_SHEET_NAMES[:4])


processes = [
    "Discharge",
    "Runoff minus accumulation",
    "Total ice-sheet contribution",
]


ice_sheet_labels = dict(VIOLIN_ICE_SHEET_LABELS)


process_labels = dict(VIOLIN_PROCESS_LABELS)




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
mc_total_ext = data["mc_total_ext"].copy()


def add_total_ais_processes(process_data):
    """
    Add Total AIS process distributions as the draw-by-draw sum
    of EAIS and WAIS process samples.
    """

    ais = (
        process_data.loc[
            process_data["ice_sheet"].isin(
                [
                    "EAIS",
                    "WAIS",
                ]
            )
        ]
        .pivot(
            index=[
                "sample_id",
                "process",
                "process_code",
                "warming",
                "warming_code",
                "year",
                "scenario_name",
            ],
            columns="ice_sheet",
            values="value",
        )
        .dropna(
            subset=[
                "EAIS",
                "WAIS",
            ]
        )
        .reset_index()
    )

    ais["value"] = (
        ais["EAIS"]
        + ais["WAIS"]
    )

    ais["variable"] = pd.NA
    ais["ice_sheet_code"] = "AIS"
    ais["ice_sheet"] = "Total AIS"

    ais = ais[
        process_data.columns
    ]

    return pd.concat(
        [
            process_data,
            ais,
        ],
        ignore_index=True,
    )


process_ext = add_total_ais_processes(
    process_ext
)


# ============================================================
# 3. LOAD AND VALIDATE ADJUSTED B19 MONTE CARLO DATA
# ============================================================

if not os.path.exists(b19_file):

    raise FileNotFoundError(
        "Adjusted B19 workbook not found:\n"
        f"{b19_file}"
    )


b19_data = pd.read_excel(
    b19_file,
    sheet_name=b19_sheet_name,
)


b19_data.columns = [
    str(column).strip()
    for column in b19_data.columns
]


print(
    "\nLoaded baseline-adjusted B19 Monte Carlo data:"
)

print(
    f"  File: {b19_file}"
)

print(
    f"  Sheet: {b19_sheet_name}"
)

print(
    f"  Rows: {len(b19_data):,}"
)

print(
    f"  Columns: {len(b19_data.columns):,}"
)


# ------------------------------------------------------------
# Expected process columns
# ------------------------------------------------------------

process_codes = [
    "EA",
    "ED",
    "GA",
    "GD",
    "GR",
    "WA",
    "WD",
    "WR",
]


required_process_columns = {
    f"PW01{code}_{year}{suffix}"
    for year in years
    for suffix in [
        "L",
        "H",
    ]
    for code in process_codes
}


# ------------------------------------------------------------
# Expected direct total columns
# ------------------------------------------------------------

total_codes = [
    "E",
    "G",
    "W",
]


required_total_columns = {
    f"USEU_PW01_{code}_{year}{suffix}"
    for year in years
    for suffix in [
        "L",
        "H",
    ]
    for code in total_codes
}


required_columns = (
    required_process_columns
    | required_total_columns
)


missing_columns = (
    required_columns
    - set(b19_data.columns)
)


if missing_columns:

    raise ValueError(
        "The adjusted B19 workbook is missing these columns:\n"
        + "\n".join(
            sorted(missing_columns)
        )
        + "\n\nColumns read by pandas:\n"
        + str(
            b19_data.columns.tolist()
        )
    )


for column in required_columns:

    b19_data[column] = pd.to_numeric(
        b19_data[column],
        errors="coerce",
    )


print(
    f"\nValidated {len(required_process_columns)} process columns "
    f"and {len(required_total_columns)} total columns."
)


# ============================================================
# 4. LOAD B19 TOTAL SUMMARY STATISTICS
# ============================================================

if not os.path.exists(
    b19_total_summary_file
):

    raise FileNotFoundError(
        "B19 total-summary workbook not found:\n"
        f"{b19_total_summary_file}"
    )


b19_total_summary = pd.read_excel(
    b19_total_summary_file,
    sheet_name=0,
)


# Convert all headers to strings so numeric Excel headers such
# as 0.05 and 0.95 can be referenced consistently.
b19_total_summary.columns = [
    str(column).strip()
    for column in b19_total_summary.columns
]


required_summary_columns = {
    "Year",
    "Ice Sheet",
    "Scenario",
    "Median (50%)",
    "0.05",
    "0.95",
}


missing_summary_columns = (
    required_summary_columns
    - set(
        b19_total_summary.columns
    )
)


if missing_summary_columns:

    raise ValueError(
        "B19_data.xlsx is missing these columns:\n"
        + "\n".join(
            sorted(
                missing_summary_columns
            )
        )
        + "\n\nColumns read by pandas:\n"
        + str(
            b19_total_summary.columns.tolist()
        )
    )


for column in [
    "Year",
    "Median (50%)",
    "0.05",
    "0.95",
]:

    b19_total_summary[column] = pd.to_numeric(
        b19_total_summary[column],
        errors="coerce",
    )


b19_total_summary["Ice Sheet"] = (
    b19_total_summary["Ice Sheet"]
    .astype(str)
    .str.strip()
)


b19_total_summary["Scenario"] = (
    b19_total_summary["Scenario"]
    .astype(str)
    .str.strip()
)


# Retain only the years used in the figures.
b19_total_summary = (
    b19_total_summary.loc[
        b19_total_summary["Year"].isin(
            years
        )
    ]
    .copy()
)


print(
    "\nLoaded B19 total summary statistics:"
)

print(
    f"  File: {b19_total_summary_file}"
)

print(
    f"  Rows for 2100–2300: "
    f"{len(b19_total_summary):,}"
)


# ============================================================
# 5. COLUMN-NAME HELPERS
# ============================================================

def b19_process_column(
    year,
    suffix,
    code,
):
    """
    Return one B19 process-column name.

    Example:
        PW01EA_2100L
    """

    return (
        f"PW01{code}_"
        f"{year}{suffix}"
    )


def b19_total_column(
    year,
    suffix,
    code,
):
    """
    Return one B19 total-column name.

    Example:
        USEU_PW01_E_2100L
    """

    return (
        f"USEU_PW01_{code}_"
        f"{year}{suffix}"
    )


# ============================================================
# 6. GENERIC SAMPLE EXTRACTION HELPERS
# ============================================================

def clean_samples(
    values,
    label,
):
    """
    Remove missing and infinite values and return a NumPy array.
    """

    samples = (
        pd.Series(values)
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if samples.size == 0:

        raise ValueError(
            f"No finite samples found for {label}."
        )

    return samples


def aligned_columns(
    column_names,
):
    """
    Return row-aligned numerical arrays for a set of B19 columns.

    Rows containing a missing value in any selected column are
    removed before draw-by-draw arithmetic.
    """

    aligned = (
        b19_data[column_names]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .astype(float)
    )

    if aligned.empty:

        raise ValueError(
            "No aligned B19 samples for:\n"
            + "\n".join(
                column_names
            )
        )

    return {
        column: aligned[column].to_numpy()
        for column in column_names
    }


def b19_process_samples(
    year,
    suffix,
    code,
):
    """
    Return one adjusted B19 process distribution in mm.
    """

    column = b19_process_column(
        year=year,
        suffix=suffix,
        code=code,
    )

    return clean_samples(
        b19_data[column],
        column,
    )


# ============================================================
# 7. B25 SAMPLE EXTRACTION
# ============================================================

def get_b25_samples(
    year,
    warming,
    process,
    ice_sheet,
):
    """
    Return one B25 process distribution in cm.
    """

    selection = process_ext.loc[
        (
            process_ext["year"]
            == year
        )
        & np.isclose(
            process_ext["warming"],
            warming,
        )
        & (
            process_ext["process"]
            == process
        )
        & (
            process_ext["ice_sheet"]
            == ice_sheet
        ),
        "value",
    ]

    return (
        clean_samples(
            selection,
            (
                f"B25 {ice_sheet}, {process}, "
                f"{year}, {warming:.1f}°C"
            ),
        )
        / 10.0
    )


def get_b25_total_samples(
    year,
    warming,
    ice_sheet,
):
    """
    Return one B25 total ice-sheet distribution in cm.
    """

    selection = mc_total_ext.loc[
        (
            mc_total_ext["year"]
            == year
        )
        & np.isclose(
            mc_total_ext["warming"],
            warming,
        )
        & (
            mc_total_ext["ice_sheet"]
            == ice_sheet
        ),
        "value",
    ]

    return (
        clean_samples(
            selection,
            (
                f"B25 total {ice_sheet}, "
                f"{year}, {warming:.1f}°C"
            ),
        )
        / 10.0
    )


# ============================================================
# 8. B19 PROCESS SAMPLE EXTRACTION
# ============================================================

def get_b19_samples(
    year,
    suffix,
    process,
    ice_sheet,
):
    """
    Return one baseline-adjusted B19 process distribution in cm.

    Accumulation is already signed as an SLR contribution.
    Therefore:

        SMB = runoff + accumulation

    EAIS has no separate runoff column.
    """

    direct_process_codes = {
        ("EAIS", "Accumulation"): "EA",
        ("EAIS", "Discharge"): "ED",

        ("WAIS", "Accumulation"): "WA",
        ("WAIS", "Discharge"): "WD",
        ("WAIS", "Runoff"): "WR",

        ("GrIS", "Accumulation"): "GA",
        ("GrIS", "Discharge"): "GD",
        ("GrIS", "Runoff"): "GR",
    }


    direct_code = direct_process_codes.get(
        (
            ice_sheet,
            process,
        )
    )


    if direct_code is not None:

        samples_mm = b19_process_samples(
            year=year,
            suffix=suffix,
            code=direct_code,
        )


    elif (
        ice_sheet == "EAIS"
        and process == "Runoff"
    ):

        return None


    elif (
        ice_sheet == "Total AIS"
        and process == "Accumulation"
    ):

        ea_column = b19_process_column(
            year,
            suffix,
            "EA",
        )

        wa_column = b19_process_column(
            year,
            suffix,
            "WA",
        )

        arrays = aligned_columns(
            [
                ea_column,
                wa_column,
            ]
        )

        samples_mm = (
            arrays[ea_column]
            + arrays[wa_column]
        )


    elif (
        ice_sheet == "Total AIS"
        and process == "Discharge"
    ):

        ed_column = b19_process_column(
            year,
            suffix,
            "ED",
        )

        wd_column = b19_process_column(
            year,
            suffix,
            "WD",
        )

        arrays = aligned_columns(
            [
                ed_column,
                wd_column,
            ]
        )

        samples_mm = (
            arrays[ed_column]
            + arrays[wd_column]
        )


    elif (
        ice_sheet == "Total AIS"
        and process == "Runoff"
    ):

        # No EAIS runoff distribution exists in B19.
        samples_mm = b19_process_samples(
            year=year,
            suffix=suffix,
            code="WR",
        )


    elif (
        ice_sheet == "EAIS"
        and process
        == "Runoff minus accumulation"
    ):

        # No EAIS runoff term exists.
        # Accumulation is already negative in SLR units.
        samples_mm = b19_process_samples(
            year=year,
            suffix=suffix,
            code="EA",
        )


    elif (
        ice_sheet == "WAIS"
        and process
        == "Runoff minus accumulation"
    ):

        runoff_column = b19_process_column(
            year,
            suffix,
            "WR",
        )

        accumulation_column = b19_process_column(
            year,
            suffix,
            "WA",
        )

        arrays = aligned_columns(
            [
                runoff_column,
                accumulation_column,
            ]
        )

        samples_mm = (
            arrays[runoff_column]
            + arrays[accumulation_column]
        )


    elif (
        ice_sheet == "GrIS"
        and process
        == "Runoff minus accumulation"
    ):

        runoff_column = b19_process_column(
            year,
            suffix,
            "GR",
        )

        accumulation_column = b19_process_column(
            year,
            suffix,
            "GA",
        )

        arrays = aligned_columns(
            [
                runoff_column,
                accumulation_column,
            ]
        )

        samples_mm = (
            arrays[runoff_column]
            + arrays[accumulation_column]
        )


    elif (
        ice_sheet == "Total AIS"
        and process
        == "Runoff minus accumulation"
    ):

        runoff_column = b19_process_column(
            year,
            suffix,
            "WR",
        )

        eais_accumulation_column = (
            b19_process_column(
                year,
                suffix,
                "EA",
            )
        )

        wais_accumulation_column = (
            b19_process_column(
                year,
                suffix,
                "WA",
            )
        )

        arrays = aligned_columns(
            [
                runoff_column,
                eais_accumulation_column,
                wais_accumulation_column,
            ]
        )

        samples_mm = (
            arrays[runoff_column]
            + arrays[
                eais_accumulation_column
            ]
            + arrays[
                wais_accumulation_column
            ]
        )


    else:

        raise ValueError(
            "Unsupported adjusted B19 combination: "
            f"{ice_sheet}, {process}"
        )


    return samples_mm / 10.0


# ============================================================
# 9. B19 TOTAL MONTE CARLO SAMPLE EXTRACTION
# ============================================================

def get_b19_total_samples(
    year,
    suffix,
    ice_sheet,
):
    """
    Return one baseline-adjusted B19 total distribution in cm.

    EAIS, WAIS, and Greenland totals are read directly from the
    corrected USEU_PW01 total columns.

    Total AIS is calculated draw by draw as:

        corrected EAIS total + corrected WAIS total

    These samples are used to draw the B19 total violin shapes.
    """

    total_code_map = {
        "EAIS": "E",
        "WAIS": "W",
        "GrIS": "G",
    }


    if ice_sheet in total_code_map:

        total_code = total_code_map[
            ice_sheet
        ]

        column = b19_total_column(
            year=year,
            suffix=suffix,
            code=total_code,
        )

        samples_mm = clean_samples(
            b19_data[column],
            column,
        )


    elif ice_sheet == "Total AIS":

        eais_column = b19_total_column(
            year=year,
            suffix=suffix,
            code="E",
        )

        wais_column = b19_total_column(
            year=year,
            suffix=suffix,
            code="W",
        )

        arrays = aligned_columns(
            [
                eais_column,
                wais_column,
            ]
        )

        samples_mm = (
            arrays[eais_column]
            + arrays[wais_column]
        )


    else:

        raise ValueError(
            "Unsupported B19 total ice sheet: "
            f"{ice_sheet}"
        )


    return samples_mm / 10.0


# ============================================================
# 10. B19 TOTAL PERCENTILE LOOKUP
# ============================================================

def get_b19_total_percentiles(
    year,
    suffix,
    ice_sheet,
):
    """
    Return B19 total 5th, 50th, and 95th percentiles in cm.

    Values are read directly from B19_data.xlsx.

    Returns:
        np.array([q05, q50, q95])
    """

    scenario = {
        "L": "Low",
        "H": "High",
    }[suffix]


    summary_ice_sheet = {
        "EAIS": "EAIS",
        "WAIS": "WAIS",
        "Total AIS": "Total AIS",
        "GrIS": "GrIS",
    }[ice_sheet]


    selection = b19_total_summary.loc[
        (
            b19_total_summary["Year"]
            == year
        )
        & (
            b19_total_summary["Scenario"]
            == scenario
        )
        & (
            b19_total_summary["Ice Sheet"]
            == summary_ice_sheet
        )
    ]


    if len(selection) != 1:

        raise ValueError(
            "Expected exactly one B19 total-summary row for:\n"
            f"  Year: {year}\n"
            f"  Scenario: {scenario}\n"
            f"  Ice sheet: {summary_ice_sheet}\n"
            f"Rows found: {len(selection)}"
        )


    row = selection.iloc[0]


    values = np.array(
        [
            row["0.05"],
            row["Median (50%)"],
            row["0.95"],
        ],
        dtype=float,
    )


    if not np.all(
        np.isfinite(values)
    ):

        raise ValueError(
            "Non-finite B19 total percentiles found for:\n"
            f"  Year: {year}\n"
            f"  Scenario: {scenario}\n"
            f"  Ice sheet: {summary_ice_sheet}"
        )


    return values


# ============================================================
# 11. VIOLIN FUNCTIONS
# ============================================================

def calculate_density(samples):

    samples = np.asarray(
        samples,
        dtype=float,
    )

    y_min, y_max = np.percentile(
        samples,
        [
            violin_lower_percentile,
            violin_upper_percentile,
        ],
    )


    if (
        np.unique(samples).size < 2
        or np.isclose(
            np.ptp(samples),
            0.0,
        )
    ):

        center = float(
            np.median(samples)
        )

        y = np.linspace(
            center - 0.25,
            center + 0.25,
            kde_points,
        )

        density = np.exp(
            -0.5
            * (
                (y - center)
                / 0.08
            ) ** 2
        )

        return (
            y,
            density / density.max(),
        )


    if np.isclose(
        y_min,
        y_max,
    ):

        padding = max(
            abs(y_min) * 0.01,
            0.25,
        )

        y_min -= padding
        y_max += padding


    y = np.linspace(
        y_min,
        y_max,
        kde_points,
    )


    try:

        kde = gaussian_kde(
            samples
        )

        kde.set_bandwidth(
            kde.factor
            * kde_bandwidth_multiplier
        )

        density = kde(y)


    except np.linalg.LinAlgError:

        warnings.warn(
            "Singular KDE; using a normal approximation."
        )

        center = float(
            np.median(samples)
        )

        scale = max(
            float(np.std(samples)),
            0.08,
        )

        density = np.exp(
            -0.5
            * (
                (y - center)
                / scale
            ) ** 2
        )


    if density.max() > 0:

        density /= density.max()


    return (
        y,
        density,
    )


def draw_half_violin(
    ax,
    samples,
    center,
    side,
    color,
    study,
):

    y, density = calculate_density(
        samples
    )

    widths = (
        density
        * violin_half_width
    )

    if side == "left":

        outer = center - widths

    else:

        outer = center + widths


    alpha = (
        b19_alpha
        if study == "B19"
        else b25_alpha
    )

    linestyle = (
        "--"
        if study == "B19"
        else "-"
    )


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

    collection.set_linestyle(
        linestyle
    )


def add_summary_markers(
    ax,
    samples,
    center,
    side,
    study,
    percentiles=None,
):
    """
    Add median and 5th/95th percentile markers.

    If percentiles is None, markers are calculated from samples.

    If percentiles is supplied, it must contain:
        [q05, q50, q95]

    The supplied values are used for B19 total markers.
    """

    if percentiles is None:

        q05, q50, q95 = np.percentile(
            samples,
            [
                5,
                50,
                95,
            ],
        )

    else:

        q05, q50, q95 = np.asarray(
            percentiles,
            dtype=float,
        )


    if side == "left":

        median_xmin = (
            center
            - median_half_width
        )

        median_xmax = center

        marker_x = (
            center
            - median_half_width
            * 0.62
        )

    else:

        median_xmin = center

        median_xmax = (
            center
            + median_half_width
        )

        marker_x = (
            center
            + median_half_width
            * 0.62
        )


    linestyle = (
        "--"
        if study == "B19"
        else "-"
    )

    facecolor = (
        "white"
        if study == "B19"
        else "black"
    )


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
        [
            q05,
            q95,
        ],
        marker_x
        - percentile_tick_half_width,
        marker_x
        + percentile_tick_half_width,
        color="black",
        linewidth=1.1,
        linestyle=linestyle,
        zorder=7,
    )


    ax.plot(
        [
            marker_x,
            marker_x,
        ],
        [
            q05,
            q95,
        ],
        linestyle="none",
        marker="o",
        markersize=3.4,
        markerfacecolor=facecolor,
        markeredgecolor="black",
        markeredgewidth=0.9,
        zorder=8,
    )


# ============================================================
# 12. POSITIONS AND Y LIMITS
# ============================================================

def create_x_positions():

    positions = {}
    group_centers = {}

    x = 0.0


    for process in processes:

        group = []

        for ice_sheet in ice_sheets:

            positions[
                (
                    process,
                    ice_sheet,
                )
            ] = x

            group.append(x)

            x += within_group_spacing


        group_centers[process] = (
            np.mean(group)
        )

        x += between_group_spacing


    return (
        positions,
        group_centers,
    )


x_positions, group_centers = (
    create_x_positions()
)


def panel_y_limits(
    year,
    warming,
    suffix,
    padding_fraction=0.08,
):
    """
    Calculate y-limits for one year and warming panel.

    B19 total hard-coded percentile markers are included so none
    of the total markers fall outside the visible axis range.
    """

    bounds = []


    for process in processes:

        for ice_sheet in ice_sheets:


            if (
                process
                == "Total ice-sheet contribution"
            ):

                b25 = get_b25_total_samples(
                    year=year,
                    warming=warming,
                    ice_sheet=ice_sheet,
                )

                b19 = get_b19_total_samples(
                    year=year,
                    suffix=suffix,
                    ice_sheet=ice_sheet,
                )

                b19_marker_values = (
                    get_b19_total_percentiles(
                        year=year,
                        suffix=suffix,
                        ice_sheet=ice_sheet,
                    )
                )


            else:

                b25 = get_b25_samples(
                    year=year,
                    warming=warming,
                    process=process,
                    ice_sheet=ice_sheet,
                )

                b19 = get_b19_samples(
                    year=year,
                    suffix=suffix,
                    process=process,
                    ice_sheet=ice_sheet,
                )

                b19_marker_values = None


            bounds.append(
                np.percentile(
                    b25,
                    [
                        violin_lower_percentile,
                        violin_upper_percentile,
                    ],
                )
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


            if b19_marker_values is not None:

                bounds.append(
                    np.array(
                        [
                            b19_marker_values[0],
                            b19_marker_values[2],
                        ]
                    )
                )


    bounds = np.asarray(
        bounds
    )

    low = bounds[:, 0].min()
    high = bounds[:, 1].max()

    span = high - low


    if np.isclose(
        span,
        0.0,
    ):

        span = 1.0


    padding = (
        padding_fraction
        * span
    )


    return (
        low - padding,
        high + padding,
    )


# ============================================================
# 13. CREATE FIGURES
# ============================================================

for year in years:


    png_file = os.path.join(
        output_dir,
        (
            "Figure_process_B19adjusted_B25_"
            "discharge_runoffminusacc_totals_"
            f"{year}_UPDATED.png"
        ),
    )


    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            11.0,
            11.0,
        ),
        sharex=True,
        sharey=False,
    )


    for ax, (
        scenario_name,
        scenario,
    ) in zip(
        axes,
        scenario_settings.items(),
    ):


        y_min, y_max = panel_y_limits(
            year=year,
            warming=scenario["warming"],
            suffix=scenario["suffix"],
        )


        for process in processes:

            for ice_sheet in ice_sheets:


                center = x_positions[
                    (
                        process,
                        ice_sheet,
                    )
                ]


                color = ICE_SHEET_COLORS[
                    ice_sheet
                ]


                if (
                    process
                    == "Total ice-sheet contribution"
                ):

                    # Adjusted MC samples determine the B19 total
                    # violin density.
                    b19 = get_b19_total_samples(
                        year=year,
                        suffix=scenario["suffix"],
                        ice_sheet=ice_sheet,
                    )

                    # Published B19 summary values determine the
                    # B19 total 5th, 50th, and 95th markers.
                    b19_marker_values = (
                        get_b19_total_percentiles(
                            year=year,
                            suffix=scenario["suffix"],
                            ice_sheet=ice_sheet,
                        )
                    )

                    b25 = get_b25_total_samples(
                        year=year,
                        warming=scenario["warming"],
                        ice_sheet=ice_sheet,
                    )


                else:

                    b19_marker_values = None

                    b19 = get_b19_samples(
                        year=year,
                        suffix=scenario["suffix"],
                        process=process,
                        ice_sheet=ice_sheet,
                    )

                    b25 = get_b25_samples(
                        year=year,
                        warming=scenario["warming"],
                        process=process,
                        ice_sheet=ice_sheet,
                    )


                if b19 is not None:

                    draw_half_violin(
                        ax=ax,
                        samples=b19,
                        center=center,
                        side="left",
                        color=color,
                        study="B19",
                    )

                    add_summary_markers(
                        ax=ax,
                        samples=b19,
                        center=center,
                        side="left",
                        study="B19",
                        percentiles=b19_marker_values,
                    )


                draw_half_violin(
                    ax=ax,
                    samples=b25,
                    center=center,
                    side="right",
                    color=color,
                    study="B25",
                )

                add_summary_markers(
                    ax=ax,
                    samples=b25,
                    center=center,
                    side="right",
                    study="B25",
                )


        # Separators between process groups.
        for i in range(
            len(processes) - 1
        ):

            left = x_positions[
                (
                    processes[i],
                    ice_sheets[-1],
                )
            ]

            right = x_positions[
                (
                    processes[i + 1],
                    ice_sheets[0],
                )
            ]

            ax.axvline(
                (
                    left + right
                )
                / 2,
                color="0.78",
                linewidth=0.9,
                zorder=0,
            )


        ax.set_ylim(
            y_min,
            y_max,
        )


        ax.yaxis.set_minor_locator(
            AutoMinorLocator(2)
        )


        ax.set_xlim(
            -0.75,
            max(
                x_positions.values()
            )
            + 0.75,
        )


        ax.set_ylabel(
            "SLR contribution (cm)",
            fontsize=12,
            fontweight="bold",
        )


        ax.tick_params(
            axis="y",
            which="minor",
            length=2.5,
            width=0.6,
        )


        ax.grid(
            which="major",
            axis="y",
            linestyle="--",
            linewidth=0.5,
            color="0.82",
            alpha=0.8,
            zorder=0,
        )


        ax.grid(
            which="minor",
            axis="y",
            linestyle=":",
            linewidth=0.4,
            color="0.82",
            alpha=0.7,
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
            (
                f"{year} — "
                f"{scenario_name}"
            ),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="top",
        )


        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)


        ax.tick_params(
            axis="y",
            labelsize=10,
            direction="out",
            length=4,
        )


    # --------------------------------------------------------
    # SHARED X-AXIS LABELS
    # --------------------------------------------------------

    tick_positions = []
    tick_labels = []


    for process in processes:

        for ice_sheet in ice_sheets:

            tick_positions.append(
                x_positions[
                    (
                        process,
                        ice_sheet,
                    )
                ]
            )

            tick_labels.append(
                ice_sheet_labels[
                    ice_sheet
                ]
            )


    for ax in axes:

        ax.set_xticks(
            tick_positions
        )

        ax.set_xticklabels(
            tick_labels,
            fontsize=10.5,
        )

        ax.tick_params(
            axis="x",
            length=0,
            pad=4,
            labelbottom=True,
        )


    # Group labels beneath the lower subplot.
    for process in processes:

        axes[-1].text(
            group_centers[
                process
            ],
            -0.1,
            process_labels[
                process
            ],
            transform=(
                axes[-1]
                .get_xaxis_transform()
            ),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
            clip_on=False,
        )


    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    ice_handles = [
        Patch(
            facecolor=(
                ICE_SHEET_COLORS[sheet]
            ),
            edgecolor=(
                ICE_SHEET_COLORS[sheet]
            ),
            alpha=0.65,
            label=(
                ice_sheet_labels[sheet]
            ),
        )
        for sheet in ice_sheets
    ]


    study_handles = [
        Patch(
            facecolor="0.65",
            edgecolor="0.25",
            linestyle="--",
            alpha=b19_alpha,
            label=(
                "Baseline-adjusted SEJ 2018 "
                "— left curve"
            ),
        ),
        Patch(
            facecolor="0.35",
            edgecolor="0.20",
            alpha=b25_alpha,
            label="SEJ 2025 — right curve",
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
            label=(
                "5th and 95th percentiles"
            ),
        ),
    ]


    axes[1].legend(
        handles=(
            ice_handles
            + study_handles
        ),
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.2,
        ),
        frameon=False,
        fontsize=10,
        ncol=4,
        columnspacing=1.3,
        handletextpad=0.5,
    )


    # --------------------------------------------------------
    # LAYOUT AND SAVE
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.14,
        top=0.91,
        hspace=0.18,
    )


    save_figure(fig, png_file)


    maybe_show()
    plt.close(fig)


    print(
        f"\nSaved {year} adjusted B19–B25 comparison:"
    )

    print(
        png_file
    )
