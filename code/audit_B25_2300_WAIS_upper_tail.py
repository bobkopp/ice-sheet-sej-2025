#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit the unusual B25 2300 WAIS / Total AIS upper tail at 3.5°C and 5.0°C.

This script:

1. Loads and preprocesses the original B25 Monte Carlo data using
   B25data_processing.py.
2. Extracts EAIS, WAIS, and Total AIS at 2300 for 3.5°C and 5.0°C.
3. Compares quantiles, maxima, and exceedance probabilities.
4. Checks whether the 3.5°C WAIS upper tail contains repeated values,
   large gaps, or strong rounding.
5. Checks draw-by-draw monotonicity from 3.5°C to 5.0°C.
6. Identifies the most severe WAIS decreases across warming levels.
7. Locates quantile-curve crossover points.
8. Verifies Total AIS = EAIS + WAIS.
9. Checks for duplicate or missing records.
10. Saves diagnostic CSV tables and figures.

Expected project setup
----------------------
This script assumes the following file is available in the project directory:

    B25data_processing.py

and that it exposes:

    load_and_prepare_data
    project_dir
    data_file

The script uses the original B25 input file defined by data_file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. LOAD B25 PROCESSING UTILITIES
# ============================================================

# Add the project directory manually here only if Python cannot find
# B25data_processing.py. Usually this is unnecessary when the script
# is saved and run from the project directory.
#
# Example:
# PROJECT_DIR_OVERRIDE = Path(
#     "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025"
# )
#
# sys.path.insert(0, str(PROJECT_DIR_OVERRIDE))

try:
    from B25data_processing import (
        load_and_prepare_data,
        project_dir,
        data_file,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import B25data_processing.py.\n\n"
        "Save this script in the same project directory as "
        "B25data_processing.py, or add the project directory to "
        "sys.path near the top of this script."
    ) from exc


# ============================================================
# 2. USER SETTINGS
# ============================================================

TARGET_YEAR = 2300
WARMING_LEVELS = [3.5, 5.0]
ICE_SHEETS = ["EAIS", "WAIS", "Total AIS"]

QUANTILES = [
    0.01,
    0.05,
    0.17,
    0.50,
    0.83,
    0.95,
    0.99,
]

TOTAL_AIS_THRESHOLDS = [
    5000,
    10000,
    15000,
    20000,
    25000,
    30000,
]

WAIS_THRESHOLDS = [
    5000,
    10000,
    15000,
    20000,
    25000,
    27500,
    28000,
]

UPPER_TAIL_THRESHOLD = 15000
ORDERED_GAP_THRESHOLD = 10000
N_LARGEST_DRAWS = 100
N_LARGEST_DECREASES = 100

OUTPUT_DIR = Path(
    project_dir
) / "output" / "temporal_coupling" / "tail_diagnostics_2300"

FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def print_heading(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_columns(
    frame: pd.DataFrame,
    required: list[str],
    frame_name: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"{frame_name} is missing required columns: {missing}\n"
            f"Available columns:\n{frame.columns.tolist()}"
        )


def get_sorted_values(
    frame: pd.DataFrame,
    warming: float,
    ice_sheet: str,
) -> np.ndarray:
    values = (
        frame.loc[
            np.isclose(
                frame["warming"].astype(float),
                warming,
            )
            & (frame["ice_sheet"] == ice_sheet),
            "value",
        ]
        .sort_values()
        .to_numpy(dtype=float)
    )

    if len(values) == 0:
        raise ValueError(
            f"No values found for warming={warming}, "
            f"ice_sheet={ice_sheet}."
        )

    return values


def exceedance_table(
    frame: pd.DataFrame,
    ice_sheet: str,
    thresholds: list[float],
) -> pd.DataFrame:
    selected = frame.loc[
        frame["ice_sheet"] == ice_sheet
    ].copy()

    records = []

    for warming, group in selected.groupby("warming"):
        values = group["value"].to_numpy(dtype=float)

        for threshold in thresholds:
            records.append(
                {
                    "ice_sheet": ice_sheet,
                    "warming": float(warming),
                    "threshold": threshold,
                    "n_above": int(np.sum(values > threshold)),
                    "percent_above": float(
                        100.0 * np.mean(values > threshold)
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# 4. LOAD AND PREPARE ORIGINAL B25 DATA
# ============================================================

print_heading("LOADING ORIGINAL B25 DATA")

print(f"Project directory:\n{project_dir}")
print(f"\nB25 input file:\n{data_file}")

b25_data = load_and_prepare_data(
    file_path=str(data_file)
)

print("\nObjects returned by load_and_prepare_data():")
print(sorted(b25_data.keys()))

if "mc_total_ext" not in b25_data:
    raise KeyError(
        "load_and_prepare_data() did not return 'mc_total_ext'.\n"
        "Available keys:\n"
        f"{sorted(b25_data.keys())}"
    )

mc_total_ext = b25_data["mc_total_ext"].copy()

require_columns(
    mc_total_ext,
    [
        "sample_id",
        "year",
        "warming",
        "ice_sheet",
        "value",
    ],
    "mc_total_ext",
)

mc_total_ext["year"] = pd.to_numeric(
    mc_total_ext["year"],
    errors="raise",
).astype(int)

mc_total_ext["warming"] = pd.to_numeric(
    mc_total_ext["warming"],
    errors="raise",
)

mc_total_ext["value"] = pd.to_numeric(
    mc_total_ext["value"],
    errors="raise",
)

if not np.isfinite(
    mc_total_ext["value"].to_numpy(dtype=float)
).all():
    raise ValueError(
        "mc_total_ext contains non-finite values."
    )

print(
    f"\nLoaded mc_total_ext with "
    f"{len(mc_total_ext):,} records."
)


# ============================================================
# 5. CREATE ANALYSIS SUBSET
# ============================================================

subset_2300 = mc_total_ext.loc[
    (mc_total_ext["year"] == TARGET_YEAR)
    & mc_total_ext["warming"].isin(WARMING_LEVELS)
    & mc_total_ext["ice_sheet"].isin(ICE_SHEETS)
].copy()

if subset_2300.empty:
    raise ValueError(
        "No matching 2300 data were found."
    )

print_heading("ANALYSIS SUBSET")

print(
    subset_2300
    .groupby(
        ["warming", "ice_sheet"],
        observed=True,
    )["value"]
    .count()
    .rename("n")
    .to_string()
)


# ============================================================
# 6. COMPONENT DISTRIBUTION SUMMARY
# ============================================================

print_heading("COMPONENT DISTRIBUTION SUMMARY")

summary_records = []

for (
    warming,
    ice_sheet,
), group in subset_2300.groupby(
    ["warming", "ice_sheet"],
    observed=True,
):
    values = group["value"]

    record = {
        "warming": float(warming),
        "ice_sheet": ice_sheet,
        "n": len(values),
        "minimum": values.min(),
        "mean": values.mean(),
        "median": values.median(),
        "maximum": values.max(),
    }

    for quantile in QUANTILES:
        record[
            f"p{int(round(quantile * 100)):02d}"
        ] = values.quantile(quantile)

    summary_records.append(record)

component_summary = (
    pd.DataFrame(summary_records)
    .sort_values(
        ["ice_sheet", "warming"]
    )
    .reset_index(drop=True)
)

print(
    component_summary
    .round(2)
    .to_string(index=False)
)

component_summary.to_csv(
    OUTPUT_DIR
    / "b25_2300_component_summary_3p5C_5p0C.csv",
    index=False,
)


# ============================================================
# 7. TOTAL AIS DIFFERENCE SUMMARY
# ============================================================

print_heading("TOTAL AIS: 5.0°C MINUS 3.5°C")

total_summary = (
    component_summary.loc[
        component_summary["ice_sheet"] == "Total AIS"
    ]
    .set_index("warming")
)

distribution_columns = [
    column
    for column in [
        "minimum",
        "p01",
        "p05",
        "p17",
        "median",
        "mean",
        "p83",
        "p95",
        "p99",
        "maximum",
    ]
    if column in total_summary.columns
]

total_comparison = pd.DataFrame(
    {
        "3.5C": total_summary.loc[
            3.5,
            distribution_columns,
        ],
        "5.0C": total_summary.loc[
            5.0,
            distribution_columns,
        ],
    }
)

total_comparison["5.0C_minus_3.5C"] = (
    total_comparison["5.0C"]
    - total_comparison["3.5C"]
)

print(
    total_comparison
    .round(2)
    .to_string()
)

total_comparison.to_csv(
    OUTPUT_DIR
    / "b25_2300_total_ais_distribution_comparison.csv"
)


# ============================================================
# 8. TOTAL AIS AND WAIS EXCEEDANCE TABLES
# ============================================================

print_heading("UPPER-TAIL EXCEEDANCE TABLES")

total_ais_exceedance = exceedance_table(
    frame=subset_2300,
    ice_sheet="Total AIS",
    thresholds=TOTAL_AIS_THRESHOLDS,
)

wais_exceedance = exceedance_table(
    frame=subset_2300,
    ice_sheet="WAIS",
    thresholds=WAIS_THRESHOLDS,
)

print("\nTotal AIS percent above threshold:")
print(
    total_ais_exceedance
    .pivot(
        index="threshold",
        columns="warming",
        values="percent_above",
    )
    .round(4)
    .to_string()
)

print("\nWAIS percent above threshold:")
print(
    wais_exceedance
    .pivot(
        index="threshold",
        columns="warming",
        values="percent_above",
    )
    .round(4)
    .to_string()
)

total_ais_exceedance.to_csv(
    OUTPUT_DIR
    / "b25_2300_total_ais_exceedance_comparison.csv",
    index=False,
)

wais_exceedance.to_csv(
    OUTPUT_DIR
    / "b25_2300_wais_exceedance_comparison.csv",
    index=False,
)


# ============================================================
# 9. 3.5°C WAIS UPPER-TAIL VALUE FREQUENCIES
# ============================================================

print_heading("3.5°C WAIS UPPER-TAIL FREQUENCIES")

wais_35_2300 = subset_2300.loc[
    np.isclose(
        subset_2300["warming"],
        3.5,
    )
    & (subset_2300["ice_sheet"] == "WAIS")
].copy()

wais_35_upper = wais_35_2300.loc[
    wais_35_2300["value"] >= UPPER_TAIL_THRESHOLD
].copy()

frequency_table = (
    wais_35_upper["value"]
    .value_counts()
    .sort_index()
    .rename_axis("WAIS_value")
    .reset_index(name="count")
)

print(
    f"WAIS draws >= {UPPER_TAIL_THRESHOLD:,.0f}: "
    f"{len(wais_35_upper):,} "
    f"({100 * len(wais_35_upper) / len(wais_35_2300):.3f}%)"
)

print("\nMost common upper-tail values:")
print(
    wais_35_upper["value"]
    .value_counts()
    .head(30)
    .to_string()
)

frequency_table.to_csv(
    OUTPUT_DIR
    / "b25_2300_3p5C_wais_upper_tail_frequencies.csv",
    index=False,
)


# ============================================================
# 10. ORDERED UPPER-TAIL GAPS
# ============================================================

print_heading("LARGEST ORDERED GAPS IN 3.5°C WAIS")

ordered_wais_35 = np.sort(
    wais_35_2300["value"].to_numpy(dtype=float)
)

upper_values = ordered_wais_35[
    ordered_wais_35 >= ORDERED_GAP_THRESHOLD
]

if len(upper_values) >= 2:
    gap_df = pd.DataFrame(
        {
            "lower_value": upper_values[:-1],
            "upper_value": upper_values[1:],
            "gap": np.diff(upper_values),
        }
    )

    largest_gaps = (
        gap_df
        .sort_values(
            "gap",
            ascending=False,
        )
        .head(100)
    )

    print(
        largest_gaps
        .round(2)
        .head(30)
        .to_string(index=False)
    )

    largest_gaps.to_csv(
        OUTPUT_DIR
        / "b25_2300_3p5C_wais_largest_ordered_gaps.csv",
        index=False,
    )
else:
    print(
        "Fewer than two values exceed the ordered-gap threshold."
    )


# ============================================================
# 11. DRAW-BY-DRAW WARMING COMPARISON
# ============================================================

print_heading("DRAW-BY-DRAW MONOTONICITY ACROSS WARMING")

warming_wide = (
    subset_2300
    .pivot(
        index=[
            "sample_id",
            "ice_sheet",
        ],
        columns="warming",
        values="value",
    )
    .reset_index()
    .rename(
        columns={
            3.5: "value_3.5C",
            5.0: "value_5.0C",
        }
    )
)

require_columns(
    warming_wide,
    [
        "sample_id",
        "ice_sheet",
        "value_3.5C",
        "value_5.0C",
    ],
    "warming_wide",
)

warming_wide["difference_5_minus_3.5"] = (
    warming_wide["value_5.0C"]
    - warming_wide["value_3.5C"]
)

warming_wide["decreases_at_5C"] = (
    warming_wide["difference_5_minus_3.5"] < 0
)

monotonicity_summary = (
    warming_wide
    .groupby(
        "ice_sheet",
        observed=True,
    )
    .agg(
        n=("sample_id", "count"),
        n_decreasing=("decreases_at_5C", "sum"),
        percent_decreasing=(
            "decreases_at_5C",
            lambda values: 100 * values.mean(),
        ),
        mean_difference=(
            "difference_5_minus_3.5",
            "mean",
        ),
        median_difference=(
            "difference_5_minus_3.5",
            "median",
        ),
        minimum_difference=(
            "difference_5_minus_3.5",
            "min",
        ),
        maximum_difference=(
            "difference_5_minus_3.5",
            "max",
        ),
    )
    .reset_index()
)

print(
    monotonicity_summary
    .round(2)
    .to_string(index=False)
)

warming_wide.to_csv(
    OUTPUT_DIR
    / "b25_2300_draw_by_draw_warming_comparison.csv",
    index=False,
)

monotonicity_summary.to_csv(
    OUTPUT_DIR
    / "b25_2300_monotonicity_summary.csv",
    index=False,
)


# ============================================================
# 12. LARGEST WAIS DECREASES FROM 3.5°C TO 5.0°C
# ============================================================

print_heading("LARGEST WAIS DECREASES FROM 3.5°C TO 5.0°C")

largest_wais_decreases = (
    warming_wide.loc[
        warming_wide["ice_sheet"] == "WAIS"
    ]
    .sort_values(
        "difference_5_minus_3.5",
        ascending=True,
    )
    .head(N_LARGEST_DECREASES)
)

print(
    largest_wais_decreases
    .head(30)
    .round(2)
    .to_string(index=False)
)

largest_wais_decreases.to_csv(
    OUTPUT_DIR
    / "b25_2300_largest_wais_decreases_3p5C_to_5p0C.csv",
    index=False,
)


# ============================================================
# 13. TOP 3.5°C TOTAL AIS DRAWS AND SAME IDS AT 5.0°C
# ============================================================

print_heading("TOP 3.5°C TOTAL AIS DRAWS")

component_wide_by_warming = (
    subset_2300
    .pivot(
        index=[
            "sample_id",
            "warming",
        ],
        columns="ice_sheet",
        values="value",
    )
    .reset_index()
)

top_35_total = (
    component_wide_by_warming.loc[
        np.isclose(
            component_wide_by_warming["warming"],
            3.5,
        )
    ]
    .sort_values(
        "Total AIS",
        ascending=False,
    )
    .head(N_LARGEST_DRAWS)
)

print(
    top_35_total
    .head(20)
    .round(2)
    .to_string(index=False)
)

extreme_sample_ids = top_35_total[
    "sample_id"
].tolist()

same_ids_both_warmings = (
    component_wide_by_warming.loc[
        component_wide_by_warming[
            "sample_id"
        ].isin(extreme_sample_ids)
    ]
    .pivot(
        index="sample_id",
        columns="warming",
        values=[
            "EAIS",
            "WAIS",
            "Total AIS",
        ],
    )
)

same_ids_both_warmings = (
    same_ids_both_warmings
    .sort_values(
        ("Total AIS", 3.5),
        ascending=False,
    )
)

top_35_total.to_csv(
    OUTPUT_DIR
    / "b25_2300_top_3p5C_total_ais_draws.csv",
    index=False,
)

same_ids_both_warmings.to_csv(
    OUTPUT_DIR
    / "b25_2300_top_3p5C_ids_compared_at_5p0C.csv"
)


# ============================================================
# 14. QUANTILE-CURVE CROSSOVERS
# ============================================================

print_heading("QUANTILE-CURVE CROSSOVERS")

crossover_records = []

for ice_sheet in [
    "WAIS",
    "Total AIS",
]:
    values_35 = get_sorted_values(
        subset_2300,
        3.5,
        ice_sheet,
    )

    values_50 = get_sorted_values(
        subset_2300,
        5.0,
        ice_sheet,
    )

    if len(values_35) != len(values_50):
        raise ValueError(
            f"Sample counts differ for {ice_sheet}: "
            f"{len(values_35)} versus {len(values_50)}."
        )

    percentile = (
        np.arange(
            1,
            len(values_35) + 1,
        )
        / len(values_35)
    )

    difference = values_50 - values_35
    sign = np.sign(difference)

    crossing_indices = np.where(
        sign[1:] != sign[:-1]
    )[0]

    print(f"\n{ice_sheet}:")

    if len(crossing_indices) == 0:
        print("  No crossover detected.")
    else:
        for index in crossing_indices:
            record = {
                "ice_sheet": ice_sheet,
                "lower_percentile": percentile[index],
                "upper_percentile": percentile[index + 1],
                "value_3.5C_lower": values_35[index],
                "value_5.0C_lower": values_50[index],
                "value_3.5C_upper": values_35[index + 1],
                "value_5.0C_upper": values_50[index + 1],
            }

            crossover_records.append(record)

            print(
                "  Between percentile ranks "
                f"{percentile[index]:.5f} and "
                f"{percentile[index + 1]:.5f}"
            )

crossover_df = pd.DataFrame(crossover_records)

crossover_df.to_csv(
    OUTPUT_DIR
    / "b25_2300_quantile_curve_crossovers.csv",
    index=False,
)


# ============================================================
# 15. VERIFY TOTAL AIS = EAIS + WAIS
# ============================================================

print_heading("VERIFY TOTAL AIS = EAIS + WAIS")

component_check = component_wide_by_warming.copy()

component_check["recalculated_total_ais"] = (
    component_check["EAIS"]
    + component_check["WAIS"]
)

component_check["total_difference"] = (
    component_check["Total AIS"]
    - component_check["recalculated_total_ais"]
)

maximum_absolute_total_error = (
    component_check["total_difference"]
    .abs()
    .max()
)

print(
    component_check["total_difference"]
    .describe()
    .to_string()
)

print(
    "\nMaximum absolute Total AIS error:",
    maximum_absolute_total_error,
)

component_check.to_csv(
    OUTPUT_DIR
    / "b25_2300_total_ais_reconstruction_check.csv",
    index=False,
)


# ============================================================
# 16. DUPLICATE AND RECORD-COMPLETENESS CHECKS
# ============================================================

print_heading("DUPLICATE AND RECORD-COMPLETENESS CHECKS")

duplicate_mask = subset_2300.duplicated(
    subset=[
        "sample_id",
        "warming",
        "ice_sheet",
    ],
    keep=False,
)

record_counts = (
    subset_2300
    .groupby(
        [
            "sample_id",
            "warming",
        ],
        observed=True,
    )["ice_sheet"]
    .nunique()
)

bad_counts = record_counts.loc[
    record_counts != len(ICE_SHEETS)
]

print(
    "Duplicated sample-warming-component records:",
    int(duplicate_mask.sum()),
)

print("\nNumber of component records per sample-warming:")
print(
    record_counts
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    "\nSample-warming combinations without exactly "
    f"{len(ICE_SHEETS)} components:",
    len(bad_counts),
)

duplicate_records = subset_2300.loc[
    duplicate_mask
].sort_values(
    [
        "sample_id",
        "warming",
        "ice_sheet",
    ]
)

duplicate_records.to_csv(
    OUTPUT_DIR
    / "b25_2300_duplicate_records.csv",
    index=False,
)

bad_counts.rename(
    "n_components"
).reset_index().to_csv(
    OUTPUT_DIR
    / "b25_2300_incomplete_component_records.csv",
    index=False,
)


# ============================================================
# 17. ROUNDING PATTERNS IN EXTREME 3.5°C WAIS VALUES
# ============================================================

print_heading("ROUNDING PATTERNS IN EXTREME 3.5°C WAIS")

extreme_wais_values = wais_35_upper[
    "value"
].to_numpy(dtype=float)

rounding_summary = pd.DataFrame(
    {
        "rounding_test": [
            "multiple_of_10",
            "multiple_of_100",
            "multiple_of_500",
            "multiple_of_1000",
        ],
        "percent": [
            100 * np.mean(
                np.isclose(
                    np.mod(extreme_wais_values, 10),
                    0,
                )
            ),
            100 * np.mean(
                np.isclose(
                    np.mod(extreme_wais_values, 100),
                    0,
                )
            ),
            100 * np.mean(
                np.isclose(
                    np.mod(extreme_wais_values, 500),
                    0,
                )
            ),
            100 * np.mean(
                np.isclose(
                    np.mod(extreme_wais_values, 1000),
                    0,
                )
            ),
        ],
    }
)

print(
    rounding_summary
    .round(2)
    .to_string(index=False)
)

rounding_summary.to_csv(
    OUTPUT_DIR
    / "b25_2300_3p5C_wais_rounding_summary.csv",
    index=False,
)


# ============================================================
# 18. FIGURE: ORDERED TOTAL AIS DISTRIBUTIONS
# ============================================================

print_heading("CREATING FIGURES")

fig, ax = plt.subplots(
    figsize=(8, 5)
)

for warming in WARMING_LEVELS:
    values = get_sorted_values(
        subset_2300,
        warming,
        "Total AIS",
    ) / 10 #for conversion to cm

    percentile = (
        np.arange(
            1,
            len(values) + 1,
        )
        / len(values)
    )

    ax.plot(
        percentile,
        values,
        linewidth=1.5,
        label=f"{warming}°C",
    )

ax.axhline(
    0,
    linewidth=0.8,
    alpha=0.6,
)

ax.set_xlabel(
    "Empirical percentile rank"
)

ax.set_ylabel(
    f"Total AIS contribution to SLR (cm) at {TARGET_YEAR}"
)

ax.set_title(
    f"B25 Total AIS distributions at {TARGET_YEAR}"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "b25_2300_total_ais_ordered_distributions.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 19. FIGURE: UPPER 10% OF TOTAL AIS
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

for warming in WARMING_LEVELS:
    values = get_sorted_values(
        subset_2300,
        warming,
        "Total AIS",
    ) / 10

    percentile = (
        np.arange(
            1,
            len(values) + 1,
        )
        / len(values)
    )

    mask = percentile >= 0.90

    ax.plot(
        percentile[mask],
        values[mask],
        linewidth=1.5,
        label=f"{warming}°C",
    )

ax.set_xlabel(
    "Empirical percentile rank"
)

ax.set_ylabel(
    f"Total AIS contribution to SLR (cm) at {TARGET_YEAR}"
)

ax.set_title(
    f"Upper tail of B25 Total AIS at {TARGET_YEAR}"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "b25_2300_total_ais_upper_tail.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 20. FIGURE: TOTAL AIS EXCEEDANCE CURVES
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

for warming in WARMING_LEVELS:
    values = get_sorted_values(
        subset_2300,
        warming,
        "Total AIS",
    ) / 10

    exceedance_probability = (
        1.0
        - np.arange(
            1,
            len(values) + 1,
        )
        / len(values)
    )

    ax.plot(
        values,
        exceedance_probability * 100,
        linewidth=1.5,
        label=f"{warming}°C",
    )

ax.set_xlim(
    0,
    None,
)

ax.set_xlabel(
    f"{TARGET_YEAR} Total AIS contribution"
)

ax.set_ylabel(
    "Percentage of samples exceeding value"
)

ax.set_title(
    f"Upper-tail exceedance of B25 Total AIS at {TARGET_YEAR}"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "b25_2300_total_ais_exceedance_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 21. FIGURE: ORDERED WAIS DISTRIBUTIONS
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

for warming in WARMING_LEVELS:
    values = get_sorted_values(
        subset_2300,
        warming,
        "WAIS",
    ) /10

    percentile = (
        np.arange(
            1,
            len(values) + 1,
        )
        / len(values)
    )

    ax.plot(
        percentile,
        values,
        linewidth=1.5,
        label=f"{warming}°C",
    )

ax.axhline(
    0,
    linewidth=0.8,
    alpha=0.6,
)

ax.set_xlabel(
    "Empirical percentile rank"
)

ax.set_ylabel(
    f"WAIS contribution to SLR (cm) at {TARGET_YEAR}"
)

ax.set_title(
    f"B25 WAIS distributions at {TARGET_YEAR}"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "b25_2300_wais_ordered_distributions.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 22. FIGURE: UPPER 10% OF WAIS
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

for warming in WARMING_LEVELS:
    values = get_sorted_values(
        subset_2300,
        warming,
        "WAIS",
    ) / 10

    percentile = (
        np.arange(
            1,
            len(values) + 1,
        )
        / len(values)
    )

    mask = percentile >= 0.90

    ax.plot(
        percentile[mask],
        values[mask],
        linewidth=1.5,
        label=f"{warming}°C",
    )

ax.set_xlabel(
    "Empirical percentile rank"
)

ax.set_ylabel(
    f"WAIS contribution to SLR (cm) at {TARGET_YEAR}"
)

ax.set_title(
    f"Upper tail of B25 WAIS at {TARGET_YEAR}"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "b25_2300_wais_upper_tail.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print_heading("FINISHED")

print(
    f"Diagnostic tables saved to:\n{OUTPUT_DIR}"
)

print(
    f"\nFigures saved to:\n{FIGURE_DIR}"
)

print(
    "\nKey validation values:"
)

print(
    f"  Maximum absolute Total AIS reconstruction error: "
    f"{maximum_absolute_total_error}"
)

print(
    f"  Duplicate records: {int(duplicate_mask.sum())}"
)

print(
    f"  Incomplete sample-warming records: {len(bad_counts)}"
)
