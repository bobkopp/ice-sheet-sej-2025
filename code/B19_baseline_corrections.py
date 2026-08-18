#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
ADJUST B19 PROCESS-LEVEL AND TOTAL MONTE CARLO SAMPLES TO THE
BASELINE REFERENCE FRAME USED IN THE B25 ELICITATION

Purpose:
  B19 experts estimated deviations from current trends.
  B25 experts were given current-trend baseline contributions
  and incorporated those contributions into their estimates.

  This script adds the B25 baseline contributions to every
  B19 process-level, allowing an
  apples-to-apples comparison between B19 and B25.

Input:
  USUK_PW01_mega_spreadsheet.xlsx

Output:
  USUK_PW01_B19_process_and_total_MC_adjusted_to_B25_baseline.xlsx

Units:
  Adjusted Monte Carlo samples remain in mm GMSL.
  Summary statistics are reported in cm GMSL.

Column examples:
  PW01EA_2100L
  PW01GD_2200H
  PW01WR_2300L
  USEU_PW01_E_2100L
  USEU_PW01_G_2200H

Column structures:
  Process: PW01 + ice-sheet code + process code + "_" + year + scenario
  Total:   USEU_PW01_ + ice-sheet code + "_" + year + scenario

Ice sheets:
  E = EAIS
  G = Greenland
  W = WAIS

Processes:
  A = Accumulation
  D = Discharge
  R = Runoff
  Total columns use USEU_PW01_E/G/W

Scenarios:
  L = Low
  H = High
============================================================

@author: elizabeth
"""


from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = Path(
    "/Users/elizabeth/Downloads/"
    "USUK_XLS_corr_271119/"
    "USUK_PW01_mega_spreadsheet.xlsx"
)

output_file = input_file.with_name(
    "B19_process_and_total_MC_adjusted_to_B25_baseline.xlsx"
)


# ============================================================
# 2. UNIT CONVERSION
# ============================================================

# Conversion explicitly provided to the B25 experts:
#
#     mm GMSL = Gt / -362.5
#
# The negative denominator converts ice-sheet mass balance into
# sea-level contribution:
#
#   positive accumulation -> negative SLR contribution
#   negative discharge    -> positive SLR contribution
#   negative runoff       -> positive SLR contribution

GT_PER_MM_GMSL = -362.5

MM_PER_CM = 10.0


# ============================================================
# 3. B25 BASELINE CONTRIBUTIONS IN GT
# ============================================================

# These are the cumulative process contributions relative to the
# 1995–2014 baseline through 2100, 2200, and 2300.
#
# Values retain the mass-balance signs used in the elicitation:
#
#   Accumulation is positive mass gain.
#   Runoff and discharge are negative mass loss.
#
# The same baseline correction applies to both the B19 Low and
# High scenarios.

baseline_gt = {
    "EAIS": {
        "Accumulation": {
            2100: 102125.0,
            2200: 209625.0,
            2300: 317125.0,
        },
        "Discharge": {
            2100: -104975.0,
            2200: -215475.0,
            2300: -325975.0,
        },
        "Total": {
            2100: -2850.0,
            2200: -5850.0,
            2300: -8850.0,
        },
    },

    "GrIS": {
        "Accumulation": {
            2100: 61750.0,
            2200: 126750.0,
            2300: 191750.0,
        },
        "Runoff": {
            2100: -47500.0,
            2200: -97500.0,
            2300: -147500.0,
        },
        "Discharge": {
            2100: -44650.0,
            2200: -91650.0,
            2300: -138650.0,
        },
        "Total": {
            2100: -30400.0,
            2200: -62400.0,
            2300: -94400.0,
        },
    },

    "WAIS": {
        "Accumulation": {
            2100: 92150.0,
            2200: 189150.0,
            2300: 286150.0,
        },
        "Runoff": {
            2100: 0.0,
            2200: 0.0,
            2300: 0.0,
        },
        "Discharge": {
            2100: -101650.0,
            2200: -208650.0,
            2300: -315650.0,
        },
        "Total": {
            2100: -9500.0,
            2200: -19500.0,
            2300: -29500.0,
        },
    },
}


# ============================================================
# 4. COLUMN CODE MAPPINGS
# ============================================================

ice_sheet_map = {
    "E": "EAIS",
    "G": "GrIS",
    "W": "WAIS",
}

process_map = {
    "A": "Accumulation",
    "D": "Discharge",
    "R": "Runoff",
}

scenario_map = {
    "L": "Low",
    "H": "High",
}


# Expected process-column format:
#
#     PW01EA_2100L
#
# Expected total-column format:
#
#     USEU_PW01_E_2100L

process_column_pattern = re.compile(
    r"^PW01"
    r"(?P<ice_sheet>[EGW])"
    r"(?P<process>[ADR])"
    r"_"
    r"(?P<year>2100|2200|2300)"
    r"(?P<scenario>[LH])"
    r"$"
)

total_column_pattern = re.compile(
    r"^USEU_PW01_"
    r"(?P<ice_sheet>[EGW])"
    r"_"
    r"(?P<year>2100|2200|2300)"
    r"(?P<scenario>[LH])"
    r"$"
)


# ============================================================
# 5. PARSE A COLUMN AND FIND ITS CORRECTION
# ============================================================

def get_baseline_correction(column_name: str) -> dict:
    """
    Parse one B19 process-level or total column and return the
    corresponding B25 baseline correction.

    The correction is calculated in mm GMSL using:

        correction_mm = signed_baseline_gt / -362.5
    """

    clean_column_name = str(column_name).strip()

    process_match = process_column_pattern.fullmatch(clean_column_name)
    total_match = total_column_pattern.fullmatch(clean_column_name)

    if process_match is not None:
        match = process_match
        process = process_map[match.group("process")]
        column_type = "Process"
    elif total_match is not None:
        match = total_match
        process = "Total"
        column_type = "Total"
    else:
        raise ValueError(
            "Column does not match an expected B19 format:\n"
            f"{clean_column_name}"
        )

    ice_sheet = ice_sheet_map[match.group("ice_sheet")]
    year = int(match.group("year"))
    scenario = scenario_map[match.group("scenario")]

    try:
        signed_baseline_gt = baseline_gt[ice_sheet][process][year]
    except KeyError as error:
        raise KeyError(
            "\nNo baseline correction was defined for:\n"
            f"  Column: {clean_column_name}\n"
            f"  Ice sheet: {ice_sheet}\n"
            f"  Quantity: {process}\n"
            f"  Year: {year}"
        ) from error

    correction_mm = signed_baseline_gt / GT_PER_MM_GMSL
    correction_cm = correction_mm / MM_PER_CM

    return {
        "column": clean_column_name,
        "column_type": column_type,
        "ice_sheet": ice_sheet,
        "process": process,
        "year": year,
        "scenario": scenario,
        "signed_baseline_gt": signed_baseline_gt,
        "conversion_gt_per_mm": GT_PER_MM_GMSL,
        "correction_mm": correction_mm,
        "correction_cm": correction_cm,
    }


# ============================================================
# 6. READ THE B19 MEGA-SPREADSHEET
# ============================================================

if not input_file.exists():
    raise FileNotFoundError(
        "Could not find the input workbook:\n"
        f"{input_file}"
    )


raw_data = pd.read_excel(input_file)

raw_data.columns = [
    str(column).strip()
    for column in raw_data.columns
]


print(
    "\nRead B19 mega-spreadsheet:"
)

print(
    f"  Rows / Monte Carlo samples: "
    f"{raw_data.shape[0]:,}"
)

print(
    f"  Columns: "
    f"{raw_data.shape[1]:,}"
)


# ============================================================
# 7. IDENTIFY PROCESS COLUMNS
# ============================================================

matched_columns = [
    column
    for column in raw_data.columns
    if (
        process_column_pattern.fullmatch(column)
        or total_column_pattern.fullmatch(column)
    )
]

unmatched_columns = [
    column
    for column in raw_data.columns
    if not (
        process_column_pattern.fullmatch(column)
        or total_column_pattern.fullmatch(column)
    )
]


if not matched_columns:
    raise ValueError(
        "No columns matched the expected PW01 column format."
    )


print(
    f"\nMatched process columns: "
    f"{len(matched_columns)}"
)


# Mega-spreadsheet should contain:
#
#   (8 process columns + 3 total columns)
#   × 3 years
#   × 2 scenarios
#   = 66 columns

expected_number_of_columns = 66

if len(matched_columns) != expected_number_of_columns:
    print(
        "\nWARNING:"
        f" Expected {expected_number_of_columns} matched process/total columns, "
        f"but found {len(matched_columns)}."
    )


if unmatched_columns:
    print(
        "\nThese columns did not match the PW01 format and will "
        "be copied without modification:"
    )

    for column in unmatched_columns:
        print(f"  {column}")


# ============================================================
# 8. BUILD THE CORRECTION TABLE
# ============================================================

correction_records = [
    get_baseline_correction(column)
    for column in matched_columns
]

correction_table = pd.DataFrame(
    correction_records
)


# Add sort-order variables for a clean output table.

ice_sheet_order = {
    "EAIS": 0,
    "WAIS": 1,
    "GrIS": 2,
}

process_order = {
    "Discharge": 0,
    "Runoff": 1,
    "Accumulation": 2,
    "Total": 3,
}

scenario_order = {
    "Low": 0,
    "High": 1,
}


correction_table["_ice_order"] = (
    correction_table["ice_sheet"].map(ice_sheet_order)
)

correction_table["_process_order"] = (
    correction_table["process"].map(process_order)
)

correction_table["_scenario_order"] = (
    correction_table["scenario"].map(scenario_order)
)


correction_table = (
    correction_table
    .sort_values(
        [
            "year",
            "_scenario_order",
            "_ice_order",
            "_process_order",
        ]
    )
    .drop(
        columns=[
            "_ice_order",
            "_process_order",
            "_scenario_order",
        ]
    )
    .reset_index(drop=True)
)


print(
    "\nBaseline corrections that will be added to every "
    "Monte Carlo sample:"
)

print(
    correction_table[
        [
            "column",
            "column_type",
            "ice_sheet",
            "process",
            "year",
            "scenario",
            "signed_baseline_gt",
            "correction_mm",
            "correction_cm",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# ============================================================
# 9. APPLY THE CORRECTIONS
# ============================================================

adjusted_data = raw_data.copy()


for record in correction_records:

    column = record["column"]
    correction_mm = record["correction_mm"]

    raw_values = pd.to_numeric(
        raw_data[column],
        errors="coerce",
    )

    # Identify any cells that contained nonblank, nonnumeric data.
    invalid_mask = (
        raw_data[column].notna()
        & raw_values.isna()
    )

    if invalid_mask.any():

        invalid_count = int(
            invalid_mask.sum()
        )

        print(
            f"\nWARNING: {column} contains "
            f"{invalid_count} nonblank cells that could not be "
            "converted to numbers. Those cells will be blank "
            "in the adjusted output."
        )

    # Add the same year/process/ice-sheet baseline correction
    # to every valid Monte Carlo sample in this column.

    adjusted_data[column] = (
        raw_values + correction_mm
    )


# ============================================================
# 10. VALIDATE SAMPLE-LEVEL CORRECTIONS
# ============================================================

validation_records = []


for record in correction_records:

    column = record["column"]
    expected_correction_mm = record["correction_mm"]

    raw_values = pd.to_numeric(
        raw_data[column],
        errors="coerce",
    )

    adjusted_values = pd.to_numeric(
        adjusted_data[column],
        errors="coerce",
    )

    observed_differences = (
        adjusted_values - raw_values
    )

    valid_differences = (
        observed_differences.dropna()
    )

    if len(valid_differences) == 0:

        minimum_change_mm = np.nan
        maximum_change_mm = np.nan
        validation_passed = False

    else:

        minimum_change_mm = (
            valid_differences.min()
        )

        maximum_change_mm = (
            valid_differences.max()
        )

        validation_passed = np.allclose(
            valid_differences,
            expected_correction_mm,
            rtol=0,
            atol=1e-9,
        )

    validation_records.append(
        {
            "column": column,
            "column_type": record["column_type"],
            "ice_sheet": record["ice_sheet"],
            "process": record["process"],
            "year": record["year"],
            "scenario": record["scenario"],
            "expected_correction_mm": (
                expected_correction_mm
            ),
            "observed_minimum_change_mm": (
                minimum_change_mm
            ),
            "observed_maximum_change_mm": (
                maximum_change_mm
            ),
            "validation_passed": validation_passed,
        }
    )


validation_table = pd.DataFrame(
    validation_records
)


if validation_table["validation_passed"].all():

    print(
        "\nValidation passed: every valid Monte Carlo sample "
        "was shifted by the expected baseline correction."
    )

else:

    failed_validation = validation_table[
        ~validation_table["validation_passed"]
    ]

    raise ValueError(
        "\nValidation failed for the following columns:\n\n"
        f"{failed_validation.to_string(index=False)}"
    )


# ============================================================
# 11. CALCULATE SUMMARY STATISTICS IN CM
# ============================================================


quantiles = {
    "p05": 0.05,
    "p17": 0.17,
    "p50": 0.50,
    "p83": 0.83,
    "p95": 0.95,
}


summary_records = []


for record in correction_records:

    column = record["column"]

    raw_values_mm = pd.to_numeric(
        raw_data[column],
        errors="coerce",
    )

    adjusted_values_mm = pd.to_numeric(
        adjusted_data[column],
        errors="coerce",
    )

    # Convert only the summary statistics to cm.
    raw_values_cm = (
        raw_values_mm / MM_PER_CM
    )

    adjusted_values_cm = (
        adjusted_values_mm / MM_PER_CM
    )

    summary_record = {
        "column": column,
        "column_type": record["column_type"],
        "ice_sheet": record["ice_sheet"],
        "process": record["process"],
        "year": record["year"],
        "scenario": record["scenario"],
        "n_samples": int(
            raw_values_mm.notna().sum()
        ),
        "baseline_correction_mm": (
            record["correction_mm"]
        ),
        "baseline_correction_cm": (
            record["correction_cm"]
        ),
        "raw_mean_cm": raw_values_cm.mean(),
        "adjusted_mean_cm": adjusted_values_cm.mean(),
    }

    for label, probability in quantiles.items():

        summary_record[f"raw_{label}_cm"] = (
            raw_values_cm.quantile(probability)
        )

        summary_record[f"adjusted_{label}_cm"] = (
            adjusted_values_cm.quantile(probability)
        )

    summary_records.append(
        summary_record
    )


summary_table = pd.DataFrame(
    summary_records
)


summary_table["_ice_order"] = (
    summary_table["ice_sheet"].map(ice_sheet_order)
)

summary_table["_process_order"] = (
    summary_table["process"].map(process_order)
)

summary_table["_scenario_order"] = (
    summary_table["scenario"].map(scenario_order)
)


summary_table = (
    summary_table
    .sort_values(
        [
            "year",
            "_scenario_order",
            "_ice_order",
            "_process_order",
        ]
    )
    .drop(
        columns=[
            "_ice_order",
            "_process_order",
            "_scenario_order",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 12. CREATE A MANUSCRIPT-STYLE SUMMARY
# ============================================================

def format_median_range(
    median,
    lower,
    upper,
    decimals=1,
):
    """
    Format a value as:

        median (lower to upper)
    """

    if any(
        pd.isna(value)
        for value in [median, lower, upper]
    ):
        return ""

    return (
        f"{median:.{decimals}f} "
        f"({lower:.{decimals}f} to "
        f"{upper:.{decimals}f})"
    )


manuscript_records = []


for _, row in summary_table.iterrows():

    manuscript_records.append(
        {
            "column": row["column"],
            "ice_sheet": row["ice_sheet"],
            "process": row["process"],
            "year": row["year"],
            "scenario": row["scenario"],
            "baseline_correction_cm": (
                row["baseline_correction_cm"]
            ),
            "B19_raw_cm": format_median_range(
                median=row["raw_p50_cm"],
                lower=row["raw_p05_cm"],
                upper=row["raw_p95_cm"],
                decimals=1,
            ),
            "B19_adjusted_to_B25_baseline_cm": (
                format_median_range(
                    median=row["adjusted_p50_cm"],
                    lower=row["adjusted_p05_cm"],
                    upper=row["adjusted_p95_cm"],
                    decimals=1,
                )
            ),
        }
    )


manuscript_summary = pd.DataFrame(
    manuscript_records
)


# ============================================================
# 13. PROCESS-SUM VERSUS TOTAL-CORRECTION CHECK
# ============================================================

# Confirm that the sum of the process-level corrections equals
# the independently supplied total correction for each ice sheet,
# year, and scenario.

correction_pivot = correction_table.pivot_table(
    index=[
        "ice_sheet",
        "year",
        "scenario",
    ],
    columns="process",
    values=[
        "correction_mm",
        "correction_cm",
    ],
    aggfunc="first",
).fillna(0.0)

correction_pivot.columns = [
    f"{unit}_{process}"
    for unit, process in correction_pivot.columns
]

baseline_sum_check = correction_pivot.reset_index()

for unit in ["correction_mm", "correction_cm"]:
    process_columns = [
        f"{unit}_Accumulation",
        f"{unit}_Discharge",
        f"{unit}_Runoff",
    ]

    for column in process_columns:
        if column not in baseline_sum_check.columns:
            baseline_sum_check[column] = 0.0

    total_column = f"{unit}_Total"
    baseline_sum_check[f"{unit}_summed_processes"] = (
        baseline_sum_check[process_columns].sum(axis=1)
    )
    baseline_sum_check[f"{unit}_difference"] = (
        baseline_sum_check[f"{unit}_summed_processes"]
        - baseline_sum_check[total_column]
    )

baseline_sum_check["check_passed"] = np.isclose(
    baseline_sum_check["correction_mm_difference"],
    0.0,
    rtol=0,
    atol=1e-9,
)


# ============================================================
# 14. SAVE THE OUTPUT WORKBOOK
# ============================================================

with pd.ExcelWriter(
    output_file,
    engine="openpyxl",
) as writer:

    # Adjusted Monte Carlo samples remain in mm.
    adjusted_data.to_excel(
        writer,
        sheet_name="B19_adjusted_MC_mm",
        index=False,
    )

    # Exact corrections used for every column.
    correction_table.to_excel(
        writer,
        sheet_name="B19_corrections",
        index=False,
    )

    # Numerical summary statistics in cm.
    summary_table.to_excel(
        writer,
        sheet_name="B19_summary_cm",
        index=False,
    )

    # Compact median (5th to 95th percentile) strings.
    manuscript_summary.to_excel(
        writer,
        sheet_name="B19_manuscript_summary",
        index=False,
    )

    # Validation that every draw received the correct shift.
    validation_table.to_excel(
        writer,
        sheet_name="B19_validation",
        index=False,
    )

    # Sum of process corrections by ice sheet/year/scenario.
    baseline_sum_check.to_excel(
        writer,
        sheet_name="Correction_sum_check",
        index=False,
    )


# ============================================================
# 15. FINAL CONSOLE OUTPUT
# ============================================================

print(
    "\nCreated adjusted B19 process-level and total workbook:\n"
    f"{output_file}"
)

print(
    "\nWorkbook sheets:"
)

print(
    "  B19_adjusted_MC_mm"
    "          - corrected process and total Monte Carlo draws in mm"
)

print(
    "  B19_corrections"
    "            - baseline correction for each column"
)

print(
    "  B19_summary_cm"
    "             - numerical summary statistics in cm"
)

print(
    "  B19_manuscript_summary"
    "     - median (5th to 95th) in cm"
)

print(
    "  B19_validation"
    "             - sample-level correction checks"
)

print(
    "  Correction_sum_check"
    "         - summed process corrections"
)