#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:02:36 2026

Create a combined B25-B19 process-comparison table using
B19 process-level and total Monte Carlo samples adjusted to the B25 baseline.

Each data cell reports:
    B25: 50th percentile (5th to 95th percentile)
    B19 adjusted to B25 baseline:
         50th percentile (5th to 95th percentile)

B25 appears on the first line in black.
Adjusted B19 appears on the second line in gray where available.

B19 Low is compared with B25 2.0°C.
B19 High is compared with B25 5.0°C.
B25 3.5°C has no B19 comparison.

Processes shown:
    Discharge
    Runoff
    Accumulation
    Surface mass balance (Runoff + Accumulation)
    Total

B25 source values and adjusted B19 Monte Carlo values are in mm.
All reported values are converted to cm.

@author: elizabeth
"""

import os

import numpy as np
import pandas as pd

from B25data_processing import (
    load_and_prepare_data,
    project_dir,
)


# ============================================================
# 1. SETTINGS
# ============================================================

# Adjusted B19 workbook created by the baseline-correction script.
b19_file = (
    "/Users/elizabeth/Downloads/"
    "USUK_XLS_corr_271119/"
    "B19_process_and_total_MC_adjusted_to_B25_baseline.xlsx"
)

# Adjusted Monte Carlo samples are stored on this worksheet.
b19_sheet_name = "B19_adjusted_MC_mm"


output_dir = os.path.join(
    project_dir,
    "output",
    "tables",
)

output_file = os.path.join(
    output_dir,
    "B25_B19_Table2_process_comparison_"
    "B19_adjusted_to_B25_baseline_UPDATED.xlsx",
)

os.makedirs(
    output_dir,
    exist_ok=True,
)


years = [
    2100,
    2200,
    2300,
]

warming_order = [
    2.0,
    3.5,
    5.0,
]

# B19 contains only Low and High scenarios.
b19_scenario_for_warming = {
    2.0: "L",
    5.0: "H",
}

ice_sheets = [
    "EAIS",
    "WAIS",
    "GrIS",
]

table_groups = [
    "Discharge",
    "Runoff",
    "Accumulation",
    "Runoff + Accumulation",
    "Total",
]

# Process names stored in the prepared B25 dataframe.
b25_process_name = {
    "Discharge": "Discharge",
    "Runoff": "Runoff",
    "Accumulation": "Accumulation",
    "Runoff + Accumulation": "Runoff minus accumulation",
}


# B19 column codes.
b19_ice_code = {
    "EAIS": "E",
    "WAIS": "W",
    "GrIS": "G",
}


# ============================================================
# 2. FORMATTING HELPERS
# ============================================================

def fmt(value):
    """
    Round to a whole centimeter and replace the standard minus
    sign with a Unicode minus sign.
    """

    if pd.isna(value):
        return ""

    return f"{value:.0f}".replace(
        "-",
        "−",
    )


def summarize_samples(samples):
    """
    Return a distribution as:

        median (5th to 95th percentile)
    """

    samples = np.asarray(
        samples,
        dtype=float,
    )

    samples = samples[
        np.isfinite(samples)
    ]

    if samples.size == 0:
        return ""

    q05, q50, q95 = np.percentile(
        samples,
        [
            5,
            50,
            95,
        ],
    )

    return (
        f"{fmt(q50)} "
        f"({fmt(q05)} to {fmt(q95)})"
    )


# ============================================================
# 3. LOAD AND SUMMARIZE B25
# ============================================================

data = load_and_prepare_data()

process_ext = data["process_ext"].copy()
mc_total_ext = data["mc_total_ext"].copy()


# B25 source values are in mm.
# Convert to cm before calculating summary statistics.

process_ext["value"] = (
    pd.to_numeric(
        process_ext["value"],
        errors="coerce",
    )
    / 10.0
)

mc_total_ext["value"] = (
    pd.to_numeric(
        mc_total_ext["value"],
        errors="coerce",
    )
    / 10.0
)


def create_b25_summary_lookup():
    """
    Create a lookup with keys:

        (year, warming, table_group, ice_sheet)

    and formatted summary strings as values.
    """

    lookup = {}

    # --------------------------------------------------------
    # Process-level quantities
    # --------------------------------------------------------

    for table_group, process_name in b25_process_name.items():

        subset = process_ext.loc[
            process_ext["process"] == process_name
        ]

        grouped = subset.groupby(
            [
                "year",
                "warming",
                "ice_sheet",
            ],
            observed=True,
        )["value"]

        for (
            year,
            warming,
            ice_sheet,
        ), samples in grouped:

            year = int(year)
            warming = float(warming)

            if (
                year in years
                and warming in warming_order
                and ice_sheet in ice_sheets
            ):
                lookup[
                    (
                        year,
                        warming,
                        table_group,
                        ice_sheet,
                    )
                ] = summarize_samples(
                    samples.to_numpy()
                )

    # --------------------------------------------------------
    # Total ice-sheet contributions
    # --------------------------------------------------------

    total_subset = mc_total_ext.loc[
        mc_total_ext["ice_sheet"].isin(
            ice_sheets
        )
    ]

    total_grouped = total_subset.groupby(
        [
            "year",
            "warming",
            "ice_sheet",
        ],
        observed=True,
    )["value"]

    for (
        year,
        warming,
        ice_sheet,
    ), samples in total_grouped:

        year = int(year)
        warming = float(warming)

        if (
            year in years
            and warming in warming_order
            and ice_sheet in ice_sheets
        ):
            lookup[
                (
                    year,
                    warming,
                    "Total",
                    ice_sheet,
                )
            ] = summarize_samples(
                samples.to_numpy()
            )

    return lookup


b25_lookup = create_b25_summary_lookup()


# ============================================================
# 4. LOAD ADJUSTED B19 MONTE CARLO DATA
# ============================================================

if not os.path.exists(b19_file):
    raise FileNotFoundError(
        "Adjusted B19 process-and-total workbook not found:\n"
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
    "\nLoaded adjusted B19 process and total Monte Carlo samples:"
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


# ============================================================
# 5. VALIDATE REQUIRED B19 COLUMNS
# ============================================================

def b19_column_name(
    year,
    suffix,
    code,
):
    """
    Construct a B19 column name.

    Example:
        year=2100, suffix='L', code='EA'
        -> PW01EA_2100L
    """

    return (
        f"PW01{code}_"
        f"{year}{suffix}"
    )


def b19_total_column_name(
    year,
    suffix,
    ice_code,
):
    """
    Construct a B19 total ice-sheet column name.

    Example:
        year=2100, suffix='L', ice_code='E'
        -> USEU_PW01_E_2100L
    """

    return (
        f"USEU_PW01_{ice_code}_"
        f"{year}{suffix}"
    )


required_b19_codes = [
    "EA",
    "ED",
    "GA",
    "GD",
    "GR",
    "WA",
    "WD",
    "WR",
]


required_b19_process_columns = {
    b19_column_name(
        year=year,
        suffix=suffix,
        code=code,
    )
    for year in years
    for suffix in [
        "L",
        "H",
    ]
    for code in required_b19_codes
}

required_b19_total_columns = {
    b19_total_column_name(
        year=year,
        suffix=suffix,
        ice_code=ice_code,
    )
    for year in years
    for suffix in [
        "L",
        "H",
    ]
    for ice_code in [
        "E",
        "G",
        "W",
    ]
}

required_b19_columns = (
    required_b19_process_columns
    | required_b19_total_columns
)


missing_b19_columns = (
    required_b19_columns
    - set(b19_data.columns)
)


if missing_b19_columns:
    raise ValueError(
        "The adjusted B19 workbook is missing these columns:\n"
        + "\n".join(
            sorted(missing_b19_columns)
        )
    )


for column in required_b19_columns:

    b19_data[column] = pd.to_numeric(
        b19_data[column],
        errors="coerce",
    )


# ============================================================
# 6. GET ROW-ALIGNED B19 ARRAYS
# ============================================================

def aligned_b19_arrays(
    year,
    suffix,
    codes,
):
    """
    Return row-aligned adjusted B19 arrays in centimeters.

    The source worksheet contains adjusted Monte Carlo values
    in mm. Values are converted to cm here.

    Rows with a missing value in any requested column are removed
    together before draw-by-draw arithmetic.
    """

    names = [
        b19_column_name(
            year=year,
            suffix=suffix,
            code=code,
        )
        for code in codes
    ]

    aligned = (
        b19_data[names]
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
        return None

    return {
        code: (
            aligned[column_name].to_numpy()
            / 10.0
        )
        for code, column_name in zip(
            codes,
            names,
        )
    }


# ============================================================
# 7. BUILD B19 PROCESS DISTRIBUTIONS
# ============================================================

def get_b19_samples(
    year,
    suffix,
    table_group,
    ice_sheet,
):
    """
    Return one adjusted B19 distribution in centimeters.

    The B19 process and total columns have already been
    adjusted to the B25 baseline.

    Accumulation is stored as a signed SLR contribution, so:

        surface mass balance = runoff + accumulation

    No EAIS runoff column exists in B19. Therefore:
        EAIS Runoff is blank.
        EAIS Runoff + Accumulation is blank.

    B19 Total is read directly from the corrected USEU_PW01_E/G/W
    Monte Carlo columns. It is not reconstructed by summing the
    process-level columns.
    """

    ice_code = b19_ice_code[ice_sheet]

    # --------------------------------------------------------
    # Discharge
    # --------------------------------------------------------

    if table_group == "Discharge":

        code = f"{ice_code}D"

        arrays = aligned_b19_arrays(
            year=year,
            suffix=suffix,
            codes=[code],
        )

        if arrays is None:
            return None

        return arrays[code]

    # --------------------------------------------------------
    # Accumulation
    # --------------------------------------------------------

    if table_group == "Accumulation":

        code = f"{ice_code}A"

        arrays = aligned_b19_arrays(
            year=year,
            suffix=suffix,
            codes=[code],
        )

        if arrays is None:
            return None

        return arrays[code]

    # --------------------------------------------------------
    # Runoff
    # --------------------------------------------------------

    if table_group == "Runoff":

        if ice_sheet == "EAIS":
            return None

        code = f"{ice_code}R"

        arrays = aligned_b19_arrays(
            year=year,
            suffix=suffix,
            codes=[code],
        )

        if arrays is None:
            return None

        return arrays[code]

    # --------------------------------------------------------
    # Surface mass balance
    # --------------------------------------------------------

    if table_group == "Runoff + Accumulation":

        if ice_sheet == "EAIS":
            return None

        runoff_code = f"{ice_code}R"
        accumulation_code = f"{ice_code}A"

        arrays = aligned_b19_arrays(
            year=year,
            suffix=suffix,
            codes=[
                runoff_code,
                accumulation_code,
            ],
        )

        if arrays is None:
            return None

        return (
            arrays[runoff_code]
            + arrays[accumulation_code]
        )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    if table_group == "Total":

        column_name = b19_total_column_name(
            year=year,
            suffix=suffix,
            ice_code=ice_code,
        )

        values = (
            pd.to_numeric(
                b19_data[column_name],
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
            .to_numpy(dtype=float)
            / 10.0
        )

        if values.size == 0:
            return None

        return values


    raise ValueError(
        "Unsupported B19 table group: "
        f"{table_group}"
    )


# ============================================================
# 8. CREATE B19 SUMMARY LOOKUP
# ============================================================

def create_b19_summary_lookup():
    """
    Create a lookup with keys:

        (year, warming, table_group, ice_sheet)

    B19 Low is assigned to B25 2.0°C.
    B19 High is assigned to B25 5.0°C.
    """

    lookup = {}

    for year in years:

        for (
            warming,
            suffix,
        ) in b19_scenario_for_warming.items():

            for table_group in table_groups:

                for ice_sheet in ice_sheets:

                    samples = get_b19_samples(
                        year=year,
                        suffix=suffix,
                        table_group=table_group,
                        ice_sheet=ice_sheet,
                    )

                    if samples is None:
                        continue

                    lookup[
                        (
                            year,
                            warming,
                            table_group,
                            ice_sheet,
                        )
                    ] = summarize_samples(
                        samples
                    )

    return lookup


b19_lookup = create_b19_summary_lookup()


# ============================================================
# 9. NUMERICAL VALIDATION TABLE
# ============================================================

def create_validation_table():
    """
    Create a long-form table of the values that will appear in
    the formatted comparison spreadsheet.

    This is useful for checking the B25 and adjusted B19 summaries
    without relying only on the publication-style worksheet.
    """

    records = []

    for year in years:

        for warming in warming_order:

            for table_group in table_groups:

                for ice_sheet in ice_sheets:

                    key = (
                        year,
                        warming,
                        table_group,
                        ice_sheet,
                    )

                    records.append(
                        {
                            "year": year,
                            "warming_C": warming,
                            "table_group": table_group,
                            "ice_sheet": ice_sheet,
                            "B25_summary_cm": b25_lookup.get(
                                key,
                                "",
                            ),
                            "B19_adjusted_summary_cm": (
                                b19_lookup.get(
                                    key,
                                    "",
                                )
                            ),
                        }
                    )

    return pd.DataFrame(records)


validation_summary = create_validation_table()


# ============================================================
# 10. EXPORT COMPARISON TABLE
# ============================================================

def export_comparison_table(
    b25_values,
    b19_values,
    validation_values,
    output_path,
):
    """
    Export a publication-style Excel comparison table.

    B25 is shown on the first line in black.

    B19 adjusted to the B25 baseline is shown below in gray where
    a matching B19 Low or High scenario exists.
    """

    first_data_column = 1

    columns_per_group = len(
        ice_sheets
    )

    final_column = (
        first_data_column
        + len(table_groups)
        * columns_per_group
        - 1
    )


    with pd.ExcelWriter(
        output_path,
        engine="xlsxwriter",
    ) as writer:

        workbook = writer.book

        worksheet = workbook.add_worksheet(
            "B25-B19 Comparison"
        )

        writer.sheets[
            "B25-B19 Comparison"
        ] = worksheet


        # ----------------------------------------------------
        # Formats
        # ----------------------------------------------------

        title_format = workbook.add_format({
            "bold": True,
            "font_size": 12,
            "align": "left",
            "valign": "vcenter",
        })

        note_format = workbook.add_format({
            "italic": True,
            "font_size": 10,
            "align": "left",
            "valign": "vcenter",
            "text_wrap": True,
        })

        group_header_format = workbook.add_format({
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "top": 1,
            "bottom": 1,
            "text_wrap": True,
        })

        subheader_format = workbook.add_format({
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "bottom": 1,
        })

        year_format = workbook.add_format({
            "bold": True,
            "align": "left",
            "valign": "vcenter",
            "top": 1,
        })

        warming_format = workbook.add_format({
            "align": "left",
            "valign": "vcenter",
        })

        warming_bottom_format = workbook.add_format({
            "align": "left",
            "valign": "vcenter",
            "bottom": 1,
        })

        cell_format = workbook.add_format({
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        })

        cell_bottom_format = workbook.add_format({
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "bottom": 1,
        })

        b25_text_format = workbook.add_format({
            "font_color": "#000000",
        })

        b19_text_format = workbook.add_format({
            "font_color": "#808080",
        })


        # ----------------------------------------------------
        # Title and note
        # ----------------------------------------------------

        worksheet.merge_range(
            0,
            0,
            0,
            final_column,
            (
                "Table 2. B25 and baseline-adjusted B19 "
                "projected sea-level-rise contributions "
                "by ice sheet and process (cm)"
            ),
            title_format,
        )

        worksheet.merge_range(
            1,
            0,
            1,
            final_column,
            (
                "Values are the 50th percentile "
                "(5th to 95th percentile). "
                "B25 is shown on the first line. "
                "B19, adjusted to the baseline used in the B25 "
                "elicitation, is shown below in gray for the "
                "2.0°C/Low and 5.0°C/High comparisons. "
                "B19 does not provide EAIS runoff."
            ),
            note_format,
        )


        # ----------------------------------------------------
        # Column headers
        # ----------------------------------------------------

        worksheet.merge_range(
            3,
            0,
            4,
            0,
            "Year and warming",
            group_header_format,
        )

        start_column = first_data_column

        for table_group in table_groups:

            end_column = (
                start_column
                + columns_per_group
                - 1
            )

            if table_group == "Runoff + Accumulation":
                display_label = (
                    "Surface mass balance\n"
                    "(Runoff + Accumulation)"
                )
            else:
                display_label = table_group

            worksheet.merge_range(
                3,
                start_column,
                3,
                end_column,
                display_label,
                group_header_format,
            )

            for offset, ice_sheet in enumerate(
                ice_sheets
            ):
                worksheet.write(
                    4,
                    start_column + offset,
                    ice_sheet,
                    subheader_format,
                )

            start_column = (
                end_column + 1
            )


        # ----------------------------------------------------
        # Table body
        # ----------------------------------------------------

        output_row = 5

        for year in years:

            worksheet.merge_range(
                output_row,
                0,
                output_row,
                final_column,
                int(year),
                year_format,
            )

            output_row += 1

            for warming_index, warming in enumerate(
                warming_order
            ):

                is_last_warming = (
                    warming_index
                    == len(warming_order) - 1
                )

                row_label_format = (
                    warming_bottom_format
                    if is_last_warming
                    else warming_format
                )

                value_cell_format = (
                    cell_bottom_format
                    if is_last_warming
                    else cell_format
                )

                worksheet.write(
                    output_row,
                    0,
                    f"{warming:.1f}°C",
                    row_label_format,
                )

                output_column = first_data_column

                for table_group in table_groups:

                    for ice_sheet in ice_sheets:

                        key = (
                            year,
                            warming,
                            table_group,
                            ice_sheet,
                        )

                        b25_text = b25_values.get(
                            key,
                            "",
                        )

                        b19_text = b19_values.get(
                            key,
                            "",
                        )

                        if b25_text and b19_text:

                            worksheet.write_rich_string(
                                output_row,
                                output_column,
                                b25_text_format,
                                b25_text,
                                "\n",
                                b19_text_format,
                                b19_text,
                                value_cell_format,
                            )

                        elif b25_text:

                            worksheet.write(
                                output_row,
                                output_column,
                                b25_text,
                                value_cell_format,
                            )

                        elif b19_text:

                            worksheet.write_rich_string(
                                output_row,
                                output_column,
                                b19_text_format,
                                b19_text,
                                value_cell_format,
                            )

                        else:

                            worksheet.write_blank(
                                output_row,
                                output_column,
                                None,
                                value_cell_format,
                            )

                        output_column += 1

                worksheet.set_row(
                    output_row,
                    36
                    if warming in b19_scenario_for_warming
                    else 24,
                )

                output_row += 1


        # ----------------------------------------------------
        # Worksheet formatting
        # ----------------------------------------------------

        worksheet.set_column(
            0,
            0,
            15,
        )

        worksheet.set_column(
            first_data_column,
            final_column,
            17,
        )

        worksheet.set_row(
            0,
            24,
        )

        worksheet.set_row(
            1,
            48,
        )

        worksheet.set_row(
            3,
            34,
        )

        worksheet.set_row(
            4,
            20,
        )

        worksheet.freeze_panes(
            5,
            1,
        )

        worksheet.hide_gridlines(
            2
        )

        worksheet.set_landscape()

        worksheet.fit_to_pages(
            1,
            0,
        )

        worksheet.set_margins(
            left=0.20,
            right=0.20,
            top=0.35,
            bottom=0.35,
        )

        worksheet.repeat_rows(
            3,
            4,
        )


        # ----------------------------------------------------
        # Add validation worksheet
        # ----------------------------------------------------

        validation_values.to_excel(
            writer,
            sheet_name="Summary check",
            index=False,
        )

        validation_worksheet = writer.sheets[
            "Summary check"
        ]

        validation_worksheet.freeze_panes(
            1,
            0,
        )

        validation_worksheet.set_column(
            0,
            0,
            10,
        )

        validation_worksheet.set_column(
            1,
            1,
            12,
        )

        validation_worksheet.set_column(
            2,
            3,
            25,
        )

        validation_worksheet.set_column(
            4,
            5,
            35,
        )


# ============================================================
# 11. RUN
# ============================================================

export_comparison_table(
    b25_values=b25_lookup,
    b19_values=b19_lookup,
    validation_values=validation_summary,
    output_path=output_file,
)


print(
    "\nCombined B25-B19 comparison table saved:"
)

print(
    output_file
)

print(
    "\nB19 source:"
)

print(
    f"  {b19_file}"
)

print(
    f"  Worksheet: {b19_sheet_name}"
)