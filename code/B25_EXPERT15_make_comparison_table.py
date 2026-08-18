#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a combined comparison table for the preferred/original B25
Monte Carlo distributions and the Expert 15 (IOP) B25 distributions.

Each data cell reports:
    Original B25: 50th percentile (5th to 95th percentile)
    Expert 15:    50th percentile (5th to 95th percentile)

Original B25 appears on the first line in black.
Expert 15 appears on the second line in purple.

Processes shown:
    Discharge
    Runoff
    Accumulation
    Surface mass balance (Runoff + Accumulation)
    Total

Both source datasets are in mm. All reported values are converted to cm.
"""

import os

import numpy as np
import pandas as pd

from B25data_processing import (
    load_and_prepare_data as load_original_b25,
    project_dir,
)

from B25_EXPERT15_data_processing import (
    load_and_prepare_data as load_expert15_b25,
)


# ============================================================
# 1. SETTINGS
# ============================================================

output_dir = os.path.join(
    project_dir,
    "output",
    "tables",
)

output_file = os.path.join(
    output_dir,
    "B25_EXPERT15_Table2_process_comparison.xlsx",
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

# Process names stored in the prepared B25 dataframes.
process_name = {
    "Discharge": "Discharge",
    "Runoff": "Runoff",
    "Accumulation": "Accumulation",
    "Runoff + Accumulation": "Runoff minus accumulation",
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


def summarize_numeric(samples):
    """
    Return unformatted 5th, 50th, and 95th percentiles for the
    validation worksheet.
    """

    samples = np.asarray(
        samples,
        dtype=float,
    )

    samples = samples[
        np.isfinite(samples)
    ]

    if samples.size == 0:
        return {
            "p05": np.nan,
            "p50": np.nan,
            "p95": np.nan,
        }

    q05, q50, q95 = np.percentile(
        samples,
        [
            5,
            50,
            95,
        ],
    )

    return {
        "p05": q05,
        "p50": q50,
        "p95": q95,
    }


# ============================================================
# 3. LOAD BOTH B25 DATASETS
# ============================================================

print("\nLoading original B25 data...")
original_data = load_original_b25()

print("\nLoading Expert 15 B25 data...")
expert15_data = load_expert15_b25()


# ============================================================
# 4. PREPARE DATAFRAMES
# ============================================================

def prepare_comparison_data(data):
    """
    Copy the prepared process and total dataframes and convert
    their values from mm to cm.
    """

    process_ext = data["process_ext"].copy()
    mc_total_ext = data["mc_total_ext"].copy()

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

    return process_ext, mc_total_ext


original_process, original_total = prepare_comparison_data(
    original_data
)

expert15_process, expert15_total = prepare_comparison_data(
    expert15_data
)


# ============================================================
# 5. BUILD SUMMARY LOOKUPS
# ============================================================

def create_summary_lookup(
    process_ext,
    mc_total_ext,
):
    """
    Create a lookup with keys:

        (year, warming, table_group, ice_sheet)

    and formatted summary strings as values.
    """

    lookup = {}

    # --------------------------------------------------------
    # Process-level quantities
    # --------------------------------------------------------

    for table_group, stored_process_name in process_name.items():

        subset = process_ext.loc[
            process_ext["process"] == stored_process_name
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


original_lookup = create_summary_lookup(
    original_process,
    original_total,
)

expert15_lookup = create_summary_lookup(
    expert15_process,
    expert15_total,
)


# ============================================================
# 6. NUMERICAL VALIDATION / DIFFERENCE TABLE
# ============================================================

def get_samples(
    process_ext,
    mc_total_ext,
    year,
    warming,
    table_group,
    ice_sheet,
):
    """Return one distribution in cm for a requested table cell."""

    if table_group == "Total":
        subset = mc_total_ext.loc[
            (mc_total_ext["year"] == year)
            & (mc_total_ext["warming"] == warming)
            & (mc_total_ext["ice_sheet"] == ice_sheet)
        ]
    else:
        stored_process_name = process_name[table_group]

        subset = process_ext.loc[
            (process_ext["year"] == year)
            & (process_ext["warming"] == warming)
            & (process_ext["ice_sheet"] == ice_sheet)
            & (process_ext["process"] == stored_process_name)
        ]

    return pd.to_numeric(
        subset["value"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)


def create_validation_table():
    """
    Create a long-form numerical table containing the original B25
    and Expert 15 percentiles plus Expert15 - Original differences.
    """

    records = []

    for year in years:
        for warming in warming_order:
            for table_group in table_groups:
                for ice_sheet in ice_sheets:

                    original_samples = get_samples(
                        process_ext=original_process,
                        mc_total_ext=original_total,
                        year=year,
                        warming=warming,
                        table_group=table_group,
                        ice_sheet=ice_sheet,
                    )

                    expert15_samples = get_samples(
                        process_ext=expert15_process,
                        mc_total_ext=expert15_total,
                        year=year,
                        warming=warming,
                        table_group=table_group,
                        ice_sheet=ice_sheet,
                    )

                    original_q = summarize_numeric(
                        original_samples
                    )
                    expert15_q = summarize_numeric(
                        expert15_samples
                    )

                    records.append(
                        {
                            "year": year,
                            "warming_C": warming,
                            "table_group": table_group,
                            "ice_sheet": ice_sheet,
                            "original_p05_cm": original_q["p05"],
                            "original_p50_cm": original_q["p50"],
                            "original_p95_cm": original_q["p95"],
                            "expert15_p05_cm": expert15_q["p05"],
                            "expert15_p50_cm": expert15_q["p50"],
                            "expert15_p95_cm": expert15_q["p95"],
                            "difference_p05_cm": (
                                expert15_q["p05"]
                                - original_q["p05"]
                            ),
                            "difference_p50_cm": (
                                expert15_q["p50"]
                                - original_q["p50"]
                            ),
                            "difference_p95_cm": (
                                expert15_q["p95"]
                                - original_q["p95"]
                            ),
                        }
                    )

    return pd.DataFrame(records)


validation_summary = create_validation_table()


# ============================================================
# 7. EXPORT COMPARISON TABLE
# ============================================================

def export_comparison_table(
    original_values,
    expert15_values,
    validation_values,
    output_path,
):
    """
    Export a publication-style Excel comparison table.

    Original B25 is shown on the first line in black.
    Expert 15 is shown on the second line in purple.
    """

    first_data_column = 1
    columns_per_group = len(ice_sheets)

    final_column = (
        first_data_column
        + len(table_groups) * columns_per_group
        - 1
    )

    with pd.ExcelWriter(
        output_path,
        engine="xlsxwriter",
    ) as writer:

        workbook = writer.book

        worksheet = workbook.add_worksheet(
            "B25-Expert15 Comparison"
        )

        writer.sheets[
            "B25-Expert15 Comparison"
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

        original_text_format = workbook.add_format({
            "font_color": "#000000",
        })

        expert15_text_format = workbook.add_format({
            "font_color": "#purple",
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
                "B25 preferred DM and Expert 15 projected "
                "sea-level-rise contributions by ice sheet "
                "and process (cm)"
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
                "Original B25 (preferred DM) is shown on the "
                "first line in black. Expert 15 (IOP; weight = 1) "
                "is shown on the second line in purple."
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

            start_column = end_column + 1

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

                        original_text = original_values.get(
                            key,
                            "",
                        )

                        expert15_text = expert15_values.get(
                            key,
                            "",
                        )

                        if original_text and expert15_text:

                            worksheet.write_rich_string(
                                output_row,
                                output_column,
                                original_text_format,
                                original_text,
                                "\n",
                                expert15_text_format,
                                expert15_text,
                                value_cell_format,
                            )

                        elif original_text:

                            worksheet.write(
                                output_row,
                                output_column,
                                original_text,
                                value_cell_format,
                            )

                        elif expert15_text:

                            worksheet.write_rich_string(
                                output_row,
                                output_column,
                                expert15_text_format,
                                expert15_text,
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

                # Both datasets are present for every warming level,
                # so every row contains two lines.
                worksheet.set_row(
                    output_row,
                    36,
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
        # Validation / numerical differences worksheet
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
            len(validation_values.columns) - 1,
            18,
        )


# ============================================================
# 8. RUN
# ============================================================

export_comparison_table(
    original_values=original_lookup,
    expert15_values=expert15_lookup,
    validation_values=validation_summary,
    output_path=output_file,
)


print(
    "\nCombined B25-Expert 15 comparison table saved:"
)

print(
    output_file
)

print(
    "\nEach cell shows:"
)

print(
    "  line 1: original B25 preferred DM"
)

print(
    "  line 2: Expert 15 / IOP"
)
