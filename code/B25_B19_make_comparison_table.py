"""
Create a combined B25-B19 process-comparison table.

Each data cell reports:
    B25: 50th percentile (5th to 95th percentile)
    B19: 50th percentile (5th to 95th percentile)

B25 appears on the first line in black. Where a matching B19 scenario
exists, B19 appears on the second line in lighter gray.

B19 Low is compared with B25 2.0°C.
B19 High is compared with B25 5.0°C.
B25 3.5°C has no B19 comparison and therefore contains one line only.

Processes shown:
    Discharge
    Runoff
    Accumulation
    Runoff + Accumulation
    Total

All source values are converted from millimeters to centimeters.

@author: elizabeth
"""

import os
import re

import numpy as np
import pandas as pd

from B25data_processing import (
    load_and_prepare_data,
    project_dir,
)


# ============================================================
# 1. SETTINGS
# ============================================================

b19_file = os.path.join(
    project_dir,
    "data",
    "B19_processdata_MC.xlsx",
)

output_dir = os.path.join(
    project_dir,
    "output",
    "tables",
)

output_file = os.path.join(
    output_dir,
    "B25_B19_Table2_process_comparison_v2.xlsx",
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

# B19 has only low and high scenarios.
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

# B25 process names as stored in process_ext.
b25_process_name = {
    "Discharge": "Discharge",
    "Runoff": "Runoff",
    "Accumulation": "Accumulation",
    "Runoff + Accumulation": "Runoff minus accumulation",
}


# ============================================================
# 2. FORMATTING HELPERS
# ============================================================

def fmt(value):
    """Round to a whole centimeter and use a Unicode minus sign."""
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

# Source B25 values are in millimeters.
process_ext["value"] = (
    process_ext["value"]
    / 10.0
)

mc_total_ext["value"] = (
    mc_total_ext["value"]
    / 10.0
)


def create_b25_summary_lookup():
    """
    Create:
        (year, warming, table_group, ice_sheet) -> formatted summary
    """
    lookup = {}

    # Process-level quantities.
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

            if (
                year in years
                and warming in warming_order
                and ice_sheet in ice_sheets
            ):
                lookup[
                    (
                        int(year),
                        float(warming),
                        table_group,
                        ice_sheet,
                    )
                ] = summarize_samples(
                    samples.to_numpy()
                )

    # Total ice-sheet contributions.
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

        if (
            year in years
            and warming in warming_order
            and ice_sheet in ice_sheets
        ):
            lookup[
                (
                    int(year),
                    float(warming),
                    "Total",
                    ice_sheet,
                )
            ] = summarize_samples(
                samples.to_numpy()
            )

    return lookup


b25_lookup = create_b25_summary_lookup()


# ============================================================
# 4. LOAD AND SUMMARIZE B19
# ============================================================

if not os.path.exists(
    b19_file
):
    raise FileNotFoundError(
        "B19 process workbook not found:\n"
        f"{b19_file}"
    )

b19_data = pd.read_excel(
    b19_file,
    sheet_name=0,
)

b19_data.columns = [
    str(column).strip()
    for column in b19_data.columns
]


def b19_column_name(
    year,
    suffix,
    code,
):
    return (
        f"PW01{code}_"
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

required_b19_columns = {
    b19_column_name(
        year,
        suffix,
        code,
    )
    for year in years
    for suffix in [
        "L",
        "H",
    ]
    for code in required_b19_codes
}

missing_b19_columns = (
    required_b19_columns
    - set(b19_data.columns)
)

if missing_b19_columns:
    raise ValueError(
        "B19_processdata_MC.xlsx is missing these columns:\n"
        + "\n".join(
            sorted(missing_b19_columns)
        )
    )

for column in required_b19_columns:
    b19_data[column] = pd.to_numeric(
        b19_data[column],
        errors="coerce",
    )


def aligned_b19_arrays(
    year,
    suffix,
    codes,
):
    """
    Return row-aligned B19 arrays in centimeters.

    Rows with a missing value in any requested column are removed
    together before draw-by-draw arithmetic.
    """
    names = [
        b19_column_name(
            year,
            suffix,
            code,
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
            aligned[name].to_numpy()
            / 10.0
        )
        for code, name in zip(
            codes,
            names,
        )
    }


def get_b19_samples(
    year,
    suffix,
    table_group,
    ice_sheet,
):
    """
    Return one B19 distribution in centimeters.

    Accumulation is already signed. Therefore:
        runoff + accumulation = R + A

    No EAIS runoff column exists, so B19 EAIS Runoff and
    Runoff + Accumulation are left blank.
    """

    ice_code = {
        "EAIS": "E",
        "WAIS": "W",
        "GrIS": "G",
    }[ice_sheet]

    if table_group == "Discharge":
        arrays = aligned_b19_arrays(
            year,
            suffix,
            [
                f"{ice_code}D",
            ],
        )

        if arrays is None:
            return None

        return arrays[
            f"{ice_code}D"
        ]

    if table_group == "Accumulation":
        arrays = aligned_b19_arrays(
            year,
            suffix,
            [
                f"{ice_code}A",
            ],
        )

        if arrays is None:
            return None

        return arrays[
            f"{ice_code}A"
        ]

    if table_group == "Runoff":

        if ice_sheet == "EAIS":
            return None

        arrays = aligned_b19_arrays(
            year,
            suffix,
            [
                f"{ice_code}R",
            ],
        )

        if arrays is None:
            return None

        return arrays[
            f"{ice_code}R"
        ]

    if table_group == "Runoff + Accumulation":

        if ice_sheet == "EAIS":
            return None

        arrays = aligned_b19_arrays(
            year,
            suffix,
            [
                f"{ice_code}R",
                f"{ice_code}A",
            ],
        )

        if arrays is None:
            return None

        return (
            arrays[f"{ice_code}R"]
            + arrays[f"{ice_code}A"]
        )

    if table_group == "Total":

        codes = [
            f"{ice_code}A",
            f"{ice_code}D",
        ]

        if ice_sheet != "EAIS":
            codes.append(
                f"{ice_code}R"
            )

        arrays = aligned_b19_arrays(
            year,
            suffix,
            codes,
        )

        if arrays is None:
            return None

        total = np.zeros(
            len(
                next(
                    iter(
                        arrays.values()
                    )
                )
            ),
            dtype=float,
        )

        for code in codes:
            total += arrays[code]

        return total

    raise ValueError(
        "Unsupported B19 table group: "
        f"{table_group}"
    )


def create_b19_summary_lookup():
    """
    Create:
        (year, warming, table_group, ice_sheet) -> formatted summary
    """
    lookup = {}

    for year in years:

        for warming, suffix in (
            b19_scenario_for_warming.items()
        ):

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
# 5. EXPORT COMPARISON TABLE
# ============================================================

def export_comparison_table(
    b25_values,
    b19_values,
    output_path,
):
    """
    Export one publication-style Excel comparison table.

    B25 is shown on the first line in black.
    B19 is shown below in lighter gray when available.
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
            "B25-B19 Table 2"
        )

        writer.sheets[
            "B25-B19 Table 2"
        ] = worksheet

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

        worksheet.merge_range(
            0,
            0,
            0,
            final_column,
            (
                "Table 2. B25 and B19 projected sea-level-rise "
                "contributions by ice sheet and process (cm)"
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
                "B25 is shown on the first line; B19 is shown "
                "below in gray for the 2.0°C/Low and 5.0°C/High "
                "comparisons. B19 does not provide EAIS runoff."
            ),
            note_format,
        )

        worksheet.merge_range(
            3,
            0,
            4,
            0,
            "Year and warming",
            group_header_format,
        )

        start_column = (
            first_data_column
        )

        for table_group in table_groups:

            end_column = (
                start_column
                + columns_per_group
                - 1
            )

            display_label = (
                "Surface mass balance\n"
                "(Runoff + Accumulation)"
                if table_group
                == "Runoff + Accumulation"
                else table_group
            )

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
                end_column
                + 1
            )

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

                output_column = (
                    first_data_column
                )

                for table_group in table_groups:

                    for ice_sheet in ice_sheets:

                        key = (
                            year,
                            warming,
                            table_group,
                            ice_sheet,
                        )

                        b25_text = (
                            b25_values.get(
                                key,
                                "",
                            )
                        )

                        b19_text = (
                            b19_values.get(
                                key,
                                "",
                            )
                        )

                        if (
                            b25_text
                            and b19_text
                        ):
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

                # Two lines for 2.0°C and 5.0°C; one line for 3.5°C.
                worksheet.set_row(
                    output_row,
                    36
                    if warming in b19_scenario_for_warming
                    else 24,
                )

                output_row += 1

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
            20,
        )

        worksheet.set_row(
            1,
            38,
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


# ============================================================
# 6. RUN
# ============================================================

export_comparison_table(
    b25_values=b25_lookup,
    b19_values=b19_lookup,
    output_path=output_file,
)

print(
    "\nCombined B25-B19 comparison table saved:"
)
print(
    output_file
)
