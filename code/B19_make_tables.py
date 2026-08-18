"""
Create a publication-style B19 process-contribution table.

The B19 source workbook:
    data/B19_processdata_MC.xlsx

Source values are stored in millimeters. This script converts them
to centimeters and reports each distribution as:

    50th percentile (5th–95th percentile)

The B19 workbook contains:
    EAIS: accumulation, discharge
    WAIS: accumulation, discharge, runoff
    GrIS: accumulation, discharge, runoff

Therefore, EAIS runoff and EAIS runoff-minus-accumulation are left blank.

@author: elizabeth
"""

import os
import re
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

project_dir = "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025"

data_file = os.path.join(
    project_dir,
    "data",
    "B19_processdata_MC.xlsx"
)

output_dir = os.path.join(
    project_dir,
    "output",
    "tables"
)

output_file = os.path.join(
    output_dir,
    "B19_Table2_process_contributions_publication_format.xlsx"
)

os.makedirs(output_dir, exist_ok=True)


scenario_order = [
    "Low",
    "High"
]

process_order = [
    "Accumulation",
    "Discharge",
    "Runoff",
    "Runoff minus accumulation"
]

ice_sheet_order = [
    "EAIS",
    "WAIS",
    "GrIS"
]


ice_sheet_map = {
    "E": "EAIS",
    "W": "WAIS",
    "G": "GrIS"
}

process_map = {
    "A": "Accumulation",
    "D": "Discharge",
    "R": "Runoff"
}

scenario_map = {
    "L": "Low",
    "H": "High"
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def fmt(value):
    """
    Format a rounded number using a Unicode minus sign.
    """

    if pd.isna(value):
        return ""

    return f"{value:.0f}".replace("-", "−")


def parse_b19_variable(column_name):
    """
    Parse a B19 workbook column name.

    Example
    -------
    PW01EA_2100L
        EAIS accumulation, 2100, Low scenario
    """

    match = re.fullmatch(
        r"PW\d+([EWG])([ADR])_(2100|2200|2300)([LH])",
        str(column_name)
    )

    if match is None:
        return None

    ice_code, process_code, year, scenario_code = match.groups()

    return {
        "variable": column_name,
        "ice_sheet_code": ice_code,
        "ice_sheet": ice_sheet_map[ice_code],
        "process_code": process_code,
        "process": process_map[process_code],
        "year": int(year),
        "scenario_code": scenario_code,
        "scenario": scenario_map[scenario_code]
    }


# ============================================================
# 3. READ AND RESHAPE B19 DATA
# ============================================================

def load_b19_process_data(file_path=data_file):
    """
    Read, validate, reshape, and convert the B19 process samples.
    """

    print(f"Reading B19 Monte Carlo samples from:\n{file_path}\n")

    wide = pd.read_excel(
        file_path,
        sheet_name=0
    )

    wide["sample_id"] = range(len(wide))

    variable_columns = [
        column
        for column in wide.columns
        if column != "sample_id"
    ]

    parsed = [
        parse_b19_variable(column)
        for column in variable_columns
    ]

    unparsed = [
        column
        for column, metadata in zip(variable_columns, parsed)
        if metadata is None
    ]

    if unparsed:
        raise ValueError(
            "The following B19 columns could not be parsed:\n"
            + "\n".join(map(str, unparsed))
        )

    metadata = pd.DataFrame(parsed)

    long = (
        wide
        .melt(
            id_vars="sample_id",
            var_name="variable",
            value_name="value"
        )
        .merge(
            metadata,
            on="variable",
            how="left",
            validate="many_to_one"
        )
    )

    # B19 source values are stored in millimeters.
    long["value"] = long["value"] / 10.0

    long["scenario"] = pd.Categorical(
        long["scenario"],
        categories=scenario_order,
        ordered=True
    )

    long["process"] = pd.Categorical(
        long["process"],
        categories=process_order,
        ordered=True
    )

    long["ice_sheet"] = pd.Categorical(
        long["ice_sheet"],
        categories=ice_sheet_order,
        ordered=True
    )

    return long


# ============================================================
# 4. DERIVE RUNOFF MINUS ACCUMULATION
# ============================================================

def add_runoff_minus_accumulation(process_long):
    """
    Add runoff minus accumulation draw by draw.

    Accumulation is already represented as a signed, negative
    sea-level contribution, so the derived value is R + A.

    B19 does not contain EAIS runoff, so this derived process is
    available only for WAIS and GrIS.
    """

    runoff_accumulation = (
        process_long
        .loc[
            process_long["process_code"].isin(["A", "R"])
        ]
        .pivot(
            index=[
                "sample_id",
                "ice_sheet_code",
                "ice_sheet",
                "year",
                "scenario_code",
                "scenario"
            ],
            columns="process_code",
            values="value"
        )
        .dropna(subset=["A", "R"])
        .reset_index()
    )

    runoff_accumulation["value"] = (
        runoff_accumulation["R"]
        + runoff_accumulation["A"]
    )

    runoff_accumulation["variable"] = pd.NA
    runoff_accumulation["process_code"] = "R-A"
    runoff_accumulation["process"] = "Runoff minus accumulation"

    runoff_accumulation = runoff_accumulation[
        process_long.columns
    ]

    combined = pd.concat(
        [
            process_long,
            runoff_accumulation
        ],
        ignore_index=True
    )

    combined["process"] = pd.Categorical(
        combined["process"],
        categories=process_order,
        ordered=True
    )

    combined["ice_sheet"] = pd.Categorical(
        combined["ice_sheet"],
        categories=ice_sheet_order,
        ordered=True
    )

    combined["scenario"] = pd.Categorical(
        combined["scenario"],
        categories=scenario_order,
        ordered=True
    )

    return combined


# ============================================================
# 5. SUMMARIZE FOR PUBLICATION TABLE
# ============================================================

def create_publication_table_data(process_data):
    """
    Format each distribution as median (5th–95th percentile).
    """

    summary = (
        process_data
        .groupby(
            [
                "year",
                "scenario",
                "process",
                "ice_sheet"
            ],
            observed=True
        )["value"]
        .agg(
            p05=lambda x: x.quantile(0.05),
            p50=lambda x: x.quantile(0.50),
            p95=lambda x: x.quantile(0.95)
        )
        .reset_index()
    )

    summary["summary"] = (
        summary["p50"].map(fmt)
        + " ("
        + summary["p05"].map(fmt)
        + "–"
        + summary["p95"].map(fmt)
        + ")"
    )

    table_wide = (
        summary
        .pivot(
            index=[
                "year",
                "scenario"
            ],
            columns=[
                "process",
                "ice_sheet"
            ],
            values="summary"
        )
        .reindex(
            columns=pd.MultiIndex.from_product(
                [
                    process_order,
                    ice_sheet_order
                ],
                names=[
                    "Process",
                    "Ice sheet"
                ]
            )
        )
    )

    row_order = pd.MultiIndex.from_product(
        [
            sorted(summary["year"].unique()),
            scenario_order
        ],
        names=[
            "year",
            "scenario"
        ]
    )

    return table_wide.reindex(row_order)


# ============================================================
# 6. EXPORT PUBLICATION-STYLE TABLE
# ============================================================

def export_publication_table(table_data, output_path):
    """
    Export the B19 process table to Excel.
    """

    first_data_column = 1
    columns_per_process = len(ice_sheet_order)

    final_column = (
        first_data_column
        + len(process_order) * columns_per_process
        - 1
    )

    with pd.ExcelWriter(
        output_path,
        engine="xlsxwriter"
    ) as writer:

        workbook = writer.book

        worksheet = workbook.add_worksheet("B19 Table 2")
        writer.sheets["B19 Table 2"] = worksheet

        title_format = workbook.add_format({
            "bold": True,
            "font_size": 12,
            "align": "left",
            "valign": "vcenter"
        })

        note_format = workbook.add_format({
            "italic": True,
            "font_size": 10,
            "align": "left",
            "valign": "vcenter",
            "text_wrap": True
        })

        group_header_format = workbook.add_format({
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "top": 1,
            "bottom": 1
        })

        subheader_format = workbook.add_format({
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "bottom": 1
        })

        row_label_format = workbook.add_format({
            "align": "left",
            "valign": "vcenter",
            "text_wrap": True
        })

        value_format = workbook.add_format({
            "align": "center",
            "valign": "vcenter"
        })

        bottom_border_format = workbook.add_format({
            "bottom": 1
        })

        worksheet.merge_range(
            0,
            0,
            0,
            final_column,
            "B19 process contributions to global mean sea level (cm)",
            title_format
        )

        worksheet.merge_range(
            1,
            0,
            1,
            final_column,
            "Values are the 50th percentile "
            "(5th–95th percentile). "
            "EAIS runoff was not included in the B19 workbook, "
            "so EAIS runoff and runoff − accumulation are blank.",
            note_format
        )

        worksheet.merge_range(
            3,
            0,
            4,
            0,
            "Year and scenario",
            group_header_format
        )

        start_column = first_data_column

        for process in process_order:

            end_column = (
                start_column
                + columns_per_process
                - 1
            )

            process_label = (
                "Runoff − Accumulation"
                if process == "Runoff minus accumulation"
                else process
            )

            worksheet.merge_range(
                3,
                start_column,
                3,
                end_column,
                process_label,
                group_header_format
            )

            for offset, ice_sheet in enumerate(ice_sheet_order):
                worksheet.write(
                    4,
                    start_column + offset,
                    ice_sheet,
                    subheader_format
                )

            start_column = end_column + 1

        output_row = 5

        years = table_data.index.get_level_values(
            "year"
        ).unique()

        for year in years:

            for scenario in scenario_order:

                if (year, scenario) not in table_data.index:
                    continue

                worksheet.write(
                    output_row,
                    0,
                    f"{int(year)}\n{scenario}",
                    row_label_format
                )

                output_column = first_data_column

                for process in process_order:
                    for ice_sheet in ice_sheet_order:

                        value = table_data.loc[
                            (year, scenario),
                            (process, ice_sheet)
                        ]

                        if pd.isna(value):
                            value = ""

                        worksheet.write(
                            output_row,
                            output_column,
                            value,
                            value_format
                        )

                        output_column += 1

                output_row += 1

        for column in range(final_column + 1):
            worksheet.write_blank(
                output_row,
                column,
                None,
                bottom_border_format
            )

        worksheet.set_column(0, 0, 18)
        worksheet.set_column(
            first_data_column,
            final_column,
            16
        )

        worksheet.set_row(1, 32)

        worksheet.freeze_panes(5, 1)
        worksheet.hide_gridlines(2)

        worksheet.set_landscape()
        worksheet.fit_to_pages(1, 0)
        worksheet.repeat_rows(3, 4)


# ============================================================
# 7. RUN WORKFLOW
# ============================================================

b19_process_long = load_b19_process_data()

b19_process_ext = add_runoff_minus_accumulation(
    b19_process_long
)

b19_table2_wide = create_publication_table_data(
    b19_process_ext
)

export_publication_table(
    b19_table2_wide,
    output_file
)

print("\nB19 table saved successfully:")
print(output_file)
