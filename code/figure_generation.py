#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:59:59 2026

@author: elizabeth
"""

## Import libraries
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter

def fmt(x):
    """Format numbers with a Unicode minus sign."""
    return f"{x:.0f}".replace("-", "−")

## File paths
project_dir = ('/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025')
data_file = os.path.join(
    project_dir,
    "data",
    "sej2025_split_panel_liz.csv"
)

# Read in data
print(f"Reading Monte Carlo samples from:\n{data_file}\n")

mc_df = pd.read_csv(data_file)
mc_df["sample_id"] = np.arange(len(mc_df))

# Parse variable names
time_lookup = {
    "21": 2100,
    "22": 2200,
    "23": 2300
}

ice_sheet_lookup = {
    "E": "EAIS",
    "W": "WAIS",
    "G": "GrIS",
    "SLR": "Total ice-sheet SLR"
}

process_lookup = {
    "A": "Accumulation",
    "D": "Discharge",
    "R": "Runoff",
    "TOTAL": "Total"
}


def parse_variable_name(column_name):
    """
    Parse names such as:
        EA2_21
        WD35_22
        G5_23
        SLR2_21
    """

    if column_name == "sample_id":
        return None

    match = re.fullmatch(
        r"(SLR|[EWG])([ADR]?)(2|35|5)_(21|22|23)",
        column_name
    )

    if match is None:
        return None

    ice_code, process_code, warming_code, time_code = match.groups()

    process_code = process_code if process_code else "TOTAL"

    return {
        "variable": column_name,
        "ice_sheet_code": ice_code,
        "ice_sheet": ice_sheet_lookup[ice_code],
        "process_code": process_code,
        "process": process_lookup[process_code],
        "warming_code": warming_code,
        "warming": {
            "2": 2.0,
            "35": 3.5,
            "5": 5.0
        }[warming_code],
        "year": time_lookup[time_code]
    }


variable_metadata = [
    parse_variable_name(column)
    for column in mc_df.columns
]

variable_metadata = [
    metadata
    for metadata in variable_metadata
    if metadata is not None
]

meta_df = pd.DataFrame(variable_metadata)

print(meta_df.head())
print(f"\nParsed variables: {len(meta_df)}")

# Change from wide to long data format
mc_long = mc_df.melt(
    id_vars="sample_id",
    var_name="variable",
    value_name="value"
)

mc_long = mc_long.merge(
    meta_df,
    on="variable",
    how="left",
    validate="many_to_one"
)

# Optional formatted warming label for plotting
mc_long["scenario_name"] = (
    mc_long["warming"]
    .map({
        2.0: "2.0°C",
        3.5: "3.5°C",
        5.0: "5.0°C"
    })
)

print(mc_long.head())
print(f"\nLong-format rows: {len(mc_long):,}")

# Add AIS (E + W)
mc_total = mc_long.loc[
    mc_long["process_code"] == "TOTAL"
].copy()

ais_df = (
    mc_total.loc[
        mc_total["ice_sheet_code"].isin(["E", "W"])
    ]
    .pivot_table(
        index=["sample_id", "warming", "warming_code", "year", "scenario_name"],
        columns="ice_sheet_code",
        values="value"
    )
    .dropna(subset=["E", "W"])
    .reset_index()
)

ais_df["value"] = ais_df["E"] + ais_df["W"]
ais_df["variable"] = pd.NA
ais_df["ice_sheet_code"] = "AIS"
ais_df["ice_sheet"] = "Total AIS"
ais_df["process_code"] = "TOTAL"
ais_df["process"] = "Total"

ais_df = ais_df[
    mc_total.columns
]

mc_total_ext = pd.concat(
    [mc_total, ais_df],
    ignore_index=True
)


# SUMMARY FUNCTIONS

def calculate_summary(values):
    """Calculate summary statistics for one Monte Carlo distribution."""

    values = values.dropna()

    return pd.Series({
        "mean": values.mean(),
        "sd": values.std(),
        "p05": values.quantile(0.05),
        "p17": values.quantile(0.17),
        "p50": values.quantile(0.50),
        "p83": values.quantile(0.83),
        "p95": values.quantile(0.95)
    })


def format_estimate(median, lower, upper):
    """Format a median and uncertainty range as median [lower, upper]."""

    return f"{median:.0f} [{lower:.0f}, {upper:.0f}]"

# ============================================================
# 6. TABLE 1: ICE-SHEET CONTRIBUTIONS
# ============================================================

ice_sheet_order = [
    "EAIS",
    "WAIS",
    "Total AIS",
    "GrIS",
    "Total ice-sheet SLR"
]

table1_stats = (
    mc_total_ext
    .groupby(
        ["ice_sheet", "warming", "scenario_name", "year"],
        observed=True
    )["value"]
    .apply(calculate_summary)
    .unstack()
    .reset_index()
)

# Add publication-style formatted columns
table1_stats["Mean ± SD"] = table1_stats.apply(
    lambda row: f"{row['mean']:.0f} ± {row['sd']:.0f}",
    axis=1
)

table1_stats["Median [5–95%]"] = table1_stats.apply(
    lambda row: format_estimate(
        row["p50"],
        row["p05"],
        row["p95"]
    ),
    axis=1
)

table1_stats["Median [17–83%]"] = table1_stats.apply(
    lambda row: format_estimate(
        row["p50"],
        row["p17"],
        row["p83"]
    ),
    axis=1
)

# Control row order
table1_stats["ice_sheet"] = pd.Categorical(
    table1_stats["ice_sheet"],
    categories=ice_sheet_order,
    ordered=True
)

table1 = (
    table1_stats[
        [
            "ice_sheet",
            "warming",
            "year",
            "Mean ± SD",
            "Median [5–95%]",
            "Median [17–83%]"
        ]
    ]
    .sort_values(["year", "warming", "ice_sheet"])
    .reset_index(drop=True)
)

table1 = table1.rename(columns={
    "ice_sheet": "Ice sheet",
    "warming": "Warming (°C)",
    "year": "Year"
})

# Put columns in the preferred order
table1 = table1[
    [
        "Year",
        "Ice sheet",
        "Warming (°C)",
        "Mean ± SD",
        "50%",
        "Median [5–95%]",
        "Median [17–83%]"
    ]
]

table1 = table1.rename(columns={
    "Median [5–95%]": "5–95%",
    "Median [17–83%]": "17–83%"
})

print(table1.head(10))




# ============================================================
# 7. ADD RUNOFF MINUS ACCUMULATION
# ============================================================

process_df = mc_long.loc[
    mc_long["process_code"].isin(["A", "D", "R"])
].copy()

runoff_accumulation = (
    process_df.loc[
        process_df["process_code"].isin(["A", "R"])
    ]
    .pivot_table(
        index=[
            "sample_id",
            "ice_sheet_code",
            "ice_sheet",
            "warming_code",
            "warming",
            "year",
            "scenario_name"
        ],
        columns="process_code",
        values="value"
    )
    .dropna(subset=["A", "R"])
    .reset_index()
)

runoff_accumulation["value"] = (
    runoff_accumulation["R"]
    - runoff_accumulation["A"]
)

runoff_accumulation["variable"] = pd.NA
runoff_accumulation["process_code"] = "R-A"
runoff_accumulation["process"] = "Runoff minus accumulation"

# Match the column structure of process_df
runoff_accumulation = runoff_accumulation[
    process_df.columns
]

process_ext = pd.concat(
    [process_df, runoff_accumulation],
    ignore_index=True
)

# ============================================================
# 8. TABLE 2: PROCESS CONTRIBUTIONS
# ============================================================

process_order = [
    "Accumulation",
    "Discharge",
    "Runoff",
    "Runoff minus accumulation"
]

process_ice_sheet_order = [
    "EAIS",
    "WAIS",
    "GrIS"
]

table2_stats = (
    process_ext
    .groupby(
        [
            "ice_sheet",
            "process",
            "warming",
            "scenario_name",
            "year"
        ],
        observed=True
    )["value"]
    .apply(calculate_summary)
    .unstack()
    .reset_index()
)

table2_stats["Mean ± SD"] = table2_stats.apply(
    lambda row: f"{row['mean']:.0f} ± {row['sd']:.0f}",
    axis=1
)

table2_stats["Median [5–95%]"] = table2_stats.apply(
    lambda row: format_estimate(
        row["p50"],
        row["p05"],
        row["p95"]
    ),
    axis=1
)

table2_stats["Median [17–83%]"] = table2_stats.apply(
    lambda row: format_estimate(
        row["p50"],
        row["p17"],
        row["p83"]
    ),
    axis=1
)

table2_stats["ice_sheet"] = pd.Categorical(
    table2_stats["ice_sheet"],
    categories=process_ice_sheet_order,
    ordered=True
)

table2_stats["process"] = pd.Categorical(
    table2_stats["process"],
    categories=process_order,
    ordered=True
)

table2 = (
    table2_stats[
        [
            "ice_sheet",
            "process",
            "scenario_name",
            "warming",
            "year",
            "Mean ± SD",
            "Median [5–95%]",
            "Median [17–83%]"
        ]
    ]
    .sort_values(
        ["year", "warming", "ice_sheet", "process"]
    )
    .reset_index(drop=True)
)

table2 = table2.rename(columns={
    "ice_sheet": "Ice sheet",
    "process": "Process",
    "scenario_name": "Warming level",
    "warming": "Warming (°C)",
    "year": "Year"
})

print(table2.head(12))

# ============================================================
# 7. PUBLICATION-STYLE TABLE 1
# ============================================================

table1_stats["5–95%"] = table1_stats.apply(
    lambda row: f"{fmt(row['p05'])}–{fmt(row['p95'])}",
    axis=1
)

table1_stats["17–83%"] = table1_stats.apply(
    lambda row: f"{fmt(row['p17'])}–{fmt(row['p83'])}",
    axis=1
)

table1_stats["50%"] = table1_stats["p50"].map(fmt)

table1_stats["Mean ± SD"] = table1_stats.apply(
    lambda row: f"{fmt(row['mean'])} ± {fmt(row['sd'])}",
    axis=1
)

table1_wide = table1_stats.pivot(
    index=["year", "ice_sheet"],
    columns="warming",
    values=[
        "Mean ± SD",
        "50%",
        "5–95%",
        "17–83%"
    ]
)

table1_wide = table1_wide.swaplevel(
    0,
    1,
    axis=1
)

warming_order = [2.0, 3.5, 5.0]

statistic_order = [
    "Mean ± SD",
    "50%",
    "5–95%",
    "17–83%"
]

desired_columns = pd.MultiIndex.from_product(
    [warming_order, statistic_order],
    names=["Warming", "Statistic"]
)

table1_wide = table1_wide.reindex(
    columns=desired_columns
)

table1_wide = table1_wide.reset_index()

table1_wide["ice_sheet"] = pd.Categorical(
    table1_wide["ice_sheet"],
    categories=ice_sheet_order,
    ordered=True
)

table1_wide = (
    table1_wide
    .sort_values(["year", "ice_sheet"])
    .reset_index(drop=True)
)

# ============================================================
# 9. EXPORT TABLES
# ============================================================

output_dir = os.path.join(project_dir, "output", "tables")
os.makedirs(output_dir, exist_ok=True)

table_file = os.path.join(
    output_dir,
    "B25_ice_sheet_summary_tables.xlsx"
)

with pd.ExcelWriter(
    table_file,
    engine="xlsxwriter"
) as writer:

    table1.to_excel(
        writer,
        sheet_name="Ice-sheet contributions",
        index=False,
        startrow=2
    )

    table2.to_excel(
        writer,
        sheet_name="Process contributions",
        index=False,
        startrow=2
    )

    workbook = writer.book

    title_format = workbook.add_format({
        "bold": True,
        "font_size": 14
    })

    header_format = workbook.add_format({
        "bold": True,
        "text_wrap": True,
        "valign": "top",
        "border": 1
    })

    numeric_format = workbook.add_format({
        "num_format": "0.0"
    })

    for sheet_name, table, title in [
        (
            "Ice-sheet contributions",
            table1,
            "Projected ice-sheet contributions to global mean sea level"
        ),
        (
            "Process contributions",
            table2,
            "Projected ice-sheet contributions by process"
        )
    ]:

        worksheet = writer.sheets[sheet_name]

        worksheet.write("A1", title, title_format)

        # Replace default headers with formatted headers
        for col_num, column_name in enumerate(table.columns):
            worksheet.write(
                2,
                col_num,
                column_name,
                header_format
            )

        worksheet.freeze_panes(3, 0)
        worksheet.autofilter(
            2,
            0,
            len(table) + 2,
            len(table.columns) - 1
        )

        # Default widths
        worksheet.set_column("A:A", 22)
        worksheet.set_column("B:B", 26)
        worksheet.set_column("C:C", 15)
        worksheet.set_column("D:D", 13, numeric_format)
        worksheet.set_column("E:E", 10)
        worksheet.set_column("F:H", 22)

print(f"Tables saved to:\n{table_file}")

# ============================================================
# 8. EXPORT PUBLICATION-STYLE TABLE 1
# ============================================================

publication_file = os.path.join(
    output_dir,
    "B25_Table1_publication_format.xlsx"
)

with pd.ExcelWriter(
    publication_file,
    engine="xlsxwriter"
) as writer:

    workbook = writer.book
    worksheet = workbook.add_worksheet("Table 1")
    writer.sheets["Table 1"] = worksheet

    # --------------------------------------------------------
    # FORMATS
    # --------------------------------------------------------

    title_format = workbook.add_format({
        "bold": True,
        "font_size": 12,
        "align": "left",
        "valign": "vcenter"
    })

    group_header_format = workbook.add_format({
        "bold": True,
        "align": "center",
        "valign": "vcenter",
        "bottom": 1
    })

    subheader_format = workbook.add_format({
        "bold": True,
        "align": "center",
        "valign": "vcenter",
        "bottom": 1
    })

    year_format = workbook.add_format({
        "bold": True,
        "align": "left"
    })

    ice_sheet_format = workbook.add_format({
        "align": "left",
        "indent": 1
    })

    value_format = workbook.add_format({
        "align": "center"
    })

    bottom_border_format = workbook.add_format({
        "bottom": 1
    })

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    worksheet.merge_range(
        0,
        0,
        0,
        12,
        "Table 1. Projected sea-level rise contributions "
        "from each ice sheet and combined",
        title_format
    )

    # --------------------------------------------------------
    # HEADERS
    # --------------------------------------------------------

    worksheet.merge_range(
        2,
        0,
        3,
        0,
        "Year and ice sheet",
        group_header_format
    )

    start_col = 1

    for warming in warming_order:

        worksheet.merge_range(
            2,
            start_col,
            2,
            start_col + 3,
            f"{warming:.1f}°C",
            group_header_format
        )

        for offset, statistic in enumerate(statistic_order):
            worksheet.write(
                3,
                start_col + offset,
                statistic,
                subheader_format
            )

        start_col += 4

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    output_row = 4

    for year, year_group in table1_wide.groupby(
        ("year", ""),
        sort=True,
        observed=True
    ):

        worksheet.write(
            output_row,
            0,
            int(year),
            year_format
        )

        output_row += 1

        for _, row in year_group.iterrows():

            ice_sheet = row[("ice_sheet", "")]

            worksheet.write(
                output_row,
                0,
                str(ice_sheet),
                ice_sheet_format
            )

            output_col = 1

            for warming in warming_order:
                for statistic in statistic_order:

                    value = row[(warming, statistic)]

                    worksheet.write(
                        output_row,
                        output_col,
                        value,
                        value_format
                    )

                    output_col += 1

            output_row += 1

    # --------------------------------------------------------
    # BOTTOM BORDER
    # --------------------------------------------------------

    for col in range(13):
        worksheet.write_blank(
            output_row,
            col,
            None,
            bottom_border_format
        )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    worksheet.set_column(0, 0, 24)
    worksheet.set_column(1, 12, 14)

    worksheet.freeze_panes(4, 1)
    worksheet.hide_gridlines(2)

print(f"Publication-style table saved to:\n{publication_file}")

# ============================================================
# 9. CREATE PUBLICATION-STYLE PROCESS TABLE
# ============================================================

process_order = [
    "Accumulation",
    "Discharge",
    "Runoff",
    "Runoff minus accumulation"
]

ice_sheet_order_process = [
    "EAIS",
    "WAIS",
    "GrIS"
]

percentile_order = [
    "5th",
    "50th",
    "95th"
]


def format_table_number(value):
    """
    Format a table value with no decimal places and use a
    proper Unicode minus sign for negative values.
    """

    if pd.isna(value):
        return ""

    return f"{value:.0f}".replace("-", "−")


quantile_frames = []

for percentile_name, quantile_value in [
    ("5th", 0.05),
    ("50th", 0.50),
    ("95th", 0.95)
]:

    temp = (
        process_ext
        .groupby(
            [
                "year",
                "warming",
                "process",
                "ice_sheet"
            ],
            observed=True
        )["value"]
        .quantile(quantile_value)
        .reset_index(name="value")
    )

    temp["percentile"] = percentile_name

    quantile_frames.append(temp)

table2_quantiles = pd.concat(
    quantile_frames,
    ignore_index=True
)

table2_quantiles["process"] = pd.Categorical(
    table2_quantiles["process"],
    categories=process_order,
    ordered=True
)

table2_quantiles["ice_sheet"] = pd.Categorical(
    table2_quantiles["ice_sheet"],
    categories=ice_sheet_order_process,
    ordered=True
)

table2_quantiles["percentile"] = pd.Categorical(
    table2_quantiles["percentile"],
    categories=percentile_order,
    ordered=True
)

table2_quantiles = (
    table2_quantiles
    .sort_values(
        [
            "year",
            "warming",
            "percentile",
            "process",
            "ice_sheet"
        ]
    )
    .reset_index(drop=True)
)

# ============================================================
# 10. EXPORT PUBLICATION-STYLE PROCESS TABLE
# ============================================================

process_table_file = os.path.join(
    output_dir,
    "B25_Table2_process_contributions_publication_format.xlsx"
)

with pd.ExcelWriter(
    process_table_file,
    engine="xlsxwriter"
) as writer:

    workbook = writer.book
    worksheet = workbook.add_worksheet("Table 2")
    writer.sheets["Table 2"] = worksheet

    # --------------------------------------------------------
    # FORMATS
    # --------------------------------------------------------

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
        "valign": "vcenter"
    })

    process_header_format = workbook.add_format({
        "bold": True,
        "align": "center",
        "valign": "vcenter",
        "top": 1,
        "bottom": 1
    })

    ice_header_format = workbook.add_format({
        "bold": True,
        "align": "center",
        "valign": "vcenter",
        "bottom": 1
    })

    left_header_format = workbook.add_format({
        "bold": True,
        "align": "center",
        "valign": "vcenter",
        "top": 1,
        "bottom": 1
    })

    year_warming_format = workbook.add_format({
        "align": "left",
        "valign": "top"
    })

    percentile_format = workbook.add_format({
        "align": "center",
        "valign": "vcenter"
    })

    value_format = workbook.add_format({
        "align": "center",
        "valign": "vcenter"
    })

    group_bottom_format = workbook.add_format({
        "align": "center",
        "valign": "vcenter",
        "bottom": 1
    })

    left_group_bottom_format = workbook.add_format({
        "align": "left",
        "valign": "vcenter",
        "bottom": 1
    })

    # --------------------------------------------------------
    # TABLE DIMENSIONS
    # --------------------------------------------------------

    first_data_col = 2
    columns_per_process = len(ice_sheet_order_process)

    final_col = (
        first_data_col
        + len(process_order) * columns_per_process
        - 1
    )

    # --------------------------------------------------------
    # TITLE AND NOTE
    # --------------------------------------------------------

    worksheet.merge_range(
        0,
        0,
        0,
        final_col,
        "Table 2. Projected sea-level rise contributions "
        "from each ice sheet and process",
        title_format
    )

    worksheet.merge_range(
        1,
        0,
        1,
        final_col,
        "Values are the 5th, 50th, and 95th percentiles "
        "of the Monte Carlo distributions.",
        note_format
    )

    # --------------------------------------------------------
    # LEFT-SIDE HEADERS
    # --------------------------------------------------------

    worksheet.merge_range(
        3,
        0,
        4,
        0,
        "Year and warming",
        left_header_format
    )

    worksheet.merge_range(
        3,
        1,
        4,
        1,
        "Percentile",
        left_header_format
    )

    # --------------------------------------------------------
    # PROCESS AND ICE-SHEET HEADERS
    # --------------------------------------------------------

    start_col = first_data_col

    for process in process_order:

        end_col = start_col + columns_per_process - 1

        worksheet.merge_range(
            3,
            start_col,
            3,
            end_col,
            process,
            process_header_format
        )

        for offset, ice_sheet in enumerate(
            ice_sheet_order_process
        ):
            worksheet.write(
                4,
                start_col + offset,
                ice_sheet,
                ice_header_format
            )

        start_col = end_col + 1

    # --------------------------------------------------------
    # DATA ROWS
    # --------------------------------------------------------

    output_row = 5

    year_warming_groups = (
        table2_quantiles[
            ["year", "warming"]
        ]
        .drop_duplicates()
        .sort_values(["year", "warming"])
    )

    for _, group_info in year_warming_groups.iterrows():

        year = int(group_info["year"])
        warming = float(group_info["warming"])

        group = table2_quantiles.loc[
            (table2_quantiles["year"] == year)
            & (table2_quantiles["warming"] == warming)
        ].copy()

        first_group_row = output_row
        last_group_row = output_row + len(percentile_order) - 1

        # Merge the year/warming label across the percentile rows
        worksheet.merge_range(
            first_group_row,
            0,
            last_group_row,
            0,
            f"{year}\n{warming:.1f}°C",
            year_warming_format
        )

        for percentile_index, percentile in enumerate(
            percentile_order
        ):

            current_row = output_row + percentile_index

            is_last_percentile = (
                percentile_index
                == len(percentile_order) - 1
            )

            current_percentile_format = (
                group_bottom_format
                if is_last_percentile
                else percentile_format
            )

            worksheet.write(
                current_row,
                1,
                percentile,
                current_percentile_format
            )

            percentile_data = group.loc[
                group["percentile"] == percentile
            ]

            output_col = first_data_col

            for process in process_order:
                for ice_sheet in ice_sheet_order_process:

                    match = percentile_data.loc[
                        (
                            percentile_data["process"]
                            == process
                        )
                        & (
                            percentile_data["ice_sheet"]
                            == ice_sheet
                        ),
                        "value"
                    ]

                    if len(match) == 0:
                        value = ""
                    else:
                        value = format_table_number(
                            match.iloc[0]
                        )

                    current_value_format = (
                        group_bottom_format
                        if is_last_percentile
                        else value_format
                    )

                    worksheet.write(
                        current_row,
                        output_col,
                        value,
                        current_value_format
                    )

                    output_col += 1

        output_row += len(percentile_order)

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    worksheet.set_column(0, 0, 17)
    worksheet.set_column(1, 1, 11)
    worksheet.set_column(
        first_data_col,
        final_col,
        11
    )

    worksheet.set_row(0, 20)
    worksheet.set_row(1, 18)
    worksheet.set_row(3, 20)
    worksheet.set_row(4, 18)

    worksheet.freeze_panes(5, 2)
    worksheet.hide_gridlines(2)

    worksheet.set_landscape()
    worksheet.fit_to_pages(1, 0)
    worksheet.set_margins(
        left=0.25,
        right=0.25,
        top=0.4,
        bottom=0.4
    )

print(
    "Publication-style process table saved to:\n"
    f"{process_table_file}"
)