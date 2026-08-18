"""
Created on Mon Jul 13 15:59:59 2026

Create summary and publication-style tables from the B25 Monte Carlo ice-sheet
projection output.

@author: elizabeth
"""

import os
import pandas as pd
from B25data_processing import (
    load_and_prepare_data,
    project_dir,
    ice_sheet_order,
    process_order,
    process_ice_sheet_order,
    warming_order)

# ============================================================
# 1. LOAD PREPARED DATA
# ============================================================

data = load_and_prepare_data()

mc_total_ext = data["mc_total_ext"].copy()
process_ext = data["process_ext"].copy()

# Convert sea-level contributions from mm to cm.
mc_total_ext["value"] = mc_total_ext["value"] / 10.0
process_ext["value"] = process_ext["value"] / 10.0

# ============================================================
# 2. TABLE SETTINGS
# ============================================================

output_dir = os.path.join(
    project_dir,
    "output",
    "tables"
)

os.makedirs(output_dir, exist_ok=True)

percentile_order = [
    "5th",
    "50th",
    "95th"
]

statistic_order = [
    "Mean ± SD",
    "50%",
    "5–95%",
    "17–83%"
]

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def fmt(value):
    """
    Format a rounded number using a Unicode minus sign.
    """

    if pd.isna(value):
        return ""

    return f"{value:.0f}".replace("-", "−")


def summarize_distributions(data, group_columns):
    """
    Calculate summary statistics for Monte Carlo distributions.
    """

    summary = (
        data
        .groupby(
            group_columns,
            observed=True
        )["value"]
        .agg(
            mean="mean",
            sd="std",
            p05=lambda x: x.quantile(0.05),
            p17=lambda x: x.quantile(0.17),
            p50=lambda x: x.quantile(0.50),
            p83=lambda x: x.quantile(0.83),
            p95=lambda x: x.quantile(0.95)
        )
        .reset_index()
    )

    summary["Mean ± SD"] = (
        summary["mean"].map(fmt)
        + " ± "
        + summary["sd"].map(fmt)
    )

    summary["50%"] = summary["p50"].map(fmt)

    summary["5–95%"] = (
        summary["p05"].map(fmt)
        + "–"
        + summary["p95"].map(fmt)
    )

    summary["17–83%"] = (
        summary["p17"].map(fmt)
        + "–"
        + summary["p83"].map(fmt)
    )

    return summary


# ============================================================
# 4. TABLE 1 SUMMARY DATA
# ============================================================

table1_stats = summarize_distributions(
    mc_total_ext,
    [
        "ice_sheet",
        "warming",
        "year"
    ]
)

table1_stats["ice_sheet"] = pd.Categorical(
    table1_stats["ice_sheet"],
    categories=ice_sheet_order,
    ordered=True
)


table1 = (
    table1_stats[
        [
            "year",
            "ice_sheet",
            "warming",
            "Mean ± SD",
            "50%",
            "5–95%",
            "17–83%"
        ]
    ]
    .sort_values([
        "year",
        "warming",
        "ice_sheet"
    ])
    .rename(columns={
        "year": "Year",
        "ice_sheet": "Ice sheet",
        "warming": "Warming (°C)"
    })
    .reset_index(drop=True)
)

print("\nTABLE 1")
print(table1.head(10))


# ============================================================
# 5. TABLE 2 SUMMARY DATA
# ============================================================

table2_stats = summarize_distributions(
    process_ext,
    [
        "ice_sheet",
        "process",
        "warming",
        "year"
    ]
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
            "year",
            "ice_sheet",
            "process",
            "warming",
            "Mean ± SD",
            "50%",
            "5–95%",
            "17–83%"
        ]
    ]
    .sort_values([
        "year",
        "warming",
        "process",
        "ice_sheet"
    ])
    .rename(columns={
        "year": "Year",
        "ice_sheet": "Ice sheet",
        "process": "Process",
        "warming": "Warming (°C)"
    })
    .reset_index(drop=True)
)

print("\nTABLE 2")
print(table2.head(12))


# ============================================================
# 6. PUBLICATION-STYLE TABLE 1 DATA
# ============================================================

table1_wide = (
    table1_stats
    .pivot(
        index=[
            "year",
            "ice_sheet"
        ],
        columns="warming",
        values=statistic_order
    )
    .swaplevel(
        0,
        1,
        axis=1
    )
    .reindex(
        columns=pd.MultiIndex.from_product(
            [
                warming_order,
                statistic_order
            ],
            names=[
                "Warming",
                "Statistic"
            ]
        )
    )
)


table1_row_order = pd.MultiIndex.from_product(
    [
        sorted(
            table1_stats["year"].unique()
        ),
        ice_sheet_order
    ],
    names=[
        "year",
        "ice_sheet"
    ]
)

table1_wide = table1_wide.reindex(
    table1_row_order
)


# ============================================================
# 7. PUBLICATION-STYLE TABLE 2 DATA
# ============================================================

# Summarize each Monte Carlo distribution as:
# median (5th–95th percentile)
table2_publication_summary = (
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
    .agg(
        p05=lambda x: x.quantile(0.05),
        p50=lambda x: x.quantile(0.50),
        p95=lambda x: x.quantile(0.95)
    )
    .reset_index()
)

# Format each result as: 50th percentile (5th–95th percentile)
table2_publication_summary["summary"] = (
    table2_publication_summary["p50"].map(fmt)
    + " ("
    + table2_publication_summary["p05"].map(fmt)
    + "–"
    + table2_publication_summary["p95"].map(fmt)
    + ")"
)

# Arrange years and warming levels as rows and
# process/ice-sheet combinations as columns.
table2_wide = (
    table2_publication_summary
    .pivot(
        index=[
            "year",
            "warming"
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
                process_ice_sheet_order
            ],
            names=[
                "Process",
                "Ice sheet"
            ]
        )
    )
)

# Enforce the desired year and warming order.
table2_row_order = pd.MultiIndex.from_product(
    [
        sorted(table2_publication_summary["year"].unique()),
        warming_order
    ],
    names=[
        "year",
        "warming"
    ]
)

table2_wide = table2_wide.reindex(
    table2_row_order
)


# ============================================================
# 8. EXPORT REGULAR SUMMARY TABLES
# ============================================================

def export_summary_tables(
    table1_data,
    table2_data,
    output_file
):
    """
    Export regular long-format summary tables to one workbook.
    """

    with pd.ExcelWriter(
        output_file,
        engine="xlsxwriter"
    ) as writer:

        table1_data.to_excel(
            writer,
            sheet_name="Ice-sheet contributions",
            index=False,
            startrow=2
        )

        table2_data.to_excel(
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

        sheet_settings = [
            (
                "Ice-sheet contributions",
                table1_data,
                "Projected ice-sheet contributions "
                "to global mean sea level"
            ),
            (
                "Process contributions",
                table2_data,
                "Projected ice-sheet contributions "
                "by process"
            )
        ]

        for sheet_name, table_data, title in sheet_settings:

            worksheet = writer.sheets[sheet_name]

            worksheet.write(
                "A1",
                title,
                title_format
            )

            for column_number, column_name in enumerate(
                table_data.columns
            ):
                worksheet.write(
                    2,
                    column_number,
                    column_name,
                    header_format
                )

            worksheet.freeze_panes(3, 0)

            worksheet.autofilter(
                2,
                0,
                len(table_data) + 2,
                len(table_data.columns) - 1
            )

            worksheet.set_column("A:A", 12)
            worksheet.set_column("B:B", 24)
            worksheet.set_column("C:C", 26)
            worksheet.set_column(
                "D:D",
                14,
                numeric_format
            )
            worksheet.set_column("E:H", 20)

            worksheet.hide_gridlines(2)


# ============================================================
# 9. EXPORT PUBLICATION-STYLE TABLE 1
# ============================================================

def export_publication_table1(
    table_data,
    output_file
):
    """
    Export publication-style ice-sheet contribution table.
    """

    number_of_columns = (
        1
        + len(warming_order)
        * len(statistic_order)
    )

    final_column = number_of_columns - 1

    with pd.ExcelWriter(
        output_file,
        engine="xlsxwriter"
    ) as writer:

        workbook = writer.book

        worksheet = workbook.add_worksheet(
            "Table 1"
        )

        writer.sheets["Table 1"] = worksheet

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

        worksheet.merge_range(
            0,
            0,
            0,
            final_column,
            "Table 1. Projected sea-level rise "
            "contributions from each ice sheet "
            "and combined",
            title_format
        )

        worksheet.merge_range(
            2,
            0,
            3,
            0,
            "Year and ice sheet",
            group_header_format
        )

        start_column = 1

        for warming in warming_order:

            end_column = (
                start_column
                + len(statistic_order)
                - 1
            )

            worksheet.merge_range(
                2,
                start_column,
                2,
                end_column,
                f"{warming:.1f}°C",
                group_header_format
            )

            for offset, statistic in enumerate(
                statistic_order
            ):
                worksheet.write(
                    3,
                    start_column + offset,
                    statistic,
                    subheader_format
                )

            start_column = end_column + 1

        output_row = 4

        years = table_data.index.get_level_values(
            "year"
        ).unique()

        for year in years:

            worksheet.write(
                output_row,
                0,
                int(year),
                year_format
            )

            output_row += 1

            year_data = table_data.xs(
                year,
                level="year"
            )

            for ice_sheet in ice_sheet_order:

                if ice_sheet not in year_data.index:
                    continue

                worksheet.write(
                    output_row,
                    0,
                    ice_sheet,
                    ice_sheet_format
                )

                output_column = 1

                for warming in warming_order:
                    for statistic in statistic_order:

                        value = year_data.loc[
                            ice_sheet,
                            (warming, statistic)
                        ]

                        worksheet.write(
                            output_row,
                            output_column,
                            value,
                            value_format
                        )

                        output_column += 1

                output_row += 1

        for column in range(number_of_columns):

            worksheet.write_blank(
                output_row,
                column,
                None,
                bottom_border_format
            )

        worksheet.set_column(0, 0, 24)
        worksheet.set_column(
            1,
            final_column,
            14
        )

        worksheet.freeze_panes(4, 1)
        worksheet.hide_gridlines(2)

        worksheet.set_landscape()
        worksheet.fit_to_pages(1, 0)


# ============================================================
# 10. EXPORT PUBLICATION-STYLE TABLE 2
# ============================================================

def export_publication_table2(
    table_data,
    output_file
):
    """
    Export publication-style process contribution table.

    Each cell reports the 50th percentile followed by the
    5th–95th percentile range in parentheses.
    """

    first_data_column = 1

    columns_per_process = len(
        process_ice_sheet_order
    )

    final_column = (
        first_data_column
        + len(process_order)
        * columns_per_process
        - 1
    )

    with pd.ExcelWriter(
        output_file,
        engine="xlsxwriter"
    ) as writer:

        workbook = writer.book

        worksheet = workbook.add_worksheet(
            "Table 2"
        )

        writer.sheets["Table 2"] = worksheet

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

        year_warming_format = workbook.add_format({
            "align": "left",
            "valign": "vcenter",
            "bottom": 1
        })

        value_format = workbook.add_format({
            "align": "center",
            "valign": "vcenter",
            "bottom": 1
        })

        worksheet.merge_range(
            0,
            0,
            0,
            final_column,
            "Table 2. Projected sea-level rise "
            "contributions from each ice sheet "
            "and process (cm)",
            title_format
        )

        worksheet.merge_range(
            1,
            0,
            1,
            final_column,
            "Values are reported as the 50th percentile "
            "(5th–95th percentile) of the Monte Carlo "
            "distributions.",
            note_format
        )

        worksheet.merge_range(
            3,
            0,
            4,
            0,
            "Year and warming",
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

            for offset, ice_sheet in enumerate(
                process_ice_sheet_order
            ):
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

            for warming in warming_order:

                if (year, warming) not in table_data.index:
                    continue

                worksheet.write(
                    output_row,
                    0,
                    f"{int(year)}\n{warming:.1f}°C",
                    year_warming_format
                )

                output_column = first_data_column

                for process in process_order:
                    for ice_sheet in process_ice_sheet_order:

                        value = table_data.loc[
                            (year, warming),
                            (process, ice_sheet)
                        ]

                        worksheet.write(
                            output_row,
                            output_column,
                            "" if pd.isna(value) else value,
                            value_format
                        )

                        output_column += 1

                worksheet.set_row(output_row, 30)
                output_row += 1

        worksheet.set_column(0, 0, 17)
        worksheet.set_column(
            first_data_column,
            final_column,
            16
        )

        worksheet.set_row(0, 20)
        worksheet.set_row(1, 28)
        worksheet.set_row(3, 20)
        worksheet.set_row(4, 18)

        worksheet.freeze_panes(5, 1)
        worksheet.hide_gridlines(2)

        worksheet.set_landscape()
        worksheet.fit_to_pages(1, 0)

        worksheet.set_margins(
            left=0.25,
            right=0.25,
            top=0.4,
            bottom=0.4
        )


# ============================================================
# 11. WRITE OUTPUT FILES
# ============================================================

summary_file = os.path.join(
    output_dir,
    "B25_ice_sheet_summary_tables_UPDATED.xlsx"
)

publication_table1_file = os.path.join(
    output_dir,
    "B25_Table1_publication_format_UPDATED.xlsx"
)

publication_table2_file = os.path.join(
    output_dir,
    "B25_Table2_process_contributions_"
    "publication_format_UPDATED.xlsx"
)


export_summary_tables(
    table1,
    table2,
    summary_file
)

export_publication_table1(
    table1_wide,
    publication_table1_file
)

export_publication_table2(
    table2_wide,
    publication_table2_file
)


print("\nFiles saved successfully:")
print(summary_file)
print(publication_table1_file)
print(publication_table2_file)