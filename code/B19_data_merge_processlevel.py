#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine B19 process-level and total ice-sheet Monte Carlo samples
from the six Low/High, 2100/2200/2300 source workbooks.

@author: elizabeth
"""

from pathlib import Path

import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

# Folder containing the six USUK Excel files.
input_dir = Path("/Users/elizabeth/Downloads/USUK_XLS_corr_271119")

# Output workbook.
output_file = input_dir / "USUK_PW01_mega_spreadsheet.xlsx"

columns_to_extract = [
    # Process-level contributions
    "PW01EA",
    "PW01ED",
    "PW01GA",
    "PW01GD",
    "PW01GR",
    "PW01WA",
    "PW01WD",
    "PW01WR",

    # Total ice-sheet contributions
    "USEU_PW01_E",
    "USEU_PW01_G",
    "USEU_PW01_W",
]

# Workbook filename mapped to the suffix that will be added
# to every extracted column.
file_suffixes = {
    "USUK_2100L_corr.xlsx": "2100L",
    "USUK_2100H_corr.xlsx": "2100H",
    "USUK_2200L_corr.xlsx": "2200L",
    "USUK_2200H_corr.xlsx": "2200H",
    "USUK_2300L_corr.xlsx": "2300L",
    "USUK_2300H_corr.xlsx": "2300H",
}


# ============================================================
# HELPER FUNCTION
# ============================================================

def extract_and_rename_columns(
    file_path: Path,
    suffix: str,
) -> pd.DataFrame:
    """
    Read one Excel workbook, extract the requested process-level and
    total ice-sheet columns, and append the year/scenario suffix to
    each column name.
    """

    print(f"Reading: {file_path.name}")

    # By default, pandas reads the first worksheet.
    dataframe = pd.read_excel(file_path)

    # Remove leading/trailing spaces that may exist in Excel headers.
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    missing_columns = [
        column
        for column in columns_to_extract
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing these columns:\n"
            f"{missing_columns}\n\n"
            f"Columns found in the workbook:\n"
            f"{dataframe.columns.tolist()}"
        )

    extracted = dataframe.loc[
        :,
        columns_to_extract
    ].copy()

    extracted = extracted.rename(
        columns={
            column: f"{column}_{suffix}"
            for column in columns_to_extract
        }
    )

    return extracted


# ============================================================
# READ AND COMBINE ALL FILES
# ============================================================

combined_dataframes = []

for filename, suffix in file_suffixes.items():

    file_path = input_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{file_path}"
        )

    extracted = extract_and_rename_columns(
        file_path=file_path,
        suffix=suffix,
    )

    combined_dataframes.append(extracted)


# Combine by row position
mega_dataframe = pd.concat(
    combined_dataframes,
    axis=1,
)


# ============================================================
# VALIDATE ROW COUNTS
# ============================================================

row_counts = {
    filename: len(dataframe)
    for filename, dataframe in zip(
        file_suffixes.keys(),
        combined_dataframes,
    )
}

print("\nRows extracted from each workbook:")

for filename, row_count in row_counts.items():
    print(f"  {filename}: {row_count:,}")

if len(set(row_counts.values())) > 1:
    print(
        "\nWARNING: The workbooks do not all contain the same "
        "number of rows. Pandas aligned them by row index, and "
        "shorter columns will contain blank cells at the bottom."
    )


# ============================================================
# SAVE THE MEGA WORKBOOK
# ============================================================

mega_dataframe.to_excel(
    output_file,
    index=False,
)

print(
    "\nCreated mega spreadsheet:\n"
    f"{output_file}"
)

print(
    f"\nFinal dimensions: "
    f"{mega_dataframe.shape[0]:,} rows × "
    f"{mega_dataframe.shape[1]:,} columns"
)