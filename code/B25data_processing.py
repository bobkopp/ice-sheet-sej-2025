#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:59:59 2026

Read and prepare the B25 Monte Carlo ice-sheet projection data.

This script:
    1. Reads the Monte Carlo samples.
    2. Parses the variable names.
    3. Reshapes the data from wide to long format.
    4. Calculates total AIS as EAIS + WAIS.
    5. Calculates runoff minus accumulation.

@author: elizabeth
"""

## Import libraries
import os
import re
import pandas as pd
import numpy as np


# ============================================================
# 1. SETTINGS
# ============================================================

project_dir = "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025"

data_file = os.path.join(
    project_dir,
    "data",
    "sej2025_SplitPanels_liz_updated.csv"
)

output_dir = os.path.join(
    project_dir,
    "output"
)


year_map = {
    "21": 2100,
    "22": 2200,
    "23": 2300
}

warming_map = {
    "2": 2.0,
    "35": 3.5,
    "5": 5.0
}

warming_label_map = {
    2.0: "2.0°C",
    3.5: "3.5°C",
    5.0: "5.0°C"
}

ice_sheet_map = {
    "E": "EAIS",
    "W": "WAIS",
    "G": "GrIS",
    "SLR": "Total ice-sheet SLR"
}

process_map = {
    "A": "Accumulation",
    "D": "Discharge",
    "R": "Runoff",
    "TOTAL": "Total"
}


# Orders used by tables and figures
ice_sheet_order = [
    "EAIS",
    "WAIS",
    "Total AIS",
    "GrIS",
    "Total ice-sheet SLR"
]

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

warming_order = [
    2.0,
    3.5,
    5.0
]


# ============================================================
# PARSE VARIABLE NAMES
# ============================================================

def parse_variable_name(column_name):
    """
    Parse a Monte Carlo variable name.

    Examples
    --------
    EA2_21
        EAIS accumulation, 2.0°C, 2100

    WD35_22
        WAIS discharge, 3.5°C, 2200

    G5_23
        Total GrIS contribution, 5.0°C, 2300

    SLR2_21
        Total ice-sheet contribution, 2.0°C, 2100
    """

    match = re.fullmatch(
        r"(SLR|[EWG])([ADR]?)(2|35|5)_(21|22|23)",
        column_name
    )

    if match is None:
        return None

    ice_code, process_code, warming_code, year_code = (
        match.groups()
    )

    process_code = process_code or "TOTAL"

    return {
        "variable": column_name,
        "ice_sheet_code": ice_code,
        "ice_sheet": ice_sheet_map[ice_code],
        "process_code": process_code,
        "process": process_map[process_code],
        "warming_code": warming_code,
        "warming": warming_map[warming_code],
        "year": year_map[year_code]
    }


# ============================================================
# READ AND RESHAPE DATA
# ============================================================

def read_monte_carlo_data(file_path=data_file):
    """
    Read the Monte Carlo samples and attach a sample ID.
    """

    print(f"Reading Monte Carlo samples from:\n{file_path}\n")

    mc_df = pd.read_csv(file_path)

    mc_df["sample_id"] = np.arange(
        len(mc_df),
        dtype=int
    )

    return mc_df


def create_variable_metadata(mc_df):
    """
    Parse all Monte Carlo variable names into a metadata table.
    """

    variable_columns = [
        column
        for column in mc_df.columns
        if column != "sample_id"
    ]

    unparsed_variables = [
        column
        for column in variable_columns
        if parse_variable_name(column) is None
    ]

    if unparsed_variables:
        raise ValueError(
            "The following variables could not be parsed:\n"
            + "\n".join(unparsed_variables)
        )

    metadata = [
        parse_variable_name(column)
        for column in variable_columns
    ]

    meta_df = pd.DataFrame(metadata)

    print(f"Parsed variables: {len(meta_df):,}")

    return meta_df


def reshape_to_long(mc_df, meta_df):
    """
    Reshape the Monte Carlo samples from wide to long format
    and attach the parsed variable metadata.
    """

    mc_long = (
        mc_df
        .melt(
            id_vars="sample_id",
            var_name="variable",
            value_name="value"
        )
        .merge(
            meta_df,
            on="variable",
            how="left",
            validate="many_to_one"
        )
    )

    mc_long["scenario_name"] = (
        mc_long["warming"]
        .map(warming_label_map)
    )

    print(f"Long-format rows: {len(mc_long):,}")

    return mc_long


# ============================================================
# DERIVED ICE-SHEET TOTALS
# ============================================================

def add_total_ais(mc_long):
    """
    Add total AIS as the draw-by-draw sum of EAIS and WAIS.

    Returns
    -------
    mc_total_ext
        Total contributions for EAIS, WAIS, GrIS, total AIS,
        and total ice-sheet SLR.
    """

    mc_total = mc_long.loc[
        mc_long["process_code"] == "TOTAL"
    ].copy()

    ais_df = (
        mc_total
        .loc[
            mc_total["ice_sheet_code"].isin(["E", "W"])
        ]
        .pivot(
            index=[
                "sample_id",
                "warming",
                "warming_code",
                "year",
                "scenario_name"
            ],
            columns="ice_sheet_code",
            values="value"
        )
        .dropna(subset=["E", "W"])
        .reset_index()
    )

    ais_df["value"] = (
        ais_df["E"]
        + ais_df["W"]
    )

    ais_df["variable"] = pd.NA
    ais_df["ice_sheet_code"] = "AIS"
    ais_df["ice_sheet"] = "Total AIS"
    ais_df["process_code"] = "TOTAL"
    ais_df["process"] = "Total"

    # Match the original long-table structure
    ais_df = ais_df[mc_total.columns]

    mc_total_ext = pd.concat(
        [
            mc_total,
            ais_df
        ],
        ignore_index=True
    )

    return mc_total_ext


# ============================================================
# DERIVED PROCESS CONTRIBUTIONS
# ============================================================

def add_runoff_minus_accumulation(mc_long):
    """
    Add runoff minus accumulation for each ice sheet,
    warming level, year, and Monte Carlo realization.

    Returns
    -------
    process_ext
        Accumulation, discharge, runoff, and
        runoff-minus-accumulation contributions.
    """

    process_df = mc_long.loc[
        mc_long["process_code"].isin(["A", "D", "R"])
    ].copy()

    runoff_accumulation = (
        process_df
        .loc[
            process_df["process_code"].isin(["A", "R"])
        ]
        .pivot(
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

    # Accumulation is already represented as a signed, negative SLR
    # contribution, so net runoff-minus-accumulation is actually R + A
    runoff_accumulation["value"] = (
        runoff_accumulation["R"]
        + runoff_accumulation["A"]
    )

    runoff_accumulation["variable"] = pd.NA
    runoff_accumulation["process_code"] = "R-A"
    runoff_accumulation["process"] = (
        "Runoff minus accumulation"
    )

    # Match the original long-table structure
    runoff_accumulation = runoff_accumulation[
        process_df.columns
    ]

    process_ext = pd.concat(
        [
            process_df,
            runoff_accumulation
        ],
        ignore_index=True
    )

    return process_ext


# ============================================================
# COMPLETE DATA-PREPARATION WORKFLOW
# ============================================================

def load_and_prepare_data(file_path=data_file):
    """
    Run the full B25 Monte Carlo data-preparation workflow.

    Returns
    -------
    data : dict
        Dictionary containing:
            mc_df
            meta_df
            mc_long
            mc_total_ext
            process_ext
    """

    mc_df = read_monte_carlo_data(file_path)

    meta_df = create_variable_metadata(mc_df)

    mc_long = reshape_to_long(
        mc_df,
        meta_df
    )

    mc_total_ext = add_total_ais(
        mc_long
    )

    process_ext = add_runoff_minus_accumulation(
        mc_long
    )

    return {
        "mc_df": mc_df,
        "meta_df": meta_df,
        "mc_long": mc_long,
        "mc_total_ext": mc_total_ext,
        "process_ext": process_ext
    }
