#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:58:12 2026

@author: elizabeth
"""

# ============================================================
# DP21 TEMPORAL CORRELATION MATRICES
#
# Calculates:
#   1. Scenario-specific correlation matrices
#   2. B19-style combined-scenario correlation matrices
#   3. Scenario-minus-combined difference matrices
#   4. Summary comparison tables
#
# Components:
#   - EAIS
#   - WAIS
#   - Total AIS
#
# Years:
#   - 2100
#   - 2200
#   - 2300
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset


# ------------------------------------------------------------
# 1. USER SETTINGS
# ------------------------------------------------------------

DATA_DIR = Path(
    "/Users/elizabeth/FACTS2/data/module_specific_input_data/deconto21-ais"
)

OUTPUT_DIR = Path(
    "/Users/elizabeth/Documents/GitHub/"
    "ice-sheet-sej-2025/data/temporal_correlation_matrices"
)

TARGET_YEARS = np.array([2100, 2200, 2300])

SCENARIOS = ["rcp26", "rcp45", "rcp85"]

COMPONENTS = ["EAIS", "WAIS", "Total_AIS"]


# ------------------------------------------------------------
# 2. LOAD ONE DP21 NETCDF FILE
# ------------------------------------------------------------

def load_dp21_file(filepath):
    """
    Load years, sample IDs, and ensemble trajectories from one
    DP21 NetCDF file.

    Expected structure
    ------------------
    years:
        Shape (n_years,)

    samples:
        Shape (n_members,)

    samps:
        Shape (n_years, n_members)

    Returns
    -------
    years : numpy.ndarray
    sample_ids : numpy.ndarray
    samps : numpy.ndarray
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with Dataset(filepath, mode="r") as ds:

        required_variables = {"years", "samples", "samps"}

        missing_variables = required_variables.difference(
            ds.variables.keys()
        )

        if missing_variables:
            raise KeyError(
                f"{filepath.name} is missing required variables: "
                f"{sorted(missing_variables)}"
            )

        years = np.asarray(ds.variables["years"][:])
        sample_ids = np.asarray(ds.variables["samples"][:])

        samps = np.ma.filled(
            ds.variables["samps"][:],
            fill_value=np.nan
        ).astype(float)

    expected_shape = (len(years), len(sample_ids))

    if samps.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape in {filepath.name}: {samps.shape}. "
            f"Expected {expected_shape}."
        )

    return years, sample_ids, samps


# ------------------------------------------------------------
# 3. LOAD EAIS, WAIS, AND TOTAL AIS FOR ONE SCENARIO
# ------------------------------------------------------------

def load_scenario(scenario):
    """
    Load paired EAIS and WAIS members for one scenario and
    construct Total AIS member by member.

    Returns
    -------
    years : numpy.ndarray

    component_data : dict
        Each value has shape:
            n_years x n_members
    """

    eais_file = DATA_DIR / f"dp21_eais_{scenario}.nc"
    wais_file = DATA_DIR / f"dp21_wais_{scenario}.nc"

    eais_years, eais_ids, eais = load_dp21_file(eais_file)
    wais_years, wais_ids, wais = load_dp21_file(wais_file)

    # Confirm that the component files can be paired.
    if not np.array_equal(eais_years, wais_years):
        raise ValueError(
            f"EAIS and WAIS years do not match for {scenario}."
        )

    if not np.array_equal(eais_ids, wais_ids):
        raise ValueError(
            f"EAIS and WAIS sample IDs do not match for {scenario}."
        )

    if eais.shape != wais.shape:
        raise ValueError(
            f"EAIS and WAIS shapes differ for {scenario}: "
            f"{eais.shape} versus {wais.shape}."
        )

    # FACTS treats EAIS member i and WAIS member i as paired.
    total_ais = eais + wais

    component_data = {
        "EAIS": eais,
        "WAIS": wais,
        "Total_AIS": total_ais
    }

    return eais_years, component_data


# ------------------------------------------------------------
# 4. EXTRACT TARGET YEARS
# ------------------------------------------------------------

def extract_target_years(years, samps, target_years):
    """
    Extract target years and return an array organized as:

        rows    = ensemble members
        columns = projection years

    Output shape:
        n_members x n_target_years
    """

    missing_years = [
        year for year in target_years
        if year not in years
    ]

    if missing_years:
        raise ValueError(
            f"Requested years are unavailable: {missing_years}"
        )

    year_indices = [
        np.flatnonzero(years == year)[0]
        for year in target_years
    ]

    # Original shape after extraction:
    # n_target_years x n_members
    #
    # Transpose to:
    # n_members x n_target_years
    selected = samps[year_indices, :].T

    return selected


# ------------------------------------------------------------
# 5. CALCULATE CORRELATION MATRIX
# ------------------------------------------------------------

def calculate_correlation_matrix(selected_samps, target_years):
    """
    Calculate the Pearson temporal correlation matrix.

    selected_samps must have:
        rows    = ensemble members
        columns = years
    """

    selected_samps = np.asarray(selected_samps, dtype=float)

    if selected_samps.ndim != 2:
        raise ValueError(
            "selected_samps must be a two-dimensional array."
        )

    if selected_samps.shape[1] != len(target_years):
        raise ValueError(
            "The number of columns must equal the number of "
            "target years."
        )

    # Remove any member containing a missing value.
    complete_rows = np.all(
        np.isfinite(selected_samps),
        axis=1
    )

    clean_samps = selected_samps[complete_rows, :]

    if clean_samps.shape[0] < 2:
        raise ValueError(
            "Fewer than two complete ensemble members remain."
        )

    correlation = np.corrcoef(
        clean_samps,
        rowvar=False
    )

    return pd.DataFrame(
        correlation,
        index=target_years,
        columns=target_years
    )


# ------------------------------------------------------------
# 6. CONVERT ONE MATRIX TO UNIQUE YEAR-PAIR RECORDS
# ------------------------------------------------------------

def matrix_to_pair_records(
    corr_df,
    component,
    matrix_type,
    scenario=None
):
    """
    Convert a symmetric correlation matrix into records for the
    unique year pairs:

        2100-2200
        2100-2300
        2200-2300
    """

    records = []

    years = list(corr_df.columns)

    for first_index in range(len(years)):

        for second_index in range(first_index + 1, len(years)):

            year_1 = years[first_index]
            year_2 = years[second_index]

            records.append({
                "component": component,
                "matrix_type": matrix_type,
                "scenario": scenario,
                "year_1": year_1,
                "year_2": year_2,
                "year_pair": f"{year_1}-{year_2}",
                "correlation": corr_df.loc[year_1, year_2]
            })

    return records


# ------------------------------------------------------------
# 7. MAIN ANALYSIS
# ------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Stores target-year samples with shape:
    # members x years
    scenario_samples = {}

    # Stores scenario-specific correlation matrices.
    scenario_correlations = {}

    print("\nLoading DP21 ensembles...")

    # --------------------------------------------------------
    # Load and process each individual scenario
    # --------------------------------------------------------

    for scenario in SCENARIOS:

        years, component_data = load_scenario(scenario)

        scenario_samples[scenario] = {}
        scenario_correlations[scenario] = {}

        for component, samps in component_data.items():

            selected = extract_target_years(
                years=years,
                samps=samps,
                target_years=TARGET_YEARS
            )

            scenario_samples[scenario][component] = selected

            corr_df = calculate_correlation_matrix(
                selected_samps=selected,
                target_years=TARGET_YEARS
            )

            scenario_correlations[scenario][component] = corr_df

            output_file = (
                OUTPUT_DIR
                / (
                    f"dp21_{component.lower()}_"
                    f"{scenario}_temporal_correlation.csv"
                )
            )

            corr_df.to_csv(
                output_file,
                float_format="%.6f"
            )

    # --------------------------------------------------------
    # Construct B19-style combined-scenario matrices
    # --------------------------------------------------------

    combined_correlations = {}
    combined_samples = {}

    for component in COMPONENTS:

        # B19-style vertical concatenation:
        #
        # [RCP2.6 members
        #  RCP4.5 members
        #  RCP8.5 members]
        #
        # With 109 members per scenario, the combined array
        # contains 327 rows and 3 year columns.
        combined = np.vstack([
            scenario_samples[scenario][component]
            for scenario in SCENARIOS
        ])

        combined_samples[component] = combined

        corr_df = calculate_correlation_matrix(
            selected_samps=combined,
            target_years=TARGET_YEARS
        )

        combined_correlations[component] = corr_df

        output_file = (
            OUTPUT_DIR
            / (
                f"dp21_{component.lower()}_"
                "combined_scenarios_temporal_correlation.csv"
            )
        )

        corr_df.to_csv(
            output_file,
            float_format="%.6f"
        )

    # --------------------------------------------------------
    # Calculate scenario-minus-combined differences
    # --------------------------------------------------------

    difference_matrices = {}

    for scenario in SCENARIOS:

        difference_matrices[scenario] = {}

        for component in COMPONENTS:

            difference_df = (
                scenario_correlations[scenario][component]
                - combined_correlations[component]
            )

            difference_matrices[scenario][component] = difference_df

            output_file = (
                OUTPUT_DIR
                / (
                    f"dp21_{component.lower()}_"
                    f"{scenario}_minus_combined_correlation.csv"
                )
            )

            difference_df.to_csv(
                output_file,
                float_format="%.6f"
            )

    # --------------------------------------------------------
    # Create one long-format comparison table
    # --------------------------------------------------------

    comparison_records = []

    for component in COMPONENTS:

        comparison_records.extend(
            matrix_to_pair_records(
                corr_df=combined_correlations[component],
                component=component,
                matrix_type="combined_scenarios",
                scenario="combined"
            )
        )

        for scenario in SCENARIOS:

            comparison_records.extend(
                matrix_to_pair_records(
                    corr_df=scenario_correlations[scenario][component],
                    component=component,
                    matrix_type="scenario_specific",
                    scenario=scenario
                )
            )

    comparison_df = pd.DataFrame(comparison_records)

    comparison_df.to_csv(
        OUTPUT_DIR
        / "dp21_temporal_correlation_comparison_long.csv",
        index=False,
        float_format="%.6f"
    )

    # --------------------------------------------------------
    # Create a wide-format comparison table
    # --------------------------------------------------------

    comparison_wide = comparison_df.pivot_table(
        index=["component", "year_pair"],
        columns="scenario",
        values="correlation"
    ).reset_index()

    # Put columns in a useful order.
    desired_columns = [
        "component",
        "year_pair",
        "combined",
        "rcp26",
        "rcp45",
        "rcp85"
    ]

    comparison_wide = comparison_wide[
        desired_columns
    ]

    # Differences relative to the combined B19-style matrix.
    for scenario in SCENARIOS:

        comparison_wide[
            f"{scenario}_minus_combined"
        ] = (
            comparison_wide[scenario]
            - comparison_wide["combined"]
        )

    comparison_wide.to_csv(
        OUTPUT_DIR
        / "dp21_temporal_correlation_comparison_wide.csv",
        index=False,
        float_format="%.6f"
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("B19-STYLE COMBINED-SCENARIO MATRICES")
    print("=" * 72)

    for component in COMPONENTS:

        display_name = component.replace("_", " ")

        n_members = combined_samples[component].shape[0]

        print(
            f"\n{display_name} "
            f"(combined ensemble members = {n_members}):"
        )

        print(
            combined_correlations[component]
            .round(4)
            .to_string()
        )

    print("\n" + "=" * 72)
    print("SCENARIO-SPECIFIC VERSUS COMBINED COMPARISON")
    print("=" * 72)

    print(
        comparison_wide
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("FINISHED")
    print("=" * 72)

    print(f"\nOutputs saved to:\n{OUTPUT_DIR}")

    return {
        "scenario_correlations": scenario_correlations,
        "combined_correlations": combined_correlations,
        "difference_matrices": difference_matrices,
        "comparison_long": comparison_df,
        "comparison_wide": comparison_wide
    }


# ------------------------------------------------------------
# 8. RUN SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":

    results = main()