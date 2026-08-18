#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 09:00:08 2026


Apply B19-style temporal dependence to the B25 ice-sheet
Structured Expert Judgment Monte Carlo samples.

This script:

    1. Uses the general B25 preprocessing workflow.
    2. Reads DP21 Total AIS temporal correlation matrices from CSV.
    3. Runs three separate temporal-coupling experiments:

           B25 2.0°C  paired with DP21 RCP2.6
           B25 3.5°C  paired with DP21 RCP4.5
           B25 5.0°C  paired with DP21 RCP8.5

    4. For each experiment:
           a. Calculates Total AIS = EAIS + WAIS.
           b. Sorts each complete yearly panel by Total AIS for the
              selected warming level.
           c. Generates a correlated latent Gaussian rank process.
           d. Uses the ranks to select complete yearly SEJ rows.
           e. Recombines the selected panels into trajectories.
           f. Saves coupled data, metadata, and diagnostics.

Each experiment creates a separate coupled dataset.

All variables belonging to a projection year move together.
Therefore, within-year multivariate relationships are preserved.

@author: elizabeth
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Change this import if your preprocessing filename differs.
from B25data_processing import (
    load_and_prepare_data,
    project_dir,
    data_file
)


# ============================================================
# 1. SETTINGS
# ============================================================

years = [
    2100,
    2200,
    2300
]

random_seed = 1234


correlation_dir = os.path.join(
    project_dir,
    "data",
    "temporal_correlation_matrices"
)


output_dir = os.path.join(
    project_dir,
    "output",
    "temporal_coupling",
    "total_ais_common_rank"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# Each entry defines one independent temporal-coupling experiment.
#
# The selected warming level determines:
#
#   1. which B25 Total AIS variables are used to rank the
#      complete yearly panels;
#   2. which DP21 scenario-specific temporal matrix is imposed.
experiment_settings = [
    {
        "warming": 2.0,
        "warming_label": "2.0C",
        "dp21_scenario": "rcp26",
        "matrix_file": os.path.join(
            correlation_dir,
            "dp21_total_ais_rcp26_temporal_correlation.csv"
        )
    },
    {
        "warming": 3.5,
        "warming_label": "3.5C",
        "dp21_scenario": "rcp45",
        "matrix_file": os.path.join(
            correlation_dir,
            "dp21_total_ais_rcp45_temporal_correlation.csv"
        )
    },
    {
        "warming": 5.0,
        "warming_label": "5.0C",
        "dp21_scenario": "rcp85",
        "matrix_file": os.path.join(
            correlation_dir,
            "dp21_total_ais_rcp85_temporal_correlation.csv"
        )
    }
]


# ============================================================
# 2. LOAD TEMPORAL CORRELATION MATRIX
# ============================================================

def load_temporal_correlation_matrix(
    file_path: str,
    years: list[int]
) -> np.ndarray:
    """
    Read a temporal correlation matrix from CSV.

    The function accommodates common CSV formats such as:

        ,2100,2200,2300
        2100,1.0,0.89,0.83
        2200,0.89,1.0,0.98
        2300,0.83,0.98,1.0

    or a CSV containing only the three numeric matrix columns.

    Returns
    -------
    correlation_matrix
        Numeric array ordered as 2100, 2200, 2300.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Temporal correlation matrix file was not found:\n"
            f"{file_path}"
        )

    matrix_df = pd.read_csv(
        file_path
    )

    year_strings = [
        str(year)
        for year in years
    ]

    # Preferred case: year labels exist as column names.
    if all(
        year in matrix_df.columns
        for year in year_strings
    ):
        matrix_df = matrix_df.loc[
            :,
            year_strings
        ]

    else:
        # Remove common CSV index columns.
        unnamed_columns = [
            column
            for column in matrix_df.columns
            if str(column).startswith("Unnamed:")
        ]

        if unnamed_columns:
            matrix_df = matrix_df.drop(
                columns=unnamed_columns
            )

        # Keep only numeric columns.
        numeric_df = matrix_df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        numeric_df = numeric_df.dropna(
            axis=1,
            how="all"
        )

        # A saved DataFrame index may appear as an additional
        # numeric column containing 2100, 2200, 2300.
        if numeric_df.shape == (
            len(years),
            len(years) + 1
        ):
            first_column = numeric_df.iloc[
                :,
                0
            ].to_numpy()

            if np.allclose(
                first_column,
                np.asarray(
                    years,
                    dtype=float
                )
            ):
                numeric_df = numeric_df.iloc[
                    :,
                    1:
                ]

        matrix_df = numeric_df

    correlation_matrix = matrix_df.to_numpy(
        dtype=float
    )

    return validate_correlation_matrix(
        correlation_matrix=correlation_matrix,
        years=years
    )


# ============================================================
# 3. VALIDATE TEMPORAL CORRELATION MATRIX
# ============================================================

def validate_correlation_matrix(
    correlation_matrix: np.ndarray,
    years: list[int],
    tolerance: float = 1e-10
) -> np.ndarray:
    """
    Validate a temporal correlation matrix.
    """

    corr = np.asarray(
        correlation_matrix,
        dtype=float
    )

    expected_shape = (
        len(years),
        len(years)
    )

    if corr.shape != expected_shape:
        raise ValueError(
            "Correlation matrix must have shape "
            f"{expected_shape}, but has shape {corr.shape}."
        )

    if not np.all(
        np.isfinite(corr)
    ):
        raise ValueError(
            "Correlation matrix contains non-finite values."
        )

    if not np.allclose(
        corr,
        corr.T,
        atol=tolerance
    ):
        raise ValueError(
            "Correlation matrix must be symmetric."
        )

    if not np.allclose(
        np.diag(corr),
        1.0,
        atol=tolerance
    ):
        raise ValueError(
            "Correlation matrix diagonal must equal one."
        )

    if np.any(
        corr < -1.0 - tolerance
    ) or np.any(
        corr > 1.0 + tolerance
    ):
        raise ValueError(
            "Correlation matrix entries must lie between -1 and 1."
        )

    eigenvalues = np.linalg.eigvalsh(
        corr
    )

    if eigenvalues.min() < -tolerance:
        raise ValueError(
            "Correlation matrix is not positive semidefinite. "
            f"Minimum eigenvalue: {eigenvalues.min():.6g}"
        )

    return corr


# ============================================================
# 4. IDENTIFY COMPLETE YEAR PANELS
# ============================================================

def get_columns_by_year(
    meta_df: pd.DataFrame,
    years: list[int]
) -> dict[int, list[str]]:
    """
    Return the original scientific variables belonging to each
    projection year.
    """

    columns_by_year = {}

    for year in years:

        year_columns = (
            meta_df
            .loc[
                meta_df["year"] == year,
                "variable"
            ]
            .tolist()
        )

        if not year_columns:
            raise ValueError(
                f"No scientific variables were found for {year}."
            )

        columns_by_year[year] = year_columns

    return columns_by_year


# ============================================================
# 5. EXTRACT TOTAL AIS RANKING VALUES
# ============================================================

def get_total_ais_by_year(
    mc_total_ext: pd.DataFrame,
    warming: float,
    years: list[int]
) -> dict[int, pd.Series]:
    """
    Extract draw-by-draw Total AIS values for one B25 warming level.

    Returns
    -------
    total_ais_by_year
        Dictionary containing one Total AIS Series per year.
        Each Series is indexed by original sample_id.
    """

    total_ais = mc_total_ext.loc[
        (mc_total_ext["ice_sheet"] == "Total AIS")
        & np.isclose(
            mc_total_ext["warming"].astype(float),
            warming
        )
        & mc_total_ext["year"].isin(years)
    ].copy()

    if total_ais.empty:
        raise ValueError(
            "No Total AIS values were found for warming level "
            f"{warming}°C."
        )

    duplicate_mask = total_ais.duplicated(
        subset=[
            "sample_id",
            "year"
        ],
        keep=False
    )

    if duplicate_mask.any():
        raise ValueError(
            "More than one Total AIS value was found for a "
            "sample_id-year combination."
        )

    total_ais_by_year = {}

    for year in years:

        year_total = (
            total_ais
            .loc[
                total_ais["year"] == year,
                [
                    "sample_id",
                    "value"
                ]
            ]
            .set_index("sample_id")
            ["value"]
            .sort_index()
        )

        if year_total.empty:
            raise ValueError(
                f"No Total AIS values were found for {year}."
            )

        total_ais_by_year[year] = year_total

    sample_counts = {
        year: len(values)
        for year, values in total_ais_by_year.items()
    }

    if len(
        set(sample_counts.values())
    ) != 1:
        raise ValueError(
            "Total AIS sample counts differ among years:\n"
            f"{sample_counts}"
        )

    return total_ais_by_year


# ============================================================
# 6. GENERATE CORRELATED LATENT RANKS
# ============================================================

def generate_correlated_rank_process(
    n_samples: int,
    correlation_matrix: np.ndarray,
    random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a temporally correlated Gaussian rank process.

    Returns
    -------
    latent_gaussian
        Shape (n_samples, n_years).

    rank_positions
        Zero-based empirical ranks.

        Each year column is a complete permutation of:

            0, 1, ..., n_samples - 1
    """

    rng = np.random.default_rng(
        random_seed
    )

    n_years = correlation_matrix.shape[0]

    latent_gaussian = rng.multivariate_normal(
        mean=np.zeros(n_years),
        cov=correlation_matrix,
        size=n_samples
    )

    rank_positions = np.empty(
        latent_gaussian.shape,
        dtype=int
    )

    for year_position in range(n_years):

        rank_positions[:, year_position] = (
            rankdata(
                latent_gaussian[:, year_position],
                method="ordinal"
            )
            .astype(int)
            - 1
        )

    return (
        latent_gaussian,
        rank_positions
    )


# ============================================================
# 7. TEMPORALLY COUPLE COMPLETE YEAR PANELS
# ============================================================

def couple_antarctica_year_panels(
    mc_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    mc_total_ext: pd.DataFrame,
    warming: float,
    years: list[int],
    correlation_matrix: np.ndarray,
    random_seed: int | None = 1234
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Couple complete B25 yearly panels using a Total-AIS-ranked
    Gaussian copula.

    The entire projection-year panel moves together. Only the
    connection among years is reconstructed.
    """

    corr = validate_correlation_matrix(
        correlation_matrix=correlation_matrix,
        years=years
    )

    columns_by_year = get_columns_by_year(
        meta_df=meta_df,
        years=years
    )

    total_ais_by_year = get_total_ais_by_year(
        mc_total_ext=mc_total_ext,
        warming=warming,
        years=years
    )

    n_samples = len(
        mc_df
    )

    expected_sample_ids = np.arange(
        n_samples,
        dtype=int
    )

    if not np.array_equal(
        np.sort(
            mc_df["sample_id"].to_numpy()
        ),
        expected_sample_ids
    ):
        raise ValueError(
            "mc_df sample_id values must be consecutive integers "
            "from 0 to N - 1."
        )

    for year in years:

        if not np.array_equal(
            total_ais_by_year[
                year
            ].index.to_numpy(),
            expected_sample_ids
        ):
            raise ValueError(
                "Total AIS sample IDs do not match the original "
                f"Monte Carlo rows for {year}."
            )

    (
        latent_gaussian,
        rank_positions
    ) = generate_correlated_rank_process(
        n_samples=n_samples,
        correlation_matrix=corr,
        random_seed=random_seed
    )

    coupled_panels = []
    metadata_records = []

    for year_position, year in enumerate(years):

        year_columns = columns_by_year[
            year
        ]

        year_panel = mc_df.loc[
            :,
            [
                "sample_id",
                *year_columns
            ]
        ].copy()

        year_panel[
            "_total_ais_rank_value"
        ] = (
            year_panel["sample_id"]
            .map(
                total_ais_by_year[
                    year
                ]
            )
        )

        if year_panel[
            "_total_ais_rank_value"
        ].isna().any():
            raise ValueError(
                f"Missing Total AIS ranking values for {year}."
            )

        # Sort the complete yearly panel using Total AIS for the
        # selected B25 warming level.
        year_panel = (
            year_panel
            .sort_values(
                by=[
                    "_total_ais_rank_value",
                    "sample_id"
                ],
                kind="mergesort"
            )
            .reset_index(drop=True)
        )

        selected_positions = rank_positions[
            :,
            year_position
        ]

        if not np.array_equal(
            np.sort(selected_positions),
            np.arange(
                n_samples,
                dtype=int
            )
        ):
            raise RuntimeError(
                f"Latent ranks for {year} are not a complete "
                "permutation."
            )

        selected_panel = (
            year_panel
            .iloc[selected_positions]
            .reset_index(drop=True)
        )

        metadata_records.append(
            pd.DataFrame({
                "trajectory_id": np.arange(
                    n_samples,
                    dtype=int
                ),
                "projection_year": year,
                "warming_used_for_ranking": warming,
                "latent_gaussian": (
                    latent_gaussian[
                        :,
                        year_position
                    ]
                ),
                "latent_rank": (
                    selected_positions
                    + 1
                ),
                "latent_percentile_rank": (
                    (
                        selected_positions
                        + 0.5
                    )
                    / n_samples
                ),
                "sorted_row_position": (
                    selected_positions
                ),
                "source_sample_id": (
                    selected_panel[
                        "sample_id"
                    ]
                    .to_numpy()
                ),
                "selected_total_ais": (
                    selected_panel[
                        "_total_ais_rank_value"
                    ]
                    .to_numpy()
                )
            })
        )

        selected_panel = selected_panel.drop(
            columns=[
                "sample_id",
                "_total_ais_rank_value"
            ]
        )

        coupled_panels.append(
            selected_panel
        )

    coupled_df = pd.concat(
        coupled_panels,
        axis=1
    )

    original_scientific_order = [
        column
        for column in mc_df.columns
        if column != "sample_id"
    ]

    coupled_df = coupled_df.loc[
        :,
        original_scientific_order
    ]

    coupled_df.insert(
        0,
        "sample_id",
        np.arange(
            n_samples,
            dtype=int
        )
    )

    coupling_metadata = pd.concat(
        metadata_records,
        ignore_index=True
    )

    latent_df = pd.DataFrame(
        latent_gaussian,
        columns=years
    )

    latent_df.index.name = "trajectory_id"

    return (
        coupled_df,
        coupling_metadata,
        latent_df
    )


# ============================================================
# 8. CALCULATE TOTAL AIS FROM WIDE DATA
# ============================================================

def calculate_wide_total_ais(
    df: pd.DataFrame,
    warming: float,
    years: list[int]
) -> pd.DataFrame:
    """
    Calculate Total AIS = EAIS + WAIS from the original wide-format
    B25 variable names.
    """

    warming_code_map = {
        2.0: "2",
        3.5: "35",
        5.0: "5"
    }

    year_code_map = {
        2100: "21",
        2200: "22",
        2300: "23"
    }

    if warming not in warming_code_map:
        raise ValueError(
            f"Unsupported warming level: {warming}"
        )

    warming_code = warming_code_map[
        warming
    ]

    totals = {}

    for year in years:

        year_code = year_code_map[
            year
        ]

        eais_variable = (
            f"E{warming_code}_{year_code}"
        )

        wais_variable = (
            f"W{warming_code}_{year_code}"
        )

        missing_variables = [
            variable
            for variable in [
                eais_variable,
                wais_variable
            ]
            if variable not in df.columns
        ]

        if missing_variables:
            raise KeyError(
                "Required variables are missing:\n"
                + "\n".join(missing_variables)
            )

        totals[year] = (
            df[eais_variable]
            + df[wais_variable]
        )

    return pd.DataFrame(
        totals
    )


# ============================================================
# 9. CORRELATION DIAGNOSTICS
# ============================================================

def create_correlation_diagnostics(
    original_df: pd.DataFrame,
    coupled_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    target_correlation: np.ndarray,
    warming: float,
    years: list[int]
) -> pd.DataFrame:
    """
    Compare target, latent, original, and coupled correlations.
    """

    original_total = calculate_wide_total_ais(
        df=original_df,
        warming=warming,
        years=years
    )

    coupled_total = calculate_wide_total_ais(
        df=coupled_df,
        warming=warming,
        years=years
    )

    matrices = {
        "target_latent_pearson": pd.DataFrame(
            target_correlation,
            index=years,
            columns=years
        ),
        "achieved_latent_pearson": latent_df.corr(
            method="pearson"
        ),
        "achieved_latent_spearman": latent_df.corr(
            method="spearman"
        ),
        "original_total_ais_pearson": original_total.corr(
            method="pearson"
        ),
        "original_total_ais_spearman": original_total.corr(
            method="spearman"
        ),
        "coupled_total_ais_pearson": coupled_total.corr(
            method="pearson"
        ),
        "coupled_total_ais_spearman": coupled_total.corr(
            method="spearman"
        )
    }

    records = []

    for matrix_name, matrix in matrices.items():

        for row_year in years:

            for column_year in years:

                records.append({
                    "matrix": matrix_name,
                    "row_year": row_year,
                    "column_year": column_year,
                    "correlation": matrix.loc[
                        row_year,
                        column_year
                    ]
                })

    return pd.DataFrame(
        records
    )


# ============================================================
# 10. EXACT MARGINAL-PRESERVATION CHECK
# ============================================================

def check_exact_marginal_preservation(
    original_df: pd.DataFrame,
    coupled_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    years: list[int]
) -> pd.DataFrame:
    """
    Verify that every variable retains its original finite-sample
    marginal distribution.
    """

    records = []

    for year in years:

        variables = (
            meta_df
            .loc[
                meta_df["year"] == year,
                "variable"
            ]
            .tolist()
        )

        for variable in variables:

            original_sorted = np.sort(
                original_df[
                    variable
                ].to_numpy()
            )

            coupled_sorted = np.sort(
                coupled_df[
                    variable
                ].to_numpy()
            )

            differences = np.abs(
                coupled_sorted
                - original_sorted
            )

            records.append({
                "projection_year": year,
                "variable": variable,
                "maximum_absolute_difference": (
                    differences.max()
                ),
                "exact_match": np.array_equal(
                    original_sorted,
                    coupled_sorted
                )
            })

    return pd.DataFrame(
        records
    )


# ============================================================
# 11. RUN ONE EXPERIMENT
# ============================================================

def run_temporal_coupling_experiment(
    experiment: dict,
    mc_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    mc_total_ext: pd.DataFrame
) -> dict:
    """
    Run and save one warming-DP21 matrix pairing.
    """

    warming = experiment[
        "warming"
    ]

    warming_label = experiment[
        "warming_label"
    ]

    dp21_scenario = experiment[
        "dp21_scenario"
    ]

    matrix_file = experiment[
        "matrix_file"
    ]

    experiment_name = (
        f"b25_{warming_label}_dp21_{dp21_scenario}"
    )

    experiment_output_dir = os.path.join(
        output_dir,
        experiment_name
    )

    os.makedirs(
        experiment_output_dir,
        exist_ok=True
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Running experiment: {experiment_name}"
    )

    print(
        f"B25 warming used for ranking: {warming}°C"
    )

    print(
        f"DP21 scenario: {dp21_scenario}"
    )

    print(
        f"Correlation matrix file:\n{matrix_file}"
    )


    correlation_matrix = (
        load_temporal_correlation_matrix(
            file_path=matrix_file,
            years=years
        )
    )


    print(
        "\nLoaded temporal correlation matrix:"
    )

    print(
        pd.DataFrame(
            correlation_matrix,
            index=years,
            columns=years
        ).round(4)
    )


    (
        coupled_df,
        coupling_metadata,
        latent_df
    ) = couple_antarctica_year_panels(
        mc_df=mc_df,
        meta_df=meta_df,
        mc_total_ext=mc_total_ext,
        warming=warming,
        years=years,
        correlation_matrix=correlation_matrix,
        random_seed=random_seed
    )


    correlation_diagnostics = (
        create_correlation_diagnostics(
            original_df=mc_df,
            coupled_df=coupled_df,
            latent_df=latent_df,
            target_correlation=correlation_matrix,
            warming=warming,
            years=years
        )
    )


    marginal_diagnostics = (
        check_exact_marginal_preservation(
            original_df=mc_df,
            coupled_df=coupled_df,
            meta_df=meta_df,
            years=years
        )
    )


    coupled_total_ais = calculate_wide_total_ais(
        df=coupled_df,
        warming=warming,
        years=years
    )


    print(
        "\nAchieved latent Gaussian Pearson correlation:"
    )

    print(
        latent_df
        .corr(method="pearson")
        .round(4)
    )


    print(
        "\nAchieved coupled Total AIS Spearman correlation:"
    )

    print(
        coupled_total_ais
        .corr(method="spearman")
        .round(4)
    )


    print(
        "\nAchieved coupled Total AIS Pearson correlation:"
    )

    print(
        coupled_total_ais
        .corr(method="pearson")
        .round(4)
    )


    n_failed_marginals = int(
        (
            ~marginal_diagnostics[
                "exact_match"
            ]
        )
        .sum()
    )


    print(
        "\nMarginal-preservation check:"
    )

    print(
        f"Variables failing exact-match test: "
        f"{n_failed_marginals}"
    )


    coupled_file = os.path.join(
        experiment_output_dir,
        f"{experiment_name}_coupled_samples.csv"
    )

    metadata_file = os.path.join(
        experiment_output_dir,
        f"{experiment_name}_coupling_metadata.csv"
    )

    correlation_file = os.path.join(
        experiment_output_dir,
        f"{experiment_name}_correlation_diagnostics.csv"
    )

    marginal_file = os.path.join(
        experiment_output_dir,
        f"{experiment_name}_marginal_diagnostics.csv"
    )

    target_matrix_file = os.path.join(
        experiment_output_dir,
        f"{experiment_name}_target_matrix.csv"
    )


    coupled_df.to_csv(
        coupled_file,
        index=False
    )

    coupling_metadata.to_csv(
        metadata_file,
        index=False
    )

    correlation_diagnostics.to_csv(
        correlation_file,
        index=False
    )

    marginal_diagnostics.to_csv(
        marginal_file,
        index=False
    )

    pd.DataFrame(
        correlation_matrix,
        index=years,
        columns=years
    ).to_csv(
        target_matrix_file,
        index=True
    )


    print(
        "\nSaved coupled samples to:\n"
        f"{coupled_file}"
    )


    return {
        "experiment_name": experiment_name,
        "warming": warming,
        "dp21_scenario": dp21_scenario,
        "matrix_file": matrix_file,
        "coupled_file": coupled_file,
        "metadata_file": metadata_file,
        "correlation_file": correlation_file,
        "marginal_file": marginal_file,
        "target_matrix_file": target_matrix_file,
        "failed_marginal_checks": n_failed_marginals
    }


# ============================================================
# 12. RUN ALL THREE EXPERIMENTS
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load and prepare the B25 samples once
    # --------------------------------------------------------

    data = load_and_prepare_data(
        file_path=data_file
    )

    mc_df = data[
        "mc_df"
    ]

    meta_df = data[
        "meta_df"
    ]

    mc_total_ext = data[
        "mc_total_ext"
    ]


    # --------------------------------------------------------
    # Run each warming-DP21 pairing
    # --------------------------------------------------------

    experiment_results = []

    for experiment in experiment_settings:

        result = run_temporal_coupling_experiment(
            experiment=experiment,
            mc_df=mc_df,
            meta_df=meta_df,
            mc_total_ext=mc_total_ext
        )

        experiment_results.append(
            result
        )


    # --------------------------------------------------------
    # Save a summary of all experiments
    # --------------------------------------------------------

    experiment_summary = pd.DataFrame(
        experiment_results
    )

    summary_file = os.path.join(
        output_dir,
        "b25_dp21_total_ais_coupling_experiment_summary.csv"
    )

    experiment_summary.to_csv(
        summary_file,
        index=False
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "All temporal-coupling experiments completed."
    )

    print(
        "\nExperiment summary:"
    )

    print(
        experiment_summary[
            [
                "experiment_name",
                "warming",
                "dp21_scenario",
                "failed_marginal_checks"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved experiment summary to:\n"
        f"{summary_file}"
    )