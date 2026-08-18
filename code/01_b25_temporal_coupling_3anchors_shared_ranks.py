#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_b25_b19style_temporal_coupling.py

Approximate the Bamber et al. (2019) temporal-coupling workflow using
B25 SEJ samples and DeConto et al. (2021; DP21) temporal dependence.

WHAT THIS DOES
--------------
1. Loads the B25 Monte Carlo samples using the existing B25data_processing.py.
2. Reads previously derived DP21 Total AIS temporal-correlation matrices.
3. Creates THREE temporally coupled B25 libraries:
       LOW      = B25 2.0 C ranked by Total AIS + DP21 RCP2.6 dependence
       MODERATE = B25 3.5 C ranked by Total AIS + DP21 RCP4.5 dependence
       HIGH     = B25 5.0 C ranked by Total AIS + DP21 RCP8.5 dependence
4. At each projection year (2100, 2200, 2300), sorts the COMPLETE B25
   year-panel by Total AIS.
5. Generates correlated Gaussian ranks and uses those ranks to reorder
   complete yearly panels.
6. Saves the temporally coupled LOW/MODERATE/HIGH libraries and diagnostics.

WHY WHOLE PANELS MOVE
---------------------
All B25 variables belonging to one projection year move together.
Therefore the original within-year multivariate relationships among
EAIS, WAIS, GrIS, processes, and warming variables are preserved.
Only the cross-year pairing of rows changes.

B19 APPROXIMATION
-----------------
B19 used a temporal correlation matrix derived from an external
process-model ensemble. Here that role is played by DP21.

Unlike B19, which used one externally derived temporal structure,
we use scenario-specific DP21 matrices:
    B25 2.0 C -> DP21 RCP2.6
    B25 3.5 C -> DP21 RCP4.5
    B25 5.0 C -> DP21 RCP8.5

The 3.5 C B25 elicitation is included as a new MODERATE anchor. This is
an intentional extension of the original B19 LOW/HIGH framework.

REQUIRES
--------
B25data_processing.py

Expected correlation-matrix files:
    data/temporal_correlation_matrices/
        dp21_total_ais_rcp26_temporal_correlation.csv
        dp21_total_ais_rcp45_temporal_correlation.csv
        dp21_total_ais_rcp85_temporal_correlation.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from B25data_processing import (
    load_and_prepare_data,
    project_dir,
    data_file,
)


# ============================================================
# 1. SETTINGS
# ============================================================

YEARS = [2100, 2200, 2300]
RANDOM_SEED = 1234

CORRELATION_DIR = os.path.join(
    project_dir,
    "data",
    "temporal_correlation_matrices",
)

OUTPUT_DIR = os.path.join(
    project_dir,
    "output",
    "b19style_baseline",
    "01_temporal_coupling",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    {
        "role": "LOW",
        "warming": 2.0,
        "warming_label": "2.0C",
        "dp21_scenario": "rcp26",
        "matrix_file": os.path.join(
            CORRELATION_DIR,
            "dp21_total_ais_rcp26_temporal_correlation.csv",
        ),
    },
    {
        "role": "MODERATE",
        "warming": 3.5,
        "warming_label": "3.5C",
        "dp21_scenario": "rcp45",
        "matrix_file": os.path.join(
            CORRELATION_DIR,
            "dp21_total_ais_rcp45_temporal_correlation.csv",
        ),
    },
    {
        "role": "HIGH",
        "warming": 5.0,
        "warming_label": "5.0C",
        "dp21_scenario": "rcp85",
        "matrix_file": os.path.join(
            CORRELATION_DIR,
            "dp21_total_ais_rcp85_temporal_correlation.csv",
        ),
    },
]


# ============================================================
# 2. LOAD / VALIDATE TEMPORAL CORRELATION MATRIX
# ============================================================

def validate_correlation_matrix(
    correlation_matrix: np.ndarray,
    years: list[int],
    tolerance: float = 1e-10,
) -> np.ndarray:

    corr = np.asarray(correlation_matrix, dtype=float)
    expected_shape = (len(years), len(years))

    if corr.shape != expected_shape:
        raise ValueError(
            f"Correlation matrix must have shape {expected_shape}, "
            f"but has shape {corr.shape}."
        )

    if not np.all(np.isfinite(corr)):
        raise ValueError("Correlation matrix contains non-finite values.")

    if not np.allclose(corr, corr.T, atol=tolerance):
        raise ValueError("Correlation matrix must be symmetric.")

    if not np.allclose(np.diag(corr), 1.0, atol=tolerance):
        raise ValueError("Correlation matrix diagonal must equal 1.")

    if np.any(corr < -1.0 - tolerance) or np.any(corr > 1.0 + tolerance):
        raise ValueError("Correlation entries must lie between -1 and 1.")

    eigvals = np.linalg.eigvalsh(corr)

    if eigvals.min() < -tolerance:
        raise ValueError(
            "Correlation matrix is not positive semidefinite. "
            f"Minimum eigenvalue = {eigvals.min():.6g}"
        )

    return corr


def load_temporal_correlation_matrix(
    file_path: str,
    years: list[int],
) -> np.ndarray:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Temporal correlation matrix was not found:\n"
            f"{file_path}"
        )

    matrix_df = pd.read_csv(file_path)
    year_strings = [str(year) for year in years]

    # Preferred format:
    # ,2100,2200,2300
    # 2100,1,...,...
    # ...
    if all(year in matrix_df.columns for year in year_strings):
        numeric = matrix_df.loc[:, year_strings]

    else:
        numeric = matrix_df.copy()

        unnamed = [
            col for col in numeric.columns
            if str(col).startswith("Unnamed:")
        ]
        if unnamed:
            numeric = numeric.drop(columns=unnamed)

        numeric = numeric.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(axis=1, how="all")

        # Handle explicit first column containing row-year labels.
        if numeric.shape == (len(years), len(years) + 1):
            first = numeric.iloc[:, 0].to_numpy(dtype=float)
            if np.allclose(first, np.asarray(years, dtype=float)):
                numeric = numeric.iloc[:, 1:]

    corr = numeric.to_numpy(dtype=float)

    return validate_correlation_matrix(
        correlation_matrix=corr,
        years=years,
    )


# ============================================================
# 3. IDENTIFY COMPLETE YEAR PANELS
# ============================================================

def get_columns_by_year(
    meta_df: pd.DataFrame,
    years: list[int],
) -> dict[int, list[str]]:

    output = {}

    for year in years:
        cols = (
            meta_df.loc[
                meta_df["year"] == year,
                "variable",
            ]
            .tolist()
        )

        if not cols:
            raise ValueError(
                f"No B25 scientific variables were found for {year}."
            )

        output[year] = cols

    return output


# ============================================================
# 4. EXTRACT TOTAL AIS VALUES USED TO RANK EACH YEAR
# ============================================================

def get_total_ais_by_year(
    mc_total_ext: pd.DataFrame,
    warming: float,
    years: list[int],
) -> dict[int, pd.Series]:

    subset = mc_total_ext.loc[
        (mc_total_ext["ice_sheet"] == "Total AIS")
        & np.isclose(
            mc_total_ext["warming"].astype(float),
            warming,
        )
        & mc_total_ext["year"].isin(years)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No Total AIS values found for warming={warming} C."
        )

    duplicated = subset.duplicated(
        subset=["sample_id", "year"],
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            "More than one Total AIS value was found for a "
            "sample_id-year combination."
        )

    out = {}

    for year in years:
        s = (
            subset.loc[
                subset["year"] == year,
                ["sample_id", "value"],
            ]
            .set_index("sample_id")["value"]
            .sort_index()
        )

        if s.empty:
            raise ValueError(f"No Total AIS values found for {year}.")

        out[year] = s

    counts = {year: len(s) for year, s in out.items()}

    if len(set(counts.values())) != 1:
        raise ValueError(
            "Total AIS sample counts differ among years:\n"
            f"{counts}"
        )

    return out


# ============================================================
# 5. GENERATE CORRELATED GAUSSIAN RANK PROCESS
# ============================================================

def generate_correlated_rank_process(
    base_normals: np.ndarray,
    correlation_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform a SHARED iid standard-normal innovation matrix into a
    scenario-specific correlated Gaussian rank process.

    Using the same base_normals for LOW, MODERATE, and HIGH gives the
    three temporal rank processes a common stochastic basis without
    assuming that B25 Monte Carlo row IDs correspond across warming
    scenarios.

    The percentile is still free to change from 2100 to 2200 to 2300
    according to the temporal correlation matrix.
    """

    corr = np.asarray(correlation_matrix, dtype=float)

    # Cholesky is preferred. Fall back to an eigen decomposition for a
    # positive-semidefinite matrix that is numerically singular.
    try:
        transform = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.clip(eigvals, 0.0, None)
        transform = eigvecs @ np.diag(np.sqrt(eigvals))

    latent = base_normals @ transform.T

    ranks = np.empty(latent.shape, dtype=int)

    for j in range(latent.shape[1]):
        ranks[:, j] = (
            rankdata(
                latent[:, j],
                method="ordinal",
            ).astype(int)
            - 1
        )

    return latent, ranks


# ============================================================
# 6. TEMPORALLY COUPLE COMPLETE B25 YEAR PANELS
# ============================================================

def couple_year_panels(
    mc_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    mc_total_ext: pd.DataFrame,
    warming: float,
    years: list[int],
    correlation_matrix: np.ndarray,
    base_normals: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    corr = validate_correlation_matrix(
        correlation_matrix,
        years,
    )

    columns_by_year = get_columns_by_year(
        meta_df,
        years,
    )

    total_ais_by_year = get_total_ais_by_year(
        mc_total_ext,
        warming,
        years,
    )

    n_samples = len(mc_df)
    expected_ids = np.arange(n_samples, dtype=int)

    if not np.array_equal(
        np.sort(mc_df["sample_id"].to_numpy(dtype=int)),
        expected_ids,
    ):
        raise ValueError(
            "mc_df sample_id must be consecutive integers 0,...,N-1."
        )

    if base_normals.shape != (n_samples, len(years)):
        raise ValueError(
            "base_normals must have shape "
            f"({n_samples}, {len(years)}), got {base_normals.shape}."
        )

    latent, rank_positions = generate_correlated_rank_process(
        base_normals=base_normals,
        correlation_matrix=corr,
    )

    coupled_panels = []
    metadata_records = []

    for year_position, year in enumerate(years):

        year_columns = columns_by_year[year]

        year_panel = mc_df.loc[
            :,
            ["sample_id", *year_columns],
        ].copy()

        year_panel["_total_ais_rank_value"] = (
            year_panel["sample_id"]
            .map(total_ais_by_year[year])
        )

        if year_panel["_total_ais_rank_value"].isna().any():
            raise ValueError(
                f"Missing Total AIS ranking values for {year}."
            )

        # Sort COMPLETE yearly panel by Total AIS.
        year_panel = (
            year_panel
            .sort_values(
                by=["_total_ais_rank_value", "sample_id"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        positions = rank_positions[:, year_position]

        if not np.array_equal(
            np.sort(positions),
            np.arange(n_samples, dtype=int),
        ):
            raise RuntimeError(
                f"Rank positions for {year} are not a full permutation."
            )

        selected = (
            year_panel
            .iloc[positions]
            .reset_index(drop=True)
        )

        metadata_records.append(
            pd.DataFrame(
                {
                    "trajectory_id": np.arange(n_samples, dtype=int),
                    "projection_year": year,
                    "warming_used_for_ranking": warming,
                    "latent_gaussian": latent[:, year_position],
                    "latent_rank": positions + 1,
                    "latent_percentile_rank": (
                        (positions + 0.5) / n_samples
                    ),
                    "source_sample_id": (
                        selected["sample_id"].to_numpy(dtype=int)
                    ),
                    "selected_total_ais": (
                        selected["_total_ais_rank_value"].to_numpy(dtype=float)
                    ),
                }
            )
        )

        selected = selected.drop(
            columns=["sample_id", "_total_ais_rank_value"]
        )

        coupled_panels.append(selected)

    coupled_df = pd.concat(coupled_panels, axis=1)

    scientific_order = [
        col for col in mc_df.columns
        if col != "sample_id"
    ]

    coupled_df = coupled_df.loc[:, scientific_order]

    coupled_df.insert(
        0,
        "trajectory_id",
        np.arange(n_samples, dtype=int),
    )

    metadata_df = pd.concat(
        metadata_records,
        ignore_index=True,
    )

    latent_df = pd.DataFrame(
        latent,
        columns=years,
    )
    latent_df.index.name = "trajectory_id"

    return coupled_df, metadata_df, latent_df


# ============================================================
# 7. CALCULATE TOTAL AIS FROM WIDE B25 DATA
# ============================================================

def calculate_wide_total_ais(
    df: pd.DataFrame,
    warming: float,
    years: list[int],
) -> pd.DataFrame:

    warming_code = {
        2.0: "2",
        3.5: "35",
        5.0: "5",
    }[warming]

    year_code = {
        2100: "21",
        2200: "22",
        2300: "23",
    }

    totals = {}

    for year in years:
        e = f"E{warming_code}_{year_code[year]}"
        w = f"W{warming_code}_{year_code[year]}"

        missing = [
            x for x in [e, w]
            if x not in df.columns
        ]

        if missing:
            raise KeyError(
                "Required B25 variables missing:\n"
                + "\n".join(missing)
            )

        totals[year] = df[e] + df[w]

    return pd.DataFrame(totals)


# ============================================================
# 8. EXACT MARGINAL-PRESERVATION CHECK
# ============================================================

def check_exact_marginal_preservation(
    original_df: pd.DataFrame,
    coupled_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    years: list[int],
) -> pd.DataFrame:

    records = []

    for year in years:

        variables = (
            meta_df.loc[
                meta_df["year"] == year,
                "variable",
            ]
            .tolist()
        )

        for variable in variables:

            original_sorted = np.sort(
                original_df[variable].to_numpy()
            )

            coupled_sorted = np.sort(
                coupled_df[variable].to_numpy()
            )

            difference = np.abs(
                coupled_sorted - original_sorted
            )

            records.append(
                {
                    "projection_year": year,
                    "variable": variable,
                    "maximum_absolute_difference": difference.max(),
                    "exact_match": np.array_equal(
                        original_sorted,
                        coupled_sorted,
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# 9. RUN ONE LOW/HIGH TEMPORAL-COUPLING EXPERIMENT
# ============================================================

def run_experiment(
    experiment: dict,
    mc_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    mc_total_ext: pd.DataFrame,
) -> dict:

    role = experiment["role"]
    warming = experiment["warming"]
    warming_label = experiment["warming_label"]
    dp21_scenario = experiment["dp21_scenario"]
    matrix_file = experiment["matrix_file"]

    print("\n" + "=" * 72)
    print(
        f"{role}: B25 {warming_label} + DP21 {dp21_scenario}"
    )
    print("=" * 72)

    corr = load_temporal_correlation_matrix(
        matrix_file,
        YEARS,
    )

    print("\nTarget DP21 temporal correlation:")
    print(
        pd.DataFrame(
            corr,
            index=YEARS,
            columns=YEARS,
        ).round(4)
    )

    coupled_df, metadata_df, latent_df = couple_year_panels(
        mc_df=mc_df,
        meta_df=meta_df,
        mc_total_ext=mc_total_ext,
        warming=warming,
        years=YEARS,
        correlation_matrix=corr,
        base_normals=base_normals,
    )

    total_ais = calculate_wide_total_ais(
        coupled_df,
        warming,
        YEARS,
    )

    print("\nAchieved latent Pearson correlation:")
    print(
        latent_df.corr(method="pearson").round(4)
    )

    print("\nAchieved coupled Total AIS Spearman correlation:")
    print(
        total_ais.corr(method="spearman").round(4)
    )

    print("\nAchieved coupled Total AIS Pearson correlation:")
    print(
        total_ais.corr(method="pearson").round(4)
    )

    marginal_df = check_exact_marginal_preservation(
        original_df=mc_df,
        coupled_df=coupled_df,
        meta_df=meta_df,
        years=YEARS,
    )

    n_failed = int(
        (~marginal_df["exact_match"]).sum()
    )

    print(
        "\nVariables failing exact marginal-preservation check:",
        n_failed,
    )

    prefix = f"b25_b19style_{role.lower()}_{warming_label}_dp21_{dp21_scenario}"

    coupled_file = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_coupled_samples.csv",
    )

    metadata_file = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_coupling_metadata.csv",
    )

    target_file = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_target_temporal_correlation.csv",
    )

    marginal_file = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_marginal_diagnostics.csv",
    )

    coupled_df.to_csv(
        coupled_file,
        index=False,
    )

    metadata_df.to_csv(
        metadata_file,
        index=False,
    )

    pd.DataFrame(
        corr,
        index=YEARS,
        columns=YEARS,
    ).to_csv(
        target_file,
        index=True,
    )

    marginal_df.to_csv(
        marginal_file,
        index=False,
    )

    return {
        "role": role,
        "warming": warming,
        "dp21_scenario": dp21_scenario,
        "coupled_file": coupled_file,
        "metadata_file": metadata_file,
        "target_file": target_file,
        "marginal_file": marginal_file,
        "failed_marginal_checks": n_failed,
    }


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("=" * 72)
    print("B25 / DP21: B19-STYLE TEMPORAL COUPLING")
    print("=" * 72)

    data = load_and_prepare_data(
        file_path=data_file
    )

    mc_df = data["mc_df"].copy()
    meta_df = data["meta_df"].copy()
    mc_total_ext = data["mc_total_ext"].copy()

    print(f"\nB25 rows: {len(mc_df):,}")
    print(f"B25 source: {data_file}")

    # One common stochastic basis for all three warming libraries.
    # This makes trajectory_id meaningful across LOW/MODERATE/HIGH
    # latent rank processes while still allowing each DP21 matrix to
    # impose its own cross-year dependence.
    rng = np.random.default_rng(RANDOM_SEED)
    base_normals = rng.standard_normal(
        size=(len(mc_df), len(YEARS))
    )

    shared_base_file = os.path.join(
        OUTPUT_DIR,
        "b25_b19style_shared_gaussian_innovations.csv",
    )

    pd.DataFrame(
        base_normals,
        columns=YEARS,
    ).rename_axis("trajectory_id").to_csv(
        shared_base_file
    )

    results = []

    for experiment in EXPERIMENTS:
        results.append(
            run_experiment(
                experiment=experiment,
                mc_df=mc_df,
                meta_df=meta_df,
                mc_total_ext=mc_total_ext,
            )
        )

    summary = pd.DataFrame(results)

    summary_file = os.path.join(
        OUTPUT_DIR,
        "b25_b19style_temporal_coupling_summary.csv",
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print("\n" + "=" * 72)
    print("TEMPORAL COUPLING COMPLETE")
    print("=" * 72)

    print(
        summary[
            [
                "role",
                "warming",
                "dp21_scenario",
                "failed_marginal_checks",
                "coupled_file",
            ]
        ].to_string(index=False)
    )

    print("\nSummary saved to:")
    print(summary_file)


if __name__ == "__main__":
    main()
