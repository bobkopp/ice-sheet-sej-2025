#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:38:56 2026

============================================================
B25 BASELINE: FAIR TRAJECTORY WEIGHTING

For every FaIR realization:
  1. Read annual GSAT from 2000-2300
  2. Calculate cumulative warming [K yr]
  3. Compare to LOW / INTERMEDIATE / HIGH reference
      cumulative-warming trajectories using Euclidean distance
  4. Convert distance to inverse-distance weights
  5. Draw ONE scenario for the entire ice-sheet trajectory

FaIR files:
  ssp119
  ssp126
  ssp245
  ssp370
  ssp585

============================================================

@author: elizabeth
"""


import os
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. SETTINGS
# ============================================================

FAIR_FILES = {
    "ssp119":
        "/Users/elizabeth/FACTS2/experiments/ssp119/output/"
        "fair-temperature/gsat.nc",

    "ssp126":
        "/Users/elizabeth/FACTS2/experiments/ssp126/output/"
        "fair-temperature/gsat.nc",

    "ssp245":
        "/Users/elizabeth/FACTS2/experiments/ssp245/output/"
        "fair-temperature/gsat.nc",

    "ssp370":
        "/Users/elizabeth/FACTS2/experiments/ssp370/output/"
        "fair-temperature/gsat.nc",

    "ssp585":
        "/Users/elizabeth/FACTS2/experiments/ssp585/output/"
        "fair-temperature/gsat.nc",
}

START_YEAR = 2000
END_YEAR = 2300

YEARS = np.arange(
    START_YEAR,
    END_YEAR + 1
)

SCENARIOS = [
    "LOW",
    "INTERMEDIATE",
    "HIGH"
]

RNG_SEED = 8172026

OUTPUT_DIR = (
    "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025/data"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. B25 REFERENCE WARMING TRAJECTORIES
# ============================================================

REFERENCE_YEARS = np.array([
    2000,
    2020,
    2050,
    2100,
    2200,
    2300
])

REFERENCE_WARMING = {

    "LOW": np.array([
        0.7,
        1.2,
        1.7,
        2.0,
        2.0,
        2.0
    ]),

    "INTERMEDIATE": np.array([
        0.7,
        1.2,
        2.0,
        3.5,
        3.5,
        3.5
    ]),

    "HIGH": np.array([
        0.7,
        1.2,
        2.5,
        5.0,
        5.0,
        5.0
    ])
}


# ============================================================
# 3. HELPER: CUMULATIVE TRAPEZOIDAL INTEGRAL
# ============================================================

def cumulative_trapezoid_annual(warming):
    """
    Convert annual warming [K] into cumulative warming [K yr].

    Assumes annual spacing.

    cumulative[0] = 0 at year 2000.
    """

    warming = np.asarray(
        warming,
        dtype=float
    )

    cumulative = np.zeros(
        len(warming),
        dtype=float
    )

    cumulative[1:] = np.cumsum(
        0.5 * (
            warming[:-1]
            + warming[1:]
        )
    )

    return cumulative


# ============================================================
# 4. BUILD ANNUAL REFERENCE TRAJECTORIES
# ============================================================

reference_rows = []

for scenario in SCENARIOS:

    annual_warming = np.interp(
        YEARS,
        REFERENCE_YEARS,
        REFERENCE_WARMING[scenario]
    )

    cumulative = (
        cumulative_trapezoid_annual(
            annual_warming
        )
    )

    for year, warming, cum in zip(
        YEARS,
        annual_warming,
        cumulative
    ):

        reference_rows.append({
            "scenario": scenario,
            "year": year,
            "warming": warming,
            "cumulative_warming": cum
        })


reference_df = pd.DataFrame(
    reference_rows
)


# ============================================================
# 5. VERIFY REFERENCE VALUES
# ============================================================

reference_check = (
    reference_df[
        reference_df["year"].isin(
            REFERENCE_YEARS
        )
    ]
    .pivot(
        index="year",
        columns="scenario",
        values="cumulative_warming"
    )
)

print("\n========================================")
print("REFERENCE CUMULATIVE WARMING [K yr]")
print("========================================")

print(
    reference_check[
        [
            "LOW",
            "INTERMEDIATE",
            "HIGH"
        ]
    ].round(1)
)


# ============================================================
# 6. BUILD REFERENCE MATRIX
# ============================================================

reference_matrix = (
    reference_df
    .pivot(
        index="scenario",
        columns="year",
        values="cumulative_warming"
    )
    .loc[
        SCENARIOS,
        YEARS
    ]
)

R = reference_matrix.to_numpy(
    dtype=float
)


# ============================================================
# 7. HELPER: INVERSE-DISTANCE WEIGHTS
# ============================================================

def inverse_distance_weights(distances):
    """
    Convert an N x 3 distance matrix into normalized
    inverse-distance weights.

    Exact matches receive 100% weight.
    """

    distances = np.asarray(
        distances,
        dtype=float
    )

    weights = np.zeros_like(
        distances
    )

    exact_rows = np.any(
        np.isclose(
            distances,
            0.0
        ),
        axis=1
    )

    # ----------------------------------------
    # Rows without exact matches
    # ----------------------------------------

    normal = ~exact_rows

    inv = (
        1.0
        / distances[normal]
    )

    weights[normal] = (
        inv
        / inv.sum(
            axis=1,
            keepdims=True
        )
    )

    # ----------------------------------------
    # Exact matches
    # ----------------------------------------

    for i in np.where(
        exact_rows
    )[0]:

        exact = np.isclose(
            distances[i],
            0.0
        )

        weights[i, exact] = (
            1.0
            / exact.sum()
        )

    return weights


# ============================================================
# 8. RANDOM NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(
    RNG_SEED
)


# ============================================================
# 9. PROCESS EACH SSP
# ============================================================

all_results = []

for ssp, filepath in FAIR_FILES.items():

    print("\n")
    print("=" * 70)
    print(f"PROCESSING {ssp}")
    print("=" * 70)

    # --------------------------------------------------------
    # Open NetCDF
    # --------------------------------------------------------

    ds = xr.open_dataset(
        filepath
    )

    # --------------------------------------------------------
    # Temperature variable:
    #
    # dimensions =
    # samples x years x locations
    #
    # locations=-1 is the global mean.
    # --------------------------------------------------------

    da = (
        ds["surface_temperature"]
        .sel(
            years=slice(
                START_YEAR,
                END_YEAR
            )
        )
        .sel(
            locations=-1
        )
    )

    # Result should now be:
    #
    # samples x years

    print(
        "Temperature array shape:",
        da.shape
    )

    print(
        "Years:",
        int(da["years"].min()),
        "-",
        int(da["years"].max())
    )

    print(
        "Samples:",
        da.sizes["samples"]
    )


    # ========================================================
    # 10. CHECK YEAR GRID
    # ========================================================

    fair_years = (
        da["years"]
        .values
        .astype(int)
    )

    if not np.array_equal(
        fair_years,
        YEARS
    ):

        raise ValueError(
            f"{ssp}: FaIR year grid does not exactly "
            f"match {START_YEAR}-{END_YEAR}."
        )


    # ========================================================
    # 11. GET FAIR TEMPERATURE MATRIX
    # ========================================================

    # Shape:
    # n_samples x 301 years

    T = da.values.astype(
        np.float64
    )

    sample_ids = (
        da["samples"]
        .values
        .astype(int)
    )


    # ========================================================
    # 12. CHECK FOR MISSING VALUES
    # ========================================================

    if np.isnan(T).any():

        n_missing = np.isnan(T).sum()

        raise ValueError(
            f"{ssp}: found {n_missing} missing "
            "temperature values."
        )


    # ========================================================
    # 13. CALCULATE CUMULATIVE WARMING
    #
    # Vectorized trapezoidal integration.
    #
    # T shape:
    #     n_samples x 301
    #
    # cumulative shape:
    #     n_samples x 301
    # ========================================================

    cumulative = np.zeros_like(
        T,
        dtype=np.float64
    )

    annual_area = (
        0.5
        * (
            T[:, :-1]
            + T[:, 1:]
        )
    )

    cumulative[:, 1:] = np.cumsum(
        annual_area,
        axis=1
    )


    # ========================================================
    # 14. EUCLIDEAN DISTANCE TO EACH REFERENCE SCENARIO
    #
    # cumulative:
    #     N x 301
    #
    # R:
    #     3 x 301
    #
    # distances:
    #     N x 3
    # ========================================================

    distances = np.sqrt(
        np.sum(
            (
                cumulative[:, None, :]
                - R[None, :, :]
            ) ** 2,
            axis=2
        )
    )


    # ========================================================
    # 15. INVERSE-DISTANCE WEIGHTS
    # ========================================================

    weights = inverse_distance_weights(
        distances
    )


    # ========================================================
    # 16. DRAW ONE SCENARIO PER FAIR REALIZATION
    # ========================================================

    selected_index = np.empty(
        len(sample_ids),
        dtype=int
    )

    for i in range(
        len(sample_ids)
    ):

        selected_index[i] = rng.choice(
            len(SCENARIOS),
            p=weights[i]
        )


    selected_scenario = np.array(
        SCENARIOS,
        dtype=object
    )[selected_index]


    # ========================================================
    # 17. SAVE RESULTS FOR THIS SSP
    # ========================================================

    result = pd.DataFrame({

        "ssp":
            ssp,

        "sample_id":
            sample_ids,

        "trajectory_id":
            [
                f"{ssp}_{sample}"
                for sample in sample_ids
            ],

        "distance_low":
            distances[:, 0],

        "distance_intermediate":
            distances[:, 1],

        "distance_high":
            distances[:, 2],

        "weight_low":
            weights[:, 0],

        "weight_intermediate":
            weights[:, 1],

        "weight_high":
            weights[:, 2],

        "selected_scenario":
            selected_scenario
    })


    # ========================================================
    # 18. ADD USEFUL TEMPERATURE DIAGNOSTICS
    # ========================================================

    year_indices = {
        year:
            np.where(
                YEARS == year
            )[0][0]

        for year in [
            2020,
            2050,
            2100,
            2200,
            2300
        ]
    }


    for year, idx in year_indices.items():

        result[
            f"warming_{year}"
        ] = T[:, idx]

        result[
            f"cumulative_warming_{year}"
        ] = cumulative[:, idx]


    all_results.append(
        result
    )


    # ========================================================
    # 19. SSP-SPECIFIC DIAGNOSTICS
    # ========================================================

    print(
        "\nMean weights:"
    )

    print(
        result[
            [
                "weight_low",
                "weight_intermediate",
                "weight_high"
            ]
        ]
        .mean()
        .round(3)
    )


    print(
        "\nSelected scenario fractions:"
    )

    print(
        result[
            "selected_scenario"
        ]
        .value_counts(
            normalize=True
        )
        .reindex(
            SCENARIOS
        )
        .fillna(0)
        .round(3)
    )


    ds.close()


# ============================================================
# 20. COMBINE ALL SSP RESULTS
# ============================================================

selection_df = pd.concat(
    all_results,
    ignore_index=True
)


print("\n")
print("=" * 70)
print("ALL FAIR TRAJECTORIES")
print("=" * 70)

print(
    "Total trajectories:",
    len(selection_df)
)


# ============================================================
# 21. MEAN WEIGHTS BY SSP
# ============================================================

mean_weights = (
    selection_df
    .groupby("ssp")[
        [
            "weight_low",
            "weight_intermediate",
            "weight_high"
        ]
    ]
    .mean()
)


print("\n========================================")
print("MEAN B25 SCENARIO WEIGHTS BY SSP")
print("========================================")

print(
    mean_weights.round(3)
)


# ============================================================
# 22. DRAWN SCENARIO FRACTIONS BY SSP
# ============================================================

scenario_fractions = pd.crosstab(
    selection_df["ssp"],
    selection_df["selected_scenario"],
    normalize="index"
)

scenario_fractions = (
    scenario_fractions
    .reindex(
        columns=SCENARIOS,
        fill_value=0
    )
)


print("\n========================================")
print("DRAWN B25 SCENARIO FRACTIONS BY SSP")
print("========================================")

print(
    scenario_fractions.round(3)
)


# ============================================================
# 23. MEAN WARMING BY SSP
# ============================================================

warming_summary = (
    selection_df
    .groupby("ssp")[
        [
            "warming_2050",
            "warming_2100",
            "warming_2200",
            "warming_2300"
        ]
    ]
    .mean()
)


print("\n========================================")
print("MEAN FAIR WARMING BY SSP")
print("========================================")

print(
    warming_summary.round(3)
)


# ============================================================
# 24. SAVE OUTPUTS
# ============================================================

selection_file = os.path.join(
    OUTPUT_DIR,
    "fair_b25_baseline_scenario_selection.csv"
)

weight_file = os.path.join(
    OUTPUT_DIR,
    "fair_b25_baseline_mean_weights_by_ssp.csv"
)

fraction_file = os.path.join(
    OUTPUT_DIR,
    "fair_b25_baseline_selection_fractions_by_ssp.csv"
)


selection_df.to_csv(
    selection_file,
    index=False
)

mean_weights.to_csv(
    weight_file
)

scenario_fractions.to_csv(
    fraction_file
)


print("\n========================================")
print("OUTPUT FILES")
print("========================================")

print(selection_file)
print(weight_file)
print(fraction_file)

# ============================================================
# DIAGNOSTIC 1: NEAREST REFERENCE SCENARIO
# ============================================================

distance_cols = [
    "distance_low",
    "distance_intermediate",
    "distance_high"
]

nearest_map = {
    "distance_low": "LOW",
    "distance_intermediate": "INTERMEDIATE",
    "distance_high": "HIGH"
}

selection_df["nearest_scenario"] = (
    selection_df[distance_cols]
    .idxmin(axis=1)
    .map(nearest_map)
)

nearest_fractions = pd.crosstab(
    selection_df["ssp"],
    selection_df["nearest_scenario"],
    normalize="index"
).reindex(
    columns=SCENARIOS,
    fill_value=0
)

print("\n========================================")
print("NEAREST REFERENCE SCENARIO BY SSP")
print("========================================")

print(nearest_fractions.round(3))

# ============================================================
# DIAGNOSTIC 2: MEAN DISTANCES BY SSP
# ============================================================

mean_distances = (
    selection_df
    .groupby("ssp")[
        [
            "distance_low",
            "distance_intermediate",
            "distance_high"
        ]
    ]
    .mean()
)

print("\n========================================")
print("MEAN EUCLIDEAN DISTANCE BY SSP")
print("========================================")

print(mean_distances.round(1))

# ============================================================
# DIAGNOSTIC 3: MEAN CUMULATIVE WARMING BY SSP
# ============================================================

cum_cols = [
    "cumulative_warming_2020",
    "cumulative_warming_2050",
    "cumulative_warming_2100",
    "cumulative_warming_2200",
    "cumulative_warming_2300"
]

mean_cumulative = (
    selection_df
    .groupby("ssp")[cum_cols]
    .mean()
)

print("\n========================================")
print("MEAN CUMULATIVE WARMING [K yr] BY SSP")
print("========================================")

print(mean_cumulative.round(1))

# ============================================================
# DIAGNOSTIC 4: MEAN WARMING BY SSP
# ============================================================

warming_cols = [
    "warming_2020",
    "warming_2050",
    "warming_2100",
    "warming_2200",
    "warming_2300"
]

mean_warming = (
    selection_df
    .groupby("ssp")[warming_cols]
    .mean()
)

print("\n========================================")
print("MEAN GSAT [K] BY SSP")
print("========================================")

print(mean_warming.round(2))