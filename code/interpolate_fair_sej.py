#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:26:35 2026

@author: elizabeth
"""

"""
Interpolate the 2025 structured expert judgment (SEJ)
ice-sheet Monte Carlo samples onto FaIR temperature trajectories.

The workflow:

    1. Reads and pools FaIR GSAT samples from five SSP experiments.
    2. Calculates, for every FaIR sample:
           a. GSAT at 2100, 2200, and 2300
           b. cumulative warming from 2000 through each year
    3. Represents the SEJ Low, Intermediate, and High pathways
       as reference points in GSAT–cumulative-warming space.
    4. Calculates Gaussian climate-space weights for each FaIR
       trajectory and projection year.
    5. Pairs each FaIR trajectory with one complete matched SEJ
       Monte Carlo row.
    6. Applies the climate weights draw-by-draw to the 2.0°C,
       3.5°C, and 5.0°C SEJ values.
    7. Calculates:
           a. Total AIS = EAIS + WAIS
           b. Runoff minus accumulation = runoff + accumulation
              because accumulation is represented as a signed
              negative sea-level contribution.
    8. Saves interpolated responses, weights, diagnostics,
       and the FaIR–SEJ pairing.

Created July 2026.

@author: elizabeth
"""

# ============================================================
# 0. IMPORTS
# ============================================================

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. SETTINGS
# ============================================================

# ----------------------------
# SEJ project and data
# ----------------------------

sej_project_dir = Path(
    "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025"
)

sej_data_file = (
    sej_project_dir
    / "data"
    / "sej2025_split_panel_liz.csv"
)

# Allows import of B25data_processing.py if it is located
# directly inside sej_project_dir.
sys.path.insert(
    0,
    str(sej_project_dir)
)

import B25data_processing as b25


# ----------------------------
# FaIR files
# ----------------------------

facts_project_dir = Path(
    "/Users/elizabeth/FACTS2"
)

fair_files = {
    "ssp119": (
        facts_project_dir
        / "experiments"
        / "ssp119"
        / "output"
        / "fair-temperature"
        / "gsat.nc"
    ),
    "ssp126": (
        facts_project_dir
        / "experiments"
        / "ssp126"
        / "output"
        / "fair-temperature"
        / "gsat.nc"
    ),
    "ssp245": (
        facts_project_dir
        / "experiments"
        / "ssp245"
        / "output"
        / "fair-temperature"
        / "gsat.nc"
    ),
    "ssp370": (
        facts_project_dir
        / "experiments"
        / "ssp370"
        / "output"
        / "fair-temperature"
        / "gsat.nc"
    ),
    "ssp585": (
        facts_project_dir
        / "experiments"
        / "ssp585"
        / "output"
        / "fair-temperature"
        / "gsat.nc"
    ),
}

scenario_labels = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}


# ----------------------------
# Output directory
# ----------------------------

output_dir = (
    sej_project_dir
    / "output"
    / "fair_sej_interpolation"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ----------------------------
# Interpolation settings
# ----------------------------

projection_years = [
    2100,
    2200,
    2300,
]

cumulative_warming_start_year = 2000

# Fixed physical scales used to make GSAT and cumulative warming
# comparable in the distance calculation.
#
# A difference of 1°C contributes one scaled distance unit.
# A difference of 200 K·yr contributes one scaled distance unit.
feature_scales = {
    "gsat": 1.0,
    "cumulative_warming": 200.0,
}

# Gaussian-kernel bandwidth in scaled climate space.
#
# Smaller bandwidth:
#     weights concentrate more strongly on the nearest pathway.
#
# Larger bandwidth:
#     weights are spread more evenly across pathways.
kernel_bandwidth = 1.0

# Seed controlling the pairing between FaIR samples and matched 
# SEJ Monte Carlo rows.
pairing_seed = 20269723 # date of the analysis

# Distance beyond which a FaIR state is flagged as far outside
# the SEJ reference domain.
outside_domain_distance = 2.0


# ============================================================
# 2. SEJ REFERENCE CLIMATE PATHWAYS
# ============================================================

# These values come directly from the supplied pathway table.
#
# Only Low, Intermediate, and High are SEJ response anchors:
#
#     Low          → 2.0°C SEJ samples
#     Intermediate → 3.5°C SEJ samples
#     High         → 5.0°C SEJ samples
#
# Overshoot and Commitment do not have separate SEJ response
# distributions and therefore are not interpolation anchors.

reference_climate = pd.DataFrame(
    [
        {
            "year": 2100,
            "pathway": "Low",
            "warming": 2.0,
            "warming_code": "2",
            "gsat": 2.0,
            "cumulative_warming": 155.0,
        },
        {
            "year": 2100,
            "pathway": "Intermediate",
            "warming": 3.5,
            "warming_code": "35",
            "gsat": 3.5,
            "cumulative_warming": 204.5,
        },
        {
            "year": 2100,
            "pathway": "High",
            "warming": 5.0,
            "warming_code": "5",
            "gsat": 5.0,
            "cumulative_warming": 262.0,
        },
        {
            "year": 2200,
            "pathway": "Low",
            "warming": 2.0,
            "warming_code": "2",
            "gsat": 2.0,
            "cumulative_warming": 355.0,
        },
        {
            "year": 2200,
            "pathway": "Intermediate",
            "warming": 3.5,
            "warming_code": "35",
            "gsat": 3.5,
            "cumulative_warming": 554.5,
        },
        {
            "year": 2200,
            "pathway": "High",
            "warming": 5.0,
            "warming_code": "5",
            "gsat": 5.0,
            "cumulative_warming": 762.0,
        },
        {
            "year": 2300,
            "pathway": "Low",
            "warming": 2.0,
            "warming_code": "2",
            "gsat": 2.0,
            "cumulative_warming": 555.0,
        },
        {
            "year": 2300,
            "pathway": "Intermediate",
            "warming": 3.5,
            "warming_code": "35",
            "gsat": 3.5,
            "cumulative_warming": 904.5,
        },
        {
            "year": 2300,
            "pathway": "High",
            "warming": 5.0,
            "warming_code": "5",
            "gsat": 5.0,
            "cumulative_warming": 1262.0,
        },
    ]
)

pathway_order = [
    "Low",
    "Intermediate",
    "High",
]

warming_code_order = [
    "2",
    "35",
    "5",
]

projection_year_code_map = {
    2100: "21",
    2200: "22",
    2300: "23",
}


# ============================================================
# 3. LOAD AND POOL FAIR TEMPERATURE TRAJECTORIES
# ============================================================

def load_fair_ensemble(
    files,
    start_year=2000,
    end_year=2300,
):
    """
    Read and pool FaIR GSAT samples from multiple scenarios.

    Returns
    -------
    pooled : xarray.DataArray
        Dimensions:
            samples
            years

        Coordinates:
            scenario
            scenario_label
            original_sample
    """

    ensembles = []

    for scenario, path in files.items():

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"FaIR file not found:\n{path}"
            )

        print(
            f"Reading FaIR data for {scenario}:"
            f"\n{path}\n"
        )

        with xr.open_dataset(path) as ds:

            if "surface_temperature" not in ds:
                raise KeyError(
                    "'surface_temperature' was not found in "
                    f"{path}"
                )

            temperature = ds[
                "surface_temperature"
            ]

            if "locations" in temperature.dims:
                temperature = temperature.squeeze(
                    "locations",
                    drop=True,
                )

            temperature = temperature.sel(
                years=slice(
                    start_year,
                    end_year,
                )
            ).load()

        required_dims = {
            "samples",
            "years",
        }

        if not required_dims.issubset(
            temperature.dims
        ):
            raise ValueError(
                f"{scenario} temperature data must contain "
                "'samples' and 'years' dimensions. "
                f"Found: {temperature.dims}"
            )

        n_samples = temperature.sizes[
            "samples"
        ]

        temperature = temperature.assign_coords(
            scenario=(
                "samples",
                np.repeat(
                    scenario,
                    n_samples,
                ),
            ),
            scenario_label=(
                "samples",
                np.repeat(
                    scenario_labels[scenario],
                    n_samples,
                ),
            ),
            original_sample=(
                "samples",
                temperature[
                    "samples"
                ].values.copy(),
            ),
        )

        ensembles.append(
            temperature
        )

    pooled = xr.concat(
        ensembles,
        dim="samples",
    )

    pooled = pooled.assign_coords(
        samples=np.arange(
            pooled.sizes["samples"],
            dtype=int,
        )
    )

    pooled.name = "surface_temperature"

    print(
        "Pooled FaIR ensemble:"
    )
    print(pooled)
    print()

    return pooled


# ============================================================
# 4. CALCULATE FAIR CLIMATE FEATURES
# ============================================================

def calculate_fair_climate_features(
    pooled,
    years=projection_years,
    start_year=2000,
):
    """
    Calculate GSAT and cumulative warming for every FaIR sample.

    Cumulative warming is the time integral of the GSAT anomaly
    beginning in 2000.

    This matches the supplied pathway table. For example, the
    Low pathway from 2000 to 2020 gives:

        ((0.7 + 1.2) / 2) * 20 = 19 K·yr

    Parameters
    ----------
    pooled : xarray.DataArray
        FaIR GSAT with dimensions samples and years.

    years : sequence of int
        Projection years.

    start_year : int
        Beginning of cumulative integration.

    Returns
    -------
    feature_df : pandas.DataFrame
        One row for each FaIR sample and projection year.
    """

    available_years = set(
        pooled["years"].values.tolist()
    )

    required_years = {
        start_year,
        *years,
    }

    missing_years = sorted(
        required_years - available_years
    )

    if missing_years:
        raise ValueError(
            "The following required years are missing from "
            f"the FaIR data: {missing_years}"
        )

    feature_frames = []

    for projection_year in years:

        subset = pooled.sel(
            years=slice(
                start_year,
                projection_year,
            )
        ).transpose(
            "samples",
            "years",
        )

        year_values = subset[
            "years"
        ].values.astype(float)

        temperature_values = subset.values

        if hasattr(np, "trapezoid"):
            cumulative_warming = np.trapezoid(
                temperature_values,
                x=year_values,
                axis=1,
            )
        else:
            cumulative_warming = np.trapz(
                temperature_values,
                x=year_values,
                axis=1,
            )

        gsat = pooled.sel(
            years=projection_year
        ).values

        frame = pd.DataFrame(
            {
                "fair_sample_id": pooled[
                    "samples"
                ].values.astype(int),
                "scenario": pooled[
                    "scenario"
                ].values,
                "scenario_label": pooled[
                    "scenario_label"
                ].values,
                "original_sample": pooled[
                    "original_sample"
                ].values,
                "year": projection_year,
                "gsat": gsat,
                "cumulative_warming": (
                    cumulative_warming
                ),
            }
        )

        feature_frames.append(
            frame
        )

    feature_df = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    if not np.isfinite(
        feature_df[
            [
                "gsat",
                "cumulative_warming",
            ]
        ].to_numpy()
    ).all():
        raise ValueError(
            "Non-finite climate features were calculated."
        )

    return feature_df


# ============================================================
# 5. CALCULATE CLIMATE-SPACE WEIGHTS
# ============================================================

def calculate_climate_weights(
    feature_df,
    reference_df=reference_climate,
    scales=feature_scales,
    bandwidth=kernel_bandwidth,
    exact_match_tolerance=1e-12,
    outside_threshold=outside_domain_distance,
):
    """
    Calculate Gaussian weights for Low, Intermediate, and High.

    Distance is calculated in scaled two-dimensional space:

        GSAT
        cumulative warming

    An exact match to a reference pathway receives a one-hot
    weight. This ensures that the reference pathways reproduce
    their corresponding SEJ samples exactly.

    Returns
    -------
    weights_df : pandas.DataFrame
        Wide table with Low, Intermediate, and High weights.

    diagnostics_df : pandas.DataFrame
        Distances and interpolation diagnostics.
    """

    if bandwidth <= 0:
        raise ValueError(
            "bandwidth must be greater than zero"
        )

    gsat_scale = float(
        scales["gsat"]
    )

    cumulative_scale = float(
        scales["cumulative_warming"]
    )

    if (
        gsat_scale <= 0
        or cumulative_scale <= 0
    ):
        raise ValueError(
            "Feature scales must be greater than zero."
        )

    weight_frames = []
    diagnostic_frames = []

    for projection_year in projection_years:

        fair_year = (
            feature_df.loc[
                feature_df["year"]
                == projection_year
            ]
            .copy()
            .reset_index(drop=True)
        )

        reference_year = (
            reference_df.loc[
                reference_df["year"]
                == projection_year
            ]
            .set_index("pathway")
            .loc[pathway_order]
            .reset_index()
        )

        if len(reference_year) != 3:
            raise ValueError(
                "Exactly three SEJ reference anchors are "
                f"required for {projection_year}."
            )

        fair_matrix = fair_year[
            [
                "gsat",
                "cumulative_warming",
            ]
        ].to_numpy(dtype=float)

        reference_matrix = reference_year[
            [
                "gsat",
                "cumulative_warming",
            ]
        ].to_numpy(dtype=float)

        scale_vector = np.array(
            [
                gsat_scale,
                cumulative_scale,
            ],
            dtype=float,
        )

        differences = (
            fair_matrix[:, np.newaxis, :]
            - reference_matrix[
                np.newaxis,
                :,
                :,
            ]
        )

        scaled_differences = (
            differences
            / scale_vector[
                np.newaxis,
                np.newaxis,
                :,
            ]
        )

        distances = np.sqrt(
            np.sum(
                scaled_differences ** 2,
                axis=2,
            )
        )

        # Gaussian log weights
        log_weights = (
            -0.5
            * (
                distances
                / bandwidth
            ) ** 2
        )

        # Numerically stable exponentiation
        log_weights = (
            log_weights
            - np.max(
                log_weights,
                axis=1,
                keepdims=True,
            )
        )

        raw_weights = np.exp(
            log_weights
        )

        weights = (
            raw_weights
            / raw_weights.sum(
                axis=1,
                keepdims=True,
            )
        )

        # Exact reference matches become one-hot weights.
        exact_matches = (
            distances
            <= exact_match_tolerance
        )

        rows_with_exact_match = np.any(
            exact_matches,
            axis=1,
        )

        if np.any(
            rows_with_exact_match
        ):
            weights[
                rows_with_exact_match,
                :,
            ] = exact_matches[
                rows_with_exact_match,
                :
            ].astype(float)

        weight_sums = weights.sum(
            axis=1
        )

        if not np.allclose(
            weight_sums,
            1.0,
        ):
            raise FloatingPointError(
                "Calculated weights do not sum to one."
            )

        nearest_index = np.argmin(
            distances,
            axis=1,
        )

        nearest_distance = np.min(
            distances,
            axis=1,
        )

        effective_number = (
            1.0
            / np.sum(
                weights ** 2,
                axis=1,
            )
        )

        weight_frame = fair_year[
            [
                "fair_sample_id",
                "scenario",
                "scenario_label",
                "original_sample",
                "year",
                "gsat",
                "cumulative_warming",
            ]
        ].copy()

        for pathway_index, pathway in enumerate(
            pathway_order
        ):
            weight_frame[
                f"weight_{pathway.lower()}"
            ] = weights[
                :,
                pathway_index,
            ]

        weight_frames.append(
            weight_frame
        )

        diagnostic_frame = fair_year[
            [
                "fair_sample_id",
                "scenario",
                "scenario_label",
                "original_sample",
                "year",
            ]
        ].copy()

        for pathway_index, pathway in enumerate(
            pathway_order
        ):
            diagnostic_frame[
                f"distance_{pathway.lower()}"
            ] = distances[
                :,
                pathway_index,
            ]

        diagnostic_frame[
            "nearest_pathway"
        ] = np.array(
            pathway_order,
            dtype=object,
        )[nearest_index]

        diagnostic_frame[
            "nearest_reference_distance"
        ] = nearest_distance

        diagnostic_frame[
            "effective_number_of_pathways"
        ] = effective_number

        diagnostic_frame[
            "outside_reference_domain"
        ] = (
            nearest_distance
            > outside_threshold
        )

        diagnostic_frames.append(
            diagnostic_frame
        )

    weights_df = pd.concat(
        weight_frames,
        ignore_index=True,
    )

    diagnostics_df = pd.concat(
        diagnostic_frames,
        ignore_index=True,
    )

    return (
        weights_df,
        diagnostics_df,
    )


# ============================================================
# 6. CREATE FAIR–SEJ SAMPLE PAIRING
# ============================================================

def create_sej_pairing(
    n_fair,
    n_sej,
    seed=1234,
):
    """
    Pair every FaIR trajectory with one complete SEJ row.

    If the numbers are equal, this is a random permutation:
    every SEJ row is used exactly once.

    If the numbers differ, SEJ rows are sampled with replacement.
    """

    rng = np.random.default_rng(
        seed
    )

    if n_fair == n_sej:

        pairing = rng.permutation(
            n_sej
        )

        replacement = False

    else:

        print(
            "WARNING: The numbers of FaIR samples and SEJ "
            "samples differ."
        )
        print(
            f"FaIR samples: {n_fair:,}"
        )
        print(
            f"SEJ samples:  {n_sej:,}"
        )
        print(
            "SEJ rows will be sampled with replacement.\n"
        )

        pairing = rng.choice(
            n_sej,
            size=n_fair,
            replace=True,
        )

        replacement = True

    pairing_df = pd.DataFrame(
        {
            "fair_sample_id": np.arange(
                n_fair,
                dtype=int,
            ),
            "sej_sample_id": pairing.astype(
                int
            ),
            "sampled_with_replacement": (
                replacement
            ),
        }
    )

    return pairing_df


# ============================================================
# 7. PREPARE SEJ RESPONSE METADATA
# ============================================================

def create_response_metadata(
    meta_df,
):
    """
    Convert the B25 variable metadata into one row per response.

    For example:

        EA2_21
        EA35_21
        EA5_21

    all correspond to the response code:

        EA

    for EAIS accumulation in 2100.
    """

    metadata = meta_df.copy()

    metadata["response_code"] = (
        metadata["variable"]
        .str.extract(
            r"^(SLR|[EWG][ADR]?)"
        )
    )

    if metadata[
        "response_code"
    ].isna().any():
        bad_variables = metadata.loc[
            metadata["response_code"].isna(),
            "variable",
        ].tolist()

        raise ValueError(
            "Could not identify response codes for:\n"
            + "\n".join(
                bad_variables
            )
        )

    response_metadata = (
        metadata[
            [
                "response_code",
                "ice_sheet_code",
                "ice_sheet",
                "process_code",
                "process",
                "year",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "year",
                "response_code",
            ]
        )
        .reset_index(drop=True)
    )

    duplicate_check = (
        response_metadata
        .groupby(
            [
                "year",
                "response_code",
            ]
        )
        .size()
    )

    if (
        duplicate_check
        > 1
    ).any():
        raise ValueError(
            "Response metadata are not unique by year and "
            "response code."
        )

    return response_metadata


# ============================================================
# 8. INTERPOLATE MATCHED SEJ RESPONSES
# ============================================================

def interpolate_sej_responses(
    mc_df,
    meta_df,
    weights_df,
    pairing_df,
):
    """
    Interpolate every SEJ response draw-by-draw.

    The same paired SEJ sample is used for:

        all projection years
        all ice sheets
        all processes
        all three warming anchors

    Returns
    -------
    interpolated_long : pandas.DataFrame
        One row for each:
            FaIR sample
            projection year
            SEJ response type
    """

    response_metadata = (
        create_response_metadata(
            meta_df
        )
    )

    n_fair = len(
        pairing_df
    )

    sej_indices = pairing_df[
        "sej_sample_id"
    ].to_numpy(dtype=int)

    if (
        sej_indices.min() < 0
        or sej_indices.max() >= len(mc_df)
    ):
        raise IndexError(
            "The SEJ pairing contains an invalid row index."
        )

    output_frames = []

    for projection_year in projection_years:

        print(
            f"Interpolating SEJ responses for "
            f"{projection_year}..."
        )

        year_weights = (
            weights_df.loc[
                weights_df["year"]
                == projection_year
            ]
            .sort_values(
                "fair_sample_id"
            )
            .reset_index(drop=True)
        )

        if len(
            year_weights
        ) != n_fair:
            raise ValueError(
                f"Expected {n_fair:,} weight rows for "
                f"{projection_year}, but found "
                f"{len(year_weights):,}."
            )

        weight_matrix = year_weights[
            [
                "weight_low",
                "weight_intermediate",
                "weight_high",
            ]
        ].to_numpy(dtype=float)

        year_metadata = (
            response_metadata.loc[
                response_metadata["year"]
                == projection_year
            ]
            .copy()
            .reset_index(drop=True)
        )

        for _, response in year_metadata.iterrows():

            response_code = response[
                "response_code"
            ]

            year_code = projection_year_code_map[
                projection_year
            ]
            
            columns = [
                f"{response_code}2_{year_code}",
                f"{response_code}35_{year_code}",
                f"{response_code}5_{year_code}",
            ]

            missing_columns = [
                column
                for column in columns
                if column not in mc_df.columns
            ]

            if missing_columns:
                raise KeyError(
                    "Missing SEJ anchor columns for "
                    f"{response_code} in {projection_year}:\n"
                    + "\n".join(
                        missing_columns
                    )
                )

            anchor_values = mc_df[
                columns
            ].to_numpy(
                dtype=float
            )

            matched_anchor_values = (
                anchor_values[
                    sej_indices,
                    :
                ]
            )

            if matched_anchor_values.shape != (
                n_fair,
                3,
            ):
                raise ValueError(
                    "Unexpected matched SEJ anchor shape for "
                    f"{response_code}: "
                    f"{matched_anchor_values.shape}"
                )

            interpolated_values = np.sum(
                matched_anchor_values
                * weight_matrix,
                axis=1,
            )

            frame = year_weights[
                [
                    "fair_sample_id",
                    "scenario",
                    "scenario_label",
                    "original_sample",
                    "year",
                    "gsat",
                    "cumulative_warming",
                    "weight_low",
                    "weight_intermediate",
                    "weight_high",
                ]
            ].copy()

            frame[
                "sej_sample_id"
            ] = sej_indices

            frame[
                "response_code"
            ] = response_code

            frame[
                "ice_sheet_code"
            ] = response[
                "ice_sheet_code"
            ]

            frame[
                "ice_sheet"
            ] = response[
                "ice_sheet"
            ]

            frame[
                "process_code"
            ] = response[
                "process_code"
            ]

            frame[
                "process"
            ] = response[
                "process"
            ]

            frame[
                "value"
            ] = interpolated_values

            output_frames.append(
                frame
            )

    interpolated_long = pd.concat(
        output_frames,
        ignore_index=True,
    )

    return interpolated_long


# ============================================================
# 9. DERIVED TOTAL AIS
# ============================================================

def add_interpolated_total_ais(
    interpolated_long,
):
    """
    Add Total AIS as EAIS + WAIS draw-by-draw.

    Returns
    -------
    total_ext : pandas.DataFrame
        Original total contributions plus Total AIS.
    """

    total_df = interpolated_long.loc[
        interpolated_long[
            "process_code"
        ] == "TOTAL"
    ].copy()

    index_columns = [
        "fair_sample_id",
        "sej_sample_id",
        "scenario",
        "scenario_label",
        "original_sample",
        "year",
        "gsat",
        "cumulative_warming",
        "weight_low",
        "weight_intermediate",
        "weight_high",
    ]

    ais = (
        total_df.loc[
            total_df[
                "ice_sheet_code"
            ].isin(
                [
                    "E",
                    "W",
                ]
            )
        ]
        .pivot(
            index=index_columns,
            columns="ice_sheet_code",
            values="value",
        )
        .dropna(
            subset=[
                "E",
                "W",
            ]
        )
        .reset_index()
    )

    ais[
        "value"
    ] = (
        ais["E"]
        + ais["W"]
    )

    ais[
        "response_code"
    ] = "AIS"

    ais[
        "ice_sheet_code"
    ] = "AIS"

    ais[
        "ice_sheet"
    ] = "Total AIS"

    ais[
        "process_code"
    ] = "TOTAL"

    ais[
        "process"
    ] = "Total"

    ais = ais[
        total_df.columns
    ]

    total_ext = pd.concat(
        [
            total_df,
            ais,
        ],
        ignore_index=True,
    )

    return total_ext


# ============================================================
# 10. DERIVED RUNOFF MINUS ACCUMULATION
# ============================================================

def add_interpolated_runoff_minus_accumulation(
    interpolated_long,
):
    """
    Add runoff minus accumulation draw-by-draw.

    Accumulation is already represented as a signed negative
    SLR contribution. Therefore:

        runoff minus accumulation contribution = R + A
    """

    process_df = interpolated_long.loc[
        interpolated_long[
            "process_code"
        ].isin(
            [
                "A",
                "D",
                "R",
            ]
        )
    ].copy()

    index_columns = [
        "fair_sample_id",
        "sej_sample_id",
        "scenario",
        "scenario_label",
        "original_sample",
        "year",
        "gsat",
        "cumulative_warming",
        "weight_low",
        "weight_intermediate",
        "weight_high",
        "ice_sheet_code",
        "ice_sheet",
    ]

    runoff_accumulation = (
        process_df.loc[
            process_df[
                "process_code"
            ].isin(
                [
                    "A",
                    "R",
                ]
            )
        ]
        .pivot(
            index=index_columns,
            columns="process_code",
            values="value",
        )
        .dropna(
            subset=[
                "A",
                "R",
            ]
        )
        .reset_index()
    )

    runoff_accumulation[
        "value"
    ] = (
        runoff_accumulation["R"]
        + runoff_accumulation["A"]
    )

    runoff_accumulation[
        "response_code"
    ] = (
        runoff_accumulation[
            "ice_sheet_code"
        ]
        + "R-A"
    )

    runoff_accumulation[
        "process_code"
    ] = "R-A"

    runoff_accumulation[
        "process"
    ] = (
        "Runoff minus accumulation"
    )

    runoff_accumulation = (
        runoff_accumulation[
            process_df.columns
        ]
    )

    process_ext = pd.concat(
        [
            process_df,
            runoff_accumulation,
        ],
        ignore_index=True,
    )

    return process_ext


# ============================================================
# 11. CONVERT OUTPUT TO XARRAY
# ============================================================

def convert_interpolated_to_xarray(
    interpolated_long,
):
    """
    Convert the interpolated responses to a compact xarray
    Dataset.

    Dataset dimensions:
        samples
        projection_year
        response
    """

    response_lookup = (
        interpolated_long[
            [
                "response_code",
                "ice_sheet_code",
                "ice_sheet",
                "process_code",
                "process",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "response_code"
        )
        .reset_index(drop=True)
    )

    response_codes = response_lookup[
        "response_code"
    ].tolist()

    sample_lookup = (
        interpolated_long[
            [
                "fair_sample_id",
                "sej_sample_id",
                "scenario",
                "scenario_label",
                "original_sample",
            ]
        ]
        .drop_duplicates(
            subset="fair_sample_id"
        )
        .sort_values(
            "fair_sample_id"
        )
    )

    value_table = (
        interpolated_long
        .pivot(
            index=[
                "fair_sample_id",
                "year",
            ],
            columns="response_code",
            values="value",
        )
        .reindex(
            columns=response_codes
        )
    )

    full_index = pd.MultiIndex.from_product(
        [
            sample_lookup[
                "fair_sample_id"
            ].values,
            projection_years,
        ],
        names=[
            "fair_sample_id",
            "year",
        ],
    )

    value_table = value_table.reindex(
        full_index
    )

    values = value_table.to_numpy().reshape(
        len(sample_lookup),
        len(projection_years),
        len(response_codes),
    )

    dataset = xr.Dataset(
        {
            "sea_level_contribution": (
                (
                    "samples",
                    "projection_year",
                    "response",
                ),
                values,
            )
        },
        coords={
            "samples": sample_lookup[
                "fair_sample_id"
            ].values,
            "projection_year": projection_years,
            "response": response_codes,
            "sej_sample_id": (
                "samples",
                sample_lookup[
                    "sej_sample_id"
                ].values,
            ),
            "scenario": (
                "samples",
                sample_lookup[
                    "scenario"
                ].values,
            ),
            "scenario_label": (
                "samples",
                sample_lookup[
                    "scenario_label"
                ].values,
            ),
            "original_sample": (
                "samples",
                sample_lookup[
                    "original_sample"
                ].values,
            ),
            "ice_sheet_code": (
                "response",
                response_lookup[
                    "ice_sheet_code"
                ].values,
            ),
            "ice_sheet": (
                "response",
                response_lookup[
                    "ice_sheet"
                ].values,
            ),
            "process_code": (
                "response",
                response_lookup[
                    "process_code"
                ].values,
            ),
            "process": (
                "response",
                response_lookup[
                    "process"
                ].values,
            ),
        },
    )

    dataset[
        "sea_level_contribution"
    ].attrs[
        "description"
    ] = (
        "SEJ ice-sheet sea-level contribution interpolated "
        "onto FaIR climate trajectories"
    )

    return dataset


# ============================================================
# 12. VALIDATION AND SUMMARY
# ============================================================

def validate_results(
    weights_df,
    interpolated_long,
):
    """
    Run basic consistency checks.
    """

    weight_columns = [
        "weight_low",
        "weight_intermediate",
        "weight_high",
    ]

    weight_sums = weights_df[
        weight_columns
    ].sum(
        axis=1
    )

    if not np.allclose(
        weight_sums,
        1.0,
    ):
        raise ValueError(
            "At least one climate-weight row does not sum "
            "to one."
        )

    if not np.isfinite(
        interpolated_long[
            "value"
        ].to_numpy()
    ).all():
        raise ValueError(
            "Interpolated responses contain non-finite values."
        )

    duplicated = interpolated_long.duplicated(
        subset=[
            "fair_sample_id",
            "year",
            "response_code",
        ]
    )

    if duplicated.any():
        raise ValueError(
            "Duplicate interpolated response records were found."
        )

    print(
        "\nValidation checks passed."
    )


def print_weight_summary(
    weights_df,
    diagnostics_df,
):
    """
    Print climate-weight summaries by SSP and year.
    """

    print(
        "\nMean climate weights by SSP and year:"
    )

    summary = (
        weights_df
        .groupby(
            [
                "scenario_label",
                "year",
            ]
        )[
            [
                "weight_low",
                "weight_intermediate",
                "weight_high",
            ]
        ]
        .mean()
        .round(3)
    )

    print(summary)

    print(
        "\nPercentage outside the reference domain:"
    )

    outside_summary = (
        diagnostics_df
        .groupby(
            [
                "scenario_label",
                "year",
            ]
        )[
            "outside_reference_domain"
        ]
        .mean()
        .mul(100)
        .round(2)
    )

    print(outside_summary)


# ============================================================
# 13. SAVE OUTPUTS
# ============================================================

def save_outputs(
    fair_features,
    weights_df,
    diagnostics_df,
    pairing_df,
    interpolated_long,
    total_ext,
    process_ext,
    interpolated_xr,
    directory=output_dir,
):
    """
    Save the interpolation outputs.
    """

    directory = Path(
        directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fair_features.to_csv(
        directory
        / "fair_climate_features.csv",
        index=False,
    )

    weights_df.to_csv(
        directory
        / "fair_sej_climate_weights.csv",
        index=False,
    )

    diagnostics_df.to_csv(
        directory
        / "fair_sej_diagnostics.csv",
        index=False,
    )

    pairing_df.to_csv(
        directory
        / "fair_sej_sample_pairing.csv",
        index=False,
    )

    interpolated_long.to_csv(
        directory
        / "fair_sej_interpolated_responses.csv",
        index=False,
    )

    total_ext.to_csv(
        directory
        / "fair_sej_interpolated_totals.csv",
        index=False,
    )

    process_ext.to_csv(
        directory
        / "fair_sej_interpolated_processes.csv",
        index=False,
    )

    interpolated_xr.to_netcdf(
        directory
        / "fair_sej_interpolated_responses.nc"
    )

    np.save(
        directory
        / "fair_sej_pairing.npy",
        pairing_df[
            "sej_sample_id"
        ].to_numpy(dtype=int),
    )

    print(
        f"\nOutputs saved to:\n{directory}"
    )


# ============================================================
# 14. RUN COMPLETE WORKFLOW
# ============================================================

def main():

    # --------------------------------------------------------
    # Load FaIR temperatures
    # --------------------------------------------------------

    pooled = load_fair_ensemble(
        files=fair_files,
        start_year=(
            cumulative_warming_start_year
        ),
        end_year=max(
            projection_years
        ),
    )

    # --------------------------------------------------------
    # Calculate GSAT and cumulative warming
    # --------------------------------------------------------

    fair_features = (
        calculate_fair_climate_features(
            pooled=pooled,
            years=projection_years,
            start_year=(
                cumulative_warming_start_year
            ),
        )
    )

    print(
        "\nFaIR climate-feature summary:"
    )

    climate_feature_summary = (
        fair_features
        .groupby(
            [
                "scenario_label",
                "year",
            ]
        )
        .agg(
            gsat_mean=("gsat", "mean"),
            gsat_std=("gsat", "std"),
            gsat_min=("gsat", "min"),
            gsat_max=("gsat", "max"),
            cumulative_warming_mean=(
                "cumulative_warming",
                "mean",
            ),
            cumulative_warming_std=(
                "cumulative_warming",
                "std",
            ),
            cumulative_warming_min=(
                "cumulative_warming",
                "min",
            ),
            cumulative_warming_max=(
                "cumulative_warming",
                "max",
            ),
        )
    )
    
    print(climate_feature_summary.round(3))

    # --------------------------------------------------------
    # Calculate climate-space weights
    # --------------------------------------------------------

    (
        weights_df,
        diagnostics_df,
    ) = calculate_climate_weights(
        feature_df=fair_features,
        reference_df=reference_climate,
        scales=feature_scales,
        bandwidth=kernel_bandwidth,
        outside_threshold=(
            outside_domain_distance
        ),
    )

    # --------------------------------------------------------
    # Load B25 Monte Carlo data
    # --------------------------------------------------------

    b25_data = b25.load_and_prepare_data(
        file_path=str(
            sej_data_file
        )
    )

    mc_df = b25_data[
        "mc_df"
    ]

    meta_df = b25_data[
        "meta_df"
    ]

    print(
        f"\nNumber of FaIR samples: "
        f"{pooled.sizes['samples']:,}"
    )

    print(
        f"Number of SEJ samples:  "
        f"{len(mc_df):,}"
    )

    # --------------------------------------------------------
    # Pair FaIR samples with complete matched SEJ rows
    # --------------------------------------------------------

    pairing_df = create_sej_pairing(
        n_fair=pooled.sizes[
            "samples"
        ],
        n_sej=len(
            mc_df
        ),
        seed=pairing_seed,
    )

    # --------------------------------------------------------
    # Interpolate all original SEJ variables
    # --------------------------------------------------------

    interpolated_long = (
        interpolate_sej_responses(
            mc_df=mc_df,
            meta_df=meta_df,
            weights_df=weights_df,
            pairing_df=pairing_df,
        )
    )

    # --------------------------------------------------------
    # Calculate derived variables
    # --------------------------------------------------------

    total_ext = (
        add_interpolated_total_ais(
            interpolated_long
        )
    )

    process_ext = (
        add_interpolated_runoff_minus_accumulation(
            interpolated_long
        )
    )

    # --------------------------------------------------------
    # Convert original interpolated variables to xarray
    # --------------------------------------------------------

    interpolated_xr = (
        convert_interpolated_to_xarray(
            interpolated_long
        )
    )

    # Attach climate variables and weights to xarray output.
    feature_xr = (
        weights_df
        .set_index(
            [
                "fair_sample_id",
                "year",
            ]
        )[
            [
                "gsat",
                "cumulative_warming",
                "weight_low",
                "weight_intermediate",
                "weight_high",
            ]
        ]
        .to_xarray()
        .rename(
            {
                "fair_sample_id": "samples",
                "year": "projection_year",
            }
        )
    )

    interpolated_xr = xr.merge(
        [
            interpolated_xr,
            feature_xr,
        ],
        compat="override",
    )

    interpolated_xr.attrs[
        "kernel_bandwidth"
    ] = kernel_bandwidth

    interpolated_xr.attrs[
        "gsat_scale_degC"
    ] = feature_scales[
        "gsat"
    ]

    interpolated_xr.attrs[
        "cumulative_warming_scale_Kyr"
    ] = feature_scales[
        "cumulative_warming"
    ]

    interpolated_xr.attrs[
        "pairing_seed"
    ] = pairing_seed

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_results(
        weights_df=weights_df,
        interpolated_long=interpolated_long,
    )

    print_weight_summary(
        weights_df=weights_df,
        diagnostics_df=diagnostics_df,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        fair_features=fair_features,
        weights_df=weights_df,
        diagnostics_df=diagnostics_df,
        pairing_df=pairing_df,
        interpolated_long=interpolated_long,
        total_ext=total_ext,
        process_ext=process_ext,
        interpolated_xr=interpolated_xr,
        directory=output_dir,
    )

    return {
        "pooled": pooled,
        "fair_features": fair_features,
        "weights": weights_df,
        "diagnostics": diagnostics_df,
        "pairing": pairing_df,
        "interpolated_long": interpolated_long,
        "total_ext": total_ext,
        "process_ext": process_ext,
        "interpolated_xr": interpolated_xr,
        "b25_data": b25_data,
    }


if __name__ == "__main__":

    results = main()