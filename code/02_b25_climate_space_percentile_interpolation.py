#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_b25_climate_space_percentile_interpolation.py

Climate-condition the temporally coupled B25 SEJ using:

    - three warming anchors: 2.0 C, 3.5 C, 5.0 C
    - FaIR GSAT
    - cumulative warming from 2000 through each projection year
    - percentile matching ACROSS warming scenarios
    - temporally correlated percentile ranks ACROSS years

CRITICAL DISTINCTION
--------------------
The original B25 Monte Carlo row IDs are NOT assumed to correspond
between warming scenarios.

At a given projection year, if the latent percentile is u = 0.73,
the script independently selects approximately the 73rd-percentile
complete B25 row from each of the 2 C, 3.5 C, and 5 C distributions,
ranked by Total AIS. It then interpolates the response values across
those three percentile-matched rows using climate-space weights.

Across years, u is NOT held fixed. A trajectory can have, for example:

    u_2100 = 0.73
    u_2200 = 0.81
    u_2300 = 0.78

The year-specific u values come from the temporally correlated rank
processes created in Script 01.

CLIMATE SPACE
-------------
At each projection year, FaIR climate state is represented by:

    GSAT
    cumulative warming from 2000 through that year

The B25 expert pathways are reference anchors:

2100:
    LOW       (2.0 C, 155.0 K yr)
    MODERATE  (3.5 C, 204.5 K yr)
    HIGH      (5.0 C, 262.0 K yr)

2200:
    LOW       (2.0 C, 355.0 K yr)
    MODERATE  (3.5 C, 554.5 K yr)
    HIGH      (5.0 C, 762.0 K yr)

2300:
    LOW       (2.0 C, 555.0 K yr)
    MODERATE  (3.5 C, 904.5 K yr)
    HIGH      (5.0 C, 1262.0 K yr)

Distance scales:
    1 C GSAT = 1 scaled unit
    200 K yr cumulative warming = 1 scaled unit

Gaussian-kernel bandwidth:
    1.0

OUTPUT
------
One climate-conditioned B25 row per FaIR trajectory with interpolated
response anchors at 2100, 2200, and 2300, plus diagnostics.

@author: elizabeth
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from B25data_processing import project_dir


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_DIR = Path(project_dir)

FACTS_DIR = Path(
    "/Users/elizabeth/FACTS2"
)

FAIR_FILES = {
    "ssp119": FACTS_DIR / "experiments" / "ssp119" / "output" / "fair-temperature" / "gsat.nc",
    "ssp126": FACTS_DIR / "experiments" / "ssp126" / "output" / "fair-temperature" / "gsat.nc",
    "ssp245": FACTS_DIR / "experiments" / "ssp245" / "output" / "fair-temperature" / "gsat.nc",
    "ssp370": FACTS_DIR / "experiments" / "ssp370" / "output" / "fair-temperature" / "gsat.nc",
    "ssp585": FACTS_DIR / "experiments" / "ssp585" / "output" / "fair-temperature" / "gsat.nc",
}

SCENARIO_LABELS = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

TEMPORAL_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "01_temporal_coupling"
)

ROLE_FILES = {
    "LOW": {
        "warming": 2.0,
        "warming_code": "2",
        "samples": TEMPORAL_DIR / "b25_b19style_low_2.0C_dp21_rcp26_coupled_samples.csv",
        "metadata": TEMPORAL_DIR / "b25_b19style_low_2.0C_dp21_rcp26_coupling_metadata.csv",
    },
    "MODERATE": {
        "warming": 3.5,
        "warming_code": "35",
        "samples": TEMPORAL_DIR / "b25_b19style_moderate_3.5C_dp21_rcp45_coupled_samples.csv",
        "metadata": TEMPORAL_DIR / "b25_b19style_moderate_3.5C_dp21_rcp45_coupling_metadata.csv",
    },
    "HIGH": {
        "warming": 5.0,
        "warming_code": "5",
        "samples": TEMPORAL_DIR / "b25_b19style_high_5.0C_dp21_rcp85_coupled_samples.csv",
        "metadata": TEMPORAL_DIR / "b25_b19style_high_5.0C_dp21_rcp85_coupling_metadata.csv",
    },
}

ROLE_ORDER = [
    "LOW",
    "MODERATE",
    "HIGH",
]

PROJECTION_YEARS = [
    2100,
    2200,
    2300,
]

YEAR_CODE = {
    2100: "21",
    2200: "22",
    2300: "23",
}

REFERENCE_CLIMATE = pd.DataFrame(
    [
        {"year": 2100, "role": "LOW",      "gsat": 2.0, "cumulative_warming": 155.0},
        {"year": 2100, "role": "MODERATE", "gsat": 3.5, "cumulative_warming": 204.5},
        {"year": 2100, "role": "HIGH",     "gsat": 5.0, "cumulative_warming": 262.0},

        {"year": 2200, "role": "LOW",      "gsat": 2.0, "cumulative_warming": 355.0},
        {"year": 2200, "role": "MODERATE", "gsat": 3.5, "cumulative_warming": 554.5},
        {"year": 2200, "role": "HIGH",     "gsat": 5.0, "cumulative_warming": 762.0},

        {"year": 2300, "role": "LOW",      "gsat": 2.0, "cumulative_warming": 555.0},
        {"year": 2300, "role": "MODERATE", "gsat": 3.5, "cumulative_warming": 904.5},
        {"year": 2300, "role": "HIGH",     "gsat": 5.0, "cumulative_warming": 1262.0},
    ]
)

FEATURE_SCALES = {
    "gsat": 1.0,
    "cumulative_warming": 200.0,
}

KERNEL_BANDWIDTH = 1.0
PAIRING_SEED = 20269723

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "02_climate_space_percentile_interpolation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. LOAD FAIR
# ============================================================

def load_fair() -> xr.DataArray:

    arrays = []

    for scenario, path in FAIR_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"FaIR file not found:\n{path}"
            )

        print(
            f"\nReading FaIR data for {scenario}:\n{path}"
        )

        with xr.open_dataset(path) as ds:

            temperature = ds[
                "surface_temperature"
            ]

            if "locations" in temperature.dims:
                temperature = temperature.squeeze(
                    "locations",
                    drop=True,
                )

            temperature = (
                temperature
                .sel(
                    years=slice(
                        2000,
                        2300,
                    )
                )
                .transpose(
                    "samples",
                    "years",
                )
                .load()
            )

        n = temperature.sizes[
            "samples"
        ]

        original_sample = temperature[
            "samples"
        ].values.copy()

        temperature = temperature.assign_coords(
            scenario=(
                "samples",
                np.repeat(
                    scenario,
                    n,
                ),
            ),
            scenario_label=(
                "samples",
                np.repeat(
                    SCENARIO_LABELS[
                        scenario
                    ],
                    n,
                ),
            ),
            original_sample=(
                "samples",
                original_sample,
            ),
        )

        arrays.append(
            temperature
        )

    pooled = xr.concat(
        arrays,
        dim="samples",
    )

    pooled = pooled.assign_coords(
        samples=np.arange(
            pooled.sizes[
                "samples"
            ],
            dtype=int,
        )
    )

    return pooled


# ============================================================
# 3. FAIR CLIMATE FEATURES
# ============================================================

def calculate_fair_features(
    pooled: xr.DataArray,
) -> pd.DataFrame:

    records = []

    for year in PROJECTION_YEARS:

        subset = (
            pooled
            .sel(
                years=slice(
                    2000,
                    year,
                )
            )
            .transpose(
                "samples",
                "years",
            )
        )

        x = subset[
            "years"
        ].values.astype(float)

        y = subset.values

        if hasattr(
            np,
            "trapezoid",
        ):
            cumulative = np.trapezoid(
                y,
                x=x,
                axis=1,
            )
        else:
            cumulative = np.trapz(
                y,
                x=x,
                axis=1,
            )

        gsat = pooled.sel(
            years=year
        ).values.astype(float)

        records.append(
            pd.DataFrame(
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
                    "year": year,
                    "gsat": gsat,
                    "cumulative_warming": cumulative,
                }
            )
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# 4. CLIMATE-SPACE WEIGHTS
# ============================================================

def calculate_climate_weights(
    fair_features: pd.DataFrame,
) -> pd.DataFrame:

    output = []

    scale_vector = np.array(
        [
            FEATURE_SCALES[
                "gsat"
            ],
            FEATURE_SCALES[
                "cumulative_warming"
            ],
        ],
        dtype=float,
    )

    for year in PROJECTION_YEARS:

        fair_year = (
            fair_features.loc[
                fair_features[
                    "year"
                ]
                == year
            ]
            .sort_values(
                "fair_sample_id"
            )
            .reset_index(
                drop=True
            )
        )

        ref_year = (
            REFERENCE_CLIMATE.loc[
                REFERENCE_CLIMATE[
                    "year"
                ]
                == year
            ]
            .set_index(
                "role"
            )
            .loc[
                ROLE_ORDER
            ]
            .reset_index()
        )

        fair_matrix = fair_year[
            [
                "gsat",
                "cumulative_warming",
            ]
        ].to_numpy(
            dtype=float
        )

        ref_matrix = ref_year[
            [
                "gsat",
                "cumulative_warming",
            ]
        ].to_numpy(
            dtype=float
        )

        differences = (
            fair_matrix[
                :,
                np.newaxis,
                :
            ]
            - ref_matrix[
                np.newaxis,
                :,
                :
            ]
        )

        scaled = (
            differences
            / scale_vector[
                np.newaxis,
                np.newaxis,
                :
            ]
        )

        distances = np.sqrt(
            np.sum(
                scaled ** 2,
                axis=2,
            )
        )

        log_weights = (
            -0.5
            * (
                distances
                / KERNEL_BANDWIDTH
            ) ** 2
        )

        log_weights -= np.max(
            log_weights,
            axis=1,
            keepdims=True,
        )

        raw = np.exp(
            log_weights
        )

        weights = (
            raw
            / raw.sum(
                axis=1,
                keepdims=True,
            )
        )

        # Exact anchor matches receive exactly one-hot weights.
        exact = (
            distances
            <= 1e-12
        )

        rows = np.any(
            exact,
            axis=1,
        )

        if np.any(
            rows
        ):
            weights[
                rows,
                :
            ] = exact[
                rows,
                :
            ].astype(float)

        frame = fair_year.copy()

        frame[
            "weight_low"
        ] = weights[
            :,
            0
        ]

        frame[
            "weight_moderate"
        ] = weights[
            :,
            1
        ]

        frame[
            "weight_high"
        ] = weights[
            :,
            2
        ]

        frame[
            "nearest_anchor_distance"
        ] = distances.min(
            axis=1
        )

        output.append(
            frame
        )

    return pd.concat(
        output,
        ignore_index=True,
    )


# ============================================================
# 5. LOAD TEMPORAL LIBRARIES / RANK METADATA
# ============================================================

def load_temporal_inputs():

    libraries = {}
    metadata = {}

    n = None

    for role in ROLE_ORDER:

        info = ROLE_FILES[
            role
        ]

        for key in [
            "samples",
            "metadata",
        ]:

            if not info[
                key
            ].exists():

                raise FileNotFoundError(
                    f"{role} {key} file not found:\n"
                    f"{info[key]}"
                )

        library = pd.read_csv(
            info[
                "samples"
            ]
        )

        meta = pd.read_csv(
            info[
                "metadata"
            ]
        )

        if n is None:
            n = len(
                library
            )
        elif len(
            library
        ) != n:
            raise ValueError(
                "Temporal libraries differ in size."
            )

        libraries[
            role
        ] = library

        metadata[
            role
        ] = meta

    return (
        libraries,
        metadata,
        n,
    )


# ============================================================
# 6. FAIR -> SHARED LATENT TRAJECTORY PAIRING
# ============================================================

def create_pairing(
    n_fair: int,
    n_trajectories: int,
) -> pd.DataFrame:

    rng = np.random.default_rng(
        PAIRING_SEED
    )

    if n_fair == n_trajectories:

        trajectory_ids = rng.permutation(
            n_trajectories
        )

        replacement = False

    else:

        trajectory_ids = rng.choice(
            n_trajectories,
            size=n_fair,
            replace=True,
        )

        replacement = True

    return pd.DataFrame(
        {
            "fair_sample_id": np.arange(
                n_fair,
                dtype=int,
            ),
            "latent_trajectory_id": trajectory_ids.astype(
                int
            ),
            "sampled_with_replacement": replacement,
        }
    )


# ============================================================
# 7. TEMPORALLY VARYING PERCENTILE RANK
# ============================================================

def calculate_interpolated_percentile_ranks(
    weights_df: pd.DataFrame,
    pairing_df: pd.DataFrame,
    temporal_metadata: dict,
) -> pd.DataFrame:
    """
    Blend the three scenario-specific latent percentile processes in
    climate space.

    This does NOT hold percentile fixed through time.

    Example:
        2100 -> 0.73
        2200 -> 0.81
        2300 -> 0.78
    """

    frames = []

    paired_ids = pairing_df[
        "latent_trajectory_id"
    ].to_numpy(
        dtype=int
    )

    for year in PROJECTION_YEARS:

        year_weights = (
            weights_df.loc[
                weights_df[
                    "year"
                ]
                == year
            ]
            .sort_values(
                "fair_sample_id"
            )
            .reset_index(
                drop=True
            )
        )

        role_percentiles = []

        for role in ROLE_ORDER:

            meta = (
                temporal_metadata[
                    role
                ]
                .loc[
                    temporal_metadata[
                        role
                    ][
                        "projection_year"
                    ]
                    == year
                ]
                .sort_values(
                    "trajectory_id"
                )
                .reset_index(
                    drop=True
                )
            )

            if len(
                meta
            ) <= paired_ids.max():
                raise IndexError(
                    f"{role} metadata does not cover paired trajectory IDs."
                )

            role_percentiles.append(
                meta.iloc[
                    paired_ids
                ][
                    "latent_percentile_rank"
                ].to_numpy(
                    dtype=float
                )
            )

        role_percentiles = np.column_stack(
            role_percentiles
        )

        climate_weights = year_weights[
            [
                "weight_low",
                "weight_moderate",
                "weight_high",
            ]
        ].to_numpy(
            dtype=float
        )

        u = np.sum(
            role_percentiles
            * climate_weights,
            axis=1,
        )

        frame = year_weights.copy()

        frame[
            "u_low_temporal"
        ] = role_percentiles[
            :,
            0
        ]

        frame[
            "u_moderate_temporal"
        ] = role_percentiles[
            :,
            1
        ]

        frame[
            "u_high_temporal"
        ] = role_percentiles[
            :,
            2
        ]

        frame[
            "interpolated_percentile"
        ] = np.clip(
            u,
            0.0,
            1.0,
        )

        frame[
            "latent_trajectory_id"
        ] = paired_ids

        frames.append(
            frame
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# 8. RESPONSE-COLUMN DISCOVERY
# ============================================================

def discover_response_prefixes(
    columns,
):
    """
    Discover B25 response families such as:
        E2_21 / E35_21 / E5_21
        WA2_21 / WA35_21 / WA5_21
        SLR2_21 / SLR35_21 / SLR5_21
    """

    pattern = re.compile(
        r"^(?P<prefix>.+?)(?P<warming>2|35|5)_(?P<year>21|22|23)$"
    )

    found = {}

    for column in columns:

        match = pattern.match(
            str(
                column
            )
        )

        if not match:
            continue

        key = (
            match.group(
                "prefix"
            ),
            match.group(
                "year"
            ),
        )

        found.setdefault(
            key,
            set(),
        ).add(
            match.group(
                "warming"
            )
        )

    complete = [
        key
        for key, warming_codes in found.items()
        if warming_codes
        == {
            "2",
            "35",
            "5",
        }
    ]

    return sorted(
        complete
    )


# ============================================================
# 9. TOTAL AIS FOR RANKING
# ============================================================

def total_ais_values(
    df: pd.DataFrame,
    warming_code: str,
    year_code: str,
) -> np.ndarray:

    e = f"E{warming_code}_{year_code}"
    w = f"W{warming_code}_{year_code}"

    missing = [
        column
        for column in [
            e,
            w,
        ]
        if column
        not in df.columns
    ]

    if missing:
        raise KeyError(
            "Missing Total AIS ranking columns:\n"
            + "\n".join(
                missing
            )
        )

    return (
        df[
            e
        ].to_numpy(
            dtype=float
        )
        + df[
            w
        ].to_numpy(
            dtype=float
        )
    )


# ============================================================
# 10. PERCENTILE-MATCHED CLIMATE INTERPOLATION
# ============================================================

def interpolate_responses(
    rank_df: pd.DataFrame,
    libraries: dict,
    n_library: int,
) -> pd.DataFrame:

    response_keys = discover_response_prefixes(
        libraries[
            "LOW"
        ].columns
    )

    frames = []

    for year in PROJECTION_YEARS:

        print(
            f"\nInterpolating percentile-matched responses for {year}..."
        )

        yc = YEAR_CODE[
            year
        ]

        ranks = (
            rank_df.loc[
                rank_df[
                    "year"
                ]
                == year
            ]
            .sort_values(
                "fair_sample_id"
            )
            .reset_index(
                drop=True
            )
        )

        u = ranks[
            "interpolated_percentile"
        ].to_numpy(
            dtype=float
        )

        # Convert percentile to zero-based rank position.
        positions = np.floor(
            u
            * n_library
        ).astype(int)

        positions = np.clip(
            positions,
            0,
            n_library - 1,
        )

        selected_rows = {}

        for role in ROLE_ORDER:

            info = ROLE_FILES[
                role
            ]

            library = libraries[
                role
            ]

            rank_values = total_ais_values(
                df=library,
                warming_code=info[
                    "warming_code"
                ],
                year_code=yc,
            )

            ordering = np.lexsort(
                (
                    np.arange(
                        n_library
                    ),
                    rank_values,
                )
            )

            selected_rows[
                role
            ] = library.iloc[
                ordering[
                    positions
                ]
            ].reset_index(
                drop=True
            )

        output = ranks[
            [
                "fair_sample_id",
                "scenario",
                "scenario_label",
                "original_sample",
                "year",
                "gsat",
                "cumulative_warming",
                "weight_low",
                "weight_moderate",
                "weight_high",
                "u_low_temporal",
                "u_moderate_temporal",
                "u_high_temporal",
                "interpolated_percentile",
                "latent_trajectory_id",
            ]
        ].copy()

        weight_matrix = ranks[
            [
                "weight_low",
                "weight_moderate",
                "weight_high",
            ]
        ].to_numpy(
            dtype=float
        )

        for prefix, key_year_code in response_keys:

            if key_year_code != yc:
                continue

            anchor_columns = {
                "LOW": f"{prefix}2_{yc}",
                "MODERATE": f"{prefix}35_{yc}",
                "HIGH": f"{prefix}5_{yc}",
            }

            values = np.column_stack(
                [
                    selected_rows[
                        role
                    ][
                        anchor_columns[
                            role
                        ]
                    ].to_numpy(
                        dtype=float
                    )
                    for role in ROLE_ORDER
                ]
            )

            interpolated = np.sum(
                values
                * weight_matrix,
                axis=1,
            )

            output[
                f"{prefix}_interp_{yc}"
            ] = interpolated

        frames.append(
            output
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# 11. WIDE OUTPUT
# ============================================================

def make_wide_output(
    interpolated_long: pd.DataFrame,
) -> pd.DataFrame:

    metadata_columns = [
        "fair_sample_id",
        "scenario",
        "scenario_label",
        "original_sample",
    ]

    base = (
        interpolated_long[
            metadata_columns
        ]
        .drop_duplicates(
            subset=[
                "fair_sample_id"
            ]
        )
        .sort_values(
            "fair_sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    response_columns = [
        column
        for column in interpolated_long.columns
        if "_interp_" in column
    ]

    for year in PROJECTION_YEARS:

        year_df = (
            interpolated_long.loc[
                interpolated_long[
                    "year"
                ]
                == year
            ]
            .sort_values(
                "fair_sample_id"
            )
            .reset_index(
                drop=True
            )
        )

        for column in response_columns:

            if column in year_df.columns:

                values = year_df[
                    column
                ]

                if values.notna().any():

                    base[
                        column
                    ] = values.to_numpy()

    return base


# ============================================================
# 12. VALIDATION / SUMMARY
# ============================================================

def print_summary(
    rank_df: pd.DataFrame,
):

    print(
        "\nMean climate-space weights by SSP and year:"
    )

    print(
        rank_df.groupby(
            [
                "scenario_label",
                "year",
            ]
        )[
            [
                "weight_low",
                "weight_moderate",
                "weight_high",
            ]
        ].mean().round(
            4
        )
    )

    print(
        "\nMean interpolated percentile by SSP and year:"
    )

    print(
        rank_df.groupby(
            [
                "scenario_label",
                "year",
            ]
        )[
            "interpolated_percentile"
        ].mean().round(
            4
        )
    )

    print(
        "\nSpearman correlation of interpolated percentile across years:"
    )

    for scenario_label, group in rank_df.groupby(
        "scenario_label"
    ):

        wide = group.pivot(
            index="fair_sample_id",
            columns="year",
            values="interpolated_percentile",
        )

        print(
            f"\n{scenario_label}"
        )

        print(
            wide.corr(
                method="spearman"
            ).round(
                4
            )
        )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "B25 CLIMATE-SPACE + PERCENTILE-MATCHED INTERPOLATION"
    )
    print(
        "=" * 72
    )

    pooled = load_fair()

    fair_features = calculate_fair_features(
        pooled
    )

    weights_df = calculate_climate_weights(
        fair_features
    )

    (
        libraries,
        temporal_metadata,
        n_library,
    ) = load_temporal_inputs()

    pairing_df = create_pairing(
        n_fair=pooled.sizes[
            "samples"
        ],
        n_trajectories=n_library,
    )

    rank_df = calculate_interpolated_percentile_ranks(
        weights_df=weights_df,
        pairing_df=pairing_df,
        temporal_metadata=temporal_metadata,
    )

    interpolated_long = interpolate_responses(
        rank_df=rank_df,
        libraries=libraries,
        n_library=n_library,
    )

    interpolated_wide = make_wide_output(
        interpolated_long
    )

    print_summary(
        rank_df
    )

    wide_file = (
        OUTPUT_DIR
        / "b25_climate_space_percentile_interpolated_anchors.csv"
    )

    long_file = (
        OUTPUT_DIR
        / "b25_climate_space_percentile_interpolated_long.csv"
    )

    rank_file = (
        OUTPUT_DIR
        / "b25_climate_space_temporal_percentile_diagnostics.csv"
    )

    pairing_file = (
        OUTPUT_DIR
        / "fair_latent_trajectory_pairing.csv"
    )

    interpolated_wide.to_csv(
        wide_file,
        index=False,
    )

    interpolated_long.to_csv(
        long_file,
        index=False,
    )

    rank_df.to_csv(
        rank_file,
        index=False,
    )

    pairing_df.to_csv(
        pairing_file,
        index=False,
    )

    print(
        "\nSaved climate-conditioned anchors to:"
    )
    print(
        wide_file
    )

    print(
        "\nSaved temporal percentile diagnostics to:"
    )
    print(
        rank_file
    )

    return {
        "wide": interpolated_wide,
        "long": interpolated_long,
        "ranks": rank_df,
        "pairing": pairing_df,
    }


if __name__ == "__main__":

    results = main()
