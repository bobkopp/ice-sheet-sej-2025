#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_b25_climate_space_categorical_selection.py

Climate-condition the temporally coupled B25 SEJ using a categorical
LOW / MODERATE / HIGH selection based on FaIR climate state in:

    GSAT
    cumulative warming from 2000 through each projection year

At each projection year:
    1. Calculate climate-space probabilities for:
           LOW       = 2.0 C
           MODERATE  = 3.5 C
           HIGH      = 5.0 C
    2. Use those probabilities as CATEGORICAL probabilities.
    3. Select ONE actual B25 warming-anchor distribution.
    4. Take the response from that selected temporally coupled library.

No weighted averaging of SLR responses is performed.

IMPORTANT
---------
The same original B25 Monte Carlo row is NOT assumed to correspond across
warming scenarios.

Temporal coupling is handled separately within the LOW, MODERATE, and HIGH
libraries by Script 01.

To reduce artificial category jitter through time, each FaIR realization
receives ONE persistent uniform selector U_category. The climate-space
probabilities change with year, so the selected category can still change
through time as the trajectory moves through climate space.

Example:
    U_category = 0.62

    2100 probabilities:
        LOW=0.70, MOD=0.25, HIGH=0.05
        -> selected LOW

    2200 probabilities:
        LOW=0.20, MOD=0.55, HIGH=0.25
        -> selected MODERATE

    2300 probabilities:
        LOW=0.05, MOD=0.20, HIGH=0.75
        -> selected HIGH

Within each selected library/year, the paired trajectory_id determines
which temporally coupled B25 row is used.

@author: elizabeth
"""

from __future__ import annotations

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
        "samples": TEMPORAL_DIR / "b25_b19style_low_2.0C_dp21_rcp26_coupled_samples.csv",
        "metadata": TEMPORAL_DIR / "b25_b19style_low_2.0C_dp21_rcp26_coupling_metadata.csv",
    },
    "MODERATE": {
        "warming": 3.5,
        "samples": TEMPORAL_DIR / "b25_b19style_moderate_3.5C_dp21_rcp45_coupled_samples.csv",
        "metadata": TEMPORAL_DIR / "b25_b19style_moderate_3.5C_dp21_rcp45_coupling_metadata.csv",
    },
    "HIGH": {
        "warming": 5.0,
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
CATEGORY_SEED = 1234

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "02_climate_space_categorical_selection"
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

            if "surface_temperature" not in ds:
                raise KeyError(
                    f"'surface_temperature' not found in:\n{path}"
                )

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

    print(
        f"\nPooled FaIR samples: "
        f"{pooled.sizes['samples']:,}"
    )

    return pooled


# ============================================================
# 3. FAIR CLIMATE FEATURES
# ============================================================

def calculate_fair_features(
    pooled: xr.DataArray,
) -> pd.DataFrame:

    frames = []

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

        frames.append(
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
        frames,
        ignore_index=True,
    )


# ============================================================
# 4. CLIMATE-SPACE PROBABILITIES
# ============================================================

def calculate_climate_probabilities(
    fair_features: pd.DataFrame,
) -> pd.DataFrame:

    frames = []

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

        log_prob = (
            -0.5
            * (
                distances
                / KERNEL_BANDWIDTH
            ) ** 2
        )

        log_prob -= np.max(
            log_prob,
            axis=1,
            keepdims=True,
        )

        raw = np.exp(
            log_prob
        )

        probabilities = (
            raw
            / raw.sum(
                axis=1,
                keepdims=True,
            )
        )

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
            probabilities[
                rows,
                :
            ] = exact[
                rows,
                :
            ].astype(float)

        frame = fair_year.copy()

        frame[
            "probability_low"
        ] = probabilities[
            :,
            0
        ]

        frame[
            "probability_moderate"
        ] = probabilities[
            :,
            1
        ]

        frame[
            "probability_high"
        ] = probabilities[
            :,
            2
        ]

        frame[
            "nearest_anchor_distance"
        ] = distances.min(
            axis=1
        )

        frames.append(
            frame
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# 5. LOAD COUPLED B25 LIBRARIES
# ============================================================

def load_libraries():
    """
    Load the three temporally coupled B25 libraries AND their coupling
    metadata.

    Important:
        trajectory_id in these coupled libraries is a NEW latent
        trajectory identifier created by Script 01. It is not an
        original B25 Monte Carlo row ID.

    Because Script 01 used the same shared Gaussian innovations for all
    three warming libraries, trajectory_id=i identifies the same latent
    stochastic trajectory across LOW / MODERATE / HIGH, while each
    warming library retains its own DP21 temporal correlation structure.
    """

    libraries = {}
    coupling_metadata = {}
    n = None

    for role in ROLE_ORDER:

        sample_path = ROLE_FILES[
            role
        ][
            "samples"
        ]

        metadata_path = ROLE_FILES[
            role
        ][
            "metadata"
        ]

        for path in [
            sample_path,
            metadata_path,
        ]:

            if not path.exists():
                raise FileNotFoundError(
                    f"{role} input not found:\n{path}"
                )

        df = pd.read_csv(
            sample_path
        )

        meta = pd.read_csv(
            metadata_path
        )

        if "trajectory_id" not in df.columns:
            raise KeyError(
                f"'trajectory_id' missing from {role} library."
            )

        required_meta = {
            "trajectory_id",
            "projection_year",
            "latent_percentile_rank",
            "source_sample_id",
            "selected_total_ais",
        }

        missing_meta = (
            required_meta
            - set(
                meta.columns
            )
        )

        if missing_meta:
            raise KeyError(
                f"{role} coupling metadata is missing:\n"
                + "\n".join(
                    sorted(
                        missing_meta
                    )
                )
            )

        if n is None:
            n = len(
                df
            )
        elif len(
            df
        ) != n:
            raise ValueError(
                "LOW/MODERATE/HIGH libraries differ in size."
            )

        expected_ids = np.arange(
            n,
            dtype=int,
        )

        if not np.array_equal(
            df[
                "trajectory_id"
            ].to_numpy(
                dtype=int
            ),
            expected_ids,
        ):
            raise ValueError(
                f"{role} trajectory_id must be 0,...,N-1."
            )

        libraries[
            role
        ] = df

        coupling_metadata[
            role
        ] = meta

    return (
        libraries,
        coupling_metadata,
        n,
    )


# ============================================================
# 6. FAIR -> B25 TRAJECTORY PAIRING
# ============================================================

def create_pairing(
    n_fair: int,
    n_library: int,
) -> pd.DataFrame:

    rng = np.random.default_rng(
        PAIRING_SEED
    )

    if n_fair == n_library:

        ids = rng.permutation(
            n_library
        )

        replacement = False

    else:

        ids = rng.choice(
            n_library,
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
            "trajectory_id": ids.astype(
                int
            ),
            "sampled_with_replacement": replacement,
        }
    )


# ============================================================
# 7. CATEGORICAL CLIMATE-SPACE SELECTION
# ============================================================

def select_categories(
    probability_df: pd.DataFrame,
    n_fair: int,
) -> pd.DataFrame:
    """
    Use one persistent U(0,1) value per FaIR realization.

    The climate-space probabilities change by year, so the category
    can change through time without introducing three independent
    random selection shocks.
    """

    rng = np.random.default_rng(
        CATEGORY_SEED
    )

    u_category = rng.random(
        n_fair
    )

    frames = []

    for year in PROJECTION_YEARS:

        year_df = (
            probability_df.loc[
                probability_df[
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

        if len(
            year_df
        ) != n_fair:
            raise ValueError(
                f"Expected {n_fair} climate rows for {year}."
            )

        p_low = year_df[
            "probability_low"
        ].to_numpy(
            dtype=float
        )

        p_mod = year_df[
            "probability_moderate"
        ].to_numpy(
            dtype=float
        )

        threshold_low = p_low
        threshold_mod = (
            p_low
            + p_mod
        )

        selected = np.where(
            u_category
            < threshold_low,
            "LOW",
            np.where(
                u_category
                < threshold_mod,
                "MODERATE",
                "HIGH",
            ),
        )

        frame = year_df.copy()

        frame[
            "category_uniform"
        ] = u_category

        frame[
            "selected_library"
        ] = selected

        frames.append(
            frame
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# 8. ATTACH SELECTED TEMPORAL PERCENTILES
# ============================================================

def attach_selected_temporal_metadata(
    selection_df: pd.DataFrame,
    pairing_df: pd.DataFrame,
    coupling_metadata: dict,
) -> pd.DataFrame:
    """
    For each FaIR sample/year, attach the temporal percentile from the
    ACTUALLY selected LOW / MODERATE / HIGH temporal process.

    Example:
        selected category at 2100 = MODERATE
        -> use MODERATE latent_percentile_rank at 2100

        selected category at 2200 = HIGH
        -> use HIGH latent_percentile_rank at 2200

    The percentile is therefore allowed to change across years.
    """

    selection = selection_df.copy()

    trajectory_ids = (
        pairing_df
        .sort_values(
            "fair_sample_id"
        )[
            "trajectory_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    selection[
        "latent_trajectory_id"
    ] = selection[
        "fair_sample_id"
    ].map(
        pd.Series(
            trajectory_ids,
            index=np.arange(
                len(
                    trajectory_ids
                ),
                dtype=int,
            ),
        )
    ).astype(
        int
    )

    selection[
        "selected_temporal_percentile"
    ] = np.nan

    selection[
        "selected_source_sample_id"
    ] = -1

    selection[
        "selected_total_ais_rank_value"
    ] = np.nan

    for role in ROLE_ORDER:

        role_meta = coupling_metadata[
            role
        ]

        for year in PROJECTION_YEARS:

            mask = (
                (
                    selection[
                        "selected_library"
                    ]
                    == role
                )
                & (
                    selection[
                        "year"
                    ]
                    == year
                )
            )

            if not mask.any():
                continue

            meta_year = (
                role_meta.loc[
                    role_meta[
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

            ids = selection.loc[
                mask,
                "latent_trajectory_id",
            ].to_numpy(
                dtype=int
            )

            selected_meta = meta_year.iloc[
                ids
            ]

            selection.loc[
                mask,
                "selected_temporal_percentile",
            ] = selected_meta[
                "latent_percentile_rank"
            ].to_numpy(
                dtype=float
            )

            selection.loc[
                mask,
                "selected_source_sample_id",
            ] = selected_meta[
                "source_sample_id"
            ].to_numpy(
                dtype=int
            )

            selection.loc[
                mask,
                "selected_total_ais_rank_value",
            ] = selected_meta[
                "selected_total_ais"
            ].to_numpy(
                dtype=float
            )

    if selection[
        "selected_temporal_percentile"
    ].isna().any():
        raise RuntimeError(
            "Some selected temporal percentiles were not assigned."
        )

    return selection


# ============================================================
# 9. BUILD CLIMATE-CONDITIONED B25 ANCHORS
# ============================================================

def build_selected_anchors(
    selection_df: pd.DataFrame,
    pairing_df: pd.DataFrame,
    libraries: dict,
) -> pd.DataFrame:

    n_fair = pairing_df.shape[
        0
    ]

    trajectory_ids = pairing_df[
        "trajectory_id"
    ].to_numpy(
        dtype=int
    )

    metadata = (
        selection_df[
            [
                "fair_sample_id",
                "scenario",
                "scenario_label",
                "original_sample",
            ]
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

    output = metadata.copy()

    # Keep the original downstream-compatible name, but clarify that this
    # is the NEW coupled latent trajectory ID, not an original B25 row ID.
    output[
        "trajectory_id"
    ] = trajectory_ids

    output[
        "latent_trajectory_id"
    ] = trajectory_ids

    for year in PROJECTION_YEARS:

        year_sel = (
            selection_df.loc[
                selection_df[
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

        selected_roles = year_sel[
            "selected_library"
        ].to_numpy()

        # Keep diagnostics in the wide output.
        output[
            f"selected_library_{year}"
        ] = selected_roles

        output[
            f"selected_temporal_percentile_{year}"
        ] = year_sel[
            "selected_temporal_percentile"
        ].to_numpy(
            dtype=float
        )

        output[
            f"selected_source_sample_id_{year}"
        ] = year_sel[
            "selected_source_sample_id"
        ].to_numpy(
            dtype=int
        )

        output[
            f"selected_total_ais_rank_value_{year}"
        ] = year_sel[
            "selected_total_ais_rank_value"
        ].to_numpy(
            dtype=float
        )

        output[
            f"gsat_{year}"
        ] = year_sel[
            "gsat"
        ].to_numpy(
            dtype=float
        )

        output[
            f"cumulative_warming_{year}"
        ] = year_sel[
            "cumulative_warming"
        ].to_numpy(
            dtype=float
        )

        for probability_name in [
            "low",
            "moderate",
            "high",
        ]:

            output[
                f"probability_{probability_name}_{year}"
            ] = year_sel[
                f"probability_{probability_name}"
            ].to_numpy(
                dtype=float
            )

        # Every temporal library contains the complete B25 year panel.
        # We choose the row from exactly ONE selected library.
        example_columns = [
            column
            for column in libraries[
                "LOW"
            ].columns
            if column
            != "trajectory_id"
        ]

        year_code = {
            2100: "_21",
            2200: "_22",
            2300: "_23",
        }[
            year
        ]

        year_columns = [
            column
            for column in example_columns
            if str(
                column
            ).endswith(
                year_code
            )
        ]

        for column in year_columns:

            values = np.empty(
                n_fair,
                dtype=float,
            )

            for role in ROLE_ORDER:

                mask = (
                    selected_roles
                    == role
                )

                if not np.any(
                    mask
                ):
                    continue

                values[
                    mask
                ] = (
                    libraries[
                        role
                    ]
                    .iloc[
                        trajectory_ids[
                            mask
                        ]
                    ][
                        column
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

            output[
                column
            ] = values

    return output


# ============================================================
# 10. DIAGNOSTICS
# ============================================================

def print_diagnostics(
    selection_df: pd.DataFrame,
):

    print(
        "\nMean climate-space probabilities by SSP and year:"
    )

    print(
        selection_df.groupby(
            [
                "scenario_label",
                "year",
            ]
        )[
            [
                "probability_low",
                "probability_moderate",
                "probability_high",
            ]
        ].mean().round(
            4
        )
    )

    print(
        "\nRealized selection fractions by SSP and year:"
    )

    realized = (
        selection_df.groupby(
            [
                "scenario_label",
                "year",
            ]
        )[
            "selected_library"
        ]
        .value_counts(
            normalize=True
        )
        .unstack(
            fill_value=0.0
        )
    )

    print(
        realized.round(
            4
        )
    )

    # Transition frequencies.
    wide = selection_df.pivot(
        index="fair_sample_id",
        columns="year",
        values="selected_library",
    )

    transitions = (
        wide.groupby(
            [
                2100,
                2200,
                2300,
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
        / len(
            wide
        )
    )

    print(
        "\nSpearman correlation of the ACTUALLY USED temporal percentile "
        "across years:"
    )

    percentile_wide = selection_df.pivot(
        index="fair_sample_id",
        columns="year",
        values="selected_temporal_percentile",
    )

    print(
        percentile_wide.corr(
            method="spearman"
        ).round(
            4
        )
    )

    print(
        "\nSelected temporal percentile summary:"
    )

    print(
        selection_df.groupby(
            [
                "scenario_label",
                "year",
            ]
        )[
            "selected_temporal_percentile"
        ].agg(
            [
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        ).round(
            4
        )
    )

    print(
        "\nMost common LOW/MODERATE/HIGH state sequences:"
    )

    print(
        transitions.head(
            15
        ).round(
            4
        )
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "B25 CATEGORICAL CLIMATE-SPACE SELECTION"
    )
    print(
        "=" * 72
    )

    pooled = load_fair()

    fair_features = calculate_fair_features(
        pooled
    )

    probabilities = calculate_climate_probabilities(
        fair_features
    )

    (
        libraries,
        coupling_metadata,
        n_library,
    ) = load_libraries()

    pairing = create_pairing(
        n_fair=pooled.sizes[
            "samples"
        ],
        n_library=n_library,
    )

    selection = select_categories(
        probability_df=probabilities,
        n_fair=pooled.sizes[
            "samples"
        ],
    )

    selection = attach_selected_temporal_metadata(
        selection_df=selection,
        pairing_df=pairing,
        coupling_metadata=coupling_metadata,
    )

    anchors = build_selected_anchors(
        selection_df=selection,
        pairing_df=pairing,
        libraries=libraries,
    )

    print_diagnostics(
        selection
    )

    anchor_file = (
        OUTPUT_DIR
        / "b25_categorical_climate_space_selected_anchors.csv"
    )

    selection_file = (
        OUTPUT_DIR
        / "b25_categorical_climate_space_selection_metadata.csv"
    )

    pairing_file = (
        OUTPUT_DIR
        / "fair_b25_trajectory_pairing.csv"
    )

    anchors.to_csv(
        anchor_file,
        index=False,
    )

    selection.to_csv(
        selection_file,
        index=False,
    )

    pairing.to_csv(
        pairing_file,
        index=False,
    )

    print(
        "\nSaved selected climate-conditioned anchors to:"
    )
    print(
        anchor_file
    )

    return {
        "anchors": anchors,
        "selection": selection,
        "pairing": pairing,
    }


if __name__ == "__main__":

    results = main()
