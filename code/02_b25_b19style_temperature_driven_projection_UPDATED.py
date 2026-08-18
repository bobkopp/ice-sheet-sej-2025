#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_b25_b19style_temperature_driven_projection.py

Apply a Bamber et al. (2019)-style temperature-driven LOW/HIGH
selection to the temporally coupled B25 libraries produced by:

    01_b25_b19style_temporal_coupling.py

====================================================================
WHAT THIS SCRIPT DOES
====================================================================

1. Reads the actual FACTS/FaIR GSAT files used in the earlier
   FaIR-B25 work:

       FACTS2/experiments/<ssp>/output/fair-temperature/gsat.nc

2. Pools the FaIR samples from:
       SSP1-1.9
       SSP1-2.6
       SSP2-4.5
       SSP3-7.0
       SSP5-8.5

3. Keeps the complete FaIR trajectories through 2300 and preserves:
       scenario
       scenario_label
       original_sample

4. For the B19-style climate selector ONLY, computes integrated
   surface temperature over 2000-2099:

       iSAT = sum(GSAT_2000, ..., GSAT_2099)

5. Maps iSAT between the original B19 LOW/HIGH markers:

       LOW  = 142.5 degC yr
       HIGH = 242.5 degC yr

   to obtain:

       p_high = clip(
           (iSAT - LOW_ISAT) / (HIGH_ISAT - LOW_ISAT),
           0,
           1
       )

6. Pairs each FaIR trajectory with one B25 trajectory index.

   If the pooled FaIR ensemble and B25 library have the same number
   of samples, every B25 trajectory is used exactly once via a random
   permutation.

   Otherwise, B25 trajectories are sampled with replacement.

7. Uses one random draw per FaIR trajectory to select either:

       LOW  = temporally coupled B25 2.0 C library
       HIGH = temporally coupled B25 5.0 C library

   The SAME paired B25 trajectory index is used in LOW and HIGH.

8. Saves one complete B25 projection row per FaIR trajectory plus
   metadata describing:
       SSP
       original FaIR sample
       integrated SAT
       p(HIGH)
       LOW/HIGH choice
       paired B25 trajectory

====================================================================
WHY THE SELECTOR STOPS AT 2099 EVEN THOUGH WE LOAD THROUGH 2300
====================================================================

This is intentional.

The original B19 temperature-driven FACTS implementation used
integrated SAT over 2000-2099 to choose between LOW and HIGH.
Using warming through 2200 or 2300 would create a different climate-
conditioning method and would allow future climate information to
affect the 2100 LOW/HIGH assignment.

Therefore:

    FaIR trajectories are loaded through 2300,
    but the B19-style LOW/HIGH selector uses 2000-2099 only.

The temporally coupled B25 LOW/HIGH libraries already contain the
2100, 2200, and 2300 cross-year dependence.

====================================================================
IMPORTANT APPROXIMATION
====================================================================

B19 used LOW/HIGH elicitation only.

Therefore B25 3.5 C is intentionally not used in this baseline.
It will be used later in the new climate-space interpolation method.

Also, the FACTS gsat.nc files are treated the same way as in the
previous FaIR-B25 interpolation workflow: surface_temperature is used
directly as GSAT anomaly. No additional 1850-1900 rebaselining is
performed here.

@author: elizabeth
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from B25data_processing import project_dir


# ============================================================
# 1. SETTINGS
# ============================================================

SEJ_PROJECT_DIR = Path(project_dir)

FACTS_PROJECT_DIR = Path(
    "/Users/elizabeth/FACTS2"
)

FAIR_FILES = {
    "ssp119": (
        FACTS_PROJECT_DIR
        / "experiments"
        / "ssp119"
        / "output"
        / "fair-temperature"
        / "gsat.nc"
    ),
    "ssp126": (
        FACTS_PROJECT_DIR
        / "experiments"
        / "ssp126"
        / "output"
        / "fair-temperature"
        / "gsat.nc"
    ),
    "ssp245": (
        FACTS_PROJECT_DIR
        / "experiments"
        / "ssp245"
        / "output"
        / "fair-temperature"
        / "gsat.nc"
    ),
    "ssp370": (
        FACTS_PROJECT_DIR
        / "experiments"
        / "ssp370"
        / "output"
        / "fair-temperature"
        / "gsat.nc"
    ),
    "ssp585": (
        FACTS_PROJECT_DIR
        / "experiments"
        / "ssp585"
        / "output"
        / "fair-temperature"
        / "gsat.nc"
    ),
}

SCENARIO_LABELS = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

TEMPORAL_DIR = (
    SEJ_PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "01_temporal_coupling"
)

LOW_FILE = (
    TEMPORAL_DIR
    / "b25_b19style_low_2.0C_dp21_rcp26_coupled_samples.csv"
)

HIGH_FILE = (
    TEMPORAL_DIR
    / "b25_b19style_high_5.0C_dp21_rcp85_coupled_samples.csv"
)

OUTPUT_DIR = (
    SEJ_PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "02_temperature_driven_projection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Same seed can be reused for reproducibility.
PAIRING_SEED = 20269723
SELECTION_SEED = 1234

# Keep full FaIR trajectories available through 2300.
FAIR_START_YEAR = 2000
FAIR_END_YEAR = 2300

# B19 climate-selection interval.
SELECTOR_START_YEAR = 2000
SELECTOR_END_YEAR = 2099

# Original B19 integrated-SAT markers.
LOW_ISAT = 142.5
HIGH_ISAT = 242.5


# ============================================================
# 2. LOAD AND POOL FAIR TRAJECTORIES
# ============================================================

def load_fair_ensemble(
    files: dict,
    start_year: int = 2000,
    end_year: int = 2300,
) -> xr.DataArray:
    """
    Read and pool FACTS/FaIR GSAT samples from multiple SSPs.

    Expected variable:
        surface_temperature

    Expected dimensions include:
        samples
        years

    A singleton 'locations' dimension is squeezed if present.

    Returns
    -------
    pooled : xarray.DataArray

        Dimensions:
            samples
            years

        Sample coordinates:
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
            f"\nReading FaIR data for {scenario}:"
            f"\n{path}"
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

            temperature = (
                temperature
                .sel(
                    years=slice(
                        start_year,
                        end_year,
                    )
                )
                .transpose(
                    "samples",
                    "years",
                )
                .load()
            )

        available_years = set(
            temperature["years"]
            .values
            .astype(int)
            .tolist()
        )

        required_years = {
            start_year,
            end_year,
            SELECTOR_START_YEAR,
            SELECTOR_END_YEAR,
        }

        missing = sorted(
            required_years
            - available_years
        )

        if missing:
            raise ValueError(
                f"{scenario} is missing required years: "
                f"{missing}"
            )

        n_samples = temperature.sizes[
            "samples"
        ]

        # Preserve original FaIR sample identifier.
        original_sample = temperature[
            "samples"
        ].values.copy()

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
                    SCENARIO_LABELS[scenario],
                    n_samples,
                ),
            ),
            original_sample=(
                "samples",
                original_sample,
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

    if not np.isfinite(
        pooled.values
    ).all():
        raise ValueError(
            "Non-finite FaIR surface temperatures were found."
        )

    print(
        "\nPooled FaIR ensemble:"
    )
    print(pooled)

    return pooled


# ============================================================
# 3. CALCULATE B19 INTEGRATED SAT
# ============================================================

def calculate_b19_integrated_sat(
    pooled: xr.DataArray,
    start_year: int = 2000,
    end_year: int = 2099,
) -> pd.DataFrame:
    """
    Calculate B19-style integrated SAT for every FaIR trajectory.

    B19 used the annual SUM over 2000-2099, not cumulative
    warming through 2200 or 2300.

    For annual data:
        units = degC yr
    """

    subset = (
        pooled
        .sel(
            years=slice(
                start_year,
                end_year,
            )
        )
        .transpose(
            "samples",
            "years",
        )
    )

    year_values = (
        subset["years"]
        .values
        .astype(int)
    )

    expected_years = np.arange(
        start_year,
        end_year + 1,
        dtype=int,
    )

    if not np.array_equal(
        year_values,
        expected_years,
    ):
        raise ValueError(
            "B19 selector requires annual FaIR data for every "
            f"year from {start_year} through {end_year}."
        )

    # Literal B19 FACTS operation:
    #     iSAT = SAT2.sum(axis=0)
    #
    # Our array is samples x years, so sum over years.
    iSAT = subset.sum(
        dim="years"
    ).values.astype(float)

    p_high = (
        (iSAT - LOW_ISAT)
        / (HIGH_ISAT - LOW_ISAT)
    )

    p_high = np.clip(
        p_high,
        0.0,
        1.0,
    )

    climate_df = pd.DataFrame(
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
            "integrated_sat_2000_2099": iSAT,
            "probability_high": p_high,
        }
    )

    return climate_df


# ============================================================
# 4. OPTIONAL CLIMATE DIAGNOSTICS THROUGH 2300
# ============================================================

def calculate_fair_projection_features(
    pooled: xr.DataArray,
    projection_years=(2100, 2200, 2300),
    start_year: int = 2000,
) -> pd.DataFrame:
    """
    Save descriptive FaIR climate information through 2300.

    IMPORTANT:
    These values are diagnostics only.
    They do NOT affect the B19 LOW/HIGH selection.

    For each FaIR sample and projection year, calculate:
        1. GSAT at that year
        2. cumulative warming from 2000 through that year
           using trapezoidal integration

    This is retained because it was useful in the previous
    FaIR-B25 interpolation analysis.
    """

    frames = []

    for projection_year in projection_years:

        subset = (
            pooled
            .sel(
                years=slice(
                    start_year,
                    projection_year,
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

        gsat = (
            pooled
            .sel(
                years=projection_year
            )
            .values
            .astype(float)
        )

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
                    "projection_year": projection_year,
                    "gsat": gsat,
                    "cumulative_warming_2000_to_year": cumulative,
                }
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# 5. LOAD TEMPORALLY COUPLED B25 LOW/HIGH LIBRARIES
# ============================================================

def load_temporal_libraries(
    low_file: Path,
    high_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not low_file.exists():
        raise FileNotFoundError(
            "LOW temporal-coupling file not found:\n"
            f"{low_file}"
        )

    if not high_file.exists():
        raise FileNotFoundError(
            "HIGH temporal-coupling file not found:\n"
            f"{high_file}"
        )

    low = pd.read_csv(
        low_file
    )

    high = pd.read_csv(
        high_file
    )

    if len(low) != len(high):
        raise ValueError(
            "LOW and HIGH libraries contain different "
            f"numbers of trajectories: "
            f"{len(low)} vs {len(high)}"
        )

    if list(low.columns) != list(high.columns):
        raise ValueError(
            "LOW and HIGH temporal libraries do not "
            "contain identical columns."
        )

    if "trajectory_id" not in low.columns:
        raise KeyError(
            "'trajectory_id' is missing from LOW/HIGH "
            "temporal libraries."
        )

    expected_ids = np.arange(
        len(low),
        dtype=int,
    )

    if not np.array_equal(
        low["trajectory_id"]
        .to_numpy(dtype=int),
        expected_ids,
    ):
        raise ValueError(
            "LOW trajectory_id must be 0,...,N-1."
        )

    if not np.array_equal(
        high["trajectory_id"]
        .to_numpy(dtype=int),
        expected_ids,
    ):
        raise ValueError(
            "HIGH trajectory_id must be 0,...,N-1."
        )

    return low, high


# ============================================================
# 6. PAIR FAIR TRAJECTORIES WITH B25 TRAJECTORIES
# ============================================================

def create_b25_pairing(
    n_fair: int,
    n_b25: int,
    seed: int,
) -> pd.DataFrame:
    """
    Pair each pooled FaIR trajectory with one B25 trajectory index.

    If n_fair == n_b25:
        random permutation
        every B25 row used exactly once

    Else:
        sample B25 rows with replacement
    """

    rng = np.random.default_rng(
        seed
    )

    if n_fair == n_b25:

        b25_indices = rng.permutation(
            n_b25
        )

        replacement = False

    else:

        print(
            "\nWARNING: FaIR and B25 sample counts differ."
        )
        print(
            f"FaIR samples: {n_fair:,}"
        )
        print(
            f"B25 samples:  {n_b25:,}"
        )
        print(
            "B25 trajectories will be sampled with replacement."
        )

        b25_indices = rng.choice(
            n_b25,
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
            "b25_trajectory_id": b25_indices.astype(int),
            "sampled_with_replacement": replacement,
        }
    )


# ============================================================
# 7. SELECT LOW OR HIGH COMPLETE B25 TRAJECTORY
# ============================================================

def build_temperature_driven_projection(
    low: pd.DataFrame,
    high: pd.DataFrame,
    climate_df: pd.DataFrame,
    pairing_df: pd.DataFrame,
    selection_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each FaIR trajectory:

        paired B25 trajectory index
                +
        B19 probability of HIGH
                ->
        select complete LOW or HIGH B25 trajectory

    The LOW/HIGH decision applies to the entire B25 trajectory.
    It is NOT made separately by projection year.
    """

    n_fair = len(
        climate_df
    )

    if len(pairing_df) != n_fair:
        raise ValueError(
            "Climate table and pairing table differ in length."
        )

    if not np.array_equal(
        climate_df[
            "fair_sample_id"
        ].to_numpy(dtype=int),
        pairing_df[
            "fair_sample_id"
        ].to_numpy(dtype=int),
    ):
        raise ValueError(
            "FaIR sample IDs do not align between climate "
            "features and B25 pairing."
        )

    b25_indices = pairing_df[
        "b25_trajectory_id"
    ].to_numpy(dtype=int)

    low_selected = (
        low
        .iloc[b25_indices]
        .reset_index(drop=True)
    )

    high_selected = (
        high
        .iloc[b25_indices]
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(
        selection_seed
    )

    uniform_selector = rng.random(
        n_fair
    )

    p_high = climate_df[
        "probability_high"
    ].to_numpy(dtype=float)

    use_high = (
        uniform_selector
        < p_high
    )

    scientific_columns = [
        col
        for col in low.columns
        if col != "trajectory_id"
    ]

    projected = low_selected.loc[
        :,
        scientific_columns,
    ].copy()

    projected.loc[
        use_high,
        scientific_columns,
    ] = (
        high_selected.loc[
            use_high,
            scientific_columns,
        ]
        .to_numpy()
    )

    projected.insert(
        0,
        "fair_sample_id",
        climate_df[
            "fair_sample_id"
        ].to_numpy(dtype=int),
    )

    projected.insert(
        1,
        "scenario",
        climate_df[
            "scenario"
        ].to_numpy(),
    )

    projected.insert(
        2,
        "scenario_label",
        climate_df[
            "scenario_label"
        ].to_numpy(),
    )

    projected.insert(
        3,
        "original_fair_sample",
        climate_df[
            "original_sample"
        ].to_numpy(),
    )

    metadata = climate_df.copy()

    metadata[
        "uniform_selector"
    ] = uniform_selector

    metadata[
        "use_high"
    ] = use_high

    metadata[
        "selected_library"
    ] = np.where(
        use_high,
        "HIGH_5C",
        "LOW_2C",
    )

    metadata[
        "b25_trajectory_id"
    ] = b25_indices

    metadata[
        "sampled_with_replacement"
    ] = pairing_df[
        "sampled_with_replacement"
    ].to_numpy()

    return (
        projected,
        metadata,
    )


# ============================================================
# 8. SUMMARY
# ============================================================

def print_summary(
    metadata: pd.DataFrame,
):

    print(
        "\nB19 integrated SAT summary by SSP:"
    )

    isat_summary = (
        metadata
        .groupby(
            "scenario_label"
        )[
            "integrated_sat_2000_2099"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
        .round(3)
    )

    print(
        isat_summary
    )

    print(
        "\nMean B19 p(HIGH) by SSP:"
    )

    probability_summary = (
        metadata
        .groupby(
            "scenario_label"
        )[
            "probability_high"
        ]
        .mean()
        .round(4)
    )

    print(
        probability_summary
    )

    print(
        "\nRealized fraction selected HIGH by SSP:"
    )

    selected_summary = (
        metadata
        .groupby(
            "scenario_label"
        )[
            "use_high"
        ]
        .mean()
        .round(4)
    )

    print(
        selected_summary
    )

    print(
        "\nOverall fraction selected HIGH:"
    )
    print(
        round(
            metadata[
                "use_high"
            ].mean(),
            4,
        )
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "B25 / DP21: B19-STYLE TEMPERATURE-DRIVEN PROJECTION"
    )
    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # A. Load all FaIR trajectories through 2300
    # --------------------------------------------------------

    pooled = load_fair_ensemble(
        files=FAIR_FILES,
        start_year=FAIR_START_YEAR,
        end_year=FAIR_END_YEAR,
    )

    # --------------------------------------------------------
    # B. Calculate literal B19 2000-2099 integrated SAT
    # --------------------------------------------------------

    climate_df = calculate_b19_integrated_sat(
        pooled=pooled,
        start_year=SELECTOR_START_YEAR,
        end_year=SELECTOR_END_YEAR,
    )

    # --------------------------------------------------------
    # C. Also save climate diagnostics through 2300
    #    These do NOT affect LOW/HIGH selection.
    # --------------------------------------------------------

    projection_features = calculate_fair_projection_features(
        pooled=pooled,
        projection_years=(
            2100,
            2200,
            2300,
        ),
        start_year=FAIR_START_YEAR,
    )

    # --------------------------------------------------------
    # D. Load temporally coupled B25 LOW/HIGH libraries
    # --------------------------------------------------------

    low, high = load_temporal_libraries(
        low_file=LOW_FILE,
        high_file=HIGH_FILE,
    )

    print(
        f"\nPooled FaIR samples: "
        f"{pooled.sizes['samples']:,}"
    )

    print(
        f"Available B25 trajectories: "
        f"{len(low):,}"
    )

    # --------------------------------------------------------
    # E. Pair one B25 trajectory with each FaIR trajectory
    # --------------------------------------------------------

    pairing_df = create_b25_pairing(
        n_fair=pooled.sizes[
            "samples"
        ],
        n_b25=len(low),
        seed=PAIRING_SEED,
    )

    # --------------------------------------------------------
    # F. Apply B19 LOW/HIGH selector
    # --------------------------------------------------------

    projected, metadata = (
        build_temperature_driven_projection(
            low=low,
            high=high,
            climate_df=climate_df,
            pairing_df=pairing_df,
            selection_seed=SELECTION_SEED,
        )
    )

    # --------------------------------------------------------
    # G. Summary
    # --------------------------------------------------------

    print_summary(
        metadata
    )

    # --------------------------------------------------------
    # H. Save
    # --------------------------------------------------------

    projection_file = (
        OUTPUT_DIR
        / "b25_b19style_temperature_driven_samples.csv"
    )

    selection_file = (
        OUTPUT_DIR
        / "b25_b19style_temperature_selection_metadata.csv"
    )

    fair_feature_file = (
        OUTPUT_DIR
        / "fair_projection_features_through_2300.csv"
    )

    pairing_file = (
        OUTPUT_DIR
        / "fair_b25_pairing.csv"
    )

    projected.to_csv(
        projection_file,
        index=False,
    )

    metadata.to_csv(
        selection_file,
        index=False,
    )

    projection_features.to_csv(
        fair_feature_file,
        index=False,
    )

    pairing_df.to_csv(
        pairing_file,
        index=False,
    )

    print(
        "\nSaved temperature-driven B25 projections to:"
    )
    print(
        projection_file
    )

    print(
        "\nSaved B19 selection metadata to:"
    )
    print(
        selection_file
    )

    print(
        "\nSaved FaIR climate diagnostics through 2300 to:"
    )
    print(
        fair_feature_file
    )

    print(
        "\nSaved FaIR-B25 pairing to:"
    )
    print(
        pairing_file
    )

    print(
        "\nDone."
    )

    return {
        "pooled_fair": pooled,
        "climate_selector": climate_df,
        "projection_features": projection_features,
        "pairing": pairing_df,
        "projected": projected,
        "selection_metadata": metadata,
    }


if __name__ == "__main__":

    results = main()
