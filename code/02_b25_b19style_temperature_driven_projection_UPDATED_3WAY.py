#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_b25_b19style_temperature_driven_projection.py

Extend the Bamber et al. (2019)-style temperature-driven selector from
LOW/HIGH to LOW/MODERATE/HIGH using B25's 2.0 C, 3.5 C, and 5.0 C
structured-expert-judgment anchors.

The selector remains intentionally one-dimensional and B19-like:

    integrated SAT = sum of annual FaIR GSAT over 2000-2099

Markers:
    LOW      = 142.5 degC yr   (original B19 LOW marker)
    MODERATE = 204.5 degC yr   (B25 Intermediate pathway marker)
    HIGH     = 242.5 degC yr   (original B19 HIGH marker)

For iSAT between LOW and MODERATE, probability is split linearly between
LOW and MODERATE. For iSAT between MODERATE and HIGH, probability is split
linearly between MODERATE and HIGH. Values outside the endpoint markers are
clipped to 100% LOW or 100% HIGH.

One categorical draw is made per FaIR trajectory, and that choice selects
ONE COMPLETE temporally coupled B25 trajectory from:

    LOW      -> B25 2.0 C + DP21 RCP2.6 temporal structure
    MODERATE -> B25 3.5 C + DP21 RCP4.5 temporal structure
    HIGH     -> B25 5.0 C + DP21 RCP8.5 temporal structure

The same paired B25 trajectory index is used for all three candidate
libraries before the categorical climate selection.

FaIR trajectories are retained through 2300 for diagnostics, but only
2000-2099 integrated SAT is used for the selector, preserving the B19-style
21st-century climate-conditioning logic.
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

SEJ_PROJECT_DIR = Path(project_dir)
FACTS_PROJECT_DIR = Path("/Users/elizabeth/FACTS2")

FAIR_FILES = {
    "ssp119": FACTS_PROJECT_DIR / "experiments" / "ssp119" / "output" / "fair-temperature" / "gsat.nc",
    "ssp126": FACTS_PROJECT_DIR / "experiments" / "ssp126" / "output" / "fair-temperature" / "gsat.nc",
    "ssp245": FACTS_PROJECT_DIR / "experiments" / "ssp245" / "output" / "fair-temperature" / "gsat.nc",
    "ssp370": FACTS_PROJECT_DIR / "experiments" / "ssp370" / "output" / "fair-temperature" / "gsat.nc",
    "ssp585": FACTS_PROJECT_DIR / "experiments" / "ssp585" / "output" / "fair-temperature" / "gsat.nc",
}

SCENARIO_LABELS = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

TEMPORAL_DIR = SEJ_PROJECT_DIR / "output" / "b19style_baseline" / "01_temporal_coupling"

LIBRARY_FILES = {
    "LOW": TEMPORAL_DIR / "b25_b19style_low_2.0C_dp21_rcp26_coupled_samples.csv",
    "MODERATE": TEMPORAL_DIR / "b25_b19style_moderate_3.5C_dp21_rcp45_coupled_samples.csv",
    "HIGH": TEMPORAL_DIR / "b25_b19style_high_5.0C_dp21_rcp85_coupled_samples.csv",
}

LIBRARY_WARMING = {
    "LOW": 2.0,
    "MODERATE": 3.5,
    "HIGH": 5.0,
}

OUTPUT_DIR = SEJ_PROJECT_DIR / "output" / "b19style_baseline" / "02_temperature_driven_projection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRING_SEED = 20269723
SELECTION_SEED = 1234

FAIR_START_YEAR = 2000
FAIR_END_YEAR = 2300
SELECTOR_START_YEAR = 2000
SELECTOR_END_YEAR = 2099

LOW_ISAT = 142.5
MODERATE_ISAT = 204.5
HIGH_ISAT = 242.5


# ============================================================
# 2. LOAD AND POOL FAIR TRAJECTORIES
# ============================================================

def load_fair_ensemble(files: dict, start_year: int = 2000, end_year: int = 2300) -> xr.DataArray:
    ensembles = []

    for scenario, path in files.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FaIR file not found:\n{path}")

        print(f"\nReading FaIR data for {scenario}:\n{path}")

        with xr.open_dataset(path) as ds:
            if "surface_temperature" not in ds:
                raise KeyError(f"'surface_temperature' was not found in {path}")

            temperature = ds["surface_temperature"]
            if "locations" in temperature.dims:
                temperature = temperature.squeeze("locations", drop=True)

            if not {"samples", "years"}.issubset(temperature.dims):
                raise ValueError(
                    f"{scenario} temperature data must contain 'samples' and 'years'. "
                    f"Found: {temperature.dims}"
                )

            temperature = (
                temperature
                .sel(years=slice(start_year, end_year))
                .transpose("samples", "years")
                .load()
            )

        required_years = {start_year, end_year, SELECTOR_START_YEAR, SELECTOR_END_YEAR}
        available_years = set(temperature["years"].values.astype(int).tolist())
        missing = sorted(required_years - available_years)
        if missing:
            raise ValueError(f"{scenario} is missing required years: {missing}")

        n_samples = temperature.sizes["samples"]
        original_sample = temperature["samples"].values.copy()

        temperature = temperature.assign_coords(
            scenario=("samples", np.repeat(scenario, n_samples)),
            scenario_label=("samples", np.repeat(SCENARIO_LABELS[scenario], n_samples)),
            original_sample=("samples", original_sample),
        )
        ensembles.append(temperature)

    pooled = xr.concat(ensembles, dim="samples")
    pooled = pooled.assign_coords(samples=np.arange(pooled.sizes["samples"], dtype=int))
    pooled.name = "surface_temperature"

    if not np.isfinite(pooled.values).all():
        raise ValueError("Non-finite FaIR surface temperatures were found.")

    print("\nPooled FaIR ensemble:")
    print(pooled)
    return pooled


# ============================================================
# 3. THREE-WAY B19-STYLE INTEGRATED-SAT SELECTOR
# ============================================================

def calculate_three_way_probabilities(iSAT: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise-linear LOW/MODERATE/HIGH probabilities."""
    iSAT = np.asarray(iSAT, dtype=float)

    p_low = np.zeros_like(iSAT)
    p_mod = np.zeros_like(iSAT)
    p_high = np.zeros_like(iSAT)

    low_region = iSAT <= LOW_ISAT
    high_region = iSAT >= HIGH_ISAT
    lower_middle = (iSAT > LOW_ISAT) & (iSAT < MODERATE_ISAT)
    upper_middle = (iSAT >= MODERATE_ISAT) & (iSAT < HIGH_ISAT)

    p_low[low_region] = 1.0
    p_high[high_region] = 1.0

    p_mod[lower_middle] = (
        (iSAT[lower_middle] - LOW_ISAT) / (MODERATE_ISAT - LOW_ISAT)
    )
    p_low[lower_middle] = 1.0 - p_mod[lower_middle]

    p_high[upper_middle] = (
        (iSAT[upper_middle] - MODERATE_ISAT) / (HIGH_ISAT - MODERATE_ISAT)
    )
    p_mod[upper_middle] = 1.0 - p_high[upper_middle]

    # Exact moderate marker should be 100% MODERATE.
    exact_mod = np.isclose(iSAT, MODERATE_ISAT, rtol=0.0, atol=1e-12)
    p_low[exact_mod] = 0.0
    p_mod[exact_mod] = 1.0
    p_high[exact_mod] = 0.0

    total = p_low + p_mod + p_high
    if not np.allclose(total, 1.0, atol=1e-12):
        raise RuntimeError("Three-way climate-selection probabilities do not sum to 1.")

    return p_low, p_mod, p_high


def calculate_b19_integrated_sat(
    pooled: xr.DataArray,
    start_year: int = 2000,
    end_year: int = 2099,
) -> pd.DataFrame:
    subset = pooled.sel(years=slice(start_year, end_year)).transpose("samples", "years")

    expected_years = np.arange(start_year, end_year + 1, dtype=int)
    actual_years = subset["years"].values.astype(int)
    if not np.array_equal(actual_years, expected_years):
        raise ValueError(
            f"Selector requires annual FaIR data for every year {start_year}-{end_year}."
        )

    iSAT = subset.sum(dim="years").values.astype(float)
    p_low, p_mod, p_high = calculate_three_way_probabilities(iSAT)

    return pd.DataFrame(
        {
            "fair_sample_id": pooled["samples"].values.astype(int),
            "scenario": pooled["scenario"].values,
            "scenario_label": pooled["scenario_label"].values,
            "original_sample": pooled["original_sample"].values,
            "integrated_sat_2000_2099": iSAT,
            "probability_low": p_low,
            "probability_moderate": p_mod,
            "probability_high": p_high,
        }
    )


# ============================================================
# 4. OPTIONAL FAIR DIAGNOSTICS THROUGH 2300
# ============================================================

def calculate_fair_projection_features(
    pooled: xr.DataArray,
    projection_years=(2100, 2200, 2300),
    start_year: int = 2000,
) -> pd.DataFrame:
    frames = []

    for projection_year in projection_years:
        subset = pooled.sel(years=slice(start_year, projection_year)).transpose("samples", "years")
        x = subset["years"].values.astype(float)
        y = subset.values

        cumulative = (
            np.trapezoid(y, x=x, axis=1)
            if hasattr(np, "trapezoid")
            else np.trapz(y, x=x, axis=1)
        )

        gsat = pooled.sel(years=projection_year).values.astype(float)

        frames.append(
            pd.DataFrame(
                {
                    "fair_sample_id": pooled["samples"].values.astype(int),
                    "scenario": pooled["scenario"].values,
                    "scenario_label": pooled["scenario_label"].values,
                    "original_sample": pooled["original_sample"].values,
                    "projection_year": projection_year,
                    "gsat": gsat,
                    "cumulative_warming_2000_to_year": cumulative,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


# ============================================================
# 5. LOAD THREE TEMPORALLY COUPLED B25 LIBRARIES
# ============================================================

def load_temporal_libraries(files: dict[str, Path]) -> dict[str, pd.DataFrame]:
    libraries = {}

    for role, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"{role} temporal-coupling file not found:\n{path}")
        libraries[role] = pd.read_csv(path)

    lengths = {role: len(df) for role, df in libraries.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Temporal libraries differ in size: {lengths}")

    column_sets = {role: list(df.columns) for role, df in libraries.items()}
    first_columns = next(iter(column_sets.values()))
    for role, cols in column_sets.items():
        if cols != first_columns:
            raise ValueError(f"{role} temporal library has different columns.")

    n = next(iter(lengths.values()))
    expected_ids = np.arange(n, dtype=int)
    for role, df in libraries.items():
        if "trajectory_id" not in df.columns:
            raise KeyError(f"'trajectory_id' missing from {role} temporal library.")
        if not np.array_equal(df["trajectory_id"].to_numpy(dtype=int), expected_ids):
            raise ValueError(f"{role} trajectory_id must be 0,...,N-1.")

    return libraries


# ============================================================
# 6. PAIR FAIR TRAJECTORIES WITH B25 TRAJECTORIES
# ============================================================

def create_b25_pairing(n_fair: int, n_b25: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if n_fair == n_b25:
        b25_indices = rng.permutation(n_b25)
        replacement = False
    else:
        print("\nWARNING: FaIR and B25 sample counts differ.")
        print(f"FaIR samples: {n_fair:,}")
        print(f"B25 samples:  {n_b25:,}")
        print("B25 trajectories will be sampled with replacement.")
        b25_indices = rng.choice(n_b25, size=n_fair, replace=True)
        replacement = True

    return pd.DataFrame(
        {
            "fair_sample_id": np.arange(n_fair, dtype=int),
            "b25_trajectory_id": b25_indices.astype(int),
            "sampled_with_replacement": replacement,
        }
    )


# ============================================================
# 7. CATEGORICAL LOW / MODERATE / HIGH SELECTION
# ============================================================

def build_temperature_driven_projection(
    libraries: dict[str, pd.DataFrame],
    climate_df: pd.DataFrame,
    pairing_df: pd.DataFrame,
    selection_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_fair = len(climate_df)
    if len(pairing_df) != n_fair:
        raise ValueError("Climate table and pairing table differ in length.")

    if not np.array_equal(
        climate_df["fair_sample_id"].to_numpy(dtype=int),
        pairing_df["fair_sample_id"].to_numpy(dtype=int),
    ):
        raise ValueError("FaIR sample IDs do not align between climate data and pairing.")

    b25_indices = pairing_df["b25_trajectory_id"].to_numpy(dtype=int)
    selected_candidates = {
        role: df.iloc[b25_indices].reset_index(drop=True)
        for role, df in libraries.items()
    }

    probabilities = climate_df[
        ["probability_low", "probability_moderate", "probability_high"]
    ].to_numpy(dtype=float)

    rng = np.random.default_rng(selection_seed)
    uniform_selector = rng.random(n_fair)

    cutoff_low = probabilities[:, 0]
    cutoff_moderate = probabilities[:, 0] + probabilities[:, 1]

    selected_role = np.full(n_fair, "HIGH", dtype=object)
    selected_role[uniform_selector < cutoff_moderate] = "MODERATE"
    selected_role[uniform_selector < cutoff_low] = "LOW"

    scientific_columns = [
        col for col in next(iter(libraries.values())).columns
        if col != "trajectory_id"
    ]

    projected = selected_candidates["LOW"].loc[:, scientific_columns].copy()
    for role in ["MODERATE", "HIGH"]:
        mask = selected_role == role
        projected.loc[mask, scientific_columns] = (
            selected_candidates[role].loc[mask, scientific_columns].to_numpy()
        )

    projected.insert(0, "fair_sample_id", climate_df["fair_sample_id"].to_numpy(dtype=int))
    projected.insert(1, "scenario", climate_df["scenario"].to_numpy())
    projected.insert(2, "scenario_label", climate_df["scenario_label"].to_numpy())
    projected.insert(3, "original_fair_sample", climate_df["original_sample"].to_numpy())
    projected.insert(4, "selected_library", selected_role)
    projected.insert(
        5,
        "selected_warming_anchor",
        np.array([LIBRARY_WARMING[x] for x in selected_role], dtype=float),
    )
    projected.insert(6, "b25_trajectory_id", b25_indices)

    metadata = climate_df.copy()
    metadata["uniform_selector"] = uniform_selector
    metadata["selected_library"] = selected_role
    metadata["selected_warming_anchor"] = np.array(
        [LIBRARY_WARMING[x] for x in selected_role], dtype=float
    )
    metadata["b25_trajectory_id"] = b25_indices
    metadata["sampled_with_replacement"] = pairing_df["sampled_with_replacement"].to_numpy()

    return projected, metadata


# ============================================================
# 8. SUMMARY
# ============================================================

def print_summary(metadata: pd.DataFrame) -> None:
    print("\nIntegrated SAT summary by SSP:")
    print(
        metadata.groupby("scenario_label")["integrated_sat_2000_2099"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(3)
    )

    print("\nMean selection probabilities by SSP:")
    print(
        metadata.groupby("scenario_label")
        [["probability_low", "probability_moderate", "probability_high"]]
        .mean()
        .round(4)
    )

    realized = (
        metadata.groupby(["scenario_label", "selected_library"])
        .size()
        .unstack(fill_value=0)
        .div(metadata.groupby("scenario_label").size(), axis=0)
        .reindex(columns=["LOW", "MODERATE", "HIGH"], fill_value=0.0)
    )

    print("\nRealized selection fractions by SSP:")
    print(realized.round(4))

    print("\nOverall realized selection fractions:")
    print(metadata["selected_library"].value_counts(normalize=True).reindex(["LOW", "MODERATE", "HIGH"]).round(4))


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("=" * 72)
    print("B25 / DP21: THREE-WAY B19-STYLE TEMPERATURE SELECTOR")
    print("=" * 72)

    pooled = load_fair_ensemble(FAIR_FILES, FAIR_START_YEAR, FAIR_END_YEAR)
    climate_df = calculate_b19_integrated_sat(pooled, SELECTOR_START_YEAR, SELECTOR_END_YEAR)
    projection_features = calculate_fair_projection_features(pooled)

    libraries = load_temporal_libraries(LIBRARY_FILES)
    n_b25 = len(next(iter(libraries.values())))

    print(f"\nPooled FaIR samples: {pooled.sizes['samples']:,}")
    print(f"Available B25 trajectories per library: {n_b25:,}")

    pairing_df = create_b25_pairing(
        n_fair=pooled.sizes["samples"],
        n_b25=n_b25,
        seed=PAIRING_SEED,
    )

    projected, metadata = build_temperature_driven_projection(
        libraries=libraries,
        climate_df=climate_df,
        pairing_df=pairing_df,
        selection_seed=SELECTION_SEED,
    )

    print_summary(metadata)

    projection_file = OUTPUT_DIR / "b25_b19style_temperature_driven_samples.csv"
    selection_file = OUTPUT_DIR / "b25_b19style_temperature_selection_metadata.csv"
    fair_feature_file = OUTPUT_DIR / "fair_projection_features_through_2300.csv"
    pairing_file = OUTPUT_DIR / "fair_b25_pairing.csv"

    projected.to_csv(projection_file, index=False)
    metadata.to_csv(selection_file, index=False)
    projection_features.to_csv(fair_feature_file, index=False)
    pairing_df.to_csv(pairing_file, index=False)

    print(f"\nSaved temperature-driven B25 projections to:\n{projection_file}")
    print(f"\nSaved three-way selection metadata to:\n{selection_file}")
    print(f"\nSaved FaIR diagnostics through 2300 to:\n{fair_feature_file}")
    print(f"\nSaved FaIR-B25 pairing to:\n{pairing_file}")
    print("\nDone.")

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
