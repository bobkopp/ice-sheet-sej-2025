#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
======================================================================
FAIR -> SEJ2025 WARMING-SCENARIO SELECTION — CORE METHOD
======================================================================

Final core-method choice
------------------------
For every FaIR realization, select ONE SEJ2025 warming scenario
(LOW / INTERMEDIATE / HIGH) for the full ice-sheet trajectory using:

    Range-normalized checkpoint Euclidean distance
    + inverse-distance weighting with p = 1
    + two-checkpoint endpoint truncation.

Checkpoint years
----------------
    2100, 2200

For scenario s and checkpoint k:

    r_s,k = (C_k - R_s,k) / (H_k - L_k)

    d_s = sqrt( sum_k r_s,k^2 )

where:
    C_k     = FaIR realization cumulative warming at checkpoint k
    R_s,k   = cumulative warming of SEJ reference scenario s at k
    H_k-L_k = LOW-HIGH reference cumulative-warming range at k

Before truncation, scenario probabilities are:

    P(s) = (1 / d_s) / sum_j (1 / d_j)

Two-checkpoint truncation rule
------------------------------
Force LOW only if the realization is at or below the LOW reference at BOTH
checkpoints:

    C2100 <= LOW C2100  AND  C2200 <= LOW C2200

Force HIGH only if the realization is at or above the HIGH reference at BOTH
checkpoints:

    C2100 >= HIGH C2100  AND  C2200 >= HIGH C2200

All other trajectories retain the probabilistic IDW p=1 weights, including
trajectories that are outside the elicited range at only one checkpoint.

One scenario is drawn per FaIR realization and is then used downstream for
all SEJ2025 ice-sheet endpoints (2100 / 2200 / 2300).

Core-chain output
-----------------
This script writes the canonical file expected by downstream scripts:

    output/CORE_METHOD_CHAIN_EMULANDICE2050_TENSION_P003_FAIR2/01_warming_selection/fair_b25_baseline_scenario_selection.csv

The output explicitly records the normalized position and outside-range flag at
both 2100 and 2200, the raw IDW weights, the effective post-truncation weights,
and the selection rule used for each realization.

For backward compatibility, reference_range_position is retained as the mean
of the normalized 2100 and 2200 positions. It is a descriptive summary only;
truncation is determined from BOTH checkpoint-specific positions.
======================================================================

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""

from pathlib import Path
import json

import core_config as core_cfg

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. SETTINGS
# ============================================================

SCRIPT_NAME = "fair_to_sej_scenario_selection"

HOME = core_cfg.HOME
PROJECT_DIR = core_cfg.PROJECT_DIR
DATA_DIR = core_cfg.DATA_DIR
OUTPUT_DIR = core_cfg.SELECTION_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Includes SSP5-3.4 overshoot (ssp534) in addition to the previous five CMIP6 paths.
FAIR_FILES = dict(core_cfg.CMIP6_FAIR_FILES)

START_YEAR = 2000
END_YEAR = 2300
YEARS = np.arange(START_YEAR, END_YEAR + 1, dtype=int)

SCENARIOS = ["LOW", "INTERMEDIATE", "HIGH"]
LOW_INDEX = SCENARIOS.index("LOW")
INT_INDEX = SCENARIOS.index("INTERMEDIATE")
HIGH_INDEX = SCENARIOS.index("HIGH")

# Final checkpoint set for warming-scenario matching.
CHECKPOINT_YEARS = [2100, 2200]
TRUNCATION_YEARS = [2100, 2200]
IDW_POWER = 1.0

# Keep the same seed used in the method-sensitivity experiments so this
# production implementation is exactly reproducible.
RNG_SEED = 8172026

SCALE_EPS = 1e-12
EXACT_DISTANCE_ATOL = 1e-12

# Canonical outputs used by the core method chain.
SELECTION_FILE = OUTPUT_DIR / "fair_b25_baseline_scenario_selection.csv"
MEAN_WEIGHTS_FILE = OUTPUT_DIR / "fair_b25_baseline_mean_weights_by_ssp.csv"
SELECTION_FRACTIONS_FILE = OUTPUT_DIR / "fair_b25_baseline_selection_fractions_by_ssp.csv"
RANGE_FRACTIONS_FILE = OUTPUT_DIR / "fair_b25_reference_range_fractions_by_ssp.csv"
METHOD_METADATA_FILE = OUTPUT_DIR / "fair_b25_scenario_selection_method_metadata.csv"
REFERENCE_CHECKPOINT_FILE = OUTPUT_DIR / "fair_b25_scenario_selection_reference_checkpoints.csv"


# ============================================================
# 2. SEJ2025 REFERENCE WARMING TRAJECTORIES
# ============================================================

REFERENCE_YEARS = np.array([2000, 2020, 2050, 2100, 2200, 2300], dtype=int)

REFERENCE_WARMING = {
    "LOW": np.array([0.7, 1.2, 1.7, 2.0, 2.0, 2.0], dtype=float),
    "INTERMEDIATE": np.array([0.7, 1.2, 2.0, 3.5, 3.5, 3.5], dtype=float),
    "HIGH": np.array([0.7, 1.2, 2.5, 5.0, 5.0, 5.0], dtype=float),
}


# ============================================================
# 3. HELPERS
# ============================================================

def cumulative_trapezoid_annual(warming):
    """Annual trapezoidal integral of warming [K] -> cumulative warming [K yr]."""
    warming = np.asarray(warming, dtype=float)
    cumulative = np.zeros(len(warming), dtype=float)
    cumulative[1:] = np.cumsum(0.5 * (warming[:-1] + warming[1:]))
    return cumulative


def cumulative_from_temperature_matrix(temperature):
    """Vectorized annual trapezoidal cumulative warming for N x year matrix."""
    temperature = np.asarray(temperature, dtype=np.float64)
    cumulative = np.zeros_like(temperature, dtype=np.float64)
    cumulative[:, 1:] = np.cumsum(
        0.5 * (temperature[:, :-1] + temperature[:, 1:]),
        axis=1,
    )
    return cumulative


def year_index(year):
    matches = np.where(YEARS == int(year))[0]
    if len(matches) != 1:
        raise ValueError(f"Could not find unique year {year} in {START_YEAR}-{END_YEAR}.")
    return int(matches[0])


def read_fair_temperature(filepath, ssp):
    """Read one SSP's global-mean FaIR GSAT matrix for 2000-2300."""
    if not filepath.exists():
        raise FileNotFoundError(f"FaIR file not found for {ssp}:\n{filepath}")

    with xr.open_dataset(filepath) as ds:
        da = (
            ds["surface_temperature"]
            .sel(years=slice(START_YEAR, END_YEAR))
            .sel(locations=-1)
        )

        fair_years = da["years"].values.astype(int)
        if not np.array_equal(fair_years, YEARS):
            raise ValueError(
                f"{ssp}: FaIR year grid does not exactly match "
                f"{START_YEAR}-{END_YEAR}."
            )

        temperature = da.values.astype(np.float64)
        sample_ids = da["samples"].values.astype(int)

    if np.isnan(temperature).any():
        raise ValueError(f"{ssp}: FaIR temperature matrix contains missing values.")

    return temperature, sample_ids


def inverse_distance_weights_p1(distances):
    """
    Convert N x 3 distances to IDW p=1 probabilities.

    Exact distance-zero matches receive all probability mass. If more than
    one scenario were ever exactly matched, probability is split equally.
    """
    distances = np.asarray(distances, dtype=float)

    if distances.ndim != 2 or distances.shape[1] != len(SCENARIOS):
        raise ValueError("distances must have shape N x 3.")
    if (~np.isfinite(distances)).any():
        raise ValueError("Non-finite distance encountered.")
    if (distances < 0).any():
        raise ValueError("Negative distance encountered.")

    weights = np.zeros_like(distances, dtype=float)

    exact_mask = np.isclose(
        distances,
        0.0,
        rtol=0.0,
        atol=EXACT_DISTANCE_ATOL,
    )
    exact_rows = exact_mask.any(axis=1)
    normal_rows = ~exact_rows

    if normal_rows.any():
        inv = 1.0 / distances[normal_rows, :]
        weights[normal_rows, :] = inv / inv.sum(axis=1, keepdims=True)

    for i in np.where(exact_rows)[0]:
        exact = exact_mask[i, :]
        weights[i, exact] = 1.0 / exact.sum()

    return weights


def draw_scenarios(weights, uniforms):
    """Draw LOW/INTERMEDIATE/HIGH using one pre-generated U(0,1) per row."""
    weights = np.asarray(weights, dtype=float)
    uniforms = np.asarray(uniforms, dtype=float)

    if len(weights) != len(uniforms):
        raise ValueError("weights and uniforms have different row counts.")

    cdf = np.cumsum(weights, axis=1)
    cdf[:, -1] = 1.0  # protect against floating-point roundoff
    selected_index = (uniforms[:, None] > cdf).sum(axis=1)
    return np.array(SCENARIOS, dtype=object)[selected_index]


def build_reference_trajectories():
    """Return annual reference dataframe and 3 x 301 cumulative matrix."""
    rows = []

    for scenario in SCENARIOS:
        annual_warming = np.interp(
            YEARS,
            REFERENCE_YEARS,
            REFERENCE_WARMING[scenario],
        )
        cumulative = cumulative_trapezoid_annual(annual_warming)

        for year, warming, cum in zip(YEARS, annual_warming, cumulative):
            rows.append(
                {
                    "scenario": scenario,
                    "year": int(year),
                    "warming": float(warming),
                    "cumulative_warming": float(cum),
                }
            )

    reference_df = pd.DataFrame(rows)
    reference_matrix = (
        reference_df
        .pivot(index="scenario", columns="year", values="cumulative_warming")
        .loc[SCENARIOS, YEARS]
        .to_numpy(dtype=float)
    )

    return reference_df, reference_matrix


def range_normalized_checkpoint_distances(cumulative, reference_matrix):
    """
    Range-normalized checkpoint Euclidean distance:

        d_s = sqrt(sum_k [ (C_k - R_s,k) / (H_k - L_k) ]^2)
    """
    indices = np.array([year_index(y) for y in CHECKPOINT_YEARS], dtype=int)

    scales = (
        reference_matrix[HIGH_INDEX, indices]
        - reference_matrix[LOW_INDEX, indices]
    )

    if (~np.isfinite(scales)).any() or (np.abs(scales) <= SCALE_EPS).any():
        raise ValueError(
            f"Invalid LOW-HIGH checkpoint scales for {CHECKPOINT_YEARS}: {scales}"
        )

    residuals = (
        cumulative[:, None, indices]
        - reference_matrix[None, :, indices]
    ) / scales[None, None, :]

    distances = np.sqrt(np.sum(residuals ** 2, axis=2))
    return distances, scales


def probability_sharpness(probabilities):
    """Small set of diagnostics for probabilistically assigned rows."""
    probabilities = np.asarray(probabilities, dtype=float)

    if len(probabilities) == 0:
        return {
            "mean_max_probability": np.nan,
            "mean_normalized_entropy": np.nan,
            "mean_effective_n_scenarios": np.nan,
            "mean_top1_top2_margin": np.nan,
        }

    max_p = probabilities.max(axis=1)
    sorted_p = np.sort(probabilities, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]

    safe = np.clip(probabilities, 1e-300, 1.0)
    entropy = -np.sum(probabilities * np.log(safe), axis=1)

    return {
        "mean_max_probability": float(np.mean(max_p)),
        "mean_normalized_entropy": float(np.mean(entropy / np.log(3.0))),
        "mean_effective_n_scenarios": float(np.mean(np.exp(entropy))),
        "mean_top1_top2_margin": float(np.mean(margin)),
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print(f"Running {SCRIPT_NAME}")
    print("Core method: range-normalized checkpoint Euclidean + IDW p=1")
    print(f"Checkpoints: {CHECKPOINT_YEARS}")
    print(
        "Truncation rule: force LOW/HIGH only when outside the elicited "
        "range in the same direction at BOTH 2100 and 2200."
    )

    reference_df, R = build_reference_trajectories()

    checkpoint_indices = np.array(
        [year_index(y) for y in CHECKPOINT_YEARS],
        dtype=int,
    )
    checkpoint_scales = (
        R[HIGH_INDEX, checkpoint_indices]
        - R[LOW_INDEX, checkpoint_indices]
    )

    # Reference checkpoint report.
    reference_checkpoint_df = (
        reference_df[reference_df["year"].isin(CHECKPOINT_YEARS)]
        .pivot(index="year", columns="scenario", values="cumulative_warming")
        .loc[CHECKPOINT_YEARS, SCENARIOS]
        .reset_index()
    )
    reference_checkpoint_df["low_high_range"] = (
        reference_checkpoint_df["HIGH"] - reference_checkpoint_df["LOW"]
    )
    reference_checkpoint_df.to_csv(REFERENCE_CHECKPOINT_FILE, index=False)

    print("\nReference cumulative warming at final checkpoints [K yr]:")
    print(reference_checkpoint_df.round(2).to_string(index=False))

    idx_2100 = year_index(2100)
    idx_2200 = year_index(2200)

    low_2100 = float(R[LOW_INDEX, idx_2100])
    high_2100 = float(R[HIGH_INDEX, idx_2100])
    low_2200 = float(R[LOW_INDEX, idx_2200])
    high_2200 = float(R[HIGH_INDEX, idx_2200])

    method_metadata = pd.DataFrame(
        [
            {
                "method": (
                    "range_normalized_checkpoint_euclidean_idw_p1_"
                    "double_checkpoint_truncation"
                ),
                "horizon": 2300,
                "checkpoint_years": json.dumps(CHECKPOINT_YEARS),
                "n_checkpoints": len(CHECKPOINT_YEARS),
                "idw_power": IDW_POWER,
                "truncation_years": json.dumps(TRUNCATION_YEARS),
                "truncation_rule": (
                    "LOW iff C2100<=LOW2100 and C2200<=LOW2200; "
                    "HIGH iff C2100>=HIGH2100 and C2200>=HIGH2200; "
                    "otherwise probabilistic IDW p=1"
                ),
                "low_reference_2100_Kyr": low_2100,
                "high_reference_2100_Kyr": high_2100,
                "low_reference_2200_Kyr": low_2200,
                "high_reference_2200_Kyr": high_2200,
                "low_high_checkpoint_ranges_Kyr": json.dumps(
                    checkpoint_scales.tolist()
                ),
                "rng_seed": RNG_SEED,
            }
        ]
    )
    method_metadata.to_csv(METHOD_METADATA_FILE, index=False)

    rng = np.random.default_rng(RNG_SEED)
    all_results = []

    for ssp, filepath in FAIR_FILES.items():
        print("\n" + "=" * 76)
        print(f"PROCESSING {ssp}")
        print("=" * 76)

        temperature, sample_ids = read_fair_temperature(filepath, ssp)
        cumulative = cumulative_from_temperature_matrix(temperature)

        distances, scales = range_normalized_checkpoint_distances(
            cumulative,
            R,
        )

        if not np.allclose(scales, checkpoint_scales):
            raise RuntimeError("Checkpoint scale inconsistency detected.")

        # Probabilities before truncation.
        raw_weights = inverse_distance_weights_p1(distances)

        # Bob's two-checkpoint truncation rule: only force an endpoint scenario
        # if the FaIR realization is outside the elicited range in the SAME
        # direction at BOTH 2100 and 2200. A trajectory outside at only one
        # checkpoint remains probabilistic.
        cumulative_2100 = cumulative[:, idx_2100]
        cumulative_2200 = cumulative[:, idx_2200]

        below_low_2100 = cumulative_2100 <= low_2100
        below_low_2200 = cumulative_2200 <= low_2200
        above_high_2100 = cumulative_2100 >= high_2100
        above_high_2200 = cumulative_2200 >= high_2200

        below_low_both = below_low_2100 & below_low_2200
        above_high_both = above_high_2100 & above_high_2200
        not_double_outside = ~(below_low_both | above_high_both)

        weights = raw_weights.copy()
        weights[below_low_both, :] = np.array([1.0, 0.0, 0.0])
        weights[above_high_both, :] = np.array([0.0, 0.0, 1.0])

        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
            raise RuntimeError(f"{ssp}: effective probabilities do not sum to 1.")
        if (weights < -1e-12).any():
            raise RuntimeError(f"{ssp}: negative scenario probability found.")

        # Same production RNG stream / approach used in the robustness test.
        selection_uniform = rng.random(len(sample_ids))
        selected_scenario = draw_scenarios(weights, selection_uniform)

        if (selected_scenario[below_low_both] != "LOW").any():
            raise RuntimeError(f"{ssp}: truncated LOW assignment failed.")
        if (selected_scenario[above_high_both] != "HIGH").any():
            raise RuntimeError(f"{ssp}: truncated HIGH assignment failed.")

        reference_range_class = np.full(
            len(sample_ids),
            "NOT_DOUBLE_OUTSIDE",
            dtype=object,
        )
        reference_range_class[below_low_both] = "BELOW_LOW_BOTH"
        reference_range_class[above_high_both] = "ABOVE_HIGH_BOTH"

        reference_range_position_2100 = (
            (cumulative_2100 - low_2100)
            / (high_2100 - low_2100)
        )
        reference_range_position_2200 = (
            (cumulative_2200 - low_2200)
            / (high_2200 - low_2200)
        )

        # Legacy summary column retained for downstream compatibility. It is
        # descriptive only; truncation uses the two checkpoint-specific values.
        reference_range_position = 0.5 * (
            reference_range_position_2100 + reference_range_position_2200
        )

        result = pd.DataFrame(
            {
                "ssp": ssp,
                "sample_id": sample_ids,
                "trajectory_id": [f"{ssp}_{sid}" for sid in sample_ids],
                "sampling_method": (
                    "range_normalized_checkpoint_euclidean_idw_p1_"
                    "double_checkpoint_truncation"
                ),
                "horizon": 2300,
                "checkpoint_years": json.dumps(CHECKPOINT_YEARS),
                "n_checkpoints": len(CHECKPOINT_YEARS),
                "distance_metric": "range_normalized_checkpoints",
                "idw_power": IDW_POWER,
                "truncation_years": json.dumps(TRUNCATION_YEARS),
                "truncation_rule": (
                    "outside_same_direction_at_both_2100_and_2200"
                ),
                "cumulative_warming_2100_for_truncation": cumulative_2100,
                "cumulative_warming_2200_for_truncation": cumulative_2200,
                "low_reference_2100": low_2100,
                "high_reference_2100": high_2100,
                "low_reference_2200": low_2200,
                "high_reference_2200": high_2200,
                "below_low_2100": below_low_2100,
                "below_low_2200": below_low_2200,
                "above_high_2100": above_high_2100,
                "above_high_2200": above_high_2200,
                "below_low_both": below_low_both,
                "above_high_both": above_high_both,
                "distance_low": distances[:, LOW_INDEX],
                "distance_intermediate": distances[:, INT_INDEX],
                "distance_high": distances[:, HIGH_INDEX],
                "raw_weight_low": raw_weights[:, LOW_INDEX],
                "raw_weight_intermediate": raw_weights[:, INT_INDEX],
                "raw_weight_high": raw_weights[:, HIGH_INDEX],
                "weight_low": weights[:, LOW_INDEX],
                "weight_intermediate": weights[:, INT_INDEX],
                "weight_high": weights[:, HIGH_INDEX],
                "reference_range_position": reference_range_position,
                "reference_range_position_2100": reference_range_position_2100,
                "reference_range_position_2200": reference_range_position_2200,
                "reference_range_class": reference_range_class,
                "selection_rule": np.where(
                    below_low_both,
                    "TRUNCATED_LOW_BOTH_CHECKPOINTS",
                    np.where(
                        above_high_both,
                        "TRUNCATED_HIGH_BOTH_CHECKPOINTS",
                        "PROBABILISTIC_IDW",
                    ),
                ),
                "selection_uniform": selection_uniform,
                "selected_scenario": selected_scenario,
            }
        )

        # Diagnostics useful downstream and for validation/plotting.
        report_years = [2020, 2050, 2100, 2150, 2200, 2250, 2300]
        for year in report_years:
            idx = year_index(year)
            result[f"warming_{year}"] = temperature[:, idx]
            result[f"cumulative_warming_{year}"] = cumulative[:, idx]

        all_results.append(result)

        selected_fraction = (
            result["selected_scenario"]
            .value_counts(normalize=True)
            .reindex(SCENARIOS, fill_value=0.0)
        )
        print(
            "Selected fractions: "
            f"LOW={selected_fraction['LOW']:.3f}  "
            f"INT={selected_fraction['INTERMEDIATE']:.3f}  "
            f"HIGH={selected_fraction['HIGH']:.3f}"
        )
        print(
            "Double-checkpoint truncation classes: "
            f"below_both={below_low_both.mean():.3f}, "
            f"not_double_outside={not_double_outside.mean():.3f}, "
            f"above_both={above_high_both.mean():.3f}"
        )

    # --------------------------------------------------------
    # Combine all SSPs.
    # --------------------------------------------------------
    selection_df = pd.concat(all_results, ignore_index=True)

    # Mean effective probabilities by SSP.
    mean_weights = (
        selection_df
        .groupby("ssp")[["weight_low", "weight_intermediate", "weight_high"]]
        .mean()
    )

    # Realized selected scenario fractions by SSP.
    selection_fractions = (
        pd.crosstab(
            selection_df["ssp"],
            selection_df["selected_scenario"],
            normalize="index",
        )
        .reindex(columns=SCENARIOS, fill_value=0.0)
    )

    # Two-checkpoint truncation-class fractions by SSP.
    range_fractions = (
        pd.crosstab(
            selection_df["ssp"],
            selection_df["reference_range_class"],
            normalize="index",
        )
        .reindex(
            columns=["BELOW_LOW_BOTH", "NOT_DOUBLE_OUTSIDE", "ABOVE_HIGH_BOTH"],
            fill_value=0.0,
        )
    )

    # --------------------------------------------------------
    # Final validations.
    # --------------------------------------------------------
    required_columns = {
        "ssp",
        "sample_id",
        "weight_low",
        "weight_intermediate",
        "weight_high",
        "reference_range_position",
        "reference_range_position_2100",
        "reference_range_position_2200",
        "reference_range_class",
        "selection_rule",
        "selected_scenario",
    }
    missing = required_columns - set(selection_df.columns)
    if missing:
        raise RuntimeError(f"Missing required downstream columns: {sorted(missing)}")

    if selection_df.duplicated(["ssp", "sample_id"]).any():
        raise RuntimeError("Duplicate (ssp, sample_id) rows found.")

    effective = selection_df[
        ["weight_low", "weight_intermediate", "weight_high"]
    ].to_numpy(dtype=float)
    if not np.allclose(effective.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("Final validation failed: probabilities do not sum to 1.")

    truncated_low = (
        selection_df["selection_rule"] == "TRUNCATED_LOW_BOTH_CHECKPOINTS"
    )
    truncated_high = (
        selection_df["selection_rule"] == "TRUNCATED_HIGH_BOTH_CHECKPOINTS"
    )
    if not (selection_df.loc[truncated_low, "selected_scenario"] == "LOW").all():
        raise RuntimeError(
            "Final validation failed: double-checkpoint LOW truncation not LOW."
        )
    if not (selection_df.loc[truncated_high, "selected_scenario"] == "HIGH").all():
        raise RuntimeError(
            "Final validation failed: double-checkpoint HIGH truncation not HIGH."
        )

    # Probability sharpness among rows where probabilities, rather than
    # double-checkpoint truncation, determine the draw.
    probabilistic = selection_df[
        selection_df["selection_rule"] == "PROBABILISTIC_IDW"
    ]
    sharpness = probability_sharpness(
        probabilistic[
            ["weight_low", "weight_intermediate", "weight_high"]
        ].to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Save canonical production outputs.
    # --------------------------------------------------------
    selection_df.to_csv(SELECTION_FILE, index=False)
    mean_weights.to_csv(MEAN_WEIGHTS_FILE)
    selection_fractions.to_csv(SELECTION_FRACTIONS_FILE)
    range_fractions.to_csv(RANGE_FRACTIONS_FILE)

    print("\n" + "=" * 76)
    print("FINAL CORE-METHOD SUMMARY")
    print("=" * 76)
    print(f"Total FaIR trajectories: {len(selection_df):,}")
    print(f"Probabilistically assigned: {len(probabilistic):,}")
    print(f"Fraction probabilistic: {len(probabilistic) / len(selection_df):.3f}")
    print("\nProbability sharpness among probabilistic rows:")
    for key, value in sharpness.items():
        print(f"  {key}: {value:.4f}")

    print("\nMean effective scenario probabilities by SSP:")
    print(mean_weights.round(3).to_string())

    print("\nRealized selected scenario fractions by SSP:")
    print(selection_fractions.round(3).to_string())

    print("\nSaved canonical selection file:")
    print(SELECTION_FILE)
    print("\nFINAL VALIDATION: PASS")


if __name__ == "__main__":
    main()
