#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_b25_b19style_temporal_interpolation_PCHIP.py

Interpolate the climate-selected B25 trajectories from Script 02 through time.

This version keeps the previously adopted interpolation method unchanged:

    2000-2013.5 : B19 historical anchoring
    2013.5-2100 : constant-acceleration bridge to the selected B25 2100 anchor
    2100-2300   : shape-preserving PCHIP through selected B25 2100/2200/2300 anchors

The ONLY climate-selection change is inherited from Script 02: each FaIR
trajectory may now select LOW (2 C), MODERATE (3.5 C), or HIGH (5 C).

Importantly, Script 03 now consumes Script 02 directly, so the chain is:

    01 temporal coupling
        -> 02 three-way climate selection
        -> 03 through-time interpolation
        -> 04 summaries/figures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

from B25data_processing import project_dir


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_DIR = Path(project_dir)

PROJECTION_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "02_temperature_driven_projection"
)

PROJECTION_FILE = PROJECTION_DIR / "b25_b19style_temperature_driven_samples.csv"

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "03_temporal_interpolation_pchip"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_YEARS = np.arange(2000, 2301, 10, dtype=int)
YEAR_CODE = {2100: "21", 2200: "22", 2300: "23"}
WARMING_CODE = {2.0: "2", 3.5: "35", 5.0: "5"}


# ============================================================
# 2. B19 HISTORICAL / BACKGROUND CONSTANTS
# ============================================================

BASELINE_RATE_MM_PER_YR = {
    "EAIS": 0.0 / 360.0,
    "GrIS": 200.0 / 360.0,
    "WAIS": 75.0 / 360.0,
}

ANOMALY_RATE_2000_MM_PER_YR = {
    "EAIS": -73.0 / 360.0,
    "GrIS": -145.0 / 360.0,
    "WAIS": -44.0 / 360.0,
}

ANOMALY_RATE_2013P5_MM_PER_YR = {
    "EAIS": -31.0 / 360.0,
    "GrIS": 100.0 / 360.0,
    "WAIS": 116.0 / 360.0,
}

ANOMALY_LEVEL_2013P5_MM = {
    component: rate * 3.0
    for component, rate in ANOMALY_RATE_2013P5_MM_PER_YR.items()
}


# ============================================================
# 3. LOAD CLIMATE-SELECTED B25 PROJECTIONS
# ============================================================

def load_selected_projections(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Script 02 output not found:\n{path}\n\nRun Script 02 first."
        )

    df = pd.read_csv(path)

    required = {
        "fair_sample_id",
        "scenario",
        "scenario_label",
        "original_fair_sample",
        "selected_library",
        "selected_warming_anchor",
        "b25_trajectory_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Script 02 output is missing required metadata columns:\n"
            + "\n".join(sorted(missing))
        )

    allowed = {"LOW": 2.0, "MODERATE": 3.5, "HIGH": 5.0}
    bad_roles = sorted(set(df["selected_library"].astype(str)) - set(allowed))
    if bad_roles:
        raise ValueError(f"Unexpected selected_library values: {bad_roles}")

    expected_warming = df["selected_library"].map(allowed).to_numpy(dtype=float)
    actual_warming = df["selected_warming_anchor"].to_numpy(dtype=float)
    if not np.allclose(actual_warming, expected_warming, rtol=0.0, atol=1e-12):
        raise ValueError("selected_library and selected_warming_anchor are inconsistent.")

    return df


# ============================================================
# 4. EXTRACT EACH ROW'S SELECTED B25 ANCHORS
# ============================================================

def extract_selected_component_anchors(
    df: pd.DataFrame,
    component: str,
) -> np.ndarray:
    prefix = {"EAIS": "E", "WAIS": "W", "GrIS": "G"}[component]
    warming = df["selected_warming_anchor"].to_numpy(dtype=float)

    out = np.empty((len(df), 3), dtype=float)

    for j, year in enumerate([2100, 2200, 2300]):
        yc = YEAR_CODE[year]
        values = np.empty(len(df), dtype=float)

        for warming_value, warming_code in WARMING_CODE.items():
            mask = np.isclose(warming, warming_value)
            column = f"{prefix}{warming_code}_{yc}"
            if column not in df.columns:
                raise KeyError(f"Required selected-anchor column missing: {column}")
            values[mask] = df.loc[mask, column].to_numpy(dtype=float)

        out[:, j] = values

    return out


# ============================================================
# 5. REMOVE B19 BACKGROUND CONTRIBUTION
# ============================================================

def debaseline_future_anchors(
    future_values_mm: np.ndarray,
    component: str,
) -> np.ndarray:
    elapsed_years = np.array([100.0, 200.0, 300.0])
    background = BASELINE_RATE_MM_PER_YR[component] * elapsed_years
    return future_values_mm - background[np.newaxis, :]


# ============================================================
# 6. HISTORICAL SEGMENT
# ============================================================

def build_historical_spline(component: str) -> CubicSpline:
    x = np.array([0.0, 10.0, 13.5])
    y = np.array([0.0, 0.0, ANOMALY_LEVEL_2013P5_MM[component]])

    return CubicSpline(
        x=x,
        y=y,
        bc_type=(
            (1, ANOMALY_RATE_2000_MM_PER_YR[component]),
            (1, ANOMALY_RATE_2013P5_MM_PER_YR[component]),
        ),
    )


# ============================================================
# 7. CONSTANT-ACCELERATION BRIDGE TO 2100
# ============================================================

def calculate_constant_acceleration(
    anomaly_2100: np.ndarray,
    component: str,
) -> np.ndarray:
    dt = 2100.0 - 2013.5
    s0 = ANOMALY_LEVEL_2013P5_MM[component]
    r0 = ANOMALY_RATE_2013P5_MM_PER_YR[component]
    return 2.0 * (anomaly_2100 - s0 - r0 * dt) / (dt ** 2)


def evaluate_constant_acceleration_bridge(
    years: np.ndarray,
    anomaly_2100: np.ndarray,
    component: str,
) -> tuple[np.ndarray, np.ndarray]:
    years = np.asarray(years, dtype=float)
    acceleration = calculate_constant_acceleration(anomaly_2100, component)
    tau = years - 2013.5
    s0 = ANOMALY_LEVEL_2013P5_MM[component]
    r0 = ANOMALY_RATE_2013P5_MM_PER_YR[component]

    values = (
        s0
        + r0 * tau[np.newaxis, :]
        + 0.5 * acceleration[:, np.newaxis] * tau[np.newaxis, :] ** 2
    )
    return values, acceleration


# ============================================================
# 8. PCHIP FUTURE SEGMENT
# ============================================================

def build_future_pchip(
    anomaly_2100: np.ndarray,
    anomaly_2200: np.ndarray,
    anomaly_2300: np.ndarray,
) -> PchipInterpolator:
    x = np.array([100.0, 200.0, 300.0])
    y = np.column_stack([anomaly_2100, anomaly_2200, anomaly_2300])
    return PchipInterpolator(x=x, y=y, axis=1, extrapolate=False)


# ============================================================
# 9. INTERPOLATE ONE COMPONENT
# ============================================================

def interpolate_component(
    df: pd.DataFrame,
    component: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    future = extract_selected_component_anchors(df, component)
    anomaly = debaseline_future_anchors(future, component)

    anomaly_2100 = anomaly[:, 0]
    anomaly_2200 = anomaly[:, 1]
    anomaly_2300 = anomaly[:, 2]

    n_samples = len(df)
    output_t = OUTPUT_YEARS.astype(float) - 2000.0
    anomaly_output = np.empty((n_samples, len(OUTPUT_YEARS)), dtype=float)

    historical_mask = OUTPUT_YEARS <= 2010
    transition_mask = (OUTPUT_YEARS > 2010) & (OUTPUT_YEARS < 2100)
    future_mask = OUTPUT_YEARS >= 2100

    historical_spline = build_historical_spline(component)
    historical_values = historical_spline(output_t[historical_mask])
    anomaly_output[:, historical_mask] = historical_values[np.newaxis, :]

    transition_years = OUTPUT_YEARS[transition_mask].astype(float)
    transition_values, acceleration = evaluate_constant_acceleration_bridge(
        transition_years,
        anomaly_2100,
        component,
    )
    anomaly_output[:, transition_mask] = transition_values

    future_pchip = build_future_pchip(anomaly_2100, anomaly_2200, anomaly_2300)
    anomaly_output[:, future_mask] = future_pchip(output_t[future_mask])

    contribution_output = (
        anomaly_output
        + BASELINE_RATE_MM_PER_YR[component] * output_t[np.newaxis, :]
    )

    anomaly_2050_matrix, _ = evaluate_constant_acceleration_bridge(
        np.array([2050.0]),
        anomaly_2100,
        component,
    )
    anomaly_2050 = anomaly_2050_matrix[:, 0]
    contribution_2050 = anomaly_2050 + BASELINE_RATE_MM_PER_YR[component] * 50.0

    dt = 2100.0 - 2013.5
    implied_rate_2100 = ANOMALY_RATE_2013P5_MM_PER_YR[component] + acceleration * dt

    diagnostics = pd.DataFrame(
        {
            "fair_sample_id": df["fair_sample_id"].to_numpy(dtype=int),
            "scenario": df["scenario"].to_numpy(),
            "scenario_label": df["scenario_label"].to_numpy(),
            "selected_library": df["selected_library"].to_numpy(),
            "selected_warming_anchor": df["selected_warming_anchor"].to_numpy(dtype=float),
            "b25_trajectory_id": df["b25_trajectory_id"].to_numpy(dtype=int),
            "component": component,
            "constant_acceleration_2013p5_2100_mm_per_yr2": acceleration,
            "implied_anomaly_rate_2100_mm_per_yr": implied_rate_2100,
            "implied_anomaly_2050_mm": anomaly_2050,
            "implied_total_contribution_2050_mm": contribution_2050,
        }
    )

    return contribution_output, diagnostics


# ============================================================
# 10. BUILD COMPLETE THROUGH-TIME DATASET
# ============================================================

def run_interpolation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arrays = {}
    diagnostics = []

    for component in ["EAIS", "WAIS", "GrIS"]:
        print(f"Interpolating selected {component} trajectories...")
        values, diag = interpolate_component(df, component)
        arrays[component] = values
        diagnostics.append(diag)

    arrays["Total AIS"] = arrays["EAIS"] + arrays["WAIS"]
    arrays["Total ice-sheet SLR"] = arrays["Total AIS"] + arrays["GrIS"]

    metadata_columns = [
        "fair_sample_id",
        "scenario",
        "scenario_label",
        "original_fair_sample",
        "selected_library",
        "selected_warming_anchor",
        "b25_trajectory_id",
    ]

    records = []
    for component, values in arrays.items():
        wide = pd.DataFrame(values, columns=OUTPUT_YEARS)
        for col in reversed(metadata_columns):
            wide.insert(0, col, df[col].to_numpy())

        long = wide.melt(
            id_vars=metadata_columns,
            var_name="year",
            value_name="value_mm",
        )
        long["year"] = long["year"].astype(int)
        long["component"] = component
        records.append(
            long[
                metadata_columns
                + ["year", "component", "value_mm"]
            ]
        )

    return pd.concat(records, ignore_index=True), pd.concat(diagnostics, ignore_index=True)


# ============================================================
# 11. VALIDATE SELECTED 2100 / 2200 / 2300 ANCHORS
# ============================================================

def validate_projection_anchors(
    interpolated: pd.DataFrame,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for component in ["EAIS", "WAIS", "GrIS"]:
        expected = extract_selected_component_anchors(source_df, component)

        for j, year in enumerate([2100, 2200, 2300]):
            reconstructed = (
                interpolated.loc[
                    (interpolated["component"] == component)
                    & (interpolated["year"] == year),
                    ["fair_sample_id", "value_mm"],
                ]
                .sort_values("fair_sample_id")["value_mm"]
                .to_numpy(dtype=float)
            )

            expected_year = expected[:, j]
            difference = np.abs(reconstructed - expected_year)

            records.append(
                {
                    "component": component,
                    "year": year,
                    "maximum_absolute_difference_mm": difference.max(),
                    "allclose": bool(
                        np.allclose(reconstructed, expected_year, rtol=0.0, atol=1e-8)
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# 12. SUMMARIES / SHAPE DIAGNOSTICS
# ============================================================

def create_quantile_summary(interpolated: pd.DataFrame) -> pd.DataFrame:
    records = []

    for (scenario_label, component, year), group in interpolated.groupby(
        ["scenario_label", "component", "year"]
    ):
        values = group["value_mm"].to_numpy(dtype=float)
        q = np.quantile(values, [0.05, 0.17, 0.50, 0.83, 0.95])
        records.append(
            {
                "scenario_label": scenario_label,
                "component": component,
                "year": year,
                "n_samples": len(values),
                "mean_mm": np.mean(values),
                "p05_mm": q[0],
                "p17_mm": q[1],
                "p50_mm": q[2],
                "p83_mm": q[3],
                "p95_mm": q[4],
            }
        )

    return pd.DataFrame(records)


def create_shape_diagnostics(interpolated: pd.DataFrame) -> pd.DataFrame:
    records = []

    for (scenario_label, component), group in interpolated.groupby(
        ["scenario_label", "component"]
    ):
        wide = (
            group.pivot(index="fair_sample_id", columns="year", values="value_mm")
            .reindex(columns=OUTPUT_YEARS)
        )
        values = wide.to_numpy(dtype=float)
        increments = np.diff(values, axis=1)
        signs = np.sign(increments)
        sign_changes = np.sum((signs[:, 1:] * signs[:, :-1]) < 0, axis=1)

        records.append(
            pd.DataFrame(
                {
                    "fair_sample_id": wide.index.to_numpy(dtype=int),
                    "scenario_label": scenario_label,
                    "component": component,
                    "minimum_value_mm": np.min(values, axis=1),
                    "maximum_value_mm": np.max(values, axis=1),
                    "minimum_decadal_change_mm": np.min(increments, axis=1),
                    "maximum_decadal_change_mm": np.max(increments, axis=1),
                    "number_of_increment_sign_changes": sign_changes,
                }
            )
        )

    return pd.concat(records, ignore_index=True)


# ============================================================
# 13. MAIN
# ============================================================

def main():
    print("=" * 72)
    print("B25: THREE-WAY CLIMATE-SELECTED THROUGH-TIME INTERPOLATION")
    print("=" * 72)

    source_df = load_selected_projections(PROJECTION_FILE)

    print(f"\nClimate-selected FaIR/B25 trajectories: {len(source_df):,}")
    print("\nSelected library counts:")
    print(source_df["selected_library"].value_counts().reindex(["LOW", "MODERATE", "HIGH"]))

    interpolated, diagnostics = run_interpolation(source_df)
    anchor_checks = validate_projection_anchors(interpolated, source_df)

    print("\nAnchor reproduction check:")
    print(anchor_checks.to_string(index=False))

    if not anchor_checks["allclose"].all():
        raise RuntimeError("Interpolation failed to reproduce one or more selected B25 anchors.")

    quantile_summary = create_quantile_summary(interpolated)
    shape_diagnostics = create_shape_diagnostics(interpolated)

    trajectory_file = OUTPUT_DIR / "b25_b19style_pchip_decadal_trajectories.csv"
    bridge_file = OUTPUT_DIR / "b25_b19style_pchip_2050_diagnostics.csv"
    anchor_file = OUTPUT_DIR / "b25_b19style_pchip_anchor_validation.csv"
    quantile_file = OUTPUT_DIR / "b25_b19style_pchip_decadal_quantile_summary.csv"
    shape_file = OUTPUT_DIR / "b25_b19style_pchip_shape_diagnostics.csv"

    interpolated.to_csv(trajectory_file, index=False)
    diagnostics.to_csv(bridge_file, index=False)
    anchor_checks.to_csv(anchor_file, index=False)
    quantile_summary.to_csv(quantile_file, index=False)
    shape_diagnostics.to_csv(shape_file, index=False)

    print(f"\nSaved decadal trajectories to:\n{trajectory_file}")
    print(f"\nSaved 2050 diagnostics to:\n{bridge_file}")
    print(f"\nSaved anchor validation to:\n{anchor_file}")
    print(f"\nSaved decadal quantile summary to:\n{quantile_file}")
    print(f"\nSaved shape diagnostics to:\n{shape_file}")
    print("\nDone.")

    return {
        "interpolated": interpolated,
        "diagnostics": diagnostics,
        "anchor_checks": anchor_checks,
        "quantile_summary": quantile_summary,
        "shape_diagnostics": shape_diagnostics,
    }


if __name__ == "__main__":
    results = main()
