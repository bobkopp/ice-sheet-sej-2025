#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_b25_b19style_temporal_interpolation_HERMITE.py

Generate decadal B25 ice-sheet sea-level contribution trajectories from
2000 through 2300 using a B19-inspired interpolation structure, while
avoiding the climate-scaled 2050 assumption.

======================================================================
METHOD OVERVIEW
======================================================================

For each B25 LOW (2 C) and HIGH (5 C) temporally coupled trajectory,
and separately for EAIS, WAIS, and GrIS:

A. HISTORICAL SEGMENT: 2000 -> 2013.5
--------------------------------------
Use the B19 historical/de-baselined anchors:

    2000   anomaly level = 0
    2010   anomaly level = 0
    2013.5 anomaly level = B19 historical anomaly level

with clamped derivatives:

    derivative at 2000   = B19 baselineanom1990
    derivative at 2013.5 = B19 baselineanom2013

This gives a small historical cubic spline that reproduces the B19
historical anchor information.

B. TRANSITION SEGMENT: 2013.5 -> 2100
--------------------------------------
Use a cubic Hermite interpolation constrained by:

    value at 2013.5 = B19 historical anomaly level
    rate  at 2013.5 = B19 historical anomaly rate

    value at 2100   = actual B25 draw, after removing the B19
                      background linear contribution
    rate  at 2100   = one-sided three-point derivative estimated
                      from that same B25 draw's 2100, 2200, 2300
                      de-baselined values:

        r_2100 = (-3*S2100 + 4*S2200 - S2300) / 200

The missing 2050 value is therefore NOT assumed from climate warming.
It is simply the value of this smooth historical-to-B25 transition
evaluated at 2050.

C. FUTURE SEGMENT: 2100 -> 2300
--------------------------------
Use a clamped cubic spline through the ACTUAL B25 anchors:

    2100
    2200
    2300

with:

    left derivative  = the same local 2100 derivative above

    right derivative = B19-style extrapolated 2300 slope:

        slope_12 = (S2200 - S2100) / 100
        slope_23 = (S2300 - S2200) / 100

        r_2300 = slope_23 + 0.5 * (slope_23 - slope_12)

This preserves every B25 2100/2200/2300 value exactly and prevents the
constructed pre-2100 trajectory from influencing the 2100-2300 spline.

D. RE-ADD B19 BACKGROUND CONTRIBUTION
--------------------------------------
All interpolation above is performed on de-baselined anomaly
contributions. The B19 background linear contribution is then added
back:

    contribution(t) = anomaly(t) + baseline_rate * years_since_2000

======================================================================
IMPORTANT
======================================================================

This is still a B19-inspired approximation because B25 did not elicit
a 2050 ice-sheet contribution. But unlike the previous version, this
script does NOT assume that fractional cumulative warming equals
fractional sea-level response.

The 2050 value is now an internally derived consequence of:
    historical ice-sheet state/rate
    + B25 2100 value
    + B25-implied 2100 rate.

@author: elizabeth
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from B25data_processing import project_dir


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_DIR = Path(project_dir)

TEMPORAL_DIR = (
    PROJECT_DIR
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
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "03_temporal_interpolation_hermite"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_YEARS = np.arange(
    2000,
    2301,
    10,
    dtype=int,
)

YEAR_CODE = {
    2100: "21",
    2200: "22",
    2300: "23",
}

WARMING_CODE = {
    "LOW": "2",
    "HIGH": "5",
}


# ============================================================
# 2. B19 HISTORICAL / BACKGROUND CONSTANTS
# ============================================================

# Original B19 MATLAB:
#
# baseline = [275 0 200 75] / 360
#
# Component order:
# total, EAIS, GrIS, WAIS
#
# Units: mm/yr
BASELINE_RATE_MM_PER_YR = {
    "EAIS": 0.0 / 360.0,
    "GrIS": 200.0 / 360.0,
    "WAIS": 75.0 / 360.0,
}

# Original B19 MATLAB:
#
# baselineanom1990 = [-262 -73 -145 -44] / 360
#
# Units: mm/yr
ANOMALY_RATE_2000_MM_PER_YR = {
    "EAIS": -73.0 / 360.0,
    "GrIS": -145.0 / 360.0,
    "WAIS": -44.0 / 360.0,
}

# Original B19 MATLAB:
#
# baselineanom2013 = [185 -31 100 116] / 360
#
# This quantity is a rate anomaly.
#
# Units: mm/yr
ANOMALY_RATE_2013P5_MM_PER_YR = {
    "EAIS": -31.0 / 360.0,
    "GrIS": 100.0 / 360.0,
    "WAIS": 116.0 / 360.0,
}

# Original B19 MATLAB:
#
# baselinelev2013 = baselineanom2013 * 3
#
# Units: mm
ANOMALY_LEVEL_2013P5_MM = {
    component: rate * 3.0
    for component, rate
    in ANOMALY_RATE_2013P5_MM_PER_YR.items()
}


# ============================================================
# 3. LOAD TEMPORALLY COUPLED LIBRARY
# ============================================================

def load_library(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "trajectory_id" not in df.columns:
        raise KeyError(
            f"'trajectory_id' not found in:\n{path}"
        )

    expected = np.arange(
        len(df),
        dtype=int,
    )

    actual = df[
        "trajectory_id"
    ].to_numpy(dtype=int)

    if not np.array_equal(
        actual,
        expected,
    ):
        raise ValueError(
            "trajectory_id must be consecutive integers "
            "0,...,N-1."
        )

    return df


# ============================================================
# 4. EXTRACT B25 2100 / 2200 / 2300 ANCHORS
# ============================================================

def extract_b25_component_anchors(
    df: pd.DataFrame,
    scenario_role: str,
    component: str,
) -> np.ndarray:

    prefix = {
        "EAIS": "E",
        "WAIS": "W",
        "GrIS": "G",
    }[component]

    warming_code = WARMING_CODE[
        scenario_role
    ]

    columns = [
        (
            f"{prefix}"
            f"{warming_code}_"
            f"{YEAR_CODE[year]}"
        )
        for year in [
            2100,
            2200,
            2300,
        ]
    ]

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "Required B25 columns are missing:\n"
            + "\n".join(missing)
        )

    return df[
        columns
    ].to_numpy(dtype=float)


# ============================================================
# 5. REMOVE B19 BACKGROUND LINEAR CONTRIBUTION
# ============================================================

def debaseline_future_anchors(
    future_values_mm: np.ndarray,
    component: str,
) -> np.ndarray:

    elapsed_years = np.array(
        [
            100.0,
            200.0,
            300.0,
        ]
    )

    background = (
        BASELINE_RATE_MM_PER_YR[
            component
        ]
        * elapsed_years
    )

    return (
        future_values_mm
        - background[
            np.newaxis,
            :
        ]
    )


# ============================================================
# 6. DERIVE LOCAL 2100 RATE FROM B25
# ============================================================

def calculate_2100_local_slope(
    anomaly_2100: np.ndarray,
    anomaly_2200: np.ndarray,
    anomaly_2300: np.ndarray,
) -> np.ndarray:
    """
    One-sided three-point first derivative at 2100 for equally
    spaced 100-year anchors:

        f'(2100) ~ (-3*f2100 + 4*f2200 - f2300) / (2*h)

    where h = 100 years.

    Units:
        mm/yr
    """

    return (
        -3.0 * anomaly_2100
        + 4.0 * anomaly_2200
        - anomaly_2300
    ) / 200.0


# ============================================================
# 7. B19-STYLE 2300 END SLOPE
# ============================================================

def calculate_2300_end_slope(
    anomaly_2100: np.ndarray,
    anomaly_2200: np.ndarray,
    anomaly_2300: np.ndarray,
) -> np.ndarray:

    slope_2100_2200 = (
        anomaly_2200
        - anomaly_2100
    ) / 100.0

    slope_2200_2300 = (
        anomaly_2300
        - anomaly_2200
    ) / 100.0

    return (
        slope_2200_2300
        + 0.5
        * (
            slope_2200_2300
            - slope_2100_2200
        )
    )


# ============================================================
# 8. BUILD HISTORICAL DE-BASELINED SEGMENT
# ============================================================

def build_historical_spline(
    component: str,
) -> CubicSpline:
    """
    Historical B19-inspired spline through:

        2000   anomaly = 0
        2010   anomaly = 0
        2013.5 anomaly = B19 historical anomaly level

    with endpoint slopes from B19 historical anomaly rates.
    """

    x = np.array(
        [
            0.0,
            10.0,
            13.5,
        ]
    )

    y = np.array(
        [
            0.0,
            0.0,
            ANOMALY_LEVEL_2013P5_MM[
                component
            ],
        ]
    )

    return CubicSpline(
        x=x,
        y=y,
        bc_type=(
            (
                1,
                ANOMALY_RATE_2000_MM_PER_YR[
                    component
                ],
            ),
            (
                1,
                ANOMALY_RATE_2013P5_MM_PER_YR[
                    component
                ],
            ),
        ),
    )


# ============================================================
# 9. INTERPOLATE ONE COMPONENT
# ============================================================

def interpolate_component(
    df: pd.DataFrame,
    scenario_role: str,
    component: str,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """
    Generate one complete 2000-2300 decadal contribution trajectory
    for every B25 draw.

    Piecewise anomaly interpolation:

        2000-2013.5:
            historical clamped cubic spline

        2013.5-2100:
            cubic Hermite

        2100-2300:
            clamped cubic spline through B25 anchors
    """

    # --------------------------------------------------------
    # A. B25 anchors
    # --------------------------------------------------------

    future = extract_b25_component_anchors(
        df=df,
        scenario_role=scenario_role,
        component=component,
    )

    anomaly = debaseline_future_anchors(
        future_values_mm=future,
        component=component,
    )

    anomaly_2100 = anomaly[:, 0]
    anomaly_2200 = anomaly[:, 1]
    anomaly_2300 = anomaly[:, 2]

    # --------------------------------------------------------
    # B. Local B25 slope at 2100
    # --------------------------------------------------------

    slope_2100 = calculate_2100_local_slope(
        anomaly_2100=anomaly_2100,
        anomaly_2200=anomaly_2200,
        anomaly_2300=anomaly_2300,
    )

    # --------------------------------------------------------
    # C. B19-style endpoint slope at 2300
    # --------------------------------------------------------

    slope_2300 = calculate_2300_end_slope(
        anomaly_2100=anomaly_2100,
        anomaly_2200=anomaly_2200,
        anomaly_2300=anomaly_2300,
    )

    n_samples = len(df)

    # --------------------------------------------------------
    # D. Historical segment: 2000 -> 2013.5
    # --------------------------------------------------------

    historical_spline = build_historical_spline(
        component=component
    )

    # --------------------------------------------------------
    # E. Hermite transition: 2013.5 -> 2100
    # --------------------------------------------------------

    transition_x = np.array(
        [
            13.5,
            100.0,
        ]
    )

    transition_y = np.vstack(
        [
            np.full(
                n_samples,
                ANOMALY_LEVEL_2013P5_MM[
                    component
                ],
                dtype=float,
            ),
            anomaly_2100,
        ]
    )

    transition_dydx = np.vstack(
        [
            np.full(
                n_samples,
                ANOMALY_RATE_2013P5_MM_PER_YR[
                    component
                ],
                dtype=float,
            ),
            slope_2100,
        ]
    )

    transition_spline = CubicHermiteSpline(
        x=transition_x,
        y=transition_y,
        dydx=transition_dydx,
        axis=0,
    )

    # --------------------------------------------------------
    # F. Future segment: 2100 -> 2300
    # --------------------------------------------------------

    future_x = np.array(
        [
            100.0,
            200.0,
            300.0,
        ]
    )

    future_y = np.column_stack(
        [
            anomaly_2100,
            anomaly_2200,
            anomaly_2300,
        ]
    )

    future_spline = CubicSpline(
        x=future_x,
        y=future_y,
        axis=1,
        bc_type=(
            (
                1,
                slope_2100,
            ),
            (
                1,
                slope_2300,
            ),
        ),
    )

    # --------------------------------------------------------
    # G. Evaluate piecewise interpolation at output decades
    # --------------------------------------------------------

    output_t = (
        OUTPUT_YEARS.astype(float)
        - 2000.0
    )

    anomaly_output = np.empty(
        (
            n_samples,
            len(OUTPUT_YEARS),
        ),
        dtype=float,
    )

    historical_mask = (
        output_t
        <= 13.5
    )

    transition_mask = (
        (output_t > 13.5)
        & (output_t < 100.0)
    )

    future_mask = (
        output_t
        >= 100.0
    )

    # Historical spline is common to all draws.
    historical_values = historical_spline(
        output_t[
            historical_mask
        ]
    )

    anomaly_output[
        :,
        historical_mask,
    ] = historical_values[
        np.newaxis,
        :
    ]

    # Hermite output shape:
    # n_times x n_samples
    transition_values = transition_spline(
        output_t[
            transition_mask
        ]
    )

    anomaly_output[
        :,
        transition_mask,
    ] = transition_values.T

    anomaly_output[
        :,
        future_mask,
    ] = future_spline(
        output_t[
            future_mask
        ]
    )

    # --------------------------------------------------------
    # H. Re-add B19 background contribution
    # --------------------------------------------------------

    contribution_output = (
        anomaly_output
        + BASELINE_RATE_MM_PER_YR[
            component
        ]
        * output_t[
            np.newaxis,
            :
        ]
    )

    # --------------------------------------------------------
    # I. Evaluate the implied 2050 contribution
    # --------------------------------------------------------

    anomaly_2050 = transition_spline(
        50.0
    )

    contribution_2050 = (
        anomaly_2050
        + BASELINE_RATE_MM_PER_YR[
            component
        ]
        * 50.0
    )

    diagnostics = pd.DataFrame(
        {
            "trajectory_id": df[
                "trajectory_id"
            ].to_numpy(dtype=int),
            "scenario_role": scenario_role,
            "component": component,
            "historical_anomaly_level_2013p5_mm": (
                ANOMALY_LEVEL_2013P5_MM[
                    component
                ]
            ),
            "historical_anomaly_rate_2013p5_mm_per_yr": (
                ANOMALY_RATE_2013P5_MM_PER_YR[
                    component
                ]
            ),
            "b25_anomaly_2100_mm": anomaly_2100,
            "b25_anomaly_2200_mm": anomaly_2200,
            "b25_anomaly_2300_mm": anomaly_2300,
            "derived_anomaly_rate_2100_mm_per_yr": slope_2100,
            "b19style_anomaly_rate_2300_mm_per_yr": slope_2300,
            "implied_anomaly_2050_mm": anomaly_2050,
            "implied_total_contribution_2050_mm": (
                contribution_2050
            ),
        }
    )

    return (
        contribution_output,
        diagnostics,
    )


# ============================================================
# 10. RUN ONE LOW/HIGH LIBRARY
# ============================================================

def run_scenario(
    df: pd.DataFrame,
    scenario_role: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    arrays = {}
    diagnostics = []

    for component in [
        "EAIS",
        "WAIS",
        "GrIS",
    ]:

        print(
            f"Interpolating {scenario_role} {component}..."
        )

        values, diag = interpolate_component(
            df=df,
            scenario_role=scenario_role,
            component=component,
        )

        arrays[
            component
        ] = values

        diagnostics.append(
            diag
        )

    # Derived totals.
    arrays[
        "Total AIS"
    ] = (
        arrays["EAIS"]
        + arrays["WAIS"]
    )

    arrays[
        "Total ice-sheet SLR"
    ] = (
        arrays["Total AIS"]
        + arrays["GrIS"]
    )

    records = []

    trajectory_ids = df[
        "trajectory_id"
    ].to_numpy(dtype=int)

    for component, values in arrays.items():

        wide = pd.DataFrame(
            values,
            columns=OUTPUT_YEARS,
        )

        wide.insert(
            0,
            "trajectory_id",
            trajectory_ids,
        )

        long = wide.melt(
            id_vars="trajectory_id",
            var_name="year",
            value_name="value_mm",
        )

        long[
            "year"
        ] = long[
            "year"
        ].astype(int)

        long[
            "scenario_role"
        ] = scenario_role

        long[
            "warming_anchor"
        ] = (
            2.0
            if scenario_role == "LOW"
            else 5.0
        )

        long[
            "component"
        ] = component

        records.append(
            long[
                [
                    "trajectory_id",
                    "scenario_role",
                    "warming_anchor",
                    "year",
                    "component",
                    "value_mm",
                ]
            ]
        )

    output = pd.concat(
        records,
        ignore_index=True,
    )

    diagnostics = pd.concat(
        diagnostics,
        ignore_index=True,
    )

    return (
        output,
        diagnostics,
    )


# ============================================================
# 11. VALIDATE B25 ANCHORS
# ============================================================

def validate_projection_anchors(
    interpolated: pd.DataFrame,
    source_df: pd.DataFrame,
    scenario_role: str,
) -> pd.DataFrame:

    records = []

    warming_code = WARMING_CODE[
        scenario_role
    ]

    prefixes = {
        "EAIS": "E",
        "WAIS": "W",
        "GrIS": "G",
    }

    for component in [
        "EAIS",
        "WAIS",
        "GrIS",
    ]:

        for year in [
            2100,
            2200,
            2300,
        ]:

            source_column = (
                f"{prefixes[component]}"
                f"{warming_code}_"
                f"{YEAR_CODE[year]}"
            )

            source = source_df[
                source_column
            ].to_numpy(dtype=float)

            reconstructed = (
                interpolated.loc[
                    (
                        interpolated[
                            "component"
                        ]
                        == component
                    )
                    & (
                        interpolated[
                            "year"
                        ]
                        == year
                    ),
                    [
                        "trajectory_id",
                        "value_mm",
                    ]
                ]
                .sort_values(
                    "trajectory_id"
                )[
                    "value_mm"
                ]
                .to_numpy(dtype=float)
            )

            difference = np.abs(
                reconstructed
                - source
            )

            records.append(
                {
                    "scenario_role": scenario_role,
                    "component": component,
                    "year": year,
                    "maximum_absolute_difference_mm": (
                        difference.max()
                    ),
                    "allclose": bool(
                        np.allclose(
                            reconstructed,
                            source,
                            rtol=0.0,
                            atol=1e-8,
                        )
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# 12. CREATE QUANTILE SUMMARY
# ============================================================

def create_quantile_summary(
    interpolated: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    quantiles = np.array(
        [
            0.05,
            0.17,
            0.50,
            0.83,
            0.95,
        ]
    )

    for (
        scenario_role,
        component,
        year,
    ), group in interpolated.groupby(
        [
            "scenario_role",
            "component",
            "year",
        ]
    ):

        values = group[
            "value_mm"
        ].to_numpy(dtype=float)

        q = np.quantile(
            values,
            quantiles,
        )

        records.append(
            {
                "scenario_role": scenario_role,
                "component": component,
                "year": year,
                "mean_mm": np.mean(values),
                "p05_mm": q[0],
                "p17_mm": q[1],
                "p50_mm": q[2],
                "p83_mm": q[3],
                "p95_mm": q[4],
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# 13. SHAPE DIAGNOSTICS
# ============================================================

def create_shape_diagnostics(
    interpolated: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simple draw-level diagnostics to flag large reversals between
    consecutive decades.

    This does not automatically declare a trajectory unphysical.
    It identifies trajectories worth inspecting.
    """

    records = []

    for (
        scenario_role,
        component,
    ), group in interpolated.groupby(
        [
            "scenario_role",
            "component",
        ]
    ):

        wide = (
            group
            .pivot(
                index="trajectory_id",
                columns="year",
                values="value_mm",
            )
            .reindex(
                columns=OUTPUT_YEARS
            )
        )

        values = wide.to_numpy(
            dtype=float
        )

        increments = np.diff(
            values,
            axis=1,
        )

        # Number of sign changes in decade-to-decade increments.
        signs = np.sign(
            increments
        )

        sign_changes = np.sum(
            (
                signs[
                    :,
                    1:
                ]
                * signs[
                    :,
                    :-1
                ]
            )
            < 0,
            axis=1,
        )

        records.append(
            pd.DataFrame(
                {
                    "trajectory_id": wide.index.to_numpy(
                        dtype=int
                    ),
                    "scenario_role": scenario_role,
                    "component": component,
                    "minimum_value_mm": np.min(
                        values,
                        axis=1,
                    ),
                    "maximum_value_mm": np.max(
                        values,
                        axis=1,
                    ),
                    "minimum_decadal_change_mm": np.min(
                        increments,
                        axis=1,
                    ),
                    "maximum_decadal_change_mm": np.max(
                        increments,
                        axis=1,
                    ),
                    "number_of_increment_sign_changes": (
                        sign_changes
                    ),
                }
            )
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "B25: B19-INSPIRED HERMITE THROUGH-TIME INTERPOLATION"
    )
    print(
        "=" * 72
    )

    low_df = load_library(
        LOW_FILE
    )

    high_df = load_library(
        HIGH_FILE
    )

    if len(low_df) != len(high_df):
        raise ValueError(
            "LOW and HIGH temporal libraries differ in size."
        )

    all_outputs = []
    all_diagnostics = []
    all_anchor_checks = []

    for scenario_role, source_df in [
        ("LOW", low_df),
        ("HIGH", high_df),
    ]:

        print(
            "\n"
            + "=" * 72
        )
        print(
            f"{scenario_role} INTERPOLATION"
        )
        print(
            "=" * 72
        )

        output, diagnostics = run_scenario(
            df=source_df,
            scenario_role=scenario_role,
        )

        anchor_checks = validate_projection_anchors(
            interpolated=output,
            source_df=source_df,
            scenario_role=scenario_role,
        )

        print(
            "\nAnchor reproduction check:"
        )

        print(
            anchor_checks.to_string(
                index=False
            )
        )

        if not anchor_checks[
            "allclose"
        ].all():
            raise RuntimeError(
                f"{scenario_role} interpolation failed "
                "to reproduce B25 anchors."
            )

        all_outputs.append(
            output
        )

        all_diagnostics.append(
            diagnostics
        )

        all_anchor_checks.append(
            anchor_checks
        )

    interpolated = pd.concat(
        all_outputs,
        ignore_index=True,
    )

    diagnostics = pd.concat(
        all_diagnostics,
        ignore_index=True,
    )

    anchor_checks = pd.concat(
        all_anchor_checks,
        ignore_index=True,
    )

    quantile_summary = create_quantile_summary(
        interpolated
    )

    shape_diagnostics = create_shape_diagnostics(
        interpolated
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    trajectory_file = (
        OUTPUT_DIR
        / "b25_b19style_hermite_decadal_trajectories.csv"
    )

    bridge_file = (
        OUTPUT_DIR
        / "b25_b19style_hermite_2050_diagnostics.csv"
    )

    anchor_file = (
        OUTPUT_DIR
        / "b25_b19style_hermite_anchor_validation.csv"
    )

    quantile_file = (
        OUTPUT_DIR
        / "b25_b19style_hermite_decadal_quantile_summary.csv"
    )

    shape_file = (
        OUTPUT_DIR
        / "b25_b19style_hermite_shape_diagnostics.csv"
    )

    interpolated.to_csv(
        trajectory_file,
        index=False,
    )

    diagnostics.to_csv(
        bridge_file,
        index=False,
    )

    anchor_checks.to_csv(
        anchor_file,
        index=False,
    )

    quantile_summary.to_csv(
        quantile_file,
        index=False,
    )

    shape_diagnostics.to_csv(
        shape_file,
        index=False,
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "INTERPOLATION COMPLETE"
    )
    print(
        "=" * 72
    )

    print(
        "\nSaved decadal trajectories to:"
    )
    print(
        trajectory_file
    )

    print(
        "\nSaved implied 2050 diagnostics to:"
    )
    print(
        bridge_file
    )

    print(
        "\nSaved B25 anchor validation to:"
    )
    print(
        anchor_file
    )

    print(
        "\nSaved decadal quantile summary to:"
    )
    print(
        quantile_file
    )

    print(
        "\nSaved shape diagnostics to:"
    )
    print(
        shape_file
    )

    print(
        "\nMethod note:"
    )
    print(
        "The 2050 contribution is no longer based on cumulative "
        "warming. It is implied by a cubic-Hermite transition from "
        "the B19 historical state/rate to the B25 2100 state/rate."
    )

    return {
        "interpolated": interpolated,
        "diagnostics": diagnostics,
        "anchor_checks": anchor_checks,
        "quantile_summary": quantile_summary,
        "shape_diagnostics": shape_diagnostics,
    }


if __name__ == "__main__":

    results = main()
