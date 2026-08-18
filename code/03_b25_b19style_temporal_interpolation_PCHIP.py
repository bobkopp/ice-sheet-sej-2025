#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_b25_b19style_temporal_interpolation_PCHIP.py

Generate decadal B25 ice-sheet sea-level contribution trajectories from
2000 through 2300 using:

    1. B19 historical anchoring,
    2. a constant-acceleration bridge from 2013.5 to the B25 2100 anchor,
    3. shape-preserving PCHIP interpolation from 2100 to 2300.

======================================================================
WHY THIS VERSION
======================================================================

B25 does not contain an elicited 2050 ice-sheet contribution.

Earlier approaches either:
    - imposed a 2050 value from cumulative warming, or
    - inferred a 2100 derivative from 2100/2200/2300 and used Hermite
      interpolation before 2100.

Both can introduce strong assumptions or spline artifacts.

This version deliberately avoids both.

======================================================================
A. HISTORICAL SEGMENT: 2000 -> 2013.5
======================================================================

Use the B19 historical/de-baselined anchors:

    2000   anomaly level = 0
    2010   anomaly level = 0
    2013.5 anomaly level = B19 historical anomaly level

with endpoint derivatives:

    derivative at 2000   = B19 baselineanom1990
    derivative at 2013.5 = B19 baselineanom2013

A clamped cubic spline is used only for this short historical segment.

======================================================================
B. TRANSITION SEGMENT: 2013.5 -> 2100
======================================================================

For each B25 draw and each component, assume constant acceleration in
the DE-BASELINED sea-level contribution rate between 2013.5 and 2100.

Known:

    S0 = de-baselined anomaly level at 2013.5
    r0 = de-baselined anomaly rate at 2013.5
    S1 = actual B25 de-baselined contribution at 2100

Let:

    dt = 2100 - 2013.5

Then the constant acceleration required to hit the B25 2100 value is:

    a = 2 * (S1 - S0 - r0*dt) / dt^2

and for any time t between 2013.5 and 2100:

    S(t) = S0 + r0*tau + 0.5*a*tau^2

where:

    tau = t - 2013.5

The implied 2050 contribution is simply S(2050).

Important:
    2200 and 2300 DO NOT influence this pre-2100 trajectory.

======================================================================
C. FUTURE SEGMENT: 2100 -> 2300
======================================================================

Use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) through
the ACTUAL B25 anchors:

    2100
    2200
    2300

PCHIP is shape-preserving and avoids the overshoot/wiggle behavior of
ordinary cubic splines.

No 2100 derivative is imposed.

======================================================================
D. RE-ADD B19 BACKGROUND CONTRIBUTION
======================================================================

Interpolation is performed on the de-baselined anomaly contribution.

The B19 background linear contribution is then added back:

    contribution(t)
        = anomaly(t)
        + baseline_rate * years_since_2000

======================================================================
OUTPUT
======================================================================

Decadal trajectories from 2000 through 2300 for:

    EAIS
    WAIS
    GrIS
    Total AIS = EAIS + WAIS
    Total ice-sheet SLR = EAIS + WAIS + GrIS

for:

    LOW  = B25 2 C temporally coupled library
    HIGH = B25 5 C temporally coupled library

@author: elizabeth
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
    / "03_temporal_interpolation_pchip"
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
# 5. REMOVE B19 BACKGROUND CONTRIBUTION
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
# 6. HISTORICAL SEGMENT
# ============================================================

def build_historical_spline(
    component: str,
) -> CubicSpline:

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
# 7. CONSTANT-ACCELERATION BRIDGE TO 2100
# ============================================================

def calculate_constant_acceleration(
    anomaly_2100: np.ndarray,
    component: str,
) -> np.ndarray:
    """
    Calculate the constant acceleration needed to connect the
    B19 historical state/rate at 2013.5 to each B25 2100 anomaly.

    Units:
        mm/yr^2
    """

    t0 = 2013.5
    t1 = 2100.0

    dt = (
        t1
        - t0
    )

    s0 = (
        ANOMALY_LEVEL_2013P5_MM[
            component
        ]
    )

    r0 = (
        ANOMALY_RATE_2013P5_MM_PER_YR[
            component
        ]
    )

    return (
        2.0
        * (
            anomaly_2100
            - s0
            - r0 * dt
        )
        / (
            dt ** 2
        )
    )


def evaluate_constant_acceleration_bridge(
    years: np.ndarray,
    anomaly_2100: np.ndarray,
    component: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Evaluate the 2013.5->2100 constant-acceleration bridge.

    Returns
    -------
    values
        Shape:
            (n_samples, n_times)

    acceleration
        Shape:
            (n_samples,)
    """

    years = np.asarray(
        years,
        dtype=float,
    )

    acceleration = (
        calculate_constant_acceleration(
            anomaly_2100=anomaly_2100,
            component=component,
        )
    )

    tau = (
        years
        - 2013.5
    )

    s0 = (
        ANOMALY_LEVEL_2013P5_MM[
            component
        ]
    )

    r0 = (
        ANOMALY_RATE_2013P5_MM_PER_YR[
            component
        ]
    )

    values = (
        s0
        + r0
        * tau[
            np.newaxis,
            :
        ]
        + 0.5
        * acceleration[
            :,
            np.newaxis,
        ]
        * (
            tau[
                np.newaxis,
                :
            ] ** 2
        )
    )

    return (
        values,
        acceleration,
    )


# ============================================================
# 8. PCHIP FUTURE SEGMENT
# ============================================================

def build_future_pchip(
    anomaly_2100: np.ndarray,
    anomaly_2200: np.ndarray,
    anomaly_2300: np.ndarray,
) -> PchipInterpolator:
    """
    Shape-preserving interpolation through the actual B25
    2100/2200/2300 anchors.
    """

    x = np.array(
        [
            100.0,
            200.0,
            300.0,
        ]
    )

    y = np.column_stack(
        [
            anomaly_2100,
            anomaly_2200,
            anomaly_2300,
        ]
    )

    return PchipInterpolator(
        x=x,
        y=y,
        axis=1,
        extrapolate=False,
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
    Piecewise interpolation:

        2000-2013.5:
            B19 historical clamped spline

        2013.5-2100:
            constant acceleration

        2100-2300:
            PCHIP through B25 anchors
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

    n_samples = len(df)

    # --------------------------------------------------------
    # B. Output-time bookkeeping
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
        OUTPUT_YEARS
        <= 2010
    )

    transition_mask = (
        (OUTPUT_YEARS > 2010)
        & (OUTPUT_YEARS < 2100)
    )

    future_mask = (
        OUTPUT_YEARS
        >= 2100
    )

    # --------------------------------------------------------
    # C. Historical segment
    # --------------------------------------------------------

    historical_spline = build_historical_spline(
        component=component
    )

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

    # --------------------------------------------------------
    # D. 2013.5 -> 2100 constant-acceleration bridge
    #
    # Decadal output starts again at 2020.
    # --------------------------------------------------------

    transition_years = OUTPUT_YEARS[
        transition_mask
    ].astype(float)

    (
        transition_values,
        acceleration,
    ) = evaluate_constant_acceleration_bridge(
        years=transition_years,
        anomaly_2100=anomaly_2100,
        component=component,
    )

    anomaly_output[
        :,
        transition_mask,
    ] = transition_values

    # --------------------------------------------------------
    # E. 2100 -> 2300 PCHIP
    # --------------------------------------------------------

    future_pchip = build_future_pchip(
        anomaly_2100=anomaly_2100,
        anomaly_2200=anomaly_2200,
        anomaly_2300=anomaly_2300,
    )

    anomaly_output[
        :,
        future_mask,
    ] = future_pchip(
        output_t[
            future_mask
        ]
    )

    # --------------------------------------------------------
    # F. Re-add B19 background contribution
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
    # G. Implied 2050 contribution / transition diagnostics
    # --------------------------------------------------------

    (
        anomaly_2050_matrix,
        _,
    ) = evaluate_constant_acceleration_bridge(
        years=np.array(
            [
                2050.0
            ]
        ),
        anomaly_2100=anomaly_2100,
        component=component,
    )

    anomaly_2050 = (
        anomaly_2050_matrix[
            :,
            0
        ]
    )

    contribution_2050 = (
        anomaly_2050
        + BASELINE_RATE_MM_PER_YR[
            component
        ]
        * 50.0
    )

    # Rate at 2100 implied by constant acceleration.
    dt = (
        2100.0
        - 2013.5
    )

    implied_rate_2100 = (
        ANOMALY_RATE_2013P5_MM_PER_YR[
            component
        ]
        + acceleration
        * dt
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
            "constant_acceleration_2013p5_2100_mm_per_yr2": (
                acceleration
            ),
            "implied_anomaly_rate_2100_mm_per_yr": (
                implied_rate_2100
            ),
            "implied_anomaly_2050_mm": (
                anomaly_2050
            ),
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
# 10. RUN ONE LOW/HIGH SCENARIO
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

    trajectory_ids = df[
        "trajectory_id"
    ].to_numpy(dtype=int)

    records = []

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

    return (
        pd.concat(
            records,
            ignore_index=True,
        ),
        pd.concat(
            diagnostics,
            ignore_index=True,
        ),
    )


# ============================================================
# 11. VALIDATE 2100 / 2200 / 2300 B25 ANCHORS
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
# 12. QUANTILE SUMMARY
# ============================================================

def create_quantile_summary(
    interpolated: pd.DataFrame,
) -> pd.DataFrame:

    records = []

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
            [
                0.05,
                0.17,
                0.50,
                0.83,
                0.95,
            ],
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
    Summarize decade-to-decade behavior for each trajectory.
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

        # Pre-2100 only.
        pre_mask = (
            OUTPUT_YEARS
            <= 2100
        )

        pre_values = values[
            :,
            pre_mask,
        ]

        pre_increments = np.diff(
            pre_values,
            axis=1,
        )

        pre_signs = np.sign(
            pre_increments
        )

        pre_sign_changes = np.sum(
            (
                pre_signs[
                    :,
                    1:
                ]
                * pre_signs[
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
                    "pre2100_increment_sign_changes": (
                        pre_sign_changes
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
        "B25: B19-HISTORICAL + CONSTANT-ACCELERATION + PCHIP"
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
        / "b25_b19style_pchip_decadal_trajectories.csv"
    )

    bridge_file = (
        OUTPUT_DIR
        / "b25_b19style_pchip_2050_diagnostics.csv"
    )

    anchor_file = (
        OUTPUT_DIR
        / "b25_b19style_pchip_anchor_validation.csv"
    )

    quantile_file = (
        OUTPUT_DIR
        / "b25_b19style_pchip_decadal_quantile_summary.csv"
    )

    shape_file = (
        OUTPUT_DIR
        / "b25_b19style_pchip_shape_diagnostics.csv"
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
        "Pre-2100 values are based only on the B19 historical "
        "state/rate and the B25 2100 anchor using constant "
        "acceleration. B25 2200/2300 do not affect the pre-2100 "
        "trajectory. Future interpolation uses shape-preserving PCHIP."
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
