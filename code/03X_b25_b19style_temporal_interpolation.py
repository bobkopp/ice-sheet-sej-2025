#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_b25_b19style_temporal_interpolation.py

Reconstruct a Bamber et al. (2019)-style through-time interpolation for
the temporally coupled B25 LOW and HIGH ice-sheet projection libraries.

======================================================================
WHAT IS REPRODUCED FROM B19
======================================================================

For each ice-sheet component and each Monte Carlo trajectory, B19 used:

    1. A background ("baseline") contribution rate.
    2. Historical anomaly anchors around 2000, 2010, and 2013.5.
    3. Elicited future anchors at 2050, 2100, 2200, and 2300.
    4. A clamped cubic spline.
    5. A prescribed derivative at 2000.
    6. An extrapolated derivative at 2300.
    7. Addition of the background linear contribution after fitting
       the spline to the de-baselined anomaly.

The original MATLAB knots were:

    t = [0, 10, 13.5, 50, 100, 200, 300]

where t is years since 2000.

This script reproduces that structure with scipy.interpolate.CubicSpline.

======================================================================
ONE REQUIRED B25 APPROXIMATION: THE MISSING 2050 SLR ANCHOR
======================================================================

B19 had an elicited 2050 ice-sheet contribution. B25 does not.

Therefore the ONLY new bridge introduced here is a 2050 de-baselined
ice-sheet contribution estimated between the historical 2013.5 anchor
and the B25 2100 anchor according to progress through the expert climate
pathway in cumulative-warming space:

    fraction_2050 =
        (CW_2050 - CW_2013.5) /
        (CW_2100 - CW_2013.5)

    anomaly_2050 =
        anomaly_2013.5
        + fraction_2050
          * (anomaly_2100 - anomaly_2013.5)

For the expert pathways:

    LOW:
        CW_2050 = 62.5 K yr
        CW_2100 = 155.0 K yr

    HIGH:
        CW_2050 = 74.5 K yr
        CW_2100 = 262.0 K yr

The common 2000-2020 expert pathway is:
    2000 = 0.7 C
    2020 = 1.2 C

Assuming linear temperature change between those years gives
CW_2013.5 = 11.728125 K yr.

This 2050 bridge is NOT an exact B19 step. It is the explicit B25
approximation required because no 2050 B25 SEJ distribution exists.

======================================================================
OUTPUT
======================================================================

Separate LOW and HIGH decadal trajectories are generated from 2000
through 2300 for:

    EAIS
    WAIS
    GrIS
    Total AIS = EAIS + WAIS
    Total ice-sheet SLR = EAIS + WAIS + GrIS

Units remain mm, matching the B25 inputs and the B19 historical rates.

@author: elizabeth
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

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
    / "03_temporal_interpolation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# B19 produced decadal values.
OUTPUT_STEP_YEARS = 10

OUTPUT_YEARS = np.arange(
    2000,
    2300 + OUTPUT_STEP_YEARS,
    OUTPUT_STEP_YEARS,
    dtype=int,
)

# Projection-year codes in B25 variable names.
YEAR_CODE = {
    2100: "21",
    2200: "22",
    2300: "23",
}

# B25 warming codes.
WARMING_CODE = {
    "LOW": "2",
    "HIGH": "5",
}


# ============================================================
# 2. B19 HISTORICAL / BASELINE CONSTANTS
# ============================================================

# Original B19 MATLAB:
#
# baseline = [275 0 200 75] / 360
#
# Component order there was:
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
# Used as the LEFT-END DERIVATIVE of the de-baselined spline.
#
# Units: mm/yr
START_ANOMALY_SLOPE_MM_PER_YR = {
    "EAIS": -73.0 / 360.0,
    "GrIS": -145.0 / 360.0,
    "WAIS": -44.0 / 360.0,
}

# Original B19 MATLAB:
#
# baselineanom2013 = [185 -31 100 116] / 360
# baselinelev2013  = baselineanom2013 * 3
#
# This is the de-baselined anomaly LEVEL at 2013.5.
#
# Units: mm
ANOMALY_LEVEL_2013P5_MM = {
    "EAIS": (-31.0 / 360.0) * 3.0,
    "GrIS": (100.0 / 360.0) * 3.0,
    "WAIS": (116.0 / 360.0) * 3.0,
}


# ============================================================
# 3. EXPERT CLIMATE PATHWAY INFORMATION
# ============================================================

# Cumulative warming supplied to the B25 experts, K yr.
CUMULATIVE_WARMING = {
    "LOW": {
        2000: 0.0,
        2020: 19.0,
        2050: 62.5,
        2100: 155.0,
        2200: 355.0,
        2300: 555.0,
    },
    "HIGH": {
        2000: 0.0,
        2020: 19.0,
        2050: 74.5,
        2100: 262.0,
        2200: 762.0,
        2300: 1262.0,
    },
}

# Expert GSAT pathway common to all scenarios from 2000 to 2020:
# 2000 = 0.7 C
# 2020 = 1.2 C
#
# Calculate cumulative warming through 2013.5 assuming linear GSAT.
def cumulative_warming_2000_to_2013p5() -> float:

    duration = 13.5

    temp_2000 = 0.7
    temp_2020 = 1.2

    temp_2013p5 = (
        temp_2000
        + (
            temp_2020 - temp_2000
        )
        * duration
        / 20.0
    )

    return (
        0.5
        * (
            temp_2000
            + temp_2013p5
        )
        * duration
    )


CW_2013P5 = cumulative_warming_2000_to_2013p5()


# ============================================================
# 4. LOAD LOW / HIGH TEMPORALLY COUPLED LIBRARIES
# ============================================================

def load_library(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    if "trajectory_id" not in df.columns:
        raise KeyError(
            f"'trajectory_id' not found in:\n{path}"
        )

    expected = np.arange(
        len(df),
        dtype=int,
    )

    if not np.array_equal(
        df[
            "trajectory_id"
        ].to_numpy(dtype=int),
        expected,
    ):
        raise ValueError(
            "trajectory_id must be 0,...,N-1."
        )

    return df


# ============================================================
# 5. EXTRACT B25 COMPONENT ANCHORS
# ============================================================

def extract_b25_component_anchors(
    df: pd.DataFrame,
    scenario_role: str,
    component: str,
) -> np.ndarray:
    """
    Return B25 contribution anchors at:
        2100
        2200
        2300

    Shape:
        (n_trajectories, 3)

    B25 prefixes:
        E -> EAIS
        W -> WAIS
        G -> GrIS
    """

    prefix = {
        "EAIS": "E",
        "WAIS": "W",
        "GrIS": "G",
    }[component]

    wc = WARMING_CODE[
        scenario_role
    ]

    columns = [
        f"{prefix}{wc}_{YEAR_CODE[year]}"
        for year in [
            2100,
            2200,
            2300,
        ]
    ]

    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            "Required B25 columns are missing:\n"
            + "\n".join(
                missing
            )
        )

    return df[
        columns
    ].to_numpy(
        dtype=float
    )


# ============================================================
# 6. CONVERT FUTURE B25 VALUES TO B19 DE-BASELINED ANOMALIES
# ============================================================

def debaseline_future_anchors(
    future_values_mm: np.ndarray,
    component: str,
) -> np.ndarray:
    """
    Reproduce the B19 operation:

        data - dataBaseline

    where:
        dataBaseline = baseline_rate * years_since_2000

    future_values_mm shape:
        (n_trajectories, 3)

    corresponding to:
        2100, 2200, 2300
    """

    elapsed = np.array(
        [
            100.0,
            200.0,
            300.0,
        ]
    )

    baseline_rate = (
        BASELINE_RATE_MM_PER_YR[
            component
        ]
    )

    baseline_contribution = (
        baseline_rate
        * elapsed
    )

    return (
        future_values_mm
        - baseline_contribution[
            np.newaxis,
            :
        ]
    )


# ============================================================
# 7. ESTIMATE MISSING 2050 ANCHOR
# ============================================================

def estimate_2050_anomaly(
    anomaly_2100_mm: np.ndarray,
    scenario_role: str,
    component: str,
) -> tuple[np.ndarray, float]:
    """
    Estimate the missing B25 2050 de-baselined anomaly using
    progress through the expert cumulative-warming pathway.

    Interpolate between:
        historical anomaly level at 2013.5
    and:
        B25 anomaly at 2100
    """

    cw = CUMULATIVE_WARMING[
        scenario_role
    ]

    fraction = (
        (
            cw[2050]
            - CW_2013P5
        )
        /
        (
            cw[2100]
            - CW_2013P5
        )
    )

    anomaly_2013p5 = (
        ANOMALY_LEVEL_2013P5_MM[
            component
        ]
    )

    anomaly_2050 = (
        anomaly_2013p5
        + fraction
        * (
            anomaly_2100_mm
            - anomaly_2013p5
        )
    )

    return (
        anomaly_2050,
        float(fraction),
    )


# ============================================================
# 8. B19 END-SLOPE EXTRAPOLATION
# ============================================================

def calculate_end_slope(
    anomaly_2100: np.ndarray,
    anomaly_2200: np.ndarray,
    anomaly_2300: np.ndarray,
) -> np.ndarray:
    """
    Reproduce B19:

        slope1 = (2200 - 2100) / 100
        slope2 = (2300 - 2200) / 100

        end_slope =
            slope2
            + 0.5 * (slope2 - slope1)

    Units:
        mm/yr
    """

    slope_2100_2200 = (
        anomaly_2200
        - anomaly_2100
    ) / 100.0

    slope_2200_2300 = (
        anomaly_2300
        - anomaly_2200
    ) / 100.0

    end_slope = (
        slope_2200_2300
        + 0.5
        * (
            slope_2200_2300
            - slope_2100_2200
        )
    )

    return end_slope


# ============================================================
# 9. INTERPOLATE ONE COMPONENT
# ============================================================

def interpolate_component(
    df: pd.DataFrame,
    scenario_role: str,
    component: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate B19-style interpolated contribution trajectories.

    Returns
    -------
    contribution_mm
        Shape:
            (n_trajectories, n_output_years)

    diagnostics
        One row per trajectory describing the constructed 2050
        anchor and endpoint slope.
    """

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

    anomaly_2050, fraction_2050 = (
        estimate_2050_anomaly(
            anomaly_2100_mm=anomaly_2100,
            scenario_role=scenario_role,
            component=component,
        )
    )

    end_slope = calculate_end_slope(
        anomaly_2100=anomaly_2100,
        anomaly_2200=anomaly_2200,
        anomaly_2300=anomaly_2300,
    )

    start_slope = (
        START_ANOMALY_SLOPE_MM_PER_YR[
            component
        ]
    )

    anomaly_2013p5 = (
        ANOMALY_LEVEL_2013P5_MM[
            component
        ]
    )

    n = len(df)

    # Exact B19 spline-knot locations, years since 2000.
    knot_t = np.array(
        [
            0.0,
            10.0,
            13.5,
            50.0,
            100.0,
            200.0,
            300.0,
        ]
    )

    # B19 de-baselined spline values:
    #
    # 2000   = 0
    # 2010   = 0
    # 2013.5 = historical anomaly level
    # 2050   = B25 approximation
    # 2100   = B25
    # 2200   = B25
    # 2300   = B25
    knot_y = np.column_stack(
        [
            np.zeros(n),
            np.zeros(n),
            np.full(
                n,
                anomaly_2013p5,
                dtype=float,
            ),
            anomaly_2050,
            anomaly_2100,
            anomaly_2200,
            anomaly_2300,
        ]
    )

    # scipy CubicSpline accepts one derivative per trajectory
    # at each boundary.
    spline = CubicSpline(
        x=knot_t,
        y=knot_y,
        axis=1,
        bc_type=(
            (
                1,
                np.full(
                    n,
                    start_slope,
                    dtype=float,
                ),
            ),
            (
                1,
                end_slope,
            ),
        ),
    )

    output_t = (
        OUTPUT_YEARS.astype(float)
        - 2000.0
    )

    anomaly_output = spline(
        output_t
    )

    # Re-add the B19 background linear contribution.
    baseline_rate = (
        BASELINE_RATE_MM_PER_YR[
            component
        ]
    )

    contribution_mm = (
        anomaly_output
        + baseline_rate
        * output_t[
            np.newaxis,
            :
        ]
    )

    diagnostics = pd.DataFrame(
        {
            "trajectory_id": df[
                "trajectory_id"
            ].to_numpy(dtype=int),
            "scenario_role": scenario_role,
            "component": component,
            "cw_2013p5": CW_2013P5,
            "cw_2050": CUMULATIVE_WARMING[
                scenario_role
            ][2050],
            "cw_2100": CUMULATIVE_WARMING[
                scenario_role
            ][2100],
            "fraction_2013p5_to_2050": (
                fraction_2050
            ),
            "estimated_anomaly_2050_mm": (
                anomaly_2050
            ),
            "b25_anomaly_2100_mm": (
                anomaly_2100
            ),
            "b25_anomaly_2200_mm": (
                anomaly_2200
            ),
            "b25_anomaly_2300_mm": (
                anomaly_2300
            ),
            "end_slope_2300_mm_per_yr": (
                end_slope
            ),
        }
    )

    return (
        contribution_mm,
        diagnostics,
    )


# ============================================================
# 10. RUN ONE SCENARIO
# ============================================================

def run_scenario(
    df: pd.DataFrame,
    scenario_role: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Interpolate EAIS, WAIS, and GrIS separately, then derive:

        Total AIS = EAIS + WAIS
        Total ice-sheet SLR = EAIS + WAIS + GrIS
    """

    component_arrays = {}
    diagnostics = []

    for component in [
        "EAIS",
        "WAIS",
        "GrIS",
    ]:

        print(
            f"Interpolating {scenario_role} {component}..."
        )

        values, diag = (
            interpolate_component(
                df=df,
                scenario_role=scenario_role,
                component=component,
            )
        )

        component_arrays[
            component
        ] = values

        diagnostics.append(
            diag
        )

    component_arrays[
        "Total AIS"
    ] = (
        component_arrays["EAIS"]
        + component_arrays["WAIS"]
    )

    component_arrays[
        "Total ice-sheet SLR"
    ] = (
        component_arrays["Total AIS"]
        + component_arrays["GrIS"]
    )

    records = []

    trajectory_ids = df[
        "trajectory_id"
    ].to_numpy(dtype=int)

    for component, values in component_arrays.items():

        frame = pd.DataFrame(
            values,
            columns=OUTPUT_YEARS,
        )

        frame.insert(
            0,
            "trajectory_id",
            trajectory_ids,
        )

        long = frame.melt(
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
# 11. VALIDATE THAT 2100 / 2200 / 2300 ARE PRESERVED
# ============================================================

def validate_projection_anchors(
    interpolated: pd.DataFrame,
    source_df: pd.DataFrame,
    scenario_role: str,
) -> pd.DataFrame:

    records = []

    wc = WARMING_CODE[
        scenario_role
    ]

    prefix = {
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

            source_col = (
                f"{prefix[component]}"
                f"{wc}_"
                f"{YEAR_CODE[year]}"
            )

            source = source_df[
                source_col
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

            diff = np.abs(
                reconstructed
                - source
            )

            records.append(
                {
                    "scenario_role": scenario_role,
                    "component": component,
                    "year": year,
                    "maximum_absolute_difference_mm": (
                        diff.max()
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
# 12. SUMMARY QUANTILES
# ============================================================

def create_quantile_summary(
    interpolated: pd.DataFrame,
) -> pd.DataFrame:

    quantiles = [
        0.05,
        0.17,
        0.50,
        0.83,
        0.95,
    ]

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
            quantiles,
        )

        records.append(
            {
                "scenario_role": scenario_role,
                "component": component,
                "year": year,
                "mean_mm": np.mean(
                    values
                ),
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
# 13. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "B25: B19-STYLE THROUGH-TIME INTERPOLATION"
    )
    print(
        "=" * 72
    )

    print(
        f"\nComputed cumulative warming through 2013.5: "
        f"{CW_2013P5:.6f} K yr"
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
            f"{scenario_role} THROUGH-TIME INTERPOLATION"
        )
        print(
            "=" * 72
        )

        output, diagnostics = (
            run_scenario(
                df=source_df,
                scenario_role=scenario_role,
            )
        )

        anchor_check = (
            validate_projection_anchors(
                interpolated=output,
                source_df=source_df,
                scenario_role=scenario_role,
            )
        )

        print(
            "\nAnchor reproduction check:"
        )
        print(
            anchor_check.to_string(
                index=False
            )
        )

        if not anchor_check[
            "allclose"
        ].all():
            raise RuntimeError(
                f"{scenario_role} interpolation failed "
                "to reproduce one or more B25 anchors."
            )

        all_outputs.append(
            output
        )

        all_diagnostics.append(
            diagnostics
        )

        all_anchor_checks.append(
            anchor_check
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

    summary = create_quantile_summary(
        interpolated
    )

    # --------------------------------------------------------
    # Save combined files
    # --------------------------------------------------------

    interpolated_file = (
        OUTPUT_DIR
        / "b25_b19style_low_high_decadal_trajectories.csv"
    )

    diagnostic_file = (
        OUTPUT_DIR
        / "b25_b19style_2050_bridge_diagnostics.csv"
    )

    anchor_file = (
        OUTPUT_DIR
        / "b25_b19style_anchor_validation.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "b25_b19style_decadal_quantile_summary.csv"
    )

    interpolated.to_csv(
        interpolated_file,
        index=False,
    )

    diagnostics.to_csv(
        diagnostic_file,
        index=False,
    )

    anchor_checks.to_csv(
        anchor_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
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
        interpolated_file
    )

    print(
        "\nSaved 2050 bridge diagnostics to:"
    )
    print(
        diagnostic_file
    )

    print(
        "\nSaved anchor validation to:"
    )
    print(
        anchor_file
    )

    print(
        "\nSaved decadal quantile summary to:"
    )
    print(
        summary_file
    )

    print(
        "\nImportant:"
    )
    print(
        "The spline structure and historical/baseline treatment "
        "follow B19. The 2050 SLR anchor is an explicit B25 "
        "approximation based on the expert cumulative-warming pathway."
    )

    return {
        "interpolated": interpolated,
        "diagnostics": diagnostics,
        "anchor_checks": anchor_checks,
        "summary": summary,
    }


if __name__ == "__main__":

    results = main()
