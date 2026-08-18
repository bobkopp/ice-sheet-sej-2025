#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_b25_climate_space_temporal_interpolation_PCHIP.py

Generate decadal ice-sheet SLR trajectories from the climate-conditioned
B25 anchors produced by:

    02_b25_climate_space_percentile_interpolation.py

The climate-conditioned anchors already incorporate:
    - FaIR GSAT
    - cumulative warming
    - LOW/MODERATE/HIGH climate-space weighting
    - percentile matching across warming scenarios
    - temporally varying percentile ranks

This script DOES NOT change that climate conditioning.

Through-time interpolation:
    2000-2013.5:
        B19 historical clamped spline

    2013.5-2100:
        constant-acceleration bridge to the climate-conditioned 2100 anchor

    2100-2300:
        PCHIP through the climate-conditioned 2100/2200/2300 anchors

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

PROJECT_DIR = Path(
    project_dir
)

INPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "02_climate_space_percentile_interpolation"
    / "b25_climate_space_percentile_interpolated_anchors.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "03_climate_space_temporal_interpolation_pchip"
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

PREFIX = {
    "EAIS": "E",
    "WAIS": "W",
    "GrIS": "G",
}


# ============================================================
# 2. B19 HISTORICAL CONSTANTS
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
    for component, rate
    in ANOMALY_RATE_2013P5_MM_PER_YR.items()
}


# ============================================================
# 3. LOAD CLIMATE-CONDITIONED ANCHORS
# ============================================================

def load_anchors() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_meta = {
        "fair_sample_id",
        "scenario",
        "scenario_label",
        "original_sample",
    }

    missing = (
        required_meta
        - set(
            df.columns
        )
    )

    if missing:
        raise KeyError(
            "Missing metadata columns:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    for component, prefix in PREFIX.items():

        for year in YEAR_CODE:

            column = (
                f"{prefix}_interp_"
                f"{YEAR_CODE[year]}"
            )

            if column not in df.columns:

                raise KeyError(
                    f"Missing climate-conditioned anchor: {column}"
                )

    return df


# ============================================================
# 4. EXTRACT / DEBASELINE ANCHORS
# ============================================================

def extract_component_anchors(
    df: pd.DataFrame,
    component: str,
) -> np.ndarray:

    prefix = PREFIX[
        component
    ]

    return np.column_stack(
        [
            df[
                f"{prefix}_interp_{YEAR_CODE[year]}"
            ].to_numpy(
                dtype=float
            )
            for year in [
                2100,
                2200,
                2300,
            ]
        ]
    )


def debaseline(
    values: np.ndarray,
    component: str,
) -> np.ndarray:

    elapsed = np.array(
        [
            100.0,
            200.0,
            300.0,
        ]
    )

    return (
        values
        - BASELINE_RATE_MM_PER_YR[
            component
        ]
        * elapsed[
            np.newaxis,
            :
        ]
    )


# ============================================================
# 5. HISTORICAL + BRIDGE
# ============================================================

def historical_spline(
    component: str,
):

    return CubicSpline(
        x=np.array(
            [
                0.0,
                10.0,
                13.5,
            ]
        ),
        y=np.array(
            [
                0.0,
                0.0,
                ANOMALY_LEVEL_2013P5_MM[
                    component
                ],
            ]
        ),
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


def constant_acceleration(
    anomaly_2100: np.ndarray,
    component: str,
) -> np.ndarray:

    dt = (
        2100.0
        - 2013.5
    )

    s0 = ANOMALY_LEVEL_2013P5_MM[
        component
    ]

    r0 = ANOMALY_RATE_2013P5_MM_PER_YR[
        component
    ]

    return (
        2.0
        * (
            anomaly_2100
            - s0
            - r0
            * dt
        )
        / dt ** 2
    )


# ============================================================
# 6. INTERPOLATE ONE COMPONENT
# ============================================================

def interpolate_component(
    df: pd.DataFrame,
    component: str,
):

    anchors = extract_component_anchors(
        df,
        component,
    )

    anomaly = debaseline(
        anchors,
        component,
    )

    a2100 = anomaly[
        :,
        0
    ]

    a2200 = anomaly[
        :,
        1
    ]

    a2300 = anomaly[
        :,
        2
    ]

    n = len(
        df
    )

    t = (
        OUTPUT_YEARS.astype(
            float
        )
        - 2000.0
    )

    output = np.empty(
        (
            n,
            len(
                OUTPUT_YEARS
            ),
        ),
        dtype=float,
    )

    # Historical.
    hist_mask = (
        OUTPUT_YEARS
        <= 2010
    )

    hist = historical_spline(
        component
    )

    output[
        :,
        hist_mask,
    ] = hist(
        t[
            hist_mask
        ]
    )[
        np.newaxis,
        :
    ]

    # Constant-acceleration transition.
    transition_mask = (
        (
            OUTPUT_YEARS
            > 2010
        )
        & (
            OUTPUT_YEARS
            < 2100
        )
    )

    years_transition = OUTPUT_YEARS[
        transition_mask
    ].astype(
        float
    )

    acceleration = constant_acceleration(
        a2100,
        component,
    )

    tau = (
        years_transition
        - 2013.5
    )

    s0 = ANOMALY_LEVEL_2013P5_MM[
        component
    ]

    r0 = ANOMALY_RATE_2013P5_MM_PER_YR[
        component
    ]

    output[
        :,
        transition_mask,
    ] = (
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
        * tau[
            np.newaxis,
            :
        ] ** 2
    )

    # Future PCHIP.
    future_mask = (
        OUTPUT_YEARS
        >= 2100
    )

    pchip = PchipInterpolator(
        x=np.array(
            [
                100.0,
                200.0,
                300.0,
            ]
        ),
        y=np.column_stack(
            [
                a2100,
                a2200,
                a2300,
            ]
        ),
        axis=1,
        extrapolate=False,
    )

    output[
        :,
        future_mask,
    ] = pchip(
        t[
            future_mask
        ]
    )

    # Re-add background contribution.
    contribution = (
        output
        + BASELINE_RATE_MM_PER_YR[
            component
        ]
        * t[
            np.newaxis,
            :
        ]
    )

    return contribution


# ============================================================
# 7. BUILD LONG TRAJECTORIES
# ============================================================

def build_trajectories(
    df: pd.DataFrame,
) -> pd.DataFrame:

    arrays = {}

    for component in [
        "EAIS",
        "WAIS",
        "GrIS",
    ]:

        print(
            f"Interpolating {component}..."
        )

        arrays[
            component
        ] = interpolate_component(
            df,
            component,
        )

    arrays[
        "Total AIS"
    ] = (
        arrays[
            "EAIS"
        ]
        + arrays[
            "WAIS"
        ]
    )

    arrays[
        "Total ice-sheet SLR"
    ] = (
        arrays[
            "Total AIS"
        ]
        + arrays[
            "GrIS"
        ]
    )

    frames = []

    meta = df[
        [
            "fair_sample_id",
            "scenario",
            "scenario_label",
            "original_sample",
        ]
    ].copy()

    for component, values in arrays.items():

        wide = pd.DataFrame(
            values,
            columns=OUTPUT_YEARS,
        )

        wide.insert(
            0,
            "fair_sample_id",
            meta[
                "fair_sample_id"
            ].to_numpy(),
        )

        long = wide.melt(
            id_vars=[
                "fair_sample_id"
            ],
            var_name="year",
            value_name="value_mm",
        )

        long[
            "year"
        ] = long[
            "year"
        ].astype(
            int
        )

        long = long.merge(
            meta,
            on="fair_sample_id",
            how="left",
            validate="many_to_one",
        )

        long[
            "component"
        ] = component

        frames.append(
            long
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# 8. VALIDATE ANCHORS
# ============================================================

def validate_anchors(
    trajectories: pd.DataFrame,
    source: pd.DataFrame,
):

    records = []

    for component, prefix in PREFIX.items():

        for year in [
            2100,
            2200,
            2300,
        ]:

            expected = source[
                f"{prefix}_interp_{YEAR_CODE[year]}"
            ].to_numpy(
                dtype=float
            )

            actual = (
                trajectories.loc[
                    (
                        trajectories[
                            "component"
                        ]
                        == component
                    )
                    & (
                        trajectories[
                            "year"
                        ]
                        == year
                    ),
                    [
                        "fair_sample_id",
                        "value_mm",
                    ]
                ]
                .sort_values(
                    "fair_sample_id"
                )[
                    "value_mm"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            diff = np.abs(
                expected
                - actual
            )

            records.append(
                {
                    "component": component,
                    "year": year,
                    "maximum_absolute_difference_mm": diff.max(),
                    "allclose": bool(
                        np.allclose(
                            expected,
                            actual,
                            rtol=0.0,
                            atol=1e-8,
                        )
                    ),
                }
            )

    validation = pd.DataFrame(
        records
    )

    print(
        "\nAnchor validation:"
    )

    print(
        validation.to_string(
            index=False
        )
    )

    if not validation[
        "allclose"
    ].all():

        raise RuntimeError(
            "One or more climate-conditioned anchors were not reproduced."
        )

    return validation


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "B25 CLIMATE-SPACE THROUGH-TIME INTERPOLATION"
    )
    print(
        "=" * 72
    )

    source = load_anchors()

    trajectories = build_trajectories(
        source
    )

    validation = validate_anchors(
        trajectories,
        source,
    )

    trajectory_file = (
        OUTPUT_DIR
        / "b25_climate_space_pchip_decadal_trajectories.csv"
    )

    validation_file = (
        OUTPUT_DIR
        / "b25_climate_space_pchip_anchor_validation.csv"
    )

    trajectories.to_csv(
        trajectory_file,
        index=False,
    )

    validation.to_csv(
        validation_file,
        index=False,
    )

    print(
        "\nSaved trajectories to:"
    )
    print(
        trajectory_file
    )

    return {
        "trajectories": trajectories,
        "validation": validation,
    }


if __name__ == "__main__":

    results = main()
