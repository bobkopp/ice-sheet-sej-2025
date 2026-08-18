#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_b25_categorical_climate_space_temporal_interpolation_PCHIP.py

Generate decadal SLR trajectories from the categorical climate-space
B25 anchors produced by:

    02_b25_climate_space_categorical_selection.py

At each projection year, each FaIR trajectory has already selected ONE
actual elicited B25 warming library:
    LOW / MODERATE / HIGH

This script does not average across warming scenarios.

Through-time interpolation remains:
    2000-2013.5 : B19 historical anchoring
    2013.5-2100 : constant-acceleration bridge
    2100-2300   : PCHIP through selected B25 anchors

@author: elizabeth
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

from B25data_processing import project_dir


PROJECT_DIR = Path(project_dir)

INPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "02_climate_space_categorical_selection"
    / "b25_categorical_climate_space_selected_anchors.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "03_categorical_climate_space_temporal_interpolation_pchip"
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


def load_selected_anchors():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    return pd.read_csv(
        INPUT_FILE
    )


def selected_component_anchors(
    df,
    component,
):

    prefix = PREFIX[
        component
    ]

    # The selected row retains all warming columns.
    # We must read the column matching the selected category at each year.
    values = []

    role_to_code = {
        "LOW": "2",
        "MODERATE": "35",
        "HIGH": "5",
    }

    for year in [
        2100,
        2200,
        2300,
    ]:

        selected = df[
            f"selected_library_{year}"
        ].to_numpy()

        yc = YEAR_CODE[
            year
        ]

        out = np.empty(
            len(
                df
            ),
            dtype=float,
        )

        for role, code in role_to_code.items():

            mask = (
                selected
                == role
            )

            column = (
                f"{prefix}{code}_{yc}"
            )

            if column not in df.columns:
                raise KeyError(
                    f"Missing selected anchor column: {column}"
                )

            out[
                mask
            ] = df.loc[
                mask,
                column,
            ].to_numpy(
                dtype=float
            )

        values.append(
            out
        )

    return np.column_stack(
        values
    )


def debaseline(
    values,
    component,
):

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


def historical_spline(
    component,
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


def interpolate_component(
    df,
    component,
):

    anchors = selected_component_anchors(
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
        )
    )

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

    acceleration = (
        2.0
        * (
            a2100
            - s0
            - r0
            * dt
        )
        / dt ** 2
    )

    tau = (
        years_transition
        - 2013.5
    )

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

    return (
        contribution,
        anchors,
    )


def build_trajectories(
    df,
):

    arrays = {}
    original_anchors = {}

    for component in [
        "EAIS",
        "WAIS",
        "GrIS",
    ]:

        print(
            f"Interpolating {component}..."
        )

        (
            arrays[
                component
            ],
            original_anchors[
                component
            ],
        ) = interpolate_component(
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

    meta = df[
        [
            "fair_sample_id",
            "scenario",
            "scenario_label",
            "original_sample",
            "trajectory_id",
            "selected_library_2100",
            "selected_library_2200",
            "selected_library_2300",
        ]
    ].copy()

    frames = []

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

    return (
        pd.concat(
            frames,
            ignore_index=True,
        ),
        original_anchors,
    )


def validate_anchors(
    trajectories,
    source,
    original_anchors,
):

    records = []

    for component in [
        "EAIS",
        "WAIS",
        "GrIS",
    ]:

        for j, year in enumerate(
            [
                2100,
                2200,
                2300,
            ]
        ):

            expected = original_anchors[
                component
            ][
                :,
                j
            ]

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
                            atol=1e-8,
                            rtol=0.0,
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
            "Selected B25 anchors were not exactly reproduced."
        )

    return validation


def main():

    print(
        "=" * 72
    )
    print(
        "B25 CATEGORICAL CLIMATE-SPACE THROUGH-TIME INTERPOLATION"
    )
    print(
        "=" * 72
    )

    source = load_selected_anchors()

    (
        trajectories,
        original_anchors,
    ) = build_trajectories(
        source
    )

    validation = validate_anchors(
        trajectories,
        source,
        original_anchors,
    )

    trajectory_file = (
        OUTPUT_DIR
        / "b25_categorical_climate_space_pchip_decadal_trajectories.csv"
    )

    validation_file = (
        OUTPUT_DIR
        / "b25_categorical_climate_space_pchip_anchor_validation.csv"
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
