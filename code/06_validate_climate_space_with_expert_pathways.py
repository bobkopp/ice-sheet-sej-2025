#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_validate_climate_space_with_expert_pathways.py

Construct synthetic annual temperature trajectories from the exact
climate pathways shown to the B25 experts and pass them through the
GSAT + cumulative-warming climate-space probability calculation.

LOW, INTERMEDIATE, and HIGH should map exactly to one-hot probabilities
at 2100, 2200, and 2300.

OVERSHOOT and COMMITMENT are not response anchors, so they should produce
mixtures based on their locations in climate space.

@author: elizabeth
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from B25data_processing import project_dir

PROJECT_DIR = Path(project_dir)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "06_climate_space_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_YEARS = np.array(
    [
        2000,
        2020,
        2050,
        2100,
        2200,
        2300,
    ],
    dtype=int,
)

PATHWAYS = {
    "LOW": np.array(
        [0.7, 1.2, 1.7, 2.0, 2.0, 2.0]
    ),
    "INTERMEDIATE": np.array(
        [0.7, 1.2, 2.0, 3.5, 3.5, 3.5]
    ),
    "HIGH": np.array(
        [0.7, 1.2, 2.5, 5.0, 5.0, 5.0]
    ),
    "OVERSHOOT": np.array(
        [0.7, 1.2, 2.0, 3.5, 2.0, 2.0]
    ),
    "COMMITMENT": np.array(
        [0.7, 1.2, 1.3, 1.3, 1.3, 1.3]
    ),
}

PROJECTION_YEARS = [
    2100,
    2200,
    2300,
]

REFERENCE_CLIMATE = pd.DataFrame(
    [
        {"year": 2100, "role": "LOW", "gsat": 2.0, "cumulative_warming": 155.0},
        {"year": 2100, "role": "MODERATE", "gsat": 3.5, "cumulative_warming": 204.5},
        {"year": 2100, "role": "HIGH", "gsat": 5.0, "cumulative_warming": 262.0},

        {"year": 2200, "role": "LOW", "gsat": 2.0, "cumulative_warming": 355.0},
        {"year": 2200, "role": "MODERATE", "gsat": 3.5, "cumulative_warming": 554.5},
        {"year": 2200, "role": "HIGH", "gsat": 5.0, "cumulative_warming": 762.0},

        {"year": 2300, "role": "LOW", "gsat": 2.0, "cumulative_warming": 555.0},
        {"year": 2300, "role": "MODERATE", "gsat": 3.5, "cumulative_warming": 904.5},
        {"year": 2300, "role": "HIGH", "gsat": 5.0, "cumulative_warming": 1262.0},
    ]
)

ROLE_ORDER = [
    "LOW",
    "MODERATE",
    "HIGH",
]

FEATURE_SCALES = {
    "gsat": 1.0,
    "cumulative_warming": 200.0,
}

KERNEL_BANDWIDTH = 1.0


def build_annual_pathways():

    years = np.arange(
        2000,
        2301,
        dtype=int,
    )

    frames = []

    for pathway, values in PATHWAYS.items():

        temp = np.interp(
            years,
            ANCHOR_YEARS,
            values,
        )

        frames.append(
            pd.DataFrame(
                {
                    "pathway": pathway,
                    "year": years,
                    "gsat": temp,
                }
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def calculate_features(annual):

    records = []

    for pathway, group in annual.groupby(
        "pathway"
    ):

        group = group.sort_values("year")

        for year in PROJECTION_YEARS:

            subset = group.loc[
                group["year"] <= year
            ]

            x = subset["year"].to_numpy(dtype=float)
            y = subset["gsat"].to_numpy(dtype=float)

            if hasattr(np, "trapezoid"):
                cumulative = np.trapezoid(
                    y,
                    x=x,
                )
            else:
                cumulative = np.trapz(
                    y,
                    x=x,
                )

            gsat = group.loc[
                group["year"] == year,
                "gsat",
            ].iloc[0]

            records.append(
                {
                    "pathway": pathway,
                    "year": year,
                    "gsat": gsat,
                    "cumulative_warming": cumulative,
                }
            )

    return pd.DataFrame(records)


def calculate_probabilities(features):

    records = []

    scales = np.array(
        [
            FEATURE_SCALES["gsat"],
            FEATURE_SCALES["cumulative_warming"],
        ],
        dtype=float,
    )

    for _, row in features.iterrows():

        refs = (
            REFERENCE_CLIMATE.loc[
                REFERENCE_CLIMATE["year"]
                == row["year"]
            ]
            .set_index("role")
            .loc[ROLE_ORDER]
            .reset_index()
        )

        target = np.array(
            [
                row["gsat"],
                row["cumulative_warming"],
            ],
            dtype=float,
        )

        ref_matrix = refs[
            [
                "gsat",
                "cumulative_warming",
            ]
        ].to_numpy(dtype=float)

        scaled = (
            target[np.newaxis, :]
            - ref_matrix
        ) / scales[np.newaxis, :]

        distances = np.sqrt(
            np.sum(
                scaled ** 2,
                axis=1,
            )
        )

        logp = (
            -0.5
            * (
                distances
                / KERNEL_BANDWIDTH
            ) ** 2
        )

        logp -= np.max(logp)

        raw = np.exp(logp)
        probabilities = raw / raw.sum()

        exact = distances <= 1e-12

        if exact.any():
            probabilities = exact.astype(float)

        records.append(
            {
                "pathway": row["pathway"],
                "year": int(row["year"]),
                "gsat": row["gsat"],
                "cumulative_warming": row[
                    "cumulative_warming"
                ],
                "distance_low": distances[0],
                "distance_moderate": distances[1],
                "distance_high": distances[2],
                "probability_low": probabilities[0],
                "probability_moderate": probabilities[1],
                "probability_high": probabilities[2],
            }
        )

    return pd.DataFrame(records)


def validate_exact_anchors(results):

    expected = {
        "LOW": "probability_low",
        "INTERMEDIATE": "probability_moderate",
        "HIGH": "probability_high",
    }

    checks = []

    for pathway, probability_column in expected.items():

        subset = results.loc[
            results["pathway"] == pathway
        ]

        passed = np.allclose(
            subset[
                probability_column
            ].to_numpy(dtype=float),
            1.0,
            atol=1e-12,
            rtol=0.0,
        )

        checks.append(
            {
                "pathway": pathway,
                "expected_probability_column": probability_column,
                "all_projection_years_exactly_one_hot": passed,
                "minimum_expected_probability": subset[
                    probability_column
                ].min(),
            }
        )

    return pd.DataFrame(checks)


def main():

    print("=" * 72)
    print("VALIDATE CLIMATE SPACE WITH EXPERT PATHWAYS")
    print("=" * 72)

    annual = build_annual_pathways()
    features = calculate_features(annual)
    results = calculate_probabilities(features)
    checks = validate_exact_anchors(results)

    annual.to_csv(
        OUTPUT_DIR
        / "synthetic_expert_temperature_pathways_annual.csv",
        index=False,
    )

    results.to_csv(
        OUTPUT_DIR
        / "synthetic_expert_pathway_climate_space_probabilities.csv",
        index=False,
    )

    checks.to_csv(
        OUTPUT_DIR
        / "synthetic_expert_pathway_exact_anchor_validation.csv",
        index=False,
    )

    print(
        "\nSynthetic pathway climate features and probabilities:"
    )

    print(
        results[
            [
                "pathway",
                "year",
                "gsat",
                "cumulative_warming",
                "probability_low",
                "probability_moderate",
                "probability_high",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nExact-anchor validation:"
    )

    print(
        checks.to_string(index=False)
    )

    if not checks[
        "all_projection_years_exactly_one_hot"
    ].all():
        raise RuntimeError(
            "LOW / INTERMEDIATE / HIGH pathways did not map exactly "
            "to their intended climate anchors."
        )

    print(
        "\nSaved validation outputs to:"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
