#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_check_categorical_climate_space_transitions.py

Diagnostics for the categorical GSAT + cumulative-warming climate-space
selection workflow.

Purpose
-------
Check whether LOW / MODERATE / HIGH category transitions through time are
reasonable and whether switching between B25 warming libraries introduces
suspicious discontinuities in the selected 2100 / 2200 / 2300 ice-sheet
anchors.

Inputs
------
1. Script 02 selection metadata:
   output/b19style_baseline/02_climate_space_categorical_selection/
       b25_categorical_climate_space_selection_metadata.csv

2. Script 02 selected-anchor output:
   output/b19style_baseline/02_climate_space_categorical_selection/
       b25_categorical_climate_space_selected_anchors.csv

Outputs
-------
1. transition_sequences_by_ssp.csv
2. category_transition_matrix_by_ssp.csv
3. climate_direction_diagnostics.csv
4. slr_jump_diagnostics.csv
5. slr_jump_summary_by_transition.csv
6. suspicious_transition_examples.csv

What it checks
--------------
A. Are category changes concentrated in SSPs where they make sense?
B. Does movement in selected category agree with movement in climate-space
   probabilities?
C. Do category-switching trajectories show unusually large jumps in:
       EAIS
       WAIS
       Total AIS
       GrIS
       Total ice-sheet SLR
   relative to trajectories that stay in the same category?

@author: elizabeth
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from B25data_processing import project_dir


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_DIR = Path(project_dir)

INPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "02_climate_space_categorical_selection"
)

SELECTION_FILE = (
    INPUT_DIR
    / "b25_categorical_climate_space_selection_metadata.csv"
)

ANCHOR_FILE = (
    INPUT_DIR
    / "b25_categorical_climate_space_selected_anchors.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "05_categorical_transition_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

YEARS = [
    2100,
    2200,
    2300,
]

ROLE_SCORE = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
}

ROLE_ORDER = [
    "LOW",
    "MODERATE",
    "HIGH",
]

COMPONENT_PREFIX = {
    "EAIS": "E",
    "WAIS": "W",
    "GrIS": "G",
}

YEAR_CODE = {
    2100: "21",
    2200: "22",
    2300: "23",
}

ROLE_CODE = {
    "LOW": "2",
    "MODERATE": "35",
    "HIGH": "5",
}


# ============================================================
# 2. LOAD INPUTS
# ============================================================

def load_inputs():

    for path in [
        SELECTION_FILE,
        ANCHOR_FILE,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    selection = pd.read_csv(
        SELECTION_FILE
    )

    anchors = pd.read_csv(
        ANCHOR_FILE
    )

    return (
        selection,
        anchors,
    )


# ============================================================
# 3. WIDE CATEGORY / CLIMATE TABLE
# ============================================================

def make_wide_selection(
    selection: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "fair_sample_id",
        "scenario",
        "scenario_label",
        "original_sample",
        "year",
        "gsat",
        "cumulative_warming",
        "probability_low",
        "probability_moderate",
        "probability_high",
        "selected_library",
        "category_uniform",
    }

    missing = required - set(
        selection.columns
    )

    if missing:
        raise KeyError(
            "Selection file is missing columns:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    meta = (
        selection[
            [
                "fair_sample_id",
                "scenario",
                "scenario_label",
                "original_sample",
                "category_uniform",
            ]
        ]
        .drop_duplicates(
            subset=[
                "fair_sample_id"
            ]
        )
        .sort_values(
            "fair_sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    wide = meta.copy()

    for year in YEARS:

        y = (
            selection.loc[
                selection[
                    "year"
                ]
                == year
            ]
            .sort_values(
                "fair_sample_id"
            )
            .reset_index(
                drop=True
            )
        )

        wide[
            f"selected_{year}"
        ] = y[
            "selected_library"
        ].to_numpy()

        wide[
            f"gsat_{year}"
        ] = y[
            "gsat"
        ].to_numpy(
            dtype=float
        )

        wide[
            f"cw_{year}"
        ] = y[
            "cumulative_warming"
        ].to_numpy(
            dtype=float
        )

        wide[
            f"p_low_{year}"
        ] = y[
            "probability_low"
        ].to_numpy(
            dtype=float
        )

        wide[
            f"p_moderate_{year}"
        ] = y[
            "probability_moderate"
        ].to_numpy(
            dtype=float
        )

        wide[
            f"p_high_{year}"
        ] = y[
            "probability_high"
        ].to_numpy(
            dtype=float
        )

    return wide


# ============================================================
# 4. TRANSITION SEQUENCES BY SSP
# ============================================================

def transition_sequences_by_ssp(
    wide: pd.DataFrame,
) -> pd.DataFrame:

    df = wide.copy()

    df[
        "sequence"
    ] = (
        df[
            "selected_2100"
        ]
        + " -> "
        + df[
            "selected_2200"
        ]
        + " -> "
        + df[
            "selected_2300"
        ]
    )

    summary = (
        df.groupby(
            [
                "scenario_label",
                "sequence",
            ]
        )
        .size()
        .rename(
            "count"
        )
        .reset_index()
    )

    totals = (
        summary.groupby(
            "scenario_label"
        )[
            "count"
        ]
        .transform(
            "sum"
        )
    )

    summary[
        "fraction_within_ssp"
    ] = (
        summary[
            "count"
        ]
        / totals
    )

    return (
        summary
        .sort_values(
            [
                "scenario_label",
                "fraction_within_ssp",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 5. PAIRWISE TRANSITION MATRICES
# ============================================================

def transition_matrix_by_ssp(
    wide: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for scenario_label, group in wide.groupby(
        "scenario_label"
    ):

        for y1, y2 in [
            (
                2100,
                2200,
            ),
            (
                2200,
                2300,
            ),
        ]:

            tab = pd.crosstab(
                group[
                    f"selected_{y1}"
                ],
                group[
                    f"selected_{y2}"
                ],
                normalize="index",
            )

            for from_role in ROLE_ORDER:

                for to_role in ROLE_ORDER:

                    value = (
                        tab.loc[
                            from_role,
                            to_role,
                        ]
                        if (
                            from_role in tab.index
                            and to_role in tab.columns
                        )
                        else 0.0
                    )

                    records.append(
                        {
                            "scenario_label": scenario_label,
                            "from_year": y1,
                            "to_year": y2,
                            "from_category": from_role,
                            "to_category": to_role,
                            "conditional_fraction": value,
                        }
                    )

    return pd.DataFrame(
        records
    )


# ============================================================
# 6. CLIMATE-DIRECTION DIAGNOSTICS
# ============================================================

def climate_direction_diagnostics(
    wide: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare category movement with changes in climate-space probabilities.

    A simple climate-severity score is defined as:
        E[category score] = 0*p_low + 1*p_mod + 2*p_high

    This is only a diagnostic summary of the probability distribution.
    """

    records = []

    for y1, y2 in [
        (
            2100,
            2200,
        ),
        (
            2200,
            2300,
        ),
    ]:

        score_1 = (
            wide[
                f"p_moderate_{y1}"
            ]
            + 2.0
            * wide[
                f"p_high_{y1}"
            ]
        )

        score_2 = (
            wide[
                f"p_moderate_{y2}"
            ]
            + 2.0
            * wide[
                f"p_high_{y2}"
            ]
        )

        selected_score_1 = wide[
            f"selected_{y1}"
        ].map(
            ROLE_SCORE
        )

        selected_score_2 = wide[
            f"selected_{y2}"
        ].map(
            ROLE_SCORE
        )

        climate_change = (
            score_2
            - score_1
        )

        category_change = (
            selected_score_2
            - selected_score_1
        )

        pair = pd.DataFrame(
            {
                "fair_sample_id": wide[
                    "fair_sample_id"
                ],
                "scenario_label": wide[
                    "scenario_label"
                ],
                "from_year": y1,
                "to_year": y2,
                "from_category": wide[
                    f"selected_{y1}"
                ],
                "to_category": wide[
                    f"selected_{y2}"
                ],
                "climate_severity_change": climate_change,
                "category_score_change": category_change,
                "gsat_change": (
                    wide[
                        f"gsat_{y2}"
                    ]
                    - wide[
                        f"gsat_{y1}"
                    ]
                ),
                "cumulative_warming_change": (
                    wide[
                        f"cw_{y2}"
                    ]
                    - wide[
                        f"cw_{y1}"
                    ]
                ),
            }
        )

        pair[
            "transition"
        ] = (
            pair[
                "from_category"
            ]
            + " -> "
            + pair[
                "to_category"
            ]
        )

        pair[
            "direction_agrees_with_probability_shift"
        ] = (
            (
                pair[
                    "category_score_change"
                ]
                == 0
            )
            | (
                np.sign(
                    pair[
                        "category_score_change"
                    ]
                )
                == np.sign(
                    pair[
                        "climate_severity_change"
                    ]
                )
            )
        )

        records.append(
            pair
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# 7. EXTRACT SELECTED COMPONENT ANCHORS
# ============================================================

def extract_selected_values(
    anchors: pd.DataFrame,
    component: str,
    year: int,
) -> np.ndarray:

    selected = anchors[
        f"selected_library_{year}"
    ].to_numpy()

    prefix = COMPONENT_PREFIX[
        component
    ]

    yc = YEAR_CODE[
        year
    ]

    values = np.empty(
        len(
            anchors
        ),
        dtype=float,
    )

    for role in ROLE_ORDER:

        mask = (
            selected
            == role
        )

        column = (
            f"{prefix}"
            f"{ROLE_CODE[role]}"
            f"_{yc}"
        )

        if column not in anchors.columns:
            raise KeyError(
                f"Missing B25 column: {column}"
            )

        values[
            mask
        ] = anchors.loc[
            mask,
            column,
        ].to_numpy(
            dtype=float
        )

    return values


# ============================================================
# 8. SLR JUMP DIAGNOSTICS
# ============================================================

def slr_jump_diagnostics(
    wide: pd.DataFrame,
    anchors: pd.DataFrame,
) -> pd.DataFrame:

    if not np.array_equal(
        wide[
            "fair_sample_id"
        ].to_numpy(
            dtype=int
        ),
        anchors.sort_values(
            "fair_sample_id"
        )[
            "fair_sample_id"
        ].to_numpy(
            dtype=int
        ),
    ):
        anchors = (
            anchors
            .sort_values(
                "fair_sample_id"
            )
            .reset_index(
                drop=True
            )
        )

    values = {}

    for component in [
        "EAIS",
        "WAIS",
        "GrIS",
    ]:

        for year in YEARS:

            values[
                (
                    component,
                    year,
                )
            ] = extract_selected_values(
                anchors=anchors,
                component=component,
                year=year,
            )

    for year in YEARS:

        values[
            (
                "Total AIS",
                year,
            )
        ] = (
            values[
                (
                    "EAIS",
                    year,
                )
            ]
            + values[
                (
                    "WAIS",
                    year,
                )
            ]
        )

        values[
            (
                "Total ice-sheet SLR",
                year,
            )
        ] = (
            values[
                (
                    "Total AIS",
                    year,
                )
            ]
            + values[
                (
                    "GrIS",
                    year,
                )
            ]
        )

    records = []

    for y1, y2 in [
        (
            2100,
            2200,
        ),
        (
            2200,
            2300,
        ),
    ]:

        transition = (
            wide[
                f"selected_{y1}"
            ]
            + " -> "
            + wide[
                f"selected_{y2}"
            ]
        )

        switched = (
            wide[
                f"selected_{y1}"
            ]
            != wide[
                f"selected_{y2}"
            ]
        )

        for component in [
            "EAIS",
            "WAIS",
            "Total AIS",
            "GrIS",
            "Total ice-sheet SLR",
        ]:

            v1 = values[
                (
                    component,
                    y1,
                )
            ]

            v2 = values[
                (
                    component,
                    y2,
                )
            ]

            absolute_change = (
                v2
                - v1
            )

            per_year_change = (
                absolute_change
                / (
                    y2
                    - y1
                )
            )

            records.append(
                pd.DataFrame(
                    {
                        "fair_sample_id": wide[
                            "fair_sample_id"
                        ].to_numpy(),
                        "scenario_label": wide[
                            "scenario_label"
                        ].to_numpy(),
                        "from_year": y1,
                        "to_year": y2,
                        "transition": transition.to_numpy(),
                        "switched_category": switched.to_numpy(),
                        "component": component,
                        "value_from_mm": v1,
                        "value_to_mm": v2,
                        "change_mm": absolute_change,
                        "absolute_change_mm": np.abs(
                            absolute_change
                        ),
                        "change_rate_mm_per_yr": per_year_change,
                    }
                )
            )

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# 9. SUMMARIZE JUMPS BY TRANSITION
# ============================================================

def summarize_slr_jumps(
    jump_df: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        jump_df.groupby(
            [
                "scenario_label",
                "from_year",
                "to_year",
                "transition",
                "switched_category",
                "component",
            ]
        )[
            "absolute_change_mm"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                lambda x: np.quantile(
                    x,
                    0.90,
                ),
                lambda x: np.quantile(
                    x,
                    0.95,
                ),
                "max",
            ]
        )
        .reset_index()
    )

    grouped.columns = [
        "scenario_label",
        "from_year",
        "to_year",
        "transition",
        "switched_category",
        "component",
        "n",
        "mean_absolute_change_mm",
        "median_absolute_change_mm",
        "p90_absolute_change_mm",
        "p95_absolute_change_mm",
        "maximum_absolute_change_mm",
    ]

    return grouped


# ============================================================
# 10. FLAG SUSPICIOUS LARGE JUMPS
# ============================================================

def flag_suspicious_examples(
    jump_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each SSP / year-pair / component, define a high-jump threshold
    from trajectories that DO NOT switch category.

    A switching trajectory is flagged when its absolute change exceeds
    the 99th percentile of same-category trajectories.

    This is a diagnostic flag, not a declaration of physical implausibility.
    """

    thresholds = (
        jump_df.loc[
            ~jump_df[
                "switched_category"
            ]
        ]
        .groupby(
            [
                "scenario_label",
                "from_year",
                "to_year",
                "component",
            ]
        )[
            "absolute_change_mm"
        ]
        .quantile(
            0.99
        )
        .rename(
            "same_category_p99_absolute_change_mm"
        )
        .reset_index()
    )

    switched = jump_df.loc[
        jump_df[
            "switched_category"
        ]
    ].copy()

    switched = switched.merge(
        thresholds,
        on=[
            "scenario_label",
            "from_year",
            "to_year",
            "component",
        ],
        how="left",
        validate="many_to_one",
    )

    switched[
        "exceeds_same_category_p99"
    ] = (
        switched[
            "absolute_change_mm"
        ]
        > switched[
            "same_category_p99_absolute_change_mm"
        ]
    )

    return (
        switched.loc[
            switched[
                "exceeds_same_category_p99"
            ]
        ]
        .sort_values(
            "absolute_change_mm",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 11. PRINT KEY SUMMARIES
# ============================================================

def print_key_summaries(
    sequences: pd.DataFrame,
    climate_diag: pd.DataFrame,
    suspicious: pd.DataFrame,
):

    print(
        "\n"
        + "=" * 72
    )
    print(
        "MOST COMMON STATE SEQUENCES BY SSP"
    )
    print(
        "=" * 72
    )

    for scenario, group in sequences.groupby(
        "scenario_label"
    ):

        print(
            f"\n{scenario}"
        )

        print(
            group.head(
                8
            )[
                [
                    "sequence",
                    "fraction_within_ssp",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "CATEGORY DIRECTION VS CLIMATE-PROBABILITY DIRECTION"
    )
    print(
        "=" * 72
    )

    agreement = (
        climate_diag.groupby(
            [
                "scenario_label",
                "from_year",
                "to_year",
            ]
        )[
            "direction_agrees_with_probability_shift"
        ]
        .mean()
    )

    print(
        agreement.round(
            4
        )
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "SUSPICIOUS SWITCHING JUMPS"
    )
    print(
        "=" * 72
    )

    if suspicious.empty:

        print(
            "\nNo switching trajectories exceed the 99th-percentile "
            "same-category jump threshold."
        )

    else:

        print(
            f"\nFlagged rows: {len(suspicious):,}"
        )

        print(
            suspicious[
                [
                    "fair_sample_id",
                    "scenario_label",
                    "from_year",
                    "to_year",
                    "transition",
                    "component",
                    "absolute_change_mm",
                    "same_category_p99_absolute_change_mm",
                ]
            ]
            .head(
                25
            )
            .to_string(
                index=False
            )
        )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "CATEGORICAL CLIMATE-SPACE TRANSITION DIAGNOSTICS"
    )
    print(
        "=" * 72
    )

    selection, anchors = load_inputs()

    wide = make_wide_selection(
        selection
    )

    anchors = (
        anchors
        .sort_values(
            "fair_sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    sequences = transition_sequences_by_ssp(
        wide
    )

    transition_matrix = transition_matrix_by_ssp(
        wide
    )

    climate_diag = climate_direction_diagnostics(
        wide
    )

    jump_df = slr_jump_diagnostics(
        wide=wide,
        anchors=anchors,
    )

    jump_summary = summarize_slr_jumps(
        jump_df
    )

    suspicious = flag_suspicious_examples(
        jump_df
    )

    # Save.
    sequences.to_csv(
        OUTPUT_DIR
        / "transition_sequences_by_ssp.csv",
        index=False,
    )

    transition_matrix.to_csv(
        OUTPUT_DIR
        / "category_transition_matrix_by_ssp.csv",
        index=False,
    )

    climate_diag.to_csv(
        OUTPUT_DIR
        / "climate_direction_diagnostics.csv",
        index=False,
    )

    jump_df.to_csv(
        OUTPUT_DIR
        / "slr_jump_diagnostics.csv",
        index=False,
    )

    jump_summary.to_csv(
        OUTPUT_DIR
        / "slr_jump_summary_by_transition.csv",
        index=False,
    )

    suspicious.to_csv(
        OUTPUT_DIR
        / "suspicious_transition_examples.csv",
        index=False,
    )

    print_key_summaries(
        sequences=sequences,
        climate_diag=climate_diag,
        suspicious=suspicious,
    )

    print(
        "\nOutputs saved to:"
    )
    print(
        OUTPUT_DIR
    )

    return {
        "wide": wide,
        "sequences": sequences,
        "transition_matrix": transition_matrix,
        "climate_diagnostics": climate_diag,
        "jump_diagnostics": jump_df,
        "jump_summary": jump_summary,
        "suspicious": suspicious,
    }


if __name__ == "__main__":

    results = main()
