#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose temporal dependence after persistent-selector resampling.

Goal
----
Test whether switching among the 2.0°C, 3.5°C, and 5.0°C temporally coupled
B25 libraries degrades the temporal dependence structure of the resulting
FaIR-conditioned ice-sheet trajectories.

The script compares:

1. Baseline temporal correlations within each Step-2 coupled B25 library.
2. Final temporal correlations after persistent-selector resampling,
   separately by SSP.
3. Final correlations split by:
       - same library at both years
       - switched library between years

Responses analyzed:
    EAIS total
    WAIS total
    GrIS total
    Total AIS

Year pairs:
    2100–2200
    2100–2300
    2200–2300

Outputs:
    CSV diagnostic tables
    heatmaps of Spearman correlations
    bar plots comparing baseline vs final correlations
    switching-effect tables

Expected prerequisite:
    Run interpolate_fair_sej_persistent_selector.py first.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_DIR = Path(
    "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025"
)

sys.path.insert(
    0,
    str(PROJECT_DIR),
)

import B25data_processing as b25


PERSISTENT_DIR = (
    PROJECT_DIR
    / "output"
    / "fair_sej_persistent_selector"
)

RESAMPLED_TOTALS_FILE = (
    PERSISTENT_DIR
    / "fair_sej_resampled_totals.csv"
)

LIBRARY_CHOICES_FILE = (
    PERSISTENT_DIR
    / "fair_sej_library_choices.csv"
)

STEP2_LIBRARIES = {
    "2.0°C": (
        PROJECT_DIR
        / "output"
        / "temporal_coupling"
        / "total_ais_common_rank"
        / "b25_2.0C_dp21_rcp26"
        / "b25_2.0C_dp21_rcp26_coupled_samples.csv"
    ),
    "3.5°C": (
        PROJECT_DIR
        / "output"
        / "temporal_coupling"
        / "total_ais_common_rank"
        / "b25_3.5C_dp21_rcp45"
        / "b25_3.5C_dp21_rcp45_coupled_samples.csv"
    ),
    "5.0°C": (
        PROJECT_DIR
        / "output"
        / "temporal_coupling"
        / "total_ais_common_rank"
        / "b25_5.0C_dp21_rcp85"
        / "b25_5.0C_dp21_rcp85_coupled_samples.csv"
    ),
}

WARMING_FOR_LIBRARY = {
    "2.0°C": 2.0,
    "3.5°C": 3.5,
    "5.0°C": 5.0,
}

RESPONSES = [
    "EAIS",
    "WAIS",
    "GrIS",
    "Total AIS",
]

YEAR_PAIRS = [
    (2100, 2200),
    (2100, 2300),
    (2200, 2300),
]

SSP_ORDER = [
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP3-7.0",
    "SSP5-8.5",
]

OUTPUT_DIR = (
    PERSISTENT_DIR
    / "temporal_dependence_diagnostics"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. HELPERS
# ============================================================

def require_file(path: Path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def corr_pair(
    wide: pd.DataFrame,
    year1: int,
    year2: int,
) -> tuple[float, float, int]:
    pair = (
        wide[
            [year1, year2]
        ]
        .dropna()
    )

    n = len(pair)

    if n < 3:
        return np.nan, np.nan, n

    spearman = pair[
        year1
    ].corr(
        pair[year2],
        method="spearman",
    )

    pearson = pair[
        year1
    ].corr(
        pair[year2],
        method="pearson",
    )

    return (
        float(spearman),
        float(pearson),
        int(n),
    )


def pivot_response(
    df: pd.DataFrame,
    value_col: str,
    index_cols: list[str],
) -> pd.DataFrame:
    return (
        df
        .pivot(
            index=index_cols,
            columns="year",
            values=value_col,
        )
        .reset_index()
    )


# ============================================================
# 3. LOAD FINAL PERSISTENT-SELECTOR OUTPUTS
# ============================================================

require_file(
    RESAMPLED_TOTALS_FILE
)

require_file(
    LIBRARY_CHOICES_FILE
)

totals = pd.read_csv(
    RESAMPLED_TOTALS_FILE
)

choices = pd.read_csv(
    LIBRARY_CHOICES_FILE
)

required_total_columns = {
    "fair_sample_id",
    "scenario_label",
    "year",
    "ice_sheet",
    "value",
}

missing = (
    required_total_columns
    - set(
        totals.columns
    )
)

if missing:
    raise KeyError(
        "Missing required columns from "
        f"{RESAMPLED_TOTALS_FILE.name}: "
        f"{sorted(missing)}"
    )

required_choice_columns = {
    "fair_sample_id",
    "year",
    "selected_library_label",
}

missing = (
    required_choice_columns
    - set(
        choices.columns
    )
)

if missing:
    raise KeyError(
        "Missing required columns from "
        f"{LIBRARY_CHOICES_FILE.name}: "
        f"{sorted(missing)}"
    )

totals = totals.loc[
    totals[
        "ice_sheet"
    ].isin(
        RESPONSES
    )
].copy()


# ============================================================
# 4. BASELINE TEMPORAL CORRELATIONS WITHIN STEP-2 LIBRARIES
# ============================================================

baseline_records = []

for library_label, path in STEP2_LIBRARIES.items():

    require_file(
        path
    )

    data = b25.load_and_prepare_data(
        file_path=str(path)
    )

    total_ext = data[
        "mc_total_ext"
    ].copy()

    target_warming = WARMING_FOR_LIBRARY[
        library_label
    ]

    subset = total_ext.loc[
        np.isclose(
            total_ext["warming"],
            target_warming,
        )
        & total_ext[
            "ice_sheet"
        ].isin(
            RESPONSES
        )
        & total_ext[
            "year"
        ].isin(
            [
                2100,
                2200,
                2300,
            ]
        )
    ].copy()

    for response in RESPONSES:

        response_df = subset.loc[
            subset[
                "ice_sheet"
            ]
            == response
        ]

        wide = (
            response_df
            .pivot(
                index="sample_id",
                columns="year",
                values="value",
            )
        )

        for year1, year2 in YEAR_PAIRS:

            spearman, pearson, n = corr_pair(
                wide,
                year1,
                year2,
            )

            baseline_records.append(
                {
                    "library": library_label,
                    "response": response,
                    "year1": year1,
                    "year2": year2,
                    "year_pair": (
                        f"{year1}-{year2}"
                    ),
                    "n": n,
                    "spearman": spearman,
                    "pearson": pearson,
                }
            )

baseline = pd.DataFrame(
    baseline_records
)

baseline.to_csv(
    OUTPUT_DIR
    / "step2_library_temporal_correlations.csv",
    index=False,
)


# ============================================================
# 5. FINAL TEMPORAL CORRELATIONS BY SSP
# ============================================================

final_records = []

for scenario_label in SSP_ORDER:

    scenario_df = totals.loc[
        totals[
            "scenario_label"
        ]
        == scenario_label
    ]

    for response in RESPONSES:

        response_df = scenario_df.loc[
            scenario_df[
                "ice_sheet"
            ]
            == response
        ]

        wide = (
            response_df
            .pivot(
                index="fair_sample_id",
                columns="year",
                values="value",
            )
        )

        for year1, year2 in YEAR_PAIRS:

            spearman, pearson, n = corr_pair(
                wide,
                year1,
                year2,
            )

            final_records.append(
                {
                    "scenario_label": scenario_label,
                    "response": response,
                    "year1": year1,
                    "year2": year2,
                    "year_pair": (
                        f"{year1}-{year2}"
                    ),
                    "n": n,
                    "spearman": spearman,
                    "pearson": pearson,
                }
            )

final_corr = pd.DataFrame(
    final_records
)

final_corr.to_csv(
    OUTPUT_DIR
    / "persistent_selector_temporal_correlations_by_ssp.csv",
    index=False,
)


# ============================================================
# 6. BUILD SWITCH / NO-SWITCH LABELS
# ============================================================

choice_wide = (
    choices[
        [
            "fair_sample_id",
            "year",
            "selected_library_label",
        ]
    ]
    .pivot(
        index="fair_sample_id",
        columns="year",
        values="selected_library_label",
    )
)

switch_frames = []

for year1, year2 in YEAR_PAIRS:

    frame = pd.DataFrame(
        {
            "fair_sample_id": (
                choice_wide.index
            ),
            "year1": year1,
            "year2": year2,
            "year_pair": (
                f"{year1}-{year2}"
            ),
            "library_year1": (
                choice_wide[
                    year1
                ].values
            ),
            "library_year2": (
                choice_wide[
                    year2
                ].values
            ),
        }
    )

    frame[
        "switched_library"
    ] = (
        frame[
            "library_year1"
        ]
        != frame[
            "library_year2"
        ]
    )

    switch_frames.append(
        frame
    )

switch_table = pd.concat(
    switch_frames,
    ignore_index=True,
)


# ============================================================
# 7. FINAL CORRELATIONS SPLIT BY SWITCHING STATUS
# ============================================================

switch_corr_records = []

for scenario_label in SSP_ORDER:

    scenario_totals = totals.loc[
        totals[
            "scenario_label"
        ]
        == scenario_label
    ]

    for response in RESPONSES:

        response_df = scenario_totals.loc[
            scenario_totals[
                "ice_sheet"
            ]
            == response
        ]

        wide = (
            response_df
            .pivot(
                index="fair_sample_id",
                columns="year",
                values="value",
            )
            .reset_index()
        )

        for year1, year2 in YEAR_PAIRS:

            status = switch_table.loc[
                switch_table[
                    "year_pair"
                ]
                == f"{year1}-{year2}"
            ][
                [
                    "fair_sample_id",
                    "switched_library",
                    "library_year1",
                    "library_year2",
                ]
            ]

            merged = wide.merge(
                status,
                on="fair_sample_id",
                how="inner",
            )

            for switched in [
                False,
                True,
            ]:

                group = merged.loc[
                    merged[
                        "switched_library"
                    ]
                    == switched
                ]

                n = len(
                    group
                )

                if n >= 3:
                    spearman = group[
                        year1
                    ].corr(
                        group[
                            year2
                        ],
                        method="spearman",
                    )

                    pearson = group[
                        year1
                    ].corr(
                        group[
                            year2
                        ],
                        method="pearson",
                    )
                else:
                    spearman = np.nan
                    pearson = np.nan

                switch_corr_records.append(
                    {
                        "scenario_label": scenario_label,
                        "response": response,
                        "year1": year1,
                        "year2": year2,
                        "year_pair": (
                            f"{year1}-{year2}"
                        ),
                        "switch_status": (
                            "switched"
                            if switched
                            else "same library"
                        ),
                        "n": n,
                        "spearman": spearman,
                        "pearson": pearson,
                    }
                )

switch_corr = pd.DataFrame(
    switch_corr_records
)

switch_corr.to_csv(
    OUTPUT_DIR
    / "temporal_correlations_split_by_library_switching.csv",
    index=False,
)


# ============================================================
# 8. SWITCH TRANSITION FREQUENCIES
# ============================================================

choice_with_scenario = (
    choices[
        [
            "fair_sample_id",
            "scenario_label",
            "year",
            "selected_library_label",
        ]
    ]
    .pivot(
        index=[
            "fair_sample_id",
            "scenario_label",
        ],
        columns="year",
        values="selected_library_label",
    )
    .reset_index()
)

transition_records = []

for year1, year2 in YEAR_PAIRS:

    transition = (
        choice_with_scenario
        .groupby(
            [
                "scenario_label",
                year1,
                year2,
            ],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
        .rename(
            columns={
                year1: "library_year1",
                year2: "library_year2",
            }
        )
    )

    transition[
        "year_pair"
    ] = (
        f"{year1}-{year2}"
    )

    transition_records.append(
        transition
    )

transitions = pd.concat(
    transition_records,
    ignore_index=True,
)

transitions[
    "percent_within_ssp_pair"
] = (
    transitions[
        "n"
    ]
    / transitions.groupby(
        [
            "scenario_label",
            "year_pair",
        ],
        observed=True,
    )[
        "n"
    ].transform(
        "sum"
    )
    * 100
)

transitions.to_csv(
    OUTPUT_DIR
    / "library_transition_frequencies.csv",
    index=False,
)


# ============================================================
# 9. SUMMARY: FINAL MINUS RELEVANT BASELINE RANGE
# ============================================================

# Since each SSP is a mixture of libraries, there is no single exact Step-2
# target correlation. As a descriptive benchmark, report whether the final
# correlation falls within the min-max range of the three Step-2 libraries.

baseline_range = (
    baseline
    .groupby(
        [
            "response",
            "year_pair",
        ],
        observed=True,
    )
    .agg(
        baseline_spearman_min=(
            "spearman",
            "min",
        ),
        baseline_spearman_max=(
            "spearman",
            "max",
        ),
        baseline_spearman_mean=(
            "spearman",
            "mean",
        ),
    )
    .reset_index()
)

comparison = final_corr.merge(
    baseline_range,
    on=[
        "response",
        "year_pair",
    ],
    how="left",
)

comparison[
    "difference_from_baseline_mean"
] = (
    comparison[
        "spearman"
    ]
    - comparison[
        "baseline_spearman_mean"
    ]
)

comparison[
    "below_step2_library_range"
] = (
    comparison[
        "spearman"
    ]
    < comparison[
        "baseline_spearman_min"
    ]
)

comparison[
    "above_step2_library_range"
] = (
    comparison[
        "spearman"
    ]
    > comparison[
        "baseline_spearman_max"
    ]
)

comparison.to_csv(
    OUTPUT_DIR
    / "final_vs_step2_temporal_correlation_comparison.csv",
    index=False,
)


# ============================================================
# 10. FIGURE: FINAL SPEARMAN HEATMAP, TOTAL AIS
# ============================================================

total_ais_final = final_corr.loc[
    final_corr[
        "response"
    ]
    == "Total AIS"
].copy()

heat = (
    total_ais_final
    .pivot(
        index="scenario_label",
        columns="year_pair",
        values="spearman",
    )
    .reindex(
        SSP_ORDER
    )
    .reindex(
        columns=[
            "2100-2200",
            "2100-2300",
            "2200-2300",
        ]
    )
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

image = ax.imshow(
    heat.to_numpy(
        dtype=float
    ),
    aspect="auto",
    vmin=0,
    vmax=1,
)

ax.set_xticks(
    np.arange(
        len(
            heat.columns
        )
    )
)

ax.set_xticklabels(
    heat.columns
)

ax.set_yticks(
    np.arange(
        len(
            heat.index
        )
    )
)

ax.set_yticklabels(
    heat.index
)

for i in range(
    heat.shape[0]
):
    for j in range(
        heat.shape[1]
    ):
        value = heat.iloc[
            i,
            j,
        ]

        if pd.notna(
            value
        ):
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

ax.set_title(
    "Total AIS temporal Spearman correlation\n"
    "after persistent-selector resampling"
)

colorbar = fig.colorbar(
    image,
    ax=ax,
)

colorbar.set_label(
    "Spearman correlation"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "total_ais_post_resampling_spearman_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 11. FIGURE: STEP-2 BASELINES BY LIBRARY, TOTAL AIS
# ============================================================

baseline_total = baseline.loc[
    baseline[
        "response"
    ]
    == "Total AIS"
].copy()

x = np.arange(
    len(
        YEAR_PAIRS
    )
)

width = 0.24

fig, ax = plt.subplots(
    figsize=(9, 5)
)

for j, library in enumerate(
    [
        "2.0°C",
        "3.5°C",
        "5.0°C",
    ]
):

    group = (
        baseline_total.loc[
            baseline_total[
                "library"
            ]
            == library
        ]
        .set_index(
            "year_pair"
        )
        .reindex(
            [
                "2100-2200",
                "2100-2300",
                "2200-2300",
            ]
        )
    )

    ax.bar(
        x
        + (
            j - 1
        )
        * width,
        group[
            "spearman"
        ],
        width=width,
        label=library,
    )

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        "2100–2200",
        "2100–2300",
        "2200–2300",
    ]
)

ax.set_ylim(
    0,
    1,
)

ax.set_ylabel(
    "Spearman correlation"
)

ax.set_title(
    "Step-2 Total AIS temporal correlation\n"
    "within each coupled B25 library"
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="y",
    alpha=0.2,
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "total_ais_step2_library_spearman.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 12. FIGURE: SWITCHED VS SAME-LIBRARY, TOTAL AIS
# ============================================================

switch_total = switch_corr.loc[
    switch_corr[
        "response"
    ]
    == "Total AIS"
].copy()

for year_pair in [
    "2100-2200",
    "2100-2300",
    "2200-2300",
]:

    subset = switch_total.loc[
        switch_total[
            "year_pair"
        ]
        == year_pair
    ]

    x = np.arange(
        len(
            SSP_ORDER
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for j, status in enumerate(
        [
            "same library",
            "switched",
        ]
    ):

        group = (
            subset.loc[
                subset[
                    "switch_status"
                ]
                == status
            ]
            .set_index(
                "scenario_label"
            )
            .reindex(
                SSP_ORDER
            )
        )

        ax.bar(
            x
            + (
                j - 0.5
            )
            * width,
            group[
                "spearman"
            ],
            width=width,
            label=status,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        SSP_ORDER
    )

    ax.set_ylim(
        -0.1,
        1,
    )

    ax.set_ylabel(
        "Spearman correlation"
    )

    ax.set_title(
        "Total AIS temporal dependence:\n"
        f"{year_pair} same-library vs switched trajectories"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / (
            "total_ais_switch_effect_"
            f"{year_pair.replace('-', '_')}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# 13. CONSOLE SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STEP-2 TOTAL AIS BASELINE SPEARMAN CORRELATIONS"
)

print(
    "=" * 80
)

print(
    baseline_total[
        [
            "library",
            "year_pair",
            "spearman",
            "pearson",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)

print(
    "\n"
    + "=" * 80
)

print(
    "POST-RESAMPLING TOTAL AIS SPEARMAN CORRELATIONS BY SSP"
)

print(
    "=" * 80
)

print(
    total_ais_final[
        [
            "scenario_label",
            "year_pair",
            "spearman",
            "pearson",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)

print(
    "\n"
    + "=" * 80
)

print(
    "TOTAL AIS: SAME LIBRARY VS SWITCHED"
)

print(
    "=" * 80
)

print(
    switch_total[
        [
            "scenario_label",
            "year_pair",
            "switch_status",
            "n",
            "spearman",
            "pearson",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL CORRELATIONS BELOW ALL STEP-2 LIBRARY BASELINES"
)

print(
    "=" * 80
)

flagged = comparison.loc[
    comparison[
        "below_step2_library_range"
    ]
].copy()

if flagged.empty:
    print(
        "None."
    )
else:
    print(
        flagged[
            [
                "scenario_label",
                "response",
                "year_pair",
                "spearman",
                "baseline_spearman_min",
                "baseline_spearman_max",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

print(
    f"\nOutputs saved to:\n"
    f"{OUTPUT_DIR}"
)
