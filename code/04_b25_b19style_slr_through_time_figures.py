#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_b25_b19style_slr_through_time_figures.py

Calculate and plot B25 ice-sheet contributions to sea-level rise through time
using the validated PCHIP-based temporal interpolation.

INPUT
-----
output/b19style_baseline/03_temporal_interpolation_pchip/
    b25_b19style_pchip_decadal_trajectories.csv

EXPECTED COLUMNS
----------------
trajectory_id
scenario_role       LOW / HIGH
warming_anchor      2.0 / 5.0
year                2000, 2010, ..., 2300
component           EAIS / WAIS / GrIS / Total AIS / Total ice-sheet SLR
value_mm

OUTPUT
------
1. A recalculated decadal percentile summary.
2. Figure 1:
       5 vertically stacked panels:
           EAIS
           WAIS
           Total AIS
           GrIS
           Total ice-sheet SLR

       LOW and HIGH shown together in each panel.
       Solid line   = median
       Dark band    = 17th-83rd percentile
       Light band   = 5th-95th percentile

3. Figure 2:
       Total AIS + GrIS + Total ice-sheet SLR only,
       for a more compact summary figure.

4. Figure 3:
       LOW and HIGH in separate columns, with:
           EAIS
           WAIS
           GrIS
           Total AIS
           Total ice-sheet SLR
       as rows.

All values are plotted in meters of global mean sea-level equivalent.

@author: elizabeth
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from B25data_processing import project_dir


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_DIR = Path(project_dir)

INPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "03_temporal_interpolation_pchip"
)

TRAJECTORY_FILE = (
    INPUT_DIR
    / "b25_b19style_pchip_decadal_trajectories.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "04_slr_through_time"
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


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

COMPONENT_ORDER = [
    "EAIS",
    "WAIS",
    "Total AIS",
    "GrIS",
    "Total ice-sheet SLR",
]

COMPACT_COMPONENT_ORDER = [
    "Total AIS",
    "GrIS",
    "Total ice-sheet SLR",
]

SCENARIO_ORDER = [
    "LOW",
    "HIGH",
]

SCENARIO_LABELS = {
    "LOW": "Low warming (2°C)",
    "HIGH": "High warming (5°C)",
}

# Keep the same colors as the user's existing low/high plotting convention
# if desired. Change only these two values if needed.
LOW_COLOR = "#4C78A8"
HIGH_COLOR = "#E45756"

MEDIAN_LINEWIDTH = 2.2

INNER_ALPHA = 0.28
OUTER_ALPHA = 0.12

FIG_DPI = 300


# ============================================================
# 2. LOAD TRAJECTORIES
# ============================================================

def load_trajectories(
    file_path: Path,
) -> pd.DataFrame:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Trajectory file not found:\n{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    required = {
        "trajectory_id",
        "scenario_role",
        "warming_anchor",
        "year",
        "component",
        "value_mm",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise KeyError(
            "Trajectory file is missing required columns:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    df = df.copy()

    df[
        "scenario_role"
    ] = df[
        "scenario_role"
    ].astype(str)

    df[
        "year"
    ] = df[
        "year"
    ].astype(int)

    df[
        "value_mm"
    ] = pd.to_numeric(
        df[
            "value_mm"
        ],
        errors="raise",
    )

    if not np.isfinite(
        df[
            "value_mm"
        ].to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Non-finite SLR values found."
        )

    return df


# ============================================================
# 3. CALCULATE THROUGH-TIME SLR SUMMARY
# ============================================================

def calculate_slr_summary(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for (
        scenario_role,
        component,
        year,
    ), group in trajectories.groupby(
        [
            "scenario_role",
            "component",
            "year",
        ],
        sort=False,
    ):

        values_mm = group[
            "value_mm"
        ].to_numpy(
            dtype=float
        )

        q = np.quantile(
            values_mm,
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
                "warming_anchor": (
                    2.0
                    if scenario_role == "LOW"
                    else 5.0
                ),
                "component": component,
                "year": year,
                "n_samples": len(
                    values_mm
                ),
                "mean_mm": np.mean(
                    values_mm
                ),
                "p05_mm": q[0],
                "p17_mm": q[1],
                "p50_mm": q[2],
                "p83_mm": q[3],
                "p95_mm": q[4],
            }
        )

    summary = pd.DataFrame(
        records
    )

    summary[
        "component"
    ] = pd.Categorical(
        summary[
            "component"
        ],
        categories=COMPONENT_ORDER,
        ordered=True,
    )

    summary[
        "scenario_role"
    ] = pd.Categorical(
        summary[
            "scenario_role"
        ],
        categories=SCENARIO_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values(
            [
                "component",
                "scenario_role",
                "year",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # Add meter versions for plotting and later use.
    for column in [
        "mean_mm",
        "p05_mm",
        "p17_mm",
        "p50_mm",
        "p83_mm",
        "p95_mm",
    ]:

        summary[
            column.replace(
                "_mm",
                "_m",
            )
        ] = (
            summary[
                column
            ]
            / 1000.0
        )

    return summary


# ============================================================
# 4. CHECK ANCHORS / BASIC CONSISTENCY
# ============================================================

def validate_summary(
    summary: pd.DataFrame,
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SLR THROUGH-TIME SUMMARY"
    )

    print(
        "=" * 72
    )

    counts = (
        summary
        .groupby(
            [
                "scenario_role",
                "component",
            ],
            observed=True,
        )[
            "year"
        ]
        .nunique()
    )

    print(
        "\nNumber of decades per scenario/component:"
    )

    print(
        counts
    )

    if not (
        counts
        == 31
    ).all():

        raise ValueError(
            "Expected 31 decadal values from 2000 through 2300."
        )

    print(
        "\nMedian Total ice-sheet SLR at elicitation years (m):"
    )

    check = (
        summary.loc[
            (
                summary[
                    "component"
                ]
                == "Total ice-sheet SLR"
            )
            & summary[
                "year"
            ].isin(
                [
                    2100,
                    2200,
                    2300,
                ]
            ),
            [
                "scenario_role",
                "year",
                "p50_m",
                "p17_m",
                "p83_m",
                "p05_m",
                "p95_m",
            ]
        ]
    )

    print(
        check.to_string(
            index=False
        )
    )


# ============================================================
# 5. HELPER: DRAW ONE SLR PANEL
# ============================================================

def draw_scenario(
    ax,
    group: pd.DataFrame,
    color: str,
    label: str,
):

    group = group.sort_values(
        "year"
    )

    years = group[
        "year"
    ].to_numpy()

    p05 = group[
        "p05_m"
    ].to_numpy()

    p17 = group[
        "p17_m"
    ].to_numpy()

    p50 = group[
        "p50_m"
    ].to_numpy()

    p83 = group[
        "p83_m"
    ].to_numpy()

    p95 = group[
        "p95_m"
    ].to_numpy()

    # 5-95%
    ax.fill_between(
        years,
        p05,
        p95,
        color=color,
        alpha=OUTER_ALPHA,
        linewidth=0,
    )

    # 17-83%
    ax.fill_between(
        years,
        p17,
        p83,
        color=color,
        alpha=INNER_ALPHA,
        linewidth=0,
    )

    # Median
    ax.plot(
        years,
        p50,
        color=color,
        linewidth=MEDIAN_LINEWIDTH,
        label=label,
    )


# ============================================================
# 6. FIGURE 1: FIVE COMPONENTS
# ============================================================

def make_five_component_figure(
    summary: pd.DataFrame,
):

    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(
            8.5,
            13.0,
        ),
        sharex=True,
    )

    colors = {
        "LOW": LOW_COLOR,
        "HIGH": HIGH_COLOR,
    }

    for ax, component in zip(
        axes,
        COMPONENT_ORDER,
    ):

        component_df = summary.loc[
            summary[
                "component"
            ]
            == component
        ]

        for scenario in SCENARIO_ORDER:

            group = component_df.loc[
                component_df[
                    "scenario_role"
                ]
                == scenario
            ]

            draw_scenario(
                ax=ax,
                group=group,
                color=colors[
                    scenario
                ],
                label=SCENARIO_LABELS[
                    scenario
                ],
            )

        ax.axhline(
            0,
            color="0.45",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )

        ax.set_ylabel(
            f"{component}\n(m)"
        )

        ax.grid(
            axis="y",
            alpha=0.18,
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

    axes[
        -1
    ].set_xlabel(
        "Year"
    )

    axes[
        -1
    ].set_xlim(
        2000,
        2300,
    )

    axes[
        -1
    ].set_xticks(
        [
            2000,
            2050,
            2100,
            2150,
            2200,
            2250,
            2300,
        ]
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=LOW_COLOR,
            linewidth=MEDIAN_LINEWIDTH,
            label=SCENARIO_LABELS[
                "LOW"
            ],
        ),
        Line2D(
            [0],
            [0],
            color=HIGH_COLOR,
            linewidth=MEDIAN_LINEWIDTH,
            label=SCENARIO_LABELS[
                "HIGH"
            ],
        ),
        Patch(
            facecolor="0.5",
            alpha=INNER_ALPHA,
            edgecolor="none",
            label="17–83%",
        ),
        Patch(
            facecolor="0.5",
            alpha=OUTER_ALPHA,
            edgecolor="none",
            label="5–95%",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.995,
        ),
        ncol=4,
        frameon=False,
    )

    fig.subplots_adjust(
        top=0.955,
        bottom=0.07,
        left=0.15,
        right=0.97,
        hspace=0.15,
    )

    outfile = (
        FIGURE_DIR
        / "b25_b19style_slr_through_time_all_components.png"
    )

    fig.savefig(
        outfile,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"\nSaved:\n{outfile}"
    )


# ============================================================
# 7. FIGURE 2: COMPACT THREE-PANEL SUMMARY
# ============================================================

def make_compact_summary_figure(
    summary: pd.DataFrame,
):

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(
            8.5,
            8.5,
        ),
        sharex=True,
    )

    colors = {
        "LOW": LOW_COLOR,
        "HIGH": HIGH_COLOR,
    }

    for ax, component in zip(
        axes,
        COMPACT_COMPONENT_ORDER,
    ):

        component_df = summary.loc[
            summary[
                "component"
            ]
            == component
        ]

        for scenario in SCENARIO_ORDER:

            group = component_df.loc[
                component_df[
                    "scenario_role"
                ]
                == scenario
            ]

            draw_scenario(
                ax=ax,
                group=group,
                color=colors[
                    scenario
                ],
                label=SCENARIO_LABELS[
                    scenario
                ],
            )

        ax.axhline(
            0,
            color="0.45",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )

        ax.set_ylabel(
            f"{component}\n(m)"
        )

        ax.grid(
            axis="y",
            alpha=0.18,
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

    axes[
        -1
    ].set_xlabel(
        "Year"
    )

    axes[
        -1
    ].set_xlim(
        2000,
        2300,
    )

    axes[
        -1
    ].set_xticks(
        [
            2000,
            2050,
            2100,
            2150,
            2200,
            2250,
            2300,
        ]
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=LOW_COLOR,
            linewidth=MEDIAN_LINEWIDTH,
            label=SCENARIO_LABELS[
                "LOW"
            ],
        ),
        Line2D(
            [0],
            [0],
            color=HIGH_COLOR,
            linewidth=MEDIAN_LINEWIDTH,
            label=SCENARIO_LABELS[
                "HIGH"
            ],
        ),
        Patch(
            facecolor="0.5",
            alpha=INNER_ALPHA,
            edgecolor="none",
            label="17–83%",
        ),
        Patch(
            facecolor="0.5",
            alpha=OUTER_ALPHA,
            edgecolor="none",
            label="5–95%",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.995,
        ),
        ncol=4,
        frameon=False,
    )

    fig.subplots_adjust(
        top=0.94,
        bottom=0.09,
        left=0.16,
        right=0.97,
        hspace=0.17,
    )

    outfile = (
        FIGURE_DIR
        / "b25_b19style_slr_through_time_compact.png"
    )

    fig.savefig(
        outfile,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"\nSaved:\n{outfile}"
    )


# ============================================================
# 8. FIGURE 3: LOW / HIGH IN SEPARATE COLUMNS
# ============================================================

def make_scenario_column_figure(
    summary: pd.DataFrame,
):

    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(
            11.5,
            13.0,
        ),
        sharex=True,
        sharey="row",
    )

    colors = {
        "LOW": LOW_COLOR,
        "HIGH": HIGH_COLOR,
    }

    for row_index, component in enumerate(
        COMPONENT_ORDER
    ):

        for col_index, scenario in enumerate(
            SCENARIO_ORDER
        ):

            ax = axes[
                row_index,
                col_index,
            ]

            group = summary.loc[
                (
                    summary[
                        "component"
                    ]
                    == component
                )
                & (
                    summary[
                        "scenario_role"
                    ]
                    == scenario
                )
            ]

            draw_scenario(
                ax=ax,
                group=group,
                color=colors[
                    scenario
                ],
                label=SCENARIO_LABELS[
                    scenario
                ],
            )

            ax.axhline(
                0,
                color="0.45",
                linewidth=0.8,
                linestyle="--",
                zorder=0,
            )

            ax.grid(
                axis="y",
                alpha=0.18,
            )

            ax.spines[
                "top"
            ].set_visible(
                False
            )

            ax.spines[
                "right"
            ].set_visible(
                False
            )

            if col_index == 0:

                ax.set_ylabel(
                    f"{component}\n(m)"
                )

            if row_index == 0:

                ax.set_title(
                    SCENARIO_LABELS[
                        scenario
                    ]
                )

            if row_index == (
                len(
                    COMPONENT_ORDER
                )
                - 1
            ):

                ax.set_xlabel(
                    "Year"
                )

                ax.set_xlim(
                    2000,
                    2300,
                )

                ax.set_xticks(
                    [
                        2000,
                        2050,
                        2100,
                        2150,
                        2200,
                        2250,
                        2300,
                    ]
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="0.35",
            linewidth=MEDIAN_LINEWIDTH,
            label="Median",
        ),
        Patch(
            facecolor="0.5",
            alpha=INNER_ALPHA,
            edgecolor="none",
            label="17–83%",
        ),
        Patch(
            facecolor="0.5",
            alpha=OUTER_ALPHA,
            edgecolor="none",
            label="5–95%",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.995,
        ),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(
        top=0.95,
        bottom=0.07,
        left=0.12,
        right=0.98,
        hspace=0.16,
        wspace=0.12,
    )

    outfile = (
        FIGURE_DIR
        / "b25_b19style_slr_through_time_low_high_columns.png"
    )

    fig.savefig(
        outfile,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"\nSaved:\n{outfile}"
    )


# ============================================================
# 9. SAVE SELECTED YEAR TABLE
# ============================================================

def save_key_year_table(
    summary: pd.DataFrame,
):

    key_years = [
        2050,
        2100,
        2200,
        2300,
    ]

    table = summary.loc[
        summary[
            "year"
        ].isin(
            key_years
        ),
        [
            "scenario_role",
            "warming_anchor",
            "component",
            "year",
            "p05_m",
            "p17_m",
            "p50_m",
            "p83_m",
            "p95_m",
        ]
    ].copy()

    outfile = (
        OUTPUT_DIR
        / "b25_b19style_slr_key_year_percentiles_m.csv"
    )

    table.to_csv(
        outfile,
        index=False,
    )

    print(
        f"\nSaved:\n{outfile}"
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "B25 B19-STYLE SLR CONTRIBUTIONS THROUGH TIME"
    )

    print(
        "=" * 72
    )

    trajectories = load_trajectories(
        TRAJECTORY_FILE
    )

    print(
        f"\nTrajectory rows: "
        f"{len(trajectories):,}"
    )

    print(
        f"Unique trajectory IDs: "
        f"{trajectories['trajectory_id'].nunique():,}"
    )

    summary = calculate_slr_summary(
        trajectories
    )

    validate_summary(
        summary
    )

    summary_file = (
        OUTPUT_DIR
        / "b25_b19style_slr_decadal_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print(
        f"\nSaved:\n{summary_file}"
    )

    save_key_year_table(
        summary
    )

    make_five_component_figure(
        summary
    )

    make_compact_summary_figure(
        summary
    )

    make_scenario_column_figure(
        summary
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DONE"
    )

    print(
        "=" * 72
    )

    return {
        "trajectories": trajectories,
        "summary": summary,
    }


if __name__ == "__main__":

    results = main()
