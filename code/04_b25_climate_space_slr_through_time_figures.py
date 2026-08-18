#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_b25_climate_space_slr_through_time_figures.py

Summarize and plot the climate-space / percentile-matched B25
ice-sheet SLR trajectories produced by Script 03.

Outputs:
    - decadal percentile summary by SSP and component
    - key-year table
    - all-component figure
    - compact Total AIS / GrIS / Total ice-sheet SLR figure
    - climate-space weight diagnostics copied into a concise summary

@author: elizabeth
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from B25data_processing import project_dir


PROJECT_DIR = Path(
    project_dir
)

TRAJECTORY_FILE = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "03_climate_space_temporal_interpolation_pchip"
    / "b25_climate_space_pchip_decadal_trajectories.csv"
)

RANK_DIAGNOSTIC_FILE = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "02_climate_space_percentile_interpolation"
    / "b25_climate_space_temporal_percentile_diagnostics.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "04_climate_space_slr_through_time"
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

SSP_ORDER = [
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP3-7.0",
    "SSP5-8.5",
]

FIG_DPI = 300


def load_data():

    if not TRAJECTORY_FILE.exists():

        raise FileNotFoundError(
            f"Trajectory file not found:\n{TRAJECTORY_FILE}"
        )

    trajectories = pd.read_csv(
        TRAJECTORY_FILE
    )

    ranks = None

    if RANK_DIAGNOSTIC_FILE.exists():

        ranks = pd.read_csv(
            RANK_DIAGNOSTIC_FILE
        )

    return (
        trajectories,
        ranks,
    )


def calculate_summary(
    trajectories,
):

    records = []

    for (
        scenario_label,
        component,
        year,
    ), group in trajectories.groupby(
        [
            "scenario_label",
            "component",
            "year",
        ]
    ):

        values = group[
            "value_mm"
        ].to_numpy(
            dtype=float
        )

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
                "scenario_label": scenario_label,
                "component": component,
                "year": year,
                "n_samples": len(
                    values
                ),
                "mean_m": np.mean(
                    values
                ) / 1000.0,
                "p05_m": q[0] / 1000.0,
                "p17_m": q[1] / 1000.0,
                "p50_m": q[2] / 1000.0,
                "p83_m": q[3] / 1000.0,
                "p95_m": q[4] / 1000.0,
            }
        )

    return pd.DataFrame(
        records
    )


def draw_panel(
    ax,
    group,
    label,
):

    group = group.sort_values(
        "year"
    )

    x = group[
        "year"
    ].to_numpy()

    ax.fill_between(
        x,
        group[
            "p05_m"
        ],
        group[
            "p95_m"
        ],
        alpha=0.08,
        linewidth=0,
    )

    ax.fill_between(
        x,
        group[
            "p17_m"
        ],
        group[
            "p83_m"
        ],
        alpha=0.18,
        linewidth=0,
    )

    ax.plot(
        x,
        group[
            "p50_m"
        ],
        linewidth=2.0,
        label=label,
    )


def make_figure(
    summary,
    components,
    filename,
    figsize,
):

    fig, axes = plt.subplots(
        len(
            components
        ),
        1,
        figsize=figsize,
        sharex=True,
    )

    if len(
        components
    ) == 1:
        axes = [
            axes
        ]

    for ax, component in zip(
        axes,
        components,
    ):

        subset = summary.loc[
            summary[
                "component"
            ]
            == component
        ]

        for scenario in SSP_ORDER:

            group = subset.loc[
                subset[
                    "scenario_label"
                ]
                == scenario
            ]

            draw_panel(
                ax,
                group,
                scenario,
            )

        ax.axhline(
            0,
            linewidth=0.8,
            linestyle="--",
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

    axes[
        0
    ].legend(
        loc="upper left",
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved:\n{path}"
    )


def summarize_rank_diagnostics(
    ranks,
):

    if ranks is None:

        return None

    weights = (
        ranks.groupby(
            [
                "scenario_label",
                "year",
            ]
        )[
            [
                "weight_low",
                "weight_moderate",
                "weight_high",
                "interpolated_percentile",
            ]
        ]
        .mean()
        .reset_index()
    )

    correlations = []

    for scenario, group in ranks.groupby(
        "scenario_label"
    ):

        wide = group.pivot(
            index="fair_sample_id",
            columns="year",
            values="interpolated_percentile",
        )

        corr = wide.corr(
            method="spearman"
        )

        for y1 in corr.index:

            for y2 in corr.columns:

                correlations.append(
                    {
                        "scenario_label": scenario,
                        "year_1": int(
                            y1
                        ),
                        "year_2": int(
                            y2
                        ),
                        "spearman_rank_correlation": corr.loc[
                            y1,
                            y2,
                        ],
                    }
                )

    return (
        weights,
        pd.DataFrame(
            correlations
        ),
    )


def main():

    trajectories, ranks = load_data()

    summary = calculate_summary(
        trajectories
    )

    summary_file = (
        OUTPUT_DIR
        / "b25_climate_space_slr_decadal_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    key_years = summary.loc[
        summary[
            "year"
        ].isin(
            [
                2050,
                2100,
                2200,
                2300,
            ]
        )
    ]

    key_years.to_csv(
        OUTPUT_DIR
        / "b25_climate_space_slr_key_years.csv",
        index=False,
    )

    make_figure(
        summary=summary,
        components=COMPONENT_ORDER,
        filename="b25_climate_space_slr_all_components.png",
        figsize=(
            8.5,
            13.0,
        ),
    )

    make_figure(
        summary=summary,
        components=COMPACT_COMPONENT_ORDER,
        filename="b25_climate_space_slr_compact.png",
        figsize=(
            8.5,
            8.5,
        ),
    )

    rank_summary = summarize_rank_diagnostics(
        ranks
    )

    if rank_summary is not None:

        weights, correlations = rank_summary

        weights.to_csv(
            OUTPUT_DIR
            / "b25_climate_space_mean_weights_by_ssp_year.csv",
            index=False,
        )

        correlations.to_csv(
            OUTPUT_DIR
            / "b25_climate_space_temporal_rank_correlations.csv",
            index=False,
        )

        print(
            "\nMean climate-space weights:"
        )

        print(
            weights.round(
                4
            ).to_string(
                index=False
            )
        )

    print(
        "\nDone."
    )

    return {
        "summary": summary,
        "ranks": ranks,
    }


if __name__ == "__main__":

    results = main()
