#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_b25_categorical_climate_space_slr_through_time_figures.py

Summarize and plot SLR trajectories from the categorical
GSAT + cumulative-warming climate-space method.

@author: elizabeth
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from B25data_processing import project_dir


PROJECT_DIR = Path(project_dir)

TRAJECTORY_FILE = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "03_categorical_climate_space_temporal_interpolation_pchip"
    / "b25_categorical_climate_space_pchip_decadal_trajectories.csv"
)

SELECTION_FILE = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "02_climate_space_categorical_selection"
    / "b25_categorical_climate_space_selection_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "04_categorical_climate_space_slr_through_time"
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

    selection = None

    if SELECTION_FILE.exists():
        selection = pd.read_csv(
            SELECTION_FILE
        )

    return (
        trajectories,
        selection,
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

    path = FIGURE_DIR / filename

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


def summarize_selection(
    selection,
):

    if selection is None:
        return None

    realized = (
        selection.groupby(
            [
                "scenario_label",
                "year",
            ]
        )[
            "selected_library"
        ]
        .value_counts(
            normalize=True
        )
        .unstack(
            fill_value=0.0
        )
        .reset_index()
    )

    probabilities = (
        selection.groupby(
            [
                "scenario_label",
                "year",
            ]
        )[
            [
                "probability_low",
                "probability_moderate",
                "probability_high",
            ]
        ]
        .mean()
        .reset_index()
    )

    return (
        realized,
        probabilities,
    )


def main():

    trajectories, selection = load_data()

    summary = calculate_summary(
        trajectories
    )

    summary.to_csv(
        OUTPUT_DIR
        / "b25_categorical_climate_space_slr_decadal_summary.csv",
        index=False,
    )

    summary.loc[
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
    ].to_csv(
        OUTPUT_DIR
        / "b25_categorical_climate_space_slr_key_years.csv",
        index=False,
    )

    make_figure(
        summary,
        COMPONENT_ORDER,
        "b25_categorical_climate_space_slr_all_components.png",
        (
            8.5,
            13.0,
        ),
    )

    make_figure(
        summary,
        COMPACT_COMPONENT_ORDER,
        "b25_categorical_climate_space_slr_compact.png",
        (
            8.5,
            8.5,
        ),
    )

    selection_summary = summarize_selection(
        selection
    )

    if selection_summary is not None:

        realized, probabilities = selection_summary

        realized.to_csv(
            OUTPUT_DIR
            / "realized_category_fractions_by_ssp_year.csv",
            index=False,
        )

        probabilities.to_csv(
            OUTPUT_DIR
            / "mean_category_probabilities_by_ssp_year.csv",
            index=False,
        )

        print(
            "\nRealized selection fractions:"
        )

        print(
            realized.round(
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
        "selection": selection,
    }


if __name__ == "__main__":

    results = main()
