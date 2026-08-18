#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_b25_categorical_climate_space_slr_through_time_figures_SIMPLIFIED.py

Simplified figures for the categorical GSAT + cumulative-warming
climate-space workflow.

The previous version overlaid all five SSPs in every component panel,
which made the figures difficult to interpret.

This version produces:

FIGURE 1
--------
A 5 x 5 small-multiples figure:
    rows    = ice-sheet component
    columns = SSP

Each panel shows ONLY ONE SSP:
    solid line = median
    dark band  = 17th-83rd percentile
    light band = 5th-95th percentile

FIGURE 2
--------
A compact 3 x 5 version using only:
    Total AIS
    GrIS
    Total ice-sheet SLR

FIGURE 3
--------
One figure per SSP, with 5 vertically stacked component panels.

This preserves the uncertainty information but avoids overlaying
multiple SSP lines/bands in the same axes.

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
    / "figures_simplified"
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

MEDIAN_LINEWIDTH = 2.0
INNER_ALPHA = 0.28
OUTER_ALPHA = 0.12

XTICKS = [
    2000,
    2050,
    2100,
    2150,
    2200,
    2250,
    2300,
]


# ============================================================
# 2. LOAD DATA
# ============================================================

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


# ============================================================
# 3. CALCULATE SUMMARY
# ============================================================

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


# ============================================================
# 4. PANEL DRAWING
# ============================================================

def draw_uncertainty_panel(
    ax,
    group,
):

    group = group.sort_values(
        "year"
    )

    x = group[
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

    ax.fill_between(
        x,
        p05,
        p95,
        alpha=OUTER_ALPHA,
        linewidth=0,
    )

    ax.fill_between(
        x,
        p17,
        p83,
        alpha=INNER_ALPHA,
        linewidth=0,
    )

    ax.plot(
        x,
        p50,
        linewidth=MEDIAN_LINEWIDTH,
    )

    ax.axhline(
        0,
        linewidth=0.7,
        linestyle="--",
    )

    ax.set_xlim(
        2000,
        2300,
    )

    ax.set_xticks(
        XTICKS
    )

    ax.grid(
        axis="y",
        alpha=0.15,
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


# ============================================================
# 5. FIGURE 1: 5 COMPONENTS x 5 SSPS
# ============================================================

def make_all_components_small_multiples(
    summary,
):

    fig, axes = plt.subplots(
        nrows=len(
            COMPONENT_ORDER
        ),
        ncols=len(
            SSP_ORDER
        ),
        figsize=(
            16.5,
            13.0,
        ),
        sharex=True,
        sharey="row",
    )

    for row_index, component in enumerate(
        COMPONENT_ORDER
    ):

        for col_index, scenario in enumerate(
            SSP_ORDER
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
                        "scenario_label"
                    ]
                    == scenario
                )
            ]

            draw_uncertainty_panel(
                ax,
                group,
            )

            if row_index == 0:
                ax.set_title(
                    scenario
                )

            if col_index == 0:
                ax.set_ylabel(
                    f"{component}\n(m)"
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

            else:
                ax.tick_params(
                    labelbottom=False
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            linewidth=MEDIAN_LINEWIDTH,
            label="Median",
        ),
        Patch(
            alpha=INNER_ALPHA,
            label="17–83%",
        ),
        Patch(
            alpha=OUTER_ALPHA,
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
        top=0.955,
        bottom=0.07,
        left=0.08,
        right=0.985,
        hspace=0.18,
        wspace=0.10,
    )

    outfile = (
        FIGURE_DIR
        / "b25_categorical_climate_space_all_components_small_multiples.png"
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
        f"Saved:\n{outfile}"
    )


# ============================================================
# 6. FIGURE 2: 3 COMPONENTS x 5 SSPS
# ============================================================

def make_compact_small_multiples(
    summary,
):

    fig, axes = plt.subplots(
        nrows=len(
            COMPACT_COMPONENT_ORDER
        ),
        ncols=len(
            SSP_ORDER
        ),
        figsize=(
            16.5,
            8.5,
        ),
        sharex=True,
        sharey="row",
    )

    for row_index, component in enumerate(
        COMPACT_COMPONENT_ORDER
    ):

        for col_index, scenario in enumerate(
            SSP_ORDER
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
                        "scenario_label"
                    ]
                    == scenario
                )
            ]

            draw_uncertainty_panel(
                ax,
                group,
            )

            if row_index == 0:
                ax.set_title(
                    scenario
                )

            if col_index == 0:
                ax.set_ylabel(
                    f"{component}\n(m)"
                )

            if row_index == (
                len(
                    COMPACT_COMPONENT_ORDER
                )
                - 1
            ):
                ax.set_xlabel(
                    "Year"
                )

            else:
                ax.tick_params(
                    labelbottom=False
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            linewidth=MEDIAN_LINEWIDTH,
            label="Median",
        ),
        Patch(
            alpha=INNER_ALPHA,
            label="17–83%",
        ),
        Patch(
            alpha=OUTER_ALPHA,
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
        top=0.94,
        bottom=0.09,
        left=0.08,
        right=0.985,
        hspace=0.18,
        wspace=0.10,
    )

    outfile = (
        FIGURE_DIR
        / "b25_categorical_climate_space_compact_small_multiples.png"
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
        f"Saved:\n{outfile}"
    )


# ============================================================
# 7. FIGURE 3: ONE FIGURE PER SSP
# ============================================================

def make_one_figure_per_ssp(
    summary,
):

    for scenario in SSP_ORDER:

        fig, axes = plt.subplots(
            nrows=len(
                COMPONENT_ORDER
            ),
            ncols=1,
            figsize=(
                8.0,
                12.0,
            ),
            sharex=True,
        )

        for ax, component in zip(
            axes,
            COMPONENT_ORDER,
        ):

            group = summary.loc[
                (
                    summary[
                        "component"
                    ]
                    == component
                )
                & (
                    summary[
                        "scenario_label"
                    ]
                    == scenario
                )
            ]

            draw_uncertainty_panel(
                ax,
                group,
            )

            ax.set_ylabel(
                f"{component}\n(m)"
            )

        axes[
            -1
        ].set_xlabel(
            "Year"
        )

        legend_handles = [
            Line2D(
                [0],
                [0],
                linewidth=MEDIAN_LINEWIDTH,
                label="Median",
            ),
            Patch(
                alpha=INNER_ALPHA,
                label="17–83%",
            ),
            Patch(
                alpha=OUTER_ALPHA,
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

        fig.suptitle(
            scenario,
            y=0.965,
        )

        fig.subplots_adjust(
            top=0.92,
            bottom=0.07,
            left=0.16,
            right=0.97,
            hspace=0.16,
        )

        safe_name = (
            scenario
            .replace(
                ".",
                "p"
            )
            .replace(
                "-",
                "_"
            )
        )

        outfile = (
            FIGURE_DIR
            / f"b25_categorical_climate_space_{safe_name}.png"
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
            f"Saved:\n{outfile}"
        )


# ============================================================
# 8. SELECTION SUMMARY
# ============================================================

def summarize_selection(
    selection,
):

    if selection is None:
        return

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

    realized.to_csv(
        OUTPUT_DIR
        / "realized_category_fractions_by_ssp_year.csv",
        index=False,
    )


# ============================================================
# 9. MAIN
# ============================================================

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

    make_all_components_small_multiples(
        summary
    )

    make_compact_small_multiples(
        summary
    )

    make_one_figure_per_ssp(
        summary
    )

    summarize_selection(
        selection
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
