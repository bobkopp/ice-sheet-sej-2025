#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:45:55 2026

@author: elizabeth
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# DIAGNOSTIC SETTINGS
# ============================================================

diagnostic_output_dir = Path(
    "/Users/elizabeth/Documents/GitHub/"
    "ice-sheet-sej-2025/output/"
    "fair_sej_interpolation/diagnostics"
)

diagnostic_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

projection_years = [
    2100,
    2200,
    2300,
]

scenario_order = [
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP3-7.0",
    "SSP5-8.5",
]

anchor_order = [
    "Low",
    "Intermediate",
    "High",
]

anchor_colors = {
    "Low": "tab:blue",
    "Intermediate": "tab:orange",
    "High": "tab:red",
}

scenario_markers = {
    "SSP1-1.9": "o",
    "SSP1-2.6": "s",
    "SSP2-4.5": "^",
    "SSP3-7.0": "D",
    "SSP5-8.5": "P",
}

# ============================================================
# SEJ CLIMATE ANCHORS
# ============================================================

sej_climate_anchors = pd.DataFrame(
    [
        {
            "anchor": "Low",
            "year": 2100,
            "gsat": 2.0,
            "cumulative_warming": 155.0,
        },
        {
            "anchor": "Intermediate",
            "year": 2100,
            "gsat": 3.5,
            "cumulative_warming": 204.5,
        },
        {
            "anchor": "High",
            "year": 2100,
            "gsat": 5.0,
            "cumulative_warming": 262.0,
        },

        {
            "anchor": "Low",
            "year": 2200,
            "gsat": 2.0,
            "cumulative_warming": 355.0,
        },
        {
            "anchor": "Intermediate",
            "year": 2200,
            "gsat": 3.5,
            "cumulative_warming": 554.5,
        },
        {
            "anchor": "High",
            "year": 2200,
            "gsat": 5.0,
            "cumulative_warming": 762.0,
        },

        {
            "anchor": "Low",
            "year": 2300,
            "gsat": 2.0,
            "cumulative_warming": 555.0,
        },
        {
            "anchor": "Intermediate",
            "year": 2300,
            "gsat": 3.5,
            "cumulative_warming": 904.5,
        },
        {
            "anchor": "High",
            "year": 2300,
            "gsat": 5.0,
            "cumulative_warming": 1262.0,
        },
    ]
)

# ============================================================
# DIAGNOSTIC 1
# FAIR CLIMATE SPACE VERSUS SEJ ANCHORS
# ============================================================

def plot_fair_climate_space(
    fair_features,
    anchor_table,
    output_dir,
    max_points_per_group=1500,
    random_seed=1234,
):
    """
    Plot GSAT versus cumulative warming for the FaIR ensemble,
    together with the Low, Intermediate, and High SEJ anchors.

    Parameters
    ----------
    fair_features : pandas.DataFrame
        Must contain:
            scenario_label
            year
            gsat
            cumulative_warming

        May also contain:
            outside_reference_domain

    anchor_table : pandas.DataFrame
        Must contain:
            anchor
            year
            gsat
            cumulative_warming

    output_dir : str or pathlib.Path
        Directory in which the figure and summaries are saved.

    max_points_per_group : int
        Maximum number of points plotted for each SSP-year group.

    random_seed : int
        Seed used only for plotting subsamples.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_columns = {
        "scenario_label",
        "year",
        "gsat",
        "cumulative_warming",
    }

    missing_columns = (
        required_columns
        - set(fair_features.columns)
    )

    if missing_columns:
        raise KeyError(
            "fair_features is missing these columns: "
            f"{sorted(missing_columns)}"
        )

    anchor_required_columns = {
        "anchor",
        "year",
        "gsat",
        "cumulative_warming",
    }

    missing_anchor_columns = (
        anchor_required_columns
        - set(anchor_table.columns)
    )

    if missing_anchor_columns:
        raise KeyError(
            "anchor_table is missing these columns: "
            f"{sorted(missing_anchor_columns)}"
        )

    rng = np.random.default_rng(
        random_seed
    )

    plot_parts = []

    for (
        scenario_label,
        year,
    ), group in fair_features.groupby(
        [
            "scenario_label",
            "year",
        ],
        sort=False,
    ):

        if len(group) > max_points_per_group:
            selected_indices = rng.choice(
                group.index.to_numpy(),
                size=max_points_per_group,
                replace=False,
            )

            group_to_plot = group.loc[
                selected_indices
            ].copy()

        else:
            group_to_plot = group.copy()

        plot_parts.append(
            group_to_plot
        )

    plot_data = pd.concat(
        plot_parts,
        ignore_index=True,
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 6),
        sharey=False,
    )

    for ax, year in zip(
        axes,
        projection_years,
    ):

        year_data = plot_data.loc[
            plot_data["year"] == year
        ]

        for scenario_label in scenario_order:

            scenario_data = year_data.loc[
                year_data["scenario_label"]
                == scenario_label
            ]

            if scenario_data.empty:
                continue

            ax.scatter(
                scenario_data["gsat"],
                scenario_data[
                    "cumulative_warming"
                ],
                s=8,
                alpha=0.18,
                marker=scenario_markers.get(
                    scenario_label,
                    "o",
                ),
                label=scenario_label,
                rasterized=True,
            )

        year_anchors = anchor_table.loc[
            anchor_table["year"] == year
        ]

        for _, anchor_row in (
            year_anchors.iterrows()
        ):

            anchor_name = anchor_row[
                "anchor"
            ]

            ax.scatter(
                anchor_row["gsat"],
                anchor_row[
                    "cumulative_warming"
                ],
                s=190,
                marker="*",
                color=anchor_colors[
                    anchor_name
                ],
                edgecolor="black",
                linewidth=0.9,
                zorder=10,
            )

            ax.annotate(
                anchor_name,
                (
                    anchor_row["gsat"],
                    anchor_row[
                        "cumulative_warming"
                    ],
                ),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(
            str(year),
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xlabel(
            "GSAT anomaly (°C)"
        )

        ax.set_ylabel(
            "Cumulative warming since 2000 "
            "(K·yr)"
        )

        ax.grid(
            alpha=0.25
        )

    handles, labels = axes[0].get_legend_handles_labels()

    unique_handles = {}
    for handle, label in zip(
        handles,
        labels,
    ):
        unique_handles[label] = handle

    fig.legend(
        unique_handles.values(),
        unique_handles.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=5,
        frameon=False,
    )

    fig.suptitle(
        "FaIR climate trajectories and SEJ climate anchors",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[
            0.0,
            0.08,
            1.0,
            0.94,
        ]
    )

    figure_path = (
        output_dir
        / "diagnostic_1_fair_climate_space.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save numerical climate-space summary
    # --------------------------------------------------------

    summary = (
        fair_features
        .groupby(
            [
                "scenario_label",
                "year",
            ]
        )
        .agg(
            n_samples=(
                "gsat",
                "size",
            ),
            gsat_mean=(
                "gsat",
                "mean",
            ),
            gsat_p05=(
                "gsat",
                lambda x: np.quantile(
                    x,
                    0.05,
                ),
            ),
            gsat_median=(
                "gsat",
                "median",
            ),
            gsat_p95=(
                "gsat",
                lambda x: np.quantile(
                    x,
                    0.95,
                ),
            ),
            cumulative_warming_mean=(
                "cumulative_warming",
                "mean",
            ),
            cumulative_warming_p05=(
                "cumulative_warming",
                lambda x: np.quantile(
                    x,
                    0.05,
                ),
            ),
            cumulative_warming_median=(
                "cumulative_warming",
                "median",
            ),
            cumulative_warming_p95=(
                "cumulative_warming",
                lambda x: np.quantile(
                    x,
                    0.95,
                ),
            ),
        )
        .reset_index()
    )

    if (
        "outside_reference_domain"
        in fair_features.columns
    ):

        outside_summary = (
            fair_features
            .groupby(
                [
                    "scenario_label",
                    "year",
                ]
            )[
                "outside_reference_domain"
            ]
            .mean()
            .mul(100.0)
            .rename(
                "percent_outside_reference_domain"
            )
            .reset_index()
        )

        summary = summary.merge(
            outside_summary,
            on=[
                "scenario_label",
                "year",
            ],
            how="left",
        )

    summary_path = (
        output_dir
        / "diagnostic_1_climate_space_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nDiagnostic 1 saved:"
    )
    print(
        figure_path
    )
    print(
        summary_path
    )

    return summary