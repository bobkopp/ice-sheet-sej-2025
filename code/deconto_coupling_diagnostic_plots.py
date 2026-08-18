#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 09:04:29 2026


Visualize the B25–DP21 Total AIS temporal-coupling experiments.

Figures produced:

1. Target and achieved correlation matrices.
2. Reconstructed Total AIS trajectories and percentile envelope.
3. Rank-persistence scatterplots.
4. Correlation comparison across all three experiments.

@author: elizabeth
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

project_dir = (
    "/Users/elizabeth/Documents/GitHub/"
    "ice-sheet-sej-2025"
)

results_dir = os.path.join(
    project_dir,
    "output",
    "temporal_coupling",
    "total_ais_common_rank"
)

figure_dir = os.path.join(
    results_dir,
    "figures"
)

os.makedirs(
    figure_dir,
    exist_ok=True
)


years = [
    2100,
    2200,
    2300
]


experiment_settings = [
    {
        "name": "b25_2.0C_dp21_rcp26",
        "label": "2.0°C / DP21 RCP2.6"
    },
    {
        "name": "b25_3.5C_dp21_rcp45",
        "label": "3.5°C / DP21 RCP4.5"
    },
    {
        "name": "b25_5.0C_dp21_rcp85",
        "label": "5.0°C / DP21 RCP8.5"
    }
]


# Number of individual trajectories shown in spaghetti plots.
n_trajectories_to_plot = 100

trajectory_random_seed = 1234


# ============================================================
# 2. READ ONE EXPERIMENT
# ============================================================

def load_experiment(
    experiment_name: str
) -> dict[str, pd.DataFrame]:
    """
    Load the coupled samples, coupling metadata, target matrix,
    and correlation diagnostics for one experiment.
    """

    experiment_dir = os.path.join(
        results_dir,
        experiment_name
    )

    files = {
        "coupled": os.path.join(
            experiment_dir,
            f"{experiment_name}_coupled_samples.csv"
        ),
        "metadata": os.path.join(
            experiment_dir,
            f"{experiment_name}_coupling_metadata.csv"
        ),
        "target": os.path.join(
            experiment_dir,
            f"{experiment_name}_target_matrix.csv"
        ),
        "correlations": os.path.join(
            experiment_dir,
            f"{experiment_name}_correlation_diagnostics.csv"
        )
    }

    for label, file_path in files.items():

        if not os.path.exists(
            file_path
        ):
            raise FileNotFoundError(
                f"Missing {label} file:\n{file_path}"
            )

    coupled_df = pd.read_csv(
        files["coupled"]
    )

    metadata_df = pd.read_csv(
        files["metadata"]
    )

    target_df = pd.read_csv(
        files["target"],
        index_col=0
    )

    target_df.index = (
        target_df.index
        .astype(int)
    )

    target_df.columns = (
        target_df.columns
        .astype(int)
    )

    correlation_df = pd.read_csv(
        files["correlations"]
    )

    return {
        "coupled": coupled_df,
        "metadata": metadata_df,
        "target": target_df,
        "correlations": correlation_df
    }


# ============================================================
# 3. CORRELATION TABLE UTILITIES
# ============================================================

def extract_correlation_matrix(
    correlation_df: pd.DataFrame,
    matrix_name: str
) -> pd.DataFrame:
    """
    Convert one matrix from the long diagnostic table to wide form.
    """

    matrix = (
        correlation_df
        .loc[
            correlation_df["matrix"] == matrix_name
        ]
        .pivot(
            index="row_year",
            columns="column_year",
            values="correlation"
        )
    )

    matrix.index = (
        matrix.index
        .astype(int)
    )

    matrix.columns = (
        matrix.columns
        .astype(int)
    )

    return matrix.loc[
        years,
        years
    ]


def draw_correlation_matrix(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str
) -> None:
    """
    Draw one annotated correlation-matrix panel.
    """

    image = ax.imshow(
        matrix.to_numpy(),
        vmin=0.0,
        vmax=1.0,
        aspect="equal"
    )

    ax.set_xticks(
        np.arange(
            len(years)
        )
    )

    ax.set_yticks(
        np.arange(
            len(years)
        )
    )

    ax.set_xticklabels(
        years
    )

    ax.set_yticklabels(
        years
    )

    ax.set_xlabel(
        "Projection year"
    )

    ax.set_ylabel(
        "Projection year"
    )

    ax.set_title(
        title
    )

    for row in range(
        len(years)
    ):

        for column in range(
            len(years)
        ):

            value = matrix.iloc[
                row,
                column
            ]

            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center"
            )

    return image


# ============================================================
# 4. FIGURE 1: CORRELATION MATRICES
# ============================================================

def plot_correlation_matrices(
    experiment_name: str,
    experiment_label: str,
    experiment_data: dict[str, pd.DataFrame]
) -> None:
    """
    Compare target and achieved dependence for one experiment.
    """

    target = experiment_data[
        "target"
    ]

    correlation_df = experiment_data[
        "correlations"
    ]

    latent_pearson = extract_correlation_matrix(
        correlation_df,
        "achieved_latent_pearson"
    )

    total_spearman = extract_correlation_matrix(
        correlation_df,
        "coupled_total_ais_spearman"
    )

    total_pearson = extract_correlation_matrix(
        correlation_df,
        "coupled_total_ais_pearson"
    )


    matrices = [
        (
            target,
            "Target DP21\nlatent Pearson"
        ),
        (
            latent_pearson,
            "Achieved latent\nPearson"
        ),
        (
            total_spearman,
            "Coupled Total AIS\nSpearman"
        ),
        (
            total_pearson,
            "Coupled Total AIS\nPearson"
        )
    ]


    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(
            15,
            4.2
        ),
        constrained_layout=True
    )

    image = None

    for ax, (
        matrix,
        title
    ) in zip(
        axes,
        matrices
    ):

        image = draw_correlation_matrix(
            ax=ax,
            matrix=matrix,
            title=title
        )


    fig.colorbar(
        image,
        ax=axes,
        shrink=0.82,
        label="Correlation"
    )

    fig.suptitle(
        experiment_label,
        fontsize=14
    )


    output_file = os.path.join(
        figure_dir,
        f"{experiment_name}_correlation_matrices.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 5. FIGURE 2: COUPLED TRAJECTORIES
# ============================================================

def plot_total_ais_trajectories(
    experiment_name: str,
    experiment_label: str,
    metadata_df: pd.DataFrame
) -> None:
    """
    Plot selected reconstructed trajectories and the ensemble
    percentile envelope.

    selected_total_ais comes directly from the Total AIS variable
    used to rank the panels.
    """

    trajectory_table = (
        metadata_df
        .pivot(
            index="trajectory_id",
            columns="projection_year",
            values="selected_total_ais"
        )
        .loc[
            :,
            years
        ]
    )


    quantiles = trajectory_table.quantile(
        [
            0.05,
            0.17,
            0.50,
            0.83,
            0.95
        ],
        axis=0
    )


    rng = np.random.default_rng(
        trajectory_random_seed
    )

    available_ids = (
        trajectory_table
        .index
        .to_numpy()
    )

    number_to_plot = min(
        n_trajectories_to_plot,
        len(available_ids)
    )

    selected_ids = rng.choice(
        available_ids,
        size=number_to_plot,
        replace=False
    )


    fig, ax = plt.subplots(
        figsize=(
            7.5,
            5.5
        )
    )


    for trajectory_id in selected_ids:

        ax.plot(
            years,
            trajectory_table.loc[
                trajectory_id,
                years
            ],
            linewidth=0.7,
            alpha=0.18
        )


    ax.fill_between(
        years,
        quantiles.loc[
            0.05,
            years
        ],
        quantiles.loc[
            0.95,
            years
        ],
        alpha=0.18,
        label="5th–95th percentile"
    )


    ax.fill_between(
        years,
        quantiles.loc[
            0.17,
            years
        ],
        quantiles.loc[
            0.83,
            years
        ],
        alpha=0.28,
        label="17th–83rd percentile"
    )


    ax.plot(
        years,
        quantiles.loc[
            0.50,
            years
        ],
        linewidth=2.5,
        marker="o",
        label="Median"
    )


    ax.set_xticks(
        years
    )

    ax.set_xlabel(
        "Projection year"
    )

    ax.set_ylabel(
        "Total AIS contribution"
    )

    ax.set_title(
        f"Reconstructed Total AIS trajectories\n"
        f"{experiment_label}"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )


    output_file = os.path.join(
        figure_dir,
        f"{experiment_name}_total_ais_trajectories.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 6. FIGURE 3: RANK-PERSISTENCE SCATTERPLOTS
# ============================================================

def calculate_percentile_ranks(
    values: pd.Series
) -> pd.Series:
    """
    Convert values to empirical percentile ranks in approximately
    the interval (0, 1).
    """

    return values.rank(
        method="average"
    ) / (
        len(values)
        + 1
    )


def plot_rank_persistence(
    experiment_name: str,
    experiment_label: str,
    metadata_df: pd.DataFrame
) -> None:
    """
    Compare Total AIS percentile rank between projection years.
    """

    trajectory_table = (
        metadata_df
        .pivot(
            index="trajectory_id",
            columns="projection_year",
            values="selected_total_ais"
        )
        .loc[
            :,
            years
        ]
    )


    rank_table = pd.DataFrame(
        {
            year: calculate_percentile_ranks(
                trajectory_table[
                    year
                ]
            )
            for year in years
        }
    )


    pairings = [
        (
            2100,
            2200
        ),
        (
            2100,
            2300
        ),
        (
            2200,
            2300
        )
    ]


    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(
            13,
            4.2
        ),
        constrained_layout=True
    )


    for ax, (
        first_year,
        second_year
    ) in zip(
        axes,
        pairings
    ):

        x = rank_table[
            first_year
        ]

        y = rank_table[
            second_year
        ]

        correlation = x.corr(
            y,
            method="spearman"
        )

        ax.scatter(
            x,
            y,
            s=9,
            alpha=0.30
        )

        ax.plot(
            [
                0,
                1
            ],
            [
                0,
                1
            ],
            linestyle="--",
            linewidth=1
        )

        ax.set_xlim(
            0,
            1
        )

        ax.set_ylim(
            0,
            1
        )

        ax.set_xlabel(
            f"{first_year} percentile rank"
        )

        ax.set_ylabel(
            f"{second_year} percentile rank"
        )

        ax.set_title(
            f"{first_year}–{second_year}\n"
            f"Spearman ρ = {correlation:.2f}"
        )

        ax.grid(
            alpha=0.20
        )


    fig.suptitle(
        f"Total AIS rank persistence\n{experiment_label}",
        fontsize=14
    )


    output_file = os.path.join(
        figure_dir,
        f"{experiment_name}_rank_persistence.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 7. FIGURE 4: COMPARE ALL THREE EXPERIMENTS
# ============================================================

def plot_experiment_correlation_comparison(
    loaded_experiments: list[dict]
) -> None:
    """
    Compare achieved coupled Total AIS rank correlations across
    the three warming–DP21 experiments.
    """

    pairings = [
        (
            2100,
            2200,
            "2100–2200"
        ),
        (
            2100,
            2300,
            "2100–2300"
        ),
        (
            2200,
            2300,
            "2200–2300"
        )
    ]


    records = []

    for item in loaded_experiments:

        experiment_label = item[
            "label"
        ]

        correlation_df = item[
            "data"
        ][
            "correlations"
        ]

        matrix = extract_correlation_matrix(
            correlation_df=correlation_df,
            matrix_name="coupled_total_ais_spearman"
        )

        for (
            first_year,
            second_year,
            pairing_label
        ) in pairings:

            records.append({
                "experiment": experiment_label,
                "year_pair": pairing_label,
                "correlation": matrix.loc[
                    first_year,
                    second_year
                ]
            })


    comparison_df = pd.DataFrame(
        records
    )


    experiment_labels = [
        item["label"]
        for item in loaded_experiments
    ]

    pair_labels = [
        item[2]
        for item in pairings
    ]


    x = np.arange(
        len(pair_labels)
    )

    width = 0.24


    fig, ax = plt.subplots(
        figsize=(
            9,
            5.3
        )
    )


    for experiment_number, experiment_label in enumerate(
        experiment_labels
    ):

        values = (
            comparison_df
            .loc[
                comparison_df["experiment"]
                == experiment_label
            ]
            .set_index("year_pair")
            .loc[
                pair_labels,
                "correlation"
            ]
            .to_numpy()
        )

        positions = (
            x
            + (
                experiment_number
                - 1
            )
            * width
        )

        bars = ax.bar(
            positions,
            values,
            width=width,
            label=experiment_label
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=3,
            fontsize=8
        )


    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        pair_labels
    )

    ax.set_ylim(
        0,
        1.08
    )

    ax.set_xlabel(
        "Projection-year pair"
    )

    ax.set_ylabel(
        "Coupled Total AIS Spearman correlation"
    )

    ax.set_title(
        "Temporal rank persistence across coupling experiments"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )


    output_file = os.path.join(
        figure_dir,
        "total_ais_spearman_comparison.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 8. RUN ALL FIGURES
# ============================================================

if __name__ == "__main__":

    loaded_experiments = []


    for experiment in experiment_settings:

        experiment_name = experiment[
            "name"
        ]

        experiment_label = experiment[
            "label"
        ]


        print(
            "\nLoading:",
            experiment_name
        )


        experiment_data = load_experiment(
            experiment_name
        )


        loaded_experiments.append({
            "name": experiment_name,
            "label": experiment_label,
            "data": experiment_data
        })


        plot_correlation_matrices(
            experiment_name=experiment_name,
            experiment_label=experiment_label,
            experiment_data=experiment_data
        )


        plot_total_ais_trajectories(
            experiment_name=experiment_name,
            experiment_label=experiment_label,
            metadata_df=experiment_data[
                "metadata"
            ]
        )


        plot_rank_persistence(
            experiment_name=experiment_name,
            experiment_label=experiment_label,
            metadata_df=experiment_data[
                "metadata"
            ]
        )


    plot_experiment_correlation_comparison(
        loaded_experiments=loaded_experiments
    )


    print(
        "\nSaved figures to:\n"
        f"{figure_dir}"
    )
