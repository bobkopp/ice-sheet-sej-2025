#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostics for the FaIR–SEJ interpolation emulator.

This standalone script reads outputs created by interpolate_fair_sej.py and
produces three diagnostics:

1. FaIR GSAT–cumulative-warming climate-space plots with SEJ anchors.
2. Distributions of Low, Intermediate, and High interpolation weights.
3. Comparisons between original SEJ anchor distributions and interpolated
   distributions for FaIR samples nearest each anchor.

Run interpolate_fair_sej.py first.

Default project directory
-------------------------
/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025

Default diagnostic output directory
-----------------------------------
<project>/output/fair_sej_interpolation/diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

DEFAULT_PROJECT_DIR = Path(
    "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025"
)

PROJECTION_YEARS = [2100, 2200, 2300]

SCENARIO_ORDER = [
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP3-7.0",
    "SSP5-8.5",
]

PATHWAY_ORDER = [
    "Low",
    "Intermediate",
    "High",
]

WEIGHT_COLUMN_MAP = {
    "Low": "weight_low",
    "Intermediate": "weight_intermediate",
    "High": "weight_high",
}

YEAR_CODE_MAP = {
    2100: "21",
    2200: "22",
    2300: "23",
}

WARMING_CODE_MAP = {
    "Low": "2",
    "Intermediate": "35",
    "High": "5",
}

DEFAULT_RESPONSE_CODES = [
    "E",     # EAIS total
    "W",     # WAIS total
    "G",     # GrIS total
    "SLR",   # Total ice-sheet SLR
]

FEATURE_SCALES = {
    "gsat": 1.0,
    "cumulative_warming": 200.0,
}

REFERENCE_CLIMATE = pd.DataFrame(
    [
        {"year": 2100, "pathway": "Low",          "gsat": 2.0, "cumulative_warming": 155.0},
        {"year": 2100, "pathway": "Intermediate", "gsat": 3.5, "cumulative_warming": 204.5},
        {"year": 2100, "pathway": "High",         "gsat": 5.0, "cumulative_warming": 262.0},

        {"year": 2200, "pathway": "Low",          "gsat": 2.0, "cumulative_warming": 355.0},
        {"year": 2200, "pathway": "Intermediate", "gsat": 3.5, "cumulative_warming": 554.5},
        {"year": 2200, "pathway": "High",         "gsat": 5.0, "cumulative_warming": 762.0},

        {"year": 2300, "pathway": "Low",          "gsat": 2.0, "cumulative_warming": 555.0},
        {"year": 2300, "pathway": "Intermediate", "gsat": 3.5, "cumulative_warming": 904.5},
        {"year": 2300, "pathway": "High",         "gsat": 5.0, "cumulative_warming": 1262.0},
    ]
)


# ============================================================
# 2. STYLE
# ============================================================

def set_publication_style() -> None:
    """Apply restrained, publication-oriented Matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ============================================================
# 3. HELPERS
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n\n"
            "Run interpolate_fair_sej.py first, or check --project-dir."
        )


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    missing = sorted(required_columns - set(dataframe.columns))

    if missing:
        raise KeyError(
            f"{dataframe_name} is missing required columns:\n"
            + "\n".join(missing)
        )


def q05(values: pd.Series) -> float:
    return float(values.quantile(0.05))


def q17(values: pd.Series) -> float:
    return float(values.quantile(0.17))


def q25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def q75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def q83(values: pd.Series) -> float:
    return float(values.quantile(0.83))


def q95(values: pd.Series) -> float:
    return float(values.quantile(0.95))


def save_figure(
    fig: plt.Figure,
    output_stem: Path,
) -> None:
    """Save matching PNG and PDF files."""
    output_stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )


# ============================================================
# 4. LOAD SAVED OUTPUTS
# ============================================================

def load_outputs(
    project_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Read files created by interpolate_fair_sej.py.

    Expected interpolation outputs:
        fair_climate_features.csv
        fair_sej_climate_weights.csv
        fair_sej_diagnostics.csv
        fair_sej_interpolated_responses.csv

    Expected original SEJ file:
        data/sej2025_split_panel_liz.csv
    """

    interpolation_dir = (
        project_dir
        / "output"
        / "fair_sej_interpolation"
    )

    paths = {
        "fair_features": interpolation_dir / "fair_climate_features.csv",
        "weights": interpolation_dir / "fair_sej_climate_weights.csv",
        "diagnostics": interpolation_dir / "fair_sej_diagnostics.csv",
        "interpolated_long": interpolation_dir / "fair_sej_interpolated_responses.csv",
        "original_sej": project_dir / "data" / "sej2025_split_panel_liz.csv",
    }

    for path in paths.values():
        require_file(path)

    print("Reading saved emulator outputs...")

    outputs = {
        name: pd.read_csv(path)
        for name, path in paths.items()
    }

    print(f"  FaIR climate-feature rows: {len(outputs['fair_features']):,}")
    print(f"  Climate-weight rows:       {len(outputs['weights']):,}")
    print(f"  Diagnostic rows:           {len(outputs['diagnostics']):,}")
    print(f"  Interpolated rows:         {len(outputs['interpolated_long']):,}")
    print(f"  Original SEJ rows:         {len(outputs['original_sej']):,}")

    return outputs


# ============================================================
# 5. DIAGNOSTIC 1
#    FAIR CLIMATE SPACE VERSUS SEJ ANCHORS
# ============================================================

def diagnostic_1_climate_space(
    fair_features: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    max_points_per_scenario_year: int = 1500,
    random_seed: int = 1234,
) -> pd.DataFrame:
    """
    Plot GSAT against cumulative warming for the FaIR ensemble,
    separately for 2100, 2200, and 2300, with SEJ climate anchors.
    """

    require_columns(
        fair_features,
        {
            "fair_sample_id",
            "scenario_label",
            "year",
            "gsat",
            "cumulative_warming",
        },
        "fair_features",
    )

    require_columns(
        diagnostics,
        {
            "fair_sample_id",
            "year",
            "outside_reference_domain",
            "nearest_reference_distance",
        },
        "diagnostics",
    )

    merged = fair_features.merge(
        diagnostics[
            [
                "fair_sample_id",
                "year",
                "outside_reference_domain",
                "nearest_reference_distance",
            ]
        ],
        on=[
            "fair_sample_id",
            "year",
        ],
        how="left",
        validate="one_to_one",
    )

    rng = np.random.default_rng(
        random_seed
    )

    plotting_groups = []

    for _, group in merged.groupby(
        [
            "scenario_label",
            "year",
        ],
        sort=False,
    ):
        if len(group) > max_points_per_scenario_year:
            positions = rng.choice(
                len(group),
                size=max_points_per_scenario_year,
                replace=False,
            )

            group = group.iloc[
                positions
            ]

        plotting_groups.append(
            group
        )

    plot_data = pd.concat(
        plotting_groups,
        ignore_index=True,
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 6),
    )

    for ax, year in zip(
        axes,
        PROJECTION_YEARS,
    ):
        year_data = plot_data.loc[
            plot_data["year"]
            == year
        ]

        for scenario_label in SCENARIO_ORDER:
            scenario_data = year_data.loc[
                year_data["scenario_label"]
                == scenario_label
            ]

            if scenario_data.empty:
                continue

            ax.scatter(
                scenario_data["gsat"],
                scenario_data["cumulative_warming"],
                s=8,
                alpha=0.20,
                label=scenario_label,
                rasterized=True,
            )

        anchors = REFERENCE_CLIMATE.loc[
            REFERENCE_CLIMATE["year"]
            == year
        ]

        ax.plot(
            anchors["gsat"],
            anchors["cumulative_warming"],
            linestyle="--",
            linewidth=1.0,
        )

        for _, anchor in anchors.iterrows():
            ax.scatter(
                anchor["gsat"],
                anchor["cumulative_warming"],
                marker="*",
                s=190,
                edgecolor="black",
                linewidth=0.7,
                zorder=10,
            )

            ax.annotate(
                anchor["pathway"],
                (
                    anchor["gsat"],
                    anchor["cumulative_warming"],
                ),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        outside_percent = (
            100.0
            * merged.loc[
                merged["year"] == year,
                "outside_reference_domain",
            ].mean()
        )

        ax.set_title(
            f"{year}\n{outside_percent:.1f}% outside reference domain",
            fontweight="bold",
        )

        ax.set_xlabel(
            "GSAT anomaly (°C)"
        )

        ax.set_ylabel(
            "Cumulative warming since 2000 (K·yr)"
        )

        ax.grid(
            alpha=0.25
        )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    unique_handles = {}

    for handle, label in zip(
        handles,
        labels,
    ):
        if label not in unique_handles:
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

    save_figure(
        fig,
        output_dir
        / "diagnostic_1_climate_space",
    )

    plt.close(fig)

    summary = (
        merged
        .groupby(
            [
                "scenario_label",
                "year",
            ],
            observed=True,
        )
        .agg(
            n_samples=("fair_sample_id", "size"),
            gsat_mean=("gsat", "mean"),
            gsat_p05=("gsat", q05),
            gsat_median=("gsat", "median"),
            gsat_p95=("gsat", q95),
            cumulative_warming_mean=("cumulative_warming", "mean"),
            cumulative_warming_p05=("cumulative_warming", q05),
            cumulative_warming_median=("cumulative_warming", "median"),
            cumulative_warming_p95=("cumulative_warming", q95),
            mean_nearest_reference_distance=(
                "nearest_reference_distance",
                "mean",
            ),
            percent_outside_reference_domain=(
                "outside_reference_domain",
                lambda x: 100.0 * x.mean(),
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        output_dir
        / "diagnostic_1_climate_space_summary.csv",
        index=False,
    )

    print("\nDiagnostic 1 complete.")

    return summary


# ============================================================
# 6. DIAGNOSTIC 2
#    WEIGHT DISTRIBUTIONS
# ============================================================

def diagnostic_2_weight_distributions(
    weights: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Plot weight distributions and save percentile and
    dominant-pathway summaries.
    """

    require_columns(
        weights,
        {
            "fair_sample_id",
            "scenario_label",
            "year",
            "weight_low",
            "weight_intermediate",
            "weight_high",
        },
        "weights",
    )

    weight_long = weights.melt(
        id_vars=[
            "fair_sample_id",
            "scenario_label",
            "year",
        ],
        value_vars=list(
            WEIGHT_COLUMN_MAP.values()
        ),
        var_name="weight_column",
        value_name="weight",
    )

    reverse_map = {
        column_name: pathway
        for pathway, column_name
        in WEIGHT_COLUMN_MAP.items()
    }

    weight_long["pathway"] = (
        weight_long["weight_column"]
        .map(reverse_map)
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(18, 14),
        sharey=True,
    )

    for row_index, pathway in enumerate(
        PATHWAY_ORDER
    ):
        for column_index, year in enumerate(
            PROJECTION_YEARS
        ):
            ax = axes[
                row_index,
                column_index,
            ]

            panel = weight_long.loc[
                (
                    weight_long["pathway"]
                    == pathway
                )
                & (
                    weight_long["year"]
                    == year
                )
            ]

            box_values = []
            box_labels = []

            for scenario_label in SCENARIO_ORDER:
                values = (
                    panel.loc[
                        panel["scenario_label"]
                        == scenario_label,
                        "weight",
                    ]
                    .dropna()
                    .to_numpy()
                )

                if len(values) == 0:
                    continue

                box_values.append(
                    values
                )

                box_labels.append(
                    scenario_label
                )

            ax.boxplot(
                box_values,
                labels=box_labels,
                showfliers=False,
                whis=(5, 95),
            )

            ax.set_ylim(
                -0.03,
                1.03,
            )

            ax.grid(
                axis="y",
                alpha=0.25,
            )

            ax.tick_params(
                axis="x",
                labelrotation=45,
            )

            if row_index == 0:
                ax.set_title(
                    str(year),
                    fontweight="bold",
                )

            if column_index == 0:
                ax.set_ylabel(
                    f"{pathway} weight",
                    fontweight="bold",
                )

    fig.suptitle(
        "Distributions of SEJ climate-anchor weights",
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.96,
        ]
    )

    save_figure(
        fig,
        output_dir
        / "diagnostic_2_weight_distributions",
    )

    plt.close(fig)

    percentile_summary = (
        weight_long
        .groupby(
            [
                "scenario_label",
                "year",
                "pathway",
            ],
            observed=True,
        )
        .agg(
            n_samples=("weight", "size"),
            mean=("weight", "mean"),
            minimum=("weight", "min"),
            p05=("weight", q05),
            p25=("weight", q25),
            median=("weight", "median"),
            p75=("weight", q75),
            p95=("weight", q95),
            maximum=("weight", "max"),
        )
        .reset_index()
    )

    percentile_summary.to_csv(
        output_dir
        / "diagnostic_2_weight_percentiles.csv",
        index=False,
    )

    weight_matrix = weights[
        [
            "weight_low",
            "weight_intermediate",
            "weight_high",
        ]
    ].to_numpy(dtype=float)

    dominant_indices = np.argmax(
        weight_matrix,
        axis=1,
    )

    dominant = weights[
        [
            "fair_sample_id",
            "scenario_label",
            "year",
        ]
    ].copy()

    dominant["dominant_pathway"] = np.array(
        PATHWAY_ORDER,
        dtype=object,
    )[dominant_indices]

    dominant_summary = (
        dominant
        .groupby(
            [
                "scenario_label",
                "year",
                "dominant_pathway",
            ],
            observed=True,
        )
        .size()
        .rename("n_samples")
        .reset_index()
    )

    totals = (
        dominant_summary
        .groupby(
            [
                "scenario_label",
                "year",
            ]
        )["n_samples"]
        .transform("sum")
    )

    dominant_summary["percent_samples"] = (
        100.0
        * dominant_summary["n_samples"]
        / totals
    )

    dominant_summary.to_csv(
        output_dir
        / "diagnostic_2_dominant_pathway.csv",
        index=False,
    )

    print("\nDiagnostic 2 complete.")

    return {
        "weight_percentiles": percentile_summary,
        "dominant_pathway": dominant_summary,
    }


# ============================================================
# 7. DIAGNOSTIC 3
#    ORIGINAL SEJ VERSUS NEAR-ANCHOR INTERPOLATED RESULTS
# ============================================================

def diagnostic_3_anchor_comparison(
    weights: pd.DataFrame,
    interpolated_long: pd.DataFrame,
    original_sej: pd.DataFrame,
    output_dir: Path,
    response_codes: list[str] | None = None,
    nearest_fraction: float = 0.05,
    feature_scales: dict[str, float] = FEATURE_SCALES,
) -> pd.DataFrame:
    """
    Compare original SEJ anchor distributions with interpolated
    distributions among FaIR samples nearest each climate anchor.

    This is a near-anchor diagnostic rather than an exact-anchor
    reproduction test. The selected FaIR states will generally not
    lie exactly on the SEJ anchors.
    """

    if response_codes is None:
        response_codes = DEFAULT_RESPONSE_CODES

    if not 0.0 < nearest_fraction <= 1.0:
        raise ValueError(
            "nearest_fraction must be greater than 0 and no more than 1."
        )

    require_columns(
        weights,
        {
            "fair_sample_id",
            "scenario_label",
            "year",
            "gsat",
            "cumulative_warming",
        },
        "weights",
    )

    require_columns(
        interpolated_long,
        {
            "fair_sample_id",
            "year",
            "response_code",
            "value",
        },
        "interpolated_long",
    )

    gsat_scale = float(
        feature_scales["gsat"]
    )

    cumulative_scale = float(
        feature_scales["cumulative_warming"]
    )

    comparison_rows = []
    selected_rows = []

    for year in PROJECTION_YEARS:
        fair_year = weights.loc[
            weights["year"]
            == year
        ].copy()

        anchors = REFERENCE_CLIMATE.loc[
            REFERENCE_CLIMATE["year"]
            == year
        ]

        for _, anchor in anchors.iterrows():
            pathway = anchor["pathway"]

            fair_year["distance_to_anchor"] = np.sqrt(
                (
                    (
                        fair_year["gsat"]
                        - anchor["gsat"]
                    )
                    / gsat_scale
                ) ** 2
                +
                (
                    (
                        fair_year["cumulative_warming"]
                        - anchor["cumulative_warming"]
                    )
                    / cumulative_scale
                ) ** 2
            )

            n_select = max(
                1,
                int(
                    np.ceil(
                        nearest_fraction
                        * len(fair_year)
                    )
                ),
            )

            nearest = fair_year.nsmallest(
                n_select,
                "distance_to_anchor",
            ).copy()

            nearest["pathway"] = pathway

            selected_rows.append(
                nearest[
                    [
                        "fair_sample_id",
                        "scenario_label",
                        "year",
                        "gsat",
                        "cumulative_warming",
                        "distance_to_anchor",
                        "pathway",
                    ]
                ]
            )

            selected_ids = nearest[
                "fair_sample_id"
            ].astype(int)

            subset = interpolated_long.loc[
                (
                    interpolated_long["year"]
                    == year
                )
                & (
                    interpolated_long["fair_sample_id"]
                    .isin(selected_ids)
                )
                & (
                    interpolated_long["response_code"]
                    .isin(response_codes)
                )
            ]

            warming_code = WARMING_CODE_MAP[
                pathway
            ]

            year_code = YEAR_CODE_MAP[
                year
            ]

            for response_code in response_codes:
                original_column = (
                    f"{response_code}"
                    f"{warming_code}_"
                    f"{year_code}"
                )

                if original_column not in original_sej.columns:
                    print(
                        "WARNING: Skipping missing original SEJ column "
                        f"{original_column}"
                    )
                    continue

                original_values = pd.to_numeric(
                    original_sej[original_column],
                    errors="coerce",
                ).dropna()

                interpolated_values = pd.to_numeric(
                    subset.loc[
                        subset["response_code"]
                        == response_code,
                        "value",
                    ],
                    errors="coerce",
                ).dropna()

                if original_values.empty:
                    print(
                        f"WARNING: {original_column} has no finite values."
                    )
                    continue

                if interpolated_values.empty:
                    print(
                        "WARNING: No interpolated values for "
                        f"{response_code}, {year}, {pathway}."
                    )
                    continue

                original_quantiles = original_values.quantile(
                    [
                        0.05,
                        0.17,
                        0.50,
                        0.83,
                        0.95,
                    ]
                )

                interpolated_quantiles = interpolated_values.quantile(
                    [
                        0.05,
                        0.17,
                        0.50,
                        0.83,
                        0.95,
                    ]
                )

                comparison_rows.append(
                    {
                        "year": year,
                        "pathway": pathway,
                        "response_code": response_code,
                        "original_column": original_column,
                        "nearest_fraction": nearest_fraction,
                        "n_original": len(original_values),
                        "n_interpolated": len(interpolated_values),
                        "mean_distance_to_anchor": nearest["distance_to_anchor"].mean(),
                        "maximum_distance_to_anchor": nearest["distance_to_anchor"].max(),
                        "original_mean": original_values.mean(),
                        "interpolated_mean": interpolated_values.mean(),
                        "mean_difference": (
                            interpolated_values.mean()
                            - original_values.mean()
                        ),
                        "original_std": original_values.std(ddof=1),
                        "interpolated_std": interpolated_values.std(ddof=1),
                        "original_p05": original_quantiles.loc[0.05],
                        "interpolated_p05": interpolated_quantiles.loc[0.05],
                        "difference_p05": (
                            interpolated_quantiles.loc[0.05]
                            - original_quantiles.loc[0.05]
                        ),
                        "original_p17": original_quantiles.loc[0.17],
                        "interpolated_p17": interpolated_quantiles.loc[0.17],
                        "difference_p17": (
                            interpolated_quantiles.loc[0.17]
                            - original_quantiles.loc[0.17]
                        ),
                        "original_p50": original_quantiles.loc[0.50],
                        "interpolated_p50": interpolated_quantiles.loc[0.50],
                        "difference_p50": (
                            interpolated_quantiles.loc[0.50]
                            - original_quantiles.loc[0.50]
                        ),
                        "original_p83": original_quantiles.loc[0.83],
                        "interpolated_p83": interpolated_quantiles.loc[0.83],
                        "difference_p83": (
                            interpolated_quantiles.loc[0.83]
                            - original_quantiles.loc[0.83]
                        ),
                        "original_p95": original_quantiles.loc[0.95],
                        "interpolated_p95": interpolated_quantiles.loc[0.95],
                        "difference_p95": (
                            interpolated_quantiles.loc[0.95]
                            - original_quantiles.loc[0.95]
                        ),
                    }
                )

    comparison = pd.DataFrame(
        comparison_rows
    )

    if comparison.empty:
        raise RuntimeError(
            "Diagnostic 3 produced no comparison rows. "
            "Check response codes and input columns."
        )

    selected_samples = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    comparison.to_csv(
        output_dir
        / "diagnostic_3_anchor_distribution_comparison.csv",
        index=False,
    )

    selected_samples.to_csv(
        output_dir
        / "diagnostic_3_nearest_fair_samples.csv",
        index=False,
    )

    for response_code in response_codes:
        response_data = comparison.loc[
            comparison["response_code"]
            == response_code
        ].copy()

        if response_data.empty:
            continue

        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(17, 5.5),
            sharey=True,
        )

        for ax, year in zip(
            axes,
            PROJECTION_YEARS,
        ):
            panel = (
                response_data.loc[
                    response_data["year"]
                    == year
                ]
                .set_index("pathway")
                .reindex(PATHWAY_ORDER)
                .reset_index()
            )

            x_positions = np.arange(
                len(PATHWAY_ORDER)
            )

            original_lower = (
                panel["original_p50"]
                - panel["original_p05"]
            )

            original_upper = (
                panel["original_p95"]
                - panel["original_p50"]
            )

            interpolated_lower = (
                panel["interpolated_p50"]
                - panel["interpolated_p05"]
            )

            interpolated_upper = (
                panel["interpolated_p95"]
                - panel["interpolated_p50"]
            )

            ax.errorbar(
                x_positions - 0.08,
                panel["original_p50"],
                yerr=np.vstack(
                    [
                        original_lower,
                        original_upper,
                    ]
                ),
                fmt="o",
                capsize=4,
                label="Original SEJ",
            )

            ax.errorbar(
                x_positions + 0.08,
                panel["interpolated_p50"],
                yerr=np.vstack(
                    [
                        interpolated_lower,
                        interpolated_upper,
                    ]
                ),
                fmt="s",
                capsize=4,
                label="Interpolated near anchor",
            )

            ax.set_xticks(
                x_positions
            )

            ax.set_xticklabels(
                PATHWAY_ORDER
            )

            ax.set_title(
                str(year),
                fontweight="bold",
            )

            ax.set_xlabel(
                "SEJ climate pathway"
            )

            ax.grid(
                axis="y",
                alpha=0.25,
            )

        axes[0].set_ylabel(
            f"{response_code} sea-level contribution"
        )

        handles, labels = (
            axes[0]
            .get_legend_handles_labels()
        )

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=2,
            frameon=False,
        )

        fig.suptitle(
            "Original SEJ versus interpolated near-anchor "
            f"distributions: {response_code}",
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

        save_figure(
            fig,
            output_dir
            / (
                "diagnostic_3_anchor_comparison_"
                f"{response_code}"
            ),
        )

        plt.close(fig)

    print("\nDiagnostic 3 complete.")

    return comparison


# ============================================================
# 8. RUN ALL DIAGNOSTICS
# ============================================================

def main(
    project_dir: Path,
    nearest_fraction: float = 0.05,
    max_points_per_scenario_year: int = 1500,
) -> dict[str, object]:
    set_publication_style()

    project_dir = (
        Path(project_dir)
        .expanduser()
        .resolve()
    )

    diagnostic_output_dir = (
        project_dir
        / "output"
        / "fair_sej_interpolation"
        / "diagnostics"
    )

    diagnostic_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = load_outputs(
        project_dir=project_dir
    )

    climate_space_summary = diagnostic_1_climate_space(
        fair_features=outputs["fair_features"],
        diagnostics=outputs["diagnostics"],
        output_dir=diagnostic_output_dir,
        max_points_per_scenario_year=(
            max_points_per_scenario_year
        ),
        random_seed=1234,
    )

    weight_results = diagnostic_2_weight_distributions(
        weights=outputs["weights"],
        output_dir=diagnostic_output_dir,
    )

    anchor_comparison = diagnostic_3_anchor_comparison(
        weights=outputs["weights"],
        interpolated_long=outputs["interpolated_long"],
        original_sej=outputs["original_sej"],
        output_dir=diagnostic_output_dir,
        response_codes=DEFAULT_RESPONSE_CODES,
        nearest_fraction=nearest_fraction,
        feature_scales=FEATURE_SCALES,
    )

    print(
        "\nAll diagnostics completed successfully."
    )

    print(
        f"\nOutputs saved to:\n{diagnostic_output_dir}"
    )

    return {
        "climate_space_summary": climate_space_summary,
        "weight_results": weight_results,
        "anchor_comparison": anchor_comparison,
    }


# ============================================================
# 9. COMMAND-LINE INTERFACE
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create diagnostics for the FaIR–SEJ interpolation emulator."
        )
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help=(
            "Path to the ice-sheet-sej-2025 project directory. "
            f"Default: {DEFAULT_PROJECT_DIR}"
        ),
    )

    parser.add_argument(
        "--nearest-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of FaIR samples nearest each SEJ anchor used "
            "in Diagnostic 3. Default: 0.05."
        ),
    )

    parser.add_argument(
        "--max-points-per-scenario-year",
        type=int,
        default=1500,
        help=(
            "Maximum plotted FaIR points per SSP-year group in "
            "Diagnostic 1. Default: 1500."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    results = main(
        project_dir=args.project_dir,
        nearest_fraction=args.nearest_fraction,
        max_points_per_scenario_year=(
            args.max_points_per_scenario_year
        ),
    )
