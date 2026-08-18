#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_b25_b19style_slr_through_time_figures.py

Calculate and plot climate-conditioned B25 ice-sheet contributions to sea-level
rise through time after the three-way B19-style LOW/MODERATE/HIGH selector.

INPUT
-----
output/b19style_baseline/03_temporal_interpolation_pchip/
    b25_b19style_pchip_decadal_trajectories.csv

The input now contains one through-time trajectory per pooled FaIR realization,
with metadata recording the FaIR SSP and which B25 library was selected:

    LOW      = 2.0 C
    MODERATE = 3.5 C
    HIGH     = 5.0 C

OUTPUTS
-------
1. Decadal SLR percentile summary by SSP, component, and year.
2. Realized LOW/MODERATE/HIGH selection fractions by SSP.
3. One five-panel figure per SSP showing EAIS, WAIS, Total AIS, GrIS, and total
   ice-sheet SLR with median, 17-83%, and 5-95% ranges.
4. One compact comparison figure showing total ice-sheet SLR across all SSPs.
5. Key-year percentile table for 2050, 2100, 2200, and 2300.

All plotted values are meters of global mean sea-level equivalent.
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

TRAJECTORY_FILE = INPUT_DIR / "b25_b19style_pchip_decadal_trajectories.csv"

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "04_slr_through_time"
)
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

COMPONENT_ORDER = [
    "EAIS",
    "WAIS",
    "Total AIS",
    "GrIS",
    "Total ice-sheet SLR",
]

SCENARIO_ORDER = [
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP3-7.0",
    "SSP5-8.5",
]

SELECTION_ORDER = ["LOW", "MODERATE", "HIGH"]

MEDIAN_LINEWIDTH = 2.2
INNER_ALPHA = 0.28
OUTER_ALPHA = 0.12
FIG_DPI = 300


# ============================================================
# 2. LOAD TRAJECTORIES
# ============================================================

def load_trajectories(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Trajectory file not found:\n{file_path}\n\nRun Script 03 first."
        )

    df = pd.read_csv(file_path)

    required = {
        "fair_sample_id",
        "scenario",
        "scenario_label",
        "selected_library",
        "selected_warming_anchor",
        "year",
        "component",
        "value_mm",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Trajectory file is missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["value_mm"] = pd.to_numeric(df["value_mm"], errors="raise")

    if not np.isfinite(df["value_mm"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite SLR values found.")

    return df


# ============================================================
# 3. CALCULATE THROUGH-TIME SLR SUMMARY BY SSP
# ============================================================

def calculate_slr_summary(trajectories: pd.DataFrame) -> pd.DataFrame:
    records = []

    for (scenario_label, component, year), group in trajectories.groupby(
        ["scenario_label", "component", "year"],
        sort=False,
    ):
        values_mm = group["value_mm"].to_numpy(dtype=float)
        q = np.quantile(values_mm, [0.05, 0.17, 0.50, 0.83, 0.95])

        records.append(
            {
                "scenario_label": scenario_label,
                "component": component,
                "year": year,
                "n_samples": len(values_mm),
                "mean_mm": np.mean(values_mm),
                "p05_mm": q[0],
                "p17_mm": q[1],
                "p50_mm": q[2],
                "p83_mm": q[3],
                "p95_mm": q[4],
            }
        )

    summary = pd.DataFrame(records)

    summary["component"] = pd.Categorical(
        summary["component"],
        categories=COMPONENT_ORDER,
        ordered=True,
    )
    summary["scenario_label"] = pd.Categorical(
        summary["scenario_label"],
        categories=SCENARIO_ORDER,
        ordered=True,
    )

    summary = summary.sort_values(
        ["scenario_label", "component", "year"]
    ).reset_index(drop=True)

    for column in ["mean_mm", "p05_mm", "p17_mm", "p50_mm", "p83_mm", "p95_mm"]:
        summary[column.replace("_mm", "_m")] = summary[column] / 1000.0

    return summary


# ============================================================
# 4. SELECTION FRACTIONS BY SSP
# ============================================================

def calculate_selection_fractions(trajectories: pd.DataFrame) -> pd.DataFrame:
    # One metadata row per FaIR sample is enough; component/year repeats are removed.
    sample_metadata = trajectories[
        ["fair_sample_id", "scenario_label", "selected_library", "selected_warming_anchor"]
    ].drop_duplicates(subset=["fair_sample_id"])

    counts = (
        sample_metadata.groupby(["scenario_label", "selected_library"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=SCENARIO_ORDER, columns=SELECTION_ORDER, fill_value=0)
    )

    fractions = counts.div(counts.sum(axis=1), axis=0)
    fractions.columns = [f"fraction_{c.lower()}" for c in fractions.columns]
    fractions = fractions.reset_index()
    return fractions


# ============================================================
# 5. BASIC VALIDATION
# ============================================================

def validate_summary(summary: pd.DataFrame) -> None:
    counts = (
        summary.groupby(["scenario_label", "component"], observed=True)["year"]
        .nunique()
    )

    print("\nNumber of decades per SSP/component:")
    print(counts)

    if not (counts == 31).all():
        raise ValueError("Expected 31 decadal values from 2000 through 2300.")

    print("\nMedian Total ice-sheet SLR at elicitation years (m):")
    check = summary.loc[
        (summary["component"] == "Total ice-sheet SLR")
        & summary["year"].isin([2100, 2200, 2300]),
        ["scenario_label", "year", "p05_m", "p17_m", "p50_m", "p83_m", "p95_m"],
    ]
    print(check.to_string(index=False))


# ============================================================
# 6. DRAW ONE UNCERTAINTY SERIES
# ============================================================

def draw_summary_series(ax, group: pd.DataFrame, label: str | None = None):
    group = group.sort_values("year")
    years = group["year"].to_numpy()
    p05 = group["p05_m"].to_numpy()
    p17 = group["p17_m"].to_numpy()
    p50 = group["p50_m"].to_numpy()
    p83 = group["p83_m"].to_numpy()
    p95 = group["p95_m"].to_numpy()

    line = ax.plot(
        years,
        p50,
        linewidth=MEDIAN_LINEWIDTH,
        label=label,
    )[0]
    color = line.get_color()

    ax.fill_between(years, p05, p95, color=color, alpha=OUTER_ALPHA, linewidth=0)
    ax.fill_between(years, p17, p83, color=color, alpha=INNER_ALPHA, linewidth=0)

    return line


# ============================================================
# 7. ONE FIVE-PANEL FIGURE PER SSP
# ============================================================

def make_ssp_component_figures(summary: pd.DataFrame) -> None:
    for scenario_label in SCENARIO_ORDER:
        scenario_df = summary.loc[summary["scenario_label"] == scenario_label]
        if scenario_df.empty:
            continue

        fig, axes = plt.subplots(
            nrows=5,
            ncols=1,
            figsize=(8.5, 13.0),
            sharex=True,
        )

        for ax, component in zip(axes, COMPONENT_ORDER):
            group = scenario_df.loc[scenario_df["component"] == component]
            draw_summary_series(ax, group)

            ax.axhline(0, linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_ylabel(f"{component}\n(m)")
            ax.grid(axis="y", alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Year")
        axes[-1].set_xlim(2000, 2300)
        axes[-1].set_xticks([2000, 2050, 2100, 2150, 2200, 2250, 2300])

        fig.suptitle(scenario_label, y=0.995)
        fig.legend(
            handles=[
                Line2D([0], [0], linewidth=MEDIAN_LINEWIDTH, label="Median"),
                Patch(alpha=INNER_ALPHA, label="17–83%"),
                Patch(alpha=OUTER_ALPHA, label="5–95%"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=3,
            frameon=False,
        )

        fig.subplots_adjust(
            top=0.94,
            bottom=0.07,
            left=0.15,
            right=0.97,
            hspace=0.15,
        )

        safe = scenario_label.lower().replace("-", "").replace(".", "").replace(" ", "_")
        outfile = FIGURE_DIR / f"b25_b19style_slr_through_time_{safe}.png"
        fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved:\n{outfile}")


# ============================================================
# 8. COMPACT TOTAL ICE-SHEET SLR COMPARISON
# ============================================================

def make_total_slr_comparison(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    total_df = summary.loc[summary["component"] == "Total ice-sheet SLR"]

    for scenario_label in SCENARIO_ORDER:
        group = total_df.loc[total_df["scenario_label"] == scenario_label]
        if group.empty:
            continue
        draw_summary_series(ax, group, label=scenario_label)

    ax.axhline(0, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total ice-sheet SLR (m)")
    ax.set_xlim(2000, 2300)
    ax.set_xticks([2000, 2050, 2100, 2150, 2200, 2250, 2300])
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    outfile = FIGURE_DIR / "b25_b19style_total_ice_sheet_slr_all_ssps.png"
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved:\n{outfile}")


# ============================================================
# 9. KEY-YEAR TABLE
# ============================================================

def save_key_year_table(summary: pd.DataFrame) -> None:
    key_years = [2050, 2100, 2200, 2300]
    table = summary.loc[
        summary["year"].isin(key_years),
        [
            "scenario_label",
            "component",
            "year",
            "p05_m",
            "p17_m",
            "p50_m",
            "p83_m",
            "p95_m",
        ],
    ].copy()

    outfile = OUTPUT_DIR / "b25_b19style_slr_key_year_percentiles_m.csv"
    table.to_csv(outfile, index=False)
    print(f"\nSaved:\n{outfile}")


# ============================================================
# 10. MAIN
# ============================================================

def main():
    print("=" * 72)
    print("B25 B19-STYLE THREE-WAY SLR CONTRIBUTIONS THROUGH TIME")
    print("=" * 72)

    trajectories = load_trajectories(TRAJECTORY_FILE)

    print(f"\nTrajectory rows: {len(trajectories):,}")
    print(f"Unique FaIR samples: {trajectories['fair_sample_id'].nunique():,}")

    summary = calculate_slr_summary(trajectories)
    selection_fractions = calculate_selection_fractions(trajectories)

    validate_summary(summary)

    print("\nRealized LOW / MODERATE / HIGH selection fractions by SSP:")
    print(selection_fractions.round(4).to_string(index=False))

    summary_file = OUTPUT_DIR / "b25_b19style_slr_decadal_summary.csv"
    selection_file = OUTPUT_DIR / "b25_b19style_selection_fractions_by_ssp.csv"

    summary.to_csv(summary_file, index=False)
    selection_fractions.to_csv(selection_file, index=False)

    print(f"\nSaved:\n{summary_file}")
    print(f"\nSaved:\n{selection_file}")

    save_key_year_table(summary)
    make_ssp_component_figures(summary)
    make_total_slr_comparison(summary)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    return {
        "trajectories": trajectories,
        "summary": summary,
        "selection_fractions": selection_fractions,
    }


if __name__ == "__main__":
    results = main()
