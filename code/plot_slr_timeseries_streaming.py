#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot the out-of-core FaIR–SEJ NetCDF time-series output.

The script reads one response/scenario subset at a time, so it does not
load the complete NetCDF cube into memory.

Expected inputs:
    fair_sej_timeseries.nc
    fair_sej_decreasing_intervals.csv

Outputs:
    PNG and PDF figures plus compact quantile-summary CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from netCDF4 import Dataset
except ImportError as exc:
    raise ImportError(
        "Install netCDF4 with `conda install netcdf4` or "
        "`pip install netCDF4`."
    ) from exc


DEFAULT_PROJECT_DIR = Path(
    "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025"
)

QUANTILES = np.array([0.05, 0.17, 0.50, 0.83, 0.95])
ANCHOR_YEARS = [2005, 2100, 2200, 2300]
SCENARIO_ORDER = [
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP3-7.0",
    "SSP5-8.5",
]
COMPONENT_ORDER = ["E", "W", "G", "AIS"]
COMPONENT_LABELS = {
    "E": "EAIS",
    "W": "WAIS",
    "G": "GrIS",
    "AIS": "Total AIS",
    "SLR": "Total ice-sheet SLR",
}


def decode_strings(values: np.ndarray) -> np.ndarray:
    output = []
    for value in values:
        if isinstance(value, bytes):
            output.append(value.decode("utf-8"))
        else:
            output.append(str(value))
    return np.asarray(output, dtype=object)


def set_style() -> None:
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
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )


def ordered_scenarios(values: np.ndarray) -> list[str]:
    available = set(values.tolist())
    ordered = [
        scenario
        for scenario in SCENARIO_ORDER
        if scenario in available
    ]
    return ordered + sorted(available - set(ordered))


def response_index(
    response_codes: np.ndarray,
    code: str,
) -> int:
    matches = np.where(response_codes == code)[0]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one response named {code}; found {len(matches)}."
        )
    return int(matches[0])


def load_subset(
    contribution,
    sample_indices: np.ndarray,
    response_position: int,
) -> np.ndarray:
    return np.asarray(
        contribution[
            sample_indices,
            :,
            response_position,
        ],
        dtype=np.float32,
    )


def quantile_frame(
    values: np.ndarray,
    years: np.ndarray,
    scenario: str,
    code: str,
) -> pd.DataFrame:
    q = np.nanquantile(values, QUANTILES, axis=0)
    return pd.DataFrame(
        {
            "scenario_label": scenario,
            "response_code": code,
            "year": years,
            "p05": q[0],
            "p17": q[1],
            "p50": q[2],
            "p83": q[3],
            "p95": q[4],
        }
    )


def plot_total_slr(
    ds: Dataset,
    years: np.ndarray,
    scenarios: list[str],
    scenario_labels: np.ndarray,
    response_codes: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    contribution = ds.variables["sea_level_contribution"]
    slr_position = response_index(response_codes, "SLR")
    summaries = []

    fig, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(10, 3.5 * len(scenarios)),
        sharex=True,
    )
    if len(scenarios) == 1:
        axes = np.asarray([axes])

    for ax, scenario in zip(axes, scenarios):
        sample_indices = np.where(
            scenario_labels == scenario
        )[0]
        values = load_subset(
            contribution,
            sample_indices,
            slr_position,
        )
        summary = quantile_frame(
            values,
            years,
            scenario,
            "SLR",
        )
        summaries.append(summary)

        ax.fill_between(
            years,
            summary["p05"],
            summary["p95"],
            alpha=0.20,
            label="5th–95th percentile",
        )
        ax.fill_between(
            years,
            summary["p17"],
            summary["p83"],
            alpha=0.35,
            label="17th–83rd percentile",
        )
        ax.plot(
            years,
            summary["p50"],
            linewidth=2.0,
            label="Median",
        )

        anchor_mask = np.isin(years, ANCHOR_YEARS)
        ax.scatter(
            years[anchor_mask],
            summary.loc[anchor_mask, "p50"],
            s=28,
            zorder=4,
            label="Temporal anchors",
        )

        for anchor_year in ANCHOR_YEARS:
            ax.axvline(
                anchor_year,
                linestyle=":",
                linewidth=0.7,
                alpha=0.45,
            )

        ax.axhline(0.0, linewidth=0.8, alpha=0.6)
        ax.set_title(scenario, fontweight="bold")
        ax.set_ylabel("Cumulative GMSL contribution (mm)")
        ax.grid(alpha=0.20)
        del values

    axes[-1].set_xlabel("Year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Total ice-sheet sea-level contribution trajectories",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.97])
    save_figure(
        fig,
        output_dir / "figure_1_total_slr_fan_plots",
    )
    plt.close(fig)

    output = pd.concat(summaries, ignore_index=True)
    output.to_csv(
        output_dir / "figure_1_total_slr_quantiles.csv",
        index=False,
    )
    return output


def plot_components(
    ds: Dataset,
    years: np.ndarray,
    scenarios: list[str],
    scenario_labels: np.ndarray,
    response_codes: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    contribution = ds.variables["sea_level_contribution"]
    available = [
        code
        for code in COMPONENT_ORDER
        if code in set(response_codes.tolist())
    ]
    summaries = []

    for scenario in scenarios:
        sample_indices = np.where(
            scenario_labels == scenario
        )[0]
        fig, ax = plt.subplots(figsize=(10, 6))

        for code in available:
            position = response_index(response_codes, code)
            values = load_subset(
                contribution,
                sample_indices,
                position,
            )
            summary = quantile_frame(
                values,
                years,
                scenario,
                code,
            )
            summaries.append(summary)
            ax.plot(
                years,
                summary["p50"],
                linewidth=2.0,
                label=COMPONENT_LABELS[code],
            )
            del values

        for anchor_year in ANCHOR_YEARS:
            ax.axvline(
                anchor_year,
                linestyle=":",
                linewidth=0.7,
                alpha=0.45,
            )

        ax.axhline(0.0, linewidth=0.8, alpha=0.6)
        ax.set_title(
            f"Ice-sheet component medians: {scenario}",
            fontweight="bold",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative GMSL contribution (mm)")
        ax.grid(alpha=0.20)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()

        safe = scenario.lower().replace(".", "p").replace("-", "_")
        save_figure(
            fig,
            output_dir / f"figure_2_component_trajectories_{safe}",
        )
        plt.close(fig)

    output = pd.concat(summaries, ignore_index=True)
    output.to_csv(
        output_dir / "figure_2_component_quantiles.csv",
        index=False,
    )
    return output


def plot_rates(
    ds: Dataset,
    years: np.ndarray,
    scenarios: list[str],
    scenario_labels: np.ndarray,
    response_codes: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    contribution = ds.variables["sea_level_contribution"]
    slr_position = response_index(response_codes, "SLR")
    rate_years = years[1:]
    year_differences = np.diff(years).astype(np.float32)
    summaries = []

    fig, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(10, 3.5 * len(scenarios)),
        sharex=True,
    )
    if len(scenarios) == 1:
        axes = np.asarray([axes])

    for ax, scenario in zip(axes, scenarios):
        sample_indices = np.where(
            scenario_labels == scenario
        )[0]
        values = load_subset(
            contribution,
            sample_indices,
            slr_position,
        )
        rates = np.diff(values, axis=1) / year_differences[None, :]
        summary = quantile_frame(
            rates,
            rate_years,
            scenario,
            "SLR_rate",
        )
        summaries.append(summary)

        ax.fill_between(
            rate_years,
            summary["p05"],
            summary["p95"],
            alpha=0.20,
            label="5th–95th percentile",
        )
        ax.fill_between(
            rate_years,
            summary["p17"],
            summary["p83"],
            alpha=0.35,
            label="17th–83rd percentile",
        )
        ax.plot(
            rate_years,
            summary["p50"],
            linewidth=2.0,
            label="Median annual rate",
        )

        for anchor_year in [2100, 2200, 2300]:
            ax.axvline(
                anchor_year,
                linestyle=":",
                linewidth=0.7,
                alpha=0.45,
            )

        ax.axhline(0.0, linewidth=0.8, alpha=0.6)
        ax.set_title(scenario, fontweight="bold")
        ax.set_ylabel("GMSL contribution rate (mm/yr)")
        ax.grid(alpha=0.20)
        del values, rates

    axes[-1].set_xlabel("Year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Annual rate implied by piecewise-linear interpolation",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.97])
    save_figure(
        fig,
        output_dir / "figure_3_total_slr_annual_rate_fans",
    )
    plt.close(fig)

    output = pd.concat(summaries, ignore_index=True)
    output.to_csv(
        output_dir / "figure_3_total_slr_rate_quantiles.csv",
        index=False,
    )
    return output


def plot_anchor_distributions(
    ds: Dataset,
    years: np.ndarray,
    scenarios: list[str],
    scenario_labels: np.ndarray,
    response_codes: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    contribution = ds.variables["sea_level_contribution"]
    slr_position = response_index(response_codes, "SLR")
    anchor_positions = np.asarray(
        [int(np.where(years == year)[0][0]) for year in ANCHOR_YEARS]
    )

    summaries = []
    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario in scenarios:
        sample_indices = np.where(
            scenario_labels == scenario
        )[0]
        values = np.asarray(
            contribution[
                sample_indices,
                anchor_positions,
                slr_position,
            ],
            dtype=np.float32,
        )
        q = np.nanquantile(values, QUANTILES, axis=0)
        summary = pd.DataFrame(
            {
                "scenario_label": scenario,
                "response_code": "SLR",
                "year": ANCHOR_YEARS,
                "p05": q[0],
                "p17": q[1],
                "p50": q[2],
                "p83": q[3],
                "p95": q[4],
            }
        )
        summaries.append(summary)

        ax.errorbar(
            ANCHOR_YEARS,
            summary["p50"],
            yerr=np.vstack(
                [
                    summary["p50"] - summary["p05"],
                    summary["p95"] - summary["p50"],
                ]
            ),
            marker="o",
            capsize=4,
            linewidth=1.7,
            label=scenario,
        )
        del values

    ax.axhline(0.0, linewidth=0.8, alpha=0.6)
    ax.set_xticks(ANCHOR_YEARS)
    ax.set_title("Temporal anchor distributions", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative GMSL contribution (mm)")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "figure_4_temporal_anchor_distributions",
    )
    plt.close(fig)

    output = pd.concat(summaries, ignore_index=True)
    output.to_csv(
        output_dir / "figure_4_temporal_anchor_quantiles.csv",
        index=False,
    )
    return output


def plot_decreasing(
    decreasing_path: Path,
    scenarios: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    data = pd.read_csv(decreasing_path)
    selected_codes = [
        code
        for code in ["E", "W", "G", "SLR"]
        if code in set(data["response_code"])
    ]
    selected = data.loc[
        data["response_code"].isin(selected_codes)
    ].copy()
    selected["interval"] = (
        selected["start_year"].astype(str)
        + "–"
        + selected["end_year"].astype(str)
    )
    intervals = ["2005–2100", "2100–2200", "2200–2300"]

    for code in selected_codes:
        panel = selected.loc[selected["response_code"] == code]
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(scenarios))
        width = 0.24

        for index, interval in enumerate(intervals):
            values = []
            for scenario in scenarios:
                match = panel.loc[
                    (panel["scenario_label"] == scenario)
                    & (panel["interval"] == interval),
                    "percent_decreasing",
                ]
                values.append(
                    float(match.iloc[0])
                    if not match.empty
                    else np.nan
                )

            ax.bar(
                x + (index - 1) * width,
                values,
                width=width,
                label=interval,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=30, ha="right")
        ax.set_ylim(0.0, 100.0)
        ax.set_title(
            "Draws with decreasing cumulative contribution: "
            f"{COMPONENT_LABELS.get(code, code)}",
            fontweight="bold",
        )
        ax.set_ylabel("Draws decreasing across interval (%)")
        ax.set_xlabel("Scenario")
        ax.grid(axis="y", alpha=0.20)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / f"figure_5_decreasing_intervals_{code}",
        )
        plt.close(fig)

    selected.to_csv(
        output_dir / "figure_5_decreasing_interval_summary.csv",
        index=False,
    )
    return selected


def main(
    project_dir: Path,
    output_dir: Path | None = None,
) -> None:
    set_style()
    project_dir = Path(project_dir).expanduser().resolve()
    timeseries_dir = (
        project_dir
        / "output"
        / "fair_sej_interpolation"
        / "timeseries"
    )
    netcdf_path = timeseries_dir / "fair_sej_timeseries.nc"
    decreasing_path = (
        timeseries_dir / "fair_sej_decreasing_intervals.csv"
    )

    for path in [netcdf_path, decreasing_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}\n\n"
                "Run generate_slr_timeseries_streaming.py first."
            )

    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else timeseries_dir / "figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = Dataset(netcdf_path, "r")
    try:
        years = np.asarray(ds.variables["year"][:], dtype=int)
        scenario_labels = decode_strings(
            ds.variables["scenario_label"][:]
        )
        response_codes = decode_strings(
            ds.variables["response_code"][:]
        )
        scenarios = ordered_scenarios(scenario_labels)

        print("Creating plots from NetCDF...")
        print(f"  Samples: {len(scenario_labels):,}")
        print(f"  Years: {len(years):,}")
        print(f"  Responses: {len(response_codes):,}")

        plot_total_slr(
            ds,
            years,
            scenarios,
            scenario_labels,
            response_codes,
            out_dir,
        )
        print("  Figure 1 complete.")

        plot_components(
            ds,
            years,
            scenarios,
            scenario_labels,
            response_codes,
            out_dir,
        )
        print("  Figure 2 complete.")

        plot_rates(
            ds,
            years,
            scenarios,
            scenario_labels,
            response_codes,
            out_dir,
        )
        print("  Figure 3 complete.")

        plot_anchor_distributions(
            ds,
            years,
            scenarios,
            scenario_labels,
            response_codes,
            out_dir,
        )
        print("  Figure 4 complete.")

        plot_decreasing(
            decreasing_path,
            scenarios,
            out_dir,
        )
        print("  Figure 5 complete.")
    finally:
        ds.close()

    print("\nAll plots completed successfully.")
    print(f"Figures saved to:\n{out_dir}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        project_dir=args.project_dir,
        output_dir=args.output_dir,
    )
