#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Decompose the original B25 2300 upper tail by ice sheet and process.

author:elizabeth
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from B25data_processing import load_and_prepare_data, project_dir, data_file
except ImportError as exc:
    raise ImportError(
        "Could not import B25data_processing.py. Save this script in the same "
        "project directory, or add that directory to sys.path."
    ) from exc

TARGET_YEAR = 2300
WARMING_LEVELS = [3.5, 5.0]
ICE_SHEET_ORDER = ["EAIS", "WAIS", "GrIS"]
PROCESS_ORDER = [
    "Accumulation",
    "Discharge",
    "Runoff",
    "Runoff minus accumulation",
]
PROCESS_CODE_TO_NAME = {
    "A": "Accumulation",
    "D": "Discharge",
    "R": "Runoff",
    "R-A": "Runoff minus accumulation",
}
QUANTILES = [0.01, 0.05, 0.17, 0.50, 0.83, 0.95, 0.99]
UPPER_TAIL_START = 0.90
UNIT_DIVISOR = 10.0
UNIT_LABEL = "cm"
EXCEEDANCE_THRESHOLDS = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]

OUTPUT_DIR = (
    Path(project_dir)
    / "output"
    / "temporal_coupling"
    / "process_tail_diagnostics_2300"
)
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def print_heading(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {frame.columns.tolist()}"
        )


def standardize_process_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    require_columns(
        output,
        ["sample_id", "year", "warming", "ice_sheet", "value"],
        "process dataframe",
    )
    output["year"] = pd.to_numeric(output["year"], errors="raise").astype(int)
    output["warming"] = pd.to_numeric(output["warming"], errors="raise")
    output["value"] = pd.to_numeric(output["value"], errors="raise")

    if "process_code" in output.columns:
        fallback = (
            output["process"]
            if "process" in output.columns
            else output["process_code"].astype(str)
        )
        output["process"] = (
            output["process_code"].astype(str).map(PROCESS_CODE_TO_NAME).fillna(fallback)
        )
    elif "process" not in output.columns:
        raise KeyError("Process data must contain 'process_code' or 'process'.")

    name_map = {
        "A": "Accumulation",
        "Accumulation": "Accumulation",
        "D": "Discharge",
        "Discharge": "Discharge",
        "R": "Runoff",
        "Runoff": "Runoff",
        "R-A": "Runoff minus accumulation",
        "Runoff minus accumulation": "Runoff minus accumulation",
    }
    output["process"] = output["process"].astype(str).map(name_map).fillna(
        output["process"].astype(str)
    )

    if (output["process"] == "Runoff minus accumulation").any():
        return output

    original = output.loc[
        output["process"].isin(["Accumulation", "Discharge", "Runoff"])
    ].copy()
    index_columns = ["sample_id", "year", "warming", "ice_sheet"]
    for optional in ["scenario_name", "ice_sheet_code"]:
        if optional in original.columns:
            index_columns.append(optional)

    net = (
        original.loc[original["process"].isin(["Accumulation", "Runoff"])]
        .pivot(index=index_columns, columns="process", values="value")
        .dropna(subset=["Accumulation", "Runoff"])
        .reset_index()
    )
    net["value"] = net["Runoff"] + net["Accumulation"]
    net["process"] = "Runoff minus accumulation"
    if "process_code" in output.columns:
        net["process_code"] = "R-A"
    for column in output.columns:
        if column not in net.columns:
            net[column] = np.nan
    net = net[output.columns]
    return pd.concat([output, net], ignore_index=True)


def extract_process_dataframe(b25_data: dict) -> tuple[pd.DataFrame, str]:
    preferred_keys = [
        "mc_process_ext",
        "process_ext",
        "mc_process_df",
        "process_df",
        "mc_long",
        "long_df",
    ]
    required = {"sample_id", "year", "warming", "ice_sheet", "value"}
    for key in preferred_keys:
        candidate = b25_data.get(key)
        if not isinstance(candidate, pd.DataFrame):
            continue
        has_process = "process" in candidate.columns or "process_code" in candidate.columns
        if required.issubset(candidate.columns) and has_process:
            return standardize_process_frame(candidate), key

    available = {
        key: value.columns.tolist() if isinstance(value, pd.DataFrame) else type(value).__name__
        for key, value in b25_data.items()
    }
    raise KeyError(
        "Could not locate a process-level dataframe returned by "
        f"load_and_prepare_data(). Available objects: {available}"
    )


def sorted_values(frame: pd.DataFrame, ice_sheet: str, process: str, warming: float) -> np.ndarray:
    values = (
        frame.loc[
            (frame["ice_sheet"] == ice_sheet)
            & (frame["process"] == process)
            & np.isclose(frame["warming"], warming),
            "display_value",
        ]
        .sort_values()
        .to_numpy(dtype=float)
    )
    if len(values) == 0:
        raise ValueError(f"No values found for {ice_sheet}, {process}, {warming}°C.")
    return values


print_heading("LOADING ORIGINAL B25 MONTE CARLO DATA")
print(f"Project directory:\n{project_dir}")
print(f"\nB25 input file:\n{data_file}")

b25_data = load_and_prepare_data(file_path=str(data_file))
print("\nObjects returned by load_and_prepare_data():")
print(sorted(b25_data.keys()))

process_ext, source_key = extract_process_dataframe(b25_data)
print(f"\nUsing process dataframe: {source_key}")
if not np.isfinite(process_ext["value"].to_numpy(dtype=float)).all():
    raise ValueError("The process dataframe contains non-finite values.")

process_2300 = process_ext.loc[
    (process_ext["year"] == TARGET_YEAR)
    & process_ext["warming"].isin(WARMING_LEVELS)
    & process_ext["ice_sheet"].isin(ICE_SHEET_ORDER)
    & process_ext["process"].isin(PROCESS_ORDER)
].copy()
process_2300["display_value"] = process_2300["value"] / UNIT_DIVISOR
if process_2300.empty:
    raise ValueError("No matching process records were found.")

print_heading("SAMPLE COUNTS")
sample_counts = (
    process_2300.groupby(["ice_sheet", "process", "warming"], observed=True)["value"]
    .count()
    .rename("n")
    .reset_index()
)
print(sample_counts.sort_values(["ice_sheet", "process", "warming"]).to_string(index=False))
sample_counts.to_csv(OUTPUT_DIR / "b25_2300_process_sample_counts.csv", index=False)

print_heading("PROCESS DISTRIBUTION SUMMARY")
summary_records = []
for (ice_sheet, process, warming), group in process_2300.groupby(
    ["ice_sheet", "process", "warming"], observed=True
):
    values = group["display_value"]
    record = {
        "ice_sheet": ice_sheet,
        "process": process,
        "warming": float(warming),
        "n": len(values),
        "minimum": values.min(),
        "mean": values.mean(),
        "median": values.median(),
        "maximum": values.max(),
        "units": UNIT_LABEL,
    }
    for q in QUANTILES:
        record[f"p{int(round(q * 100)):02d}"] = values.quantile(q)
    summary_records.append(record)

process_summary = (
    pd.DataFrame(summary_records)
    .sort_values(["ice_sheet", "process", "warming"])
    .reset_index(drop=True)
)
print(process_summary.round(2).to_string(index=False))
process_summary.to_csv(
    OUTPUT_DIR / "b25_2300_process_distribution_summary.csv", index=False
)

print_heading("PROCESS DIFFERENCES: 5.0°C MINUS 3.5°C")
statistics = ["minimum", "p01", "p05", "p17", "median", "mean", "p83", "p95", "p99", "maximum"]
comparison_records = []
for (ice_sheet, process), group in process_summary.groupby(
    ["ice_sheet", "process"], observed=True
):
    by_warming = group.set_index("warming")
    if not {3.5, 5.0}.issubset(by_warming.index):
        continue
    for statistic in statistics:
        v35 = by_warming.loc[3.5, statistic]
        v50 = by_warming.loc[5.0, statistic]
        comparison_records.append(
            {
                "ice_sheet": ice_sheet,
                "process": process,
                "statistic": statistic,
                "value_3.5C": v35,
                "value_5.0C": v50,
                "difference_5.0C_minus_3.5C": v50 - v35,
                "units": UNIT_LABEL,
            }
        )

process_comparison = pd.DataFrame(comparison_records)
focus = process_comparison.loc[
    process_comparison["statistic"].isin(["median", "p95", "p99", "maximum"])
]
print(focus.sort_values(["ice_sheet", "process", "statistic"]).round(2).to_string(index=False))
process_comparison.to_csv(
    OUTPUT_DIR / "b25_2300_process_5p0C_minus_3p5C.csv", index=False
)

print_heading("PROCESS EXCEEDANCE TABLES")
exceedance_records = []
for (ice_sheet, process, warming), group in process_2300.groupby(
    ["ice_sheet", "process", "warming"], observed=True
):
    values = group["display_value"].to_numpy(dtype=float)
    for threshold in EXCEEDANCE_THRESHOLDS:
        exceedance_records.append(
            {
                "ice_sheet": ice_sheet,
                "process": process,
                "warming": float(warming),
                "threshold": threshold,
                "n_above": int(np.sum(values > threshold)),
                "percent_above": float(100 * np.mean(values > threshold)),
                "units": UNIT_LABEL,
            }
        )
exceedance_df = pd.DataFrame(exceedance_records)
exceedance_df.to_csv(
    OUTPUT_DIR / "b25_2300_process_exceedance_table.csv", index=False
)
print(
    exceedance_df.loc[exceedance_df["threshold"].isin([1000, 1500, 2000, 2500, 3000])]
    .sort_values(["ice_sheet", "process", "threshold", "warming"])
    .round(3)
    .to_string(index=False)
)

print_heading("CREATING FIGURES")

# Figure 1: upper-tail empirical quantile curves.
fig, axes = plt.subplots(
    nrows=len(ICE_SHEET_ORDER),
    ncols=len(PROCESS_ORDER),
    figsize=(18, 12),
    sharex=True,
)
for row, ice_sheet in enumerate(ICE_SHEET_ORDER):
    for col, process in enumerate(PROCESS_ORDER):
        ax = axes[row, col]
        available = process_2300.loc[
            (process_2300["ice_sheet"] == ice_sheet)
            & (process_2300["process"] == process)
        ]
        if available.empty:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        for warming in WARMING_LEVELS:
            values = sorted_values(process_2300, ice_sheet, process, warming)
            percentile = np.arange(1, len(values) + 1) / len(values)
            mask = percentile >= UPPER_TAIL_START
            ax.plot(percentile[mask], values[mask], linewidth=1.5, label=f"{warming}°C")
        ax.axhline(0, linewidth=0.7, alpha=0.5)
        ax.grid(alpha=0.20)
        if row == 0:
            ax.set_title(process, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"{ice_sheet}\nContribution ({UNIT_LABEL})")
        if row == len(ICE_SHEET_ORDER) - 1:
            ax.set_xlabel("Empirical percentile rank")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Original B25 process distributions at 2300: upper 10%", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIGURE_DIR / "b25_2300_process_upper_tail_quantiles.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "b25_2300_process_upper_tail_quantiles.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 2: exceedance curves.
fig, axes = plt.subplots(
    nrows=len(ICE_SHEET_ORDER),
    ncols=len(PROCESS_ORDER),
    figsize=(18, 12),
)
for row, ice_sheet in enumerate(ICE_SHEET_ORDER):
    for col, process in enumerate(PROCESS_ORDER):
        ax = axes[row, col]
        available = process_2300.loc[
            (process_2300["ice_sheet"] == ice_sheet)
            & (process_2300["process"] == process)
        ]
        if available.empty:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        for warming in WARMING_LEVELS:
            values = sorted_values(process_2300, ice_sheet, process, warming)
            exceedance = (1 - np.arange(1, len(values) + 1) / len(values)) * 100
            ax.plot(values, exceedance, linewidth=1.5, label=f"{warming}°C")
        ax.axvline(0, linewidth=0.7, alpha=0.5)
        ax.grid(alpha=0.20)
        if row == 0:
            ax.set_title(process, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"{ice_sheet}\nPercent exceeding")
        if row == len(ICE_SHEET_ORDER) - 1:
            ax.set_xlabel(f"Contribution ({UNIT_LABEL})")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Original B25 process distributions at 2300: exceedance curves", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIGURE_DIR / "b25_2300_process_exceedance_curves.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "b25_2300_process_exceedance_curves.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 3: scenario differences in upper-tail statistics.
bar_statistics = ["p95", "p99", "maximum"]
bar_data = process_comparison.loc[
    process_comparison["statistic"].isin(bar_statistics)
].copy()
bar_data["category"] = bar_data["ice_sheet"] + " — " + bar_data["process"]
category_order = [
    f"{ice_sheet} — {process}"
    for ice_sheet in ICE_SHEET_ORDER
    for process in PROCESS_ORDER
    if (
        (process_summary["ice_sheet"] == ice_sheet)
        & (process_summary["process"] == process)
    ).any()
]
x = np.arange(len(category_order))
width = 0.24
fig, ax = plt.subplots(figsize=(16, 7))
for offset, statistic in enumerate(bar_statistics):
    values = (
        bar_data.loc[bar_data["statistic"] == statistic]
        .set_index("category")
        .reindex(category_order)["difference_5.0C_minus_3.5C"]
        .to_numpy(dtype=float)
    )
    ax.bar(x + (offset - 1) * width, values, width=width, label=statistic.upper())
ax.axhline(0, linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels(category_order, rotation=45, ha="right")
ax.set_ylabel(f"5.0°C minus 3.5°C ({UNIT_LABEL})")
ax.set_title("Difference in upper-tail process statistics at 2300", fontweight="bold")
ax.grid(axis="y", alpha=0.20)
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "b25_2300_process_upper_tail_differences.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "b25_2300_process_upper_tail_differences.pdf", bbox_inches="tight")
plt.close(fig)

print_heading("LARGEST UPPER-TAIL REVERSALS")
largest_reversals = (
    process_comparison.loc[
        process_comparison["statistic"].isin(["p95", "p99", "maximum"])
    ]
    .sort_values("difference_5.0C_minus_3.5C", ascending=True)
)
print(largest_reversals.head(20).round(2).to_string(index=False))
largest_reversals.to_csv(
    OUTPUT_DIR / "b25_2300_largest_process_upper_tail_reversals.csv", index=False
)

print_heading("FINISHED")
print(f"Tables saved to:\n{OUTPUT_DIR}")
print(f"\nFigures saved to:\n{FIGURE_DIR}")
print("\nMost useful outputs:")
print("  b25_2300_process_distribution_summary.csv")
print("  b25_2300_largest_process_upper_tail_reversals.csv")
print("  figures/b25_2300_process_upper_tail_quantiles.png")
print("  figures/b25_2300_process_upper_tail_differences.png")
