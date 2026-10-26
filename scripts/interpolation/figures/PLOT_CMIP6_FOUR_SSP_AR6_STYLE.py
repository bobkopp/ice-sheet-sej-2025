#!/usr/bin/env python3
"""Four-pathway CMIP6 ice-sheet projections, inspired by AR6 WGI Fig. 9.27.

Standalone: place next to figure_settings.py, figure_labels.py,
figure_style.py and figure_helpers.py. Run directly (also works in an IDE).

Data are the *final* emulandice2050 tension-p0.03 FaIR2 annual outputs.
No resampling, baseline shifting, or interpolation occurs in this script.
Per-realization sums precede percentile calculation (never add quantiles).

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from netCDF4 import Dataset

from figure_settings import FINAL_CORE_METHOD_ROOT, fair2_output_for
from figure_labels import SSP_LABELS
from figure_helpers import save_figure
from figure_style import BLUE, GREEN, ORANGE, PURPLE

# ============================= EDIT THESE =============================
INPUT_DIR = Path(FINAL_CORE_METHOD_ROOT) / "03_cmip6_annual"
ANNUAL_NETCDF = INPUT_DIR / "annual_ice_sheet_trajectories_2000_2300.nc"
SAMPLED_CSV = INPUT_DIR / "sampled_sej2025_realizations.csv"
IMBIE_CSV = INPUT_DIR / "imbie_observed_slr_1979_2023_monthly_rebased_to_1995_2014.csv"
OUTPUT_DIR = fair2_output_for("consolidated_final_suite", "manuscript", "four_ssp_comparison")

# Alternate: "SPLIT" = [2000,2150] and [2150,2300]; "OVERVIEW" =
# [2000,2150] and [2000,2300]. Use "BOTH" to write both arrangements.
LAYOUT = "BOTH"  # "SPLIT", "OVERVIEW", "BOTH"

# Requested four middle scenarios, displayed from lower to higher forcing.
SCENARIOS = ("ssp126", "ssp245", "ssp534-over", "ssp370")
SCENARIO_COLORS = {
    "ssp126": BLUE,
    "ssp245": GREEN,
    "ssp534-over": PURPLE,
    "ssp370": ORANGE,
}

ROWS = (("G", "GrIS"), ("E", "EAIS"), ("W", "WAIS"),
        ("TOTAL", "Total ice-sheet contribution"))

# Center interval: 17th–83rd. Optional outer 5th–95th interval.
SHOW_5_95 = False
SHOW_IMBIE = True
INNER_ALPHA = 0.16
OUTER_ALPHA = 0.055
MEDIAN_WIDTH = 2.0
FIGSIZE = (13.4, 13.0)
SAVE_PDF = True

# Optional manual y limits in cm; None = automatic with same limits across
# both columns for a given component.
Y_LIMITS = {"G": None, "E": None, "W": None, "TOTAL": None}
# ======================================================================


def canonical_ssp(s):
    s = str(s).strip().lower().replace("_", "-")
    if s in {"ssp534", "ssp534-over", "ssp534over"}:
        return "ssp534-over"
    return s


def load_data():
    for path in (ANNUAL_NETCDF, SAMPLED_CSV):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")
    meta = pd.read_csv(SAMPLED_CSV)
    if "ssp" not in meta:
        raise ValueError(f"Missing 'ssp' column in {SAMPLED_CSV}")
    labels = meta["ssp"].map(canonical_ssp).to_numpy()
    missing = [s for s in SCENARIOS if not np.any(labels == s)]
    if missing:
        raise ValueError(f"Scenarios absent from sampled input: {missing}; found: {sorted(set(labels))}")

    with Dataset(ANNUAL_NETCDF, "r") as nc:
        for v in ("years", "E", "W", "G"):
            if v not in nc.variables:
                raise ValueError(f"NetCDF missing variable '{v}': {ANNUAL_NETCDF}")
        years = np.asarray(nc.variables["years"][:], dtype=int)
        arr = {v: np.ma.filled(nc.variables[v][:], np.nan).astype(float)
               for v in ("E", "W", "G")}
    if years.ndim != 1 or len(np.unique(years)) != len(years) or np.any(np.diff(years) <= 0):
        raise ValueError("Expected strictly increasing, unique annual years")
    if any(a.shape != (len(meta), len(years)) for a in arr.values()):
        raise ValueError("Annual component arrays must be (number of CSV rows, number of years)")
    if any(not np.isfinite(a).all() for a in arr.values()):
        raise ValueError("Non-finite annual ice-sheet realizations")
    arr["TOTAL"] = arr["G"] + arr["E"] + arr["W"]
    print("Annual input:", ANNUAL_NETCDF)
    print("Sample metadata:", SAMPLED_CSV)
    for s in SCENARIOS:
        print(f"  {SSP_LABELS[s]}: {(labels == s).sum():,} realizations")
    return years, labels, arr


def read_imbie():
    if not SHOW_IMBIE:
        return None
    if not IMBIE_CSV.is_file():
        print(f"NOTE: historical IMBIE file absent; drawing projections only: {IMBIE_CSV}")
        return None
    df = pd.read_csv(IMBIE_CSV)
    needed = {"Date", "G_cm", "E_cm", "W_cm"}
    if not needed.issubset(df):
        raise ValueError(f"Missing IMBIE columns: {sorted(needed - set(df.columns))}")
    dates = pd.DatetimeIndex(pd.to_datetime(df["Date"], errors="raise"))
    starts = pd.DatetimeIndex(pd.to_datetime(dates.year.astype(str) + "-01-01"))
    ends = pd.DatetimeIndex(pd.to_datetime((dates.year + 1).astype(str) + "-01-01"))
    decimal = dates.year.to_numpy() + ((dates - starts).days / (ends - starts).days)
    obs = {v: df[f"{v}_cm"].to_numpy(dtype=float) for v in ("G", "E", "W")}
    obs["TOTAL"] = obs["G"] + obs["E"] + obs["W"]
    return decimal, obs


def quantiles(years, labels, arr):
    summary = {}
    percentiles = [5, 17, 50, 83, 95] if SHOW_5_95 else [17, 50, 83]
    for key, _ in ROWS:
        summary[key] = {}
        for scenario in SCENARIOS:
            selected = arr[key][labels == scenario, :]
            q = np.percentile(selected, percentiles, axis=0)
            summary[key][scenario] = dict(zip(percentiles, q))
    return summary


def make_figure(layout, years, summary, historical):
    windows = ((2000, 2150), (2150, 2300)) if layout == "SPLIT" else ((2000, 2150), (2000, 2300))
    fig, axes = plt.subplots(4, 2, figsize=FIGSIZE, sharex=False, sharey="row")
    for row_index, (component, name) in enumerate(ROWS):
        # Calculate y ranges using the percentiles actually displayed.
        ends = [summary[component][s][95 if SHOW_5_95 else 83] for s in SCENARIOS]
        starts = [summary[component][s][5 if SHOW_5_95 else 17] for s in SCENARIOS]
        low = min(float(np.min(x)) for x in starts)
        high = max(float(np.max(x)) for x in ends)
        if historical is not None:
            low = min(low, float(np.min(historical[1][component])))
            high = max(high, float(np.max(historical[1][component])))
        pad = max((high - low) * 0.07, 1.0)
        ylim = Y_LIMITS[component] if Y_LIMITS[component] is not None else (low - pad, high + pad)
        for col, (start, stop) in enumerate(windows):
            ax = axes[row_index, col]
            t = (years >= start) & (years <= stop)
            if not np.any(t):
                raise ValueError(f"No projection years within {start}–{stop}")
            for scenario in SCENARIOS:
                q = summary[component][scenario]
                color = SCENARIO_COLORS[scenario]
                if SHOW_5_95:
                    ax.fill_between(years[t], q[5][t], q[95][t], color=color,
                                    alpha=OUTER_ALPHA, linewidth=0, zorder=1)
                ax.fill_between(years[t], q[17][t], q[83][t], color=color,
                                alpha=INNER_ALPHA, linewidth=0, zorder=2)
                ax.plot(years[t], q[50][t], color=color, linewidth=MEDIAN_WIDTH,
                        zorder=4, solid_capstyle="round")
            if historical is not None:
                ty, obs = historical
                o = (ty >= start) & (ty <= stop)
                if np.any(o):
                    ax.plot(ty[o], obs[component][o], color="black", lw=1.9, zorder=6)
            ax.axhline(0, color="0.65", linewidth=0.7, zorder=0)
            ax.set_xlim(start, stop)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", color="0.9", linewidth=0.7)
            ax.tick_params(direction="out", labelsize=9)
            ax.set_xticks(np.arange(start, stop + 1, 50))
            if col == 0:
                ax.set_ylabel(f"{name}\nSLR contribution (cm)", fontsize=10.5, fontweight="bold")
            if row_index == 0:
                ax.set_title(f"{start}–{stop}", fontweight="bold", fontsize=12)
            if row_index == len(ROWS) - 1:
                ax.set_xlabel("Year", fontweight="bold")
    handles = [Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2.8,
                      label=SSP_LABELS[s]) for s in SCENARIOS]
    handles.append(Patch(facecolor="0.5", alpha=INNER_ALPHA, label="17th–83rd percentile"))
    if SHOW_5_95:
        handles.append(Patch(facecolor="0.5", alpha=OUTER_ALPHA, label="5th–95th percentile"))
    if historical is not None:
        handles.append(Line2D([0], [0], color="black", lw=2, label="IMBIE observations"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.967),
               ncol=min(len(handles), 6), frameon=False, fontsize=10)
    fig.suptitle("Ice-sheet contributions to global mean sea-level rise", fontsize=16,
                 fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.905, bottom=0.065, left=0.145, right=0.985,
                        hspace=0.29, wspace=0.10)
    outfile = OUTPUT_DIR / f"CMIP6_four_SSPs_GrIS_EAIS_WAIS_Total_{layout.lower()}.png"
    saved = save_figure(fig, outfile)
    print("Saved:", saved)
    if SAVE_PDF:
        pdf = outfile.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        print("Saved:", pdf)
    plt.close(fig)


def main():
    if LAYOUT not in {"SPLIT", "OVERVIEW", "BOTH"}:
        raise ValueError("LAYOUT must be SPLIT, OVERVIEW, or BOTH")
    years, labels, arrays = load_data()
    summary = quantiles(years, labels, arrays)
    history = read_imbie()
    for mode in (("SPLIT", "OVERVIEW") if LAYOUT == "BOTH" else (LAYOUT,)):
        make_figure(mode, years, summary, history)


if __name__ == "__main__":
    main()
