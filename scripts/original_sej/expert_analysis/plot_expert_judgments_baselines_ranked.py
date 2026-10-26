"""
Expert-judgment plots for the 2025 split-panel elicitation workbook.

-----------------------
1. Reads the supplied workbook: 2025_FINAL_Split_Panels(4).xls
2. Uses separate DTT expert-ID mappings for WEA (WAIS/EAIS) and GrIS.
3. Orders all 18 experts by combined score (Statistical Accuracy x mean
   relative information for calibration variables), highest to lowest.
4. Highlights the top six experts used by the IW(0.1) DM in green.
5. Keeps only EW and IW(0.1) among the displayed DMs; IW(optimal) is omitted.
6. Uses independent x-axis limits for every subplot (no shared x-axis).
7. Adds the baseline process contribution supplied to experts as a vertical
   dashed line on target plots, using the applicable ice sheet/process/year.
8. Preserves original calibration-question units; only target questions are
   converted from mm GMSL to cm GMSL.

Input workbook columns A:F:
    expert, item, five, fifty, ninetyfive, realiz

Dependencies:
    pandas, numpy, matplotlib, xlrd

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""

from __future__ import annotations

from figure_settings import PROJECT_ROOT, DATA_ROOT, UPSTREAM_ROOT, FIGURE_ROOT, output_for, DPI, SHOW_FIGURES, B25_MC_FILE, B19_ADJUSTED_FILE, CMIP6_ANNUAL_DIR, CMIP7_ANNUAL_DIR, CMIP6_SELECTION_FILE, CMIP7_SELECTION_FILE
from figure_settings import B19_SUMMARY_FILE, EXPERT_QUESTIONS_FILE, EXPERT_WORKBOOK_FILE, PERFORMANCE_FILE, AR6_WORKBOOK_FILE, COUPLED_SEJ_FILE
from figure_helpers import save_figure, maybe_show
from figure_style import DM_COLORS, EXPERT_COLOR, WEIGHTED_COLOR, REALIZATION_COLOR, IR_COLOR, FRAME_COLOR, BASELINE_COLOR

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator


# ============================================================================
# 1. USER SETTINGS
# ============================================================================

DATA_FILE = EXPERT_WORKBOOK_FILE
OUTPUT_DIR = output_for("03_expert_judgments")

MAKE_CALIBRATION_PLOTS = True
MAKE_TARGET_PLOTS = True
SHOW_BASELINE_LINES = True

IR_SCALE = 0.10
PANELS_PER_PAGE = 3
N_REAL_EXPERTS = 18
MAX_X_TICK_INTERVALS = 10

# ---------------------------------------------------------------------------
# Expert IDs and performance ranking
# ---------------------------------------------------------------------------
# IMPORTANT: the numeric values in the workbook's `expert` column are graph
# row numbers (1-18), not the final DTT expert IDs. The WEA and GrIS panels use
# different row-number -> DTT-ID mappings.

WEA_EXPERT_ID_LABELS = {
    1: "001",
    2: "002",
    3: "007",
    4: "008",
    5: "009",
    6: "011",
    7: "015",
    8: "017",
    9: "018",
    10: "019",
    11: "020",
    12: "022",
    13: "023",
    14: "024",
    15: "025",
    16: "027",
    17: "030",
    18: "031",
}

GRIS_EXPERT_ID_LABELS = {
    1: "001",
    2: "002",
    3: "005",
    4: "007",
    5: "008",
    6: "009",
    7: "011",
    8: "013",
    9: "015",
    10: "018",
    11: "020",
    12: "022",
    13: "023",
    14: "025",
    15: "027",
    16: "029",
    17: "030",
    18: "031",
}

# Combined scores from the performance summary table shown in the 2/20/2026
# DTT output. These are the GLOBAL summary scores Roger described:
#     combined score = Statistical Accuracy x Mean rel. Info (calib vbls)
# They are used only to establish one stable expert ordering across plots.
WEA_COMBINED_SCORE = {
    1: 0.3071,
    2: 0.0235,
    3: 0.1972,
    4: 0.0407,
    5: 0.0203,
    6: 0.3312,
    7: 1.2470,
    8: 0.8049,
    9: 0.1057,
    10: 0.0006,
    11: 0.0377,
    12: 0.0180,
    13: 0.0008,
    14: 0.2498,
    15: 0.0347,
    16: 0.5570,
    17: 0.0504,
    18: 0.1513,
}

GRIS_COMBINED_SCORE = {
    1: 0.2995,
    2: 0.0229,
    3: 0.2165,
    4: 0.1938,
    5: 0.03995,
    6: 0.01994,
    7: 0.3216,
    8: 0.9930,
    9: 1.2170,
    10: 0.1029,
    11: 0.03706,
    12: 0.01765,
    13: 0.000747,
    14: 0.03409,
    15: 0.5426,
    16: 0.00000455,
    17: 0.04931,
    18: 0.1482,
}

# The top six combined-score experts are the six experts included in IW(0.1).
# These are workbook row numbers, not DTT display IDs.
WEA_TOP_SIX = [7, 8, 16, 6, 1, 14]   # 015,017,027,011,001,024
GRIS_TOP_SIX = [9, 8, 15, 7, 1, 3]   # 015,013,027,011,001,005

SHEET_CONFIG = {
    "Gr_only": {
        "expert_id_labels": GRIS_EXPERT_ID_LABELS,
        "combined_scores": GRIS_COMBINED_SCORE,
        "weighted_experts": GRIS_TOP_SIX,
        "calibration_prefix": "Greenland",
        "calibration_panel_label": "GrIS Expert Panel",
    },
    "WEA_only": {
        "expert_id_labels": WEA_EXPERT_ID_LABELS,
        "combined_scores": WEA_COMBINED_SCORE,
        "weighted_experts": WEA_TOP_SIX,
        "calibration_prefix": "Antarctic",
        "calibration_panel_label": "AIS Expert Panel",
    },
}

# The workbook contains additional DM rows after the 18 real experts. We plot
# only the first two: EW and IW(0.1). IW(optimal) remains intentionally omitted.
DM_LABELS = ["EW", "IW (0.1)"]

# ---------------------------------------------------------------------------
# Visual settings
# ---------------------------------------------------------------------------
BASELINE_LINESTYLE = (0, (5, 3))
XLABEL = "cm GMSL"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
    }
)


# ---------------------------------------------------------------------------
# Baseline process contributions supplied to experts
# ---------------------------------------------------------------------------
# Conversion: approximately 362.5 Gt of land-ice mass loss = 1 mm GMSL.
GT_PER_MM_GMSL = 362.5

# Values below are cumulative process magnitudes in Gt.
# Sign convention in the plotted GMSL contribution:
#   accumulation -> negative GMSL (mass gain)
#   runoff       -> positive GMSL (mass loss)
#   discharge    -> positive GMSL (mass loss)
#
BASELINE_PROCESS_GT = {
    "GrIS": {
        "Accumulation": {2100: 61750.0, 2200: 126750.0, 2300: 191750.0},
        "Runoff":       {2100: 47500.0, 2200:  97500.0, 2300: 147500.0},
        "Discharge":    {2100: 44650.0, 2200:  91650.0, 2300: 138650.0},
    },
    "WAIS": {
        "Accumulation": {2100:  92150.0, 2200: 189150.0, 2300: 286150.0},
        "Runoff":       {2100:      0.0, 2200:      0.0, 2300:      0.0},
        "Discharge":    {2100: 101650.0, 2200: 208650.0, 2300: 315650.0},
    },
    "EAIS": {
        "Accumulation": {2100: 102125.0, 2200: 209625.0, 2300: 317125.0},
        "Runoff":       {2100:      0.0, 2200:      0.0, 2300:      0.0},
        "Discharge":    {2100: 104975.0, 2200: 215475.0, 2300: 325975.0},
    },
}

BASELINE_GMSL_SIGN = {
    "Accumulation": -1.0,
    "Runoff": 1.0,
    "Discharge": 1.0,
}


# ============================================================================
# 2. DATA HANDLING
# ============================================================================

def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read A:F, clean missing values, and preserve calibration units."""
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"

    try:
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            usecols="A:F",
            engine=engine,
        )
    except ImportError as exc:
        if path.suffix.lower() == ".xls":
            raise ImportError(
                "Reading .xls files requires xlrd. Install it with:\n"
                "    python -m pip install xlrd\n"
                "Or save the workbook as .xlsx and point DATA_FILE to that copy."
            ) from exc
        raise

    df = df.dropna(subset=["item"]).copy()

    df["expert"] = pd.to_numeric(df["expert"], errors="coerce")
    for col in ["five", "fifty", "ninetyfive", "realiz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Match the original R treatment of the missing-value sentinel.
    bad = (
        np.isclose(df["five"], -999.6, equal_nan=False)
        & np.isclose(df["fifty"], -999.6, equal_nan=False)
        & np.isclose(df["ninetyfive"], -999.6, equal_nan=False)
    )
    df.loc[bad, ["five", "fifty", "ninetyfive"]] = np.nan

    # Calibration questions stay in their original elicitation units.
    # Target questions are stored in mm GMSL and plotted in cm GMSL.
    value_cols = ["five", "fifty", "ninetyfive", "realiz"]
    is_target = ~df["item"].astype(str).str.startswith("Calib")
    df.loc[is_target, value_cols] = df.loc[is_target, value_cols] / 10.0

    return df


def calibration_sort_key(item: str) -> int:
    match = re.search(r"(\d+)", str(item))
    return int(match.group(1)) if match else 10**9


CALIBRATION_UNITS = {
    1: "mm SLE",
    2: "Gt/yr",
    3: "Gt",
    4: "km",
    5: "mm/yr",
    6: "m",
    7: "%",
    8: "cm SLE",
    9: r"Gt CO$_2$/yr",
    10: "cm",
    11: "cm",
    12: "m",
    13: "Gt/yr",
}


def calibration_number(item: str) -> int:
    match = re.search(r"(\d+)", str(item))
    if match is None:
        raise ValueError(f"Could not determine calibration number from: {item}")
    return int(match.group(1))


def calibration_title(item: str) -> str:
    return f"Q{calibration_number(item)}"


def calibration_xlabel(item: str) -> str:
    q = calibration_number(item)
    if q not in CALIBRATION_UNITS:
        raise ValueError(f"No calibration unit specified for Q{q}")
    return CALIBRATION_UNITS[q]


def parse_target_item(item: str) -> dict[str, str] | None:
    """Parse compact target code into ice sheet, process, warming, and year."""
    s = str(item).rstrip("_")

    patterns = {
        "WAA": ("WAIS", "Accumulation"),
        "WAR": ("WAIS", "Runoff"),
        "WAD": ("WAIS", "Discharge"),
        "EAA": ("EAIS", "Accumulation"),
        "EAR": ("EAIS", "Runoff"),
        "EAD": ("EAIS", "Discharge"),
        "GrA": ("GrIS", "Accumulation"),
        "GrR": ("GrIS", "Runoff"),
        "GrD": ("GrIS", "Discharge"),
    }

    prefix = next((p for p in patterns if s.startswith(p + "_")), None)
    if prefix is None:
        return None

    remainder = s[len(prefix) + 1 :]
    match = re.fullmatch(r"(3_5|2|5)_(2100|2200|2300)", remainder)
    if match is None:
        return None

    warming_code, year = match.groups()
    warming = {"2": "2.0°C", "3_5": "3.5°C", "5": "5.0°C"}[warming_code]
    ice_sheet, process = patterns[prefix]

    return {
        "ice_sheet": ice_sheet,
        "process": process,
        "warming": warming,
        "year": year,
    }


def target_filename_label(item: str) -> str:
    parsed = parse_target_item(item)
    if parsed is None:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(item).rstrip("_"))
    warming_file = parsed["warming"].replace("°", "")
    return f"{parsed['ice_sheet']}_{parsed['process']}_{warming_file}"


def split_item_rows(item_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one item into the 18 real experts and displayed DM rows."""
    s = item_df.reset_index(drop=True).copy()
    real = s.iloc[:N_REAL_EXPERTS].copy()
    dm = s.iloc[N_REAL_EXPERTS : N_REAL_EXPERTS + len(DM_LABELS)].copy()
    return real, dm


def order_real_experts(
    real: pd.DataFrame,
    combined_scores: dict[int, float],
) -> pd.DataFrame:
    """Sort all real experts by combined score, highest to lowest."""
    ordered = real.copy()

    def score_for_row(x: float) -> float:
        if not np.isfinite(x):
            return -np.inf
        return combined_scores.get(int(x), -np.inf)

    ordered["_combined_score"] = ordered["expert"].map(score_for_row)
    ordered = (
        ordered.sort_values(
            ["_combined_score", "expert"],
            ascending=[False, True],
            kind="stable",
        )
        .drop(columns="_combined_score")
        .reset_index(drop=True)
    )
    return ordered


def get_baseline_cm(item: str) -> float | None:
    """Return applicable baseline process contribution in cm GMSL."""
    if not SHOW_BASELINE_LINES:
        return None

    parsed = parse_target_item(item)
    if parsed is None:
        return None

    ice_sheet = parsed["ice_sheet"]
    process = parsed["process"]
    year = int(parsed["year"])

    try:
        gt = BASELINE_PROCESS_GT[ice_sheet][process][year]
        sign = BASELINE_GMSL_SIGN[process]
    except KeyError:
        return None

    mm_gmsl = sign * gt / GT_PER_MM_GMSL
    return mm_gmsl / 10.0


# ============================================================================
# 3. AXIS / RANGE UTILITIES
# ============================================================================

def finite_minmax(values: pd.Series) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite values available for plotting.")
    return float(arr.min()), float(arr.max())


def compute_intrinsic_range(item_df: pd.DataFrame) -> tuple[float, float]:
    min_five, _ = finite_minmax(item_df["five"])
    _, max_95 = finite_minmax(item_df["ninetyfive"])
    span = max_95 - min_five
    return min_five - IR_SCALE * abs(span), max_95 + IR_SCALE * abs(span)


def compute_xlim(
    item_df: pd.DataFrame,
    realization: float | None = None,
    baseline: float | None = None,
) -> tuple[float, float]:
    """Compute x-limits for ONE subplot only; no cross-panel sharing."""
    min_five, _ = finite_minmax(item_df["five"])
    _, max_95 = finite_minmax(item_df["ninetyfive"])
    span = max(max_95 - min_five, 1e-12)

    lo = min_five - 1.5 * IR_SCALE * abs(span)
    hi = max_95 + 3.3 * IR_SCALE * abs(span)

    # Keep calibration realization visible.
    if realization is not None and np.isfinite(realization):
        pad = 0.05 * max(hi - lo, 1.0)
        lo = min(lo, realization - pad)
        hi = max(hi, realization + pad)

    # Keep target baseline visible without forcing any other panel to share it.
    if baseline is not None and np.isfinite(baseline):
        pad = 0.05 * max(hi - lo, 1.0)
        lo = min(lo, baseline - pad)
        hi = max(hi, baseline + pad)

    # Round outward to clean axis bounds.
    lo = math.floor(lo)
    hi = math.ceil(hi)

    if hi <= lo:
        hi = lo + 1

    return float(lo), float(hi)


# ============================================================================
# 4. ONE PANEL
# ============================================================================

def plot_item(
    ax: plt.Axes,
    item: str,
    item_df: pd.DataFrame,
    title: str,
    x_label: str,
    expert_id_labels: dict[int, str],
    combined_scores: dict[int, float],
    weighted_experts: list[int],
    is_calibration: bool,
) -> None:
    real, dm = split_item_rows(item_df)

    # Every panel uses the same performance-based ranking for its expert panel.
    real = order_real_experts(real, combined_scores)

    realization_values = item_df["realiz"].dropna().to_numpy(dtype=float)
    realization = (
        float(realization_values[0]) if realization_values.size else np.nan
    )

    baseline = None if is_calibration else get_baseline_cm(item)

    # Axis limits are computed independently for this subplot only.
    visible_df = pd.concat([real, dm], ignore_index=True)
    xlo, xhi = compute_xlim(
        visible_df,
        realization=realization if is_calibration else None,
        baseline=baseline,
    )
    ir_lo, ir_hi = compute_intrinsic_range(visible_df)

    # Highest-ranked expert is placed at the top of the expert section.
    y_real = np.arange(N_REAL_EXPERTS, 0, -1)
    y_dm = np.arange(
        N_REAL_EXPERTS + 2,
        N_REAL_EXPERTS + 2 + len(dm),
    )
    ymax = N_REAL_EXPERTS + len(DM_LABELS) + 2

    # Individual expert intervals + medians.
    for y, (_, row) in zip(y_real, real.iterrows()):
        if not all(
            np.isfinite(row[col]) for col in ["five", "fifty", "ninetyfive"]
        ):
            continue

        graph_expert = int(row["expert"]) if np.isfinite(row["expert"]) else None
        color = (
            WEIGHTED_COLOR
            if graph_expert in weighted_experts
            else EXPERT_COLOR
        )

        ax.hlines(
            y,
            row["five"],
            row["ninetyfive"],
            color=color,
            lw=2.0,
            zorder=2,
        )
        ax.plot(
            row["fifty"],
            y,
            "o",
            ms=5.4,
            color=color,
            zorder=3,
        )

    # Decision-maker solutions: EW and IW(0.1) only.
    for k, (y, (_, row)) in enumerate(zip(y_dm, dm.iterrows())):
        if not all(
            np.isfinite(row[col]) for col in ["five", "fifty", "ninetyfive"]
        ):
            continue

        color = DM_COLORS[k]
        ax.hlines(
            y,
            row["five"],
            row["ninetyfive"],
            color=color,
            lw=3.0,
            zorder=2,
        )
        ax.plot(
            row["fifty"],
            y,
            "s",
            ms=5.3,
            color=color,
            zorder=3,
        )

        ax.text(
            0.01,
            y,
            DM_LABELS[k],
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.5,
            zorder=4,
        )

    # Intrinsic-range bounds.
    ax.vlines(
        [ir_lo, ir_hi],
        ymin=0.7,
        ymax=N_REAL_EXPERTS + 1.4,
        colors=IR_COLOR,
        linestyles="--",
        lw=1.5,
        zorder=1,
    )

    # Baseline process contribution supplied to experts.
    if baseline is not None and np.isfinite(baseline):
        ax.axvline(
            baseline,
            color=BASELINE_COLOR,
            linestyle=BASELINE_LINESTYLE,
            lw=1.8,
            zorder=1.5,
        )


    # Calibration realization.
    if is_calibration and np.isfinite(realization):
        ax.axvline(
            realization,
            color=REALIZATION_COLOR,
            lw=1.8,
            zorder=1,
        )
        ax.annotate(
            "realiz.",
            xy=(realization, ymax - 0.15),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            color=REALIZATION_COLOR,
            fontsize=8.5,
        )

    # Independent x-axis for this subplot.
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0.5, ymax)

    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=MAX_X_TICK_INTERVALS,
            steps=[1, 2, 2.5, 5, 10],
            min_n_ticks=4,
        )
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    # Apply the correct sheet-specific DTT expert IDs in performance order.
    ax.set_yticks(y_real)
    ax.set_yticklabels(
        [
            expert_id_labels.get(int(e), str(int(e)))
            if np.isfinite(e)
            else ""
            for e in real["expert"]
        ]
    )

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("Expert ID", fontweight="bold")

    # Compact identifier: Q# for calibration, projection year for targets.
    ax.text(
        0.985,
        0.965,
        title,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        zorder=10,
    )

    ax.set_axisbelow(True)
    ax.grid(
        axis="both",
        which="major",
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )
    ax.grid(
        axis="x",
        which="minor",
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    ax.axhline(
        N_REAL_EXPERTS + 1,
        color="0.87",
        lw=0.8,
    )

    for spine in ax.spines.values():
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.9)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3,
        width=0.7,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        length=3,
        width=0.7,
    )


# ============================================================================
# 5. PAGINATION / SAVING
# ============================================================================

def save_pages(
    df: pd.DataFrame,
    items: list[str],
    expert_id_labels: dict[int, str],
    combined_scores: dict[int, float],
    weighted_experts: list[int],
    output_prefix: str,
    kind: str,
    title_func,
    xlabel_func,
    is_calibration: bool,
    calibration_panel_label: str | None = None,
) -> None:
    if not items:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_pages = math.ceil(len(items) / PANELS_PER_PAGE)

    for page_idx in range(n_pages):
        page_items = items[
            page_idx * PANELS_PER_PAGE : (page_idx + 1) * PANELS_PER_PAGE
        ]

        # sharex=False is intentional. Every panel gets its own x-axis scale.
        fig, axes = plt.subplots(
            PANELS_PER_PAGE,
            1,
            figsize=(8.27, 11.69),
            constrained_layout=False,
            sharex=False,
        )
        axes = np.atleast_1d(axes)

        if is_calibration:
            if calibration_panel_label is None:
                raise ValueError(
                    "calibration_panel_label must be provided for calibration plots."
                )
            fig.suptitle(
                f"Calibration Question — {calibration_panel_label}",
                fontsize=13,
                fontweight="bold",
                y=0.975,
            )
        else:
            meta = parse_target_item(page_items[0])
            if meta is not None:
                fig.suptitle(
                    f'{meta["ice_sheet"]} {meta["process"]} — {meta["warming"]}',
                    fontsize=13,
                    fontweight="bold",
                    y=0.975,
                )

        # No shared x-limit calculation here. Each call computes its own x-range.
        for ax, item in zip(axes, page_items):
            s_item = df.loc[df["item"] == item].copy()
            plot_item(
                ax=ax,
                item=item,
                item_df=s_item,
                title=title_func(item),
                x_label=xlabel_func(item),
                expert_id_labels=expert_id_labels,
                combined_scores=combined_scores,
                weighted_experts=weighted_experts,
                is_calibration=is_calibration,
            )

        for ax in axes[len(page_items) :]:
            ax.axis("off")

        fig.subplots_adjust(
            left=0.09,
            right=0.975,
            top=0.94,
            bottom=0.055,
            hspace=0.20,
        )

        if kind == "Target":
            filename = f"{target_filename_label(page_items[0])}.png"
        else:
            filename = f"{output_prefix}_{kind}_{page_idx + 1:02d}.png"

        png_path = OUTPUT_DIR / filename
        save_figure(fig, png_path)
        plt.close(fig)


# ============================================================================
# 6. MAIN
# ============================================================================

def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find DATA_FILE: {DATA_FILE.resolve()}\n"
            "Update DATA_FILE in section 1 to point to the workbook."
        )

    for sheet_name, cfg in SHEET_CONFIG.items():
        print(f"\nReading {sheet_name} ...")
        df = read_sheet(DATA_FILE, sheet_name)

        calibration_items = sorted(
            [
                x
                for x in df["item"].dropna().unique()
                if str(x).startswith("Calib")
            ],
            key=calibration_sort_key,
        )

        target_items = sorted(
            [
                x
                for x in df["item"].dropna().unique()
                if not str(x).startswith("Calib")
            ]
        )

        print(f"  rows: {len(df):,}")
        print(f"  calibration items: {len(calibration_items)}")
        print(f"  target items: {len(target_items)}")

        ranked_rows = sorted(
            cfg["combined_scores"],
            key=cfg["combined_scores"].get,
            reverse=True,
        )
        ranked_labels = [cfg["expert_id_labels"][x] for x in ranked_rows]
        top_six_labels = [
            cfg["expert_id_labels"][x] for x in cfg["weighted_experts"]
        ]
        print(f"  combined-score order: {ranked_labels}")
        print(f"  top six / green: {top_six_labels}")

        if MAKE_CALIBRATION_PLOTS:
            save_pages(
                df=df,
                items=calibration_items,
                expert_id_labels=cfg["expert_id_labels"],
                combined_scores=cfg["combined_scores"],
                weighted_experts=cfg["weighted_experts"],
                output_prefix=cfg["calibration_prefix"],
                kind="Calibration",
                title_func=calibration_title,
                xlabel_func=calibration_xlabel,
                is_calibration=True,
                calibration_panel_label=cfg["calibration_panel_label"],
            )

        if MAKE_TARGET_PLOTS:
            save_pages(
                df=df,
                items=target_items,
                expert_id_labels=cfg["expert_id_labels"],
                combined_scores=cfg["combined_scores"],
                weighted_experts=cfg["weighted_experts"],
                output_prefix=cfg["calibration_prefix"],
                kind="Target",
                title_func=lambda item: f"{parse_target_item(item)['year']}",
                xlabel_func=lambda item: XLABEL,
                is_calibration=False,
                calibration_panel_label=None,
            )

    print(f"\nFinished. Outputs are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
