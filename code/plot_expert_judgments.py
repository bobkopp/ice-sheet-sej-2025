"""
Created on Mon Aug 17 09:42:59 2026

Reproduce and clean up the February 2026 R expert-judgment plots originally
created by Willy Aspinall.

Input workbook columns (A:F):
    expert, item, five, fifty, ninetyfive, realiz

Outputs:
    PNG only.

    Target figures use descriptive names such as:
        WAIS_Discharge_2.0C.png
        WAIS_Discharge_3.5C.png
        WAIS_Discharge_5.0C.png
        EAIS_Accumulation_2.0C.png
        GrIS_Runoff_5.0C.png

    Calibration figures use:
        GrIS_Calibration_01.png, ...
        Antarctica_Calibration_01.png, ...

Dependencies:
    pandas, numpy, matplotlib
    xlrd       (only if reading legacy .xls directly)
    openpyxl   (if using .xlsx)

author: elizabeth
"""

from __future__ import annotations

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

DATA_FILE = Path(
    "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025/data/2025_FINAL_Split_Panels.xls"
)
OUTPUT_DIR = Path(
    "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025/output/figures/"
)

# Choose which plot groups to make.
MAKE_CALIBRATION_PLOTS = True
MAKE_TARGET_PLOTS = True

# 10% overshoot for the intrinsic range
IR_SCALE = 0.10

# Three panels per PNG
PANELS_PER_PAGE = 3

# There are 18 actual experts in these split-panel sheets.
N_REAL_EXPERTS = 18

# Configuration is explicit so the weighted-expert lists cannot accidentally
# overwrite one another
SHEET_CONFIG = {
    "Gr_only": {
        "weighted_experts": [1, 3, 7, 8, 9, 15],
        "calibration_prefix": "GrIS",
    },
    "WEA_only": {
        "weighted_experts": [1, 6, 7, 8, 14, 16],
        "calibration_prefix": "Antarctica",
    },
}

# DM rows are taken as the rows after the first N_REAL_EXPERTS within each item,
# rather than assuming particular numeric expert IDs.
DM_LABELS = [
    "EW",
    "IW (0.1)",
    "IW (optimal)",
]
DM_COLORS = ["0.55", "forestgreen", "0.55"]

# Visual settings
EXPERT_COLOR = "0.62"
WEIGHTED_COLOR = "forestgreen"
REALIZATION_COLOR = "firebrick"
IR_COLOR = "0.20"
FRAME_COLOR = "lightsteelblue"
XLABEL = "cm GMSL"

# Maximum number of major x-axis intervals. MaxNLocator will choose a nice,
# readable spacing based on the actual shared x-range of each figure.
MAX_X_TICK_INTERVALS = 10

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


# ============================================================================
# 2. DATA HANDLING
# ============================================================================

def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read columns A:F, clean missing values, and convert mm to cm."""
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

    # Equivalent to R's filter(!is.na(.[[2]])).
    df = df.dropna(subset=["item"]).copy()

    # Standardize types.
    df["expert"] = pd.to_numeric(df["expert"], errors="coerce")
    for col in ["five", "fifty", "ninetyfive", "realiz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # In the R code, only rows where all three elicited percentiles equal
    # -999.6 are converted to NA.
    bad = (
        np.isclose(df["five"], -999.6, equal_nan=False)
        & np.isclose(df["fifty"], -999.6, equal_nan=False)
        & np.isclose(df["ninetyfive"], -999.6, equal_nan=False)
    )
    df.loc[bad, ["five", "fifty", "ninetyfive"]] = np.nan

    # Convert all judgment/realization values from mm to cm once, here, so all
    # downstream calculations (IR, x-limits, DMs, etc.) are also in cm.
    value_cols = ["five", "fifty", "ninetyfive", "realiz"]
    df[value_cols] = df[value_cols] / 10.0

    return df


def calibration_sort_key(item: str) -> int:
    """Natural numeric ordering: Calib1, Calib2, ..., Calib13."""
    match = re.search(r"(\d+)", str(item))
    return int(match.group(1)) if match else 10**9


def parse_target_item(item: str) -> dict[str, str] | None:
    """Parse a compact target code into ice sheet, process, warming, and year."""
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
    warming = {
        "2": "2.0°C",
        "3_5": "3.5°C",
        "5": "5.0°C",
    }[warming_code]

    ice_sheet, process = patterns[prefix]

    return {
        "ice_sheet": ice_sheet,
        "process": process,
        "warming": warming,
        "year": year,
    }


def pretty_item_name(item: str) -> str:
    """Translate compact target codes to readable plot titles."""
    parsed = parse_target_item(item)

    if parsed is None:
        return str(item).rstrip("_").replace("_", " ")

    return (
        f"{parsed['ice_sheet']} {parsed['process']}: "
        f"{parsed['warming']}, {parsed['year']}"
    )


def target_filename_label(item: str) -> str:
    """Create a concise filename such as WAIS_Discharge_5.0C."""
    parsed = parse_target_item(item)

    if parsed is None:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(item).rstrip("_"))
        return safe

    # Avoid the degree symbol in filenames while keeping the warming explicit.
    warming_file = parsed["warming"].replace("°", "")

    return f"{parsed['ice_sheet']}_{parsed['process']}_{warming_file}"


def split_item_rows(item_df: pd.DataFrame):
    """Split one item's rows into actual experts and the three DM solutions."""
    s = item_df.reset_index(drop=True).copy()
    real = s.iloc[:N_REAL_EXPERTS].copy()
    dm = s.iloc[N_REAL_EXPERTS : N_REAL_EXPERTS + len(DM_LABELS)].copy()
    return real, dm


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
    """Same 10% intrinsic-range overshoot used by the R script."""
    min_five, _ = finite_minmax(item_df["five"])
    _, max_95 = finite_minmax(item_df["ninetyfive"])
    span = max_95 - min_five
    return min_five - IR_SCALE * abs(span), max_95 + IR_SCALE * abs(span)


def compute_xlim(
    item_df: pd.DataFrame,
    realization: float | None = None,
) -> tuple[float, float]:
    """Match the asymmetric R padding, while keeping realizations visible."""
    min_five, _ = finite_minmax(item_df["five"])
    _, max_95 = finite_minmax(item_df["ninetyfive"])
    span = max(max_95 - min_five, 1e-12)

    lo = math.floor(min_five - 1.5 * IR_SCALE * abs(span))
    hi = math.ceil(max_95 + 3.3 * IR_SCALE * abs(span))

    # If a calibration realization lies outside the elicited range, keep it
    # visible rather than clipping it.
    if realization is not None and np.isfinite(realization):
        pad = 0.05 * max(hi - lo, 1)
        lo = min(lo, math.floor(realization - pad))
        hi = max(hi, math.ceil(realization + pad))

    if hi <= lo:
        hi = lo + 1

    return lo, hi


# ============================================================================
# 4. ONE PANEL
# ============================================================================

def plot_item(
    ax: plt.Axes,
    item_df: pd.DataFrame,
    title: str,
    weighted_experts: list[int],
    is_calibration: bool,
    shared_xlim: tuple[float, float] | None = None,
) -> None:
    real, dm = split_item_rows(item_df)

    # Use a single calibration realization (it is repeated down the rows).
    realization_values = item_df["realiz"].dropna().to_numpy(dtype=float)
    realization = (
        float(realization_values[0])
        if realization_values.size
        else np.nan
    )

    if shared_xlim is None:
        xlo, xhi = compute_xlim(
            item_df,
            realization if is_calibration else None,
        )
    else:
        xlo, xhi = shared_xlim

    ir_lo, ir_hi = compute_intrinsic_range(item_df)

    # Real experts at 1..18; DMs are placed after one blank spacer row.
    y_real = np.arange(1, len(real) + 1)
    y_dm = np.arange(
        N_REAL_EXPERTS + 2,
        N_REAL_EXPERTS + 2 + len(dm),
    )
    ymax = N_REAL_EXPERTS + len(DM_LABELS) + 2

    # Real expert ranges and medians.
    for y, (_, row) in zip(y_real, real.iterrows()):
        if (
            not np.isfinite(row["five"])
            or not np.isfinite(row["fifty"])
            or not np.isfinite(row["ninetyfive"])
        ):
            continue

        expert_id = (
            int(row["expert"])
            if np.isfinite(row["expert"])
            else None
        )
        color = (
            WEIGHTED_COLOR
            if expert_id in weighted_experts
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

    # Decision Maker (DM) solutions.
    for k, (y, (_, row)) in enumerate(zip(y_dm, dm.iterrows())):
        if k >= len(DM_LABELS):
            break

        if (
            not np.isfinite(row["five"])
            or not np.isfinite(row["fifty"])
            or not np.isfinite(row["ninetyfive"])
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
        # Keep the DM labels safely inside the plotting area.
        label_x = xlo + 0.015 * (xhi - xlo)
        ax.text(
            label_x,
            y,
            DM_LABELS[k],
            ha="left",
            va="center",
            fontsize=8.5,
            zorder=4,
        )

    # Intrinsic Range (IR) bounds.
    ax.vlines(
        [ir_lo, ir_hi],
        ymin=0.7,
        ymax=N_REAL_EXPERTS + 1.4,
        colors=IR_COLOR,
        linestyles="--",
        lw=1.5,
        zorder=1,
    )

    # Calibration realization only when one exists.
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

    # Axes.
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0.5, ymax)

    # Use a limited number of automatically chosen, readable x ticks.
    # Because xlo/xhi are shared within a figure, all three panels get the same
    # tick locations without overcrowding very wide ranges.
    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=MAX_X_TICK_INTERVALS,
            steps=[1, 2, 2.5, 5, 10],
            min_n_ticks=4,
        )
    )
    
    # One unlabeled minor tick between each pair of major ticks
    ax.xaxis.set_minor_locator(
        AutoMinorLocator(2)
        )

    ax.set_yticks(y_real)
    ax.set_yticklabels(
        [
            str(int(e)) if np.isfinite(e) else ""
            for e in real["expert"]
        ]
    )

    ax.set_xlabel(XLABEL, fontweight="bold")
    ax.set_ylabel("Expert ID", fontweight="bold")

    # Bold subplot title.
    if is_calibration:
        ax.set_title(
            title,
            pad=5,
            fontweight="bold",
        )
    else:
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
    
        # Major gridlines on both axes.
        ax.set_axisbelow(True)
        ax.grid(
            axis="both",
            which="major",
            linestyle="--",
            linewidth=0.6,
            alpha=0.25,
        )
        
        # Minor x-axis grid
        ax.grid(
            axis="x",
            which="minor",
            linestyle="--",
            linewidth=0.6,
            alpha=0.25,
        )

    # Subtle separator between individual experts and DM rows.
    ax.axhline(
        N_REAL_EXPERTS + 1,
        color="0.87",
        lw=0.8,
    )

    # Light frame to match Willy's original plots
    for spine in ax.spines.values():
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.9)

    ax.tick_params(
        axis="both",
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
    weighted_experts: list[int],
    output_prefix: str,
    kind: str,
    title_func,
    is_calibration: bool,
) -> None:
    if not items:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_pages = math.ceil(len(items) / PANELS_PER_PAGE)

    for page_idx in range(n_pages):
        page_items = items[
            page_idx * PANELS_PER_PAGE : (page_idx + 1) * PANELS_PER_PAGE
        ]

        fig, axes = plt.subplots(
            PANELS_PER_PAGE,
            1,
            figsize=(8.27, 11.69),
            constrained_layout=False,
        )
        axes = np.atleast_1d(axes)

        # Figure-level title for target plots.
        if not is_calibration:
            meta = parse_target_item(page_items[0])
            if meta is not None:
                fig.suptitle(
                    f'{meta["ice_sheet"]} {meta["process"]} — {meta["warming"]}',
                    fontsize=13,
                    fontweight="bold",
                    y=0.975,
                )

        # Determine common x-axis limits for all panels on this page.
        page_xlims = []
        for item in page_items:
            s_item = df.loc[df["item"] == item].copy()

            if is_calibration:
                realization_values = s_item["realiz"].dropna().to_numpy(dtype=float)
                realization = (
                    float(realization_values[0])
                    if realization_values.size
                    else None
                )
            else:
                realization = None

            page_xlims.append(compute_xlim(s_item, realization))

        shared_xlo = min(x[0] for x in page_xlims)
        shared_xhi = max(x[1] for x in page_xlims)

        # Reserve extra room on the left for the full Decision Maker labels.
        if not is_calibration:
            span = shared_xhi - shared_xlo
            shared_xlo -= 0.075 * span

        shared_xlim = (shared_xlo, shared_xhi)

        # Plot each panel using the common x-axis.
        for ax, item in zip(axes, page_items):
            s_item = df.loc[df["item"] == item].copy()
            plot_item(
                ax=ax,
                item_df=s_item,
                title=title_func(item),
                weighted_experts=weighted_experts,
                is_calibration=is_calibration,
                shared_xlim=shared_xlim,
            )

        # Hide unused panel(s) on the last page.
        for ax in axes[len(page_items):]:
            ax.axis("off")

        fig.subplots_adjust(
            left=0.09,
            right=0.975,
            top=0.94 if not is_calibration else 0.965,
            bottom=0.055,
            hspace=0.20,
        )

        if kind == "Target":
            filename = f"{target_filename_label(page_items[0])}.png"
        else:
            filename = f"{output_prefix}_{kind}_{page_idx + 1:02d}.png"

        png_path = OUTPUT_DIR / filename
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ============================================================================
# 6. MAIN
# ============================================================================

def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find DATA_FILE: {DATA_FILE.resolve()}\n"
            "Update DATA_FILE in section 1 to point to your workbook."
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
        print(f"  weighted experts: {cfg['weighted_experts']}")

        if MAKE_CALIBRATION_PLOTS:
            save_pages(
                df=df,
                items=calibration_items,
                weighted_experts=cfg["weighted_experts"],
                output_prefix=cfg["calibration_prefix"],
                kind="Calibration",
                title_func=lambda item: re.sub(r"_+$", "", str(item)),
                is_calibration=True,
            )

        if MAKE_TARGET_PLOTS:
            save_pages(
                df=df,
                items=target_items,
                weighted_experts=cfg["weighted_experts"],
                output_prefix=cfg["calibration_prefix"],
                kind="Target",
                title_func=lambda item: (
                    f"{parse_target_item(item)['year']}"
                    ),
                is_calibration=False,
            )

    print(f"\nFinished. Outputs are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
