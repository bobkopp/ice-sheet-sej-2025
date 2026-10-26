#!/usr/bin/env python3
"""Four-SSP AR6-inspired figure with 2100–2150 and 2000–2300 columns.

Based on PLOT_CMIP6_FOUR_SSP_AR6_STYLE_2300.py. Reads the *actual*
CMIP6 NetCDF E/W/G realization arrays and the companion sampled CSV's
'ssp' column. Never synthesizes scenario trajectories or sums percentiles.

The zero lines align across the full figure (both columns and all rows),
while each component retains its own y-axis range. Endpoint uncertainty
bars are included in automatic range calculations.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from netCDF4 import Dataset

# ============================ USER SETTINGS ============================
# Use the same isolated core and figure roots as the validated metarepo.
import os
PROJECT_DIR = Path(os.environ.get('SEJ_PROJECT_DIR', Path(__file__).resolve().parents[3])).expanduser().resolve()
CORE_ROOT = Path(os.environ.get('SEJ_CORE_OUTPUT_ROOT', PROJECT_DIR / 'output' / 'CORE_METHOD_CHAIN_02I_B_FINAL_TENSION_P003_FAIR2')).expanduser().resolve()
ANNUAL_DIR = CORE_ROOT / '03_cmip6_annual'
ANNUAL_NETCDF = ANNUAL_DIR / 'annual_ice_sheet_trajectories_2000_2300.nc'
SAMPLED_CSV = ANNUAL_DIR / 'sampled_sej2025_realizations.csv'
IMBIE_CSV = ANNUAL_DIR / 'imbie_observed_slr_1979_2023_monthly_rebased_to_1995_2014.csv'
FIGURE_ROOT = Path(os.environ.get('FIG_FAIR2_OUTPUT_DIR', PROJECT_DIR / 'output' / 'manuscript_figures' / 'stage02I_B_final_tension_p003')).expanduser().resolve()
OUTPUT_PNG = (FIGURE_ROOT / 'AR6_style_four_SSPs'
              / 'CMIP6_four_SSPs_2100_2150_and_2000_2300.png')
SAVE_PDF = True
SHOW_IMBIE = True
SHOW_OUTER_BANDS = False     # 5–95 shading over time; bars always show 5–95
FIGSIZE = (8, 10)
DPI = 300
INNER_ALPHA = 0.16
OUTER_ALPHA = 0.05
MEDIAN_LW = 2.1
HISTORICAL_LW = 1.9
# Optional shared y-axis overrides by component, e.g. 'G': (-20, 300).
MANUAL_YLIMS = {}
# Fraction of each subplot height below y=0. The same value is applied to
# all rows and both time windows, so every horizontal zero line aligns.
ZERO_BOTTOM_FRACTION = 0.15  # Less empty space below zero; equal across all panels
# Preferred upper bounds for the first three rows. These keep the left column
# compact while preserving the right column interval space, yet zero still aligns
# across both columns because the same ZERO_BOTTOM_FRACTION is applied everywhere.
LEFT_COLUMN_UPPER_FIRST3 = 175
RIGHT_COLUMN_UPPER_FIRST3 = 400
# Thin divider between WAIS (row 3) and total ice sheet (row 4).
SHOW_TOTAL_SEPARATOR = True
SEPARATOR_COLOR = '0.65'
SEPARATOR_LW = 0.6
SEPARATOR_DOWN = 0.008  # Figure-height fraction; positive moves line toward Total row
SEPARATOR_LEFT = 0.012  # Figure fraction; extends through left y-axis labels
# ======================================================================

BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
RED_ORANGE = "#D55E00"
PURPLE = "#6A3D9A"
PINK = "#CC79A7"
BLACK = "#000000"

CLIMATE_SCENARIO_COLORS = {
    "ssp119": BLUE,
    "ssp126": LIGHT_BLUE,
    "ssp245": GREEN,
    "ssp370": ORANGE,
    "ssp534os": PINK,
    "ssp585": RED_ORANGE,
    # optional alternate spellings if your dataframe uses them
    "ssp5-3.4-os": PINK,
    "SSP1-1.9": BLUE,
    "SSP1-2.6": LIGHT_BLUE,
    "SSP2-4.5": GREEN,
    "SSP3-7.0": ORANGE,
    "SSP5-3.4-OS": PINK,
    "SSP5-8.5": RED_ORANGE,
}
# The supplied climate palette has no SSP5-3.4-OS key; retain the
# established magenta/pink color for that fourth scenario.
SCENARIO_COLORS = {**CLIMATE_SCENARIO_COLORS, 'ssp534-over': PINK}
SCENARIOS = ('ssp126', 'ssp534-over', 'ssp245', 'ssp370')
SCENARIO_LABELS = {
    'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5',
    'ssp534-over': 'SSP5-3.4-OS', 'ssp370': 'SSP3-7.0',
}
ROWS = (('G', 'GrIS'), ('E', 'EAIS'), ('W', 'WAIS'),
        ('TOTAL', 'Total ice sheet'))
WINDOWS = ((2000, 2150), (2150, 2300))
PERCENTILES = (5, 17, 50, 83, 95)


def canonical_ssp(value):
    s = str(value).strip().lower().replace('_', '-')
    return 'ssp534-over' if s in ('ssp534', 'ssp534over', 'ssp534-over') else s


def load_annual():
    for path in (ANNUAL_NETCDF, SAMPLED_CSV):
        if not path.is_file():
            raise FileNotFoundError(f'Missing final CMIP6 input: {path}')
    meta = pd.read_csv(SAMPLED_CSV)
    if 'ssp' not in meta.columns:
        raise ValueError('Sampled CSV must contain an ssp column.')
    labels = meta['ssp'].map(canonical_ssp).to_numpy()
    with Dataset(ANNUAL_NETCDF, 'r') as nc:
        for name in ('years', 'E', 'W', 'G'):
            if name not in nc.variables:
                raise ValueError(f'NetCDF missing variable {name}: {ANNUAL_NETCDF}')
        years = np.asarray(nc.variables['years'][:], dtype=int)
        arrays = {c: np.ma.filled(nc.variables[c][:], np.nan).astype(np.float64)
                  for c in ('E', 'W', 'G')}
    if (years.ndim != 1 or years[0] != 2000 or years[-1] != 2300
            or not np.all(np.diff(years) == 1)):
        raise ValueError('Expected consecutive annual years 2000–2300.')
    for c, a in arrays.items():
        if a.shape != (len(labels), len(years)) or not np.isfinite(a).all():
            raise ValueError(f'{c}: shape {a.shape} or values do not match sampled CSV.')
    # Sum realization-by-realization BEFORE taking quantiles.
    arrays['TOTAL'] = arrays['E'] + arrays['W'] + arrays['G']
    for s in SCENARIOS:
        n = int(np.sum(labels == s))
        if n == 0:
            raise ValueError(f'No samples for {s}. Available: {sorted(set(labels))}')
        print(f'{SCENARIO_LABELS[s]}: {n:,} samples')
    return years, labels, arrays


def load_imbie():
    if not SHOW_IMBIE:
        return None
    if not IMBIE_CSV.is_file():
        raise FileNotFoundError(f'Historical IMBIE requested but missing: {IMBIE_CSV}')
    df = pd.read_csv(IMBIE_CSV)
    required = {'Date', 'G_cm', 'E_cm', 'W_cm'}
    if not required.issubset(df.columns):
        raise ValueError(f'IMBIE missing: {sorted(required - set(df.columns))}')
    dates = pd.DatetimeIndex(pd.to_datetime(df['Date'], errors='raise'))
    starts = pd.DatetimeIndex(pd.to_datetime(dates.year.astype(str) + '-01-01'))
    ends = pd.DatetimeIndex(pd.to_datetime((dates.year + 1).astype(str) + '-01-01'))
    x = dates.year.to_numpy(float) + ((dates - starts).days / (ends - starts).days).to_numpy(float)
    obs = {c: df[f'{c}_cm'].to_numpy(float) for c in ('G', 'E', 'W')}
    obs['TOTAL'] = obs['E'] + obs['W'] + obs['G']
    return x, obs


def compute_quantiles(labels, arrays):
    result = {}
    for comp, _ in ROWS:
        result[comp] = {}
        for s in SCENARIOS:
            q = np.percentile(arrays[comp][labels == s, :], PERCENTILES, axis=0)
            result[comp][s] = dict(zip(PERCENTILES, q))
    return result


def axis_limits(comp, years, quantiles, history, start, end):
    override = MANUAL_YLIMS.get(comp)
    if override is not None:
        return override
    # Within-window 5–95 envelope, including the endpoint bars.
    keep = (years >= start) & (years <= end)
    lo = min(float(quantiles[comp][s][5][keep].min()) for s in SCENARIOS)
    hi = max(float(quantiles[comp][s][95][keep].max()) for s in SCENARIOS)
    if history is not None:
        hx, obs = history
        hkeep = (hx >= start) & (hx <= min(end, 2023.99))
        if hkeep.any():
            lo = min(lo, float(np.nanmin(obs[comp][hkeep])))
            hi = max(hi, float(np.nanmax(obs[comp][hkeep])))
    lo, hi = min(lo, 0.), max(hi, 0.)
    pad = max((hi-lo)*0.07, 0.5)
    return lo-pad, hi+pad


def align_zero(ylim, fraction=ZERO_BOTTOM_FRACTION):
    """Expand a y range, without clipping it, to put zero at one height.

    This does not impose a common numerical y scale across ice sheets.
    """
    if not 0 < fraction < 1:
        raise ValueError('ZERO_BOTTOM_FRACTION must be strictly between 0 and 1.')
    lo, hi = map(float, ylim)
    if not lo < hi:
        raise ValueError(f'Invalid y-axis limits: {ylim}')
    # Keep both bounds at least as large in magnitude as originally specified.
    extent = max(-lo / fraction, hi / (1 - fraction))
    return (-fraction * extent, (1 - fraction) * extent)


def plot():
    years, labels, arrays = load_annual()
    history = load_imbie()
    quantiles = compute_quantiles(labels, arrays)
    fig = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(
        nrows=4,
        ncols=5,
        width_ratios=(6.3, 1.1, 0.65, 6.3, 1.1),
        left=0.075,
        right=0.985,
        top=0.845,
        bottom=0.07,
        hspace=0.29,
        wspace=0.18,
    )

    for row, (comp, name) in enumerate(ROWS):
        for group, (start, end) in enumerate(WINDOWS):
    
            # Keep the first three left-column rows compact (top ≈150 cm) and
            # preserve the existing right-column headroom for endpoint intervals,
            # while using one shared zero fraction so y=0 aligns across both columns.
            if comp == 'TOTAL':
                ylim = align_zero(axis_limits(comp, years, quantiles, history, start, end))
            else:
                data_lo, data_hi = axis_limits(comp, years, quantiles, history, start, end)
                preferred_top = LEFT_COLUMN_UPPER_FIRST3 if group == 0 else RIGHT_COLUMN_UPPER_FIRST3
                # Hold the preferred top when the displayed data fit within it; if not,
                # allow the top to expand rather than clipping uncertainty intervals.
                if data_hi <= preferred_top:
                    implied_bottom = -ZERO_BOTTOM_FRACTION * preferred_top / (1 - ZERO_BOTTOM_FRACTION)
                    ylim = (implied_bottom, preferred_top)
                else:
                    ylim = align_zero((data_lo, data_hi))

            main_col, box_col = ((0, 1), (3, 4))[group]

            ax = fig.add_subplot(grid[row, main_col])
            box = fig.add_subplot(grid[row, box_col], sharey=ax)
            
            keep = (years >= start) & (years <= end)
            for s in SCENARIOS:
                q = quantiles[comp][s]
                color = SCENARIO_COLORS[s]
                if SHOW_OUTER_BANDS:
                    ax.fill_between(years[keep], q[5][keep], q[95][keep],
                                    color=color, alpha=OUTER_ALPHA, lw=0, zorder=1)
                ax.fill_between(years[keep], q[17][keep], q[83][keep],
                                color=color, alpha=INNER_ALPHA, lw=0, zorder=2)
                ax.plot(years[keep], q[50][keep], color=color, lw=MEDIAN_LW, zorder=4)

            if history is not None:
                hx, obs = history
                hkeep = (hx >= start) & (hx <= min(end, 2023.99))
                if hkeep.any():
                    ax.plot(hx[hkeep], obs[comp][hkeep], color=BLACK,
                            lw=HISTORICAL_LW, zorder=6)
            ax.axhline(0, color='0.55', lw=0.8, zorder=0)
            ax.set_xlim(start, end)
            ax.set_ylim(*ylim)
            ax.set_xticks(
                [2000, 2050, 2100, 2150] if end == 2150
                else [2150, 2200, 2250, 2300]
            )
            ax.set_axisbelow(True)
            ax.grid(True, axis='both', linestyle='--', color='0.86', lw=0.75)
            ax.tick_params(direction='out', labelsize=9)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if group == 0:
                ax.set_ylabel(
                    f'{name}\nSLE (cm)',
                    fontsize=10,
                    fontweight='bold'
                )
            else:
                ax.tick_params(axis='y', left=True, labelleft=True)
            if row == 0:
                ax.set_title(f'{start}–{end}', fontsize=11, pad=8, fontweight = 'bold')
            if row == len(ROWS)-1:
                ax.set_xlabel('Year', fontsize=10, fontweight = 'bold')

            end_index = int(np.where(years == end)[0][0])
            for i, s in enumerate(SCENARIOS):
                q = quantiles[comp][s]
                color = SCENARIO_COLORS[s]
                box.vlines(i, q[5][end_index], q[95][end_index],
                           color=color, lw=2.3, alpha=0.50, zorder=2)
                box.vlines(i, q[17][end_index], q[83][end_index],
                           color=color, lw=6, alpha=0.95, zorder=3)
                box.scatter(i, q[50][end_index], color=color, s=45, zorder=4)
            box.axhline(0, color='0.55', lw=0.8)
            box.set_xlim(-0.6, len(SCENARIOS)-0.4)
            box.set_ylim(*ylim)
            box.set_xticks(range(len(SCENARIOS)))
            if row == len(ROWS)-1:
                box.set_xticklabels(['2.6', '3.4-OS', '4.5', '7.0'],
                                    rotation=90, fontsize=8)
            else:
                box.set_xticklabels([])
            box.set_axisbelow(True)
            box.grid(
                True,
                axis='y',
                linestyle='--',
                color='0.86',
                lw=0.75
            )
            box.tick_params(left=False, labelleft=False, bottom=False, length=0)
            for spine in ('top', 'right', 'bottom', 'left'):
                box.spines[spine].set_visible(False)
            if row == 0:
                box.set_title(f'{end}', fontsize=10, pad=8)

    if SHOW_TOTAL_SEPARATOR:
        # Draw across the full figure width, including the y-axis labels,
        # both time windows, and their endpoint interval panels.
        above = grid[2, 0].get_position(fig).y0
        below = grid[3, 0].get_position(fig).y1
        separator_y = (above + below) / 2 - SEPARATOR_DOWN
        fig.add_artist(Line2D([SEPARATOR_LEFT,
                               grid[0, 4].get_position(fig).x1],
                              [separator_y, separator_y],
                              transform=fig.transFigure,
                              color=SEPARATOR_COLOR, lw=SEPARATOR_LW,
                              zorder=20, clip_on=False))

    handles = [Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2.8,
                      label=SCENARIO_LABELS[s]) for s in SCENARIOS]
    handles += [Patch(color='0.5', alpha=INNER_ALPHA, label='17th–83rd percentile'),
                Line2D([0], [0], color='0.5', lw=2.4, alpha=0.5,
                       label='Endpoint: 5th–95th percentile')]
    if history is not None:
        handles += [Line2D([0], [0], color=BLACK, lw=HISTORICAL_LW,
                           label='IMBIE observations')]
    fig.suptitle('Projected ice-sheet contributions under four CMIP6 SSPs',
                 fontsize=15, fontweight='bold', y=0.99)
    fig.legend(
        handles=handles,
        ncol=4,
        loc='upper center',
        bbox_to_anchor=(0.53, 0.955),
        fontsize=9,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.4,
        labelspacing=0.5
    )
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches='tight')
    print('Saved PNG:', OUTPUT_PNG)
    if SAVE_PDF:
        pdf = OUTPUT_PNG.with_suffix('.pdf')
        fig.savefig(pdf, bbox_inches='tight')
        print('Saved PDF:', pdf)
    plt.close(fig)

if __name__ == '__main__':
    plot()
