#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIP6 and CMIP7: filled contours of REALIZED SEJ scenario assignment fractions.

Metarepo adaptation: only paths and non-interactive rendering were changed.

Recreates the style of the earlier three-panel 'Selected LOW / INTERMEDIATE /
HIGH' filled-contour figure, but uses normalized C2100 (x) and C2200 (y),
not raw C2100 and C2300. Reads the already-run [2100, 2200] checkpoint +
C2200 hard-boundary selection CSVs. Does not resample or modify assignments.

Within each normalized-space bin:
    realized LOW fraction = n(realizations assigned LOW) / n(all realizations)
(and analogously for INTERMEDIATE and HIGH).

These are empirical realized assignment fractions, NOT analytic IDW probabilities.
Bins with fewer than MIN_COUNT_PER_BIN realizations remain blank. The same
colormap (0-1) and bin edges are used in the CMIP6 and CMIP7 figures.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects

# =====================================================================
# 1. SETTINGS
# =====================================================================
# Read only the verified Stage 01/01B outputs from the selected test run.
CORE_OUTPUT_ROOT = Path(os.environ['SEJ_CORE_OUTPUT_ROOT']).expanduser().resolve()
FIG_OUTPUT_ROOT = Path(os.environ['FIG_OUTPUT_DIR']).expanduser().resolve()
METHOD_ID = 'B_2100_2200_boundary_2200'

PLOT_CMIP6 = True
PLOT_CMIP7 = True
SHOW_FIGURES_IN_SPYDER = False

CMIP6_SCENARIOS = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
CMIP7_SCENARIOS = ['L', 'ML', 'M', 'H']

# The original example is a filled contour plot, without scatter points.
# Turn on to overlay small colored SSP / CMIP7 pathway points and a legend.
SHOW_SCENARIO_POINTS = False
POINT_SIZE = 1.2
POINT_ALPHA = 0.12
SCENARIO_COLORS = {
    'ssp119': '#0072B2', 'ssp126': '#56B4E9',
    'ssp245': '#009E73', 'ssp370': '#E69F00', 'ssp585': '#D55E00',
    'L': '#0072B2', 'ML': '#009E73', 'M': '#E69F00', 'H': '#D55E00',
}
SCENARIO_LABELS = {
    'ssp119': 'SSP1-1.9', 'ssp126': 'SSP1-2.6',
    'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0',
    'ssp585': 'SSP5-8.5',
    'L': 'Low (L)', 'ML': 'Medium-to-Low (ML)',
    'M': 'Medium (M)', 'H': 'High (H)',
}

# Identical display limits and bin edges for CMIP6 and CMIP7.
X_LIM = (-0.6, 1.6)
Y_LIM = (-0.6, 1.6)
N_BINS_X = 26
# Split y bins at exactly 0 and 1: no bin crosses the hard boundary.
N_Y_BELOW_0 = 8
N_Y_0_TO_1 = 16
N_Y_ABOVE_1 = 8
MIN_COUNT_PER_BIN = 5

FILL_LEVELS = np.linspace(0.0, 1.0, 21)   # Filled steps of 0.05
LINE_LEVELS = np.arange(0.1, 1.0, 0.1)  # Isolines every 0.10
CMAP = 'viridis'
DPI = 300

OUTPUT_DIR = (
    FIG_OUTPUT_ROOT / 'supplementary' / 'scenario_selection_realized_fractions'
)

CMIP6_SELECTION_FILE = (
    CORE_OUTPUT_ROOT / '01_warming_selection'
    / 'fair_b25_baseline_scenario_selection.csv'
)
CMIP7_SELECTION_FILE = (
    CORE_OUTPUT_ROOT / '01_warming_selection'
    / 'cmip7_fair_b25_baseline_scenario_selection.csv'
)

# =====================================================================
# 2. NORMALIZATION AND REFERENCE POINTS
# =====================================================================
REF_C2100 = {'LOW': 155.0, 'INTERMEDIATE': 204.5, 'HIGH': 262.0}
REF_C2200 = {'LOW': 355.0, 'INTERMEDIATE': 554.5, 'HIGH': 762.0}
SCENARIOS = ('LOW', 'INTERMEDIATE', 'HIGH')

RANGE_2100 = REF_C2100['HIGH'] - REF_C2100['LOW']
RANGE_2200 = REF_C2200['HIGH'] - REF_C2200['LOW']

REFERENCE_POINTS = {
    scenario: (
        (REF_C2100[scenario] - REF_C2100['LOW']) / RANGE_2100,
        (REF_C2200[scenario] - REF_C2200['LOW']) / RANGE_2200,
    ) for scenario in SCENARIOS
}


def normalized_bin_edges():
    """Edges at y=0 and y=1 prevent mixing forced/interior assignments."""
    x_edges = np.linspace(*X_LIM, N_BINS_X + 1)
    y_edges = np.concatenate([
        np.linspace(Y_LIM[0], 0.0, N_Y_BELOW_0 + 1)[:-1],
        np.linspace(0.0, 1.0, N_Y_0_TO_1 + 1)[:-1],
        np.linspace(1.0, Y_LIM[1], N_Y_ABOVE_1 + 1),
    ])
    return x_edges, y_edges


# =====================================================================
# 3. LOAD EXISTING REALIZED ASSIGNMENTS
# =====================================================================
def load_selection_data(filepath, family, climate_scenarios):
    if not filepath.is_file():
        raise FileNotFoundError(
            f'{family}: missing C2200-boundary selection CSV:\n{filepath}'
        )

    df = pd.read_csv(filepath)
    required = {
        'ssp', 'selected_scenario',
        'cumulative_warming_2100', 'cumulative_warming_2200',
        'weight_low', 'weight_intermediate', 'weight_high',
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{family}: missing columns: {sorted(missing)}')

    df['ssp'] = df['ssp'].astype(str).str.strip()
    df = df[df['ssp'].isin(climate_scenarios)].copy()
    if df.empty:
        raise ValueError(f'{family}: no requested pathways found in {filepath}')
    absent = sorted(set(climate_scenarios) - set(df['ssp']))
    if absent:
        raise ValueError(f'{family}: missing pathways: {absent}')

    if 'sampling_method_id' in df.columns:
        if not df['sampling_method_id'].eq(METHOD_ID).all():
            raise ValueError(f'{family}: wrong selection method; expected {METHOD_ID}')
    if 'checkpoint_years' in df.columns:
        checkpoints = df['checkpoint_years'].astype(str).str.replace(
            ' ', '', regex=False
        )
        if not checkpoints.eq('[2100,2200]').all():
            raise ValueError(f'{family}: checkpoints are not [2100,2200]')
    if 'boundary_year' in df.columns:
        if not pd.to_numeric(df['boundary_year'], errors='raise').eq(2200).all():
            raise ValueError(f'{family}: wrong boundary year')
    if 'boundary_rule' in df.columns:
        if not df['boundary_rule'].eq('C2200_LOW_HIGH_HARD_CUTOFF').all():
            raise ValueError(f'{family}: wrong boundary rule')
    if family == 'CMIP7' and 'scenario_family' in df.columns:
        if not df['scenario_family'].eq('CMIP7').all():
            raise ValueError('CMIP7: incorrect scenario_family metadata')

    df['selected_scenario'] = (
        df['selected_scenario'].astype(str).str.strip().str.upper()
    )
    if not df['selected_scenario'].isin(SCENARIOS).all():
        raise ValueError(f'{family}: unexpected selected_scenario values')

    numeric = [
        'cumulative_warming_2100', 'cumulative_warming_2200',
        'weight_low', 'weight_intermediate', 'weight_high',
    ]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f'{family}: non-finite cumulative warming or weights')

    p = df[['weight_low', 'weight_intermediate', 'weight_high']].to_numpy()
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f'{family}: probabilities do not sum to 1')

    below = df['cumulative_warming_2200'].to_numpy() <= REF_C2200['LOW']
    above = df['cumulative_warming_2200'].to_numpy() >= REF_C2200['HIGH']
    selected = df['selected_scenario'].to_numpy()
    if not np.all(selected[below] == 'LOW'):
        raise ValueError(f'{family}: C2200<=LOW was not always assigned LOW')
    if not np.all(selected[above] == 'HIGH'):
        raise ValueError(f'{family}: C2200>=HIGH was not always assigned HIGH')
    if not np.allclose(p[below], [1, 0, 0], atol=1e-10):
        raise ValueError(f'{family}: below-LOW probabilities violate hard boundary')
    if not np.allclose(p[above], [0, 0, 1], atol=1e-10):
        raise ValueError(f'{family}: above-HIGH probabilities violate hard boundary')

    df['C2100_norm'] = (
        df['cumulative_warming_2100'] - REF_C2100['LOW']
    ) / RANGE_2100
    df['C2200_norm'] = (
        df['cumulative_warming_2200'] - REF_C2200['LOW']
    ) / RANGE_2200
    print(f'{family}: {len(df):,} saved realized assignments loaded.')
    return df


# =====================================================================
# 4. EMPIRICAL FRACTION IN EACH NORMALIZED-SPACE BIN
# =====================================================================
def calculate_realized_fractions(df, x_edges, y_edges):
    x = df['C2100_norm'].to_numpy(dtype=float)
    y = df['C2200_norm'].to_numpy(dtype=float)
    assigned = df['selected_scenario'].to_numpy()

    n_total, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    keep = n_total >= MIN_COUNT_PER_BIN
    fractions = {}
    for scenario in SCENARIOS:
        n_selected, _, _ = np.histogram2d(
            x[assigned == scenario], y[assigned == scenario],
            bins=[x_edges, y_edges],
        )
        z = np.full(n_total.shape, np.nan, dtype=float)
        z[keep] = n_selected[keep] / n_total[keep]
        fractions[scenario] = z
    return n_total, fractions


def save_grid_table(family, x_edges, y_edges, counts, fractions):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ix in range(len(x_edges) - 1):
        for iy in range(len(y_edges) - 1):
            rows.append({
                'C2100_norm_left': x_edges[ix],
                'C2100_norm_right': x_edges[ix + 1],
                'C2200_norm_bottom': y_edges[iy],
                'C2200_norm_top': y_edges[iy + 1],
                'n_realizations': int(counts[ix, iy]),
                'fraction_LOW': fractions['LOW'][ix, iy],
                'fraction_INTERMEDIATE': fractions['INTERMEDIATE'][ix, iy],
                'fraction_HIGH': fractions['HIGH'][ix, iy],
            })
    outfile = OUTPUT_DIR / f'{family}_realized_selected_fraction_grid_NORMALIZED_C2100_C2200.csv'
    pd.DataFrame(rows).to_csv(outfile, index=False)
    print(f'Saved grid: {outfile}')


# =====================================================================
# 5. THREE-PANEL FIGURE IN THE STYLE OF THE OLD CONTOUR PLOT
# =====================================================================
def add_reference_structure(ax):
    xs = [REFERENCE_POINTS[s][0] for s in SCENARIOS]
    ys = [REFERENCE_POINTS[s][1] for s in SCENARIOS]

    # Connecting line and reference-star symbols, as in the old figure.
    ax.plot(xs, ys, '-', color='#1f77b4', linewidth=1.6, zorder=7)

    # Reference-guide lines.
    for scenario in SCENARIOS:
        rx, ry = REFERENCE_POINTS[scenario]
        ax.plot([rx, rx], [Y_LIM[0], ry], '--',
                linewidth=1.0, color='0.30', alpha=0.75, zorder=5)
        ax.plot([X_LIM[0], rx], [ry, ry], '--',
                linewidth=1.0, color='0.30', alpha=0.75, zorder=5)

    # Full horizontal lines explicitly indicate the C2200 hard cutoffs.
    for bound in (0.0, 1.0):
        ax.axhline(bound, linestyle=':', color='0.20', linewidth=1.15,
                   alpha=0.9, zorder=6)

    for scenario in SCENARIOS:
        rx, ry = REFERENCE_POINTS[scenario]
        ax.plot(rx, ry, '*', markersize=14, markerfacecolor='white',
                markeredgecolor='black', markeredgewidth=1.0, zorder=9)

        if scenario == 'LOW':
            offset, ha, va = (0, -14), 'center', 'top'
        elif scenario == 'INTERMEDIATE':
            offset, ha, va = (12, 0), 'left', 'center'
        else:
            offset, ha, va = (5, 7), 'left', 'bottom'

        ax.annotate(scenario, (rx, ry), xytext=offset,
                    textcoords='offset points', ha=ha, va=va,
                    fontsize=9, fontweight='bold', color='black', zorder=10)


def make_combined_figure(df, family, climate_scenarios):
    x_edges, y_edges = normalized_bin_edges()
    n_total, fractions = calculate_realized_fractions(df, x_edges, y_edges)
    save_grid_table(family, x_edges, y_edges, n_total, fractions)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8),
                             sharex=True, sharey=True)
    filled = None

    for ax, scenario in zip(axes, SCENARIOS):
        z = np.ma.masked_invalid(fractions[scenario].T)
        if z.count() < 4:
            raise ValueError(
                f'{family}/{scenario}: too few populated bins; '
                'reduce MIN_COUNT_PER_BIN or the bin count.'
            )

        # Filled contours reproduce the old binned REALIZED fractions.
        # No analytical probability surface or smoothing is substituted.
        filled = ax.contourf(
            X, Y, z, levels=FILL_LEVELS, cmap=CMAP,
            vmin=0.0, vmax=1.0, extend='neither', zorder=1,
        )
        finite_z = fractions[scenario][np.isfinite(fractions[scenario])]
        levels = [lv for lv in LINE_LEVELS
                  if finite_z.min() < lv < finite_z.max()]
        if levels:
            lines = ax.contour(
                X, Y, z, levels=levels, colors='white',
                linewidths=0.85, alpha=0.95, zorder=3,
            )
            labels = ax.clabel(lines, inline=True, fontsize=9,
                               colors='white', fmt='%.1f')
            for label in labels:
                label.set_fontweight('bold')
                label.set_path_effects([
                    patheffects.Stroke(linewidth=1.4, foreground='black'),
                    patheffects.Normal(),
                ])

        # Optional transparent climate-pathway dots, disabled by default
        # so the figure matches the original filled-contour example.
        if SHOW_SCENARIO_POINTS:
            for climate_scenario in climate_scenarios:
                sub = df[df['ssp'] == climate_scenario]
                ax.scatter(sub['C2100_norm'], sub['C2200_norm'],
                           s=POINT_SIZE, alpha=POINT_ALPHA,
                           color=SCENARIO_COLORS[climate_scenario],
                           edgecolors='none', rasterized=True, zorder=4)

        ax.set_xlim(X_LIM)
        ax.set_ylim(Y_LIM)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Selected {scenario}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Normalized cumulative warming\nthrough 2100',
                      fontsize=10, fontweight='bold')
        if ax is axes[0]:
            ax.set_ylabel('Normalized cumulative warming\nthrough 2200',
                          fontsize=10, fontweight='bold')
        ax.set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        ax.tick_params(labelsize=10, labelbottom=True,
                       labelleft=(ax is axes[0]))
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.28, zorder=0)
        add_reference_structure(ax)

    # All panels have the same probability scale (0 to 1).
    cbar = fig.colorbar(filled, ax=axes, fraction=0.025, pad=0.018,
                        ticks=np.arange(0.0, 1.01, 0.1))
    cbar.set_label('Fraction selected', fontsize=11)

    fig.suptitle(
        f'{family}: realized warming-scenario assignment fractions\n'
        '[2100, 2200] normalized checkpoints + C2200 hard boundary',
        fontsize=14, y=0.995,
    )

    if SHOW_SCENARIO_POINTS:
        handles = [Line2D([0], [0], marker='o', linestyle='None',
                          color=SCENARIO_COLORS[s], markersize=6,
                          label=SCENARIO_LABELS[s])
                   for s in climate_scenarios]
        fig.legend(handles=handles, loc='lower center',
                   bbox_to_anchor=(0.50, 0.005),
                   ncol=len(handles), frameon=False, fontsize=9)

    fig.subplots_adjust(left=0.07, right=0.90,
                        bottom=0.20 if SHOW_SCENARIO_POINTS else 0.13,
                        top=0.82, wspace=0.12)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = (
        OUTPUT_DIR
        / f'{family}_REALIZED_selected_fraction_contours_NORMALIZED_C2100_C2200_boundary2200.png'
    )
    fig.savefig(outfile, dpi=DPI, bbox_inches='tight')
    if not SHOW_FIGURES_IN_SPYDER:
        plt.close(fig)
    print(f'Saved figure: {outfile}')


# =====================================================================
# 6. MAIN
# =====================================================================
def main():
    if not PLOT_CMIP6 and not PLOT_CMIP7:
        raise ValueError('Enable PLOT_CMIP6 and/or PLOT_CMIP7.')
    if PLOT_CMIP6:
        df6 = load_selection_data(CMIP6_SELECTION_FILE, 'CMIP6', CMIP6_SCENARIOS)
        make_combined_figure(df6, 'CMIP6', CMIP6_SCENARIOS)
    if PLOT_CMIP7:
        df7 = load_selection_data(CMIP7_SELECTION_FILE, 'CMIP7', CMIP7_SCENARIOS)
        make_combined_figure(df7, 'CMIP7', CMIP7_SCENARIOS)
    print('\nDone. Outputs saved to:', OUTPUT_DIR)
    if SHOW_FIGURES_IN_SPYDER:
        plt.show()


if __name__ == '__main__':
    main()
