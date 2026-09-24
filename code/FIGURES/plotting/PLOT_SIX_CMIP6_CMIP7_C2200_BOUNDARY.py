#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Six scenario-panel figures for C2200-BOUNDARY CMIP6 and CMIP7 ice-sheet SLR.

Run in Spyder: open THIS file and press F5. No command-line arguments.

Outputs (six figures total, filenames end in _C2200_boundary.png):
    GrIS_CMIP6_SSP_panels_2000_2300.png
    GrIS_CMIP7_L_ML_M_H_panels_2000_2300.png
    Total_AIS_CMIP6_SSP_panels_2000_2300.png
    Total_AIS_CMIP7_L_ML_M_H_panels_2000_2300.png
    Total_ice_sheet_SLR_CMIP6_SSP_panels_2000_2300.png
    Total_ice_sheet_SLR_CMIP7_L_ML_M_H_panels_2000_2300.png

Each subplot contains ONE climate scenario's median and 17-83% / 5-95%
uncertainty bands, plus the same 2000-2023 IMBIE observed contribution.
CMIP6: five SSP panels in a 2x3 grid (sixth panel empty).
CMIP7: exactly L, ML, M and H in a 2x2 grid. Other CMIP7 pathways
may exist in the input file but are NOT computed, plotted, or written to
figure-data CSVs. No physical ice-sheet sampling is rerun. The plotter does not create C2200-boundary results.

All SLR shown here is ICE-SHEET contribution (cm sea-level equivalent),
NOT total global sea-level rise from glaciers, thermal expansion, etc.
"""

from __future__ import annotations

from figure_labels import ANNUAL_YEARS, CMIP6_SCENARIO_PAIRS, CMIP7_SCENARIO_PAIRS, PERCENTILE_LEVELS

from figure_settings import PROJECT_ROOT, DATA_ROOT, UPSTREAM_ROOT, FIGURE_ROOT, output_for, DPI, SHOW_FIGURES, B25_MC_FILE, B19_ADJUSTED_FILE, CMIP6_ANNUAL_DIR, CMIP7_ANNUAL_DIR, CMIP6_SELECTION_FILE, CMIP7_SELECTION_FILE
from figure_helpers import save_figure, maybe_show
from figure_style import ICE_SHEET_COLORS, CMIP_PANEL_TOTAL_COLOR

from pathlib import Path
import csv

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# ---------------------------------------------------------------------
# 1. EDIT THESE SETTINGS IN SPYDER
# ---------------------------------------------------------------------
PROJECT_DIR = PROJECT_ROOT

PLOT_CMIP6 = True
PLOT_CMIP7 = True
SHOW_FIGURES_IN_SPYDER = SHOW_FIGURES
SHOW_WIDE_RANGE = True          # True -> 5-95% AND 17-83%; False -> only 17-83%
SHARE_Y_WITHIN_FIGURE = True   # False -> individual y axes make low scenarios legible
SHOW_2050_TRANSITION = False    # Optional subtle dashed guide

# Use a NEW output directory; older plots containing seven CMIP7 pathways
# elsewhere in the project will not be mistaken for these four-pathway plots.
OUTPUT_DIR = output_for("06_cmip6_cmip7", "scenario_panels")

# This script reads ONLY [2100, 2200] checkpoints with the C2200 hard boundary.
METHOD_ID = 'B_2100_2200_boundary_2200'

# The earlier CMIP7 pipeline generated NO-BOUNDARY data only.
# After you generate a full CMIP7 C2200-boundary annual run, set this to its
# actual annual_trajectory_method folder. NEVER point it at the no-boundary run.
CMIP7_BOUNDARY_ANNUAL_DIR = CMIP7_ANNUAL_DIR
YEARS_EXPECTED = np.arange(ANNUAL_YEARS[0], ANNUAL_YEARS[1] + 1)
QUANTILES = PERCENTILE_LEVELS

CMIP6_SCENARIOS = CMIP6_SCENARIO_PAIRS

# *** THE ONLY CMIP7 SCENARIOS EVER PLOTTED ***
CMIP7_SCENARIOS = CMIP7_SCENARIO_PAIRS

# The NetCDF contains E, W, G. AIS and Total are summed per joint draw,
# THEN percentiles are calculated; never add separate percentile curves.
COMPONENTS = (
    ('G', 'GrIS', 'GrIS', ICE_SHEET_COLORS['GrIS']),
    ('AIS', 'Total AIS', 'Total_AIS', ICE_SHEET_COLORS['Total AIS']),
    ('TOTAL', 'Total ice-sheet SLR (AIS + GrIS)', 'Total_ice_sheet_SLR', CMIP_PANEL_TOTAL_COLOR),
)


OBS_COMPONENT_COLUMNS = {
    'G': 'GrIS_SLR_mm_rebased_2005_cm',
    'AIS': 'observed_total_ais_cm',
    'TOTAL': 'observed_total_ice_sheets_cm',
}


def method_root(family: str) -> Path:
    if family == 'CMIP6':
        return CMIP6_ANNUAL_DIR
    if family == 'CMIP7':
        # This folder must contain a GENUINE CMIP7 C2200-boundary run.
        return CMIP7_BOUNDARY_ANNUAL_DIR
    raise ValueError(f'Unexpected family: {family}')


def locate_observations(families: list[str]) -> Path:
    # This is the already REBASED (2005 reference) observation record from
    # the annual trajectory method, NOT the unrebased raw monthly IMBIE CSV.
    name = 'imbie_observed_slr_2000_2023_monthly_rebased_to_2005.csv'
    candidates = [method_root(family) / name for family in families]
    # Only look inside the selected C2200-boundary annual directories.
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        'Cannot find the rebased IMBIE file. Looked in:\n' +
        '\n'.join(str(p) for p in candidates) +
        '\nThe script will not silently plot a different-baseline observational series.'
    )


def load_observations(path: Path) -> pd.DataFrame:
    obs = pd.read_csv(path)
    columns = (
        'Date',
        'EAIS_SLR_mm_rebased_2005_cm',
        'WAIS_SLR_mm_rebased_2005_cm',
        'GrIS_SLR_mm_rebased_2005_cm',
    )
    missing = sorted(set(columns) - set(obs.columns))
    if missing:
        raise ValueError(f'IMBIE observation file missing {missing}: {path}')
    obs['Date'] = pd.to_datetime(obs['Date'], errors='raise')
    if obs['Date'].isna().any():
        raise ValueError('Missing IMBIE observation dates')
    obs = obs.sort_values('Date')
    e, w, g = (obs[c].to_numpy(dtype=float) for c in columns[1:])
    if not all(np.isfinite(a).all() for a in (e, w, g)):
        raise ValueError('Non-finite observations in rebased IMBIE input')
    obs['observed_total_ais_cm'] = e + w
    obs['observed_total_ice_sheets_cm'] = e + w + g
    obs['decimal_year'] = (
        obs['Date'].dt.year
        + (obs['Date'].dt.dayofyear - 1)
        / np.where(obs['Date'].dt.is_leap_year, 366.0, 365.0)
    )
    return obs


def load_input_metadata(family: str, root: Path) -> pd.DataFrame:
    filename = root / 'sampled_sej2025_realizations.csv'
    if not filename.is_file():
        raise FileNotFoundError(
            f'Missing {family} C2200-boundary metadata: {filename}\n'
            'The prior CMIP7 package created only no-boundary results. '
            'Generate the C2200-boundary CMIP7 selection AND annual NetCDF, '
            'then set CMIP7_BOUNDARY_ANNUAL_DIR to that run.'
        )
    meta = pd.read_csv(filename)
    if 'ssp' not in meta:
        raise ValueError(f'Missing scenario-code column ssp: {filename}')
    meta['ssp'] = meta['ssp'].astype(str).str.strip()
    expected = {code for code, _ in (CMIP6_SCENARIOS if family == 'CMIP6' else CMIP7_SCENARIOS)}
    missing = expected - set(meta['ssp'])
    if missing:
        raise ValueError(f'{family} is missing required scenario(s): {sorted(missing)}')
    if family == 'CMIP6' and set(meta['ssp']) != expected:
        raise ValueError('Unexpected SSP values; verify that the CMIP6 C2200-boundary file is being used.')
    if 'sampling_method_id' in meta:
        bad = ~meta['sampling_method_id'].astype(str).eq(METHOD_ID)
        if bad.any():
            raise ValueError(
                f'Refusing a dataset with non-C2200-boundary sampling_method_id in {filename}'
            )
    if 'sampling_method_id' not in meta:
        raise ValueError(f'Missing sampling_method_id; cannot verify C2200 boundary: {filename}')
    if 'boundary_applied' not in meta or 'boundary_year' not in meta:
        raise ValueError(f'Missing boundary metadata; refusing to plot an unverified method: {filename}')
    applied = meta['boundary_applied'].astype(str).str.strip().str.lower()
    if not applied.eq('true').all():
        raise ValueError(f'Expected boundary_applied=True for every row: {filename}')
    byear = pd.to_numeric(meta['boundary_year'], errors='coerce')
    if not byear.eq(2200).all():
        raise ValueError(f'Expected boundary_year=2200 for every row: {filename}')
    if 'checkpoint_years' in meta:
        normalized = meta['checkpoint_years'].astype(str).str.replace(' ', '', regex=False)
        if not normalized.eq('[2100,2200]').all():
            raise ValueError(f'Expected checkpoints [2100,2200]: {filename}')
    if family == 'CMIP7':
        if 'scenario_family' not in meta or not meta['scenario_family'].astype(str).eq('CMIP7').all():
            raise ValueError(f'CMIP7 metadata must contain scenario_family=CMIP7: {filename}')
    return meta


def quantiles_for_family(family: str) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
    root = method_root(family)
    nc_file = root / 'annual_ice_sheet_trajectories_2000_2300.nc'
    if not nc_file.is_file():
        raise FileNotFoundError(
            f'Missing {family} C2200-boundary annual trajectories: {nc_file}\n'
            'This script plots existing C2200-boundary outputs only. '
            'It cannot convert the no-boundary NetCDF by changing a filename.'
        )
    meta = load_input_metadata(family, root)
    scenario_list = CMIP6_SCENARIOS if family == 'CMIP6' else CMIP7_SCENARIOS
    codes = meta['ssp'].to_numpy(dtype=str)
    quantiles = {}

    with Dataset(nc_file, 'r') as nc:
        for variable in ('years', 'E', 'W', 'G'):
            if variable not in nc.variables:
                raise ValueError(f'Variable {variable} missing in {nc_file}')
        years = np.asarray(nc.variables['years'][:], dtype=int)
        if not np.array_equal(years, YEARS_EXPECTED):
            raise ValueError(f'Expected 301 annual years 2000–2300 in {nc_file}')
        expected_shape = (len(meta), len(years))
        for variable in ('E', 'W', 'G'):
            if nc.variables[variable].shape != expected_shape:
                raise ValueError(
                    f'{family}: {variable} shape {nc.variables[variable].shape} '
                    f'does not match metadata/year shape {expected_shape}'
                )

        # Only the requested scenarios are processed. Other CMIP7 pathway
        # records remain in the NetCDF but never appear in output plots/CSVs.
        for code, _ in scenario_list:
            indexes = np.flatnonzero(codes == code)
            if len(indexes) == 0:
                raise ValueError(f'{family} has no samples for {code}')
            print(f'  {family}: computing quantiles for {code} ({len(indexes):,} samples)')

            # netCDF4 supports ascending integer index arrays on the first axis.
            g = np.asarray(nc.variables['G'][indexes, :], dtype=float)
            e = np.asarray(nc.variables['E'][indexes, :], dtype=float)
            w = np.asarray(nc.variables['W'][indexes, :], dtype=float)
            ais = e + w
            total = ais + g
            del e, w

            for comp, data in (('G', g), ('AIS', ais), ('TOTAL', total)):
                if not np.isfinite(data).all():
                    raise ValueError(f'{family}/{code}/{comp} has non-finite annual values')
                qq = np.percentile(data, QUANTILES, axis=0)
                if qq.shape != (5, len(years)):
                    raise ValueError(f'Unexpected percentile shape for {family}/{code}/{comp}')
                quantiles[(code, comp)] = qq
            del g, ais, total

    # Explicit CMIP7 guard: no additional plotted scenario codes, even if
    # the source NetCDF has more. Helps diagnose confusion with older figures.
    actual_codes = {code for code, comp in quantiles}
    requested_codes = {code for code, _ in scenario_list}
    if actual_codes != requested_codes:
        raise AssertionError(f'Unexpected plotted scenario set: {actual_codes}')
    print(f'  {family} scenarios in NEW figures: {", ".join(code for code, _ in scenario_list)}')
    return years, quantiles


def save_annual_figure_data(
    family: str, years: np.ndarray, quantiles: dict[tuple[str, str], np.ndarray]
) -> Path:
    output = OUTPUT_DIR / f'{family}_plotted_scenario_annual_quantiles_C2200_boundary_cm.csv'
    scenario_list = CMIP6_SCENARIOS if family == 'CMIP6' else CMIP7_SCENARIOS
    with output.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(('family', 'scenario_code', 'component', 'year',
                         'p05_cm', 'p17_cm', 'p50_cm', 'p83_cm', 'p95_cm'))
        for code, _ in scenario_list:
            for comp, _, _, _ in COMPONENTS:
                q = quantiles[(code, comp)]
                for j, year in enumerate(years):
                    writer.writerow((family, code, comp, int(year),
                                     *[float(q[k, j]) for k in range(5)]))
    return output


def make_component_figure(
    family: str,
    component: tuple[str, str, str, str],
    scenarios: tuple[tuple[str, str], ...],
    years: np.ndarray,
    quantiles: dict[tuple[str, str], np.ndarray],
    obs: pd.DataFrame,
) -> Path:
    comp, component_label, filename_part, color = component
    if family == 'CMIP6':
        fig, axs = plt.subplots(2, 3, figsize=(15.4, 8.0), sharex=True,
                                sharey=SHARE_Y_WITHIN_FIGURE)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(12.3, 8.0), sharex=True,
                                sharey=SHARE_Y_WITHIN_FIGURE)
    axes = np.asarray(axs).reshape(-1)

    x_obs = obs['decimal_year'].to_numpy(dtype=float)
    y_obs = obs[OBS_COMPONENT_COLUMNS[comp]].to_numpy(dtype=float)
    for ax, (code, label) in zip(axes, scenarios):
        q = quantiles[(code, comp)]
        if SHOW_WIDE_RANGE:
            ax.fill_between(years, q[0], q[4], color=color, alpha=0.17, lw=0,
                            label='5–95% interval')
        ax.fill_between(years, q[1], q[3], color=color, alpha=0.37, lw=0,
                        label='17–83% interval')
        ax.plot(years, q[2], color=color, lw=2.0, label='Median', zorder=4)
        ax.plot(x_obs, y_obs, color='black', lw=2.3,            # historical observations
                label='IMBIE observations', zorder=8)
        ax.axhline(0, color='0.72', lw=0.65, zorder=0)
        if SHOW_2050_TRANSITION:
            ax.axvline(2050, color='0.55', lw=0.8, ls=':', zorder=0)
        ax.set_xlim(2000, 2300)
        ax.set_xticks((2000, 2100, 2200, 2300))
        ax.tick_params(axis='both', labelsize=11, labelbottom = True, labelleft = True)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(which = 'both', alpha=0.40, lw=0.6, linestyle = '--')
        ax.margins(y=0.05)

    for ax in axes[len(scenarios):]:
        ax.set_visible(False)

    # Use common x and y labels in the figure rather than repeating labels
    fig.supxlabel('Year', fontsize=12, y=0.075, fontweight = 'bold')
    fig.supylabel('Sea-level contribution (cm)', fontsize=12, x=0.014, fontweight = 'bold')
    fig.suptitle(f'{component_label} — {family} climate scenarios',
                 fontsize=15, y=0.995)

    legend_handles = [
        Line2D([0], [0], color=color, lw=2.2, label='Median'),
        Patch(facecolor=color, alpha=0.37, label='17–83% interval'),
    ]
    if SHOW_WIDE_RANGE:
        legend_handles.append(Patch(facecolor=color, alpha=0.17,
                                    label='5–95% interval'))
    legend_handles.append(Line2D([0], [0], color='black', lw=2.3,
                                 label='IMBIE observations'))
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.91,
                        wspace=0.20, hspace=0.28)

    filename = (f'{filename_part}_{family}_'
                f'{"SSP" if family == "CMIP6" else "L_ML_M_H"}_'
                'panels_2000_2300_C2200_boundary.png')
    target = OUTPUT_DIR / filename
    save_figure(fig, target)
    if not SHOW_FIGURES_IN_SPYDER:
        plt.close(fig)
    return target


def main() -> None:
    families = [family for family, enabled in (
        ('CMIP6', PLOT_CMIP6), ('CMIP7', PLOT_CMIP7)) if enabled]
    if not families:
        raise ValueError('Enable at least one of PLOT_CMIP6 and PLOT_CMIP7')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    obs_file = locate_observations(families)
    obs = load_observations(obs_file)
    saved_files = []
    for family in families:
        scenarios = CMIP6_SCENARIOS if family == 'CMIP6' else CMIP7_SCENARIOS
        print(f'\nReading {family} C2200-boundary annual trajectories: {method_root(family)}')
        years, q = quantiles_for_family(family)
        for component in COMPONENTS:
            path = make_component_figure(family, component, scenarios, years, q, obs)
            saved_files.append(path)
            print('  Saved figure:', path.name)
        data_file = save_annual_figure_data(family, years, q)
        print('  Saved figure data:', data_file.name)
        del q

    print('\nCOMPLETE: saved', len(saved_files), 'figures.')
    print('CMIP7 figure scenario codes (if enabled): L, ML, M, H ONLY')
    print('Plotting method: C2200 HARD BOUNDARY; [2100,2200] checkpoints')
    print('Observations:', obs_file)
    print('Output directory:', OUTPUT_DIR)
    if SHOW_FIGURES_IN_SPYDER:
        maybe_show()


if __name__ == '__main__':
    main()
