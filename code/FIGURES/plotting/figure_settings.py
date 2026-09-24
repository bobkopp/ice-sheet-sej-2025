"""Single configuration adapter for all manuscript plotting scripts.

RUN_FIGURES.py sets these environment values for each subprocess. To execute an
individual plotting script, edit RUN_FIGURES.py's USER SETTINGS and use --only,
or export the FIG_* variables listed below before running it directly.
"""
from __future__ import annotations
import os
from pathlib import Path


def location(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()

PROJECT_ROOT = location('FIG_PROJECT_DIR', Path.home() / 'Documents/GitHub/ice-sheet-sej-2025')
DATA_ROOT = location('FIG_DATA_DIR', PROJECT_ROOT / 'data')
UPSTREAM_ROOT = location('FIG_UPSTREAM_DIR', PROJECT_ROOT / 'output')
FIGURE_ROOT = location('FIG_OUTPUT_DIR', UPSTREAM_ROOT / 'manuscript_figures')
SHOW_FIGURES = os.environ.get('FIG_SHOW', '0') == '1'
DPI = int(os.environ.get('FIG_DPI', '300'))

B25_MC_FILE = location('FIG_B25_MC_FILE', DATA_ROOT / 'sej2025_SplitPanels_liz_updated.csv')
B19_SUMMARY_FILE = location('FIG_B19_SUMMARY_FILE', DATA_ROOT / 'B19_data.xlsx')
B19_ADJUSTED_FILE = location('FIG_B19_ADJUSTED_FILE', DATA_ROOT / 'B19_process_and_total_MC_adjusted_to_B25_baseline.xlsx')
EXPERT_QUESTIONS_FILE = location('FIG_EXPERT_QUESTIONS_FILE', DATA_ROOT / 'ICE+SHEET+SEJ+target+questions_July+15,+2026_CLEANER_LIZ.csv')
EXPERT_WORKBOOK_FILE = location('FIG_EXPERT_WORKBOOK_FILE', DATA_ROOT / '2025_FINAL_Split_Panels.xls')
PERFORMANCE_FILE = location('FIG_PERFORMANCE_FILE', DATA_ROOT / 'SA_Inf_MAPE.csv')
AR6_WORKBOOK_FILE = location('FIG_AR6_WORKBOOK_FILE', DATA_ROOT / 'ipcc_ar6_sea_level_projection_global.xlsx')
COUPLED_SEJ_FILE = location('FIG_COUPLED_SEJ_FILE', UPSTREAM_ROOT / 'scenario_specific_temporal_coupling_by_warming' / 'SEJ25_scenario_specific_temporally_coupled_by_warming.csv')
METHOD_ID = 'B_2100_2200_boundary_2200'
CMIP6_METHOD_ROOT = UPSTREAM_ROOT / 'checkpoint_2100_2200_boundary_sensitivity_three_methods' / METHOD_ID
CMIP7_METHOD_ROOT = UPSTREAM_ROOT / 'cmip7_boundary_2200' / METHOD_ID
CMIP6_ANNUAL_DIR = location('FIG_CMIP6_ANNUAL_DIR', CMIP6_METHOD_ROOT / 'annual_trajectory_method')
CMIP7_ANNUAL_DIR = location('FIG_CMIP7_ANNUAL_DIR', CMIP7_METHOD_ROOT / 'annual_trajectory_method')
CMIP6_SELECTION_FILE = location('FIG_CMIP6_SELECTION_FILE', CMIP6_METHOD_ROOT / 'scenario_selection' / 'fair_b25_baseline_scenario_selection.csv')
CMIP7_SELECTION_FILE = location('FIG_CMIP7_SELECTION_FILE', CMIP7_METHOD_ROOT / 'scenario_selection' / 'fair_b25_baseline_scenario_selection.csv')


def output_for(*subfolders: str) -> Path:
    """Create and return a safe destination below the configured figure root."""
    if not subfolders or any(x in ('', '.', '..') or '/' in x or '\\' in x for x in subfolders):
        raise ValueError('Use nonempty subfolder names without separators or ..')
    target = FIGURE_ROOT.joinpath(*subfolders)
    target.mkdir(parents=True, exist_ok=True)
    return target
