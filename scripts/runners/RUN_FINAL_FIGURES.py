#!/usr/bin/env python3
"""Unified runner for validated SEJ manuscript and supplementary figures.

Created on: 2026-10-26

Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parents[1]  # scripts root
REPO = HERE.parent.resolve()
OUTPUT = (REPO / 'output').resolve()
# Each figure gets its own root. Existing scripts still determine filenames.
FIGURES = {
    'warming': ('main_text', 'figure_01_warming', 'warming_scenario_plot.py'),
    'performance': ('main_text', 'figure_02_performance', 'SA_Inf_MAPE_fig.py'),
    'cluster': ('main_text', 'figure_03_slr_by_year', 'SLRxYEARxwarminglevel_Cluster_fig.py'),
    'violin': ('main_text', 'figures_04_05_violin', 'split_violin_plot_adjustedB19_B25.py'),
    'drivers': ('main_text', 'figure_06_drivers', 'drivers_scaled.py'),
    'pdfs': ('supplementary', 'b19_b25_pdfs', 'PDF_individual_B19_comparison_all.py'),
    'experts': ('supplementary', 'expert_judgments_calibration', 'plot_expert_judgments_baselines_ranked.py'),
    'mass_change': ('supplementary', 'mass_change_observations', 'mass_change_obs.py'),
    'copulas': ('supplementary', 'copula_dependence', 'copula_dependence_matrices.py'),
    'selection_probabilities': ('supplementary', 'scenario_selection_probabilities', 'PLOT_NORMALIZED_SELECTION_CONTOURS_CURRENT_METHOD_metarepo.py'),
    'four_ssp_exact': ('supplementary', 'four_ssp_ar6_style', 'PLOT_CMIP6_FOUR_SSP_AR6_STYLE_2150_2300_SHARED_Y_left150_zero_aligned_across_columns.py'),
}
# The original consolidated suite also creates figures the user does not want
# (e.g. diagnostics, realized fractions). Keep it separately opt-in.
OPTIONAL = {'consolidated_legacy': ('legacy_optional', 'consolidated_suite', 'MASTER_FAIR2_ICE_SHEET_FIGURES_CONSOLIDATED.py')}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--only',default='',help='Comma-separated figure group names; default is all eleven validated groups')
    parser.add_argument('--execute',action='store_true',help='Run figures (default: preview only)')
    parser.add_argument('--allow-overwrite',action='store_true',help='Allow rerunning into an existing group directory')
    parser.add_argument('--output-root',default=str(OUTPUT/'metarepo_figures_organized'),help='Isolated root inside the starter output directory')
    args=parser.parse_args()
    catalog={**FIGURES,**OPTIONAL}
    if args.list:
        for key,(category,folder,script) in catalog.items():
            print(f'{key:25} {category}/{folder}  {script}'+ (' [OPTIONAL]' if key in OPTIONAL else ''))
        return
    requested=[x.strip() for x in args.only.split(',') if x.strip()] if args.only else list(FIGURES)
    unknown=set(requested)-set(catalog)
    if unknown:parser.error('Unknown groups: '+', '.join(sorted(unknown)))
    if len(set(requested))!=len(requested):parser.error('Repeated figure group')
    root=Path(args.output_root).expanduser().resolve()
    if OUTPUT not in root.parents:parser.error(f'Unsafe output root: {root}; must be a subdirectory of {OUTPUT}')
    base=os.environ.copy()
    core=base.get('SEJ_CORE_OUTPUT_ROOT')
    if not core or OUTPUT not in Path(core).expanduser().resolve().parents:
        parser.error('Run source test_environment.sh first; SEJ_CORE_OUTPUT_ROOT must be inside starter/output')
    base['FIG_PROJECT_DIR']=base.get('FIG_PROJECT_DIR') or base.get('SEJ_PROJECT_DIR') or str(REPO)
    base['SEJ_ORGANIZED_FIGURES']='1'
    for key in requested:
        category,folder,filename=catalog[key]
        script=HERE/("original_sej" if key in {"warming","performance","cluster","violin","drivers","pdfs","experts","mass_change","copulas"} else "interpolation") / (
            {"warming":"warming_scenarios", "performance":"expert_analysis", "cluster":"distributions", "violin":"distributions", "drivers":"expert_analysis", "pdfs":"distributions", "experts":"expert_analysis", "mass_change":"expert_analysis", "copulas":"distributions"}.get(key,"figures")
        )/filename
        if not script.is_file():parser.error(f'Missing script: {script}')
        target=root/category/folder
        if args.execute and not args.allow_overwrite and target.exists() and any(target.rglob('*')):
            parser.error(f'Existing outputs under {target}; choose a new --output-root or explicitly --allow-overwrite')
        env=base.copy()
        env['FIG_OUTPUT_DIR']=str(target)
        env['FIG_FAIR2_OUTPUT_DIR']=str(target)
        env['SEJ_FIGURE_GROUP_OUTPUT']=str(target)
        env['PYTHONPATH']=str(HERE)+os.pathsep+env.get('PYTHONPATH','')
        print(f"{'RUN' if args.execute else 'PREVIEW'} {key}: {target}",flush=True)
        if args.execute:
            # Make organized shared modules available to figure subprocesses.

            script_root = Path(__file__).resolve().parents[1]

            module_dirs = [
                script_root / "shared",
                script_root / "original_sej" / "data_processing",
                script_root / "interpolation" / "inputs",
                script_root / "interpolation" / "annual_projections",
                script_root / "interpolation" / "tables",
            ]
            
            existing_pythonpath = env.get("PYTHONPATH", "")
            
            env["PYTHONPATH"] = os.pathsep.join(
                [str(path) for path in module_dirs]
                + ([existing_pythonpath] if existing_pythonpath else [])
            )
            subprocess.run([sys.executable,str(script)],cwd=HERE,env=env,check=True)
    if not args.execute:print('Dry run; no scripts executed.')
if __name__=='__main__':main()
