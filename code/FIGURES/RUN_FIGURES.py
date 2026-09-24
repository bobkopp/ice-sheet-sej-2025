#!/usr/bin/env python3
"""
Run the existing SEJ manuscript plotting workflows with one configuration.

Edit USER SETTINGS below, then from Spyder press F5 or in Terminal use:
    python RUN_FIGURES.py                 # every figure group
    python RUN_FIGURES.py --check         # inspect inputs, do not plot
    python RUN_FIGURES.py --list          # list group names
    python RUN_FIGURES.py --only warming performance
    python RUN_FIGURES.py --continue-on-error

No physical SEJ realizations, climate trajectories or scenario assignments
are regenerated.

author: ed881
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

# ============================================================================
# USER SETTINGS
# ============================================================================
PROJECT_DIR = Path.home() / 'Documents/GitHub/ice-sheet-sej-2025'
DATA_DIR = PROJECT_DIR / 'data'                 # observational / elicitation inputs
UPSTREAM_OUTPUT_DIR = PROJECT_DIR / 'output'   # EXISTING projection results (read-only)
FIGURE_OUTPUT_DIR = UPSTREAM_OUTPUT_DIR / 'manuscript_figures'  # NEW figures/tables

# Change any filename here if your input data use a different name/location.
B25_MONTE_CARLO_FILE = DATA_DIR / 'sej2025_SplitPanels_liz_updated.csv'
B19_SUMMARY_FILE = DATA_DIR / 'B19_data.xlsx'
B19_ADJUSTED_FILE = DATA_DIR / 'B19_process_and_total_MC_adjusted_to_B25_baseline.xlsx'
EXPERT_QUESTIONS_FILE = DATA_DIR / 'ICE+SHEET+SEJ+target+questions_July+15,+2026_CLEANER_LIZ.csv'
EXPERT_WORKBOOK_FILE = DATA_DIR / '2025_FINAL_Split_Panels.xls'
PERFORMANCE_FILE = DATA_DIR / 'SA_Inf_MAPE.csv'
AR6_WORKBOOK_FILE = DATA_DIR / 'ipcc_ar6_sea_level_projection_global.xlsx'
COUPLED_SEJ_FILE = (UPSTREAM_OUTPUT_DIR / 'scenario_specific_temporal_coupling_by_warming'
                    / 'SEJ25_scenario_specific_temporally_coupled_by_warming.csv')

METHOD_ID = 'B_2100_2200_boundary_2200'  # fixed scientific method for these plots
CMIP6_METHOD_DIR = (UPSTREAM_OUTPUT_DIR
    / 'checkpoint_2100_2200_boundary_sensitivity_three_methods' / METHOD_ID)
CMIP7_METHOD_DIR = UPSTREAM_OUTPUT_DIR / 'cmip7_boundary_2200' / METHOD_ID
CMIP6_ANNUAL_DIR = CMIP6_METHOD_DIR / 'annual_trajectory_method'
CMIP7_ANNUAL_DIR = CMIP7_METHOD_DIR / 'annual_trajectory_method'
CMIP6_SELECTION_FILE = CMIP6_METHOD_DIR / 'scenario_selection/fair_b25_baseline_scenario_selection.csv'
CMIP7_SELECTION_FILE = CMIP7_METHOD_DIR / 'scenario_selection/fair_b25_baseline_scenario_selection.csv'

DPI = 300
SHOW_FIGURES = False   # False saves PNG only
INCLUDE_UNSCALED_DRIVERS = False
# ============================================================================

PLOT_DIR = Path(__file__).resolve().parent / 'plotting'
TASKS = (
    ('warming', 'warming_scenario_plot.py', ()),
    ('performance', 'SA_Inf_MAPE_fig.py', ('FIG_PERFORMANCE_FILE',)),
    ('expert_judgments', 'plot_expert_judgments_baselines_ranked.py', ('FIG_EXPERT_WORKBOOK_FILE',)),
    ('drivers', 'drivers_scaled.py', ('FIG_B25_MC_FILE', 'FIG_EXPERT_QUESTIONS_FILE')),
    ('clustered', 'SLRxYEARxwarminglevel_Cluster_fig.py', ('FIG_B25_MC_FILE',)),
    ('b19_densities', 'PDF_individual_B19_comparison_all.py', ('FIG_B25_MC_FILE', 'FIG_B19_SUMMARY_FILE')),
    ('b19_violins', 'split_violin_plot_adjustedB19_B25.py', ('FIG_B25_MC_FILE', 'FIG_B19_SUMMARY_FILE', 'FIG_B19_ADJUSTED_FILE')),
    ('annual_trajectories', 'plot_final_ice_sheet_results_AR6_three_panel_C2100_C2200_boundary_2200.py',
        ('FIG_CMIP6_ANNUAL_DIR/annual_ice_sheet_trajectories_2000_2300.nc',
         'FIG_CMIP6_ANNUAL_DIR/sampled_sej2025_realizations.csv',
         'FIG_CMIP6_ANNUAL_DIR/imbie_observed_slr_2000_2023_monthly_rebased_to_2005.csv',
         'FIG_COUPLED_SEJ_FILE', 'FIG_AR6_WORKBOOK_FILE')),
    ('cmip_panels', 'PLOT_SIX_CMIP6_CMIP7_C2200_BOUNDARY.py',
        ('FIG_CMIP6_ANNUAL_DIR/annual_ice_sheet_trajectories_2000_2300.nc',
         'FIG_CMIP7_ANNUAL_DIR/annual_ice_sheet_trajectories_2000_2300.nc',
         'FIG_CMIP6_ANNUAL_DIR/sampled_sej2025_realizations.csv',
         'FIG_CMIP7_ANNUAL_DIR/sampled_sej2025_realizations.csv',
         'FIG_CMIP6_ANNUAL_DIR/imbie_observed_slr_2000_2023_monthly_rebased_to_2005.csv')),
    ('realized_fractions', 'PLOT_CMIP6_CMIP7_NORMALIZED_REALIZED_FRACTIONS_C2200_BOUNDARY.py',
        ('FIG_CMIP6_SELECTION_FILE', 'FIG_CMIP7_SELECTION_FILE')),
)


def settings(args):
    project = (args.project_dir or PROJECT_DIR).expanduser().resolve()
    # When changing the project via --project-dir, default input/output roots follow.
    data = (args.data_dir or (project / 'data' if args.project_dir else DATA_DIR)).expanduser().resolve()
    upstream = (args.upstream_dir or (project / 'output' if args.project_dir else UPSTREAM_OUTPUT_DIR)).expanduser().resolve()
    output = (args.output_dir or (upstream / 'manuscript_figures' if (args.project_dir or args.upstream_dir) else FIGURE_OUTPUT_DIR)).expanduser().resolve()
    # The explicit top-of-file file overrides apply when using the top-of-file paths;
    # CLI root overrides use conventional paths under their new roots by default.
    override = any((args.project_dir, args.data_dir, args.upstream_dir))
    def file_default(explicit, default):
        return default if override else explicit
    method6 = upstream / 'checkpoint_2100_2200_boundary_sensitivity_three_methods' / METHOD_ID
    method7 = upstream / 'cmip7_boundary_2200' / METHOD_ID
    default_annual6 = method6 / 'annual_trajectory_method'
    default_annual7 = method7 / 'annual_trajectory_method'
    mappings = {
        'FIG_PROJECT_DIR':project, 'FIG_DATA_DIR':data, 'FIG_UPSTREAM_DIR':upstream,
        'FIG_OUTPUT_DIR':output,
        'FIG_B25_MC_FILE':file_default(B25_MONTE_CARLO_FILE, data / 'sej2025_SplitPanels_liz_updated.csv'),
        'FIG_B19_SUMMARY_FILE':file_default(B19_SUMMARY_FILE, data / 'B19_data.xlsx'),
        'FIG_B19_ADJUSTED_FILE':file_default(B19_ADJUSTED_FILE, data / 'B19_process_and_total_MC_adjusted_to_B25_baseline.xlsx'),
        'FIG_EXPERT_QUESTIONS_FILE':file_default(EXPERT_QUESTIONS_FILE, data / 'ICE+SHEET+SEJ+target+questions_July+15,+2026_CLEANER_LIZ.csv'),
        'FIG_EXPERT_WORKBOOK_FILE':file_default(EXPERT_WORKBOOK_FILE, data / '2025_FINAL_Split_Panels.xls'),
        'FIG_PERFORMANCE_FILE':file_default(PERFORMANCE_FILE, data / 'SA_Inf_MAPE.csv'),
        'FIG_AR6_WORKBOOK_FILE':file_default(AR6_WORKBOOK_FILE, data / 'ipcc_ar6_sea_level_projection_global.xlsx'),
        'FIG_COUPLED_SEJ_FILE':file_default(COUPLED_SEJ_FILE, upstream / 'scenario_specific_temporal_coupling_by_warming' / 'SEJ25_scenario_specific_temporally_coupled_by_warming.csv'),
        'FIG_CMIP6_ANNUAL_DIR':file_default(CMIP6_ANNUAL_DIR, default_annual6),
        'FIG_CMIP7_ANNUAL_DIR':file_default(CMIP7_ANNUAL_DIR, default_annual7),
        'FIG_CMIP6_SELECTION_FILE':file_default(CMIP6_SELECTION_FILE, method6 / 'scenario_selection/fair_b25_baseline_scenario_selection.csv'),
        'FIG_CMIP7_SELECTION_FILE':file_default(CMIP7_SELECTION_FILE, method7 / 'scenario_selection/fair_b25_baseline_scenario_selection.csv'),
    }
    env = os.environ.copy()
    env.update({k:str(Path(v).expanduser().resolve()) for k,v in mappings.items()})
    env.update({'FIG_DPI':str(args.dpi if args.dpi is not None else DPI),
                'FIG_SHOW':'1' if (args.show or SHOW_FIGURES) else '0',
                'FIG_INCLUDE_UNSCALED_DRIVERS':'1' if (args.include_unscaled_drivers or INCLUDE_UNSCALED_DRIVERS) else '0'})
    if not (args.show or SHOW_FIGURES):
        env['MPLBACKEND'] = 'Agg'
    env['PYTHONPATH'] = str(PLOT_DIR) + os.pathsep + env.get('PYTHONPATH', '')
    return env


def required_paths(group, env):
    inputs = dict((name, inputs) for name, _script, inputs in TASKS)[group]
    return [(name, Path(env[name.split('/')[0]]).joinpath(*name.split('/')[1:]))
            for name in inputs]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--list', action='store_true', help='List available figure groups')
    ap.add_argument('--only', nargs='+', metavar='GROUP', help='Generate only selected groups (see --list)')
    ap.add_argument('--skip', nargs='*', default=[], metavar='GROUP', help='Skip selected groups')
    ap.add_argument('--check', action='store_true', help='Check input paths without making outputs')
    ap.add_argument('--continue-on-error', action='store_true', help='Run independent groups even if one fails')
    ap.add_argument('--project-dir', type=Path)
    ap.add_argument('--data-dir', type=Path)
    ap.add_argument('--upstream-dir', type=Path)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--dpi', type=int)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--include-unscaled-drivers', action='store_true')
    args = ap.parse_args(argv)
    names = [item[0] for item in TASKS]
    if args.list:
        for group, script, _ in TASKS:print(f'{group:22s} {script}')
        return 0
    selected = names if not args.only else args.only
    unknown = set(selected + args.skip) - set(names)
    if unknown:ap.error(f'Unknown group(s): {sorted(unknown)}. Run --list.')
    selected = [item for item in names if item in selected and item not in args.skip]
    if not selected:ap.error('No groups selected')
    env = settings(args)
    out = Path(env['FIG_OUTPUT_DIR'])
    reserved = [Path(env[x]) for x in ('FIG_PROJECT_DIR','FIG_DATA_DIR','FIG_UPSTREAM_DIR')]
    if out in reserved or any(out == Path(env[x]) for x in ('FIG_CMIP6_ANNUAL_DIR','FIG_CMIP7_ANNUAL_DIR')):
        ap.error('Output directory must be separate from project, data, upstream and annual-data directories.')
    missing = {}
    for group in selected:
        absent = [str(path) for _,path in required_paths(group,env) if not path.is_file()]
        if absent:missing[group]=absent
    if args.check:
        print('Input preflight (no plotting, no output folders created):')
        for group in selected:
            print(f'  {group}: '+('MISSING INPUTS' if group in missing else 'OK'))
            for path in missing.get(group,[]):print('    '+path)
        return 1 if missing else 0
    if missing and not args.continue_on_error:
        print('Missing input files; no plotting started. Edit USER SETTINGS or run --only GROUP.', file=sys.stderr)
        for group,paths in missing.items():
            print(f'  {group}:',file=sys.stderr)
            for path in paths:print('    '+path,file=sys.stderr)
        return 2
    out.mkdir(parents=True,exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    records=[]
    for group,filename,_ in TASKS:
        if group not in selected:continue
        if group in missing:
            records.append({'group':group,'status':'missing_inputs','inputs':missing[group]})
            print(f'\nSKIP {group}: missing inputs',file=sys.stderr)
            continue
        print(f'\n=== {group}: {filename} ===',flush=True)
        script=PLOT_DIR/filename
        try:
            completed=subprocess.run([sys.executable,str(script)],cwd=PLOT_DIR,env=env,check=False)
            status='ok' if completed.returncode==0 else 'failed'
            records.append({'group':group,'script':filename,'status':status,'exit_code':completed.returncode})
            if status!='ok' and not args.continue_on_error:break
        except OSError as exc:
            records.append({'group':group,'script':filename,'status':'failed','error':str(exc)})
            if not args.continue_on_error:break
    manifest={'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
              'settings':{key:env[key] for key in env if key.startswith('FIG_')},'runs':records}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    good=sum(item['status']=='ok' for item in records)
    print(f'\nFinished: {good}/{len(selected)} groups succeeded. Outputs: {out}')
    print('Run report:',out/'run_manifest.json')
    return 0 if good == len(selected) else 1

if __name__=='__main__':
    sys.exit(main())
