#!/usr/bin/env python3
"""Diagnostic only: expected CMIP7 SEJ mixture vs realized and CRN-counterfactual draws.

Read-only verification of the adopted CMIP7 common-random-number sampling rule.
Does not alter coupling, scenario selection, sampled trajectories or production files.
Exact mixture CDF is calculated from ALL rows of the coupled SEJ warming ensembles.
At 2050/2100/2200/2300, reference mixtures use each pathway's MEAN effective
scenario weights (not selected fractions). A recreated CRN row map (required to match the final sampled file) shares
SEJ row indices by FaIR sample_id *within the three warming scenarios*.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core_config as cfg

OUTPUT_DIR = cfg.CORE_OUTPUT_ROOT / 'diagnostics_cmip7_mixture_crn'
YEARS = (2050, 2100, 2200, 2300)
COMPONENTS = ('E', 'W', 'G', 'AIS', 'TOTAL')
SCENARIOS = ('LOW', 'INTERMEDIATE', 'HIGH')
CODES = ('2', '35', '5')
PATHWAYS = ('LN', 'VL', 'L', 'ML', 'M', 'HL', 'H')
PERCENTILES = (5, 17, 50, 83, 95)
CRN_SEED = cfg.CMIP7_CRN_SEED
from core_annual_tension import crn_indices_by_sample_id


def load_inputs():
    selection = pd.read_csv(cfg.CMIP7_SELECTION_FILE)
    sampled = pd.read_csv(cfg.CMIP7_ANNUAL_OUTPUT_DIR / 'sampled_sej2025_realizations.csv')
    coupled = pd.read_csv(cfg.COUPLED_SEJ_FILE)
    imbie = pd.read_csv(cfg.CMIP7_ANNUAL_OUTPUT_DIR / 'imbie_observed_slr_1979_2023_monthly_rebased_to_1995_2014.csv')
    for name, frame in [('selection',selection), ('sampled', sampled)]:
        required = {'ssp', 'sample_id', 'selected_scenario'}
        if not required.issubset(frame.columns):
            raise ValueError(f'{name}: missing {required - set(frame)}')
    if 'sej2025_row_index' not in sampled:
        raise ValueError('Sampled CSV missing sej2025_row_index')
    for c in ('weight_low','weight_intermediate','weight_high'):
        if c not in selection: raise ValueError(f'Selection CSV missing {c}')
    if len(selection) != len(sampled):
        raise ValueError('CMIP7 selection and sampled row counts differ')
    cols = ['ssp','sample_id','selected_scenario']
    left = selection[cols].astype(str).reset_index(drop=True)
    right = sampled[cols].astype(str).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError('Selection and sampled metadata row alignment differs; refusing to compare by position')
    month = pd.to_datetime(imbie['Date']).dt.year.eq(2005)
    if month.sum() != 12: raise ValueError('IMBIE reference shifts require exactly 12 months in 2005')
    shifts = {v: float(imbie.loc[month, f'{v}_cm'].mean()) for v in ('E','W','G')}
    if len(coupled) < 2: raise ValueError('Insufficient coupled samples')
    return selection, sampled, coupled, shifts


def source_knots(coupled, shifts):
    """Return stacked original coupled values (cm, 1995–2014 baseline)."""
    result = {}
    for year in YEARS:
        result[year] = {}
        for scenario, code in zip(SCENARIOS, CODES):
            components = {}
            for v in ('E','W','G'):
                if year == 2050:
                    col = f'EMU2050_{v}{code}_cm'
                    if col not in coupled: raise ValueError(f'Missing {col}')
                    # Stage 02 EMU columns are stored in cm in the 1995–2014 frame.
                    values = coupled[col].to_numpy(float)
                else:
                    col = f'{v}{code}_{ {2100: '21', 2200: '22', 2300: '23'}[year] }'
                    if col not in coupled: raise ValueError(f'Missing {col}')
                    values = coupled[col].to_numpy(float)*0.1 + shifts[v]
                if not np.isfinite(values).all(): raise ValueError(f'Nonfinite {col}')
                components[v] = values
            components['AIS'] = components['E'] + components['W']
            components['TOTAL'] = components['AIS'] + components['G']
            result[year][scenario] = components
    return result


def weighted_percentiles(values, scenarios, probs):
    """Exact finite-ensemble mixture quantiles of equal-sized SEJ scenario panels."""
    values = np.concatenate(values)
    panel_ids = np.concatenate([np.full(len(x), j, dtype=np.int8) for j,x in enumerate(scenarios)])
    n_each = np.array([len(x) for x in scenarios])
    if not np.all(n_each == n_each[0]): raise ValueError('Expected equally sized scenario panels')
    order = np.argsort(values, kind='stable')
    sorted_vals = values[order]
    sorted_weights = np.asarray(probs, float)[panel_ids[order]] / n_each[0]
    cdf = np.cumsum(sorted_weights)
    cdf[-1] = 1.
    return np.interp(np.asarray(PERCENTILES)/100., cdf, sorted_vals)


def summarize(vals):
    return np.percentile(np.asarray(vals, float), PERCENTILES)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selection, sampled, coupled, shifts = load_inputs()
    knots = source_knots(coupled, shifts)
    for frame in (selection, sampled):
        frame['ssp'] = frame['ssp'].astype(str).str.strip()
        frame['sample_id'] = pd.to_numeric(frame['sample_id'], errors='raise').astype(int)
        frame['selected_scenario'] = frame['selected_scenario'].astype(str).str.strip().str.upper()
    # Reconstruct with the EXACT same algorithm used in Stage 03B.
    expected_indices = crn_indices_by_sample_id(selection, len(coupled))
    pathway_sets = {p: set(selection.loc[selection.ssp.eq(p), 'sample_id']) for p in PATHWAYS}
    common_ids = set.intersection(*pathway_sets.values())
    aligned_all = all(x == pathway_sets[PATHWAYS[0]] for x in pathway_sets.values())
    if not aligned_all:
        raise ValueError('CMIP7 FaIR sample IDs are not aligned across all seven pathways')
    shared_ids = sorted(common_ids)
    # A map is retained solely for the diagnostic's per-pathway summaries.
    crn_lookup = selection[['ssp', 'sample_id', 'selected_scenario']].copy()
    crn_lookup['expected_index'] = expected_indices
    crn_rows = {sc: dict(zip(crn_lookup.loc[crn_lookup.selected_scenario.eq(sc), 'sample_id'],
                             crn_lookup.loc[crn_lookup.selected_scenario.eq(sc), 'expected_index']))
                for sc in SCENARIOS}
    frac_rows=[]; rows=[]; samplemap=[]
    for p in PATHWAYS:
        select = selection.loc[selection.ssp.eq(p)].copy()
        real = sampled.loc[sampled.ssp.eq(p)].copy()
        if len(select)==0: raise ValueError(f'No CMIP7 selection for pathway {p}')
        probs = select[['weight_low','weight_intermediate','weight_high']].to_numpy(float).mean(axis=0)
        counts = select['selected_scenario'].value_counts(normalize=True)
        for i, scenario in enumerate(SCENARIOS):
            frac_rows.append({'pathway':p,'scenario':scenario,'expected_weight':probs[i], 'realized_selection_fraction':counts.get(scenario,0.),'difference':counts.get(scenario,0.)-probs[i], 'n_samples':len(select)})
        source_index = real['sej2025_row_index'].to_numpy(dtype=int)
        if (source_index<0).any() or (source_index>=len(coupled)).any(): raise ValueError('Out-of-range sampled row index')
        chosen = real['selected_scenario'].to_numpy()
        crn_select = select.loc[select.sample_id.isin(common_ids)].copy()
        crn_source = crn_lookup.loc[crn_select.index, 'expected_index'].to_numpy(dtype=int)
        # The final 03B workflow now USES CRN. Confirm exact reproduction of
        # its sample-row assignments rather than merely displaying a hypothetical run.
        current_map = real.set_index('sample_id')['sej2025_row_index']
        actual_crn = current_map.loc[crn_select.sample_id].to_numpy(dtype=int)
        if not np.array_equal(actual_crn, crn_source):
            mismatched = int(np.count_nonzero(actual_crn != crn_source))
            raise AssertionError(
                f'{p}: final CMIP7 sampling is not the expected FaIR-ID CRN rule '
                f'({mismatched} row mismatches). Rerun Stage 03B from THIS package; verify both files come from the same run.'
            )
        for sid,sc,idx in zip(crn_select.sample_id,crn_select.selected_scenario,crn_source):
            samplemap.append({'pathway':p,'sample_id':int(sid),'selected_scenario':sc,'crn_counterfactual_sej_row_index':int(idx)})
        for year in YEARS:
            for v in COMPONENTS:
                original = np.empty(len(real)); original[:] = np.nan
                crn = np.empty(len(crn_select)); crn[:] = np.nan
                for sc in SCENARIOS:
                    mask = chosen==sc
                    original[mask] = knots[year][sc][v][source_index[mask]]
                    cmask = crn_select['selected_scenario'].eq(sc).to_numpy()
                    crn[cmask] = knots[year][sc][v][crn_source[cmask]]
                if not np.isfinite(original).all() or not np.isfinite(crn).all(): raise ValueError('Unassigned rows')
                expected = weighted_percentiles([knots[year][sc][v] for sc in SCENARIOS], [knots[year][sc][v] for sc in SCENARIOS], probs)
                realized = summarize(original)
                counterfactual = summarize(crn)
                for q,ex,re,cf in zip(PERCENTILES,expected,realized,counterfactual):
                    rows.append({'pathway':p,'year':year,'component':v,'percentile':q,
                        'expected_mixture_cm':ex,'realized_sample_cm':re,'crn_counterfactual_cm':cf,
                        'realized_minus_expected_cm':re-ex,'crn_minus_expected_cm':cf-ex,
                        'realized_minus_crn_cm':re-cf,'n_realized':len(real),'n_crn':len(crn_select)})
    result=pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR/'CMIP7_02IB_mixture_vs_realized_vs_CRN_quantiles.csv',index=False)
    pd.DataFrame(frac_rows).to_csv(OUTPUT_DIR/'CMIP7_02IB_expected_vs_realized_scenario_fractions.csv',index=False)
    pd.DataFrame(samplemap).to_csv(OUTPUT_DIR/'CMIP7_02IB_CRN_counterfactual_row_map.csv',index=False)
    diffs=[]
    for year in YEARS:
        for comp in COMPONENTS:
            for pct in (50,95):
                s=result[(result.year==year)&(result.component==comp)&(result.percentile==pct)].set_index('pathway')
                for a,b in [('LN','L'),('VL','L'),('LN','VL')]:
                    rec={'year':year,'component':comp,'percentile':pct,'contrast':f'{a} minus {b}'}
                    for label,column in [('expected','expected_mixture_cm'),('realized','realized_sample_cm'),('crn','crn_counterfactual_cm')]:
                        rec[f'{label}_difference_cm']=s.loc[a,column]-s.loc[b,column]
                    diffs.append(rec)
    df=pd.DataFrame(diffs)
    df.to_csv(OUTPUT_DIR/'CMIP7_02IB_LN_VL_L_pairwise_differences.csv',index=False)
    # Output intuitive focused plots, medians and p95, with NO modification to production.
    for comp in ('G','AIS','TOTAL'):
        fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharex=True)
        for ax,pct in zip(axes,(50,95)):
            view=df[(df.component==comp)&(df.percentile==pct)&(df.contrast.isin(['LN minus L','VL minus L']))]
            for contrast,ls in [('LN minus L','-'),('VL minus L','--')]:
                part=view[view.contrast==contrast].sort_values('year')
                for key,col,marker in [('Expected','expected_difference_cm','o'),('Realized','realized_difference_cm','s'),('CRN','crn_difference_cm','^')]:
                    ax.plot(part.year,part[col],marker=marker,ls=ls,label=f'{contrast}: {key}')
            ax.axhline(0,color='0.55',lw=.8)
            ax.set_title(f'{comp}: {pct}th percentile')
            ax.set_xlabel('Year'); ax.set_ylabel('Difference (cm)'); ax.grid(axis='y',alpha=.2)
        h,l=axes[0].get_legend_handles_labels()
        fig.legend(h,l,loc='lower center',ncol=3,fontsize=8,frameon=False)
        fig.subplots_adjust(bottom=.26,top=.88,wspace=.28)
        path=OUTPUT_DIR/f'CMIP7_02IB_LN_VL_vs_L_{comp}.png'
        fig.savefig(path,dpi=200,bbox_inches='tight');plt.close(fig)
    (OUTPUT_DIR/'method_and_scope.json').write_text(json.dumps({
      'input_root':str(cfg.CORE_OUTPUT_ROOT),'coupling':str(cfg.COUPLED_SEJ_FILE),
      'source_selection':str(cfg.CMIP7_SELECTION_FILE),
      'source_sampled':str(cfg.CMIP7_ANNUAL_OUTPUT_DIR/'sampled_sej2025_realizations.csv'),
      'sample_id_sets_match':aligned_all, 'common_sample_id_count':len(common_ids),
      'crn_seed':CRN_SEED,'scope':'knot years only; full exact mixture from 3 complete coupled ensembles',
      'note':'Read-only check of CRN already used by Stage 03B; does not regenerate annual trajectories'
    },indent=2))
    print('\nSaved CMIP7 02I-B mixture/realized/CRN diagnostics:',OUTPUT_DIR)
    focus=df[(df.component=='TOTAL')&(df.percentile==50)]
    print(focus[['year','contrast','expected_difference_cm','realized_difference_cm','crn_difference_cm']].to_string(index=False))

if __name__=='__main__':main()
