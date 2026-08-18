#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCRIPT A: Component-specific DP21 temporal-copula diagnostics

Implements the diagnostic/fitting stage of Copilot Alternative 1.

For EAIS, WAIS, and Total AIS under RCP2.6/RCP4.5/RCP8.5:
  - load paired DP21 trajectories
  - extract 2100, 2200, 2300
  - convert each year's ensemble values to percentile ranks
  - Gaussianize ranks with Phi^{-1}
  - estimate interval-specific Gaussian-copula correlations
  - compare empirical Spearman vs Gaussian-implied Spearman
  - diagnose first-order Markov adequacy by comparing:
        observed rho(2100,2300)
    with
        rho(2100,2200) * rho(2200,2300)

DP21 does not include GrIS, so this script explicitly leaves GrIS unresolved.

Outputs:
  dp21_component_interval_copulas.csv
  dp21_component_full_correlations.csv
  dp21_markov_diagnostic.csv
  dp21_component_copula_parameters_for_scriptB.csv
  dp21_gris_requirement.csv

Figures:
  component_interval_latent_correlations.png
  component_empirical_spearman_correlations.png
  markov_2100_2300_diagnostic.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.stats import norm, rankdata

DATA_DIR = Path("/Users/elizabeth/FACTS2/data/module_specific_input_data/deconto21-ais")
PROJECT_DIR = Path("/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025")
OUTPUT_DIR = PROJECT_DIR / "output" / "component_specific_temporal_copulas"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_YEARS = np.array([2100, 2200, 2300], dtype=int)
SCENARIOS = ["rcp26", "rcp45", "rcp85"]
SCENARIO_LABELS = {"rcp26":"RCP2.6","rcp45":"RCP4.5","rcp85":"RCP8.5"}
COMPONENTS = ["EAIS", "WAIS", "Total AIS"]
ADJACENT_PAIRS = [(2100,2200),(2200,2300)]
ALL_PAIRS = [(2100,2200),(2100,2300),(2200,2300)]

def load_dp21_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with Dataset(path, "r") as ds:
        years = np.asarray(ds.variables["years"][:]).astype(int)
        ids = np.asarray(ds.variables["samples"][:])
        samps = np.ma.filled(ds.variables["samps"][:], np.nan).astype(float)
    if samps.shape != (len(years), len(ids)):
        raise ValueError(f"Unexpected shape in {path.name}: {samps.shape}")
    if not np.isfinite(samps).all():
        raise ValueError(f"Non-finite values in {path.name}")
    return years, ids, samps

def load_scenario(scenario):
    ey, ei, e = load_dp21_file(DATA_DIR / f"dp21_eais_{scenario}.nc")
    wy, wi, w = load_dp21_file(DATA_DIR / f"dp21_wais_{scenario}.nc")
    if not np.array_equal(ey, wy):
        raise ValueError(f"EAIS/WAIS years differ for {scenario}")
    if not np.array_equal(ei, wi):
        raise ValueError(f"EAIS/WAIS sample IDs differ for {scenario}")
    return {"years": ey, "EAIS": e, "WAIS": w, "Total AIS": e + w}

def select_target_years(years, samps):
    lookup = {int(y): i for i, y in enumerate(years)}
    missing = [y for y in TARGET_YEARS if int(y) not in lookup]
    if missing:
        raise ValueError(f"Missing years: {missing}")
    idx = [lookup[int(y)] for y in TARGET_YEARS]
    return samps[idx, :].T

def to_uniform_ranks(x):
    n = len(x)
    return rankdata(x, method="average") / (n + 1.0)

def gaussianize(X):
    U = np.column_stack([to_uniform_ranks(X[:,j]) for j in range(X.shape[1])])
    Z = norm.ppf(U)
    return U, Z

def pair_idx(y1, y2):
    m = {int(y): i for i, y in enumerate(TARGET_YEARS)}
    return m[y1], m[y2]

def implied_spearman(rho):
    return (6.0 / np.pi) * np.arcsin(rho / 2.0)

full_records = []
interval_records = []
markov_records = []
parameter_records = []

for scenario in SCENARIOS:
    data = load_scenario(scenario)

    for component in COMPONENTS:
        X = select_target_years(data["years"], data[component])
        U, Z = gaussianize(X)

        rawP = np.corrcoef(X, rowvar=False)
        latentP = np.corrcoef(Z, rowvar=False)
        spear = pd.DataFrame(X, columns=TARGET_YEARS).corr(method="spearman").to_numpy()

        for y1, y2 in ALL_PAIRS:
            i, j = pair_idx(y1, y2)
            rho = float(latentP[i,j])
            full_records.append({
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "component": component,
                "year_pair": f"{y1}-{y2}",
                "year1": y1,
                "year2": y2,
                "n_members": X.shape[0],
                "raw_pearson": float(rawP[i,j]),
                "empirical_spearman": float(spear[i,j]),
                "latent_gaussian_pearson": rho,
                "gaussian_implied_spearman": implied_spearman(rho),
            })

        rhos = {}
        for y1, y2 in ADJACENT_PAIRS:
            i, j = pair_idx(y1, y2)
            rho = float(latentP[i,j])
            rhos[(y1,y2)] = rho
            interval_records.append({
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "component": component,
                "year_pair": f"{y1}-{y2}",
                "year1": y1,
                "year2": y2,
                "latent_gaussian_rho": rho,
                "empirical_spearman": float(spear[i,j]),
                "gaussian_implied_spearman": implied_spearman(rho),
                "raw_pearson": float(rawP[i,j]),
                "n_members": X.shape[0],
            })

        rho12 = rhos[(2100,2200)]
        rho23 = rhos[(2200,2300)]
        i, j = pair_idx(2100,2300)
        observed = float(latentP[i,j])
        implied = rho12 * rho23

        markov_records.append({
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "component": component,
            "rho_2100_2200": rho12,
            "rho_2200_2300": rho23,
            "observed_latent_rho_2100_2300": observed,
            "markov_implied_latent_rho_2100_2300": implied,
            "difference_observed_minus_markov": observed - implied,
            "abs_difference": abs(observed - implied),
            "observed_spearman_2100_2300": float(spear[i,j]),
            "markov_implied_gaussian_spearman_2100_2300": implied_spearman(implied),
        })

        parameter_records.append({
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "component": component,
            "rho_2100_2200": rho12,
            "rho_2200_2300": rho23,
            "direct_rho_2100_2300_diagnostic": observed,
            "markov_implied_rho_2100_2300": implied,
        })

full_df = pd.DataFrame(full_records)
interval_df = pd.DataFrame(interval_records)
markov_df = pd.DataFrame(markov_records)
param_df = pd.DataFrame(parameter_records)

full_df.to_csv(OUTPUT_DIR/"dp21_component_full_correlations.csv", index=False)
interval_df.to_csv(OUTPUT_DIR/"dp21_component_interval_copulas.csv", index=False)
markov_df.to_csv(OUTPUT_DIR/"dp21_markov_diagnostic.csv", index=False)
param_df.to_csv(OUTPUT_DIR/"dp21_component_copula_parameters_for_scriptB.csv", index=False)

pd.DataFrame([{
    "component":"GrIS",
    "status":"not available in DP21 Antarctic ensemble",
    "action_needed":"Identify a Greenland process-model ensemble with matched 2100/2200/2300 trajectories.",
    "do_not_do":"Do not copy EAIS/WAIS/Total AIS correlations onto GrIS without an explicit assumption."
}]).to_csv(OUTPUT_DIR/"dp21_gris_requirement.csv", index=False)

# Figure 1: latent interval correlations
plot_components = ["EAIS","WAIS","Total AIS"]
plot_pairs = ["2100-2200","2200-2300"]
x = np.arange(len(plot_components))
width = 0.12
fig, ax = plt.subplots(figsize=(11,6))
k = 0
for scenario in SCENARIOS:
    for pair in plot_pairs:
        g = (interval_df[(interval_df.scenario == scenario) & (interval_df.year_pair == pair)]
             .set_index("component").reindex(plot_components))
        ax.bar(x + (k-2.5)*width, g["latent_gaussian_rho"], width=width,
               label=f"{SCENARIO_LABELS[scenario]} {pair}")
        k += 1
ax.set_xticks(x)
ax.set_xticklabels(plot_components)
ax.set_ylim(0,1)
ax.set_ylabel("Latent Gaussian correlation")
ax.set_title("DP21 component-specific interval copulas")
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(FIGURE_DIR/"component_interval_latent_correlations.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2: empirical Spearman
fig, ax = plt.subplots(figsize=(11,6))
k = 0
for scenario in SCENARIOS:
    for pair in plot_pairs:
        g = (interval_df[(interval_df.scenario == scenario) & (interval_df.year_pair == pair)]
             .set_index("component").reindex(plot_components))
        ax.bar(x + (k-2.5)*width, g["empirical_spearman"], width=width,
               label=f"{SCENARIO_LABELS[scenario]} {pair}")
        k += 1
ax.set_xticks(x)
ax.set_xticklabels(plot_components)
ax.set_ylim(0,1)
ax.set_ylabel("Empirical Spearman correlation")
ax.set_title("DP21 component-specific temporal rank persistence")
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(FIGURE_DIR/"component_empirical_spearman_correlations.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 3: Markov direct vs implied
labels, observed, implied = [], [], []
for scenario in SCENARIOS:
    for component in plot_components:
        r = markov_df[(markov_df.scenario == scenario) & (markov_df.component == component)].iloc[0]
        labels.append(f"{SCENARIO_LABELS[scenario]}\n{component}")
        observed.append(r["observed_latent_rho_2100_2300"])
        implied.append(r["markov_implied_latent_rho_2100_2300"])

xx = np.arange(len(labels))
bw = 0.38
fig, ax = plt.subplots(figsize=(13,6))
ax.bar(xx-bw/2, observed, width=bw, label="Observed DP21 2100–2300")
ax.bar(xx+bw/2, implied, width=bw, label="Implied by adjacent Markov copulas")
ax.set_xticks(xx)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylim(0,1)
ax.set_ylabel("Latent Gaussian correlation")
ax.set_title("First-order Markov diagnostic for component-specific copulas")
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURE_DIR/"markov_2100_2300_diagnostic.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nADJACENT COMPONENT-SPECIFIC COPULA PARAMETERS")
print(interval_df[[
    "scenario_label","component","year_pair",
    "latent_gaussian_rho","empirical_spearman","gaussian_implied_spearman"
]].round(4).to_string(index=False))

print("\nFIRST-ORDER MARKOV DIAGNOSTIC")
print(markov_df[[
    "scenario_label","component","rho_2100_2200","rho_2200_2300",
    "observed_latent_rho_2100_2300",
    "markov_implied_latent_rho_2100_2300",
    "difference_observed_minus_markov"
]].round(4).to_string(index=False))

print("\nNOTE: DP21 supplies EAIS and WAIS only; GrIS still needs an external process-model source.")
print(f"\nOutputs saved to:\n{OUTPUT_DIR}")
