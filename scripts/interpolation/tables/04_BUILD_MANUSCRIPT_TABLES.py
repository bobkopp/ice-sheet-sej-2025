#!/usr/bin/env python3
"""Build compact, canonical manuscript tables from the final CMIP6/CMIP7 outputs.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from pathlib import Path
import pandas as pd
import core_config as cfg

cmip6_file = (
    cfg.CMIP6_ANNUAL_OUTPUT_DIR / "tables" /
    "all_ice_sheet_components_SSP_projection_table_2020_2050_2100_2150_2200_2300.csv"
)
cmip7_file = (
    cfg.CMIP7_ANNUAL_OUTPUT_DIR / "tables" /
    "all_ice_sheet_components_CMIP7_projection_table_2020_2050_2100_2150_2200_2300.csv"
)

for p in (cmip6_file, cmip7_file):
    if not p.is_file():
        raise FileNotFoundError(p)

cmip6 = pd.read_csv(cmip6_file)
cmip7 = pd.read_csv(cmip7_file)
all_long = pd.concat([cmip6, cmip7], ignore_index=True)

out = cfg.MANUSCRIPT_OUTPUT_DIR
out.mkdir(parents=True, exist_ok=True)

all_long.to_csv(out / "MANUSCRIPT_all_components_all_pathways_long.csv", index=False)

# Wide table: one row per family/pathway/component and percentile-year columns.
value_cols = ["p05_cm", "p17_cm", "p50_cm", "p83_cm", "p95_cm"]
wide = all_long.pivot_table(
    index=["climate_family", "ssp", "component", "reference_period"],
    columns="year",
    values=value_cols,
    aggfunc="first",
)
wide.columns = [f"{stat}_{year}" for stat, year in wide.columns]
wide = wide.reset_index()
wide.to_csv(out / "MANUSCRIPT_all_components_all_pathways_wide.csv", index=False)

# Table-3-friendly total ice-sheet subset.
total = all_long.loc[all_long["component"].eq("Total ice sheet")].copy()
total.to_csv(out / "MANUSCRIPT_total_ice_sheet_all_pathways_long.csv", index=False)

total_wide = total.pivot_table(
    index=["climate_family", "ssp", "reference_period"],
    columns="year",
    values=value_cols,
    aggfunc="first",
)
total_wide.columns = [f"{stat}_{year}" for stat, year in total_wide.columns]
total_wide = total_wide.reset_index()
total_wide.to_csv(out / "MANUSCRIPT_total_ice_sheet_all_pathways_wide.csv", index=False)

# Component table at the principal manuscript endpoint years.
components = all_long.loc[all_long["year"].isin([2050, 2100, 2200, 2300])].copy()
components.to_csv(out / "MANUSCRIPT_component_summary_2050_2100_2200_2300.csv", index=False)

print("Saved manuscript tables to:", out)
