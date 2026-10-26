#!/usr/bin/env python3
"""Shared paths/settings for the final emulandice-2050 + tension core chain.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from pathlib import Path
import os

HOME = Path.home()
PROJECT_DIR = Path(os.environ.get("SEJ_PROJECT_DIR", Path(__file__).resolve().parents[2])).expanduser().resolve()
DATA_DIR = Path(os.environ.get("SEJ_DATA_DIR", PROJECT_DIR / "data")).expanduser().resolve()
FACTS2_ROOT = Path(os.environ.get("SEJ_FACTS2_ROOT", HOME / "FACTS2")).expanduser().resolve()
CORE_OUTPUT_ROOT = Path(os.environ.get("SEJ_CORE_OUTPUT_ROOT", PROJECT_DIR / "output" / "CORE_METHOD_CHAIN_02I_B_FINAL_TENSION_P003_FAIR2")).expanduser().resolve()
SELECTION_OUTPUT_DIR = CORE_OUTPUT_ROOT / "01_warming_selection"
COUPLING_OUTPUT_DIR = CORE_OUTPUT_ROOT / "02_temporal_coupling"
CMIP6_ANNUAL_OUTPUT_DIR = CORE_OUTPUT_ROOT / "03_cmip6_annual"
CMIP7_ANNUAL_OUTPUT_DIR = CORE_OUTPUT_ROOT / "03B_cmip7_annual"
MANUSCRIPT_OUTPUT_DIR = CORE_OUTPUT_ROOT / "04_manuscript_tables"

for p in (
    CORE_OUTPUT_ROOT,
    SELECTION_OUTPUT_DIR,
    COUPLING_OUTPUT_DIR,
    CMIP6_ANNUAL_OUTPUT_DIR,
    CMIP7_ANNUAL_OUTPUT_DIR,
    MANUSCRIPT_OUTPUT_DIR,
):
    p.mkdir(parents=True, exist_ok=True)

CMIP6_SELECTION_FILE = SELECTION_OUTPUT_DIR / "fair_b25_baseline_scenario_selection.csv"
CMIP7_SELECTION_FILE = SELECTION_OUTPUT_DIR / "cmip7_fair_b25_baseline_scenario_selection.csv"
COUPLED_SEJ_FILE = COUPLING_OUTPUT_DIR / "SEJ25_emulandice2050_temporally_coupled_by_warming.csv"

SEJ_DATA_FILE = DATA_DIR / "sej2025_SplitPanels_liz_updated.csv"

CMIP6_SSPS = [
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp534-over",
    "ssp585",
]

FAIR2_ROOT = FACTS2_ROOT / "experiments" / "fair2"
CMIP6_FAIR_FILES = {
    "ssp119": FAIR2_ROOT / "ssp119_fair2" / "output" / "fair2-climate" / "gsat.nc",
    "ssp126": FAIR2_ROOT / "ssp126_fair2" / "output" / "fair2-climate" / "gsat.nc",
    "ssp245": FAIR2_ROOT / "ssp245_fair2" / "output" / "fair2-climate" / "gsat.nc",
    "ssp370": FAIR2_ROOT / "ssp370_fair2" / "output" / "fair2-climate" / "gsat.nc",
    "ssp534-over": FAIR2_ROOT / "ssp534_fair2" / "output" / "fair2-climate" / "gsat.nc",
    "ssp585": FAIR2_ROOT / "ssp585_fair2" / "output" / "fair2-climate" / "gsat.nc",
}

CMIP7_CODES = ["H", "HL", "L", "LN", "M", "ML", "VL"]

REFERENCE_START_YEAR = 1995
REFERENCE_END_YEAR = 2014
FIT_START_YEAR = 1979
FIT_END_YEAR = 2022
LAST_OBS_YEAR = 2023
INTERMEDIATE_KNOT_YEAR = 2050.0
TENSION_P = 0.03

# emulandice workflow-1e scenario proxies used to define warming-level 2050 knots.
EMULANDICE_EXPERIMENTS = {
    2.0: "emulandice1_ssp126",
    3.5: "emulandice1_ssp245",
    5.0: "emulandice1_ssp585",
}
EMULANDICE_ROOT = FACTS2_ROOT / "experiments"

# DP21 raw component ensembles are used directly so 2050-2300 EAIS and WAIS
# temporal correlations do not depend on the retired Nerem/quadratic coupling.
DP21_ROOT = FACTS2_ROOT / "data" / "module_specific_input_data" / "deconto21-ais"
DP21_SCENARIOS = {
    2.0: "rcp26",
    3.5: "rcp45",
    5.0: "rcp85",
}

GRIS_CORRELATION_DIR = (
    DATA_DIR / "temporal_correlation_matrices" / "GrIS_FittedISMIP_scenario_specific"
)
GRIS_CORRELATION_FILES = {
    2.0: GRIS_CORRELATION_DIR / "GrIS_FittedISMIP_ssp126_Pearson.csv",
    3.5: GRIS_CORRELATION_DIR / "GrIS_FittedISMIP_ssp245_Pearson.csv",
    5.0: GRIS_CORRELATION_DIR / "GrIS_FittedISMIP_ssp585_Pearson.csv",
}

COUPLING_YEARS = [2050, 2100, 2200, 2300]
SEJ_YEARS = [2100, 2200, 2300]
COUPLING_RNG_SEED = 1234
SAMPLE_RNG_SEED = 83126
CMIP7_CRN_SEED = 100626  # CMIP7 CRN ONLY. Do not change COUPLING_RNG_SEED or SAMPLE_RNG_SEED.
MATCH_BLOCK_SIZE = 64
MATCH_LOCAL_OFFSETS = [0, 32, 16, 48, 0, 32]
CORRELATION_EIGEN_FLOOR = 1e-4
MAXDET_PD_EPS = 1e-8
MAXDET_MAXITER = 6000

# Final preferred self-contained 3D balanced matching (02I-B).
STAGE02I_BLOCK_SIZE = 5000
STAGE02I_REFINE_ROUNDS = 30
STAGE02I_PROPOSALS_PER_YEAR_ROUND = 18000
STAGE02I_SWAP_BATCH = 1500
STAGE02I_SEED = 20261014
STAGE02I_ADJACENT_ERROR_ALLOWANCE = 0.015
