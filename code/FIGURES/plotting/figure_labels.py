"""Shared scientific names, scenario identifiers and reference values.

These definitions reproduce the existing plotting scripts. They do not implement
scenario assignment, change observational baselines, or compute any percentiles.
Keep distinct display conventions (e.g. 2°C versus 2.0°C) where needed.

Changing a reference value or unit factor here is a scientific change, NOT
merely a style change: check the analysis methods before editing it.
"""
from __future__ import annotations

# Physical components: Antarctic = EAIS + WAIS; total ice sheets = AIS + GrIS.
ICE_SHEET_NAMES = ("EAIS", "WAIS", "Total AIS", "GrIS", "Total ice-sheet SLR")
B25_ICE_SHEET_CODE_MAP = {"E": "EAIS", "W": "WAIS", "G": "GrIS", "SLR": "Total ice-sheet SLR"}
COMPONENT_LABELS = {"E": "EAIS", "W": "WAIS", "G": "GrIS", "AIS": "Total AIS", "TOTAL": "Total ice-sheet SLR"}
PROCESS_CODE_LABELS = {"A": "Accumulation", "D": "Discharge", "R": "Runoff", "TOTAL": "Total"}
PROCESS_ORDER = ("Accumulation", "Discharge", "Runoff", "Runoff minus accumulation")
PROCESS_ICE_SHEETS = ("EAIS", "WAIS", "GrIS")
VIOLIN_ICE_SHEET_LABELS = {"EAIS": "EAIS", "WAIS": "WAIS", "Total AIS": "AIS", "GrIS": "GrIS"}
VIOLIN_PROCESS_LABELS = {"Discharge": "Discharge", "Runoff minus accumulation": "Runoff − Accumulation", "Total ice-sheet contribution": "Total ice-sheet contribution"}

PROJECTION_YEARS = (2100, 2200, 2300)
WARMING_LEVELS = (2.0, 3.5, 5.0)
PERCENTILE_LEVELS = (5, 17, 50, 83, 95)
B25_YEAR_CODE_MAP = {"21": 2100, "22": 2200, "23": 2300}
B25_WARMING_CODE_MAP = {"2": 2.0, "35": 3.5, "5": 5.0}
WARMING_LABELS_DECIMAL = {2.0: "2.0°C", 3.5: "3.5°C", 5.0: "5.0°C"}
WARMING_LABELS_COMPACT = {2.0: "2°C", 3.5: "3.5°C", 5.0: "5°C"}
COMPACT_WARMING_TO_VALUE = {label: value for value, label in WARMING_LABELS_COMPACT.items()}
WARMING_SCENARIO_ORDER = ("LOW", "INTERMEDIATE", "HIGH")
WARMING_SCENARIO_CODES = dict(zip(WARMING_SCENARIO_ORDER, ("2", "35", "5")))
WARMING_SCENARIO_LABELS = dict(zip(WARMING_SCENARIO_ORDER, WARMING_LABELS_DECIMAL.values()))

SSP_ORDER = ("ssp119", "ssp126", "ssp245", "ssp370", "ssp585")
SSP_LABELS = {"ssp119": "SSP1-1.9", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
CMIP7_ORDER = ("L", "ML", "M", "H")   # Intentionally only four C2200-boundary pathways.
CMIP7_LABELS = {"L": "Low (L)", "ML": "Medium-to-Low (ML)", "M": "Medium (M)", "H": "High (H)"}
CLIMATE_SCENARIO_LABELS = {**SSP_LABELS, **CMIP7_LABELS}
CMIP6_SCENARIO_PAIRS = tuple((key, SSP_LABELS[key]) for key in SSP_ORDER)
CMIP7_SCENARIO_PAIRS = tuple((key, CMIP7_LABELS[key]) for key in CMIP7_ORDER)

MM_TO_CM = 0.1   # Only for inputs explicitly in millimetres.
MM_PER_CM = 10.0  # Use for division-based conversion to retain previous rounding.
M_TO_CM = 100.0  # Only for inputs explicitly in metres.
ANNUAL_YEARS = (2000, 2300)  # inclusive endpoints for precomputed trajectories

# The following were hardcoded in the uploaded manuscript scripts. Their
# provenance is the reference warming / cumulative-warming scenario assumptions;
# these are NOT parameters to infer from or refit to a new CMIP dataset.
WARMING_REFERENCE_YEARS = (2000, 2020, 2050, 2100, 2200, 2300)
WARMING_REFERENCE_TRAJECTORIES = {
    "Low": (0.7, 1.2, 1.7, 2.0, 2.0, 2.0),
    "Intermediate": (0.7, 1.2, 2.0, 3.5, 3.5, 3.5),
    "High": (0.7, 1.2, 2.5, 5.0, 5.0, 5.0),
}
REF_CUMULATIVE_WARMING = {
    2100: {"LOW": 155.0, "INTERMEDIATE": 204.5, "HIGH": 262.0},
    2200: {"LOW": 355.0, "INTERMEDIATE": 554.5, "HIGH": 762.0},
}
