"""Central configuration for final manuscript/SI table generation.

Both CMIP6 and CMIP7 now read from the FINAL production core chain:

    CORE_METHOD_CHAIN_02I_B_FINAL_TENSION_P003_FAIR2

Scientific production method:
- cumulative-warming selection at 2100/2200 with IDW p=1 and double-checkpoint truncation;
- component-resolved 12-variable EAIS/WAIS/GrIS temporal coupling;
- emulandice workflow-1e model-derived 2050 knots;
- classical spline under tension with p=0.03 and the observed 2023 endpoint rate;
- Total AIS and total ice sheet derived draw-by-draw.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from pathlib import Path
import os


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


PROJECT_DIR = _path_from_env(
    "TABLE_PROJECT_DIR",
    Path(os.environ.get("SEJ_PROJECT_DIR", Path(__file__).resolve().parents[3])),
)
OUTPUT_ROOT = PROJECT_DIR / "output"

# -----------------------------------------------------------------------------
# FINAL production core-chain input
# -----------------------------------------------------------------------------
FINAL_CORE_ROOT = _path_from_env(
    "TABLE_FINAL_CORE_ROOT",
    Path(os.environ.get("SEJ_CORE_OUTPUT_ROOT", OUTPUT_ROOT / "CORE_METHOD_CHAIN_02I_B_FINAL_TENSION_P003_FAIR2")),
)

CMIP6_SOURCE = _path_from_env(
    "TABLE_CMIP6_SOURCE",
    FINAL_CORE_ROOT
    / "03_cmip6_annual"
    / "tables"
    / "all_ice_sheet_components_SSP_projection_table_2020_2050_2100_2150_2200_2300.csv",
)

CMIP7_SOURCE = _path_from_env(
    "TABLE_CMIP7_SOURCE",
    FINAL_CORE_ROOT
    / "03B_cmip7_annual"
    / "tables"
    / "all_ice_sheet_components_CMIP7_projection_table_2020_2050_2100_2150_2200_2300.csv",
)

# -----------------------------------------------------------------------------
# Table output destination
# -----------------------------------------------------------------------------
TABLE_OUTPUT_ROOT = _path_from_env(
    "TABLE_OUTPUT_ROOT",
    OUTPUT_ROOT / "manuscript_tables_02I_B_final_tension_p003",
)
CMIP6_OUTPUT_DIR = TABLE_OUTPUT_ROOT / "cmip6_ssp"
CMIP7_OUTPUT_DIR = TABLE_OUTPUT_ROOT / "cmip7_pathways"

# -----------------------------------------------------------------------------
# Table contents
# -----------------------------------------------------------------------------
TABLE_YEARS = (2050, 2100, 2150, 2200, 2300)
COMPONENTS = ("EAIS", "WAIS", "GrIS", "Total AIS", "Total ice sheet")
QUANTILES = ("p05_cm", "p17_cm", "p50_cm", "p83_cm", "p95_cm")

CMIP6_CODES = (
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp534-over",
    "ssp585",
)
CMIP6_LABELS = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp534-over": "SSP5-3.4-OS",
    "ssp585": "SSP5-8.5",
}

# Preserve the user's established CMIP7 pathway ordering.
CMIP7_CODES = ("LN", "VL", "L", "ML", "M", "HL", "H")

# -----------------------------------------------------------------------------
# Production-method provenance / validation
# -----------------------------------------------------------------------------
COUPLING_METADATA_FILE = FINAL_CORE_ROOT / "02_temporal_coupling" / "method_metadata.csv"
EXPECTED_COUPLING_METHOD = "02I-B balanced 3D Hungarian 5000 + guarded weighted swaps 30"
EXPECTED_CMIP7_CRN_SEED = 100626
EXPECTED_CMIP7_SAMPLING_RULE = "CMIP7_CRN_shared_FaIR_sample_id_separate_warming_streams"
CMIP7_ANNUAL_NETCDF = FINAL_CORE_ROOT / "03B_cmip7_annual" / "annual_ice_sheet_trajectories_2000_2300.nc"
CMIP7_SAMPLED_CSV = FINAL_CORE_ROOT / "03B_cmip7_annual" / "sampled_sej2025_realizations.csv"

FINAL_SAMPLING_METHOD = (
    "range_normalized_checkpoint_euclidean_idw_p1_double_checkpoint_truncation"
)
FINAL_CHECKPOINTS = (2100, 2200)
FINAL_TRUNCATION_YEARS = (2100, 2200)
FINAL_TRUNCATION_RULE = "outside_same_direction_at_both_2100_and_2200"
FINAL_TENSION_P = 0.03
FINAL_REFERENCE_PERIOD = "1995-2014"

EXPECTED_SOURCE_BY_YEAR = {
    2050: "recoupled emulandice workflow-1e 2050 knot",
    2100: "sampled temporally coupled SEJ endpoint",
    2150: "tension-spline interpolation",
    2200: "sampled temporally coupled SEJ endpoint",
    2300: "sampled temporally coupled SEJ endpoint",
}

METHOD_NOTE_COMMON = (
    "Climate trajectories are mapped to the 2025 SEJ assessment using cumulative-"
    "warming checkpoints at 2100 and 2200, range-normalized Euclidean distance, "
    "inverse-distance weighting (p = 1), and double-checkpoint endpoint truncation. "
    "Ice-sheet trajectories use model-derived emulandice workflow-1e 2050 knots for "
    "EAIS, WAIS, and GrIS; component-resolved temporal coupling uses scenario-specific "
    "DP21 dependence for EAIS and WAIS and FittedISMIP dependence for GrIS, with "
    "unspecified cross-component/cross-time relationships completed by maximum entropy; "
    "the selected 02I-B method applies balanced three-component Hungarian rank matching "
    "in 5,000-row blocks with 30 rounds of priority-weighted refinement. "
    "Annual values from 2024 onward are interpolated with a classical spline under "
    "tension (p = 0.03) constrained by the observed 2023 endpoint rate. Contributions "
    "are in cm sea-level equivalent relative to the 1995–2014 baseline. Total AIS and "
    "total ice-sheet distributions are derived draw-by-draw from component realizations; "
    "their percentiles are therefore not obtained by summing component percentiles."
)

METHOD_NOTE_CMIP6 = (
    "CMIP6 FaIR2 trajectories: " + METHOD_NOTE_COMMON
)

METHOD_NOTE_CMIP7 = (
    "CMIP7 pathways (FaIR sample-ID-aligned common random numbers; "
    "independent LOW/INTERMEDIATE/HIGH streams; CRN seed 100626): " + METHOD_NOTE_COMMON
)
