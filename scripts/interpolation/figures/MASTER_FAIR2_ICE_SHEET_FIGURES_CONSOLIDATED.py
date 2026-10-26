#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
======================================================================
CONSOLIDATED FAIR2 ICE-SHEET FIGURE SUITE
======================================================================

Purpose
-------
This is the single active plotting entry point for the FaIR2 CMIP6/SSP results.

Input branches
--------------
CMIP6 / SSP:
    output/CORE_METHOD_CHAIN_EMULANDICE2050_TENSION_P003_FAIR2/

CMIP7:
    output/CORE_METHOD_CHAIN_EMULANDICE2050_TENSION_P003_FAIR2/

CMIP6 and CMIP7 use the emulandice-informed 2050, tension-p003 core outputs.
No Nerem/natural-cubic or common-random-number branch is used.

Figure groups
-------------
MANUSCRIPT
    - 3 x 3 GrIS / WAIS / EAIS warming-scenario grid
    - AR6 three-panel comparisons
    - CMIP6 SSP pathway projections
    - CMIP7 pathway projections

SUPPLEMENTARY
    - standalone Total AIS and total ice-sheet warming-scenario distributions
      (GrIS / WAIS / EAIS are omitted here because they already appear in the
      3 x 3 grid)
    - temporal-correlation figures and CSV
    - CMIP6 / CMIP7 realized-selection-fraction heat maps

DIAGNOSTIC
    - random individual trajectories for all five physical quantities

The top-level switches below select whole figure groups; smaller sub-switches
allow individual figure families to be disabled without editing plotting logic.

Outputs
-------
All outputs from this consolidated script are isolated under:

    output/manuscript_figures/emulandice2050_tension_p003/consolidated_final_suite/

with separate manuscript/, supplementary/, and diagnostics/ subdirectories.
This prevents overwriting both the pre-FaIR2 figures and earlier FaIR2 plotting
runs.

This script performs plotting/post-processing only. It does not modify the core
method, redraw SEJ endpoints, or resample the annual trajectories.
======================================================================

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""

from figure_labels import (
    MM_TO_CM as SHARED_MM_TO_CM,
    SSP_ORDER as SHARED_SSP_ORDER,
    SSP_LABELS as SHARED_SSP_LABELS,
    COMPONENT_LABELS,
    M_TO_CM,
    PERCENTILE_LEVELS,
    WARMING_SCENARIO_CODES,
    WARMING_SCENARIO_LABELS,
    WARMING_SCENARIO_ORDER,
    CLIMATE_SCENARIO_LABELS,
    CMIP7_ORDER,
    REF_CUMULATIVE_WARMING,
)

from figure_settings import PROJECT_ROOT, DATA_ROOT, UPSTREAM_ROOT, FIGURE_ROOT, output_for, DPI, SHOW_FIGURES, B25_MC_FILE, B19_ADJUSTED_FILE
from figure_settings import B19_SUMMARY_FILE, EXPERT_QUESTIONS_FILE, EXPERT_WORKBOOK_FILE, PERFORMANCE_FILE, AR6_WORKBOOK_FILE, COUPLED_SEJ_FILE
from figure_settings import FAIR2_CORE_METHOD_ROOT, FINAL_CORE_METHOD_ROOT, fair2_output_for
from figure_helpers import save_figure, maybe_show
from figure_style import (
    AR6_OUTER_COLOR, AR6_INNER_COLOR, AR6_MEDIAN_COLOR,
    DIFF_PERCENTILE_STYLES, STUDY_COMPONENT_COLORS,
    CLIMATE_SCENARIO_COLORS, REFERENCE_PATH_COLOR,
)

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import patheffects
from netCDF4 import Dataset
from matplotlib.ticker import MaxNLocator, MultipleLocator


# ============================================================
# 1. PATHS
# ============================================================

HOME = Path.home()

PROJECT_DIR = PROJECT_ROOT

DATA_DIR = DATA_ROOT

# Save the supplied IPCC AR6 workbook in data/ with this exact filename.
AR6_PROJECTION_FILE = AR6_WORKBOOK_FILE

# Read ONLY the FINAL core-chain annual trajectories generated after the
# double-checkpoint-truncation FaIR -> SEJ selection and 12-variable 02I-B coupling.
# Do not use the older C2200-only boundary sensitivity output.
EXPECTED_SELECTION_METHOD = (
    "range_normalized_checkpoint_euclidean_idw_p1_"
    "double_checkpoint_truncation"
)
EXPECTED_CHECKPOINTS = [2100, 2200]
EXPECTED_TRUNCATION_YEARS = [2100, 2200]
EXPECTED_TRUNCATION_RULE = "outside_same_direction_at_both_2100_and_2200"

# Both families now use the SAME scientific core/method configuration.
CMIP6_CORE_METHOD_ROOT = FAIR2_CORE_METHOD_ROOT
CMIP7_CORE_METHOD_ROOT = FAIR2_CORE_METHOD_ROOT

CMIP6_SELECTION_OUTPUT_DIR = CMIP6_CORE_METHOD_ROOT / "01_warming_selection"
CMIP7_SELECTION_OUTPUT_DIR = CMIP7_CORE_METHOD_ROOT / "01_warming_selection"
COUPLING_OUTPUT_DIR = CMIP6_CORE_METHOD_ROOT / "02_temporal_coupling"
METHOD_OUTPUT_DIR = CMIP6_CORE_METHOD_ROOT / "03_cmip6_annual"
CMIP7_METHOD_OUTPUT_DIR = CMIP7_CORE_METHOD_ROOT / "03B_cmip7_annual"

ANNUAL_TRAJECTORY_FILE = (
    METHOD_OUTPUT_DIR
    / "annual_ice_sheet_trajectories_2000_2300.nc"
)

SAMPLED_SEJ_FILE = (
    METHOD_OUTPUT_DIR
    / "sampled_sej2025_realizations.csv"
)

REBASED_IMBIE_FILE = (
    METHOD_OUTPUT_DIR
    / "imbie_observed_slr_1979_2023_monthly_rebased_to_1995_2014.csv"
)

# FINAL CMIP7 annual output from the same double-checkpoint / 12-variable 02I-B core chain.

CMIP7_ANNUAL_TRAJECTORY_FILE = (
    CMIP7_METHOD_OUTPUT_DIR
    / "annual_ice_sheet_trajectories_2000_2300.nc"
)

CMIP7_SAMPLED_SEJ_FILE = (
    CMIP7_METHOD_OUTPUT_DIR
    / "sampled_sej2025_realizations.csv"
)

CMIP7_REBASED_IMBIE_FILE = (
    CMIP7_METHOD_OUTPUT_DIR
    / "imbie_observed_slr_1979_2023_monthly_rebased_to_1995_2014.csv"
)

# Use the new emulandice-informed coupled endpoints, never the old SEJ file.
# Resolve by an unambiguous filename prefix to avoid fabricating a suffix.
SOURCE_SEJ_FILE = COUPLING_OUTPUT_DIR / "SEJ25_emulandice2050_temporally_coupled_by_warming.csv"


# Consolidated output root. This is intentionally different from both the
# pre-FaIR2 manuscript figure tree and the earlier FaIR2 plotting outputs.
SUITE_OUTPUT_ROOT = fair2_output_for("consolidated_final_suite")

MANUSCRIPT_OUTPUT_ROOT = SUITE_OUTPUT_ROOT / "manuscript"
SUPPLEMENTARY_OUTPUT_ROOT = SUITE_OUTPUT_ROOT / "supplementary"
DIAGNOSTIC_OUTPUT_ROOT = SUITE_OUTPUT_ROOT / "diagnostics"

# Keep FIGURE_DIR as a general suite-level alias for existing status messages.
FIGURE_DIR = SUITE_OUTPUT_ROOT

# Manuscript figures
GRID_FIGURE_DIR = MANUSCRIPT_OUTPUT_ROOT / "warming_scenario_grids"
AR6_COMPARISON_DIR = MANUSCRIPT_OUTPUT_ROOT / "ar6_comparisons"
SSP_FIGURE_DIR = MANUSCRIPT_OUTPUT_ROOT / "ssp_projections"
CMIP7_PATHWAY_FIGURE_DIR = MANUSCRIPT_OUTPUT_ROOT / "cmip7_pathway_projections"

# Supplementary figures
WARMING_FIGURE_DIR = SUPPLEMENTARY_OUTPUT_ROOT / "warming_scenarios"
CORRELATION_FIGURE_DIR = SUPPLEMENTARY_OUTPUT_ROOT / "correlations"
CORRELATION_CSV = CORRELATION_FIGURE_DIR / "temporal_correlation_through_time.csv"

# Diagnostic figures
RANDOM_TRAJECTORY_DIR = DIAGNOSTIC_OUTPUT_ROOT / "random_trajectories"

for directory in [
    SUITE_OUTPUT_ROOT,
    MANUSCRIPT_OUTPUT_ROOT,
    SUPPLEMENTARY_OUTPUT_ROOT,
    DIAGNOSTIC_OUTPUT_ROOT,
    WARMING_FIGURE_DIR,
    GRID_FIGURE_DIR,
    AR6_COMPARISON_DIR,
    SSP_FIGURE_DIR,
    CORRELATION_FIGURE_DIR,
    CMIP7_PATHWAY_FIGURE_DIR,
    RANDOM_TRAJECTORY_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. USER FIGURE-GROUP SWITCHES
# ============================================================
RUN_MANUSCRIPT_FIGURES = True
RUN_SUPPLEMENTARY_FIGURES = True
RUN_DIAGNOSTIC_FIGURES = False

# ------------------------------------------------------------
# Manuscript family switches
# ------------------------------------------------------------
MANUSCRIPT_MAKE_3X3_GRID = True
MANUSCRIPT_MAKE_AR6_COMPARISONS = True
MANUSCRIPT_MAKE_CMIP6_SSP_FIGURES = True
MANUSCRIPT_MAKE_CMIP7_PATHWAY_FIGURES = True

# ------------------------------------------------------------
# Supplementary family switches
# ------------------------------------------------------------
# Standalone warming plots are intentionally limited to Total AIS and
# Total ice sheet. GrIS / WAIS / EAIS already appear in the 3 x 3 grid.
SUPPLEMENT_MAKE_TOTAL_WARMING_DISTRIBUTIONS = True
SUPPLEMENT_MAKE_CORRELATION_FIGURES = True
SUPPLEMENT_MAKE_REALIZED_FRACTION_HEATMAPS = True

# ------------------------------------------------------------
# Diagnostic family switches
# ------------------------------------------------------------
DIAGNOSTIC_MAKE_RANDOM_TRAJECTORY_FIGURES = True
N_RANDOM_TRAJECTORIES = 100
RANDOM_TRAJECTORY_SEED = 1234

# Derived switches used by the existing plotting functions.
MAKE_WARMING_SCENARIO_FIGURES = (
    RUN_SUPPLEMENTARY_FIGURES
    and SUPPLEMENT_MAKE_TOTAL_WARMING_DISTRIBUTIONS
)
MAKE_3X3_GRID = RUN_MANUSCRIPT_FIGURES and MANUSCRIPT_MAKE_3X3_GRID
MAKE_AR6_COMPARISON_FIGURES = (
    RUN_MANUSCRIPT_FIGURES
    and MANUSCRIPT_MAKE_AR6_COMPARISONS
)
MAKE_SSP_FIGURES = (
    RUN_MANUSCRIPT_FIGURES
    and MANUSCRIPT_MAKE_CMIP6_SSP_FIGURES
)
MAKE_CORRELATION_FIGURES = (
    RUN_SUPPLEMENTARY_FIGURES
    and SUPPLEMENT_MAKE_CORRELATION_FIGURES
)
MAKE_RANDOM_TRAJECTORY_FIGURES = (
    RUN_DIAGNOSTIC_FIGURES
    and DIAGNOSTIC_MAKE_RANDOM_TRAJECTORY_FIGURES
)

# The old standalone warming-scenario figure had an extra guide at 2200.
SHOW_2200_GUIDE_IN_STANDALONE = True

# 2050 observationally constrained knot used by the natural cubic.
TRANSITION_YEAR = 2050
SHOW_2050_TRANSITION_GUIDE = False

# Correlation metric used in the displayed correlation figures.
# Both Pearson and Spearman are saved to the combined CSV.
CORRELATION_KIND = "pearson"
# options:
#   "pearson"
#   "spearman"


# ============================================================
# 3. LABELS / ORDER
# ============================================================

MM_TO_CM = SHARED_MM_TO_CM
AR6_M_TO_CM = M_TO_CM

# The AR6 workbook uses temperature-limited scenarios. For the dedicated
# comparison figure, match the SEJ LOW and HIGH scenarios to 2.0 C and 5.0 C.
AR6_SCENARIO_BY_SEJ = {
    "LOW": "tlim2.0win0.25",
    "HIGH": "tlim5.0win0.25",
}

AR6_SHEET_BY_VARIABLE = {
    "G": "GIS",
    "AIS": "AIS",
}

AR6_COMPARE_SCENARIOS = [
    "LOW",
    "HIGH",
]

AR6_COMPARE_VARIABLES = [
    "G",
    "AIS",
]

AR6_VARIABLE_FILE_LABELS = {
    "G": "GrIS",
    "AIS": "Total_AIS",
}

AR6_SCENARIO_FILE_LABELS = {
    "LOW": "2C",
    "HIGH": "5C",
}

AR6_XLIM = (2000, 2150)
AR6_XTICKS = [2000, 2050, 2100, 2150]

# ------------------------------------------------------------
# AR6 three-panel comparison styling
# ------------------------------------------------------------
# This study keeps the ice-sheet-specific colors, while AR6 is shown in
# grayscale so the two distributions remain visually distinct even though
# they are displayed in separate stacked panels.
AR6_OUTER_ALPHA = 0.28
AR6_INNER_ALPHA = 0.42
AR6_MEDIAN_LW = 2.4

# Percentile-specific styling for the difference panel.
# Lower-tail percentiles use blues, upper-tail percentiles use warm colors,
# and the median is black for emphasis.

PERCENTILES = list(PERCENTILE_LEVELS)

SCENARIO_ORDER = list(WARMING_SCENARIO_ORDER)

SCENARIO_CODES = dict(WARMING_SCENARIO_CODES)

SCENARIO_LABELS = dict(WARMING_SCENARIO_LABELS)

def canonicalize_cmip6_ssp(code):
    """Canonicalize the SSP5-3.4 overshoot identifier used by plotting."""
    code = str(code).strip()
    if code.lower() in {"ssp534", "ssp534-over", "ssp534_over", "ssp534over"}:
        return "ssp534-over"
    return code


# Final CMIP6 pathway set, including SSP5-3.4 overshoot.  Canonicalize any
# older shared-label spelling (e.g., ``ssp534``) to ``ssp534-over``.
SSP_ORDER = []
for _ssp in list(SHARED_SSP_ORDER):
    _canon = canonicalize_cmip6_ssp(_ssp)
    if _canon not in SSP_ORDER:
        SSP_ORDER.append(_canon)
if "ssp534-over" not in SSP_ORDER:
    _insert_at = SSP_ORDER.index("ssp585") if "ssp585" in SSP_ORDER else len(SSP_ORDER)
    SSP_ORDER.insert(_insert_at, "ssp534-over")

SSP_LABELS = {canonicalize_cmip6_ssp(k): v for k, v in dict(SHARED_SSP_LABELS).items()}
SSP_LABELS.setdefault("ssp534-over", "SSP5-3.4-OS")

# Full CMIP7 pathway set
CMIP7_ALL_SCENARIOS = ["VL", "LN",  "L", "ML", "M", "HL", "H"]

# Prefer shared descriptive labels when available, but never omit a pathway if
# figure_labels.py has not yet defined a friendly label for it.
CMIP7_ALL_LABELS = {
    code: dict(CLIMATE_SCENARIO_LABELS).get(code, f"CMIP7 {code}")
    for code in CMIP7_ALL_SCENARIOS
}

CMIP7_PATHWAY_VARIABLE_ORDER = ["G", "AIS", "TOTAL"]

VARIABLE_LABELS = dict(COMPONENT_LABELS)

VARIABLE_FILE_LABELS = {
    "E": "EAIS",
    "W": "WAIS",
    "G": "GrIS",
    "AIS": "AIS",
    "TOTAL": "Total_ice_sheet",
}

# Standalone distribution figures contain only quantities NOT already represented
# in the 3 x 3 GrIS / WAIS / EAIS grid. This avoids generating the same
# distribution twice within the consolidated suite.
STANDALONE_VARIABLE_ORDER = [
    "AIS",
    "TOTAL",
]

GRID_VARIABLE_ORDER = [
    "G",
    "W",
    "E",
]

# Diagnostics remain comprehensive even though the publication plots are deduplicated.
RANDOM_TRAJECTORY_VARIABLE_ORDER = [
    "G",
    "W",
    "E",
    "AIS",
    "TOTAL",
]

SSP_VARIABLE_ORDER = [
    "G",
    "W",
    'E',
    "AIS",
    'TOTAL'
]

CORRELATION_VARIABLE_ORDER = [
    "G",
    "AIS",
]

CORRELATION_TARGET_YEARS = [
    2200,
    2300,
]

# ------------------------------------------------------------
# Ice-sheet colors
# ------------------------------------------------------------
# Keep component identities consistent with the other SEJ2025 figures.

VARIABLE_STUDY_COLORS = STUDY_COMPONENT_COLORS

# Nested uncertainty intervals use the SAME component hue.
# Opacity distinguishes the wider and narrower ranges.
THIS_STUDY_OUTER_ALPHA = 0.18   # 5–95%
THIS_STUDY_INNER_ALPHA = 0.38   # 17–83%
THIS_STUDY_MEDIAN_LW = 2.4



# ============================================================
# MASTER SUITE SWITCHES
# ============================================================
# The consolidated master is the only active plotting entry point.
# These execution flags are derived from the user-facing group/sub-switches above.
RUN_FINAL_RESULTS_SUITE = any([
    MAKE_WARMING_SCENARIO_FIGURES,
    MAKE_3X3_GRID,
    MAKE_AR6_COMPARISON_FIGURES,
    MAKE_SSP_FIGURES,
    MAKE_CORRELATION_FIGURES,
    MAKE_RANDOM_TRAJECTORY_FIGURES,
])
RUN_CMIP7_PATHWAY_FIGURES = (
    RUN_MANUSCRIPT_FIGURES
    and MANUSCRIPT_MAKE_CMIP7_PATHWAY_FIGURES
)
RUN_REALIZED_FRACTION_HEATMAPS = (
    RUN_SUPPLEMENTARY_FIGURES
    and SUPPLEMENT_MAKE_REALIZED_FRACTION_HEATMAPS
)

# Heat-map family switches. These use already-generated scenario-selection CSVs.
HEATMAP_PLOT_CMIP6 = True
HEATMAP_PLOT_CMIP7 = True
HEATMAP_SHOW_SCENARIO_POINTS = False
HEATMAP_POINT_SIZE = 1.2
HEATMAP_POINT_ALPHA = 0.12

# Canonical FINAL core-chain selection outputs. Do not point these at the older
# older sensitivity folders.
HEATMAP_CMIP6_SELECTION_FILE = CMIP6_SELECTION_OUTPUT_DIR / "fair_b25_baseline_scenario_selection.csv"
HEATMAP_CMIP7_SELECTION_FILE = CMIP7_SELECTION_OUTPUT_DIR / "cmip7_fair_b25_baseline_scenario_selection.csv"

# Heat-map display/bin settings. The edges are forced to split exactly at
# normalized positions 0 and 1 so no bin straddles a truncation corner.
HEATMAP_X_LIM = (-0.6, 1.6)
HEATMAP_Y_LIM = (-0.6, 1.6)
HEATMAP_N_X_BELOW_0 = 8
HEATMAP_N_X_0_TO_1 = 16
HEATMAP_N_X_ABOVE_1 = 8
HEATMAP_N_Y_BELOW_0 = 8
HEATMAP_N_Y_0_TO_1 = 16
HEATMAP_N_Y_ABOVE_1 = 8
HEATMAP_MIN_COUNT_PER_BIN = 5
HEATMAP_FILL_LEVELS = np.linspace(0.0, 1.0, 21)
HEATMAP_LINE_LEVELS = np.arange(0.1, 1.0, 0.1)
HEATMAP_CMAP = "viridis"
HEATMAP_OUTPUT_DIR = (
    SUPPLEMENTARY_OUTPUT_ROOT
    / "realized_fractions"
    / "double_checkpoint_truncation"
)


def _parse_year_list(value):
    """Parse a JSON-style year list such as '[2100, 2200]'."""
    return [int(v) for v in json.loads(str(value))]


def validate_core_method_metadata(df, family, expected_pathways=None, require_idw_power=False):
    """Validate metadata shared by final sampled files and raw selection files."""
    required = {
        "sampling_method", "checkpoint_years", "truncation_years",
        "truncation_rule", "ssp", "selected_scenario",
        "reference_range_class", "selection_rule",
    }
    if require_idw_power:
        required.add("idw_power")

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{family}: missing core-method metadata: {missing}")

    if not df["sampling_method"].astype(str).eq(EXPECTED_SELECTION_METHOD).all():
        found = sorted(df["sampling_method"].astype(str).unique())
        raise ValueError(
            f"{family}: wrong selection method.\n"
            f"Expected: {EXPECTED_SELECTION_METHOD}\nFound: {found}"
        )

    checkpoints = df["checkpoint_years"].apply(_parse_year_list)
    if not checkpoints.apply(lambda x: x == EXPECTED_CHECKPOINTS).all():
        found = sorted({tuple(x) for x in checkpoints})
        raise ValueError(
            f"{family}: wrong checkpoints; expected {EXPECTED_CHECKPOINTS}, found {found}"
        )

    truncation_years = df["truncation_years"].apply(_parse_year_list)
    if not truncation_years.apply(lambda x: x == EXPECTED_TRUNCATION_YEARS).all():
        found = sorted({tuple(x) for x in truncation_years})
        raise ValueError(
            f"{family}: wrong truncation years; expected {EXPECTED_TRUNCATION_YEARS}, "
            f"found {found}"
        )

    if not df["truncation_rule"].astype(str).eq(EXPECTED_TRUNCATION_RULE).all():
        found = sorted(df["truncation_rule"].astype(str).unique())
        raise ValueError(
            f"{family}: wrong truncation rule.\n"
            f"Expected: {EXPECTED_TRUNCATION_RULE}\nFound: {found}"
        )

    if require_idw_power:
        if not np.allclose(pd.to_numeric(df["idw_power"], errors="raise"), 1.0):
            raise ValueError(f"{family}: unexpected inverse-distance weighting exponent")

    selected = df["selected_scenario"].astype(str).str.strip().str.upper()
    if not selected.isin(SCENARIO_ORDER).all():
        found = sorted(selected.unique())
        raise ValueError(f"{family}: unexpected selected_scenario values: {found}")

    if expected_pathways is not None:
        actual = set(df["ssp"].astype(str).str.strip())
        expected = set(expected_pathways)
        missing_pathways = sorted(expected - actual)
        extra_pathways = sorted(actual - expected)
        if missing_pathways or extra_pathways:
            raise ValueError(
                f"{family}: pathway mismatch; missing={missing_pathways}, extra={extra_pathways}"
            )

    expected_forced = {
        "BELOW_LOW_BOTH": ("LOW", "TRUNCATED_LOW_BOTH_CHECKPOINTS"),
        "ABOVE_HIGH_BOTH": ("HIGH", "TRUNCATED_HIGH_BOTH_CHECKPOINTS"),
    }
    for range_label, (scenario, rule) in expected_forced.items():
        mask = df["reference_range_class"].astype(str).eq(range_label)
        if not selected[mask].eq(scenario).all():
            raise ValueError(f"{family}: {range_label} rows are not always {scenario}")
        if not df.loc[mask, "selection_rule"].astype(str).eq(rule).all():
            raise ValueError(f"{family}: invalid selection_rule for {range_label}")

    interior = df["reference_range_class"].astype(str).eq("NOT_DOUBLE_OUTSIDE")
    if not df.loc[interior, "selection_rule"].astype(str).eq("PROBABILISTIC_IDW").all():
        raise ValueError(
            f"{family}: NOT_DOUBLE_OUTSIDE rows must remain PROBABILISTIC_IDW"
        )

    unexpected_classes = set(df["reference_range_class"].astype(str).unique()) - {
        "BELOW_LOW_BOTH", "NOT_DOUBLE_OUTSIDE", "ABOVE_HIGH_BOTH"
    }
    if unexpected_classes:
        raise ValueError(
            f"{family}: unexpected reference_range_class values: {sorted(unexpected_classes)}"
        )

    return df


def run_final_results_suite():
    """Run the enabled CMIP6-based manuscript, supplementary, and diagnostic families."""
    # ============================================================
    # 4. LOAD FINAL OUTPUTS
    # ============================================================

    # ------------------------------------------------------------
    # Load AR6 comparison data only when the dedicated comparison figure is
    # requested. The clean main figures do NOT use these tables.
    # ------------------------------------------------------------

    AR6_TABLES = {}

    if MAKE_AR6_COMPARISON_FIGURES:

        if not AR6_PROJECTION_FILE.exists():

            raise FileNotFoundError(
                "AR6 comparison figure is enabled, but the workbook was not found:\n"
                f"  {AR6_PROJECTION_FILE}\n\n"
                "Save the supplied workbook in the project data directory as "
                "'ipcc_ar6_sea_level_projection_global.xlsx', or set "
                "MAKE_AR6_COMPARISON_FIGURES = False."
            )

        AR6_TABLES = pd.read_excel(
            AR6_PROJECTION_FILE,
            sheet_name=list(
                AR6_SHEET_BY_VARIABLE.values()
            ),
        )

        print(
            "Loaded AR6 comparison workbook: "
            f"{AR6_PROJECTION_FILE}"
        )

    print("\n" + "=" * 72)
    print("LOADING FINAL METHOD OUTPUTS")
    print("=" * 72)

    # Fail early instead of quietly using an old or different method's results.
    for required_file in (SAMPLED_SEJ_FILE, ANNUAL_TRAJECTORY_FILE, REBASED_IMBIE_FILE):
        if not required_file.is_file():
            raise FileNotFoundError(
                f"Missing FINAL core-chain CMIP6 input: {required_file}\n"
                "Generate the emulandice2050_tension_p003 core outputs first; see README."
            )

    sampled_df = pd.read_csv(SAMPLED_SEJ_FILE)
    sampled_df["ssp"] = sampled_df["ssp"].map(canonicalize_cmip6_ssp)

    # Shared validator used by BOTH the final annual-results suite and the
    # CMIP6/CMIP7 realized-selection heat maps.
    validate_core_method_metadata(
        sampled_df,
        family="CMIP6 final sampled annual realizations",
        expected_pathways=SSP_ORDER,
        require_idw_power=True,
    )

    print(
        "Verified FINAL core method: [2100,2200] normalized checkpoints; "
        "IDW p=1; endpoint truncation only when outside in the same direction "
        "at BOTH checkpoints."
    )
    print("CMIP6 input:", METHOD_OUTPUT_DIR)
    print("AR6 figures will be saved to:", AR6_COMPARISON_DIR)

    if not SOURCE_SEJ_FILE.is_file():
        raise FileNotFoundError(
            f"Cannot uniquely identify new emulandice temporal-coupling file: {SOURCE_SEJ_FILE}\n"
            "Run the self-contained Stage 02 coupling first."
        )
    source_sej_df = pd.read_csv(SOURCE_SEJ_FILE)

    imbie_df = pd.read_csv(
        REBASED_IMBIE_FILE
    )

    imbie_df[
        "Date"
    ] = pd.to_datetime(
        imbie_df[
            "Date"
        ]
    )


    with Dataset(
        ANNUAL_TRAJECTORY_FILE,
        "r",
    ) as nc:

        years = np.asarray(
            nc.variables[
                "years"
            ][:],
            dtype=int,
        )

        E = np.asarray(
            nc.variables[
                "E"
            ][
                :,
                :,
            ],
            dtype=float,
        )

        W = np.asarray(
            nc.variables[
                "W"
            ][
                :,
                :,
            ],
            dtype=float,
        )

        G = np.asarray(
            nc.variables[
                "G"
            ][
                :,
                :,
            ],
            dtype=float,
        )

        # Keep the direct total trajectory as a diagnostic if present.
        DIRECT_TOTAL = (
            np.asarray(
                nc.variables[
                    "SLR"
                ][
                    :,
                    :,
                ],
                dtype=float,
            )
            if "SLR" in nc.variables
            else None
        )


    AIS = (
        E
        + W
    )

    TOTAL = (
        AIS
        + G
    )


    # Verify the 2005 -> 1995–2014 reference shift from FINAL 2100 hard knots.
    # The source SEJ endpoints are in mm relative to the 2005 zero;
    # the annual trajectories are in cm relative to the 1995–2014 zero.
    # Compare matched rows, not quantiles, to avoid conflating selection
    # differences with an additive reference-period shift.
    if 2100 not in years:
        raise ValueError("Final annual trajectories do not contain a 2100 hard knot")
    _i2100 = int(np.flatnonzero(years == 2100)[0])
    _scenario_codes = {
        "LOW": SCENARIO_CODES["LOW"],
        "INTERMEDIATE": SCENARIO_CODES["INTERMEDIATE"],
        "HIGH": SCENARIO_CODES["HIGH"],
    }
    _sample_ids = pd.to_numeric(sampled_df["sej2025_row_index"], errors="raise").to_numpy()
    if not np.all(np.isfinite(_sample_ids)) or not np.all(_sample_ids == np.floor(_sample_ids)):
        raise ValueError("Invalid SEJ row indices in final sampled CSV")
    _sample_ids = _sample_ids.astype(int)
    if np.any(_sample_ids < 0) or np.any(_sample_ids >= len(source_sej_df)):
        raise ValueError("SEJ row indices are outside the coupled source CSV; check index conventions")
    _selected = sampled_df["selected_scenario"].astype(str).str.strip().str.upper().to_numpy()
    _shifts = {}
    for _variable, _annual in (("E", E), ("W", W), ("G", G)):
        _src_mm = np.full(len(sampled_df), np.nan, dtype=float)
        for _scenario, _code in _scenario_codes.items():
            _mask = _selected == _scenario
            if not np.any(_mask):
                continue
            _col = f"{_variable}{_code}_21"
            if _col not in source_sej_df:
                raise ValueError(f"Missing coupled SEJ hard-knot column: {_col}")
            _src_mm[_mask] = pd.to_numeric(
                source_sej_df.iloc[_sample_ids[_mask]][_col], errors="raise"
            ).to_numpy(dtype=float)
        _offsets = _annual[:, _i2100] - _src_mm * MM_TO_CM
        if not np.all(np.isfinite(_offsets)):
            raise ValueError(f"Non-finite 2100 endpoint offsets for {_variable}")
        _shift = float(np.median(_offsets))
        _max_residual = float(np.max(np.abs(_offsets - _shift)))
        if _max_residual > 0.05:  # cm, rounding tolerance
            raise ValueError(
                f"{_variable}: final 2100 annual hard knots do not match sampled "
                f"SEJ endpoints plus a constant reference shift; "
                f"median offset={_shift:.5f} cm, max residual={_max_residual:.5f} cm. "
                "Check row indexing, component definitions, and unit conventions."
            )
        _shifts[_variable] = _shift
        print(f"Verified {_variable} reference-period shift: {_shift:+.5f} cm "
              f"(max 2100 residual {_max_residual:.6f} cm)")
    REFERENCE_SHIFTS_CM = {
        **_shifts,
        "AIS": _shifts["E"] + _shifts["W"],
        "SLR": _shifts["E"] + _shifts["W"] + _shifts["G"],
    }

    TRAJECTORIES = {
        "E": E,
        "W": W,
        "G": G,
        "AIS": AIS,
        "TOTAL": TOTAL,
    }


    if len(
        sampled_df
    ) != E.shape[0]:

        raise ValueError(
            "sampled_sej2025_realizations.csv and annual NetCDF "
            "do not contain the same number of realizations:\n"
            f"  sampled CSV: {len(sampled_df)}\n"
            f"  annual NetCDF: {E.shape[0]}"
        )


    required_sampled_columns = {
        "ssp",
        "sample_id",
        "selected_scenario",
        "sej2025_row_index",
    }

    missing_sampled_columns = (
        required_sampled_columns
        - set(
            sampled_df.columns
        )
    )

    if missing_sampled_columns:

        raise ValueError(
            "Sampled SEJ2025 file is missing expected columns: "
            f"{sorted(missing_sampled_columns)}"
        )


    print(
        f"Annual trajectories: {E.shape[0]:,} realizations x "
        f"{E.shape[1]} years"
    )

    print(
        f"Years: {years.min()}-{years.max()}"
    )

    # ------------------------------------------------------------
    # Check direct-total versus component-summed total.
    # ------------------------------------------------------------

    if DIRECT_TOTAL is not None:

        total_difference = np.abs(
            DIRECT_TOTAL
            - TOTAL
        )

        print(
            "\nDirect SLR vs. E + W + G:"
        )

        print(
            "  max absolute difference [cm]: "
            f"{np.nanmax(total_difference):.6e}"
        )

        print(
            "  mean absolute difference [cm]: "
            f"{np.nanmean(total_difference):.6e}"
        )


    # ============================================================
    # 5. IMBIE OBSERVATIONAL SERIES
    # ============================================================

    def decimal_year(dates):
        """
        Convert pandas datetime values to numeric decimal years.

        IMPORTANT:
        Projection trajectories use numeric years (2000...2300). Plotting
        pandas datetimes on the same axis causes Matplotlib to switch to a
        date axis, which can make the numeric projection years render as
        unexpected calendar dates. Converting IMBIE dates to decimal years
        keeps the entire x-axis in one numeric coordinate system.
        """

        dates = pd.DatetimeIndex(
            dates
        )

        year = dates.year.astype(
            float
        )

        start = pd.to_datetime(
            dates.year.astype(str)
            + "-01-01"
        )

        end = pd.to_datetime(
            (dates.year + 1).astype(str)
            + "-01-01"
        )

        fraction = (
            (dates - start).days.astype(float)
            / (end - start).days.astype(float)
        )

        return (
            year
            + fraction
        )



    required_imbie_columns = {
        "Date",
        "E_cm",
        "W_cm",
        "G_cm",
    }

    missing_imbie_columns = (
        required_imbie_columns
        - set(
            imbie_df.columns
        )
    )

    if missing_imbie_columns:

        raise ValueError(
            "1995–2014-rebased IMBIE file is missing expected columns: "
            f"{sorted(missing_imbie_columns)}"
        )


    IMBIE_SERIES = {
        "E": (
            imbie_df[
                "E_cm"
            ].to_numpy(
                dtype=float
            )
        ),

        "W": (
            imbie_df[
                "W_cm"
            ].to_numpy(
                dtype=float
            )
        ),

        "G": (
            imbie_df[
                "G_cm"
            ].to_numpy(
                dtype=float
            )
        ),
    }

    IMBIE_SERIES[
        "AIS"
    ] = (
        IMBIE_SERIES[
            "E"
        ]
        + IMBIE_SERIES[
            "W"
        ]
    )

    IMBIE_SERIES[
        "TOTAL"
    ] = (
        IMBIE_SERIES[
            "AIS"
        ]
        + IMBIE_SERIES[
            "G"
        ]
    )

    IMBIE_DATES = (
        imbie_df[
            "Date"
        ]
    )

    IMBIE_YEARS = decimal_year(
        IMBIE_DATES
    )

    FINAL_OBSERVATION_YEAR = float(
        IMBIE_YEARS[-1]
    )

    print(
        "IMBIE numeric plotting range: "
        f"{IMBIE_YEARS.min():.2f}-{IMBIE_YEARS.max():.2f}"
    )


    # ============================================================
    # 6. GENERIC HELPERS
    # ============================================================

    def get_study_color(variable):
        """
        Return the plotting color for one ice-sheet variable.
        """

        if variable not in VARIABLE_STUDY_COLORS:
            raise ValueError(
                f"No plotting color defined for variable: {variable}"
            )

        return VARIABLE_STUDY_COLORS[variable]


    def year_index(
        year,
    ):
        """
        Column index of one year in the annual trajectory arrays.
        """

        match = np.where(
            years
            == year
        )[0]

        if len(
            match
        ) != 1:

            raise ValueError(
                f"Could not locate unique year {year}."
            )

        return int(
            match[0]
        )


    def scenario_mask(
        scenario,
    ):
        """
        Rows whose upstream FaIR -> SEJ2025 selection chose one
        LOW / INTERMEDIATE / HIGH warming scenario.
        """

        return (
            sampled_df[
                "selected_scenario"
            ]
            .eq(
                scenario
            )
            .to_numpy()
        )


    def ssp_mask(
        ssp,
    ):
        """
        Rows belonging to one SSP.
        """

        return (
            sampled_df[
                "ssp"
            ]
            .eq(
                ssp
            )
            .to_numpy()
        )


    def distribution_quantiles(
        values,
    ):
        """
        Return P5 / P17 / P50 / P83 / P95 through time.
        """

        return np.percentile(
            values,
            PERCENTILES,
            axis=0,
        )


    def source_sej_endpoint_values(
        variable,
        scenario,
        year_code,
    ):
        """
        Original source SEJ2025 endpoint distribution in cm.

        These are the SOURCE temporally coupled SEJ2025 marginals, not
        the sampled FaIR-driven subset.

        AIS:
            E + W draw-by-draw

        TOTAL:
            use source SLR column if present; otherwise E + W + G
        """

        warming_code = (
            SCENARIO_CODES[
                scenario
            ]
        )

        if variable == "AIS":

            values_mm = (
                source_sej_df[
                    f"E{warming_code}_{year_code}"
                ].to_numpy(
                    dtype=float
                )
                + source_sej_df[
                    f"W{warming_code}_{year_code}"
                ].to_numpy(
                    dtype=float
                )
            )

        elif variable == "TOTAL":

            direct_column = (
                f"SLR{warming_code}_{year_code}"
            )

            if direct_column in source_sej_df.columns:

                values_mm = (
                    source_sej_df[
                        direct_column
                    ].to_numpy(
                        dtype=float
                    )
                )

            else:

                values_mm = (
                    source_sej_df[
                        f"E{warming_code}_{year_code}"
                    ].to_numpy(
                        dtype=float
                    )
                    + source_sej_df[
                        f"W{warming_code}_{year_code}"
                    ].to_numpy(
                        dtype=float
                    )
                    + source_sej_df[
                        f"G{warming_code}_{year_code}"
                    ].to_numpy(
                        dtype=float
                    )
                )

        else:

            column = (
                f"{variable}{warming_code}_{year_code}"
            )

            if column not in source_sej_df.columns:

                raise ValueError(
                    f"Missing source SEJ2025 column: {column}"
                )

            values_mm = (
                source_sej_df[
                    column
                ].to_numpy(
                    dtype=float
                )
            )

        shift_key = "SLR" if variable == "TOTAL" else variable
        if shift_key not in REFERENCE_SHIFTS_CM:
            raise ValueError(f"Missing reference-period shift for {shift_key}")

        return (
            values_mm
            * MM_TO_CM
            + float(REFERENCE_SHIFTS_CM[shift_key])
        )


    def add_original_sej_endpoint_markers(
        ax,
        variable,
        scenario,
        max_year=None,
    ):
        """
        Add original source SEJ2025 P5/P17/P83/P95 as open circles
        and P50 as a filled black diamond at 2100/2200/2300.

        If max_year is supplied, only endpoints at or before max_year are
        plotted. This is used for the dedicated AR6 comparison figures,
        which end at 2150.
        """

        for endpoint_year, year_code in [
            (2100, "21"),
            (2200, "22"),
            (2300, "23"),
        ]:

            if (
                max_year is not None
                and endpoint_year > max_year
            ):
                continue

            values = source_sej_endpoint_values(
                variable=variable,
                scenario=scenario,
                year_code=year_code,
            )

            q05, q17, q50, q83, q95 = np.percentile(
                values,
                PERCENTILES,
            )

            ax.scatter(
                [
                    endpoint_year,
                    endpoint_year,
                    endpoint_year,
                    endpoint_year,
                ],
                [
                    q05,
                    q17,
                    q83,
                    q95,
                ],
                marker="o",
                s=38,
                facecolors="white",
                edgecolors="black",
                linewidths=1.2,
                zorder=8,
            )

            ax.scatter(
                endpoint_year,
                q50,
                marker="D",
                s=48,
                facecolors="black",
                edgecolors="black",
                linewidths=1.0,
                zorder=9,
            )


    def get_ar6_projection(
        variable,
        scenario,
    ):
        """
        Return AR6 P5/P17/P50/P83/P95 trajectories for GrIS or AIS.

        The supplied workbook reports values in meters and uses the scenario
        identifiers tlim2.0win0.25 and tlim5.0win0.25 for the comparisons used
        here. Values are converted to centimeters before plotting.
        """

        if scenario not in AR6_SCENARIO_BY_SEJ:
            return None

        if variable not in AR6_SHEET_BY_VARIABLE:
            return None

        sheet_name = AR6_SHEET_BY_VARIABLE[
            variable
        ]

        ar6_scenario = AR6_SCENARIO_BY_SEJ[
            scenario
        ]

        table = AR6_TABLES[
            sheet_name
        ].copy()

        subset = table[
            table[
                "scenario"
            ].eq(
                ar6_scenario
            )
        ].copy()

        if "confidence" in subset.columns:
            subset = subset[
                subset[
                    "confidence"
                ].astype(str).str.lower().eq(
                    "medium"
                )
            ].copy()

        if subset.empty:
            raise ValueError(
                f"No AR6 rows found for {variable=} / {scenario=} "
                f"({sheet_name=}, {ar6_scenario=})."
            )

        year_columns = sorted(
            [
                column
                for column in subset.columns
                if isinstance(
                    column,
                    (int, np.integer),
                )
            ]
        )

        if not year_columns:
            raise ValueError(
                f"Could not identify numeric AR6 year columns in sheet {sheet_name}."
            )

        quantile_curves = {}

        for percentile in PERCENTILES:

            row = subset[
                subset[
                    "quantile"
                ].eq(
                    percentile
                )
            ]

            if len(row) != 1:
                raise ValueError(
                    f"Expected exactly one AR6 row for {variable}, {scenario}, "
                    f"quantile {percentile}; found {len(row)}."
                )

            quantile_curves[
                percentile
            ] = (
                row[
                    year_columns
                ]
                .iloc[0]
                .to_numpy(
                    dtype=float
                )
                * AR6_M_TO_CM
            )

        return {
            "years": np.asarray(
                year_columns,
                dtype=float,
            ),
            "quantiles": quantile_curves,
        }


    def plot_ar6_distribution_panel(
        ax,
        variable,
        scenario,
    ):
        """
        Plot the AR6 P5-P95 and P17-P83 uncertainty ranges plus median in a
        standalone panel. Because AR6 is no longer overlaid on this study, the
        same ice-sheet color can be used cleanly in both panels.
        """

        ar6 = get_ar6_projection(
            variable=variable,
            scenario=scenario,
        )

        if ar6 is None:
            return None

        ar6_years = ar6["years"]
        q = ar6["quantiles"]

        ax.fill_between(
            ar6_years,
            q[5],
            q[95],
            color=AR6_OUTER_COLOR,
            alpha=AR6_OUTER_ALPHA,
            label="AR6 5–95%",
            zorder=1,
        )

        ax.fill_between(
            ar6_years,
            q[17],
            q[83],
            color=AR6_INNER_COLOR,
            alpha=AR6_INNER_ALPHA,
            label="AR6 17–83%",
            zorder=2,
        )

        ax.plot(
            ar6_years,
            q[50],
            color=AR6_MEDIAN_COLOR,
            linewidth=AR6_MEDIAN_LW,
            label="AR6 median",
            zorder=3,
        )

        ax.axhline(
            0,
            color="0.50",
            linewidth=0.8,
            linestyle="--",
            alpha=0.35,
        )

        ax.grid(alpha=0.4, linestyle = '--')

        return ar6


    def study_quantiles_at_years(
        values,
        target_years,
    ):
        """
        Return this-study P5/P17/P50/P83/P95 at the requested years.

        The annual trajectory file contains integer years, while the AR6 workbook
        reports decadal values from 2020 through 2150. Differences are therefore
        calculated only at those common reported years; no interpolation of AR6
        percentiles is introduced.
        """

        result = {}

        for percentile in PERCENTILES:
            result[percentile] = []

        for target_year in target_years:

            idx = year_index(
                int(target_year)
            )

            q = np.percentile(
                values[:, idx],
                PERCENTILES,
            )

            for percentile, value in zip(PERCENTILES, q):
                result[percentile].append(float(value))

        for percentile in PERCENTILES:
            result[percentile] = np.asarray(
                result[percentile],
                dtype=float,
            )

        return result


    def plot_percentile_difference_panel(
        ax,
        values,
        ar6,
    ):
        """
        Plot this study minus AR6 for all five reported percentiles.

        Positive values mean this study projects a larger SLR contribution than
        AR6 at that percentile; negative values mean this study projects a smaller
        contribution.
        """

        ar6_years = ar6["years"]
        ar6_q = ar6["quantiles"]

        study_q = study_quantiles_at_years(
            values=values,
            target_years=ar6_years,
        )

        for percentile in PERCENTILES:

            difference = (
                study_q[percentile]
                - ar6_q[percentile]
            )

            style = DIFF_PERCENTILE_STYLES[percentile]

            ax.plot(
                ar6_years,
                difference,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=f"P{percentile}",
                zorder=3 if percentile == 50 else 2,
            )

        ax.axhline(
            0,
            color="0.35",
            linewidth=1.0,
            linestyle="-",
            alpha=0.75,
            zorder=1,
        )

        ax.grid(alpha=0.4, linestyle = '--')


    def study_panel_legend_handles(variable):
        """Legend for the top panel of the three-panel comparison figure."""

        study_color = get_study_color(variable)

        return [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.4,
                label="IMBIE observations",
            ),
            Patch(
                facecolor=study_color,
                alpha=THIS_STUDY_OUTER_ALPHA,
                label="2025 SEJ 5–95%",
            ),
            Patch(
                facecolor=study_color,
                alpha=THIS_STUDY_INNER_ALPHA,
                label="2025 SEJ 17–83%",
            ),
            Line2D(
                [0],
                [0],
                color=study_color,
                linewidth=THIS_STUDY_MEDIAN_LW,
                label="2025 SEJ median",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize=6,
                label="Original SEJ2025 percentiles",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=6,
                label="Original SEJ2025 median",
            ),
        ]


    def ar6_panel_legend_handles(variable=None):
        """Legend for the middle AR6 panel."""

        return [
            Patch(
                facecolor=AR6_OUTER_COLOR,
                alpha=AR6_OUTER_ALPHA,
                label="AR6 5–95%",
            ),
            Patch(
                facecolor=AR6_INNER_COLOR,
                alpha=AR6_INNER_ALPHA,
                label="AR6 17–83%",
            ),
            Line2D(
                [0],
                [0],
                color=AR6_MEDIAN_COLOR,
                linewidth=AR6_MEDIAN_LW,
                label="AR6 median",
            ),
        ]

    def plot_imbie(
        ax,
        variable,
        linewidth=1.4,
    ):
        """
        Plot monthly IMBIE observations.
        """

        ax.plot(
            IMBIE_YEARS,
            IMBIE_SERIES[
                variable
            ],
            color="black",
            linewidth=linewidth,
            zorder=7,
            label="IMBIE observations",
        )


    def plot_projection_distribution(
        ax,
        values,
        variable,
        scenario=None,
        include_original_endpoints=False,
        include_imbie=True,
        show_observation_end=True,
        show_zero_line=True,
        year_window=None,
    ):
        """
        Shared uncertainty-distribution plotting helper.

        values:
            realizations x years

        year_window:
            Optional (start_year, end_year) tuple. When supplied, only that
            portion of the projection trajectory is drawn. This lets the
            dedicated AR6 comparison figures autoscale their y-axes using
            only the visible 2000–2150 data rather than hidden 2200/2300
            projection values.

        The 5–95%, 17–83%, and median all use the same ice-sheet hue, with
        opacity distinguishing the nested uncertainty intervals.
        """

        q05, q17, q50, q83, q95 = (
            distribution_quantiles(
                values
            )
        )

        plot_years = years
        plot_q05 = q05
        plot_q17 = q17
        plot_q50 = q50
        plot_q83 = q83
        plot_q95 = q95

        endpoint_max_year = None

        if year_window is not None:

            start_year, end_year = year_window

            year_mask = (
                (years >= start_year)
                & (years <= end_year)
            )

            if not year_mask.any():
                raise ValueError(
                    f"No trajectory years fall inside {year_window}."
                )

            plot_years = years[year_mask]
            plot_q05 = q05[year_mask]
            plot_q17 = q17[year_mask]
            plot_q50 = q50[year_mask]
            plot_q83 = q83[year_mask]
            plot_q95 = q95[year_mask]

            endpoint_max_year = end_year

        study_color = get_study_color(variable)

        if include_imbie:

            plot_imbie(
                ax,
                variable,
            )

        ax.fill_between(
            plot_years,
            plot_q05,
            plot_q95,
            color=study_color,
            alpha=THIS_STUDY_OUTER_ALPHA,
            label="Projection 5–95%",
            zorder=1,
        )

        ax.fill_between(
            plot_years,
            plot_q17,
            plot_q83,
            color=study_color,
            alpha=THIS_STUDY_INNER_ALPHA,
            label="Projection 17–83%",
            zorder=2,
        )

        ax.plot(
            plot_years,
            plot_q50,
            color=study_color,
            linewidth=THIS_STUDY_MEDIAN_LW,
            label="Projection median",
            zorder=3,
        )

        if (
            include_original_endpoints
            and scenario is not None
        ):

            add_original_sej_endpoint_markers(
                ax=ax,
                variable=variable,
                scenario=scenario,
                max_year=endpoint_max_year,
            )

        if show_observation_end:

            ax.axvline(
                FINAL_OBSERVATION_YEAR,
                color="0.55",
                linewidth=1.0,
                linestyle="--",
                alpha=0.55,
            )

        if SHOW_2050_TRANSITION_GUIDE:

            ax.axvline(
                TRANSITION_YEAR,
                color="0.35",
                linewidth=1.0,
                linestyle=":",
                alpha=0.65,
                zorder=6,
            )

        if show_zero_line:

            ax.axhline(
                0,
                color="0.50",
                linewidth=0.8,
                linestyle="--",
                alpha=0.35,
            )

        ax.grid(alpha=0.4, linestyle = '--')


    def standard_endpoint_legend_handles(variable=None):
        """
        Handles shared by warming-scenario figures.

        If variable is supplied, the legend uses that ice-sheet color. For
        mixed-component figures (e.g., the 3 x 3 grid), use neutral gray so
        the legend communicates interval style rather than one component color.
        """

        if variable is None:
            study_color = "0.35"
        else:
            study_color = get_study_color(variable)

        return [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.4,
                label="IMBIE observations",
            ),

            Patch(
                facecolor=study_color,
                alpha=THIS_STUDY_OUTER_ALPHA,
                label="Projection 5–95%",
            ),

            Patch(
                facecolor=study_color,
                alpha=THIS_STUDY_INNER_ALPHA,
                label="Projection 17–83%",
            ),

            Line2D(
                [0],
                [0],
                color=study_color,
                linewidth=THIS_STUDY_MEDIAN_LW,
                label="Projection median",
            ),

            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize=6,
                label="Original SEJ2025 percentiles",
            ),

            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=6,
                label="Original SEJ2025 median",
            ),
        ]


    def ssp_legend_handles(variable):
        """
        Legend for the SSP projection figures.
        """

        study_color = get_study_color(variable)

        return [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.4,
                label="IMBIE observations",
            ),

            Patch(
                facecolor=study_color,
                alpha=THIS_STUDY_OUTER_ALPHA,
                label="Projection 5–95%",
            ),

            Patch(
                facecolor=study_color,
                alpha=THIS_STUDY_INNER_ALPHA,
                label="Projection 17–83%",
            ),

            Line2D(
                [0],
                [0],
                color=study_color,
                linewidth=THIS_STUDY_MEDIAN_LW,
                label="Projection median",
            ),
        ]


    def safe_pearson(
        x,
        y,
    ):
        """
        Pearson correlation with finite-value / zero-variance checks.
        """

        x = np.asarray(
            x,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        finite = (
            np.isfinite(
                x
            )
            & np.isfinite(
                y
            )
        )

        x = x[
            finite
        ]

        y = y[
            finite
        ]

        if (
            len(
                x
            ) < 3
            or np.std(
                x
            ) == 0
            or np.std(
                y
            ) == 0
        ):

            return np.nan

        return float(
            np.corrcoef(
                x,
                y,
            )[0, 1]
        )


    def safe_spearman(
        x,
        y,
    ):
        """
        Spearman correlation with finite-value / zero-variance checks.
        """

        x = np.asarray(
            x,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        finite = (
            np.isfinite(
                x
            )
            & np.isfinite(
                y
            )
        )

        x = x[
            finite
        ]

        y = y[
            finite
        ]

        if (
            len(
                x
            ) < 3
            or np.std(
                x
            ) == 0
            or np.std(
                y
            ) == 0
        ):

            return np.nan

        return float(
            pd.Series(
                x
            ).corr(
                pd.Series(
                    y
                ),
                method="spearman",
            )
        )


    # ============================================================
    # 7. WARMING-SCENARIO DISTRIBUTION FIGURES
    # ============================================================

    if MAKE_WARMING_SCENARIO_FIGURES:

        print("\n" + "=" * 72)
        print("CREATING WARMING-SCENARIO DISTRIBUTION FIGURES")
        print("=" * 72)

        for variable in STANDALONE_VARIABLE_ORDER:

            variable_dir = (
                WARMING_FIGURE_DIR
                / VARIABLE_FILE_LABELS[
                    variable
                ]
            )

            variable_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for scenario in SCENARIO_ORDER:

                mask = scenario_mask(
                    scenario
                )

                if not mask.any():

                    print(
                        f"WARNING: no {scenario} trajectories; skipping."
                    )

                    continue

                values = (
                    TRAJECTORIES[
                        variable
                    ][
                        mask,
                        :,
                    ]
                )

                fig, ax = plt.subplots(
                    figsize=(
                        9,
                        5.4,
                    )
                )

                plot_projection_distribution(
                    ax=ax,
                    values=values,
                    variable=variable,
                    scenario=scenario,
                    include_original_endpoints=True,
                    include_imbie=True,
                )

                if SHOW_2200_GUIDE_IN_STANDALONE:

                    ax.axvline(
                        2200,
                        linewidth=1.0,
                        linestyle="--",
                        alpha=0.30,
                    )

                ax.set_xlim(
                    2000,
                    2300,
                )

                ax.set_xticks(
                    [
                        2000,
                        2050,
                        2100,
                        2150,
                        2200,
                        2250,
                        2300,
                    ]
                )

                ax.set_xlabel(
                    "Year"
                )

                ax.set_ylabel(
                    (
                        f"{VARIABLE_LABELS[variable]}\n"
                        "SLR contribution (cm)"
                    )
                )

                ax.set_title(
                    SCENARIO_LABELS[
                        scenario
                    ],
                    fontsize=15,
                )

                ax.legend(
                    handles=standard_endpoint_legend_handles(variable),
                    frameon=False,
                    loc="upper left",
                )

                fig.tight_layout()

                outfile = (
                    variable_dir
                    / (
                        f"{VARIABLE_FILE_LABELS[variable]}_"
                        f"distribution_{scenario}.png"
                    )
                )

                save_figure(fig, outfile)

                plt.close(
                    fig
                )


    # ============================================================
    # 8. RANDOM INDIVIDUAL TRAJECTORY FIGURES
    # ============================================================

    if MAKE_RANDOM_TRAJECTORY_FIGURES:

        print("\n" + "=" * 72)
        print("CREATING RANDOM INDIVIDUAL TRAJECTORY FIGURES")
        print("=" * 72)

        def random_unique_scenario_indices(scenario, n=N_RANDOM_TRAJECTORIES):
            """Choose reproducible, non-duplicate coupled trajectories where possible."""
            mask = scenario_mask(scenario)
            candidates = np.flatnonzero(mask)
            if len(candidates) == 0:
                raise ValueError(f"No trajectories found for {scenario}")

            if "coupled_trajectory_id" in sampled_df.columns:
                physical_ids = sampled_df.loc[mask, "coupled_trajectory_id"].to_numpy()
            else:
                physical_ids = sampled_df.loc[mask, "sej2025_row_index"].to_numpy()

            _, first = np.unique(physical_ids, return_index=True)
            candidates = candidates[np.sort(first)]

            # Scenario-specific deterministic seed so LOW/INT/HIGH are reproducible
            # but do not all draw the same position sequence.
            scenario_offset = SCENARIO_ORDER.index(scenario) * 10000
            rng = np.random.default_rng(RANDOM_TRAJECTORY_SEED + scenario_offset)
            n_take = min(int(n), len(candidates))
            return rng.choice(candidates, size=n_take, replace=False)

        # Select once per warming scenario, then reuse the SAME realization rows
        # for E/W/G/AIS/TOTAL to preserve draw-by-draw physical correspondence.
        random_indices = {
            scenario: random_unique_scenario_indices(scenario)
            for scenario in SCENARIO_ORDER
        }

        future = years >= 2024

        for variable in RANDOM_TRAJECTORY_VARIABLE_ORDER:
            fig, axes = plt.subplots(
                1, 3, figsize=(14.5, 4.8), sharex=True, sharey=True
            )
            color = get_study_color(variable)

            for ax, scenario in zip(axes, SCENARIO_ORDER):
                idx = random_indices[scenario]

                # Individual realization trajectories.
                for i in idx:
                    ax.plot(
                        years[future],
                        TRAJECTORIES[variable][i, future],
                        color=color,
                        linewidth=0.7,
                        alpha=0.12,
                    )

                # Median of the FULL warming-scenario ensemble for orientation.
                values = TRAJECTORIES[variable][scenario_mask(scenario), :]
                median = np.median(values, axis=0)
                ax.plot(
                    years[future],
                    median[future],
                    color=color,
                    linewidth=2.4,
                    label="Projection median",
                )

                # Observations through 2023.
                ax.plot(
                    IMBIE_YEARS,
                    IMBIE_SERIES[variable],
                    color="black",
                    linewidth=1.4,
                    label="IMBIE observations",
                )

                ax.axvline(2050, linewidth=0.9, linestyle="--", alpha=0.25)
                ax.set_xlim(2000, 2300)
                ax.set_xticks([2000, 2050, 2100, 2200, 2300])
                ax.set_xlabel("Year")
                ax.set_title(SCENARIO_LABELS[scenario], fontsize=13)

            axes[0].set_ylabel(
                f"{VARIABLE_LABELS[variable]}\nSLR contribution (cm)"
            )
            axes[0].legend(frameon=False, loc="upper left")
            fig.suptitle(
                f"{VARIABLE_LABELS[variable]}: {N_RANDOM_TRAJECTORIES} random trajectories",
                fontsize=15,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            outfile = (
                RANDOM_TRAJECTORY_DIR
                / f"{VARIABLE_FILE_LABELS[variable]}_{N_RANDOM_TRAJECTORIES}_random_trajectories.png"
            )
            save_figure(fig, outfile)
            plt.close(fig)


    # ============================================================
    # 9. 3 x 3 WARMING-SCENARIO GRID
    # ============================================================

    if MAKE_3X3_GRID:

        print("\n" + "=" * 72)
        print("CREATING 3 x 3 WARMING-SCENARIO GRID")
        print("=" * 72)

        fig, axes = plt.subplots(
            nrows=3,
            ncols=3,
            figsize=(
                13,
                13,
            ),
            sharex=True,
            sharey="row",
        )

        for row_i, variable in enumerate(
            GRID_VARIABLE_ORDER
        ):

            for col_i, scenario in enumerate(
                SCENARIO_ORDER
            ):

                ax = axes[
                    row_i,
                    col_i,
                ]

                mask = scenario_mask(
                    scenario
                )

                values = (
                    TRAJECTORIES[
                        variable
                    ][
                        mask,
                        :,
                    ]
                )

                plot_projection_distribution(
                    ax=ax,
                    values=values,
                    variable=variable,
                    scenario=scenario,
                    include_original_endpoints=True,
                    include_imbie=True,
                )

                ax.set_xlim(
                    2000,
                    2300,
                )

                ax.set_xticks(
                    [
                        2000,
                        2050,
                        2100,
                        2150,
                        2200,
                        2250,
                        2300,
                    ]
                )

                if row_i == 0:

                    ax.set_title(
                        SCENARIO_LABELS[
                            scenario
                        ],
                        fontsize=13,
                    )

                if col_i == 0:

                    ax.set_ylabel(
                        (
                            f"{VARIABLE_LABELS[variable]}\n"
                            "SLR contribution (cm)"
                        )
                    )

                if row_i == (
                    len(
                        GRID_VARIABLE_ORDER
                    )
                    - 1
                ):

                    ax.set_xlabel(
                        "Year"
                    )


        fig.legend(
            handles=standard_endpoint_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                0.99,
            ),
            ncol=3,
            frameon=False,
        )

        fig.suptitle(
            "Ice-sheet SLR through time by warming scenario",
            fontsize=16,
            y=1.015,
        )

        fig.tight_layout(
            rect=[
                0,
                0,
                1,
                0.94,
            ]
        )

        outfile = (
            GRID_FIGURE_DIR
            / "ice_sheet_3x3_by_warming_scenario.png"
        )

        save_figure(fig, outfile)

        plt.close(
            fig
        )


    # ============================================================
    # 10. COMMON-AXIS AR6 COMPARISON FIGURES
    # ============================================================

    if MAKE_AR6_COMPARISON_FIGURES:

        print("\n" + "=" * 72)
        print("CREATING COMMON-AXIS AR6 COMPARISON FIGURES")
        print("=" * 72)

        # --------------------------------------------------------
        # Styling
        # --------------------------------------------------------
        # 2025 SEJ:
        #   - filled 5–95% and 17–83% uncertainty ranges
        #   - solid colored median
        #
        # AR6:
        #   - gray percentile lines only
        #   - dotted = 5th / 95th percentiles
        #   - dashed = 17th / 83rd percentiles
        #   - solid = median
        # --------------------------------------------------------

        AR6_LINE_COLOR = "0.40"

        STUDY_MEDIAN_LW = 2.6

        AR6_MEDIAN_LW = 2.3
        AR6_INNER_LW = 1.6
        AR6_OUTER_LW = 1.4

        for variable in AR6_COMPARE_VARIABLES:

            study_color = get_study_color(variable)

            for scenario in AR6_COMPARE_SCENARIOS:

                # ------------------------------------------------
                # 2025 SEJ trajectories
                # ------------------------------------------------
                mask = scenario_mask(scenario)

                if not mask.any():
                    raise ValueError(
                        f"No sampled trajectories found for {scenario}."
                    )

                values = TRAJECTORIES[variable][mask, :]

                study_q = dict(
                    zip(
                        PERCENTILES,
                        np.percentile(
                            values,
                            PERCENTILES,
                            axis=0,
                        ),
                    )
                )

                # ------------------------------------------------
                # AR6 trajectories
                # ------------------------------------------------
                ar6 = get_ar6_projection(
                    variable=variable,
                    scenario=scenario,
                )

                if ar6 is None:
                    raise ValueError(
                        f"No AR6 comparison data found for "
                        f"{variable=} / {scenario=}."
                    )

                ar6_years = ar6["years"]
                ar6_q = ar6["quantiles"]

                # ------------------------------------------------
                # Create common-axis figure
                # ------------------------------------------------
                fig, ax = plt.subplots(
                    figsize=(8.6, 5.6)
                )

                # ------------------------------------------------
                # IMBIE observations
                # ------------------------------------------------
                plot_imbie(
                    ax,
                    variable,
                    linewidth=1.8,
                )

                # ------------------------------------------------
                # 2025 SEJ — normal filled projection
                # ------------------------------------------------
                study_year_mask = (
                    (years >= FINAL_OBSERVATION_YEAR + 1)
                    & (years <= AR6_XLIM[1])
                )

                study_plot_years = years[study_year_mask]

                # 5–95% range
                ax.fill_between(
                    study_plot_years,
                    study_q[5][study_year_mask],
                    study_q[95][study_year_mask],
                    color=study_color,
                    alpha=THIS_STUDY_OUTER_ALPHA,
                    linewidth=0,
                    zorder=1,
                )

                # 17–83% range
                ax.fill_between(
                    study_plot_years,
                    study_q[17][study_year_mask],
                    study_q[83][study_year_mask],
                    color=study_color,
                    alpha=THIS_STUDY_INNER_ALPHA,
                    linewidth=0,
                    zorder=2,
                )

                # Median
                ax.plot(
                    study_plot_years,
                    study_q[50][study_year_mask],
                    color=study_color,
                    linestyle="-",
                    linewidth=STUDY_MEDIAN_LW,
                    zorder=5,
                )

                # ------------------------------------------------
                # AR6 — gray line overlay only
                # ------------------------------------------------
                ar6_year_mask = (
                    (ar6_years >= AR6_XLIM[0])
                    & (ar6_years <= AR6_XLIM[1])
                )

                ar6_plot_years = ar6_years[ar6_year_mask]

                # 5th / 95th percentiles
                for percentile in (5, 95):
                    ax.plot(
                        ar6_plot_years,
                        ar6_q[percentile][ar6_year_mask],
                        color=AR6_LINE_COLOR,
                        linestyle=":",
                        linewidth=AR6_OUTER_LW,
                        zorder=6,
                    )

                # 17th / 83rd percentiles
                for percentile in (17, 83):
                    ax.plot(
                        ar6_plot_years,
                        ar6_q[percentile][ar6_year_mask],
                        color=AR6_LINE_COLOR,
                        linestyle="--",
                        linewidth=AR6_INNER_LW,
                        zorder=6,
                    )

                # Median
                ax.plot(
                    ar6_plot_years,
                    ar6_q[50][ar6_year_mask],
                    color=AR6_LINE_COLOR,
                    linestyle="-",
                    linewidth=AR6_MEDIAN_LW,
                    zorder=7,
                )

                # ------------------------------------------------
                # Axes
                # ------------------------------------------------
                ax.axhline(
                    0,
                    color="0.75",
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.6,
                    zorder=0,
                )

                ax.set_xlim(
                    AR6_XLIM
                )

                ax.set_xticks(
                    AR6_XTICKS
                )

                ax.set_xlabel(
                    "Year",
                    fontweight="bold",
                    fontsize=12,
                )

                ax.set_ylabel(
                    "SLR contribution (cm)",
                    fontweight="bold",
                    fontsize=12,
                )

                ax.set_title(
                    (
                        f"{VARIABLE_LABELS[variable]} — "
                        f"{SCENARIO_LABELS[scenario]}"
                    ),
                    fontsize=15,
                    fontweight="bold",
                )

                ax.grid(
                    alpha=0.30,
                    linestyle="--",
                )

                # ------------------------------------------------
                # Legend
                # ------------------------------------------------
                legend_handles = [

                    Line2D(
                        [0], [0],
                        color="black",
                        linewidth=1.8,
                        label="IMBIE observations",
                    ),

                    Patch(
                        facecolor=study_color,
                        alpha=THIS_STUDY_OUTER_ALPHA,
                        label="2025 SEJ 5–95%",
                    ),

                    Patch(
                        facecolor=study_color,
                        alpha=THIS_STUDY_INNER_ALPHA,
                        label="2025 SEJ 17–83%",
                    ),

                    Line2D(
                        [0], [0],
                        color=study_color,
                        linestyle="-",
                        linewidth=STUDY_MEDIAN_LW,
                        label="2025 SEJ median",
                    ),

                    Line2D(
                        [0], [0],
                        color=AR6_LINE_COLOR,
                        linestyle=":",
                        linewidth=AR6_OUTER_LW,
                        label="AR6 5–95%",
                    ),

                    Line2D(
                        [0], [0],
                        color=AR6_LINE_COLOR,
                        linestyle="--",
                        linewidth=AR6_INNER_LW,
                        label="AR6 17–83%",
                    ),

                    Line2D(
                        [0], [0],
                        color=AR6_LINE_COLOR,
                        linestyle="-",
                        linewidth=AR6_MEDIAN_LW,
                        label="AR6 median",
                    ),
                ]

                ax.legend(
                    handles=legend_handles,
                    frameon=False,
                    loc="upper left",
                    fontsize=9.5,
                    ncol=2,
                )

                fig.tight_layout()

                # ------------------------------------------------
                # Save
                # ------------------------------------------------
                outfile = (
                    AR6_COMPARISON_DIR
                    / (
                        "AR6_common_axis_comparison_"
                        f"{AR6_VARIABLE_FILE_LABELS[variable]}_"
                        f"{AR6_SCENARIO_FILE_LABELS[scenario]}.png"
                    )
                )

                save_figure(
                    fig,
                    outfile,
                )

                plt.close(
                    fig
                )

                print(
                    f"Saved: {outfile}"
                )


    # ============================================================
    # 11. SSP PROJECTION FIGURES
    # ============================================================

    if MAKE_SSP_FIGURES:

        print("\n" + "=" * 72)
        print("CREATING SSP PROJECTION FIGURES")
        print("=" * 72)

        for variable in SSP_VARIABLE_ORDER:

            fig, axes = plt.subplots(
                nrows=2,
                ncols=3,
                figsize=(
                    12.5,
                    8.0,
                ),
                sharex=True,
                sharey=True,
            )
            axes = np.asarray(axes).reshape(-1)

            for ax, ssp in zip(
                axes,
                SSP_ORDER,
            ):

                mask = ssp_mask(
                    ssp
                )

                values = (
                    TRAJECTORIES[
                        variable
                    ][
                        mask,
                        :,
                    ]
                )

                plot_projection_distribution(
                    ax=ax,
                    values=values,
                    variable=variable,
                    scenario=None,
                    include_original_endpoints=False,
                    include_imbie=True,
                )

                ax.set_xlim(
                    2000,
                    2300,
                )

                ax.set_xticks(
                    [
                        2000,
                        2050,
                        2100,
                        2150,
                        2200,
                        2250,
                        2300,
                    ]
                )

                ax.set_title(
                    SSP_LABELS[
                        ssp
                    ],
                    fontsize=12,
                    fontweight = 'bold'
                )

                ax.set_xlabel(
                    "Year",
                    fontweight = 'bold'
                )

            for _ax in (axes[0], axes[3]):
                _ax.set_ylabel(
                    "SLR contribution (cm)",
                    fontweight = 'bold')

            fig.legend(
                handles=ssp_legend_handles(variable),
                loc="upper center",
                bbox_to_anchor=(
                    0.5,
                    0.945,
                ),
                ncol=4,
                frameon=False,
            )

            fig.suptitle(
                (
                    f"{VARIABLE_LABELS[variable]} "
                    "sea-level projections — CMIP6 climate scenarios"
                ),
                fontsize=16,
                y=0.985,
            )

            fig.tight_layout(
                rect=[
                    0,
                    0,
                    1,
                    0.94,
                ]
            )

            outfile = (
                SSP_FIGURE_DIR
                / (
                    f"{VARIABLE_FILE_LABELS[variable]}_"
                    "projections_by_SSP.png"
                )
            )

            save_figure(fig, outfile)

            plt.close(
                fig
            )


    # ============================================================
    # 11. RECALCULATE TEMPORAL CORRELATIONS FROM FINAL ARRAYS
    # ============================================================

    if MAKE_CORRELATION_FIGURES:
        print("\n" + "=" * 72)
        print("CALCULATING FINAL TEMPORAL CORRELATIONS")
        print("=" * 72)

        correlation_rows = []

        for target_year in CORRELATION_TARGET_YEARS:

            target_index = year_index(
                target_year
            )

            for ssp in SSP_ORDER:

                mask = ssp_mask(
                    ssp
                )

                for variable in CORRELATION_VARIABLE_ORDER:

                    values = (
                        TRAJECTORIES[
                            variable
                        ][
                            mask,
                            :,
                        ]
                    )

                    target_values = (
                        values[
                            :,
                            target_index,
                        ]
                    )

                    for year in years:

                        if (
                            year
                            < 2024
                            or year
                            > target_year
                        ):

                            continue

                        current_values = (
                            values[
                                :,
                                year_index(
                                    year
                                ),
                            ]
                        )

                        correlation_rows.append({
                            "ssp": ssp,
                            "component": VARIABLE_LABELS[
                                variable
                            ],
                            "variable": variable,
                            "target_year": target_year,
                            "year": int(
                                year
                            ),
                            "n_realizations": int(
                                mask.sum()
                            ),
                            "pearson_r": safe_pearson(
                                current_values,
                                target_values,
                            ),
                            "spearman_rho": safe_spearman(
                                current_values,
                                target_values,
                            ),
                        })


        correlation_df = pd.DataFrame(
            correlation_rows
        )

        CORRELATION_CSV = (
            CORRELATION_FIGURE_DIR
            / "temporal_correlation_through_time.csv"
        )

        correlation_df.to_csv(
            CORRELATION_CSV,
            index=False,
        )


        # ============================================================
        # 12. CORRELATION FIGURES
        # ============================================================

        if MAKE_CORRELATION_FIGURES:

            print("\n" + "=" * 72)
            print("CREATING TEMPORAL-CORRELATION FIGURES")
            print("=" * 72)

            if CORRELATION_KIND == "pearson":

                correlation_column = (
                    "pearson_r"
                )

                correlation_label = (
                    "Pearson correlation"
                )

            elif CORRELATION_KIND == "spearman":

                correlation_column = (
                    "spearman_rho"
                )

                correlation_label = (
                    "Spearman correlation"
                )

            else:

                raise ValueError(
                    "CORRELATION_KIND must be 'pearson' or 'spearman'."
                )


            for variable in CORRELATION_VARIABLE_ORDER:

                for target_year in CORRELATION_TARGET_YEARS:

                    fig, ax = plt.subplots(
                        figsize=(
                            9,
                            5.8,
                        )
                    )

                    for ssp in SSP_ORDER:

                        subset = correlation_df[
                            (
                                correlation_df[
                                    "variable"
                                ]
                                == variable
                            )
                            & (
                                correlation_df[
                                    "ssp"
                                ]
                                == ssp
                            )
                            & (
                                correlation_df[
                                    "target_year"
                                ]
                                == target_year
                            )
                        ].sort_values(
                            "year"
                        )

                        ax.plot(
                            subset[
                                "year"
                            ],
                            subset[
                                correlation_column
                            ],
                            linewidth=2.0,
                            label=SSP_LABELS[
                                ssp
                            ],
                        )


                    # Match the previous figure convention.
                    if target_year == 2200:

                        ax.axvline(
                            2100,
                            linewidth=1.0,
                            linestyle="--",
                            alpha=0.45,
                        )

                    elif target_year == 2300:

                        ax.axvline(
                            2100,
                            linewidth=1.0,
                            linestyle="--",
                            alpha=0.45,
                        )

                        ax.axvline(
                            2200,
                            linewidth=1.0,
                            linestyle="--",
                            alpha=0.45,
                        )


                    ax.set_ylim(
                        -0.05,
                        1.05,
                    )

                    ax.set_xlim(
                        2024,
                        target_year,
                    )

                    ax.set_xlabel(
                        "Year"
                    )

                    ax.set_ylabel(
                        (
                            f"{correlation_label} with "
                            f"{target_year} contribution"
                        )
                    )

                    ax.set_title(
                        (
                            f"{VARIABLE_LABELS[variable]}: "
                            f"evolution of correlation with "
                            f"{target_year} uncertainty"
                        ),
                        fontsize=15,
                    )

                    ax.grid(alpha=0.4, linestyle = '--')

                    ax.legend(
                        frameon=False,
                    )

                    fig.tight_layout()

                    outfile = (
                        CORRELATION_FIGURE_DIR
                        / (
                            f"{VARIABLE_FILE_LABELS[variable]}_"
                            f"correlation_with_{target_year}_"
                            f"through_time_{CORRELATION_KIND}.png"
                        )
                    )

                    save_figure(fig, outfile)

                    plt.close(
                        fig
                    )



    # ============================================================
    # 13. FINAL SUMMARY
    # ============================================================

    print("\n" + "=" * 72)
    print("FINAL FIGURE SCRIPT COMPLETE")
    print("=" * 72)

    print(
        "\nWarming-scenario figures:"
    )

    print(
        WARMING_FIGURE_DIR
    )

    print(
        "\n3 x 3 grid:"
    )

    print(
        GRID_FIGURE_DIR
    )

    print(
        "\nAR6 three-panel comparison figures:"
    )

    print(
        AR6_COMPARISON_DIR
    )

    print(
        "\nSSP figures:"
    )

    print(
        SSP_FIGURE_DIR
    )

    print(
        "\nCorrelation figures + combined correlation CSV:"
    )

    print(
        CORRELATION_FIGURE_DIR
    )

    print(
        "\nCombined correlation CSV:"
    )

    print(
        CORRELATION_CSV
    )


# ============================================================
# CMIP7 PATHWAY PROJECTION FIGURES — ALL SEVEN PATHWAYS
# ============================================================

def _decimal_year_for_cmip7(dates):
    """Convert pandas datetimes to decimal years on the numeric projection axis."""
    dates = pd.DatetimeIndex(dates)
    year = dates.year.astype(float)
    start = pd.to_datetime(dates.year.astype(str) + "-01-01")
    end = pd.to_datetime((dates.year + 1).astype(str) + "-01-01")
    return year + (dates - start).days.astype(float) / (end - start).days.astype(float)


def _cmip7_observation_series(imbie_df):
    required = {
        "Date",
        "E_cm",
        "W_cm",
        "G_cm",
    }
    missing = sorted(required.difference(imbie_df.columns))
    if missing:
        raise ValueError(f"CMIP7 1995–2014-rebased IMBIE file is missing columns: {missing}")

    obs = imbie_df.copy()
    obs["Date"] = pd.to_datetime(obs["Date"], errors="raise")
    e = obs["E_cm"].to_numpy(dtype=float)
    w = obs["W_cm"].to_numpy(dtype=float)
    g = obs["G_cm"].to_numpy(dtype=float)
    if not all(np.isfinite(a).all() for a in (e, w, g)):
        raise ValueError("CMIP7 1995–2014-rebased IMBIE file contains non-finite values")

    return (
        _decimal_year_for_cmip7(obs["Date"]),
        {"E": e, "W": w, "G": g, "AIS": e + w, "TOTAL": e + w + g},
    )


def run_cmip7_pathway_figures():
    """Plot GrIS, Total AIS, and Total ice sheet for ALL seven CMIP7 pathways."""
    print("\n" + "=" * 72)
    print("CREATING CMIP7 PATHWAY FIGURES — ALL SEVEN PATHWAYS")
    print("=" * 72)

    for required_file in (
        CMIP7_SAMPLED_SEJ_FILE,
        CMIP7_ANNUAL_TRAJECTORY_FILE,
        CMIP7_REBASED_IMBIE_FILE,
    ):
        if not required_file.is_file():
            raise FileNotFoundError(
                "Missing FINAL CMIP7 core-chain input:\n"
                f"  {required_file}\n"
                "Generate the emulandice2050_tension_p003 CMIP7 annual branch first."
            )

    meta = pd.read_csv(CMIP7_SAMPLED_SEJ_FILE)
    meta["ssp"] = meta["ssp"].astype(str).str.strip()
    validate_core_method_metadata(
        meta,
        family="CMIP7 final sampled annual realizations",
        expected_pathways=CMIP7_ALL_SCENARIOS,
        require_idw_power=True,
    )
    if "scenario_family" in meta.columns:
        if not meta["scenario_family"].astype(str).eq("CMIP7").all():
            found = sorted(meta["scenario_family"].astype(str).unique())
            raise ValueError(
                f"CMIP7 sampled file has unexpected scenario_family values: {found}"
            )

    with Dataset(CMIP7_ANNUAL_TRAJECTORY_FILE, "r") as nc:
        missing_vars = sorted({"years", "E", "W", "G"}.difference(nc.variables))
        if missing_vars:
            raise ValueError(
                f"CMIP7 annual NetCDF is missing variables {missing_vars}: "
                f"{CMIP7_ANNUAL_TRAJECTORY_FILE}"
            )
        cmip7_years = np.asarray(nc.variables["years"][:], dtype=int)
        e = np.asarray(nc.variables["E"][:, :], dtype=float)
        w = np.asarray(nc.variables["W"][:, :], dtype=float)
        g = np.asarray(nc.variables["G"][:, :], dtype=float)

    if not np.array_equal(cmip7_years, np.arange(2000, 2301, dtype=int)):
        raise ValueError("CMIP7 annual NetCDF must contain every year 2000-2300")
    if e.shape != w.shape or e.shape != g.shape:
        raise ValueError("CMIP7 E/W/G annual arrays do not have identical shapes")
    if e.shape != (len(meta), len(cmip7_years)):
        raise ValueError(
            "CMIP7 sampled metadata and annual NetCDF dimensions do not match: "
            f"metadata={len(meta)}, arrays={e.shape}"
        )
    if not all(np.isfinite(a).all() for a in (e, w, g)):
        raise ValueError("CMIP7 annual arrays contain non-finite values")

    trajectories = {
        "E": e,
        "W": w,
        "G": g,
        "AIS": e + w,
        "TOTAL": e + w + g,
    }

    obs_df = pd.read_csv(CMIP7_REBASED_IMBIE_FILE)
    obs_years, obs_series = _cmip7_observation_series(obs_df)

    quantiles = {}
    quantile_rows = []
    for code in CMIP7_ALL_SCENARIOS:
        mask = meta["ssp"].eq(code).to_numpy()
        n = int(mask.sum())
        if n == 0:
            raise ValueError(f"CMIP7 pathway {code} has no annual realizations")
        print(f"  {code}: {n:,} realizations")

        for variable in CMIP7_PATHWAY_VARIABLE_ORDER:
            q = np.percentile(
                trajectories[variable][mask, :],
                PERCENTILES,
                axis=0,
            )
            quantiles[(code, variable)] = q
            for j, year in enumerate(cmip7_years):
                quantile_rows.append({
                    "scenario_family": "CMIP7",
                    "scenario": code,
                    "component": variable,
                    "year": int(year),
                    "p05_cm": float(q[0, j]),
                    "p17_cm": float(q[1, j]),
                    "p50_cm": float(q[2, j]),
                    "p83_cm": float(q[3, j]),
                    "p95_cm": float(q[4, j]),
                })

    CMIP7_PATHWAY_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    quantile_file = (
        CMIP7_PATHWAY_FIGURE_DIR
        / "CMIP7_all_7_pathways_annual_quantiles_2000_2300_cm.csv"
    )
    pd.DataFrame(quantile_rows).to_csv(quantile_file, index=False)
    print("Saved CMIP7 figure data:", quantile_file)

    for variable in CMIP7_PATHWAY_VARIABLE_ORDER:
        fig, axs = plt.subplots(
            4, 2, figsize=(8, 8), sharex=True, sharey=True
        )
        axes = np.asarray(axs).reshape(-1)
        color = VARIABLE_STUDY_COLORS[variable]

        for ax, code in zip(axes, CMIP7_ALL_SCENARIOS):
            q = quantiles[(code, variable)]
            ax.fill_between(
                cmip7_years, q[0], q[4],
                color=color, alpha=THIS_STUDY_OUTER_ALPHA, linewidth=0,
            )
            ax.fill_between(
                cmip7_years, q[1], q[3],
                color=color, alpha=THIS_STUDY_INNER_ALPHA, linewidth=0,
            )
            ax.plot(
                cmip7_years, q[2], color=color,
                linewidth=THIS_STUDY_MEDIAN_LW, zorder=4,
            )
            ax.plot(
                obs_years, obs_series[variable],
                color="black", linewidth=2.0, zorder=6,
            )
            if SHOW_2050_TRANSITION_GUIDE:
                ax.axvline(2050, linestyle=":", linewidth=0.9, color="0.55")
            ax.axhline(0.0, linewidth=0.7, color="0.75", zorder=0)
            ax.set_xlim(2000, 2300)
            ax.set_xticks([2000, 2100, 2200, 2300])

            if variable == "G":   # GrIS
                ax.yaxis.set_major_locator(MultipleLocator(100))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
            
            ax.set_title(
                CMIP7_ALL_LABELS[code], fontsize=11, fontweight="semibold"
            )
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.30)
            ax.tick_params(labelbottom=True, labelleft=True)

        for ax in axes[len(CMIP7_ALL_SCENARIOS):]:
            ax.set_visible(False)

        fig.supxlabel("Year", fontsize=12, fontweight="bold", y=0.055)
        fig.supylabel(
            "Sea-level contribution (cm)", fontsize=12, fontweight="bold", x=0.015
        )
        fig.suptitle(
            f"{VARIABLE_LABELS[variable]} — CMIP7 climate pathways",
            fontsize=14, fontweight="bold", y=0.985,
        )
        handles = [
            Line2D(
                [0], [0], color=color, linewidth=THIS_STUDY_MEDIAN_LW,
                label="Median",
            ),
            Patch(
                facecolor=color, alpha=THIS_STUDY_INNER_ALPHA,
                label="17–83% interval",
            ),
            Patch(
                facecolor=color, alpha=THIS_STUDY_OUTER_ALPHA,
                label="5–95% interval",
            ),
            Line2D(
                [0], [0], color="black", linewidth=2.0,
                label="IMBIE observations",
            ),
        ]
        fig.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncol=4, frameon=False, fontsize=9,
        )
        fig.subplots_adjust(
            left=0.1, right=0.99, bottom=0.13, top=0.91,
            wspace=0.25, hspace=0.43,
        )

        outfile = (
            CMIP7_PATHWAY_FIGURE_DIR
            / f"{VARIABLE_FILE_LABELS[variable]}_CMIP7_all_7_pathways_panels_2000_2300.png"
        )
        save_figure(fig, outfile)
        if not SHOW_FIGURES:
            plt.close(fig)
        print("Saved CMIP7 pathway figure:", outfile)

    print("CMIP7 pathways plotted:", ", ".join(CMIP7_ALL_SCENARIOS))
    print("CMIP7 pathway outputs:", CMIP7_PATHWAY_FIGURE_DIR)


# ============================================================
# REALIZED SCENARIO-SELECTION FRACTION HEAT MAPS
# ============================================================
HEATMAP_CMIP6_SCENARIOS = list(SSP_ORDER)
HEATMAP_CMIP7_SCENARIOS = list(CMIP7_ALL_SCENARIOS)
HEATMAP_SCENARIO_LABELS = {**dict(CLIMATE_SCENARIO_LABELS), **CMIP7_ALL_LABELS}
HEATMAP_SCENARIO_LABELS.setdefault("ssp534-over", "SSP5-3.4-OS")
HEATMAP_SCENARIO_COLORS = dict(CLIMATE_SCENARIO_COLORS)
if "ssp534-over" not in HEATMAP_SCENARIO_COLORS and "ssp534" in HEATMAP_SCENARIO_COLORS:
    HEATMAP_SCENARIO_COLORS["ssp534-over"] = HEATMAP_SCENARIO_COLORS["ssp534"]

_HEATMAP_REF_C2100 = dict(REF_CUMULATIVE_WARMING[2100])
_HEATMAP_REF_C2200 = dict(REF_CUMULATIVE_WARMING[2200])
_HEATMAP_RANGE_2100 = _HEATMAP_REF_C2100["HIGH"] - _HEATMAP_REF_C2100["LOW"]
_HEATMAP_RANGE_2200 = _HEATMAP_REF_C2200["HIGH"] - _HEATMAP_REF_C2200["LOW"]
_HEATMAP_REFERENCE_POINTS = {
    scenario: (
        (_HEATMAP_REF_C2100[scenario] - _HEATMAP_REF_C2100["LOW"]) / _HEATMAP_RANGE_2100,
        (_HEATMAP_REF_C2200[scenario] - _HEATMAP_REF_C2200["LOW"]) / _HEATMAP_RANGE_2200,
    )
    for scenario in SCENARIO_ORDER
}


def _heatmap_segmented_edges(limits, n_below, n_middle, n_above):
    lo, hi = limits
    return np.concatenate([
        np.linspace(lo, 0.0, n_below + 1)[:-1],
        np.linspace(0.0, 1.0, n_middle + 1)[:-1],
        np.linspace(1.0, hi, n_above + 1),
    ])


def heatmap_normalized_bin_edges():
    x_edges = _heatmap_segmented_edges(
        HEATMAP_X_LIM, HEATMAP_N_X_BELOW_0,
        HEATMAP_N_X_0_TO_1, HEATMAP_N_X_ABOVE_1,
    )
    y_edges = _heatmap_segmented_edges(
        HEATMAP_Y_LIM, HEATMAP_N_Y_BELOW_0,
        HEATMAP_N_Y_0_TO_1, HEATMAP_N_Y_ABOVE_1,
    )
    return x_edges, y_edges


def _heatmap_resolve_cumulative_column(df, year):
    candidates = [
        f"cumulative_warming_{year}_for_truncation",
        f"cumulative_warming_{year}",
    ]
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Could not find cumulative-warming column for {year}; tried {candidates}"
    )


def load_heatmap_selection_data(filepath, family, climate_scenarios):
    if not filepath.is_file():
        raise FileNotFoundError(
            f"{family}: missing double-checkpoint-truncation selection CSV:\n{filepath}"
        )

    print(f"{family} heat-map selection input: {filepath}")
    df = pd.read_csv(filepath)

    required_extra = {
        "weight_low", "weight_intermediate", "weight_high",
    }
    missing_extra = sorted(required_extra.difference(df.columns))
    if missing_extra:
        raise ValueError(f"{family}: missing heat-map columns: {missing_extra}")

    df["ssp"] = df["ssp"].astype(str).str.strip()
    if family == "CMIP6":
        df["ssp"] = df["ssp"].map(canonicalize_cmip6_ssp)
    df = df[df["ssp"].isin(climate_scenarios)].copy()
    if df.empty:
        raise ValueError(f"{family}: no requested pathways found in {filepath}")

    validate_core_method_metadata(
        df, family=family, expected_pathways=climate_scenarios, require_idw_power=False
    )

    if family == "CMIP7" and "scenario_family" in df.columns:
        if not df["scenario_family"].astype(str).eq("CMIP7").all():
            raise ValueError("CMIP7: incorrect scenario_family metadata")

    c2100_col = _heatmap_resolve_cumulative_column(df, 2100)
    c2200_col = _heatmap_resolve_cumulative_column(df, 2200)

    numeric = [
        c2100_col, c2200_col,
        "weight_low", "weight_intermediate", "weight_high",
    ]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{family}: non-finite cumulative warming or weights")

    p = df[["weight_low", "weight_intermediate", "weight_high"]].to_numpy(dtype=float)
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f"{family}: probabilities do not sum to 1")

    c2100 = df[c2100_col].to_numpy(dtype=float)
    c2200 = df[c2200_col].to_numpy(dtype=float)
    below_both = (
        (c2100 <= _HEATMAP_REF_C2100["LOW"])
        & (c2200 <= _HEATMAP_REF_C2200["LOW"])
    )
    above_both = (
        (c2100 >= _HEATMAP_REF_C2100["HIGH"])
        & (c2200 >= _HEATMAP_REF_C2200["HIGH"])
    )

    selected = df["selected_scenario"].astype(str).str.strip().str.upper().to_numpy()
    if not np.all(selected[below_both] == "LOW"):
        raise ValueError(f"{family}: below-both trajectories were not always assigned LOW")
    if not np.all(selected[above_both] == "HIGH"):
        raise ValueError(f"{family}: above-both trajectories were not always assigned HIGH")
    if not np.allclose(p[below_both], [1, 0, 0], atol=1e-10):
        raise ValueError(f"{family}: below-both probabilities violate truncation")
    if not np.allclose(p[above_both], [0, 0, 1], atol=1e-10):
        raise ValueError(f"{family}: above-both probabilities violate truncation")

    df["C2100_norm"] = (c2100 - _HEATMAP_REF_C2100["LOW"]) / _HEATMAP_RANGE_2100
    df["C2200_norm"] = (c2200 - _HEATMAP_REF_C2200["LOW"]) / _HEATMAP_RANGE_2200

    print(
        f"{family}: {len(df):,} saved realized assignments loaded; "
        f"below-both={below_both.mean():.3f}, above-both={above_both.mean():.3f}."
    )
    return df


def calculate_heatmap_realized_fractions(df, x_edges, y_edges):
    x = df["C2100_norm"].to_numpy(dtype=float)
    y = df["C2200_norm"].to_numpy(dtype=float)
    assigned = df["selected_scenario"].astype(str).str.upper().to_numpy()

    n_total, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    keep = n_total >= HEATMAP_MIN_COUNT_PER_BIN
    fractions = {}
    for scenario in SCENARIO_ORDER:
        n_selected, _, _ = np.histogram2d(
            x[assigned == scenario], y[assigned == scenario],
            bins=[x_edges, y_edges],
        )
        z = np.full(n_total.shape, np.nan, dtype=float)
        z[keep] = n_selected[keep] / n_total[keep]
        fractions[scenario] = z
    return n_total, fractions


def save_heatmap_grid_table(family, x_edges, y_edges, counts, fractions):
    rows = []
    for ix in range(len(x_edges) - 1):
        for iy in range(len(y_edges) - 1):
            rows.append({
                "C2100_norm_left": x_edges[ix],
                "C2100_norm_right": x_edges[ix + 1],
                "C2200_norm_bottom": y_edges[iy],
                "C2200_norm_top": y_edges[iy + 1],
                "n_realizations": int(counts[ix, iy]),
                "fraction_LOW": fractions["LOW"][ix, iy],
                "fraction_INTERMEDIATE": fractions["INTERMEDIATE"][ix, iy],
                "fraction_HIGH": fractions["HIGH"][ix, iy],
            })
    HEATMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = HEATMAP_OUTPUT_DIR / (
        f"{family}_realized_selected_fraction_grid_NORMALIZED_C2100_C2200.csv"
    )
    pd.DataFrame(rows).to_csv(outfile, index=False)
    print(f"Saved grid: {outfile}")


def add_heatmap_reference_structure(ax):
    xs = [_HEATMAP_REFERENCE_POINTS[s][0] for s in SCENARIO_ORDER]
    ys = [_HEATMAP_REFERENCE_POINTS[s][1] for s in SCENARIO_ORDER]
    ax.plot(xs, ys, "-", color=REFERENCE_PATH_COLOR, linewidth=1.6, zorder=7)

    for scenario in SCENARIO_ORDER:
        rx, ry = _HEATMAP_REFERENCE_POINTS[scenario]
        ax.plot([rx, rx], [HEATMAP_Y_LIM[0], ry], "--",
                linewidth=1.0, color="0.30", alpha=0.75, zorder=5)
        ax.plot([HEATMAP_X_LIM[0], rx], [ry, ry], "--",
                linewidth=1.0, color="0.30", alpha=0.75, zorder=5)

    # Only the two corners are deterministic under the double-checkpoint rule.
    ax.plot([HEATMAP_X_LIM[0], 0.0], [0.0, 0.0], ":",
            color="0.20", linewidth=1.4, alpha=0.95, zorder=6)
    ax.plot([0.0, 0.0], [HEATMAP_Y_LIM[0], 0.0], ":",
            color="0.20", linewidth=1.4, alpha=0.95, zorder=6)
    ax.plot([1.0, HEATMAP_X_LIM[1]], [1.0, 1.0], ":",
            color="0.20", linewidth=1.4, alpha=0.95, zorder=6)
    ax.plot([1.0, 1.0], [1.0, HEATMAP_Y_LIM[1]], ":",
            color="0.20", linewidth=1.4, alpha=0.95, zorder=6)

    for scenario in SCENARIO_ORDER:
        rx, ry = _HEATMAP_REFERENCE_POINTS[scenario]
        ax.plot(rx, ry, "*", markersize=14, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.0, zorder=9)
        if scenario == "LOW":
            offset, ha, va = (0, -14), "center", "top"
        elif scenario == "INTERMEDIATE":
            offset, ha, va = (12, 0), "left", "center"
        else:
            offset, ha, va = (5, 7), "left", "bottom"
        ax.annotate(
            scenario, (rx, ry), xytext=offset, textcoords="offset points",
            ha=ha, va=va, fontsize=9, fontweight="bold", color="black", zorder=10,
        )


def make_heatmap_combined_figure(df, family, climate_scenarios):
    x_edges, y_edges = heatmap_normalized_bin_edges()
    n_total, fractions = calculate_heatmap_realized_fractions(df, x_edges, y_edges)
    save_heatmap_grid_table(family, x_edges, y_edges, n_total, fractions)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), sharex=True, sharey=True)
    filled = None

    for ax, scenario in zip(axes, SCENARIO_ORDER):
        z = np.ma.masked_invalid(fractions[scenario].T)
        if z.count() < 4:
            raise ValueError(
                f"{family}/{scenario}: too few populated bins; "
                "reduce HEATMAP_MIN_COUNT_PER_BIN or the bin count."
            )

        filled = ax.contourf(
            X, Y, z, levels=HEATMAP_FILL_LEVELS, cmap=HEATMAP_CMAP,
            vmin=0.0, vmax=1.0, extend="neither", zorder=1,
        )
        finite_z = fractions[scenario][np.isfinite(fractions[scenario])]
        levels = [
            lv for lv in HEATMAP_LINE_LEVELS
            if finite_z.min() < lv < finite_z.max()
        ]
        if levels:
            lines = ax.contour(
                X, Y, z, levels=levels, colors="white",
                linewidths=0.85, alpha=0.95, zorder=3,
            )
            labels = ax.clabel(lines, inline=True, fontsize=9, colors="white", fmt="%.1f")
            for label in labels:
                label.set_fontweight("bold")
                label.set_path_effects([
                    patheffects.Stroke(linewidth=1.4, foreground="black"),
                    patheffects.Normal(),
                ])

        if HEATMAP_SHOW_SCENARIO_POINTS:
            for climate_scenario in climate_scenarios:
                sub = df[df["ssp"] == climate_scenario]
                scatter_kwargs = {}
                if climate_scenario in HEATMAP_SCENARIO_COLORS:
                    scatter_kwargs["color"] = HEATMAP_SCENARIO_COLORS[climate_scenario]
                ax.scatter(
                    sub["C2100_norm"], sub["C2200_norm"],
                    s=HEATMAP_POINT_SIZE, alpha=HEATMAP_POINT_ALPHA,
                    edgecolors="none", rasterized=True, zorder=4,
                    **scatter_kwargs,
                )

        ax.set_xlim(HEATMAP_X_LIM)
        ax.set_ylim(HEATMAP_Y_LIM)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Selected {scenario}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Normalized cumulative warming\nthrough 2100",
                      fontsize=10, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Normalized cumulative warming\nthrough 2200",
                          fontsize=10, fontweight="bold")
        ax.set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        ax.tick_params(labelsize=10, labelbottom=True, labelleft=(ax is axes[0]))
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.28, zorder=0)
        add_heatmap_reference_structure(ax)

    cbar = fig.colorbar(
        filled, ax=axes, fraction=0.025, pad=0.018,
        ticks=np.arange(0.0, 1.01, 0.1),
    )
    cbar.set_label("Fraction selected", fontsize=11)

    fig.suptitle(
        f"{family}: realized warming-scenario assignment fractions\n"
        "[2100, 2200] normalized checkpoints + double-checkpoint endpoint truncation",
        fontsize=14, y=0.995,
    )

    if HEATMAP_SHOW_SCENARIO_POINTS:
        handles = []
        for s in climate_scenarios:
            kwargs = {}
            if s in HEATMAP_SCENARIO_COLORS:
                kwargs["color"] = HEATMAP_SCENARIO_COLORS[s]
            handles.append(
                Line2D([0], [0], marker="o", linestyle="None", markersize=6,
                       label=HEATMAP_SCENARIO_LABELS.get(s, s), **kwargs)
            )
        fig.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.005),
            ncol=len(handles), frameon=False, fontsize=9,
        )

    fig.subplots_adjust(
        left=0.07, right=0.90,
        bottom=0.20 if HEATMAP_SHOW_SCENARIO_POINTS else 0.13,
        top=0.82, wspace=0.12,
    )

    HEATMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = HEATMAP_OUTPUT_DIR / (
        f"{family}_REALIZED_selected_fraction_contours_"
        "NORMALIZED_C2100_C2200_double_checkpoint_truncation.png"
    )
    save_figure(fig, outfile)
    if not SHOW_FIGURES:
        plt.close(fig)
    print(f"Saved heat map: {outfile}")


def run_realized_fraction_heatmaps():
    if not HEATMAP_PLOT_CMIP6 and not HEATMAP_PLOT_CMIP7:
        print("Heat maps enabled, but both HEATMAP_PLOT_CMIP6 and HEATMAP_PLOT_CMIP7 are False.")
        return

    print("\n" + "=" * 72)
    print("CREATING REALIZED SCENARIO-SELECTION FRACTION HEAT MAPS")
    print("=" * 72)

    if HEATMAP_PLOT_CMIP6:
        df6 = load_heatmap_selection_data(
            HEATMAP_CMIP6_SELECTION_FILE, "CMIP6", HEATMAP_CMIP6_SCENARIOS
        )
        make_heatmap_combined_figure(df6, "CMIP6", HEATMAP_CMIP6_SCENARIOS)

    if HEATMAP_PLOT_CMIP7:
        df7 = load_heatmap_selection_data(
            HEATMAP_CMIP7_SELECTION_FILE, "CMIP7", HEATMAP_CMIP7_SCENARIOS
        )
        make_heatmap_combined_figure(df7, "CMIP7", HEATMAP_CMIP7_SCENARIOS)

    print("Heat-map outputs:", HEATMAP_OUTPUT_DIR)


def main():
    enabled_groups = [
        name
        for name, enabled in [
            ("manuscript", RUN_MANUSCRIPT_FIGURES),
            ("supplementary", RUN_SUPPLEMENTARY_FIGURES),
            ("diagnostic", RUN_DIAGNOSTIC_FIGURES),
        ]
        if enabled
    ]

    print("\n" + "=" * 72)
    print("CONSOLIDATED FAIR2 FIGURE SUITE")
    print("=" * 72)
    print("Enabled groups:", ", ".join(enabled_groups) if enabled_groups else "none")
    print("CMIP6 / SSP input root:", CMIP6_CORE_METHOD_ROOT)
    print("CMIP7 input root:", CMIP7_CORE_METHOD_ROOT)
    print("Output root:", SUITE_OUTPUT_ROOT)

    if not (
        RUN_FINAL_RESULTS_SUITE
        or RUN_CMIP7_PATHWAY_FIGURES
        or RUN_REALIZED_FRACTION_HEATMAPS
    ):
        raise ValueError(
            "Enable at least one top-level group: RUN_MANUSCRIPT_FIGURES, "
            "RUN_SUPPLEMENTARY_FIGURES, or RUN_DIAGNOSTIC_FIGURES."
        )

    if RUN_FINAL_RESULTS_SUITE:
        run_final_results_suite()

    if RUN_CMIP7_PATHWAY_FIGURES:
        run_cmip7_pathway_figures()

    if RUN_REALIZED_FRACTION_HEATMAPS:
        run_realized_fraction_heatmaps()

    if SHOW_FIGURES:
        maybe_show()

    print("\nConsolidated FAIR2 figure suite completed.")


if __name__ == "__main__":
    main()
