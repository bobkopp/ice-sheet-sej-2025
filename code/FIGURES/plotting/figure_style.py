"""Shared manuscript figure colors, keyed by *meaning* rather than panel.

Edit palette values here to change every plot using that category. These values
reproduce the previously assigned colors; note that the two existing TOTAL
ice-sheet plot families deliberately used different charcoal shades.

This file is presentation-only: it must not contain sampling, unit conversion,
quantile calculations, or data-path settings.
"""

# Underlying colorblind-friendly hues used in the current manuscript figures.
BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
RED_ORANGE = "#D55E00"
PURPLE = "#6A3D9A"
PINK = "#CC79A7"
BLACK = "#000000"

# Use the same color for a physical component everywhere it appears.
ICE_SHEET_COLORS = {
    "EAIS": BLUE,
    "WAIS": ORANGE,
    "Total AIS": PINK,
    "GrIS": GREEN,
}
WARMING_COLORS = {2.0: BLUE, 3.5: PURPLE, 5.0: RED_ORANGE}
PERFORMANCE_YEAR_COLORS = {2018: BLUE, 2025: ORANGE}
CLIMATE_SCENARIO_COLORS = {
    "ssp119": BLUE, "ssp126": LIGHT_BLUE, "ssp245": GREEN,
    "ssp370": ORANGE, "ssp585": RED_ORANGE,
    "L": BLUE, "ML": GREEN, "M": ORANGE, "H": RED_ORANGE,
}

# Total ice-sheet SLR had two slightly different shades in the original plots.
STUDY_COMPONENT_COLORS = {
    "E": ICE_SHEET_COLORS["EAIS"],
    "W": ICE_SHEET_COLORS["WAIS"],
    "AIS": ICE_SHEET_COLORS["Total AIS"],
    "G": ICE_SHEET_COLORS["GrIS"],
    "TOTAL": "#4D4D4D",
}
CMIP_PANEL_TOTAL_COLOR = "#444444"   # Preserve existing six-panel figure shade.

# Driver-response categories: their codes are distinct from climate/ice-sheet IDs.
DRIVER_RESPONSE_COLORS = {
    "1": "#332288", "2": "#88CCEE", "3": "#44AA99",
    "4": "#117733", "5": "#DDCC77", "6": "#CC6677",
    "AM": "#4477AA", "AC": "#EE6677", "SI": "#CCBB44",
    "MOC": "#AA3377", "CDW": "#66CCEE", "ACC": "#228833",
}

# AR6 is deliberately grayscale in dedicated three-panel comparison figures.
AR6_OUTER_COLOR = "#BDBDBD"
AR6_INNER_COLOR = "#7F7F7F"
AR6_MEDIAN_COLOR = "#2F2F2F"
DIFF_PERCENTILE_STYLES = {
    5:  {"color": BLUE,       "linestyle": ":",  "linewidth": 1.8},
    17: {"color": LIGHT_BLUE, "linestyle": "--", "linewidth": 1.9},
    50: {"color": BLACK,      "linestyle": "-",  "linewidth": 2.4},
    83: {"color": ORANGE,     "linestyle": "--", "linewidth": 1.9},
    95: {"color": RED_ORANGE, "linestyle": ":",  "linewidth": 1.8},
}

# Expert-judgment target and calibration plots retain their separate visual roles.
DM_COLORS = ["0.55", "forestgreen"]
EXPERT_COLOR = "0.62"
WEIGHTED_COLOR = "forestgreen"
REALIZATION_COLOR = "firebrick"
IR_COLOR = "0.20"
FRAME_COLOR = "lightsteelblue"
BASELINE_COLOR = "steelblue"

# The normalized-fractions figure uses a separate reference-line color.
REFERENCE_PATH_COLOR = "#1f77b4"
