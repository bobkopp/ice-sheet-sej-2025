#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:05:51 2026

@author: elizabeth

Expert judgments of ice-sheet uncertainty drivers.

Figures produced
----------------
1. Scaled DP + grounded-SMB figures:
       One figure for each warming scenario.
       DP is scaled by the B25 median discharge contribution.
       SMB-G is scaled by the B25 median grounded-SMB contribution.

2. Unscaled SMB figures:
       One figure for each warming scenario.
       Each figure includes:
           - grounded-ice SMB
           - ice-shelf SMB/thinning
       for GrIS, WAIS, and EAIS at 2100, 2200, and 2300.

3. Unscaled ocean-process figures:
       One figure for each warming scenario.
       Each figure includes GrIS, WAIS, and EAIS at
       2100, 2200, and 2300.

Scaling
-------
For each ice sheet × year × warming level:

    SMB = Runoff + Accumulation

Accumulation is already negative in SLR-contribution units.

    dynamic_weight =
        |median discharge|
        ------------------------------------------
        |median discharge| + |median SMB|

    smb_weight =
        |median SMB|
        ------------------------------------------
        |median discharge| + |median SMB|

Within each scaled bar, expert-response percentages are multiplied
by the corresponding weight.

The unscaled SMB and ocean-process figures show the percentage of
valid expert responses within each question and therefore sum to 100%.
"""

import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import textwrap

from B25data_processing import (
    load_and_prepare_data,
    project_dir,
)


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = os.path.join(
    project_dir,
    "data",
    "ICE+SHEET+SEJ+target+questions_"
    "July+15,+2026_CLEANER_LIZ.csv",
)

figure_output_dir = os.path.join(
    project_dir,
    "output",
    "figures",
)

table_output_dir = os.path.join(
    project_dir,
    "output",
    "tables",
)

os.makedirs(
    figure_output_dir,
    exist_ok=True,
)

os.makedirs(
    table_output_dir,
    exist_ok=True,
)


# ============================================================
# 2. BASIC SETTINGS
# ============================================================

ice_sheet_order = [
    "GrIS",
    "WAIS",
    "EAIS",
]

year_order = [
    2100,
    2200,
    2300,
]

warming_order = [
    "2°C",
    "3.5°C",
    "5°C",
]

warming_numeric_map = {
    "2°C": 2.0,
    "3.5°C": 3.5,
    "5°C": 5.0,
}

warming_label_map = {
    2.0: "2°C",
    3.5: "3.5°C",
    5.0: "5°C",
}

process_group_order = [
    "DP",
    "SMB-G",
    "SMB-S",
    "OP",
]

process_group_titles = {
    "DP": "Dynamical processes",
    "SMB-G": "Grounded-ice SMB",
    "SMB-S": "Ice-shelf SMB and thinning",
    "OP": "Ocean processes",
}

process_group_short_labels = {
    "DP": "Dynamical processes",
    "SMB-G": "Grounded-ice\nSMB",
    "SMB-S": "Ice shelves\nSMB",
    "OP": "Ocean processes",
}


# ============================================================
# 3. QUESTION DEFINITIONS
# ============================================================

question_map = {
    # Dynamical processes
    "Q5": {
        "warming": "2°C",
        "process_group": "DP",
    },
    "Q7": {
        "warming": "3.5°C",
        "process_group": "DP",
    },
    "Q9": {
        "warming": "5°C",
        "process_group": "DP",
    },

    # Grounded-ice SMB
    "Q12": {
        "warming": "2°C",
        "process_group": "SMB-G",
    },
    "Q13": {
        "warming": "3.5°C",
        "process_group": "SMB-G",
    },
    "Q14": {
        "warming": "5°C",
        "process_group": "SMB-G",
    },

    # Ice-shelf SMB and thinning
    "Q16": {
        "warming": "2°C",
        "process_group": "SMB-S",
    },
    "Q17": {
        "warming": "3.5°C",
        "process_group": "SMB-S",
    },
    "Q18": {
        "warming": "5°C",
        "process_group": "SMB-S",
    },

    # Ocean processes
    "Q20": {
        "warming": "2°C",
        "process_group": "OP",
    },
    "Q21": {
        "warming": "3.5°C",
        "process_group": "OP",
    },
    "Q22": {
        "warming": "5°C",
        "process_group": "OP",
    },
}


# ============================================================
# 4. RESPONSE LABELS AND ORDERS
# ============================================================

response_labels = {
    # Dynamical processes
    "1": "Buttressing by ice shelves",
    "2": "Basal traction",
    "3": "Transverse stresses",
    "4": "Hydrofracturing",
    "5": "Ice cliff instability",
    "6": "Dissipation after iceberg formation",

    # Surface mass balance
    "AM": "Atmospheric circulation / moisture transport",
    "AC": "Albedo",
    "SI": "Summer sea-ice extent",

    # Ocean processes
    "MOC": "Meridional overturning circulation",
    "CDW": "Circumpolar Deep Water",
    "ACC": "Antarctic Circumpolar Current",
}

response_order = {
    "DP": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ],
    "SMB-G": [
        "AM",
        "AC",
    ],
    "SMB-S": [
        "AM",
        "AC",
        "SI",
    ],
    "OP": [
        "MOC",
        "CDW",
        "ACC",
    ],
}


# ============================================================
# 5. COLORS
# ============================================================

response_colors = {
    # Dynamical processes
    "1": "#332288",  # indigo
    "2": "#88CCEE",  # light blue
    "3": "#44AA99",  # teal
    "4": "#117733",  # dark green
    "5": "#DDCC77",  # sand
    "6": "#CC6677",  # muted rose

    # Surface mass balance
    "AM": "#4477AA",  # blue
    "AC": "#EE6677",  # coral
    "SI": "#CCBB44",  # mustard

    # Ocean processes
    "MOC": "#AA3377",  # purple
    "CDW": "#66CCEE",  # cyan
    "ACC": "#228833",  # green
}


# ============================================================
# 6. LOAD B25 DATA AND CALCULATE SCALING WEIGHTS
# ============================================================

b25_data = load_and_prepare_data()

process_ext = b25_data[
    "process_ext"
].copy()


def calculate_b25_scaling_weights(process_data):
    """
    Calculate dynamic and grounded-SMB scaling weights from the
    B25 median process contributions.

    B25data_processing calculates:

        Runoff minus accumulation = Runoff + Accumulation

    because accumulation is already signed as a negative
    contribution to sea-level rise.
    """

    required_processes = [
        "Discharge",
        "Runoff minus accumulation",
    ]

    subset = process_data.loc[
        process_data["process"].isin(
            required_processes
        )
        & process_data["ice_sheet"].isin(
            ice_sheet_order
        )
        & process_data["year"].isin(
            year_order
        )
    ].copy()

    subset["value"] = pd.to_numeric(
        subset["value"],
        errors="coerce",
    )

    median_table = (
        subset
        .groupby(
            [
                "ice_sheet",
                "year",
                "warming",
                "process",
            ],
            observed=True,
        )["value"]
        .median()
        .unstack("process")
        .reset_index()
    )

    missing_processes = (
        set(required_processes)
        - set(median_table.columns)
    )

    if missing_processes:
        raise ValueError(
            "Could not calculate B25 scaling weights. "
            "Missing process fields:\n"
            + "\n".join(
                sorted(missing_processes)
            )
        )

    median_table = median_table.rename(
        columns={
            "Discharge": "median_discharge_mm",
            "Runoff minus accumulation": "median_smb_mm",
        }
    )

    median_table["abs_discharge_mm"] = (
        median_table["median_discharge_mm"].abs()
    )

    median_table["abs_smb_mm"] = (
        median_table["median_smb_mm"].abs()
    )

    median_table["weight_denominator_mm"] = (
        median_table["abs_discharge_mm"]
        + median_table["abs_smb_mm"]
    )

    invalid_denominator = (
        median_table["weight_denominator_mm"].isna()
        | np.isclose(
            median_table["weight_denominator_mm"],
            0.0,
        )
    )

    if invalid_denominator.any():

        problematic = median_table.loc[
            invalid_denominator,
            [
                "ice_sheet",
                "year",
                "warming",
                "median_discharge_mm",
                "median_smb_mm",
            ],
        ]

        raise ValueError(
            "Invalid scaling denominator for:\n"
            f"{problematic.to_string(index=False)}"
        )

    median_table["dynamic_weight"] = (
        median_table["abs_discharge_mm"]
        / median_table["weight_denominator_mm"]
    )

    median_table["smb_weight"] = (
        median_table["abs_smb_mm"]
        / median_table["weight_denominator_mm"]
    )

    median_table["dynamic_percent"] = (
        100.0
        * median_table["dynamic_weight"]
    )

    median_table["smb_percent"] = (
        100.0
        * median_table["smb_weight"]
    )

    median_table["warming_label"] = (
        median_table["warming"].map(
            warming_label_map
        )
    )

    weight_sum = (
        median_table["dynamic_weight"]
        + median_table["smb_weight"]
    )

    if not np.allclose(
        weight_sum,
        1.0,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(
            "Dynamic and SMB scaling weights do not sum to 1."
        )

    return median_table


scaling_weights = calculate_b25_scaling_weights(
    process_ext
)


scaling_weights_file = os.path.join(
    table_output_dir,
    "SEJ_driver_B25_scaling_weights.csv",
)

scaling_weights.to_csv(
    scaling_weights_file,
    index=False,
)

print(
    "\nSaved B25 scaling weights:\n"
    f"{scaling_weights_file}"
)

print(
    "\nB25 median-based scaling weights:"
)

print(
    scaling_weights
    .sort_values(
        [
            "warming",
            "ice_sheet",
            "year",
        ]
    )
    [
        [
            "warming_label",
            "ice_sheet",
            "year",
            "median_discharge_mm",
            "median_smb_mm",
            "dynamic_percent",
            "smb_percent",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}",
    )
)


# ============================================================
# 7. READ EXPERT-RESPONSE CSV
# ============================================================

raw = pd.read_csv(
    input_file
)

print(
    "\nCSV dimensions:",
    raw.shape,
)

print(
    "Number of question columns:",
    len(raw.columns) - 1,
)


# First four rows contain:
#
# row 0 = ice sheet
# row 1 = warming scenario
# row 2 = year
# row 3 = process type
#
# Expert responses begin at row 4.

metadata_rows = 4

responses = raw.iloc[
    metadata_rows:
].copy()


# ============================================================
# 8. CLEAN RESPONSE VALUES
# ============================================================

def clean_response(value):
    """
    Convert spreadsheet responses to consistent string codes.

    Examples:
        1.0  -> "1"
        AM   -> "AM"
        NaN  -> NaN
    """

    if pd.isna(value):
        return np.nan

    clean_value = (
        str(value)
        .strip()
        .upper()
    )

    if re.fullmatch(
        r"\d+\.0",
        clean_value,
    ):
        clean_value = clean_value.split(
            "."
        )[0]

    return clean_value


# ============================================================
# 9. RESHAPE EXPERT RESPONSES
# ============================================================

long_parts = []


for question, question_info in question_map.items():

    question_columns = [
        column
        for column in raw.columns
        if column.startswith(
            f"{question}_"
        )
    ]

    if not question_columns:

        print(
            f"WARNING: no columns found for {question}"
        )

        continue

    for column in question_columns:

        ice_sheet = str(
            raw.loc[
                0,
                column,
            ]
        ).strip()

        try:

            year = int(
                float(
                    raw.loc[
                        2,
                        column,
                    ]
                )
            )

        except (
            TypeError,
            ValueError,
        ) as error:

            raise ValueError(
                f"Could not interpret the year for "
                f"{column}: {raw.loc[2, column]}"
            ) from error

        selected_processes = (
            responses[column]
            .apply(
                clean_response
            )
        )

        temporary = pd.DataFrame(
            {
                "expert_row": responses.index,
                "question": question,
                "column": column,
                "warming": question_info[
                    "warming"
                ],
                "process_group": question_info[
                    "process_group"
                ],
                "ice_sheet": ice_sheet,
                "year": year,
                "selected_process": selected_processes,
            }
        )

        long_parts.append(
            temporary
        )


if not long_parts:

    raise ValueError(
        "No expert-response columns were found."
    )


long_data = pd.concat(
    long_parts,
    ignore_index=True,
)

long_data = long_data.dropna(
    subset=[
        "selected_process"
    ]
).copy()

long_data["process_label"] = (
    long_data["selected_process"]
    .map(
        response_labels
    )
)


# ============================================================
# 10. VALIDATE RESPONSE CODES
# ============================================================

unknown_responses = (
    long_data.loc[
        long_data["process_label"].isna(),
        [
            "question",
            "column",
            "process_group",
            "selected_process",
        ],
    ]
    .drop_duplicates()
)

if not unknown_responses.empty:

    print(
        "\nWARNING: Unrecognized response codes:"
    )

    print(
        unknown_responses.to_string(
            index=False
        )
    )

else:

    print(
        "\nAll response codes were recognized."
    )


invalid_group_responses = []


for process_group, valid_codes in response_order.items():

    group_subset = long_data.loc[
        long_data["process_group"]
        == process_group
    ]

    invalid = group_subset.loc[
        ~group_subset["selected_process"].isin(
            valid_codes
        )
    ]

    if not invalid.empty:

        invalid_group_responses.append(
            invalid
        )


if invalid_group_responses:

    invalid_group_responses = pd.concat(
        invalid_group_responses,
        ignore_index=True,
    )

    print(
        "\nWARNING: Response codes appeared in an "
        "unexpected question group:"
    )

    print(
        invalid_group_responses[
            [
                "question",
                "column",
                "process_group",
                "selected_process",
            ]
        ]
        .drop_duplicates()
        .to_string(
            index=False
        )
    )


# ============================================================
# 11. COUNT RESPONSES AND CALCULATE UNSCALED PERCENTAGES
# ============================================================

counts = (
    long_data
    .groupby(
        [
            "warming",
            "process_group",
            "ice_sheet",
            "year",
            "selected_process",
        ],
        observed=True,
    )
    .size()
    .reset_index(
        name="n"
    )
)

response_totals = (
    long_data
    .groupby(
        [
            "warming",
            "process_group",
            "ice_sheet",
            "year",
        ],
        observed=True,
    )
    .size()
    .reset_index(
        name="n_responses"
    )
)

plot_data = counts.merge(
    response_totals,
    on=[
        "warming",
        "process_group",
        "ice_sheet",
        "year",
    ],
    how="left",
)

plot_data["percent"] = (
    100.0
    * plot_data["n"]
    / plot_data["n_responses"]
)


# ============================================================
# 12. ADD ZERO-COUNT CATEGORIES
# ============================================================

complete_rows = []


for warming in warming_order:

    for process_group in process_group_order:

        for ice_sheet in ice_sheet_order:

            for year in year_order:

                for selected_process in response_order[
                    process_group
                ]:

                    complete_rows.append(
                        {
                            "warming": warming,
                            "process_group": process_group,
                            "ice_sheet": ice_sheet,
                            "year": year,
                            "selected_process": selected_process,
                        }
                    )


complete_grid = pd.DataFrame(
    complete_rows
)

plot_data = complete_grid.merge(
    plot_data,
    on=[
        "warming",
        "process_group",
        "ice_sheet",
        "year",
        "selected_process",
    ],
    how="left",
)

plot_data["n"] = (
    plot_data["n"]
    .fillna(0)
)

plot_data["percent"] = (
    plot_data["percent"]
    .fillna(0)
)

plot_data["process_label"] = (
    plot_data["selected_process"]
    .map(
        response_labels
    )
)

plot_data = plot_data.drop(
    columns=[
        "n_responses"
    ],
    errors="ignore",
)

plot_data = plot_data.merge(
    response_totals,
    on=[
        "warming",
        "process_group",
        "ice_sheet",
        "year",
    ],
    how="left",
)


# ============================================================
# 13. ATTACH B25 WEIGHTS AND CALCULATE SCALED VALUES
# ============================================================

weight_lookup = (
    scaling_weights[
        [
            "ice_sheet",
            "year",
            "warming_label",
            "median_discharge_mm",
            "median_smb_mm",
            "dynamic_weight",
            "smb_weight",
            "dynamic_percent",
            "smb_percent",
        ]
    ]
    .rename(
        columns={
            "warming_label": "warming",
        }
    )
)

plot_data = plot_data.merge(
    weight_lookup,
    on=[
        "warming",
        "ice_sheet",
        "year",
    ],
    how="left",
    validate="many_to_one",
)

missing_weights = (
    plot_data.loc[
        plot_data["dynamic_weight"].isna(),
        [
            "warming",
            "ice_sheet",
            "year",
        ],
    ]
    .drop_duplicates()
)

if not missing_weights.empty:

    raise ValueError(
        "Missing B25 scaling weights for:\n"
        f"{missing_weights.to_string(index=False)}"
    )


# Only DP and SMB-G are used in the scaled figure.
plot_data["scale_weight"] = np.select(
    [
        plot_data["process_group"] == "DP",
        plot_data["process_group"] == "SMB-G",
    ],
    [
        plot_data["dynamic_weight"],
        plot_data["smb_weight"],
    ],
    default=np.nan,
)

plot_data["scaled_percent"] = (
    plot_data["percent"]
    * plot_data["scale_weight"]
)


# ============================================================
# 14. SORT AND SAVE TABLES
# ============================================================

warming_rank = {
    value: index
    for index, value in enumerate(
        warming_order
    )
}

group_rank = {
    value: index
    for index, value in enumerate(
        process_group_order
    )
}

ice_sheet_rank = {
    value: index
    for index, value in enumerate(
        ice_sheet_order
    )
}

plot_data["_warming_order"] = (
    plot_data["warming"]
    .map(
        warming_rank
    )
)

plot_data["_group_order"] = (
    plot_data["process_group"]
    .map(
        group_rank
    )
)

plot_data["_ice_sheet_order"] = (
    plot_data["ice_sheet"]
    .map(
        ice_sheet_rank
    )
)

plot_data["_response_order"] = (
    plot_data.apply(
        lambda row: response_order[
            row["process_group"]
        ].index(
            row["selected_process"]
        ),
        axis=1,
    )
)

plot_data = (
    plot_data
    .sort_values(
        [
            "_warming_order",
            "_ice_sheet_order",
            "_group_order",
            "year",
            "_response_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


unscaled_table_file = os.path.join(
    table_output_dir,
    "SEJ_driver_response_percentages_unscaled.csv",
)

scaled_table_file = os.path.join(
    table_output_dir,
    "SEJ_DP_SMBG_response_percentages_scaled.csv",
)


unscaled_columns = [
    "warming",
    "ice_sheet",
    "year",
    "process_group",
    "selected_process",
    "process_label",
    "n",
    "n_responses",
    "percent",
]

scaled_columns = [
    "warming",
    "ice_sheet",
    "year",
    "process_group",
    "selected_process",
    "process_label",
    "n",
    "n_responses",
    "percent",
    "median_discharge_mm",
    "median_smb_mm",
    "dynamic_weight",
    "smb_weight",
    "scale_weight",
    "scaled_percent",
]


plot_data[
    unscaled_columns
].to_csv(
    unscaled_table_file,
    index=False,
)

plot_data.loc[
    plot_data["process_group"].isin(
        [
            "DP",
            "SMB-G",
        ]
    ),
    scaled_columns,
].to_csv(
    scaled_table_file,
    index=False,
)

print(
    "\nSaved unscaled response table:\n"
    f"{unscaled_table_file}"
)

print(
    "\nSaved scaled DP/SMB-G table:\n"
    f"{scaled_table_file}"
)


# ============================================================
# 15. PRINT RESPONSE TOTALS
# ============================================================

print(
    "\nNumber of valid responses:"
)

print(
    response_totals
    .sort_values(
        [
            "warming",
            "ice_sheet",
            "process_group",
            "year",
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 16. GENERAL PLOTTING HELPERS
# ============================================================

tick_label_fontsize = 14
y_axis_label_fontsize = 18

def draw_bracket(
    ax,
    left,
    right,
    y,
    height=2.0,
    linewidth=1.1,
):
    """Draw a bracket above a set of bars."""

    ax.plot(
        [
            left,
            left,
            right,
            right,
        ],
        [
            y - height,
            y,
            y,
            y - height,
        ],
        color="black",
        linewidth=linewidth,
        clip_on=False,
    )


def safe_warming_name(warming):
    """Convert a warming label into a filename-safe string."""

    return (
        warming
        .replace(
            "°",
            "",
        )
        .replace(
            ".",
            "p",
        )
    )


def get_bar_height(
    data,
    warming,
    ice_sheet,
    year,
    process_group,
    selected_process,
    value_column,
):
    """Return one plotting value, or zero if no row exists."""

    selection = data.loc[
        (
            data["warming"]
            == warming
        )
        & (
            data["ice_sheet"]
            == ice_sheet
        )
        & (
            data["year"]
            == year
        )
        & (
            data["process_group"]
            == process_group
        )
        & (
            data["selected_process"]
            == selected_process
        ),
        value_column,
    ]

    if selection.empty:
        return 0.0

    value = selection.iloc[0]

    if pd.isna(value):
        return 0.0

    return float(value)


def build_legend_handles(
    group_code_pairs,
    wrap_width=30,
):
    """
    Build a legend containing group headings and wrapped
    response labels.
    """

    handles = []

    for heading, codes in group_code_pairs:

        handles.append(
            Patch(
                facecolor="none",
                edgecolor="none",
                label=heading,
            )
        )

        for code in codes:

            handles.append(
                Patch(
                    facecolor=response_colors[code],
                    edgecolor="none",
                    label=wrap_legend_label(
                        response_labels[code],
                        width=wrap_width,
                    ),
                )
            )

    return handles


def format_legend_headings(
    legend,
    headings,
    heading_fontsize=12.5,
):
    """Make legend group headings larger and bold."""

    for text in legend.get_texts():

        if text.get_text() in headings:

            text.set_fontweight("bold")
            text.set_fontsize(heading_fontsize)

def wrap_legend_label(label, width=30):
    """Wrap long legend labels onto multiple lines."""
    return textwrap.fill(
        label,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )

# ============================================================
# 17. SCALED DP + GROUNDED-SMB FIGURE
#     Combined 100% bars; 2°C and 5°C panels
# ============================================================

def plot_scaled_dp_smbg_combined(
    data,
    warmings=("2°C", "5°C"),
    years=(2100, 2300),
    save=True,
):
    """
    Plot scaled DP + grounded-ice SMB uncertainty drivers.

    Each ice sheet × year is represented by ONE stacked bar.

    Within each bar:
        - DP responses are scaled by the B25 discharge weight.
        - SMB-G responses are scaled by the B25 grounded-SMB weight.
        - DP + SMB-G therefore sum to 100%.

    Existing response colors are retained.

    DP and SMB-G are visually distinguished by pattern:
        - DP: solid fill
        - SMB-G: hatched fill

    Two panels show the 2°C and 5°C warming scenarios.
    Only 2100 and 2300 are plotted.
    """

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(warmings),
        figsize=(14, 7.5),
        sharey=True,
    )

    if len(warmings) == 1:
        axes = [axes]

    bar_width = 0.62
    within_sheet_spacing = 1.3
    between_sheet_gap = 1.3

    # --------------------------------------------------------
    # Loop over warming panels
    # --------------------------------------------------------

    for ax, warming in zip(
        axes,
        warmings,
    ):

        scenario_data = data.loc[
            data["warming"] == warming
        ].copy()

        x_positions = {}
        sheet_centers = {}
        sheet_bounds = {}

        current_x = 0.0

        # ----------------------------------------------------
        # Assign x positions
        # ----------------------------------------------------

        for ice_sheet in ice_sheet_order:

            first_x = current_x

            for year in years:

                x_positions[
                    (
                        ice_sheet,
                        year,
                    )
                ] = current_x

                current_x += within_sheet_spacing

            last_x = x_positions[
                (
                    ice_sheet,
                    years[-1],
                )
            ]

            sheet_bounds[
                ice_sheet
            ] = (
                first_x - bar_width / 1.35,
                last_x + bar_width / 1.35,
            )

            sheet_centers[
                ice_sheet
            ] = (
                first_x + last_x
            ) / 2.0

            current_x += between_sheet_gap

        # ----------------------------------------------------
        # Draw one combined bar for each
        # ice sheet × year
        # ----------------------------------------------------

        for ice_sheet in ice_sheet_order:

            for year in years:

                x = x_positions[
                    (
                        ice_sheet,
                        year,
                    )
                ]

                bottom = 0.0

                # ====================================================
                # DYNAMICAL PROCESSES
                # ====================================================

                for selected_process in response_order["DP"]:

                    height = get_bar_height(
                        data=scenario_data,
                        warming=warming,
                        ice_sheet=ice_sheet,
                        year=year,
                        process_group="DP",
                        selected_process=selected_process,
                        value_column="scaled_percent",
                    )

                    ax.bar(
                        x,
                        height,
                        width=bar_width,
                        bottom=bottom,
                        color=response_colors[
                            selected_process
                        ],
                        edgecolor="white",
                        linewidth=0.6,
                    )

                    bottom += height

                # Save the DP/SMB boundary
                dp_top = bottom

                # ====================================================
                # GROUNDED-ICE SMB
                # ====================================================

                for selected_process in response_order["SMB-G"]:

                    height = get_bar_height(
                        data=scenario_data,
                        warming=warming,
                        ice_sheet=ice_sheet,
                        year=year,
                        process_group="SMB-G",
                        selected_process=selected_process,
                        value_column="scaled_percent",
                    )

                    ax.bar(
                        x,
                        height,
                        width=bar_width,
                        bottom=bottom,
                        color=response_colors[
                            selected_process
                        ],
                        edgecolor="white",
                        linewidth=0.6,
                        hatch="///",
                    )

                    bottom += height

                # ------------------------------------------------
                # Boundary between DP and SMB
                # ------------------------------------------------

                ax.hlines(
                    y=dp_top,
                    xmin=x - bar_width / 2,
                    xmax=x + bar_width / 2,
                    color="black",
                    linewidth=1.0,
                    zorder=5,
                )

        # ----------------------------------------------------
        # X-axis labels
        # ----------------------------------------------------

        ordered_keys = [
            (
                ice_sheet,
                year,
            )
            for ice_sheet in ice_sheet_order
            for year in years
        ]

        ax.set_xticks(
            [
                x_positions[key]
                for key in ordered_keys
            ]
        )

        ax.set_xticklabels(
            [
                str(key[1])
                for key in ordered_keys
            ],
            rotation=0,
            ha="center",
            fontsize=tick_label_fontsize,
        )

        # ----------------------------------------------------
        # Ice-sheet labels and brackets
        # ----------------------------------------------------
        
        for ice_sheet in ice_sheet_order:
        
            left, right = sheet_bounds[
                ice_sheet
            ]
        
            center = sheet_centers[
                ice_sheet
            ]
        
            draw_bracket(
                ax=ax,
                left=left,
                right=right,
                y=105.5,
                height=2.0,
                linewidth=1.2,
            )
        
            ax.text(
                center,
                109,
                ice_sheet,
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
                clip_on=False,
            )
        
        
        # ----------------------------------------------------
        # Warming scenario label
        # ----------------------------------------------------
        
        panel_center = (
            min(x_positions.values())
            + max(x_positions.values())
        ) / 2.0
        
        ax.text(
            panel_center,
            116.5,
            warming,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            clip_on=False,
        )

        # ----------------------------------------------------
        # Axes
        # ----------------------------------------------------

        ax.set_ylim(
            0,
            118,
        )

        ax.set_yticks(
            np.arange(
                0,
                101,
                10,
            )
        )

        ax.yaxis.set_major_formatter(
            PercentFormatter(
                xmax=100,
                decimals=0,
            )
        )

        ax.tick_params(
            axis="y",
            labelsize=tick_label_fontsize,
        )

        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.5,
            alpha=0.5,
        )

        ax.set_axisbelow(True)

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

    # --------------------------------------------------------
    # Shared y-axis label
    # --------------------------------------------------------

    axes[0].set_ylabel(
        "Scaled share of expert responses",
        fontsize=y_axis_label_fontsize,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_handles = []

    # DP heading
    legend_handles.append(
        Patch(
            facecolor="none",
            edgecolor="none",
            label="Dynamical processes",
        )
    )

    for code in response_order["DP"]:

        legend_handles.append(
            Patch(
                facecolor=response_colors[
                    code
                ],
                edgecolor="white",
                linewidth=0.6,
                label=wrap_legend_label(
                    response_labels[
                        code
                    ],
                    width=28,
                ),
            )
        )

    # SMB-G heading
    legend_handles.append(
        Patch(
            facecolor="none",
            edgecolor="none",
            label="Grounded-ice SMB",
        )
    )

    for code in response_order["SMB-G"]:

        legend_handles.append(
            Patch(
                facecolor=response_colors[
                    code
                ],
                edgecolor="white",
                linewidth=0.6,
                hatch="///",
                label=wrap_legend_label(
                    response_labels[
                        code
                    ],
                    width=28,
                ),
            )
        )

    # Put legend directly beside the second panel
    legend = axes[-1].legend(
        handles=legend_handles,
        frameon=False,
        bbox_to_anchor=(
            1.01,
            0.5,
        ),
        loc="center left",
        fontsize=11,
        handlelength=1.3,
        handletextpad=0.6,
        labelspacing=0.75,
        borderaxespad=0.2,
    )

    format_legend_headings(
        legend,
        [
            "Dynamical processes",
            "Grounded-ice SMB",
        ],
        heading_fontsize=12.5,
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout(
        w_pad=2.0,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save:

        output_file = os.path.join(
            figure_output_dir,
            "scaled_DP_SMBG_combined_2C_5C.png",
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            "\nSaved combined scaled DP + SMB-G figure:\n"
            f"{output_file}"
        )

    return (
        fig,
        axes,
    )

# ============================================================
# 18. UNSCALED GROUNDED-ICE + ICE-SHELF SMB FIGURE
# ============================================================

def plot_unscaled_smb(
    data,
    warming,
    save=True,
):
    """
    Plot unscaled grounded-ice and ice-shelf SMB driver responses.

    Each bar sums to 100% of valid expert responses within the
    corresponding question.
    """

    selected_groups = [
        "SMB-G",
        "SMB-S",
    ]

    scenario_data = data.loc[
        data["warming"]
        == warming
    ].copy()

    fig, ax = plt.subplots(
        figsize=(
            15,
            8.5,
        )
    )

    bar_width = 0.68
    gap_between_groups = 0.75
    gap_between_ice_sheets = 2.0

    x_positions = {}
    group_centers = {}
    sheet_centers = {}
    sheet_bounds = {}

    current_x = 0.0


    # --------------------------------------------------------
    # Assign x positions
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        sheet_start = current_x

        for process_group in selected_groups:

            group_start = current_x

            for year in year_order:

                x_positions[
                    (
                        ice_sheet,
                        process_group,
                        year,
                    )
                ] = current_x

                current_x += 1.0

            group_end = (
                current_x
                - 1.0
            )

            group_centers[
                (
                    ice_sheet,
                    process_group,
                )
            ] = (
                group_start
                + group_end
            ) / 2.0

            current_x += gap_between_groups

        current_x -= gap_between_groups

        sheet_end = (
            current_x
            - 1.0
        )

        sheet_bounds[
            ice_sheet
        ] = (
            sheet_start
            - bar_width / 1.4,
            sheet_end
            + bar_width / 1.4,
        )

        sheet_centers[
            ice_sheet
        ] = (
            sheet_start
            + sheet_end
        ) / 2.0

        current_x += gap_between_ice_sheets


    # --------------------------------------------------------
    # Draw bars
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        for process_group in selected_groups:

            for year in year_order:

                x = x_positions[
                    (
                        ice_sheet,
                        process_group,
                        year,
                    )
                ]

                bottom = 0.0

                for selected_process in response_order[
                    process_group
                ]:

                    height = get_bar_height(
                        data=scenario_data,
                        warming=warming,
                        ice_sheet=ice_sheet,
                        year=year,
                        process_group=process_group,
                        selected_process=selected_process,
                        value_column="percent",
                    )

                    ax.bar(
                        x,
                        height,
                        width=bar_width,
                        bottom=bottom,
                        color=response_colors[
                            selected_process
                        ],
                        edgecolor="white",
                        linewidth=0.5,
                    )

                    bottom += height


    # --------------------------------------------------------
    # X-axis labels
    # --------------------------------------------------------

    ordered_keys = [
        (
            ice_sheet,
            process_group,
            year,
        )
        for ice_sheet in ice_sheet_order
        for process_group in selected_groups
        for year in year_order
    ]

    ax.set_xticks(
        [
            x_positions[key]
            for key in ordered_keys
        ]
    )

    ax.set_xticklabels(
        [
            str(key[2])
            for key in ordered_keys
        ],
        rotation=45,
        ha="right",
        fontsize=tick_label_fontsize,
    )


    # --------------------------------------------------------
    # SMB-domain labels
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        for process_group in selected_groups:

            ax.text(
                group_centers[
                    (
                        ice_sheet,
                        process_group,
                    )
                ],
                -2,
                process_group_short_labels[
                    process_group
                ],
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
                clip_on=False,
            )


    # --------------------------------------------------------
    # Ice-sheet labels and brackets
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        left, right = sheet_bounds[
            ice_sheet
        ]

        center = sheet_centers[
            ice_sheet
        ]

        draw_bracket(
            ax=ax,
            left=left,
            right=right,
            y=106,
            height=3,
            linewidth=1.3,
        )

        ax.text(
            center,
            111,
            ice_sheet,
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
        )


    # --------------------------------------------------------
    # Axes and title
    # --------------------------------------------------------

    ax.set_ylim(
        -10,
        117,
    )

    ax.set_ylabel(
        "Share of expert responses",
        fontsize= y_axis_label_fontsize,
        fontweight = "bold",
    )

    ax.set_yticks(
        np.arange(
            0,
            101,
            10,
        )
    )
    
    ax.tick_params(
        axis="y",
        labelsize=tick_label_fontsize,
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=100,
            decimals=0,
        )
    )

    ax.text(
        0.98,
        0.98,
        warming,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.5,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    handles = build_legend_handles(
        [
            (
                "SMB processes",
                [
                    "AM",
                    "AC",
                    "SI",
                ],
            ),
        ],
        wrap_width=30,
    )
    
    legend = ax.legend(
        handles=handles,
        frameon=False,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        fontsize=11.5,
        handlelength=1.2,
        handletextpad=0.7,
        labelspacing=0.9,
    )
    
    format_legend_headings(
        legend,
        [
            "SMB processes",
        ],
        heading_fontsize=13.5,
    )


    if save:

        output_file = os.path.join(
            figure_output_dir,
            (
                "unscaled_SMB_grounded_shelves_"
                f"{safe_warming_name(warming)}.png"
            ),
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            "\nSaved unscaled SMB figure:\n"
            f"{output_file}"
        )

    return (
        fig,
        ax,
    )


# ============================================================
# 19. UNSCALED OCEAN-PROCESS FIGURE
# ============================================================

def plot_unscaled_ocean(
    data,
    warming,
    save=True,
):
    """
    Plot unscaled ocean-process driver responses.

    Each bar sums to 100% of valid expert responses.
    """

    process_group = "OP"

    scenario_data = data.loc[
        (
            data["warming"]
            == warming
        )
        & (
            data["process_group"]
            == process_group
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            7.5,
        )
    )

    bar_width = 0.68
    gap_between_ice_sheets = 1.8

    x_positions = {}
    sheet_centers = {}
    sheet_bounds = {}

    current_x = 0.0


    # --------------------------------------------------------
    # Assign x positions
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        first_x = current_x

        for year in year_order:

            x_positions[
                (
                    ice_sheet,
                    year,
                )
            ] = current_x

            current_x += 1.0

        last_x = (
            current_x
            - 1.0
        )

        sheet_bounds[
            ice_sheet
        ] = (
            first_x
            - bar_width / 1.5,
            last_x
            + bar_width / 1.5,
        )

        sheet_centers[
            ice_sheet
        ] = (
            first_x
            + last_x
        ) / 2.0

        current_x += (
            gap_between_ice_sheets
        )


    # --------------------------------------------------------
    # Draw bars
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        for year in year_order:

            x = x_positions[
                (
                    ice_sheet,
                    year,
                )
            ]

            bottom = 0.0

            for selected_process in response_order[
                process_group
            ]:

                height = get_bar_height(
                    data=scenario_data,
                    warming=warming,
                    ice_sheet=ice_sheet,
                    year=year,
                    process_group=process_group,
                    selected_process=selected_process,
                    value_column="percent",
                )

                ax.bar(
                    x,
                    height,
                    width=bar_width,
                    bottom=bottom,
                    color=response_colors[
                        selected_process
                    ],
                    edgecolor="white",
                    linewidth=0.5,
                )

                bottom += height


    # --------------------------------------------------------
    # X-axis labels
    # --------------------------------------------------------

    ordered_keys = [
        (
            ice_sheet,
            year,
        )
        for ice_sheet in ice_sheet_order
        for year in year_order
    ]

    ax.set_xticks(
        [
            x_positions[key]
            for key in ordered_keys
        ]
    )

    ax.set_xticklabels(
        [
            str(key[1])
            for key in ordered_keys
        ],
        rotation=45,
        ha="right",
        fontsize=tick_label_fontsize,
    )


    # --------------------------------------------------------
    # Ice-sheet labels and brackets
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        left, right = sheet_bounds[
            ice_sheet
        ]

        center = sheet_centers[
            ice_sheet
        ]

        draw_bracket(
            ax=ax,
            left=left,
            right=right,
            y=106,
            height=3,
            linewidth=1.2,
        )

        ax.text(
            center,
            111,
            ice_sheet,
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
        )


    # --------------------------------------------------------
    # Axes and title
    # --------------------------------------------------------

    ax.set_ylim(
        0,
        117,
    )

    ax.set_ylabel(
        "Share of expert responses",
        fontsize= y_axis_label_fontsize,
        fontweight = 'bold',
    )

    ax.set_yticks(
        np.arange(
            0,
            101,
            10,
        )
    )
    
    ax.tick_params(
        axis="y",
        labelsize=tick_label_fontsize,
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=100,
            decimals=0,
        )
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.5,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    handles = build_legend_handles(
        [
            (
                "Ocean processes",
                response_order["OP"],
            ),
        ],
        wrap_width=30,
    )
    
    legend = ax.legend(
        handles=handles,
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=11.5,
        handlelength=1.2,
        handletextpad=0.7,
        labelspacing=0.9,
    )
    
    format_legend_headings(
        legend,
        [
            "Ocean processes",
        ],
        heading_fontsize=13.5,
    )

    plt.tight_layout()


    if save:

        output_file = os.path.join(
            figure_output_dir,
            (
                "unscaled_ocean_processes_"
                f"{safe_warming_name(warming)}.png"
            ),
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            "\nSaved unscaled ocean-process figure:\n"
            f"{output_file}"
        )

    return (
        fig,
        ax,
    )


# ============================================================
# 20. CREATE ALL FIGURES
# ============================================================

# for warming in warming_order:

#     # Scaled dynamical-process and grounded-SMB figure
#     fig, ax = plot_scaled_dp_smbg(
#         data=plot_data,
#         warming=warming,
#         save=True,
#     )

#     plt.show()
#     plt.close(
#         fig
#     )


#     # Unscaled grounded-ice and ice-shelf SMB figure
#     fig, ax = plot_unscaled_smb(
#         data=plot_data,
#         warming=warming,
#         save=True,
#     )

#     plt.show()
#     plt.close(
#         fig
#     )


#     # Unscaled ocean-process figure
#     fig, ax = plot_unscaled_ocean(
#         data=plot_data,
#         warming=warming,
#         save=True,
#     )

#     plt.show()
#     plt.close(
#         fig
#     )

# Combined scaled DP + SMB-G figure:
# 2°C and 5°C only; 2100 and 2300 only

fig, axes = plot_scaled_dp_smbg_combined(
    data=plot_data,
    warmings=("2°C", "5°C"),
    years=(2100, 2300),
    save=True,
)

plt.show()
plt.close(fig)


# ============================================================
# 21. OPTIONAL EXAMPLE OUTPUT
# ============================================================

example = plot_data.loc[
    (
        plot_data["warming"]
        == "5°C"
    )
    & (
        plot_data["ice_sheet"]
        == "WAIS"
    )
    & (
        plot_data["year"]
        == 2100
    ),
    [
        "warming",
        "ice_sheet",
        "year",
        "process_group",
        "selected_process",
        "process_label",
        "n",
        "n_responses",
        "percent",
        "dynamic_weight",
        "smb_weight",
        "scale_weight",
        "scaled_percent",
    ],
]

print(
    "\nExample output: WAIS, 2100, 5°C\n"
)

print(
    example.to_string(
        index=False
    )
)