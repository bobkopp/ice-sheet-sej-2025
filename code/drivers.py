#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:17:14 2026

@author: elizabeth
"""


# ============================================================
# EXPERT JUDGMENTS OF ICE-SHEET UNCERTAINTY DRIVERS
#
# Question groups:
#
#   Dynamical processes:
#       Q5  = 2°C
#       Q7  = 3.5°C
#       Q9  = 5°C
#
#   Surface mass balance of grounded ice:
#       Q12 = 2°C
#       Q13 = 3.5°C
#       Q14 = 5°C
#
#   Surface mass balance and thinning of ice shelves:
#       Q16 = 2°C
#       Q17 = 3.5°C
#       Q18 = 5°C
#
#   Ocean processes:
#       Q20 = 2°C
#       Q21 = 3.5°C
#       Q22 = 5°C
#
# Output:
#   1. One combined B22-style figure per warming scenario
#   2. One separate figure per process category and warming scenario
#   3. CSV table of response counts and percentages
# ============================================================


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter



# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

input_file = (
    "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025/data/ICE+SHEET+SEJ+target+questions_July+15,+2026_CLEANER_LIZ.csv"
)

output_dir = "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025/output/figures"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. BASIC SETTINGS
# ============================================================

ice_sheet_order = ["GrIS", "WAIS", "EAIS"]
year_order = [2100, 2200, 2300]
warming_order = ["2°C", "3.5°C", "5°C"]

process_group_order = [
    "DP",
    "SMB-G",
    "SMB-S",
    "OP",
]

process_group_titles = {
    "DP": "Dynamical processes",
    "SMB-G": "Grounded-ice SMB",
    "SMB-S": "Ice-shelf SMB/thinning",
    "OP": "Ocean processes",
}


# ============================================================
# 3. QUESTION DEFINITIONS
# ============================================================

question_map = {
    # --------------------------------------------------------
    # Dynamical processes
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Surface mass balance of grounded ice
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Surface mass balance and thinning of ice shelves
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Ocean processes
    # --------------------------------------------------------
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
# 4. RESPONSE LABELS
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
    "DP": ["1", "2", "3", "4", "5", "6"],
    "SMB-G": ["AM", "AC"],
    "SMB-S": ["AM", "AC", "SI"],
    "OP": ["MOC", "CDW", "ACC"],
}


# ============================================================
# 5. COLORS
# ============================================================

# These are chosen to roughly resemble the distinctions in the
# B22 figure while keeping related process groups recognizable.

response_colors = {
    # DP
    "1": "#3973AC",
    "2": "#B5423D",
    "3": "#8AAA43",
    "4": "#6A4C93",
    "5": "#3E9AAF",
    "6": "#E68532",

    # SMB
    "AM": "#8DA7CF",
    "AC": "#CE9191",
    "SI": "#67A9CF",

    # OP
    "MOC": "#446CB3",
    "CDW": "#55A868",
    "ACC": "#C17C3A",
}


# ============================================================
# 6. READ THE CSV
# ============================================================

raw = pd.read_csv(input_file)

print("CSV dimensions:", raw.shape)
print("Number of question columns:", len(raw.columns) - 1)


# The first four rows contain:
#
# row 0 = ice sheet
# row 1 = warming scenario
# row 2 = year
# row 3 = process type
#
# Expert responses begin at row 4.

metadata_rows = 4
responses = raw.iloc[metadata_rows:].copy()


# ============================================================
# 7. HELPER FUNCTION FOR CLEANING RESPONSES
# ============================================================

def clean_response(value):
    """
    Convert spreadsheet responses into consistent string codes.

    Examples:
        1.0 -> "1"
        "AM " -> "AM"
        NaN -> NaN
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    # Convert values such as "1.0" to "1"
    if re.fullmatch(r"\d+\.0", value):
        value = value.split(".")[0]

    return value


# ============================================================
# 8. RESHAPE THE DATA
# ============================================================

long_parts = []

for question, question_info in question_map.items():

    question_columns = [
        column
        for column in raw.columns
        if column.startswith(f"{question}_")
    ]

    if len(question_columns) == 0:
        print(f"Warning: no columns found for {question}")
        continue

    for column in question_columns:

        ice_sheet = str(raw.loc[0, column]).strip()

        try:
            year = int(float(raw.loc[2, column]))
        except (TypeError, ValueError):
            raise ValueError(
                f"Could not interpret year for column {column}: "
                f"{raw.loc[2, column]}"
            )

        selected_processes = responses[column].apply(clean_response)

        temp = pd.DataFrame(
            {
                "expert_row": responses.index,
                "question": question,
                "column": column,
                "warming": question_info["warming"],
                "process_group": question_info["process_group"],
                "ice_sheet": ice_sheet,
                "year": year,
                "selected_process": selected_processes,
            }
        )

        long_parts.append(temp)


long_data = pd.concat(long_parts, ignore_index=True)

# Remove missing responses
long_data = long_data.dropna(subset=["selected_process"]).copy()

# Add full descriptive labels
long_data["process_label"] = (
    long_data["selected_process"].map(response_labels)
)


# ============================================================
# 9. VALIDATE RESPONSE CODES
# ============================================================

unknown_responses = long_data[
    long_data["process_label"].isna()
][
    [
        "question",
        "column",
        "process_group",
        "selected_process",
    ]
].drop_duplicates()


if len(unknown_responses) > 0:

    print("\nWARNING: Unrecognized response codes were found:")
    print(unknown_responses.to_string(index=False))

else:

    print("\nAll response codes were recognized.")


# Check whether responses are valid within each process group

invalid_group_responses = []

for process_group, valid_codes in response_order.items():

    subset = long_data[
        long_data["process_group"] == process_group
    ]

    invalid = subset[
        ~subset["selected_process"].isin(valid_codes)
    ]

    if len(invalid) > 0:
        invalid_group_responses.append(invalid)


if invalid_group_responses:

    invalid_group_responses = pd.concat(
        invalid_group_responses,
        ignore_index=True,
    )

    print("\nWARNING: Some codes appeared in an unexpected group:")
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
        .to_string(index=False)
    )


# ============================================================
# 10. COUNT RESPONSES AND CALCULATE PERCENTAGES
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
    .reset_index(name="n")
)


totals = (
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
    .reset_index(name="n_responses")
)


plot_data = counts.merge(
    totals,
    on=[
        "warming",
        "process_group",
        "ice_sheet",
        "year",
    ],
    how="left",
)


plot_data["percent"] = (
    100
    * plot_data["n"]
    / plot_data["n_responses"]
)


plot_data["process_label"] = (
    plot_data["selected_process"].map(response_labels)
)


# ============================================================
# 11. ADD ZERO-COUNT RESPONSE CATEGORIES
# ============================================================

complete_rows = []

for warming in warming_order:

    for process_group in process_group_order:

        for ice_sheet in ice_sheet_order:

            for year in year_order:

                for selected_process in response_order[process_group]:

                    complete_rows.append(
                        {
                            "warming": warming,
                            "process_group": process_group,
                            "ice_sheet": ice_sheet,
                            "year": year,
                            "selected_process": selected_process,
                        }
                    )


complete_grid = pd.DataFrame(complete_rows)


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


plot_data["n"] = plot_data["n"].fillna(0)
plot_data["percent"] = plot_data["percent"].fillna(0)

plot_data["process_label"] = (
    plot_data["selected_process"].map(response_labels)
)


# Add response totals back to zero-count rows
plot_data = plot_data.drop(
    columns="n_responses",
    errors="ignore",
)

plot_data = plot_data.merge(
    totals,
    on=[
        "warming",
        "process_group",
        "ice_sheet",
        "year",
    ],
    how="left",
)


# ============================================================
# 12. SORT AND SAVE SUMMARY TABLE
# ============================================================

warming_rank = {
    warming: i
    for i, warming in enumerate(warming_order)
}

group_rank = {
    group: i
    for i, group in enumerate(process_group_order)
}

ice_sheet_rank = {
    ice_sheet: i
    for i, ice_sheet in enumerate(ice_sheet_order)
}


plot_data["_warming_order"] = (
    plot_data["warming"].map(warming_rank)
)

plot_data["_group_order"] = (
    plot_data["process_group"].map(group_rank)
)

plot_data["_ice_sheet_order"] = (
    plot_data["ice_sheet"].map(ice_sheet_rank)
)


plot_data["_response_order"] = plot_data.apply(
    lambda row: response_order[row["process_group"]].index(
        row["selected_process"]
    ),
    axis=1,
)


plot_data = plot_data.sort_values(
    [
        "_warming_order",
        "_ice_sheet_order",
        "_group_order",
        "year",
        "_response_order",
    ]
).reset_index(drop=True)


summary_columns = [
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


summary_file = os.path.join(
    output_dir,
    "SEJ_process_response_percentages.csv",
)

plot_data[summary_columns].to_csv(
    summary_file,
    index=False,
)

print(f"\nSaved summary table:\n{summary_file}")


# ============================================================
# 13. PRINT RESPONSE TOTALS
# ============================================================

print("\nNumber of valid responses in each bar:")

print(
    totals.sort_values(
        [
            "warming",
            "ice_sheet",
            "process_group",
            "year",
        ]
    ).to_string(index=False)
)


# ============================================================
# 14. HELPER FUNCTION: DRAW A BRACKET
# ============================================================

def draw_bracket(
    ax,
    left,
    right,
    y,
    height=2.0,
    linewidth=1.1,
):
    """
    Draw a simple bracket over a group of bars.
    """

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


# ============================================================
# 15. COMBINED B22-STYLE FIGURE
# ============================================================

def plot_combined_figure(
    data,
    warming,
    save=True,
):
    """
    Create one combined figure for a warming scenario.

    For each ice sheet, the figure contains four process blocks:

        DP
        SMB-G
        SMB-S
        OP

    Each block contains one bar for 2100, 2200, and 2300.
    Each stacked bar sums to 100%.
    """

    scenario_data = data[
        data["warming"] == warming
    ].copy()

    fig, ax = plt.subplots(figsize=(19, 9))


    # --------------------------------------------------------
    # Bar-position settings
    # --------------------------------------------------------

    bar_width = 0.68

    gap_between_process_groups = 0.70
    gap_between_ice_sheets = 2.0

    x_positions = {}

    process_bounds = {}
    process_centers = {}

    ice_sheet_bounds = {}
    ice_sheet_centers = {}

    current_x = 0.0


    # --------------------------------------------------------
    # Assign x positions
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        ice_start = current_x

        for process_group in process_group_order:

            process_start = current_x

            for year in year_order:

                x_positions[
                    (
                        ice_sheet,
                        process_group,
                        year,
                    )
                ] = current_x

                current_x += 1.0

            process_end = current_x - 1.0

            process_bounds[
                (
                    ice_sheet,
                    process_group,
                )
            ] = (
                process_start - bar_width / 1.6,
                process_end + bar_width / 1.6,
            )

            process_centers[
                (
                    ice_sheet,
                    process_group,
                )
            ] = (
                process_start + process_end
            ) / 2

            current_x += gap_between_process_groups

        # Remove final within-sheet gap before calculating bounds
        current_x -= gap_between_process_groups

        ice_end = current_x - 1.0

        ice_sheet_bounds[ice_sheet] = (
            ice_start - bar_width / 1.4,
            ice_end + bar_width / 1.4,
        )

        ice_sheet_centers[ice_sheet] = (
            ice_start + ice_end
        ) / 2

        current_x += gap_between_ice_sheets


    # --------------------------------------------------------
    # Draw stacked bars
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        for process_group in process_group_order:

            for year in year_order:

                x = x_positions[
                    (
                        ice_sheet,
                        process_group,
                        year,
                    )
                ]

                bottom = 0.0

                for selected_process in response_order[process_group]:

                    subset = scenario_data[
                        (
                            scenario_data["ice_sheet"]
                            == ice_sheet
                        )
                        & (
                            scenario_data["process_group"]
                            == process_group
                        )
                        & (
                            scenario_data["year"]
                            == year
                        )
                        & (
                            scenario_data["selected_process"]
                            == selected_process
                        )
                    ]

                    if len(subset) == 0:
                        height = 0.0
                    else:
                        height = subset["percent"].iloc[0]

                    ax.bar(
                        x,
                        height,
                        width=bar_width,
                        bottom=bottom,
                        color=response_colors[selected_process],
                        edgecolor="white",
                        linewidth=0.45,
                    )

                    bottom += height


    # --------------------------------------------------------
    # X-axis year labels
    # --------------------------------------------------------

    ordered_keys = []

    for ice_sheet in ice_sheet_order:
        for process_group in process_group_order:
            for year in year_order:

                ordered_keys.append(
                    (
                        ice_sheet,
                        process_group,
                        year,
                    )
                )


    ordered_positions = [
        x_positions[key]
        for key in ordered_keys
    ]

    ordered_year_labels = [
        str(key[2])
        for key in ordered_keys
    ]


    ax.set_xticks(ordered_positions)

    ax.set_xticklabels(
        ordered_year_labels,
        rotation=60,
        ha="right",
        fontsize=8.5,
    )


    # --------------------------------------------------------
    # Process-group labels
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        for process_group in process_group_order:

            center = process_centers[
                (
                    ice_sheet,
                    process_group,
                )
            ]

            ax.text(
                center,
                -15,
                process_group,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                clip_on=False,
            )


    # --------------------------------------------------------
    # Ice-sheet labels and brackets
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        left, right = ice_sheet_bounds[ice_sheet]
        center = ice_sheet_centers[ice_sheet]

        draw_bracket(
            ax=ax,
            left=left,
            right=right,
            y=108,
            height=3,
            linewidth=1.3,
        )

        ax.text(
            center,
            112,
            ice_sheet,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_ylim(-20, 118)

    ax.set_ylabel(
        "Share of expert responses",
        fontsize=13,
    )

    ax.set_yticks(np.arange(0, 101, 10))

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=100,
            decimals=0,
        )
    )

    ax.set_title(
        (
            "Expert judgments of uncertainty drivers\n"
            f"{warming} warming scenario"
        ),
        fontsize=17,
        pad=28,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="y",
        labelsize=11,
    )


    # --------------------------------------------------------
    # Build legend
    # --------------------------------------------------------

    legend_handles = []

    legend_sequence = [
        ("DP", response_order["DP"]),
        ("SMB", ["AM", "AC", "SI"]),
        ("OP", response_order["OP"]),
    ]


    for legend_group, codes in legend_sequence:

        # Dummy heading
        legend_handles.append(
            Patch(
                facecolor="none",
                edgecolor="none",
                label=legend_group,
            )
        )

        for code in codes:

            legend_handles.append(
                Patch(
                    facecolor=response_colors[code],
                    edgecolor="none",
                    label=response_labels[code],
                )
            )


    legend = ax.legend(
        handles=legend_handles,
        frameon=False,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        fontsize=10,
        handlelength=1.1,
        handletextpad=0.5,
        labelspacing=0.75,
    )


    # Make legend group headings bold
    for text in legend.get_texts():

        if text.get_text() in ["DP", "SMB", "OP"]:
            text.set_fontweight("bold")


    plt.tight_layout()


    if save:

        safe_warming = (
            warming
            .replace("°", "")
            .replace(".", "p")
        )

        output_file = os.path.join(
            output_dir,
            (
                "combined_uncertainty_drivers_"
                f"{safe_warming}.png"
            ),
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Saved combined figure:\n{output_file}")


    return fig, ax


# ============================================================
# 16. SEPARATE PROCESS-GROUP FIGURE
# ============================================================

def plot_process_group(
    data,
    warming,
    process_group,
    save=True,
):
    """
    Create a B22-style plot for one process group and one
    warming scenario.

    Each ice sheet has bars for 2100, 2200, and 2300.
    """

    subset_data = data[
        (data["warming"] == warming)
        & (data["process_group"] == process_group)
    ].copy()


    fig, ax = plt.subplots(figsize=(11.5, 7.5))


    bar_width = 0.68
    group_gap = 1.8

    x_positions = {}
    group_bounds = {}
    group_centers = {}

    current_x = 0.0


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

        last_x = current_x - 1.0

        group_bounds[ice_sheet] = (
            first_x - bar_width / 1.5,
            last_x + bar_width / 1.5,
        )

        group_centers[ice_sheet] = (
            first_x + last_x
        ) / 2

        current_x += group_gap


    # --------------------------------------------------------
    # Plot bars
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

            for selected_process in response_order[process_group]:

                row = subset_data[
                    (
                        subset_data["ice_sheet"]
                        == ice_sheet
                    )
                    & (
                        subset_data["year"]
                        == year
                    )
                    & (
                        subset_data["selected_process"]
                        == selected_process
                    )
                ]

                if len(row) == 0:
                    height = 0.0
                else:
                    height = row["percent"].iloc[0]

                ax.bar(
                    x,
                    height,
                    width=bar_width,
                    bottom=bottom,
                    color=response_colors[selected_process],
                    edgecolor="white",
                    linewidth=0.5,
                )

                bottom += height


    # --------------------------------------------------------
    # X-axis
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
        fontsize=10,
    )


    # --------------------------------------------------------
    # Ice-sheet headings
    # --------------------------------------------------------

    for ice_sheet in ice_sheet_order:

        left, right = group_bounds[ice_sheet]
        center = group_centers[ice_sheet]

        draw_bracket(
            ax,
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
    # Axis formatting
    # --------------------------------------------------------

    ax.set_ylim(0, 117)

    ax.set_ylabel(
        "Share of expert responses",
        fontsize=12,
    )

    ax.set_yticks(np.arange(0, 101, 10))

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=100,
            decimals=0,
        )
    )

    ax.set_title(
        (
            f"{process_group_titles[process_group]}\n"
            f"{warming} warming scenario"
        ),
        fontsize=16,
        pad=28,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    handles = [
        Patch(
            facecolor=response_colors[code],
            edgecolor="none",
            label=response_labels[code],
        )
        for code in response_order[process_group]
    ]


    ax.legend(
        handles=handles,
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=10,
    )


    plt.tight_layout()


    if save:

        safe_warming = (
            warming
            .replace("°", "")
            .replace(".", "p")
        )

        output_file = os.path.join(
            output_dir,
            (
                f"{process_group}_"
                f"{safe_warming}.png"
            ),
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Saved process figure:\n{output_file}")


    return fig, ax


# ============================================================
# 17. CREATE THE COMBINED FIGURES
# ============================================================

for warming in warming_order:

    fig, ax = plot_combined_figure(
        data=plot_data,
        warming=warming,
        save=True,
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 18. CREATE THE SEPARATE FIGURES
# ============================================================

make_separate_figures = True

if make_separate_figures:

    for warming in warming_order:

        for process_group in process_group_order:

            fig, ax = plot_process_group(
                data=plot_data,
                warming=warming,
                process_group=process_group,
                save=True,
            )

            plt.show()
            plt.close(fig)


# ============================================================
# 19. OPTIONAL: INSPECT ONE SUMMARY
# ============================================================

example = plot_data[
    (plot_data["warming"] == "5°C")
    & (plot_data["ice_sheet"] == "WAIS")
    & (plot_data["year"] == 2100)
][summary_columns]


print(
    "\nExample summary: WAIS, 2100, 5°C\n"
)

print(
    example.to_string(index=False)
)