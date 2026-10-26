#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
======================================================================
NORMALIZED SCENARIO-SELECTION PROBABILITY CONTOURS
CMIP6 + CMIP7 — CURRENT DOUBLE-CHECKPOINT METHOD
======================================================================

Creates two standalone figures:
    1) CMIP6
    2) CMIP7

Each figure has THREE LARGE VERTICALLY STACKED PANELS:
    P(LOW)
    P(INTERMEDIATE)
    P(HIGH)

Axes are the CURRENT normalized cumulative-warming checkpoints:
    x = normalized cumulative warming through 2100
    y = normalized cumulative warming through 2200

Normalization:
    x = (C2100 - LOW2100) / (HIGH2100 - LOW2100)
    y = (C2200 - LOW2200) / (HIGH2200 - LOW2200)

Therefore:
    LOW reference  = (0, 0)
    HIGH reference = (1, 1)

The INTERMEDIATE reference lies between them.

Current scenario-selection method:
    - range-normalized Euclidean distance in [2100, 2200] checkpoint space
    - inverse-distance weighting, p = 1
    - DOUBLE-CHECKPOINT endpoint truncation

Truncation:
    LOW is forced ONLY when x <= 0 AND y <= 0.
    HIGH is forced ONLY when x >= 1 AND y >= 1.
    Everywhere else, ordinary probabilistic IDW weights are retained.

Scatter points are the actual climate realizations and are colored by:
    CMIP6 -> SSP
    CMIP7 -> pathway

The contours show the analytical EFFECTIVE selection probabilities after
applying the current double-checkpoint truncation rule.

Run in Spyder with F5, or:
    python PLOT_NORMALIZED_SELECTION_CONTOURS_CURRENT_METHOD.py
======================================================================

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ======================================================================
# 1. USER SETTINGS
# ======================================================================

PROJECT_DIR = Path(os.environ.get(
    "SEJ_PROJECT_DIR", Path(__file__).resolve().parents[3]
)).expanduser().resolve()
CORE_OUTPUT_ROOT = Path(os.environ.get(
    "SEJ_CORE_OUTPUT_ROOT",
    PROJECT_DIR / "output" / "CORE_METHOD_CHAIN_02I_B_FINAL_TENSION_P003_FAIR2"
)).expanduser().resolve()
FIGURE_OUTPUT_ROOT = Path(os.environ.get(
    "FIG_OUTPUT_DIR", PROJECT_DIR / "output" / "manuscript_figures"
)).expanduser().resolve()

PLOT_CMIP6 = True
PLOT_CMIP7 = True

SHOW_FIGURES = True
SAVE_PDF = False
DPI = 300

# ----------------------------------------------------------------------
# Manuscript-sized vertical layout
# ----------------------------------------------------------------------
#
# 8 x 10 was a little too compressed once the title, subtitle, legend,
# and three panel titles were included.
#
FIGSIZE = (8.0, 11.5)

# Scatter appearance
POINT_SIZE = 5
POINT_ALPHA = 0.18

# Shared normalized view
X_LIM = (-0.60, 1.60)
Y_LIM = (-0.60, 1.60)

# Probability contours
CONTOUR_LEVELS = (0.2, 0.4, 0.6, 0.8)

# Analytical grid
GRID_N = 500

OUTPUT_DIR = Path(os.environ.get("SEJ_FIGURE_GROUP_OUTPUT") or (FIGURE_OUTPUT_ROOT / "supplementary" / "normalized_selection_probability_contours")).expanduser().resolve()

# The verified Stage 01 and 01B outputs are the only inputs.
CMIP6_SELECTION_CANDIDATES = [
    CORE_OUTPUT_ROOT / "01_warming_selection" / "fair_b25_baseline_scenario_selection.csv"
]
CMIP7_SELECTION_CANDIDATES = [
    CORE_OUTPUT_ROOT / "01_warming_selection" / "cmip7_fair_b25_baseline_scenario_selection.csv"
]


# ======================================================================
# 3. SCENARIO ORDER / LABELS / COLORS
# ======================================================================

CMIP6_ORDER = (
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

CMIP6_COLORS = {
    "ssp119": "#0072B2",
    "ssp126": "#56B4E9",
    "ssp245": "#009E73",
    "ssp370": "#E69F00",
    "ssp534-over": "#CC79A7",
    "ssp585": "#D55E00",
}


CMIP7_ORDER = (
    "VL",
    "LN",
    "L",
    "ML",
    "M",
    "HL",
    "H",
)

CMIP7_LABELS = {
    "VL": "Very Low (VL)",
    "LN": "Low-to-Negative (LN)",
    "L": "Low (L)",
    "ML": "Medium-to-Low (ML)",
    "M": "Medium (M)",
    "HL": "High-to-Low (HL)",
    "H": "High (H)",
}

CMIP7_COLORS = {
    "VL": "#332288",
    "LN": "#0072B2",
    "L": "#56B4E9",
    "ML": "#009E73",
    "M": "#E69F00",
    "HL": "#CC79A7",
    "H": "#D55E00",
}


# ======================================================================
# 4. NORMALIZED REFERENCE POINTS
# ======================================================================

# Current cumulative-warming anchors [K yr]

LOW_2100 = 155.0
INT_2100 = 204.5
HIGH_2100 = 262.0

LOW_2200 = 355.0
INT_2200 = 554.5
HIGH_2200 = 762.0


REFERENCE_POINTS = {
    "LOW": (
        0.0,
        0.0,
    ),

    "INTERMEDIATE": (
        (INT_2100 - LOW_2100)
        / (HIGH_2100 - LOW_2100),

        (INT_2200 - LOW_2200)
        / (HIGH_2200 - LOW_2200),
    ),

    "HIGH": (
        1.0,
        1.0,
    ),
}


SEJ_SCENARIOS = (
    "LOW",
    "INTERMEDIATE",
    "HIGH",
)


# ----------------------------------------------------------------------
# Reference-label offsets
# ----------------------------------------------------------------------
#
# These are deliberately different so LOW / INTERMEDIATE / HIGH labels
# do not sit directly on top of contour labels.
#
REFERENCE_LABEL_OFFSETS = {
    "LOW": (6, 5),
    "INTERMEDIATE": (7, 9),
    "HIGH": (6, 6),
}


# ======================================================================
# 5. HELPERS
# ======================================================================

def first_existing(
    candidates,
    label,
):
    for path in candidates:

        if path.is_file():
            return path

    attempted = "\n".join(
        f"  {p}"
        for p in candidates
    )

    raise FileNotFoundError(
        f"Could not find current {label} selection CSV. Tried:\n"
        f"{attempted}"
    )


def load_selection(
    path,
    expected_pathways,
    family,
):

    df = pd.read_csv(
        path
    )

    df["ssp"] = (
        df["ssp"]
        .astype(str)
        .str.strip()
    )

    required = {
        "ssp",
        "reference_range_position_2100",
        "reference_range_position_2200",
        "weight_low",
        "weight_intermediate",
        "weight_high",
        "sampling_method",
        "truncation_rule",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{family}: missing required columns: "
            f"{sorted(missing)}"
        )


    expected_method = (
        "range_normalized_checkpoint_euclidean_idw_p1_"
        "double_checkpoint_truncation"
    )

    expected_rule = (
        "outside_same_direction_at_both_2100_and_2200"
    )


    if not (
        df["sampling_method"]
        .astype(str)
        .eq(expected_method)
        .all()
    ):

        found = sorted(
            df["sampling_method"]
            .astype(str)
            .unique()
        )

        raise ValueError(
            f"{family}: wrong scenario-selection method: "
            f"{found}"
        )


    if not (
        df["truncation_rule"]
        .astype(str)
        .eq(expected_rule)
        .all()
    ):

        found = sorted(
            df["truncation_rule"]
            .astype(str)
            .unique()
        )

        raise ValueError(
            f"{family}: wrong truncation rule: "
            f"{found}"
        )


    found_pathways = set(
        df["ssp"].unique()
    )

    expected = set(
        expected_pathways
    )


    if found_pathways != expected:

        raise ValueError(
            f"{family}: unexpected pathway set.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found:    {sorted(found_pathways)}"
        )


    return df


def analytical_effective_probabilities():
    """
    Analytical IDW p=1 probabilities in normalized checkpoint space,
    followed by the CURRENT double-checkpoint truncation rule.
    """

    x = np.linspace(
        X_LIM[0],
        X_LIM[1],
        GRID_N,
    )

    y = np.linspace(
        Y_LIM[0],
        Y_LIM[1],
        GRID_N,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )


    pts = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
        ]
    )


    refs = np.array(
        [
            REFERENCE_POINTS[s]
            for s in SEJ_SCENARIOS
        ],
        dtype=float,
    )


    # ----------------------------------------------------------
    # Euclidean distance in normalized 2100 / 2200 space
    # ----------------------------------------------------------

    d = np.sqrt(
        (
            (
                pts[:, None, :]
                - refs[None, :, :]
            )
            ** 2
        )
        .sum(axis=2)
    )


    # ----------------------------------------------------------
    # IDW p = 1
    # ----------------------------------------------------------

    weights = np.zeros_like(
        d
    )


    # Handle exact reference locations robustly.

    exact = (
        d <= 1e-12
    )

    exact_rows = (
        exact.any(axis=1)
    )


    if exact_rows.any():

        exact_index = (
            exact[exact_rows]
            .argmax(axis=1)
        )

        weights[
            exact_rows,
            exact_index,
        ] = 1.0


    nonexact_rows = (
        ~exact_rows
    )


    inv = (
        1.0
        / d[nonexact_rows]
    )

    weights[
        nonexact_rows
    ] = (
        inv
        / inv.sum(
            axis=1,
            keepdims=True,
        )
    )


    # ----------------------------------------------------------
    # CURRENT DOUBLE-CHECKPOINT ENDPOINT TRUNCATION
    # ----------------------------------------------------------
    #
    # LOW forced only if below/equal LOW at BOTH checkpoints.
    #
    # HIGH forced only if above/equal HIGH at BOTH checkpoints.
    #
    # Everything else retains probabilistic IDW weights.
    # ----------------------------------------------------------

    low_both = (
        (pts[:, 0] <= 0.0)
        &
        (pts[:, 1] <= 0.0)
    )


    high_both = (
        (pts[:, 0] >= 1.0)
        &
        (pts[:, 1] >= 1.0)
    )


    weights[
        low_both,
        :
    ] = 0.0

    weights[
        low_both,
        0
    ] = 1.0


    weights[
        high_both,
        :
    ] = 0.0

    weights[
        high_both,
        2
    ] = 1.0


    surfaces = {
        scenario: (
            weights[:, i]
            .reshape(xx.shape)
        )
        for i, scenario
        in enumerate(SEJ_SCENARIOS)
    }


    return (
        xx,
        yy,
        surfaces,
    )


# ======================================================================
# 6. PLOTTING
# ======================================================================

def plot_family(
    df,
    family,
    scenario_order,
    scenario_labels,
    scenario_colors,
    output_name,
):

    xx, yy, surfaces = (
        analytical_effective_probabilities()
    )


    fig, axes = plt.subplots(
        3,
        1,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
    )


    x = pd.to_numeric(
        df[
            "reference_range_position_2100"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )


    y = pd.to_numeric(
        df[
            "reference_range_position_2200"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )


    # ==========================================================
    # PANELS
    # ==========================================================

    for ax, sej_scenario in zip(
        axes,
        SEJ_SCENARIOS,
    ):

        # ------------------------------------------------------
        # Actual climate realizations
        # ------------------------------------------------------

        for pathway in scenario_order:

            mask = (
                df["ssp"]
                .eq(pathway)
                .to_numpy()
            )

            ax.scatter(
                x[mask],
                y[mask],
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                color=scenario_colors[
                    pathway
                ],
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )


        # ------------------------------------------------------
        # Effective probability contours
        # ------------------------------------------------------

        cs = ax.contour(
            xx,
            yy,
            surfaces[
                sej_scenario
            ],
            levels=CONTOUR_LEVELS,
            colors="black",
            linewidths=1.15,
            zorder=4,
        )


        ax.clabel(
            cs,
            inline=True,
            fontsize=12,
            fmt="%.1f",
            inline_spacing=4,
        )


        # ------------------------------------------------------
        # LOW/HIGH normalized reference boundaries
        # ------------------------------------------------------

        ax.axvline(
            0.0,
            color="0.50",
            linestyle=":",
            linewidth=0.9,
            zorder=1,
        )

        ax.axhline(
            0.0,
            color="0.50",
            linestyle=":",
            linewidth=0.9,
            zorder=1,
        )

        ax.axvline(
            1.0,
            color="0.50",
            linestyle=":",
            linewidth=0.9,
            zorder=1,
        )

        ax.axhline(
            1.0,
            color="0.50",
            linestyle=":",
            linewidth=0.9,
            zorder=1,
        )


        # ------------------------------------------------------
        # LOW / INTERMEDIATE / HIGH reference trajectory
        # ------------------------------------------------------

        ref_x = [
            REFERENCE_POINTS[s][0]
            for s in SEJ_SCENARIOS
        ]

        ref_y = [
            REFERENCE_POINTS[s][1]
            for s in SEJ_SCENARIOS
        ]


        ax.plot(
            ref_x,
            ref_y,
            linestyle="--",
            linewidth=1.25,
            color="black",
            alpha=0.55,
            zorder=5,
        )


        ax.scatter(
            ref_x,
            ref_y,
            marker="*",
            s=90,
            color="black",
            edgecolor="white",
            linewidth=0.5,
            zorder=6,
        )


        # ------------------------------------------------------
        # Reference labels
        # ------------------------------------------------------

        for s in SEJ_SCENARIOS:

            rx, ry = (
                REFERENCE_POINTS[s]
            )

            dx, dy = (
                REFERENCE_LABEL_OFFSETS[s]
            )

            label = (
                "INTERMEDIATE"
                if s == "INTERMEDIATE"
                else s
            )


            ax.annotate(
                label,
                (rx, ry),
                xytext=(
                    dx,
                    dy,
                ),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                zorder=7,
            )


        # ------------------------------------------------------
        # Axes
        # ------------------------------------------------------

        ax.set_xlim(
            X_LIM
        )

        ax.set_ylim(
            Y_LIM
        )


        # Smaller panel titles + tighter pad
        ax.set_title(
            f"P({sej_scenario})",
            fontsize=12.5,
            fontweight="bold",
            pad=4,
        )


        ax.set_ylabel(
            "Normalized cumulative\nwarming through 2200",
            fontsize=10.5,
            fontweight="bold",
        )


        ax.grid(
            True,
            color="0.92",
            linewidth=0.65,
            zorder=0,
        )


        ax.tick_params(
            axis="both",
            labelsize=9.5,
            direction="out",
        )


        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )


    # ==========================================================
    # SHARED X AXIS
    # ==========================================================

    axes[-1].set_xlabel(
        "Normalized cumulative warming through 2100",
        fontsize=10.5,
        fontweight="bold",
    )


    # ==========================================================
    # LEGEND
    # ==========================================================

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=(
                scenario_colors[p]
            ),
            markeredgecolor="none",
            label=(
                scenario_labels[p]
            ),
        )
        for p in scenario_order
    ]


    # Both figures naturally use two legend rows at this manuscript size.
    ncol = (
        3
        if family == "CMIP6"
        else 4
    )


    # ==========================================================
    # MAIN TITLE
    # ==========================================================
    #
    # Keep the main title short.
    # Method details are separated into a smaller subtitle below it.
    # ==========================================================

    fig.suptitle(
        f"{family}: normalized scenario-selection probabilities",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),   # move legend UP toward title
        ncol=ncol,
        frameon=False,
        fontsize=9,
        columnspacing=1.10,
        handletextpad=0.35,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    
    fig.subplots_adjust(
        left=0.145,
        right=0.985,
        bottom=0.065,
        top=0.865,        # move first subplot UP toward legend
        hspace=0.32,
    )


    # ==========================================================
    # SAVE
    # ==========================================================

    png = (
        OUTPUT_DIR
        / f"{output_name}.png"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
    )

    print(
        "Saved:",
        png,
    )


    if SAVE_PDF:

        pdf = (
            OUTPUT_DIR
            / f"{output_name}.pdf"
        )

        fig.savefig(
            pdf,
            bbox_inches="tight",
        )

        print(
            "Saved:",
            pdf,
        )


    if SHOW_FIGURES:

        plt.show()


    plt.close(
        fig
    )


# ======================================================================
# 7. MAIN
# ======================================================================

def main():

    print(
        "\n"
        + "=" * 78
    )

    print(
        "CURRENT NORMALIZED SCENARIO-SELECTION PROBABILITY CONTOURS"
    )

    print(
        "=" * 78
    )


    # ------------------------------------------------------------------
    # CMIP6
    # ------------------------------------------------------------------

    if PLOT_CMIP6:

        cmip6_file = (
            first_existing(
                CMIP6_SELECTION_CANDIDATES,
                "CMIP6",
            )
        )


        print(
            "\nCMIP6 input:"
        )

        print(
            " ",
            cmip6_file,
        )


        cmip6 = (
            load_selection(
                cmip6_file,
                CMIP6_ORDER,
                "CMIP6",
            )
        )


        plot_family(
            cmip6,
            family="CMIP6",
            scenario_order=(
                CMIP6_ORDER
            ),
            scenario_labels=(
                CMIP6_LABELS
            ),
            scenario_colors=(
                CMIP6_COLORS
            ),
            output_name=(
                "CMIP6_normalized_selection_probability_contours_"
                "double_checkpoint_truncation"
            ),
        )


    # ------------------------------------------------------------------
    # CMIP7
    # ------------------------------------------------------------------

    if PLOT_CMIP7:

        cmip7_file = (
            first_existing(
                CMIP7_SELECTION_CANDIDATES,
                "CMIP7",
            )
        )


        print(
            "\nCMIP7 input:"
        )

        print(
            " ",
            cmip7_file,
        )


        cmip7 = (
            load_selection(
                cmip7_file,
                CMIP7_ORDER,
                "CMIP7",
            )
        )


        plot_family(
            cmip7,
            family="CMIP7",
            scenario_order=(
                CMIP7_ORDER
            ),
            scenario_labels=(
                CMIP7_LABELS
            ),
            scenario_colors=(
                CMIP7_COLORS
            ),
            output_name=(
                "CMIP7_normalized_selection_probability_contours_"
                "double_checkpoint_truncation"
            ),
        )


    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()