"""
Comparison of B19 and B25 individual ice-sheet contributions
to sea-level rise in 2100.

The figure contains two vertically stacked panels:
    A. Low warming (2.0°C)
    B. High warming (5.0°C)

Each panel shows B25 probability density functions for:
    GrIS
    Total AIS

Above the B25 probability density functions, percentile bars
compare B19 and B25 projections.

Each percentile bar shows:
    Thin line: 5th–95th percentile range
    Thick line: 17th–83rd percentile range
    Circle: 50th percentile (median)

B25:
    Solid percentile bars

B19:
    Dashed percentile bars

Because the underlying B19 Monte Carlo samples are unavailable,
only B19 percentile bars are shown; no B19 probability density
functions are plotted.

@author: elizabeth
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from B25data_processing import (
    load_and_prepare_data,
    project_dir
)


# ============================================================
# 1. SETTINGS
# ============================================================

output_dir = os.path.join(
    project_dir,
    "output",
    "figures"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


b19_file = os.path.join(
    project_dir,
    "data",
    "B19_data.xlsx"
)


years = [
    2100,
    2200,
    2300
]


warming_scenarios = {
    "Low warming (2.0°C)": 2.0,
    "High warming (5.0°C)": 5.0
}


# The dictionary keys are the names displayed in the figure.
# The values are the exact ice-sheet names in mc_total_ext.
ice_sheets = {
    "GrIS": "GrIS",
    "AIS": "Total AIS"
}


# Names used in the B19 spreadsheet.
b19_ice_sheet_names = {
    "GrIS": "GrIS",
    "AIS": "Total AIS"
}


b19_scenario_names = {
    "Low warming (2.0°C)": "Low",
    "High warming (5.0°C)": "High"
}


# Colorblind-friendly colors matching the earlier figure.
ice_sheet_colors = {
    "GrIS": "#009E73",
    "AIS": "#CC79A7"
}


panel_labels = {
    "Low warming (2.0°C)": "A",
    "High warming (5.0°C)": "B"
}


percentile_levels = [
    5,
    17,
    50,
    83,
    95
]


# Multiplier applied to scipy's default Scott bandwidth.
# Increase for smoother KDE curves; decrease for more detail.
kde_bandwidth = 1.0


# Year-specific x-axis limits in centimeters.
x_limits = {
    2100: (-100, 175),
    2200: (-100, 625),
    2300: (-100, 1000)
}


# Set to False to suppress the numerical percentile labels.
annotate_percentiles = True


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

data = load_and_prepare_data()

mc_total_ext = data["mc_total_ext"]


b19_data = pd.read_excel(
    b19_file
)

def clean_excel_column_name(column):
    """
    Convert Excel percentage headers such as 0.05 back to '5%'.
    """

    if isinstance(column, (int, float, np.integer, np.floating)):
        percentage_headers = {
            0.05: "5%",
            0.17: "17%",
            0.50: "Median (50%)",
            0.83: "83%",
            0.95: "95%"
        }

        for numeric_value, label in percentage_headers.items():
            if np.isclose(
                float(column),
                numeric_value
            ):
                return label

    text = str(column).strip()

    text_map = {
        "0.05": "5%",
        "0.17": "17%",
        "0.5": "Median (50%)",
        "0.50": "Median (50%)",
        "0.83": "83%",
        "0.95": "95%"
    }

    return text_map.get(
        text,
        text
    )


b19_data.columns = [
    clean_excel_column_name(column)
    for column in b19_data.columns
]

print(
    "B19 spreadsheet columns:",
    b19_data.columns.tolist()
)

# Standardize spreadsheet column names and text fields.
b19_data.columns = (
    b19_data.columns
    .astype(str)
    .str.strip()
)

required_b19_columns = {
    "Year",
    "Ice Sheet",
    "Scenario",
    "Median (50%)",
    "5%",
    "95%",
    "17%",
    "83%"
}

missing_b19_columns = (
    required_b19_columns
    - set(b19_data.columns)
)

if missing_b19_columns:
    raise ValueError(
        "The B19 spreadsheet is missing these columns: "
        + ", ".join(sorted(missing_b19_columns))
    )

b19_data["Year"] = pd.to_numeric(
    b19_data["Year"],
    errors="raise"
).astype(int)

b19_data["Ice Sheet"] = (
    b19_data["Ice Sheet"]
    .astype(str)
    .str.strip()
)

b19_data["Scenario"] = (
    b19_data["Scenario"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_ice_sheet_samples(
    dataframe,
    ice_sheet,
    year,
    warming
):
    """
    Extract B25 samples for one ice sheet and convert values
    from millimeters to centimeters.
    """

    samples = dataframe.loc[
        (
            dataframe["ice_sheet"] == ice_sheet
        )
        & (
            dataframe["year"] == year
        )
        & (
            dataframe["warming"] == warming
        ),
        "value"
    ].dropna()

    if samples.empty:

        raise ValueError(
            "No samples found for "
            f"{ice_sheet}, {year}, {warming:.1f}°C."
        )

    return (
        samples.to_numpy()
        / 10.0
    )


def get_b19_percentiles(
    dataframe,
    year,
    ice_sheet,
    scenario
):
    """
    Extract B19 percentiles in this order:
    5th, 17th, 50th, 83rd, and 95th.
    """

    selection = dataframe.loc[
        (
            dataframe["Year"] == year
        )
        & (
            dataframe["Ice Sheet"].str.lower()
            == ice_sheet.lower()
        )
        & (
            dataframe["Scenario"]
            == scenario.lower()
        )
    ]

    if selection.empty:
        available = (
            dataframe.loc[
                dataframe["Year"] == year,
                [
                    "Ice Sheet",
                    "Scenario"
                ]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

        raise ValueError(
            f"No B19 row found for {year}, "
            f"{ice_sheet}, {scenario}.\n\n"
            f"Available rows for {year}:\n{available}"
        )

    if len(selection) > 1:
        raise ValueError(
            f"Multiple B19 rows found for {year}, "
            f"{ice_sheet}, {scenario}."
        )

    row = selection.iloc[0]

    return np.array(
        [
            row["5%"],
            row["17%"],
            row["Median (50%)"],
            row["83%"],
            row["95%"]
        ],
        dtype=float
    )


def calculate_kde(
    samples,
    x_values,
    bandwidth_multiplier=1.0
):
    """
    Calculate a Gaussian kernel density estimate.
    """

    kde = gaussian_kde(
        samples
    )

    kde.set_bandwidth(
        bw_method=(
            kde.factor
            * bandwidth_multiplier
        )
    )

    return kde(
        x_values
    )


def fmt(value):
    """
    Format percentile values as whole numbers and use a
    typographic minus sign.
    """

    return f"{value:.0f}".replace(
        "-",
        "−"
    )


def spread_positions(
    values,
    minimum_spacing
):
    """
    Shift annotation positions horizontally so adjacent
    labels maintain a minimum separation.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    positions = values.copy()

    # Forward pass.
    for index in range(
        1,
        len(positions)
    ):

        if (
            positions[index]
            - positions[index - 1]
            < minimum_spacing
        ):

            positions[index] = (
                positions[index - 1]
                + minimum_spacing
            )

    # Recenter the adjusted labels around their original mean.
    positions -= (
        positions.mean()
        - values.mean()
    )

    return positions


def add_percentile_annotations(
    ax,
    percentiles,
    y_position,
    color,
    annotation_offset,
    x_range
):
    """
    Annotate the five percentile values.

    The 5th, 50th, and 95th percentiles are shown above
    the bar. The 17th and 83rd percentiles are shown below.
    """

    q05, q17, q50, q83, q95 = percentiles

    minimum_spacing = (
        0.045
        * x_range
    )

    above_values = np.array(
        [
            q05,
            q50,
            q95
        ],
        dtype=float
    )

    below_values = np.array(
        [
            q17,
            q83
        ],
        dtype=float
    )

    above_positions = spread_positions(
        values=above_values,
        minimum_spacing=minimum_spacing
    )

    below_positions = spread_positions(
        values=below_values,
        minimum_spacing=minimum_spacing
    )

    # --------------------------------------------------------
    # LABELS ABOVE: 5TH, 50TH, AND 95TH
    # --------------------------------------------------------

    for value, x_position, is_median in zip(
        above_values,
        above_positions,
        [
            False,
            True,
            False
        ]
    ):

        ax.text(
            x_position,
            y_position + annotation_offset,
            fmt(value),
            color=color,
            fontsize=7.5,
            fontweight=(
                "bold"
                if is_median
                else "normal"
            ),
            ha="center",
            va="bottom",
            clip_on=False,
            zorder=20
        )

    # --------------------------------------------------------
    # LABELS BELOW: 17TH AND 83RD
    # --------------------------------------------------------

    for value, x_position in zip(
        below_values,
        below_positions
    ):

        ax.text(
            x_position,
            y_position - annotation_offset,
            fmt(value),
            color=color,
            fontsize=7.5,
            ha="center",
            va="top",
            clip_on=False,
            zorder=20
        )


def add_percentile_bar(
    ax,
    percentiles,
    y_position,
    color,
    study,
    annotation_offset,
    x_range
):
    """
    Draw one B19 or B25 percentile bar.

    B19:
        Dashed ranges and open median marker.

    B25:
        Solid ranges and filled median marker.
    """

    q05, q17, q50, q83, q95 = np.asarray(
        percentiles,
        dtype=float
    )

    if study == "B19":

        linestyle = "--"
        median_facecolor = "white"
        thin_linewidth = 1.2
        thick_linewidth = 3.0

    elif study == "B25":

        linestyle = "-"
        median_facecolor = color
        thin_linewidth = 1.3
        thick_linewidth = 3.5

    else:

        raise ValueError(
            "study must be either 'B19' or 'B25'."
        )

    # --------------------------------------------------------
    # THIN 5TH–95TH PERCENTILE RANGE
    # --------------------------------------------------------

    ax.hlines(
        y=y_position,
        xmin=q05,
        xmax=q95,
        color=color,
        linewidth=thin_linewidth,
        linestyle=linestyle,
        zorder=4
    )

    # --------------------------------------------------------
    # THICK 17TH–83RD PERCENTILE RANGE
    # --------------------------------------------------------

    ax.hlines(
        y=y_position,
        xmin=q17,
        xmax=q83,
        color=color,
        linewidth=thick_linewidth,
        linestyle=linestyle,
        zorder=5
    )

    # --------------------------------------------------------
    # MEDIAN MARKER
    # --------------------------------------------------------

    ax.plot(
        q50,
        y_position,
        marker="o",
        markersize=4.5,
        markerfacecolor=median_facecolor,
        markeredgecolor=color,
        markeredgewidth=1.0,
        linestyle="none",
        zorder=7
    )

    # --------------------------------------------------------
    # PERCENTILE TICKS
    # --------------------------------------------------------

    tick_height = (
        annotation_offset
        * 0.24
    )

    for percentile in [
        q05,
        q17,
        q50,
        q83,
        q95
    ]:

        ax.vlines(
            x=percentile,
            ymin=(
                y_position
                - tick_height
            ),
            ymax=(
                y_position
                + tick_height
            ),
            color=color,
            linewidth=0.8,
            linestyle=linestyle,
            zorder=6
        )

    # --------------------------------------------------------
    # OPTIONAL NUMERICAL ANNOTATIONS
    # --------------------------------------------------------

    if annotate_percentiles:

        add_percentile_annotations(
            ax=ax,
            percentiles=[
                q05,
                q17,
                q50,
                q83,
                q95
            ],
            y_position=y_position,
            color=color,
            annotation_offset=annotation_offset,
            x_range=x_range
        )


# ============================================================
# 4. CREATE ONE FIGURE FOR EACH YEAR
# ============================================================

for year in years:

    figure_png = os.path.join(
        output_dir,
        f"Figure02_B19_B25_GrIS_AIS_PDFs_{year}.png"
    )

    figure_pdf = os.path.join(
        output_dir,
        f"Figure02_B19_B25_GrIS_AIS_PDFs_{year}.pdf"
    )

    x_min, x_max = x_limits[
        year
    ]

    x_range = (
        x_max
        - x_min
    )

    x_values = np.linspace(
        x_min,
        x_max,
        1200
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7.0, 9.0),
        sharex=True
    )

    for ax, (
        scenario_name,
        warming
    ) in zip(
        axes,
        warming_scenarios.items()
    ):

        # ----------------------------------------------------
        # EXTRACT B25 DISTRIBUTIONS AND PLOT KDE CURVES
        # ----------------------------------------------------

        distributions = {}
        densities = {}

        for display_name, data_name in ice_sheets.items():

            samples = get_ice_sheet_samples(
                dataframe=mc_total_ext,
                ice_sheet=data_name,
                year=year,
                warming=warming
            )

            density = calculate_kde(
                samples=samples,
                x_values=x_values,
                bandwidth_multiplier=kde_bandwidth
            )

            distributions[
                display_name
            ] = samples

            densities[
                display_name
            ] = density

            ax.plot(
                x_values,
                density,
                color=ice_sheet_colors[
                    display_name
                ],
                linewidth=1.5,
                label=display_name,
                zorder=3
            )

        max_density = max(
            density.max()
            for density in densities.values()
        )

        # ----------------------------------------------------
        # PERCENTILE-BAR POSITIONS
        # ----------------------------------------------------

        curve_gap = (
            max_density
            * 0.18
        )

        bar_gap = (
            max_density
            * 0.19
        )

        lowest_bar = (
            max_density
            + curve_gap
        )

        bar_positions = {
            "AIS": {
                "B19": lowest_bar,
                "B25": lowest_bar + bar_gap
            },
            "GrIS": {
                "B19": lowest_bar + 2 * bar_gap,
                "B25": lowest_bar + 3 * bar_gap
            }
        }

        annotation_offset = (
            max_density
            * 0.025
        )

        # ----------------------------------------------------
        # DRAW B19 AND B25 PERCENTILE BARS
        # ----------------------------------------------------

        for display_name in [
            "AIS",
            "GrIS"
        ]:

            color = ice_sheet_colors[
                display_name
            ]

            b19_values = get_b19_percentiles(
                dataframe=b19_data,
                year=year,
                ice_sheet=b19_ice_sheet_names[
                    display_name
                ],
                scenario=b19_scenario_names[
                    scenario_name
                ]
            )

            add_percentile_bar(
                ax=ax,
                percentiles=b19_values,
                y_position=bar_positions[
                    display_name
                ][
                    "B19"
                ],
                color=color,
                study="B19",
                annotation_offset=annotation_offset,
                x_range=x_range
            )

            b25_values = np.percentile(
                distributions[
                    display_name
                ],
                percentile_levels
            )

            add_percentile_bar(
                ax=ax,
                percentiles=b25_values,
                y_position=bar_positions[
                    display_name
                ][
                    "B25"
                ],
                color=color,
                study="B25",
                annotation_offset=annotation_offset,
                x_range=x_range
            )

        # ----------------------------------------------------
        # ICE-SHEET AND STUDY LABELS
        # ----------------------------------------------------

        label_x = (
            x_min
            + 0.02 * x_range
        )

        for display_name in [
            "AIS",
            "GrIS"
        ]:

            color = ice_sheet_colors[
                display_name
            ]

            for study in [
                "B19",
                "B25"
            ]:

                ax.text(
                    label_x,
                    bar_positions[
                        display_name
                    ][
                        study
                    ],
                    f"{display_name} {study}",
                    color=color,
                    fontsize=8.5,
                    fontweight=(
                        "bold"
                        if study == "B25"
                        else "normal"
                    ),
                    ha="left",
                    va="center",
                    clip_on=True,
                    zorder=20
                )

        # ----------------------------------------------------
        # AXES AND LABELS
        # ----------------------------------------------------

        ax.set_xlim(
            x_min,
            x_max
        )

        highest_bar = bar_positions[
            "GrIS"
        ][
            "B25"
        ]

        ax.set_ylim(
            0,
            highest_bar
            + max_density * 0.18
        )

        ax.set_ylabel(
            "Probability density, p[SLR]",
            fontsize=11,
            fontweight="bold"
        )

        ax.text(
            0.01,
            1.0,
            panel_labels[
                scenario_name
            ],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top"
        )

        ax.text(
            0.98,
            0.98,
            scenario_name,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="right",
            va="top"
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

        ax.tick_params(
            axis="both",
            direction="out",
            labelsize=10,
            length=4
        )

    # ========================================================
    # 5. SHARED X-AXIS LABEL
    # ========================================================

    axes[0].set_xlabel(
        ""
    )

    axes[1].set_xlabel(
        "Ice-sheet contribution to SLR (cm)",
        fontsize=11,
        fontweight="bold"
    )

    # ========================================================
    # 6. FIGURE TITLE
    # ========================================================

    fig.suptitle(
        str(year),
        fontsize=14,
        fontweight="bold",
        y=0.985
    )

    # ========================================================
    # 7. LEGENDS
    # ========================================================

    ice_sheet_handles = [
        Line2D(
            [0],
            [0],
            color=ice_sheet_colors[
                "GrIS"
            ],
            linewidth=1.8,
            label="GrIS"
        ),
        Line2D(
            [0],
            [0],
            color=ice_sheet_colors[
                "AIS"
            ],
            linewidth=1.8,
            label="AIS"
        )
    ]

    study_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="0.25",
            markersize=4.5,
            label="B19"
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="-",
            linewidth=3.0,
            marker="o",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            markersize=4.5,
            label="B25"
        )
    ]

    ice_sheet_legend = fig.legend(
        handles=ice_sheet_handles,
        title="Ice sheet",
        loc="lower center",
        bbox_to_anchor=(0.36, 0.012),
        ncol=2,
        frameon=False,
        fontsize=10,
        title_fontsize=10,
        handlelength=2.5
    )

    fig.add_artist(
        ice_sheet_legend
    )

    fig.legend(
        handles=study_handles,
        title="Study",
        loc="lower center",
        bbox_to_anchor=(0.68, 0.012),
        ncol=2,
        frameon=False,
        fontsize=10,
        title_fontsize=10,
        handlelength=2.5
    )

    # ========================================================
    # 8. LAYOUT AND SAVE
    # ========================================================

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.13,
        top=0.94,
        hspace=0.26
    )

    fig.savefig(
        figure_png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        figure_pdf,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(
        fig
    )

    print(
        f"\nSaved {year} B19-B25 comparison figure:"
    )

    print(
        figure_png
    )

    print(
        figure_pdf
    )
