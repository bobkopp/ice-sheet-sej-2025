"""
Create B25 sea-level-rise figures against cumulative warming.

Outputs:
    1. Total ice-sheet SLR vs cumulative warming
    2. EAIS, WAIS, Total AIS, and GrIS vs cumulative warming
    3. CSV containing the plotted percentile summaries

The original Monte Carlo sea-level values are converted from
millimeters to centimeters before calculating percentiles.

Cumulative warming is expressed in K·yr and is integrated from 2000.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from B25data_processing import (
    load_and_prepare_data,
    project_dir,
)


# ============================================================
# 1. SETTINGS
# ============================================================

output_dir = os.path.join(
    project_dir,
    "output/figures",
)

os.makedirs(
    output_dir,
    exist_ok=True,
)


# Cumulative-warming values from the B25 scenario table.
# Units: K·yr, integrated from 2000.
cumulative_warming_map = {
    (2.0, 2100): 155.0,
    (2.0, 2200): 355.0,
    (2.0, 2300): 555.0,

    (3.5, 2100): 204.5,
    (3.5, 2200): 554.5,
    (3.5, 2300): 904.5,

    (5.0, 2100): 262.0,
    (5.0, 2200): 762.0,
    (5.0, 2300): 1262.0,
}


# Match the warming-scenario colors used previously.
warming_colors = {
    2.0: "#0072B2",   # Low: blue
    3.5: "#6A3D9A",   # Intermediate: purple
    5.0: "#E66101",   # High: orange
}


warming_labels = {
    2.0: "Low (2.0°C)",
    3.5: "Intermediate (3.5°C)",
    5.0: "High (5.0°C)",
}


# Marker shape distinguishes projection year.
year_markers = {
    2100: "o",
    2200: "s",
    2300: "^",
}


projection_years = [
    2100,
    2200,
    2300,
]


components = [
    "EAIS",
    "WAIS",
    "Total AIS",
    "GrIS",
]


total_component_name = "Total ice-sheet SLR"


# ============================================================
# 2. LOAD PREPARED MONTE CARLO DATA
# ============================================================

data = load_and_prepare_data()

mc_total_ext = data["mc_total_ext"].copy()


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {
    "ice_sheet",
    "warming",
    "year",
    "value",
}

missing_columns = required_columns.difference(
    mc_total_ext.columns
)

if missing_columns:
    raise KeyError(
        "mc_total_ext is missing the following required columns: "
        f"{sorted(missing_columns)}\n\n"
        f"Available columns:\n{list(mc_total_ext.columns)}"
    )


# Convert relevant fields to numeric values.
mc_total_ext["warming"] = pd.to_numeric(
    mc_total_ext["warming"],
    errors="coerce",
)

mc_total_ext["year"] = pd.to_numeric(
    mc_total_ext["year"],
    errors="coerce",
)

mc_total_ext["value"] = pd.to_numeric(
    mc_total_ext["value"],
    errors="coerce",
)


# Keep only the scenarios and years used in the figure.
plot_data = mc_total_ext.loc[
    mc_total_ext["warming"].isin(warming_colors)
    & mc_total_ext["year"].isin(projection_years)
].copy()


if plot_data.empty:
    raise ValueError(
        "No Monte Carlo samples matched the selected warming "
        "scenarios and projection years."
    )


# ============================================================
# 4. CONVERT SEA-LEVEL VALUES FROM MM TO CM
# ============================================================

plot_data["value_cm"] = plot_data["value"] / 10.0


# ============================================================
# 5. ADD CUMULATIVE WARMING
# ============================================================

plot_data["cumulative_warming"] = plot_data.apply(
    lambda row: cumulative_warming_map.get(
        (
            float(row["warming"]),
            int(row["year"]),
        ),
        np.nan,
    ),
    axis=1,
)


missing_mapping = (
    plot_data.loc[
        plot_data["cumulative_warming"].isna(),
        ["warming", "year"],
    ]
    .drop_duplicates()
    .sort_values(["warming", "year"])
)

if not missing_mapping.empty:
    raise ValueError(
        "The following warming/year combinations do not have "
        "cumulative-warming values:\n\n"
        f"{missing_mapping.to_string(index=False)}"
    )


# ============================================================
# 6. CALCULATE MONTE CARLO PERCENTILES
# ============================================================

def percentile(values, q):
    """Return a NaN-safe percentile."""

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric_values.empty:
        return np.nan

    return np.nanpercentile(
        numeric_values,
        q,
    )


summary = (
    plot_data
    .groupby(
        [
            "ice_sheet",
            "warming",
            "year",
            "cumulative_warming",
        ],
        as_index=False,
    )
    .agg(
        n=("value_cm", "count"),
        p05_cm=("value_cm", lambda x: percentile(x, 5)),
        p17_cm=("value_cm", lambda x: percentile(x, 17)),
        p50_cm=("value_cm", lambda x: percentile(x, 50)),
        p83_cm=("value_cm", lambda x: percentile(x, 83)),
        p95_cm=("value_cm", lambda x: percentile(x, 95)),
    )
    .sort_values(
        [
            "ice_sheet",
            "warming",
            "year",
        ]
    )
)


summary_file = os.path.join(
    output_dir,
    "slr_vs_cumulative_warming_summary_cm.csv",
)

summary.to_csv(
    summary_file,
    index=False,
)

print("\nPercentile summary in centimeters:")
print(summary.to_string(index=False))

print(f"\nSaved summary:\n{summary_file}")


# ============================================================
# 7. PLOTTING FUNCTION
# ============================================================

def plot_component(
    ax,
    component_data,
    title,
    show_xlabel=True,
    show_ylabel=True,
):
    """
    Plot one ice-sheet component against cumulative warming.

    Thin vertical bars:
        5th–95th percentile range

    Thick vertical bars:
        17th–83rd percentile range

    Markers:
        Median values

    Colors:
        Warming scenarios

    Marker shapes:
        Projection years
    """

    for warming in sorted(warming_colors):

        scenario_data = (
            component_data.loc[
                component_data["warming"] == warming
            ]
            .sort_values("cumulative_warming")
        )

        if scenario_data.empty:
            continue

        color = warming_colors[warming]

        # Connect the medians for the three projection years.
        # This line is only a visual guide between modeled points.
        ax.plot(
            scenario_data["cumulative_warming"],
            scenario_data["p50_cm"],
            color=color,
            linewidth=1.5,
            alpha=0.9,
            zorder=1,
        )

        for _, row in scenario_data.iterrows():

            year = int(row["year"])
            marker = year_markers[year]
            x_value = row["cumulative_warming"]

            # 5th–95th percentile interval.
            ax.vlines(
                x=x_value,
                ymin=row["p05_cm"],
                ymax=row["p95_cm"],
                color=color,
                linewidth=1.2,
                alpha=0.95,
                zorder=2,
            )

            # 17th–83rd percentile interval.
            ax.vlines(
                x=x_value,
                ymin=row["p17_cm"],
                ymax=row["p83_cm"],
                color=color,
                linewidth=4.0,
                alpha=0.95,
                zorder=3,
            )

            # Median.
            ax.plot(
                x_value,
                row["p50_cm"],
                marker=marker,
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.7,
                linestyle="none",
                zorder=4,
            )

    ax.axhline(
        y=0,
        color="black",
        linewidth=0.8,
        zorder=0,
    )

    ax.set_title(
        title,
        fontweight="bold",
    )

    if show_xlabel:
        ax.set_xlabel(
            "Cumulative warming since 2000 (K·yr)"
        )

    if show_ylabel:
        ax.set_ylabel(
            "Sea-level contribution (cm)"
        )

    ax.grid(
        True,
        linewidth=0.5,
        alpha=0.3,
    )


# ============================================================
# 8. LEGEND HANDLES
# ============================================================

warming_handles = [
    Line2D(
        [0],
        [0],
        color=warming_colors[warming],
        marker="o",
        markerfacecolor=warming_colors[warming],
        markeredgecolor="black",
        markeredgewidth=0.7,
        linewidth=1.5,
        markersize=7,
        label=warming_labels[warming],
    )
    for warming in sorted(warming_colors)
]


year_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        marker=year_markers[year],
        markerfacecolor="white",
        markeredgecolor="black",
        linestyle="none",
        markersize=7,
        label=str(year),
    )
    for year in projection_years
]


# ============================================================
# 9. FIGURE A: TOTAL ICE-SHEET SLR
# ============================================================

total_data = summary.loc[
    summary["ice_sheet"] == total_component_name
].copy()


if total_data.empty:

    available_components = sorted(
        summary["ice_sheet"]
        .dropna()
        .unique()
    )

    raise ValueError(
        f'No data were found for "{total_component_name}".\n\n'
        "Available ice-sheet labels:\n"
        f"{available_components}"
    )


fig, ax = plt.subplots(
    figsize=(9, 6.5),
)


plot_component(
    ax=ax,
    component_data=total_data,
    title="Total ice-sheet sea-level contribution",
    show_xlabel=True,
    show_ylabel=True,
)


# Warming-scenario legend.
warming_legend = ax.legend(
    handles=warming_handles,
    title="Warming scenario",
    loc="upper left",
    frameon=False,
)

ax.add_artist(
    warming_legend
)


# Projection-year legend.
ax.legend(
    handles=year_handles,
    title="Projection year",
    loc="lower right",
    frameon=False,
)


fig.tight_layout()


total_figure_file = os.path.join(
    output_dir,
    "total_ice_sheet_slr_vs_cumulative_warming_cm_UPDATDE.png",
)

fig.savefig(
    total_figure_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print(
    "\nSaved total ice-sheet figure:"
    f"\n{total_figure_file}"
)


# ============================================================
# 10. FIGURE B: INDIVIDUAL ICE-SHEET COMPONENTS
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(12, 9),
    sharex=True,
)

axes = axes.flatten()


for index, component in enumerate(components):

    ax = axes[index]

    component_data = summary.loc[
        summary["ice_sheet"] == component
    ].copy()

    if component_data.empty:

        ax.text(
            0.5,
            0.5,
            f"No data found for\n{component}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(
            component,
            fontweight="bold",
        )

        continue

    plot_component(
        ax=ax,
        component_data=component_data,
        title=component,
        show_xlabel=False,
        show_ylabel=False,
    )


# Figure-level warming-scenario legend.
fig.legend(
    handles=warming_handles,
    title="Warming scenario",
    loc="upper center",
    bbox_to_anchor=(0.39, 1.01),
    ncol=3,
    frameon=False,
)


# Figure-level projection-year legend.
fig.legend(
    handles=year_handles,
    title="Projection year",
    loc="upper center",
    bbox_to_anchor=(0.82, 1.01),
    ncol=3,
    frameon=False,
)


fig.supxlabel(
    "Cumulative warming since 2000 (K·yr)",
    y=0.03,
)

fig.supylabel(
    "Sea-level contribution (cm)",
    x=0.03,
)


fig.tight_layout(
    rect=[
        0.04,
        0.05,
        1.00,
        0.92,
    ]
)


components_figure_file = os.path.join(
    output_dir,
    "ice_sheet_components_vs_cumulative_warming_cm.png",
)

fig.savefig(
    components_figure_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print(
    "\nSaved component figure:"
    f"\n{components_figure_file}"
)