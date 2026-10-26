#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:29:09 2026

Create figure showing statistical accuracy v. (a) informativeness
and (b) mean absolute percentage error. Compare 2018 and 2025 SEJ
expert performance. 

Figure 2 in manuscript.

Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""

from figure_settings import PROJECT_ROOT, DATA_ROOT, UPSTREAM_ROOT, FIGURE_ROOT, output_for, DPI, SHOW_FIGURES, B25_MC_FILE, B19_ADJUSTED_FILE, CMIP6_ANNUAL_DIR, CMIP7_ANNUAL_DIR, CMIP6_SELECTION_FILE, CMIP7_SELECTION_FILE
from figure_settings import B19_SUMMARY_FILE, EXPERT_QUESTIONS_FILE, EXPERT_WORKBOOK_FILE, PERFORMANCE_FILE, AR6_WORKBOOK_FILE, COUPLED_SEJ_FILE
from figure_helpers import save_figure, maybe_show
from figure_style import PERFORMANCE_YEAR_COLORS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = str(PERFORMANCE_FILE)
df = pd.read_csv(file_path)
# Harmonize file coding (2019/2026) with actual elicitation years (2018/2025).
df["Year"] = df["Year"].replace({2019: 2018, 2026: 2025})

# ============================================================
# 2. PLOT SETTINGS
# ============================================================

years = sorted(df["Year"].unique())

# Assign a different marker to each year
markers = {
    2018: "o",
    2025: "^"
}

# Color-blind friendly color palette
colors = PERFORMANCE_YEAR_COLORS

# ============================================================
# 3. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(7, 5),
    sharey=True
)

# Plot each year separately
for year in years:

    year_data = df[df["Year"] == year]

    # Year-specific SA threshold
    threshold = 0.01 if year == 2018 else 0.1

    # Filled if above threshold, open if below
    facecolors = [
        colors[year] if sa >= threshold else "none"
        for sa in year_data["SA"]
    ]

    # A. SA vs Informativeness
    axes[0].scatter(
        year_data["Inf"],
        year_data["SA"],
        label=str(year),
        marker=markers[year],
        facecolors=facecolors,
        edgecolors=colors[year],
        s=65,
        linewidth=1.3,
    )

    # B. SA vs MAPE
    axes[1].scatter(
        year_data["MAPE"],
        year_data["SA"],
        marker=markers[year],
        facecolors=facecolors,
        edgecolors=colors[year],
        s=65,
        linewidth=1.3,
    )

# ============================================================
# 4. FORMAT AXES
# ============================================================

axes[0].set_xlabel(
    "Informativeness",
    fontsize=12,
    fontweight="bold"
)

axes[1].set_xlabel(
    "Mean Absolute\nPercentage Error",
    fontsize=12,
    fontweight="bold"
)

axes[0].set_ylabel(
    "Statistical Accuracy",
    fontsize=12,
    fontweight="bold"
)

for ax in axes:

    # Horizontal threshold line
    ax.axhline(
        y=0.01,                  #B19
        color=PERFORMANCE_YEAR_COLORS[2018],
        linestyle="--",
        linewidth=1.2
    )
    
    ax.axhline(
        y=0.1,                  #B25
        color=PERFORMANCE_YEAR_COLORS[2025],
        linestyle="--",
        linewidth=1.2
    )

    ax.tick_params(
        axis="both",
        labelsize=12
    )
    
    ax.set_yscale("log")
    
    ax.grid(
        True,
        which="major",
        color="lightgray",
        linestyle="--",
        linewidth=0.6,
        alpha=0.7
    )
    
axes[1].tick_params(
    axis="y",
    labelleft=True
)

# Panel labels
axes[0].text(
    0.03,
    1.02,
    "A.",
    transform=axes[0].transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="left"
)

axes[1].text(
    0.03,
    1.02,
    "B.",
    transform=axes[1].transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="left"
)

# One shared legend
legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='none',
        label='2018', # match year of SEJ elicitation
        markerfacecolor=colors[2018],
        markeredgecolor=colors[2018],
        markeredgewidth=1.3,
        markersize=8,
        linestyle='None'
    ),
    Line2D(
        [0], [0],
        marker='^',
        color='none',
        label='2025', # SAA
        markerfacecolor=colors[2025],
        markeredgecolor=colors[2025],
        markeredgewidth=1.3,
        markersize=8,
        linestyle='None'
    )
]

axes[1].legend(
    handles=legend_elements,
    title="Year",
    loc="lower right",
    bbox_to_anchor = (1.0, 0.08),
    frameon=True,
    fontsize=10,
    title_fontsize=10
)

plt.tight_layout()
sns.despine()

save_figure(fig, output_for("002_expert_performance") / "SA_Inf_MAPE.png", bbox_inches=None)
maybe_show()