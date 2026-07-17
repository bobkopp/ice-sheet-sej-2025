#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:52:27 2026

Figure of dynamical processes assumptions.

Question posed to experts:
    How will near-field gravitational changes and vertical land motion due to 
    past and present-day ice sheet unloading affect marine ice sheet instability
    through 2300? Decrease instability, increase instability, or no significant
    change?

@author: elizabeth
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

fig_dir = "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025/output/figures/"
os.makedirs(fig_dir, exist_ok = True)

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

df_dyn = pd.DataFrame({
    "GrIS": [
        "N", "N", "N", "N", None,
        "N", "N", "N", "N", "N",
        "N", "D", "N", "N", "I"
    ],
    "WAIS": [
        "D", "I", "D", "N", "D",
        "N", "D", "D", "I", "N",
        "D", "D", "D", "D", "D"
    ],
    "EAIS": [
        "N", "N", "N", "N", "N",
        "N", "D", "D", "N", "N",
        "N", "D", "N", "N", "N"
    ]
})

# ------------------------------------------------------------
# CALCULATE PERCENTAGES
# ------------------------------------------------------------

summary_dyn = pd.DataFrame(index=["D", "I", "N"])

for sheet in df_dyn.columns:
    counts = df_dyn[sheet].dropna().value_counts(normalize=True) * 100

    summary_dyn[sheet] = [
        counts.get("D", 0),
        counts.get("I", 0),
        counts.get("N", 0)
    ]

print(summary_dyn.round(1))

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 5))

# Greyscale
d_color = "0.72"
i_color = "0.45"
n_color = "0.20"

bar_width = 0.48

ax.bar(
    summary_dyn.columns,
    summary_dyn.loc["D"],
    width=bar_width,
    color=d_color
)

ax.bar(
    summary_dyn.columns,
    summary_dyn.loc["I"],
    bottom=summary_dyn.loc["D"],
    width=bar_width,
    color=i_color
)

ax.bar(
    summary_dyn.columns,
    summary_dyn.loc["N"],
    bottom=summary_dyn.loc["D"] + summary_dyn.loc["I"],
    width=bar_width,
    color=n_color
)

# ------------------------------------------------------------
# ANNOTATIONS
# ------------------------------------------------------------

for i, sheet in enumerate(summary_dyn.columns):

    decrease = summary_dyn.loc["D", sheet]
    increase = summary_dyn.loc["I", sheet]
    no_change = summary_dyn.loc["N", sheet]

    # Decrease instability
    if decrease >= 1:
        ax.text(
            i,
            decrease / 2,
            "dec.",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="semibold",
            color="black"
        )

    # Increase instability
    if increase >= 1:
        ax.text(
            i,
            decrease + increase / 2,
            "inc.",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="semibold",
            color="white"
        )

    # No significant change
    if no_change >= 8:
        y = decrease + increase + no_change / 2

        ax.text(
            i,
            y + 3,
            "no sig.",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="semibold",
            color="white"
        )

        ax.text(
            i,
            y - 3,
            "change",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="semibold",
            color="white"
        )

# ------------------------------------------------------------
# FORMATTING
# ------------------------------------------------------------

ax.set_ylim(0, 100)
ax.set_ylabel("Expert responses", fontsize=12)

ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f"{value}%" for value in range(0, 101, 20)])

ax.set_xticks(range(len(summary_dyn.columns)))
ax.set_xticklabels(
    summary_dyn.columns,
    fontsize=12,
    fontweight="bold"
)
ax.set_title("Changes to marine ice sheet instability")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

outfile = os.path.join(
    fig_dir,
    "dynamical_processes_gravity_vlm.png"
)

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)

print(f"Figure saved to: {outfile}")

plt.show()