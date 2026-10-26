#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:17:04 2026

Figure of mass change observations and assumptions.

Question posed to experts:
    Are recent decadal trends (since 1990) in mass balance largely due to
    natural variability (NV) or anthropogenic forcing (AF) for each ice sheet
    (i.e., GrIS and WAIS) overall?

Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

from pathlib import Path

# Metarepo output: place supplementary figures inside the isolated FIG_OUTPUT_DIR.
if not os.environ.get("FIG_OUTPUT_DIR"):
    raise RuntimeError("Set FIG_OUTPUT_DIR to an isolated output directory first")
fig_dir = str(Path(os.environ.get("SEJ_FIGURE_GROUP_OUTPUT") or (Path(os.environ["FIG_OUTPUT_DIR"]) / "supplementary" / "mass_change")).expanduser().resolve())
os.makedirs(fig_dir, exist_ok = True)

# Data
df = pd.DataFrame({
    "GrIS": [
        "AF","AF","AF","AF",None,"AF","AF","AF",
        "AF","AF","AF","AF","AF","AF","AF"
    ],
    "WAIS": [
        "AF","AF","AF","NV","NV","AF","AF","NV",
        "AF","NV","AF","AF","AF","AF","NV"
    ]
})

# Calculate percentages (omit blanks)
summary = pd.DataFrame(index=["NV", "AF"])

for sheet in df.columns:
    counts = df[sheet].dropna().value_counts(normalize=True) * 100
    summary[sheet] = [counts.get("NV", 0), counts.get("AF", 0)]

# Plot
fig, ax = plt.subplots(figsize=(4,5))

# Greyscale colors
nv_color = "0.65"   # light gray
af_color = "0.30" 

ax.bar(summary.columns,
       summary.loc["NV"],
       color=nv_color,
       width=0.42)

ax.bar(summary.columns,
       summary.loc["AF"],
       bottom=summary.loc["NV"],
       color=af_color,
       width=0.42)

# Labels inside bars
for i, sheet in enumerate(summary.columns):

    nv = summary.loc["NV", sheet]
    af = summary.loc["AF", sheet]

    if nv > 5:
        ax.text(i, nv/2,
                "natural\nvariability",
                ha="center", va="center",
                fontsize=9.5,
                fontweight = 'bold',
                linespacing = 1.5)

    y = nv + af/2

    ax.text(i, y + 5,
            "anthro-\npogenic",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="semibold",
            color="white")
    
    ax.text(i, y - 3,
            "forcing",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="semibold",
            color="white")

# Formatting
ax.set_ylim(0, 100)
ax.set_title("Mass change observations")
ax.set_ylabel("Expert responses",
              fontweight = 'bold',
              fontsize = 12)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f"{i}%" for i in range(0, 101, 20)])
ax.set_xticks(range(len(summary.columns)))
ax.set_xticklabels(summary.columns, fontsize=12, fontweight="bold")


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


outfile = os.path.join(fig_dir, "mass_change_observations.png")
plt.savefig(outfile, dpi=300, bbox_inches="tight")

print(f"Figure saved to: {outfile}")
plt.close(fig)  # non-interactive plotting for reproducible runs