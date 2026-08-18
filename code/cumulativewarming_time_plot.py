#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:23:20 2026

@author: elizabeth
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CUMULATIVE WARMING TRAJECTORIES
# ============================================================

years = np.array([2000, 2020, 2050, 2100, 2200, 2300])

cumulative_warming = {
    "Low":          [0, 19, 62.5, 155, 355, 555],
    "Intermediate": [0, 19, 67, 204.5, 554.5, 904.5],
    "High":         [0, 19, 74.5, 262, 762, 1262],
    #"Overshoot":    [0, 19, 67, 204.5, 479.5, 679.5],
    #"Commitment":   [0, 19, 56.5, 121.5, 251.5, 381.5],
}

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

for scenario, values in cumulative_warming.items():
    ax.plot(
        years,
        values,
        marker="o",
        linewidth=2,
        markersize=5,
        label=scenario,
    )

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Cumulative warming (K·yr)", fontsize=12)

ax.set_xlim(2000, 2300)
ax.set_xticks(years)

ax.grid(True, alpha=0.3, linestyle = '--')
ax.legend(frameon=False)

sns.despine()
fig.tight_layout()
plt.show()