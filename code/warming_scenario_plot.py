#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:01:00 2026

@author: elizabeth
"""

import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# Colors
low_color  = "#0072B2"   # Low
med_color  = "#6A3D9A"   # Intermediate
high_color = "#D55E00"   # High


# Data
years = [2000, 2020, 2050, 2100, 2200, 2300]

warming = {
    "Low": [0.7, 1.2, 1.7, 2.0, 2.0, 2.0],
    "Intermediate": [0.7, 1.2, 2.0, 3.5, 3.5, 3.5],
    "High": [0.7, 1.2, 2.5, 5.0, 5.0, 5.0],
}

fig, ax = plt.subplots(figsize=(6,4))

ax.plot(years, warming["Low"], color=low_color, linestyle="-", linewidth=2.5, label="Low")
ax.plot(years, warming["Intermediate"], color=med_color, linestyle="--", linewidth=2.5, label="Intermediate")
ax.plot(years, warming["High"], color=high_color, linestyle="-.", linewidth=2.5, label="High")

ax.set_xlabel("Year",
              fontweight = 'bold',
              fontsize = 12)
ax.set_ylabel("Global mean surface air\ntemperature(°C above 1850–1900)",
              fontweight = 'bold',
              fontsize = 12)
ax.set_xlim(2000, 2300)
ax.set_ylim(0, 5.5)

ax.tick_params(axis="both", labelsize=12)

ax.legend(frameon=False)
ax.grid(linestyle = '--',
        alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/warming_scenarios.png", dpi=300, bbox_inches="tight")
plt.show()