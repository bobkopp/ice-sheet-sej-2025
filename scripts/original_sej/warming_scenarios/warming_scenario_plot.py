#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:01:00 2026

Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""

from figure_labels import WARMING_REFERENCE_TRAJECTORIES, WARMING_REFERENCE_YEARS

from figure_settings import PROJECT_ROOT, DATA_ROOT, UPSTREAM_ROOT, FIGURE_ROOT, output_for, DPI, SHOW_FIGURES, B25_MC_FILE, B19_ADJUSTED_FILE, CMIP6_ANNUAL_DIR, CMIP7_ANNUAL_DIR, CMIP6_SELECTION_FILE, CMIP7_SELECTION_FILE
from figure_helpers import save_figure, maybe_show
from figure_style import WARMING_COLORS

import matplotlib.pyplot as plt
import os
import seaborn as sns

# Colors
low_color = WARMING_COLORS[2.0]
med_color = WARMING_COLORS[3.5]
high_color = WARMING_COLORS[5.0]


# Data
years = list(WARMING_REFERENCE_YEARS)

warming = {key: list(values) for key, values in WARMING_REFERENCE_TRAJECTORIES.items()}

fig, ax = plt.subplots(figsize=(6,4))

ax.plot(years, warming["High"], color=high_color, linestyle="-.", linewidth=2.5, label="High")
ax.plot(years, warming["Intermediate"], color=med_color, linestyle="--", linewidth=2.5, label="Intermediate")
ax.plot(years, warming["Low"], color=low_color, linestyle="-", linewidth=2.5, label="Low")


ax.set_xlabel("Year",
              fontweight = 'bold',
              fontsize = 12)
ax.set_ylabel("Global mean surface air\ntemperature (°C above 1850–1900)",
              fontweight = 'bold',
              fontsize = 12)
ax.set_xlim(2000, 2300)
ax.set_ylim(0, 5.5)

ax.tick_params(axis="both", labelsize=12)

ax.legend(frameon=False)
ax.grid(linestyle = '--',
        alpha=0.3)

sns.despine()
plt.tight_layout()
save_figure(fig, output_for("001_warming_reference") / "warming_scenarios.png")
maybe_show()