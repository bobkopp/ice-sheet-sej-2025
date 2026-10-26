#!/usr/bin/env python3
"""Plot *selection probabilities* (not realized fractions), using final 01/01B output.

The plotted surfaces are computed from the same normalized Euclidean IDW p=1
formula and two-checkpoint deterministic boundary specified in stages 01/01B.


Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def selection_weights(X, refs):
    # Coordinates and reference points are normalized by (HIGH - LOW) at each checkpoint.
    d = np.sqrt(((X[:, None, :] - refs[None, :, :]) ** 2).sum(axis=2))
    tiny = d < 1e-12
    raw = np.divide(1., d, out=np.zeros_like(d), where=~tiny)
    raw = np.where(tiny.any(axis=1)[:, None], tiny.astype(float), raw)
    w = raw / raw.sum(axis=1, keepdims=True)
    low_both = (X[:, 0] <= 0) & (X[:, 1] <= 0)
    high_both = (X[:, 0] >= 1) & (X[:, 1] >= 1)
    w[low_both, :] = [1, 0, 0]
    w[high_both, :] = [0, 0, 1]
    return w


def run(family, selection_dir, output_dir):
    prefix = "cmip7_" if family == "CMIP7" else ""
    sel = pd.read_csv(selection_dir / (prefix + "fair_b25_baseline_scenario_selection.csv"))
    reference = pd.read_csv(selection_dir / (prefix + "fair_b25_scenario_selection_reference_checkpoints.csv"))
    reference = reference.set_index("year").loc[[2100, 2200]]
    for field in ("LOW", "INTERMEDIATE", "HIGH"):
        if field not in reference:
            raise ValueError("Missing reference checkpoint: " + field)
    scale = (reference["HIGH"] - reference["LOW"]).to_numpy(dtype=float)
    origin = reference["LOW"].to_numpy(dtype=float)
    refs = np.column_stack([(reference[k].to_numpy(dtype=float) - origin) / scale
                            for k in ("LOW", "INTERMEDIATE", "HIGH")]).T
    pos = sel[["reference_range_position_2100", "reference_range_position_2200"]].to_numpy(float)
    # Validate the formula against actual stage-01/01B output before plotting.
    sample = np.linspace(0, len(sel)-1, min(500, len(sel)), dtype=int)
    calculated = selection_weights(pos[sample], refs)
    original = sel[["weight_low", "weight_intermediate", "weight_high"]].to_numpy(float)[sample]
    error = np.max(np.abs(calculated - original))
    if error > 1e-8:
        raise ValueError(f"{family} probability formula did not reproduce saved weights (max diff {error:.3e})")
    xlo, xhi = min(-0.4, np.quantile(pos[:, 0], .01)), max(1.4, np.quantile(pos[:, 0], .99))
    ylo, yhi = min(-0.4, np.quantile(pos[:, 1], .01)), max(1.4, np.quantile(pos[:, 1], .99))
    gx, gy = np.meshgrid(np.linspace(xlo, xhi, 150), np.linspace(ylo, yhi, 150))
    weights = selection_weights(np.column_stack([gx.ravel(), gy.ravel()]), refs).reshape(*gx.shape, 3)
    rng = np.random.default_rng(1234)
    shown = rng.choice(len(pos), min(3000, len(pos)), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharex=True, sharey=True, constrained_layout=True)
    for i, (ax, name) in enumerate(zip(axes, ("LOW", "INTERMEDIATE", "HIGH"))):
        ax.contourf(gx, gy, weights[:, :, i], levels=np.linspace(0, 1, 21), cmap="Greys", vmin=0, vmax=1)
        contour = ax.contour(gx, gy, weights[:, :, i], levels=[.1, .25, .5, .75, .9], colors="black", linewidths=.65)
        ax.clabel(contour, fontsize=7, fmt="%.2g")
        ax.scatter(pos[shown, 0], pos[shown, 1], s=1, c="tab:blue", alpha=.12, rasterized=True)
        ax.plot(refs[:, 0], refs[:, 1], marker="x", ls="", c="tab:red", ms=6)
        ax.set_title(f"{name} selection probability")
        ax.set_xlabel("Normalized cumulative warming at 2100")
        ax.axvline(0, color="gray", ls=":", lw=.5)
        ax.axvline(1, color="gray", ls=":", lw=.5)
        ax.axhline(0, color="gray", ls=":", lw=.5)
        ax.axhline(1, color="gray", ls=":", lw=.5)
    axes[0].set_ylabel("Normalized cumulative warming at 2200")
    fig.suptitle(f"{family} FaIR realizations — SEJ warming-scenario selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{family.lower()}_selection_probabilities.png"
    fig.savefig(dest, dpi=220)
    plt.close(fig)
    print("Validated stage-selection probabilities (maximum sampled error", f"{error:.2e})")
    print("Saved:", dest)


def main():
    root = os.environ.get("SEJ_CORE_OUTPUT_ROOT")
    fig_root = os.environ.get("FIG_OUTPUT_DIR")
    if not root or not fig_root:
        raise SystemExit("Source test_environment.sh and set FIG_OUTPUT_DIR first")
    selection_dir = Path(root).expanduser().resolve() / "01_warming_selection"
    output_dir = Path(fig_root).expanduser().resolve() / "supplementary" / "scenario_selection_probabilities"
    for family in ("CMIP6", "CMIP7"):
        run(family, selection_dir, output_dir)

if __name__ == "__main__":
    main()
