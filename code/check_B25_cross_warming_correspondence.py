#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
B25 cross-warming draw-level correspondence diagnostics
========================================================

Tests whether Monte Carlo row i at 2.0°C, 3.5°C, and 5.0°C appears to
carry meaningful draw-level correspondence across warming scenarios.

IMPORTANT:
These diagnostics can show whether same-row values are statistically
associated across warming levels. They cannot prove that the association
was intentionally constructed as a physically coherent cross-warming draw.
The definitive test remains inspection of the Monte Carlo generation code.

Analyzes process-level responses for EAIS, WAIS, and GrIS plus total
responses for EAIS, WAIS, GrIS, Total AIS, and Total ice-sheet SLR at
2100, 2200, and 2300.

For each response/year/warming pair, calculates:
    - Spearman rank correlation
    - Pearson correlation of raw values
    - Pearson correlation of percentile ranks
    - mean / median / p90 / p95 absolute percentile-rank difference
    - fraction within 0.05 and 0.10 percentile-rank difference
    - fraction in the same decile and same quintile

Independent-rank benchmark:
    E[|U-V|] = 1/3 ≈ 0.333 for independent U,V ~ Uniform(0,1)

Outputs are saved under:
    output/cross_warming_correspondence/

No temporal coupling or FaIR interpolation is used here.
This operates on the original B25 Monte Carlo samples.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import B25data_processing as b25
except ImportError as exc:
    raise ImportError(
        "Could not import B25data_processing.py. Save this script in the "
        "same project directory as B25data_processing.py, or add that "
        "directory to sys.path."
    ) from exc


# ============================================================
# 1. SETTINGS
# ============================================================

YEARS = [2100, 2200, 2300]
WARMINGS = [2.0, 3.5, 5.0]
WARMING_PAIRS = [(2.0, 3.5), (2.0, 5.0), (3.5, 5.0)]

PROCESS_ICE_SHEETS = ["EAIS", "WAIS", "GrIS"]
PROCESS_ORDER = [
    "Accumulation",
    "Discharge",
    "Runoff",
    "Runoff minus accumulation",
]
TOTAL_ICE_SHEETS = [
    "EAIS",
    "WAIS",
    "GrIS",
    "Total AIS",
    "Total ice-sheet SLR",
]

OUTPUT_DIR = Path(b25.project_dir) / "output" / "cross_warming_correspondence"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MAKE_INDIVIDUAL_SCATTERS = False
SCATTER_N = 5000
SCATTER_SEED = 20260810


# ============================================================
# 2. HELPERS
# ============================================================

def print_heading(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def warming_label(value):
    return f"{value:.1f}°C"


def pair_label(w1, w2):
    return f"{warming_label(w1)}–{warming_label(w2)}"


def percentile_ranks(values):
    """Average fractional ranks, placed strictly inside (0,1)."""
    numeric = pd.to_numeric(values, errors="raise")
    n = len(numeric)
    ranks = numeric.rank(method="average", pct=False).to_numpy(dtype=float)
    return (ranks - 0.5) / n


def safe_corr(x, y):
    if len(x) < 2:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def build_response_table(process_ext, total_ext):
    process = process_ext.loc[
        process_ext["ice_sheet"].isin(PROCESS_ICE_SHEETS)
        & process_ext["process"].isin(PROCESS_ORDER)
        & process_ext["year"].isin(YEARS)
        & process_ext["warming"].isin(WARMINGS)
    ].copy()
    process["response_type"] = "Process"
    process["response_label"] = process["ice_sheet"] + " — " + process["process"]

    totals = total_ext.loc[
        total_ext["ice_sheet"].isin(TOTAL_ICE_SHEETS)
        & total_ext["year"].isin(YEARS)
        & total_ext["warming"].isin(WARMINGS)
    ].copy()
    totals["response_type"] = "Total"
    totals["response_label"] = totals["ice_sheet"] + " — Total"

    keep = [
        "sample_id", "year", "warming", "ice_sheet", "process", "value",
        "response_type", "response_label",
    ]
    combined = pd.concat([process[keep], totals[keep]], ignore_index=True)
    combined["value"] = pd.to_numeric(combined["value"], errors="raise")
    return combined


def correspondence_metrics(subset, w1, w2):
    wide = (
        subset.loc[
            subset["warming"].isin([w1, w2]),
            ["sample_id", "warming", "value"],
        ]
        .pivot(index="sample_id", columns="warming", values="value")
        .dropna(subset=[w1, w2])
        .sort_index()
    )

    x = wide[w1].to_numpy(dtype=float)
    y = wide[w2].to_numpy(dtype=float)
    rank_x = percentile_ranks(wide[w1])
    rank_y = percentile_ranks(wide[w2])
    abs_diff = np.abs(rank_x - rank_y)

    decile_x = np.minimum((rank_x * 10).astype(int), 9)
    decile_y = np.minimum((rank_y * 10).astype(int), 9)
    quintile_x = np.minimum((rank_x * 5).astype(int), 4)
    quintile_y = np.minimum((rank_y * 5).astype(int), 4)

    return {
        "n": len(wide),
        "spearman_rho": safe_corr(rank_x, rank_y),
        "pearson_raw": safe_corr(x, y),
        "pearson_rank": safe_corr(rank_x, rank_y),
        "mean_abs_rank_diff": float(np.mean(abs_diff)),
        "median_abs_rank_diff": float(np.median(abs_diff)),
        "p90_abs_rank_diff": float(np.quantile(abs_diff, 0.90)),
        "p95_abs_rank_diff": float(np.quantile(abs_diff, 0.95)),
        "fraction_within_0.05_rank": float(np.mean(abs_diff <= 0.05)),
        "fraction_within_0.10_rank": float(np.mean(abs_diff <= 0.10)),
        "fraction_same_decile": float(np.mean(decile_x == decile_y)),
        "fraction_same_quintile": float(np.mean(quintile_x == quintile_y)),
    }


def draw_heatmap(table, metric, title, output_name, response_order, vmin=None, vmax=None):
    column_order = [
        f"{year}\n{pair_label(w1, w2)}"
        for year in YEARS
        for w1, w2 in WARMING_PAIRS
    ]
    heat = (
        table.pivot(index="response_label", columns="diagnostic_column", values=metric)
        .reindex(index=response_order, columns=column_order)
    )

    fig_height = max(7, 0.42 * len(heat.index))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title(title)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(image, ax=ax, shrink=0.8)
    cbar.set_label(metric)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{output_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{output_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_rank_scatter(response_subset, response_label, year, output_path):
    rng = np.random.default_rng(SCATTER_SEED)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)

    for ax, (w1, w2) in zip(axes, WARMING_PAIRS):
        wide = (
            response_subset.loc[
                response_subset["warming"].isin([w1, w2]),
                ["sample_id", "warming", "value"],
            ]
            .pivot(index="sample_id", columns="warming", values="value")
            .dropna(subset=[w1, w2])
        )
        rank_x = percentile_ranks(wide[w1])
        rank_y = percentile_ranks(wide[w2])
        n = len(wide)
        idx = rng.choice(n, size=min(SCATTER_N, n), replace=False)
        rho = safe_corr(rank_x, rank_y)

        ax.scatter(rank_x[idx], rank_y[idx], s=5, alpha=0.25)
        ax.plot([0, 1], [0, 1], linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{pair_label(w1, w2)}\nSpearman ρ = {rho:.2f}")
        ax.set_xlabel(f"{warming_label(w1)} percentile rank")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Higher-warming percentile rank")
    fig.suptitle(
        f"{response_label}, {year}\nSame-row cross-warming rank correspondence"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. LOAD ORIGINAL B25 MONTE CARLO DATA
# ============================================================

print_heading("LOADING ORIGINAL B25 MONTE CARLO DATA")
print(f"Project directory:\n{b25.project_dir}")
print(f"\nMonte Carlo input file:\n{b25.data_file}")

data = b25.load_and_prepare_data(file_path=b25.data_file)
process_ext = data["process_ext"].copy()
total_ext = data["mc_total_ext"].copy()
responses = build_response_table(process_ext, total_ext)

print(f"\nCombined diagnostic records: {len(responses):,}")
print(f"Unique response labels: {responses['response_label'].nunique()}")


# ============================================================
# 4. RESPONSE ORDER
# ============================================================

response_order = []
for ice_sheet in PROCESS_ICE_SHEETS:
    for process in PROCESS_ORDER:
        response_order.append(f"{ice_sheet} — {process}")
for ice_sheet in TOTAL_ICE_SHEETS:
    response_order.append(f"{ice_sheet} — Total")

available = set(responses["response_label"].unique())
response_order = [r for r in response_order if r in available]


# ============================================================
# 5. COMPLETENESS CHECK
# ============================================================

print_heading("COMPLETENESS CHECK")
counts = (
    responses.groupby(["response_label", "year", "warming"], observed=True)["sample_id"]
    .nunique()
    .rename("n")
    .reset_index()
)
print(counts.groupby(["year", "warming"], observed=True)["n"].agg(["min", "max"]).to_string())
counts.to_csv(OUTPUT_DIR / "cross_warming_sample_counts.csv", index=False)


# ============================================================
# 6. ALL CROSS-WARMING DIAGNOSTICS
# ============================================================

print_heading("CALCULATING CROSS-WARMING CORRESPONDENCE")
records = []

grouped = responses.groupby(
    ["response_label", "response_type", "ice_sheet", "process", "year"],
    observed=True,
)

for (response_label, response_type, ice_sheet, process, year), subset in grouped:
    for w1, w2 in WARMING_PAIRS:
        metrics = correspondence_metrics(subset, w1, w2)
        records.append({
            "response_label": response_label,
            "response_type": response_type,
            "ice_sheet": ice_sheet,
            "process": process,
            "year": int(year),
            "warming_1": w1,
            "warming_2": w2,
            "warming_pair": pair_label(w1, w2),
            "diagnostic_column": f"{int(year)}\n{pair_label(w1, w2)}",
            **metrics,
        })

diagnostics = pd.DataFrame(records).sort_values(
    ["year", "response_type", "ice_sheet", "process", "warming_1", "warming_2"]
).reset_index(drop=True)

diagnostics.to_csv(
    OUTPUT_DIR / "cross_warming_correspondence_all_responses.csv",
    index=False,
)

print(
    diagnostics[
        [
            "response_label", "year", "warming_pair", "n", "spearman_rho",
            "mean_abs_rank_diff", "fraction_same_decile",
        ]
    ].round(3).to_string(index=False)
)


# ============================================================
# 7. SUMMARY BY YEAR / WARMING PAIR
# ============================================================

print_heading("SUMMARY ACROSS RESPONSES")
summary_by_pair = (
    diagnostics.groupby(["year", "warming_pair"], observed=True)
    .agg(
        n_responses=("response_label", "nunique"),
        spearman_mean=("spearman_rho", "mean"),
        spearman_median=("spearman_rho", "median"),
        spearman_min=("spearman_rho", "min"),
        spearman_max=("spearman_rho", "max"),
        mean_abs_rank_diff_mean=("mean_abs_rank_diff", "mean"),
        mean_abs_rank_diff_median=("mean_abs_rank_diff", "median"),
        same_decile_mean=("fraction_same_decile", "mean"),
    )
    .reset_index()
)
print(summary_by_pair.round(3).to_string(index=False))
summary_by_pair.to_csv(
    OUTPUT_DIR / "cross_warming_summary_by_year_and_pair.csv",
    index=False,
)


# ============================================================
# 8. STRONGEST / WEAKEST APPARENT MATCHING
# ============================================================

print_heading("STRONGEST AND WEAKEST APPARENT CORRESPONDENCE")
ranked = diagnostics.copy()
ranked["correspondence_score"] = (
    ranked["spearman_rho"] - ranked["mean_abs_rank_diff"]
)

strongest = ranked.sort_values(
    ["correspondence_score", "spearman_rho"], ascending=False
).head(25)
weakest = ranked.sort_values(
    ["correspondence_score", "spearman_rho"], ascending=True
).head(25)

show_cols = [
    "response_label", "year", "warming_pair", "spearman_rho",
    "mean_abs_rank_diff", "fraction_same_decile",
]
print("\nStrongest apparent correspondence:")
print(strongest[show_cols].round(3).to_string(index=False))
print("\nWeakest apparent correspondence:")
print(weakest[show_cols].round(3).to_string(index=False))

strongest.to_csv(OUTPUT_DIR / "cross_warming_strongest_correspondence.csv", index=False)
weakest.to_csv(OUTPUT_DIR / "cross_warming_weakest_correspondence.csv", index=False)


# ============================================================
# 9. HEATMAPS
# ============================================================

print_heading("CREATING DIAGNOSTIC FIGURES")

draw_heatmap(
    diagnostics,
    "spearman_rho",
    "B25 same-row cross-warming Spearman correlation\nOriginal Monte Carlo samples",
    "cross_warming_spearman_heatmap",
    response_order,
    vmin=-1,
    vmax=1,
)

draw_heatmap(
    diagnostics,
    "mean_abs_rank_diff",
    "B25 same-row mean absolute percentile-rank difference\nIndependent-rank benchmark = 0.333",
    "cross_warming_mean_abs_rank_difference_heatmap",
    response_order,
    vmin=0,
    vmax=0.5,
)

draw_heatmap(
    diagnostics,
    "fraction_same_decile",
    "B25 fraction of same-row draws in the same decile across warming",
    "cross_warming_same_decile_heatmap",
    response_order,
    vmin=0,
    vmax=1,
)


# ============================================================
# 10. SPEARMAN VS RANK-DIFFERENCE FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))
for warming_pair, group in diagnostics.groupby("warming_pair", observed=True):
    ax.scatter(
        group["spearman_rho"],
        group["mean_abs_rank_diff"],
        alpha=0.65,
        label=warming_pair,
    )

ax.axhline(1 / 3, linestyle="--", linewidth=1, label="Independent-rank mean |Δrank| = 1/3")
ax.axvline(0, linewidth=1)
ax.set_xlabel("Same-row Spearman correlation")
ax.set_ylabel("Mean absolute percentile-rank difference")
ax.set_title("Cross-warming row correspondence across B25 responses")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "cross_warming_spearman_vs_rank_difference.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    FIGURE_DIR / "cross_warming_spearman_vs_rank_difference.pdf",
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 11. DISTRIBUTION OF SPEARMAN VALUES
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))
for warming_pair, group in diagnostics.groupby("warming_pair", observed=True):
    ax.hist(
        group["spearman_rho"].dropna(),
        bins=np.linspace(-1, 1, 21),
        alpha=0.45,
        label=warming_pair,
    )
ax.axvline(0, linewidth=1)
ax.set_xlabel("Same-row Spearman correlation")
ax.set_ylabel("Number of response-year combinations")
ax.set_title("Distribution of cross-warming Spearman correlations")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "cross_warming_spearman_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 12. MEDIAN SPEARMAN BY YEAR
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(YEARS))
width = 0.24

for j, (w1, w2) in enumerate(WARMING_PAIRS):
    label = pair_label(w1, w2)
    group = (
        summary_by_pair.loc[summary_by_pair["warming_pair"] == label]
        .set_index("year")
        .reindex(YEARS)
    )
    ax.bar(
        x + (j - 1) * width,
        group["spearman_median"],
        width=width,
        label=label,
    )

ax.axhline(0, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(YEARS)
ax.set_ylim(-1, 1)
ax.set_xlabel("Projection year")
ax.set_ylabel("Median Spearman correlation across responses")
ax.set_title("Median same-row cross-warming rank correlation")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "cross_warming_median_spearman_by_year.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 13. AUTOMATIC EXAMPLE RANK-SCATTERS
# ============================================================

ranked_clean = ranked.dropna(subset=["spearman_rho", "mean_abs_rank_diff"]).copy()
example_rows = []

if len(ranked_clean) > 0:
    weakest_row = ranked_clean.sort_values("correspondence_score").iloc[0]
    strongest_row = ranked_clean.sort_values("correspondence_score").iloc[-1]
    median_target = ranked_clean["correspondence_score"].median()
    median_index = (
        ranked_clean["correspondence_score"].sub(median_target).abs().idxmin()
    )
    median_row = ranked_clean.loc[median_index]
    example_rows = [
        ("weakest", weakest_row),
        ("middle", median_row),
        ("strongest", strongest_row),
    ]

for descriptor, row in example_rows:
    response_label = row["response_label"]
    year = int(row["year"])
    response_subset = responses.loc[
        (responses["response_label"] == response_label)
        & (responses["year"] == year)
    ]
    safe_name = (
        response_label.replace(" ", "_").replace("—", "-").replace("/", "-")
    )
    make_rank_scatter(
        response_subset,
        response_label,
        year,
        FIGURE_DIR / f"rank_scatter_{descriptor}_{safe_name}_{year}.png",
    )


# ============================================================
# 14. OPTIONAL ALL INDIVIDUAL SCATTERS
# ============================================================

if MAKE_INDIVIDUAL_SCATTERS:
    scatter_dir = FIGURE_DIR / "individual_rank_scatters"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    for response_label in response_order:
        for year in YEARS:
            response_subset = responses.loc[
                (responses["response_label"] == response_label)
                & (responses["year"] == year)
            ]
            if response_subset.empty:
                continue
            safe_name = (
                response_label.replace(" ", "_").replace("—", "-").replace("/", "-")
            )
            make_rank_scatter(
                response_subset,
                response_label,
                year,
                scatter_dir / f"{safe_name}_{year}.png",
            )


# ============================================================
# 15. SIMPLE DESCRIPTIVE FLAGS
# ============================================================

print_heading("SIMPLE INTERPRETATION FLAGS")
interpretation = diagnostics.copy()

# Descriptive flags only; not formal hypothesis tests.
interpretation["near_independent_rank_pattern"] = (
    interpretation["spearman_rho"].abs() <= 0.10
) & (
    interpretation["mean_abs_rank_diff"].between(0.30, 0.36)
)

interpretation["strong_rank_correspondence"] = (
    interpretation["spearman_rho"] >= 0.80
) & (
    interpretation["mean_abs_rank_diff"] <= 0.15
)

flag_summary = pd.DataFrame({
    "n_diagnostics": [len(interpretation)],
    "n_near_independent_pattern": [int(interpretation["near_independent_rank_pattern"].sum())],
    "percent_near_independent_pattern": [100 * interpretation["near_independent_rank_pattern"].mean()],
    "n_strong_rank_correspondence": [int(interpretation["strong_rank_correspondence"].sum())],
    "percent_strong_rank_correspondence": [100 * interpretation["strong_rank_correspondence"].mean()],
})

print(flag_summary.round(2).to_string(index=False))
interpretation.to_csv(
    OUTPUT_DIR / "cross_warming_correspondence_with_flags.csv",
    index=False,
)
flag_summary.to_csv(
    OUTPUT_DIR / "cross_warming_correspondence_flag_summary.csv",
    index=False,
)


# ============================================================
# 16. FINISHED
# ============================================================

print_heading("FINISHED")
print(f"Diagnostic tables saved to:\n{OUTPUT_DIR}")
print(f"\nFigures saved to:\n{FIGURE_DIR}")
print("\nMost useful outputs:")
print("  1. cross_warming_correspondence_all_responses.csv")
print("  2. cross_warming_summary_by_year_and_pair.csv")
print("  3. cross_warming_weakest_correspondence.csv")
print("  4. figures/cross_warming_spearman_heatmap.png")
print("  5. figures/cross_warming_mean_abs_rank_difference_heatmap.png")
print("  6. figures/cross_warming_spearman_vs_rank_difference.png")
print("\nReference benchmark:")
print("  If row alignment were essentially independent across warming, expect")
print("  Spearman rho near 0 and mean absolute rank difference near 1/3.")
