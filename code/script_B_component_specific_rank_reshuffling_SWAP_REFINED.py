#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCRIPT B: Component-specific copula + intact-row rank reshuffling prototype
===========================================================================

Purpose
-------
Prototype Copilot Alternative 1 for ONE B25 warming experiment:

    B25 3.5°C  <->  DP21 RCP4.5

This script REPLACES the single Total-AIS common-rank temporal coupling with:

    1. a full 3 x 3 EAIS Gaussian temporal copula from DP21,
    2. a full 3 x 3 WAIS Gaussian temporal copula from DP21,
    3. separate simulated EAIS and WAIS percentile-rank histories,
    4. a one-to-one rank-vector matching step at each projection year,
    5. movement of the COMPLETE original B25 yearly panel.

Why this matters
----------------
The desired temporal state for trajectory i at year t is now:

    (u_EAIS[i,t], u_WAIS[i,t])

rather than one common Total-AIS percentile.

For each projection year, every original B25 row is characterized by its
3.5°C empirical ranks:

    (r_EAIS[j,t], r_WAIS[j,t])

The script finds a one-to-one assignment between desired trajectories and
actual B25 rows that approximately minimizes squared rank-vector distance.

Once a B25 row is assigned, the ENTIRE row/panel for that year moves
together, including:

    EAIS
    WAIS
    GrIS
    process variables
    Total ice-sheet SLR
    all warming-level columns belonging to that projection year

Thus:

    - no new B25 sea-level values are generated;
    - every original yearly B25 row is used exactly once;
    - every year-specific marginal distribution is preserved exactly;
    - the original within-year joint panel is preserved exactly;
    - EAIS and WAIS can have different temporal rank persistence.

Important GrIS limitation
-------------------------
DP21 is Antarctic and does not supply a GrIS temporal copula.

Therefore GrIS is NOT used as a target dimension in this prototype.
GrIS values move with the intact B25 row selected from the joint EAIS/WAIS
rank match. The resulting GrIS temporal dependence is diagnostic only and
is not claimed to be physically targeted.

This script uses the FULL 3-year Gaussian copula matrices estimated by
Script A, rather than the first-order Markov approximation.

Expected prerequisite
---------------------
Run:

    script_A_component_specific_dp21_copulas.py

first, producing:

    output/component_specific_temporal_copulas/
        dp21_component_full_correlations.csv

Outputs
-------
Under:

    output/component_specific_temporal_coupling/
        b25_3.5C_dp21_rcp45_component_specific/

the script saves:

    *_coupled_samples.csv
    *_coupling_metadata.csv
    *_target_copulas.csv
    *_temporal_correlation_diagnostics.csv
    *_within_year_dependence_diagnostics.csv
    *_assignment_quality.csv
    *_swap_refinement_summary.csv
    *_marginal_preservation.csv

Figures:
    temporal_correlation_comparison.png
    assignment_rank_scatter_2100.png
    assignment_rank_scatter_2200.png
    assignment_rank_scatter_2300.png
    assignment_distance_by_year.png

Method note
-----------
This script uses a fast local/bin one-to-one matcher followed by deterministic
pairwise swap refinement of poorly matched trajectories. The final assignment
remains a complete permutation at every stage, so exact B25 marginal and
within-year panel preservation are retained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.stats import norm, rankdata


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

PROJECT_DIR = Path(
    "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025"
)

sys.path.insert(
    0,
    str(PROJECT_DIR),
)

from B25data_processing import (
    load_and_prepare_data,
    project_dir,
    data_file,
)


# ============================================================
# 2. EXPERIMENT SETTINGS
# ============================================================

WARMING = 3.5
WARMING_LABEL = "3.5C"
DP21_SCENARIO = "rcp45"
DP21_SCENARIO_LABEL = "RCP4.5"

YEARS = [
    2100,
    2200,
    2300,
]

RANDOM_SEED = 1234

# Local/bin matching settings.
N_RANK_BINS = 100
LOCAL_K = 64
MATCHING_SEED = 24680

# Swap-refinement settings.
# The fast local matcher already handles most trajectories well; these settings
# target the poorly matched tail while preserving the one-to-one permutation.
SWAP_REFINE_THRESHOLD = 0.10
SWAP_PARTNER_K = 64
SWAP_MAX_PASSES = 8
SWAP_MIN_IMPROVEMENT = 1e-10

# Number of points shown in each diagnostic scatterplot.
N_SCATTER = 5000
SCATTER_SEED = 20260810


# ============================================================
# 3. INPUT / OUTPUT PATHS
# ============================================================

SCRIPT_A_DIR = (
    PROJECT_DIR
    / "output"
    / "component_specific_temporal_copulas"
)

SCRIPT_A_CORRELATION_FILE = (
    SCRIPT_A_DIR
    / "dp21_component_full_correlations.csv"
)

OUTPUT_ROOT = (
    PROJECT_DIR
    / "output"
    / "component_specific_temporal_coupling"
)

EXPERIMENT_NAME = (
    f"b25_{WARMING_LABEL}_dp21_{DP21_SCENARIO}"
    "_component_specific"
)

OUTPUT_DIR = (
    OUTPUT_ROOT
    / EXPERIMENT_NAME
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def print_heading(
    title: str,
):
    print(
        "\n"
        + "=" * 88
    )

    print(
        title
    )

    print(
        "=" * 88
    )


def require_file(
    path: Path,
):
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def empirical_uniform_ranks(
    values: np.ndarray,
):
    """
    Return percentile-style empirical ranks in (0,1).

    Ties receive average ranks.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(
        values
    )

    return (
        rankdata(
            values,
            method="average",
        )
        - 0.5
    ) / n


def gaussian_implied_spearman(
    rho: float,
):
    return (
        6.0
        / np.pi
        * np.arcsin(
            rho
            / 2.0
        )
    )


# ============================================================
# 5. LOAD FULL 3 x 3 COMPONENT COPULAS FROM SCRIPT A
# ============================================================

def load_component_copula(
    correlation_file: Path,
    scenario: str,
    component: str,
    years: list[int],
):
    """
    Reconstruct the full latent Gaussian correlation matrix for one
    component/scenario from Script A's pairwise output.
    """

    require_file(
        correlation_file
    )

    corr_df = pd.read_csv(
        correlation_file
    )

    subset = corr_df.loc[
        (
            corr_df[
                "scenario"
            ]
            == scenario
        )
        & (
            corr_df[
                "component"
            ]
            == component
        )
    ].copy()

    if subset.empty:
        raise ValueError(
            "No Script-A correlations found for "
            f"{scenario}, {component}."
        )

    matrix = np.eye(
        len(
            years
        ),
        dtype=float,
    )

    year_index = {
        year: i
        for i, year in enumerate(
            years
        )
    }

    for _, row in subset.iterrows():

        year1 = int(
            row[
                "year1"
            ]
        )

        year2 = int(
            row[
                "year2"
            ]
        )

        if (
            year1 not in year_index
            or year2 not in year_index
        ):
            continue

        rho = float(
            row[
                "latent_gaussian_pearson"
            ]
        )

        i = year_index[
            year1
        ]

        j = year_index[
            year2
        ]

        matrix[
            i,
            j,
        ] = rho

        matrix[
            j,
            i,
        ] = rho

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    if np.min(
        eigenvalues
    ) <= 0:
        raise ValueError(
            f"{component} copula matrix is not positive definite.\n"
            f"Eigenvalues: {eigenvalues}\n"
            f"Matrix:\n{matrix}"
        )

    return matrix


# ============================================================
# 6. SIMULATE COMPONENT-SPECIFIC FULL 3-YEAR RANK HISTORIES
# ============================================================

def simulate_component_rank_histories(
    correlation_matrix: np.ndarray,
    n_samples: int,
    seed: int,
):
    """
    Draw a full multivariate Gaussian vector across 2100/2200/2300,
    then convert each year to an exact empirical rank permutation.

    The Gaussian values themselves are latent only.
    """

    rng = np.random.default_rng(
        seed
    )

    latent = rng.multivariate_normal(
        mean=np.zeros(
            len(
                YEARS
            )
        ),
        cov=correlation_matrix,
        size=n_samples,
    )

    desired_ranks = np.empty_like(
        latent,
        dtype=float,
    )

    rank_positions = np.empty_like(
        latent,
        dtype=int,
    )

    for j in range(
        latent.shape[1]
    ):

        order = np.argsort(
            latent[
                :,
                j,
            ],
            kind="mergesort",
        )

        positions = np.empty(
            n_samples,
            dtype=int,
        )

        positions[
            order
        ] = np.arange(
            n_samples,
            dtype=int,
        )

        rank_positions[
            :,
            j,
        ] = positions

        desired_ranks[
            :,
            j,
        ] = (
            positions
            + 0.5
        ) / n_samples

    return (
        latent,
        desired_ranks,
        rank_positions,
    )


# ============================================================
# 7. IDENTIFY COMPLETE YEAR PANELS
# ============================================================

def get_columns_by_year(
    meta_df: pd.DataFrame,
    years: list[int],
):
    columns_by_year = {}

    for year in years:

        year_columns = (
            meta_df.loc[
                meta_df[
                    "year"
                ]
                == year,
                "variable",
            ]
            .tolist()
        )

        if not year_columns:
            raise ValueError(
                f"No B25 scientific variables found for {year}."
            )

        columns_by_year[
            year
        ] = year_columns

    return columns_by_year


# ============================================================
# 8. BUILD B25 COMPONENT TOTALS AND RANK VECTORS
# ============================================================

def build_component_rank_table(
    mc_total_ext: pd.DataFrame,
    warming: float,
    year: int,
):
    """
    For one warming/year, return one row per original B25 sample with:

        sample_id
        EAIS
        WAIS
        GrIS
        Total AIS
        rank_EAIS
        rank_WAIS
        rank_GrIS
        rank_Total_AIS
    """

    subset = mc_total_ext.loc[
        np.isclose(
            mc_total_ext[
                "warming"
            ],
            warming,
        )
        & (
            mc_total_ext[
                "year"
            ]
            == year
        )
        & mc_total_ext[
            "ice_sheet"
        ].isin(
            [
                "EAIS",
                "WAIS",
                "GrIS",
                "Total AIS",
            ]
        )
    ][
        [
            "sample_id",
            "ice_sheet",
            "value",
        ]
    ].copy()

    wide = (
        subset
        .pivot(
            index="sample_id",
            columns="ice_sheet",
            values="value",
        )
        .reset_index()
        .sort_values(
            "sample_id"
        )
        .reset_index(drop=True)
    )

    required = {
        "EAIS",
        "WAIS",
        "GrIS",
        "Total AIS",
    }

    missing = (
        required
        - set(
            wide.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing B25 totals at {year}: "
            f"{sorted(missing)}"
        )

    for component in [
        "EAIS",
        "WAIS",
        "GrIS",
        "Total AIS",
    ]:

        wide[
            f"rank_{component.replace(' ', '_')}"
        ] = empirical_uniform_ranks(
            wide[
                component
            ].to_numpy(
                dtype=float
            )
        )

    return wide


# ============================================================
# 9. LOCAL/BINNED MATCHING + SWAP REFINEMENT
# ============================================================

def binned_local_unique_assignment(
    desired_points: np.ndarray,
    candidate_points: np.ndarray,
    n_bins: int = N_RANK_BINS,
    local_k: int = LOCAL_K,
    seed: int = MATCHING_SEED,
):
    """Fast approximate one-to-one match in 2-D EAIS/WAIS rank space.

    Stage 1 matches points falling in the same fine rank bin.
    Stage 2 uses local K-nearest-neighbor matching for the remainder.
    Stage 3 greedily completes any small remainder.

    The returned assignment is always a complete permutation of the original
    B25 rows, so exact marginals and intact within-year panels are preserved.
    """
    rng = np.random.default_rng(seed)
    desired_points = np.asarray(desired_points, dtype=float)
    candidate_points = np.asarray(candidate_points, dtype=float)

    if desired_points.shape != candidate_points.shape or desired_points.ndim != 2:
        raise ValueError("desired_points and candidate_points must have identical 2-D shapes.")
    if desired_points.shape[1] != 2:
        raise ValueError("This matcher expects exactly two dimensions: EAIS and WAIS.")
    if not np.isfinite(desired_points).all() or not np.isfinite(candidate_points).all():
        raise ValueError("Non-finite rank values found.")

    n_samples = desired_points.shape[0]
    print(f"    local/bin matching with {n_bins} x {n_bins} rank grid", flush=True)

    desired_bins = np.clip(np.floor(desired_points * n_bins).astype(int), 0, n_bins - 1)
    candidate_bins = np.clip(np.floor(candidate_points * n_bins).astype(int), 0, n_bins - 1)
    desired_bin_id = desired_bins[:, 0] * n_bins + desired_bins[:, 1]
    candidate_bin_id = candidate_bins[:, 0] * n_bins + candidate_bins[:, 1]

    assignment = np.full(n_samples, -1, dtype=int)
    candidate_used = np.zeros(n_samples, dtype=bool)

    print("    stage 1: matching within identical rank bins...", flush=True)
    desired_by_bin = {}
    candidate_by_bin = {}
    for i, bid in enumerate(desired_bin_id):
        desired_by_bin.setdefault(int(bid), []).append(i)
    for j, bid in enumerate(candidate_bin_id):
        candidate_by_bin.setdefault(int(bid), []).append(j)

    same_bin_matches = 0
    for bid in set(desired_by_bin).intersection(candidate_by_bin):
        di = np.asarray(desired_by_bin[bid], dtype=int)
        ci = np.asarray(candidate_by_bin[bid], dtype=int)
        rng.shuffle(di)
        rng.shuffle(ci)
        n_match = min(len(di), len(ci))
        if n_match:
            assignment[di[:n_match]] = ci[:n_match]
            candidate_used[ci[:n_match]] = True
            same_bin_matches += n_match

    print(
        f"        same-bin matches: {same_bin_matches:,} / {n_samples:,} "
        f"({100 * same_bin_matches / n_samples:.2f}%)",
        flush=True,
    )

    unresolved = np.where(assignment < 0)[0]
    unused = np.where(~candidate_used)[0]
    if len(unresolved) != len(unused):
        raise RuntimeError("Desired/candidate remainder sizes differ.")

    print(f"    stage 2: local matching for {len(unresolved):,} remaining rows...", flush=True)
    if len(unresolved):
        tree = cKDTree(candidate_points[unused])
        k_eff = min(local_k, len(unused))
        distances, neighbors = tree.query(desired_points[unresolved], k=k_eff, workers=-1)
        if k_eff == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]

        # Give the harder points first access to their local candidates.
        order = np.argsort(distances[:, 0])[::-1]
        locally_used = np.zeros(len(unused), dtype=bool)
        local_matches = 0
        for pos in order:
            desired_index = unresolved[pos]
            for local_candidate in np.atleast_1d(neighbors[pos]):
                local_candidate = int(local_candidate)
                if not locally_used[local_candidate]:
                    original_candidate = unused[local_candidate]
                    assignment[desired_index] = original_candidate
                    locally_used[local_candidate] = True
                    candidate_used[original_candidate] = True
                    local_matches += 1
                    break
        print(f"        local KNN matches: {local_matches:,}", flush=True)

    unresolved = np.where(assignment < 0)[0]
    unused = np.where(~candidate_used)[0]
    if len(unresolved) != len(unused):
        raise RuntimeError("Final desired/candidate remainder sizes differ.")

    print(f"    stage 3: final remainder = {len(unresolved):,}", flush=True)
    if len(unresolved):
        unresolved = unresolved.copy()
        rng.shuffle(unresolved)
        remaining_candidates = set(unused.tolist())
        for count, desired_index in enumerate(unresolved, start=1):
            candidate_array = np.fromiter(remaining_candidates, dtype=int)
            delta = candidate_points[candidate_array] - desired_points[desired_index]
            best = candidate_array[np.argmin(np.sum(delta ** 2, axis=1))]
            assignment[desired_index] = best
            candidate_used[best] = True
            remaining_candidates.remove(int(best))
            if count % 1000 == 0 or count == len(unresolved):
                print(f"        final fallback: {count:,} / {len(unresolved):,}", flush=True)

    if np.any(assignment < 0):
        raise RuntimeError("Some desired trajectories remain unmatched.")
    if len(np.unique(assignment)) != n_samples:
        raise RuntimeError("Duplicate B25 rows were assigned.")
    if not np.array_equal(np.sort(assignment), np.arange(n_samples)):
        raise RuntimeError("Final assignment is not a complete permutation.")

    assigned_points = candidate_points[assignment]
    rank_errors = assigned_points - desired_points
    distances = np.sqrt(np.sum(rank_errors ** 2, axis=1))

    print("    initial local matching complete", flush=True)
    print(f"        mean rank distance   = {np.mean(distances):.5f}", flush=True)
    print(f"        median rank distance = {np.median(distances):.5f}", flush=True)
    print(f"        p95 rank distance    = {np.quantile(distances, 0.95):.5f}", flush=True)

    return assignment, assigned_points, rank_errors, distances


def refine_assignment_with_swaps(
    desired_points: np.ndarray,
    candidate_points: np.ndarray,
    assignment: np.ndarray,
    distance_threshold: float = SWAP_REFINE_THRESHOLD,
    partner_k: int = SWAP_PARTNER_K,
    max_passes: int = SWAP_MAX_PASSES,
    min_improvement: float = SWAP_MIN_IMPROVEMENT,
):
    """Refine a complete assignment using deterministic improving pair swaps.

    Only swaps that reduce the combined squared EAIS/WAIS rank error are
    accepted. Because candidate rows are only exchanged, the assignment stays
    a complete permutation at every step.
    """
    desired_points = np.asarray(desired_points, dtype=float)
    candidate_points = np.asarray(candidate_points, dtype=float)
    assignment = np.asarray(assignment, dtype=int).copy()
    n_samples = len(assignment)

    if len(np.unique(assignment)) != n_samples or not np.array_equal(
        np.sort(assignment), np.arange(n_samples)
    ):
        raise ValueError("Starting assignment must be a complete one-to-one permutation.")

    desired_tree = cKDTree(desired_points)
    candidate_tree = cKDTree(candidate_points)
    pass_records = []

    assigned_points = candidate_points[assignment]
    sqerr = np.sum((assigned_points - desired_points) ** 2, axis=1)
    distances = np.sqrt(sqerr)
    initial_total_cost = float(np.sum(sqerr))

    print("\n    starting swap refinement...", flush=True)
    print(f"        initial mean distance   = {np.mean(distances):.5f}", flush=True)
    print(f"        initial median distance = {np.median(distances):.5f}", flush=True)
    print(f"        initial p95 distance    = {np.quantile(distances, 0.95):.5f}", flush=True)
    print(
        f"        initial > {distance_threshold:.3f} = "
        f"{100 * np.mean(distances > distance_threshold):.2f}%",
        flush=True,
    )

    for pass_number in range(1, max_passes + 1):
        assigned_points = candidate_points[assignment]
        sqerr = np.sum((assigned_points - desired_points) ** 2, axis=1)
        distances = np.sqrt(sqerr)
        bad = np.where(distances > distance_threshold)[0]
        bad = bad[np.argsort(distances[bad])[::-1]]
        if not len(bad):
            print(f"        pass {pass_number}: no assignments above threshold", flush=True)
            break

        inverse_assignment = np.empty(n_samples, dtype=int)
        inverse_assignment[assignment] = np.arange(n_samples, dtype=int)
        locked = np.zeros(n_samples, dtype=bool)
        accepted = 0
        total_improvement = 0.0

        for i in bad:
            if locked[i]:
                continue

            cand_i = assignment[i]
            point_i = candidate_points[cand_i]
            current_cost_i = np.sum((point_i - desired_points[i]) ** 2)
            partners = set()

            k1 = min(partner_k, n_samples)
            _, desired_neighbors = desired_tree.query(point_i, k=k1)
            partners.update(int(j) for j in np.atleast_1d(desired_neighbors))

            _, candidate_neighbors = candidate_tree.query(desired_points[i], k=k1)
            partners.update(
                int(inverse_assignment[int(c)]) for c in np.atleast_1d(candidate_neighbors)
            )

            partners.discard(int(i))
            partners = np.asarray([j for j in partners if not locked[j]], dtype=int)
            if not len(partners):
                continue

            cand_j = assignment[partners]
            points_j = candidate_points[cand_j]
            desired_j = desired_points[partners]

            current_cost_j = np.sum((points_j - desired_j) ** 2, axis=1)
            current_pair_cost = current_cost_i + current_cost_j
            swapped_cost_i = np.sum((points_j - desired_points[i]) ** 2, axis=1)
            swapped_cost_j = np.sum((point_i - desired_j) ** 2, axis=1)
            improvement = current_pair_cost - (swapped_cost_i + swapped_cost_j)

            best_pos = int(np.argmax(improvement))
            best_improvement = float(improvement[best_pos])
            if best_improvement <= min_improvement:
                continue

            j = int(partners[best_pos])
            cj = assignment[j]
            assignment[i], assignment[j] = cj, cand_i
            inverse_assignment[cand_i] = j
            inverse_assignment[cj] = i
            locked[i] = True
            locked[j] = True
            accepted += 1
            total_improvement += best_improvement

        assigned_points = candidate_points[assignment]
        sqerr = np.sum((assigned_points - desired_points) ** 2, axis=1)
        distances = np.sqrt(sqerr)
        pass_records.append(
            {
                "pass": pass_number,
                "accepted_swaps": accepted,
                "total_squared_error": float(np.sum(sqerr)),
                "improvement_this_pass": total_improvement,
                "mean_rank_distance": float(np.mean(distances)),
                "median_rank_distance": float(np.median(distances)),
                "p90_rank_distance": float(np.quantile(distances, 0.90)),
                "p95_rank_distance": float(np.quantile(distances, 0.95)),
                "p99_rank_distance": float(np.quantile(distances, 0.99)),
                "fraction_above_threshold": float(np.mean(distances > distance_threshold)),
            }
        )
        print(
            f"        pass {pass_number}: {accepted:,} swaps | "
            f"mean={np.mean(distances):.5f} | p95={np.quantile(distances, 0.95):.5f} | "
            f">{distance_threshold:.2f}={100 * np.mean(distances > distance_threshold):.2f}%",
            flush=True,
        )
        if accepted == 0:
            print("        no further improving swaps found", flush=True)
            break

    if len(np.unique(assignment)) != n_samples or not np.array_equal(
        np.sort(assignment), np.arange(n_samples)
    ):
        raise RuntimeError("Swap refinement broke the one-to-one permutation.")

    assigned_points = candidate_points[assignment]
    rank_errors = assigned_points - desired_points
    distances = np.sqrt(np.sum(rank_errors ** 2, axis=1))
    final_total_cost = float(np.sum(distances ** 2))

    print("\n    swap refinement finished", flush=True)
    print(
        f"        total squared error reduction = {initial_total_cost - final_total_cost:.4f}",
        flush=True,
    )
    print(f"        final mean distance            = {np.mean(distances):.5f}", flush=True)
    print(f"        final median distance          = {np.median(distances):.5f}", flush=True)
    print(f"        final p95 distance             = {np.quantile(distances, 0.95):.5f}", flush=True)
    print(
        f"        final > {distance_threshold:.3f}            = "
        f"{100 * np.mean(distances > distance_threshold):.2f}%",
        flush=True,
    )

    return assignment, assigned_points, rank_errors, distances, pd.DataFrame(pass_records)


# ============================================================
# 10. CALCULATE WIDE TOTALS FROM ANY B25 WIDE DATAFRAME
# ============================================================

def calculate_wide_totals(
    df: pd.DataFrame,
    warming: float,
    years: list[int],
):
    """
    Use the standard B25 processing function on an in-memory dataframe
    written temporarily only if necessary is intentionally avoided here.

    Instead extract total responses from variable naming via the existing
    preprocessing metadata by reusing load_and_prepare_data later on the
    saved coupled file for final diagnostics.

    This placeholder is not used during matching.
    """

    raise NotImplementedError


# ============================================================
# 11. RUN COMPONENT-SPECIFIC COUPLING
# ============================================================

def couple_component_specific_panels(
    mc_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    mc_total_ext: pd.DataFrame,
    warming: float,
    eais_desired_ranks: np.ndarray,
    wais_desired_ranks: np.ndarray,
):
    """Rebuild complete B25 trajectories using local matching + swap refinement."""
    n_samples = len(mc_df)
    expected_shape = (n_samples, len(YEARS))
    if eais_desired_ranks.shape != expected_shape:
        raise ValueError("Unexpected EAIS desired-rank array shape.")
    if wais_desired_ranks.shape != expected_shape:
        raise ValueError("Unexpected WAIS desired-rank array shape.")

    columns_by_year = get_columns_by_year(meta_df=meta_df, years=YEARS)
    selected_panels = []
    metadata_frames = []
    assignment_quality_records = []
    swap_summary_frames = []

    for year_position, year in enumerate(YEARS):
        print_heading(f"MATCHING B25 PANEL: {year}")

        rank_table = build_component_rank_table(
            mc_total_ext=mc_total_ext,
            warming=warming,
            year=year,
        )
        if len(rank_table) != n_samples:
            raise ValueError(
                f"Expected {n_samples} B25 rank rows at {year}; found {len(rank_table)}."
            )

        desired_points = np.column_stack(
            [eais_desired_ranks[:, year_position], wais_desired_ranks[:, year_position]]
        )
        candidate_points = rank_table[["rank_EAIS", "rank_WAIS"]].to_numpy(dtype=float)

        # Fast initial permutation.
        (
            assignment,
            assigned_points,
            rank_errors,
            assignment_distances,
        ) = binned_local_unique_assignment(
            desired_points=desired_points,
            candidate_points=candidate_points,
            n_bins=N_RANK_BINS,
            local_k=LOCAL_K,
            seed=MATCHING_SEED + int(year),
        )

        initial_assignment = assignment.copy()
        initial_assignment_distances = assignment_distances.copy()

        # Deterministic improving-swap refinement.
        (
            assignment,
            assigned_points,
            rank_errors,
            assignment_distances,
            swap_pass_summary,
        ) = refine_assignment_with_swaps(
            desired_points=desired_points,
            candidate_points=candidate_points,
            assignment=assignment,
            distance_threshold=SWAP_REFINE_THRESHOLD,
            partner_k=SWAP_PARTNER_K,
            max_passes=SWAP_MAX_PASSES,
            min_improvement=SWAP_MIN_IMPROVEMENT,
        )
        if not swap_pass_summary.empty:
            swap_pass_summary.insert(0, "year", year)
            swap_summary_frames.append(swap_pass_summary)

        year_columns = columns_by_year[year]
        year_panel = mc_df.loc[:, ["sample_id", *year_columns]].copy()
        source_sample_ids = rank_table["sample_id"].to_numpy(dtype=int)
        selected_source_sample_ids = source_sample_ids[assignment]

        sample_to_position = pd.Series(
            np.arange(len(mc_df), dtype=int),
            index=mc_df["sample_id"].to_numpy(dtype=int),
        )
        source_positions = sample_to_position.loc[selected_source_sample_ids].to_numpy(dtype=int)

        selected_panel = year_panel.iloc[source_positions].reset_index(drop=True)
        selected_panel = selected_panel.drop(columns=["sample_id"])
        selected_panels.append(selected_panel)

        assigned_rank_table = rank_table.iloc[assignment].reset_index(drop=True)
        metadata = pd.DataFrame(
            {
                "trajectory_id": np.arange(n_samples, dtype=int),
                "year": year,
                "source_sample_id": selected_source_sample_ids,
                "desired_rank_EAIS": desired_points[:, 0],
                "desired_rank_WAIS": desired_points[:, 1],
                "assigned_rank_EAIS": assigned_points[:, 0],
                "assigned_rank_WAIS": assigned_points[:, 1],
                "rank_error_EAIS": rank_errors[:, 0],
                "rank_error_WAIS": rank_errors[:, 1],
                "rank_distance": assignment_distances,
                "initial_rank_distance": initial_assignment_distances,
                "final_rank_distance": assignment_distances,
                "rank_distance_improvement": initial_assignment_distances - assignment_distances,
                "initial_assignment_changed": initial_assignment != assignment,
                "assigned_EAIS_value": assigned_rank_table["EAIS"].to_numpy(dtype=float),
                "assigned_WAIS_value": assigned_rank_table["WAIS"].to_numpy(dtype=float),
                "assigned_GrIS_value": assigned_rank_table["GrIS"].to_numpy(dtype=float),
                "assigned_Total_AIS_value": assigned_rank_table["Total AIS"].to_numpy(dtype=float),
            }
        )
        metadata_frames.append(metadata)

        assignment_quality_records.append(
            {
                "year": year,
                "n_samples": n_samples,
                "initial_mean_rank_distance": float(np.mean(initial_assignment_distances)),
                "initial_median_rank_distance": float(np.median(initial_assignment_distances)),
                "initial_p95_rank_distance": float(np.quantile(initial_assignment_distances, 0.95)),
                "initial_percent_above_0.10": float(100 * np.mean(initial_assignment_distances > 0.10)),
                "initial_percent_above_0.25": float(100 * np.mean(initial_assignment_distances > 0.25)),
                "final_mean_rank_distance": float(np.mean(assignment_distances)),
                "final_median_rank_distance": float(np.median(assignment_distances)),
                "final_p90_rank_distance": float(np.quantile(assignment_distances, 0.90)),
                "final_p95_rank_distance": float(np.quantile(assignment_distances, 0.95)),
                "final_p99_rank_distance": float(np.quantile(assignment_distances, 0.99)),
                "final_max_rank_distance": float(np.max(assignment_distances)),
                "final_percent_above_0.10": float(100 * np.mean(assignment_distances > 0.10)),
                "final_percent_above_0.25": float(100 * np.mean(assignment_distances > 0.25)),
                "percent_assignments_changed": float(100 * np.mean(initial_assignment != assignment)),
                "mean_abs_rank_error_EAIS": float(np.mean(np.abs(rank_errors[:, 0]))),
                "mean_abs_rank_error_WAIS": float(np.mean(np.abs(rank_errors[:, 1]))),
            }
        )

    coupled_df = pd.concat(selected_panels, axis=1)
    original_scientific_order = [c for c in mc_df.columns if c != "sample_id"]
    coupled_df = coupled_df.loc[:, original_scientific_order]
    coupled_df.insert(0, "sample_id", np.arange(n_samples, dtype=int))

    coupling_metadata = pd.concat(metadata_frames, ignore_index=True)
    assignment_quality = pd.DataFrame(assignment_quality_records)
    swap_summary = (
        pd.concat(swap_summary_frames, ignore_index=True)
        if swap_summary_frames
        else pd.DataFrame()
    )

    return coupled_df, coupling_metadata, assignment_quality, swap_summary


# ============================================================
# 12. EXACT MARGINAL-PRESERVATION CHECK
# ============================================================

def check_exact_marginal_preservation(
    original_df: pd.DataFrame,
    coupled_df: pd.DataFrame,
    meta_df: pd.DataFrame,
):
    records = []

    for _, row in meta_df.iterrows():

        variable = row[
            "variable"
        ]

        original_sorted = np.sort(
            original_df[
                variable
            ].to_numpy(
                dtype=float
            )
        )

        coupled_sorted = np.sort(
            coupled_df[
                variable
            ].to_numpy(
                dtype=float
            )
        )

        difference = np.abs(
            original_sorted
            - coupled_sorted
        )

        records.append(
            {
                "variable": variable,
                "year": int(
                    row[
                        "year"
                    ]
                ),
                "ice_sheet": row[
                    "ice_sheet"
                ],
                "process": row[
                    "process"
                ],
                "maximum_absolute_difference": float(
                    np.max(
                        difference
                    )
                ),
                "exact_match": bool(
                    np.array_equal(
                        original_sorted,
                        coupled_sorted,
                    )
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# 13. LOAD COUPLED FILE THROUGH STANDARD PROCESSING
# ============================================================

def calculate_final_diagnostics(
    original_file: Path,
    coupled_file: Path,
    target_eais: np.ndarray,
    target_wais: np.ndarray,
):
    """
    Reload original and coupled samples through standard B25 processing
    so all Total AIS / GrIS totals are calculated consistently.
    """

    original_data = load_and_prepare_data(
        file_path=str(
            original_file
        )
    )

    coupled_data = load_and_prepare_data(
        file_path=str(
            coupled_file
        )
    )

    original_total = original_data[
        "mc_total_ext"
    ].copy()

    coupled_total = coupled_data[
        "mc_total_ext"
    ].copy()

    temporal_records = []

    target_by_component = {
        "EAIS": target_eais,
        "WAIS": target_wais,
    }

    for component in [
        "EAIS",
        "WAIS",
        "GrIS",
        "Total AIS",
    ]:

        subset = coupled_total.loc[
            np.isclose(
                coupled_total[
                    "warming"
                ],
                WARMING,
            )
            & (
                coupled_total[
                    "ice_sheet"
                ]
                == component
            )
            & coupled_total[
                "year"
            ].isin(
                YEARS
            )
        ][
            [
                "sample_id",
                "year",
                "value",
            ]
        ].copy()

        wide = (
            subset
            .pivot(
                index="sample_id",
                columns="year",
                values="value",
            )
            .reindex(
                columns=YEARS
            )
        )

        achieved_spearman = wide.corr(
            method="spearman"
        )

        achieved_pearson = wide.corr(
            method="pearson"
        )

        for i, year1 in enumerate(
            YEARS
        ):
            for j, year2 in enumerate(
                YEARS
            ):
                if j <= i:
                    continue

                record = {
                    "component": component,
                    "year1": year1,
                    "year2": year2,
                    "year_pair": (
                        f"{year1}-{year2}"
                    ),
                    "achieved_spearman": float(
                        achieved_spearman.loc[
                            year1,
                            year2,
                        ]
                    ),
                    "achieved_pearson": float(
                        achieved_pearson.loc[
                            year1,
                            year2,
                        ]
                    ),
                }

                if component in target_by_component:

                    target_matrix = target_by_component[
                        component
                    ]

                    rho = float(
                        target_matrix[
                            i,
                            j,
                        ]
                    )

                    record[
                        "target_latent_gaussian_rho"
                    ] = rho

                    record[
                        "target_gaussian_implied_spearman"
                    ] = (
                        gaussian_implied_spearman(
                            rho
                        )
                    )

                    record[
                        "spearman_minus_target_implied"
                    ] = (
                        record[
                            "achieved_spearman"
                        ]
                        - record[
                            "target_gaussian_implied_spearman"
                        ]
                    )

                else:

                    record[
                        "target_latent_gaussian_rho"
                    ] = np.nan

                    record[
                        "target_gaussian_implied_spearman"
                    ] = np.nan

                    record[
                        "spearman_minus_target_implied"
                    ] = np.nan

                temporal_records.append(
                    record
                )

    temporal_df = pd.DataFrame(
        temporal_records
    )

    # Within-year cross-component dependence.
    within_records = []

    for year in YEARS:

        for dataset_name, total_df in [
            (
                "original",
                original_total,
            ),
            (
                "coupled",
                coupled_total,
            ),
        ]:

            subset = total_df.loc[
                np.isclose(
                    total_df[
                        "warming"
                    ],
                    WARMING,
                )
                & (
                    total_df[
                        "year"
                    ]
                    == year
                )
                & total_df[
                    "ice_sheet"
                ].isin(
                    [
                        "EAIS",
                        "WAIS",
                        "GrIS",
                        "Total AIS",
                    ]
                )
            ][
                [
                    "sample_id",
                    "ice_sheet",
                    "value",
                ]
            ]

            wide = (
                subset
                .pivot(
                    index="sample_id",
                    columns="ice_sheet",
                    values="value",
                )
            )

            spear = wide.corr(
                method="spearman"
            )

            for c1, c2 in [
                (
                    "EAIS",
                    "WAIS",
                ),
                (
                    "EAIS",
                    "GrIS",
                ),
                (
                    "WAIS",
                    "GrIS",
                ),
                (
                    "EAIS",
                    "Total AIS",
                ),
                (
                    "WAIS",
                    "Total AIS",
                ),
            ]:

                within_records.append(
                    {
                        "dataset": dataset_name,
                        "year": year,
                        "component_1": c1,
                        "component_2": c2,
                        "spearman": float(
                            spear.loc[
                                c1,
                                c2,
                            ]
                        ),
                    }
                )

    within_df = pd.DataFrame(
        within_records
    )

    original_wide = (
        within_df.loc[
            within_df[
                "dataset"
            ]
            == "original"
        ]
        .rename(
            columns={
                "spearman": "original_spearman"
            }
        )
        .drop(
            columns=[
                "dataset"
            ]
        )
    )

    coupled_wide = (
        within_df.loc[
            within_df[
                "dataset"
            ]
            == "coupled"
        ]
        .rename(
            columns={
                "spearman": "coupled_spearman"
            }
        )
        .drop(
            columns=[
                "dataset"
            ]
        )
    )

    within_compare = original_wide.merge(
        coupled_wide,
        on=[
            "year",
            "component_1",
            "component_2",
        ],
        how="inner",
    )

    within_compare[
        "difference"
    ] = (
        within_compare[
            "coupled_spearman"
        ]
        - within_compare[
            "original_spearman"
        ]
    )

    return (
        temporal_df,
        within_compare,
    )


# ============================================================
# 14. TARGET COPULA TABLE
# ============================================================

def make_target_copula_table(
    component: str,
    matrix: np.ndarray,
):
    return pd.DataFrame(
        matrix,
        index=YEARS,
        columns=YEARS,
    ).rename_axis(
        index="year"
    )


# ============================================================
# 15. FIGURES
# ============================================================

def plot_assignment_scatter(
    coupling_metadata: pd.DataFrame,
    year: int,
):
    rng = np.random.default_rng(
        SCATTER_SEED
    )

    subset = (
        coupling_metadata.loc[
            coupling_metadata[
                "year"
            ]
            == year
        ]
        .copy()
        .reset_index(drop=True)
    )

    n = len(
        subset
    )

    if n > N_SCATTER:

        idx = rng.choice(
            n,
            size=N_SCATTER,
            replace=False,
        )

        subset = subset.iloc[
            idx
        ]

    fig, ax = plt.subplots(
        figsize=(
            7,
            7,
        )
    )

    ax.scatter(
        subset[
            "desired_rank_EAIS"
        ],
        subset[
            "desired_rank_WAIS"
        ],
        s=5,
        alpha=0.25,
        label="Desired rank vectors",
    )

    ax.scatter(
        subset[
            "assigned_rank_EAIS"
        ],
        subset[
            "assigned_rank_WAIS"
        ],
        s=5,
        alpha=0.25,
        label="Assigned B25 row ranks",
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_xlabel(
        "EAIS percentile rank"
    )

    ax.set_ylabel(
        "WAIS percentile rank"
    )

    ax.set_title(
        f"Desired vs assigned B25 rank vectors: {year}"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / f"assignment_rank_scatter_{year}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_assignment_distances(
    coupling_metadata: pd.DataFrame,
):
    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    data = [
        coupling_metadata.loc[
            coupling_metadata[
                "year"
            ]
            == year,
            "rank_distance",
        ].to_numpy(
            dtype=float
        )
        for year in YEARS
    ]

    ax.boxplot(
        data,
        tick_labels=[
            str(
                year
            )
            for year in YEARS
        ],
        showfliers=False,
    )

    ax.set_xlabel(
        "Projection year"
    )

    ax.set_ylabel(
        "EAIS–WAIS rank-vector distance"
    )

    ax.set_title(
        "Quality of intact-row rank matching"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "assignment_distance_by_year.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_temporal_diagnostics(
    temporal_df: pd.DataFrame,
):
    components = [
        "EAIS",
        "WAIS",
        "Total AIS",
        "GrIS",
    ]

    year_pairs = [
        "2100-2200",
        "2100-2300",
        "2200-2300",
    ]

    x = np.arange(
        len(
            year_pairs
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    width = 0.18

    for j, component in enumerate(
        components
    ):

        group = (
            temporal_df.loc[
                temporal_df[
                    "component"
                ]
                == component
            ]
            .set_index(
                "year_pair"
            )
            .reindex(
                year_pairs
            )
        )

        ax.bar(
            x
            + (
                j
                - 1.5
            )
            * width,
            group[
                "achieved_spearman"
            ],
            width=width,
            label=component,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "2100–2200",
            "2100–2300",
            "2200–2300",
        ]
    )

    ax.set_ylim(
        -0.1,
        1,
    )

    ax.set_ylabel(
        "Achieved Spearman correlation"
    )

    ax.set_title(
        "Temporal rank persistence after component-specific reshuffling"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "temporal_correlation_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print_heading(
        "SCRIPT B: COMPONENT-SPECIFIC RANK RESHUFFLING"
    )

    print(
        f"Experiment: B25 {WARMING}°C / "
        f"DP21 {DP21_SCENARIO_LABEL}"
    )

    print(
        f"\nOriginal B25 file:\n{data_file}"
    )

    # --------------------------------------------------------
    # A. Load original B25 data
    # --------------------------------------------------------

    original_data = load_and_prepare_data(
        file_path=str(
            data_file
        )
    )

    mc_df = original_data[
        "mc_df"
    ].copy()

    meta_df = original_data[
        "meta_df"
    ].copy()

    mc_total_ext = original_data[
        "mc_total_ext"
    ].copy()

    n_samples = len(
        mc_df
    )

    print(
        f"\nB25 Monte Carlo samples: "
        f"{n_samples:,}"
    )

    # --------------------------------------------------------
    # B. Load full component-specific DP21 copulas
    # --------------------------------------------------------

    target_eais = load_component_copula(
        correlation_file=(
            SCRIPT_A_CORRELATION_FILE
        ),
        scenario=DP21_SCENARIO,
        component="EAIS",
        years=YEARS,
    )

    target_wais = load_component_copula(
        correlation_file=(
            SCRIPT_A_CORRELATION_FILE
        ),
        scenario=DP21_SCENARIO,
        component="WAIS",
        years=YEARS,
    )

    print_heading(
        "TARGET EAIS LATENT COPULA"
    )

    print(
        pd.DataFrame(
            target_eais,
            index=YEARS,
            columns=YEARS,
        ).round(
            4
        )
    )

    print_heading(
        "TARGET WAIS LATENT COPULA"
    )

    print(
        pd.DataFrame(
            target_wais,
            index=YEARS,
            columns=YEARS,
        ).round(
            4
        )
    )

    # --------------------------------------------------------
    # C. Simulate separate full 3-year rank histories
    # --------------------------------------------------------

    (
        latent_eais,
        desired_eais,
        rank_positions_eais,
    ) = simulate_component_rank_histories(
        correlation_matrix=target_eais,
        n_samples=n_samples,
        seed=RANDOM_SEED,
    )

    (
        latent_wais,
        desired_wais,
        rank_positions_wais,
    ) = simulate_component_rank_histories(
        correlation_matrix=target_wais,
        n_samples=n_samples,
        seed=RANDOM_SEED + 1,
    )

    # Save latent histories for complete reproducibility.
    latent_history_df = pd.DataFrame(
        {
            "trajectory_id": np.arange(
                n_samples,
                dtype=int,
            ),
            "EAIS_z_2100": latent_eais[
                :,
                0,
            ],
            "EAIS_z_2200": latent_eais[
                :,
                1,
            ],
            "EAIS_z_2300": latent_eais[
                :,
                2,
            ],
            "WAIS_z_2100": latent_wais[
                :,
                0,
            ],
            "WAIS_z_2200": latent_wais[
                :,
                1,
            ],
            "WAIS_z_2300": latent_wais[
                :,
                2,
            ],
            "EAIS_rank_2100": desired_eais[
                :,
                0,
            ],
            "EAIS_rank_2200": desired_eais[
                :,
                1,
            ],
            "EAIS_rank_2300": desired_eais[
                :,
                2,
            ],
            "WAIS_rank_2100": desired_wais[
                :,
                0,
            ],
            "WAIS_rank_2200": desired_wais[
                :,
                1,
            ],
            "WAIS_rank_2300": desired_wais[
                :,
                2,
            ],
        }
    )

    latent_history_df.to_csv(
        OUTPUT_DIR
        / (
            f"{EXPERIMENT_NAME}"
            "_latent_rank_histories.csv"
        ),
        index=False,
    )

    # --------------------------------------------------------
    # D. Match intact B25 yearly panels
    # --------------------------------------------------------

    (
        coupled_df,
        coupling_metadata,
        assignment_quality,
        swap_summary,
    ) = couple_component_specific_panels(
        mc_df=mc_df,
        meta_df=meta_df,
        mc_total_ext=mc_total_ext,
        warming=WARMING,
        eais_desired_ranks=desired_eais,
        wais_desired_ranks=desired_wais,
    )

    # --------------------------------------------------------
    # E. Exact marginal check BEFORE writing output
    # --------------------------------------------------------

    marginal_diagnostics = (
        check_exact_marginal_preservation(
            original_df=mc_df,
            coupled_df=coupled_df,
            meta_df=meta_df,
        )
    )

    if not marginal_diagnostics[
        "exact_match"
    ].all():

        failed = marginal_diagnostics.loc[
            ~marginal_diagnostics[
                "exact_match"
            ]
        ]

        raise RuntimeError(
            "Exact B25 marginal preservation failed.\n"
            + failed.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # F. Save coupled wide file
    # --------------------------------------------------------

    coupled_file = (
        OUTPUT_DIR
        / (
            f"{EXPERIMENT_NAME}"
            "_coupled_samples.csv"
        )
    )

    coupled_df.to_csv(
        coupled_file,
        index=False,
    )

    coupling_metadata.to_csv(
        OUTPUT_DIR
        / (
            f"{EXPERIMENT_NAME}"
            "_coupling_metadata.csv"
        ),
        index=False,
    )

    assignment_quality.to_csv(
        OUTPUT_DIR
        / (
            f"{EXPERIMENT_NAME}"
            "_assignment_quality.csv"
        ),
        index=False,
    )

    if not swap_summary.empty:
        swap_summary.to_csv(
            OUTPUT_DIR
            / (
                f"{EXPERIMENT_NAME}"
                "_swap_refinement_summary.csv"
            ),
            index=False,
        )

    marginal_diagnostics.to_csv(
        OUTPUT_DIR
        / (
            f"{EXPERIMENT_NAME}"
            "_marginal_preservation.csv"
        ),
        index=False,
    )

    # Target matrices.
    eais_target_df = make_target_copula_table(
        "EAIS",
        target_eais,
    )

    wais_target_df = make_target_copula_table(
        "WAIS",
        target_wais,
    )

    target_long = pd.concat(
        [
            (
                eais_target_df
                .reset_index()
                .melt(
                    id_vars="year",
                    var_name="target_year",
                    value_name="latent_rho",
                )
                .assign(
                    component="EAIS"
                )
            ),
            (
                wais_target_df
                .reset_index()
                .melt(
                    id_vars="year",
                    var_name="target_year",
                    value_name="latent_rho",
                )
                .assign(
                    component="WAIS"
                )
            ),
        ],
        ignore_index=True,
    )

    target_long.to_csv(
        OUTPUT_DIR
        / (
            f"{EXPERIMENT_NAME}"
            "_target_copulas.csv"
        ),
        index=False,
    )

    # --------------------------------------------------------
    # G. Reload coupled output and calculate final diagnostics
    # --------------------------------------------------------

    (
        temporal_diagnostics,
        within_year_diagnostics,
    ) = calculate_final_diagnostics(
        original_file=Path(
            data_file
        ),
        coupled_file=coupled_file,
        target_eais=target_eais,
        target_wais=target_wais,
    )

    temporal_diagnostics.to_csv(
        OUTPUT_DIR
        / (
            f"{EXPERIMENT_NAME}"
            "_temporal_correlation_diagnostics.csv"
        ),
        index=False,
    )

    within_year_diagnostics.to_csv(
        OUTPUT_DIR
        / (
            f"{EXPERIMENT_NAME}"
            "_within_year_dependence_diagnostics.csv"
        ),
        index=False,
    )

    # --------------------------------------------------------
    # H. Figures
    # --------------------------------------------------------

    for year in YEARS:

        plot_assignment_scatter(
            coupling_metadata=(
                coupling_metadata
            ),
            year=year,
        )

    plot_assignment_distances(
        coupling_metadata
    )

    plot_temporal_diagnostics(
        temporal_diagnostics
    )

    # --------------------------------------------------------
    # I. Console diagnostics
    # --------------------------------------------------------

    print_heading(
        "ASSIGNMENT QUALITY"
    )

    print(
        assignment_quality
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print_heading(
        "TEMPORAL CORRELATION DIAGNOSTICS"
    )

    print(
        temporal_diagnostics[
            [
                "component",
                "year_pair",
                "target_latent_gaussian_rho",
                "target_gaussian_implied_spearman",
                "achieved_spearman",
                "achieved_pearson",
                "spearman_minus_target_implied",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print_heading(
        "WITHIN-YEAR DEPENDENCE PRESERVATION"
    )

    print(
        within_year_diagnostics
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    print_heading(
        "MARGINAL PRESERVATION"
    )

    print(
        "Variables checked:",
        len(
            marginal_diagnostics
        ),
    )

    print(
        "Failed exact checks:",
        int(
            (
                ~marginal_diagnostics[
                    "exact_match"
                ]
            ).sum()
        ),
    )

    print(
        "Maximum absolute difference:",
        marginal_diagnostics[
            "maximum_absolute_difference"
        ].max(),
    )

    print_heading(
        "IMPORTANT GrIS NOTE"
    )

    print(
        "GrIS is carried with the intact B25 row but is NOT "
        "temporally targeted in this prototype because DP21 "
        "does not provide Greenland trajectories."
    )

    print_heading(
        "FINISHED"
    )

    print(
        f"Outputs saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
