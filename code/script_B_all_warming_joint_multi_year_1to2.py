#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCRIPT B: Component-specific copula + intact-row rank reshuffling prototype
===========================================================================

Purpose
-------
Run the component-specific joint multi-year coupling for ALL THREE B25 warming libraries:

    B25 2.0°C  <->  DP21 RCP2.6
    B25 3.5°C  <->  DP21 RCP4.5
    B25 5.0°C  <->  DP21 RCP8.5

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
empirical ranks within the CURRENT warming library:

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
Under separate experiment folders within:

    output/component_specific_temporal_coupling/

one folder is written for each 2.0°C/RCP2.6, 3.5°C/RCP4.5,
and 5.0°C/RCP8.5 coupling.

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
This script uses a fast local/bin one-to-one matcher plus rank-distance swap
refinement as a warm start, followed by a JOINT multi-year permutation optimizer.
The joint optimizer directly penalizes error in the EAIS and WAIS temporal
Spearman targets while retaining the full copula-generated rank histories as a
secondary objective. Every operation is an intact-row swap within one year, so
exact B25 marginals and within-year panel dependence are retained by construction.
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

# Each B25 warming library is coupled independently using the
# component-specific DP21 temporal copulas from the corresponding
# forcing experiment.
EXPERIMENTS = [
    {
        "warming": 2.0,
        "warming_label": "2.0C",
        "dp21_scenario": "rcp26",
        "dp21_scenario_label": "RCP2.6",
        "seed_offset": 0,
    },
    {
        "warming": 3.5,
        "warming_label": "3.5C",
        "dp21_scenario": "rcp45",
        "dp21_scenario_label": "RCP4.5",
        "seed_offset": 1000,
    },
    {
        "warming": 5.0,
        "warming_label": "5.0C",
        "dp21_scenario": "rcp85",
        "dp21_scenario_label": "RCP8.5",
        "seed_offset": 2000,
    },
]

# These globals are set by configure_experiment() before each run.
# Functions elsewhere in the script read them at runtime.
WARMING = None
WARMING_LABEL = None
DP21_SCENARIO = None
DP21_SCENARIO_LABEL = None
EXPERIMENT_NAME = None
OUTPUT_DIR = None
FIGURE_DIR = None
CURRENT_RANDOM_SEED = None

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

# Joint multi-year optimization settings.
#
# The objective has two dimensionless pieces:
#   1. mean squared error between achieved and target EAIS/WAIS
#      temporal Spearman correlations;
#   2. mean squared EAIS/WAIS rank-vector mismatch relative to the
#      copula-generated desired rank histories.
#
# Equal weights are a transparent starting point. They are tuning
# parameters and should be sensitivity-tested before a production choice.
JOINT_TEMPORAL_WEIGHT = 1.0
JOINT_RANK_WEIGHT = 2.0

# Maximum number of joint passes over 2100, 2200, and 2300.
JOINT_MAX_PASSES = 8

# Per year/pass, directly examine the worst rank-matched trajectories
# plus a random subset so the optimizer can also improve temporal
# dependence through swaps involving already-good local matches.
JOINT_N_WORST = 6000
JOINT_N_RANDOM = 2000

# Promising partner search sizes.
JOINT_PARTNER_K = 48
JOINT_RANDOM_PARTNERS = 16

# Minimum reduction in the combined objective required to accept a swap.
JOINT_MIN_IMPROVEMENT = 1e-12

JOINT_SEED = 97531

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

# Experiment-specific output paths are configured immediately
# before each run so the three warming libraries never overwrite
# one another.

COMBINED_SUMMARY_DIR = (
    OUTPUT_ROOT
    / "all_warming_joint_multi_year_summary"
)

COMBINED_SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def configure_experiment(
    experiment: dict,
):
    """
    Set the current warming/scenario globals and create the
    experiment-specific output folders.
    """

    global WARMING
    global WARMING_LABEL
    global DP21_SCENARIO
    global DP21_SCENARIO_LABEL
    global EXPERIMENT_NAME
    global OUTPUT_DIR
    global FIGURE_DIR
    global CURRENT_RANDOM_SEED

    WARMING = float(
        experiment[
            "warming"
        ]
    )

    WARMING_LABEL = str(
        experiment[
            "warming_label"
        ]
    )

    DP21_SCENARIO = str(
        experiment[
            "dp21_scenario"
        ]
    )

    DP21_SCENARIO_LABEL = str(
        experiment[
            "dp21_scenario_label"
        ]
    )

    CURRENT_RANDOM_SEED = (
        RANDOM_SEED
        + int(
            experiment.get(
                "seed_offset",
                0,
            )
        )
    )

    EXPERIMENT_NAME = (
        f"b25_{WARMING_LABEL}_dp21_{DP21_SCENARIO}"
        "_component_specific_joint_multi_year"
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
# 10. JOINT MULTI-YEAR PERMUTATION OPTIMIZATION
# ============================================================

def target_spearman_from_latent(
    latent_matrix: np.ndarray,
):
    """
    Convert a Gaussian-copula latent correlation matrix to its
    Gaussian-implied Spearman correlation matrix.
    """
    latent_matrix = np.asarray(
        latent_matrix,
        dtype=float,
    )

    return (
        6.0
        / np.pi
        * np.arcsin(
            latent_matrix
            / 2.0
        )
    )


def _build_assigned_rank_matrices(
    assignments: dict[int, np.ndarray],
    candidate_points_by_year: dict[int, np.ndarray],
):
    """
    Convert year-specific permutations into N x 3 matrices of
    assigned EAIS and WAIS percentile ranks.
    """

    eais = np.column_stack(
        [
            candidate_points_by_year[
                year
            ][
                assignments[
                    year
                ],
                0,
            ]
            for year in YEARS
        ]
    )

    wais = np.column_stack(
        [
            candidate_points_by_year[
                year
            ][
                assignments[
                    year
                ],
                1,
            ]
            for year in YEARS
        ]
    )

    return (
        eais,
        wais,
    )


def _joint_objective_summary(
    assignments: dict[int, np.ndarray],
    candidate_points_by_year: dict[int, np.ndarray],
    desired_eais_ranks: np.ndarray,
    desired_wais_ranks: np.ndarray,
    target_eais_spearman: np.ndarray,
    target_wais_spearman: np.ndarray,
    temporal_weight: float,
    rank_weight: float,
):
    """
    Recompute the complete joint objective from scratch.

    This is used for diagnostics and at the end of each optimization
    pass to prevent accumulation of numerical bookkeeping error.
    """

    assigned_eais, assigned_wais = (
        _build_assigned_rank_matrices(
            assignments=assignments,
            candidate_points_by_year=(
                candidate_points_by_year
            ),
        )
    )

    achieved_eais = np.corrcoef(
        assigned_eais,
        rowvar=False,
    )

    achieved_wais = np.corrcoef(
        assigned_wais,
        rowvar=False,
    )

    upper = np.triu_indices(
        len(
            YEARS
        ),
        k=1,
    )

    temporal_errors = np.concatenate(
        [
            (
                achieved_eais[
                    upper
                ]
                - target_eais_spearman[
                    upper
                ]
            ),
            (
                achieved_wais[
                    upper
                ]
                - target_wais_spearman[
                    upper
                ]
            ),
        ]
    )

    temporal_mse = float(
        np.mean(
            temporal_errors ** 2
        )
    )

    rank_sse = float(
        np.sum(
            (
                assigned_eais
                - desired_eais_ranks
            ) ** 2
        )
        + np.sum(
            (
                assigned_wais
                - desired_wais_ranks
            ) ** 2
        )
    )

    rank_mse = (
        rank_sse
        / (
            2.0
            * assigned_eais.size
        )
    )

    objective = (
        temporal_weight
        * temporal_mse
        + rank_weight
        * rank_mse
    )

    return {
        "objective": float(
            objective
        ),
        "temporal_mse": float(
            temporal_mse
        ),
        "rank_mse": float(
            rank_mse
        ),
        "assigned_eais": assigned_eais,
        "assigned_wais": assigned_wais,
        "achieved_eais": achieved_eais,
        "achieved_wais": achieved_wais,
    }


def _joint_summary_record(
    stage: str,
    pass_number: int,
    stats: dict,
    accepted_swaps: int,
):
    """
    Flatten the joint objective and achieved temporal correlations
    into one diagnostic record.
    """

    e = stats[
        "achieved_eais"
    ]

    w = stats[
        "achieved_wais"
    ]

    return {
        "stage": stage,
        "pass": pass_number,
        "accepted_swaps": int(
            accepted_swaps
        ),
        "objective": stats[
            "objective"
        ],
        "temporal_mse": stats[
            "temporal_mse"
        ],
        "rank_mse": stats[
            "rank_mse"
        ],
        "EAIS_spearman_2100_2200": e[
            0,
            1,
        ],
        "EAIS_spearman_2100_2300": e[
            0,
            2,
        ],
        "EAIS_spearman_2200_2300": e[
            1,
            2,
        ],
        "WAIS_spearman_2100_2200": w[
            0,
            1,
        ],
        "WAIS_spearman_2100_2300": w[
            0,
            2,
        ],
        "WAIS_spearman_2200_2300": w[
            1,
            2,
        ],
    }


def optimize_joint_multi_year_permutations(
    assignments: dict[int, np.ndarray],
    candidate_points_by_year: dict[int, np.ndarray],
    desired_eais_ranks: np.ndarray,
    desired_wais_ranks: np.ndarray,
    target_eais_latent: np.ndarray,
    target_wais_latent: np.ndarray,
    temporal_weight: float = JOINT_TEMPORAL_WEIGHT,
    rank_weight: float = JOINT_RANK_WEIGHT,
    max_passes: int = JOINT_MAX_PASSES,
    n_worst: int = JOINT_N_WORST,
    n_random: int = JOINT_N_RANDOM,
    partner_k: int = JOINT_PARTNER_K,
    random_partners: int = JOINT_RANDOM_PARTNERS,
    min_improvement: float = JOINT_MIN_IMPROVEMENT,
    seed: int = JOINT_SEED,
):
    """
    Jointly optimize the 2100, 2200, and 2300 B25 permutations.

    HARD CONSTRAINT
    ---------------
    At each projection year the assignment always remains a complete
    permutation of the original B25 rows. Therefore:

        - every B25 row is used exactly once per year;
        - all B25 marginals remain exactly unchanged;
        - all within-year cross-component/process dependence remains exact;
        - only cross-time row connections are altered.

    OBJECTIVE
    ---------
    The combined objective is:

        J =
            temporal_weight * temporal_MSE
            + rank_weight * rank_MSE

    where temporal_MSE is the mean squared discrepancy between the
    achieved and DP21-target EAIS/WAIS temporal Spearman correlations,
    and rank_MSE measures departure from the component-specific
    copula-generated desired EAIS/WAIS rank histories.

    METHOD
    ------
    Starting from the fast local + rank-swap warm-start permutations,
    the optimizer considers pairwise swaps WITHIN one projection year.
    Each proposed swap is evaluated against the full three-year
    objective. A swap is accepted only if it improves the joint
    objective.

    Because a swap only exchanges two intact rows within the same year,
    the within-year B25 cloud is invariant by construction.
    """

    rng = np.random.default_rng(
        seed
    )

    n_samples = len(
        next(
            iter(
                assignments.values()
            )
        )
    )

    n_years = len(
        YEARS
    )

    year_to_position = {
        year: j
        for j, year in enumerate(
            YEARS
        )
    }

    # Defensive copies.
    assignments = {
        year: np.asarray(
            assignment,
            dtype=int,
        ).copy()
        for year, assignment in assignments.items()
    }

    # Verify every warm-start assignment is a full permutation.
    for year in YEARS:

        a = assignments[
            year
        ]

        if len(
            np.unique(
                a
            )
        ) != n_samples:

            raise ValueError(
                f"{year} warm-start assignment is not one-to-one."
            )

        if not np.array_equal(
            np.sort(
                a
            ),
            np.arange(
                n_samples
            ),
        ):

            raise ValueError(
                f"{year} warm-start assignment is not a "
                "complete permutation."
            )

    target_eais_spearman = (
        target_spearman_from_latent(
            target_eais_latent
        )
    )

    target_wais_spearman = (
        target_spearman_from_latent(
            target_wais_latent
        )
    )

    print_heading(
        "JOINT MULTI-YEAR PERMUTATION OPTIMIZATION"
    )

    print(
        "Hard constraint: each yearly B25 panel remains an exact "
        "permutation of the original rows.",
        flush=True,
    )

    print(
        f"Objective weights: temporal={temporal_weight}, "
        f"rank={rank_weight}",
        flush=True,
    )

    # Search structures do not change because desired points and
    # candidate clouds are fixed; only the holder of each candidate
    # row changes.
    desired_points_by_year = {
        year: np.column_stack(
            [
                desired_eais_ranks[
                    :,
                    year_to_position[
                        year
                    ],
                ],
                desired_wais_ranks[
                    :,
                    year_to_position[
                        year
                    ],
                ],
            ]
        )
        for year in YEARS
    }

    desired_trees = {
        year: cKDTree(
            desired_points_by_year[
                year
            ]
        )
        for year in YEARS
    }

    candidate_trees = {
        year: cKDTree(
            candidate_points_by_year[
                year
            ]
        )
        for year in YEARS
    }

    summary_records = []

    stats = _joint_objective_summary(
        assignments=assignments,
        candidate_points_by_year=(
            candidate_points_by_year
        ),
        desired_eais_ranks=(
            desired_eais_ranks
        ),
        desired_wais_ranks=(
            desired_wais_ranks
        ),
        target_eais_spearman=(
            target_eais_spearman
        ),
        target_wais_spearman=(
            target_wais_spearman
        ),
        temporal_weight=temporal_weight,
        rank_weight=rank_weight,
    )

    summary_records.append(
        _joint_summary_record(
            stage="warm_start",
            pass_number=0,
            stats=stats,
            accepted_swaps=0,
        )
    )

    print(
        "\nWarm-start achieved temporal Spearman correlations:",
        flush=True,
    )

    print(
        f"    EAIS: "
        f"{stats['achieved_eais'][0,1]:.4f}, "
        f"{stats['achieved_eais'][0,2]:.4f}, "
        f"{stats['achieved_eais'][1,2]:.4f}",
        flush=True,
    )

    print(
        f"    WAIS: "
        f"{stats['achieved_wais'][0,1]:.4f}, "
        f"{stats['achieved_wais'][0,2]:.4f}, "
        f"{stats['achieved_wais'][1,2]:.4f}",
        flush=True,
    )

    print(
        f"    objective={stats['objective']:.6f} | "
        f"temporal MSE={stats['temporal_mse']:.6f} | "
        f"rank MSE={stats['rank_mse']:.6f}",
        flush=True,
    )

    # --------------------------------------------------------
    # Joint passes
    # --------------------------------------------------------

    for pass_number in range(
        1,
        max_passes + 1,
    ):

        # Recompute exact state at the start of every pass.
        stats = _joint_objective_summary(
            assignments=assignments,
            candidate_points_by_year=(
                candidate_points_by_year
            ),
            desired_eais_ranks=(
                desired_eais_ranks
            ),
            desired_wais_ranks=(
                desired_wais_ranks
            ),
            target_eais_spearman=(
                target_eais_spearman
            ),
            target_wais_spearman=(
                target_wais_spearman
            ),
            temporal_weight=temporal_weight,
            rank_weight=rank_weight,
        )

        assigned_eais = stats[
            "assigned_eais"
        ].copy()

        assigned_wais = stats[
            "assigned_wais"
        ].copy()

        # Means, centered sum-of-squares, and denominators are fixed
        # under within-year permutations.
        e_means = np.mean(
            assigned_eais,
            axis=0,
        )

        w_means = np.mean(
            assigned_wais,
            axis=0,
        )

        e_centered = (
            assigned_eais
            - e_means
        )

        w_centered = (
            assigned_wais
            - w_means
        )

        e_ss = np.sum(
            e_centered ** 2,
            axis=0,
        )

        w_ss = np.sum(
            w_centered ** 2,
            axis=0,
        )

        e_denom = np.sqrt(
            e_ss[
                :,
                None,
            ]
            * e_ss[
                None,
                :,
            ]
        )

        w_denom = np.sqrt(
            w_ss[
                :,
                None,
            ]
            * w_ss[
                None,
                :,
            ]
        )

        e_dots = (
            e_centered.T
            @ e_centered
        )

        w_dots = (
            w_centered.T
            @ w_centered
        )

        current_e_corr = (
            e_dots
            / e_denom
        )

        current_w_corr = (
            w_dots
            / w_denom
        )

        # Current objective bookkeeping.
        upper = np.triu_indices(
            n_years,
            k=1,
        )

        temporal_sse = float(
            np.sum(
                (
                    current_e_corr[
                        upper
                    ]
                    - target_eais_spearman[
                        upper
                    ]
                ) ** 2
            )
            + np.sum(
                (
                    current_w_corr[
                        upper
                    ]
                    - target_wais_spearman[
                        upper
                    ]
                ) ** 2
            )
        )

        current_rank_sse = float(
            np.sum(
                (
                    assigned_eais
                    - desired_eais_ranks
                ) ** 2
            )
            + np.sum(
                (
                    assigned_wais
                    - desired_wais_ranks
                ) ** 2
            )
        )

        accepted_this_pass = 0

        # ----------------------------------------------------
        # Optimize each yearly permutation while evaluating the
        # impact on ALL temporal pairs involving that year.
        # ----------------------------------------------------

        for year in YEARS:

            t = year_to_position[
                year
            ]

            desired_points = (
                desired_points_by_year[
                    year
                ]
            )

            candidate_points = (
                candidate_points_by_year[
                    year
                ]
            )

            assignment = assignments[
                year
            ]

            inverse_assignment = np.empty(
                n_samples,
                dtype=int,
            )

            inverse_assignment[
                assignment
            ] = np.arange(
                n_samples,
                dtype=int,
            )

            # Current row-level rank distance. This is used only
            # to identify a useful targeted subset; the acceptance
            # criterion is always the FULL JOINT objective.
            current_points = (
                candidate_points[
                    assignment
                ]
            )

            row_distance = np.sqrt(
                np.sum(
                    (
                        current_points
                        - desired_points
                    ) ** 2,
                    axis=1,
                )
            )

            n_worst_eff = min(
                n_worst,
                n_samples,
            )

            worst_indices = np.argpartition(
                row_distance,
                -n_worst_eff,
            )[
                -n_worst_eff:
            ]

            available_for_random = np.setdiff1d(
                np.arange(
                    n_samples,
                    dtype=int,
                ),
                worst_indices,
                assume_unique=False,
            )

            n_random_eff = min(
                n_random,
                len(
                    available_for_random
                ),
            )

            if n_random_eff > 0:

                random_indices = rng.choice(
                    available_for_random,
                    size=n_random_eff,
                    replace=False,
                )

                active_indices = np.unique(
                    np.concatenate(
                        [
                            worst_indices,
                            random_indices,
                        ]
                    )
                )

            else:

                active_indices = np.unique(
                    worst_indices
                )

            # Work on the worst points first, followed by the
            # random exploratory subset.
            active_indices = active_indices[
                np.argsort(
                    row_distance[
                        active_indices
                    ]
                )[
                    ::-1
                ]
            ]

            locked = np.zeros(
                n_samples,
                dtype=bool,
            )

            accepted_year = 0

            for i in active_indices:

                i = int(
                    i
                )

                if locked[
                    i
                ]:
                    continue

                candidate_i = int(
                    assignment[
                        i
                    ]
                )

                current_point_i = (
                    candidate_points[
                        candidate_i
                    ]
                )

                # --------------------------------------------
                # Build promising partner set.
                # --------------------------------------------

                partner_set = set()

                k_eff = min(
                    partner_k,
                    n_samples,
                )

                # Which trajectories would like i's current row?
                _, desired_neighbors = (
                    desired_trees[
                        year
                    ].query(
                        current_point_i,
                        k=k_eff,
                    )
                )

                for j in np.atleast_1d(
                    desired_neighbors
                ):

                    partner_set.add(
                        int(
                            j
                        )
                    )

                # Which candidate rows fit i's desired point?
                # Add the trajectories currently holding them.
                _, candidate_neighbors = (
                    candidate_trees[
                        year
                    ].query(
                        desired_points[
                            i
                        ],
                        k=k_eff,
                    )
                )

                for candidate_j in np.atleast_1d(
                    candidate_neighbors
                ):

                    holder = inverse_assignment[
                        int(
                            candidate_j
                        )
                    ]

                    partner_set.add(
                        int(
                            holder
                        )
                    )

                # Random partners make it possible to improve
                # temporal correlation even when a useful swap
                # is not locally obvious in rank space.
                if random_partners > 0:

                    random_js = rng.integers(
                        0,
                        n_samples,
                        size=random_partners,
                    )

                    for j in random_js:

                        partner_set.add(
                            int(
                                j
                            )
                        )

                partner_set.discard(
                    i
                )

                partners = np.asarray(
                    [
                        j
                        for j in partner_set
                        if not locked[
                            j
                        ]
                    ],
                    dtype=int,
                )

                if len(
                    partners
                ) == 0:
                    continue

                candidate_js = assignment[
                    partners
                ]

                points_j = candidate_points[
                    candidate_js
                ]

                # --------------------------------------------
                # Rank-error change for each proposed swap.
                # --------------------------------------------

                desired_i = desired_points[
                    i
                ]

                desired_j = desired_points[
                    partners
                ]

                current_rank_pair = (
                    np.sum(
                        (
                            current_point_i
                            - desired_i
                        ) ** 2
                    )
                    + np.sum(
                        (
                            points_j
                            - desired_j
                        ) ** 2,
                        axis=1,
                    )
                )

                swapped_rank_pair = (
                    np.sum(
                        (
                            points_j
                            - desired_i
                        ) ** 2,
                        axis=1,
                    )
                    + np.sum(
                        (
                            current_point_i
                            - desired_j
                        ) ** 2,
                        axis=1,
                    )
                )

                delta_rank_sse = (
                    swapped_rank_pair
                    - current_rank_pair
                )

                # --------------------------------------------
                # Temporal-correlation change.
                #
                # Swapping year t values at i/j changes only
                # correlations involving year t. Because each
                # yearly marginal is merely permuted, means and
                # variances are fixed.
                # --------------------------------------------

                delta_temporal_sse = np.zeros(
                    len(
                        partners
                    ),
                    dtype=float,
                )

                e_i_t = assigned_eais[
                    i,
                    t,
                ]

                e_j_t = assigned_eais[
                    partners,
                    t,
                ]

                w_i_t = assigned_wais[
                    i,
                    t,
                ]

                w_j_t = assigned_wais[
                    partners,
                    t,
                ]

                for s in range(
                    n_years
                ):

                    if s == t:
                        continue

                    # EAIS dot-product update for each partner.
                    delta_dot_e = (
                        e_j_t
                        - e_i_t
                    ) * (
                        assigned_eais[
                            i,
                            s,
                        ]
                        - assigned_eais[
                            partners,
                            s,
                        ]
                    )

                    old_rho_e = (
                        e_dots[
                            t,
                            s,
                        ]
                        / e_denom[
                            t,
                            s,
                        ]
                    )

                    new_rho_e = (
                        e_dots[
                            t,
                            s,
                        ]
                        + delta_dot_e
                    ) / e_denom[
                        t,
                        s,
                    ]

                    target_rho_e = (
                        target_eais_spearman[
                            t,
                            s,
                        ]
                    )

                    delta_temporal_sse += (
                        (
                            new_rho_e
                            - target_rho_e
                        ) ** 2
                        - (
                            old_rho_e
                            - target_rho_e
                        ) ** 2
                    )

                    # WAIS.
                    delta_dot_w = (
                        w_j_t
                        - w_i_t
                    ) * (
                        assigned_wais[
                            i,
                            s,
                        ]
                        - assigned_wais[
                            partners,
                            s,
                        ]
                    )

                    old_rho_w = (
                        w_dots[
                            t,
                            s,
                        ]
                        / w_denom[
                            t,
                            s,
                        ]
                    )

                    new_rho_w = (
                        w_dots[
                            t,
                            s,
                        ]
                        + delta_dot_w
                    ) / w_denom[
                        t,
                        s,
                    ]

                    target_rho_w = (
                        target_wais_spearman[
                            t,
                            s,
                        ]
                    )

                    delta_temporal_sse += (
                        (
                            new_rho_w
                            - target_rho_w
                        ) ** 2
                        - (
                            old_rho_w
                            - target_rho_w
                        ) ** 2
                    )

                # Six temporal correlation targets total:
                # 3 pairs x 2 components.
                delta_temporal_mse = (
                    delta_temporal_sse
                    / 6.0
                )

                delta_rank_mse = (
                    delta_rank_sse
                    / (
                        2.0
                        * n_samples
                        * n_years
                    )
                )

                delta_objective = (
                    temporal_weight
                    * delta_temporal_mse
                    + rank_weight
                    * delta_rank_mse
                )

                best_position = int(
                    np.argmin(
                        delta_objective
                    )
                )

                best_delta = float(
                    delta_objective[
                        best_position
                    ]
                )

                if best_delta >= (
                    -min_improvement
                ):
                    continue

                j = int(
                    partners[
                        best_position
                    ]
                )

                candidate_j = int(
                    assignment[
                        j
                    ]
                )

                # --------------------------------------------
                # Store PRE-SWAP values for exact dot updates.
                # --------------------------------------------

                old_e_i_t = float(
                    assigned_eais[
                        i,
                        t,
                    ]
                )

                old_e_j_t = float(
                    assigned_eais[
                        j,
                        t,
                    ]
                )

                old_w_i_t = float(
                    assigned_wais[
                        i,
                        t,
                    ]
                )

                old_w_j_t = float(
                    assigned_wais[
                        j,
                        t,
                    ]
                )

                # Exact temporal SSE update for the accepted j.
                accepted_delta_temporal = float(
                    delta_temporal_sse[
                        best_position
                    ]
                )

                accepted_delta_rank = float(
                    delta_rank_sse[
                        best_position
                    ]
                )

                # --------------------------------------------
                # Perform intact-row swap.
                # --------------------------------------------

                assignment[
                    i
                ] = candidate_j

                assignment[
                    j
                ] = candidate_i

                inverse_assignment[
                    candidate_i
                ] = j

                inverse_assignment[
                    candidate_j
                ] = i

                # Both EAIS and WAIS ranks travel with the same
                # intact B25 row.
                assigned_eais[
                    i,
                    t,
                ] = old_e_j_t

                assigned_eais[
                    j,
                    t,
                ] = old_e_i_t

                assigned_wais[
                    i,
                    t,
                ] = old_w_j_t

                assigned_wais[
                    j,
                    t,
                ] = old_w_i_t

                # Update temporal dot products exactly.
                for s in range(
                    n_years
                ):

                    if s == t:
                        continue

                    delta_dot_e_scalar = (
                        old_e_j_t
                        - old_e_i_t
                    ) * (
                        assigned_eais[
                            i,
                            s,
                        ]
                        - assigned_eais[
                            j,
                            s,
                        ]
                    )

                    e_dots[
                        t,
                        s,
                    ] += delta_dot_e_scalar

                    e_dots[
                        s,
                        t,
                    ] = e_dots[
                        t,
                        s,
                    ]

                    delta_dot_w_scalar = (
                        old_w_j_t
                        - old_w_i_t
                    ) * (
                        assigned_wais[
                            i,
                            s,
                        ]
                        - assigned_wais[
                            j,
                            s,
                        ]
                    )

                    w_dots[
                        t,
                        s,
                    ] += delta_dot_w_scalar

                    w_dots[
                        s,
                        t,
                    ] = w_dots[
                        t,
                        s,
                    ]

                temporal_sse += (
                    accepted_delta_temporal
                )

                current_rank_sse += (
                    accepted_delta_rank
                )

                locked[
                    i
                ] = True

                locked[
                    j
                ] = True

                accepted_year += 1

                accepted_this_pass += 1

            print(
                f"    pass {pass_number}, {year}: "
                f"{accepted_year:,} accepted joint swaps",
                flush=True,
            )

        # ----------------------------------------------------
        # Exact full recomputation after the pass.
        # ----------------------------------------------------

        stats = _joint_objective_summary(
            assignments=assignments,
            candidate_points_by_year=(
                candidate_points_by_year
            ),
            desired_eais_ranks=(
                desired_eais_ranks
            ),
            desired_wais_ranks=(
                desired_wais_ranks
            ),
            target_eais_spearman=(
                target_eais_spearman
            ),
            target_wais_spearman=(
                target_wais_spearman
            ),
            temporal_weight=temporal_weight,
            rank_weight=rank_weight,
        )

        summary_records.append(
            _joint_summary_record(
                stage="joint_optimization",
                pass_number=pass_number,
                stats=stats,
                accepted_swaps=(
                    accepted_this_pass
                ),
            )
        )

        print(
            f"\n    pass {pass_number} complete: "
            f"{accepted_this_pass:,} swaps | "
            f"objective={stats['objective']:.6f} | "
            f"temporal MSE={stats['temporal_mse']:.6f} | "
            f"rank MSE={stats['rank_mse']:.6f}",
            flush=True,
        )

        print(
            f"        EAIS rho: "
            f"{stats['achieved_eais'][0,1]:.4f}, "
            f"{stats['achieved_eais'][0,2]:.4f}, "
            f"{stats['achieved_eais'][1,2]:.4f}",
            flush=True,
        )

        print(
            f"        WAIS rho: "
            f"{stats['achieved_wais'][0,1]:.4f}, "
            f"{stats['achieved_wais'][0,2]:.4f}, "
            f"{stats['achieved_wais'][1,2]:.4f}",
            flush=True,
        )

        if accepted_this_pass == 0:

            print(
                "    no further joint-objective-improving swaps found",
                flush=True,
            )

            break

    # Final permutation validation.
    for year in YEARS:

        a = assignments[
            year
        ]

        if len(
            np.unique(
                a
            )
        ) != n_samples:

            raise RuntimeError(
                f"Joint optimization broke the {year} "
                "one-to-one assignment."
            )

        if not np.array_equal(
            np.sort(
                a
            ),
            np.arange(
                n_samples
            ),
        ):

            raise RuntimeError(
                f"Joint optimization broke the {year} "
                "complete permutation."
            )

    return (
        assignments,
        pd.DataFrame(
            summary_records
        ),
    )


# ============================================================
# 11. CALCULATE WIDE TOTALS FROM ANY B25 WIDE DATAFRAME
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
    target_eais_latent: np.ndarray,
    target_wais_latent: np.ndarray,
):
    """
    Rebuild complete B25 trajectories in three stages:

        1. fast local/bin one-to-one match independently at each year;
        2. deterministic rank-distance swap refinement at each year
           as a warm start;
        3. JOINT optimization of all three yearly permutations using
           the full three-year temporal-correlation objective.

    At all stages every yearly assignment remains a complete
    permutation of the original B25 rows.
    """

    n_samples = len(
        mc_df
    )

    expected_shape = (
        n_samples,
        len(
            YEARS
        ),
    )

    if eais_desired_ranks.shape != expected_shape:
        raise ValueError(
            "Unexpected EAIS desired-rank array shape."
        )

    if wais_desired_ranks.shape != expected_shape:
        raise ValueError(
            "Unexpected WAIS desired-rank array shape."
        )

    columns_by_year = get_columns_by_year(
        meta_df=meta_df,
        years=YEARS,
    )

    # Store all yearly matching objects first. We cannot assemble
    # final panels until AFTER the joint multi-year optimization.
    rank_tables = {}
    desired_points_by_year = {}
    candidate_points_by_year = {}
    warm_start_assignments = {}
    local_assignments = {}
    local_distances_by_year = {}
    warm_distances_by_year = {}
    warm_swap_summary_frames = []

    # --------------------------------------------------------
    # A. Build independent warm-start permutations.
    # --------------------------------------------------------

    for year_position, year in enumerate(
        YEARS
    ):

        print_heading(
            f"WARM-START MATCHING B25 PANEL: {year}"
        )

        rank_table = build_component_rank_table(
            mc_total_ext=mc_total_ext,
            warming=warming,
            year=year,
        )

        if len(
            rank_table
        ) != n_samples:

            raise ValueError(
                f"Expected {n_samples} B25 rank rows at {year}; "
                f"found {len(rank_table)}."
            )

        desired_points = np.column_stack(
            [
                eais_desired_ranks[
                    :,
                    year_position,
                ],
                wais_desired_ranks[
                    :,
                    year_position,
                ],
            ]
        )

        candidate_points = rank_table[
            [
                "rank_EAIS",
                "rank_WAIS",
            ]
        ].to_numpy(
            dtype=float
        )

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
            seed=MATCHING_SEED + int(
                year
            ),
        )

        local_assignments[
            year
        ] = assignment.copy()

        local_distances_by_year[
            year
        ] = assignment_distances.copy()

        # Rank-only swap refinement is retained only as a warm start.
        # The final scientific optimization is the joint stage below.
        (
            assignment,
            assigned_points,
            rank_errors,
            assignment_distances,
            warm_swap_summary,
        ) = refine_assignment_with_swaps(
            desired_points=desired_points,
            candidate_points=candidate_points,
            assignment=assignment,
            distance_threshold=SWAP_REFINE_THRESHOLD,
            partner_k=SWAP_PARTNER_K,
            max_passes=SWAP_MAX_PASSES,
            min_improvement=SWAP_MIN_IMPROVEMENT,
        )

        if not warm_swap_summary.empty:

            warm_swap_summary.insert(
                0,
                "year",
                year,
            )

            warm_swap_summary_frames.append(
                warm_swap_summary
            )

        rank_tables[
            year
        ] = rank_table

        desired_points_by_year[
            year
        ] = desired_points

        candidate_points_by_year[
            year
        ] = candidate_points

        warm_start_assignments[
            year
        ] = assignment.copy()

        warm_distances_by_year[
            year
        ] = assignment_distances.copy()

    # --------------------------------------------------------
    # B. JOINT MULTI-YEAR PERMUTATION OPTIMIZATION.
    # --------------------------------------------------------

    (
        final_assignments,
        joint_summary,
    ) = optimize_joint_multi_year_permutations(
        assignments=warm_start_assignments,
        candidate_points_by_year=(
            candidate_points_by_year
        ),
        desired_eais_ranks=eais_desired_ranks,
        desired_wais_ranks=wais_desired_ranks,
        target_eais_latent=target_eais_latent,
        target_wais_latent=target_wais_latent,
    )

    # --------------------------------------------------------
    # C. Assemble final intact B25 panels and metadata.
    # --------------------------------------------------------

    selected_panels = []
    metadata_frames = []
    assignment_quality_records = []

    sample_to_position = pd.Series(
        np.arange(
            len(
                mc_df
            ),
            dtype=int,
        ),
        index=mc_df[
            "sample_id"
        ].to_numpy(
            dtype=int
        ),
    )

    for year_position, year in enumerate(
        YEARS
    ):

        rank_table = rank_tables[
            year
        ]

        desired_points = (
            desired_points_by_year[
                year
            ]
        )

        candidate_points = (
            candidate_points_by_year[
                year
            ]
        )

        local_assignment = (
            local_assignments[
                year
            ]
        )

        warm_assignment = (
            warm_start_assignments[
                year
            ]
        )

        final_assignment = (
            final_assignments[
                year
            ]
        )

        final_assigned_points = (
            candidate_points[
                final_assignment
            ]
        )

        final_rank_errors = (
            final_assigned_points
            - desired_points
        )

        final_distances = np.sqrt(
            np.sum(
                final_rank_errors ** 2,
                axis=1,
            )
        )

        source_sample_ids = rank_table[
            "sample_id"
        ].to_numpy(
            dtype=int
        )

        selected_source_sample_ids = (
            source_sample_ids[
                final_assignment
            ]
        )

        year_columns = columns_by_year[
            year
        ]

        year_panel = mc_df.loc[
            :,
            [
                "sample_id",
                *year_columns,
            ]
        ].copy()

        source_positions = sample_to_position.loc[
            selected_source_sample_ids
        ].to_numpy(
            dtype=int
        )

        selected_panel = (
            year_panel.iloc[
                source_positions
            ]
            .reset_index(
                drop=True
            )
            .drop(
                columns=[
                    "sample_id"
                ]
            )
        )

        selected_panels.append(
            selected_panel
        )

        assigned_rank_table = (
            rank_table.iloc[
                final_assignment
            ]
            .reset_index(
                drop=True
            )
        )

        local_distances = (
            local_distances_by_year[
                year
            ]
        )

        warm_distances = (
            warm_distances_by_year[
                year
            ]
        )

        metadata = pd.DataFrame(
            {
                "trajectory_id": np.arange(
                    n_samples,
                    dtype=int,
                ),
                "year": year,
                "source_sample_id": (
                    selected_source_sample_ids
                ),
                "desired_rank_EAIS": (
                    desired_points[
                        :,
                        0,
                    ]
                ),
                "desired_rank_WAIS": (
                    desired_points[
                        :,
                        1,
                    ]
                ),
                "assigned_rank_EAIS": (
                    final_assigned_points[
                        :,
                        0,
                    ]
                ),
                "assigned_rank_WAIS": (
                    final_assigned_points[
                        :,
                        1,
                    ]
                ),
                "rank_error_EAIS": (
                    final_rank_errors[
                        :,
                        0,
                    ]
                ),
                "rank_error_WAIS": (
                    final_rank_errors[
                        :,
                        1,
                    ]
                ),
                "rank_distance": (
                    final_distances
                ),
                "local_rank_distance": (
                    local_distances
                ),
                "warm_start_rank_distance": (
                    warm_distances
                ),
                "final_joint_rank_distance": (
                    final_distances
                ),
                "changed_by_rank_swap": (
                    local_assignment
                    != warm_assignment
                ),
                "changed_by_joint_optimization": (
                    warm_assignment
                    != final_assignment
                ),
                "assigned_EAIS_value": (
                    assigned_rank_table[
                        "EAIS"
                    ].to_numpy(
                        dtype=float
                    )
                ),
                "assigned_WAIS_value": (
                    assigned_rank_table[
                        "WAIS"
                    ].to_numpy(
                        dtype=float
                    )
                ),
                "assigned_GrIS_value": (
                    assigned_rank_table[
                        "GrIS"
                    ].to_numpy(
                        dtype=float
                    )
                ),
                "assigned_Total_AIS_value": (
                    assigned_rank_table[
                        "Total AIS"
                    ].to_numpy(
                        dtype=float
                    )
                ),
            }
        )

        metadata_frames.append(
            metadata
        )

        assignment_quality_records.append(
            {
                "year": year,
                "n_samples": n_samples,

                "local_mean_rank_distance": float(
                    np.mean(
                        local_distances
                    )
                ),
                "warm_start_mean_rank_distance": float(
                    np.mean(
                        warm_distances
                    )
                ),
                "final_joint_mean_rank_distance": float(
                    np.mean(
                        final_distances
                    )
                ),

                "local_median_rank_distance": float(
                    np.median(
                        local_distances
                    )
                ),
                "warm_start_median_rank_distance": float(
                    np.median(
                        warm_distances
                    )
                ),
                "final_joint_median_rank_distance": float(
                    np.median(
                        final_distances
                    )
                ),

                "local_p95_rank_distance": float(
                    np.quantile(
                        local_distances,
                        0.95,
                    )
                ),
                "warm_start_p95_rank_distance": float(
                    np.quantile(
                        warm_distances,
                        0.95,
                    )
                ),
                "final_joint_p95_rank_distance": float(
                    np.quantile(
                        final_distances,
                        0.95,
                    )
                ),

                "final_joint_p99_rank_distance": float(
                    np.quantile(
                        final_distances,
                        0.99,
                    )
                ),
                "final_joint_max_rank_distance": float(
                    np.max(
                        final_distances
                    )
                ),

                "final_percent_above_0.10": float(
                    100
                    * np.mean(
                        final_distances
                        > 0.10
                    )
                ),
                "final_percent_above_0.25": float(
                    100
                    * np.mean(
                        final_distances
                        > 0.25
                    )
                ),

                "percent_changed_by_rank_swap": float(
                    100
                    * np.mean(
                        local_assignment
                        != warm_assignment
                    )
                ),
                "percent_changed_by_joint_optimization": float(
                    100
                    * np.mean(
                        warm_assignment
                        != final_assignment
                    )
                ),

                "mean_abs_rank_error_EAIS": float(
                    np.mean(
                        np.abs(
                            final_rank_errors[
                                :,
                                0,
                            ]
                        )
                    )
                ),
                "mean_abs_rank_error_WAIS": float(
                    np.mean(
                        np.abs(
                            final_rank_errors[
                                :,
                                1,
                            ]
                        )
                    )
                ),
            }
        )

    coupled_df = pd.concat(
        selected_panels,
        axis=1,
    )

    original_scientific_order = [
        column
        for column in mc_df.columns
        if column != "sample_id"
    ]

    coupled_df = coupled_df.loc[
        :,
        original_scientific_order
    ]

    coupled_df.insert(
        0,
        "sample_id",
        np.arange(
            n_samples,
            dtype=int,
        ),
    )

    coupling_metadata = pd.concat(
        metadata_frames,
        ignore_index=True,
    )

    assignment_quality = pd.DataFrame(
        assignment_quality_records
    )

    warm_swap_summary = (
        pd.concat(
            warm_swap_summary_frames,
            ignore_index=True,
        )
        if warm_swap_summary_frames
        else pd.DataFrame()
    )

    return (
        coupled_df,
        coupling_metadata,
        assignment_quality,
        warm_swap_summary,
        joint_summary,
    )


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

def run_experiment(
    original_data: dict,
    experiment: dict,
):
    """
    Run one complete warming-library temporal-coupling experiment.

    Returns compact diagnostic tables for the cross-experiment summary.
    """

    configure_experiment(
        experiment
    )

    print_heading(
        "SCRIPT B: COMPONENT-SPECIFIC JOINT MULTI-YEAR COUPLING"
    )

    print(
        f"Experiment: B25 {WARMING}°C / "
        f"DP21 {DP21_SCENARIO_LABEL}"
    )

    print(
        f"Objective weights: temporal={JOINT_TEMPORAL_WEIGHT}, "
        f"rank={JOINT_RANK_WEIGHT}"
    )

    print(
        f"\nOriginal B25 file:\n{data_file}"
    )

    print(
        f"\nExperiment output directory:\n{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # A. Reuse original B25 data loaded once by main()
    # --------------------------------------------------------

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
        seed=CURRENT_RANDOM_SEED,
    )

    (
        latent_wais,
        desired_wais,
        rank_positions_wais,
    ) = simulate_component_rank_histories(
        correlation_matrix=target_wais,
        n_samples=n_samples,
        seed=CURRENT_RANDOM_SEED + 1,
    )

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
    # D. Match / jointly optimize intact B25 yearly panels
    # --------------------------------------------------------

    (
        coupled_df,
        coupling_metadata,
        assignment_quality,
        swap_summary,
        joint_summary,
    ) = couple_component_specific_panels(
        mc_df=mc_df,
        meta_df=meta_df,
        mc_total_ext=mc_total_ext,
        warming=WARMING,
        eais_desired_ranks=desired_eais,
        wais_desired_ranks=desired_wais,
        target_eais_latent=target_eais,
        target_wais_latent=target_wais,
    )

    # --------------------------------------------------------
    # E. Exact marginal check
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
    # F. Save experiment outputs
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
                "_warm_start_swap_refinement_summary.csv"
            ),
            index=False,
        )

    joint_summary.to_csv(
        OUTPUT_DIR
        / (
            f"{EXPERIMENT_NAME}"
            "_joint_optimization_summary.csv"
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
        "JOINT OPTIMIZATION SUMMARY"
    )

    print(
        joint_summary
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

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
        "temporally targeted because DP21 does not provide "
        "Greenland trajectories."
    )

    print_heading(
        f"FINISHED: {WARMING}°C / {DP21_SCENARIO_LABEL}"
    )

    print(
        f"Outputs saved to:\n{OUTPUT_DIR}"
    )

    # Add experiment labels to compact summaries.
    temporal_summary = (
        temporal_diagnostics.copy()
    )

    temporal_summary.insert(
        0,
        "warming",
        WARMING,
    )

    temporal_summary.insert(
        1,
        "warming_label",
        WARMING_LABEL,
    )

    temporal_summary.insert(
        2,
        "dp21_scenario",
        DP21_SCENARIO,
    )

    temporal_summary.insert(
        3,
        "dp21_scenario_label",
        DP21_SCENARIO_LABEL,
    )

    assignment_summary = (
        assignment_quality.copy()
    )

    assignment_summary.insert(
        0,
        "warming",
        WARMING,
    )

    assignment_summary.insert(
        1,
        "warming_label",
        WARMING_LABEL,
    )

    assignment_summary.insert(
        2,
        "dp21_scenario",
        DP21_SCENARIO,
    )

    joint_summary_out = (
        joint_summary.copy()
    )

    joint_summary_out.insert(
        0,
        "warming",
        WARMING,
    )

    joint_summary_out.insert(
        1,
        "warming_label",
        WARMING_LABEL,
    )

    joint_summary_out.insert(
        2,
        "dp21_scenario",
        DP21_SCENARIO,
    )

    validation_record = pd.DataFrame(
        [
            {
                "warming": WARMING,
                "warming_label": WARMING_LABEL,
                "dp21_scenario": DP21_SCENARIO,
                "dp21_scenario_label": (
                    DP21_SCENARIO_LABEL
                ),
                "n_variables_checked": len(
                    marginal_diagnostics
                ),
                "n_failed_marginal_checks": int(
                    (
                        ~marginal_diagnostics[
                            "exact_match"
                        ]
                    ).sum()
                ),
                "maximum_absolute_marginal_difference": float(
                    marginal_diagnostics[
                        "maximum_absolute_difference"
                    ].max()
                ),
                "maximum_absolute_within_year_spearman_difference": float(
                    np.max(
                        np.abs(
                            within_year_diagnostics[
                                "difference"
                            ].to_numpy(
                                dtype=float
                            )
                        )
                    )
                ),
            }
        ]
    )

    return {
        "temporal": temporal_summary,
        "assignment": assignment_summary,
        "joint": joint_summary_out,
        "validation": validation_record,
    }


def main():

    print_heading(
        "SCRIPT B: ALL-WARMING JOINT MULTI-YEAR COUPLING"
    )

    print(
        "Coupling experiments:"
    )

    for experiment in EXPERIMENTS:

        print(
            f"    B25 {experiment['warming']}°C "
            f"<-> DP21 {experiment['dp21_scenario_label']}"
        )

    print(
        f"\nFinal objective weights: "
        f"temporal={JOINT_TEMPORAL_WEIGHT}, "
        f"rank={JOINT_RANK_WEIGHT}"
    )

    print(
        f"\nLoading original B25 data once from:\n{data_file}"
    )

    original_data = load_and_prepare_data(
        file_path=str(
            data_file
        )
    )

    all_temporal = []
    all_assignment = []
    all_joint = []
    all_validation = []

    for experiment_number, experiment in enumerate(
        EXPERIMENTS,
        start=1,
    ):

        print_heading(
            f"EXPERIMENT {experiment_number} OF {len(EXPERIMENTS)}"
        )

        result = run_experiment(
            original_data=original_data,
            experiment=experiment,
        )

        all_temporal.append(
            result[
                "temporal"
            ]
        )

        all_assignment.append(
            result[
                "assignment"
            ]
        )

        all_joint.append(
            result[
                "joint"
            ]
        )

        all_validation.append(
            result[
                "validation"
            ]
        )

    # --------------------------------------------------------
    # Combined cross-warming summaries
    # --------------------------------------------------------

    combined_temporal = pd.concat(
        all_temporal,
        ignore_index=True,
    )

    combined_assignment = pd.concat(
        all_assignment,
        ignore_index=True,
    )

    combined_joint = pd.concat(
        all_joint,
        ignore_index=True,
    )

    combined_validation = pd.concat(
        all_validation,
        ignore_index=True,
    )

    combined_temporal.to_csv(
        COMBINED_SUMMARY_DIR
        / "all_warming_temporal_correlation_diagnostics.csv",
        index=False,
    )

    combined_assignment.to_csv(
        COMBINED_SUMMARY_DIR
        / "all_warming_assignment_quality.csv",
        index=False,
    )

    combined_joint.to_csv(
        COMBINED_SUMMARY_DIR
        / "all_warming_joint_optimization_summary.csv",
        index=False,
    )

    combined_validation.to_csv(
        COMBINED_SUMMARY_DIR
        / "all_warming_validation_summary.csv",
        index=False,
    )

    print_heading(
        "ALL THREE WARMING LIBRARIES FINISHED"
    )

    print(
        "Cross-warming validation summary:"
    )

    print(
        combined_validation
        .round(
            8
        )
        .to_string(
            index=False
        )
    )

    print(
        f"\nCombined summary files saved to:\n"
        f"{COMBINED_SUMMARY_DIR}"
    )


if __name__ == "__main__":
    main()
