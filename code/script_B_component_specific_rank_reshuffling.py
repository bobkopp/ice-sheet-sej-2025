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
    *_marginal_preservation.csv

Figures:
    temporal_correlation_comparison.png
    assignment_rank_scatter_2100.png
    assignment_rank_scatter_2200.png
    assignment_rank_scatter_2300.png
    assignment_distance_by_year.png

Method note
-----------
A full Hungarian assignment on 50,000 x 50,000 rows is infeasible.
This script uses a scalable approximate one-to-one matcher:

    - KD-tree nearest-neighbor candidate search,
    - greedy unique assignment,
    - progressively expanded candidate sets,
    - exact Hungarian solution only for a small unresolved remainder.

The final assignment is always a permutation, so exact B25 marginal
preservation does not depend on matching optimality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
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

# Initial number of nearest B25 candidates retained per desired rank vector.
INITIAL_K = 64

# Candidate search is progressively enlarged for unresolved trajectories.
K_SEQUENCE = [
    64,
    128,
    256,
    512,
    1024,
]

# If the final unresolved set is this size or smaller, solve that remainder
# exactly with the Hungarian algorithm.
MAX_HUNGARIAN_REMAINDER = 2500

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
# 9. LOCAL / BINNED ONE-TO-ONE RANK-VECTOR MATCHING
# ============================================================

from scipy.spatial import cKDTree


# Number of bins along EACH rank dimension.
#
# 100 means:
#     EAIS percentile rank: 100 bins
#     WAIS percentile rank: 100 bins
#
# so rank space is divided into a 100 x 100 grid.
#
# Each bin is 0.01 wide in percentile-rank space.
N_RANK_BINS = 100

# When a desired point cannot be matched inside its own bin,
# search among nearby unused B25 rows.
#
# This is only used for the unmatched remainder.
LOCAL_K = 64

# Seed controls random ordering within bins.
MATCHING_SEED = 24680


def binned_local_unique_assignment(
    desired_points: np.ndarray,
    candidate_points: np.ndarray,
    n_bins: int = N_RANK_BINS,
    local_k: int = LOCAL_K,
    seed: int = MATCHING_SEED,
):
    """
    Approximate one-to-one matching of desired EAIS/WAIS rank
    vectors to intact B25 rows.

    Strategy
    --------
    1. Divide [0,1] x [0,1] EAIS-WAIS rank space into bins.

    2. Match desired points and B25 candidate rows that occupy
       the same bin.

    3. Remove those matched desired points and candidate rows.

    4. For the remaining points, perform local nearest-neighbor
       matching among the remaining unused B25 rows.

    5. Finish any small remainder by assigning the remaining
       desired points to the remaining candidate rows using a
       distance-aware greedy procedure.

    Every candidate B25 row is used exactly once.

    Therefore the result is a permutation of the original B25
    yearly rows.

    This preserves:
        - all original B25 marginal distributions exactly;
        - all variables belonging to each intact B25 row;
        - the original within-year B25 joint sample cloud.

    It only changes:
        - which B25 row is connected to which temporal trajectory.
    """

    rng = np.random.default_rng(
        seed
    )

    desired_points = np.asarray(
        desired_points,
        dtype=float,
    )

    candidate_points = np.asarray(
        candidate_points,
        dtype=float,
    )

    if desired_points.shape != candidate_points.shape:
        raise ValueError(
            "desired_points and candidate_points must "
            f"have identical shapes. Got "
            f"{desired_points.shape} and "
            f"{candidate_points.shape}."
        )

    if desired_points.ndim != 2:
        raise ValueError(
            "Rank arrays must be two-dimensional."
        )

    n_samples, n_dimensions = (
        desired_points.shape
    )

    if n_dimensions != 2:
        raise ValueError(
            "This matcher currently expects exactly two "
            "rank dimensions: EAIS and WAIS."
        )

    if not np.isfinite(
        desired_points
    ).all():
        raise ValueError(
            "Non-finite desired ranks found."
        )

    if not np.isfinite(
        candidate_points
    ).all():
        raise ValueError(
            "Non-finite B25 candidate ranks found."
        )

    print(
        f"    local/bin matching with "
        f"{n_bins} x {n_bins} rank grid",
        flush=True,
    )

    # --------------------------------------------------------
    # A. Convert percentile ranks into integer bin coordinates
    # --------------------------------------------------------

    desired_bins = np.floor(
        desired_points * n_bins
    ).astype(int)

    candidate_bins = np.floor(
        candidate_points * n_bins
    ).astype(int)

    # Rank = 1.0 would otherwise become bin n_bins.
    desired_bins = np.clip(
        desired_bins,
        0,
        n_bins - 1,
    )

    candidate_bins = np.clip(
        candidate_bins,
        0,
        n_bins - 1,
    )

    # Convert 2-D bin coordinate to one integer bin ID.
    desired_bin_id = (
        desired_bins[:, 0] * n_bins
        + desired_bins[:, 1]
    )

    candidate_bin_id = (
        candidate_bins[:, 0] * n_bins
        + candidate_bins[:, 1]
    )

    # Final mapping:
    #
    # assignment[i] = B25 candidate row assigned
    #                 to desired trajectory i
    assignment = np.full(
        n_samples,
        -1,
        dtype=int,
    )

    candidate_used = np.zeros(
        n_samples,
        dtype=bool,
    )

    # --------------------------------------------------------
    # B. SAME-BIN MATCHING
    # --------------------------------------------------------

    print(
        "    stage 1: matching within identical rank bins...",
        flush=True,
    )

    desired_by_bin = {}
    candidate_by_bin = {}

    for i, bin_id in enumerate(
        desired_bin_id
    ):
        desired_by_bin.setdefault(
            int(bin_id),
            [],
        ).append(
            i
        )

    for j, bin_id in enumerate(
        candidate_bin_id
    ):
        candidate_by_bin.setdefault(
            int(bin_id),
            [],
        ).append(
            j
        )

    same_bin_matches = 0

    all_bins = set(
        desired_by_bin.keys()
    ).intersection(
        candidate_by_bin.keys()
    )

    for bin_id in all_bins:

        desired_idx = np.asarray(
            desired_by_bin[
                bin_id
            ],
            dtype=int,
        )

        candidate_idx = np.asarray(
            candidate_by_bin[
                bin_id
            ],
            dtype=int,
        )

        # Randomize within the tiny local bin so there is no
        # arbitrary dependence on original dataframe ordering.
        rng.shuffle(
            desired_idx
        )

        rng.shuffle(
            candidate_idx
        )

        n_match = min(
            len(
                desired_idx
            ),
            len(
                candidate_idx
            ),
        )

        if n_match == 0:
            continue

        assignment[
            desired_idx[
                :n_match
            ]
        ] = candidate_idx[
            :n_match
        ]

        candidate_used[
            candidate_idx[
                :n_match
            ]
        ] = True

        same_bin_matches += n_match

    print(
        f"        same-bin matches: "
        f"{same_bin_matches:,} / "
        f"{n_samples:,} "
        f"({100 * same_bin_matches / n_samples:.2f}%)",
        flush=True,
    )

    # --------------------------------------------------------
    # C. IDENTIFY REMAINDER
    # --------------------------------------------------------

    unresolved_desired = np.where(
        assignment < 0
    )[0]

    unused_candidates = np.where(
        ~candidate_used
    )[0]

    if (
        len(
            unresolved_desired
        )
        != len(
            unused_candidates
        )
    ):
        raise RuntimeError(
            "Desired/candidate remainder sizes differ."
        )

    print(
        f"    stage 2: local matching for "
        f"{len(unresolved_desired):,} remaining rows...",
        flush=True,
    )

    # --------------------------------------------------------
    # D. LOCAL NEAREST-NEIGHBOR MATCHING OF REMAINDER
    # --------------------------------------------------------

    if len(
        unresolved_desired
    ) > 0:

        remaining_candidate_points = (
            candidate_points[
                unused_candidates
            ]
        )

        tree = cKDTree(
            remaining_candidate_points
        )

        k_eff = min(
            local_k,
            len(
                unused_candidates
            ),
        )

        distances, neighbors = tree.query(
            desired_points[
                unresolved_desired
            ],
            k=k_eff,
            workers=-1,
        )

        if k_eff == 1:

            distances = distances[
                :,
                np.newaxis,
            ]

            neighbors = neighbors[
                :,
                np.newaxis,
            ]

        # ----------------------------------------------------
        # Match the most constrained desired points first.
        #
        # We use nearest-neighbor distance as a simple measure:
        # points with relatively poor local matches get priority.
        # ----------------------------------------------------

        nearest_distance = distances[
            :,
            0
        ]

        order = np.argsort(
            nearest_distance
        )[
            ::-1
        ]

        locally_used = np.zeros(
            len(
                unused_candidates
            ),
            dtype=bool,
        )

        local_matches = 0

        for local_position in order:

            desired_index = (
                unresolved_desired[
                    local_position
                ]
            )

            candidate_choices = neighbors[
                local_position
            ]

            for candidate_local_index in (
                candidate_choices
            ):

                if not locally_used[
                    candidate_local_index
                ]:

                    original_candidate_index = (
                        unused_candidates[
                            candidate_local_index
                        ]
                    )

                    assignment[
                        desired_index
                    ] = (
                        original_candidate_index
                    )

                    locally_used[
                        candidate_local_index
                    ] = True

                    candidate_used[
                        original_candidate_index
                    ] = True

                    local_matches += 1

                    break

        print(
            f"        local KNN matches: "
            f"{local_matches:,}",
            flush=True,
        )

    # --------------------------------------------------------
    # E. FINAL REMAINDER
    # --------------------------------------------------------

    unresolved_desired = np.where(
        assignment < 0
    )[0]

    unused_candidates = np.where(
        ~candidate_used
    )[0]

    if (
        len(
            unresolved_desired
        )
        != len(
            unused_candidates
        )
    ):
        raise RuntimeError(
            "Final desired/candidate remainder sizes differ."
        )

    print(
        f"    stage 3: final remainder = "
        f"{len(unresolved_desired):,}",
        flush=True,
    )

    # If anything remains, use an iterative nearest-neighbor
    # greedy assignment over only the remaining points.
    #
    # This is intentionally a fallback, not the primary matcher.
    if len(
        unresolved_desired
    ) > 0:

        # Randomize desired order first so ties do not depend
        # systematically on trajectory ID.
        unresolved_desired = (
            unresolved_desired.copy()
        )

        rng.shuffle(
            unresolved_desired
        )

        remaining_candidates = set(
            unused_candidates.tolist()
        )

        for count, desired_index in enumerate(
            unresolved_desired,
            start=1,
        ):

            candidate_array = np.fromiter(
                remaining_candidates,
                dtype=int,
            )

            delta = (
                candidate_points[
                    candidate_array
                ]
                - desired_points[
                    desired_index
                ]
            )

            squared_distance = np.sum(
                delta ** 2,
                axis=1,
            )

            best_position = np.argmin(
                squared_distance
            )

            best_candidate = (
                candidate_array[
                    best_position
                ]
            )

            assignment[
                desired_index
            ] = best_candidate

            candidate_used[
                best_candidate
            ] = True

            remaining_candidates.remove(
                int(
                    best_candidate
                )
            )

            if (
                count % 1000 == 0
                or count
                == len(
                    unresolved_desired
                )
            ):

                print(
                    f"        final fallback: "
                    f"{count:,} / "
                    f"{len(unresolved_desired):,}",
                    flush=True,
                )

    # --------------------------------------------------------
    # F. EXACT PERMUTATION VALIDATION
    # --------------------------------------------------------

    if np.any(
        assignment < 0
    ):
        raise RuntimeError(
            "Some desired trajectories remain unmatched."
        )

    if len(
        np.unique(
            assignment
        )
    ) != n_samples:
        raise RuntimeError(
            "Duplicate B25 candidate rows were assigned."
        )

    if not np.array_equal(
        np.sort(
            assignment
        ),
        np.arange(
            n_samples
        ),
    ):
        raise RuntimeError(
            "Final assignment is not a complete permutation "
            "of the B25 rows."
        )

    # --------------------------------------------------------
    # G. MATCH QUALITY
    # --------------------------------------------------------

    assigned_points = candidate_points[
        assignment
    ]

    rank_errors = (
        assigned_points
        - desired_points
    )

    assignment_distances = np.sqrt(
        np.sum(
            rank_errors ** 2,
            axis=1,
        )
    )

    print(
        "    matching complete",
        flush=True,
    )

    print(
        f"        mean rank distance: "
        f"{np.mean(assignment_distances):.5f}",
        flush=True,
    )

    print(
        f"        median rank distance: "
        f"{np.median(assignment_distances):.5f}",
        flush=True,
    )

    print(
        f"        p95 rank distance: "
        f"{np.quantile(assignment_distances, 0.95):.5f}",
        flush=True,
    )

    print(
        f"        p99 rank distance: "
        f"{np.quantile(assignment_distances, 0.99):.5f}",
        flush=True,
    )

    print(
        f"        maximum rank distance: "
        f"{np.max(assignment_distances):.5f}",
        flush=True,
    )

    return (
        assignment,
        assigned_points,
        rank_errors,
        assignment_distances,
    )


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
    """
    Rebuild complete B25 trajectories by matching desired EAIS/WAIS
    rank vectors to intact B25 yearly rows.
    """

    n_samples = len(
        mc_df
    )

    if (
        eais_desired_ranks.shape
        != (
            n_samples,
            len(
                YEARS
            ),
        )
    ):
        raise ValueError(
            "Unexpected EAIS desired-rank array shape."
        )

    if (
        wais_desired_ranks.shape
        != (
            n_samples,
            len(
                YEARS
            ),
        )
    ):
        raise ValueError(
            "Unexpected WAIS desired-rank array shape."
        )

    columns_by_year = get_columns_by_year(
        meta_df=meta_df,
        years=YEARS,
    )

    selected_panels = []
    metadata_frames = []
    assignment_quality_records = []

    for year_position, year in enumerate(
        YEARS
    ):

        print_heading(
            f"MATCHING B25 PANEL: {year}"
        )
        
        import time

        t0 = time.time()
        print("    building B25 component rank table...", flush=True)
        
        rank_table = build_component_rank_table(
            mc_total_ext=mc_total_ext,
            warming=warming,
            year=year,
        )
        
        print(
            f"    rank table finished in {time.time() - t0:.1f} seconds",
            flush=True
        )
        
        t1 = time.time()
        print("    building desired/candidate rank arrays...", flush=True)
        
        desired_points = np.column_stack(
            [
                eais_desired_ranks[:, year_position],
                wais_desired_ranks[:, year_position],
            ]
        )
        
        candidate_points = rank_table[
            ["rank_EAIS", "rank_WAIS"]
        ].to_numpy(dtype=float)
        
        print(
            f"    arrays finished in {time.time() - t1:.1f} seconds",
            flush=True
        )
        
        print("    entering sparse assignment...", flush=True)

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
            seed=MATCHING_SEED + int(year),
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

        # rank_table is sorted by sample_id; confirm the B25 wide frame is too.
        source_sample_ids = rank_table[
            "sample_id"
        ].to_numpy(
            dtype=int
        )

        selected_source_sample_ids = (
            source_sample_ids[
                assignment
            ]
        )

        # Map sample_id -> row position in mc_df safely.
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
        )

        selected_panel = selected_panel.drop(
            columns=[
                "sample_id"
            ]
        )

        selected_panels.append(
            selected_panel
        )

        assigned_rank_table = (
            rank_table.iloc[
                assignment
            ]
            .reset_index(
                drop=True
            )
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
                    assigned_points[
                        :,
                        0,
                    ]
                ),
                "assigned_rank_WAIS": (
                    assigned_points[
                        :,
                        1,
                    ]
                ),
                "rank_error_EAIS": (
                    rank_errors[
                        :,
                        0,
                    ]
                ),
                "rank_error_WAIS": (
                    rank_errors[
                        :,
                        1,
                    ]
                ),
                "rank_distance": (
                    assignment_distances
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
                "mean_rank_distance": float(
                    np.mean(
                        assignment_distances
                    )
                ),
                "median_rank_distance": float(
                    np.median(
                        assignment_distances
                    )
                ),
                "p90_rank_distance": float(
                    np.quantile(
                        assignment_distances,
                        0.90,
                    )
                ),
                "p95_rank_distance": float(
                    np.quantile(
                        assignment_distances,
                        0.95,
                    )
                ),
                "p99_rank_distance": float(
                    np.quantile(
                        assignment_distances,
                        0.99,
                    )
                ),
                "max_rank_distance": float(
                    np.max(
                        assignment_distances
                    )
                ),
                "mean_abs_rank_error_EAIS": float(
                    np.mean(
                        np.abs(
                            rank_errors[
                                :,
                                0,
                            ]
                        )
                    )
                ),
                "mean_abs_rank_error_WAIS": float(
                    np.mean(
                        np.abs(
                            rank_errors[
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

    return (
        coupled_df,
        coupling_metadata,
        assignment_quality,
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
