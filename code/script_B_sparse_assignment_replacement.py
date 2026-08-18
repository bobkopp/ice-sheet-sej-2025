# ============================================================
# REPLACEMENT FOR SECTION 9 OF SCRIPT B
# Sparse one-to-one rank-vector matching
# ============================================================

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree


# Try progressively larger sparse candidate graphs.
# With 50,000 samples, these remain far smaller than a dense
# 50,000 x 50,000 cost matrix.
K_SEQUENCE = [
    32,
    64,
    128,
    256,
    512,
    1024,
]


def sparse_unique_assignment(
    desired_points: np.ndarray,
    candidate_points: np.ndarray,
    k_sequence=K_SEQUENCE,
):
    """
    Find a one-to-one assignment between desired rank vectors and
    intact B25 rows using a sparse minimum-cost bipartite matching.

    Parameters
    ----------
    desired_points : ndarray, shape (N, D)
        Desired component rank vectors for N reconstructed trajectories.

    candidate_points : ndarray, shape (N, D)
        Rank vectors of the N original B25 rows for the same year.

    k_sequence : iterable of int
        Number of nearest candidate rows to expose to each desired
        trajectory. The sparse graph is enlarged until a complete
        one-to-one matching exists.

    Returns
    -------
    assignment : ndarray, shape (N,)
        assignment[i] gives the candidate-row position assigned to
        desired trajectory i.

    assigned_points : ndarray, shape (N, D)

    rank_errors : ndarray, shape (N, D)

    distances : ndarray, shape (N,)

    used_k : int
        K at which a complete sparse matching was found.

    Notes
    -----
    This solves the key problem with the previous greedy matcher:
    greedy choices could consume locally attractive B25 rows early,
    leaving thousands of trajectories unmatched even though a valid
    global one-to-one assignment existed.

    Here, all assignments are solved jointly over a sparse K-nearest-
    neighbor bipartite graph.

    The final result is still a complete permutation of the original
    B25 rows, so every yearly row is used exactly once.
    """

    desired_points = np.asarray(
        desired_points,
        dtype=float,
    )

    candidate_points = np.asarray(
        candidate_points,
        dtype=float,
    )

    if desired_points.ndim != 2:
        raise ValueError(
            "desired_points must be a 2-D array."
        )

    if candidate_points.ndim != 2:
        raise ValueError(
            "candidate_points must be a 2-D array."
        )

    if desired_points.shape != candidate_points.shape:
        raise ValueError(
            "desired_points and candidate_points must have "
            f"the same shape. Got {desired_points.shape} and "
            f"{candidate_points.shape}."
        )

    if not np.isfinite(desired_points).all():
        raise ValueError(
            "Non-finite desired rank values found."
        )

    if not np.isfinite(candidate_points).all():
        raise ValueError(
            "Non-finite candidate rank values found."
        )

    n_samples = desired_points.shape[0]

    tree = cKDTree(
        candidate_points
    )

    last_error = None

    for k in k_sequence:

        k_eff = min(
            int(k),
            n_samples,
        )

        print(
            f"    trying sparse full matching with K={k_eff}..."
        )

        distances, neighbors = tree.query(
            desired_points,
            k=k_eff,
            workers=-1,
        )

        if k_eff == 1:
            distances = distances[:, np.newaxis]
            neighbors = neighbors[:, np.newaxis]

        # Build sparse cost matrix:
        #
        # rows    = desired trajectories
        # columns = original B25 candidate rows
        #
        # Squared Euclidean distance is sufficient because sqrt()
        # is monotonic and the optimizer only needs relative costs.
        row_indices = np.repeat(
            np.arange(
                n_samples,
                dtype=int,
            ),
            k_eff,
        )

        col_indices = neighbors.reshape(
            -1
        ).astype(int)

        costs = (
            distances.reshape(-1) ** 2
        )

        # scipy sparse matching treats stored zeros awkwardly because
        # zero can mean "no edge". Add a numerically negligible offset.
        costs = costs + 1e-15

        graph = coo_matrix(
            (
                costs,
                (
                    row_indices,
                    col_indices,
                ),
            ),
            shape=(
                n_samples,
                n_samples,
            ),
        ).tocsr()

        try:

            matched_rows, matched_cols = (
                min_weight_full_bipartite_matching(
                    graph
                )
            )

        except ValueError as exc:

            last_error = exc

            print(
                f"      no complete matching at K={k_eff}"
            )

            continue

        # scipy should return one match per desired row.
        if len(matched_rows) != n_samples:

            print(
                f"      incomplete matching at K={k_eff}: "
                f"{len(matched_rows):,} / {n_samples:,}"
            )

            continue

        assignment = np.full(
            n_samples,
            -1,
            dtype=int,
        )

        assignment[
            matched_rows
        ] = matched_cols

        if np.any(
            assignment < 0
        ):
            print(
                f"      unresolved desired rows remain at K={k_eff}"
            )
            continue

        if len(
            np.unique(
                assignment
            )
        ) != n_samples:
            raise RuntimeError(
                "Sparse matching returned duplicate candidate rows."
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
                "Sparse matching is not a complete permutation."
            )

        assigned_points = candidate_points[
            assignment
        ]

        rank_errors = (
            assigned_points
            - desired_points
        )

        final_distances = np.sqrt(
            np.sum(
                rank_errors ** 2,
                axis=1,
            )
        )

        print(
            f"      COMPLETE one-to-one matching found at K={k_eff}"
        )

        print(
            f"      mean rank distance   = "
            f"{np.mean(final_distances):.5f}"
        )

        print(
            f"      median rank distance = "
            f"{np.median(final_distances):.5f}"
        )

        print(
            f"      95th percentile      = "
            f"{np.quantile(final_distances, 0.95):.5f}"
        )

        print(
            f"      maximum              = "
            f"{np.max(final_distances):.5f}"
        )

        return (
            assignment,
            assigned_points,
            rank_errors,
            final_distances,
            k_eff,
        )

    raise RuntimeError(
        "Could not find a complete one-to-one sparse matching "
        f"for any K in {list(k_sequence)}."
        + (
            f"\nLast scipy error: {last_error}"
            if last_error is not None
            else ""
        )
        + "\nIncrease K_SEQUENCE if necessary."
    )
