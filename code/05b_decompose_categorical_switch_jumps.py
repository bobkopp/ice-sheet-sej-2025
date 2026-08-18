#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05b_decompose_categorical_switch_jumps.py

Decompose category-switching jumps into:

    actual change
      = within-category temporal evolution
      + destination-year category-switch effect

For example:

    MODERATE at 2200 -> HIGH at 2300

with temporal percentiles u_2200 and u_2300:

    actual
      = HIGH_2300(u_2300) - MOD_2200(u_2200)

    temporal component
      = MOD_2300(u_2300) - MOD_2200(u_2200)

    category-switch component
      = HIGH_2300(u_2300) - MOD_2300(u_2300)

Thus the destination-year percentile is held fixed when comparing
warming categories.

@author: elizabeth
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from B25data_processing import project_dir

PROJECT_DIR = Path(project_dir)

TEMPORAL_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "01_temporal_coupling"
)

SELECTION_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "02_climate_space_categorical_selection"
)

SELECTION_FILE = (
    SELECTION_DIR
    / "b25_categorical_climate_space_selection_metadata.csv"
)

ROLE_FILES = {
    "LOW": TEMPORAL_DIR / "b25_b19style_low_2.0C_dp21_rcp26_coupled_samples.csv",
    "MODERATE": TEMPORAL_DIR / "b25_b19style_moderate_3.5C_dp21_rcp45_coupled_samples.csv",
    "HIGH": TEMPORAL_DIR / "b25_b19style_high_5.0C_dp21_rcp85_coupled_samples.csv",
}

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "b19style_baseline"
    / "05_categorical_transition_diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROLE_CODE = {
    "LOW": "2",
    "MODERATE": "35",
    "HIGH": "5",
}

YEAR_CODE = {
    2100: "21",
    2200: "22",
    2300: "23",
}

COMPONENT_PREFIX = {
    "EAIS": "E",
    "WAIS": "W",
    "GrIS": "G",
}


def load_inputs():

    if not SELECTION_FILE.exists():
        raise FileNotFoundError(
            f"Selection metadata not found:\n{SELECTION_FILE}"
        )

    selection = pd.read_csv(SELECTION_FILE)

    required = {
        "fair_sample_id",
        "scenario_label",
        "year",
        "selected_library",
        "selected_temporal_percentile",
    }

    missing = required - set(selection.columns)

    if missing:
        raise KeyError(
            "Selection metadata is missing:\n"
            + "\n".join(sorted(missing))
        )

    libraries = {}

    for role, path in ROLE_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{role} temporal library not found:\n{path}"
            )

        libraries[role] = pd.read_csv(path)

    return selection, libraries


def total_ais(df, role, year):

    code = ROLE_CODE[role]
    yc = YEAR_CODE[year]

    return (
        df[f"E{code}_{yc}"].to_numpy(dtype=float)
        + df[f"W{code}_{yc}"].to_numpy(dtype=float)
    )


def build_rank_orderings(libraries):

    orderings = {}

    for role, df in libraries.items():

        for year in YEAR_CODE:

            values = total_ais(df, role, year)

            orderings[(role, year)] = np.lexsort(
                (
                    np.arange(len(df)),
                    values,
                )
            )

    return orderings


def rows_at_percentile(library, ordering, percentile):

    n = len(library)

    position = np.floor(
        np.asarray(percentile, dtype=float) * n
    ).astype(int)

    position = np.clip(position, 0, n - 1)

    return library.iloc[
        ordering[position]
    ].reset_index(drop=True)


def component_values(rows, role, year):

    code = ROLE_CODE[role]
    yc = YEAR_CODE[year]

    out = {}

    for component, prefix in COMPONENT_PREFIX.items():

        out[component] = rows[
            f"{prefix}{code}_{yc}"
        ].to_numpy(dtype=float)

    out["Total AIS"] = (
        out["EAIS"]
        + out["WAIS"]
    )

    out["Total ice-sheet SLR"] = (
        out["Total AIS"]
        + out["GrIS"]
    )

    return out


def decompose_switches(selection, libraries, orderings):

    frames = []

    for y1, y2 in [
        (2100, 2200),
        (2200, 2300),
    ]:

        s1 = (
            selection.loc[
                selection["year"] == y1
            ]
            .sort_values("fair_sample_id")
            .reset_index(drop=True)
        )

        s2 = (
            selection.loc[
                selection["year"] == y2
            ]
            .sort_values("fair_sample_id")
            .reset_index(drop=True)
        )

        switched = (
            s1["selected_library"].to_numpy()
            != s2["selected_library"].to_numpy()
        )

        if not np.any(switched):
            continue

        pairs = pd.DataFrame(
            {
                "fair_sample_id": s1.loc[
                    switched,
                    "fair_sample_id",
                ].to_numpy(dtype=int),
                "scenario_label": s1.loc[
                    switched,
                    "scenario_label",
                ].to_numpy(),
                "from_role": s1.loc[
                    switched,
                    "selected_library",
                ].to_numpy(),
                "to_role": s2.loc[
                    switched,
                    "selected_library",
                ].to_numpy(),
                "u_from": s1.loc[
                    switched,
                    "selected_temporal_percentile",
                ].to_numpy(dtype=float),
                "u_to": s2.loc[
                    switched,
                    "selected_temporal_percentile",
                ].to_numpy(dtype=float),
            }
        )

        for (from_role, to_role), group in pairs.groupby(
            ["from_role", "to_role"],
            sort=False,
        ):

            u1 = group["u_from"].to_numpy(dtype=float)
            u2 = group["u_to"].to_numpy(dtype=float)

            from_y1_rows = rows_at_percentile(
                libraries[from_role],
                orderings[(from_role, y1)],
                u1,
            )

            from_y2_rows = rows_at_percentile(
                libraries[from_role],
                orderings[(from_role, y2)],
                u2,
            )

            to_y2_rows = rows_at_percentile(
                libraries[to_role],
                orderings[(to_role, y2)],
                u2,
            )

            from_y1 = component_values(
                from_y1_rows,
                from_role,
                y1,
            )

            from_y2 = component_values(
                from_y2_rows,
                from_role,
                y2,
            )

            to_y2 = component_values(
                to_y2_rows,
                to_role,
                y2,
            )

            for component in [
                "EAIS",
                "WAIS",
                "Total AIS",
                "GrIS",
                "Total ice-sheet SLR",
            ]:

                temporal_part = (
                    from_y2[component]
                    - from_y1[component]
                )

                category_part = (
                    to_y2[component]
                    - from_y2[component]
                )

                actual_change = (
                    to_y2[component]
                    - from_y1[component]
                )

                residual = (
                    actual_change
                    - temporal_part
                    - category_part
                )

                denom = np.abs(actual_change)

                category_fraction = np.where(
                    denom > 1e-12,
                    np.abs(category_part) / denom,
                    np.nan,
                )

                frames.append(
                    pd.DataFrame(
                        {
                            "fair_sample_id": group[
                                "fair_sample_id"
                            ].to_numpy(dtype=int),
                            "scenario_label": group[
                                "scenario_label"
                            ].to_numpy(),
                            "from_year": y1,
                            "to_year": y2,
                            "transition": (
                                from_role
                                + " -> "
                                + to_role
                            ),
                            "component": component,
                            "u_from": u1,
                            "u_to": u2,
                            "value_from_mm": from_y1[component],
                            "counterfactual_same_category_to_mm": from_y2[component],
                            "actual_to_category_mm": to_y2[component],
                            "actual_change_mm": actual_change,
                            "temporal_component_mm": temporal_part,
                            "category_switch_component_mm": category_part,
                            "decomposition_residual_mm": residual,
                            "absolute_actual_change_mm": np.abs(actual_change),
                            "absolute_temporal_component_mm": np.abs(temporal_part),
                            "absolute_category_switch_component_mm": np.abs(category_part),
                            "absolute_category_fraction_of_actual": category_fraction,
                        }
                    )
                )

    return pd.concat(frames, ignore_index=True)


def summarize(decomposition):

    return (
        decomposition.groupby(
            [
                "scenario_label",
                "from_year",
                "to_year",
                "transition",
                "component",
            ]
        )
        .agg(
            n=("fair_sample_id", "size"),
            median_abs_actual_mm=(
                "absolute_actual_change_mm",
                "median",
            ),
            p95_abs_actual_mm=(
                "absolute_actual_change_mm",
                lambda x: np.quantile(x, 0.95),
            ),
            median_abs_temporal_mm=(
                "absolute_temporal_component_mm",
                "median",
            ),
            median_abs_category_mm=(
                "absolute_category_switch_component_mm",
                "median",
            ),
            median_category_fraction=(
                "absolute_category_fraction_of_actual",
                "median",
            ),
            max_abs_residual_mm=(
                "decomposition_residual_mm",
                lambda x: np.max(np.abs(x)),
            ),
        )
        .reset_index()
    )


def main():

    print("=" * 72)
    print("DECOMPOSE CATEGORY-SWITCHING SLR JUMPS")
    print("=" * 72)

    selection, libraries = load_inputs()

    orderings = build_rank_orderings(
        libraries
    )

    decomposition = decompose_switches(
        selection,
        libraries,
        orderings,
    )

    summary = summarize(
        decomposition
    )

    largest = (
        decomposition.loc[
            decomposition["component"].isin(
                [
                    "Total AIS",
                    "Total ice-sheet SLR",
                ]
            )
        ]
        .sort_values(
            "absolute_actual_change_mm",
            ascending=False,
        )
        .head(100)
        .reset_index(drop=True)
    )

    decomposition.to_csv(
        OUTPUT_DIR
        / "category_switch_jump_decomposition_all.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "category_switch_jump_decomposition_summary.csv",
        index=False,
    )

    largest.to_csv(
        OUTPUT_DIR
        / "category_switch_jump_decomposition_largest_examples.csv",
        index=False,
    )

    print(
        "\nLargest Total AIS / total ice-sheet SLR switching changes:"
    )

    print(
        largest[
            [
                "fair_sample_id",
                "scenario_label",
                "from_year",
                "to_year",
                "transition",
                "component",
                "u_from",
                "u_to",
                "actual_change_mm",
                "temporal_component_mm",
                "category_switch_component_mm",
                "absolute_category_fraction_of_actual",
            ]
        ]
        .head(30)
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nMedian decomposition by transition/component:"
    )

    print(
        summary.loc[
            summary["component"].isin(
                [
                    "Total AIS",
                    "Total ice-sheet SLR",
                ]
            )
        ][
            [
                "scenario_label",
                "from_year",
                "to_year",
                "transition",
                "component",
                "n",
                "median_abs_actual_mm",
                "median_abs_temporal_mm",
                "median_abs_category_mm",
                "median_category_fraction",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nSaved decomposition diagnostics to:"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
