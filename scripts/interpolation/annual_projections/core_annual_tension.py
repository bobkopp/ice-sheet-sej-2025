#!/usr/bin/env python3
"""Shared CMIP6/CMIP7 annual-trajectory engine for the final core chain.

Construction
------------
Historical output through 2023 uses observed IMBIE values rebased to 1995-2014.
For EAIS, WAIS, and GrIS, the projection spline begins at the fitted Dec-2023
historical boundary value, is constrained to the observed year-end 2023 rate,
and passes through:

    recoupled emulandice 2050
    sampled/coupled SEJ 2100
    sampled/coupled SEJ 2200
    sampled/coupled SEJ 2300

The interpolation is the adopted classical spline under tension with p=0.03.
Total AIS and total ice sheet are derived draw-by-draw from component arrays.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

import core_config as cfg
from core_imbie_helpers import (
    load_imbie_common,
    evaluate_best_fit,
    observed_endpoint_rate,
    reference_shifts_from_core_2005,
)

MM_TO_CM = 0.1
ANNUAL_YEARS = np.arange(2000, 2301, dtype=int)
TABLE_YEARS = [2020, 2050, 2100, 2150, 2200, 2300]
TABLE_PERCENTILES = [5, 17, 50, 83, 95]
SCENARIO_CODES = {"LOW": "2", "INTERMEDIATE": "35", "HIGH": "5"}
SCENARIO_ORDER = ["LOW", "INTERMEDIATE", "HIGH"]
REPLACE = True

VARIABLE_LABELS = {
    "E": "EAIS",
    "W": "WAIS",
    "G": "GrIS",
    "AIS": "Total AIS",
    "SLR": "Total ice sheet",
}


def year_index(year: int) -> int:
    m = np.where(ANNUAL_YEARS == int(year))[0]
    if len(m) != 1:
        raise ValueError(f"Could not find unique annual year {year}.")
    return int(m[0])


def parse_sej_column(column: str):
    m = re.fullmatch(r"(.+?)(2|35|5)_(21|22|23)", column)
    return None if m is None else m.groups()


def neutral_sej_column_name(column: str) -> str:
    parsed = parse_sej_column(column)
    if parsed is None:
        raise ValueError(f"Could not parse SEJ column {column}")
    variable, _, year_code = parsed
    return f"{variable}_{year_code}"


def validate_selection(selection_df: pd.DataFrame):
    required = {
        "ssp", "sample_id", "selected_scenario", "reference_range_class",
        "reference_range_position_2100", "reference_range_position_2200",
        "selection_rule", "sampling_method", "horizon", "checkpoint_years",
        "idw_power", "truncation_years", "truncation_rule",
    }
    missing = required - set(selection_df.columns)
    if missing:
        raise ValueError(f"Scenario-selection file missing columns: {sorted(missing)}")
    expected_method = "range_normalized_checkpoint_euclidean_idw_p1_double_checkpoint_truncation"
    if not selection_df["sampling_method"].eq(expected_method).all():
        raise ValueError("Wrong upstream warming-selection method.")
    if not selection_df["horizon"].astype(int).eq(2300).all():
        raise ValueError("Wrong upstream warming-selection horizon.")
    cp = selection_df["checkpoint_years"].apply(lambda x: [int(v) for v in json.loads(x)])
    tr = selection_df["truncation_years"].apply(lambda x: [int(v) for v in json.loads(x)])
    if not cp.apply(lambda x: x == [2100, 2200]).all():
        raise ValueError("Wrong upstream checkpoint years.")
    if not tr.apply(lambda x: x == [2100, 2200]).all():
        raise ValueError("Wrong upstream truncation years.")
    if not selection_df["truncation_rule"].eq("outside_same_direction_at_both_2100_and_2200").all():
        raise ValueError("Wrong upstream truncation rule.")
    if not np.allclose(selection_df["idw_power"].to_numpy(float), 1.0):
        raise ValueError("Wrong upstream IDW power.")


def crn_indices_by_sample_id(selection_df: pd.DataFrame, n_sej: int) -> np.ndarray:
    """Shared FaIR member IDs, independent SEJ streams per elicited warming level.

    Identical (sample_id, selected_scenario) pairs get identical coupled SEJ row
    indices across all seven CMIP7 pathways. IDs and probabilities are NOT changed.
    """
    pathway_names = tuple(cfg.CMIP7_CODES)
    if "sample_id" not in selection_df or "ssp" not in selection_df:
        raise ValueError("CMIP7 CRN requires sample_id and ssp columns")
    ids = pd.to_numeric(selection_df["sample_id"], errors="raise").to_numpy()
    if not np.all(np.isfinite(ids)) or not np.all(ids == np.floor(ids)):
        raise ValueError("CMIP7 CRN requires integral, finite FaIR sample IDs")
    ids = ids.astype(np.int64)
    pathways = selection_df["ssp"].astype(str).str.strip().to_numpy()
    selected = selection_df["selected_scenario"].astype(str).str.strip().str.upper().to_numpy()
    if set(pathways) != set(pathway_names):
        raise ValueError(f"Unexpected CMIP7 pathways: {sorted(set(pathways))}")
    unique_sets = []
    for pathway in pathway_names:
        subset = ids[pathways == pathway]
        if len(np.unique(subset)) != len(subset):
            raise ValueError(f"CMIP7 {pathway}: duplicate FaIR sample IDs")
        unique_sets.append(set(subset.tolist()))
    if any(x != unique_sets[0] for x in unique_sets[1:]):
        raise ValueError("CRN requires identical FaIR sample-ID sets across CMIP7 pathways")
    ordered_ids = np.array(sorted(unique_sets[0]), dtype=np.int64)
    rng = np.random.default_rng(cfg.CMIP7_CRN_SEED)
    # Same indexed sample_id has one draw per warming stream, reused by pathway.
    streams = {scenario: rng.integers(0, n_sej, size=len(ordered_ids))
               for scenario in SCENARIO_ORDER}
    id_positions = np.searchsorted(ordered_ids, ids)
    if not np.array_equal(ordered_ids[id_positions], ids):
        raise ValueError("CMIP7 CRN sample-id alignment failure")
    result = np.full(len(selection_df), -1, dtype=np.int64)
    for scenario in SCENARIO_ORDER:
        mask = selected == scenario
        result[mask] = streams[scenario][id_positions[mask]]
    if np.any(result < 0):
        raise ValueError("Invalid CMIP7 warming scenario encountered")
    return result


def sample_coupled_rows(selection_df: pd.DataFrame, coupled_df: pd.DataFrame,
                        family: str = "CMIP6"):

    scenario_columns = {s: [] for s in SCENARIO_ORDER}
    code_to_scenario = {v: k for k, v in SCENARIO_CODES.items()}
    for c in coupled_df.columns:
        p = parse_sej_column(c)
        if p is None:
            continue
        _, code, _ = p
        scenario_columns[code_to_scenario[code]].append(c)

    neutral_by_scenario = {
        s: [neutral_sej_column_name(c) for c in cols]
        for s, cols in scenario_columns.items()
    }
    neutral_columns = neutral_by_scenario["LOW"]
    refset = set(neutral_columns)
    for s in ("INTERMEDIATE", "HIGH"):
        if set(neutral_by_scenario[s]) != refset:
            raise ValueError(f"{s} SEJ columns differ from LOW after removing warming code.")

    rng = np.random.default_rng(cfg.SAMPLE_RNG_SEED)
    n_fair = len(selection_df)
    n_sej = len(coupled_df)
    if family.upper() == "CMIP7":
        sample_indices = crn_indices_by_sample_id(selection_df, n_sej)
    elif family.upper() == "CMIP6":
        sample_indices = rng.choice(n_sej, size=n_fair, replace=REPLACE)
    else:
        raise ValueError(f"Unknown sampling family: {family}")

    sampled_sej = pd.DataFrame(np.nan, index=np.arange(n_fair), columns=neutral_columns, dtype=float)
    knot_cols = ["KNOT_E_2050_cm", "KNOT_W_2050_cm", "KNOT_G_2050_cm"]
    sampled_knot = pd.DataFrame(np.nan, index=np.arange(n_fair), columns=knot_cols, dtype=float)
    sampled_coupled_id = np.full(n_fair, -1, dtype=int)

    for scenario in SCENARIO_ORDER:
        mask = selection_df["selected_scenario"].eq(scenario).to_numpy()
        if not mask.any():
            continue
        source_rows = sample_indices[mask]
        sampled_sej.loc[mask, neutral_by_scenario[scenario]] = (
            coupled_df.iloc[source_rows][scenario_columns[scenario]].to_numpy(float)
        )
        code = SCENARIO_CODES[scenario]
        src = [f"EMU2050_E{code}_cm", f"EMU2050_W{code}_cm", f"EMU2050_G{code}_cm"]
        missing = [c for c in src if c not in coupled_df.columns]
        if missing:
            raise ValueError(f"Coupled file missing final emulandice 2050 columns: {missing}")
        sampled_knot.loc[mask, knot_cols] = coupled_df.iloc[source_rows][src].to_numpy(float)
        if "coupled_trajectory_id" in coupled_df.columns:
            sampled_coupled_id[mask] = coupled_df.iloc[source_rows]["coupled_trajectory_id"].to_numpy(int)
        else:
            sampled_coupled_id[mask] = source_rows

    if sampled_sej.isna().any().any() or sampled_knot.isna().any().any():
        raise ValueError("Sampling produced missing values.")

    combined = selection_df.reset_index(drop=True).copy()
    combined["sej2025_row_index"] = sample_indices
    combined["sej_sampling_rule"] = (
        "CMIP7_CRN_shared_FaIR_sample_id_separate_warming_streams"
        if family.upper() == "CMIP7" else "CMIP6_independent_rng_with_replacement"
    )
    combined["coupled_trajectory_id"] = sampled_coupled_id
    combined = pd.concat(
        [combined, sampled_sej.reset_index(drop=True), sampled_knot.reset_index(drop=True)],
        axis=1,
    )
    return combined


def historical_context():
    raw = load_imbie_common(cfg.PROJECT_DIR)
    x2023 = float(raw["decimal_year"].iloc[-1])
    boundary = {}
    for v in ("E", "W", "G"):
        y2023 = float(evaluate_best_fit(raw, v, np.array([x2023]))[0][0])
        r2023 = observed_endpoint_rate(raw, v)
        boundary[v] = (x2023, y2023, r2023)
    shifts = reference_shifts_from_core_2005(raw)
    annual_hist = (
        raw.loc[raw["calendar_year"].between(2000, 2023)]
        .groupby("calendar_year", as_index=False)
        .tail(1)
        .sort_values("calendar_year")
        .reset_index(drop=True)
    )
    return raw, annual_hist, boundary, shifts


class TensionSpline:
    """Classical interpolating spline under tension with continuous first derivative."""

    def __init__(self, x, y, p, left_slope):
        self.x = np.asarray(x, float)
        y = np.asarray(y, float)
        if y.ndim == 1:
            y = y[:, None]
            self.squeeze = True
        else:
            self.squeeze = False
        self.y = y
        self.p = float(p)
        slope = np.asarray(left_slope, float)
        if slope.ndim == 0:
            slope = np.full(y.shape[1], float(slope))
        self.left_slope = slope
        self.M = self._solve()

    def _solve(self):
        x, y, p = self.x, self.y, self.p
        h = np.diff(x)
        delta = np.diff(y, axis=0) / h[:, None]
        n = len(x)
        A = np.zeros((n - 1, n - 1), float)
        rhs = np.empty((n - 1, y.shape[1]), float)

        h0 = h[0]
        s0 = np.sinh(p * h0)
        c0 = np.cosh(p * h0) / s0
        A[0, 0] = c0 / p - 1.0 / (p**2 * h0)
        if n - 1 > 1:
            A[0, 1] = 1.0 / (p**2 * h0) - 1.0 / (p * s0)
        rhs[0] = delta[0] - self.left_slope

        for i in range(1, n - 1):
            hl, hr = h[i - 1], h[i]
            sl, sr = np.sinh(p * hl), np.sinh(p * hr)
            cl, cr = np.cosh(p * hl) / sl, np.cosh(p * hr) / sr
            A[i, i - 1] = 1.0 / (p**2 * hl) - 1.0 / (p * sl)
            A[i, i] = cl / p - 1.0 / (p**2 * hl) + cr / p - 1.0 / (p**2 * hr)
            if i + 1 < n - 1:
                A[i, i + 1] = 1.0 / (p**2 * hr) - 1.0 / (p * sr)
            rhs[i] = delta[i] - delta[i - 1]

        M = np.zeros_like(y)
        M[:-1] = np.linalg.solve(A, rhs)
        return M

    def _eval(self, xx, derivative=False):
        xx = np.atleast_1d(np.asarray(xx, float))
        out = np.empty((len(xx), self.y.shape[1]), float)
        idx = np.searchsorted(self.x, xx, side="right") - 1
        idx = np.clip(idx, 0, len(self.x) - 2)
        p = self.p
        for k in np.unique(idx):
            mask = idx == k
            u = xx[mask] - self.x[k]
            h = self.x[k + 1] - self.x[k]
            y0, y1 = self.y[k], self.y[k + 1]
            M0, M1 = self.M[k], self.M[k + 1]
            sh = np.sinh(p * h)
            if derivative:
                term0 = -np.cosh(p * (h - u))[:, None] / (p * sh)
                term1 = np.cosh(p * u)[:, None] / (p * sh)
                linear = (y1 - y0) / h + (M0 - M1) / (p**2 * h)
                out[mask] = term0 * M0 + term1 * M1 + linear
            else:
                lh = np.sinh(p * (h - u))[:, None] / (p**2 * sh)
                rh = np.sinh(p * u)[:, None] / (p**2 * sh)
                ll = ((h - u) / h)[:, None]
                rl = (u / h)[:, None]
                out[mask] = lh * M0 + rh * M1 + ll * (y0 - M0 / p**2) + rl * (y1 - M1 / p**2)
        return out[:, 0] if self.squeeze else out

    def value(self, xx):
        return self._eval(xx, derivative=False)

    def rate(self, xx):
        return self._eval(xx, derivative=True)


def build_component_trajectory(variable, combined, annual_hist, boundary, shifts):
    n = len(combined)
    full = np.full((n, len(ANNUAL_YEARS)), np.nan, dtype=np.float32)

    for row in annual_hist.itertuples():
        full[:, year_index(int(row.calendar_year))] = np.float32(getattr(row, f"{variable}_cm"))

    x2023, y2023, rate2023 = boundary[variable]
    knot2050 = combined[f"KNOT_{variable}_2050_cm"].to_numpy(float)
    endpoint2100 = combined[f"{variable}_21"].to_numpy(float) * MM_TO_CM + shifts[variable]
    endpoint2200 = combined[f"{variable}_22"].to_numpy(float) * MM_TO_CM + shifts[variable]
    endpoint2300 = combined[f"{variable}_23"].to_numpy(float) * MM_TO_CM + shifts[variable]

    xk = np.array([x2023, 2050.0, 2100.0, 2200.0, 2300.0], float)
    yk = np.vstack([
        np.full(n, y2023), knot2050, endpoint2100, endpoint2200, endpoint2300
    ])
    spline = TensionSpline(xk, yk, p=cfg.TENSION_P, left_slope=rate2023)
    future = ANNUAL_YEARS >= 2024
    years = ANNUAL_YEARS[future].astype(float)
    full[:, future] = spline.value(years).T.astype(np.float32)

    validation = []
    for year, expected in {2050: knot2050, 2100: endpoint2100, 2200: endpoint2200, 2300: endpoint2300}.items():
        actual = full[:, year_index(year)].astype(float)
        diff = np.abs(actual - expected)
        validation.append({
            "variable": variable, "year": year,
            "max_abs_difference_cm": float(diff.max()),
            "mean_abs_difference_cm": float(diff.mean()),
            "allclose": bool(np.allclose(actual, expected, rtol=1e-6, atol=1e-5)),
        })

    # Derivative continuity at hard knots.
    eps = 1e-5
    for year in (2050, 2100, 2200):
        gap = np.abs(spline.rate([year + eps])[0] - spline.rate([year - eps])[0])
        validation.append({
            "variable": variable, "year": f"rate_{year}",
            "max_abs_difference_cm": float(gap.max()),
            "mean_abs_difference_cm": float(gap.mean()),
            "allclose": bool(gap.max() < 1e-3),
        })
    return full, validation


def save_netcdf(path: Path, combined: pd.DataFrame, arrays: dict[str, np.ndarray], family: str):
    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("samples", len(combined))
        nc.createDimension("years", len(ANNUAL_YEARS))
        yv = nc.createVariable("years", "i4", ("years",))
        yv[:] = ANNUAL_YEARS
        rv = nc.createVariable("sej2025_row_index", "i8", ("samples",))
        rv[:] = combined["sej2025_row_index"].to_numpy(np.int64)
        kv = nc.createVariable("intermediate_knot_year", "f4")
        kv.assignValue(np.float32(cfg.INTERMEDIATE_KNOT_YEAR))
        kv.units = "year"
        for v, arr in arrays.items():
            ov = nc.createVariable(v, "f4", ("samples", "years"), zlib=True)
            ov[:, :] = arr
            ov.units = "cm"
            ov.long_name = VARIABLE_LABELS[v]
        nc.description = (
            f"{family} SEJ2025 ice-sheet trajectories: observed IMBIE through 2023; "
            "recoupled emulandice workflow-1e 2050 EAIS/WAIS/GrIS knot; coupled SEJ "
            "2100/2200/2300 endpoints; classical spline under tension p=0.03."
        )
        nc.reference_period = f"{cfg.REFERENCE_START_YEAR}-{cfg.REFERENCE_END_YEAR}"
        nc.infill_rule = (
            "Observed through 2023; projection spline starts from fitted Dec-2023 level with "
            "observed endpoint rate, then passes through 2050/2100/2200/2300; tension p=0.03."
        )
        nc.sej_sampling_rule = str(combined["sej_sampling_rule"].iloc[0])
        if family.upper() == "CMIP7":
            nc.cmip7_crn_seed = int(cfg.CMIP7_CRN_SEED)
        nc.total_ais_rule = "EAIS + WAIS draw-by-draw"
        nc.total_ice_sheet_rule = "EAIS + WAIS + GrIS draw-by-draw"
        nc.scenario_selection = (
            "Range-normalized cumulative-warming distance at 2100 and 2200 with IDW p=1; "
            "LOW/HIGH forced only when outside in the same direction at both checkpoints."
        )


def make_projection_table(arrays, combined, pathways, family):
    comps = {
        "EAIS": arrays["E"],
        "WAIS": arrays["W"],
        "Total AIS": arrays["AIS"],
        "GrIS": arrays["G"],
        "Total ice sheet": arrays["SLR"],
    }
    rows = []
    for cname, all_values in comps.items():
        for pathway in pathways:
            idx = np.where(combined["ssp"].eq(pathway).to_numpy())[0]
            if len(idx) == 0:
                continue
            values = all_values[idx]
            for year in TABLE_YEARS:
                q = np.percentile(values[:, year_index(year)], TABLE_PERCENTILES)
                if year <= 2023:
                    source = "IMBIE observation"
                elif year == 2050:
                    source = "recoupled emulandice workflow-1e 2050 knot"
                elif year in (2100, 2200, 2300):
                    source = "sampled temporally coupled SEJ endpoint"
                else:
                    source = "tension-spline interpolation"
                rows.append({
                    "component": cname,
                    "ssp": pathway,
                    "year": year,
                    "p05_cm": q[0], "p17_cm": q[1], "p50_cm": q[2],
                    "p83_cm": q[3], "p95_cm": q[4],
                    "source": source,
                    "reference_period": f"{cfg.REFERENCE_START_YEAR}-{cfg.REFERENCE_END_YEAR}",
                    "climate_family": family,
                })
    return pd.DataFrame(rows)


def run_family(family: str):
    family = family.upper()
    if family == "CMIP6":
        selection_file = cfg.CMIP6_SELECTION_FILE
        outdir = cfg.CMIP6_ANNUAL_OUTPUT_DIR
        pathways = cfg.CMIP6_SSPS
        table_stub = "SSP"
    elif family == "CMIP7":
        selection_file = cfg.CMIP7_SELECTION_FILE
        outdir = cfg.CMIP7_ANNUAL_OUTPUT_DIR
        pathways = cfg.CMIP7_CODES
        table_stub = "CMIP7"
    else:
        raise ValueError(f"Unknown family {family}")

    outdir.mkdir(parents=True, exist_ok=True)
    tabledir = outdir / "tables"
    tabledir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print(f"{family}: EMULANDICE 2050 + TENSION p={cfg.TENSION_P:g} ANNUAL TRAJECTORIES")
    print("=" * 92)

    selection = pd.read_csv(selection_file)
    validate_selection(selection)
    coupled = pd.read_csv(cfg.COUPLED_SEJ_FILE)
    combined = sample_coupled_rows(selection, coupled, family=family)
    combined.to_csv(outdir / "sampled_sej2025_realizations.csv", index=False)

    raw, annual_hist, boundary, shifts = historical_context()
    raw.to_csv(outdir / "imbie_observed_slr_1979_2023_monthly_rebased_to_1995_2014.csv", index=False)

    arrays = {}
    validation = []
    for v in ("E", "W", "G"):
        print(f"Building {VARIABLE_LABELS[v]}...")
        arrays[v], rows = build_component_trajectory(v, combined, annual_hist, boundary, shifts)
        validation.extend(rows)
    arrays["AIS"] = (arrays["E"] + arrays["W"]).astype(np.float32)
    arrays["SLR"] = (arrays["AIS"] + arrays["G"]).astype(np.float32)

    valdf = pd.DataFrame(validation)
    valdf.to_csv(outdir / "trajectory_knot_and_rate_continuity.csv", index=False)
    if not valdf["allclose"].all():
        raise RuntimeError("Trajectory knot/rate-continuity validation failed.")

    ncfile = outdir / "annual_ice_sheet_trajectories_2000_2300.nc"
    save_netcdf(ncfile, combined, arrays, family)

    table = make_projection_table(arrays, combined, pathways, family)
    table_file = tabledir / f"all_ice_sheet_components_{table_stub}_projection_table_2020_2050_2100_2150_2200_2300.csv"
    table.to_csv(table_file, index=False)

    pd.DataFrame([{
        "climate_family": family,
        "scenario_selection_method": "range_normalized_checkpoint_euclidean_idw_p1_double_checkpoint_truncation",
        "checkpoint_years": "[2100, 2200]",
        "2050_knot": "recoupled emulandice workflow-1e EAIS/WAIS/GrIS",
        "temporal_coupling": "Stage 02I-B: 3D balanced Hungarian matching (block 5000) + 30 rounds guarded temporal refinement",
        "projection_infill": f"classical spline under tension p={cfg.TENSION_P}",
        "projection_knots": "fitted Dec-2023 boundary; emulandice 2050; SEJ 2100; SEJ 2200; SEJ 2300",
        "rate_2023_constrained": True,
        "total_ais": "EAIS+WAIS draw-by-draw",
        "total_ice_sheet": "EAIS+WAIS+GrIS draw-by-draw",
        "sample_rng_seed": cfg.SAMPLE_RNG_SEED,
    }]).to_csv(outdir / "method_metadata.csv", index=False)

    print("Saved:")
    print(" ", ncfile)
    print(" ", table_file)
    print("Done.")
