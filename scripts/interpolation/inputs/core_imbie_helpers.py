#!/usr/bin/env python3
"""IMBIE loading, rebasing, and historical boundary helpers for the final core chain.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import core_config as cfg

COMPONENTS = ("E", "W", "G")
MM_TO_CM = 0.1
CUMULATIVE_COLUMN = "Cumulative mass balance anomaly (mm)"
UNCERTAINTY_COLUMN = "Cumulative mass balance anomaly uncertainty (mm)"


def raw_files(project_dir: Path | None = None):
    data_dir = cfg.DATA_DIR if project_dir is None else Path(project_dir) / "data"
    root = data_dir / "historical_observations" / "imbie_2023"
    return {
        "E": root / "imbie3_east_antarctica_mm_partitioned.csv",
        "W": root / "imbie3_west_antarctica_mm_partitioned.csv",
        "G": root / "imbie3_greenland_mm_partitioned.csv",
    }


def decimal_year(dates) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    starts = pd.to_datetime(dates.year.astype(str) + "-01-01")
    ends = pd.to_datetime((dates.year + 1).astype(str) + "-01-01")
    return np.asarray(
        dates.year.astype(float)
        + (dates - starts).days.astype(float) / (ends - starts).days.astype(float),
        dtype=float,
    )


def _find_header(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith("Date,"):
                return i
    raise ValueError(f"Could not locate CSV header beginning with 'Date,' in {path}")


def read_component(variable: str, project_dir: Path | None = None) -> pd.DataFrame:
    path = raw_files(project_dir)[variable]
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, skiprows=_find_header(path))
    df.columns = [str(c).strip() for c in df.columns]
    needed = ["Date", CUMULATIVE_COLUMN, UNCERTAINTY_COLUMN]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    out = df[needed].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    out = out.loc[
        out["Date"].dt.year.between(cfg.FIT_START_YEAR, cfg.LAST_OBS_YEAR)
    ].reset_index(drop=True)
    out[f"{variable}_unrebased_cm"] = (
        -pd.to_numeric(out[CUMULATIVE_COLUMN], errors="raise") * MM_TO_CM
    )
    out[f"{variable}_sigma_cm"] = (
        pd.to_numeric(out[UNCERTAINTY_COLUMN], errors="raise") * MM_TO_CM
    )
    return out[["Date", f"{variable}_unrebased_cm", f"{variable}_sigma_cm"]]


def load_imbie_common(project_dir: Path | None = None) -> pd.DataFrame:
    frames = [read_component(v, project_dir) for v in COMPONENTS]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Date", how="inner", validate="one_to_one")

    expected_start = pd.Timestamp(f"{cfg.FIT_START_YEAR}-01-01")
    expected_end = pd.Timestamp(f"{cfg.LAST_OBS_YEAR}-12-01")
    if merged.empty:
        raise ValueError("No common E/W/G IMBIE record.")
    if merged["Date"].iloc[0] != expected_start or merged["Date"].iloc[-1] != expected_end:
        raise ValueError(
            "Common E/W/G record does not span Jan-1979 through Dec-2023. "
            f"Found {merged['Date'].min()} to {merged['Date'].max()}."
        )

    ref = merged["Date"].dt.year.between(cfg.REFERENCE_START_YEAR, cfg.REFERENCE_END_YEAR)
    for v in COMPONENTS:
        baseline = float(merged.loc[ref, f"{v}_unrebased_cm"].mean())
        merged[f"{v}_cm"] = merged[f"{v}_unrebased_cm"] - baseline

    merged["AIS_cm"] = merged["E_cm"] + merged["W_cm"]
    merged["SLR_cm"] = merged["AIS_cm"] + merged["G_cm"]
    merged["decimal_year"] = decimal_year(merged["Date"])
    merged["calendar_year"] = merged["Date"].dt.year
    return merged


def annual_means(raw: pd.DataFrame) -> pd.DataFrame:
    out = (
        raw.groupby("calendar_year", as_index=False)
        .agg(
            decimal_year=("decimal_year", "mean"),
            E_cm=("E_cm", "mean"),
            W_cm=("W_cm", "mean"),
            G_cm=("G_cm", "mean"),
        )
        .sort_values("calendar_year")
        .reset_index(drop=True)
    )
    out["AIS_cm"] = out["E_cm"] + out["W_cm"]
    out["SLR_cm"] = out["AIS_cm"] + out["G_cm"]
    return out


def four_year_means(raw: pd.DataFrame) -> pd.DataFrame:
    annual = annual_means(raw)
    a = annual.loc[annual["calendar_year"].between(cfg.FIT_START_YEAR, cfg.FIT_END_YEAR)].copy()
    a["block"] = (a["calendar_year"] - cfg.FIT_START_YEAR) // 4
    counts = a.groupby("block")["calendar_year"].transform("count")
    a = a.loc[counts.eq(4)].copy()
    out = (
        a.groupby("block", as_index=False)
        .agg(
            decimal_year=("decimal_year", "mean"),
            E_cm=("E_cm", "mean"),
            W_cm=("W_cm", "mean"),
            G_cm=("G_cm", "mean"),
        )
        .sort_values("block")
        .reset_index(drop=True)
    )
    out["AIS_cm"] = out["E_cm"] + out["W_cm"]
    out["SLR_cm"] = out["AIS_cm"] + out["G_cm"]
    return out


def _design_matrix(times, anchor_decimal):
    dt = np.asarray(times, float) - float(anchor_decimal)
    return np.column_stack([np.ones_like(dt), dt, 0.5 * dt**2])


def fit_best_quadratic(raw: pd.DataFrame):
    four = four_year_means(raw)
    last_decimal = float(raw["decimal_year"].iloc[-1])
    X = _design_matrix(four["decimal_year"].to_numpy(float), last_decimal)
    Y = four[["E_cm", "W_cm", "G_cm"]].to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return beta, last_decimal


def evaluate_best_fit(raw: pd.DataFrame, variable: str, x_decimal):
    beta, last_decimal = fit_best_quadratic(raw)
    dt = np.asarray(x_decimal, float) - last_decimal
    if variable in COMPONENTS:
        b = beta[:, COMPONENTS.index(variable)]
    elif variable == "AIS":
        b = beta[:, 0] + beta[:, 1]
    elif variable == "SLR":
        b = beta.sum(axis=1)
    else:
        raise ValueError(variable)
    value = b[0] + b[1] * dt + 0.5 * b[2] * dt**2
    slope = b[1] + b[2] * dt
    return np.asarray(value, float), np.asarray(slope, float)


def observed_endpoint_rate(raw: pd.DataFrame, variable: str) -> float:
    annual = (
        raw.groupby("calendar_year", as_index=False)
        .tail(1)
        .sort_values("calendar_year")
        .reset_index(drop=True)
    )
    x = annual["decimal_year"].to_numpy(float)
    y = annual[f"{variable}_cm"].to_numpy(float)
    return float(np.gradient(y, x, edge_order=2)[-1])


def reference_shifts_from_core_2005(raw: pd.DataFrame) -> dict[str, float]:
    """Additive shifts converting SEJ values referenced to 2005 to 1995-2014."""
    mask = raw["Date"].dt.year.eq(2005)
    if int(mask.sum()) != 12:
        raise ValueError("Expected 12 monthly observations in 2005.")
    out = {v: float(raw.loc[mask, f"{v}_cm"].mean()) for v in COMPONENTS}
    out["AIS"] = out["E"] + out["W"]
    out["SLR"] = out["AIS"] + out["G"]
    return out
