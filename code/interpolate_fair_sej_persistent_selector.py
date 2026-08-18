#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Persistent-selector FaIR–SEJ resampling.

This replaces cross-warming weighted averaging with probabilistic selection
from three independently sampled, temporally coupled B25 warming libraries.

Method:
1. Compute FaIR climate features at 2100, 2200, 2300.
2. Compute Gaussian climate-state weights for 2.0°C, 3.5°C, 5.0°C libraries.
3. Draw one persistent U(0,1) selector per FaIR trajectory and reuse it at
   all three anchor years.
4. Convert each year's weights + persistent selector into a categorical
   library choice.
5. Independently pair each FaIR trajectory with one candidate coupled B25 row
   in EACH warming library. These row identities are not assumed to correspond
   across warming.
6. At each anchor year, extract the selected warming-specific variables from
   the selected library's candidate row, preserving the intact B25 panel.
7. Save diagnostics, switching summaries, and resampled outputs.

No cross-warming arithmetic averaging is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SEJ_PROJECT_DIR = Path("/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025")
FACTS_PROJECT_DIR = Path("/Users/elizabeth/FACTS2")
sys.path.insert(0, str(SEJ_PROJECT_DIR))

import B25data_processing as b25

LIBRARIES = {
    "low": {
        "label": "2.0°C", "warming": 2.0, "warming_code": "2",
        "file": SEJ_PROJECT_DIR / "output" / "temporal_coupling" / "total_ais_common_rank"
        / "b25_2.0C_dp21_rcp26" / "b25_2.0C_dp21_rcp26_coupled_samples.csv",
    },
    "intermediate": {
        "label": "3.5°C", "warming": 3.5, "warming_code": "35",
        "file": SEJ_PROJECT_DIR / "output" / "temporal_coupling" / "total_ais_common_rank"
        / "b25_3.5C_dp21_rcp45" / "b25_3.5C_dp21_rcp45_coupled_samples.csv",
    },
    "high": {
        "label": "5.0°C", "warming": 5.0, "warming_code": "5",
        "file": SEJ_PROJECT_DIR / "output" / "temporal_coupling" / "total_ais_common_rank"
        / "b25_5.0C_dp21_rcp85" / "b25_5.0C_dp21_rcp85_coupled_samples.csv",
    },
}
LIBRARY_ORDER = ["low", "intermediate", "high"]

FAIR_FILES = {
    "ssp119": FACTS_PROJECT_DIR / "experiments" / "ssp119" / "output" / "fair-temperature" / "gsat.nc",
    "ssp126": FACTS_PROJECT_DIR / "experiments" / "ssp126" / "output" / "fair-temperature" / "gsat.nc",
    "ssp245": FACTS_PROJECT_DIR / "experiments" / "ssp245" / "output" / "fair-temperature" / "gsat.nc",
    "ssp370": FACTS_PROJECT_DIR / "experiments" / "ssp370" / "output" / "fair-temperature" / "gsat.nc",
    "ssp585": FACTS_PROJECT_DIR / "experiments" / "ssp585" / "output" / "fair-temperature" / "gsat.nc",
}
SCENARIO_LABELS = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

PROJECTION_YEARS = [2100, 2200, 2300]
CUMULATIVE_WARMING_START_YEAR = 2000
FEATURE_SCALES = {"gsat": 1.0, "cumulative_warming": 200.0}
KERNEL_BANDWIDTH = 1.0
OUTSIDE_DOMAIN_DISTANCE = 2.0
SELECTOR_SEED = 20260810
PAIRING_SEED = 20260811

REFERENCE_CLIMATE = pd.DataFrame([
    {"year": 2100, "pathway": "low", "warming": 2.0, "warming_code": "2", "gsat": 2.0, "cumulative_warming": 155.0},
    {"year": 2100, "pathway": "intermediate", "warming": 3.5, "warming_code": "35", "gsat": 3.5, "cumulative_warming": 204.5},
    {"year": 2100, "pathway": "high", "warming": 5.0, "warming_code": "5", "gsat": 5.0, "cumulative_warming": 262.0},
    {"year": 2200, "pathway": "low", "warming": 2.0, "warming_code": "2", "gsat": 2.0, "cumulative_warming": 355.0},
    {"year": 2200, "pathway": "intermediate", "warming": 3.5, "warming_code": "35", "gsat": 3.5, "cumulative_warming": 554.5},
    {"year": 2200, "pathway": "high", "warming": 5.0, "warming_code": "5", "gsat": 5.0, "cumulative_warming": 762.0},
    {"year": 2300, "pathway": "low", "warming": 2.0, "warming_code": "2", "gsat": 2.0, "cumulative_warming": 555.0},
    {"year": 2300, "pathway": "intermediate", "warming": 3.5, "warming_code": "35", "gsat": 3.5, "cumulative_warming": 904.5},
    {"year": 2300, "pathway": "high", "warming": 5.0, "warming_code": "5", "gsat": 5.0, "cumulative_warming": 1262.0},
])

YEAR_CODE_MAP = {2100: "21", 2200: "22", 2300: "23"}
OUTPUT_DIR = SEJ_PROJECT_DIR / "output" / "fair_sej_persistent_selector"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

def load_fair_ensemble():
    ensembles = []
    for scenario, path in FAIR_FILES.items():
        require_file(path)
        with xr.open_dataset(path) as ds:
            t = ds["surface_temperature"]
            if "locations" in t.dims:
                t = t.squeeze("locations", drop=True)
            t = t.sel(years=slice(CUMULATIVE_WARMING_START_YEAR, max(PROJECTION_YEARS))).load()
        n = t.sizes["samples"]
        t = t.assign_coords(
            scenario=("samples", np.repeat(scenario, n)),
            scenario_label=("samples", np.repeat(SCENARIO_LABELS[scenario], n)),
            original_sample=("samples", t["samples"].values.copy()),
        )
        ensembles.append(t)
    pooled = xr.concat(ensembles, dim="samples")
    pooled = pooled.assign_coords(samples=np.arange(pooled.sizes["samples"], dtype=int))
    return pooled

def climate_features(pooled):
    frames = []
    for year in PROJECTION_YEARS:
        sub = pooled.sel(years=slice(CUMULATIVE_WARMING_START_YEAR, year)).transpose("samples", "years")
        x = sub["years"].values.astype(float)
        y = sub.values
        cumulative = np.trapezoid(y, x=x, axis=1) if hasattr(np, "trapezoid") else np.trapz(y, x=x, axis=1)
        frames.append(pd.DataFrame({
            "fair_sample_id": pooled["samples"].values.astype(int),
            "scenario": pooled["scenario"].values,
            "scenario_label": pooled["scenario_label"].values,
            "original_sample": pooled["original_sample"].values,
            "year": year,
            "gsat": pooled.sel(years=year).values,
            "cumulative_warming": cumulative,
        }))
    return pd.concat(frames, ignore_index=True)

def climate_weights(feature_df):
    wf, df = [], []
    scale = np.array([FEATURE_SCALES["gsat"], FEATURE_SCALES["cumulative_warming"]], dtype=float)
    for year in PROJECTION_YEARS:
        f = feature_df.loc[feature_df["year"] == year].sort_values("fair_sample_id").reset_index(drop=True)
        r = REFERENCE_CLIMATE.loc[REFERENCE_CLIMATE["year"] == year].set_index("pathway").loc[LIBRARY_ORDER].reset_index()
        fm = f[["gsat", "cumulative_warming"]].to_numpy(float)
        rm = r[["gsat", "cumulative_warming"]].to_numpy(float)
        d = np.sqrt(np.sum(((fm[:, None, :] - rm[None, :, :]) / scale[None, None, :]) ** 2, axis=2))
        logw = -0.5 * (d / KERNEL_BANDWIDTH) ** 2
        logw -= np.max(logw, axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        exact = d <= 1e-12
        rows = np.any(exact, axis=1)
        if np.any(rows):
            w[rows, :] = exact[rows, :].astype(float)

        a = f.copy()
        for j, lib in enumerate(LIBRARY_ORDER):
            a[f"weight_{lib}"] = w[:, j]
        wf.append(a)

        nearest = np.argmin(d, axis=1)
        diag = f[["fair_sample_id", "scenario", "scenario_label", "original_sample", "year"]].copy()
        for j, lib in enumerate(LIBRARY_ORDER):
            diag[f"distance_{lib}"] = d[:, j]
        diag["nearest_library"] = np.array(LIBRARY_ORDER, dtype=object)[nearest]
        diag["nearest_reference_distance"] = np.min(d, axis=1)
        diag["outside_reference_domain"] = diag["nearest_reference_distance"] > OUTSIDE_DOMAIN_DISTANCE
        df.append(diag)

    weights = pd.concat(wf, ignore_index=True)
    diagnostics = pd.concat(df, ignore_index=True)
    if not np.allclose(weights[["weight_low", "weight_intermediate", "weight_high"]].sum(axis=1), 1.0):
        raise ValueError("Climate weights do not sum to one.")
    return weights, diagnostics

def persistent_choices(weights_df):
    ids = np.sort(weights_df["fair_sample_id"].unique())
    rng = np.random.default_rng(SELECTOR_SEED)
    selectors = pd.DataFrame({"fair_sample_id": ids, "persistent_selector_u": rng.random(len(ids))})
    c = weights_df.merge(selectors, on="fair_sample_id", validate="many_to_one")
    u = c["persistent_selector_u"].to_numpy(float)
    low = c["weight_low"].to_numpy(float)
    mid = (c["weight_low"] + c["weight_intermediate"]).to_numpy(float)
    c["selected_library"] = np.where(u < low, "low", np.where(u < mid, "intermediate", "high"))
    c["selected_library_label"] = c["selected_library"].map({k: v["label"] for k, v in LIBRARIES.items()})
    c["selected_warming"] = c["selected_library"].map({k: v["warming"] for k, v in LIBRARIES.items()})
    return selectors, c

def load_libraries():
    out = {}
    for lib in LIBRARY_ORDER:
        path = LIBRARIES[lib]["file"]
        require_file(path)
        data = b25.load_and_prepare_data(file_path=str(path))
        out[lib] = {"mc_df": data["mc_df"].copy(), "meta_df": data["meta_df"].copy(), "path": path}
    return out

def pair_libraries(fair_ids, libraries):
    fair_ids = np.asarray(fair_ids, dtype=int)
    master = np.random.default_rng(PAIRING_SEED)
    out = pd.DataFrame({"fair_sample_id": fair_ids})
    for lib in LIBRARY_ORDER:
        n_fair, n_sej = len(fair_ids), len(libraries[lib]["mc_df"])
        seed = int(master.integers(0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(seed)
        rows = rng.permutation(n_sej) if n_fair == n_sej else rng.choice(n_sej, size=n_fair, replace=True)
        out[f"{lib}_sej_row"] = rows.astype(int)
        out[f"{lib}_pairing_seed"] = seed
    return out

def response_metadata(meta_df):
    m = meta_df.copy()
    m["response_code"] = m["variable"].str.extract(r"^(SLR|[EWG][ADR]?)")
    cols = ["response_code", "ice_sheet_code", "ice_sheet", "process_code", "process", "year"]
    return m[cols].drop_duplicates().sort_values(["year", "response_code"]).reset_index(drop=True)

def resample_responses(libraries, choices, pairings):
    rm = response_metadata(libraries["low"]["meta_df"])
    merged = choices.merge(pairings, on="fair_sample_id", validate="many_to_one")
    frames = []

    for year in PROJECTION_YEARS:
        yc = merged.loc[merged["year"] == year].sort_values("fair_sample_id").reset_index(drop=True)
        ycode = YEAR_CODE_MAP[year]
        year_meta = rm.loc[rm["year"] == year].copy()

        for lib in LIBRARY_ORDER:
            selected = yc.loc[yc["selected_library"] == lib].copy().reset_index(drop=True)
            if selected.empty:
                continue
            mc = libraries[lib]["mc_df"]
            rows = selected[f"{lib}_sej_row"].to_numpy(int)
            wcode = LIBRARIES[lib]["warming_code"]

            for _, r in year_meta.iterrows():
                var = f"{r['response_code']}{wcode}_{ycode}"
                if var not in mc.columns:
                    raise KeyError(f"Missing variable {var} in {lib} library.")
                vals = mc.iloc[rows][var].to_numpy(float)

                frame = selected[[
                    "fair_sample_id", "scenario", "scenario_label", "original_sample", "year",
                    "gsat", "cumulative_warming", "weight_low", "weight_intermediate",
                    "weight_high", "persistent_selector_u", "selected_library",
                    "selected_library_label", "selected_warming"
                ]].copy()
                frame["sej_sample_id"] = rows
                frame["selected_variable"] = var
                frame["response_code"] = r["response_code"]
                frame["ice_sheet_code"] = r["ice_sheet_code"]
                frame["ice_sheet"] = r["ice_sheet"]
                frame["process_code"] = r["process_code"]
                frame["process"] = r["process"]
                frame["value"] = vals
                frames.append(frame)

    return pd.concat(frames, ignore_index=True)

def add_total_ais(df):
    total = df.loc[df["process_code"] == "TOTAL"].copy()
    idx = [
        "fair_sample_id","scenario","scenario_label","original_sample","year","gsat","cumulative_warming",
        "weight_low","weight_intermediate","weight_high","persistent_selector_u","selected_library",
        "selected_library_label","selected_warming","sej_sample_id"
    ]
    ais = total.loc[total["ice_sheet_code"].isin(["E","W"])].pivot(
        index=idx, columns="ice_sheet_code", values="value"
    ).dropna(subset=["E","W"]).reset_index()
    ais["value"] = ais["E"] + ais["W"]
    ais["selected_variable"] = pd.NA
    ais["response_code"] = "AIS"
    ais["ice_sheet_code"] = "AIS"
    ais["ice_sheet"] = "Total AIS"
    ais["process_code"] = "TOTAL"
    ais["process"] = "Total"
    for c in total.columns:
        if c not in ais.columns:
            ais[c] = pd.NA
    return pd.concat([total, ais[total.columns]], ignore_index=True)

def add_runoff_minus_accumulation(df):
    process = df.loc[df["process_code"].isin(["A","D","R"])].copy()
    idx = [
        "fair_sample_id","scenario","scenario_label","original_sample","year","gsat","cumulative_warming",
        "weight_low","weight_intermediate","weight_high","persistent_selector_u","selected_library",
        "selected_library_label","selected_warming","sej_sample_id","ice_sheet_code","ice_sheet"
    ]
    net = process.loc[process["process_code"].isin(["A","R"])].pivot(
        index=idx, columns="process_code", values="value"
    ).dropna(subset=["A","R"]).reset_index()
    net["value"] = net["R"] + net["A"]
    net["selected_variable"] = pd.NA
    net["response_code"] = net["ice_sheet_code"] + "R-A"
    net["process_code"] = "R-A"
    net["process"] = "Runoff minus accumulation"
    for c in process.columns:
        if c not in net.columns:
            net[c] = pd.NA
    return pd.concat([process, net[process.columns]], ignore_index=True)

def switching_summary(choices):
    w = choices[["fair_sample_id","scenario_label","year","selected_library"]].pivot(
        index=["fair_sample_id","scenario_label"], columns="year", values="selected_library"
    ).reset_index()
    w["switch_2100_2200"] = w[2100] != w[2200]
    w["switch_2200_2300"] = w[2200] != w[2300]
    w["switch_2100_2300"] = w[2100] != w[2300]
    w["n_switches_adjacent"] = w["switch_2100_2200"].astype(int) + w["switch_2200_2300"].astype(int)
    s = w.groupby("scenario_label", observed=True).agg(
        n_samples=("fair_sample_id","count"),
        percent_switch_2100_2200=("switch_2100_2200", lambda x: 100*x.mean()),
        percent_switch_2200_2300=("switch_2200_2300", lambda x: 100*x.mean()),
        percent_switch_2100_2300=("switch_2100_2300", lambda x: 100*x.mean()),
        mean_adjacent_switches=("n_switches_adjacent","mean"),
    ).reset_index()
    return w, s

def to_xarray(df):
    meta_cols = ["response_code","ice_sheet_code","ice_sheet","process_code","process"]
    rmeta = df[meta_cols].drop_duplicates().sort_values("response_code").reset_index(drop=True)
    codes = rmeta["response_code"].tolist()
    samples = np.sort(df["fair_sample_id"].unique())
    years = np.array(PROJECTION_YEARS, dtype=int)
    values = np.full((len(samples), len(years), len(codes)), np.nan, dtype=np.float32)
    si = df["fair_sample_id"].map({v:i for i,v in enumerate(samples)}).to_numpy(int)
    yi = df["year"].map({v:i for i,v in enumerate(years)}).to_numpy(int)
    ri = df["response_code"].map({v:i for i,v in enumerate(codes)}).to_numpy(int)
    values[si, yi, ri] = df["value"].to_numpy(np.float32)
    if np.isnan(values).any():
        raise ValueError("Missing sample-year-response values during xarray conversion.")
    ds = xr.Dataset(
        {"sea_level_contribution": (("samples","projection_year","response"), values)},
        coords={
            "samples": samples,
            "projection_year": years,
            "response": np.arange(len(codes), dtype=int),
            "response_code": ("response", rmeta["response_code"].astype(str).to_numpy()),
            "ice_sheet_code": ("response", rmeta["ice_sheet_code"].astype(str).to_numpy()),
            "ice_sheet": ("response", rmeta["ice_sheet"].astype(str).to_numpy()),
            "process_code": ("response", rmeta["process_code"].astype(str).to_numpy()),
            "process": ("response", rmeta["process"].astype(str).to_numpy()),
        }
    )
    ds["sea_level_contribution"].attrs["units"] = "mm"
    ds.attrs["method"] = "Persistent-selector weighted resampling from temporally coupled B25 warming libraries"
    ds.attrs["selector_seed"] = SELECTOR_SEED
    ds.attrs["pairing_seed"] = PAIRING_SEED
    return ds

def main():
    pooled = load_fair_ensemble()
    features = climate_features(pooled)
    weights, climate_diag = climate_weights(features)
    selectors, choices = persistent_choices(weights)
    libraries = load_libraries()
    pairings = pair_libraries(pooled["samples"].values.astype(int), libraries)
    resampled = resample_responses(libraries, choices, pairings)

    totals = add_total_ais(resampled)
    processes = add_runoff_minus_accumulation(resampled)
    switch_detail, switch_summary = switching_summary(choices)

    # Core validation.
    if resampled.duplicated(["fair_sample_id","year","response_code"]).any():
        raise ValueError("Duplicate sample-year-response records found.")
    if not np.isfinite(resampled["value"].to_numpy(float)).all():
        raise ValueError("Non-finite resampled values found.")

    validation = pd.DataFrame([
        {"check":"climate_weights_sum_to_one","passed":bool(np.allclose(weights[["weight_low","weight_intermediate","weight_high"]].sum(axis=1),1.0))},
        {"check":"one_persistent_selector_per_fair_sample","passed":bool(selectors["fair_sample_id"].is_unique)},
        {"check":"all_library_choices_valid","passed":bool(choices["selected_library"].isin(LIBRARY_ORDER).all())},
        {"check":"one_candidate_row_set_per_fair_sample","passed":bool(pairings["fair_sample_id"].is_unique)},
        {"check":"no_duplicate_sample_year_response","passed":True},
        {"check":"all_resampled_values_finite","passed":True},
    ])

    choice_freq = choices.groupby(
        ["scenario_label","year","selected_library_label"], observed=True
    ).size().rename("n").reset_index()
    choice_freq["percent"] = (
        choice_freq["n"]
        / choice_freq.groupby(["scenario_label","year"], observed=True)["n"].transform("sum")
        * 100
    )

    ds = to_xarray(resampled)

    features.to_csv(OUTPUT_DIR/"fair_climate_features.csv", index=False)
    weights.to_csv(OUTPUT_DIR/"fair_sej_climate_weights.csv", index=False)
    climate_diag.to_csv(OUTPUT_DIR/"fair_sej_climate_diagnostics.csv", index=False)
    selectors.to_csv(OUTPUT_DIR/"fair_sej_persistent_selector.csv", index=False)
    choices.to_csv(OUTPUT_DIR/"fair_sej_library_choices.csv", index=False)
    pairings.to_csv(OUTPUT_DIR/"fair_sej_library_pairings.csv", index=False)
    resampled.to_csv(OUTPUT_DIR/"fair_sej_resampled_responses.csv", index=False)
    totals.to_csv(OUTPUT_DIR/"fair_sej_resampled_totals.csv", index=False)
    processes.to_csv(OUTPUT_DIR/"fair_sej_resampled_processes.csv", index=False)
    switch_detail.to_csv(OUTPUT_DIR/"fair_sej_library_switching_detail.csv", index=False)
    switch_summary.to_csv(OUTPUT_DIR/"fair_sej_library_switching_summary.csv", index=False)
    choice_freq.to_csv(OUTPUT_DIR/"fair_sej_library_choice_frequencies.csv", index=False)
    validation.to_csv(OUTPUT_DIR/"fair_sej_validation.csv", index=False)
    ds.to_netcdf(OUTPUT_DIR/"fair_sej_resampled_responses.nc")

    print("\nLibrary choice frequencies:")
    print(choice_freq.round(2).to_string(index=False))
    print("\nLibrary switching summary:")
    print(switch_summary.round(2).to_string(index=False))
    print("\nValidation:")
    print(validation.to_string(index=False))
    print(f"\nOutputs saved to:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
