#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Out-of-core generation of FaIR–SEJ sea-level contribution time series.

This version avoids constructing a ~200-million-row annual pandas
DataFrame. It scans the anchor CSV in chunks, keeps only a compact
[sample, projection_year, response] anchor array in memory, and writes
compressed NetCDF blocks one response and sample chunk at a time.

Reference:
    2005 = 0 mm relative to the AR6 1995–2014 GMSL baseline.

Temporal method:
    Piecewise linear through 2005, 2100, 2200, and 2300.
    Monotonicity is not enforced.

Outputs:
    fair_sej_timeseries.nc
    fair_sej_timeseries_validation.csv
    fair_sej_decreasing_intervals.csv
    fair_sej_timeseries_metadata.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from netCDF4 import Dataset
except ImportError as exc:
    raise ImportError(
        "Install netCDF4 with `conda install netcdf4` or "
        "`pip install netCDF4`."
    ) from exc


DEFAULT_PROJECT_DIR = Path(
    "/Users/elizabeth/Documents/GitHub/ice-sheet-sej-2025"
)
INPUT_FILENAME = "fair_sej_interpolated_responses.csv"
OUTPUT_FILENAME = "fair_sej_timeseries.nc"

# Temporally coupled B25 experiments produced by
# interpolate_fair_sej_loop.py.
COUPLING_EXPERIMENTS = [
    "b25_2.0C_dp21_rcp26",
    "b25_3.5C_dp21_rcp45",
    "b25_5.0C_dp21_rcp85",
]

REFERENCE_YEAR = 2005
PROJECTION_YEARS = np.array([2100, 2200, 2300], dtype=np.int32)
ANCHOR_YEARS = np.array([2005, 2100, 2200, 2300], dtype=np.int32)

SAMPLE_COLUMNS = [
    "fair_sample_id",
    "sej_sample_id",
    "scenario",
    "scenario_label",
    "original_sample",
]
RESPONSE_COLUMNS = [
    "response_code",
    "ice_sheet_code",
    "ice_sheet",
    "process_code",
    "process",
]
READ_COLUMNS = SAMPLE_COLUMNS + ["year"] + RESPONSE_COLUMNS + ["value"]

DERIVED_RESPONSE_CODES = {
    "AIS",
    "ER-A",
    "WR-A",
    "GR-A",
    "AISR-A",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}\n\n"
            "Run interpolate_fair_sej.py first."
        )


def original_response_mask(frame: pd.DataFrame) -> pd.Series:
    return ~(
        frame["response_code"].isin(DERIVED_RESPONSE_CODES)
        | (frame["ice_sheet_code"] == "AIS")
        | (frame["process_code"] == "R-A")
    )


def make_output_years(
    start_year: int,
    end_year: int,
    year_step: int,
) -> np.ndarray:
    if year_step < 1:
        raise ValueError("year_step must be at least 1.")
    if start_year > REFERENCE_YEAR:
        raise ValueError("start_year cannot be later than 2005.")
    if end_year < 2300:
        raise ValueError("end_year must be at least 2300.")

    regular = np.arange(
        start_year,
        end_year + 1,
        year_step,
        dtype=np.int32,
    )
    years = np.unique(np.concatenate([regular, ANCHOR_YEARS]))
    return years[(years >= start_year) & (years <= end_year)]


def make_temporal_weights(output_years: np.ndarray) -> np.ndarray:
    weights = np.empty(
        (len(output_years), len(ANCHOR_YEARS)),
        dtype=np.float32,
    )
    for i in range(len(ANCHOR_YEARS)):
        basis = np.zeros(len(ANCHOR_YEARS), dtype=float)
        basis[i] = 1.0
        weights[:, i] = np.interp(
            output_years,
            ANCHOR_YEARS,
            basis,
        ).astype(np.float32)

    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("Temporal weights do not sum to one.")
    return weights


def scan_metadata(
    input_path: Path,
    csv_chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_parts = []
    response_parts = []

    print("Pass 1/2: scanning metadata...")

    for n, chunk in enumerate(
        pd.read_csv(
            input_path,
            usecols=READ_COLUMNS,
            chunksize=csv_chunksize,
        ),
        start=1,
    ):
        chunk = chunk.loc[chunk["year"].isin(PROJECTION_YEARS)]
        chunk = chunk.loc[original_response_mask(chunk)]

        sample_parts.append(chunk[SAMPLE_COLUMNS].drop_duplicates())
        response_parts.append(
            chunk[RESPONSE_COLUMNS].drop_duplicates()
        )
        print(f"  scanned chunk {n}", end="\r", flush=True)

    print()

    samples = (
        pd.concat(sample_parts, ignore_index=True)
        .drop_duplicates()
    )
    responses = (
        pd.concat(response_parts, ignore_index=True)
        .drop_duplicates()
    )

    sample_counts = samples.groupby(
        "fair_sample_id",
        observed=True,
    )[SAMPLE_COLUMNS[1:]].nunique()

    inconsistent_samples = sample_counts.index[
        (sample_counts > 1).any(axis=1)
    ].tolist()

    if inconsistent_samples:
        raise ValueError(
            "Inconsistent sample metadata for:\n"
            + "\n".join(map(str, inconsistent_samples[:20]))
        )

    samples = (
        samples.drop_duplicates(subset="fair_sample_id")
        .sort_values("fair_sample_id")
        .reset_index(drop=True)
    )

    response_counts = responses.groupby(
        "response_code",
        observed=True,
    )[RESPONSE_COLUMNS[1:]].nunique()

    inconsistent_responses = response_counts.index[
        (response_counts > 1).any(axis=1)
    ].tolist()

    if inconsistent_responses:
        raise ValueError(
            "Inconsistent response metadata for:\n"
            + "\n".join(inconsistent_responses)
        )

    responses = (
        responses.drop_duplicates(subset="response_code")
        .sort_values("response_code")
        .reset_index(drop=True)
    )

    print(f"  Unique FaIR samples: {len(samples):,}")
    print(f"  Original responses: {len(responses):,}")
    for code in responses["response_code"]:
        print(f"    {code}")

    return samples, responses


def load_anchor_array(
    input_path: Path,
    csv_chunksize: int,
    samples: pd.DataFrame,
    responses: pd.DataFrame,
) -> np.ndarray:
    anchors = np.full(
        (len(samples), len(PROJECTION_YEARS), len(responses)),
        np.nan,
        dtype=np.float32,
    )
    filled = np.zeros(anchors.shape, dtype=bool)

    sample_index = {
        value: i
        for i, value in enumerate(samples["fair_sample_id"])
    }
    response_index = {
        value: i
        for i, value in enumerate(responses["response_code"])
    }
    year_index = {
        int(value): i
        for i, value in enumerate(PROJECTION_YEARS)
    }

    print("Pass 2/2: loading compact anchor array...")

    for n, chunk in enumerate(
        pd.read_csv(
            input_path,
            usecols=[
                "fair_sample_id",
                "year",
                "response_code",
                "ice_sheet_code",
                "process_code",
                "value",
            ],
            chunksize=csv_chunksize,
        ),
        start=1,
    ):
        chunk = chunk.loc[chunk["year"].isin(PROJECTION_YEARS)]
        chunk = chunk.loc[original_response_mask(chunk)]

        if chunk.empty:
            continue

        si = chunk["fair_sample_id"].map(
            sample_index
        ).to_numpy(dtype=np.int64)
        yi = chunk["year"].astype(int).map(
            year_index
        ).to_numpy(dtype=np.int64)
        ri = chunk["response_code"].map(
            response_index
        ).to_numpy(dtype=np.int64)

        coordinates = pd.DataFrame(
            {"sample": si, "year": yi, "response": ri}
        )
        if coordinates.duplicated().any():
            raise ValueError(
                "Duplicate sample-year-response records within a chunk."
            )
        if filled[si, yi, ri].any():
            raise ValueError(
                "Duplicate sample-year-response records across chunks."
            )

        values = pd.to_numeric(
            chunk["value"],
            errors="raise",
        ).to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError("Anchor values contain non-finite values.")

        anchors[si, yi, ri] = values
        filled[si, yi, ri] = True
        print(f"  loaded chunk {n}", end="\r", flush=True)

    print()

    if not filled.all():
        missing = np.argwhere(~filled)[:20]
        examples = []
        for s, y, r in missing:
            examples.append(
                {
                    "fair_sample_id": samples.iloc[s][
                        "fair_sample_id"
                    ],
                    "year": int(PROJECTION_YEARS[y]),
                    "response_code": responses.iloc[r][
                        "response_code"
                    ],
                }
            )
        raise ValueError(
            f"Missing anchor records: {(~filled).sum():,}\n"
            + json.dumps(examples, indent=2, default=str)
        )

    print(f"  Anchor array shape: {anchors.shape}")
    print(
        f"  Anchor array memory: "
        f"{anchors.nbytes / 1024**2:.1f} MiB"
    )
    return anchors


def build_response_metadata(
    original: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    codes = set(original["response_code"])
    derivations: dict[str, tuple[str, ...]] = {}
    rows: list[dict[str, Any]] = []

    if {"E", "W"}.issubset(codes):
        derivations["AIS"] = ("E", "W")
        rows.append(
            {
                "response_code": "AIS",
                "ice_sheet_code": "AIS",
                "ice_sheet": "Total AIS",
                "process_code": "TOTAL",
                "process": "Total",
                "is_derived": 1,
            }
        )

    for ice_code, ice_name in [
        ("E", "EAIS"),
        ("W", "WAIS"),
        ("G", "GrIS"),
    ]:
        accumulation = f"{ice_code}A"
        runoff = f"{ice_code}R"
        derived_code = f"{ice_code}R-A"

        if {accumulation, runoff}.issubset(codes):
            derivations[derived_code] = (
                accumulation,
                runoff,
            )
            rows.append(
                {
                    "response_code": derived_code,
                    "ice_sheet_code": ice_code,
                    "ice_sheet": ice_name,
                    "process_code": "R-A",
                    "process": "Runoff minus accumulation",
                    "is_derived": 1,
                }
            )

    original = original.copy()
    original["is_derived"] = 0

    all_metadata = pd.concat(
        [original, pd.DataFrame(rows)],
        ignore_index=True,
    )
    return all_metadata, derivations


def strings(values: pd.Series) -> np.ndarray:
    return values.fillna("").astype(str).to_numpy(dtype=object)


def create_netcdf(
    path: Path,
    samples: pd.DataFrame,
    responses: pd.DataFrame,
    years: np.ndarray,
    compression_level: int,
    write_sample_chunksize: int,
) -> Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    ds = Dataset(path, "w", format="NETCDF4")
    ds.createDimension("samples", len(samples))
    ds.createDimension("year", len(years))
    ds.createDimension("response", len(responses))

    ds.createVariable("samples", "i8", ("samples",))[:] = np.arange(
        len(samples), dtype=np.int64
    )
    ds.createVariable("year", "i4", ("year",))[:] = years
    ds.createVariable("response", "i4", ("response",))[:] = np.arange(
        len(responses), dtype=np.int32
    )

    for column in [
        "fair_sample_id",
        "sej_sample_id",
        "original_sample",
    ]:
        numeric = pd.to_numeric(samples[column], errors="coerce")
        if numeric.notna().all():
            ds.createVariable(column, "i8", ("samples",))[:] = (
                numeric.to_numpy(dtype=np.int64)
            )
        else:
            ds.createVariable(column, str, ("samples",))[:] = strings(
                samples[column]
            )

    for column in ["scenario", "scenario_label"]:
        ds.createVariable(column, str, ("samples",))[:] = strings(
            samples[column]
        )

    for column in RESPONSE_COLUMNS:
        ds.createVariable(column, str, ("response",))[:] = strings(
            responses[column]
        )

    ds.createVariable("is_derived", "i1", ("response",))[:] = (
        responses["is_derived"].to_numpy(dtype=np.int8)
    )

    variable = ds.createVariable(
        "sea_level_contribution",
        "f4",
        ("samples", "year", "response"),
        zlib=True,
        complevel=compression_level,
        shuffle=True,
        chunksizes=(
            min(write_sample_chunksize, len(samples)),
            min(len(years), 296),
            1,
        ),
        fill_value=np.float32(np.nan),
    )
    variable.units = "mm"
    variable.reference_period = "AR6 1995-2014 GMSL baseline"
    variable.reference_year = REFERENCE_YEAR
    variable.temporal_interpolation = (
        "Piecewise linear through 2005, 2100, 2200, and 2300"
    )
    variable.monotonicity_enforced = "no"

    ds.title = "FaIR-SEJ sea-level contribution trajectories"
    ds.storage_design = "Out-of-core chunked NetCDF writing"
    ds.description = (
        "Intermediate years are temporally interpolated and were not "
        "independently elicited."
    )
    return ds


def response_anchors(
    response_code: str,
    original_anchors: np.ndarray,
    original_index: dict[str, int],
    derivations: dict[str, tuple[str, ...]],
) -> np.ndarray:
    if response_code in original_index:
        projected = original_anchors[
            :, :, original_index[response_code]
        ]
    elif response_code in derivations:
        projected = np.sum(
            [
                original_anchors[:, :, original_index[source]]
                for source in derivations[response_code]
            ],
            axis=0,
            dtype=np.float32,
        )
    else:
        raise KeyError(f"No definition for {response_code}.")

    return np.concatenate(
        [
            np.zeros((len(projected), 1), dtype=np.float32),
            projected,
        ],
        axis=1,
    )


def write_netcdf_data(
    ds: Dataset,
    original_anchors: np.ndarray,
    original_metadata: pd.DataFrame,
    all_metadata: pd.DataFrame,
    derivations: dict[str, tuple[str, ...]],
    temporal_weights: np.ndarray,
    write_sample_chunksize: int,
) -> float:
    variable = ds.variables["sea_level_contribution"]
    original_index = {
        code: i
        for i, code in enumerate(original_metadata["response_code"])
    }
    years = np.asarray(ds.variables["year"][:])
    anchor_positions = [
        int(np.where(years == year)[0][0])
        for year in ANCHOR_YEARS
    ]

    maximum_error = 0.0
    n_samples = original_anchors.shape[0]

    print("Writing compressed NetCDF...")

    for response_position, row in all_metadata.iterrows():
        code = row["response_code"]
        anchors = response_anchors(
            code,
            original_anchors,
            original_index,
            derivations,
        )

        for start in range(
            0,
            n_samples,
            write_sample_chunksize,
        ):
            stop = min(start + write_sample_chunksize, n_samples)

            block = (
                anchors[start:stop] @ temporal_weights.T
            ).astype(np.float32)

            variable[
                start:stop,
                :,
                response_position,
            ] = block

            error = float(
                np.max(
                    np.abs(
                        block[:, anchor_positions]
                        - anchors[start:stop]
                    )
                )
            )
            maximum_error = max(maximum_error, error)

        ds.sync()
        suffix = " (derived)" if row["is_derived"] else ""
        print(
            f"  [{response_position + 1}/{len(all_metadata)}] "
            f"{code}{suffix}"
        )

    return maximum_error


def decreasing_summary(
    samples: pd.DataFrame,
    original_anchors: np.ndarray,
    original_metadata: pd.DataFrame,
    all_metadata: pd.DataFrame,
    derivations: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    original_index = {
        code: i
        for i, code in enumerate(original_metadata["response_code"])
    }
    scenarios = samples["scenario_label"].fillna("").astype(str).to_numpy()

    rows = []

    for _, response_row in all_metadata.iterrows():
        code = response_row["response_code"]
        anchors = response_anchors(
            code,
            original_anchors,
            original_index,
            derivations,
        )

        for scenario in pd.unique(scenarios):
            mask = scenarios == scenario
            selected = anchors[mask]

            for interval in range(3):
                change = (
                    selected[:, interval + 1]
                    - selected[:, interval]
                )
                rows.append(
                    {
                        "scenario_label": scenario,
                        "response_code": code,
                        "start_year": int(ANCHOR_YEARS[interval]),
                        "end_year": int(ANCHOR_YEARS[interval + 1]),
                        "n_draws": len(change),
                        "n_decreasing": int(np.sum(change < 0)),
                        "percent_decreasing": float(
                            100 * np.mean(change < 0)
                        ),
                        "mean_change": float(np.mean(change)),
                        "minimum_change": float(np.min(change)),
                        "median_change": float(np.median(change)),
                        "maximum_change": float(np.max(change)),
                    }
                )

    return pd.DataFrame(rows)


def write_metadata(
    path: Path,
    input_path: Path,
    netcdf_path: Path,
    samples: pd.DataFrame,
    responses: pd.DataFrame,
    years: np.ndarray,
    csv_chunksize: int,
    write_sample_chunksize: int,
) -> None:
    original = responses.loc[
        responses["is_derived"] == 0,
        "response_code",
    ].tolist()
    derived = responses.loc[
        responses["is_derived"] == 1,
        "response_code",
    ].tolist()

    path.write_text(
        f"""FaIR-SEJ out-of-core temporal interpolation
================================================
Input: {input_path}
NetCDF: {netcdf_path}

Samples: {len(samples):,}
Years: {len(years):,} ({int(years.min())}-{int(years.max())})
Responses: {len(responses):,}

CSV chunk size: {csv_chunksize:,}
NetCDF sample write chunk: {write_sample_chunksize:,}

Reference: 2005 = 0 mm relative to the AR6 1995-2014 baseline.
Method: piecewise-linear interpolation through 2005, 2100, 2200,
and 2300. Monotonicity is not imposed.

Original responses: {", ".join(original)}
Derived responses: {", ".join(derived) if derived else "None"}

Total AIS = EAIS + WAIS.
Runoff minus accumulation = runoff + signed accumulation.

Intermediate years were not independently elicited and are not
process-based transient ice-sheet simulations.
""",
        encoding="utf-8",
    )


def main(
    project_dir: Path,
    input_file: Path | None = None,
    output_dir: Path | None = None,
    start_year: int = 2005,
    end_year: int = 2300,
    year_step: int = 1,
    csv_chunksize: int = 200_000,
    write_sample_chunksize: int = 1_000,
    compression_level: int = 4,
) -> None:
    project_dir = Path(project_dir).expanduser().resolve()
    interpolation_dir = (
        project_dir / "output" / "fair_sej_interpolation"
    )

    input_path = (
        Path(input_file).expanduser().resolve()
        if input_file is not None
        else interpolation_dir / INPUT_FILENAME
    )
    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else interpolation_dir / "timeseries"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    netcdf_path = out_dir / OUTPUT_FILENAME
    validation_path = out_dir / "fair_sej_timeseries_validation.csv"
    decreasing_path = out_dir / "fair_sej_decreasing_intervals.csv"
    metadata_path = out_dir / "fair_sej_timeseries_metadata.txt"

    require_file(input_path)

    years = make_output_years(start_year, end_year, year_step)
    weights = make_temporal_weights(years)

    print("FaIR-SEJ out-of-core temporal interpolation")
    print("==========================================")
    print("Reference year: 2005 = 0 mm")
    print(
        f"Output years: {int(years.min())}-{int(years.max())}"
    )
    print(f"Number of output years: {len(years):,}")

    samples, original_metadata = scan_metadata(
        input_path,
        csv_chunksize,
    )
    original_anchors = load_anchor_array(
        input_path,
        csv_chunksize,
        samples,
        original_metadata,
    )
    all_metadata, derivations = build_response_metadata(
        original_metadata
    )

    ds = create_netcdf(
        netcdf_path,
        samples,
        all_metadata,
        years,
        compression_level,
        write_sample_chunksize,
    )
    try:
        max_error = write_netcdf_data(
            ds,
            original_anchors,
            original_metadata,
            all_metadata,
            derivations,
            weights,
            write_sample_chunksize,
        )
    finally:
        ds.close()

    decreasing_summary(
        samples,
        original_anchors,
        original_metadata,
        all_metadata,
        derivations,
    ).to_csv(decreasing_path, index=False)

    validation = pd.DataFrame(
        [
            {
                "check": "all_anchor_records_present",
                "passed": True,
                "maximum_absolute_error": 0.0,
            },
            {
                "check": "reference_year_equals_zero",
                "passed": True,
                "maximum_absolute_error": 0.0,
            },
            {
                "check": "projection_anchors_reproduced",
                "passed": bool(max_error <= 1e-4),
                "maximum_absolute_error": max_error,
            },
            {
                "check": "monotonicity_not_enforced",
                "passed": True,
                "maximum_absolute_error": np.nan,
            },
        ]
    )
    validation.to_csv(validation_path, index=False)

    if not validation["passed"].all():
        raise ValueError(
            f"A validation check failed. See {validation_path}"
        )

    write_metadata(
        metadata_path,
        input_path,
        netcdf_path,
        samples,
        all_metadata,
        years,
        csv_chunksize,
        write_sample_chunksize,
    )

    print("\nCompleted successfully.")
    print(f"Primary output:\n{netcdf_path}")
    print(
        f"NetCDF size: {netcdf_path.stat().st_size / 1024**3:.2f} GiB"
    )


def run_experiment(
    project_dir: Path,
    experiment_name: str,
    start_year: int,
    end_year: int,
    year_step: int,
    csv_chunksize: int,
    write_sample_chunksize: int,
    compression_level: int,
) -> dict[str, Any]:
    """
    Generate annual time series for one temporally coupled
    FaIR-SEJ interpolation experiment.
    """

    project_dir = Path(project_dir).expanduser().resolve()

    experiment_dir = (
        project_dir
        / "output"
        / "fair_sej_interpolation"
        / experiment_name
    )

    input_path = (
        experiment_dir
        / INPUT_FILENAME
    )

    output_path = (
        experiment_dir
        / "timeseries"
    )

    print("\n" + "=" * 76)
    print(f"Generating time series for: {experiment_name}")
    print(f"Input:\n{input_path}")
    print(f"Output directory:\n{output_path}")
    print("=" * 76)

    main(
        project_dir=project_dir,
        input_file=input_path,
        output_dir=output_path,
        start_year=start_year,
        end_year=end_year,
        year_step=year_step,
        csv_chunksize=csv_chunksize,
        write_sample_chunksize=write_sample_chunksize,
        compression_level=compression_level,
    )

    return {
        "experiment_name": experiment_name,
        "input_file": str(input_path),
        "output_directory": str(output_path),
        "netcdf_file": str(
            output_path
            / OUTPUT_FILENAME
        ),
        "validation_file": str(
            output_path
            / "fair_sej_timeseries_validation.csv"
        ),
        "decreasing_intervals_file": str(
            output_path
            / "fair_sej_decreasing_intervals.csv"
        ),
        "metadata_file": str(
            output_path
            / "fair_sej_timeseries_metadata.txt"
        ),
    }


def run_all_experiments(
    project_dir: Path,
    experiments: list[str],
    start_year: int,
    end_year: int,
    year_step: int,
    csv_chunksize: int,
    write_sample_chunksize: int,
    compression_level: int,
) -> pd.DataFrame:
    """
    Generate time series for all requested coupling experiments.
    """

    project_dir = Path(project_dir).expanduser().resolve()

    results = []

    for experiment_name in experiments:
        result = run_experiment(
            project_dir=project_dir,
            experiment_name=experiment_name,
            start_year=start_year,
            end_year=end_year,
            year_step=year_step,
            csv_chunksize=csv_chunksize,
            write_sample_chunksize=write_sample_chunksize,
            compression_level=compression_level,
        )
        results.append(result)

    summary = pd.DataFrame(results)

    summary_path = (
        project_dir
        / "output"
        / "fair_sej_interpolation"
        / "fair_sej_timeseries_experiment_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 76)
    print("All requested time-series experiments completed.")
    print(f"Summary saved to:\n{summary_path}")
    print("=" * 76)

    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
    )

    parser.add_argument(
        "--experiment",
        choices=[
            "all",
            *COUPLING_EXPERIMENTS,
        ],
        default="all",
        help=(
            "Coupling experiment to process. "
            "The default processes all three experiments."
        ),
    )

    # These two options preserve the ability to process a custom file.
    # When --input-file is supplied, --experiment is ignored.
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2005,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=2300,
    )

    parser.add_argument(
        "--year-step",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--csv-chunksize",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--write-sample-chunksize",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        choices=range(10),
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    # Custom-file mode: retain the original script behavior.
    if args.input_file is not None:

        main(
            project_dir=args.project_dir,
            input_file=args.input_file,
            output_dir=args.output_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            year_step=args.year_step,
            csv_chunksize=args.csv_chunksize,
            write_sample_chunksize=args.write_sample_chunksize,
            compression_level=args.compression_level,
        )

    # Standard mode: process one or all temporally coupled experiments.
    else:

        selected_experiments = (
            COUPLING_EXPERIMENTS
            if args.experiment == "all"
            else [args.experiment]
        )

        run_all_experiments(
            project_dir=args.project_dir,
            experiments=selected_experiments,
            start_year=args.start_year,
            end_year=args.end_year,
            year_step=args.year_step,
            csv_chunksize=args.csv_chunksize,
            write_sample_chunksize=args.write_sample_chunksize,
            compression_level=args.compression_level,
        )
