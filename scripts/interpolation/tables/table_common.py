"""Shared loading, strict production validation, and formatting for manuscript tables.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from table_config import (
    COMPONENTS,
    COUPLING_METADATA_FILE,
    EXPECTED_COUPLING_METHOD,
    EXPECTED_SOURCE_BY_YEAR,
    FINAL_REFERENCE_PERIOD,
    QUANTILES,
    TABLE_YEARS,
    CMIP7_ANNUAL_NETCDF, CMIP7_SAMPLED_CSV, EXPECTED_CMIP7_CRN_SEED, EXPECTED_CMIP7_SAMPLING_RULE,
)


def canonicalize_cmip6_ssp(value: str) -> str:
    code = str(value).strip().lower()
    if code in {"ssp534", "ssp534-over", "ssp534_over", "ssp534over"}:
        return "ssp534-over"
    return code


def display_number(value: float, decimals: int | None = None) -> str:
    """
    Format manuscript-table values according to magnitude:

      |value| < 10 cm  -> 1 decimal place
      |value| >= 10 cm -> nearest whole cm

    Additional display rules:
      * values that round to 0.0 are displayed as "0"
      * negative values use the true minus sign "−" (U+2212),
        not a hyphen-minus "-"

    Examples:
         0.03  -> "0"
        -0.03  -> "0"
         0.64  -> "0.6"
        -3.42  -> "−3.4"
         9.81  -> "9.8"
         9.96  -> "10"
        10.2   -> "10"
       -24.4   -> "−24"
       466.7   -> "467"

    `decimals` is retained only for compatibility with the existing
    table-generation functions.
    """
    value = float(value)
    magnitude = abs(value)

    if magnitude < 10:
        rounded = round(magnitude, 1)

        # Do not display 0.0 or −0.0.
        if rounded == 0:
            return "0"

        # Avoid displaying 10.0 when e.g. 9.96 rounds upward.
        if rounded >= 10:
            text = f"{rounded:.0f}"
        else:
            text = f"{rounded:.1f}"

    else:
        rounded = round(magnitude)

        if rounded == 0:
            return "0"

        text = f"{rounded:.0f}"

    sign = "−" if value < 0 else ""
    return sign + text


def formatted_entries(row: pd.Series, decimals: int) -> tuple[str, str, str]:
    median = display_number(row["p50_cm"], decimals)
    inner = (
        f"{display_number(row['p17_cm'], decimals)} to "
        f"{display_number(row['p83_cm'], decimals)}"
    )
    outer = (
        f"{display_number(row['p05_cm'], decimals)} to "
        f"{display_number(row['p95_cm'], decimals)}"
    )
    return median, inner, outer


def _validate_final_method_provenance(frame: pd.DataFrame, family: str) -> None:
    """Prevent the table scripts from silently consuming a retired core-chain table."""
    if "reference_period" not in frame.columns:
        raise ValueError(
            f"{family}: source table has no reference_period column; this does not look "
            "like the final emulandice-2050 production output."
        )

    ref = frame["reference_period"].astype(str).str.strip()
    if not ref.eq(FINAL_REFERENCE_PERIOD).all():
        bad = sorted(ref.unique().tolist())
        raise ValueError(
            f"{family}: expected reference period {FINAL_REFERENCE_PERIOD}; found {bad}."
        )

    if "source" not in frame.columns:
        raise ValueError(
            f"{family}: source table has no source column; cannot verify final method provenance."
        )

    for year, expected_source in EXPECTED_SOURCE_BY_YEAR.items():
        rows = frame.loc[frame["year"].eq(year)]
        if rows.empty:
            continue
        observed = set(rows["source"].astype(str).str.strip())
        if observed != {expected_source}:
            raise ValueError(
                f"{family}: year {year} does not have the expected final-method source.\n"
                f"Expected: {expected_source!r}\n"
                f"Observed: {sorted(observed)}"
            )


def validate_selected_coupling_metadata() -> None:
    """Refuse stale core outputs from another temporal-coupling implementation."""
    path = COUPLING_METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot verify selected 02I-B temporal coupling; metadata missing: {path}"
        )
    metadata = pd.read_csv(path)
    if "method" not in metadata.columns or metadata.empty:
        raise ValueError(f"Temporal-coupling metadata has no method field: {path}")
    methods = metadata["method"].astype(str).str.strip().unique().tolist()
    if methods != [EXPECTED_COUPLING_METHOD]:
        raise ValueError(
            f"Unexpected temporal-coupling method in {path}: {methods!r}; "
            f"expected {EXPECTED_COUPLING_METHOD!r}"
        )


def validate_cmip7_crn_provenance() -> None:
    """Require that Stage 03B was regenerated with the selected CRN seed.

    This checks the actual trajectory NetCDF metadata and row-level sampling
    labels, not merely a configured seed in the table package.
    """
    from netCDF4 import Dataset
    for path in (CMIP7_ANNUAL_NETCDF, CMIP7_SAMPLED_CSV):
        if not path.is_file():
            raise FileNotFoundError(f"CMIP7 CRN output missing: {path}. Rerun Stage 03B.")
    with Dataset(CMIP7_ANNUAL_NETCDF, 'r') as nc:
        actual_seed = getattr(nc, 'cmip7_crn_seed', None)
        actual_rule = str(getattr(nc, 'sej_sampling_rule', ''))
        if actual_seed is None or int(actual_seed) != EXPECTED_CMIP7_CRN_SEED:
            raise ValueError(f"CMIP7 NetCDF uses CRN seed {actual_seed!r}, expected {EXPECTED_CMIP7_CRN_SEED}; rerun Stage 03B.")
        if actual_rule != EXPECTED_CMIP7_SAMPLING_RULE:
            raise ValueError(f"CMIP7 NetCDF sampling rule {actual_rule!r} is not selected CRN rule.")
        netcdf_rows = np.asarray(nc.variables['sej2025_row_index'][:], dtype=np.int64)
    sample = pd.read_csv(CMIP7_SAMPLED_CSV, usecols=['ssp','sample_id','selected_scenario','sej2025_row_index','sej_sampling_rule'])
    if not sample['sej_sampling_rule'].astype(str).eq(EXPECTED_CMIP7_SAMPLING_RULE).all():
        raise ValueError('CMIP7 sampled CSV does not contain the selected CRN rule.')
    indices = pd.to_numeric(sample['sej2025_row_index'], errors='raise').to_numpy(dtype=np.int64)
    if len(indices) != len(netcdf_rows) or not np.array_equal(indices, netcdf_rows):
        raise ValueError('CMIP7 sampled CSV and annual NetCDF row indices disagree.')
    # Reconstruct the exact CRN mapping using the same code as Stage 03B.
    from core_annual_tension import crn_indices_by_sample_id
    import core_config as cfg
    if int(cfg.CMIP7_CRN_SEED) != EXPECTED_CMIP7_CRN_SEED:
        raise ValueError('core_config.py CRN seed differs from table-package expected seed.')
    coupled = pd.read_csv(cfg.COUPLED_SEJ_FILE, usecols=['coupled_trajectory_id'])
    reconstructed = crn_indices_by_sample_id(sample, len(coupled))
    if not np.array_equal(reconstructed, indices):
        mismatch = int(np.count_nonzero(reconstructed != indices))
        raise ValueError(f'CMIP7 sampled row mapping differs from CRN seed {EXPECTED_CMIP7_CRN_SEED}: {mismatch} mismatches.')


def load_projection_table(
    path: Path,
    *,
    scenario_codes: tuple[str, ...],
    family: str,
) -> pd.DataFrame:
    """Load one final selected-year quantile CSV and strictly validate its grid."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing final {family} selected-year projection table:\n{path}"
        )

    validate_selected_coupling_metadata()
    if family.upper() == "CMIP7":
        validate_cmip7_crn_provenance()
    frame = pd.read_csv(path)
    required = {"component", "ssp", "year", *QUANTILES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{family}: source table is missing columns: {missing}")

    frame = frame.copy()
    frame["component"] = frame["component"].astype(str).str.strip()
    frame["ssp"] = frame["ssp"].astype(str).str.strip()
    if family.upper() == "CMIP6":
        frame["ssp"] = frame["ssp"].map(canonicalize_cmip6_ssp)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)

    # Validate provenance before filtering so an accidentally selected legacy file
    # cannot pass merely because its numerical grid happens to look right.
    _validate_final_method_provenance(frame, family)

    frame = frame.loc[frame["year"].isin(TABLE_YEARS)].copy()

    frame[list(QUANTILES)] = frame[list(QUANTILES)].apply(
        pd.to_numeric, errors="raise"
    )
    q = frame[list(QUANTILES)].to_numpy(dtype=float)
    if not np.isfinite(q).all():
        raise ValueError(f"{family}: non-finite percentile values in source table")
    if np.any(np.diff(q, axis=1) < 0):
        raise ValueError(
            f"{family}: percentile order violation; require p05<=p17<=p50<=p83<=p95"
        )

    keys = ["ssp", "year", "component"]
    if frame.duplicated(keys).any():
        duplicate = frame.loc[frame.duplicated(keys, keep=False), keys].head(10)
        raise ValueError(f"{family}: duplicate pathway/year/component rows:\n{duplicate}")

    expected = pd.MultiIndex.from_product(
        (scenario_codes, TABLE_YEARS, COMPONENTS),
        names=("ssp", "year", "component"),
    )
    observed = pd.MultiIndex.from_frame(frame[keys])
    missing_rows = expected.difference(observed)
    extra_rows = observed.difference(expected)
    if len(missing_rows) or len(extra_rows):
        raise ValueError(
            f"{family}: source pathway/year/component grid does not match the final table set.\n"
            f"Missing (first 10): {list(missing_rows[:10])}\n"
            f"Extra (first 10): {list(extra_rows[:10])}"
        )

    indexed = frame.set_index(keys)
    ordered = indexed.loc[expected].reset_index()
    return ordered
