#!/usr/bin/env python3
"""Preflight checks for the final emulandice-2050 + tension core chain.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from pathlib import Path
import core_config as cfg
from core_imbie_helpers import raw_files

checks = []
checks.append(("SEJ2025 split-panel file", cfg.SEJ_DATA_FILE))

for ssp, path in cfg.CMIP6_FAIR_FILES.items():
    checks.append((f"CMIP6 {ssp} FaIR2 GSAT", path))

for code in cfg.CMIP7_CODES:
    checks.append((
        f"CMIP7 {code} FaIR GSAT",
        cfg.DATA_DIR / "fair_output" / f"ar7.{code}.temperature.fair.temperature_gsat.nc",
    ))

for v, path in raw_files().items():
    checks.append((f"IMBIE raw {v}", path))

for warming, rcp in cfg.DP21_SCENARIOS.items():
    checks.append((f"DP21 EAIS {warming:g}C", cfg.DP21_ROOT / f"dp21_eais_{rcp}.nc"))
    checks.append((f"DP21 WAIS {warming:g}C", cfg.DP21_ROOT / f"dp21_wais_{rcp}.nc"))

for warming, path in cfg.GRIS_CORRELATION_FILES.items():
    checks.append((f"FittedISMIP GrIS {warming:g}C", path))

for warming, experiment in cfg.EMULANDICE_EXPERIMENTS.items():
    exp = cfg.EMULANDICE_ROOT / experiment
    checks.append((f"emulandice EAIS {warming:g}C", exp / "output" / "emulandice-ais" / "output-gslr-eais.nc"))
    checks.append((f"emulandice WAIS {warming:g}C", exp / "output" / "emulandice-ais" / "output-gslr-wais.nc"))
    gris_dir = exp / "output" / "emulandice-gris"
    gris_candidates = [
        gris_dir / "output-gslr-gris.nc",
        gris_dir / "output-gslr-gis.nc",
        gris_dir / "output-gslr-greenland.nc",
    ]
    hits = [p for p in gris_candidates if p.is_file()]
    if not hits and gris_dir.is_dir():
        hits = sorted(gris_dir.glob("output-gslr-*.nc"))
    checks.append((f"emulandice GrIS {warming:g}C", hits[0] if len(hits) == 1 else gris_candidates[0]))

missing = []
for label, path in checks:
    ok = Path(path).is_file()
    print(f"{'OK' if ok else 'MISSING':7s}  {label}: {path}")
    if not ok:
        missing.append((label, path))

if missing:
    raise FileNotFoundError(f"{len(missing)} required input(s) are missing. See lines above.")

print("\nAll required inputs found.")
print("Final knot: emulandice workflow-1e at 2050 only")
print("Final interpolation: classical spline under tension, p=", cfg.TENSION_P)
