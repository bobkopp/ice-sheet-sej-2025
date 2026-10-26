#!/usr/bin/env python3
"""Safely preview or execute selected stages of the final 02I-B workflow.

Examples (from the starter repository root):
    source test_environment.sh
    python scripts/RUN_CORE_METHOD_CHAIN.py --list
    python scripts/RUN_CORE_METHOD_CHAIN.py --stages 03B,CRN --dry-run
    python scripts/RUN_CORE_METHOD_CHAIN.py --stages 00 --execute

Dry-run is the default. --execute requires isolated output directories,
existing external inputs, and no existing files that a stage might overwrite.
Scientific stage scripts are not changed by this runner.

Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)

Created on: 2026-10-26
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]  # scripts root
STARTER_ROOT = HERE.parent.resolve()
TEST_OUTPUT_ROOT = (STARTER_ROOT / "output").resolve()

STAGES = [
    ("00", "00_check_inputs.py"),
    ("01", "01_FaIR_to_SEJ_SCENARIO_SELECTION_DOUBLE_CHECKPOINT_TRUNCATION.py"),
    ("01B", "01B_FaIR_to_SEJ_SCENARIO_SELECTION_CMIP7_DOUBLE_CHECKPOINT_TRUNCATION.py"),
    ("02", "02_TEMPORAL_COUPLING_EMULANDICE2050_12VAR_MAXDET.py"),
    ("03", "03_SAMPLE_SEJ_AND_GENERATE_ANNUAL_TRAJECTORIES.py"),
    ("03B", "03B_SAMPLE_SEJ_AND_GENERATE_ANNUAL_TRAJECTORIES_CMIP7.py"),
    ("CRN", "CMIP7_02IB_MIXTURE_AND_CRN_DIAGNOSTIC.py"),
    ("04", "04_BUILD_MANUSCRIPT_TABLES.py"),
    ("05", "05_BUILD_FORMATTED_MANUSCRIPT_TABLES.py"),
]
STAGE_MAP = dict(STAGES)
STAGE_FOLDER = {'00': 'interpolation/inputs', '01': 'interpolation/scenario_selection', '01B': 'interpolation/scenario_selection', '02': 'interpolation/temporal_dependence', '03': 'interpolation/annual_projections', '03B': 'interpolation/annual_projections', 'CRN': 'interpolation/diagnostics', '04': 'interpolation/tables', '05': 'interpolation/tables'}

def stage_script(code, script):
    return HERE / STAGE_FOLDER[code] / script
# These are true data dependencies, not necessarily run-order dependencies.
DEPENDENCIES = {
    "03": ("01", "02"),
    "03B": ("01B", "02"),
    "CRN": ("01B", "02", "03B"),
    "04": ("03", "03B"),
    "05": ("02", "03", "03B"),
}


def selected_stages(raw: str) -> list[tuple[str, str]]:
    if raw.lower() == "all":
        return STAGES
    names = [s.strip().upper() for s in raw.split(",") if s.strip()]
    unknown = sorted(set(names) - STAGE_MAP.keys())
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(unknown)}")
    if not names or len(names) != len(set(names)):
        raise ValueError("Provide distinct stage codes")
    chosen = set(names)
    return [(code, script) for code, script in STAGES if code in chosen]


def under(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def isolated_paths(selected: list[str]) -> dict[str, Path]:
    """Validate only destinations that the selected scientific stages may write.

    Figure-output configuration is irrelevant to this computational runner.
    Stages 00-04 and CRN use SEJ_CORE_OUTPUT_ROOT; Stage 05 additionally
    writes formatted tables to TABLE_OUTPUT_ROOT.
    """
    raw_core = os.environ.get("SEJ_CORE_OUTPUT_ROOT")
    if not raw_core:
        raise ValueError("SEJ_CORE_OUTPUT_ROOT is unset. Set it to an isolated output directory")
    core = Path(raw_core).expanduser().resolve()
    if not under(core, TEST_OUTPUT_ROOT):
        raise ValueError(f"Unsafe SEJ_CORE_OUTPUT_ROOT: {core} (must be under {TEST_OUTPUT_ROOT})")

    raw_tables = os.environ.get("TABLE_OUTPUT_ROOT")
    if "05" in selected and not raw_tables:
        raise ValueError("Stage 05 requires TABLE_OUTPUT_ROOT in the isolated output directory")
    tables = (Path(raw_tables).expanduser().resolve() if raw_tables
              else TEST_OUTPUT_ROOT / "metarepo_runner_tables_unused")
    if "05" in selected:
        if not under(tables, TEST_OUTPUT_ROOT):
            raise ValueError(f"Unsafe TABLE_OUTPUT_ROOT: {tables} (must be under {TEST_OUTPUT_ROOT})")
        if tables == core:
            raise ValueError("TABLE_OUTPUT_ROOT and SEJ_CORE_OUTPUT_ROOT must differ")
        if os.environ.get("TABLE_FINAL_CORE_ROOT") and Path(os.environ["TABLE_FINAL_CORE_ROOT"]).expanduser().resolve() != core:
            raise ValueError("TABLE_FINAL_CORE_ROOT conflicts with SEJ_CORE_OUTPUT_ROOT")
        if os.environ.get("TABLE_PROJECT_DIR"):
            raise ValueError("Unset TABLE_PROJECT_DIR before Stage 05; it adds an unverified override")
    return {"SEJ_CORE_OUTPUT_ROOT": core, "TABLE_OUTPUT_ROOT": tables}


def stage_products(core: Path, tables: Path) -> dict[str, tuple[Path, ...]]:
    select = core / "01_warming_selection"
    cm6 = core / "03_cmip6_annual"
    cm7 = core / "03B_cmip7_annual"
    return {
        "01": (select / "fair_b25_baseline_scenario_selection.csv",),
        "01B": (select / "cmip7_fair_b25_baseline_scenario_selection.csv",),
        "02": (core / "02_temporal_coupling" / "SEJ25_emulandice2050_temporally_coupled_by_warming.csv",),
        "03": (cm6 / "sampled_sej2025_realizations.csv", cm6 / "annual_ice_sheet_trajectories_2000_2300.nc",
               cm6 / "tables" / "all_ice_sheet_components_SSP_projection_table_2020_2050_2100_2150_2200_2300.csv"),
        "03B": (cm7 / "sampled_sej2025_realizations.csv", cm7 / "annual_ice_sheet_trajectories_2000_2300.nc",
                cm7 / "tables" / "all_ice_sheet_components_CMIP7_projection_table_2020_2050_2100_2150_2200_2300.csv"),
        "CRN": (core / "diagnostics_cmip7_mixture_crn" / "CMIP7_02IB_mixture_vs_realized_vs_CRN_quantiles.csv",),
        "04": (core / "04_manuscript_tables" / "MANUSCRIPT_all_components_all_pathways_long.csv",),
        "05": (tables / "cmip6_ssp", tables / "cmip7_pathways"),
    }


def stage_output_areas(core: Path, tables: Path) -> dict[str, tuple[Path, ...]]:
    return {
        "01": (core / "01_warming_selection",),
        "01B": (core / "01_warming_selection",),
        "02": (core / "02_temporal_coupling",),
        "03": (core / "03_cmip6_annual",),
        "03B": (core / "03B_cmip7_annual",),
        "CRN": (core / "diagnostics_cmip7_mixture_crn",),
        "04": (core / "04_manuscript_tables",),
        "05": (tables / "cmip6_ssp", tables / "cmip7_pathways"),
    }


def existing_outputs(code: str, core: Path, tables: Path) -> list[Path]:
    areas = stage_output_areas(core, tables)[code]
    if code in ("01", "01B"):
        # Shared directory: identify only this stage's output prefix.
        prefix = "cmip7_" if code == "01B" else "fair_b25_"
        shared = areas[0]
        return sorted(p for p in shared.iterdir() if p.name.startswith(prefix)) if shared.is_dir() else []
    # For other stages, ANY pre-existing file in their dedicated output tree
    # blocks execution, including partially completed runs.
    found = []
    for area in areas:
        if area.is_file():
            found.append(area)
        elif area.is_dir():
            found.extend(p for p in area.rglob("*") if p.is_file())
    return sorted(found)


def external_inputs(project: Path, facts2: Path, stage: str) -> list[Path]:
    data = Path(os.environ.get("SEJ_DATA_DIR", project / "data")).expanduser().resolve()
    fair = facts2 / "experiments" / "fair2"
    if stage == "01":
        exps = ("ssp119", "ssp126", "ssp245", "ssp370", "ssp534", "ssp585")
        return [fair / f"{s}_fair2" / "output" / "fair2-climate" / "gsat.nc" for s in exps]
    if stage == "01B":
        return [data / "fair_output" / f"ar7.{s}.temperature.fair.temperature_gsat.nc"
                for s in ("H", "HL", "L", "LN", "M", "ML", "VL")]
    if stage == "02":
        dp21 = facts2 / "data" / "module_specific_input_data" / "deconto21-ais"
        gris = data / "temporal_correlation_matrices" / "GrIS_FittedISMIP_scenario_specific"
        inputs = [data / "sej2025_SplitPanels_liz_updated.csv"]
        for code in ("rcp26", "rcp45", "rcp85"):
            inputs += [dp21 / f"dp21_eais_{code}.nc", dp21 / f"dp21_wais_{code}.nc"]
        for scenario in ("ssp126", "ssp245", "ssp585"):
            inputs.append(gris / f"GrIS_FittedISMIP_{scenario}_Pearson.csv")
        for scenario in ("ssp126", "ssp245", "ssp585"):
            base = facts2 / "experiments" / f"emulandice1_{scenario}" / "output"
            inputs += [base / "emulandice-ais" / "output-gslr-eais.nc",
                       base / "emulandice-ais" / "output-gslr-wais.nc"]
        return inputs  # GrIS candidate filenames and IMBIE are checked by stage 00.
    return []


def check_dependencies(chosen: list[str], core: Path, tables: Path) -> list[str]:
    products = stage_products(core, tables)
    completed = set()
    problems = []
    for code in chosen:
        for prereq in DEPENDENCIES.get(code, ()):
            if prereq in completed:
                continue
            missing = [p for p in products[prereq] if not p.is_file()]
            if missing:
                problems.append(f"{code} needs {prereq}: missing {missing[0]}")
        completed.add(code)
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List stages and dependencies")
    parser.add_argument("--stages", default="all", help="Comma-separated stage codes, or all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    mode.add_argument("--execute", action="store_true", help="Run after passing safety checks")
    args = parser.parse_args()
    if args.list:
        for code, script in STAGES:
            print(f"{code:>3}  {script:72}  needs: {', '.join(DEPENDENCIES.get(code, ())) or 'none'}")
        return
    try:
        stages = selected_stages(args.stages)
    except ValueError as e:
        parser.error(str(e))
    missing_scripts = [script for code, script in stages if not stage_script(code, script).is_file()]
    if missing_scripts:
        parser.error("Missing scripts: " + ", ".join(missing_scripts))
    print("Mode:", "EXECUTE" if args.execute else "DRY RUN (no scripts execute)")
    print("Stages:", ", ".join(code for code, _ in stages))
    for code, script in stages:
        print(f"  {code:>3}: {script}")
    if not args.execute:
        print("\nNo calculations performed. To check execution safety, source test_environment.sh then use --execute.")
        return
    try:
        selected = [code for code, _ in stages]
        paths = isolated_paths(selected)
        core = paths["SEJ_CORE_OUTPUT_ROOT"]
        tables = paths["TABLE_OUTPUT_ROOT"]
        project = Path(os.environ.get("SEJ_PROJECT_DIR", STARTER_ROOT)).expanduser().resolve()
        facts2 = Path(os.environ.get("SEJ_FACTS2_ROOT", Path.home() / "FACTS2")).expanduser().resolve()
        errors = check_dependencies(selected, core, tables)
        # For 00, rely on its own full preflight, which also checks IMBIE and
        # multiple possible emulandice GrIS filenames.
        for code in selected:
            for p in external_inputs(project, facts2, code):
                if not p.is_file():
                    errors.append(f"{code}: required external input missing: {p}")
            if code != "00":
                found = existing_outputs(code, core, tables)
                if found:
                    errors.append(f"{code}: output already exists ({len(found)} file(s)); first: {found[0]}")
        if errors:
            raise ValueError("Execution blocked:\n  - " + "\n  - ".join(errors))
    except ValueError as e:
        parser.error(str(e))
    print("\nPreflight passed: isolated paths, dependencies, external inputs, and no existing output files.")
    print("NOTE: Stage 00 performs the full scientific input preflight when selected.")
    for code, script in stages:
        # Check again immediately before each run; important for partial retries.
        if code != "00" and existing_outputs(code, core, tables):
            parser.error(f"Stage {code} already has outputs; refusing to overwrite")
        print(f"\nRUNNING {code}: {script}", flush=True)
        env=os.environ.copy()
        env['PYTHONPATH']=str(HERE)+os.pathsep+env.get('PYTHONPATH','')
        # Make organized shared modules available to computational subprocesses.
        script_root = Path(__file__).resolve().parents[1]
        
        module_dirs = [
            script_root / "shared",
            script_root / "interpolation" / "inputs",
            script_root / "interpolation" / "annual_projections",
            script_root / "interpolation" / "tables",
        ]
        
        existing_pythonpath = env.get("PYTHONPATH", "")
        
        env["PYTHONPATH"] = os.pathsep.join(
            [str(path) for path in module_dirs]
            + ([existing_pythonpath] if existing_pythonpath else [])
        )
        
        
        subprocess.run([sys.executable, str(stage_script(code, script))], check=True, cwd=HERE, env=env)
    print("\nSelected workflow stages completed.")


if __name__ == "__main__":
    main()
