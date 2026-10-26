# Workflow structure

Two scientific workflows are maintained separately to make the paper reproducible without mixing original elicitation analyses with subsequent climate-scenario interpolation.

## Original SEJ (`scripts/original_sej/`)

Externally produced SEJ Monte Carlo input, expert calibration and driver analyses, original warming references, endpoint distributions and corresponding figures. This workflow **does not** recreate the expert elicitation itself or regenerate Monte Carlo samples from the original expert instrument.

## Interpolation (`scripts/interpolation/`)

- `inputs/`: external input checks and IMBIE loading/rebasing helpers
- `scenario_selection/`: CMIP6/CMIP7 scenario selection from FaIR trajectories
- `temporal_dependence/`: SEJ temporal coupling informed by DP21, fitted ISMIP6 and emulandice
- `annual_projections/`: annual trajectories from historical records, intermediate constraints and SEJ endpoints
- `tables/`: manuscript and supplementary table generation
- `figures/`: interpolation and pathway figures
- `diagnostics/`: CMIP7 mixture and common-random-number diagnostics

## Common machinery

- `scripts/shared/`: plotting constants, labels, helpers, core configuration
- `scripts/runners/RUN_CORE_METHOD_CHAIN.py`: preview and guarded stage execution
- `scripts/runners/RUN_FINAL_FIGURES.py`: preview and organized figure generation

Both runners execute organized source files. The runner subprocesses place `scripts/shared/` and the necessary helper directories on Python's import path. Configure `SEJ_DATA_DIR` and `SEJ_FACTS2_ROOT` to use external inputs. Set `SEJ_CORE_OUTPUT_ROOT` under this repository's `output/` before executing; Stage 05 also needs `TABLE_OUTPUT_ROOT` under `output/`.

The core runner blocks output overwrites. For exact commands and input-layout requirements see the top-level README.
