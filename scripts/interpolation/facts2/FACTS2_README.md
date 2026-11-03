# FACTS2 Experiment Configurations

This directory contains the FACTS2 experiment configurations used to generate climate and ice-sheet model inputs for the 2025 structured expert judgement (SEJ) ice-sheet projections.

## Experiments

Two sets of experiments are provided:

- `fair2/`: FaIR2 climate simulations for six CMIP6 SSP pathways (SSP1-1.9, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-3.4-OS, and SSP5-8.5).
- `emulandice/`: emulandice simulations for the same six SSP pathways, producing Antarctic and Greenland ice-sheet projections.

Each of the 12 experiment directories includes:

- `experiment-config.yaml`: FACTS2 experiment settings and input/output paths.
- `experiment-compose.yaml`: Docker Compose services, module commands, images, and volume mounts.

## Role in the interpolation workflow

FaIR2 global temperature trajectories inform the selection and weighting of SEJ warming scenarios. Emulandice outputs provide model-based ice-sheet projections used in constructing the 2050 temporal constraints. These outputs are subsequently used by the scripts under `scripts/interpolation/`.

## Reproducing the experiment configurations

The experiments require a compatible FACTS2 installation and its associated input datasets. The archived YAML files use `${FACTS2_ROOT}` in place of machine-specific absolute paths. Docker Compose supports this variable substitution, but FACTS2's interpretation of environment variables inside `experiment-config.yaml` has not been verified.

To generate local copies with the paths explicitly expanded, run the following commands **from the metarepo root**:

```bash
python scripts/interpolation/facts2/prepare_experiments.py \
  --facts2-root "$HOME/FACTS2" \
  --output-root "$HOME/FACTS2/prepared_metarepo_experiments"
```

The preparation script reads the archived files and writes expanded copies under `prepared_metarepo_experiments/fair2/` and `prepared_metarepo_experiments/emulandice/`. It does not overwrite the archived YAML files or the original FACTS2 experiments. To check that no placeholders remain:

```bash
grep -Rn '\${FACTS2_ROOT}' "$HOME/FACTS2/prepared_metarepo_experiments" || true
```

Before executing an experiment, inspect the prepared files and verify that the referenced input directories exist.


## Data and software

The repository preserves experiment definitions but does not include generated NetCDF outputs, Docker images, or model input datasets. Scientific parameters, scenario identifiers, random seeds, and Docker image versions in the copied configurations have otherwise been retained.

See the [FACTS project on GitHub](https://github.com/fact-sealevel) for the software and installation information.
