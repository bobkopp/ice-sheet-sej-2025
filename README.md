# Ice-sheet projections from structured expert judgement

> **Publication metadata pending.** Add the final paper title, coauthors, journal citation, publication DOI, code-release DOI, and finalized licenses once confirmed. This is a pre-release research repository, not a claim that every workflow has been independently reproduced.

This repository provides the software, input datasets, figures, and instructions used in a study of Greenland and Antarctic ice-sheet contributions to global mean sea-level change through 2300. The study uses structured expert judgement (SEJ), with scenario-based interpolation informed by FaIR, FACTS2, IMBIE observations, DeConto et al. (2021), fitted ISMIP6, and emulandice.

**Manuscript:** [Insert final manuscript title, authors, journal, year, and publication URL/DOI]

## Abstract

Structured expert judgement (SEJ) is an established method for estimating the likelihood of deeply uncertain outcomes, such as century-to-multicentury ice-sheet changes. We used SEJ to estimate ice-sheet changes through 2300 under scenarios stabilizing in 2100 at 2°C, 3.5°C, and 5°C of global warming. Under a 2°C scenario, the projected median (5th–95th percentile) ice-sheet contribution to GMSL change is 23 (−16 to 76) cm by 2100 and 131 (−45 to 420) cm by 2300; under a 5°C scenario, these values rise to 42 (−34 to 181) cm by 2100 and 365 (10 to 1120) cm by 2300. Relative to a comparable 2018 study, upper-end projections are similar while low-end projections reflect a greater probability of a significant increase in Antarctic accumulation. Combining elicitation results with probabilistic interpolation indicates a GMSL contribution from the Greenland ice sheet under SSP2-4.5 of 16 (−4 to 43) cm by 2100, 33 (−6 to 93) cm by 2150, and 77 (−17 to 234) cm by 2300, exceeding corresponding Intergovernmental Panel on Climate Change Sixth Assessment Report (AR6) medium-confidence estimates in uncertainty and, for 2100 and 2150, median values. For Antarctica, projections of 11 (−34 to 85) cm by 2100, 26 (−54 to 191) cm by 2150, and 105 (−84 to 532) cm by 2300 are more uncertain than AR6 medium-confidence projections. Accounting for tail dependence, 95th-percentile total ice-sheet contributions reach 1.1 m by 2100 and 2.5 m by 2150 under SSP2-4.5, highlighting the potential for multi-meter GMSL rise in the 22nd century under intermediate emissions.

## Journal reference

[Insert final full journal citation and DOI when available.]

## Code reference

- **This metarepo:** [Insert GitHub URL, versioned Zenodo release DOI, and software license after publication.]
- **Underlying modeling systems:** see *Contributing modeling software* below. Refer to their specific releases and citations, not just this repository.

## Data reference

### Input data

| Input | Purpose | Where to obtain it |
| --- | --- | --- |
| 2025 SEJ split-panel Monte Carlo data | Original elicitation distributions and draws | [Study data repository: Zenodo record 10.5281/zenodo.23543612](https://doi.org/10.5281/zenodo.23543612); verify final filenames and release status |
| Expert elicitation responses, calibration variables, process drivers | Original SEJ analysis and figures | Same study repository; verify file mapping |
| IMBIE 2023 observational records | Antarctic and Greenland historical contributions | [Insert authoritative data/software citation, license, and download instructions] |
| FaIR2 CMIP6 GSAT and CMIP7 pathway GSAT | Scenario selection and interpolation | Generated with FACTS2/FaIR; archived files are intended for study data repository; verify provenance and names |
| DeConto et al. (2021) EAIS and WAIS ensembles | Antarctic target temporal dependence | FACTS2 `deconto21-ais` inputs; [add dataset identifier/release] |
| Fitted ISMIP6 Greenland correlation matrices | Greenland target temporal dependence | Study data repository/underlying fitted ISMIP6 resource; [confirm citation] |
| emulandice EAIS, WAIS, GrIS outputs | Intermediate 2050 probabilistic constraints | FACTS2 emulandice experiments; [add version, DOI, and archived outputs] |
| IPCC AR6 comparison inputs | Reference projections for selected figures | [Insert AR6 projection release and prescribed acknowledgement/citation] |


### Output data

- Scenario-selection tables, temporally coupled SEJ draws, annual trajectories, and manuscript tables are created under `output/` (excluded from Git).

## Contributing modeling software

| Software | Role | Repository | Exact version / DOI |
| --- | --- | --- | --- |
| [FACTS2](https://github.com/fact-sealevel) | Framework for emulandice, FaIR-derived inputs and sea-level workflows | [fact-sealevel](https://github.com/fact-sealevel) | [Fill in release/commit and citation] |
| FaIR 2 | Climate temperature trajectories | [FaIR repository](https://github.com/OMS-NetZero/FAIR) | [Fill in release and DOI] |
| DeConto et al. (2021) inputs | Antarctic temporal correlations | Within FACTS2 data module | [Fill in exact dataset identifier] |
| Fitted ISMIP6 | Greenland temporal correlations | [Link to exact implementation/data] | [Fill in version/citation] |
| Python scientific stack | Analyses and figures | See [`requirements.txt`](requirements.txt) | Versions were not pinned from the historical production environment |

## Reproduce my experiment

### Repository layout

```text
scripts/
  original_sej/       Original expert analysis, distributions and figures
  interpolation/      Inputs, scenario selection, coupling, annual projections, tables and figures
  shared/             Shared scientific labels, settings and path configuration
  runners/            Two documented execution entry points
figures/              Figure archive guidance (generated files not yet included)
data/                 External-data staging instructions (large files excluded)
docs/                 Figure manifest, portability notes and workflow documentation
output/               Local generated products, ignored by Git
```

**The two workflows are distinct:**
`original_sej/` operates on pre-generated expert judgement / Monte Carlo input data; it does **not** regenerate the elicitation or original draws.
`interpolation/` starts from those Monte Carlo draws and externally generated FaIR/FACTS2 inputs to construct scenario-based, time-dependent projections.

### 1. Install Python dependencies

Use a fresh Python environment, then install this repository's direct Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
python -m pip install -r requirements.txt
```

`requirements.txt` lists the imported packages without pinned versions: the historical production environment has not been captured here. Exact versions and upstream software installations should be documented before a reproducibility release.

### 2. Acquire external inputs and configure locations

The directory names and required source filenames are encoded in `scripts/shared/core_config.py` and checked by `scripts/interpolation/inputs/00_check_inputs.py`.

```bash
# From this repository root; substitute locations on your own system.
export SEJ_DATA_DIR="/absolute/path/to/external/sej-data"
export SEJ_FACTS2_ROOT="/absolute/path/to/FACTS2"
export SEJ_CORE_OUTPUT_ROOT="$PWD/output/reproduction_run_01"
# Optional: leave SEJ_PROJECT_DIR unset to resolve the repository root automatically.
```

A valid `SEJ_DATA_DIR` contains `sej2025_SplitPanels_liz_updated.csv`, `historical_observations/imbie_2023/`, `fair_output/`, and `temporal_correlation_matrices/GrIS_FittedISMIP_scenario_specific/`. A valid `SEJ_FACTS2_ROOT` contains the referenced emulandice experiments, FaIR2 GSAT experiment outputs, and DeConto21 model inputs. The complete checks also inspect pathway-specific files.


### 3. Inspect and execute the interpolation chain

The runner starts in preview mode and prevents writing outside this repository's `output/` tree. It also refuses to overwrite previously generated stage results.

```bash
python scripts/runners/RUN_CORE_METHOD_CHAIN.py --list
python scripts/runners/RUN_CORE_METHOD_CHAIN.py --stages 00 --dry-run
python scripts/runners/RUN_CORE_METHOD_CHAIN.py --stages 00 --execute
python scripts/runners/RUN_CORE_METHOD_CHAIN.py --stages 01 --execute
```

Select stage codes with `--stages` (comma-separated). Follow the runner's `--list` dependency information. Stage descriptions:

| Stage | Purpose |
| --- | --- |
| `00` | Check external scientific input files |
| `01`, `01B` | CMIP6 and CMIP7 FaIR-to-SEJ scenario selection |
| `02` | Temporal coupling with 2050 emulandice and multi-year targets |
| `03`, `03B` | CMIP6 and CMIP7 annual trajectories |
| `04` | Manuscript tables |
| `05` | Formatted manuscript and supplementary tables; also requires `TABLE_OUTPUT_ROOT` under `output/` |


### 4. Reproduce original SEJ analyses

The `scripts/original_sej/` scripts generate the original expert performance, driver, distribution, warming-reference and other figures from externally supplied elicitation/Monte Carlo files. Use the figure runner below or inspect each script's input configuration. Original expert judgement input creation is outside the scope of this repository.


## Reproduce my figures

Use the named figure groups rather than executing every plotting file indiscriminately:

```bash
python scripts/runners/RUN_FINAL_FIGURES.py --list
python scripts/runners/RUN_FINAL_FIGURES.py \
  --only warming --execute \
  --output-root "$PWD/output/reproduced_figures_01"
```

The figure runner additionally requires `SEJ_CORE_OUTPUT_ROOT` to be under this repository's `output/`; configure it in the step above. Figures reading core interpolated trajectories require their upstream stages to have completed. The `consolidated_legacy` figure suite is opt-in, not part of the default validated group set.

For specific original and interpolation manuscript figure assignments, see [`docs/FIGURE_SCRIPT_MANIFEST.csv`](docs/FIGURE_SCRIPT_MANIFEST.csv) and [`docs/FIGURE_INVENTORY.md`](docs/FIGURE_INVENTORY.md). Final numbering/archived figure image files should be updated when the manuscript is frozen. The [IMMM-SFA metarepo guidelines](https://immm-sfa.github.io/metarepo/) recommend including final figure image files alongside the generation code; these are not yet included in this pre-release snapshot.

## Citation, license and release checklist

- [ ] Finalize manuscript title, abstract, authors, journal details, and paper DOI.
- [ ] Verify the study data release, filenames, licenses, and citations (including IMBIE, fitted ISMIP6, DeConto21, FACTS2, FaIR 2, and IPCC AR6).
- [ ] Record exact Python/package versions and upstream software releases used for manuscript output.
- [ ] Confirm authorship metadata, ORCIDs if available, and the intended code license; update `CITATION.cff` and add `LICENSE` once approved.
- [ ] Archive manuscript figure files in `figures/` with a definitive figure-to-script manifest.
- [ ] Publish the GitHub repository and mint a **separate code DOI** via a versioned Zenodo GitHub release; do not use the study-data DOI as a code-release DOI.
- [ ] Cite both the appropriate data release and the metarepo software release in the paper's Data Availability/Code Availability sections.
