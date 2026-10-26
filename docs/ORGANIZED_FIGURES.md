# Figure output organization

The unified runner is `scripts/RUN_FINAL_FIGURES.py`. The original computational pipeline is untouched; figures are generated into a fresh `output/metarepo_figures_organized/` root by default.

- `main_text/figure_01_warming/`
- `main_text/figure_02_performance/`
- `main_text/figure_03_slr_by_year/`
- `main_text/figures_04_05_violin/`
- `main_text/figure_06_drivers/`
- `supplementary/b19_b25_pdfs/`
- `supplementary/expert_judgments_calibration/`
- `supplementary/mass_change_observations/`
- `supplementary/copula_dependence/`
- `supplementary/scenario_selection_probabilities/`
- `supplementary/four_ssp_ar6_style/`

Some generators retain a meaningful leaf folder such as `figures/`, `tables/`, `b19_comparisons/`, `process_violin/`, or `AR6_style_four_SSPs/`. When run through the unified runner, redundant numbered folder prefixes are suppressed by `figure_settings.output_for`. Standalone runs without the runner preserve the old paths.

Legacy `MASTER_FAIR2_ICE_SHEET_FIGURES_CONSOLIDATED.py` is available as **`consolidated_legacy`**, not enabled by default: it currently generates additional diagnostic, realized-fraction, and related output groups. Further selective refactoring would be required to include just its manuscript subsets. The original script remains unchanged except for its descriptive module header.

The figure filenames and calculation logic are not intentionally changed. The main grouping is by scientific role, and a fresh output root prevents overwrites. The old `output/metarepo_test_main_figures/` files are **not moved or deleted**.

## Commands

From `ice_sheet_sej_metarepo_starter`:

```bash
source test_environment.sh
python scripts/RUN_FINAL_FIGURES.py --list
python scripts/RUN_FINAL_FIGURES.py --only warming,performance --dry-run
python scripts/RUN_FINAL_FIGURES.py --only warming --execute
```

`--execute` is mandatory; otherwise no figure scripts run. By default, all 11 approved groups are selected. Running to a nonempty group destination requires `--allow-overwrite`; preferably use `--output-root` with a new isolated folder. Legacy consolidated suite runs only if explicitly specified using `--only consolidated_legacy`.