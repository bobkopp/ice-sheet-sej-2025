# Working figure inventory (2026-10-26)

This is an integration inventory, not a claim that all figures have been executed or visually validated. Captions are intentionally out of scope.

| Figure | Script | Status |
|---|---|---|
| Main 1: warming scenarios | `warming_scenario_plot.py` | Added; test pending |
| Main 2: SA vs informativeness/MAPE | `SA_Inf_MAPE_fig.py` | Added; test pending |
| Main 3: year/warming clustered SLR | `SLRxYEARxwarminglevel_Cluster_fig.py` | Added; test pending |
| Main 4 and 5: 2100/2300 B19–B25 split violins | `split_violin_plot_adjustedB19_B25.py` | Added; test pending |
| Main 6: expert assessments of drivers | `drivers_scaled.py` | Added; test pending |
| Supplemental: 2200 split violin | `split_violin_plot_adjustedB19_B25.py` | Same script; test pending |
| Supplemental: 2100/2200/2300 PDFs | `PDF_individual_B19_comparison_all.py` | Added; test pending |
| Supplemental: expert judgments/calibration | `plot_expert_judgments_baselines_ranked.py` | Added; test pending |
| Specifically requested four-SSP AR6 style | `PLOT_CMIP6_FOUR_SSP_AR6_STYLE_2150_2300_SHARED_Y_left150_zero_aligned_across_columns.py` | Added explicitly; only input/output roots adapted; test pending |
| CMIP6/CMIP7 time series, warming grids, AR6-style figures | `MASTER_FAIR2_ICE_SHEET_FIGURES_CONSOLIDATED.py` | Already present, execution passed; visual diff pending |
| Correlation, realized fractions, random trajectories, diagnostics plots | Various | Excluded from default figure plan; scientific diagnostics retained |

**Shared dependency:** `B25data_processing.py` is added unchanged. Continue using existing `figure_labels.py`, `figure_settings.py`, `figure_helpers.py`, and `figure_style.py`, rather than replacing the versions adapted in your local working copy.

Do not replace your validated `RUN_FINAL_FIGURES.py` or any scripts from stages 00–05. This addition is additive. Some newly added scripts retain older implicit inputs through shared figure settings; verify inputs and output roots before execution. The historical `figure_settings.py` supplied in the ZIP has deliberately NOT been added.
