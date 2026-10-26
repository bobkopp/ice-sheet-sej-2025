# Working figure inventory (2026-10-26)

| Figure | Script | Status |
|---|---|---|
| Figure 1: warming scenarios | `warming_scenario_plot.py` | Added; test pending |
| Figure 2: SA vs informativeness/MAPE | `SA_Inf_MAPE_fig.py` | Added; test pending |
| Figure 3: year/warming clustered SLR | `SLRxYEARxwarminglevel_Cluster_fig.py` | Added; test pending |
| Figures 4 and 5: 2100/2300 B19–B25 split violins | `split_violin_plot_adjustedB19_B25.py` | Added; test pending |
| Figure 6: SSP ARG style | `PLOT_CMIP6_FOUR_SSP_AR6_STYLE_2150_2300_SHARED_Y_left150_zero_aligned_across_columns.py` | Added; test pending |
| Figure 7: expert assessments of drivers | `drivers_scaled.py` | Added; test pending |
| Figure S1: dependence structure copula | `copula_dependence_matrices.py` | Added; test pending |
| Figures S2 and S3: scenario-selection probabilities | `scenario_selection_probabilities.py` | Added; test pending |
| Figures S4-S6: 2100/2200/2300 PDFs | `PDF_individual_B19_comparison_all.py` | Added; test pending |
| Figure S7: 2200 split violin | `split_violin_plot_adjustedB19_B25.py` | Added; test pending |
| CMIP6/CMIP7 time series, warming grids, AR6-style figures | `MASTER_FAIR2_ICE_SHEET_FIGURES_CONSOLIDATED.py` | Already present, execution passed; visual diff pending |
| Figures S18-S54: expert judgments/calibration | `plot_expert_judgments_baselines_ranked.py` | Added; test pending |
| Figure S55: internal variability v. external forcing | `mass_change_obs.py` | Added; test pending |

**Shared dependency:** `B25data_processing.py` is added unchanged. Continue using existing `figure_labels.py`, `figure_settings.py`, `figure_helpers.py`, and `figure_style.py`.