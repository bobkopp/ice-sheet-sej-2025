# Scientific inputs (external)

Do not commit large scientific data here. Set `SEJ_DATA_DIR` to a separately downloaded or archived data folder. The full input inventory and source references are described in the top-level [`README.md`](../README.md).

The Stage 00 input check expects, among others:

```text
$SEJ_DATA_DIR/
  sej2025_SplitPanels_liz_updated.csv
  historical_observations/imbie_2023/
    imbie3_east_antarctica_mm_partitioned.csv
    imbie3_west_antarctica_mm_partitioned.csv
    imbie3_greenland_mm_partitioned.csv
  fair_output/ar7.*.temperature.fair.temperature_gsat.nc
  temporal_correlation_matrices/GrIS_FittedISMIP_scenario_specific/
```

Additional input products come from `SEJ_FACTS2_ROOT`, including FaIR2 GSAT trajectories, DeConto21 ensembles and emulandice module outputs. The user should obtain these from the appropriate credited software/data releases; the repository does not redistribute them.
