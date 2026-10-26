# Paths and portability

The default project directory is the root of this standalone repository, located from the script path. Set `SEJ_PROJECT_DIR` to override it when working with a separate project/data directory.

- `SEJ_DATA_DIR`: externally supplied SEJ Monte Carlo data and supporting inputs; defaults to `<project>/data`.
- `SEJ_FACTS2_ROOT`: FACTS2 checkout and experiment outputs; defaults to `~/FACTS2`.
- `SEJ_CORE_OUTPUT_ROOT`: computational stage output directory; defaults to `<project>/output/CORE_METHOD_CHAIN_02I_B_FINAL_TENSION_P003_FAIR2`.
- `FIG_DATA_DIR`: figure input-data override; if absent follows `SEJ_DATA_DIR`.
- `FIG_OUTPUT_DIR`, `FIG_FAIR2_OUTPUT_DIR`: figure output directories.
- `TABLE_OUTPUT_ROOT`: formatted manuscript table output directory.

The core runner requires an explicit isolated `SEJ_CORE_OUTPUT_ROOT` inside this repository's `output/` for `--execute`, and Stage 05 additionally requires an isolated `TABLE_OUTPUT_ROOT`. This deliberate safety restriction prevents accidental overwrite of production outputs. Figure execution likewise requires `SEJ_CORE_OUTPUT_ROOT` within the repository's `output/`. See runner `--help` for further options.

The Monte Carlo data were prepared externally; this repository does not regenerate them. FACTS2, FaIR, emulandice, IMBIE, and temporal-correlation inputs remain external dependencies.
