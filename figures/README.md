# Manuscript figures

This directory is reserved for final, frozen manuscript and supplementary figure images. Figure-generation source code is under `scripts/original_sej/` and `scripts/interpolation/figures/`, with group entry points in `scripts/runners/RUN_FINAL_FIGURES.py`.

Run `python scripts/runners/RUN_FINAL_FIGURES.py --list` from the repository root to inspect supported figure groups. Set `SEJ_DATA_DIR`, `SEJ_FACTS2_ROOT` and `SEJ_CORE_OUTPUT_ROOT` first; the latter must be under the repository's `output/` folder. Choose a fresh `--output-root` when executing. Generated images belong in ignored `output/` until selected for a publication release.

See `docs/FIGURE_INVENTORY.md` and `docs/FIGURE_SCRIPT_MANIFEST.csv` for figure/script mappings. Before publication add the definitive figure images and revise `.gitignore` to allow them.
