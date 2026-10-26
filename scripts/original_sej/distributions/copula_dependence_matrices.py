#!/usr/bin/env python3
"""Ice-sheet rank scatterplot matrices, with multiple layout choices.

Edit PATHS / SETTINGS below.
Creates the selected compact layout: standalone 2100 and 2300 figures
and a combined vertical figure. All subplot borders have major/minor tick marks.

Each Monte Carlo sample is transformed to its normalized rank,
so both axes range from 0 to 1.

Axes: labels ONLY on the leftmost column (y) and bottom row (x) of EACH matrix.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # works from IDEs and saves figures without opening windows
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

# ============================= EDIT THESE =============================
if not os.environ.get("FIG_OUTPUT_DIR"):
    raise RuntimeError("Set FIG_OUTPUT_DIR to an isolated output directory first")
PROJECT_ROOT = Path(os.environ.get("FIG_PROJECT_DIR", os.environ.get("SEJ_PROJECT_DIR", Path(__file__).resolve().parents[3]))).expanduser().resolve()
INPUT_CSV = Path(os.environ.get("FIG_B25_MC_FILE", str(Path(os.environ.get("SEJ_DATA_DIR", PROJECT_ROOT / "data")) / "sej2025_SplitPanels_liz_updated.csv"))).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("SEJ_FIGURE_GROUP_OUTPUT") or (Path(os.environ["FIG_OUTPUT_DIR"]) / "supplementary" / "copula_matrices")).expanduser().resolve()
N_POINTS = 10_000
SEED = 1234
DOT_SIZE = 1.0
DOT_ALPHA = 0.70

VERSIONS = ('balanced',)  # Selected layout
SAVE_PDF = False
SAVE_COMPARISON = False
# =====================================================================

SCENARIOS = (
    ('2_21', '2°C warming in 2100', ('E2_21', 'G2_21', 'W2_21', 'SLR2_21')),
    ('5_23', '5°C warming in 2300', ('E5_23', 'G5_23', 'W5_23', 'SLR5_23')),
)
LABELS = ('EAIS', 'GrIS', 'WAIS', 'Total IS SLR')

# Dimensions are fractions of the figure, except figsize in inches.
# Matrix positions are calculated explicitly instead of relying on gridspec's
# title layout, so titles and tick labels cannot collide between scenarios.
STYLES = {
    'compact':  dict(gap_x=.036, gap_y=.018, matrix_height=.390, block_gap=.108,
                     single_size=(9.4, 9.4), combined_size=(9.4, 18.5)),
    'balanced': dict(gap_x=.044, gap_y=.022, matrix_height=.388, block_gap=.075,
                     single_size=(9.4, 9.4), combined_size=(9.4, 18.5)),
    'spacious': dict(gap_x=.055, gap_y=.027, matrix_height=.384, block_gap=.120,
                     single_size=(9.4, 9.4), combined_size=(9.4, 18.5)),
}


def get_ranks(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f'Missing columns: {missing}\nAvailable: {df.columns.tolist()}'
        )

    values = df.loc[:, cols].apply(
        pd.to_numeric, errors='coerce'
    ).dropna()

    if len(values) < 2:
        raise ValueError(f'Not enough valid paired rows for {cols}')

    if len(values) > N_POINTS:
        values = values.sample(n=N_POINTS, random_state=SEED)

    n = len(values)

    # Normalize ranks to [0, 1] for empirical copula plots.
    # All variables retain their original paired sample relationships.
    ranks = values.rank(method='first').to_numpy(dtype=float) / n

    return ranks, n


def style_axis(ax, row, col, n):
    # Empirical copula axes run from 0 to 1.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Major ticks and labels
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

    # Minor ticks (unlabeled)
    ax.xaxis.set_minor_locator(FixedLocator([0.25, 0.75]))
    ax.yaxis.set_minor_locator(FixedLocator([0.25, 0.75]))

    # Display 0, 0.5, 1 without unnecessary decimal places.
    fmt = FuncFormatter(lambda val, _: f'{val:g}')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    # Tick marks on all four sides of every panel.
    # Numeric labels only on the left column and bottom row.
    ax.tick_params(
        axis='both',
        which='both',
        top=True,
        right=True,
        bottom=True,
        left=True,
        labeltop=False,
        labelright=False,
        labelbottom=(row == 3),
        labelleft=(col == 0),
        direction='out',
        labelsize=8,
        pad=4
    )

    # Tick lengths
    ax.tick_params(which='major', length=6.5, width=0.9)
    ax.tick_params(which='minor', length=4, width=0.8)

    # Normal-weight tick labels
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight('normal')

    for spine in ax.spines.values():
        spine.set_linewidth(0.95)


def add_matrix(fig, ranks, n, title, *, left, bottom, width, height, gap_x, gap_y):
    """Draw matrix within an exactly-defined bounding rectangle."""
    cell_w = (width - 3 * gap_x) / 4
    cell_h = (height - 3 * gap_y) / 4
    for row in range(4):
        for col in range(4):
            x = left + col * (cell_w + gap_x)
            y = bottom + (3 - row) * (cell_h + gap_y)
            ax = fig.add_axes([x, y, cell_w, cell_h])
            if row == col:
                ax.text(.5, .5, LABELS[row], transform=ax.transAxes,
                        ha='center', va='center', fontsize=13)
            else:
                ax.scatter(ranks[:, col], ranks[:, row], s=DOT_SIZE,
                           c='black', alpha=DOT_ALPHA, linewidths=0,
                           rasterized=True)
            style_axis(ax, row, col, n)
    # Figure text is anchored to the matrix bounds rather than any square.
    fig.text(left + width / 2, bottom + height + .030,
             title, ha='center', va='bottom', fontsize=15, fontweight='bold')


def save(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / filename
    fig.savefig(png, dpi=240, facecolor='white')
    if SAVE_PDF:
        fig.savefig(png.with_suffix('.pdf'), facecolor='white')
    print('Saved:', png)
    plt.close(fig)
    return png


def single_figure(title, ranks, n, style, filename):
    fig = plt.figure(figsize=style['single_size'], facecolor='white')
    # Generous margins for labels, title, and square appearance.
    add_matrix(fig, ranks, n, title,
               left=.12, bottom=.105, width=.82, height=.79,
               gap_x=style['gap_x'], gap_y=style['gap_y'])
    return save(fig, filename)


def combined_figure(data, style, filename):
    fig = plt.figure(figsize=style['combined_size'], facecolor='white')
    # Independent well-separated blocks. Each matrix has its own LEFT and
    # BOTTOM tick labels; the two title baselines are explicitly separated.
    left, width = .12, .82
    matrix_height = style['matrix_height']
    bottom_lower = .055
    bottom_upper = bottom_lower + matrix_height + style['block_gap']
    for key, title, _ in SCENARIOS:
        bottom = bottom_upper if key == '2_21' else bottom_lower
        ranks, n = data[key]
        add_matrix(fig, ranks, n, title,
                   left=left, bottom=bottom, width=width, height=matrix_height,
                   gap_x=style['gap_x'], gap_y=style['gap_y'])
    return save(fig, filename)


def comparison_sheet(png_paths):
    """Combine the three finished vertically-stacked versions side-by-side."""
    from PIL import Image, ImageOps, ImageDraw
    photos = []
    for label, path in png_paths:
        with Image.open(path) as im:
            im = im.convert('RGB')
            im.thumbnail((640, 1350))
            tile = Image.new('RGB', (660, 1410), 'white')
            x = (660 - im.width) // 2
            tile.paste(im, (x, 45))
            ImageDraw.Draw(tile).text((20, 14), label.upper(), fill='black')
            photos.append(tile)
    if not photos:
        return
    result = Image.new('RGB', (660 * len(photos), 1410), 'white')
    for i, im in enumerate(photos):
        result.paste(im, (660 * i, 0))
    dest = OUTPUT_DIR / 'compare_layouts.png'
    result.save(dest)
    print('Saved:', dest)


def main():
    df = pd.read_csv(INPUT_CSV)
    data = {key: get_ranks(df, cols) for key, _, cols in SCENARIOS}
    comparisons = []
    for variant in VERSIONS:
        style = STYLES[variant]
        for key, title, _ in SCENARIOS:
            ranks, n = data[key]
            single_figure(title, ranks, n, style, f'correlations_{key}_{variant}.png')
        combined = combined_figure(data, style, f'correlations_stacked_{variant}.png')
        comparisons.append((variant, combined))
    if SAVE_COMPARISON:
        comparison_sheet(comparisons)
    print('Finished compact layout.')


if __name__ == '__main__':
    main()
