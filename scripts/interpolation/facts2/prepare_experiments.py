#!/usr/bin/env python3
"""
Prepare local FACTS2 experiment configurations.

Created on: 2026-10-27
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)

Expands FACTS2_ROOT in archived YAML files and writes runnable copies
into the local FACTS2 experiments directory.
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--facts2-root",
        type=Path,
        default=Path.home() / "FACTS2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory for prepared experiment files.",
    )
    args = parser.parse_args()

    facts2_root = args.facts2_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_root = Path(__file__).resolve().parent

    if not facts2_root.is_dir():
        parser.error(f"FACTS2 directory does not exist: {facts2_root}")

    for category in ("fair2", "emulandice"):
        for experiment in sorted((source_root / category).iterdir()):
            if not experiment.is_dir():
                continue

            for name in ("experiment-config.yaml", "experiment-compose.yaml"):
                src = experiment / name
                if not src.is_file():
                    raise FileNotFoundError(src)

                text = src.read_text(encoding="utf-8")
                text = text.replace("${FACTS2_ROOT}", str(facts2_root))

                destination = output_root / category / experiment.name / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text, encoding="utf-8")

                print(f"Wrote: {destination}")


if __name__ == "__main__":
    main()
