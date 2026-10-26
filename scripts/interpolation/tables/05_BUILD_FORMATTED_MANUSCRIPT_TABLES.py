#!/usr/bin/env python3
"""Final formatted manuscript/SI tables, using CMIP6 and CMIP7 CRN100626 outputs.

Requires completed Stage 03, Stage 03B, and Stage 02I-B. Does not resample.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from make_cmip6_manuscript_tables import run as run_cmip6
from make_cmip7_si_long_tables import run as run_cmip7

if __name__ == "__main__":
    run_cmip6()
    run_cmip7()
    print("\nStage 05 formatted manuscript/SI tables completed.")
