#!/usr/bin/env python3
"""CMIP7 annual trajectories using the selected 02I-B coupled panels and FaIR-ID CRN.

Run this script to regenerate CMIP7 sampled realizations, annual NetCDF and tables.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
import core_config as cfg
from core_annual_tension import crn_indices_by_sample_id, run_family

if not hasattr(cfg, 'CMIP7_CRN_SEED'):
    raise RuntimeError('This Stage 03B requires the bundled core_config.py defining CMIP7_CRN_SEED. Replace BOTH files.')

if __name__ == '__main__':
    print('CMIP7 CRN seed:', cfg.CMIP7_CRN_SEED)
    print('Running shared FaIR-sample-ID CRN sampler (separate warming streams)')
    run_family('CMIP7')
