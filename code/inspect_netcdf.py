#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:53:31 2026

@author: elizabeth
"""

from netCDF4 import Dataset
import numpy as np

path = (
    "output/fair_sej_interpolation/timeseries/"
    "fair_sej_timeseries.nc"
)

def decode(values):
    return np.array([
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in values
    ])

with Dataset(path, "r") as ds:
    print("\nFULL DATASET")
    print("============")
    print(ds)

    print("\nGLOBAL ATTRIBUTES")
    print("=================")
    for name in ds.ncattrs():
        print(f"{name}: {getattr(ds, name)}")

    print("\nDIMENSIONS")
    print("==========")
    for name, dimension in ds.dimensions.items():
        print(f"{name}: {len(dimension)}")

    print("\nVARIABLES")
    print("=========")
    for name, variable in ds.variables.items():
        print(
            f"{name}: "
            f"shape={variable.shape}, "
            f"dtype={variable.dtype}, "
            f"dimensions={variable.dimensions}"
        )

    for variable_name in [
        "scenario",
        "scenario_label",
        "response_code",
        "ice_sheet",
        "process",
    ]:
        if variable_name in ds.variables:
            values = decode(
                ds.variables[variable_name][:]
            )

            print(
                f"\nUNIQUE {variable_name.upper()} VALUES"
            )
            print("=" * (7 + len(variable_name)))
            print(np.unique(values))