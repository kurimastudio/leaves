#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.table import Table
from spectral_cube import SpectralCube
import numpy as np

leaf_catalog = Table.read("leaf_catalog_with_dist_mass_sigma_v0.fits")


# Load the leaf catalog we generated
leaf_catalog = Table.read("leaf_catalog_minimal.fits")

for i in range(14, 47):

    print(f"🔹 Cleaning leaf assignments for REGION {i}")

    # -------------------------------
    # Load leaf-assignment cube
    # -------------------------------
    try:
        leaf_cube = SpectralCube.read(f"region{i}_snr_ultra_5_leaf_asgn.fits")
    except:
        print(f"⛔ leaves_region_{i}.fits not found, skipping.")
        continue

    data = leaf_cube._data.astype(int)

    # -------------------------------
    # Extract valid leaves for this region
    # -------------------------------
    mask = leaf_catalog["REGION"] == i
    valid_ids = set(leaf_catalog["LEAF_ID"][mask])

    if len(valid_ids) == 0:
        print(f"⚠ No leaves in catalog for region {i} — all set to -1")
        data[:] = -1
    else:
        # Fast mask: keep only entries from valid_ids
        keep_mask = np.isin(data, list(valid_ids))
        data[~keep_mask] = -1

    # -------------------------------
    # Save cleaned cube
    # -------------------------------
    cleaned = SpectralCube(
        data=data,
        wcs=leaf_cube.wcs,
        meta=leaf_cube.meta,
        mask=leaf_cube.mask,
        header=leaf_cube.header,
    )

    outname = f"leaves_catalog_clean_{i}.fits"
    cleaned.write(outname, overwrite=True)

    print(f"✔ Saved {outname}")

print("\n🎉 DONE — All leaf cubes cleaned according to catalog.\n")
