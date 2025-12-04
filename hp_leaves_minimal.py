#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.table import Table
from spectral_cube import SpectralCube
import numpy as np

# ============================
# Collect rows here
# ============================
rows = []

for i in range(14, 47):  # regions 14 .. 46
    print(f"\n🔹 Processing REGION {i}")

    # ---------------------------------
    # 1) Load cubes (clean cloud + leaves)
    # ---------------------------------
    try:
        cloud = SpectralCube.read(f"clouds_catalog_cube_{i}.fits")
    except:
        print(f"⛔ Missing clouds_catalog_cube_{i}.fits, skipping.")
        continue

    try:
        leaves = SpectralCube.read(f"region{i}_snr_ultra_5_leaf_asgn.fits")
    except:
        print(f"⛔ Missing leaves_region_{i}.fits, skipping.")
        continue

    cloud_data = cloud._data.astype(int)
    leaf_data = leaves._data.astype(int)

    # ---------------------------------
    # 2) Mask leaves outside valid clouds (only once!)
    # ---------------------------------
    leaf_data[cloud_data == -1] = -1

    # ---------------------------------
    # 3) Valid leaf IDs
    # ---------------------------------
    leaf_ids = np.unique(leaf_data)
    leaf_ids = leaf_ids[leaf_ids >= 0]  # remove –1

    if len(leaf_ids) == 0:
        print(f"⚠ No valid leaves in region {i}.")
        continue

    # ---------------------------------
    # 4) Determine parent cloud per leaf (fast)
    # ---------------------------------
    valid_vox = leaf_data >= 0
    leaf_vals = leaf_data[valid_vox]
    cloud_vals = cloud_data[valid_vox]

    for leaf_id in leaf_ids:
        # voxels where this leaf is present
        parent_clouds = cloud_vals[leaf_vals == leaf_id]

        if len(parent_clouds) == 0:
            continue

        # majority vote → dominant parent cloud
        parent = np.bincount(parent_clouds).argmax()

        # save minimal record
        rows.append((i, leaf_id, parent))

    print(f"✔ REGION {i}: {len(leaf_ids)} leaves linked.")


# ============================
# Final Table & Write
# ============================
print("\n💾 Saving FITS catalog...")

leaf_table = Table(rows=rows, names=("REGION", "LEAF_ID", "CLOUD_ID"))

leaf_table.write("leaf_catalog_minimal.fits", overwrite=True)
print("🎉 DONE — saved as leaf_catalog_minimal.fits\n")
