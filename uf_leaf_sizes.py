from astropy.io import fits
from astropy.table import Table
import numpy as np

# === Load the master leaf catalog ===
leafcat = Table.read("leaf_catalog_kurima_f.fits")

# Prepare new columns
Lsize_px = np.zeros(len(leafcat), dtype=int)
Bsize_px = np.zeros(len(leafcat), dtype=int)
Vsize_px = np.zeros(len(leafcat), dtype=int)

print("Computing CHIMPS2 leaf L/B/V sizes...")

# === Loop through regions ===
for REG in range(14, 47):

    print(f"Processing REGION {REG} ...")

    # Load corresponding leaf assignment cube
    try:
        cube = fits.getdata(f"leaves_catalog_clean_{REG}.fits")
    except FileNotFoundError:
        print(f"  ⚠ REGION {REG}: leaf cube not found — skipping.")
        continue

    # Select leaves belonging to this region
    mask_region = leafcat["REGION"] == REG
    leaf_ids = leafcat["LEAF_ID"][mask_region]

    # Loop over the leaves in this region
    for leaf_id in leaf_ids:

        idx = np.where((leafcat["REGION"] == REG) & (leafcat["LEAF_ID"] == leaf_id))[0][
            0
        ]

        # Boolean mask of voxels belonging to this leaf
        vox = cube == leaf_id

        if not np.any(vox):
            # No voxels — assign zero
            Lsize_px[idx] = 0
            Bsize_px[idx] = 0
            Vsize_px[idx] = 0
            continue

        # Get voxel coordinates (v, b, l)
        v_idx, b_idx, l_idx = np.where(vox)

        # Extents in pixels
        Lsize_px[idx] = l_idx.max() - l_idx.min() + 1
        Bsize_px[idx] = b_idx.max() - b_idx.min() + 1
        Vsize_px[idx] = v_idx.max() - v_idx.min() + 1

# Add columns to catalog
leafcat["Lsize_px"] = Lsize_px
leafcat["Bsize_px"] = Bsize_px
leafcat["Vsize_px"] = Vsize_px

leafcat.write("leaf_catalog_with_LBV_sizes.fits", overwrite=True)
print("✅ Saved: leaf_catalog_with_LBV_sizes.fits")
