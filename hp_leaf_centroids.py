#!/usr/bin/env python3
import numpy as np
from astropy.table import Table
from spectral_cube import SpectralCube
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------
# Load catalog
# ----------------------------------------------------
catalog = Table.read("leaf_catalog_kurima_f.fits")
print(f"📦 Loaded catalog with {len(catalog)} leaves.")

# Prepare output columns if not present
for col in ["cen_l", "cen_b", "cen_v"]:
    if col not in catalog.colnames:
        catalog[col] = np.full(len(catalog), np.nan)


# ----------------------------------------------------
# Loop over CHIMPS2 regions 14–46
# ----------------------------------------------------
for region in range(14, 47):

    idx_region = np.where(catalog["REGION"] == region)[0]
    n_leaves = len(idx_region)

    print(f"\n➡ REGION {region}: {n_leaves} leaves")

    if n_leaves == 0:
        continue

    try:
        asgn = SpectralCube.read(f"leaves_catalog_clean_{region}.fits")
        emis = SpectralCube.read(f"region{region}_smooth.fits")
    except Exception as e:
        print(f"❌ Skipping region {region}: {e}")
        continue

    # Extract raw data arrays (fast)
    asgn_data = asgn._data.astype(np.int32)
    emis_data = np.nan_to_num(emis.filled_data[:].value, nan=0.0)

    # Precompute coordinate grids (only once per region)
    ypix, xpix = np.indices(asgn_data.shape[1:])
    vel_axis = emis.spectral_axis.to("km/s").value

    # -----------------------------------------------
    # Compute flux-weighted centroid for each leaf
    # -----------------------------------------------
    for idx in idx_region:

        leaf_id = catalog["LEAF_ID"][idx]
        mask = asgn_data == leaf_id

        if not np.any(mask):
            continue

        flux = emis_data * mask
        total_flux = flux.sum()

        if total_flux == 0:
            continue

        # ----- Velocity centroid -----
        flux_v = flux.sum(axis=(1, 2))
        v_centroid = np.sum(flux_v * vel_axis) / np.sum(flux_v)

        # ----- Spatial centroid -----
        flux_xy = flux.sum(axis=0)
        xcen = np.sum(flux_xy * xpix) / np.sum(flux_xy)
        ycen = np.sum(flux_xy * ypix) / np.sum(flux_xy)

        # Convert pixels → WCS (Galactic lon/lat)
        skycoord = emis.wcs.celestial.pixel_to_world(xcen, ycen)

        catalog["cen_l"][idx] = skycoord.l.deg
        catalog["cen_b"][idx] = skycoord.b.deg
        catalog["cen_v"][idx] = v_centroid


# ----------------------------------------------------
# Save output
# ----------------------------------------------------
outname = "leaf_catalog_with_PPV_centroids.fits"
catalog.write(outname, overwrite=True)

print(f"\n🎉 Finished — PPV centroids saved to: {outname}\n")
