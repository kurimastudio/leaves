#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from astropy.table import Table
from astropy.io import fits
from spectral_cube import SpectralCube
import astropy.units as u

# ============================
# LOAD CATALOG
# ============================
cat = Table.read("leaf_catalog_with_dist_and_mass.fits")

# Allocate arrays
sigma_v = np.full(len(cat), np.nan)
v_cent = np.full(len(cat), np.nan)

# ============================
# LOOP OVER REGIONS
# ============================
regions = np.unique(cat["REGION"])

for reg in regions:
    print(f"\n🔹 REGION {reg}: computing σ_v and v₀")

    # === Load smoothed emission cube once ===
    em_name = f"region{reg}_smooth.fits"
    try:
        cube = SpectralCube.read(em_name).with_spectral_unit(u.km / u.s)
    except:
        print(f"  ⚠ Missing cube {em_name}")
        continue

    # ===== COMPUTE MOMENTS (only once per region) =====
    M0 = cube.moment(order=0).value  # ∫I dv
    M1 = cube.moment(order=1).value  # intensity-weighted velocity
    LWS = cube.linewidth_sigma().value  # per-pixel σ_v

    # clean NaNs
    for arr in (M0, M1, LWS):
        arr[np.isnan(arr)] = 0.0

    # === Load leaf assignment cube ===
    leaf_asgn_name = f"leaves_catalog_clean_{reg}.fits"
    try:
        labels = fits.getdata(leaf_asgn_name).astype(np.int32)
    except:
        print(f"  ⚠ Missing leaf assignments {leaf_asgn_name}")
        continue

    # === Find leaves belonging to this region ===
    idxs = np.where(cat["REGION"] == reg)[0]

    # ============================
    # ULTRA-FAST LEAF LOOP
    # ============================
    for idx in idxs:
        leaf_id = cat["LEAF_ID"][idx]

        # Collapse along velocity to a 2D footprint
        mask2d = (labels == leaf_id).any(axis=0)
        if not mask2d.any():
            continue

        w = M0[mask2d]  # weight
        if w.sum() <= 0:
            continue

        # ===== velocity centroid =====
        vmap = M1[mask2d]
        v_cent[idx] = np.sum(vmap * w) / np.sum(w)

        # ===== linewidth (sigma) =====
        lwmap = LWS[mask2d]
        sigma_v[idx] = np.sum(lwmap * w) / np.sum(w)

# ============================
# SAVE
# ============================
cat["sigma_v"] = sigma_v
cat["v_centroid"] = v_cent
cat.write("leaf_catalog_with_dist_mass_sigma_v0.fits", overwrite=True)

print("\n🎉 DONE → leaf_catalog_with_dist_mass_sigma_v0.fits\n")
