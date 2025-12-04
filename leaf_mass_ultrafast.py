#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from astropy.table import Table
from astropy.io import fits
import astropy.units as u

# ============================
# CONSTANTS
# ============================
mu = 1.36  # mean molecular weight factor
mH = 1.6735575e-24  # hydrogen mass [g]
Msun = 1.98847e33  # solar mass [g]

# Milam et al. (2005) 12C/13C Gradient
A, B = 6.21, 18.71  # best fit
sA, sB = 1.00, 1.87  # 1σ uncertainties

# ============================
# LOAD CATALOG WITH DISTANCES
# ============================
t2 = Table.read("leaf_catalog_with_distances.fits")

# Prepare output arrays
mass_leaf = np.zeros(len(t2))
mass_err = np.zeros(len(t2))

# ============================
# GROUP LEAVES BY REGION
# ============================
regions = np.unique(t2["REGION"])

for reg in regions:
    print(f"\n🔹 REGION {reg} -> computing leaf masses")

    # === Load cloud assignment cube (integer labels) ===
    lab = fits.getdata(f"leaves_catalog_clean_{reg}.fits").astype(int)
    # === Load corresponding 13CO column density cube ===
    cd = fits.getdata(f"CHIMPS_13CO_G0{reg}_colte_sm19_N13.fits")
    cd[np.isnan(cd)] = 0.0

    # === Pixel area (steradians) from header ===
    hd = fits.getheader(f"CHIMPS_13CO_G0{reg}_colte_sm19_N13.fits")
    cdelt_rad = abs(hd["CDELT2"]) * np.pi / 180.0
    pix_area_sr = cdelt_rad**2  # small-angle approx

    # === Get array indices for leaves in this region ===
    idxs = np.where(t2["REGION"] == reg)[0]

    # ==============================
    # 🔁 LOOP OVER LEAVES IN REGION
    # ==============================
    for idx in idxs:
        leaf_id = t2["_idx"][idx]  # cloud ID of leaf
        Rgal = t2["galcen_distance"][idx]  # kpc
        D_kpc = t2["distance"][idx]  # kpc

        # --- Mask for this leaf ---
        mask = lab == leaf_id

        if not np.any(mask):
            mass_leaf[idx] = np.nan
            mass_err[idx] = np.nan
            continue

        # --- Sum 13CO column density in leaf ---
        Nsum = np.sum(cd[mask])

        # --- Abundance gradient (Milam+05) ---
        denom = A * Rgal + B
        X13 = 1e-4 / denom  # 13CO/H2 abundance
        NH2 = Nsum / X13  # convert to H2 column

        # --- Distance → cm ---
        D_cm = (D_kpc * u.kpc).to(u.cm).value

        # ====================
        # 💠 TOTAL MASS
        # ====================
        mass = mu * mH * (D_cm**2) * (NH2 * pix_area_sr)
        mass_leaf[idx] = mass / Msun

        # ====================
        # 📉 UNCERTAINTY (from abundance gradient)
        # ====================
        sigma_X = 1e-4 * np.sqrt((Rgal * sA) ** 2 + sB**2) / (denom**2)
        frac_X = sigma_X / X13  # fractional error propagates to mass

        mass_err[idx] = mass_leaf[idx] * frac_X

# ============================
# SAVE OUTPUT
# ============================
t2["mass"] = mass_leaf
t2["mass_err"] = mass_err
t2.write("leaf_catalog_with_dist_and_mass.fits", overwrite=True)

print("\n🎉 DONE → leaf_catalog_with_dist_and_mass.fits\n")
