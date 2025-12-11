#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hi-GAL matching and evolutionary classification for CHIMPS2 leaves.
"""

from astropy.table import Table
from astropy.io import fits
import numpy as np
import os

# -------------------------
# SETTINGS
# -------------------------
os.chdir("/home/yuwei/Data/bench")

LEAF_FILE = "leaf_catalog_kurima_f.fits"
HIGAL_FILE = "hg360c.fits"
OUTPUT_FILE = "leaf_catalog_kurima_f.fits"

# Evolution ranking (adjust if needed)
ACTIVE_STAGES = {"PROT", "YSO", "HII", "UCHII", "EMBEDDED"}  # editable

# -------------------------
# LOAD DATA
# -------------------------
leaf = Table(fits.getdata(LEAF_FILE, 1))
higal = Table(fits.getdata(HIGAL_FILE, 1))

# Ensure no NaNs in coordinates
mask = ~np.isnan(higal["glon"]) & ~np.isnan(higal["glat"]) & ~np.isnan(higal["vlsr"])
higal = higal[mask]

# -------------------------
# STORAGE ARRAYS
# -------------------------
L70, L160, L250, L350, L500, Tbol = [], [], [], [], [], []
evol_lists = []
category = []  # final column


# -------------------------
# CLASSIFICATION FUNCTION
# -------------------------
def classify_stage(flag_list):
    """
    Categorize leaves based on evolutionary composition.
    """
    flags = set(flag_list)

    if flags == {"NONE"} or len(flags) == 0:
        return "UNCLASSIFIED"

    if len(flags) == 1:
        return "SINGLE_PHASE"

    # If more than one type and contains early + late phase
    if flags & ACTIVE_STAGES:
        return "CLUSTER_FORMING"

    return "MIXED_PHASE"


# -------------------------
# MAIN MATCHING LOOP
# -------------------------
for i in range(len(leaf)):

    # leaf centroid
    x = leaf["cen_l"][i]
    y = leaf["cen_b"][i]
    z = leaf["cen_v"][i] / 1000  # m/s → km/s

    # matching ellipsoid
    rx = (leaf["Lsize"][i] / 2 + 5) * 0.0017  # deg
    ry = (leaf["Bsize"][i] / 2 + 5) * 0.0017  # deg
    rz = (leaf["Vsize"][i] / 2 + 5) * 0.5  # km/s

    # storage per leaf
    l70, l160, l250, l350, l500, tbol = [], [], [], [], [], []
    match_flags = []

    for j in range(len(higal)):
        dx = (higal["glon"][j] - x) / rx
        dy = (higal["glat"][j] - y) / ry
        dz = (higal["vlsr"][j] - z) / rz

        # ellipsoid inclusion test
        if dx * dx + dy * dy + dz * dz <= 1:

            l70.append(higal["L70"][j])
            l160.append(higal["L160"][j])
            l250.append(higal["L250"][j])
            l350.append(higal["L350"][j])
            l500.append(higal["L500"][j])
            tbol.append(higal["t_bol"][j])

            match_flags.append(higal["evol_flag"][j])

    # assign values
    L70.append(np.sum(l70) if l70 else 0.0)
    L160.append(np.sum(l160) if l160 else 0.0)
    L250.append(np.sum(l250) if l250 else 0.0)
    L350.append(np.sum(l350) if l350 else 0.0)
    L500.append(np.sum(l500) if l500 else 0.0)
    Tbol.append(np.nanmean(tbol) if tbol else np.nan)

    # store evolution list
    evol_lists.append(sorted(list(set(match_flags))) if match_flags else ["NONE"])

    # classify leaf
    category.append(classify_stage(match_flags if match_flags else ["NONE"]))


# -------------------------
# WRITE BACK TO CATALOG
# -------------------------
leaf["L70"] = L70
leaf["L160"] = L160
leaf["L250"] = L250
leaf["L350"] = L350
leaf["L500"] = L500
leaf["Tbol"] = Tbol

# Compute SFE
leaf["SFE70"] = leaf["L70"] / leaf["mass"]
leaf["SFE160"] = leaf["L160"] / leaf["mass"]
leaf["SFE250"] = leaf["L250"] / leaf["mass"]
leaf["SFE350"] = leaf["L350"] / leaf["mass"]
leaf["SFE500"] = leaf["L500"] / leaf["mass"]

leaf["evol_matches"] = evol_lists
leaf["evol_category"] = category

leaf.write(OUTPUT_FILE, format="fits", overwrite=True)

print(
    "✔ Leaf catalog updated with Hi-GAL luminosities, SFE, evolutionary associations, and category flags."
)
