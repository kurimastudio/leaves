import os
import numpy as np
from astropy.table import Table
from astropy.io import fits
from spectral_cube import SpectralCube
import astropy.units as u

os.chdir("/home/kurima/Data/v5")


os.chdir('/data/yuwei/v5')

# ============================
# LOAD CATALOG
# ============================
cat = Table.read("new_leaves.fits")

sigma_v = np.full(len(cat), np.nan, dtype=float)
v_cent  = np.full(len(cat), np.nan, dtype=float)

regions = np.unique(cat["REGION"])

for reg in regions:
    print(f"\n🔹 REGION {reg}: computing sigma_v and v0")

    em_name = f"region{reg}_smooth.fits"
    leaf_asgn_name = f"leaves_catalog_clean_{reg}.fits"

    # ----------------------------
    # Read cube
    # ----------------------------
    try:
        cube = SpectralCube.read(em_name).with_spectral_unit(u.km/u.s)
    except Exception as e:
        print(f"  ⚠ Missing or unreadable cube {em_name}: {e}")
        continue

    try:
        labels = fits.getdata(leaf_asgn_name).astype(np.int32)
    except Exception as e:
        print(f"  ⚠ Missing or unreadable leaf assignments {leaf_asgn_name}: {e}")
        continue

    # ----------------------------
    # Get data
    # ----------------------------
    data = cube.filled_data[:].value   # shape = (nv, ny, nx)
    data = np.asarray(data, dtype=np.float64)

    # Replace non-finite and negative intensities
    bad = ~np.isfinite(data)
    if np.any(bad):
        data[bad] = 0.0

    # Optional: clip negative noise values
    data[data < 0] = 0.0

    if data.shape != labels.shape:
        print(f"  ⚠ Shape mismatch: cube {data.shape}, labels {labels.shape}")
        continue

    vel = cube.spectral_axis.to_value(u.km/u.s)   # shape = (nv,)

    # ----------------------------
    # Flatten arrays
    # ----------------------------
    lab_flat = labels.ravel()
    I_flat   = data.ravel()

    # Build velocity cube lazily via repeat
    nv, ny, nx = data.shape
    v_flat = np.repeat(vel, ny * nx)

    # ----------------------------
    # Keep only labeled voxels with positive intensity
    # ----------------------------
    good = (lab_flat > 0) & np.isfinite(I_flat) & (I_flat > 0)

    if not np.any(good):
        print("  ⚠ No valid labeled voxels in this region")
        continue

    lab = lab_flat[good]
    I   = I_flat[good]
    v   = v_flat[good]

    # ----------------------------
    # Weighted sums per leaf label
    # ----------------------------
    max_label = lab.max()

    S0 = np.bincount(lab, weights=I, minlength=max_label + 1)
    S1 = np.bincount(lab, weights=I * v, minlength=max_label + 1)
    S2 = np.bincount(lab, weights=I * v * v, minlength=max_label + 1)

    # centroid and variance
    with np.errstate(divide='ignore', invalid='ignore'):
        v0_all = S1 / S0
        var_all = S2 / S0 - v0_all**2

    # Numerical cleanup
    var_all[var_all < 0] = 0.0
    sig_all = np.sqrt(var_all)

    # ----------------------------
    # Write back to catalog
    # ----------------------------
    idxs = np.where(cat["REGION"] == reg)[0]
    leaf_ids = np.array(cat["LEAF_ID"][idxs], dtype=int)

    valid_leaf = (
        (leaf_ids >= 0) &
        (leaf_ids < len(v0_all)) &
        np.isfinite(S0[leaf_ids]) &
        (S0[leaf_ids] > 0)
    )

    v_cent[idxs[valid_leaf]]  = v0_all[leaf_ids[valid_leaf]]
    sigma_v[idxs[valid_leaf]] = sig_all[leaf_ids[valid_leaf]]

    print(f"  ✓ assigned {np.sum(valid_leaf)} / {len(idxs)} leaves")
    print(f"  sigma_v range in region: "
          f"{np.nanmin(sig_all[1:]):.3f} to {np.nanmax(sig_all[1:]):.3f} km/s")

# ============================
# SAVE
# ============================
cat["sigma_v"] = sigma_v
cat["v_centroid"] = v_cent
cat.write("leaf_catalog_sigma_v0.fits", overwrite=True)

print("\n🎉 DONE -> leaf_catalog_sigma_v0.fits\n")











