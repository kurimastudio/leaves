#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.table import Table
from astropy.io import fits
import numpy as np
import sys


def clean_leaves_from_chunk(tbl):
    regs = np.unique(tbl["REGION"])
    valid = {r: set(tbl["LEAF_ID"][tbl["REGION"] == r]) for r in regs}

    for r in regs:
        fname = f"region{r}_snr_ultra_5_leaf_asgn.fits"
        try:
            hdul = fits.open(fname, memmap=True)
        except:
            continue

        data = hdul[0].data
        if data is None:
            hdul.close()
            continue

        ids = valid[r]
        if not ids:
            data[:] = -1
        else:
            # FAST LOOKUP TABLE
            vmax = int(data.max())
            lut = np.zeros(vmax + 1, dtype=bool)
            for v in ids:
                if v >= 0:
                    lut[v] = True
            data[~lut[data]] = -1

        hdul.writeto(f"leaves_catalog_clean_{r}.fits", overwrite=True)
        hdul.close()


if __name__ == "__main__":
    tbl = Table.read(sys.argv[1])
    clean_leaves_from_chunk(tbl)
