"""
Utility functions for ARCSIX LW radiative transfer simulations.
"""

import numpy as np
from scipy.signal import convolve


def gaussian(x, mu, sig):
    y = (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )
    return y / np.max(y)


def ssfr_slit_convolve(wvl, flux_orig, wvl_joint):
    dwvl = wvl[1] - wvl[0]
    xx = np.linspace(-12, 12, int(24 / dwvl + 1))
    yy_gaussian_vis = gaussian(xx, 0, 3.8251)
    yy_gaussian_nir = gaussian(xx, 0, 4.5046)

    flux_conv = flux_orig.copy()
    flux_convolved_vis = convolve(flux_orig, yy_gaussian_vis, mode='same') / np.sum(yy_gaussian_vis)
    flux_convolved_nir = convolve(flux_orig, yy_gaussian_nir, mode='same') / np.sum(yy_gaussian_nir)
    flux_conv[wvl <= wvl_joint] = flux_convolved_vis[wvl <= 950]
    flux_conv[wvl > wvl_joint]  = flux_convolved_nir[wvl > 950]

    return flux_conv


if __name__ == '__main__':
    pass
