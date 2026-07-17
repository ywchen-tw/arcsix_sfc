"""Mean bias ERA5 fal minus SSFR TOA-solar-weighted broadband albedo.

All legs (no spirals, alt <= 1.6 km), per season and campaign-mean, using the
same data loading and selection logic as analysis/ssfr_era5_alb_si_fig.py.
"""
import os
import sys

_BASE_DIR_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../lrt_sim
sys.path.insert(0, _BASE_DIR_)
os.chdir(_BASE_DIR_)

import pickle
import numpy as np

from ssfr_atm_corr.settings import _fdir_general_

_SOLAR_FLUX_FILE = os.path.join(_BASE_DIR_, 'arcsix_ssfr_solar_flux_slit.dat')
_ALT_MAX_KM = 1.6

combined_file = f'{_fdir_general_}/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl'
with open(combined_file, 'rb') as f:
    d = pickle.load(f)

solar_wvl, solar_flux = np.loadtxt(_SOLAR_FLUX_FILE, comments='#', unpack=True)

all_ssfr, all_era5 = [], []
for season in ('spring', 'summer'):
    wvl = np.asarray(d[f'ext_wvl_{season}'], dtype=float)
    alb = np.clip(np.asarray(d[f'alb_final_ext_all_{season}'], dtype=float), 0.0, 1.0)
    flux = np.interp(wvl, solar_wvl, solar_flux, left=0.0, right=0.0)
    bb_toa = np.trapz(alb * flux, wvl, axis=1) / np.trapz(flux, wvl)

    bb_flux = np.asarray(d[f'broadband_alb_final_ext_{season}_all'], dtype=float)
    tags = np.asarray(d[f'case_tags_{season}_all']).astype(str)
    alt = np.asarray(d[f'alt_all_{season}'], dtype=float)
    era5 = np.asarray(d[f'era5_alb_{season}_all'], dtype=float)

    sel = ((np.char.find(tags, 'spiral') < 0)
           & (alt <= _ALT_MAX_KM)
           & np.isfinite(bb_toa) & np.isfinite(bb_flux)
           & np.isfinite(era5))

    ssfr_m = np.nanmean(bb_toa[sel])
    era5_m = np.nanmean(era5[sel])
    bias = era5_m - ssfr_m
    per_pt = np.nanmean(era5[sel] - bb_toa[sel])
    print(f'{season:6s}: n={sel.sum():6d}  SSFR(TOA)={ssfr_m:.4f}  '
          f'ERA5={era5_m:.4f}  bias(ERA5-SSFR)={bias:+.4f}  '
          f'per-point mean bias={per_pt:+.4f}')

    all_ssfr.append(bb_toa[sel])
    all_era5.append(era5[sel])

ssfr_c = np.concatenate(all_ssfr)
era5_c = np.concatenate(all_era5)
print(f'both  : n={ssfr_c.size:6d}  SSFR(TOA)={ssfr_c.mean():.4f}  '
      f'ERA5={era5_c.mean():.4f}  bias(ERA5-SSFR)={era5_c.mean() - ssfr_c.mean():+.4f}  '
      f'per-point mean bias={np.mean(era5_c - ssfr_c):+.4f}')
