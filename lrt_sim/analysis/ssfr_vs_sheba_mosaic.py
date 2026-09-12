"""ARCSIX SSFR albedo vs SHEBA (Perovich et al. 2002) and MOSAiC (Light et al. 2022).

Computes, from the combined SSFR product (all legs: no spirals, alt <= 1.6 km):
  * per-season date ranges and daily-mean broadband albedo (both weightings)
  * per-season broadband stats (mean/std/percentiles)
  * campaign-mean spectral albedo per season + values at 500/800/1064 nm
  * summer distribution across surface-type albedo ranges
Panel (a) overlays representative MOSAiC surface-type spectra (melting snow,
bare melting ice, dark pond, autumn snow) extracted from the archived survey
data (Smith et al. 2021, Arctic Data Center doi:10.18739/A2FT8DK8Z; sites as
documented in Light et al. 2022, Figs 2/8), read from
``data/SI_data/mosaic_ref_spectra.csv``.
Panel (b) overlays the SHEBA-1998 (Perovich et al. 2002) and MOSAiC-2020
(Light et al. 2022) surface-type bands plus the N-ICE2015 early-spring broadband
level (0.82; Di Biagio et al. 2021 Sec. 3.1.2, measured by Walden et al. 2017),
read from ``data/SI_data/endmember_lit_values.csv``. Each campaign carries its
own color, shared by its bands and their labels.

The N-ICE2015 value is a noon-centred *point* albedo over snow-covered sea ice
(Kipp & Zonen CMP22, 200-3600 nm), that is an ice/snow end member, whereas the
ARCSIX daily means here are flight-track averages over mixed scenes on the
extended 300-4000 nm grid. The like-for-like ARCSIX comparison is the SIF=1 end
member in ``analysis/ssfr_ice_frac_alb_analysis.py``, where the same 0.82 is
drawn; being noon-centred it is also a lower limit on the daily albedo
(Wiscombe & Warren 1980).

Figure geometry/fonts/colors follow the shared GRL style (``plot_style``);
season colors match ``ssfr_era5_alb_si_fig.py`` (spring blue, summer vermillion).

Run: python analysis/ssfr_vs_sheba_mosaic.py   (any cwd; it chdirs to lrt_sim/)
Output: fig/SI/arcsix_vs_sheba_mosaic.{png,pdf}
"""
import os
import sys

_BASE_DIR_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../lrt_sim
sys.path.insert(0, _BASE_DIR_)
os.chdir(_BASE_DIR_)

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ssfr_atm_corr.settings import _fdir_general_
from plot_style import (
    FULL_WIDTH_MM, OKABE_ITO, add_panel_label, apply_grl_style, figsize_mm, save_grl,
)

_SOLAR_FLUX_FILE = os.path.join(_BASE_DIR_, 'arcsix_ssfr_solar_flux_slit.dat')
_ALT_MAX_KM = 1.6

with open(f'{_fdir_general_}/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl', 'rb') as f:
    d = pickle.load(f)

solar_wvl, solar_flux = np.loadtxt(_SOLAR_FLUX_FILE, comments='#', unpack=True)

# N-ICE2015 broadband reference, read from the same table that
# analysis/ssfr_ice_frac_alb_analysis.py uses so the two figures cannot drift.
_lit = pd.read_csv(f'{_fdir_general_}/SI_data/endmember_lit_values.csv', comment='#')
_nice_row = _lit.loc[_lit['source'] == 'dibiagio2021'].iloc[0]
_nice = float(_nice_row['albedo'])
_nice_band = f"{_nice_row['band_nm_lo']:.0f}-{_nice_row['band_nm_hi']:.0f} nm"

out = {}
for season in ('spring', 'summer'):
    wvl = np.asarray(d[f'ext_wvl_{season}'], dtype=float)
    alb = np.clip(np.asarray(d[f'alb_final_ext_all_{season}'], dtype=float), 0.0, 1.0)
    flux = np.interp(wvl, solar_wvl, solar_flux, left=0.0, right=0.0)
    bb_toa = np.trapz(alb * flux, wvl, axis=1) / np.trapz(flux, wvl)
    bb_flx = np.asarray(d[f'broadband_alb_final_ext_{season}_all'], dtype=float)

    tags = np.asarray(d[f'case_tags_{season}_all']).astype(str)
    alt = np.asarray(d[f'alt_all_{season}'], dtype=float)
    era5 = np.asarray(d[f'era5_alb_{season}_all'], dtype=float)
    dates = np.asarray(d[f'dates_{season}_all']).astype(int)

    sel = ((np.char.find(tags, 'spiral') < 0) & (alt <= _ALT_MAX_KM)
           & np.isfinite(bb_toa) & np.isfinite(bb_flx) & np.isfinite(era5))

    bt, bf, dt = bb_toa[sel], bb_flx[sel], dates[sel]
    udates = np.unique(dt)
    print(f'\n=== {season.upper()} ===')
    print(f'n={sel.sum()}, flight days={udates.size}: {udates.min()}..{udates.max()}')
    for w, b in (('TOA', bt), ('flux', bf)):
        q = np.percentile(b, [5, 25, 50, 75, 95])
        print(f'  {w:4s}: mean={b.mean():.3f} std={b.std():.3f} '
              f'p5={q[0]:.3f} p25={q[1]:.3f} med={q[2]:.3f} p75={q[3]:.3f} p95={q[4]:.3f}')

    # surface-type-range distribution (TOA-weighted broadband)
    bins = [(0.0, 0.15, 'open-water-like (<0.15)'),
            (0.15, 0.45, 'pond-like / mixed (0.15-0.45)'),
            (0.45, 0.60, 'mixed bare/ponded (0.45-0.60)'),
            (0.60, 0.70, 'bare melting ice (0.60-0.70)'),
            (0.70, 1.01, 'snow-covered (>0.70)')]
    for lo, hi, lab in bins:
        frac = ((bt >= lo) & (bt < hi)).mean() * 100
        print(f'    {lab:32s}: {frac:5.1f}%')

    # daily means
    day_toa = np.array([bt[dt == u].mean() for u in udates])
    day_flx = np.array([bf[dt == u].mean() for u in udates])
    day_n = np.array([(dt == u).sum() for u in udates])
    print('  daily means (date, n, TOA, flux):')
    for u, n, a, b_ in zip(udates, day_n, day_toa, day_flx):
        print(f'    {u}  n={n:5d}  TOA={a:.3f}  flux={b_:.3f}')

    # mean spectral albedo
    spec_mean = np.nanmean(alb[sel], axis=0)
    spec_std = np.nanstd(alb[sel], axis=0)
    for w0 in (500, 800, 1064):
        i = np.argmin(np.abs(wvl - w0))
        print(f'  spectral alb @ {w0:4d} nm: {spec_mean[i]:.3f} +/- {spec_std[i]:.3f}')
    out[season] = dict(wvl=wvl, spec_mean=spec_mean, spec_std=spec_std,
                       udates=udates, day_toa=day_toa, day_flx=day_flx,
                       bt=bt, bf=bf)

# --- N-ICE2015 vs the ARCSIX daily means ---------------------------------
# Not like-for-like: 0.82 is a point value over snow (an end member), the ARCSIX
# daily means are track averages over mixed scenes, so part of any gap is scene
# composition rather than a darker surface. See the module docstring.
print(f'\n=== N-ICE2015 (Di Biagio et al. 2021, {_nice_band}) ===')
print(f'  snow-covered sea ice, early spring: {_nice:.2f}')
for season in ('spring', 'summer'):
    dm = out[season]['day_toa']
    print(f'  ARCSIX {season:6s} daily mean (TOA): {dm.min():.3f}-{dm.max():.3f}, '
          f'mean {dm.mean():.3f} ({dm.mean() - _nice:+.3f} vs N-ICE2015)')

# ---------------------------------------------------------------- figure
import datetime as _dt
def doy(yyyymmdd):
    s = str(yyyymmdd)
    return _dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])).timetuple().tm_yday

apply_grl_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_mm(FULL_WIDTH_MM, 75.0))

# Season colors match ssfr_era5_alb_si_fig.py: spring blue, summer vermillion.
colors = {'spring': OKABE_ITO[0], 'summer': OKABE_ITO[1]}
for season in ('spring', 'summer'):
    o = out[season]
    m = o['wvl'] <= 2500
    ax1.fill_between(o['wvl'][m], (o['spec_mean'] - o['spec_std'])[m],
                     (o['spec_mean'] + o['spec_std'])[m],
                     color=colors[season], alpha=0.2, lw=0)
    ax1.plot(o['wvl'][m], o['spec_mean'][m], color=colors[season],
             label=f'{season.capitalize()} mean $\\pm$ 1$\\sigma$')

# Representative MOSAiC surface-type spectra (Smith et al. 2021 dataset,
# doi:10.18739/A2FT8DK8Z; sites documented in Light et al. 2022, Figs 2/8),
# extracted by data/SI_data provenance notes. Thin gray/black lines so the
# ARCSIX seasonal means stay dominant; autumn snow ends ~1250 nm (Sep NIR
# data failed the dataset's quality control).
_mosaic_csv = f'{_fdir_general_}/SI_data/mosaic_ref_spectra.csv'
_mref = pd.read_csv(_mosaic_csv, comment='#')
_mstyles = [
    ('autumn_snow_0917_K20', 'MOSAiC autumn snow', 'dimgray', '-'),
    ('melting_snow_0620_LD165', 'MOSAiC melting snow', 'dimgray', '--'),
    ('bare_ice_0724_LD165', 'MOSAiC bare melting ice', 'k', '-.'),
    ('dark_pond_0724_LD175', 'MOSAiC dark pond', 'k', ':'),
]
for key, lab, c, ls in _mstyles:
    ax1.plot(_mref['wvl_nm'], _mref[key], color=c, ls=ls, lw=0.9, alpha=0.8,
             zorder=1, label=lab)
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Spectral Albedo')
ax1.set_xlim(350, 2500)
ax1.set_ylim(0, 1)
ax1.legend(loc='upper right', fontsize=6.5)
add_panel_label(ax1, '(a)')

# panel b: daily mean broadband vs DOY + SHEBA/MOSAiC references
for season in ('spring', 'summer'):
    o = out[season]
    x = [doy(u) for u in o['udates']]
    ax2.plot(x, o['day_toa'], 'o-', color=colors[season],
             label=f'{season.capitalize()} daily mean (TOA)')
    ax2.plot(x, o['day_flx'], '^--', color=colors[season], alpha=0.5,
             label=f'{season.capitalize()} daily mean (flux)')

# One color per campaign, shared by each band and its label, so a label's color
# says which campaign it belongs to. SHEBA stays neutral gray (the remaining
# Okabe-Ito entries are too close to the ARCSIX season colors); MOSAiC keeps the
# green of its bare-ice band; N-ICE2015 takes the reddish purple it already has
# in analysis/ssfr_ice_frac_alb_analysis.py.
_C_SHEBA     = 'gray'
_C_SHEBA_TXT = 'dimgray'          # darker than the 0.3-alpha band, legible at 6 pt
_C_MOSAIC    = OKABE_ITO[2]
_C_NICE      = OKABE_ITO[5]
_band_color  = {'sheba': (_C_SHEBA, _C_SHEBA_TXT, 0.3),
                'mosaic': (_C_MOSAIC, _C_MOSAIC, 0.18)}   # light: the N-ICE2015
#                                    label sits partly over the MOSAiC CO1 band

# Reference bands (line-averaged albedo from the surface campaigns). Labels
# are placed per band (inside / beside) to avoid the flight-track markers.
refs = [
    ('sheba',   91, 135, 0.80, 0.90, 'Dry snow\n0.8-0.9', 113, 0.85, 'center', 'center'),
    ('mosaic',  97, 132, 0.76, 0.88, None, 0, 0, '', ''),  # MOSAiC CO1 line means overlap
    ('sheba',  152, 166, 0.70, 0.75, 'Melting snow 0.70-0.75', 147, 0.70, 'right', 'center'),
    ('sheba',  172, 211, 0.45, 0.65, 'Pond formation-\nevolution', 191.5, 0.685, 'center', 'bottom'),
    ('sheba',  204, 213, 0.38, 0.42, 'Line avg ~0.4', 216, 0.40, 'left', 'center'),
    ('sheba',  224, 242, 0.36, 0.38, 'Minimum 0.37', 233, 0.348, 'center', 'top'),
]
for camp, x0, x1, y0, y1, lab, tx, ty, ha, va in refs:
    c_band, c_txt, a_band = _band_color[camp]
    ax2.fill_between([x0, x1], y0, y1, color=c_band, alpha=a_band, lw=0)
    if lab:
        ax2.text(tx, ty, lab, fontsize=6, ha=ha, va=va, color=c_txt)
ax2.axhspan(0.60, 0.68, color=_C_MOSAIC, alpha=0.12)
ax2.text(93, 0.59, 'MOSAiC bare melting\nice 0.64$\\pm$0.04', fontsize=6,
         va='top', color=_C_MOSAIC)

# N-ICE2015: a single early-spring level, not a time series, so a line rather
# than a band. Drawn over April (DOY 91-120), the month Di Biagio et al. (2021)
# compare directly with SHEBA; their "day 145" is a day of year, the same axis
# used here. The CSV span starts 1 Feb (DOY 32), left of this panel's x-limit.
ax2.plot([91, 120], [_nice, _nice], color=_C_NICE, ls='--', lw=1.0, zorder=2)
# Label anchored under the segment's left end: to the right of DOY ~146 it would
# run into the spring flight-track markers.
ax2.text(92, 0.79, f'N-ICE2015 snow-\ncovered ice {_nice:.2f}', fontsize=6,
         ha='left', va='top', color=_C_NICE)

# Color key: one text object per campaign, since a single string cannot carry
# three colors (this replaces the former one-line 'SHEBA-1998 / MOSAiC-2020
# references' caption).
ax2.text(0.98, 0.985, 'Surface-campaign references', transform=ax2.transAxes,
         ha='right', va='top', fontsize=6.5, color='0.35')
for _dy, _name, _c in ((0.055, 'SHEBA-1998', _C_SHEBA_TXT),
                       (0.105, 'MOSAiC-2020', _C_MOSAIC),
                       (0.155, 'N-ICE2015', _C_NICE)):
    ax2.text(0.98, 0.985 - _dy, _name, transform=ax2.transAxes,
             ha='right', va='top', fontsize=7, color=_c)
ax2.set_xlabel('Day of Year')
ax2.set_ylabel('Broadband Albedo')
ax2.set_xlim(88, 265)
ax2.set_ylim(0.2, 1.0)
ax2.legend(fontsize=6.5, loc='lower left')
add_panel_label(ax2, '(b)')

fig.tight_layout()
os.makedirs('./fig/SI', exist_ok=True)
written = save_grl(fig, './fig/SI/arcsix_vs_sheba_mosaic')
print('\nSaved ' + ', '.join(written))
