"""Two-panel SI figure: SSFR extended broadband albedo vs collocated ERA5 albedo.

Panel (a): per-second scatter for the Fig-4 CRE case (case_004, 2024-06-03
cloudy_atm_corr_2, 14.711-14.868 UTC h), replacing the legacy single-panel
``{date}_{case_tag}_sfc_alb_ssfr_vs_era5.png`` that was built from the old
albedo product (means 0.704/0.655).

Panels (b)/(c): every point in the combined product from both seasons, excluding
spiral legs (case_tag contains 'spiral') and high-altitude legs (alt > 1.6 km),
split by weighting so the two point clouds stay distinguishable: (b) TOA-solar
weighted, (c) actual-sky flux weighted.

The SSFR broadband is shown under two weightings:
  * TOA-solar weighted - recomputed per second from the extended (300-4000 nm)
    spectra with the slit-convolved solar flux, i.e. the same solar weighting
    as ``cre.ext_alb_cases`` / ``cre.cre_plot`` (the CRE broadband definition).
  * Actual-sky flux weighted - the ``broadband_alb_final_ext_*`` arrays stored
    in the combined product, weighted by the simulated downward flux at the
    surface under each leg's own (clear or cloudy) sky. This is the convention
    closest to how ERA5 defines its forecast albedo (fal).

Figure geometry/fonts/colors follow the shared GRL style (``plot_style``).

Run: python analysis/ssfr_era5_alb_si_fig.py   (any cwd; it chdirs to lrt_sim/)
Output: fig/SI/sfc_alb_ssfr_vs_era5_2panel.{png,pdf}
        fig/SI/sfc_alb_ssfr_vs_era5_2panel_cloudy.{png,pdf}  (cloudy legs only in
        panels b/c, where the TOA vs actual-sky weighting difference is largest)
"""

import os
import sys

_BASE_DIR_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../lrt_sim
sys.path.insert(0, _BASE_DIR_)
os.chdir(_BASE_DIR_)

import pickle

import numpy as np
import matplotlib.pyplot as plt

from ssfr_atm_corr.settings import _fdir_general_
from plot_style import (
    FULL_WIDTH_MM, OKABE_ITO, add_panel_label, apply_grl_style, figsize_mm, save_grl,
)

_SOLAR_FLUX_FILE = os.path.join(_BASE_DIR_, 'arcsix_ssfr_solar_flux_slit.dat')

# Panel (a) case: the Fig-4 CRE case (case_004).
_CASE_DATE = 20240603
_CASE_TAG = 'cloudy_atm_corr_2'

_ALT_MAX_KM = 1.6

# Okabe-Ito assignments: weighting -> marker, season -> color.
_C_TOA = OKABE_ITO[0]        # blue
_C_FLUX = OKABE_ITO[2]       # bluish green
_SEASON_COLORS = {'spring': OKABE_ITO[0], 'summer': OKABE_ITO[1]}  # blue / vermillion


def _per_second_broadband(d, season, solar_wvl, solar_flux):
    """Solar-flux-weighted broadband albedo per second from the extended spectra."""
    wvl = np.asarray(d[f'ext_wvl_{season}'], dtype=float)
    alb = np.clip(np.asarray(d[f'alb_final_ext_all_{season}'], dtype=float), 0.0, 1.0)
    flux = np.interp(wvl, solar_wvl, solar_flux, left=0.0, right=0.0)
    return np.trapz(alb * flux, wvl, axis=1) / np.trapz(flux, wvl)


def _style_axis(ax, panel_label, lim=(0.2, 0.9), ylabel=True):
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal')
    ax.set_xlabel('SSFR Broadband Albedo')
    if ylabel:
        ax.set_ylabel('ERA5 Broadband Albedo')
    add_panel_label(ax, panel_label)


def load_combined():
    combined_file = f'{_fdir_general_}/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl'
    print(f'Loading combined product: {combined_file}')
    with open(combined_file, 'rb') as f:
        return pickle.load(f)


def plot_ssfr_vs_era5_2panel(d=None, cloudy_only=False):
    """Build the figure; ``cloudy_only`` restricts panels (b)/(c) to cloudy legs."""
    if d is None:
        d = load_combined()
    variant = ' (cloudy legs only)' if cloudy_only else ''

    solar_wvl, solar_flux = np.loadtxt(_SOLAR_FLUX_FILE, comments='#', unpack=True)

    bb = {s: _per_second_broadband(d, s, solar_wvl, solar_flux)
          for s in ('spring', 'summer')}
    # Actual-sky (simulated surface downward flux) weighted broadband, as stored
    # in the combined product.
    bb_flux = {s: np.asarray(d[f'broadband_alb_final_ext_{s}_all'], dtype=float)
               for s in ('spring', 'summer')}

    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize_mm(FULL_WIDTH_MM, 75.0))

    # --- (a) Fig-4 case leg -------------------------------------------------
    tags = np.asarray(d['case_tags_spring_all'])
    dates = np.asarray(d['dates_spring_all'])
    era5_spring = np.asarray(d['era5_alb_spring_all'], dtype=float)
    case_mask = (dates == _CASE_DATE) & (tags == _CASE_TAG)

    x_toa = bb['spring'][case_mask]
    x_flux = bb_flux['spring'][case_mask]
    y = era5_spring[case_mask]
    ax1.scatter(x_toa, y, c=_C_TOA, s=6, alpha=0.5, linewidths=0)
    ax1.scatter(x_flux, y, c=_C_FLUX, marker='^', s=6, alpha=0.5, linewidths=0)
    ax1.scatter(np.nanmean(x_toa), np.nanmean(y), c=_C_TOA, s=60, marker='o',
                edgecolors='k', linewidths=0.8, zorder=3,
                label='TOA-solar weighted (mean)')
    ax1.scatter(np.nanmean(x_flux), np.nanmean(y), c=_C_FLUX, s=60, marker='^',
                edgecolors='k', linewidths=0.8, zorder=3,
                label='Actual-sky flux weighted (mean)')
    _style_axis(ax1, '(a)')
    # Legend below the axes so it never occludes the scatter.
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=1,
               fontsize=6.5, handletextpad=0.4, frameon=False)
    print(f'(a) {_CASE_DATE} {_CASE_TAG}: n={np.isfinite(x_toa).sum()}, '
          f'SSFR TOA mean {np.nanmean(x_toa):.3f}, '
          f'SSFR flux-weighted mean {np.nanmean(x_flux):.3f}, '
          f'ERA5 mean {np.nanmean(y):.3f}')

    # --- (b)/(c) all combined cases, no spirals, alt <= 1.6 km --------------
    # One panel per weighting so the point clouds stay distinguishable:
    # (b) TOA-solar weighted, (c) actual-sky flux weighted. The same point
    # selection is used in both panels. ``cloudy_only`` keeps only cloudy legs
    # (case_tag contains 'cloudy'), where the downward flux at the surface is
    # most NIR-depleted and the two weightings differ the most.
    era5_all, sels = {}, {}
    for season in ('spring', 'summer'):
        tags_s = np.asarray(d[f'case_tags_{season}_all']).astype(str)
        alt_s = np.asarray(d[f'alt_all_{season}'], dtype=float)
        era5_all[season] = np.asarray(d[f'era5_alb_{season}_all'], dtype=float)
        sels[season] = ((np.char.find(tags_s, 'spiral') < 0)
                        & (alt_s <= _ALT_MAX_KM)
                        & np.isfinite(bb[season]) & np.isfinite(bb_flux[season])
                        & np.isfinite(era5_all[season]))
        if cloudy_only:
            sels[season] &= np.char.find(tags_s, 'cloudy') >= 0
        print(f'(b)/(c){variant} {season}: n={sels[season].sum()}, '
              f'SSFR TOA mean {np.nanmean(bb[season][sels[season]]):.3f}, '
              f'SSFR flux-weighted mean {np.nanmean(bb_flux[season][sels[season]]):.3f}, '
              f'ERA5 mean {np.nanmean(era5_all[season][sels[season]]):.3f}')

    # Marker encodes the weighting in every panel: circle = TOA-solar,
    # triangle = actual-sky flux weighted (same convention as panel a).
    for ax, bb_w, wname, panel, marker in (
            (ax2, bb, 'TOA-solar weighted', '(b)', 'o'),
            (ax3, bb_flux, 'Actual-sky flux weighted', '(c)', '^')):
        for season in ('spring', 'summer'):
            sel = sels[season]
            xs, ys = bb_w[season][sel], era5_all[season][sel]
            color = _SEASON_COLORS[season]
            ax.scatter(xs, ys, c=color, s=2, alpha=0.25, marker=marker,
                       linewidths=0, rasterized=True,
                       label=season.capitalize())
            ax.scatter(np.nanmean(xs), np.nanmean(ys), c=color, s=70,
                       marker=marker, edgecolors='k', linewidths=0.8, zorder=3,
                       label=f'{season.capitalize()} mean')
        # Open-water points reach albedo ~0.05, so (b)/(c) start at 0 instead of
        # panel (a)'s 0.2 (which would silently clip ~1400 summer points).
        _style_axis(ax, panel, lim=(0.0, 0.9), ylabel=(ax is ax2))
        # Weighting name inside the axes (a centered title would collide with
        # the panel label above the axes).
        wtext = f'{wname}\n(cloudy legs only)' if cloudy_only else wname
        ax.text(0.5, 0.97, wtext, transform=ax.transAxes, ha='center', va='top',
                fontsize=7.5)
        # Legend below the axes so it never occludes the scatter.
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2,
                        fontsize=6.5, handletextpad=0.4, columnspacing=0.8,
                        frameon=False)
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)

    fig.tight_layout()
    os.makedirs('./fig/SI', exist_ok=True)
    suffix = '_cloudy' if cloudy_only else ''
    written = save_grl(fig, f'./fig/SI/sfc_alb_ssfr_vs_era5_2panel{suffix}')
    plt.close(fig)
    print('Saved ' + ', '.join(written))


if __name__ == '__main__':
    apply_grl_style()
    d = load_combined()
    plot_ssfr_vs_era5_2panel(d)
    plot_ssfr_vs_era5_2panel(d, cloudy_only=True)
