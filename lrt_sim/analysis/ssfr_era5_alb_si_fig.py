"""SI figures: SSFR extended broadband albedo vs collocated ERA5 albedo.

Panel (a): per-second scatter for the Fig-4 CRE case (case_004, 2024-06-03
cloudy_atm_corr_2, 14.711-14.868 UTC h), replacing the legacy single-panel
``{date}_{case_tag}_sfc_alb_ssfr_vs_era5.png`` that was built from the old
albedo product (means 0.704/0.655).

Weighting panels (b)/(c)/...: every point in the combined product from both
seasons, excluding spiral legs (case_tag contains 'spiral') and high-altitude
legs (alt > 1.6 km), one panel per weighting so the point clouds stay
distinguishable. Each panel also shows the OTHER weighting's season means as
lighter, dashed-edge ghost markers for direct comparison.

The SSFR broadband is shown under two weightings:
  * TOA-solar weighted (circles) - recomputed per second from the extended
    (300-4000 nm) spectra with the slit-convolved solar flux, i.e. the same
    solar weighting as ``cre.ext_alb_cases`` / ``cre.cre_plot`` (the CRE
    broadband definition).
  * Actual-sky flux weighted (triangles) - the ``broadband_alb_final_ext_*``
    arrays stored in the combined product, weighted by the simulated downward
    flux at the surface under each leg's own (clear or cloudy) sky.

ERA5 comparison notes (ECMWF radiation documentation; Hogan 2015):
  * ``fal`` (forecast albedo, param 243) is a diagnostic broadband albedo:
    the model's DIFFUSE UV-visible and near-IR surface albedos averaged with
    a FIXED top-of-atmosphere solar spectrum. The TOA-solar weighted SSFR
    broadband (circles) is therefore the apples-to-apples comparison with
    ``fal``; the ERA5 counterpart of the actual-sky flux-weighted SSFR
    broadband (triangles) would be the model's true all-sky albedo
    1 - SSR/SSRD (surface net / downward shortwave fluxes), not ``fal``.
  * Over sea ice, ``fal`` follows the Ebert & Curry (1993) monthly climatology
    (dry snow Sep-May, melting snow Jun, bare ice Jul-Aug; mid-month values
    linearly interpolated), blended with open water by sea-ice concentration.
    It cannot respond to the actual surface state - which is exactly what the
    SSFR - fal difference quantifies.
  * Sources:
    https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf
    https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation
    https://tc.copernicus.org/articles/14/165/2020/  (Pohl et al. 2020, same
    fal-based evaluation approach)

Figure geometry/fonts/colors follow the shared GRL style (``plot_style``).

Run: python analysis/ssfr_era5_alb_si_fig.py   (any cwd; it chdirs to lrt_sim/)
Output: fig/SI/sfc_alb_ssfr_vs_era5_2panel.{png,pdf}          (a + all legs)
        fig/SI/sfc_alb_ssfr_vs_era5_2panel_cloudy.{png,pdf}   (a + cloudy legs,
        where the NIR-depleted surface flux separates the weightings the most)
        fig/SI/sfc_alb_ssfr_vs_era5_5panel.{png,pdf}          (a; b/c cloudy;
        d/e all legs)
"""

import os
import sys

_BASE_DIR_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../lrt_sim
sys.path.insert(0, _BASE_DIR_)
os.chdir(_BASE_DIR_)

import pickle

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
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

# Marker encodes the weighting in every panel.
_WEIGHTINGS = {
    'toa': {'marker': 'o', 'name': 'TOA-solar weighted', 'tag': 'TOA'},
    'flux': {'marker': '^', 'name': 'Actual-sky flux weighted', 'tag': 'flux'},
}


def _per_second_broadband(d, season, solar_wvl, solar_flux):
    """Solar-flux-weighted broadband albedo per second from the extended spectra."""
    wvl = np.asarray(d[f'ext_wvl_{season}'], dtype=float)
    alb = np.clip(np.asarray(d[f'alb_final_ext_all_{season}'], dtype=float), 0.0, 1.0)
    flux = np.interp(wvl, solar_wvl, solar_flux, left=0.0, right=0.0)
    return np.trapz(alb * flux, wvl, axis=1) / np.trapz(flux, wvl)


def _lighten(color, toward_white=0.6):
    """Blend a color toward white for the ghost (other-weighting) mean markers."""
    r, g, b = mcolors.to_rgb(color)
    return tuple(c + (1.0 - c) * toward_white for c in (r, g, b))


def _style_axis(ax, panel_label, lim=(0.2, 0.9), ylabel=True, xlabel=True):
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal')
    if xlabel:
        ax.set_xlabel('SSFR Broadband Albedo')
    if ylabel:
        ax.set_ylabel('ERA5 Broadband Albedo')
    add_panel_label(ax, panel_label)


def load_combined():
    combined_file = f'{_fdir_general_}/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl'
    print(f'Loading combined product: {combined_file}')
    with open(combined_file, 'rb') as f:
        return pickle.load(f)


def _season_broadbands(d):
    """Per-second broadband albedo under both weightings, per season."""
    solar_wvl, solar_flux = np.loadtxt(_SOLAR_FLUX_FILE, comments='#', unpack=True)
    bb = {
        'toa': {s: _per_second_broadband(d, s, solar_wvl, solar_flux)
                for s in ('spring', 'summer')},
        # Actual-sky (simulated surface downward flux) weighted broadband, as
        # stored in the combined product.
        'flux': {s: np.asarray(d[f'broadband_alb_final_ext_{s}_all'], dtype=float)
                 for s in ('spring', 'summer')},
    }
    return bb


def _season_selections(d, bb, cloudy_only):
    """Shared per-season point selection (and collocated ERA5) for the sweep panels."""
    variant = ' (cloudy legs only)' if cloudy_only else ' (all legs)'
    era5_all, sels = {}, {}
    for season in ('spring', 'summer'):
        tags_s = np.asarray(d[f'case_tags_{season}_all']).astype(str)
        alt_s = np.asarray(d[f'alt_all_{season}'], dtype=float)
        era5_all[season] = np.asarray(d[f'era5_alb_{season}_all'], dtype=float)
        sels[season] = ((np.char.find(tags_s, 'spiral') < 0)
                        & (alt_s <= _ALT_MAX_KM)
                        & np.isfinite(bb['toa'][season]) & np.isfinite(bb['flux'][season])
                        & np.isfinite(era5_all[season]))
        if cloudy_only:
            # Cloudy legs: the surface downward flux is most NIR-depleted, so
            # the two weightings differ the most.
            sels[season] &= np.char.find(tags_s, 'cloudy') >= 0
        print(f'{variant.strip()} {season}: n={sels[season].sum()}, '
              f'SSFR TOA mean {np.nanmean(bb["toa"][season][sels[season]]):.3f}, '
              f'SSFR flux-weighted mean {np.nanmean(bb["flux"][season][sels[season]]):.3f}, '
              f'ERA5 mean {np.nanmean(era5_all[season][sels[season]]):.3f}')
    return era5_all, sels


def _draw_case_panel(ax, d, bb, legend=True):
    """Panel (a): the Fig-4 case leg under both weightings."""
    tags = np.asarray(d['case_tags_spring_all'])
    dates = np.asarray(d['dates_spring_all'])
    era5_spring = np.asarray(d['era5_alb_spring_all'], dtype=float)
    case_mask = (dates == _CASE_DATE) & (tags == _CASE_TAG)

    x_toa = bb['toa']['spring'][case_mask]
    x_flux = bb['flux']['spring'][case_mask]
    y = era5_spring[case_mask]
    ax.scatter(x_toa, y, c=_C_TOA, s=6, alpha=0.5, linewidths=0)
    ax.scatter(x_flux, y, c=_C_FLUX, marker='^', s=6, alpha=0.5, linewidths=0)
    ax.scatter(np.nanmean(x_toa), np.nanmean(y), c=_C_TOA, s=60, marker='o',
               edgecolors='k', linewidths=0.8, zorder=3,
               label='TOA-solar weighted (mean)')
    ax.scatter(np.nanmean(x_flux), np.nanmean(y), c=_C_FLUX, s=60, marker='^',
               edgecolors='k', linewidths=0.8, zorder=3,
               label='Actual-sky flux weighted (mean)')
    _style_axis(ax, '(a)')
    if legend:
        # Legend below the axes so it never occludes the scatter.
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=1,
                  fontsize=6.5, handletextpad=0.4, frameon=False)
    print(f'(a) {_CASE_DATE} {_CASE_TAG}: n={np.isfinite(x_toa).sum()}, '
          f'SSFR TOA mean {np.nanmean(x_toa):.3f}, '
          f'SSFR flux-weighted mean {np.nanmean(x_flux):.3f}, '
          f'ERA5 mean {np.nanmean(y):.3f}')


def _draw_weighting_panel(ax, which, bb, era5_all, sels, panel, note='',
                          ylabel=True, xlabel=True, legend=True):
    """One weighting's scatter + season means, plus the other weighting's means
    as lighter, dashed-edge ghost markers."""
    other = 'flux' if which == 'toa' else 'toa'
    w, w_other = _WEIGHTINGS[which], _WEIGHTINGS[other]
    for season in ('spring', 'summer'):
        sel = sels[season]
        xs, ys = bb[which][season][sel], era5_all[season][sel]
        color = _SEASON_COLORS[season]
        ax.scatter(xs, ys, c=color, s=2, alpha=0.25, marker=w['marker'],
                   linewidths=0, rasterized=True, label=season.capitalize())
        ax.scatter(np.nanmean(xs), np.nanmean(ys), c=color, s=70,
                   marker=w['marker'], edgecolors='k', linewidths=0.8, zorder=3,
                   label=f'{season.capitalize()} mean')
        # Ghost: the other weighting's mean, lighter with a dashed edge.
        ax.scatter(np.nanmean(bb[other][season][sel]), np.nanmean(ys),
                   facecolors=_lighten(color), s=70, marker=w_other['marker'],
                   edgecolors='k', linewidths=0.8, linestyle='--', zorder=3,
                   label=f"{season.capitalize()} mean ({w_other['tag']})")
    # Open-water points reach albedo ~0.05, so these panels start at 0 instead
    # of panel (a)'s 0.2 (which would silently clip ~1400 summer points).
    _style_axis(ax, panel, lim=(0.0, 0.9), ylabel=ylabel, xlabel=xlabel)
    # Weighting name inside the axes (a centered title would collide with the
    # panel label above the axes).
    wtext = f"{w['name']}\n{note}" if note else w['name']
    ax.text(0.5, 0.97, wtext, transform=ax.transAxes, ha='center', va='top',
            fontsize=7.5)
    if legend:
        # Legend below the axes so it never occludes the scatter. Single column
        # keeps it narrower than the panel, so neighboring legends never collide.
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=1,
                        fontsize=6.5, handletextpad=0.4, frameon=False)
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)


def plot_ssfr_vs_era5_2panel(d=None, cloudy_only=False):
    """(a) case leg + (b)/(c) one panel per weighting; ``cloudy_only`` restricts
    (b)/(c) to cloudy legs."""
    if d is None:
        d = load_combined()
    bb = _season_broadbands(d)
    era5_all, sels = _season_selections(d, bb, cloudy_only)

    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize_mm(FULL_WIDTH_MM, 75.0))

    _draw_case_panel(ax1, d, bb)
    note = '(cloudy legs only)' if cloudy_only else ''
    _draw_weighting_panel(ax2, 'toa', bb, era5_all, sels, '(b)', note=note)
    _draw_weighting_panel(ax3, 'flux', bb, era5_all, sels, '(c)', note=note,
                          ylabel=False)

    fig.tight_layout()
    os.makedirs('./fig/SI', exist_ok=True)
    suffix = '_cloudy' if cloudy_only else ''
    written = save_grl(fig, f'./fig/SI/sfc_alb_ssfr_vs_era5_2panel{suffix}')
    plt.close(fig)
    print('Saved ' + ', '.join(written))


def plot_ssfr_vs_era5_5panel(d=None):
    """(a) case leg; (b)/(c) cloudy legs; (d)/(e) all legs (TOA / flux-weighted)."""
    if d is None:
        d = load_combined()
    bb = _season_broadbands(d)
    era5_cld, sels_cld = _season_selections(d, bb, cloudy_only=True)
    era5_all, sels_all = _season_selections(d, bb, cloudy_only=False)

    plt.close('all')
    fig = plt.figure(figsize=figsize_mm(FULL_WIDTH_MM, 125.0))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    _draw_case_panel(ax_a, d, bb)
    # Top row: cloudy legs only. Same legend entries as the bottom row, so the
    # legend is drawn only under the bottom panels.
    _draw_weighting_panel(ax_b, 'toa', bb, era5_cld, sels_cld, '(b)',
                          note='(cloudy legs only)', xlabel=False, legend=False)
    _draw_weighting_panel(ax_c, 'flux', bb, era5_cld, sels_cld, '(c)',
                          note='(cloudy legs only)', ylabel=False, xlabel=False,
                          legend=False)
    # Bottom row: all legs.
    _draw_weighting_panel(ax_d, 'toa', bb, era5_all, sels_all, '(d)',
                          note='(all legs)')
    _draw_weighting_panel(ax_e, 'flux', bb, era5_all, sels_all, '(e)',
                          note='(all legs)', ylabel=False)

    os.makedirs('./fig/SI', exist_ok=True)
    written = save_grl(fig, './fig/SI/sfc_alb_ssfr_vs_era5_5panel')
    plt.close(fig)
    print('Saved ' + ', '.join(written))


if __name__ == '__main__':
    apply_grl_style()
    d = load_combined()
    plot_ssfr_vs_era5_2panel(d)
    plot_ssfr_vs_era5_2panel(d, cloudy_only=True)
    plot_ssfr_vs_era5_5panel(d)
