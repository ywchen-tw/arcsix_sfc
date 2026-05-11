"""Product-level plotting and array helpers for SSFR atmospheric correction."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .ssfr_atm_corr_settings import gas_bands
except ImportError:
    from ssfr_atm_corr_settings import gas_bands


def ssfr_alb_plot(date_s, tmhr_ranges_select, wvl, alb, color_series,
                  alt_avg_all,
                  modis_bands_nm, modis_alb_legs, modis_alb_file,
                  case_tag,
                  ylabel='SSFR upward/downward ratio',
                  title='SSFR measurement',
                  suptitle='SSFR upward/downward ratio Comparison',
                  file_description='',
                  lon_avg_all=None,
                  lat_avg_all=None,
                  aviris_file=None,
                  aviris_closest=False,
                  aviris_reflectance_wvl=None,
                  aviris_reflectance_spectrum=None,
                  aviris_reflectance_spectrum_unc=None):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if aviris_file is not None:
        aviris_label = 'AVIRIS Reflectance' if aviris_closest else 'All AVIRIS mean'
        ax.scatter(aviris_reflectance_wvl, aviris_reflectance_spectrum, s=5, c='m', label=aviris_label, alpha=0.7)
        ax.fill_between(
            aviris_reflectance_wvl,
            aviris_reflectance_spectrum - aviris_reflectance_spectrum_unc,
            aviris_reflectance_spectrum + aviris_reflectance_spectrum_unc,
            color='m',
            alpha=0.3,
        )
        if modis_alb_file is not None:
            ax.scatter(modis_bands_nm, modis_alb_legs, s=50, c='g', marker='*', label='MODIS Albedo', edgecolors='k')

    for i in range(len(tmhr_ranges_select)):
        if lon_avg_all is not None and lat_avg_all is not None:
            label = 'Z=%.2fkm, lon:%.2f, lat: %.2f' % (alt_avg_all[i], lon_avg_all[i], lat_avg_all[i])
        else:
            label = 'Z=%.2fkm' % (alt_avg_all[i])
        ax.plot(wvl, alb[i, :], '-', color=color_series[i], label=label)
        if aviris_file is None and modis_alb_file is not None:
            ax.scatter(modis_bands_nm, modis_alb_legs[i], s=50, color=color_series[i], marker='*', edgecolors='k')

    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)

    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=13)
    fig.suptitle(suptitle, fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_%s_comparison.png' % (date_s, date_s, case_tag, file_description), bbox_inches='tight', dpi=150)


def ssfr_up_dn_ratio_plot(date_s, tmhr_ranges_select, wvl, up_dn_ratio, color_series, alt_avg_all, case_tag,
                          albedo_used='albedo used: SSFR upward/downward ratio',
                          file_suffix=''):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(tmhr_ranges_select)):
        ax.plot(wvl, up_dn_ratio[i, :], '-', color=color_series[i], label='Z=%.2fkm' % (alt_avg_all[i]))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('SSFR upward/downward ratio', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(albedo_used, fontsize=13)
    fig.suptitle(f'P3 level upward/downward ratio Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_up_dn_ratio_comparison%s.png' % (date_s, date_s, case_tag, file_suffix), bbox_inches='tight', dpi=150)


def weighted_broadband_alb(alb, toa, wvl):
    """TOA-weighted broadband albedo via trapezoidal integration."""
    return np.trapz(alb * toa, x=wvl, axis=-1) / np.trapz(toa, x=wvl, axis=-1)


def fill_nan_ffill_bfill(arr, limit=2):
    """Fill NaNs in a 1-D array using repeated forward/backward fill."""
    s = pd.Series(arr)
    s = s.ffill(limit=limit).bfill(limit=limit)
    while s.isna().any():
        s = s.ffill(limit=limit).bfill(limit=limit)
    return s.values
