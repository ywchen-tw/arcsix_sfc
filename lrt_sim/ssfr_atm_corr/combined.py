import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_LRT_SIM_ROOT = str(_THIS_FILE.parents[1])
_REPO_ROOT = str(_THIS_FILE.parents[2])
for _path in (_REPO_ROOT, _LRT_SIM_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import glob
import copy
import time
from collections import OrderedDict
import datetime
import multiprocessing as mp
import pickle
from dataclasses import dataclass
import logging
from typing import Optional
import h5py
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata, NearestNDInterpolator
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.axes as maxes
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import bisect
import pandas as pd
import xarray as xr
from collections import defaultdict
import gc
from pyproj import Transformer, CRS
from util import *
import types
# mpl.use('Agg')

try:
    if __package__:
        from .helpers import gas_abs_masking
        from .settings import (
            _aspect_,
            _cam_,
            _fdir_cam_img_,
            _fdir_data_,
            _fdir_general_,
            _fdir_main_,
            _fdir_sat_data_,
            _fdir_sat_img_,
            _fdir_sat_img_vn_,
            _fdir_tmp_,
            _fdir_tmp_graph_,
            _hsk_,
            _mission_,
            _platform_,
            _preferred_region_,
            _ssfr1_,
            _ssfr2_,
            _spns_,
            _title_extra_,
            _tmhr_range_,
            _wavelength_,
            gas_bands,
        )
    else:
        from helpers import gas_abs_masking
        from settings import (
            _aspect_,
            _cam_,
            _fdir_cam_img_,
            _fdir_data_,
            _fdir_general_,
            _fdir_main_,
            _fdir_sat_data_,
            _fdir_sat_img_,
            _fdir_sat_img_vn_,
            _fdir_tmp_,
            _fdir_tmp_graph_,
            _hsk_,
            _mission_,
            _platform_,
            _preferred_region_,
            _ssfr1_,
            _ssfr2_,
            _spns_,
            _title_extra_,
            _tmhr_range_,
            _wavelength_,
            gas_bands,
        )
except ImportError:
    from lrt_sim.ssfr_atm_corr.helpers import gas_abs_masking
    from lrt_sim.ssfr_atm_corr.settings import (
        _aspect_,
        _cam_,
        _fdir_cam_img_,
        _fdir_data_,
        _fdir_general_,
        _fdir_main_,
        _fdir_sat_data_,
        _fdir_sat_img_,
        _fdir_sat_img_vn_,
        _fdir_tmp_,
        _fdir_tmp_graph_,
        _hsk_,
        _mission_,
        _platform_,
        _preferred_region_,
        _ssfr1_,
        _ssfr2_,
        _spns_,
        _title_extra_,
        _tmhr_range_,
        _wavelength_,
        gas_bands,
    )


# ---------------------------------------------------------------------------
# Grain size retrieval constants (ART theory, 1700 nm)
# ---------------------------------------------------------------------------
_GS_WVL_ = 1700.0    # nm — retrieval wavelength
_GS_A_   = 5.8       # form factor for randomly oriented hexagonal plates/columns
# Temperature-dependent k at 1700 nm (Warren & Brandt 2008, Table 2).
# NOTE: verify these node values against the paper before production use.
_GS_CHI_T_NODES_ = np.array([213.0, 233.0, 253.0, 266.0, 273.0])     # K
_GS_CHI_K_NODES_ = np.array([5.8e-4, 7.8e-4, 1.04e-3, 1.38e-3, 1.61e-3])  # imaginary index

# ---------------------------------------------------------------------------
# Grain size retrieval constants — ratio method (AART)
# Warren & Brandt 2008 ice optical constants at 266 K
# NOTE: verify k values for 865/1020/1650 nm against the paper.
# ---------------------------------------------------------------------------
# 1280 / 1100 nm pair
_GS_RATIO_1280_WVL1_ = 1280.0;  _GS_RATIO_1280_CHI1_ = 1.39e-5   # imaginary k at 1280 nm
_GS_RATIO_1280_WVL2_ = 1100.0;  _GS_RATIO_1280_CHI2_ = 2.89e-7   # imaginary k at 1100 nm
# 1650 / 1020 nm pair (Painter et al. / Nolin & Dozier style)
_GS_RATIO_1650_WVL1_ = 1650.0;  _GS_RATIO_1650_CHI1_ = 1.2e-3    # imaginary k at 1650 nm
_GS_RATIO_1650_WVL2_ = 1020.0;  _GS_RATIO_1650_CHI2_ = 2.0e-7    # imaginary k at 1020 nm
# 1020 / 865 nm pair (Kokhanovsky / Sentinel-3 OLCI style)
_GS_RATIO_1020_WVL1_ = 1020.0;  _GS_RATIO_1020_CHI1_ = 2.0e-7    # imaginary k at 1020 nm
_GS_RATIO_1020_WVL2_ =  865.0;  _GS_RATIO_1020_CHI2_ = 2.4e-9    # imaginary k at 865 nm
# _GS_A_ = 5.8 is shared across all pairs

# ---------------------------------------------------------------------------
# Grain size retrieval constants — single-band AART at 1240 nm
# 1240 nm sits in the shoulder of the 1.25 μm ice band: enough absorption
# for reliable single-band AART without excessive T-sensitivity.
# Temperature-dependent k from Warren & Brandt 2008 (Table 2).
# NOTE: verify node values against the paper before production use.
# ---------------------------------------------------------------------------
_GS_1240_WVL_ = 1240.0   # nm
_GS_1240_CHI_T_NODES_ = np.array([213.0, 233.0, 253.0, 266.0, 273.0])      # K
_GS_1240_CHI_K_NODES_ = np.array([4.5e-6, 5.5e-6, 6.4e-6, 7.5e-6, 8.2e-6])  # imaginary index


# ---------------------------------------------------------------------------
# SeasonData dataclass
# ---------------------------------------------------------------------------

@dataclass
class SeasonData:
    name: str                            # 'spring' or 'summer'
    wvl: np.ndarray                      # (n_wvl,) — season-specific wavelength grid
    lon: np.ndarray                      # (n_pts,)
    lat: np.ndarray                      # (n_pts,)
    alt: np.ndarray                      # (n_pts,)
    time: np.ndarray                     # (n_pts,)
    dates: np.ndarray                    # (n_pts,) int
    conditions: np.ndarray               # (n_pts,) str
    case_tags: np.ndarray                # (n_pts,) str
    case_tags_leg: np.ndarray            # (n_legs,) str — per-leg, not per-point
    lon_avg: np.ndarray                  # (n_legs,)
    lat_avg: np.ndarray                  # (n_legs,)
    alt_avg: np.ndarray                  # (n_legs,)
    icing: np.ndarray                    # (n_pts,)
    icing_pre: np.ndarray                # (n_pts,)
    fdn: np.ndarray                      # (n_pts, n_wvl)
    fup: np.ndarray                      # (n_pts, n_wvl)
    toa_expand: np.ndarray               # (n_pts, n_wvl)
    alb_iter1: np.ndarray                # (n_pts, n_wvl)
    alb_iter2: np.ndarray                # (n_pts, n_wvl)
    broadband_alb_iter2: np.ndarray      # (n_pts,)
    bb_alb_iter2_690_1190: np.ndarray    # (n_pts,) TOA-weighted broadband albedo 690–1190 nm
    kt19_sfc_T: np.ndarray               # (n_pts,)  surface temperature from KT19 (deg C)
    sza: np.ndarray                      # (n_pts,)
    # filled later by collocation:
    ice_frac: np.ndarray = None          # (n_pts,)
    nad_hdrf: np.ndarray = None          # (n_pts,)
    nad_rad: np.ndarray = None           # (n_pts,)
    myi_ratio: np.ndarray = None         # (n_pts,)
    fyi_ratio: np.ndarray = None         # (n_pts,)
    yi_ratio: np.ndarray = None          # (n_pts,)
    ice_ratio: np.ndarray = None         # (n_pts,)
    ow_ratio: np.ndarray = None          # (n_pts,)
    ice_age: np.ndarray = None           # (n_pts,)
    era5_alb: np.ndarray = None          # (n_pts,)
    grain_size: np.ndarray = None        # (n_pts,)  r_opt via single-band AART at 1700 nm (m)
    grain_size_1240: np.ndarray = None   # (n_pts,)  r_opt via single-band AART at 1240 nm (m)
    grain_size_ratio: np.ndarray = None           # (n_pts,)  r_opt via 1280/1100 nm ratio (m)
    grain_size_ratio_1650_1020: np.ndarray = None # (n_pts,)  r_opt via 1650/1020 nm ratio (m)
    grain_size_ratio_1020_865: np.ndarray = None  # (n_pts,)  r_opt via 1020/865 nm ratio (m)
    brt19h: np.ndarray = None            # (n_pts,)  brightness temperature 19 GHz H-pol (K)
    brt37h: np.ndarray = None            # (n_pts,)  brightness temperature 37 GHz H-pol (K)
    brt37v: np.ndarray = None            # (n_pts,)  brightness temperature 37 GHz V-pol (K)


# ---------------------------------------------------------------------------
# Helper: extract date_s and case_tag from filename
# ---------------------------------------------------------------------------

def extract_date_casetag(filepath):
    """Extract date_s and case_tag from a processed atmospheric-correction pickle."""
    base_name = os.path.basename(filepath)
    stem = base_name.replace('sfc_alb_update_', '').replace('.pkl', '')
    parts = stem.split('_', 1)
    date_s = parts[0]
    case_tag = parts[1].split('_time_')[0] if len(parts) > 1 else 'default'
    return date_s, case_tag


def first_present(mapping, keys, default=None, required=True):
    """Return the first present key from a processed pickle dictionary."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    if required:
        raise KeyError(f"Missing required key; tried: {', '.join(keys)}")
    return default


# ---------------------------------------------------------------------------
# Helper: load one season worth of files and return a SeasonData
# ---------------------------------------------------------------------------

def load_season(files, name):
    """
    Load a list of sfc_alb pkl files for one season and return a SeasonData.

    Parameters
    ----------
    files : list of str
        Sorted list of pkl file paths belonging to this season.
    name : str
        'spring' or 'summer'.

    Returns
    -------
    SeasonData
    """
    if not files:
        raise FileNotFoundError(f"No {name} sfc_alb_update_*.pkl files found for combination.")

    # Collect lists of per-file arrays
    wvl = None
    lons, lats, alts, times = [], [], [], []
    dates_all, conditions_all, case_tags_all, case_tags_leg_all = [], [], [], []
    lon_avgs, lat_avgs, alt_avgs = [], [], []
    icings, icing_pres = [], []
    fdns, fups, toa_expands = [], [], []
    alb_iter1s, alb_iter2s = [], []
    # also keep broadband_alb_iter1 arrays (from file) for saving
    bb_alb_iter1_list, bb_alb_iter2_list = [], []
    bb_alb_iter1_filter_list, bb_alb_iter2_filter_list = [], []
    kt19_sfc_T_list = []
    sza_list = []

    for filepath in files:
        print(f"Processing surface albedo file: {filepath}")
        with open(filepath, 'rb') as f:
            d = pickle.load(f)

        date_s, case_tag = extract_date_casetag(filepath)

        condition = 'cloudy'
        if 'spiral' in case_tag.lower():
            condition = 'spiral'
        elif 'clear' in case_tag.lower():
            condition = 'clear'

        lon_all = np.asarray(first_present(d, ('lon_all',)))
        lat_all = np.asarray(first_present(d, ('lat_all',)))
        alt_all = np.asarray(first_present(d, ('alt_all',)))
        time_all = np.asarray(first_present(d, ('time_all',)))
        native_wvl = np.asarray(first_present(d, ('native_wvl', 'wvl')))

        if wvl is None:
            wvl = native_wvl

        n_pts = len(lon_all)
        n_legs = len(first_present(d, ('lon_avg',)))

        lons.append(lon_all)
        lats.append(lat_all)
        alts.append(alt_all)
        times.append(time_all)
        dates_all.extend([int(date_s)] * n_pts)
        conditions_all.extend([condition] * n_pts)
        case_tags_all.extend([case_tag] * n_pts)
        case_tags_leg_all.extend([case_tag] * n_legs)

        lon_avgs.append(first_present(d, ('lon_avg',)))
        lat_avgs.append(first_present(d, ('lat_avg',)))
        alt_avgs.append(first_present(d, ('alt_avg',)))

        icings.append(first_present(d, ('icing_all',), np.zeros(n_pts, dtype=bool), required=False))
        icing_pres.append(first_present(d, ('icing_pre_all',), np.zeros(n_pts, dtype=bool), required=False))
        fdns.append(first_present(d, ('fdn_all',)))
        fups.append(first_present(d, ('fup_all',)))
        toa_expands.append(first_present(d, ('toa_expand_all',)))
        alb_iter1s.append(first_present(d, ('alb_iter1_all_1s', 'alb_iter1_all')))
        alb_iter2s.append(first_present(d, ('alb_final_all_1s', 'alb_final_all', 'alb_iter2_all_1s', 'alb_iter2_all')))
        bb_alb_iter1 = first_present(d, ('broadband_alb_iter1_all_1s', 'broadband_alb_iter1_all'), required=True)
        bb_alb_iter2 = first_present(
            d,
            ('broadband_alb_final_all_1s', 'broadband_alb_final_all', 'broadband_alb_iter2_all_1s', 'broadband_alb_iter2_all'),
            required=True,
        )
        bb_alb_iter1_list.append(bb_alb_iter1)
        bb_alb_iter2_list.append(bb_alb_iter2)
        bb_alb_iter1_filter_list.append(
            first_present(d, ('broadband_alb_iter1_all_filter_1s', 'broadband_alb_iter1_all_filter'), bb_alb_iter1, required=False)
        )
        bb_alb_iter2_filter_list.append(
            first_present(
                d,
                ('broadband_alb_final_all_filter_1s', 'broadband_alb_iter2_all_filter_1s', 'broadband_alb_iter2_all_filter'),
                bb_alb_iter2,
                required=False,
            )
        )
        kt19_sfc_T_list.append(
            first_present(d, ('kt19_sfc_T_all',), np.full(n_pts, np.nan), required=False)
        )
        sza_list.append(first_present(d, ('sza_all',), np.full(n_pts, np.nan), required=False))

    lon_arr       = np.concatenate(lons, axis=0)
    lat_arr       = np.concatenate(lats, axis=0)
    alt_arr       = np.concatenate(alts, axis=0)
    time_arr      = np.concatenate(times, axis=0)
    dates_arr     = np.array(dates_all)
    cond_arr      = np.array(conditions_all)
    ctag_arr      = np.array(case_tags_all)
    ctag_leg_arr  = np.array(case_tags_leg_all)
    lon_avg_arr   = np.concatenate(lon_avgs, axis=0)
    lat_avg_arr   = np.concatenate(lat_avgs, axis=0)
    alt_avg_arr   = np.concatenate(alt_avgs, axis=0)
    icing_arr     = np.concatenate(icings, axis=0)
    icing_pre_arr = np.concatenate(icing_pres, axis=0)
    fdn_arr       = np.concatenate(fdns, axis=0)
    fup_arr       = np.concatenate(fups, axis=0)
    toa_arr       = np.concatenate(toa_expands, axis=0)
    alb1_arr      = np.concatenate(alb_iter1s, axis=0)
    alb2_arr      = np.concatenate(alb_iter2s, axis=0)

    # Compute TOA-weighted broadband albedo from the concatenated arrays
    bb_alb_iter2 = (
        np.trapz(alb2_arr * toa_arr, wvl, axis=1) /
        np.trapz(toa_arr, wvl, axis=1)
    )
    
    # Compute TOA-weighted broadband albedo from the concatenated arrays for wavelength range 690 to 1190 nm
    wvl_mask = (wvl >= 690) & (wvl <= 1190)
    bb_alb_iter2_690_1190 = (
        np.trapz(alb2_arr[:, wvl_mask] * toa_arr[:, wvl_mask], wvl[wvl_mask], axis=1) /
        np.trapz(toa_arr[:, wvl_mask], wvl[wvl_mask], axis=1)
    )
    

    # Store the file-provided broadband arrays on the object as extra attributes
    # (needed for the output dicts)
    season = SeasonData(
        name=name,
        wvl=wvl,
        lon=lon_arr,
        lat=lat_arr,
        alt=alt_arr,
        time=time_arr,
        dates=dates_arr,
        conditions=cond_arr,
        case_tags=ctag_arr,
        case_tags_leg=ctag_leg_arr,
        lon_avg=lon_avg_arr,
        lat_avg=lat_avg_arr,
        alt_avg=alt_avg_arr,
        icing=icing_arr,
        icing_pre=icing_pre_arr,
        fdn=fdn_arr,
        fup=fup_arr,
        toa_expand=toa_arr,
        alb_iter1=alb1_arr,
        alb_iter2=alb2_arr,
        broadband_alb_iter2=bb_alb_iter2,
        bb_alb_iter2_690_1190=bb_alb_iter2_690_1190,
        kt19_sfc_T=np.concatenate(kt19_sfc_T_list, axis=0),
        sza=np.concatenate(sza_list, axis=0),
    )
    # Attach extra broadband arrays (from files) for saving
    season._bb_alb_iter1        = np.concatenate(bb_alb_iter1_list, axis=0)
    season._bb_alb_iter2_file   = np.concatenate(bb_alb_iter2_list, axis=0)
    season._bb_alb_iter1_filter = np.concatenate(bb_alb_iter1_filter_list, axis=0)
    season._bb_alb_iter2_filter = np.concatenate(bb_alb_iter2_filter_list, axis=0)

    return season


# ---------------------------------------------------------------------------
# Helper: make a north-polar-stereo map figure
# ---------------------------------------------------------------------------

def make_polar_map(lon_all, lat_all, figsize=(8, 4)):
    """
    Create a NorthPolarStereo figure/ax centered on the data extent.

    Parameters
    ----------
    lon_all : array-like
        All longitudes (used to determine extent and central longitude).
    lat_all : array-like
        All latitudes (used to determine extent).
    figsize : tuple, optional

    Returns
    -------
    fig, ax
    """
    plt.close('all')
    central_lon = np.mean(lon_all)
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)}
    )
    ax.coastlines()
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
    ax.set_extent(
        [np.min(lon_all) - 2, np.max(lon_all) + 2,
         np.min(lat_all) - 2, np.max(lat_all) + 2],
        crs=ccrs.PlateCarree()
    )
    return fig, ax


# ---------------------------------------------------------------------------
# Helper: diagnostic ECICE maps for one date
# ---------------------------------------------------------------------------

def plot_ecice_diagnostics(date_s, lon, lat, lonlat_shape,
                           myi, fyi, yi, ice, ow,
                           brt19h, brt37h, brt37v,
                           lon_flight, lat_flight, myi_flight,
                           era5_lon_mesh, era5_lat_mesh, era5_alb_date,
                           era5_alb_flight, bb_alb_flight,
                           lon_all, lat_all):
    """
    Generate all ECICE/ERA5 diagnostic maps for one (date, season) combination.
    Saves figures to ./fig/ice_age/.
    """
    bt_min, bt_max = 100, 250

    # --- MYI concentration ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 4))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       myi.reshape(lonlat_shape),
                       transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c1, ax=ax, label='Multi-year Ice Conc (%)')
    fig.suptitle(f"{date_s}", fontsize=16)
    fig.savefig(f'./fig/ice_age/myi_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- BRT19H ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 4))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       brt19h,
                       transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=bt_min, vmax=bt_max)
    fig.colorbar(c1, ax=ax, label='BRT19H (K)')
    fig.savefig(f'./fig/ice_age/brt19h_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- BRT37H ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 4))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       brt37h,
                       transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=bt_min, vmax=bt_max)
    fig.colorbar(c1, ax=ax, label='BRT37H (K)')
    fig.savefig(f'./fig/ice_age/brt37h_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- BRT37V ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 4))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       brt37v,
                       transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=bt_min, vmax=bt_max)
    fig.colorbar(c1, ax=ax, label='BRT37V (K)')
    fig.savefig(f'./fig/ice_age/brt37v_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- MYI + FYI combined conc (3-panel) ---
    central_lon = np.mean(lon_all)
    lon_min = np.min(lon_all) - 2
    lon_max = np.max(lon_all) + 2
    lat_min = np.min(lat_all) - 2
    lat_max = np.max(lat_all) + 2
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(24, 4),
        subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)}
    )
    c1 = ax1.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                        myi.reshape(lonlat_shape),
                        transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c1, ax=ax1, label='Multi-year (%)')
    ax1.set_title('Multi-year Ice Conc.')
    c2 = ax2.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                        fyi.reshape(lonlat_shape),
                        transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c2, ax=ax2, label='First-year (%)')
    ax2.set_title('First-year Ice Conc.')
    c3 = ax3.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                        (myi + fyi).reshape(lonlat_shape),
                        transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c3, ax=ax3, label='Multi-year + First-year (%)')
    ax3.set_title('Multi-year + First-year Ice Conc.')
    for ax in [ax1, ax2, ax3]:
        ax.coastlines()
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    fig.suptitle(f"{date_s}", fontsize=16)
    fig.savefig(f'./fig/ice_age/myi_fyi_combined_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- MYI + FYI combined perc (3-panel) ---
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(24, 4),
        subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)}
    )
    c1 = ax1.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                        (myi / ice * 100).reshape(lonlat_shape),
                        transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c1, ax=ax1, label='Multi-year perc (%)')
    ax1.set_title('Multi-year Ice Perc.')
    c2 = ax2.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                        (fyi / ice * 100).reshape(lonlat_shape),
                        transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c2, ax=ax2, label='First-year perc (%)')
    ax2.set_title('First-year Ice Perc.')
    c3 = ax3.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                        ((myi + fyi) / ice * 100).reshape(lonlat_shape),
                        transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c3, ax=ax3, label='Multi-year + First-year perc (%)')
    ax3.set_title('Multi-year + First-year Ice Perc.')
    for ax in [ax1, ax2, ax3]:
        ax.coastlines()
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    fig.suptitle(f"{date_s}", fontsize=16)
    fig.savefig(f'./fig/ice_age/myi_fyi_combined_perc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- FYI conc ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 4))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       fyi.reshape(lonlat_shape),
                       transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c1, ax=ax, label='First-year (%)')
    fig.suptitle(f"{date_s}", fontsize=16)
    fig.savefig(f'./fig/ice_age/fyi_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- MYI+FYI conc ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 4))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       (myi + fyi).reshape(lonlat_shape),
                       transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c1, ax=ax, label='Multi-year + First-year (%)')
    fig.suptitle(f"{date_s}", fontsize=16)
    fig.savefig(f'./fig/ice_age/myi_fyi_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- MYI+FYI+YI conc ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 4))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       (myi + fyi + yi).reshape(lonlat_shape),
                       transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c1, ax=ax, label='Multi-year + First-year + Young Ice Conc (%)')
    fig.suptitle(f"{date_s}", fontsize=16)
    fig.savefig(f'./fig/ice_age/myi_fyi_yi_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Total ice conc ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 6))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       ice.reshape(lonlat_shape),
                       transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c1, ax=ax, label='Total Ice Conc (%)')
    fig.savefig(f'./fig/ice_age/total_ice_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Open water conc ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 6))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       ow.reshape(lonlat_shape),
                       transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c1, ax=ax, label='Open Water Conc (%)')
    fig.savefig(f'./fig/ice_age/ow_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Total ice + OW conc ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 6))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       (ice + ow).reshape(lonlat_shape),
                       transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    fig.colorbar(c1, ax=ax, label='Total ice + Open water Conc (%)')
    fig.savefig(f'./fig/ice_age/total_ice_ow_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- MYI percentage with flight scatter ---
    myi_fyi_yi_total = ice
    myi_fyi_yi_total_flight = myi_flight  # already: myi+fyi+yi for flight points
    myi_to_total_ratio = myi / (myi_fyi_yi_total + 1e-7) * 100
    myi_to_total_ratio_flight = myi_flight / (myi_fyi_yi_total_flight + 1e-7) * 100
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 4))
    c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape),
                       myi_to_total_ratio.reshape(lonlat_shape),
                       transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
    ax.scatter(lon_flight, lat_flight, c=myi_to_total_ratio_flight,
               transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100, edgecolors='k')
    fig.colorbar(c1, ax=ax, label='Multi-year Ice Percentage (%)')
    fig.suptitle(f"{date_s}", fontsize=16)
    fig.savefig(f'./fig/ice_age/myi_percentage_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- ERA5 albedo map + flight scatter ---
    fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 6))
    c1 = ax.pcolormesh(era5_lon_mesh, era5_lat_mesh, era5_alb_date[0],
                       transform=ccrs.PlateCarree(), cmap='jet', vmin=0.4, vmax=0.8)
    ax.scatter(lon_flight, lat_flight, c=era5_alb_flight,
               transform=ccrs.PlateCarree(), cmap='jet', vmin=0.4, vmax=0.8, edgecolors='k')
    fig.colorbar(c1, ax=ax, label='ERA5 Forecast Albedo')
    fig.savefig(f'./fig/ice_age/era5_alb_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- ERA5 vs broadband albedo scatter ---
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(era5_alb_flight, bb_alb_flight, c='k', alpha=0.5)
    ax.plot([0, 1], [0, 1], 'k--', label='1:1 Line')
    ax.set_xlabel('ERA5 Forecast Albedo')
    ax.set_ylabel('Retrieved Broadband Surface Albedo')
    fig.savefig(f'./fig/ice_age/bb_alb_{date_s}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Collocation: ice_frac
# ---------------------------------------------------------------------------

def collocate_ice_frac(season, ice_frac_data, time_offsets):
    """
    Match ice_frac_data (time-series) to each point in season by closest time.

    Parameters
    ----------
    season : SeasonData
    ice_frac_data : dict with keys 'date', 'time', 'ice_frac', 'nad_hdrf', 'nad_rad'
    time_offsets : dict mapping date_str -> float offset in hours

    Side-effects: assigns season.ice_frac, season.nad_hdrf, season.nad_rad
    Returns: np.ndarray (n_pts,)
    """
    ice_frac_date   = ice_frac_data['date']
    ice_frac_time   = ice_frac_data['time']
    ice_frac_values = ice_frac_data['ice_frac']
    nad_hdrf_values = ice_frac_data['nad_hdrf']
    nad_rad_values  = ice_frac_data['nad_rad']

    result       = np.full(len(season.time), np.nan)
    nad_hdrf_out = np.full(len(season.time), np.nan)
    nad_rad_out  = np.full(len(season.time), np.nan)

    for date_s in sorted(set(season.dates)):
        date_mask = season.dates == date_s
        time_pts = season.time[date_mask]
        t_offset = time_offsets.get(str(date_s), 0)

        date_idx           = ice_frac_date == int(date_s)
        ice_frac_all_date  = ice_frac_values[date_idx]
        nad_hdrf_all_date  = nad_hdrf_values[date_idx]
        nad_rad_all_date   = nad_rad_values[date_idx]
        ice_frac_time_date = ice_frac_time[date_idx].copy()
        ice_frac_time_date += t_offset

        tmp          = np.full(len(time_pts), np.nan)
        tmp_nad_hdrf = np.full(len(time_pts), np.nan)
        tmp_nad_rad  = np.full(len(time_pts), np.nan)
        for i, t in enumerate(time_pts):
            time_diff = np.abs(ice_frac_time_date - t)
            if i % 1000 == 0:
                print(f"  time index {i}, t: {t}, time_diff min: {np.min(time_diff)}")
            if np.min(time_diff) <= 1. / 60 / 60:   # within 1 s
                closest_index = np.argmin(time_diff)
                tmp[i]          = ice_frac_all_date[closest_index].copy()
                tmp_nad_hdrf[i] = nad_hdrf_all_date[closest_index].copy()
                tmp_nad_rad[i]  = nad_rad_all_date[closest_index].copy()
        result[date_mask]       = tmp.copy()
        nad_hdrf_out[date_mask] = tmp_nad_hdrf.copy()
        nad_rad_out[date_mask]  = tmp_nad_rad.copy()

    season.ice_frac  = result
    season.nad_hdrf  = nad_hdrf_out
    season.nad_rad   = nad_rad_out
    return result


# ---------------------------------------------------------------------------
# Collocation: ECICE (myi, fyi, yi, ice, ow, era5_alb)
# ---------------------------------------------------------------------------

def collocate_ecice(season, data_dir,
                    era5_lon_mesh, era5_lat_mesh, era5_alb, era5_time_dates_str,
                    lon_all, lat_all):
    """
    For each date in the season, load the ECICE NetCDF, griddata-interpolate
    to flight track, and assign to season fields.
    Also calls plot_ecice_diagnostics for each date.

    Parameters
    ----------
    season : SeasonData
    data_dir : str   path to directory containing ECICE-IcetypesUncorrected-*.nc
    era5_lon_mesh, era5_lat_mesh : 2-D meshgrids for ERA5
    era5_alb : 3-D array (time, lat, lon)
    era5_time_dates_str : array of date strings matching era5_alb time axis
    lon_all, lat_all : combined (spring+summer) lon/lat for map extents
    """
    n = len(season.lon)
    myi_out    = np.full(n, np.nan)
    fyi_out    = np.full(n, np.nan)
    yi_out     = np.full(n, np.nan)
    ice_out    = np.full(n, np.nan)
    ow_out     = np.full(n, np.nan)
    era5_out   = np.full(n, np.nan)
    brt19h_out = np.full(n, np.nan)
    brt37h_out = np.full(n, np.nan)
    brt37v_out = np.full(n, np.nan)

    for date_s in sorted(set(season.dates)):
        date_s_str = str(date_s)
        print(f"Processing ice age data for date: {date_s_str}")
        print(f"Ice age data file: {data_dir}/ECICE-IcetypesUncorrected-{date_s_str}.nc")

        with Dataset(f'{data_dir}/ECICE-IcetypesUncorrected-{date_s_str}.nc', 'r') as nc:
            lon_nc        = nc.variables['LON'][:]
            lat_nc        = nc.variables['LAT'][:]
            myi_nc        = nc.variables['MYI'][:]
            fyi_nc        = nc.variables['FYI'][:]
            yi_nc         = nc.variables['YI'][:]
            ice_nc        = nc.variables['TOTAL_ICE'][:]
            ow_nc         = nc.variables['OW'][:]
            brt19h_nc     = nc.variables['BRT19H'][:]
            brt37h_nc     = nc.variables['BRT37H'][:]
            brt37v_nc     = nc.variables['BRT37V'][:]

        lonlat_shape = lon_nc.shape
        lon_f    = lon_nc.flatten()
        lat_f    = lat_nc.flatten()
        myi_f    = myi_nc.flatten()
        fyi_f    = fyi_nc.flatten()
        yi_f     = yi_nc.flatten()
        ice_f    = ice_nc.flatten()
        ow_f     = ow_nc.flatten()
        brt19h_f = brt19h_nc.flatten()
        brt37h_f = brt37h_nc.flatten()
        brt37v_f = brt37v_nc.flatten()

        date_mask = season.dates == date_s
        if date_mask.sum() > 0:
            lon_pts = season.lon[date_mask]
            lat_pts = season.lat[date_mask]

            myi_out[date_mask]  = griddata((lon_f, lat_f), myi_f,  (lon_pts, lat_pts), method='nearest')
            fyi_out[date_mask]  = griddata((lon_f, lat_f), fyi_f,  (lon_pts, lat_pts), method='nearest')
            yi_out[date_mask]   = griddata((lon_f, lat_f), yi_f,   (lon_pts, lat_pts), method='nearest')
            ice_out[date_mask]  = griddata((lon_f, lat_f), ice_f,  (lon_pts, lat_pts), method='nearest')
            ow_out[date_mask]     = griddata((lon_f, lat_f), ow_f,     (lon_pts, lat_pts), method='nearest')
            brt19h_out[date_mask] = griddata((lon_f, lat_f), brt19h_f, (lon_pts, lat_pts), method='nearest')
            brt37h_out[date_mask] = griddata((lon_f, lat_f), brt37h_f, (lon_pts, lat_pts), method='nearest')
            brt37v_out[date_mask] = griddata((lon_f, lat_f), brt37v_f, (lon_pts, lat_pts), method='nearest')

            era5_alb_date = era5_alb[era5_time_dates_str == date_s_str]
            era5_out[date_mask] = griddata(
                (era5_lon_mesh.flatten(), era5_lat_mesh.flatten()),
                era5_alb_date[0].flatten(),
                (lon_pts, lat_pts),
                method='nearest'
            )

            # Build myi_flight for the percentage diagnostic map:
            # original used myi+fyi+yi as denominator for percentage
            myi_fyi_yi_flight = myi_out[date_mask] + fyi_out[date_mask] + yi_out[date_mask]

            plot_ecice_diagnostics(
                date_s=date_s_str,
                lon=lon_f, lat=lat_f,
                lonlat_shape=lonlat_shape,
                myi=myi_f, fyi=fyi_f, yi=yi_f, ice=ice_f, ow=ow_f,
                brt19h=brt19h_nc, brt37h=brt37h_nc, brt37v=brt37v_nc,
                lon_flight=lon_pts, lat_flight=lat_pts,
                myi_flight=myi_fyi_yi_flight,
                era5_lon_mesh=era5_lon_mesh, era5_lat_mesh=era5_lat_mesh,
                era5_alb_date=era5_alb_date,
                era5_alb_flight=era5_out[date_mask],
                bb_alb_flight=season.broadband_alb_iter2[date_mask],
                lon_all=lon_all, lat_all=lat_all,
            )

    season.myi_ratio = myi_out
    season.fyi_ratio = fyi_out
    season.yi_ratio  = yi_out
    season.ice_ratio = ice_out
    season.ow_ratio  = ow_out
    season.era5_alb  = era5_out
    season.brt19h    = brt19h_out
    season.brt37h    = brt37h_out
    season.brt37v    = brt37v_out


# ---------------------------------------------------------------------------
# Collocation: AMSR2 ice concentration
# ---------------------------------------------------------------------------

def collocate_amsr2_ice(season, data_dir):
    """
    For each date in the season, load the matching AMSR2 NSIDC-0803 NetCDF,
    reproject from polar stereographic to lon/lat, nearest-neighbor interpolate
    to flight track, and assign to season.amsr2_ice_conc.

    Parameters
    ----------
    season   : SeasonData
    data_dir : str  path to directory containing NSIDC-0803_SEAICE_AMSR2_N_2024*.nc
    """
    pattern    = 'NSIDC-0803_SEAICE_AMSR2_N_2024*.nc'
    amsr2_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    amsr2_dates = [os.path.basename(f).split('_')[4][0:8] for f in amsr2_files]

    file_by_date = {d: f for d, f in zip(amsr2_dates, amsr2_files)}

    n       = len(season.lon)
    ice_out = np.full(n, np.nan)

    for date_s in sorted(set(season.dates)):
        date_s_str = str(date_s)
        fn = file_by_date.get(date_s_str)
        if fn is None:
            print(f"collocate_amsr2_ice: no file found for date {date_s_str}, skipping.")
            continue

        print(f"Processing AMSR2 ice concentration for date: {date_s_str}")

        with Dataset(fn, 'r') as ds:
            x = ds.variables['x'][:]
            y = ds.variables['y'][:]

            v = ds.variables['ICECON']
            if 'time' in v.dimensions:
                idx = tuple(0 if d == 'time' else slice(None) for d in v.dimensions)
                arr = np.array(v[idx], dtype=np.float32)
            else:
                arr = np.array(v[:], dtype=np.float32)

            fill = getattr(v, '_FillValue', None) or getattr(v, 'missing_value', None)
            if fill is not None:
                arr = np.where(arr == fill, np.nan, arr)
            arr[arr > 250] = np.nan

            ice_conc_pct = arr * 100.0

            if 'crs' in ds.variables:
                crs_var = ds.variables['crs']
                wkt     = getattr(crs_var, 'crs_wkt',    None) or getattr(crs_var, 'spatial_ref', None)
                proj4   = getattr(crs_var, 'proj4',      None) or getattr(crs_var, 'proj4text',   None)
                if wkt:
                    src_crs = CRS.from_wkt(wkt)
                elif proj4:
                    src_crs = CRS.from_string(proj4)
                else:
                    epsg    = getattr(crs_var, 'epsg', None)
                    src_crs = CRS.from_epsg(int(epsg)) if epsg else CRS.from_epsg(3411)
            else:
                src_crs = CRS.from_epsg(3411)

        xx, yy = np.meshgrid(x, y)
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
        lon_flat, lat_flat = transformer.transform(xx.ravel(), yy.ravel())

        date_mask = season.dates == date_s
        if date_mask.sum() > 0:
            ice_out[date_mask] = griddata(
                (lon_flat, lat_flat),
                ice_conc_pct.ravel(),
                (season.lon[date_mask], season.lat[date_mask]),
                method='nearest'
            )

    season.amsr2_ice_conc = ice_out


# ---------------------------------------------------------------------------
# Collocation: NSIDC ice age
# ---------------------------------------------------------------------------

def collocate_nsidc_ice_age(season, nsidc_lon, nsidc_lat, nsidc_ice_age, time_nc_dates,
                             lon_all, lat_all):
    """
    Match NSIDC weekly ice age to flight track by closest date, then griddata.

    Parameters
    ----------
    season : SeasonData
    nsidc_lon, nsidc_lat : 2-D arrays
    nsidc_ice_age : 3-D array (time, lat, lon)
    time_nc_dates : list/array of datetime objects
    lon_all, lat_all : combined lon/lat for map extents

    Side-effects: assigns season.ice_age
    """
    n = len(season.lon)
    ice_age_out = np.full(n, np.nan)
    nsidc_lon_f = nsidc_lon.flatten()
    nsidc_lat_f = nsidc_lat.flatten()

    for date_s in sorted(set(season.dates)):
        date_s_str = str(date_s)
        date_s_dt  = datetime.datetime.strptime(date_s_str, '%Y%m%d')
        time_diff  = np.abs(np.array([(t - date_s_dt).days for t in time_nc_dates]))
        closest_index = np.argmin(time_diff)
        print(f"  Closest date in ice age data: {time_nc_dates[closest_index]}, "
              f"index: {closest_index}, time diff: {time_diff[closest_index]} days")

        ice_age_nc = nsidc_ice_age[closest_index, :, :].flatten()
        ice_age_nc[ice_age_nc == 20] = np.nan  # land
        ice_age_nc[ice_age_nc == 21] = np.nan  # near coast

        date_mask = season.dates == date_s
        ice_age_mesh = griddata(
            (nsidc_lon_f, nsidc_lat_f), ice_age_nc,
            (season.lon[date_mask], season.lat[date_mask]),
            method='nearest'
        )
        ice_age_out[date_mask] = ice_age_mesh.copy()

    ice_age_out[np.isnan(ice_age_out)] = 0  # set ice age to 0 if nan

    # Plot ice age map for each date
    for date_s in sorted(set(season.dates)):
        date_s_str = str(date_s)
        date_s_dt  = datetime.datetime.strptime(date_s_str, '%Y%m%d')
        time_diff  = np.abs(np.array([(t - date_s_dt).days for t in time_nc_dates]))
        closest_index = np.argmin(time_diff)
        ice_age_nc = nsidc_ice_age[closest_index, :, :].flatten()
        ice_age_nc[ice_age_nc == 20] = np.nan
        ice_age_nc[ice_age_nc == 21] = np.nan

        fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 4))
        c1 = ax.pcolormesh(nsidc_lon, nsidc_lat, ice_age_nc.reshape(nsidc_lon.shape),
                           transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=5)
        fig.colorbar(c1, ax=ax, label='Ice Age (years)')
        fig.suptitle(f"{date_s_str}", fontsize=16)
        fig.savefig(f'./fig/ice_age/ice_age_{date_s_str}_collocate.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    season.ice_age = ice_age_out


# ---------------------------------------------------------------------------
# Helper: spectral albedo plot
# ---------------------------------------------------------------------------

def _plot_spectral_alb(dates, wvls, albs, alb_stds, labels, colors,
                       title, fname, fig_dir, exclude=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, d in enumerate(dates):
        if exclude and d in exclude:
            continue
        ax.plot(wvls[i], albs[i], label=labels[i], color=colors[i])
        ax.fill_between(wvls[i], albs[i] - alb_stds[i], albs[i] + alb_stds[i],
                        color=colors[i], alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(350, 2000)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/{fname}', bbox_inches='tight', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grain size retrieval (ART theory)
# ---------------------------------------------------------------------------

def compute_grain_size(season):
    """
    Retrieve optical-equivalent snow grain size (r_opt) via ART theory at 1700 nm.

    White-sky (cloudy):
        r_opt = λ / (4π χ(T)) * (ln(α) / A)²

    Black-sky (clear, spiral — includes SZA correction via escape function K0):
        K0(θ) = 3/7 * (1 + 2*cos(θ))
        r_opt = λ / (4π χ(T)) * (ln(α) / (A * K0))²

    χ(T) is interpolated per-point from the Warren & Brandt 2008 lookup table
    using KT19 surface temperature.  NaN KT19 values fall back to 266 K.

    Result stored in season.grain_size in meters.
    """
    lam = _GS_WVL_ * 1e-9   # 1700 nm → meters
    A   = _GS_A_

    # Per-point temperature-dependent χ from KT19 (deg C → K, clipped to table range)
    T_C = season.kt19_sfc_T
    T_K = np.where(np.isfinite(T_C), T_C + 273.15, 266.0)
    T_K = np.clip(T_K, _GS_CHI_T_NODES_[0], _GS_CHI_T_NODES_[-1])
    chi = np.interp(T_K, _GS_CHI_T_NODES_, _GS_CHI_K_NODES_)   # shape (n_pts,)

    # Vectorised linear interpolation of alb_iter2 to 1700 nm
    wvl = season.wvl
    idx_hi = np.searchsorted(wvl, _GS_WVL_)          # first index where wvl >= 1700
    idx_hi = np.clip(idx_hi, 1, len(wvl) - 1)
    idx_lo = idx_hi - 1
    t = (_GS_WVL_ - wvl[idx_lo]) / (wvl[idx_hi] - wvl[idx_lo])
    alb_1700 = (season.alb_iter2[:, idx_lo] * (1.0 - t)
                + season.alb_iter2[:, idx_hi] * t)

    alb_1700 = np.clip(alb_1700, 1e-6, 1.0)
    ln_alb   = np.log(alb_1700)               # ≤ 0

    prefactor = lam / (4.0 * np.pi * chi)    # array (n_pts,)
    grain_size = np.full(len(season.lon), np.nan)

    # White-sky: cloudy
    ws = season.conditions == 'cloudy'
    if ws.any():
        grain_size[ws] = prefactor[ws] * (ln_alb[ws] / A) ** 2

    # Black-sky: clear + spiral (apply K0 escape-function correction)
    bs = (season.conditions == 'clear') | (season.conditions == 'spiral')
    if bs.any():
        K0 = (3.0 / 7.0) * (1.0 + 2.0 * np.cos(np.deg2rad(season.sza[bs])))
        K0 = np.where(K0 > 0, K0, np.nan)    # guard: SZA >= 90° → NaN
        grain_size[bs] = prefactor[bs] * (ln_alb[bs] / (A * K0)) ** 2

    season.grain_size = grain_size


def compute_grain_size_1240(season):
    """
    Retrieve optical-equivalent snow grain size (r_opt) via single-band AART at 1240 nm.

    1240 nm sits in the shoulder of the 1.25 μm ice absorption band, providing
    enough absorption contrast for reliable single-band AART retrieval while
    avoiding the over-sensitivity issues of 1020 nm (too transparent) or the
    strong T-dependence of 1700 nm.

    White-sky (cloudy):
        r_opt = λ / (4π χ(T)) * (ln(α) / A)²

    Black-sky (clear, spiral):
        r_opt = λ / (4π χ(T)) * (ln(α) / (A * K0))²

    χ(T) is interpolated per-point from the Warren & Brandt 2008 lookup table
    using KT19 surface temperature.  NaN KT19 values fall back to 266 K.

    Result stored in season.grain_size_1240 in meters.
    """
    lam = _GS_1240_WVL_ * 1e-9   # 1240 nm → meters
    A   = _GS_A_

    # Per-point temperature-dependent χ from KT19 (deg C → K, clipped to table range)
    T_C = season.kt19_sfc_T
    T_K = np.where(np.isfinite(T_C), T_C + 273.15, 266.0)
    T_K = np.clip(T_K, _GS_1240_CHI_T_NODES_[0], _GS_1240_CHI_T_NODES_[-1])
    chi = np.interp(T_K, _GS_1240_CHI_T_NODES_, _GS_1240_CHI_K_NODES_)   # shape (n_pts,)

    wvl = season.wvl
    idx_hi = np.searchsorted(wvl, _GS_1240_WVL_)
    idx_hi = np.clip(idx_hi, 1, len(wvl) - 1)
    idx_lo = idx_hi - 1
    t = (_GS_1240_WVL_ - wvl[idx_lo]) / (wvl[idx_hi] - wvl[idx_lo])
    alb_1240 = (season.alb_iter2[:, idx_lo] * (1.0 - t)
                + season.alb_iter2[:, idx_hi] * t)

    alb_1240 = np.clip(alb_1240, 1e-6, 1.0)
    ln_alb   = np.log(alb_1240)               # ≤ 0

    prefactor = lam / (4.0 * np.pi * chi)    # array (n_pts,)
    grain_size_1240 = np.full(len(season.lon), np.nan)

    ws = season.conditions == 'cloudy'
    if ws.any():
        grain_size_1240[ws] = prefactor[ws] * (ln_alb[ws] / A) ** 2

    bs = (season.conditions == 'clear') | (season.conditions == 'spiral')
    if bs.any():
        K0 = (3.0 / 7.0) * (1.0 + 2.0 * np.cos(np.deg2rad(season.sza[bs])))
        K0 = np.where(K0 > 0, K0, np.nan)
        grain_size_1240[bs] = prefactor[bs] * (ln_alb[bs] / (A * K0)) ** 2

    season.grain_size_1240 = grain_size_1240


def _grain_size_from_ratio(season, wvl1_nm, chi1, wvl2_nm, chi2):
    """
    Core AART ratio-method retrieval for an arbitrary wavelength pair.

        R = α(λ1) / α(λ2),   λ1 more absorptive than λ2  →  R < 1

        r_opt = 1/(4π) * [ ln(R) / (A * K0 * (√(χ1/λ1) − √(χ2/λ2))) ]²

    White-sky (cloudy): K0 = 1
    Black-sky (clear/spiral): K0 = 3/7 * (1 + 2*cos(θ0))

    Returns grain size array in metres.
    """
    A    = _GS_A_
    lam1 = wvl1_nm * 1e-9
    lam2 = wvl2_nm * 1e-9
    D    = np.sqrt(chi1 / lam1) - np.sqrt(chi2 / lam2)   # m^-0.5, > 0

    wvl = season.wvl

    def _interp_alb(target_nm):
        idx_hi = np.searchsorted(wvl, target_nm)
        idx_hi = np.clip(idx_hi, 1, len(wvl) - 1)
        idx_lo = idx_hi - 1
        t = (target_nm - wvl[idx_lo]) / (wvl[idx_hi] - wvl[idx_lo])
        return (season.alb_iter2[:, idx_lo] * (1.0 - t)
                + season.alb_iter2[:, idx_hi] * t)

    alb1 = np.clip(_interp_alb(wvl1_nm), 1e-6, 1.0)
    alb2 = np.clip(_interp_alb(wvl2_nm), 1e-6, 1.0)
    ln_R = np.log(alb1 / alb2)   # ≤ 0

    prefactor = 1.0 / (4.0 * np.pi)
    gs = np.full(len(season.lon), np.nan)

    ws = season.conditions == 'cloudy'
    if ws.any():
        gs[ws] = prefactor * (ln_R[ws] / (A * D)) ** 2

    bs = (season.conditions == 'clear') | (season.conditions == 'spiral')
    if bs.any():
        K0 = (3.0 / 7.0) * (1.0 + 2.0 * np.cos(np.deg2rad(season.sza[bs])))
        K0 = np.where(K0 > 0, K0, np.nan)
        gs[bs] = prefactor * (ln_R[bs] / (A * K0 * D)) ** 2

    return gs


def compute_grain_size_ratio(season):
    """Compute ratio-method grain size for all three wavelength pairs."""
    season.grain_size_ratio = _grain_size_from_ratio(
        season,
        _GS_RATIO_1280_WVL1_, _GS_RATIO_1280_CHI1_,
        _GS_RATIO_1280_WVL2_, _GS_RATIO_1280_CHI2_,
    )
    season.grain_size_ratio_1650_1020 = _grain_size_from_ratio(
        season,
        _GS_RATIO_1650_WVL1_, _GS_RATIO_1650_CHI1_,
        _GS_RATIO_1650_WVL2_, _GS_RATIO_1650_CHI2_,
    )
    season.grain_size_ratio_1020_865 = _grain_size_from_ratio(
        season,
        _GS_RATIO_1020_WVL1_, _GS_RATIO_1020_CHI1_,
        _GS_RATIO_1020_WVL2_, _GS_RATIO_1020_CHI2_,
    )


# ---------------------------------------------------------------------------
# Main combined function
# ---------------------------------------------------------------------------

def combined_atm_corr():
    log = logging.getLogger("atm corr combined")

    output_dir = f'{_fdir_general_}/sfc_alb_combined_smooth_450nm'
    sfc_alb_files = sorted(glob.glob(f'{output_dir}/sfc_alb_update_*.pkl'))
    print(f"Found {len(sfc_alb_files)} surface albedo files for combination.")
    
    combined_output_file = f'{output_dir}/sfc_alb_combined_spring_summer.pkl'
    combined_output_alb_file = f'{output_dir}/alb_atm_corr_combined_spring_summer.h5'

    if 0:#os.path.exists(combined_output_file) and os.path.exists(combined_output_alb_file):
        print(f"Both output files found, loading from {combined_output_file} and skipping processing ...")
        with open(combined_output_file, 'rb') as f:
            d = pickle.load(f)
        spring = types.SimpleNamespace(
            name='spring',
            wvl=d['wvl_spring'],
            lon=d['lon_all_spring'],
            lat=d['lat_all_spring'],
            alt=d['alt_all_spring'],
            time=d['time_spring_all'],
            dates=d['dates_spring_all'],
            conditions=d['leg_contidions_all_spring'],
            case_tags=d['case_tags_spring_all'],
            alb_iter2=d['alb_iter2_all_spring'],
            broadband_alb_iter2=d['broadband_alb_iter2_spring_all'],
            bb_alb_iter2_690_1190=d['broadband_alb_690_1190_spring_all'],
            ice_frac=d['ice_frac_all_spring'],
            nad_hdrf=d['nad_hdrf_spring_all'],
            nad_rad=d['nad_rad_spring_all'],
            myi_ratio=d['myi_ratio_spring_all'],
            ice_ratio=d['ice_ratio_spring_all'],
            ice_age=d['ice_age_spring_all'],
            era5_alb=d['era5_alb_spring_all'],
            lon_avg=d['lon_avg_spring'],
            amsr2_ice_conc=d['amsr2_ice_conc_spring_all'],
        )
        summer = types.SimpleNamespace(
            name='summer',
            wvl=d['wvl_summer'],
            lon=d['lon_all_summer'],
            lat=d['lat_all_summer'],
            alt=d['alt_all_summer'],
            time=d['time_summer_all'],
            dates=d['dates_summer_all'],
            conditions=d['leg_contidions_all_summer'],
            case_tags=d['case_tags_summer_all'],
            alb_iter2=d['alb_iter2_all_summer'],
            broadband_alb_iter2=d['broadband_alb_iter2_summer_all'],
            bb_alb_iter2_690_1190=d['broadband_alb_690_1190_summer_all'],
            ice_frac=d['ice_frac_all_summer'],
            nad_hdrf=d['nad_hdrf_summer_all'],
            nad_rad=d['nad_rad_summer_all'],
            myi_ratio=d['myi_ratio_summer_all'],
            ice_ratio=d['ice_ratio_summer_all'],
            ice_age=d['ice_age_summer_all'],
            era5_alb=d['era5_alb_summer_all'],
            lon_avg=d['lon_avg_summer'],
            amsr2_ice_conc=d['amsr2_ice_conc_summer_all'],
        )
    else:
        # 1. Split files by season and build SeasonData objects
        spring_files = [f for f in sfc_alb_files if extract_date_casetag(f)[0] < '20240630']
        summer_files = [f for f in sfc_alb_files if extract_date_casetag(f)[0] >= '20240630']

        spring = load_season(spring_files, 'spring')
        summer = load_season(summer_files, 'summer')

        compute_grain_size(spring)
        compute_grain_size(summer)
        compute_grain_size_1240(spring)
        compute_grain_size_1240(summer)
        compute_grain_size_ratio(spring)
        compute_grain_size_ratio(summer)

        print(f"Combined total of {len(spring.lon_avg)} spring flight legs and {len(spring.lon)} total points.")
        print("alb_iter1 spring shape:", spring.alb_iter1.shape)
        print(f"Combined total of {len(summer.lon_avg)} summer flight legs and {len(summer.lon)} total points.")
        print("alb_iter1 summer shape:", summer.alb_iter1.shape)

        # Combined (cross-season) arrays for map extents / summary plots
        lon_all = np.concatenate((spring.lon, summer.lon))
        lat_all = np.concatenate((spring.lat, summer.lat))

        os.makedirs('./fig/ice_age', exist_ok=True)

        # 2. Load ice_frac data and collocate for each season
        file = f'{_fdir_general_}/cam_icefrac_rad/ice_frac_all.pkl'
        with open(file, 'rb') as f:
            ice_frac_data = pickle.load(f)

        ice_frac_time_offset = {
            '20240528': 0,
            '20240530': 0,
            '20240531': 0,
            '20240603': -0.50/3600,
            '20240605': -0.80/3600,
            '20240606': -0.75/3600,
            '20240607':  0.35/3600,
            '20240610': 0,
            '20240611': -0.15/3600,
            '20240613':  0.55/3600,
            '20240725':  0.30/3600,
            '20240729': -0.95/3600,
            '20240730':  1.0/3600,
            '20240801': -0.50/3600,
            '20240802': -0.15/3600,
            '20240807': -0.70/3600,
            '20240808': -0.25/3600,
            '20240809': -0.05/3600,
            '20240815': -0.85/3600,
        }

        collocate_ice_frac(spring, ice_frac_data, ice_frac_time_offset)
        collocate_ice_frac(summer, ice_frac_data, ice_frac_time_offset)

        amsr2_data_dir = f'{_fdir_general_}/ice_conc'
        collocate_amsr2_ice(spring, amsr2_data_dir)
        collocate_amsr2_ice(summer, amsr2_data_dir)

        # 3. Load NSIDC and ERA5, then collocate ECICE + NSIDC ice age for each season
        with Dataset(f'{_fdir_general_}/ice_age/iceage_nh_12.5km_20240101_20250923_ql.nc', 'r') as nc:
            nsidc_lon     = nc.variables['longitude'][:]
            nsidc_lat     = nc.variables['latitude'][:]
            time_nc       = nc.variables['time'][:]   # days since 1970-01-01
            nsidc_ice_age = nc.variables['age_of_sea_ice'][:]
        time_nc_dates = np.array([
            datetime.datetime(1970, 1, 1) + datetime.timedelta(days=t) for t in time_nc
        ])
        nsidc_ice_age = np.array(nsidc_ice_age, dtype=np.float32)

        with Dataset(f'{_fdir_general_}/era5/forecast_albedo_0_daily-mean.nc', 'r') as nc:
            era5_lon  = nc.variables['longitude'][:]
            era5_lat  = nc.variables['latitude'][:]
            era5_time = nc.variables['valid_time'][:]
            era5_alb  = nc.variables['fal'][:]
        era5_time_dates     = np.array([
            datetime.datetime(2024, 5, 1) + datetime.timedelta(days=int(t)) for t in era5_time
        ])
        era5_time_dates_str = np.array([t.strftime('%Y%m%d') for t in era5_time_dates])
        era5_alb            = np.array(era5_alb, dtype=np.float32)
        era5_lat_mesh, era5_lon_mesh = np.meshgrid(era5_lat, era5_lon, indexing='ij')

        ecice_data_dir = f'{_fdir_general_}/ice_age'

        collocate_ecice(spring, ecice_data_dir,
                        era5_lon_mesh, era5_lat_mesh, era5_alb, era5_time_dates_str,
                        lon_all, lat_all)
        collocate_ecice(summer, ecice_data_dir,
                        era5_lon_mesh, era5_lat_mesh, era5_alb, era5_time_dates_str,
                        lon_all, lat_all)

        collocate_nsidc_ice_age(spring, nsidc_lon, nsidc_lat, nsidc_ice_age, time_nc_dates,
                                 lon_all, lat_all)
        collocate_nsidc_ice_age(summer, nsidc_lon, nsidc_lat, nsidc_ice_age, time_nc_dates,
                                 lon_all, lat_all)

        # 4. Save outputs (pkl + h5)
        output_all_dict = {
            'native_wvl_spring': spring.wvl,
            'native_wvl_summer': summer.wvl,
            'wvl_spring': spring.wvl,
            'wvl_summer': summer.wvl,
            'time_spring_all': spring.time,
            'lon_all_spring': spring.lon,
            'lat_all_spring': spring.lat,
            'alt_all_spring': spring.alt,
            'sza_spring_all': spring.sza,
            'kt19_sfc_T_spring_all': spring.kt19_sfc_T,
            'dates_spring_all': spring.dates,
            'leg_conditions_all_spring': spring.conditions,
            'leg_contidions_all_spring': spring.conditions,
            'case_tags_spring': spring.case_tags_leg,
            'case_tags_spring_all': spring.case_tags,
            'fdn_all_spring': spring.fdn,
            'fup_all_spring': spring.fup,
            'toa_expand_all_spring': spring.toa_expand,
            'icing_all_spring': spring.icing,
            'icing_pre_all_spring': spring.icing_pre,
            'alb_iter1_all_spring': spring.alb_iter1,
            'alb_iter2_all_spring': spring.alb_iter2,
            'alb_final_all_spring': spring.alb_iter2,
            'alb_iter1_all_1s_spring': spring.alb_iter1,
            'alb_iter2_all_1s_spring': spring.alb_iter2,
            'alb_final_all_1s_spring': spring.alb_iter2,
            'broadband_alb_iter1_all_spring': spring._bb_alb_iter1,
            'broadband_alb_iter2_all_spring': spring._bb_alb_iter2_file,
            'broadband_alb_iter1_all_1s_spring': spring._bb_alb_iter1,
            'broadband_alb_iter2_all_1s_spring': spring._bb_alb_iter2_file,
            'broadband_alb_final_all_1s_spring': spring._bb_alb_iter2_file,
            'broadband_alb_final_file_spring': spring._bb_alb_iter2_file,
            'broadband_alb_iter1_all_filter_spring': spring._bb_alb_iter1_filter,
            'broadband_alb_iter2_all_filter_spring': spring._bb_alb_iter2_filter,
            'broadband_alb_iter1_all_filter_1s_spring': spring._bb_alb_iter1_filter,
            'broadband_alb_iter2_all_filter_1s_spring': spring._bb_alb_iter2_filter,
            'broadband_alb_final_all_filter_1s_spring': spring._bb_alb_iter2_filter,
            'lon_avg_spring': spring.lon_avg,
            'lat_avg_spring': spring.lat_avg,
            'alt_avg_spring': spring.alt_avg,
            'broadband_alb_iter2_spring_all': spring.broadband_alb_iter2,
            'broadband_alb_final_spring_all': spring.broadband_alb_iter2,
            'broadband_alb_690_1190_spring_all': spring.bb_alb_iter2_690_1190,
            'ice_frac_all_spring': spring.ice_frac,
            'nad_hdrf_spring_all': spring.nad_hdrf,
            'nad_rad_spring_all': spring.nad_rad,
            'myi_ratio_spring_all': spring.myi_ratio,
            'fyi_ratio_spring_all': spring.fyi_ratio,
            'yi_ratio_spring_all': spring.yi_ratio,
            'ice_ratio_spring_all': spring.ice_ratio,
            'ow_ratio_spring_all': spring.ow_ratio,
            'ice_age_spring_all': spring.ice_age,
            'era5_alb_spring_all': spring.era5_alb,
            'grain_size_spring_all': spring.grain_size,
            'grain_size_1240_spring_all': spring.grain_size_1240,
            'grain_size_ratio_spring_all': spring.grain_size_ratio,
            'grain_size_ratio_1650_1020_spring_all': spring.grain_size_ratio_1650_1020,
            'grain_size_ratio_1020_865_spring_all': spring.grain_size_ratio_1020_865,
            'brt19h_spring_all': spring.brt19h,
            'brt37h_spring_all': spring.brt37h,
            'brt37v_spring_all': spring.brt37v,
            'amsr2_ice_conc_spring_all': spring.amsr2_ice_conc,

            'time_summer_all': summer.time,
            'lon_all_summer': summer.lon,
            'lat_all_summer': summer.lat,
            'alt_all_summer': summer.alt,
            'sza_summer_all': summer.sza,
            'kt19_sfc_T_summer_all': summer.kt19_sfc_T,
            'dates_summer_all': summer.dates,
            'leg_conditions_all_summer': summer.conditions,
            'leg_contidions_all_summer': summer.conditions,
            'case_tags_summer': summer.case_tags_leg,
            'case_tags_summer_all': summer.case_tags,
            'fdn_all_summer': summer.fdn,
            'fup_all_summer': summer.fup,
            'toa_expand_all_summer': summer.toa_expand,
            'icing_all_summer': summer.icing,
            'icing_pre_all_summer': summer.icing_pre,
            'alb_iter1_all_summer': summer.alb_iter1,
            'alb_iter2_all_summer': summer.alb_iter2,
            'alb_final_all_summer': summer.alb_iter2,
            'alb_iter1_all_1s_summer': summer.alb_iter1,
            'alb_iter2_all_1s_summer': summer.alb_iter2,
            'alb_final_all_1s_summer': summer.alb_iter2,
            'broadband_alb_iter1_all_summer': summer._bb_alb_iter1,
            'broadband_alb_iter2_all_summer': summer._bb_alb_iter2_file,
            'broadband_alb_iter1_all_1s_summer': summer._bb_alb_iter1,
            'broadband_alb_iter2_all_1s_summer': summer._bb_alb_iter2_file,
            'broadband_alb_final_all_1s_summer': summer._bb_alb_iter2_file,
            'broadband_alb_final_file_summer': summer._bb_alb_iter2_file,
            'broadband_alb_iter1_all_filter_summer': summer._bb_alb_iter1_filter,
            'broadband_alb_iter2_all_filter_summer': summer._bb_alb_iter2_filter,
            'broadband_alb_iter1_all_filter_1s_summer': summer._bb_alb_iter1_filter,
            'broadband_alb_iter2_all_filter_1s_summer': summer._bb_alb_iter2_filter,
            'broadband_alb_final_all_filter_1s_summer': summer._bb_alb_iter2_filter,
            'lon_avg_summer': summer.lon_avg,
            'lat_avg_summer': summer.lat_avg,
            'alt_avg_summer': summer.alt_avg,
            'broadband_alb_iter2_summer_all': summer.broadband_alb_iter2,
            'broadband_alb_final_summer_all': summer.broadband_alb_iter2,
            'broadband_alb_690_1190_summer_all': summer.bb_alb_iter2_690_1190,
            'ice_frac_all_summer': summer.ice_frac,
            'nad_hdrf_summer_all': summer.nad_hdrf,
            'nad_rad_summer_all': summer.nad_rad,
            'myi_ratio_summer_all': summer.myi_ratio,
            'fyi_ratio_summer_all': summer.fyi_ratio,
            'yi_ratio_summer_all': summer.yi_ratio,
            'ice_ratio_summer_all': summer.ice_ratio,
            'ow_ratio_summer_all': summer.ow_ratio,
            'ice_age_summer_all': summer.ice_age,
            'era5_alb_summer_all': summer.era5_alb,
            'grain_size_summer_all': summer.grain_size,
            'grain_size_1240_summer_all': summer.grain_size_1240,
            'grain_size_ratio_summer_all': summer.grain_size_ratio,
            'grain_size_ratio_1650_1020_summer_all': summer.grain_size_ratio_1650_1020,
            'grain_size_ratio_1020_865_summer_all': summer.grain_size_ratio_1020_865,
            'brt19h_summer_all': summer.brt19h,
            'brt37h_summer_all': summer.brt37h,
            'brt37v_summer_all': summer.brt37v,
            'amsr2_ice_conc_summer_all': summer.amsr2_ice_conc,
        }

    
        with open(combined_output_file, 'wb') as f:
            pickle.dump(output_all_dict, f)
        print(f"Combined surface albedo data saved to {combined_output_file}")

        output_alb_all_dict = {
            'time_spring_all': spring.time,
            'time_springl': spring.time,
            'lon_spring': spring.lon,
            'lat_spring': spring.lat,
            'alt_spring': spring.alt,
            'dates_spring': spring.dates,
            'native_wvl_spring': spring.wvl,
            'wvl_spring': spring.wvl,
            'alb_final_spring': spring.alb_iter2,
            'alb_atm_corr_spring': spring.alb_iter2,
            'broadband_alb_atm_corr_spring': spring.broadband_alb_iter2,
            'broadband_alb_alb_tm_corr_spring': spring.broadband_alb_iter2,

            'time_summer_all': summer.time,
            'lon_summer': summer.lon,
            'lat_summer': summer.lat,
            'alt_summer': summer.alt,
            'dates_summer': summer.dates,
            'native_wvl_summer': summer.wvl,
            'wvl_summer': summer.wvl,
            'alb_final_summer': summer.alb_iter2,
            'alb_atm_corr_summer': summer.alb_iter2,
            'broadband_alb_atm_corr_summer': summer.broadband_alb_iter2,
            'broadband_alb_alb_tm_corr_summer': summer.broadband_alb_iter2,
        }
    
        with h5py.File(combined_output_alb_file, 'w') as hf:
            for key, value in output_alb_all_dict.items():
                hf.create_dataset(key, data=value)
        print(f"Combined surface albedo HDF5 data saved to {combined_output_alb_file}")

    # --- ERA5 vs broadband scatter (all seasons combined) ---

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(spring.era5_alb, spring.broadband_alb_iter2, c='blue', label='Spring', alpha=0.5)
    ax.scatter(summer.era5_alb, summer.broadband_alb_iter2, c='red', label='Summer', alpha=0.5)
    ax.plot([0, 1], [0, 1], 'k--', label='1:1 Line')
    ax.set_xlabel('ERA5 Forecast Albedo')
    ax.set_ylabel('Retrieved Broadband Surface Albedo')
    fig.savefig(f'./fig/ice_age/bb_alb_all.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 5. Generate summary plots (date-averaged spectral albedo, AMSR2 map, scatter)
    fig_dir = f'fig/sfc_alb_corr_lonlat'
    os.makedirs(fig_dir, exist_ok=True)

    date_all_list       = []
    date_alb            = []
    date_alb_std        = []
    date_broadband_alb  = []
    date_broadband_alb_std = []
    date_broadband_alb_p10 = []
    date_broadband_alb_p90 = []
    date_broadband_alb_690_1190     = []
    date_broadband_alb_690_1190_p10 = []
    date_broadband_alb_690_1190_p90 = []
    date_ice_frac       = []
    date_ice_frac_std   = []
    date_ice_frac_p10   = []
    date_ice_frac_p90   = []
    date_myi_ratio      = []
    date_myi_ratio_std  = []
    date_ice_age_avg    = []
    date_ice_age_std    = []
    date_ice_age_myi_ratio = []
    date_clear_all      = []
    date_alb_clear      = []
    date_alb_clear_std  = []
    date_ice_frac_clear = []
    date_ice_frac_clear_std = []
    date_myi_ratio_clear = []
    date_cloudy_all     = []
    date_alb_cloudy     = []
    date_alb_cloudy_std = []
    date_ice_frac_cloudy = []
    date_ice_frac_cloudy_std = []
    date_myi_ratio_cloudy = []
    date_alb_wvl        = []
    date_alb_clear_wvl  = []
    date_alb_cloudy_wvl = []
    date_amsr2_ice_conc     = []
    date_amsr2_ice_conc_p10 = []
    date_amsr2_ice_conc_p90 = []

    for season_obj in [spring, summer]:
        for date in sorted(set(season_obj.dates)):
            date_mask = season_obj.dates == date
            alt_mask  = season_obj.alt <= 1.6
            date_mask = date_mask & alt_mask

            date_alb_avg = np.nanmean(season_obj.alb_iter2[date_mask], axis=0)
            if np.isnan(date_alb_avg).all():
                print(f"All NaN for date {date}, skipping.")
                continue
            date_all_list.append(str(date)[4:])
            date_alb_std_  = np.nanstd(season_obj.alb_iter2[date_mask], axis=0)
            date_alb.append(date_alb_avg)
            date_alb_std.append(date_alb_std_)
            date_alb_wvl.append(season_obj.wvl)

            date_broadband_alb.append(np.nanmean(season_obj.broadband_alb_iter2[date_mask]))
            date_broadband_alb_std.append(np.nanstd(season_obj.broadband_alb_iter2[date_mask]))
            date_broadband_alb_p10.append(np.nanpercentile(season_obj.broadband_alb_iter2[date_mask], 10))
            date_broadband_alb_p90.append(np.nanpercentile(season_obj.broadband_alb_iter2[date_mask], 90))
            date_broadband_alb_690_1190.append(np.nanmean(season_obj.bb_alb_iter2_690_1190[date_mask]))
            date_broadband_alb_690_1190_p10.append(np.nanpercentile(season_obj.bb_alb_iter2_690_1190[date_mask], 10))
            date_broadband_alb_690_1190_p90.append(np.nanpercentile(season_obj.bb_alb_iter2_690_1190[date_mask], 90))

            date_ice_frac.append(np.nanmean(season_obj.ice_frac[date_mask]))
            date_ice_frac_std.append(np.nanstd(season_obj.ice_frac[date_mask]))
            date_ice_frac_p10.append(np.nanpercentile(season_obj.ice_frac[date_mask], 10))
            date_ice_frac_p90.append(np.nanpercentile(season_obj.ice_frac[date_mask], 90))

            date_myi_ratio.append(np.nanmean(season_obj.myi_ratio[date_mask] / season_obj.ice_ratio[date_mask]))
            date_myi_ratio_std.append(np.nanstd(season_obj.myi_ratio[date_mask] / season_obj.ice_ratio[date_mask]))

            date_ice_age_avg.append(np.nanmean(season_obj.ice_age[date_mask]))
            date_ice_age_std.append(np.nanstd(season_obj.ice_age[date_mask]))
            date_ice_age_myi_ratio.append(
                (season_obj.ice_age[date_mask] >= 2).sum() / len(season_obj.ice_age[date_mask])
            )

            date_amsr2_ice_conc.append(np.nanmean(season_obj.amsr2_ice_conc[date_mask]))
            date_amsr2_ice_conc_p10.append(np.nanpercentile(season_obj.amsr2_ice_conc[date_mask], 10))
            date_amsr2_ice_conc_p90.append(np.nanpercentile(season_obj.amsr2_ice_conc[date_mask], 90))

            clear_mask  = date_mask & (season_obj.conditions == 'clear')
            if np.any(clear_mask):
                date_clear_all.append(str(date)[4:])
                date_alb_clear.append(np.nanmean(season_obj.alb_iter2[clear_mask], axis=0))
                date_alb_clear_std.append(np.nanstd(season_obj.alb_iter2[clear_mask], axis=0))
                date_alb_clear_wvl.append(season_obj.wvl)
                date_ice_frac_clear.append(np.nanmean(season_obj.ice_frac[clear_mask]))
                date_ice_frac_clear_std.append(np.nanstd(season_obj.ice_frac[clear_mask]))
                date_myi_ratio_clear.append(
                    np.nanmean(season_obj.myi_ratio[clear_mask] / season_obj.ice_ratio[clear_mask])
                )

            cloudy_mask = date_mask & (season_obj.conditions == 'cloudy')
            if np.any(cloudy_mask):
                date_cloudy_all.append(str(date)[4:])
                date_alb_cloudy.append(np.nanmean(season_obj.alb_iter2[cloudy_mask], axis=0))
                date_alb_cloudy_std.append(np.nanstd(season_obj.alb_iter2[cloudy_mask], axis=0))
                date_alb_cloudy_wvl.append(season_obj.wvl)
                date_ice_frac_cloudy.append(np.nanmean(season_obj.ice_frac[cloudy_mask]))
                date_ice_frac_cloudy_std.append(np.nanstd(season_obj.ice_frac[cloudy_mask]))
                date_myi_ratio_cloudy.append(
                    np.nanmean(season_obj.myi_ratio[cloudy_mask] / season_obj.ice_ratio[cloudy_mask])
                )

    print("date_all length:", len(date_all_list))
    print("date_alb length:", len(date_alb))
    print("date_alb_wvl length:", len(date_alb_wvl))

    if 0:
        plt.close('all')
        n_dates        = len(date_all_list)
        n_dates_clear  = len(date_clear_all)
        n_dates_cloudy = len(date_cloudy_all)
        if n_dates == 0:
            color_series        = []
            color_series_clear  = []
            color_series_cloudy = []
        else:
            cmap_name = 'jet'
            cmap = mpl.colormaps.get_cmap(cmap_name)
            if n_dates == 1:
                color_series = [cmap(0.5)]
            else:
                color_series        = [cmap(i / (n_dates - 1))        for i in range(n_dates)]
                color_series_clear  = [cmap(i / (n_dates_clear - 1))  for i in range(len(date_clear_all))]
                color_series_cloudy = [cmap(i / (n_dates_cloudy - 1)) for i in range(len(date_cloudy_all))]

        norm = mpl.colors.Normalize(vmin=0, vmax=max(1, n_dates - 1))
        sm   = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)  # type: ignore[attr-defined]
        sm.set_array([])

        EX = ['0808', '0809']
        ice_frac_labels  = [f'{d}, ice fraction={f:.3f}+/-{s:.3f}' for d, f, s in zip(date_all_list,   date_ice_frac,       date_ice_frac_std)]
        myi_labels       = [f'{d}, MYI={m*100:.1f}%'               for d, m    in zip(date_all_list,   date_myi_ratio)]
        age_labels       = [f'{d}, Ice Age={a:.1f} +/- {s:.1f} y'  for d, a, s in zip(date_all_list,   date_ice_age_avg,    date_ice_age_std)]
        age_myi_labels   = [f'{d}, MYI={r*100:.1f}%'               for d, r    in zip(date_all_list,   date_ice_age_myi_ratio)]
        clear_labels     = [f'{d}, ice fraction={f:.3f}+/-{s:.3f}' for d, f, s in zip(date_clear_all,  date_ice_frac_clear, date_ice_frac_clear_std)]
        cloudy_labels    = [f'{d}, ice fraction={f:.3f}+/-{s:.3f}' for d, f, s in zip(date_cloudy_all, date_ice_frac_cloudy,date_ice_frac_cloudy_std)]

        _D  = (date_all_list,   date_alb_wvl,        date_alb,        date_alb_std)
        _CL = (date_clear_all,  date_alb_clear_wvl,  date_alb_clear,  date_alb_clear_std)
        _CU = (date_cloudy_all, date_alb_cloudy_wvl, date_alb_cloudy, date_alb_cloudy_std)

        plot_configs = [
            (*_D,  ice_frac_labels, color_series,        'Surface Albedo (atm corr + fit) for All Flights',    'arcsix_albedo_all_flights.png',                        None),
            (*_D,  ice_frac_labels, color_series,        'Surface Albedo (atm corr + fit)\nexclude 0808, 0809','arcsix_albedo_all_flights_partial.png',                 EX),
            (*_CL, clear_labels,    color_series_clear,  'Clear Sky Surface Albedo (atm corr + fit)\nexclude 0808, 0809', 'arcsix_albedo_all_flights_clear_partial.png', EX),
            (*_CU, cloudy_labels,   color_series_cloudy, 'Below Cloud Surface Albedo (atm corr + fit)\nexclude 0808, 0809','arcsix_albedo_all_flights_cloudy_partial.png',EX),
            (*_D,  myi_labels,      color_series,        'Surface Albedo (atm corr + fit)',                    'arcsix_albedo_all_flights_myi.png',                     None),
            (*_D,  myi_labels,      color_series,        'Surface Albedo (atm corr + fit)\nexclude 0808, 0809','arcsix_albedo_all_flights_partial_myi.png',             EX),
            (*_D,  age_labels,      color_series,        'Surface Albedo (atm corr + fit)\nexclude 0808, 0809','arcsix_albedo_all_flights_partial_ice_age.png',         EX),
            (*_D,  age_myi_labels,  color_series,        'Surface Albedo (atm corr + fit)\nexclude 0808, 0809','arcsix_albedo_all_flights_partial_ice_age_myi_ratio.png',EX),
        ]
        for dates, wvls, albs, stds, labels, colors, title, fname, exc in plot_configs:
            _plot_spectral_alb(dates, wvls, albs, stds, labels, colors, title, fname, fig_dir, exclude=exc)

        # --- AMSR2 ice concentration map + broadband albedo scatter ---
        SF_into = {
            '0528': 'RF01', '0530': 'RF02', '0531': 'RF03', '0603': 'RF04',
            '0605': 'RF05', '0606': 'RF06', '0607': 'RF07', '0610': 'RF08',
            '0611': 'RF09', '0613': 'RF10', '0725': 'RF11', '0729': 'RF12',
            '0730': 'RF13', '0801': 'RF14', '0802': 'RF15', '0807': 'RF16',
            '0808': 'RF17', '0809': 'RF18', '0815': 'RF19',
        }

        patern = 'NSIDC-0803_SEAICE_AMSR2_N_2024*.nc'
        amsr2_data_dir  = '../data/ice_conc/'
        amsr2_files     = sorted(glob.glob(os.path.join(amsr2_data_dir, patern)))
        amsr2_dates     = [os.path.basename(f).split('_')[4][0:8] for f in amsr2_files]
        amsr2_dates_int = np.array([int(d[4:8]) for d in amsr2_dates])

        ice_conc_all = None
        for fn in amsr2_files:
            with Dataset(fn, 'r') as ds:
                x = ds.variables['x'][:]
                y = ds.variables['y'][:]

                v = ds.variables['ICECON']
                print("var dims:", v.dimensions, "shape:", v.shape)
                if 'time' in v.dimensions:
                    idx = [0 if d == 'time' else slice(None) for d in v.dimensions]
                    arr = v[tuple(idx)]
                else:
                    arr = v[:]

                arr  = np.array(arr, dtype=np.float32)
                print("original arr min/max:", np.nanmin(arr), np.nanmax(arr))

                fill = getattr(v, '_FillValue', None) or getattr(v, 'missing_value', None)
                print("fill value:", fill)
                if fill is not None:
                    arr = np.ma.masked_equal(arr, fill)

                arr[arr > 250] = np.nan

                print("arr min/max after fill mask:", np.nanmin(arr), np.nanmax(arr))

                ice_conc = arr.copy()
                print("Data subset shape:", ice_conc.shape)

                if 'crs' in ds.variables:
                    crs_var = ds.variables['crs']
                    wkt   = getattr(crs_var, 'crs_wkt', None) or getattr(crs_var, 'spatial_ref', None)
                    proj4 = getattr(crs_var, 'proj4', None) or getattr(crs_var, 'proj4text', None)
                    if wkt:
                        src_crs = CRS.from_wkt(wkt)
                    elif proj4:
                        src_crs = CRS.from_string(proj4)
                    else:
                        epsg    = getattr(crs_var, 'epsg', None)
                        src_crs = CRS.from_epsg(int(epsg)) if epsg else None
                else:
                    src_crs = None

                if src_crs is None:
                    src_crs = CRS.from_epsg(3411)

                print("ice_conc min/max:", np.nanmin(ice_conc), np.nanmax(ice_conc))
                ice_conc_scale = ice_conc * 100

                xx, yy = np.meshgrid(x, y)

                transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
                lon_flat, lat_flat = transformer.transform(xx.ravel(), yy.ravel())
                lon_amsr = np.array(lon_flat).reshape(xx.shape)
                lat_amsr = np.array(lat_flat).reshape(xx.shape)
                print("lon/lat shape:", lon_amsr.shape, lat_amsr.shape)
                print("ice_conc_scale shape:", ice_conc_scale.shape)
                print("ice_conc_scale min/max before flatten:", np.nanmin(ice_conc_scale), np.nanmax(ice_conc_scale))

                if ice_conc_all is None:
                    ice_conc_all = np.array(ice_conc_scale.copy(), dtype=np.float32)
                else:
                    ice_conc_all = np.dstack((ice_conc_all, ice_conc_scale.copy()))

                del ice_conc_scale, xx, yy, lon_flat, lat_flat, transformer

        spring_mask = amsr2_dates_int <= 630
        summer_mask = amsr2_dates_int >= 701
        ice_conc_spring_avg = np.nanmean(ice_conc_all[:, :, spring_mask], axis=2) if spring_mask.any() else np.full(ice_conc_all.shape[:2], np.nan, dtype=np.float32)
        ice_conc_summer_avg = np.nanmean(ice_conc_all[:, :, summer_mask], axis=2) if summer_mask.any() else np.full(ice_conc_all.shape[:2], np.nan, dtype=np.float32)
        ice_conc_spring_std = np.nanstd(ice_conc_all[:, :, spring_mask], axis=2) if spring_mask.any() else np.full(ice_conc_all.shape[:2], np.nan, dtype=np.float32)
        ice_conc_summer_std = np.nanstd(ice_conc_all[:, :, summer_mask], axis=2) if summer_mask.any() else np.full(ice_conc_all.shape[:2], np.nan, dtype=np.float32)

        plt.close('all')
        central_lon = float(np.nanmean(np.concatenate((spring.lon_avg, summer.lon_avg)))) \
            if len(spring.lon_avg) + len(summer.lon_avg) > 0 else 0.0
        proj = ccrs.NorthPolarStereo(central_longitude=central_lon)

        fig = plt.figure(figsize=(18, 9))
        gs  = gridspec.GridSpec(2, 2, width_ratios=[3, 2], wspace=0.1, hspace=0.3)
        ax11 = fig.add_subplot(gs[0, 0], projection=proj)
        ax12 = fig.add_subplot(gs[0, 1])
        ax21 = fig.add_subplot(gs[1, 0], projection=proj)
        ax22 = fig.add_subplot(gs[1, 1])

        for ax1 in [ax11, ax21]:
            ax1.coastlines(resolution='10m', linewidth=0.8)
            ax1.add_feature(cartopy.feature.LAND, facecolor="#8e8e8e", zorder=0)
            ax1.add_feature(cartopy.feature.OCEAN, facecolor="#8e8e8e", zorder=0)
            gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
            gl.top_labels   = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}

        im11 = ax11.pcolormesh(lon_amsr, lat_amsr, ice_conc_spring_avg,
                            cmap='Blues_r', vmin=0, vmax=100,
                            transform=ccrs.PlateCarree(), zorder=1)
        fig.colorbar(im11, ax=ax11, orientation='vertical', pad=0.01, shrink=0.9).set_label('Ice Concentration (%)', fontsize=10)
        im21 = ax21.pcolormesh(lon_amsr, lat_amsr, ice_conc_summer_avg,
                            cmap='Blues_r', vmin=0, vmax=100,
                            transform=ccrs.PlateCarree(), zorder=1)
        fig.colorbar(im21, ax=ax21, orientation='vertical', pad=0.01, shrink=0.9).set_label('Ice Concentration (%)', fontsize=10)

        mask_sp = ~np.isnan(spring.broadband_alb_iter2)
        sc11 = ax11.scatter(spring.lon[mask_sp], spring.lat[mask_sp],
                            s=5, c=spring.broadband_alb_iter2[mask_sp],
                            cmap='jet', transform=ccrs.PlateCarree(),
                            zorder=3, edgecolor=None, vmin=0.1, vmax=0.9)
        fig.colorbar(sc11, ax=ax11, orientation='vertical', pad=0.05, shrink=0.9).set_label('Broadband Albedo', fontsize=10)

        mask_su = ~np.isnan(summer.broadband_alb_iter2)
        sc21 = ax21.scatter(summer.lon[mask_su], summer.lat[mask_su],
                            s=5, c=summer.broadband_alb_iter2[mask_su],
                            cmap='jet', transform=ccrs.PlateCarree(),
                            zorder=3, edgecolor=None, vmin=0.1, vmax=0.9)
        fig.colorbar(sc21, ax=ax21, orientation='vertical', pad=0.05, shrink=0.9).set_label('Broadband Albedo', fontsize=10)

        lon_all_combined = np.concatenate((spring.lon, summer.lon))
        lat_all_combined = np.concatenate((spring.lat, summer.lat))
        lon_min_c, lon_max_c = np.nanmin(lon_all_combined), np.nanmax(lon_all_combined)
        lat_min_c, lat_max_c = np.nanmin(lat_all_combined), np.nanmax(lat_all_combined)
        for ax in [ax11, ax21]:
            pad_lon = max(0.2, (lon_max_c - lon_min_c) * 0.05)
            pad_lat = max(0.2, (lat_max_c - lat_min_c) * 0.05)
            ax.set_extent([lon_min_c - pad_lon, lon_max_c + pad_lon,
                        lat_min_c - pad_lat, lat_max_c + pad_lat],
                        crs=ccrs.PlateCarree())
            ax.tick_params('both', labelsize=10)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        ax11.set_title(f'RF01-10 (May 28 - June 13)', fontsize=12)
        ax21.set_title(f'RF11-19 (July 25 - Aug 15)', fontsize=12)

        n_dates = len(date_all_list)
        if n_dates == 0:
            color_series = []
        else:
            cmap_name = 'jet'
            cmap = mpl.colormaps.get_cmap(cmap_name)
            color_series = [cmap(0.5)] if n_dates == 1 else [cmap(i / (n_dates - 1)) for i in range(n_dates)]

        norm = mpl.colors.Normalize(vmin=0, vmax=max(1, n_dates - 1))
        sm   = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)  # type: ignore[attr-defined]
        sm.set_array([])

        eff_alb_ = gas_abs_masking(date_alb_wvl[0], np.ones_like(date_alb_wvl[0]), alt=5)
        for i in range(len(date_all_list)):
            ax12.plot(date_alb_wvl[i], date_alb[i],
                    label=f'{SF_into[date_all_list[i]]} ({date_all_list[i]})',
                    color=color_series[i])
            ax12.fill_between(date_alb_wvl[i], date_alb[i]-date_alb_std[i], date_alb[i]+date_alb_std[i],
                            color=color_series[i], alpha=0.1)
        ax12.fill_between(date_alb_wvl[0], -0.05, 1.05,
                        where=np.isnan(eff_alb_), color='gray', alpha=0.2)
        ax12.set_xlabel('Wavelength (nm)', fontsize=14)
        ax12.set_ylabel('Surface Albedo', fontsize=14)
        ax12.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.05, -0.1))
        ax12.tick_params(labelsize=12)
        ax12.set_ylim(-0.05, 1.05)
        ax12.set_xlim(350, 2000)

        ice_frac_arr   = np.array(date_ice_frac)
        bb_alb_arr     = np.array(date_broadband_alb)
        xerr_lo = np.maximum(0, ice_frac_arr   - np.array(date_ice_frac_p10))
        xerr_hi = np.maximum(0, np.array(date_ice_frac_p90)   - ice_frac_arr)
        yerr_lo = np.maximum(0, bb_alb_arr     - np.array(date_broadband_alb_p10))
        yerr_hi = np.maximum(0, np.array(date_broadband_alb_p90) - bb_alb_arr)
        ax22.errorbar(date_ice_frac, date_broadband_alb,
                    xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
                    fmt='o', color='black', ecolor='lightgray',
                    markersize=3, markerfacecolor='none',
                    elinewidth=1.5, capsize=1.5, zorder=2)
        ax22.scatter(date_ice_frac, date_broadband_alb, s=50, c=color_series, zorder=3)
        ax22.set_xlabel('Sea Ice Fraction', fontsize=14)
        ax22.set_ylabel('Broadband Albedo', fontsize=14)
        ax22.tick_params(labelsize=12)
        ax22.set_xlim(0.0, 1.10)
        ax22.set_ylim(0.05, 0.9)

        for ax, cap in zip([ax11, ax12, ax21, ax22], ['(a)', '(c)', '(b)', '(d)']):
            ax.text(0.0, 1.01, cap, transform=ax.transAxes, fontsize=14,
                    verticalalignment='bottom', horizontalalignment='left')

        fig.savefig(
            f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_spring_summer_combined.png',
            bbox_inches='tight', dpi=300
        )
        plt.close(fig)
    
    
    plt.close('all')

    n_dates = len(date_all_list)
    if n_dates == 0:
        color_series = []
    else:
        cmap_name = 'jet'
        cmap = mpl.colormaps.get_cmap(cmap_name)
        color_series = [cmap(0.5)] if n_dates == 1 else [cmap(i / (n_dates - 1)) for i in range(n_dates)]

    central_lon = float(np.nanmean(np.concatenate((spring.lon_avg, summer.lon_avg)))) \
        if len(spring.lon_avg) + len(summer.lon_avg) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)

    fig = plt.figure(figsize=(9, 5))
    gs  = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.3)
    ax22 = fig.add_subplot(gs[0, 0])

    # only plot 20240801
    mask = np.array(date_all_list) == '0801'
    date_ice_frac_mask   = np.array(date_ice_frac)[mask]
    date_broadband_alb_690_1190_mask = np.array(date_broadband_alb_690_1190)[mask]
    date_ice_frac_p10_mask   = np.array(date_ice_frac_p10)[mask]
    date_ice_frac_p90_mask   = np.array(date_ice_frac_p90)[mask]
    date_broadband_alb_690_1190_p10_mask = np.array(date_broadband_alb_690_1190_p10)[mask]
    date_broadband_alb_690_1190_p90_mask = np.array(date_broadband_alb_690_1190_p90)[mask]
    ice_frac_arr_mask   = np.array(date_ice_frac_mask)
    bb_alb_arr_mask     = np.array(date_broadband_alb_690_1190_mask)
    ice_frac_arr   = np.array(date_ice_frac)
    bb_alb_arr     = np.array(date_broadband_alb_690_1190)
    xerr_lo = np.maximum(0, ice_frac_arr_mask   - np.array(date_ice_frac_p10_mask))
    xerr_hi = np.maximum(0, np.array(date_ice_frac_p90_mask)   - ice_frac_arr_mask)
    yerr_lo = np.maximum(0, bb_alb_arr_mask     - np.array(date_broadband_alb_690_1190_p10_mask))
    yerr_hi = np.maximum(0, np.array(date_broadband_alb_690_1190_p90_mask) - bb_alb_arr_mask)
    ax22.errorbar(date_ice_frac_mask, date_broadband_alb_690_1190_mask,
                  xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
                  fmt='o', color='black', ecolor='lightgray',
                  markersize=3, markerfacecolor='none',
                  elinewidth=1.5, capsize=1.5, zorder=2)
    color_series_mask = [color_series[i] for i in range(len(date_all_list)) if date_all_list[i] == '0801']
    ax22.scatter(date_ice_frac_mask, date_broadband_alb_690_1190_mask, s=50, c=color_series_mask, zorder=3)
    ax22.set_xlabel('Camera Sea Ice Fraction', fontsize=14)
    ax22.set_ylabel('Broadband Albedo (690–1190 nm)', fontsize=14)
    ax22.tick_params(labelsize=12)
    ax22.set_xlim(-0.05, 1.05) 
    ax22.set_ylim(0.05, 0.9)

    # Band 2 (0.69–1.19 µm) reference albedos from Ebert & Curry (1993)
    # SZA = 60° → µ₀ = cos(60°) = 0.5; diffuse_ratio = 0.1
    mu0 = np.cos(np.deg2rad(60.0))
    diffuse_ratio = 0.1
    alb_dry_snow = diffuse_ratio * 0.832 + (1 - diffuse_ratio) * (0.902 - 0.116 * mu0)
    alb_melting_snow = 0.702  # Band 2, h_s >= 0.1 m (fixed)
    alb_w_star = 0.026 / (mu0**1.7 + 0.065) + 0.015 * (mu0 - 0.1) * (mu0 - 0.5) * (mu0 - 1.0)
    alb_water = diffuse_ratio * 0.060 + (1 - diffuse_ratio) * (alb_w_star - 0.007)
    frac_arr = np.arange(0, 1.01, 0.01)
    alb_dry_snow_frac = alb_dry_snow * frac_arr + alb_water * (1 - frac_arr)
    alb_melting_snow_frac = alb_melting_snow * frac_arr + alb_water * (1 - frac_arr)
    # ax22.plot(frac_arr, alb_dry_snow_frac, color='steelblue', linestyle='-', linewidth=1.5, label='ERA5 Dry snow (Band 2, SZA=60°)')
    # ax22.plot(frac_arr, alb_melting_snow_frac, color='darkorange', linestyle='-', linewidth=1.5, label='ERA5 Melting snow (Band 2, $h_s$>=0.1 m)')
    
    # ax22.legend(fontsize=10)

    fig.savefig(
        f'{fig_dir}/arcsix_broadband_albedo_690_1190_vs_Sea_Ice_Fraction_test.png',
        bbox_inches='tight', dpi=300
    )
    plt.close(fig)
    
    plt.close('all')

    n_dates = len(date_all_list)
    if n_dates == 0:
        color_series = []
    else:
        cmap_name = 'jet'
        cmap = mpl.colormaps.get_cmap(cmap_name)
        color_series = [cmap(0.5)] if n_dates == 1 else [cmap(i / (n_dates - 1)) for i in range(n_dates)]

    central_lon = float(np.nanmean(np.concatenate((spring.lon_avg, summer.lon_avg)))) \
        if len(spring.lon_avg) + len(summer.lon_avg) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)

    fig = plt.figure(figsize=(9, 5))
    gs  = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.3)
    ax22 = fig.add_subplot(gs[0, 0])

    ice_frac_arr   = np.array(date_ice_frac)
    bb_alb_arr     = np.array(date_broadband_alb_690_1190)
    xerr_lo = np.maximum(0, ice_frac_arr   - np.array(date_ice_frac_p10))
    xerr_hi = np.maximum(0, np.array(date_ice_frac_p90)   - ice_frac_arr)
    yerr_lo = np.maximum(0, bb_alb_arr     - np.array(date_broadband_alb_690_1190_p10))
    yerr_hi = np.maximum(0, np.array(date_broadband_alb_690_1190_p90) - bb_alb_arr)
    ax22.errorbar(date_ice_frac, date_broadband_alb_690_1190,
                  xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
                  fmt='o', color='black', ecolor='lightgray',
                  markersize=3, markerfacecolor='none',
                  elinewidth=1.5, capsize=1.5, zorder=2)
    ax22.scatter(date_ice_frac, date_broadband_alb_690_1190, s=50, c=color_series, zorder=3)
    ax22.set_xlabel('Camera Sea Ice Fraction', fontsize=14)
    ax22.set_ylabel('Broadband Albedo (690–1190 nm)', fontsize=14)
    ax22.tick_params(labelsize=12)
    ax22.set_xlim(-0.05, 1.05) 
    ax22.set_ylim(0.05, 0.9)

    # Band 2 (0.69–1.19 µm) reference albedos from Ebert & Curry (1993)
    # SZA = 60° → µ₀ = cos(60°) = 0.5; diffuse_ratio = 0.1
    # mu0 = np.cos(np.deg2rad(60.0))
    # diffuse_ratio = 0.1
    # alb_dry_snow = diffuse_ratio * 0.832 + (1 - diffuse_ratio) * (0.902 - 0.116 * mu0)
    # alb_melting_snow = 0.702  # Band 2, h_s >= 0.1 m (fixed)
    # alb_w_star = 0.026 / (mu0**1.7 + 0.065) + 0.015 * (mu0 - 0.1) * (mu0 - 0.5) * (mu0 - 1.0)
    # alb_water = diffuse_ratio * 0.060 + (1 - diffuse_ratio) * (alb_w_star - 0.007)
    # frac_arr = np.arange(0, 1.01, 0.01)
    # alb_dry_snow_frac = alb_dry_snow * frac_arr + alb_water * (1 - frac_arr)
    # alb_melting_snow_frac = alb_melting_snow * frac_arr + alb_water * (1 - frac_arr)
    # ax22.plot(frac_arr, alb_dry_snow_frac, color='steelblue', linestyle='-', linewidth=1.5, label='ERA5 Dry snow (Band 2, SZA=60°)')
    # ax22.plot(frac_arr, alb_melting_snow_frac, color='darkorange', linestyle='-', linewidth=1.5, label='ERA5 Melting snow (Band 2, $h_s$>=0.1 m)')
    
    # ax22.legend(fontsize=10)

    fig.savefig(
        f'{fig_dir}/arcsix_broadband_albedo_690_1190_vs_Sea_Ice_Fraction_test2.png',
        bbox_inches='tight', dpi=300
    )
    plt.close(fig)
    
    plt.close('all')

    n_dates = len(date_all_list)
    if n_dates == 0:
        color_series = []
    else:
        cmap_name = 'jet'
        cmap = mpl.colormaps.get_cmap(cmap_name)
        color_series = [cmap(0.5)] if n_dates == 1 else [cmap(i / (n_dates - 1)) for i in range(n_dates)]

    central_lon = float(np.nanmean(np.concatenate((spring.lon_avg, summer.lon_avg)))) \
        if len(spring.lon_avg) + len(summer.lon_avg) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)

    fig = plt.figure(figsize=(9, 5))
    gs  = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.3)
    ax22 = fig.add_subplot(gs[0, 0])

    ice_frac_arr   = np.array(date_ice_frac)
    bb_alb_arr     = np.array(date_broadband_alb_690_1190)
    xerr_lo = np.maximum(0, ice_frac_arr   - np.array(date_ice_frac_p10))
    xerr_hi = np.maximum(0, np.array(date_ice_frac_p90)   - ice_frac_arr)
    yerr_lo = np.maximum(0, bb_alb_arr     - np.array(date_broadband_alb_690_1190_p10))
    yerr_hi = np.maximum(0, np.array(date_broadband_alb_690_1190_p90) - bb_alb_arr)
    ax22.errorbar(date_ice_frac, date_broadband_alb_690_1190,
                  xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
                  fmt='o', color='black', ecolor='lightgray',
                  markersize=3, markerfacecolor='none',
                  elinewidth=1.5, capsize=1.5, zorder=2)
    ax22.scatter(date_ice_frac, date_broadband_alb_690_1190, s=50, c=color_series, zorder=3)
    ax22.set_xlabel('Camera Sea Ice Fraction', fontsize=14)
    ax22.set_ylabel('Broadband Albedo (690–1190 nm)', fontsize=14)
    ax22.tick_params(labelsize=12)
    ax22.set_xlim(-0.05, 1.05) 
    ax22.set_ylim(0.05, 0.9)

    # Band 2 (0.69–1.19 µm) reference albedos from Ebert & Curry (1993)
    # SZA = 60° → µ₀ = cos(60°) = 0.5; diffuse_ratio = 0.1
    mu0 = np.cos(np.deg2rad(60.0))
    diffuse_ratio = 0.1
    alb_dry_snow = diffuse_ratio * 0.832 + (1 - diffuse_ratio) * (0.902 - 0.116 * mu0)
    alb_melting_snow = 0.702  # Band 2, h_s >= 0.1 m (fixed)
    alb_w_star = 0.026 / (mu0**1.7 + 0.065) + 0.015 * (mu0 - 0.1) * (mu0 - 0.5) * (mu0 - 1.0)
    alb_water = diffuse_ratio * 0.060 + (1 - diffuse_ratio) * (alb_w_star - 0.007)
    frac_arr = np.arange(0, 1.01, 0.01)
    alb_dry_snow_frac = alb_dry_snow * frac_arr + alb_water * (1 - frac_arr)
    alb_melting_snow_frac = alb_melting_snow * frac_arr + alb_water * (1 - frac_arr)
    ax22.plot(frac_arr, alb_dry_snow_frac, color='steelblue', linestyle='-', linewidth=1.5, label='ERA5 Dry snow (Band 2, SZA=60°)')
    ax22.plot(frac_arr, alb_melting_snow_frac, color='darkorange', linestyle='-', linewidth=1.5, label='ERA5 Melting snow (Band 2, $h_s$>=0.1 m)')
    
    ax22.legend(fontsize=10)

    fig.savefig(
        f'{fig_dir}/arcsix_broadband_albedo_690_1190_vs_Sea_Ice_Fraction.png',
        bbox_inches='tight', dpi=300
    )
    plt.close(fig)
    
    
    # arcsix_broadband_albedo_vs_NSIDC_Sea_Ice_Fraction
    plt.close('all')

    n_dates = len(date_all_list)
    if n_dates == 0:
        color_series = []
    else:
        cmap_name = 'jet'
        cmap = mpl.colormaps.get_cmap(cmap_name)
        color_series = [cmap(0.5)] if n_dates == 1 else [cmap(i / (n_dates - 1)) for i in range(n_dates)]

    central_lon = float(np.nanmean(np.concatenate((spring.lon_avg, summer.lon_avg)))) \
        if len(spring.lon_avg) + len(summer.lon_avg) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)

    fig = plt.figure(figsize=(9, 5))
    gs  = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.3)
    ax22 = fig.add_subplot(gs[0, 0])

    print("date_amsr2_ice_conc min/max:", np.nanmin(date_amsr2_ice_conc), np.nanmax(date_amsr2_ice_conc))
    amsr2_ice_frac_arr   = np.array(date_amsr2_ice_conc)/100
    bb_alb_arr     = np.array(date_broadband_alb)
    xerr_lo = np.maximum(0,  amsr2_ice_frac_arr - np.array(date_amsr2_ice_conc_p10)/100)
    xerr_hi = np.maximum(0, np.array(date_amsr2_ice_conc_p90)/100   - amsr2_ice_frac_arr)
    yerr_lo = np.maximum(0, bb_alb_arr     - np.array(date_broadband_alb_p10))
    yerr_hi = np.maximum(0, np.array(date_broadband_alb_p90) - bb_alb_arr)
    ax22.errorbar(amsr2_ice_frac_arr, date_broadband_alb,
                  xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
                  fmt='o', color='black', ecolor='lightgray',
                  markersize=3, markerfacecolor='none',
                  elinewidth=1.5, capsize=1.5, zorder=2)
    ax22.scatter(amsr2_ice_frac_arr, date_broadband_alb, s=50, c=color_series, zorder=3)
    ax22.set_xlabel('NSIDC Sea Ice Fraction', fontsize=14)
    ax22.set_ylabel('Broadband Albedo (nm)', fontsize=14)
    ax22.tick_params(labelsize=12)
    ax22.set_xlim(-0.05, 1.05) 
    ax22.set_ylim(0.05, 0.9)

    # Band 2 (0.69–1.19 µm) reference albedos from Ebert & Curry (1993)
    # SZA = 60° → µ₀ = cos(60°) = 0.5; diffuse_ratio = 0.1
    # mu0 = np.cos(np.deg2rad(60.0))
    # diffuse_ratio = 0.1
    # alb_dry_snow = diffuse_ratio * 0.832 + (1 - diffuse_ratio) * (0.902 - 0.116 * mu0)
    # alb_melting_snow = 0.702  # Band 2, h_s >= 0.1 m (fixed)
    # alb_w_star = 0.026 / (mu0**1.7 + 0.065) + 0.015 * (mu0 - 0.1) * (mu0 - 0.5) * (mu0 - 1.0)
    # alb_water = diffuse_ratio * 0.060 + (1 - diffuse_ratio) * (alb_w_star - 0.007)
    # frac_arr = np.arange(0, 1.01, 0.01)
    # alb_dry_snow_frac = alb_dry_snow * frac_arr + alb_water * (1 - frac_arr)
    # alb_melting_snow_frac = alb_melting_snow * frac_arr + alb_water * (1 - frac_arr)
    # ax22.plot(frac_arr, alb_dry_snow_frac, color='steelblue', linestyle='-', linewidth=1.5, label='ERA5 Dry snow (Band 2, SZA=60°)')
    # ax22.plot(frac_arr, alb_melting_snow_frac, color='darkorange', linestyle='-', linewidth=1.5, label='ERA5 Melting snow (Band 2, $h_s$>=0.1 m)')
    
    # ax22.legend(fontsize=10)

    fig.savefig(
        f'{fig_dir}/arcsix_broadband_albedo_vs_NSIDC_Sea_Ice_Fraction.png',
        bbox_inches='tight', dpi=300
    )
    plt.close(fig)

    # AMSR2 ice concentration vs NSIDC ice fraction (per-point, per season)
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, season_obj, color, label in zip(
        axes,
        [spring, summer],
        ['royalblue', 'tomato'],
        ['Spring', 'Summer'],
    ):
        alt_mask  = season_obj.alt <= 1.6
        ax.scatter(
            season_obj.amsr2_ice_conc[alt_mask] / 100.0,
            season_obj.ice_frac[alt_mask],
            s=5, color=color, alpha=0.3, label=label,
        )
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='1:1')
        ax.set_xlabel('AMSR2 Ice Concentration', fontsize=13)
        ax.set_ylabel('NSIDC Ice Fraction', fontsize=13)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(label, fontsize=13)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=11)
    fig.suptitle('AMSR2 Ice Concentration vs NSIDC Ice Fraction', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/amsr2_ice_conc_vs_ice_frac.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # date-averaged AMSR2 ice concentration vs camera ice fraction (with p10/p90 error bars)
    plt.close('all')
    amsr2_ice_frac_arr = np.array(date_amsr2_ice_conc) / 100.0
    ice_frac_arr       = np.array(date_ice_frac)
    xerr_lo = np.maximum(0, amsr2_ice_frac_arr - np.array(date_amsr2_ice_conc_p10) / 100.0)
    xerr_hi = np.maximum(0, np.array(date_amsr2_ice_conc_p90) / 100.0 - amsr2_ice_frac_arr)
    yerr_lo = np.maximum(0, ice_frac_arr - np.array(date_ice_frac_p10))
    yerr_hi = np.maximum(0, np.array(date_ice_frac_p90) - ice_frac_arr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(amsr2_ice_frac_arr, ice_frac_arr,
                xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
                fmt='o', color='black', ecolor='lightgray',
                markersize=3, markerfacecolor='none',
                elinewidth=1.5, capsize=1.5, zorder=2)
    ax.scatter(amsr2_ice_frac_arr, ice_frac_arr, s=50, c=color_series, zorder=3)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='1:1')
    ax.set_xlabel('AMSR2 Ice Concentration', fontsize=14)
    ax.set_ylabel('Camera Ice Fraction', fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/amsr2_ice_frac_vs_camera_ice_frac.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # date-averaged AMSR2 ice concentration vs camera ice fraction (with p10/p90 error bars)
    plt.close('all')
    amsr2_ice_frac_arr = np.array(date_amsr2_ice_conc) / 100.0
    ice_frac_arr       = np.array(date_ice_frac)
    xerr_lo = np.maximum(0, amsr2_ice_frac_arr - np.array(date_amsr2_ice_conc_p10) / 100.0)
    xerr_hi = np.maximum(0, np.array(date_amsr2_ice_conc_p90) / 100.0 - amsr2_ice_frac_arr)
    yerr_lo = np.maximum(0, ice_frac_arr - np.array(date_ice_frac_p10))
    yerr_hi = np.maximum(0, np.array(date_ice_frac_p90) - ice_frac_arr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(amsr2_ice_frac_arr, ice_frac_arr,
                xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
                fmt='o', color='black', ecolor='lightgray',
                markersize=3, markerfacecolor='none',
                elinewidth=1.5, capsize=1.5, zorder=2)
    ax.scatter(amsr2_ice_frac_arr, ice_frac_arr, s=50, c=color_series, zorder=3)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='1:1')
    ax.set_xlabel('AMSR2 Ice Concentration', fontsize=14)
    ax.set_ylabel('Camera Ice Fraction', fontsize=14)
    ax.set_xlim(0.65, 1.05)
    ax.set_ylim(0.65, 1.05)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/amsr2_ice_frac_vs_camera_ice_frac_zoomin.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':


    dir_fig = './fig'
    os.makedirs(dir_fig, exist_ok=True)

    config = FlightConfig(mission='ARCSIX',
                            platform='P3B',
                            data_root=_fdir_data_,
                            root_mac=_fdir_general_,
                            root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data',)

    """
    # IMPORTANT
    # need to run arcsix_gas_insitu.py first to generate gas files for each date
    """

    # surface albedo derivation
    # ------------------------------------------


    combined_atm_corr()
