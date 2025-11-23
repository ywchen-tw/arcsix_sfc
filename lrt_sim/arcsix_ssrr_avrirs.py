"""
Inheritated from Hong Chen (hong.chen@lasp.colorado.edu)
Edited by Yu-Wen Chen (yu-wen.chen@colorado.edu)

This code serves as an example code to reproduce 3D irradiance simulation for App. 3 in Chen et al. (2022).
Special note: due to large data volume, only partial flight track simulation is provided for illustration purpose.

The processes include:
    1) `main_run()`: pre-process aircraft and satellite data and run simulations
        a) partition flight track into mini flight track segments and collocate satellite imagery data
        b) run simulations based on satellite imagery cloud retrievals
            i) 3D mode
            ii) IPA mode

    2) `main_post()`: post-process data
        a) extract radiance observations from pre-processed data
        b) extract radiance simulations of EaR3T
        c) plot

This code has been tested under:
    1) Linux on 2023-06-27 by Hong Chen
      Operating System: Red Hat Enterprise Linux
           CPE OS Name: cpe:/o:redhat:enterprise_linux:7.7:GA:workstation
                Kernel: Linux 3.10.0-1062.9.1.el7.x86_64
          Architecture: x86-64
"""

import os
import sys
import glob
import copy
import time
from collections import OrderedDict
import datetime
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from pathlib import Path
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
from geopy import distance
import bisect
import pandas as pd
import xarray as xr
from collections import defaultdict
import platform
import gc
from pyproj import Transformer
# mpl.use('Agg')


import er3t

from util.util import *
from util.arcsix_atm import prepare_atmospheric_profile

_mission_      = 'arcsix'
_platform_     = 'p3b'

_hsk_          = 'hsk'
_alp_          = 'alp'
_spns_         = 'spns-a'
_ssfr1_        = 'ssfr-a'
_ssfr2_        = 'ssfr-b'
_cam_          = 'nac'

# _fdir_main_       = 'data/%s/flt-vid' % _mission_
_fdir_main_       = 'data/flt-vid'
_fdir_sat_img_    = 'data/%s/sat-img' % _mission_
_fdir_sat_data_   = 'data/%s/sat' % _mission_
_fdir_cam_img_    = 'data/%s/2024-Spring/p3' % _mission_
_wavelength_      = 555.0

_fdir_sat_img_vn_ = 'data/%s/sat-img-vn' % _mission_

_preferred_region_ = 'ca_archipelago'
_aspect_ = 'equal'

if platform.system() == 'Darwin':
    _fdir_data_ = '/Volumes/argus/field/%s/processed' % _mission_
    _fdir_data_ = '../data/processed' 
    _fdir_general_ = '../data'
    _fdir_tmp_ = './tmp'
elif platform.system() == 'Linux':
    _fdir_data_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data/processed"
    _fdir_general_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data"
    _fdir_tmp_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/tmp"
_fdir_tmp_graph_ = 'tmp-graph_flt-vid'

_title_extra_ = 'ARCSIX RF#1'

_tmhr_range_ = {
        '20240517': [19.20, 23.00],
        '20240521': [14.80, 17.50],
        }

# dates for ARCSIX-1
#╭────────────────────────────────────────────────────────────────────────────╮#
_dates1_ = [
        datetime.datetime(2024, 5, 28),
        datetime.datetime(2024, 5, 30),
        datetime.datetime(2024, 5, 31),
        datetime.datetime(2024, 6,  3),
        datetime.datetime(2024, 6,  5),
        datetime.datetime(2024, 6,  6),
        datetime.datetime(2024, 6,  7),
        datetime.datetime(2024, 6, 10),
        datetime.datetime(2024, 6, 11),
        datetime.datetime(2024, 6, 13),
    ]
#╰────────────────────────────────────────────────────────────────────────────╯#

# dates for ARCSIX-2
#╭────────────────────────────────────────────────────────────────────────────╮#
_dates2_ = [
        datetime.datetime(2024, 7, 25),
        datetime.datetime(2024, 7, 29),
        datetime.datetime(2024, 7, 30),
        datetime.datetime(2024, 8,  1),
        datetime.datetime(2024, 8,  2),
        datetime.datetime(2024, 8,  7),
        datetime.datetime(2024, 8,  8),
        datetime.datetime(2024, 8,  9),
        datetime.datetime(2024, 8,  15),
    ]
#╰────────────────────────────────────────────────────────────────────────────╯#

_dates_ = _dates1_


o2a_1_start, o2a_1_end = 748, 776
h2o_1_start, h2o_1_end = 672, 706
h2o_2_start, h2o_2_end = 705, 746
h2o_3_start, h2o_3_end = 884, 996
h2o_4_start, h2o_4_end = 1084, 1175
h2o_5_start, h2o_5_end = 1230, 1286
h2o_6_start, h2o_6_end = 1290, 1509
h2o_7_start, h2o_7_end = 1748, 2050
final_start, final_end = 2110, 2200

gas_bands = [(o2a_1_start, o2a_1_end), (h2o_1_start, h2o_1_end), (h2o_2_start, h2o_2_end),
                (h2o_3_start, h2o_3_end), (h2o_4_start, h2o_4_end), (h2o_5_start, h2o_5_end),
                (h2o_6_start, h2o_6_end), (h2o_7_start, h2o_7_end), (final_start, final_end)]

 
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
                   aviris_reflectance_spectrum_unc=None
                   ):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if aviris_file is not None:
        aviris_label = 'AVIRIS Reflectance' if ((aviris_file is not None) and (aviris_closest)) else 'All AVIRIS mean'
        ax.scatter(aviris_reflectance_wvl, aviris_reflectance_spectrum, s=5, c='m', label=aviris_label, alpha=0.7) if ((aviris_file is not None)) else None
        ax.fill_between(aviris_reflectance_wvl, aviris_reflectance_spectrum-aviris_reflectance_spectrum_unc, aviris_reflectance_spectrum+aviris_reflectance_spectrum_unc, color='m', alpha=0.3) if aviris_file is not None else None
        if modis_alb_file is not None:
            ax.scatter(modis_bands_nm, modis_alb_legs, s=50, c='g', marker='*', label='MODIS Albedo', edgecolors='k')

    for i in range(len(tmhr_ranges_select)):
        if lon_avg_all is not None and lat_avg_all is not None:
            ax.plot(wvl, alb[i, :], '-', color=color_series[i], label='Z=%.2fkm, lon:%.2f, lat: %.2f' % (alt_avg_all[i], lon_avg_all[i], lat_avg_all[i]))
        else:
            ax.plot(wvl, alb[i, :], '-', color=color_series[i], label='Z=%.2fkm' % (alt_avg_all[i]))
        if aviris_file is None and modis_alb_file is not None:
            ax.scatter(modis_bands_nm, modis_alb_legs[i], s=50, color=color_series[i], marker='*', edgecolors='k')
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
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


def lonlat_dist(ref_lon, ref_lat, xy_lon, xy_lat):
    # Approximate radius of earth in km
    R = 6373.0

    xy_lat_radians = np.radians(xy_lat)
    xy_lon_radians = np.radians(xy_lon)
    ref_lat_radians = np.radians(ref_lat)
    ref_lon_radians = np.radians(ref_lon)

    dlon = xy_lon_radians - ref_lon_radians
    dlat = xy_lat_radians - ref_lat_radians

    a = np.sin(dlat / 2)**2 + np.cos(ref_lat_radians) * np.cos(xy_lat_radians) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # im = ax.pcolormesh(xy_lon, xy_lat, distance, cmap='jet', shading='auto')
    # plt.colorbar(im, ax=ax, label='Distance')
    # ax.scatter(ref_lon, ref_lat, c='r', marker='x', s=100, label='Flight Leg Center')
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # ax.legend(fontsize=12)
    # fig.tight_layout()
    # plt.show()
    # sys.exit()

    return distance

def atm_corr_plot(date=datetime.datetime(2024, 5, 31),
                     tmhr_ranges_select=[[14.10, 14.27]],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                     simulation_interval: Optional[float] = None,  # in minutes, e.g., 10 for 10 minutes
                     rsp_plot=False,
                     aviris_id=None,
                            ):
    log = logging.getLogger("atm corr spiral plot")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    doy_s = date.timetuple().tm_yday
    print(f"Processing date: {date_s}, DOY: {doy_s}")
    
    if simulation_interval is not None:
        # split tmhr_ranges_select into smaller intervals
        tmhr_ranges_select_new = []
        for lo, hi in tmhr_ranges_select:
            t_start = lo
            while t_start < hi and t_start < (hi - 0.0167/6):  # 1 minute
                t_end = min(t_start + simulation_interval/60.0, hi)
                tmhr_ranges_select_new.append([t_start, t_end])
                t_start = t_end
    tmhr_ranges_select = tmhr_ranges_select_new

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    
    modis_alb_dir = f'{_fdir_general_}/modis_albedo'
    # list all modis albedo files
    modis_alb_files = sorted(glob.glob(os.path.join(modis_alb_dir, f'M*.nc')))
    for fname in modis_alb_files:
        print("Checking modis file:", os.path.basename(fname).split('.')[1])
        if str(doy_s) in os.path.basename(fname).split('.')[1]:
            modis_alb_file = fname
            break
    else:
        modis_alb_file = None

        
    if modis_alb_file is not None:
        with Dataset(modis_alb_file, 'r') as ds:
            modis_lon = ds.variables['Longitude'][:]
            modis_lat = ds.variables['Latitude'][:]
            modis_bands = ds.variables['Bands'][:]
            modis_sur_alb = ds.variables['Albedo_1km'][:]
        
        modis_alb_legs = []
        modis_bands_nm = np.array([float(i) for i in modis_bands[:7]])*1000  # in nm   
        
    aviris_rad_file = None
    aviris_file = None
    aviris_dir = f'{_fdir_general_}/aviris_ng'
    if aviris_id is not None:
        aviris_files = sorted(glob.glob(os.path.join(aviris_dir, f'ang*.nc')))
    else:
        aviris_files = sorted(glob.glob(os.path.join(aviris_dir, f'{aviris_id}*.nc')))
    for fname in aviris_files:
        if date_s in os.path.basename(fname):
            if 'RFL' in os.path.basename(fname):
                aviris_file = fname
            if 'RDN' in os.path.basename(fname):
                aviris_rad_file = fname
            # break
    if aviris_file is None and aviris_rad_file is None:
        aviris_closest = False
    
    aviris_file = None
            
    print("modis_alb_file:", modis_alb_file)
    
    
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    for i in range(len(tmhr_ranges_select)):
        fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
        with open(fname_pkl, 'rb') as f:
            cld_leg = pickle.load(f)
        time = cld_leg['time']
        time_start = tmhr_ranges_select[i][0]
        time_end = tmhr_ranges_select[i][1]
        
 
        alt = cld_leg['alt']
        lon_all = cld_leg['lon']
        lat_all = cld_leg['lat']
        heading = cld_leg['heading']
        hsr1_tot = cld_leg['hsr1_tot']
        hsr1_dif = cld_leg['hsr1_dif']
        hsr1_wvl = cld_leg['hsr1_wvl']
        rsp_lon = cld_leg['rsp_lon']
        rsp_lat = cld_leg['rsp_lat']
        rsp_rad = cld_leg['rsp_rad']
        rsp_rad_norm = cld_leg['rsp_rad_norm']
        rsp_ref = cld_leg['rsp_ref']
        rsp_wvl = cld_leg['rsp_wvl']
        rsp_mu0 = cld_leg['rsp_mu0']
        rsp_sd = cld_leg['rsp_sd']
        sza = cld_leg['sza']
        saa = cld_leg['saa']
        ssfr_wvl_zen = cld_leg['ssfr_zen_wvl']
        ssfr_wvl_nad = cld_leg['ssfr_nad_wvl']
        ssfr_flux_zen = cld_leg['ssfr_zen']
        ssfr_flux_nad = cld_leg['ssfr_nad']
        ssrr_wvl_zen = cld_leg['ssrr_zen_wvl']
        ssrr_rad_zen = cld_leg['ssrr_zen_rad']
        ssrr_wvl_nad = cld_leg['ssrr_nad_wvl']
        ssrr_rad_nad = cld_leg['ssrr_nad_rad']
        toa_flux = cld_leg['ssfr_toa']
        
        alt_avg = np.nanmean(alt)  # in km
        lon_avg = np.nanmean(lon_all)
        lat_avg = np.nanmean(lat_all)
        
        if i == 0:
            time_all = []
            fdn_550_all = []
            fup_550_all = []
            fdn_1600_all = []
            fup_1600_all = []
            
            lon_avg_all = np.zeros(len(tmhr_ranges_select))
            lat_avg_all = np.zeros(len(tmhr_ranges_select))
            lon_min_all = np.zeros(len(tmhr_ranges_select))
            lon_max_all = np.zeros(len(tmhr_ranges_select))
            lat_min_all = np.zeros(len(tmhr_ranges_select))
            lat_max_all = np.zeros(len(tmhr_ranges_select))
            alt_avg_all = np.zeros(len(tmhr_ranges_select))

            
            flux_wvl = ssfr_wvl_zen
            ssfr_fup_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fup_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            toa_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_up_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_dn_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_up_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_dn_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            
            rsp_rad_mean_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_rad_std_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_ref_mean_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_ref_std_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_mu0_mean_all = np.zeros(len(tmhr_ranges_select))
            rsp_sd_mean_all = np.zeros(len(tmhr_ranges_select))

            

        time_all.append(time)
        ssfr_wvl = cld_leg['ssfr_zen_wvl']
        ssfr_550_ind = np.argmin(np.abs(ssfr_wvl - 550))
        ssfr_1600_ind = np.argmin(np.abs(ssfr_wvl - 1600))
        fdn_550_all.append(cld_leg['ssfr_zen'][:, ssfr_550_ind])
        fup_550_all.append(cld_leg['ssfr_nad'][:, ssfr_550_ind])
        fdn_1600_all.append(cld_leg['ssfr_zen'][:, ssfr_1600_ind])
        fup_1600_all.append(cld_leg['ssfr_nad'][:, ssfr_1600_ind])
        
        
        lon_avg_all[i] = lon_avg.copy()
        lat_avg_all[i] = lat_avg.copy()
        lon_min_all[i] = lon_all.min()
        lon_max_all[i] = lon_all.max()
        lat_min_all[i] = lat_all.min()
        lat_max_all[i] = lat_all.max()
        alt_avg_all[i] = alt_avg.copy()
        
        
        ssfr_fup_mean_all[i, :] = np.nanmean(ssfr_flux_nad, axis=0)
        ssfr_fdn_mean_all[i, :] = np.nanmean(ssfr_flux_zen, axis=0)
        ssfr_fup_std_all[i, :] = np.nanstd(ssfr_flux_nad, axis=0)
        ssfr_fdn_std_all[i, :] = np.nanstd(ssfr_flux_zen, axis=0)
        toa_mean_all[i, :] = np.nanmean(toa_flux, axis=0)
        
        ssrr_rad_up_mean_all[i, :] = np.nanmean(ssrr_rad_nad, axis=0)
        ssrr_rad_dn_mean_all[i, :] = np.nanmean(ssrr_rad_zen, axis=0)
        ssrr_rad_up_std_all[i, :] = np.nanstd(ssrr_rad_nad, axis=0)
        ssrr_rad_dn_std_all[i, :] = np.nanstd(ssrr_rad_zen, axis=0)
        ssrr_rad_up_wvl = ssrr_wvl_nad
        ssrr_rad_dn_wvl = ssrr_wvl_zen
        
        rsp_wvl = cld_leg['rsp_wvl']
        rsp_mu0 = cld_leg['rsp_mu0']
        rsp_sd = cld_leg['rsp_sd']
        
        if cld_leg['rsp_rad'] is not None:
            print("cld_leg['rsp_rad'] shape:", cld_leg['rsp_rad'].shape)
            
            rsp_rad_mean_all[i, :] = np.nanmean(cld_leg['rsp_rad'], axis=0)/1000  # in W m-2 sr-1 nm-1
            rsp_rad_std_all[i, :] = np.nanstd(cld_leg['rsp_rad'], axis=0)/1000  # in W m-2 sr-1 nm-1
            
            rsp_ref_mean_all[i, :] = np.nanmean(cld_leg['rsp_ref'], axis=0)
            rsp_ref_std_all[i, :] = np.nanstd(cld_leg['rsp_ref'], axis=0)
            
            rsp_mu0_mean_all[i] = np.nanmean(cld_leg['rsp_mu0'])
            rsp_sd_mean_all[i] = np.nanmean(cld_leg['rsp_sd'])
        
        print(f"date_s: {date_s}, time: {time_start:.2f}-{time_end:.2f}, alt_avg: {alt_avg:.2f} km")
    
        # find the modis location closest to the flight leg center
        if modis_alb_file is not None:
            dist = np.sqrt((modis_lon - lon_avg)**2 + (modis_lat - lat_avg)**2)
            min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            modis_alb_leg = modis_sur_alb[min_idx[0], min_idx[1], :7]
            modis_alb_legs.append(modis_alb_leg)
    

    if modis_alb_file is None:
        modis_bands_nm = None
        modis_alb_legs = None

    log.info(f"Saved surface albedo updates to {_fdir_general_}/sfc_alb/sfc_alb_update_{date_s}_{case_tag}.pkl")

    ssrr_ref = ssrr_rad_up_mean_all * np.pi / toa_mean_all / rsp_mu0_mean_all[:, np.newaxis] * (rsp_sd_mean_all[:, np.newaxis]**2)
    ssfr_fup_ref = ssfr_fup_mean_all / toa_mean_all /  rsp_mu0_mean_all[:, np.newaxis] * (rsp_sd_mean_all[:, np.newaxis]**2)
       
    print("rsp_wvl:", rsp_wvl)
    
    print("ssrr_rad_up_mean_all:", ssrr_rad_up_mean_all[0,:])
        
    print("lon avg all mean:", lon_avg_all.mean())
    print("lat avg all mean:", lat_avg_all.mean())
    
    
    if aviris_file is not None:
        
        # 1) Open your NetCDF
        ds = Dataset(aviris_file)
        easting  = ds.variables["easting"][:]   # shape (1665,)
        northing = ds.variables["northing"][:]  # shape (1207,)
        
        # 2) Read GeoTransform
        gt = list(map(float, ds['transverse_mercator'].getncattr("GeoTransform").split()))
        crs_info_ind = ds['transverse_mercator'].getncattr("crs_wkt").find('AUTHORITY["EPSG","326')
        crs_wkt = ds['transverse_mercator'].getncattr("crs_wkt")
        print("crs_info_ind:", crs_info_ind)
        # AUTHORITY["EPSG","32622"]
        epsg_code = int(ds['transverse_mercator'].getncattr("crs_wkt")[crs_info_ind+18:crs_info_ind+23])
        print("AVIRIS EPSG code:", epsg_code)
        GT0, GT1, GT2, GT3, GT4, GT5 = gt

        # 3) Prepare transformer
        # transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
        transformer = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)

        # Example: compute lon/lat for point at (row, col)
        row, col = 0, 0

        # Projected coords
        x = GT0 + col * GT1 + row * GT2
        y = GT3 + col * GT4 + row * GT5

        # Geographic coords
        lon, lat = transformer.transform(x, y)

        print(f"Row {row}, Col {col} → x={x:.2f} m, y={y:.2f} m → lon={lon:.6f}, lat={lat:.6f}")
        
        

        # 2) Build a 2D meshgrid of coordinates
        #    X will have shape (1207, 1665) where each row is a copy of easting
        #    Y will have shape (1207, 1665) where each column is a copy of northing
        X, Y = np.meshgrid(easting, northing)


        # 4) Transform the entire grid in one go
        #    lon, lat will each be arrays of shape (1207, 1665)
        lon, lat = transformer.transform(X, Y)
        
        aviris_reflectance_wvl = ds.groups['reflectance'].variables['wavelength'][:]  # in nm
        aviris_reflectance_data = ds.groups['reflectance'].variables['reflectance'][:]  # shape (wvl, northing, easting)
        aviris_reflectance_data = np.where(aviris_reflectance_data<0, np.nan, aviris_reflectance_data)
        # aviris_reflectance_data[~aviris_reflectance_data>0] = np.nan
        
        wvl_nancheck = 550  # nm
        wvl_nancheck_idx = np.argmin(np.abs(aviris_reflectance_wvl - wvl_nancheck))
        wvl_nancheck_mask = ~np.isnan(aviris_reflectance_data[wvl_nancheck_idx, :, :])
        
        wvl_plot = 550  # nm
        wvl_idx = np.argmin(np.abs(aviris_reflectance_wvl - wvl_plot))
        reflectance_550 = aviris_reflectance_data[wvl_idx, :, :]  # shape (northing, easting)
        
        aviris_ref_all = np.zeros((len(tmhr_ranges_select), len(aviris_reflectance_wvl)))
        aviris_ref_all_unc = np.zeros((len(tmhr_ranges_select), len(aviris_reflectance_wvl)))
        for j in range(len(tmhr_ranges_select)):
            # find lon, lat mean index in the aviris data
            lon_mean = lon_avg_all[j]
            lat_mean = lat_avg_all[j]
            dist = np.sqrt((lon - lon_mean)**2 + (lat - lat_mean)**2)
            min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            print("lon_mean, lat_mean:", lon_mean, lat_mean)
            print("Closest AVIRIS pixel lon, lat:", lon[min_idx], lat[min_idx])
            print("Closest AVIRIS pixel index (northing, easting):", min_idx)
        

            
            
            print("aviris_reflectance_data shape:", aviris_reflectance_data.shape)
            print("min_idx[0], min_idx[1]:", min_idx[0], min_idx[1])
            idx0_start = max(0, min_idx[0]-3)
            idx0_end = min(aviris_reflectance_data.shape[1], min_idx[0]+4)
            idx1_start = max(0, min_idx[1]-3)
            idx1_end = min(aviris_reflectance_data.shape[2], min_idx[1]+4)
            ref_data_subset = aviris_reflectance_data[:, idx0_start:idx0_end, idx1_start:idx1_end]
            print("ref_data_subset shape:", ref_data_subset.shape)
            # aviris_reflectance_spectrum = np.nanmean(ref_data_subset, axis=(1,2))
            # aviris_reflectance_spectrum_unc = np.nanstd(ref_data_subset, axis=(1,2))
            
            if np.all(np.isnan(ref_data_subset)):
                while np.all(np.isnan(ref_data_subset)):
                    print("Closest pixel reflectance spectrum is all NaN, finding closer pixels and averaging...")
                    valid_lon, valid_lat = lon[wvl_nancheck_mask], lat[wvl_nancheck_mask]
                    valid_aviris_reflectance_data = aviris_reflectance_data[:, wvl_nancheck_mask]
                    valid_dist = np.sqrt((valid_lon - lon_mean)**2 + (valid_lat - lat_mean)**2)
                    min_idx_list = np.unravel_index(np.argsort(valid_dist, axis=None)[:25], valid_dist.shape)
                    # min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                    print("New min_idx:", min_idx_list)
                    print("New closest AVIRIS pixel lon, lat:", valid_lon[min_idx_list][0], valid_lat[min_idx_list][0])
                    ref_data_subset = valid_aviris_reflectance_data[:, min_idx_list]
                print("New ref_data_subset shape:", ref_data_subset.shape)
                aviris_reflectance_spectrum = np.nanmean(ref_data_subset, axis=(1, 2))
                print("New aviris_reflectance_spectrum shape:", aviris_reflectance_spectrum.shape)
                aviris_reflectance_spectrum_unc = np.nanstd(ref_data_subset, axis=(1, 2))
                aviris_closest = False
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.pcolormesh(lon, lat, reflectance_550, cmap='jet', shading='auto')
                plt.colorbar(im, ax=ax, label='Reflectance at 550 nm')
                ax.scatter(valid_lon[min_idx_list[0][0]], valid_lat[min_idx_list[0][0]], facecolors='none', edgecolor='black', s=100, label='Closest Pixel')
                ax.scatter(lon_mean, lat_mean, color='red', marker='x', s=100, label='Flight Leg Center')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f'AVIRIS Reflectance at {wvl_plot} nm')
                ax.legend(fontsize=12)
                fig.tight_layout()
                fig.savefig('fig/%s/%s_%s_aviris_reflectance_550nm_leg_%d.png' % (date_s, date_s, case_tag, j), bbox_inches='tight', dpi=150)
                # plt.show()
            else:
                aviris_reflectance_spectrum = np.nanmean(ref_data_subset, axis=(1,2))
                aviris_reflectance_spectrum_unc = np.nanstd(ref_data_subset, axis=(1,2))
                aviris_closest = True
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.pcolormesh(lon, lat, reflectance_550, cmap='jet', shading='auto')
                plt.colorbar(im, ax=ax, label='Reflectance at 550 nm')
                ax.scatter(lon[min_idx], lat[min_idx], facecolors='none', edgecolor='black', s=100, label='Closest Pixel')
                ax.scatter(lon_mean, lat_mean, color='red', marker='x', s=100, label='Flight Leg Center')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f'AVIRIS Reflectance at {wvl_plot} nm')
                ax.legend(fontsize=12)
                fig.tight_layout()
                fig.savefig('fig/%s/%s_%s_aviris_reflectance_550nm_leg_d.png' % (date_s, date_s, case_tag, j), bbox_inches='tight', dpi=150)
            
            aviris_ref_all[j, :] = aviris_reflectance_spectrum
            aviris_ref_all_unc[j, :] = aviris_reflectance_spectrum_unc

    
    if aviris_rad_file is not None:
        ds_rad = Dataset(aviris_rad_file)
        aviris_rad_wvl = ds_rad.groups['radiance'].variables['wavelength'][:]
        aviris_rad_wvl = np.array([float(i) for i in aviris_rad_wvl])  # in nm
        aviris_rad = ds_rad.groups['radiance'].variables['radiance'][:]  # shape (wvl, lat, lon) # in uW nm-1 cm-2 sr-1
        aviris_rad = aviris_rad * 1e-6 * 1e4  # convert to W nm-1 m-2 sr-1
        aviris_rad_lon = ds_rad.variables['lon'][:]
        aviris_rad_lat = ds_rad.variables['lat'][:]
        ds_rad.close()
        
        
        wvl_plot = 550  # nm
        rad_wvl_idx = np.argmin(np.abs(aviris_rad_wvl - wvl_plot))
        rad_550 = aviris_rad[rad_wvl_idx, :, :]  # shape (northing, easting)
        
        aviris_nearest_lon_all = np.zeros(len(tmhr_ranges_select))
        aviris_nearest_lat_all = np.zeros(len(tmhr_ranges_select))
        aviris_rad_all = np.zeros((len(tmhr_ranges_select), len(aviris_rad_wvl)))
        aviris_rad_all_unc = np.zeros((len(tmhr_ranges_select), len(aviris_rad_wvl)))
        for j in range(len(tmhr_ranges_select)):
            # find the closest pixel in aviris_rad to the flight leg center
            # find lon, lat mean index in the aviris data
            lon_mean = lon_avg_all[j]
            lat_mean = lat_avg_all[j]
            # dist_rad = np.sqrt((aviris_rad_lon - lon_mean)**2 + (aviris_rad_lat - lat_mean)**2)
            dist_rad = lonlat_dist(lon_mean, lat_mean, aviris_rad_lon, aviris_rad_lat)
            min_rad_idx = np.unravel_index(np.argmin(dist_rad, axis=None), dist_rad.shape)
        
            idx0_start = max(0, min_rad_idx[0]-4)
            idx0_end = min(aviris_rad.shape[1], min_rad_idx[0]+5)
            idx1_start = max(0, min_rad_idx[1]-4)
            idx1_end = min(aviris_rad.shape[2], min_rad_idx[1]+5)
            rad_data_subset = aviris_rad[:, idx0_start:idx0_end, idx1_start:idx1_end]
            
            wvl_nancheck = 550  # nm
            wvl_nancheck_idx = np.argmin(np.abs(aviris_rad_wvl - wvl_nancheck))
            wvl_nancheck_mask = ~np.isnan(aviris_rad_wvl[wvl_nancheck_idx])
            
            if np.all(np.isnan(rad_data_subset)):
                while np.all(np.isnan(rad_data_subset)):
                    print("Closest pixel rad spectrum is all NaN, finding closer pixels and averaging...")
                    valid_lon, valid_lat = aviris_rad_lon[wvl_nancheck_mask], aviris_rad_lat[wvl_nancheck_mask]
                    valid_aviris_rad_data = aviris_rad[:, wvl_nancheck_mask]
                    valid_dist = np.sqrt((valid_lon - lon_mean)**2 + (valid_lat - lat_mean)**2)
                    min_idx_list = np.unravel_index(np.argsort(valid_dist, axis=None)[:25], valid_dist.shape)
                    # min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                    print("New min_idx:", min_idx_list)
                    print("New closest AVIRIS pixel lon, lat:", valid_lon[min_idx_list][0], valid_lat[min_idx_list][0])
                    rad_data_subset = valid_aviris_rad_data[:, min_idx_list]
                print("New rad_data_subset shape:", rad_data_subset.shape)
                aviris_rad_spectrum = np.nanmean(rad_data_subset, axis=(1, 2))
                print("New aviris_rad_spectrum shape:", aviris_rad_spectrum.shape)
                aviris_rad_spectrum_unc = np.nanstd(rad_data_subset, axis=(1, 2))
                aviris_closest = False
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.pcolormesh(aviris_rad_lon, aviris_rad_lat, rad_550, cmap='jet', shading='auto')
                plt.colorbar(im, ax=ax, label='Radiance at 550 nm')
                ax.scatter(valid_lon[min_idx_list[0][0]], valid_lat[min_idx_list[0][0]], facecolors='none', edgecolor='black', s=100, label='Closest Pixel')
                ax.scatter(lon_mean, lat_mean, color='red', marker='x', s=100, label='Flight Leg Center')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f'AVIRIS L1B Radiance at {wvl_plot} nm')
                ax.legend(fontsize=12)
                fig.tight_layout()
                fig.savefig('fig/%s/%s_%s_aviris_rad_550nm_leg_%d.png' % (date_s, date_s, case_tag, j), bbox_inches='tight', dpi=150)
                
                aviris_nearest_lon_all[j] = valid_lon[min_idx_list[0][0]]
                aviris_nearest_lat_all[j] = valid_lat[min_idx_list[0][0]]
            else:
                aviris_rad_spectrum = np.nanmean(rad_data_subset, axis=(1,2))
                aviris_rad_spectrum_unc = np.nanstd(rad_data_subset, axis=(1,2))
                aviris_closest = True
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.contourf(aviris_rad_lon, aviris_rad_lat, rad_550, cmap='jet',)# shading='auto')
                plt.colorbar(im, ax=ax, label='Radiance at 550 nm')
                ax.scatter(aviris_rad_lon[min_rad_idx], aviris_rad_lat[min_rad_idx], facecolors='none', edgecolor='black', s=100, label='Closest Pixel')
                ax.scatter(lon_mean, lat_mean, color='red', marker='x', s=100, label='Flight Leg Center')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f'AVIRIS Radiance at {wvl_plot} nm')
                ax.legend(fontsize=12)
                fig.tight_layout()
                fig.savefig('fig/%s/%s_%s_aviris_rad_550nm_leg_%d.png' % (date_s, date_s, case_tag, j), bbox_inches='tight', dpi=150)
                
                aviris_nearest_lon_all[j] = aviris_rad_lon[min_rad_idx]
                aviris_nearest_lat_all[j] = aviris_rad_lat[min_rad_idx]
            
            aviris_rad_all[j, :] = aviris_rad_spectrum
            aviris_rad_all_unc[j, :] = aviris_rad_spectrum_unc
            
        
        


    # Create a ScalarMappable
    data_min, data_max = np.log(alt_avg_all/alt_avg_all.max()).min(), np.log(alt_avg_all/alt_avg_all.max()).max()
    lon_lat_sqrt = np.sqrt(lon_avg_all**2 + lat_avg_all**2)
    data_min, data_max = lon_lat_sqrt.min(), lon_lat_sqrt.max()
    norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
    cmap = cm.jet # Or any other built-in colormap like cm.viridis
    s_m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_series = s_m.to_rgba(np.log(alt_avg_all/alt_avg_all.max()))    
    color_series = s_m.to_rgba(lon_lat_sqrt)
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(aviris_rad_lon, aviris_rad_lat, rad_550, cmap='jet', shading='auto')
    plt.colorbar(im, ax=ax, label='Radiance at 550 nm')
    ax.scatter(aviris_nearest_lon_all, aviris_nearest_lat_all, facecolors='none', edgecolor='black', s=100, label='Closest Pixel')
    ax.scatter(lon_avg_all, lat_avg_all, c=color_series, marker='x', s=100, label='Flight Leg Center')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'AVIRIS L1B Radiance at {wvl_plot} nm')
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_aviris_rad_550nm_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    toa_mean = toa_mean_all[0, :]
    for alt_ind in range(len(tmhr_ranges_select)):
        ax.plot(flux_wvl, ssfr_fdn_mean_all[alt_ind, :]/toa_mean, color=color_series[alt_ind], 
                 label='Z=%.2fkm, lon:%.2f to %.2f' % (alt_avg_all[alt_ind], lon_min_all[alt_ind], lon_max_all[alt_ind]))    

        ax.fill_between(flux_wvl,
                        (ssfr_fdn_mean_all[alt_ind, :]-ssfr_fdn_std_all[alt_ind, :])/toa_mean,
                        (ssfr_fdn_mean_all[alt_ind, :]+ssfr_fdn_std_all[alt_ind, :])/toa_mean, color='bisque', alpha=0.75)
        
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Downward flux / TOA ratio', fontsize=12)
    
    ax.set_xlim(360, 2000)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)
    ax.set_ylim(0.75, 1.15)

    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f'{date_s} Downward flux / TOA ratio', fontsize=10)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_toa_dnflux_toa_ratio.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), sharex=True, sharey='row')
    ax1.plot(flux_wvl, toa_mean_all[0, :], '--', color='gray', label='TOA')
        
    for alt_ind in range(len(tmhr_ranges_select)):
        ax1.plot(flux_wvl, ssfr_fdn_mean_all[alt_ind, :], color=color_series[alt_ind], 
                 label='Z=%.2fkm, lon:%.2f to %.2f' % (alt_avg_all[alt_ind], lon_min_all[alt_ind], lon_max_all[alt_ind]))
        ax2.plot(flux_wvl, ssfr_fup_mean_all[alt_ind, :], color=color_series[alt_ind], 
                 label='Z=%.2fkm, lon:%.2f to %.2f' % (alt_avg_all[alt_ind], lon_min_all[alt_ind], lon_max_all[alt_ind]))     
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax1.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    
    for ax in [ax1, ax2]:
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
    
    ax2.legend(fontsize=10,)
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_obs_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    
    wvl_list = [550, 650, 760, 940, 1050, 1250, 1600]
    wvl_idx_list = [np.argmin(np.abs(flux_wvl - wvl)) for wvl in wvl_list]
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'orange']
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.5))
    for ind in range(len(tmhr_ranges_select)):
        ax1.plot(time_all[ind], fdn_550_all[ind], '-', color='b', label='550 nm downward' if ind==0 else None)
        ax1.plot(time_all[ind], fup_550_all[ind], '-', color='r', label='550 nm upward' if ind==0 else None)
        ax2.plot(time_all[ind], fdn_1600_all[ind], '-', color='b', label='1600 nm downward' if ind==0 else None)
        ax2.plot(time_all[ind], fup_1600_all[ind], '-', color='r', label='1600 nm upward' if ind==0 else None)
    ax1.set_xlabel('Time (UTC)', fontsize=14)
    ax1.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax1.set_title('SSFR 550 nm Flux', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=12)
    ax2.set_xlabel('Time (UTC)', fontsize=14)
    ax2.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_title('SSFR 1600 nm Flux', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=12)
    fig.suptitle(f'SSFR Flux Time Series {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_time_series_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
        
    if rsp_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax2 = ax.twinx()
        for alt_ind in range(len(tmhr_ranges_select)):
            ax.plot(flux_wvl, ssrr_ref[alt_ind, :], '-', color=color_series[alt_ind], 
                    label='Z=%.2fkm, lon:%.2f to %.2f' % (alt_avg_all[alt_ind], lon_min_all[alt_ind], lon_max_all[alt_ind]))
            ax.scatter(rsp_wvl, rsp_ref_mean_all[alt_ind, :], marker='o', color=color_series[alt_ind], edgecolors='k', s=50, alpha=0.7)
                
            ax2.plot(flux_wvl, ssfr_fup_ref[alt_ind, :], '--', color=color_series[alt_ind], alpha=0.7)
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('SSRR radiance x $\pi$ / TOA', fontsize=14)
        ax2.set_ylabel('SSFR Upward flux / TOA', fontsize=14)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.16, 0.5))
        # plt.grid(True)
        ax.set_ylim(-0.05, 1.2)
        ax2.set_ylim(-0.05, 1.2)
        ax.tick_params(labelsize=12)
        # ax.set_title(f'Simulated - Observed Downward Flux Difference Percentage\nalb=SSFR up/down flux ratio', fontsize=13)
        fig.suptitle(f'Upward Radiance and Flux Ref Comparison {date_s}', fontsize=16, y=0.98)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssfr_ssrr_ref_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
        
        for alt_ind in range(len(tmhr_ranges_select)):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax2 = ax.twinx()
            l1 = ax.plot(flux_wvl, ssrr_ref[alt_ind, :], '-', color=color_series[alt_ind], label='SSRR')
            
            l2 = ax.scatter(rsp_wvl, rsp_ref_mean_all[alt_ind, :], marker='o', color=color_series[alt_ind], edgecolors='k', s=50, alpha=0.7, label='RSP')
            l5 = ax2.plot(flux_wvl, ssfr_fup_ref[alt_ind, :], '--', color=color_series[alt_ind], alpha=0.7, label='SSFR')
            
            ll = l1 + [l2] + l5
            labs = [l.get_label() for l in ll]
            ax.legend(ll, labs, fontsize=10, loc='upper right', ) 
            ax.set_xlabel('Wavelength (nm)', fontsize=14)
            ax.set_ylabel('SSRR radiance x $\pi$ / TOA', fontsize=14)
            ax2.set_ylabel('SSFR Upward flux / TOA', fontsize=14)
            ax.set_ylim(-0.05, 1.2)
            ax2.set_ylim(-0.05, 1.2)
            ax.tick_params(labelsize=12)
            fig.suptitle(f'Upward Radiance and Flux Ref Comparison {date_s}\nZ={alt_avg_all[alt_ind]:.2f}km, lon: {lon_min_all[alt_ind]:.3f}-{lon_max_all[alt_ind]:.3f}', 
                        fontsize=16, y=0.98)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_ssfr_ssrr_ref_comparison_Z%.2fkm.png' % (date_s, date_s, case_tag, alt_avg_all[alt_ind]), bbox_inches='tight', dpi=150)
            plt.close('all')
    
   
    
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for alt_ind in range(len(tmhr_ranges_select)):
            # plot ssrr
            ax.tick_params(labelsize=12)
            # ax.errorbar(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
            #             yerr=ssrr_rad_up_std_all[alt_ind, :], color=color_series[alt_ind], markersize=4, label='SSRR Z=%.2fkm' % (alt_avg_all[alt_ind]))
            ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
                    color=color_series[alt_ind], label='Z=%.2fkm' % (alt_avg_all[alt_ind]))
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        # ax.set_ylim(-0.05, 1.05)
        fig.suptitle(f'SSRR Upward Radiance Comparison {date_s}', fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssrr_rad_up_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
        plt.close('all')
        
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax2 = ax.twinx()
        for alt_ind in range(len(tmhr_ranges_select)):
            # plot ssrr
            ax.tick_params(labelsize=12)
            ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
                    color=color_series[alt_ind], label='SSRR Z=%.2fkm' % (alt_avg_all[alt_ind]))

            ax2.plot(flux_wvl, ssfr_fup_mean_all[alt_ind, :],
                    linestyle='--', label='SSRR Ref Z=%.2fkm' % (alt_avg_all[alt_ind]))
            
            for rsp_wvl_ind in range(len(rsp_wvl)):
                rsp_wvl_val = rsp_wvl[rsp_wvl_ind]
                # find the closest wavelength in flux_wvl
                rsp_rad_mean = rsp_rad_mean_all[alt_ind, rsp_wvl_ind]
                rsp_rad_std = rsp_rad_std_all[alt_ind, rsp_wvl_ind]
                ax.errorbar(rsp_wvl_val, rsp_rad_mean, yerr=rsp_rad_std, fmt='s', color=color_series[alt_ind], markersize=6,)# label='SSRR RSP Z=%.2fkm' % (alt_avg_all[alt_ind]))
 
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
        ax2.set_ylabel('SSFR Ref Upward Flux (W/m$^2$/nm)', fontsize=14)
        ax.legend(fontsize=10, loc='upper right')
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        # ax.set_ylim(-0.05, 1.05)
        fig.suptitle(f'SSRR Upward Radiance Comparison {date_s}', fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssrr_ssfr_rad_up_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
        plt.close('all')
        
    if aviris_file is not None:
        ssfr_toa_data = pd.read_csv('arcsix_ssfr_solar_flux.dat', delim_whitespace=True, header=2,
                                    names=['wvl', 'flux'])
        ssfr_toa_wvl = ssfr_toa_data['wvl'].values
        ssfr_toa_flux = ssfr_toa_data['flux'].values
        # print("ssfr_toa_flux:", ssfr_toa_flux)
        ssfr_toa_flux /= 1000  # convert to W/m2/nm
        toa_interp = interp1d(ssfr_toa_wvl, ssfr_toa_flux, kind='linear', bounds_error=False, fill_value='extrapolate')
        rsp_sd_mean = np.nanmean(rsp_sd_mean_all, axis=0)
        rsp_mu0_mean = np.nanmean(rsp_mu0_mean_all, axis=0)
        print("aviris_rad_wvl:", aviris_rad_wvl)
        toa_at_aviris_wvl = toa_interp(aviris_rad_wvl)
        aviris_rad_to_ref = aviris_rad_spectrum * np.pi / toa_at_aviris_wvl / np.cos(rsp_mu0_mean) * rsp_sd_mean**2
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for alt_ind in range(len(tmhr_ranges_select)):
        # plot ssrr
        ax.tick_params(labelsize=12)
        # ax.errorbar(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
        #             yerr=ssrr_rad_up_std_all[alt_ind, :], color=color_series[alt_ind], markersize=4, label='SSRR Z=%.2fkm' % (alt_avg_all[alt_ind]))
        ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
                 color=color_series[alt_ind], label='Z=%.2fkm' % (alt_avg_all[alt_ind]))
        if aviris_rad_file is not None:
            # ax.errorbar(aviris_rad_wvl, aviris_rad_spectrum, yerr=aviris_rad_spectrum_unc, 
            #             fmt='o', color='m', markersize=2, label='AVIRIS', alpha=0.7)
            ax.plot(aviris_rad_wvl, aviris_rad_all[alt_ind, :], '-', color='m', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    fig.suptitle(f'SSRR Upward Radiance Comparison {date_s}', fontsize=13)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssrr_rad_up_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax2 = ax.twinx()
    for alt_ind in range(len(tmhr_ranges_select)):
        # plot ssrr
        ax.tick_params(labelsize=12)
        ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
                 color=color_series[alt_ind], label='SSRR Z=%.2fkm' % (alt_avg_all[alt_ind]))

    
        ax2.plot(flux_wvl, ssfr_fup_mean_all[alt_ind, :],
                 linestyle='--', label='SSRR Ref Z=%.2fkm' % (alt_avg_all[alt_ind]))
        
        for rsp_wvl_ind in range(len(rsp_wvl)):
            rsp_wvl_val = rsp_wvl[rsp_wvl_ind]
            # find the closest wavelength in flux_wvl
            rsp_rad_mean = rsp_rad_mean_all[alt_ind, rsp_wvl_ind]
            rsp_rad_std = rsp_rad_std_all[alt_ind, rsp_wvl_ind]
            ax.errorbar(rsp_wvl_val, rsp_rad_mean, yerr=rsp_rad_std, fmt='s', color=color_series[alt_ind], markersize=6,)# label='SSRR RSP Z=%.2fkm' % (alt_avg_all[alt_ind]))

        if aviris_rad_file is not None:
            # ax.errorbar(aviris_rad_wvl, aviris_rad_spectrum, yerr=aviris_rad_spectrum_unc, 
            #             fmt='o', color='m', markersize=2, label='AVIRIS', alpha=0.7)
            ax.plot(aviris_rad_wvl, aviris_rad_all[alt_ind], '-', color='m', alpha=0.7)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
    ax2.set_ylabel('SSFR Ref Upward Flux (W/m$^2$/nm)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    fig.suptitle(f'SSRR Upward Radiance Comparison {date_s}', fontsize=13)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssrr_ssfr_rad_up_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    for j in range(len(tmhr_ranges_select)):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if aviris_rad_file is not None:
            # ax.errorbar(aviris_rad_wvl, aviris_rad_spectrum, yerr=aviris_rad_spectrum_unc, 
            #             fmt='o', color='m', markersize=2, label='AVIRIS', alpha=0.7)
            l1 = ax.plot(aviris_rad_wvl, aviris_rad_all[j, :], '-', color='m', alpha=0.7, label='AVIRIS')
            
        ax2 = ax.twinx()
        # plot ssrr
        ax.tick_params(labelsize=12)
        l2 = ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[j, :],
                color=color_series[j], label='SSRR')

    
        l3 = ax2.plot(flux_wvl, ssfr_fup_mean_all[j, :],
                linestyle='--', label='SSFR')
        
        for rsp_wvl_ind in range(len(rsp_wvl)):
            rsp_wvl_val = rsp_wvl[rsp_wvl_ind]
            # find the closest wavelength in flux_wvl
            rsp_rad_mean = rsp_rad_mean_all[j, rsp_wvl_ind]
            rsp_rad_std = rsp_rad_std_all[j, rsp_wvl_ind]
            if rsp_wvl_ind == 0:
                l4 = ax.errorbar(rsp_wvl_val, rsp_rad_mean, yerr=rsp_rad_std, fmt='s', color=color_series[j], markersize=6, label='RSP')
            else:
                ax.errorbar(rsp_wvl_val, rsp_rad_mean, yerr=rsp_rad_std, fmt='s', color=color_series[j], markersize=6,) 
        
        # collect handles+labels from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        
        handles = h1 + h2
        labels  = l1 + l2
        ax.legend(handles, labels, fontsize=10, loc='upper right')
        
        
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
        ax2.set_ylabel('SSFR Ref Upward Flux (W/m$^2$/nm)', fontsize=14)
        
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        # ax.set_ylim(-0.05, 1.05)
        fig.suptitle(f'SSRR Upward Radiance Comparison {date_s}\nZ={alt_avg_all[j]:.2f}, lat={lat_avg_all[j]:.2f}$^o$N',
                     fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssrr_ssfr_rad_up_toa_ratio_comparison_leg_%d.png' % (date_s, date_s, case_tag, j), bbox_inches='tight', dpi=150)
        plt.close('all')
    
    print('stop here!')
    
    if aviris_rad_file is not None:
        aviris_rad_at_rsp_wvl = np.zeros((len(tmhr_ranges_select), len(rsp_wvl)))
        aviris_rad_at_ssrr_wvl = np.zeros((len(tmhr_ranges_select), len(ssrr_rad_up_wvl)))
        for j in range(len(tmhr_ranges_select)):
            aviris_rad_interp = interp1d(aviris_rad_wvl, aviris_rad_all[j, :], kind='linear', bounds_error=False, fill_value='extrapolate')
            aviris_rad_at_rsp_wvl[j, :] = aviris_rad_interp(rsp_wvl)
            aviris_rad_at_ssrr_wvl[j, :] = aviris_rad_interp(ssrr_rad_up_wvl)
    
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(aviris_rad_wvl, aviris_rad_all.mean(axis=0), yerr=aviris_rad_spectrum_unc.mean(axis=0), 
                    fmt='o', color='m', markersize=2, label='AVIRIS Radiance', alpha=0.7)
        l1 = ax.plot(aviris_rad_wvl, aviris_rad_all.mean(axis=0), '-', color='m', label='AVIRIS Radiance', alpha=0.7)
        ax_t = ax.twinx()
        if aviris_file is not None:
            l2 = ax_t.plot(aviris_reflectance_wvl, aviris_reflectance_spectrum, c='b', label='AVIRIS Reflectance', alpha=0.7)
            ax_t.fill_between(aviris_reflectance_wvl, aviris_reflectance_spectrum-aviris_reflectance_spectrum_unc, aviris_reflectance_spectrum+aviris_reflectance_spectrum_unc, color='b', alpha=0.3)
        ll = l1+l2 if aviris_file is not None else l1
        labs = [l.get_label() for l in ll]
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
        ax_t.set_ylabel('Reflectance', fontsize=14)
        ax.legend(ll, labs, fontsize=10,)
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        ax_t.tick_params(labelsize=12)
        # ax.set_ylim(-0.05, 1.05)
        fig.suptitle(f'SSRR Upward Radiance & Reflectance Comparison {date_s}', fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_aviris_rad_reflectance_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for j in range(len(tmhr_ranges_select)):
            ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[j, :]/aviris_rad_at_ssrr_wvl[j, :], '-', color=color_series[j], label='Z=%.2fkm' % (alt_avg_all[j]))
            ax.scatter(rsp_wvl, rsp_rad_mean_all[j, :]/aviris_rad_at_rsp_wvl[j, :], marker='s', color=color_series[j], edgecolors='k', s=50, alpha=0.7)
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance Ratio', fontsize=14)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        ax.set_ylim(00.7, 1.3)
        ax.set_xlim(350, 2200)
        ax.hlines(1.0, 350, 2200, color='gray', linestyle='--')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.suptitle(f'SSRR & RSP Upward Radiance to AVIRIS Ratio Comparison {date_s}', fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssrr_rsp_aviris_rad_ratio_comparison_all.png' % (date_s, date_s, case_tag,), bbox_inches='tight', dpi=150)
        
        for j in range(len(tmhr_ranges_select)):

                
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[j, :]/aviris_rad_at_ssrr_wvl[j, :], '-', color=color_series[j], label='SSRR/AVIRIS')
            ax.scatter(rsp_wvl, rsp_rad_mean_all[j, :]/aviris_rad_at_rsp_wvl[j, :], marker='s', color=color_series[j], edgecolors='k', s=50, alpha=0.7, label='RSP/AVIRIS')
            ax.set_xlabel('Wavelength (nm)', fontsize=14)
            ax.set_ylabel('Radiance Ratio', fontsize=14)
            ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
            # plt.grid(True)
            ax.tick_params(labelsize=12)
            ax.set_ylim(0.7, 1.3)
            ax.set_xlim(350, 2200)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.hlines(1.0, 350, 2200, color='gray', linestyle='--') 
            fig.suptitle(f'SSRR & RSP Upward Radiance to AVIRIS Ratio Comparison {date_s}\nZ={alt_avg_all[j]:.2f}km, lon={lon_avg_all[j]:.2f}, lat={lat_avg_all[j]:.2f}', fontsize=13)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_ssrr_rsp_aviris_rad_ratio_comparison_leg_%d.png' % (date_s, date_s, case_tag, j), bbox_inches='tight', dpi=150)
            
    if aviris_rad_file is not None:
        aviris_rad_at_rsp_wvl = np.zeros((len(tmhr_ranges_select), len(rsp_wvl)))
        aviris_rad_at_ssrr_wvl = np.zeros((len(tmhr_ranges_select), len(ssrr_rad_up_wvl)))
        ssrr_rad_at_rsp_wvl = np.zeros((len(tmhr_ranges_select), len(rsp_wvl)))
        rsp_wvl_fwhm = np.array([27, 20, 20, 20, 20, 20, 60, 90, 130])
        for j in range(len(tmhr_ranges_select)):
            aviris_rad_interp = interp1d(aviris_rad_wvl, aviris_rad_all[j, :], kind='linear', bounds_error=False, fill_value='extrapolate')
            aviris_rad_at_rsp_wvl[j, :] = aviris_rad_interp(rsp_wvl)
            aviris_rad_at_ssrr_wvl[j, :] = aviris_rad_interp(ssrr_rad_up_wvl)
            ssrr_rad_interp = interp1d(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[j, :], kind='linear', bounds_error=False, fill_value=np.nan)
            for k in range(len(rsp_wvl)):
                rsp_wvl_ = rsp_wvl[k]
                rsl_wvl_fwhm_ = rsp_wvl_fwhm[k]
                rsp_wvl_range = np.arange(rsp_wvl_ - rsl_wvl_fwhm_, rsp_wvl_ + rsl_wvl_fwhm_+1, 1)
                ssrr_rad_at_rsp_wvl_range = ssrr_rad_interp(rsp_wvl_range)
                # average ssrr rad over the fwhm range with gaussian
                sigma = rsl_wvl_fwhm_ / (2 * np.sqrt(2 * np.log(2)))
                weights = np.exp(-0.5 * ((rsp_wvl_range - rsp_wvl_) / sigma) ** 2)
                weights /= np.sum(weights)
                ssrr_rad_at_rsp_wvl[j, k] = np.nansum(ssrr_rad_at_rsp_wvl_range * weights)

        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for j in range(len(tmhr_ranges_select)):
            ax.scatter(rsp_wvl, rsp_rad_mean_all[j, :]/ssrr_rad_at_rsp_wvl[j, :], marker='s', color=color_series[j], edgecolors='k', s=50, alpha=0.7, label='Z=%.2fkm, lat=%.2f $^o$N' % (alt_avg_all[j], lat_avg_all[j]))
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance Ratio', fontsize=14)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        ax.set_ylim(0.8, 1.2)
        ax.set_xlim(350, 1000)
        ax.hlines(1.0, 350, 2200, color='gray', linestyle='--')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.suptitle(f'RSP to SSRR Upward Radiance Ratio Comparison {date_s}', fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_rsp_ssrr_rad_ratio_comparison_all.png' % (date_s, date_s, case_tag,), bbox_inches='tight', dpi=150)

    
    
    
def atm_corr_spiral_plot(date=datetime.datetime(2024, 5, 31),
                     tmhr_ranges_select=[[14.10, 14.27]],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                            ):

    
    
    log = logging.getLogger("atm corr spiral plot")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    doy_s = date.timetuple().tm_yday
    print(f"Processing date: {date_s}, DOY: {doy_s}")

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    
    modis_alb_dir = f'{_fdir_general_}/modis_albedo'
    # list all modis albedo files
    modis_alb_files = sorted(glob.glob(os.path.join(modis_alb_dir, f'M*.nc')))
    for fname in modis_alb_files:
        print("Checking modis file:", os.path.basename(fname).split('.')[1])
        if str(doy_s) in os.path.basename(fname).split('.')[1]:
            modis_alb_file = fname
            break
    else:
        modis_alb_file = None
    
    aviris_rad_file = None
    aviris_file = None
    aviris_dir = f'{_fdir_general_}/aviris_ng'
    aviris_files = sorted(glob.glob(os.path.join(aviris_dir, f'ang*.nc')))
    for fname in aviris_files:
        if date_s in os.path.basename(fname):
            if 'RFL' in os.path.basename(fname):
                aviris_file = fname
            if 'RDN' in os.path.basename(fname):
                aviris_rad_file = fname
            # break
    if aviris_file is None and aviris_rad_file is None:
        aviris_closest = False
    
        
    if modis_alb_file is not None:
        with Dataset(modis_alb_file, 'r') as ds:
            modis_lon = ds.variables['Longitude'][:]
            modis_lat = ds.variables['Latitude'][:]
            modis_bands = ds.variables['Bands'][:]
            modis_sur_alb = ds.variables['Albedo_1km'][:]
            
    print("aviris_file:", aviris_file)
    print("aviris_rad_file:", aviris_rad_file)
    print("modis_alb_file:", modis_alb_file)
    
    
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    for i in range(len(tmhr_ranges_select)):
        fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
        with open(fname_pkl, 'rb') as f:
            cld_leg = pickle.load(f)
        time = cld_leg['time']
        time_start = tmhr_ranges_select[i][0]
        time_end = tmhr_ranges_select[i][1]
        
 
        alt = cld_leg['alt']
        lon_all = cld_leg['lon']
        lat_all = cld_leg['lat']
        heading = cld_leg['heading']
        hsr1_tot = cld_leg['hsr1_tot']
        hsr1_dif = cld_leg['hsr1_dif']
        hsr1_wvl = cld_leg['hsr1_wvl']
        rsp_lon = cld_leg['rsp_lon']
        rsp_lat = cld_leg['rsp_lat']
        rsp_rad = cld_leg['rsp_rad']
        rsp_rad_norm = cld_leg['rsp_rad_norm']
        rsp_ref = cld_leg['rsp_ref']
        rsp_wvl = cld_leg['rsp_wvl']
        rsp_mu0 = cld_leg['rsp_mu0']
        rsp_sd = cld_leg['rsp_sd']
        sza = cld_leg['sza']
        saa = cld_leg['saa']
        ssfr_wvl_zen = cld_leg['ssfr_zen_wvl']
        ssfr_wvl_nad = cld_leg['ssfr_nad_wvl']
        ssfr_flux_zen = cld_leg['ssfr_zen']
        ssfr_flux_nad = cld_leg['ssfr_nad']
        ssrr_wvl_zen = cld_leg['ssrr_zen_wvl']
        ssrr_rad_zen = cld_leg['ssrr_zen_rad']
        ssrr_wvl_nad = cld_leg['ssrr_nad_wvl']
        ssrr_rad_nad = cld_leg['ssrr_nad_rad']
        toa_flux = cld_leg['ssfr_toa']
        
        alt_avg = np.nanmean(alt)  # in km
        lon_avg = np.nanmean(lon_all)
        lat_avg = np.nanmean(lat_all)
        
        

        
        if i == 0:
            time_all = []
            fdn_550_all = []
            fup_550_all = []
            fdn_1600_all = []
            fup_1600_all = []
            
            lon_avg_all = np.zeros(len(tmhr_ranges_select))
            lat_avg_all = np.zeros(len(tmhr_ranges_select))
            lon_min_all = np.zeros(len(tmhr_ranges_select))
            lon_max_all = np.zeros(len(tmhr_ranges_select))
            lat_min_all = np.zeros(len(tmhr_ranges_select))
            lat_max_all = np.zeros(len(tmhr_ranges_select))
            alt_avg_all = np.zeros(len(tmhr_ranges_select))

            
            flux_wvl = ssfr_wvl_zen
            ssfr_fup_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fup_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            toa_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_up_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_dn_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_up_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_dn_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            
            rsp_rad_mean_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_rad_std_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_ref_mean_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_ref_std_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_mu0_mean_all = np.zeros(len(tmhr_ranges_select))
            rsp_sd_mean_all = np.zeros(len(tmhr_ranges_select))

            

        time_all.append(time)
        ssfr_wvl = cld_leg['ssfr_zen_wvl']
        ssfr_550_ind = np.argmin(np.abs(ssfr_wvl - 550))
        ssfr_1600_ind = np.argmin(np.abs(ssfr_wvl - 1600))
        fdn_550_all.append(cld_leg['ssfr_zen'][:, ssfr_550_ind])
        fup_550_all.append(cld_leg['ssfr_nad'][:, ssfr_550_ind])
        fdn_1600_all.append(cld_leg['ssfr_zen'][:, ssfr_1600_ind])
        fup_1600_all.append(cld_leg['ssfr_nad'][:, ssfr_1600_ind])
        
        
        lon_avg_all[i] = lon_avg.copy()
        lat_avg_all[i] = lat_avg.copy()
        lon_min_all[i] = lon_all.min()
        lon_max_all[i] = lon_all.max()
        lat_min_all[i] = lat_all.min()
        lat_max_all[i] = lat_all.max()
        alt_avg_all[i] = alt_avg.copy()
        
        
        ssfr_fup_mean_all[i, :] = np.nanmean(ssfr_flux_nad, axis=0)
        ssfr_fdn_mean_all[i, :] = np.nanmean(ssfr_flux_zen, axis=0)
        ssfr_fup_std_all[i, :] = np.nanstd(ssfr_flux_nad, axis=0)
        ssfr_fdn_std_all[i, :] = np.nanstd(ssfr_flux_zen, axis=0)
        toa_mean_all[i, :] = np.nanmean(toa_flux, axis=0)
        
        ssrr_rad_up_mean_all[i, :] = np.nanmean(ssrr_rad_nad, axis=0)
        ssrr_rad_dn_mean_all[i, :] = np.nanmean(ssrr_rad_zen, axis=0)
        ssrr_rad_up_std_all[i, :] = np.nanstd(ssrr_rad_nad, axis=0)
        ssrr_rad_dn_std_all[i, :] = np.nanstd(ssrr_rad_zen, axis=0)
        ssrr_rad_up_wvl = ssrr_wvl_nad
        ssrr_rad_dn_wvl = ssrr_wvl_zen
        
        rsp_wvl = cld_leg['rsp_wvl']
        rsp_mu0 = cld_leg['rsp_mu0']
        rsp_sd = cld_leg['rsp_sd']
        print("cld_leg['rsp_rad'] shape:", cld_leg['rsp_rad'].shape)
        
        if cld_leg['rsp_rad'] is not None:
            print("cld_leg['rsp_rad'] shape:", cld_leg['rsp_rad'].shape)
            
            rsp_rad_mean_all[i, :] = np.nanmean(cld_leg['rsp_rad'], axis=0)/1000  # in W m-2 sr-1 nm-1
            rsp_rad_std_all[i, :] = np.nanstd(cld_leg['rsp_rad'], axis=0)/1000  # in W m-2 sr-1 nm-1
            
            rsp_ref_mean_all[i, :] = np.nanmean(cld_leg['rsp_ref'], axis=0)
            rsp_ref_std_all[i, :] = np.nanstd(cld_leg['rsp_ref'], axis=0)
            
            rsp_mu0_mean_all[i] = np.nanmean(cld_leg['rsp_mu0'])
            rsp_sd_mean_all[i] = np.nanmean(cld_leg['rsp_sd'])
        
        # plt.close('all')
        # plt.plot(rsp_wvl, rsp_rad_mean_all[i, :], '.-', label=f'Leg {i} mean')
        # plt.plot(cld_leg['ssrr_nad_wvl'], np.nanmean(cld_leg['ssrr_nad_rad'], axis=0), 'x--', label=f'Leg {i} SSFR nadir')
        # plt.xlabel('Wavelength (nm)', fontsize=14)
        # plt.ylabel('Radiance (mW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=14)
        # plt.show()
        
        # plt.close('all')
        # plt.scatter(cld_leg['rsp_lon'], cld_leg['rsp_lat'], c='b', s=20, marker='o')
        # plt.scatter(cld_leg['lon'], cld_leg['lat'], c='r', s=20, marker='x')
        # plt.show()
        # sys.exit()
        
        
        print(f"date_s: {date_s}, time: {time_start:.2f}-{time_end:.2f}, alt_avg: {alt_avg:.2f} km")
    

    ssrr_ref = ssrr_rad_up_mean_all * np.pi / toa_mean_all / rsp_mu0_mean_all[:, np.newaxis] * (rsp_sd_mean_all[:, np.newaxis]**2)
    ssfr_fup_ref = ssfr_fup_mean_all / toa_mean_all /  rsp_mu0_mean_all[:, np.newaxis] * (rsp_sd_mean_all[:, np.newaxis]**2)
       
    print("rsp_wvl:", rsp_wvl)
    
    # print("ssfr_fup_ref shape:", ssfr_fup_ref.shape)
    # sys.exit()
        
    print("lon avg all mean:", lon_avg_all.mean())
    print("lat avg all mean:", lat_avg_all.mean())
        
    

    if aviris_file is not None:
        # 1) Open your NetCDF
        ds = Dataset(aviris_file)
        easting  = ds.variables["easting"][:]   # shape (1665,)
        northing = ds.variables["northing"][:]  # shape (1207,)
        
        # 2) Read GeoTransform
        gt = list(map(float, ds['transverse_mercator'].getncattr("GeoTransform").split()))
        crs_info_ind = ds['transverse_mercator'].getncattr("crs_wkt").find('AUTHORITY["EPSG","326')
        print("crs_info_ind:", crs_info_ind)
        # AUTHORITY["EPSG","32622"]
        epsg_code = int(ds['transverse_mercator'].getncattr("crs_wkt")[crs_info_ind+18:crs_info_ind+23])
        print("AVIRIS EPSG code:", epsg_code)
        GT0, GT1, GT2, GT3, GT4, GT5 = gt

        # 3) Prepare transformer
        transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)

        # Example: compute lon/lat for point at (row, col)
        row, col = 0, 0

        # Projected coords
        x = GT0 + col * GT1 + row * GT2
        y = GT3 + col * GT4 + row * GT5

        # Geographic coords
        lon, lat = transformer.transform(x, y)

        print(f"Row {row}, Col {col} → x={x:.2f} m, y={y:.2f} m → lon={lon:.6f}, lat={lat:.6f}")
        
        

        # 2) Build a 2D meshgrid of coordinates
        #    X will have shape (1207, 1665) where each row is a copy of easting
        #    Y will have shape (1207, 1665) where each column is a copy of northing
        X, Y = np.meshgrid(easting, northing)


        # 4) Transform the entire grid in one go
        #    lon, lat will each be arrays of shape (1207, 1665)
        lon, lat = transformer.transform(X, Y)
        
        aviris_reflectance_wvl = ds.groups['reflectance'].variables['wavelength'][:]  # in nm
        aviris_reflectance_data = ds.groups['reflectance'].variables['reflectance'][:]  # shape (wvl, northing, easting)
        aviris_reflectance_data = np.where(aviris_reflectance_data<0, np.nan, aviris_reflectance_data)
        # aviris_reflectance_data[~aviris_reflectance_data>0] = np.nan
        
        wvl_nancheck = 550  # nm
        wvl_nancheck_idx = np.argmin(np.abs(aviris_reflectance_wvl - wvl_nancheck))
        wvl_nancheck_mask = ~np.isnan(aviris_reflectance_data[wvl_nancheck_idx, :, :])
        
        
        
        # find lon, lat mean index in the aviris data
        lon_mean = lon_avg_all.mean()
        lat_mean = lat_avg_all.mean()
        dist = np.sqrt((lon - lon_mean)**2 + (lat - lat_mean)**2)
        min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        print("lon_mean, lat_mean:", lon_mean, lat_mean)
        print("Closest AVIRIS pixel lon, lat:", lon[min_idx], lat[min_idx])
        print("Closest AVIRIS pixel index (northing, easting):", min_idx)
        
        
        wvl_plot = 550  # nm
        wvl_idx = np.argmin(np.abs(aviris_reflectance_wvl - wvl_plot))
        reflectance_550 = aviris_reflectance_data[wvl_idx, :, :]  # shape (northing, easting)
        
        print("aviris_reflectance_data shape:", aviris_reflectance_data.shape)
        print("min_idx[0], min_idx[1]:", min_idx[0], min_idx[1])
        idx0_start = max(0, min_idx[0]-3)
        idx0_end = min(aviris_reflectance_data.shape[1], min_idx[0]+4)
        idx1_start = max(0, min_idx[1]-3)
        idx1_end = min(aviris_reflectance_data.shape[2], min_idx[1]+4)
        ref_data_subset = aviris_reflectance_data[:, idx0_start:idx0_end, idx1_start:idx1_end]
        print("ref_data_subset shape:", ref_data_subset.shape)
        # aviris_reflectance_spectrum = np.nanmean(ref_data_subset, axis=(1,2))
        # aviris_reflectance_spectrum_unc = np.nanstd(ref_data_subset, axis=(1,2))
        
        if np.all(np.isnan(ref_data_subset)):
            while np.all(np.isnan(ref_data_subset)):
                print("Closest pixel reflectance spectrum is all NaN, finding closer pixels and averaging...")
                valid_lon, valid_lat = lon[wvl_nancheck_mask], lat[wvl_nancheck_mask]
                valid_aviris_reflectance_data = aviris_reflectance_data[:, wvl_nancheck_mask]
                valid_dist = np.sqrt((valid_lon - lon_mean)**2 + (valid_lat - lat_mean)**2)
                min_idx_list = np.unravel_index(np.argsort(valid_dist, axis=None)[:25], valid_dist.shape)
                # min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                print("New min_idx:", min_idx_list)
                print("New closest AVIRIS pixel lon, lat:", valid_lon[min_idx_list][0], valid_lat[min_idx_list][0])
                ref_data_subset = valid_aviris_reflectance_data[:, min_idx_list]
            print("New ref_data_subset shape:", ref_data_subset.shape)
            aviris_reflectance_spectrum = np.nanmean(ref_data_subset, axis=(1, 2))
            print("New aviris_reflectance_spectrum shape:", aviris_reflectance_spectrum.shape)
            aviris_reflectance_spectrum_unc = np.nanstd(ref_data_subset, axis=(1, 2))
            aviris_closest = False
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.pcolormesh(lon, lat, reflectance_550, cmap='jet', shading='auto')
            plt.colorbar(im, ax=ax, label='Reflectance at 550 nm')
            ax.scatter(valid_lon[min_idx_list[0][0]], valid_lat[min_idx_list[0][0]], facecolors='none', edgecolor='black', s=100, label='Closest Pixel')
            ax.scatter(lon_mean, lat_mean, color='red', marker='x', s=100, label='Flight Leg Center')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'AVIRIS Reflectance at {wvl_plot} nm')
            ax.legend(fontsize=12)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_aviris_reflectance_550nm.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
            plt.close('all')
            # plt.show()
        else:
            aviris_reflectance_spectrum = np.nanmean(ref_data_subset, axis=(1,2))
            aviris_reflectance_spectrum_unc = np.nanstd(ref_data_subset, axis=(1,2))
            aviris_closest = True
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.pcolormesh(lon, lat, reflectance_550, cmap='jet', shading='auto')
            plt.colorbar(im, ax=ax, label='Reflectance at 550 nm')
            ax.scatter(lon[min_idx], lat[min_idx], facecolors='none', edgecolor='black', s=100, label='Closest Pixel')
            ax.scatter(lon_mean, lat_mean, color='red', marker='x', s=100, label='Flight Leg Center')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'AVIRIS Reflectance at {wvl_plot} nm')
            ax.legend(fontsize=12)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_aviris_reflectance_550nm.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
            plt.close('all')

    
    if aviris_rad_file is not None:
        ds_rad = Dataset(aviris_rad_file)
        aviris_rad_wvl = ds_rad.groups['radiance'].variables['wavelength'][:]
        aviris_rad_wvl = np.array([float(i) for i in aviris_rad_wvl])  # in nm
        aviris_rad = ds_rad.groups['radiance'].variables['radiance'][:]  # shape (wvl, lat, lon) # in uW nm-1 cm-2 sr-1
        aviris_rad = aviris_rad * 1e-6 * 1e4  # convert to W nm-1 m-2 sr-1
        aviris_rad_lon = ds_rad.variables['lon'][:]
        aviris_rad_lat = ds_rad.variables['lat'][:]
        ds_rad.close()
        # find the closest pixel in aviris_rad to the flight leg center
        # find lon, lat mean index in the aviris data
        lon_mean = lon_avg_all.mean()
        lat_mean = lat_avg_all.mean()
        dist_rad = np.sqrt((aviris_rad_lon - lon_mean)**2 + (aviris_rad_lat - lat_mean)**2)
        min_rad_idx = np.unravel_index(np.argmin(dist_rad, axis=None), dist_rad.shape)
        
        wvl_plot = 550  # nm
        rad_wvl_idx = np.argmin(np.abs(aviris_rad_wvl - wvl_plot))
        rad_550 = aviris_rad[rad_wvl_idx, :, :]  # shape (northing, easting)
        
        idx0_start = max(0, min_rad_idx[0]-3)
        idx0_end = min(aviris_rad.shape[1], min_rad_idx[0]+4)
        idx1_start = max(0, min_rad_idx[1]-3)
        idx1_end = min(aviris_rad.shape[2], min_rad_idx[1]+4)
        rad_data_subset = aviris_rad[:, idx0_start:idx0_end, idx1_start:idx1_end]
        
        wvl_nancheck = 550  # nm
        wvl_nancheck_idx = np.argmin(np.abs(aviris_rad_wvl - wvl_nancheck))
        wvl_nancheck_mask = ~np.isnan(aviris_rad_wvl[wvl_nancheck_idx])
        
        if np.all(np.isnan(rad_data_subset)):
            while np.all(np.isnan(rad_data_subset)):
                print("Closest pixel rad spectrum is all NaN, finding closer pixels and averaging...")
                valid_lon, valid_lat = aviris_rad_lon[wvl_nancheck_mask], aviris_rad_lat[wvl_nancheck_mask]
                valid_aviris_rad_data = aviris_rad[:, wvl_nancheck_mask]
                valid_dist = np.sqrt((valid_lon - lon_mean)**2 + (valid_lat - lat_mean)**2)
                min_idx_list = np.unravel_index(np.argsort(valid_dist, axis=None)[:25], valid_dist.shape)
                # min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                print("New min_idx:", min_idx_list)
                print("New closest AVIRIS pixel lon, lat:", valid_lon[min_idx_list][0], valid_lat[min_idx_list][0])
                rad_data_subset = valid_aviris_rad_data[:, min_idx_list]
            print("New rad_data_subset shape:", rad_data_subset.shape)
            aviris_rad_spectrum = np.nanmean(rad_data_subset, axis=(1, 2))
            print("New aviris_rad_spectrum shape:", aviris_rad_spectrum.shape)
            aviris_rad_spectrum_unc = np.nanstd(rad_data_subset, axis=(1, 2))
            aviris_closest = False
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.pcolormesh(aviris_rad_lon, aviris_rad_lat, rad_550, cmap='jet', shading='auto')
            plt.colorbar(im, ax=ax, label='Radiance at 550 nm')
            ax.scatter(valid_lon[min_idx_list[0][0]], valid_lat[min_idx_list[0][0]], facecolors='none', edgecolor='black', s=100, label='Closest Pixel')
            ax.scatter(lon_mean, lat_mean, color='red', marker='x', s=100, label='Flight Leg Center')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'AVIRIS L1B Radiance at {wvl_plot} nm')
            ax.legend(fontsize=12)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_aviris_rad_550nm.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
            plt.close('all')
        else:
            aviris_rad_spectrum = np.nanmean(rad_data_subset, axis=(1,2))
            aviris_rad_spectrum_unc = np.nanstd(rad_data_subset, axis=(1,2))
            aviris_closest = True
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.contourf(aviris_rad_lon, aviris_rad_lat, rad_550, cmap='jet',)# shading='auto')
            plt.colorbar(im, ax=ax, label='Radiance at 550 nm')
            ax.scatter(aviris_rad_lon[min_rad_idx], aviris_rad_lat[min_rad_idx], facecolors='none', edgecolor='black', s=100, label='Closest Pixel')
            ax.scatter(lon_mean, lat_mean, color='red', marker='x', s=100, label='Flight Leg Center')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'AVIRIS Radiance at {wvl_plot} nm')
            ax.legend(fontsize=12)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_aviris_rad_550nm.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
            plt.close('all')
        
    
    # find the modis location closest to the flight leg center
    if modis_alb_file is not None:
        dist = np.sqrt((modis_lon - lon_avg)**2 + (modis_lat - lat_avg)**2)
        min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        modis_alb_leg = modis_sur_alb[min_idx[0], min_idx[1], :7]
        modis_bands_nm = np.array([float(i) for i in modis_bands[:7]])*1000  # in nm


    # Create a ScalarMappable
    data_min, data_max = np.log(alt_avg_all/alt_avg_all.max()).min(), np.log(alt_avg_all/alt_avg_all.max()).max()
    norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
    cmap = cm.jet # Or any other built-in colormap like cm.viridis
    s_m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_series = s_m.to_rgba(np.log(alt_avg_all/alt_avg_all.max()))
    
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), sharex=True, sharey='row')
    ax1.plot(flux_wvl, toa_mean_all[0, :], '--', color='gray', label='TOA')
    for alt_ind in range(len(tmhr_ranges_select)):
        ax1.plot(flux_wvl, ssfr_fdn_mean_all[alt_ind, :], color=color_series[alt_ind], label='Z=%.2fkm' % (alt_avg_all[alt_ind]))
        ax2.plot(flux_wvl, ssfr_fup_mean_all[alt_ind, :], color=color_series[alt_ind], label='Z=%.2fkm' % (alt_avg_all[alt_ind]))
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax1.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    for ax in [ax1, ax2]:
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.tick_params(labelsize=12)
        
    ax2.legend(fontsize=10,)
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_obs_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    wvl_list = [550, 650, 760, 940, 1050, 1250, 1600]
    wvl_idx_list = [np.argmin(np.abs(flux_wvl - wvl)) for wvl in wvl_list]
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'orange']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.5))
    for ind in range(len(tmhr_ranges_select)):
        ax1.plot(time_all[ind], fdn_550_all[ind], '-', color='b', label='550 nm downward' if ind==0 else None)
        ax1.plot(time_all[ind], fup_550_all[ind], '-', color='r', label='550 nm upward' if ind==0 else None)
        ax2.plot(time_all[ind], fdn_1600_all[ind], '-', color='b', label='1600 nm downward' if ind==0 else None)
        ax2.plot(time_all[ind], fup_1600_all[ind], '-', color='r', label='1600 nm upward' if ind==0 else None)
    ax1.set_title('SSFR 550 nm Flux', fontsize=12)
    ax2.set_title('SSFR 1600 nm Flux', fontsize=12)
    for ax in [ax1, ax2]:
        ax.set_xlabel('Time (UTC)', fontsize=14)
        ax.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=12)
    fig.suptitle(f'SSFR Flux Time Series {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_time_series_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    plt.close('all')


    
    if aviris_file is not None:
        ssfr_toa_data = pd.read_csv('arcsix_ssfr_solar_flux.dat', delim_whitespace=True, header=2,
                                    names=['wvl', 'flux'])
        ssfr_toa_wvl = ssfr_toa_data['wvl'].values
        ssfr_toa_flux = ssfr_toa_data['flux'].values
        # print("ssfr_toa_flux:", ssfr_toa_flux)
        ssfr_toa_flux /= 1000  # convert to W/m2/nm
        toa_interp = interp1d(ssfr_toa_wvl, ssfr_toa_flux, kind='linear', bounds_error=False, fill_value='extrapolate')
        rsp_sd_mean = np.nanmean(rsp_sd_mean_all, axis=0)
        rsp_mu0_mean = np.nanmean(rsp_mu0_mean_all, axis=0)
        print("aviris_rad_wvl:", aviris_rad_wvl)
        toa_at_aviris_wvl = toa_interp(aviris_rad_wvl)
        aviris_rad_to_ref = aviris_rad_spectrum * np.pi / toa_at_aviris_wvl / np.cos(rsp_mu0_mean) * rsp_sd_mean**2
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax2 = ax.twinx()
    for alt_ind in range(len(tmhr_ranges_select)):
        ax.plot(flux_wvl, ssrr_ref[alt_ind, :], '-', color=color_series[alt_ind], label='Z=%.2fkm' % (alt_avg_all[alt_ind]))
        ax.scatter(rsp_wvl, rsp_ref_mean_all[alt_ind, :], marker='o', color=color_series[alt_ind], edgecolors='k', s=50, alpha=0.7)
            
        ax2.plot(flux_wvl, ssfr_fup_ref[alt_ind, :], '--', color=color_series[alt_ind], alpha=0.7)
    if aviris_file is not None:
        ax.plot(aviris_reflectance_wvl, aviris_rad_to_ref, c='m', label='AVIRIS Rad x $\pi$ / TOA', alpha=0.7)
        ax.plot(aviris_reflectance_wvl, aviris_reflectance_spectrum, c='darkviolet', label='AVIRIS L2 Ref', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('SSRR radiance x $\pi$ / TOA', fontsize=14)
    ax2.set_ylabel('SSFR Upward flux / TOA', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.16, 0.5))
    # plt.grid(True)
    ax.set_ylim(-0.05, 1.2)
    ax2.set_ylim(-0.05, 1.2)
    ax.tick_params(labelsize=12)
    # ax.set_title(f'Simulated - Observed Downward Flux Difference Percentage\nalb=SSFR up/down flux ratio', fontsize=13)
    fig.suptitle(f'Upward Radiance and Flux Ref Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_ssrr_ref_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    for alt_ind in range(len(tmhr_ranges_select)):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax2 = ax.twinx()
        l1 = ax.plot(flux_wvl, ssrr_ref[alt_ind, :], '-', color=color_series[alt_ind], label='SSRR')
        
        l2 = ax.scatter(rsp_wvl, rsp_ref_mean_all[alt_ind, :], marker='o', color=color_series[alt_ind], edgecolors='k', s=50, alpha=0.7, label='RSP')
        if aviris_file is not None:
            l3 = ax.plot(aviris_reflectance_wvl, aviris_rad_to_ref, c='m', label='AVIRIS Rad x $\pi$ / TOA', alpha=0.7)
            l4 = ax.plot(aviris_reflectance_wvl, aviris_reflectance_spectrum, c='darkviolet', label='AVIRIS L2 Ref', alpha=0.7)
        l5 = ax2.plot(flux_wvl, ssfr_fup_ref[alt_ind, :], '--', color=color_series[alt_ind], alpha=0.7, label='SSFR')
        
        ll = l1 + [l2] + l5 + l3 + l4 if aviris_file is not None else l1 + [l2] + l5
        labs = [l.get_label() for l in ll]
        ax.legend(ll, labs, fontsize=10, loc='upper right', ) 
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('SSRR radiance x $\pi$ / TOA', fontsize=14)
        ax2.set_ylabel('SSFR Upward flux / TOA', fontsize=14)
        ax.set_ylim(-0.05, 1.2)
        ax2.set_ylim(-0.05, 1.2)
        ax.tick_params(labelsize=12)
        fig.suptitle(f'Upward Radiance and Flux Ref Comparison {date_s}\nZ={alt_avg_all[alt_ind]:.2f}km', fontsize=16, y=0.98)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssfr_ssrr_ref_comparison_Z%.2fkm.png' % (date_s, date_s, case_tag, alt_avg_all[alt_ind]), bbox_inches='tight', dpi=150)
    
    
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if aviris_rad_file is not None:
        # ax.errorbar(aviris_rad_wvl, aviris_rad_spectrum, yerr=aviris_rad_spectrum_unc, 
        #             fmt='o', color='m', markersize=2, label='AVIRIS', alpha=0.7)
        ax.plot(aviris_rad_wvl, aviris_rad_spectrum, '-', color='m', alpha=0.7)
    for alt_ind in range(len(tmhr_ranges_select)):
        # plot ssrr
        ax.tick_params(labelsize=12)
        # ax.errorbar(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
        #             yerr=ssrr_rad_up_std_all[alt_ind, :], color=color_series[alt_ind], markersize=4, label='SSRR Z=%.2fkm' % (alt_avg_all[alt_ind]))
        ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
                 color=color_series[alt_ind], label='Z=%.2fkm' % (alt_avg_all[alt_ind]))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    fig.suptitle(f'SSRR Upward Radiance Comparison {date_s}', fontsize=13)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssrr_rad_up_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if aviris_rad_file is not None:
        # ax.errorbar(aviris_rad_wvl, aviris_rad_spectrum, yerr=aviris_rad_spectrum_unc, 
        #             fmt='o', color='m', markersize=2, label='AVIRIS', alpha=0.7)
        ax.plot(aviris_rad_wvl, aviris_rad_spectrum, '-', color='m', alpha=0.7)
        
    ax2 = ax.twinx()
    for alt_ind in range(len(tmhr_ranges_select)):
        # plot ssrr
        ax.tick_params(labelsize=12)
        ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
                 color=color_series[alt_ind], label='SSRR Z=%.2fkm' % (alt_avg_all[alt_ind]))

    
        ax2.plot(flux_wvl, ssfr_fup_mean_all[alt_ind, :],
                 linestyle='--', label='SSRR Ref Z=%.2fkm' % (alt_avg_all[alt_ind]))
        
        for rsp_wvl_ind in range(len(rsp_wvl)):
            rsp_wvl_val = rsp_wvl[rsp_wvl_ind]
            # find the closest wavelength in flux_wvl
            rsp_rad_mean = rsp_rad_mean_all[alt_ind, rsp_wvl_ind]
            rsp_rad_std = rsp_rad_std_all[alt_ind, rsp_wvl_ind]
            ax.errorbar(rsp_wvl_val, rsp_rad_mean, yerr=rsp_rad_std, fmt='s', color=color_series[alt_ind], markersize=6,)# label='SSRR RSP Z=%.2fkm' % (alt_avg_all[alt_ind]))
    
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
    ax2.set_ylabel('SSFR Ref Upward Flux (W/m$^2$/nm)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    fig.suptitle(f'SSRR Upward Radiance Comparison {date_s}', fontsize=13)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssrr_ssfr_rad_up_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    plt.close('all')
    
    for j in range(len(tmhr_ranges_select)):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if aviris_rad_file is not None:
            # ax.errorbar(aviris_rad_wvl, aviris_rad_spectrum, yerr=aviris_rad_spectrum_unc, 
            #             fmt='o', color='m', markersize=2, label='AVIRIS', alpha=0.7)
            ax.plot(aviris_rad_wvl, aviris_rad_spectrum[j, :], '-', color='m', alpha=0.7)
            
        ax2 = ax.twinx()
        # plot ssrr
        ax.tick_params(labelsize=12)
        ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[j, :],
                color=color_series[j], label='SSRR$\cdot \pi$/TOA')

    
        ax2.plot(flux_wvl, ssfr_fup_mean_all[j, :],
                linestyle='--', label='SSFR/TOA')
        
        for rsp_wvl_ind in range(len(rsp_wvl)):
            rsp_wvl_val = rsp_wvl[rsp_wvl_ind]
            # find the closest wavelength in flux_wvl
            rsp_rad_mean = rsp_rad_mean_all[j, rsp_wvl_ind]
            rsp_rad_std = rsp_rad_std_all[j, rsp_wvl_ind]
            ax.errorbar(rsp_wvl_val, rsp_rad_mean, yerr=rsp_rad_std, fmt='s', color=color_series[j], markersize=6, label='RSP$\cdot \pi$/TOA')
        
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
        ax2.set_ylabel('SSFR Ref Upward Flux (W/m$^2$/nm)', fontsize=14)
        ax.legend(fontsize=10, loc='upper right')
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        # ax.set_ylim(-0.05, 1.05)
        fig.suptitle(f'SSRR Upward Radiance Comparison {date_s}\nZ={alt_avg_all[j]:.2f}, lat={lat_avg_all[j]:.2f}$^o$N',
                     fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssrr_ssfr_rad_up_toa_ratio_comparison_leg_%d.png' % (date_s, date_s, case_tag, j), bbox_inches='tight', dpi=150)
        plt.close('all')
    
    print('stop here!')

    if aviris_rad_file is not None:
        aviris_rad_interp = interp1d(aviris_rad_wvl, aviris_rad_spectrum, kind='linear', bounds_error=False, fill_value='extrapolate')
        aviris_rad_at_rsp_wvl = aviris_rad_interp(rsp_wvl)
        aviris_rad_at_ssrr_wvl = aviris_rad_interp(ssrr_rad_up_wvl)
    
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(aviris_rad_wvl, aviris_rad_spectrum, yerr=aviris_rad_spectrum_unc, 
                    fmt='o', color='m', markersize=2, label='AVIRIS Radiance', alpha=0.7)
        l1 = ax.plot(aviris_rad_wvl, aviris_rad_spectrum, '-', color='m', label='AVIRIS Radiance', alpha=0.7)
        ax_t = ax.twinx()
        if aviris_file is not None:
            l2 = ax_t.plot(aviris_reflectance_wvl, aviris_reflectance_spectrum, c='b', label='AVIRIS Reflectance', alpha=0.7)
            ax_t.fill_between(aviris_reflectance_wvl, aviris_reflectance_spectrum-aviris_reflectance_spectrum_unc, aviris_reflectance_spectrum+aviris_reflectance_spectrum_unc, color='b', alpha=0.3)
        ll = l1+l2 if aviris_file is not None else l1
        labs = [l.get_label() for l in ll]
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
        ax_t.set_ylabel('Reflectance', fontsize=14)
        ax.legend(ll, labs, fontsize=10,)
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        ax_t.tick_params(labelsize=12)
        # ax.set_ylim(-0.05, 1.05)
        fig.suptitle(f'SSRR Upward Radiance & Reflectance Comparison {date_s}', fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_aviris_rad_reflectance_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for alt_ind in range(len(tmhr_ranges_select)):
            ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :]/aviris_rad_at_ssrr_wvl, '-', color=color_series[alt_ind], label='Z=%.2fkm' % (alt_avg_all[alt_ind]))
            ax.scatter(rsp_wvl, rsp_rad_mean_all[alt_ind, :]/aviris_rad_at_rsp_wvl, marker='s', color=color_series[alt_ind], edgecolors='k', s=50, alpha=0.7)
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance Ratio', fontsize=14)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        ax.set_ylim(00.7, 1.3)
        ax.set_xlim(350, 2200)
        ax.hlines(1.0, 350, 2200, color='gray', linestyle='--')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.suptitle(f'SSRR & RSP Upward Radiance to AVIRIS Ratio Comparison {date_s}', fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssrr_rsp_aviris_rad_ratio_comparison_all.png' % (date_s, date_s, case_tag,), bbox_inches='tight', dpi=150)
        
        for alt_ind in range(len(tmhr_ranges_select)):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :]/aviris_rad_at_ssrr_wvl, '-', color=color_series[alt_ind], label='SSRR/AVIRIS')
            ax.scatter(rsp_wvl, rsp_rad_mean_all[alt_ind, :]/aviris_rad_at_rsp_wvl, marker='s', color=color_series[alt_ind], edgecolors='k', s=50, alpha=0.7, label='RSP/AVIRIS')
            ax.set_xlabel('Wavelength (nm)', fontsize=14)
            ax.set_ylabel('Radiance Ratio', fontsize=14)
            ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
            # plt.grid(True)
            ax.tick_params(labelsize=12)
            ax.set_ylim(0.7, 1.3)
            ax.set_xlim(350, 2200)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.hlines(1.0, 350, 2200, color='gray', linestyle='--') 
            fig.suptitle(f'SSRR & RSP Upward Radiance to AVIRIS Ratio Comparison {date_s}\nZ={alt_avg_all[alt_ind]:.2f}km', fontsize=13)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_ssrr_rsp_aviris_rad_ratio_comparison_Z%.2fkm.png' % (date_s, date_s, case_tag, alt_avg_all[alt_ind]), bbox_inches='tight', dpi=150)


def fit_1d_poly(x, y, order=1, dx=None, x0=None):
    """
    Fit 1D polynomial to data
    Input:
        x: 1D array of x values
        y: 1D array of y values
        order: order of polynomial
    Output:
        poly1d object
    """
    mask = ~np.isnan(x) & ~np.isnan(y)    
    # fit polynomial
    coeffs = np.polyfit(x[mask], y[mask], order)
    
    if x0 is None:
        x0 = np.nanmean(x[mask][:2])
    y0 = np.nanmean(y[mask][:2])
    x1 = np.nanmean(x[mask][-2:])
    y1 = np.nanmean(y[mask][-2:])
    if dx is None:
        dx = x1 - x0
    slope = (y1 - y0) / dx
    intercept = y0 - slope * x0
    coeffs = [slope, intercept]
    
    p = np.poly1d(coeffs)
    return p


def ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_ranges_select, date_s, case_tag):
    t_hsk = np.array(data_hsk["tmhr"])    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_hsr1 = data_hsr1['time']/3600.0  # convert to hours
    
    pitch_ang = data_hsk['ang_pit']
    roll_ang = data_hsk['ang_rol']
    
    tmhr_hsk_mask = (t_hsk >= tmhr_ranges_select[0][0]) & (t_hsk <= tmhr_ranges_select[-1][1])
    
    hsr_wvl = data_hsr1['wvl_dn_tot']
    hsr_ftot = data_hsr1['f_dn_tot']
    hsr_fdif = data_hsr1['f_dn_dif']
    hsr_dif_ratio = hsr_fdif/hsr_ftot
    hsr_550_ind = np.argmin(np.abs(hsr_wvl - 550))
    hsr_530_ind = np.argmin(np.abs(hsr_wvl - 530))
    hsr_570_ind = np.argmin(np.abs(hsr_wvl - 570))
    hsr1_diff_ratio = data_hsr1["f_dn_dif"]/data_hsr1["f_dn_tot"]
    hsr1_diff_ratio_530_570_mean = np.nanmean(hsr1_diff_ratio[:, hsr_530_ind:hsr_570_ind+1], axis=1)
    
    pitch_roll_mask = np.sqrt(pitch_ang**2 + roll_ang**2) < 2.5  # pitch and roll greater < 2.5 deg
    # pitch_roll_mask = np.logical_and(np.abs(pitch_ang) < 2.5, np.abs(roll_ang) < 2.5)  # pitch and roll greater < 2.5 deg
    pitch_roll_mask = pitch_roll_mask[tmhr_hsk_mask]
    
    
    ssfr_zen_wvl = data_ssfr['wvl_dn']
    ssfr_fdn = data_ssfr['f_dn']
    ssfr_nad_wvl = data_ssfr['wvl_up']
    ssfr_fup = data_ssfr['f_up']
    
    t_ssfr_tmhr_mask = (t_ssfr >= tmhr_ranges_select[0][0]) & (t_ssfr <= tmhr_ranges_select[-1][1])
    t_ssfr_tmhr = t_ssfr[t_ssfr_tmhr_mask]
    ssfr_fdn_tmhr = ssfr_fdn[t_ssfr_tmhr_mask]
    ssfr_fup_tmhr = ssfr_fup[t_ssfr_tmhr_mask]
    # ssfr_fdn_tmhr[~pitch_roll_mask] = np.nan
    # ssfr_fup_tmhr[~pitch_roll_mask] = np.nan
    
    t_hsr1_tmhr_mask = (t_hsr1 >= tmhr_ranges_select[0][0]) & (t_hsr1 <= tmhr_ranges_select[-1][1])
    t_hsr1_tmhr = t_hsr1[t_hsr1_tmhr_mask]
    hsr_dif_ratio = hsr_dif_ratio[t_hsr1_tmhr_mask]
    hsr1_diff_ratio_530_570_mean = hsr1_diff_ratio_530_570_mean[t_hsr1_tmhr_mask]
    
    hsr1_530_570_thresh = 0.18
    cloud_mask_hsr1 = hsr1_diff_ratio_530_570_mean > hsr1_530_570_thresh
    # ssfr_fup_tmhr[cloud_mask_hsr1] = np.nan
    # ssfr_fdn_tmhr[cloud_mask_hsr1] = np.nan
    
    
    print("ssfr_fdn_tmhr shape:", ssfr_fdn_tmhr.shape)
    print("pitch_roll_mask shape:", pitch_roll_mask.shape)
    print("ssfr_fdn_tmhr[pitch_roll_mask, ssfr_zen_550_ind] shape:", ssfr_fdn_tmhr[pitch_roll_mask, :].shape)
    
    ssfr_zen_550_ind = np.argmin(np.abs(ssfr_zen_wvl - 550))
    ssfr_nad_550_ind = np.argmin(np.abs(ssfr_nad_wvl - 550))
    
    
    
    fig, (ax10, ax20) = plt.subplots(2, 1, figsize=(16, 12))
    ax11 = ax10.twinx()
    
    
    l1 = ax10.plot(t_ssfr, ssfr_fdn[:, ssfr_zen_550_ind], '--', color='k', alpha=0.85)
    l2 = ax10.plot(t_ssfr, ssfr_fup[:, ssfr_nad_550_ind], '--', color='k', alpha=0.85)
    
    ax10.plot(t_ssfr_tmhr, ssfr_fdn_tmhr[:, ssfr_zen_550_ind], 'r-', label='SSFR Down 550nm', linewidth=3)
    ax10.plot(t_ssfr_tmhr, ssfr_fup_tmhr[:, ssfr_nad_550_ind], 'b-', label='SSFR Up 550nm', linewidth=3)
    
    l3 = ax11.plot(t_hsr1_tmhr, hsr_dif_ratio[:, hsr_550_ind], 'm-', label='HSR1 Diff Ratio 550nm')
    ax11.set_ylabel('HSR1 Diff Ratio 550nm', fontsize=14)
    
    # ax2.plot(t_hsk, data_hsk['ang_hed'], 'r-', label='HSK Heading')
    ax20.plot(t_hsk, pitch_ang, 'g-', label='HSK Pitch')
    ax20.plot(t_hsk, roll_ang, 'b-', label='HSK Roll')
    for i, (lo, hi) in enumerate(tmhr_ranges_select):
        for ax in [ax10, ax20]:
            ax.fill_betweenx([0, 6], lo, hi, color='gray', alpha=0.3, transform=ax.get_xaxis_transform(),)# label=f'Leg {i+1}' if i==0 else None)
            ax.set_xlabel('Time (UTC)')
            ax.set_xlim(tmhr_ranges_select[0][0]*0.999, tmhr_ranges_select[-1][1]*1.001)
    ll = l1 + l2 + l3
    labs = [l.get_label() for l in ll]
    ax10.legend(ll, labs, fontsize=10, )
    ax20.legend()
    ax10.set_ylim(0.4, 1.2)
    ax11.set_ylim(0.0, 0.65)
    ax10.set_ylabel('SSFR Flux (W/m$^2$/nm)')
    
    ax20.set_ylabel('HSK Pitch/Roll (deg)')
    ax20.set_ylim(-5, 5)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_pitch_roll_heading_550nm.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)

def solar_interpolation_func(solar_flux_file, date):
    """Solar spectrum interpolation function"""
    from scipy.interpolate import interp1d
    f_solar = pd.read_csv(solar_flux_file, delim_whitespace=True, comment='#', names=['wvl', 'flux'])
    wvl_solar = f_solar['wvl'].values
    flux_solar = f_solar['flux'].values/1000 # in W/m^2/nm
    flux_solar *= er3t.util.cal_sol_fac(date)
    return interp1d(wvl_solar, flux_solar, bounds_error=False, fill_value=0.0)

def write_2col_file(filename, wvl, val, header):
    """Write two-column data to a file with a header"""
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(val)):
            f.write(f'{wvl[i]:11.3f} {val[i]:12.3e}\n')


    

def flt_trk_data_collect(date=datetime.datetime(2024, 5, 31),
                     tmhr_ranges_select=[[14.10, 14.27]],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                     simulation_interval=None, # in minute
                    ):

    # case specification
    #/----------------------------------------------------------------------------\#
    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#
    
    
    log = logging.getLogger("lrt")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    
    os.makedirs(f'fig/{date_s}', exist_ok=True)
    
    if simulation_interval is not None:
        # split tmhr_ranges_select into smaller intervals
        tmhr_ranges_select_new = []
        for lo, hi in tmhr_ranges_select:
            t_start = lo
            while t_start < hi and t_start < (hi - 0.0167/6):  # 10s
                t_end = min(t_start + simulation_interval/60.0, hi)
                tmhr_ranges_select_new.append([t_start, t_end])
                t_start = t_end
        tmhr_ranges_select = tmhr_ranges_select_new

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_ssrr = load_h5(config.ssrr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    
    if date_s != '20240603':
        # MARLI netCDF
        with Dataset(str(config.marli(date_s))) as ds:
            data_marli = {var: ds.variables[var][:] for var in ("time","Alt","H","T","LSR","WVMR")}
    else:
        data_marli = {'time': np.array([]), 'Alt': np.array([]), 'H': np.array([]), 'T': np.array([]), 'LSR': np.array([]), 'WVMR': np.array([])}
    
    log.info("ssfr filename:", config.ssfr(date_s))
    
    # plot ssfr time series for checking sable legs selection
    ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_ranges_select, date_s, case_tag)

    # Build leg masks
    t_hsk = np.array(data_hsk["tmhr"])
    leg_masks = [(t_hsk>=lo)&(t_hsk<=hi) for lo,hi in tmhr_ranges_select]
    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_hsr1 = data_hsr1['time']/3600.0  # convert to hours
    t_ssrr = data_ssrr['time']/3600.0  # convert to hours
    t_marli = data_marli['time'] # in hours

          
    # Solar spectrum interpolation function
    flux_solar_interp = solar_interpolation_func(solar_flux_file='arcsix_ssfr_solar_flux_slit.dat', date=date)

    # check rsp l1b folder
    rsp_l1b_dir = f'{_fdir_general_}/rsp/ARCSIX-RSP-L1B_P3B_{date_s}_R01'
    if os.path.exists(rsp_l1b_dir):
        rsp_l1b_files = sorted(glob.glob(f'{rsp_l1b_dir}/ARCSIX-RSP-L1B_P3B_{date_s}*.h5'))
        print("rsp_l1b_files:", rsp_l1b_files)
        if len(rsp_l1b_files) == 0:
            print(f"No RSP L1B files found in {rsp_l1b_dir}")
            rsp_1lb_avail = False
        else:
            rsp_l1b_files = np.array(rsp_l1b_files)
            log.info(f"Found {len(rsp_l1b_files)} RSP L1B files in {rsp_l1b_dir}")
            # ARCSIX-RSP-L1B_P3B_20240605111213_R01.h5
            print([os.path.basename(f).split('_')[2] for f in rsp_l1b_files])
            rsp_l1b_times = np.array([int(os.path.basename(f).split('_')[2][8:10]) + int(os.path.basename(f).split('_')[2][10:12])/60.0 + int(os.path.basename(f).split('_')[2][12:14])/3600.0 for f in rsp_l1b_files])
            rsp_1lb_avail = True
    else:
        print(f"RSP L1B directory {rsp_l1b_dir} does not exist")
        rsp_1lb_avail = False

    # read satellite granule
    #/----------------------------------------------------------------------------\#
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)
    fname_cld_obs_info = '%s/%s_cld_obs_info_%s_%s_%s_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag)
    if 1:#not os.path.exists(fname_cld_obs_info):      
        
        # Loop legs: load raw NC, apply cloud logic, interpolate, plot
        for i, mask in enumerate(leg_masks):
            
            # find index arrays in one go
            times_leg = t_hsk[mask]
            print(f"Leg {i+1}: time range {times_leg.min()}-{times_leg.max()}h")
            
            sel_ssfr, sel_hsr1, sel_ssrr = (
                nearest_indices(t_hsk, mask, arr)
                for arr in (t_ssfr, t_hsr1, t_ssrr)
            )
            
            if len(t_marli) > 0:
                sel_marli = nearest_indices(t_hsk, mask, t_marli)
            
            # choose the rsp l1b file for this leg
            if rsp_1lb_avail:
                time_leg_start = times_leg.min()
                time_leg_end = times_leg.max()
                rsp_l1b_sel = np.zeros(rsp_l1b_times.shape, dtype=bool)
                rsp_file_start = np.where(rsp_l1b_times==rsp_l1b_times[((rsp_l1b_times - time_leg_start)<0)][-1])[0][0]
                try:
                    rsp_file_end = np.where(rsp_l1b_times==rsp_l1b_times[((rsp_l1b_times - time_leg_end)>0)][0])[0][0]
                    rsp_l1b_sel[rsp_file_start:rsp_file_end] = True
                except IndexError:
                    rsp_l1b_sel[rsp_file_start:] = True
                if len(rsp_l1b_sel) == 0:
                    print(f"No RSP L1B files found for leg {i+1} time range {time_leg_start}-{time_leg_end}h")
                    rsp_l1b_files_leg = None
                rsp_l1b_files_leg = rsp_l1b_files[rsp_l1b_sel]
                
                rsp_time_all = []
                rsp_rad_all = []
                rsp_ref_all = []
                rsp_rad_norm_all = []
                rsp_lon_all = []
                rsp_lat_all = []
                rsp_mu0_all = []
                rsp_sd_all = []
                for rsp_file_name in rsp_l1b_files_leg:
                    log.info(f"Reading RSP L1B file: {rsp_file_name}")
                    data_rsp = load_h5(rsp_file_name)
                    t_rsp = data_rsp['Platform/Fraction_of_Day']*24.0  # in hours, (dim_Scans)
                    rsp_vza = data_rsp['Geometry/Viewing_Zenith']  # in degrees, (dim_Scans, dim_Scene_Sectors)
                    rsp_ground_lat = data_rsp['Geometry/Ground_Latitude']  # (dim_Scans, dim_Scene_Sectors)
                    rsp_ground_lon = data_rsp['Geometry/Ground_Longitude']  # (dim_Scans, dim_Scene_Sectors)
                    rsp_sza = data_rsp['Platform/Platform_Solar_Zenith']  # (dim_Scans), in radians
                    rsp_sd = data_rsp['Platform/Solar_Distance']  # (dim_Scans), in AU
                    
                    intensity_1 = data_rsp['Data/Intensity_1']  # (dim_Scans, dim_Scene_Sectors, bands)
                    intensity_2 = data_rsp['Data/Intensity_2']  # (dim_Scans, dim_Scene_Sectors, bands)
                    rsp_wvl = data_rsp['Data/Wavelength']  # in nm, (bands,)
                    rsp_solar_const = data_rsp['Calibration/Solar_Constant']  # in W/m^2/nm, (bands,)
                    
                    
                    nadir_select = np.argmax(rsp_vza, axis=1)  # select the nadir sector
                    sel_rsp = nearest_indices(t_hsk, mask, t_rsp)
                    
                    rsp_time_sel = t_rsp[sel_rsp]
                    rsp_int_1_sel = intensity_1[sel_rsp, nadir_select[sel_rsp], :]  # (time, bands)
                    rsp_int_2_sel = intensity_2[sel_rsp, nadir_select[sel_rsp], :]  # (time, bands)
                    rsp_lon_sel = rsp_ground_lon[sel_rsp, nadir_select[sel_rsp]]
                    rsp_lat_sel = rsp_ground_lat[sel_rsp, nadir_select[sel_rsp]]
                    rsp_sza_sel = rsp_sza[sel_rsp]
                    rsp_sd_sel = rsp_sd[sel_rsp]
                    
                    rsp_rad_norm = (rsp_int_1_sel + rsp_int_2_sel) / 2  # (time, bands), in counts
                    rsp_rad = rsp_rad_norm * rsp_solar_const[np.newaxis, :] / np.pi
                    rsp_rad[rsp_rad < 0] = np.nan
                    rsp_ref_cal = rsp_rad_norm * rsp_sd_sel[:, np.newaxis]**2 / np.cos(rsp_sza_sel[:, np.newaxis])  # (time, bands), in W/m^2/sr/nm
                    
                    rsp_time_all.extend(rsp_time_sel)
                    rsp_rad_all.extend(rsp_rad)
                    rsp_rad_norm_all.extend(rsp_rad_norm)
                    rsp_ref_all.extend(rsp_ref_cal)
                    rsp_lon_all.extend(rsp_lon_sel)
                    rsp_lat_all.extend(rsp_lat_sel)
                    rsp_mu0_all.extend(np.cos(np.deg2rad(rsp_sza_sel)))
                    rsp_sd_all.extend(rsp_sd_sel)
                    


            # assemble a small dict for this leg
            leg = {
                "time":    times_leg,
                "alt":     data_hsk["alt"][mask] / 1000.0,
                "heading": data_hsk["ang_hed"][mask],
                "hsr1_tot": data_hsr1["f_dn_tot"][sel_hsr1],
                "hsr1_dif": data_hsr1["f_dn_dif"][sel_hsr1],
                "hsr1_wvl": data_hsr1["wvl_dn_tot"],
                "lon":     data_hsk["lon"][mask],
                "lat":     data_hsk["lat"][mask],
                "sza":     data_hsk["sza"][mask],
                "saa":     data_hsk["saa"][mask],
                "p3_alt":  data_hsk["alt"][mask] / 1000.0,
                
                "rsp_lon": np.array(rsp_lon_all) if rsp_1lb_avail else None,
                "rsp_lat": np.array(rsp_lat_all) if rsp_1lb_avail else None,
                "rsp_rad": np.array(rsp_rad_all) if rsp_1lb_avail else None,
                "rsp_rad_norm": np.array(rsp_rad_norm_all) if rsp_1lb_avail else None,
                "rsp_ref": np.array(rsp_ref_all) if rsp_1lb_avail else None,
                "rsp_wvl": rsp_wvl if rsp_1lb_avail else None,
                "rsp_mu0": np.array(rsp_mu0_all) if rsp_1lb_avail else None,
                "rsp_sd":  np.array(rsp_sd_all) if rsp_1lb_avail else None,   
            }

            if len(data_marli['time']) > 0:
                marli_wvmr = data_marli["WVMR"][sel_marli, :]
                marli_wvmr[marli_wvmr == 9999] = np.nan
                marli_wvmr[marli_wvmr > 50] = np.nan  # filter out extremely high values
                marli_wvmr[marli_wvmr < 0] = 0
                marli_h = data_marli["H"][...]
                marli_mask = np.any(np.isfinite(marli_wvmr), axis=0)
                marli_wvmr = marli_wvmr[:, marli_mask]
                marli_h = marli_h[marli_mask]
                marli_wvmr_mean = np.nanmean(marli_wvmr, axis=0)
                
                leg.update({
                    "marli_h": marli_h,
                    "marli_wvmr": marli_wvmr_mean,
                })
            else:
                leg.update({
                    "marli_h": None,
                    "marli_wvmr": None,
                })

            
            sza_hsk = data_hsk['sza'][mask]

            ssfr_zen_flux = data_ssfr['f_dn'][sel_ssfr, :]
            ssfr_nad_flux = data_ssfr['f_up'][sel_ssfr, :]
            ssfr_zen_toa = flux_solar_interp(data_ssfr['wvl_dn']) * np.cos(np.deg2rad(sza_hsk))[:, np.newaxis]  # W/m^2/nm
            ssfr_zen_wvl = data_ssfr['wvl_dn']
            ssfr_nad_wvl = data_ssfr['wvl_up']
            
            ssfr_nad_flux_interp = ssfr_zen_flux.copy()
            for j in range(ssfr_nad_flux.shape[0]):
                f_nad_flux_interp = interp1d(ssfr_nad_wvl, ssfr_nad_flux[j, :], axis=0, bounds_error=False, fill_value='extrapolate')
                ssfr_nad_flux_interp[j, :] = f_nad_flux_interp(ssfr_zen_wvl)
        
            pitch_roll_mask = np.sqrt(data_hsk["ang_pit"][mask]**2 + data_hsk["ang_rol"][mask]**2) < 2.0
            # ssfr_zen_flux[~pitch_roll_mask, :] = np.nan
            # ssfr_nad_flux_interp[~pitch_roll_mask, :] = np.nan
            # ssfr_zen_toa[~pitch_roll_mask, :] = np.nan
            
            hsr1_530nm_ind = np.argmin(np.abs(leg['hsr1_wvl'] - 530.0))
            hsr1_570nm_ind = np.argmin(np.abs(leg['hsr1_wvl'] - 570.0))
            hsr1_diff_ratio = data_hsr1["f_dn_dif"][sel_hsr1]/data_hsr1["f_dn_tot"][sel_hsr1]
            hsr1_diff_ratio_530_570_mean = np.nanmean(hsr1_diff_ratio[:, hsr1_530nm_ind:hsr1_570nm_ind+1], axis=1)
            hsr1_530_570_thresh = 0.18
            cloud_mask_hsr1 = hsr1_diff_ratio_530_570_mean > hsr1_530_570_thresh
            # ssfr_zen_flux[cloud_mask_hsr1, :] = np.nan
            # ssfr_nad_flux_interp[cloud_mask_hsr1, :] = np.nan
            
            
            leg['ssfr_zen'] = ssfr_zen_flux
            leg['ssfr_nad'] = ssfr_nad_flux_interp
            leg['ssfr_zen_wvl'] = ssfr_zen_wvl
            leg['ssfr_nad_wvl'] = ssfr_zen_wvl
            leg['ssfr_toa'] = ssfr_zen_toa
            
            
            # ssrr
            # interpolate ssrr zenith radiance to nadir wavelength grid
            f_zen_rad_interp = interp1d(data_ssrr["wvl_dn"], data_ssrr["i_dn"][sel_ssrr, :], axis=1, bounds_error=False, fill_value=np.nan)
            ssrr_rad_zen_i = f_zen_rad_interp(ssfr_zen_wvl)
            f_nad_rad_interp = interp1d(data_ssrr["wvl_up"], data_ssrr["i_up"][sel_ssrr, :], axis=1, bounds_error=False, fill_value=np.nan)
            ssrr_rad_nad_i = f_nad_rad_interp(ssfr_zen_wvl)
            
            leg['ssrr_zen_rad'] = ssrr_rad_zen_i
            leg['ssrr_nad_rad'] = ssrr_rad_nad_i
            leg['ssrr_zen_wvl'] = ssfr_zen_wvl
            leg['ssrr_nad_wvl'] = ssfr_zen_wvl

            vars()["cld_leg_%d" % i] = leg
        
            # save the cloud observation information to a pickle file
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
            with open(fname_pkl, 'wb') as f:
                pickle.dump(vars()["cld_leg_%d" % i], f, protocol=pickle.HIGHEST_PROTOCOL)

            del leg  # free memory
            del sel_ssfr, sel_ssrr, sel_hsr1
            if rsp_1lb_avail:
                del rsp_time_all, rsp_rad_all, rsp_ref_all, rsp_lon_all, rsp_lat_all
                del rsp_mu0_all, rsp_sd_all
                del data_rsp
                del t_rsp, intensity_1, intensity_2, rsp_solar_const
                del rsp_ground_lon, rsp_ground_lat, rsp_sza, rsp_vza
            gc.collect()
        
    else:
        print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
        for i in range(len(tmhr_ranges_select)):
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
            with open(fname_pkl, 'rb') as f:
                vars()[f"cld_leg_{i}"] = pickle.load(f)   
    
    
 

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


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

    # atm_corr_plot(date=datetime.datetime(2024, 6, 7),
    #                 tmhr_ranges_select=[[15.3400, 15.7583], [15.8403, 16.2653]],
    #                 case_tag='cloudy_track_2_atm_corr',
    #                 config=config,
    #                 )    

    # atm_corr_plot(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[15.84, 15.88], [15.94, 15.98]],
    #                 case_tag='cloudy_track_1_atm_corr',
    #                 config=config,
    #                 )


    # atm_corr_plot(date=datetime.datetime(2024, 6, 3),
    #                 tmhr_ranges_select=[[14.72, 14.86], [14.95, 15.09]],
    #                 case_tag='cloudy_track_1_atm_corr',
    #                 config=config,
    #                 )

    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                 tmhr_ranges_select=[[13.99, 14.18], [14.26, 14.46]],
    #                 case_tag='cloudy_track_2_atm_corr',
    #                 config=config,
    #                 )

    # atm_corr_plot(date=datetime.datetime(2024, 6, 11),
    #                 tmhr_ranges_select=[[13.9111, 14.3417], [15.3528, 15.7139]],
    #                 case_tag='clear_sky_track_1_atm_corr',
    #                 config=config,
    #                 )
    

    # atm_corr_plot(date=datetime.datetime(2024, 5, 31),
    #                 tmhr_ranges_select=[[14.10, 14.27], [16.49, 16.72]],
    #                 case_tag='clear_sky_track_1_atm_corr',
    #                 config=config,
    #                 )

    
    # atm_corr_plot(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[16.78, 16.85], [16.91, 17.00]],
    #                 case_tag='clear_sky_track_1_atm_corr',
    #                 config=config,
    #                 )
    
    

    # atm_corr_plot(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[[15.55, 15.9292], [16.0431, 16.32]],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 config=config,
    #                 )
    
    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                 tmhr_ranges_select=[[16.54, 16.62], [16.85, 16.94]],
    #                 case_tag='clear_sky_track_1_atm_corr',
    #                 config=config,
    #                 )
    
    
    # atm_corr_plot(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[
    #                                         [14.594, 14.747],
    #                                         [14.760, 14.913], # cloud probably
    #                                         [14.926, 15.062], # cloud probably
    #                                         [15.560, 15.580],
    #                                         [15.593, 15.746],
    #                                         [15.760, 15.912],
    #                                         [16.050, 16.080],
    #                                         [16.093, 16.247], # cloud shadow
    #                                         [16.260, 16.413], # cloud shadow
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 config=config,
    #                 )
    # sys.exit()
    
    
    
    
    # atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[[17.0833, 17.1028],
    #                                         [17.1264, 17.1333],
    #                                         [17.1542, 17.1625],
    #                                         [17.1833, 17.1931],
    #                                         [17.2153, 17.2181],
    #                                         [17.2403, 17.2500],
    #                                         ],
    #                 case_tag='clear_sky_spiral_atm_corr',
    #                 config=config,
    #                 )
    
    # atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 11),
    #                     tmhr_ranges_select=[[14.5667, 14.5694],
    #                                         [14.5986, 14.6097],
    #                                         [14.6375, 14.6486], # cloud shadow
    #                                         [14.6778, 14.6903],
    #                                         [14.7208, 14.7403],
    #                                         [14.7653, 14.7875],
    #                                         [14.8125, 14.8278],
    #                                         [14.8542, 14.8736],
    #                                         [14.8986, 14.9389], # more cracks
    #                                         ],
    #                 case_tag='clear_sky_spiral_atm_corr',
    #                 config=config,
    #                 )
    
    # atm_corr_spiral_plot(date=datetime.datetime(2024, 5, 31),
    #                     tmhr_ranges_select=[[15.1903, 15.2083],
    #                                         [15.2389, 15.2528],
    #                                         [15.2806, 15.3014],
    #                                         [15.3292, 15.3431],
    #                                         [15.3694, 15.3944],
    #                                         [15.4167, 15.4458],
    #                                         [15.4736, 15.5056],
    #                                         [15.5264, 15.5556],
    #                                         [15.5792, 15.6056],
    #                                         [15.6486, 15.6636],
    #                                         [15.6878, 15.7042],
    #                                         ],
    #                 case_tag='clear_sky_spiral_atm_corr',
    #                 config=config,
    #                 )


    
    
    # atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[[13.7889, 13.8010],
    #                                         [13.8350, 13.8395],
    #                                         [13.8780, 13.8885],
    #                                         [13.9240, 13.9255],
    #                                         # [13.9389, 13.9403],
    #                                         [13.9540, 13.9715],
    #                                         [13.9980, 14.0153],
    #                                         # [14.0417, 14.0575],
    #                                         [14.0417, 14.0475],
    #                                         [14.0560, 14.0590],
    #                                         [14.0825, 14.0975],
    #                                         [14.1264, 14.1525],
    #                                         [14.1762, 14.1975],
    #                                         [14.2194, 14.2420],
    #                                         [14.2605, 14.2810]
    #                                         ],
    #                 case_tag='clear_sky_spiral_atm_corr_R0',
    #                 config=config,
    #                 )


    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),
    #                     tmhr_ranges_select=[[15.689, 15.737], 
    #                                         [15.760, 15.776],
    #                                         [15.855, 15.909],
    #                                         [15.921, 16.076],
    #                                         [16.088, 16.227],
    #                                         [16.306, 16.313],
    #                                         [16.319, 16.409],
    #                                         [16.421, 16.475],
    #                                         [16.501, 16.576],
    #                                         [16.588, 16.715]
    #                                         ],
    #                     case_tag='clear_sky_track_1_atm_corr',
    #                     config=config,
    #                     simulation_interval=1,
    #                     clear_sky=True,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=0.0,
    #                     manual_cloud_cwp=0.0,
    #                     manual_cloud_cth=0.0,
    #                     manual_cloud_cbh=0.0,
    #                     manual_cloud_cot=0.0,
    #                     iter=iter,
    #                     )
    
    # done
    # atm_corr_plot(date=datetime.datetime(2024, 5, 31),
    #                     tmhr_ranges_select=[[15.689, 15.737], 
    #                                         [15.760, 15.776],
    #                                         [15.855, 15.909],
    #                                         [15.921, 16.076],
    #                                         [16.088, 16.227],
    #                                         [16.306, 16.313],
    #                                         [16.319, 16.409],
    #                                         [16.421, 16.475],
    #                                         [16.501, 16.576],
    #                                         [16.588, 16.715]
    #                                         ],
    #                 case_tag='clear_sky_track_1_atm_corr',
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[
    #                                         [13.606, 13.629],
    #                                         [13.642, 13.712],
    #                                         [13.725, 13.743],
                                            
    #                                         ],
    #                     case_tag='cloudy_track_atm_corr_1',
    #                     config=config,
    #                     simulation_interval=1,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.8, 1.0]),
    #                                            np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.2, 2.3, 2.35, 2.4, 2.5,]),
    #                                         np.array([3.0, 3.5, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                     clear_sky=False,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=4.1 ,
    #                     manual_cloud_cwp=0.10438,
    #                     manual_cloud_cth=1.170,
    #                     manual_cloud_cbh=0.466,
    #                     manual_cloud_cot=38.355,
    #                     iter=iter,
    #                     )
    
    # done
    # atm_corr_plot(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[
    #                                         [13.606, 13.629],
    #                                         [13.642, 13.712],
    #                                         [13.725, 13.743],
    #                                         ],
    #                 case_tag='cloudy_track_atm_corr_1',
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.725, 14.796], 
    #                                         [14.808, 14.830],
    #                                         [14.836, 14.867],
    #                                         ],
    #                     case_tag='cloudy_track_atm_corr_2',
    #                     config=config,
    #                     simulation_interval=1,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.8, 1.0]),
    #                                            np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.2, 2.3, 2.35, 2.4, 2.5,]),
    #                                         np.array([3.0, 3.5, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                     clear_sky=False,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=10.7,
    #                     manual_cloud_cwp=0.06186,
    #                     manual_cloud_cth=2.328,
    #                     manual_cloud_cbh=1.226,
    #                     manual_cloud_cot=8.698,
    #                     iter=iter,
    #                     )
    
    # done
    # atm_corr_plot(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.725, 14.796], 
    #                                         [14.808, 14.830],
    #                                         [14.836, 14.867],
    #                                         ],
    #                 case_tag='cloudy_track_atm_corr_2', 
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
    
    # flt_trk_data_collect(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[
    #                                     [14.26, 14.413], 
    #                                     [14.426, 14.486],
    #                                     [14.5061, 14.5083],
    #                                     [14.594, 14.747],
    #                                     [14.760, 14.913], # cloud probably
    #                                     [14.926, 15.062], # cloud probably
    #                                     [16.850, 16.913],
    #                                     [16.929, 17.080],
    #                                     [17.093, 17.190],
    #                                     [17.200, 17.247],
    #                                     [17.260, 17.404],
    #                                     [17.411, 17.414],
    #                                     [17.426, 17.477],
    #                                     [17.488, 17.493],
    #                                     [17.498, 17.502], # unstable roll
    #                                     [17.506, 17.520], # unstable roll
    #                                     [17.533, 17.551],
    #                                     [17.570, 17.580],
    #                                     [17.599, 17.747],
    #                                     [17.760, 17.913],
    #                                     [17.926, 18.000],
    #                                     ],
    #                 case_tag='clear_sky_spiral_data_collect',
    #                 config=config,
    #                 simulation_interval=30,
    #                 )
    
    # done
    # atm_corr_plot(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[
    #                                         [14.26, 14.413], 
    #                                         [14.426, 14.486],
    #                                         [14.5061, 14.5083],
    #                                         [14.594, 14.747],
    #                                         [14.760, 14.913], # cloud probably
    #                                         [14.926, 15.062], # cloud probably
    #                                         [16.850, 16.913],
    #                                         [16.929, 17.080],
    #                                         [17.093, 17.190],
    #                                         [17.200, 17.247],
    #                                         [17.260, 17.404],
    #                                         [17.411, 17.414],
    #                                         [17.426, 17.477],
    #                                         [17.488, 17.493],
    #                                         [17.498, 17.502], # unstable roll
    #                                         [17.506, 17.520], # unstable roll
    #                                         [17.533, 17.551],
    #                                         [17.570, 17.580],
    #                                         [17.599, 17.747],
    #                                         [17.760, 17.913],
    #                                         [17.926, 18.000],
    #                                         ],
    #                 case_tag='clear_sky_spiral_data_collect',
    #                 simulation_interval=30,
    #                 config=config,
    #                 )
    
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[
    #                                         [12+25/60, 12+50/60],
    #                                         [13+15/60, 13+50/60],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=10,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[
    #                                         [12+25/60, 12+50/60],
    #                                         [13+15/60, 13+50/60],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=10,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),
    #                     tmhr_ranges_select=[
    #                                         [17.39, 17.58],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=10,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 8, 7),
    #                     tmhr_ranges_select=[
    #                                         [17.39, 17.58],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=10,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 5, 28),
    #                     tmhr_ranges_select=[
    #                                         [12.62, 15.18],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=15,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 5, 28),
    #                     tmhr_ranges_select=[
    #                                         [12.62, 15.18],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=15,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 5, 30),
    #                     tmhr_ranges_select=[
    #                                         [11.30, 12.29],
    #                                         [12.40, 12.79],
    #                                         [16.38, 17.42],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=15,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 5, 30),
    #                     tmhr_ranges_select=[
    #                                         [11.30, 12.29],
    #                                         [12.40, 12.79],
    #                                         [16.38, 17.42],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=15,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),
    #                     tmhr_ranges_select=[
    #                                         [12.77, 13.04],
    #                                         [13.20, 13.55],
    #                                         [14.50, 15.04],
    #                                         [16.89, 17.43],
                                            
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=15,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 5, 31),
    #                     tmhr_ranges_select=[
    #                                         [12.77, 13.04],
    #                                         [13.20, 13.55],
    #                                         [14.50, 15.04],
    #                                         [16.89, 17.43],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=15,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[
    #                                         [11.29, 11.86],
    #                                         [11.87, 13.23],
    #                                         [13.23, 13.44],
    #                                         [16.38, 17.80],
                                            
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=15,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[
    #                                         [11.29, 11.86],
    #                                         [11.87, 13.23],
    #                                         [13.23, 13.44],
    #                                         [16.38, 17.80],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=15,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[
    #                                         [11.33, 11.88],
    #                                         [12.00, 12.20],
    #                                         [12.33, 13.80],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=15,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[
    #                                         [11.33, 11.88],
    #                                         [12.00, 12.20],
    #                                         [12.33, 13.80],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=15,
    #                 config=config,
    #                 )
    
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[
    #                                         [11.29, 13.31],
    #                                         [17.26, 18.32],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=15,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[
    #                                         [11.29, 13.31],
    #                                         [17.26, 18.32],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=15,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[
    #                                         [13.61, 14.10],
    #                                         [14.17, 14.30],
    #                                         [14.60, 14.92],
    #                                         [17.67, 18.25],
    #                                         [18.33, 18.52]
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=15,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[
    #                                         [13.61, 14.10],
    #                                         [14.17, 14.30],
    #                                         [14.60, 14.92],
    #                                         [17.67, 18.25],
    #                                         [18.33, 18.52]
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=15,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 10),
    #                     tmhr_ranges_select=[
    #                                         [11.28, 11.51],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=15,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 6, 10),
    #                     tmhr_ranges_select=[
    #                                         [11.28, 11.51],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=15,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),
    #                     tmhr_ranges_select=[
    #                                         [11.28, 11.51],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=15,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 6, 10),
    #                     tmhr_ranges_select=[
    #                                         [11.28, 11.51],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=15,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),
    #                     tmhr_ranges_select=[
    #                                         [17.70, 17.87],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=6,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 8, 9),
    #                     tmhr_ranges_select=[
    #                                         [17.70, 17.87],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=6,
    #                 config=config,
    #                 )
    
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),
    #                     tmhr_ranges_select=[
    #                                         [13.05, 13.45],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=10,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 7, 29),
    #                     tmhr_ranges_select=[
    #                                         [13.05, 13.45],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=10,
    #                 config=config,
    #                 )


    # for iter in range(3):
        # flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
        #                 tmhr_ranges_select=[[13.99, 14.18], [14.26, 14.46]],
        #                 case_tag='cloudy_track_4_atm_corr_before',
        #                 config=config,
        #                 simulation_interval=10,
        #                 levels=np.concatenate((np.arange(0.0, 1.61, 0.1),
        #                                     np.array([1.8, 2.0, 2.5, 3.0, 4.0]), 
        #                                     np.arange(5.0, 10.1, 2.5),
        #                                     np.array([15, 20, 30., 40., 45.]))),
        #                 clear_sky=False,
        #                 overwrite_atm=False,
        #                 overwrite_alb=False,
        #                 overwrite_lrt=True,
        #                 manual_cloud=True,
        #                 manual_cloud_cer=6.9,
        #                 manual_cloud_cwp=0.0231,
        #                 manual_cloud_cth=0.3,
        #                 manual_cloud_cbh=0.101,
        #                 manual_cloud_cot=5.01,
        #                 iter=iter,
        #                 )
    
    # done
    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                 tmhr_ranges_select=[[13.99, 14.18], [14.26, 14.46]],
    #                 case_tag='cloudy_track_4_atm_corr_before',
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
    
    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                 tmhr_ranges_select=[[13.99, 14.18], [14.26, 14.46]],
    #                 case_tag='cloudy_track_4_atm_corr_after',
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
    
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.3400, 15.7583], [15.8403, 16.2653]],
    #                     case_tag='cloudy_track_2_atm_corr',
    #                     config=config,
    #                     simulation_interval=1,
    #                     levels=np.concatenate((np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]),
    #                                         np.array([1.5, 2.0, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                     clear_sky=False,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=8.0,
    #                     manual_cloud_cwp=0.0229,
    #                     manual_cloud_cth=0.47,
    #                     manual_cloud_cbh=0.25,
    #                     manual_cloud_cot=4.3,
    #                     iter=iter,
    #                     )    

    # atm_corr_plot(date=datetime.datetime(2024, 6, 7),
    #                 tmhr_ranges_select=[[15.3400, 15.7583], [15.8403, 16.2653]],
    #                 case_tag='cloudy_track_2_atm_corr',
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),
    #                     tmhr_ranges_select=[[16.076, 16.109],
    #                                         [16.123, 16.255]],
    #                     case_tag='cloudy_track_1_atm_corr',
    #                     config=config,
    #                     simulation_interval=1,
    #                     levels=np.concatenate((np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]),
    #                                         np.array([1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                     clear_sky=False,
    #                     overwrite_atm=True,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=3.4,
    #                     manual_cloud_cwp=0.03209,
    #                     manual_cloud_cth=1.678,
    #                     manual_cloud_cbh=1.262,
    #                     manual_cloud_cot=14.173,
    #                     iter=iter,
    #                     )
    
    # atm_corr_plot(date=datetime.datetime(2024, 6, 11),
    #                 tmhr_ranges_select=[[16.076, 16.109],
    #                                     [16.123, 16.255]],
    #                 case_tag='cloudy_track_1_atm_corr',
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[15.85, 15.882], [16.057, 16.060]],
    #                     case_tag='cloudy_track_1_atm_corr',
    #                     config=config,
    #                     simulation_interval=1,
    #                     levels=np.concatenate((np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]),
    #                                         np.array([1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                     clear_sky=False,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.4,
    #                     manual_cloud_cwp=0.08572,
    #                     manual_cloud_cth=0.637,
    #                     manual_cloud_cbh=0.119,
    #                     manual_cloud_cot=9.57,
    #                     iter=iter,
    #                     )
    
    # atm_corr_plot(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[15.85, 15.882], [16.057, 16.060]],
    #                 case_tag='cloudy_track_1_atm_corr',
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[15.85, 15.882], [16.057, 16.060]],
    #                     case_tag='cloudy_track_2_atm_corr',
    #                     config=config,
    #                     simulation_interval=1,
    #                     levels=np.concatenate((np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]),
    #                                         np.array([1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                     clear_sky=False,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=22.5,
    #                     manual_cloud_cwp=0.03711,
    #                     manual_cloud_cth=0.919,
    #                     manual_cloud_cbh=0.609,
    #                     manual_cloud_cot=2.48,
    #                     iter=iter,
    #                     )
    
    # atm_corr_plot(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[15.85, 15.882], [16.057, 16.060]],
    #                 case_tag='cloudy_track_2_atm_corr',
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[16.0555, 16.0585], [16.207, 16.213]],
    #                     case_tag='cloudy_track_3_atm_corr',
    #                     config=config,
    #                     simulation_interval=1,
    #                     levels=np.concatenate((np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1]),
    #                                         np.array([1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                     clear_sky=False,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=12.5,
    #                     manual_cloud_cwp=0.03308,
    #                     manual_cloud_cth=1.023,
    #                     manual_cloud_cbh=0.677,
    #                     manual_cloud_cot=3.98,
    #                     iter=iter,
    #                     )

    # atm_corr_plot(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[16.057, 16.060], [16.208, 16.214]],
    #                 case_tag='cloudy_track_3_atm_corr',
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
    

    # for iter in range(1):   
    #         flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[
    #                                         [16.557, 16.580], 
    #                                         [16.591, 16.640], 
    #                                         [16.656, 16.740],
    #                                         [16.907, 16.962],
    #                                         [16.972, 16.976],
    #                                         [16.989, 16.995],
    #                                         [17.017, 17.026],
    #                                         [17.067, 17.142],
    #                                         [17.156, 17.206],
    #                                         [17.375, 17.405],
    #                                         ],
    #                     case_tag='clear_track_1_atm_corr',
    #                     config=config,
    #                     simulation_interval=15,
    #                     levels=np.concatenate((np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0,]),
    #                                         np.array([1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=0.0,
    #                     manual_cloud_cwp=0.0,
    #                     manual_cloud_cth=0.0,
    #                     manual_cloud_cbh=0.0,
    #                     manual_cloud_cot=0.0,
    #                     iter=iter,
    #                     )
    
    # done
    # atm_corr_plot(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[
    #                                     [16.557, 16.580], 
    #                                     [16.591, 16.640], 
    #                                     [16.656, 16.740],
    #                                     [16.907, 16.962],
    #                                     [16.972, 16.976],
    #                                     [16.989, 16.995],
    #                                     [17.017, 17.026],
    #                                     [17.067, 17.142],
    #                                     [17.156, 17.206],
    #                                     [17.375, 17.405],
    #                                     ],
    #                 case_tag='clear_track_1_atm_corr',
    #                 simulation_interval=1,
    #                 config=config,
    #                 )
    
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[
    #                                         [14.594, 14.747],
    #                                         [14.760, 14.913], # cloud probably
    #                                         [14.926, 15.062], # cloud probably
    #                                         [15.560, 15.580],
    #                                         [15.593, 15.746],
    #                                         [15.760, 15.912],
    #                                         [16.050, 16.080],
    #                                         [16.093, 16.247], # cloud shadow
    #                                         [16.260, 16.413], # cloud shadow
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=1,
    #                     clear_sky=True,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=0.0,
    #                     manual_cloud_cwp=0.0,
    #                     manual_cloud_cth=0.0,
    #                     manual_cloud_cbh=0.0,
    #                     manual_cloud_cot=0.0,
    #                     iter=iter,
    #                     )
    # sys.exit()
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[
    #                                         [16.251, 16.280], 
    #                                         [16.293, 16.325],
    #                                         [16.704, 16.780],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr_before',
    #                     config=config,
    #                     simulation_interval=10, # in minute
    #                     clear_sky=True,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=0.0,
    #                     manual_cloud_cwp=0.0,
    #                     manual_cloud_cth=0.0,
    #                     manual_cloud_cbh=0.0,
    #                     manual_cloud_cot=0.0,
    #                     iter=iter,
    #                     )
        
    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[
    #                                         [16.251, 16.280], 
    #                                         [16.293, 16.325],
    #                                         [16.704, 16.780],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr_before',
    #                 config=config,
    #                 simulation_interval=1, # in minute
    #                 )
    
    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[
    #                                         [16.251, 16.280], 
    #                                         [16.293, 16.325],
    #                                         [16.704, 16.780],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr_after',
    #                 config=config,
    #                 simulation_interval=1, # in minute
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[
    #                                         [12.84, 12.92],
    #                                         [16.86, 16.93],
    #                                         [17.03, 17.09],
    #                                         [17.31, 17.41],
    #                                         [17.63, 17.69],
    #                                         ],
    #                     case_tag='clear_sky_track_2_atm_corr_after',
    #                     config=config,
    #                     simulation_interval=10, # in minute
    #                     clear_sky=True,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=0.0,
    #                     manual_cloud_cwp=0.0,
    #                     manual_cloud_cth=0.0,
    #                     manual_cloud_cbh=0.0,
    #                     manual_cloud_cot=0.0,
    #                     iter=iter,
    #                     )
    
    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[
    #                                         [16.251, 16.280], 
    #                                         [16.293, 16.325],
    #                                         [16.704, 16.780],
    #                                         ],
    #                 case_tag='clear_sky_track_2_atm_corr_after',
    #                 config=config,
    #                 simulation_interval=1, # in minute
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13), # more popcorn clouds
    #                     tmhr_ranges_select=[[13.0194, 13.0569],
    #                                         [13.0792, 13.0937],
    #                                         [13.1153, 13.1306],
    #                                         [13.1569, 13.1653],
    #                                         [13.1944, 13.2069],
    #                                         [13.2319, 13.2514],
    #                                         [13.2736, 13.2889],
    #                                         [13.3125, 13.3278],
    #                                         [13.3500, 13.3708],
    #                                         [13.3889, 13.4208],
    #                                         [13.4417, 13.4708],
    #                                         [13.5181, 13.5667], # below clouds?
    #                                         ],
    #                     case_tag='clear_sky_spiral_atm_corr',
    #                     config=config,
    #                     simulation_interval=1,
    #                     clear_sky=True,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=0.0,
    #                     manual_cloud_cwp=0.0,
    #                     manual_cloud_cth=0.0,
    #                     manual_cloud_cbh=0.0,
    #                     manual_cloud_cot=0.0,
    #                     iter=iter,
    #                     )
    
    
    
    
    
    # for iter in range(4):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),
    #                     tmhr_ranges_select=[[14.5667, 14.5694],
    #                                         [14.5986, 14.6097],
    #                                         [14.6375, 14.6486], # cloud shadow
    #                                         [14.6778, 14.6903],
    #                                         [14.7208, 14.7403],
    #                                         [14.7653, 14.7875],
    #                                         [14.8125, 14.8278],
    #                                         [14.8542, 14.8736],
    #                                         [14.8986, 14.9389], # more cracks
    #                                         ],
    #                     case_tag='clear_sky_spiral_atm_corr',
    #                     config=config,
    #                     simulation_interval=1,
    #                     clear_sky=True,
    #                     overwrite_atm=False,
    #                     overwrite_alb=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=0.0,
    #                     manual_cloud_cwp=0.0,
    #                     manual_cloud_cth=0.0,
    #                     manual_cloud_cbh=0.0,
    #                     manual_cloud_cot=0.0,
    #                     iter=iter,
    #                     )
    
    # flt_trk_data_collect(date=datetime.datetime(2024, 6, 6),
    #                 tmhr_ranges_select=[[17.022, 17.067],
    #                                     ],
    #                 case_tag='clear_sky_data_collect_2',
    #                 config=config,
    #                 simulation_interval=1/4,
    #                 )
        
    atm_corr_plot(date=datetime.datetime(2024, 6, 6),
                    tmhr_ranges_select=[[17.022, 17.067],
                                        ],
                    case_tag='clear_sky_data_collect_2',
                    config=config,
                    simulation_interval=1/4,
                    aviris_id='ang20240606t161246_002',
                    )
    
 
    # flt_trk_data_collect(date=datetime.datetime(2024, 6, 6),
    #                 tmhr_ranges_select=[[17.0833, 17.0986],
    #                                     [17.1264, 17.1333],
    #                                     [17.1542, 17.1601],
    #                                     [17.1833, 17.1931],
    #                                     [17.2153, 17.2181],
    #                                     [17.2403, 17.2500],
    #                                     ],
    #                 case_tag='clear_sky_spiral_data_collect',
    #                 config=config,
    #                 )
        
    # atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 6),
    #                 tmhr_ranges_select=[[17.0833, 17.0986],
    #                                     [17.1264, 17.1333],
    #                                     [17.1542, 17.1601],
    #                                     [17.1833, 17.1931],
    #                                     [17.2153, 17.2181],
    #                                     [17.2403, 17.2500],
    #                                     ],
    #                 case_tag='clear_sky_spiral_data_collect',
    #                 config=config,
    #                 )
    
        
    # flt_trk_data_collect(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[[13.7889, 13.8010],
    #                                     [13.8350, 13.8395],
    #                                     [13.8780, 13.8885],
    #                                     [13.9240, 13.9255],
    #                                     # [13.9389, 13.9403],
    #                                     [13.9540, 13.9715],
    #                                     [13.9980, 14.0153],
    #                                     # [14.0417, 14.0575],
    #                                     [14.0417, 14.0475],
    #                                     [14.0560, 14.0590],
    #                                     [14.0825, 14.0975],
    #                                     [14.1264, 14.1525],
    #                                     [14.1762, 14.1975],
    #                                     [14.2194, 14.2420],
    #                                     [14.2605, 14.2810]
    #                                     ],
    #                 case_tag='clear_sky_spiral_data_collect',
    #                 config=config,
    #                 )
    
    # atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[[13.7889, 13.8010],
    #                                         [13.8350, 13.8395],
    #                                         [13.8780, 13.8885],
    #                                         [13.9240, 13.9255],
    #                                         # [13.9389, 13.9403],
    #                                         [13.9540, 13.9715],
    #                                         [13.9980, 14.0153],
    #                                         # [14.0417, 14.0575],
    #                                         [14.0417, 14.0475],
    #                                         [14.0560, 14.0590],
    #                                         [14.0825, 14.0975],
    #                                         [14.1264, 14.1525],
    #                                         [14.1762, 14.1975],
    #                                         [14.2194, 14.2420],
    #                                         [14.2605, 14.2810]
    #                                         ],
    #                 case_tag='clear_sky_spiral_data_collect',
    #                 config=config,
    #                 )

        

