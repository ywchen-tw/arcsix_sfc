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
import platform

if platform.system() == 'Linux':
    # Define the path to your module directory
    # Use os.path.abspath and os.path.join for platform independence
    # module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'util'))

    # Add the directory to the Python search path
    # sys.path.insert(0, module_path)
    sys.path.append("/projects/yuch8913/arcsix_sfc/lrt_sim/")  

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
import bisect
import pandas as pd
import xarray as xr
from collections import defaultdict
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


o2a_1_start, o2a_1_end = 748, 780
h2o_1_start, h2o_1_end = 672, 706
h2o_2_start, h2o_2_end = 705, 746
h2o_3_start, h2o_3_end = 884, 996
h2o_4_start, h2o_4_end = 1084, 1175
h2o_5_start, h2o_5_end = 1230, 1286
h2o_6_start, h2o_6_end = 1290, 1509
h2o_7_start, h2o_7_end = 1748, 2050
h2o_8_start, h2o_8_end = 801, 843
final_start, final_end = 2110, 2200

gas_bands = [(o2a_1_start, o2a_1_end), (h2o_1_start, h2o_1_end), (h2o_2_start, h2o_2_end),
                (h2o_3_start, h2o_3_end), (h2o_4_start, h2o_4_end), (h2o_5_start, h2o_5_end),
                (h2o_6_start, h2o_6_end), (h2o_7_start, h2o_7_end), (h2o_8_start, h2o_8_end),
                (final_start, final_end)]

 
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

def atm_corr_plot(date=datetime.datetime(2024, 5, 31),
                     tmhr_ranges_select=[[14.10, 14.27]],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                     simulation_interval: Optional[float] = None,  # in minutes, e.g., 10 for 10 minutes
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
            
            
    print("modis_alb_file:", modis_alb_file)
    
    
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    for i in range(len(tmhr_ranges_select)):
        fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
        
        with open(fname_pkl, 'rb') as f:
            cld_leg = pickle.load(f)
        time = cld_leg['time']
        time_start = tmhr_ranges_select[i][0]
        time_end = tmhr_ranges_select[i][1]
        alt_avg = np.nanmean(data_hsk['alt'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])/1000  # in km
        lon_avg = np.nanmean(data_hsk['lon'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])
        lat_avg = np.nanmean(data_hsk['lat'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])
        lon_all = data_hsk['lon'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)]
        lat_all = data_hsk['lat'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)]
        
    
        ratio_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_0.dat'
        update_1_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_1.dat'
        update_2_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_2.dat'
        update_p3_1_fname = f'{_fdir_general_}/sfc_alb/p3_up_dn_ratio_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}_1.dat'
        update_p3_2_fname = f'{_fdir_general_}/sfc_alb/p3_up_dn_ratio_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}_2.dat'
        
        if "cloudy" in case_tag:
            fdir_spiral = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'
        else:
            fdir_spiral = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
        ori_csv_name = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_0.csv'
        updated_csv_name_1 = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_1.csv'
        updated_csv_name_2 = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_2.csv'
        
        if not os.path.exists(ori_csv_name):
            log.error("Original CSV file not found: %s", ori_csv_name)
            continue
        
        
        # p3_up_dn_ratio_20240605_13.79_13.81_5.80_1.dat
        
        alb_ratio = pd.read_csv(ratio_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        alb_1 = pd.read_csv(update_1_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        alb_2 = pd.read_csv(update_2_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        p3_ratio_1 = pd.read_csv(update_p3_1_fname, delim_whitespace=True, comment='#', names=['wvl', 'ratio'])
        p3_ratio_2 = pd.read_csv(update_p3_2_fname, delim_whitespace=True, comment='#', names=['wvl', 'ratio'])
        
        df_ori = pd.read_csv(ori_csv_name)
        df_upd1 = pd.read_csv(updated_csv_name_1)
        df_upd2 = pd.read_csv(updated_csv_name_2)
        
        if i == 0:
            time_all = []
            fdn_550_all = []
            fup_550_all = []
            fdn_1600_all = []
            fup_1600_all = []
            alb_wvl = alb_ratio['wvl'].values[1:-1] # skip the first value at 348 nm and last value at 2050 nm
            alb_ratio_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            alb1_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            alb2_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            p3_ratio1_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            p3_ratio2_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            lon_avg_all = np.zeros(len(tmhr_ranges_select))
            lat_avg_all = np.zeros(len(tmhr_ranges_select))
            lon_min_all = np.zeros(len(tmhr_ranges_select))
            lon_max_all = np.zeros(len(tmhr_ranges_select))
            lat_min_all = np.zeros(len(tmhr_ranges_select))
            lat_max_all = np.zeros(len(tmhr_ranges_select))
            alt_avg_all = np.zeros(len(tmhr_ranges_select))

            
            flux_wvl = df_ori['wvl'].values
            ssfr_fup_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fup_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_toa_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_toa_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_toa_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            toa_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))

        time_all.append(time)
        ssfr_wvl = cld_leg['ssfr_zen_wvl']
        ssfr_550_ind = np.argmin(np.abs(ssfr_wvl - 550))
        ssfr_1600_ind = np.argmin(np.abs(ssfr_wvl - 1600))
        fdn_550_all.append(cld_leg['ssfr_zen'][:, ssfr_550_ind])
        fup_550_all.append(cld_leg['ssfr_nad'][:, ssfr_550_ind])
        fdn_1600_all.append(cld_leg['ssfr_zen'][:, ssfr_1600_ind])
        fup_1600_all.append(cld_leg['ssfr_nad'][:, ssfr_1600_ind])
        
        print("i:", i)
        print("alb_ratio['alb'] shape:", alb_ratio['alb'].shape)
        print("alb_ratio_all shape:", alb_ratio_all.shape)
        alb_ratio_all[i, :] = alb_ratio['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        alb1_all[i, :] = alb_1['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        alb2_all[i, :] = alb_2['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        p3_ratio1_all[i, :] = p3_ratio_1['ratio'].values
        p3_ratio2_all[i, :] = p3_ratio_2['ratio'].values
        lon_avg_all[i] = lon_avg.copy()
        lat_avg_all[i] = lat_avg.copy()
        lon_min_all[i] = lon_all.min()
        lon_max_all[i] = lon_all.max()
        lat_min_all[i] = lat_all.min()
        lat_max_all[i] = lat_all.max()
        alt_avg_all[i] = alt_avg.copy()
        
        
        ssfr_fup_mean_all[i, :] = df_ori['ssfr_fup_mean'].values
        ssfr_fdn_mean_all[i, :] = df_ori['ssfr_fdn_mean'].values
        ssfr_fup_std_all[i, :] = df_ori['ssfr_fup_std'].values
        ssfr_fdn_std_all[i, :] = df_ori['ssfr_fdn_std'].values
        simu_fup_mean_all_iter0[i, :] = df_ori['simu_fup_mean'].values
        simu_fdn_mean_all_iter0[i, :] = df_ori['simu_fdn_mean'].values
        simu_fup_mean_all_iter1[i, :] = df_upd1['simu_fup_mean'].values
        simu_fdn_mean_all_iter1[i, :] = df_upd1['simu_fdn_mean'].values
        simu_fup_mean_all_iter2[i, :] = df_upd2['simu_fup_mean'].values
        simu_fdn_mean_all_iter2[i, :] = df_upd2['simu_fdn_mean'].values
        simu_fup_toa_mean_all_iter0[i, :] = df_ori['simu_fup_toa_mean'].values
        simu_fup_toa_mean_all_iter1[i, :] = df_upd1['simu_fup_toa_mean'].values
        simu_fup_toa_mean_all_iter2[i, :] = df_upd2['simu_fup_toa_mean'].values
        toa_mean_all[i, :] = df_ori['toa_mean'].values
        
        
        print(f"date_s: {date_s}, time: {time_start:.2f}-{time_end:.2f}, alt_avg: {alt_avg:.2f} km")
    
    
    
        
    
        # find the modis location closest to the flight leg center
        if modis_alb_file is not None:
            dist = np.sqrt((modis_lon - lon_avg)**2 + (modis_lat - lat_avg)**2)
            min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            modis_alb_leg = modis_sur_alb[min_idx[0], min_idx[1], :7]
            modis_alb_legs.append(modis_alb_leg)
    
    
    time_all = np.array(time_all)
    fdn_550_all = np.array(fdn_550_all)
    fup_550_all = np.array(fup_550_all)
    fdn_1600_all = np.array(fdn_1600_all)
    fup_1600_all = np.array(fup_1600_all)

    print("lon avg all mean:", lon_avg_all.mean())
    print("lat avg all mean:", lat_avg_all.mean())
    
    select = np.full(len(tmhr_ranges_select), True)
    # remove invalid time
    for i in range(len(tmhr_ranges_select)):
        if np.all(alb_ratio_all[i, :] == 0):
            select[i] = False
            
    for data_arr in [alb_ratio_all, alb1_all, alb2_all, p3_ratio1_all, p3_ratio2_all, 
                     lon_avg_all, lat_avg_all, lon_min_all, lon_max_all, lat_min_all, lat_max_all, alt_avg_all,
                     ssfr_fup_mean_all, ssfr_fdn_mean_all, ssfr_fup_std_all, ssfr_fdn_std_all, 
                     simu_fup_mean_all_iter0, simu_fdn_mean_all_iter0,
                     simu_fup_mean_all_iter1, simu_fdn_mean_all_iter1,
                     simu_fup_mean_all_iter2, simu_fdn_mean_all_iter2, 
                     simu_fup_toa_mean_all_iter0, simu_fup_toa_mean_all_iter1, simu_fup_toa_mean_all_iter2, toa_mean_all]:
        data_arr = data_arr[select]
        
    broadband_alb_iter0 = np.sum(alb_ratio_all * toa_mean_all, axis=1) / np.sum(toa_mean_all, axis=1)
    broadband_alb_iter1 = np.sum(alb1_all * toa_mean_all, axis=1) / np.sum(toa_mean_all, axis=1)
    broadband_alb_iter2 = np.sum(alb2_all * toa_mean_all, axis=1) / np.sum(toa_mean_all, axis=1)
    gas_mask = np.isfinite(gas_abs_masking(alb_wvl, np.ones_like(alb_wvl, dtype=float)), alt=1)
    broadband_alb_iter0_filter = np.sum(alb_ratio_all[:, gas_mask] * toa_mean_all[:, gas_mask], axis=1) / np.sum(toa_mean_all[:, gas_mask], axis=1)
    broadband_alb_iter1_filter = np.sum(alb1_all[:, gas_mask] * toa_mean_all[:, gas_mask], axis=1) / np.sum(toa_mean_all[:, gas_mask], axis=1)
    broadband_alb_iter2_filter = np.sum(alb2_all[:, gas_mask] * toa_mean_all[:, gas_mask], axis=1) / np.sum(toa_mean_all[:, gas_mask], axis=1)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(lon_avg_all, broadband_alb_iter0, 'o-', label='Iteration 0')
    ax.plot(lon_avg_all, broadband_alb_iter1, 's-', label='Iteration 1')
    ax.plot(lon_avg_all, broadband_alb_iter2, 'd-', label='Iteration 2')
    ax.plot(lon_avg_all, broadband_alb_iter0_filter, 'o-.', label='Iteration 0 (gas masked)')
    ax.plot(lon_avg_all, broadband_alb_iter1_filter, 's--', label='Iteration 1 (gas masked)')
    ax.plot(lon_avg_all, broadband_alb_iter2_filter, 'd--', label='Iteration 2 (gas masked)')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Broadband Surface Albedo', fontsize=12)
    ax.set_title(f'Broadband Surface Albedo vs Longitude {date_s} {case_tag}', fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f'fig/{date_s}/{date_s}_{case_tag}_broadband_albedo_vs_longitude.png', bbox_inches='tight', dpi=150)
    log.info("Saved broadband albedo vs longitude plot to fig/%s/%s_%s_broadband_albedo_vs_longitude.png", date_s, date_s, case_tag)
    
    # save alb1 and alb2 to a pkl file
    alb_update_dict = {
        'wvl': alb_wvl,
        'alb_iter0': alb_ratio_all,
        'alb_iter1': alb1_all,
        'alb_iter2': alb2_all,
        'p3_up_dn_ratio_1': p3_ratio1_all,
        'p3_up_dn_ratio_2': p3_ratio2_all,
        'lon_avg': lon_avg_all,
        'lat_avg': lat_avg_all,
        'lon_min': lon_min_all,
        'lon_max': lon_max_all,
        'lat_min': lat_min_all,
        'lat_max': lat_max_all,
        'alt_avg': alt_avg_all,
        'modis_alb_legs': np.array(modis_alb_legs) if modis_alb_file is not None else None,
        'modis_bands_nm': modis_bands_nm if modis_alb_file is not None else None,
    }
    if modis_alb_file is None:
        modis_bands_nm = None
        modis_alb_legs = None
    with open(f'{_fdir_general_}/sfc_alb/sfc_alb_update_{date_s}_{case_tag}.pkl', 'wb') as f:
        pickle.dump(alb_update_dict, f)
    log.info(f"Saved surface albedo updates to {_fdir_general_}/sfc_alb/sfc_alb_update_{date_s}_{case_tag}.pkl")


    
        
    print("lon avg all mean:", lon_avg_all.mean())
    print("lat avg all mean:", lat_avg_all.mean())


    # Create a ScalarMappable
    data_min, data_max = np.log(alt_avg_all/alt_avg_all.max()).min(), np.log(alt_avg_all/alt_avg_all.max()).max()
    norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
    cmap = cm.jet # Or any other built-in colormap like cm.viridis
    s_m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_series = s_m.to_rgba(np.log(alt_avg_all/alt_avg_all.max()))               
    
    ssfr_alb_plot(date_s, tmhr_ranges_select, alb_wvl, alb_ratio_all, color_series, 
                   alt_avg_all,
                   modis_bands_nm, modis_alb_legs, modis_alb_file,
                   case_tag,
                   ylabel='SSFR upward/downward ratio',
                   title='SSFR measurement',
                   suptitle='SSFR upward/downward ratio Comparison',
                   file_description='SSFR_ratio',
                   lon_avg_all=lon_avg_all,
                   lat_avg_all=lat_avg_all,
                   )
        
    ssfr_alb_plot(date_s, tmhr_ranges_select, alb_wvl, alb1_all, color_series, 
                   alt_avg_all,
                   modis_bands_nm, modis_alb_legs, modis_alb_file,
                   case_tag,
                   ylabel='Atmospheric corrected Surface Albedo',
                   title='Albedo used: atmospheric corrected surface albedo (smooth and fit)',
                   suptitle='Atmospheric corrected Surface Albedo (smooth and fit) Comparison',
                   file_description='corrected_alb',
                   lon_avg_all=lon_avg_all,
                   lat_avg_all=lat_avg_all,
                   )
    
    ssfr_alb_plot(date_s, tmhr_ranges_select, alb_wvl, alb2_all, color_series, 
                   alt_avg_all,
                   modis_bands_nm, modis_alb_legs, modis_alb_file,
                   case_tag,
                   ylabel='Atmospheric corrected Surface Albedo (smooth and fit)',
                   title='Albedo used: atmospheric corrected surface albedo (smooth and fit)',
                   suptitle='Atmospheric corrected Surface Albedo (smooth and fit) Comparison',
                   file_description='corrected_alb_smooth_fit',
                   lon_avg_all=lon_avg_all,
                   lat_avg_all=lat_avg_all,
                   )
    
    
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
    
    
    toa_alt = 550  # km
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '-x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.scatter(toa_mean_all[0, wvl_idx], alt_avg_all, '-x', label='TOA', color=color)
        # ax1.boxplot(simu_fup_toa_mean_all_iter0[:, wvl_idx], positions=[alt_avg_all.mean()], widths=0.3, vert=True, patch_artist=True, label='Sim Iter0 TOA {wvl} nm', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax1.plot(simu_fdn_mean_all_iter0[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter0 {wvl} nm', color=color, alpha=0.7)
        
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax2.plot(simu_fup_mean_all_iter0[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter0 {wvl} nm', color=color, alpha=0.7)
        
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax3.plot(simu_fup_mean_all_iter0[:, wvl_idx]/simu_fdn_mean_all_iter0[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter0 {wvl} nm', color=color, alpha=0.7)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_in_alt_all_iter0.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '--x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax1.plot(simu_fdn_mean_all_iter1[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter1 {wvl} nm', color=color, alpha=0.7)
        
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax2.plot(simu_fup_mean_all_iter1[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter1 {wvl} nm', color=color, alpha=0.7)
        
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax3.plot(simu_fup_mean_all_iter1[:, wvl_idx]/simu_fdn_mean_all_iter1[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter1 {wvl} nm', color=color, alpha=0.7)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_in_alt_all_iter1.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '--x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax1.plot(simu_fdn_mean_all_iter2[:, wvl_idx], alt_avg_all, '--x', label=f'Sim Iter2 {wvl} nm', color=color, alpha=0.7)
        
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax2.plot(simu_fup_mean_all_iter2[:, wvl_idx], alt_avg_all, '--x', label=f'Sim Iter12 {wvl} nm', color=color, alpha=0.7)
        
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax3.plot(simu_fup_mean_all_iter2[:, wvl_idx]/simu_fdn_mean_all_iter2[:, wvl_idx], alt_avg_all, '--x', label=f'Sim Iter2 {wvl} nm', color=color, alpha=0.7)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_in_alt_all_iter2.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sim0_obs_dn_diff_perc = (simu_fdn_mean_all_iter0 - ssfr_fdn_mean_all)/ssfr_fdn_mean_all * 100
    for alt_ind in range(len(tmhr_ranges_select)):
        ax.plot(flux_wvl, sim0_obs_dn_diff_perc[alt_ind, :], '-', color=color_series[alt_ind], 
                label='Z=%.2fkm, lon:%.2f to %.2f' % (alt_avg_all[alt_ind], lon_min_all[alt_ind], lon_max_all[alt_ind]))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Sim-Obs Down/Obs (%)', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.axhline(0, color='k', linestyle='--', alpha=0.7)
    ax.set_ylim(-50, 50)
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    # ax.set_title(f'Simulated - Observed Downward Flux Difference Percentage\nalb=SSFR up/down flux ratio', fontsize=13)
    fig.suptitle(f'SSFR Downward Flux Difference Percentage Comparison {date_s}', fontsize=16, y=0.98)
    fig.suptitle(f'Downward Flux Sim vs Obs {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_dn_diff_perc_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    
    
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
        alt_avg = np.nanmean(data_hsk['alt'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])/1000  # in km
        lon_avg = np.nanmean(data_hsk['lon'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])
        lat_avg = np.nanmean(data_hsk['lat'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])
        lon_all = data_hsk['lon'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)]
        lat_all = data_hsk['lat'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)]
        
    
        ratio_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_0.dat'
        update_1_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_1.dat'
        update_2_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_2.dat'
        update_p3_1_fname = f'{_fdir_general_}/sfc_alb/p3_up_dn_ratio_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}_1.dat'
        update_p3_2_fname = f'{_fdir_general_}/sfc_alb/p3_up_dn_ratio_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}_2.dat'
        
        if "cloudy" in case_tag:
            fdir_spiral = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'
        else:
            fdir_spiral = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
        ori_csv_name = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_0.csv'
        updated_csv_name_1 = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_1.csv'
        updated_csv_name_2 = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_2.csv'
        
        if not os.path.exists(ori_csv_name):
            log.error("Original CSV file not found: %s", ori_csv_name)
            continue
        
        
        # p3_up_dn_ratio_20240605_13.79_13.81_5.80_1.dat
        
        alb_ratio = pd.read_csv(ratio_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        alb_1 = pd.read_csv(update_1_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        alb_2 = pd.read_csv(update_2_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        p3_ratio_1 = pd.read_csv(update_p3_1_fname, delim_whitespace=True, comment='#', names=['wvl', 'ratio'])
        p3_ratio_2 = pd.read_csv(update_p3_2_fname, delim_whitespace=True, comment='#', names=['wvl', 'ratio'])
        
        df_ori = pd.read_csv(ori_csv_name)
        df_upd1 = pd.read_csv(updated_csv_name_1)
        df_upd2 = pd.read_csv(updated_csv_name_2)
        
        if i == 0:
            time_all = []
            fdn_550_all = []
            fup_550_all = []
            fdn_1600_all = []
            fup_1600_all = []
            alb_wvl = alb_ratio['wvl'].values[1:-1] # skip the first value at 348 nm and last value at 2050 nm
            alb_ratio_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            alb1_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            alb2_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            p3_ratio1_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            p3_ratio2_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            lon_avg_all = np.zeros(len(tmhr_ranges_select))
            lat_avg_all = np.zeros(len(tmhr_ranges_select))
            lon_min_all = np.zeros(len(tmhr_ranges_select))
            lon_max_all = np.zeros(len(tmhr_ranges_select))
            lat_min_all = np.zeros(len(tmhr_ranges_select))
            lat_max_all = np.zeros(len(tmhr_ranges_select))
            alt_avg_all = np.zeros(len(tmhr_ranges_select))

            
            flux_wvl = df_ori['wvl'].values
            ssfr_fup_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fup_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_toa_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_toa_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_toa_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            toa_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))

        time_all.append(time)
        ssfr_wvl = cld_leg['ssfr_zen_wvl']
        ssfr_550_ind = np.argmin(np.abs(ssfr_wvl - 550))
        ssfr_1600_ind = np.argmin(np.abs(ssfr_wvl - 1600))
        fdn_550_all.append(cld_leg['ssfr_zen'][:, ssfr_550_ind])
        fup_550_all.append(cld_leg['ssfr_nad'][:, ssfr_550_ind])
        fdn_1600_all.append(cld_leg['ssfr_zen'][:, ssfr_1600_ind])
        fup_1600_all.append(cld_leg['ssfr_nad'][:, ssfr_1600_ind])
        
        print("i:", i)
        print("alb_ratio['alb'] shape:", alb_ratio['alb'].shape)
        print("alb_ratio_all shape:", alb_ratio_all.shape)
        alb_ratio_all[i, :] = alb_ratio['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        alb1_all[i, :] = alb_1['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        alb2_all[i, :] = alb_2['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        p3_ratio1_all[i, :] = p3_ratio_1['ratio'].values
        p3_ratio2_all[i, :] = p3_ratio_2['ratio'].values
        lon_avg_all[i] = lon_avg.copy()
        lat_avg_all[i] = lat_avg.copy()
        lon_min_all[i] = lon_all.min()
        lon_max_all[i] = lon_all.max()
        lat_min_all[i] = lat_all.min()
        lat_max_all[i] = lat_all.max()
        alt_avg_all[i] = alt_avg.copy()
        
        
        ssfr_fup_mean_all[i, :] = df_ori['ssfr_fup_mean'].values
        ssfr_fdn_mean_all[i, :] = df_ori['ssfr_fdn_mean'].values
        ssfr_fup_std_all[i, :] = df_ori['ssfr_fup_std'].values
        ssfr_fdn_std_all[i, :] = df_ori['ssfr_fdn_std'].values
        simu_fup_mean_all_iter0[i, :] = df_ori['simu_fup_mean'].values
        simu_fdn_mean_all_iter0[i, :] = df_ori['simu_fdn_mean'].values
        simu_fup_mean_all_iter1[i, :] = df_upd1['simu_fup_mean'].values
        simu_fdn_mean_all_iter1[i, :] = df_upd1['simu_fdn_mean'].values
        simu_fup_mean_all_iter2[i, :] = df_upd2['simu_fup_mean'].values
        simu_fdn_mean_all_iter2[i, :] = df_upd2['simu_fdn_mean'].values
        simu_fup_toa_mean_all_iter0[i, :] = df_ori['simu_fup_toa_mean'].values
        simu_fup_toa_mean_all_iter1[i, :] = df_upd1['simu_fup_toa_mean'].values
        simu_fup_toa_mean_all_iter2[i, :] = df_upd2['simu_fup_toa_mean'].values
        toa_mean_all[i, :] = df_ori['toa_mean'].values
        
        
        print(f"date_s: {date_s}, time: {time_start:.2f}-{time_end:.2f}, alt_avg: {alt_avg:.2f} km")
    
    time_all = np.array(time_all)
    fdn_550_all = np.array(fdn_550_all)
    fup_550_all = np.array(fup_550_all)
    fdn_1600_all = np.array(fdn_1600_all)
    fup_1600_all = np.array(fup_1600_all)

    print("lon avg all mean:", lon_avg_all.mean())
    print("lat avg all mean:", lat_avg_all.mean())
    
    select = np.full(len(tmhr_ranges_select), True)
    # remove invalid time
    for i in range(len(tmhr_ranges_select)):
        if np.all(alb_ratio_all[i, :] == 0):
            select[i] = False
            
    for data_arr in [alb_ratio_all, alb1_all, alb2_all, p3_ratio1_all, p3_ratio2_all, 
                     lon_avg_all, lat_avg_all, lon_min_all, lon_max_all, lat_min_all, lat_max_all, alt_avg_all,
                     ssfr_fup_mean_all, ssfr_fdn_mean_all, ssfr_fup_std_all, ssfr_fdn_std_all, 
                     simu_fup_mean_all_iter0, simu_fdn_mean_all_iter0,
                     simu_fup_mean_all_iter1, simu_fdn_mean_all_iter1,
                     simu_fup_mean_all_iter2, simu_fdn_mean_all_iter2, 
                     simu_fup_toa_mean_all_iter0, simu_fup_toa_mean_all_iter1, simu_fup_toa_mean_all_iter2, toa_mean_all]:
        data_arr = data_arr[select]
            
        
    

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
        
    
    # find the modis location closest to the flight leg center
    if modis_alb_file is not None:
        dist = np.sqrt((modis_lon - lon_avg)**2 + (modis_lat - lat_avg)**2)
        min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        modis_alb_leg = modis_sur_alb[min_idx[0], min_idx[1], :7]
        modis_bands_nm = np.array([float(i) for i in modis_bands[:7]])*1000  # in nm
    else:
        modis_bands_nm = None
        modis_alb_leg = None


    # Create a ScalarMappable
    data_min, data_max = np.log(alt_avg_all/alt_avg_all.max()).min(), np.log(alt_avg_all/alt_avg_all.max()).max()
    norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
    cmap = cm.jet # Or any other built-in colormap like cm.viridis
    s_m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_series = s_m.to_rgba(np.log(alt_avg_all/alt_avg_all.max()))
    
    ssfr_up_dn_ratio_plot(date_s, tmhr_ranges_select, alb_wvl, alb_ratio_all, color_series, alt_avg_all, case_tag,
                          albedo_used='albedo used: SSFR upward/downward ratio',
                          file_suffix='')

    ssfr_up_dn_ratio_plot(date_s, tmhr_ranges_select, alb_wvl, p3_ratio1_all, color_series, alt_avg_all, case_tag,
                          albedo_used='albedo used: Atmospheric corrected albedo',
                          file_suffix='_updated_1')
    
    ssfr_up_dn_ratio_plot(date_s, tmhr_ranges_select, alb_wvl, p3_ratio2_all, color_series, alt_avg_all, case_tag,
                          albedo_used='albedo used: Atmospheric corrected albedo (smoothed)',
                          file_suffix='_updated_2')
    
    ssfr_alb_plot(date_s, tmhr_ranges_select, alb_wvl, alb_ratio_all, color_series, 
                   alt_avg_all,
                   modis_bands_nm, modis_alb_leg, modis_alb_file,
                   case_tag,
                   ylabel='SSFR upward/downward ratio',
                   title='SSFR measurement',
                   suptitle='SSFR upward/downward ratio Comparison',
                   file_description='SSFR_ratio_aviris',
                   lon_avg_all=lon_avg_all,
                   lat_avg_all=lat_avg_all,
                   aviris_file=aviris_file,
                   aviris_closest=aviris_closest,
                   aviris_reflectance_wvl=aviris_reflectance_wvl,
                   aviris_reflectance_spectrum=aviris_reflectance_spectrum,
                   aviris_reflectance_spectrum_unc=aviris_reflectance_spectrum_unc)
        
    ssfr_alb_plot(date_s, tmhr_ranges_select, alb_wvl, alb1_all, color_series, 
                   alt_avg_all,
                   modis_bands_nm, modis_alb_leg, modis_alb_file,
                   case_tag,
                   ylabel='Atmospheric corrected Surface Albedo',
                   title='Albedo used: atmospheric corrected surface albedo (smooth and fit)',
                   suptitle='Atmospheric corrected Surface Albedo (smooth and fit) Comparison',
                   file_description='corrected_alb_aviris',
                   lon_avg_all=lon_avg_all,
                   lat_avg_all=lat_avg_all,
                   aviris_file=aviris_file,
                   aviris_closest=aviris_closest,
                   aviris_reflectance_wvl=aviris_reflectance_wvl,
                   aviris_reflectance_spectrum=aviris_reflectance_spectrum,
                   aviris_reflectance_spectrum_unc=aviris_reflectance_spectrum_unc)
    
    ssfr_alb_plot(date_s, tmhr_ranges_select, alb_wvl, alb2_all, color_series, 
                   alt_avg_all,
                   modis_bands_nm, modis_alb_leg, modis_alb_file,
                   case_tag,
                   ylabel='Atmospheric corrected Surface Albedo (smooth and fit)',
                   title='Albedo used: atmospheric corrected surface albedo (smooth and fit)',
                   suptitle='Atmospheric corrected Surface Albedo (smooth and fit) Comparison',
                   file_description='corrected_alb_smooth_fit_aviris',
                   lon_avg_all=lon_avg_all,
                   lat_avg_all=lat_avg_all,
                   aviris_file=aviris_file,
                   aviris_closest=aviris_closest,
                   aviris_reflectance_wvl=aviris_reflectance_wvl,
                   aviris_reflectance_spectrum=aviris_reflectance_spectrum,
                   aviris_reflectance_spectrum_unc=aviris_reflectance_spectrum_unc)
    
     
    # output_dict = {
    #             'wvl': cld_leg['ssfr_zen_wvl'],
    #             'ssfr_fup_mean': fup_mean,
    #             'ssfr_fdn_mean': fdn_mean,
    #             'ssfr_fup_std': fup_std,
    #             'ssfr_fdn_std': fdn_std,
    #             'simu_fup_mean': Fup_p3_mean_interp,
    #             'simu_fdn_mean': Fdn_p3_mean_interp,
    #             'toa_mean':
    #         }
            
    #         output_df = pd.DataFrame(output_dict)
    #         output_df.to_csv(f'{fdir}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_{iter}.csv', index=False)
    
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
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '--x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '--x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax1.plot(simu_fdn_mean_all_iter0[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter0 {wvl} nm', color=color, alpha=0.7)
        
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax2.plot(simu_fup_mean_all_iter0[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter0 {wvl} nm', color=color, alpha=0.7)
        
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax3.plot(simu_fup_mean_all_iter0[:, wvl_idx]/simu_fdn_mean_all_iter0[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter0 {wvl} nm', color=color, alpha=0.7)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_in_alt_all_iter0.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '--x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax1.plot(simu_fdn_mean_all_iter1[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter1 {wvl} nm', color=color, alpha=0.7)
        
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax2.plot(simu_fup_mean_all_iter1[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter1 {wvl} nm', color=color, alpha=0.7)
        
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax3.plot(simu_fup_mean_all_iter1[:, wvl_idx]/simu_fdn_mean_all_iter1[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter1 {wvl} nm', color=color, alpha=0.7)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_in_alt_all_iter1.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '--x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax1.plot(simu_fdn_mean_all_iter2[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter2 {wvl} nm', color=color, alpha=0.7)
        
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax2.plot(simu_fup_mean_all_iter2[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter12 {wvl} nm', color=color, alpha=0.7)
        
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax3.plot(simu_fup_mean_all_iter2[:, wvl_idx]/simu_fdn_mean_all_iter2[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter2 {wvl} nm', color=color, alpha=0.7)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_in_alt_all_iter2.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sim0_obs_dn_diff_perc = (simu_fdn_mean_all_iter0 - ssfr_fdn_mean_all)/ssfr_fdn_mean_all * 100
    for alt_ind in range(len(tmhr_ranges_select)):
        ax.plot(flux_wvl, sim0_obs_dn_diff_perc[alt_ind, :], '-', color=color_series[alt_ind], label='Z=%.2fkm' % (alt_avg_all[alt_ind]))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Sim-Obs Down/Obs (%)', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.axhline(0, color='k', linestyle='--', alpha=0.7)
    ax.set_ylim(-50, 50)
    ax.tick_params(labelsize=12)
    # ax.set_title(f'Simulated - Observed Downward Flux Difference Percentage\nalb=SSFR up/down flux ratio', fontsize=13)
    fig.suptitle(f'SSFR Downward Flux Difference Percentage Comparison {date_s}', fontsize=16, y=0.98)
    fig.suptitle(f'Downward Flux Sim vs Obs {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_dn_diff_perc_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    

    
        
    
    fig, ((ax1, ax2, ax3),
          (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey='row')
    for ax in [ax1, ax2, ax3]:
        ax.plot(flux_wvl, toa_mean_all[0, :], '--', color='gray', label='TOA')
        
    for alt_ind in range(len(tmhr_ranges_select)):
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(labelsize=12)
            ax.errorbar(flux_wvl, ssfr_fdn_mean_all[alt_ind, :], yerr=ssfr_fup_std_all[alt_ind, :], 
                        # fmt='o', 
                        color=color_series[alt_ind], markersize=4, label='Observed Fup Z=%.2fkm' % (alt_avg_all[alt_ind]))
            
            
        for ax in [ax4, ax5, ax6]:
            ax.tick_params(labelsize=12)
            ax.errorbar(flux_wvl, ssfr_fup_mean_all[alt_ind, :], yerr=ssfr_fdn_std_all[alt_ind, :], 
                        # fmt='o', 
                        color=color_series[alt_ind], markersize=4, label='Observed Fdn Z=%.2fkm' % (alt_avg_all[alt_ind]))
    
        ax1.plot(flux_wvl, simu_fdn_mean_all_iter0[alt_ind, :], 'r-', label='Simulated Fup Z=%.2fkm' % (alt_avg_all[alt_ind]))
        ax2.plot(flux_wvl, simu_fdn_mean_all_iter1[alt_ind, :], 'r-', label='Simulated Fup Z=%.2fkm' % (alt_avg_all[alt_ind]))
        ax3.plot(flux_wvl, simu_fdn_mean_all_iter2[alt_ind, :], 'r-', label='Simulated Fup Z=%.2fkm' % (alt_avg_all[alt_ind]))
        
        ax4.plot(flux_wvl, simu_fup_mean_all_iter0[alt_ind, :], 'g-', label='Simulated Fdn Z=%.2fkm' % (alt_avg_all[alt_ind]))
        ax5.plot(flux_wvl, simu_fup_mean_all_iter1[alt_ind, :], 'g-', label='Simulated Fdn Z=%.2fkm' % (alt_avg_all[alt_ind]))
        ax6.plot(flux_wvl, simu_fup_mean_all_iter2[alt_ind, :], 'g-', label='Simulated Fdn Z=%.2fkm' % (alt_avg_all[alt_ind]))
        
    ax1.set_title('SSFR Downward Flux\nalb=SSFR up/down flux ratio', fontsize=12)
    ax2.set_title('SSFR Downward Flux\nalb=updated surface albedo (1)', fontsize=12)
    ax3.set_title('SSFR Downward Flux\nalb=updated surface albedo (2)', fontsize=12)
    ax4.set_title('SSFR Upward Flux\nalb=SSFR up/down flux ratio', fontsize=12)
    ax5.set_title('SSFR Upward Flux\nalb=updated surface albedo (1)', fontsize=12)
    ax6.set_title('SSFR Upward Flux\nalb=updated surface albedo (2)', fontsize=12)
    
    ax4.set_xlabel('Wavelength (nm)', fontsize=14)
    ax5.set_xlabel('Wavelength (nm)', fontsize=14)
    ax6.set_xlabel('Wavelength (nm)', fontsize=14)
    
    ax1.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax4.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.), ncol=1)
    # ax6.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
            


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


from enum import IntFlag, auto
class ssfr_flags(IntFlag):
    pitcth_roll_exceed_threshold = auto()  #condition when the pitch or roll angle exceeded a certain threshold
    camera_icing = auto()  #condition when the camera experienced icing issues at the moment of measurement
    camera_icing_pre = auto()  #condition when the camera experienced icing issues within 1 hour prior to the moment of measurement
    zen_toa_over_threshold = auto()  #condition when the zenith TOA irradiance is over a certain threshold 
    alp_ang_pit_rol_issue = auto()  #condition when the leveling platform angle is over a certain threshold
    
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

def gas_abs_masking(wvl, alb, alt):
    o2a_1_start, o2a_1_end = 748, 780
    h2o_1_start, h2o_1_end = 672, 706
    h2o_2_start, h2o_2_end = 705, 746
    h2o_3_start, h2o_3_end = 884, 996
    h2o_4_start, h2o_4_end = 1084, 1175
    h2o_5_start, h2o_5_end = 1230, 1286
    h2o_6_start, h2o_6_end = 1290, 1509
    h2o_7_start, h2o_7_end = 1748, 2050
    h2o_8_start, h2o_8_end = 801, 843
    final_start, final_end = 2110, 2200
    
    effective_mask_ = np.ones_like(alb)
    alb_mask = alb.copy()
    if alt > 0.5:
        alb_mask[
                ((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
                ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
                ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
                ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
                ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
                ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
                ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
                ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
                ((wvl>=h2o_8_start) & (wvl<=h2o_8_end)) |
                ((wvl>=final_start) & (wvl<=final_end))
                ] = np.nan
        effective_mask_[
                ((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
                ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
                ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
                ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
                ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
                ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
                ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
                ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
                ((wvl>=h2o_8_start) & (wvl<=h2o_8_end)) |
                ((wvl>=final_start) & (wvl<=final_end))
                ] = np.nan
    else: 
        # Not mask O2 band and water abs band at VIS and NIR if altitude is low
        alb_mask[
                # ((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
                # ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
                # ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
                # ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
                ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
                ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
                ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
                ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
                # ((wvl>=h2o_8_start) & (wvl<=h2o_8_end)) |
                ((wvl>=final_start) & (wvl<=final_end))
                ] = np.nan
        effective_mask_[
                # ((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
                # ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
                # ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
                # ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
                ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
                ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
                ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
                ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
                # ((wvl>=h2o_8_start) & (wvl<=h2o_8_end)) |
                ((wvl>=final_start) & (wvl<=final_end))
                ] = np.nan
    
    # interpolation if nan in effective_mask_ range
    if np.sum(~np.isnan(effective_mask_)) != np.isfinite(alb_mask).sum():
        eff_wvl_real_mask = np.logical_and(~np.isnan(effective_mask_), np.isfinite(alb_mask))
        fit_wvl_mask = np.logical_and(~np.isnan(effective_mask_), np.isnan(alb_mask))
        # effective_mask_func = interp1d(wvl[eff_wvl_real_mask], effective_mask_[eff_wvl_real_mask], bounds_error=False, fill_value=np.nan)
        
        
        # alb_mask[fit_wvl_mask] = effective_mask_func(wvl[fit_wvl_mask])
        
        s = pd.Series(alb_mask[effective_mask_==1])
        s_mask = np.isnan(alb_mask[effective_mask_==1])
        # Fills NaN with the value immediately preceding it
        s_ffill = s.fillna(method='ffill', limit=2)
        s_ffill = s_ffill.fillna(method='bfill', limit=2)
        while np.any(np.isnan(s_ffill)):
            s_ffill = s_ffill.fillna(method='ffill', limit=2)
            s_ffill = s_ffill.fillna(method='bfill', limit=2)
        
        
        alb_mask[fit_wvl_mask] = np.array(s_ffill)[s_mask]
        
        
    
    return alb_mask
    
def find_best_fit(model_library, obs_wvl, obs_albedo):
    """
    Finds the best-fit model spectrum from a library by minimizing RMSE
    at *only* the provided obs_wvl points.
    """
    
    best_fit_params = None
    best_fit_spectrum = None
    min_rmse = np.inf
    
    obs_wvl_nanmask = np.isfinite(obs_albedo)
    
    # Loop through every pre-run model spectrum in your library
    for key in model_library.keys():
        
        model_run = model_library[key]
        model_wvl = model_run['wvl']      # Full wvl (0.2-5.0 um)
        model_wvl *= 1000  # Convert to nm
        model_albedo = model_run['albedo']  # Full albedo spectrum
        
        # -----------------------------------------------------------------
        # THIS IS THE KEY STEP:
        # It takes the full model (model_wvl, model_albedo) and
        # interpolates it, pulling out *only* the values at the
        # exact points you have in obs_wvl.
        # -----------------------------------------------------------------
        
        interpolated_model_albedo = np.interp(obs_wvl, model_wvl, model_albedo)
        
        # The gaps (0.9-1.1, 1.3-1.5) are automatically
        # and correctly ignored in the next step.
        
        # 3. Calculate RMSE *only* on the valid, non-gap data
        rmse = np.sqrt(np.mean((interpolated_model_albedo[obs_wvl_nanmask] - obs_albedo[obs_wvl_nanmask])**2))
        
        # 4. Check if this is the best fit so far
        if rmse < min_rmse:
            min_rmse = rmse
            best_fit_params = key
            # best_fit_spectrum = model_run # Store the whole run
            best_fit_spectrum = interpolated_model_albedo # Store the interpolation
        
    # plt.close('all')
    # plt.plot(obs_wvl, best_fit_spectrum, '-', color='r', label='Best Fit Model')
    # plt.plot(obs_wvl, obs_albedo, '-', color='k', label='Observed')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.title(f'Best Fit Model: {best_fit_params}, RMSE: {min_rmse:.4f}')
    # plt.show()
    
        
    return best_fit_params, best_fit_spectrum, min_rmse

def snowice_alb_fitting(alb_wvl, alb_corr, alt, clear_sky=False):
    # snicar_albedo_list = []
    if clear_sky:
        snicar_filename = 'snicar_model_results_direct.pkl'
    else:
        snicar_filename = 'snicar_model_results_diffuse.pkl'
    with open(snicar_filename, 'rb') as f:
        snicar_data = pickle.load(f)
    #     wvl = list(snicar_data.values())[0]['wvl']
    #     for key in snicar_data:
    #         snicar_albedo_list.append((key, snicar_data[key]['albedo']))
    # snicar_albedo_arr = np.array(snicar_albedo_list)  
          
    alb_corr_mask = alb_corr.copy()
    alb_corr_mask = gas_abs_masking(alb_wvl, alb_corr_mask, alt=alt)
    best_fit_key, best_fit_spectrum, min_rmse = find_best_fit(
        model_library=snicar_data,
        obs_wvl=alb_wvl,
        obs_albedo=alb_corr_mask
    )
    
    alb_corr_best_fit = np.copy(alb_corr_mask)
    mask_bands = np.isnan(alb_corr_mask)
    alb_corr_best_fit[mask_bands] = best_fit_spectrum[mask_bands]
    
    # plt.close('all')
    # plt.plot(alb_wvl, alb_corr, '-', color='k', label='Corrected Albedo')
    # plt.plot(alb_wvl, alb_corr_mask, '-', color='g', label='Masked Corrected Albedo')
    # plt.plot(alb_wvl, best_fit_spectrum, '-', color='r', label='Best Fitted Albedo')
    # plt.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(alb_corr_mask), color='gray', alpha=0.2, label='Mask Gas absorption bands')

    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.title(f'SNICAR Best Fit Model: {best_fit_key}, RMSE: {min_rmse:.4f}')
    # plt.show()
    
    
    
    alb_wvl_sep_1nd_s, alb_wvl_sep_1nd_e = 370, 795
    alb_wvl_sep_2nd_s, alb_wvl_sep_2nd_e = 795, 850
    alb_wvl_sep_3rd_s, alb_wvl_sep_3rd_e = 850, 1050
    alb_wvl_sep_4th_s, alb_wvl_sep_4th_e = 1050, 1210
    alb_wvl_sep_5th_s, alb_wvl_sep_5th_e = 1185, 1700
    alb_wvl_sep_6th_s, alb_wvl_sep_6th_e = 1550, 2100
    
    band_1_fit = (alb_wvl >= alb_wvl_sep_1nd_s) & (alb_wvl < alb_wvl_sep_1nd_e)
    band_2_fit = (alb_wvl >= alb_wvl_sep_2nd_s) & (alb_wvl < alb_wvl_sep_2nd_e)
    band_3_fit = (alb_wvl >= alb_wvl_sep_3rd_s) & (alb_wvl < alb_wvl_sep_3rd_e)
    band_4_fit = (alb_wvl >= alb_wvl_sep_4th_s) & (alb_wvl < alb_wvl_sep_4th_e)
    band_5_fit = (alb_wvl >= alb_wvl_sep_5th_s) & (alb_wvl < alb_wvl_sep_5th_e)
    band_6_fit = (alb_wvl >= alb_wvl_sep_6th_s) & (alb_wvl <= alb_wvl_sep_6th_e)
    
    alb_corr_fit = copy.deepcopy(alb_corr_mask)
    for bands_fit in [band_1_fit, band_2_fit, band_3_fit, band_4_fit, band_5_fit, band_6_fit]:
        
        # if np.isnan(alb_corr_mask[bands_fit]).any():
            # best_fit_key, best_fit_spectrum, min_rmse = find_best_fit(
            #     model_library=snicar_data,
            #     obs_wvl=alb_wvl[bands_fit],
            #     obs_albedo=alb_corr_mask[bands_fit]
            #     )
        bandfit_nan = np.isnan(alb_corr_mask[bands_fit])
        if bandfit_nan.sum() == 0:
            continue
        bandfit_nan_ind = np.where(bandfit_nan)[0]
        if bandfit_nan_ind[-1] == len(bandfit_nan)-1:
            bandfit_nan_ind = bandfit_nan_ind[:-1]
        left_mean_ind_num = 5
        if bandfit_nan_ind[0] < left_mean_ind_num:
            left_mean_ind_num = bandfit_nan_ind[0]
        xl_origin = alb_corr_fit[bands_fit][bandfit_nan_ind[0]-left_mean_ind_num:bandfit_nan_ind[0]-1].mean()
        right_mean_ind_num = 5
        if (len(bandfit_nan) - bandfit_nan_ind[-1] -1) < right_mean_ind_num:
            right_mean_ind_num = len(bandfit_nan) - bandfit_nan_ind[-1] -1
            
        print("bandfit_nan_ind[-1]:", bandfit_nan_ind[-1])
        print("len(bandfit_nan):", len(bandfit_nan))
        print("xr_origin start end:", bandfit_nan_ind[-1]+1, bandfit_nan_ind[-1]+right_mean_ind_num)
        xr_origin = alb_corr_fit[bands_fit][bandfit_nan_ind[-1]+1:bandfit_nan_ind[-1]+right_mean_ind_num].mean()
        xl_fit, xr_fit = best_fit_spectrum[bands_fit][bandfit_nan_ind[0]-1], best_fit_spectrum[bands_fit][bandfit_nan_ind[-1]+1]
        xfit_base = np.min([xl_fit, xr_fit])
        if np.isfinite(xl_origin) and np.isfinite(xr_origin):
            base = np.min([xl_origin, xr_origin])
            scale = (xr_origin - xl_origin) / (xr_fit - xl_fit)
            scale = np.abs(scale)
            replace_array = base + (best_fit_spectrum[bands_fit][bandfit_nan] - xfit_base) * scale
            print("rescale replace_array shape:", replace_array.shape)
        elif np.isfinite(xl_origin) and not np.isfinite(xr_origin):
            # only have value on xl_origin
            # use valid point to scale
            xl_origin_new = alb_corr_fit[bands_fit][~bandfit_nan][0:5].mean()
            xr_origin_new = alb_corr_fit[bands_fit][~bandfit_nan][-6:-1].mean()
            xl_fit_new = best_fit_spectrum[bands_fit][~bandfit_nan][0:6].mean()
            xr_fit_new = best_fit_spectrum[bands_fit][~bandfit_nan][-6:-1].mean()
            xfit_base_new = xl_fit_new
            base = np.min([xl_origin_new, xr_origin_new])
            scale = (xr_origin_new - xl_origin_new) / (xr_fit_new - xl_fit_new)
            scale = np.abs(scale)
            replace_array_all = base + (best_fit_spectrum[bands_fit] - xfit_base_new) * scale
            replace_array = replace_array_all[bandfit_nan]
            print("rescale replace_array_all shape:", replace_array_all.shape)
            print("rescale replace_array shape:", replace_array.shape)
            # plt.close('all')
            # plt.plot(alb_wvl[bands_fit], alb_corr_mask[bands_fit], 'o', color='k', label='Corrected Albedo')
            # plt.plot(alb_wvl[bands_fit], replace_array_all, '--', color='b', label='Replace All')
            # plt.legend()
            # plt.show()
            # sys.exit()
        elif not np.isfinite(xl_origin) and np.isfinite(xr_origin):
            # only have value on xr_origin
            # not supported yet
            raise NotImplementedError("Only have value on right side is not supported yet.")
        
        alb_corr_fit_replace = copy.deepcopy(alb_corr_fit[bands_fit])
        print('replace_array shape:', replace_array.shape)
        print('alb_corr_fit_replace shape:', alb_corr_fit_replace.shape)
        print('bandfit_nan sum:', print(np.sum(bandfit_nan)))
        
        alb_corr_fit_replace[bandfit_nan] = copy.deepcopy(replace_array)
        alb_corr_fit[bands_fit] = copy.deepcopy(alb_corr_fit_replace)
            
        # print("base, scale:", base, scale)
        # print("base + (best_fit_spectrum[bands_fit][bandfit_nan] - xfit_base) * scale:", replace_array)
        # print("alb_corr_fit[bands_fit][bandfit_nan] after adjustment:", alb_corr_fit[bands_fit][bandfit_nan])
        # plt.close('all')
        # plt.plot(alb_wvl[bands_fit], alb_corr_mask[bands_fit], 'o', color='k', label='Corrected Albedo')
        # plt.plot(alb_wvl[bands_fit], alb_corr_fit_replace, '--', color='b', label='Replace')
        # plt.plot(alb_wvl[bands_fit], alb_corr_fit[bands_fit], '-', color='r', label='Fitted Albedo')
        # plt.xlabel('Wavelength (nm)')
        # plt.ylabel('Albedo')
        # plt.legend()
        # plt.show()
            
    
    
    alb_corr_fit[alb_corr_fit<0.0] = 0.0
    alb_corr_fit[alb_corr_fit>1.0] = 1.0
    
    # plt.close('all')
    # plt.plot(alb_wvl, alb_corr, '-', color='k', label='Corrected Albedo')
    # plt.plot(alb_wvl, alb_corr_fit, '-', color='r', label='Fitted Albedo')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.title(f'SNICAR Best Fit Model: {best_fit_key}, RMSE: {min_rmse:.4f}')
    # plt.show()
    
    # smooth with window size of 5
    alb_corr_fit_smooth = alb_corr_fit.copy()
    alb_corr_fit_smooth = uniform_filter1d(alb_corr_fit_smooth, size=5)
    alb_corr_fit_smooth[:2] = alb_corr_fit_smooth[2]
    alb_corr_fit_smooth[-2:] = alb_corr_fit_smooth[-2]
    alb_corr_fit_smooth[alb_corr_fit_smooth<0.0] = 0.0
    alb_corr_fit_smooth[alb_corr_fit_smooth>1.0] = 1.0
    
    alb_corr_fit_smooth[np.isfinite(alb_corr_fit_smooth)] = alb_corr_fit_smooth
    
    # print("alb_wvl shape:", alb_wvl.shape)
    # print("alb_corr shape:", alb_corr.shape)
    # print("alb_corr_mask shape:", alb_corr_mask.shape)
    # print("alb_corr_fit shape:", alb_corr_fit.shape)
    # print("alb_corr_fit_smooth shape:", alb_corr_fit_smooth.shape)
    
    return alb_corr_fit_smooth


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
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
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
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=1,
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
    #                     simulation_interval=3,
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
    #                 simulation_interval=3,
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
    #                                         # [11.33, 11.88],
    #                                         [12.00, 12.20],
    #                                         # [12.33, 13.80],
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
    #                                         # [11.33, 11.88],
    #                                         [12.00, 12.20],
    #                                         # [12.33, 13.80],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=15,
    #                 config=config,
    #                 )
    
    
    # for iter in range(2):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[
    #                                         # [11.29, 13.31],
    #                                         # [17.26, 18.32],
    #                                         [11.29, 11.40],
    #                                         ],
    #                     case_tag='clear_sky_track_atm_corr',
    #                     config=config,
    #                     simulation_interval=1,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     iter=iter,
    #                     )
        
    # sys.exit()

    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[
    #                                         # [11.29, 13.31],
    #                                         # [17.26, 18.32],
    #                                         [11.29, 11.40]
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=1,
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
    #                     simulation_interval=20, # in minute
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
        
    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[
    #                                         [16.251, 16.280], 
    #                                         [16.293, 16.325],
    #                                         [16.704, 16.780],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr_before',
    #                 config=config,
    #                 simulation_interval=20, # in minute
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
    
    
    
    
    
    
    
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[[17.0833, 17.0986],
    #                                         [17.1264, 17.1333],
    #                                         [17.1542, 17.1601],
    #                                         [17.1833, 17.1931],
    #                                         [17.2153, 17.2181],
    #                                         [17.2403, 17.2500],
    #                                         ],
    #                     case_tag='clear_sky_spiral_atm_corr',
    #                     config=config,
    #                     simulation_interval=10,
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
        
    # for iter in range(6):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[[13.7889, 13.8097],
    #                                         [13.8347, 13.8500],
    #                                         [13.8764, 13.8903],
    #                                         [13.9236, 13.9264],
    #                                         [13.9389, 13.9403],
    #                                         [13.9528, 13.9722],
    #                                         [13.9958, 14.0153],
    #                                         [14.0417, 14.0597],
    #                                         [14.0819, 14.1000],
    #                                         [14.1264, 14.1542],
    #                                         [14.1762, 14.2000],
    #                                         [14.2194, 14.2444],
    #                                         [14.2597, 14.2833]
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
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[[13.7889, 13.8010],
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
    #                     case_tag='clear_sky_spiral_atm_corr_after_corr_R3_v2',
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[
    #                                         [13.7889, 13.8010],
    #                                         [13.8350, 13.8395],
    #                                         [13.8780, 13.8885],
    #                                         # [13.9240, 13.9255],
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
    #                     case_tag='clear_sky_spiral_atm_corr_R1',
    #                     config=config,
    #                     # simulation_interval=50,
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
    
    # atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[[13.7889, 13.8010],
    #                                         [13.8350, 13.8395],
    #                                         [13.8780, 13.8885],
    #                                         # [13.9240, 13.9255],
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
    #                 case_tag='clear_sky_spiral_atm_corr_R1',
    #                 config=config,
    #                 )
        
    # surface albedo derivation
    # ------------------------------------------

    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 5, 28),
    #                     tmhr_ranges_select=[[15.610, 15.822],
    #                                         [16.905, 17.404] 
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=3,
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

        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),
    #                     tmhr_ranges_select=[[13.839, 15.180],  # 5.6 km
    #                                         [16.905, 17.404] 
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=3,
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
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                            np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0 ,
    #                     manual_cloud_cwp=77.82,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     iter=iter,
    #                     )

    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                            np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     iter=iter,
    #                     )
        
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[[12.405, 13.812], # 5.7m, 
    #                                         [14.250, 15.036], # 100m
    #                                         [15.535, 15.931], # 450m
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=3,
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
        
    
    # atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[[13.7889, 13.8010],
    #                                         [13.8350, 13.8395],
    #                                         [13.8780, 13.8885],
    #                                         [13.9240, 13.9255],
    #                                         [13.9389, 13.9403],
    #                                         [13.9540, 13.9715],
    #                                         [13.9980, 14.0153],
    #                                         [14.0417, 14.0575],
    #                                         [14.0417, 14.0475],
    #                                         [14.0560, 14.0590],
    #                                         [14.0825, 14.0975],
    #                                         [14.1264, 14.1525],
    #                                         [14.1762, 14.1975],
    #                                         [14.2194, 14.2420],
    #                                         [14.2605, 14.2810]
    #                                         ],
    #                 case_tag='clear_sky_spiral_atm_corr',
    #                 config=config,
    #                 )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[[16.250, 16.325], # 100m, 
    #                                         [16.375, 16.632], # 450m
    #                                         [16.700, 16.794], # 100m
    #                                         [16.850, 16.952], # 1.2km
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=3,
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     iter=iter,
    #                     )
        
    # for iter in range(3):
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
    #                     simulation_interval=None,
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),
    #                     tmhr_ranges_select=[
    #                                         [14.968, 16.115], # 100-450m, clear, some cloud
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=3,
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[13.704, 13.817], # 100-450m, clear, some cloud
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=3,
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[14.109, 14.140], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.4, 0.52, 0.6, 0.8, 1.0,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=17.4,
    #                     manual_cloud_cwp=90.51,
    #                     manual_cloud_cth=0.52,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=7.82,
    #                     iter=iter,
    #                     )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[15.834, 15.883], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.28, 0.3, 0.5, 0.58, 0.8, 1.0,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=22.4,
    #                     manual_cloud_cwp=35.6 ,
    #                     manual_cloud_cth=0.58,
    #                     manual_cloud_cbh=0.28,
    #                     manual_cloud_cot=2.39,
    #                     iter=iter,
    #                     )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[16.043, 16.067], # 100-200m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.38, 0.5, 0.68, 0.8, 1.0,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=8.9,
    #                     manual_cloud_cwp=21.29,
    #                     manual_cloud_cth=0.68,
    #                     manual_cloud_cbh=0.38,
    #                     manual_cloud_cot=3.59,
    #                     iter=iter,
    #                     )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[16.550, 17.581], # 100-500m, clear
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=3,
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


    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 7, 25),
    #                     tmhr_ranges_select=[[15.094, 15.300], # 100m, some low clouds or fog below
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=11.4,
    #                     manual_cloud_cwp=9.94,
    #                     manual_cloud_cth=0.30,
    #                     manual_cloud_cbh=0.16,
    #                     manual_cloud_cot=1.31,
    #                     iter=iter,
    #                     )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 7, 25),
    #                     tmhr_ranges_select=[[15.881, 15.903], # 200-500m
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=1,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=11.4,
    #                     manual_cloud_cwp=9.94,
    #                     manual_cloud_cth=0.30,
    #                     manual_cloud_cbh=0.16,
    #                     manual_cloud_cot=1.31,
    #                     iter=iter,
    #                     )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),
    #                     tmhr_ranges_select=[[13.442, 13.465],
    #                                         [13.490, 13.514],
    #                                         [13.536, 13.554],
    #                                         [13.580, 13.611],
    #                                         [13.639, 13.654],
    #                                         [13.676, 13.707],
    #                                         [13.733, 13.775],
    #                                         [13.793, 13.836],
    #                                         ],
    #                     case_tag='clear_sky_spiral_atm_corr',
    #                     config=config,
    #                     simulation_interval=None,
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),
    #                     tmhr_ranges_select=[[13.939, 14.200], # 100m, clear
    #                                         [14.438, 14.714], # 3.7km
    #                                         [15.214, 15.804], # 1.3km
    #                                         [16.176, 16.304], # 1.3km
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=3,
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 7, 30),
    #                     tmhr_ranges_select=[[13.886, 13.908],
    #                                         [13.934, 13.950],
    #                                         [13.976, 14.000],
    #                                         [14.031, 14.051],
    #                                         [14.073, 14.096],
    #                                         [14.115, 14.134],
    #                                         [14.157, 14.179],
    #                                         [14.202, 14.219],
    #                                         [14.239, 14.254],
    #                                         [14.275, 14.294],
    #                                         ],
    #                     case_tag='clear_sky_spiral_atm_corr',
    #                     config=config,
    #                     simulation_interval=None,
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 7, 30),
    #                     tmhr_ranges_select=[[14.318, 14.936], # 100-450m, clear
    #                                         [15.043, 15.140], # 1.5km
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=3,
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 1),
    #                     tmhr_ranges_select=[[13.843, 14.361], # 100-450m, clear, some open ocean
    #                                         [14.739, 15.053], # 550m
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=3,
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
        
   
        
    for iter in range(3):
        flt_trk_atm_corr(date=datetime.datetime(2024, 8, 2),
                        tmhr_ranges_select=[
                                            # [14.557, 15.100], # 100m
                                            # [15.244, 16.635], # 1km
                                            [15.49, 15.635], # 1km
                                            ],
                        case_tag='clear_atm_corr',
                        config=config,
                        simulation_interval=3,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        manual_cloud_cer=0.0,
                        manual_cloud_cwp=0.0,
                        manual_cloud_cth=0.0,
                        manual_cloud_cbh=0.0,
                        manual_cloud_cot=0.0,
                        iter=iter,
                        )
    
    
    sys.exit()
        
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),
    #                     tmhr_ranges_select=[[13.344, 13.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.65, 0.69, 0.78, 1.0,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=10.7,
    #                     manual_cloud_cwp=11.28,
    #                     manual_cloud_cth=0.78,
    #                     manual_cloud_cbh=0.69,
    #                     manual_cloud_cot=1.59,
    #                     iter=iter,
    #                     )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),
    #                     tmhr_ranges_select=[
    #                                         [15.472, 15.567], # 180m, cloudy
    #                                         [15.580, 15.921], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.62, 0.8, 0.96,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.2,
    #                     manual_cloud_cwp=77.5,
    #                     manual_cloud_cth=0.96,
    #                     manual_cloud_cbh=0.62,
    #                     manual_cloud_cot=16.21,
    #                     iter=iter,
    #                     )
        
    for iter in range(3):
        flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
                        tmhr_ranges_select=[
                                            [12.990, 13.180], # 180m, clear
                                            ],
                        case_tag='clear_atm_corr',
                        config=config,
                        simulation_interval=3,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        manual_cloud_cer=0.0,
                        manual_cloud_cwp=0.0,
                        manual_cloud_cth=0.0,
                        manual_cloud_cbh=0.0,
                        manual_cloud_cot=0.0,
                        iter=iter,
                        )
        
    for iter in range(3):
        flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
                        tmhr_ranges_select=[
                                            [14.250, 14.373], # 180m, clear
                                            ],
                        case_tag='clear_atm_corr',
                        config=config,
                        simulation_interval=3,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        manual_cloud_cer=0.0,
                        manual_cloud_cwp=0.0,
                        manual_cloud_cth=0.0,
                        manual_cloud_cbh=0.0,
                        manual_cloud_cot=0.0,
                        iter=iter,
                        )
        
    for iter in range(3):
        flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
                        tmhr_ranges_select=[
                                            [16.471, 16.601], # 180m, clear
                                            ],
                        case_tag='clear_atm_corr',
                        config=config,
                        simulation_interval=3,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        manual_cloud_cer=0.0,
                        manual_cloud_cwp=0.0,
                        manual_cloud_cth=0.0,
                        manual_cloud_cbh=0.0,
                        manual_cloud_cot=0.0,
                        iter=iter,
                        )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
    #                     tmhr_ranges_select=[
    #                                         [13.212, 13.347], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.67, 0.8, 1.0,]),
    #                                            np.array([1.5, 1.98, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=15.3,
    #                     manual_cloud_cwp=143.94,
    #                     manual_cloud_cth=1.98,
    #                     manual_cloud_cbh=0.67,
    #                     manual_cloud_cot=14.12,
    #                     iter=iter,
    #                     )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
    #                     tmhr_ranges_select=[
    #                                         [15.314, 15.504], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.78, 1.0,]),
    #                                            np.array([1.5, 1.81, 2.21, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=True,
    #                     overwrite_lrt=True,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=7.8,
    #                     manual_cloud_cwp=64.18,
    #                     manual_cloud_cth=2.21,
    #                     manual_cloud_cbh=1.81,
    #                     manual_cloud_cot=12.41,
    #                     iter=iter,
    #                     )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),
    #                     tmhr_ranges_select=[
    #                                         [13.376, 13.600], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.34, 0.4, 0.6, 0.77, 1.0,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=9.0,
    #                     manual_cloud_cwp=83.49,
    #                     manual_cloud_cth=0.77,
    #                     manual_cloud_cbh=0.34,
    #                     manual_cloud_cot=13.93,
    #                     iter=iter,
    #                     )
        
    for iter in range(3):
        flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),
                        tmhr_ranges_select=[
                                            [14.750, 15.060], # 100m, clear
                                            [15.622, 15.887], # 100m, clear
                                            ],
                        case_tag='clear_atm_corr',
                        config=config,
                        simulation_interval=3,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        manual_cloud_cer=0.0,
                        manual_cloud_cwp=0.0,
                        manual_cloud_cth=0.0,
                        manual_cloud_cbh=0.0,
                        manual_cloud_cot=0.0,
                        iter=iter,
                        )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),
    #                     tmhr_ranges_select=[
    #                                         [16.029, 16.224], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.29, 0.4, 0.62, 0.8, 1.0,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=3,
    #                     clear_sky=False,
    #                     overwrite_lrt=True,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=8.3,
    #                     manual_cloud_cwp=49.10,
    #                     manual_cloud_cth=0.62,
    #                     manual_cloud_cbh=0.29,
    #                     manual_cloud_cot=8.93,
    #                     iter=iter,
    #                     )
        
    
    for iter in range(3):
        flt_trk_atm_corr(date=datetime.datetime(2024, 8, 15),
                        tmhr_ranges_select=[
                                            [14.085, 14.396], # 100m, clear
                                            [14.550, 14.968], # 3.5km, clear
                                            [15.078, 15.163], # 1.7km, clear
                                            ],
                        case_tag='clear_atm_corr',
                        config=config,
                        simulation_interval=3,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        manual_cloud_cer=0.0,
                        manual_cloud_cwp=0.0,
                        manual_cloud_cth=0.0,
                        manual_cloud_cbh=0.0,
                        manual_cloud_cot=0.0,
                        iter=iter,
                        )