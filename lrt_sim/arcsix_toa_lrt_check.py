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

    if modis_alb_file is not None:
        print("modis_alb_legs shape:", np.array(modis_alb_legs).shape)
        print("color_series shape:", np.array(color_series).shape)
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
                     rsp_plot=False,
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
    
    first_series = True
    time_all = []
    fdn_550_all = []
    fup_550_all = []
    fdn_1600_all = []
    fup_1600_all = []
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    for i in range(len(tmhr_ranges_select)):
        print(f"Processing time range {i+1}/{len(tmhr_ranges_select)}: {tmhr_ranges_select[i][0]:.2f} - {tmhr_ranges_select[i][1]:.2f} UTC")
        fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
        if not os.path.exists(fname_pkl):
            print(f"File not found: {fname_pkl}, skipping...")
            continue
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
        
        
        # p3_up_dn_ratio_20240605_13.79_13.81_5.80_1.dat
        
        if not os.path.exists(ratio_fname):
            print(f"File not found: {ratio_fname}, skipping...")
            if modis_alb_file is not None:
                modis_alb_legs.append([np.nan]*7)
                time_all.append([np.nan])
                fdn_550_all.append([np.nan])
                fup_550_all.append([np.nan])
                fdn_1600_all.append([np.nan])
                fup_1600_all.append([np.nan])
            continue
        
        alb_ratio = pd.read_csv(ratio_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        alb_1 = pd.read_csv(update_1_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        alb_2 = pd.read_csv(update_2_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        p3_ratio_1 = pd.read_csv(update_p3_1_fname, delim_whitespace=True, comment='#', names=['wvl', 'ratio'])
        p3_ratio_2 = pd.read_csv(update_p3_2_fname, delim_whitespace=True, comment='#', names=['wvl', 'ratio'])
        
        df_ori = pd.read_csv(ori_csv_name)
        df_upd1 = pd.read_csv(updated_csv_name_1)
        df_upd2 = pd.read_csv(updated_csv_name_2)
        
        if first_series:
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
            
            first_series = False

            

        time_all.append(time)
        ssfr_wvl = cld_leg['ssfr_zen_wvl']
        ssfr_550_ind = np.argmin(np.abs(ssfr_wvl - 550))
        ssfr_1600_ind = np.argmin(np.abs(ssfr_wvl - 1600))
        fdn_550_all.append(cld_leg['ssfr_zen'][:, ssfr_550_ind])
        fup_550_all.append(cld_leg['ssfr_nad'][:, ssfr_550_ind])
        fdn_1600_all.append(cld_leg['ssfr_zen'][:, ssfr_1600_ind])
        fup_1600_all.append(cld_leg['ssfr_nad'][:, ssfr_1600_ind])
        
        
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
        
        ssrr_rad_up_mean_all[i, :] = df_ori['ssrr_rad_up_mean'].values
        ssrr_rad_dn_mean_all[i, :] = df_ori['ssrr_rad_dn_mean'].values
        ssrr_rad_up_std_all[i, :] = df_ori['ssrr_rad_up_std'].values
        ssrr_rad_dn_std_all[i, :] = df_ori['ssrr_rad_dn_std'].values
        ssrr_rad_up_wvl = df_ori['ssrr_nad_wvl'].values
        ssrr_rad_dn_wvl = df_ori['ssrr_zen_wvl'].values
        
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

    ssrr_ref = ssrr_rad_up_mean_all * np.pi / toa_mean_all / rsp_mu0_mean_all[:, np.newaxis] * (rsp_sd_mean_all[:, np.newaxis]**2)
    ssfr_fup_ref = ssfr_fup_mean_all / toa_mean_all /  rsp_mu0_mean_all[:, np.newaxis] * (rsp_sd_mean_all[:, np.newaxis]**2)
       
    print("rsp_wvl:", rsp_wvl)
    
        
    print("lon avg all mean:", lon_avg_all.mean())
    print("lat avg all mean:", lat_avg_all.mean())


    # Create a ScalarMappable
    data_min, data_max = np.log(alt_avg_all/alt_avg_all.max()).min(), np.log(alt_avg_all/alt_avg_all.max()).max()
    norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
    cmap = cm.jet # Or any other built-in colormap like cm.viridis
    s_m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_series = s_m.to_rgba(np.log(alt_avg_all/alt_avg_all.max()))               
    
    alb_wvl = alb_ratio['wvl'].values[1:-1] # skip the first value at 348 nm and last value at 2050 nm
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
    ssfr_fdn_tmhr[~pitch_roll_mask] = np.nan
    ssfr_fup_tmhr = ssfr_fup[t_ssfr_tmhr_mask]
    ssfr_fup_tmhr[~pitch_roll_mask] = np.nan
    
    t_hsr1_tmhr_mask = (t_hsr1 >= tmhr_ranges_select[0][0]) & (t_hsr1 <= tmhr_ranges_select[-1][1])
    t_hsr1_tmhr = t_hsr1[t_hsr1_tmhr_mask]
    hsr_dif_ratio = hsr_dif_ratio[t_hsr1_tmhr_mask]
    hsr1_diff_ratio_530_570_mean = hsr1_diff_ratio_530_570_mean[t_hsr1_tmhr_mask]
    
    hsr1_530_570_thresh = 0.18
    cloud_mask_hsr1 = hsr1_diff_ratio_530_570_mean > hsr1_530_570_thresh
    ssfr_fup_tmhr[cloud_mask_hsr1] = np.nan
    ssfr_fdn_tmhr[cloud_mask_hsr1] = np.nan
    
    
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

def alb_masking(wvl, alb):
    o2a_1_start, o2a_1_end = 748, 776
    h2o_1_start, h2o_1_end = 672, 706
    h2o_2_start, h2o_2_end = 705, 746
    h2o_3_start, h2o_3_end = 884, 996
    h2o_4_start, h2o_4_end = 1084, 1175
    h2o_5_start, h2o_5_end = 1230, 1286
    h2o_6_start, h2o_6_end = 1290, 1509
    h2o_7_start, h2o_7_end = 1748, 2050
    final_start, final_end = 2110, 2200
    
    alb_mask = alb.copy()
    alb_mask[((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
             ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
             ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
             ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
             ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
             ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
             ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
             ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
             ((wvl>=final_start) & (wvl<=final_end))
            ] = np.nan
    
    return alb_mask
    

def ice_alb_fitting(alb_wvl, alb_corr):
    ice_alb_model_data = pd.read_csv('ice_alb_prior.dat', delim_whitespace=True, comment='#', header=None, names=['wvl', 'alb', 'res'])
    f_ice_alb_model = interp1d(ice_alb_model_data['wvl']*1000, ice_alb_model_data['alb'], bounds_error=False, fill_value='extrapolate')
    ice_alb_model_i = f_ice_alb_model(alb_wvl)
    
    alb_ice_fit = ice_alb_model_i.copy()
    alb_wvl_sep_1nd = 370
    alb_wvl_sep_2nd = 525
    alb_wvl_sep_2nd_2 = 625
    alb_wvl_sep_3rd = 800
    alb_wvl_sep_4th = 880
    alb_wvl_sep_5th = 1015
    alb_wvl_sep_6th = 1030 
    alb_wvl_sep_7th = 1035
    alb_wvl_sep_8th = 1080 
    alb_wvl_sep_9th = 1200
    alb_wvl_sep_10th = 1288
    alb_wvl_sep_11th = 1520
    alb_wvl_sep_12th = 1700
    alb_wvl_sep_13th =  2100
    band_12 = (alb_wvl >= alb_wvl_sep_1nd) & (alb_wvl < alb_wvl_sep_2nd)
    band_225 = (alb_wvl >= alb_wvl_sep_2nd) & (alb_wvl < alb_wvl_sep_2nd_2)
    band_253 = (alb_wvl >= alb_wvl_sep_2nd_2) & (alb_wvl < alb_wvl_sep_3rd)
    band_23 = (alb_wvl >= alb_wvl_sep_2nd) & (alb_wvl < alb_wvl_sep_3rd)
    band_34 = (alb_wvl >= alb_wvl_sep_3rd) & (alb_wvl < alb_wvl_sep_4th)
    band_45 = (alb_wvl >= alb_wvl_sep_4th) & (alb_wvl < alb_wvl_sep_5th)   
    band_56 = (alb_wvl >= alb_wvl_sep_5th) & (alb_wvl < alb_wvl_sep_6th)
    band_67 = (alb_wvl >= alb_wvl_sep_6th) & (alb_wvl < alb_wvl_sep_7th)
    band_78 = (alb_wvl >= alb_wvl_sep_7th) & (alb_wvl < alb_wvl_sep_8th)
    band_89 = (alb_wvl >= alb_wvl_sep_8th) & (alb_wvl <= alb_wvl_sep_9th)
    band_910 = (alb_wvl > alb_wvl_sep_9th) & (alb_wvl <= alb_wvl_sep_10th)
    band_1011 = (alb_wvl > alb_wvl_sep_10th) & (alb_wvl <= alb_wvl_sep_11th)
    band_1112 = (alb_wvl > alb_wvl_sep_11th) & (alb_wvl <= alb_wvl_sep_12th)
    band_1213 = (alb_wvl > alb_wvl_sep_12th) & (alb_wvl <= alb_wvl_sep_13th)
    scale_factors = np.ones_like(alb_ice_fit)
    scale_factors[...] = np.nan
    alb_corr_to_ice_alb_ratio = (alb_corr / ice_alb_model_i).copy()
    # smooth the ratio
    
    alb_corr_to_ice_alb_ratio = uniform_filter1d(alb_corr_to_ice_alb_ratio, size=3)
    alb_corr_to_ice_alb_ratio[:2] = alb_corr_to_ice_alb_ratio[2]
    alb_corr_to_ice_alb_ratio[-2:] = alb_corr_to_ice_alb_ratio[-2]
    
    alb_corr_mask_to_ice_alb_ratio = alb_corr_to_ice_alb_ratio.copy()
    

    
    alb_corr_mask_to_ice_alb_ratio = interp1d(alb_wvl, alb_corr_mask_to_ice_alb_ratio, bounds_error=False, fill_value=np.nan)
    fit_wvl = np.arange(alb_wvl[0], alb_wvl[-1], 0.5)
    alb_corr_mask_to_ice_alb_ratio = alb_corr_mask_to_ice_alb_ratio(fit_wvl)
    
    band_12_fit = (fit_wvl >= alb_wvl_sep_1nd) & (fit_wvl < alb_wvl_sep_2nd)
    band_23_fit = (fit_wvl >= alb_wvl_sep_2nd) & (fit_wvl < alb_wvl_sep_3rd)
    band_34_fit = (fit_wvl >= alb_wvl_sep_3rd) & (fit_wvl < alb_wvl_sep_4th)
    band_45_fit = (fit_wvl >= alb_wvl_sep_4th) & (fit_wvl < alb_wvl_sep_5th)   
    band_56_fit = (fit_wvl >= alb_wvl_sep_5th) & (fit_wvl < alb_wvl_sep_6th)
    band_67_fit = (fit_wvl >= alb_wvl_sep_6th) & (fit_wvl < alb_wvl_sep_7th)
    band_78_fit = (fit_wvl >= alb_wvl_sep_7th) & (fit_wvl < alb_wvl_sep_8th)
    band_89_fit = (fit_wvl >= alb_wvl_sep_8th) & (fit_wvl <= alb_wvl_sep_9th)
    band_910_fit = (fit_wvl > alb_wvl_sep_9th) & (fit_wvl <= alb_wvl_sep_10th)
    band_1011_fit = (fit_wvl > alb_wvl_sep_10th) & (fit_wvl <= alb_wvl_sep_11th)
    band_1112_fit = (fit_wvl > alb_wvl_sep_11th) & (fit_wvl <= alb_wvl_sep_12th)
    band_1213_fit = (fit_wvl > alb_wvl_sep_12th) & (fit_wvl <= alb_wvl_sep_13th)
    
    exp_width = 2.0
    band_12_fit = (fit_wvl >= alb_wvl_sep_1nd-exp_width) & (fit_wvl < alb_wvl_sep_2nd+exp_width)
    band_225_fit = (fit_wvl >= alb_wvl_sep_2nd-exp_width) & (fit_wvl < alb_wvl_sep_2nd_2+exp_width)
    band_253_fit = (fit_wvl >= alb_wvl_sep_2nd_2-exp_width) & (fit_wvl < alb_wvl_sep_3rd+exp_width)
    band_23_fit = (fit_wvl >= alb_wvl_sep_2nd-exp_width) & (fit_wvl < alb_wvl_sep_3rd+exp_width)
    band_34_fit = (fit_wvl >= alb_wvl_sep_3rd-exp_width) & (fit_wvl < alb_wvl_sep_4th+exp_width)
    band_45_fit = (fit_wvl >= alb_wvl_sep_4th-exp_width) & (fit_wvl < alb_wvl_sep_5th+exp_width)
    band_56_fit = (fit_wvl >= alb_wvl_sep_5th-exp_width) & (fit_wvl < alb_wvl_sep_6th+exp_width)
    band_67_fit = (fit_wvl >= alb_wvl_sep_6th-exp_width) & (fit_wvl < alb_wvl_sep_7th+exp_width)
    band_78_fit = (fit_wvl >= alb_wvl_sep_7th-exp_width) & (fit_wvl < alb_wvl_sep_8th+exp_width)
    band_89_fit = (fit_wvl >= alb_wvl_sep_8th-exp_width) & (fit_wvl <= alb_wvl_sep_9th+exp_width)
    band_910_fit = (fit_wvl > alb_wvl_sep_9th-exp_width) & (fit_wvl <= alb_wvl_sep_10th+exp_width)
    band_1011_fit = (fit_wvl > alb_wvl_sep_10th-exp_width) & (fit_wvl <= alb_wvl_sep_11th+exp_width)
    band_1112_fit = (fit_wvl > alb_wvl_sep_11th-exp_width) & (fit_wvl <= alb_wvl_sep_12th+exp_width)
    band_1213_fit = (fit_wvl > alb_wvl_sep_12th-exp_width) & (fit_wvl <= alb_wvl_sep_13th+exp_width)
    
    alb_corr_mask_to_ice_alb_ratio = alb_masking(fit_wvl, alb_corr_mask_to_ice_alb_ratio)
    
                
    for band, band_fit in zip([band_12, band_225, band_253,
                            #    band_23, 
                                band_34, band_45, band_56, band_67, band_78, band_89, band_910, band_1112, band_1213],
                                [band_12_fit, band_225_fit, band_253_fit,
                            #    band_23_fit, 
                                band_34_fit, band_45_fit, band_56_fit, band_67_fit, band_78_fit, band_89_fit, band_910_fit, band_1112_fit, band_1213_fit]):
        
        scale_factors[band] = fit_1d_poly(fit_wvl[band_fit], alb_corr_mask_to_ice_alb_ratio[band_fit], order=1)(alb_wvl[band])
        alb_ice_fit[band] = alb_ice_fit[band] * fit_1d_poly(fit_wvl[band_fit], alb_corr_mask_to_ice_alb_ratio[band_fit], order=1)(alb_wvl[band])
        
    # special treatment for band 10 and 11 due to the discontinuity around 1375 nm
    div = 1375
    band_1011_first = (alb_wvl > alb_wvl_sep_10th) & (alb_wvl <= div)
    band_1011_last = (alb_wvl > div) & (alb_wvl <= alb_wvl_sep_11th)
    band_1011_fit_first = (fit_wvl > alb_wvl_sep_10th - exp_width) & (fit_wvl <= div + exp_width)
    band_1011_fit_last = (fit_wvl > div - exp_width) & (fit_wvl <= alb_wvl_sep_11th + exp_width)
    dx = np.nanmean(fit_wvl[band_1011_fit_last][-2:]) - np.nanmean(fit_wvl[band_1011_fit_last][:2])
    x0 = np.nanmean(fit_wvl[band_1011_fit_last][:2])
    scales = fit_1d_poly(fit_wvl[band_1011_fit], alb_corr_mask_to_ice_alb_ratio[band_1011_fit], order=1, dx=dx, x0=x0)(alb_wvl[band_1011_last])
    
    
    scale_factors[band_1011_last] = scales
    scale_factors[band_1011_first] = scales[0]
    alb_ice_fit[band_1011_last] = alb_ice_fit[band_1011_last] * scales
    alb_ice_fit[band_1011_first] = alb_ice_fit[band_1011_first] * scales[0]
    # end of special treatment
    
    del band_12, band_23, band_34, band_45, band_56, band_67, band_78, band_89, band_910, band_1011, band_1112, band_1213
    del band_12_fit, band_23_fit, band_34_fit, band_45_fit, band_56_fit, band_67_fit, band_78_fit, band_89_fit, band_910_fit, band_1011_fit, band_1112_fit, band_1213_fit
    del band_225, band_253
    del band_225_fit, band_253_fit
    
    gc.collect()
    
    alb_ice_fit[alb_wvl<370] = np.nan
    alb_ice_fit[np.logical_and(alb_wvl>=360, alb_wvl<370)] = alb_ice_fit[np.where(np.logical_and(np.isfinite(alb_ice_fit), (alb_wvl>=370)))[0][0]]
    alb_ice_fit[alb_wvl>2100] = np.nan
    alb_ice_fit[alb_ice_fit<0.0] = 0.0
    alb_ice_fit[alb_ice_fit>1.0] = 1.0
    
    # smooth with window size of 7
    alb_ice_fit_smooth = alb_ice_fit[np.isfinite(alb_ice_fit)].copy()
    alb_ice_fit_smooth = uniform_filter1d(alb_ice_fit_smooth, size=3)
    alb_ice_fit_smooth[:1] = alb_ice_fit_smooth[1]
    alb_ice_fit_smooth[-1:] = alb_ice_fit_smooth[-1]
    alb_ice_fit_smooth[alb_ice_fit_smooth<0.0] = 0.0
    alb_ice_fit_smooth[alb_ice_fit_smooth>1.0] = 1.0
    
    alb_ice_fit[np.isfinite(alb_ice_fit)] = alb_ice_fit_smooth
    
    return alb_ice_fit

def plot_toa_check(wvl_zen, mean_flux_zen_clear, mean_toa_zen_clear,
                   toa_ratio, toa_ratio_mask_sim_mean, toa_ratio_toa_flux_mean, toa_ratio_mask_quantile_50, toa_ratio_mask_quantile_90,
                   eff_wvl_bands_start, eff_wvl_bands_end, w1, w2, w3, w4,
                   tmhr_ssfr, flux_green, flux_swir,
                   toa_green, toa_swir,
                   tmhr_hsk, alt, hed,
                   clear_times, level_leg,
                   diff_thresh, alt_thresh,
                   date_s, fig_name):
    # Plotting SSFR zenith mean clear-sky fluxes and TOA comparison
    plt.close('all')
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[1, 1], wspace=0.25, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax1.plot(wvl_zen, mean_flux_zen_clear, label='SSFR Zenith', color='red')
    ax1.plot(wvl_zen, mean_toa_zen_clear, label='TOA * cos(SZA)', color='black', ls='--')
    ax1.set_ylim([0.0, 1.25])
    ax1.set_ylabel('Downwelling Flux (W m$^{-2}$ nm$^{-1}$)')
    ax1.legend()
    ax1.grid()
    ax1.axvspan(w1, w2, color='coral', alpha=0.15)
    ax1.axvspan(w3, w4, color='purple', alpha=0.10)

    ax2.plot(wvl_zen, toa_ratio, label='SSFR Zenith / (TOA * cos(SZA))', color='red')
    ax2.axhline(1.0, color='gray', ls='--')
    ax2.set_ylim([0.8, 1.2])
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Ratio')
    ax2.legend()
    ax2.grid()
    ax2.axvspan(w1, w2, color='coral', alpha=0.15)
    ax2.axvspan(w3, w4, color='purple', alpha=0.10)
    for band_start, band_end in zip(eff_wvl_bands_start, eff_wvl_bands_end):
        ax2.axvspan(band_start, band_end, color='cyan', alpha=0.25)
    ax2.set_title(f'Selected bands Mean TOA Ratio: {toa_ratio_mask_sim_mean:.3f}, weighted: {toa_ratio_toa_flux_mean:.3f}\n                                 median: { toa_ratio_mask_quantile_50:.3f}, percentile 90: {toa_ratio_mask_quantile_90:.3f}',#
                  y=1.02)
    

    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3)

    ax3.plot(tmhr_ssfr[level_leg], flux_green[level_leg], color='coral', alpha=0.2)
    ax3.scatter(tmhr_ssfr[clear_times], flux_green[clear_times], label='SSFR %d-%d nm' % (w1, w2), s=3, color='coral')
    ax3.axhline(toa_green, label='TOA %d-%d nm' % (w1, w2), color='black', ls='--')
    ax3.set_ylim([np.nanmean(flux_green[clear_times])*0.8, np.nanmean(flux_green[clear_times])*1.3])
    ax3.set_ylabel('SZA-normalized Flux (W m$^{-2}$ nm$^{-1}$)')
    ax3.grid()

    ax3_alt = ax3.twinx()
    ax3_alt.plot(tmhr_hsk, alt, color='#87ceeb', alpha=0.5, label='Altitude')
    ax3_alt.set_ylabel('Altitude (m)', color='#87ceeb')
    ax3_alt.tick_params(axis='y', labelcolor='#87ceeb')
    ax3_alt.legend(loc='upper right', fontsize=8)

    ax3.legend()

    ax4.plot(tmhr_ssfr[level_leg], flux_swir[level_leg], color='purple', alpha=0.2)
    ax4.scatter(tmhr_ssfr[clear_times], flux_swir[clear_times], label='SSFR %d-%d nm' % (w3, w4), s=3, color='purple')
    ax4.axhline(toa_swir, label='TOA %d-%d nm' % (w3, w4), color='black', ls='--')
    ax4.set_ylim([np.nanmean(flux_swir[clear_times])*0.8, np.nanmean(flux_swir[clear_times])*1.3])
    ax4.set_xlabel('Time (hr)')
    ax4.set_ylabel('SZA-normalized Flux (W m$^{-2}$ nm$^{-1}$)')
    ax4.grid()

    ax4_hed = ax4.twinx()
    ax4_hed.plot(tmhr_hsk, hed, color='#ffcc99', alpha=0.5, label='Heading')
    ax4_hed.set_ylabel('Heading (deg)', color='#ffcc99')
    ax4_hed.tick_params(axis='y', labelcolor='#ffcc99')
    ax4_hed.legend(loc='lower right', fontsize=8)
    
    ax4.legend()
    
    fig.suptitle(f'{date_s} SSFR Zenith Average Clear-Sky and high-alt\n(green diff ratio < {diff_thresh}, alt > {alt_thresh} m)', fontsize=16)# y=0.98)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=300)

def flt_toa_check(date=datetime.datetime(2024, 5, 31),
                  config: Optional[FlightConfig] = None):
    """Flight TOA reflectance check plot"""
    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    
    output_dir = f'fig/zen_flux_toa_eval'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_ssrr = load_h5(config.ssrr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    
    
    # Evaluation
    tmhr_ssfr = data_ssfr['time']/3600.0  # convert to hours
    tmhr_hsr1 = data_hsr1['time']/3600.0  # convert to hours

    wvl_zen = data_ssfr['wvl_dn']
    flux_zen = data_ssfr['f_dn']
    icing = data_ssfr['flag'] == 1
    scale_factor = 1.0
    if date_s == '20240808':
        scale_factor = 1.316  # scaling factor for 20240808 flight due to SSFR calibration issue
    elif date_s == '20240809':
        scale_factor = 1.198  # scaling factor for 20240809 flight due to SSFR calibration issue
    
    
    
    
    tmhr_hsk = data_hsk['tmhr']
    if len(tmhr_hsk) > len(tmhr_ssfr):
        hsk_select = (tmhr_hsk >= tmhr_ssfr[0]) & (tmhr_hsk <= tmhr_ssfr[-1])
        tmhr_hsk = tmhr_hsk[hsk_select]
        sza = data_hsk["sza"][hsk_select]
        alt = data_hsk['alt'][hsk_select]
        hed = data_hsk['ang_hed'][hsk_select]
        pit = data_hsk['ang_pit'][hsk_select]
        rol = data_hsk['ang_rol'][hsk_select]
    else:
    
        sza = data_hsk["sza"]
        alt = data_hsk['alt']
        hed = data_hsk['ang_hed']
        pit = data_hsk['ang_pit']
        rol = data_hsk['ang_rol']
    
    # Solar spectrum interpolation function
    flux_solar_interp = solar_interpolation_func(solar_flux_file='arcsix_ssfr_solar_flux_slit.dat', date=date)
    toa0_zen = flux_solar_interp(wvl_zen)
    toa_flight = toa0_zen[np.newaxis, :] * np.cos(np.radians(sza[:, np.newaxis]))

    level_leg = np.sqrt(pit**2 + rol**2) < 2.0
    
    w1, w2 = 780.0, 820.0
    w3, w4 = 1620.0, 1660.0
    green_wvl = (wvl_zen>=w1) & (wvl_zen<=w2)
    swir_wvl = (wvl_zen>=w3) & (wvl_zen<=w4)
    flux_green = np.nanmean(flux_zen[:, green_wvl], axis=1)/np.cos(np.radians(sza))
    flux_swir = np.nanmean(flux_zen[:, swir_wvl], axis=1)/np.cos(np.radians(sza))
    toa_green = np.nanmean(toa0_zen[green_wvl])
    toa_swir = np.nanmean(toa0_zen[swir_wvl])
    
    wvl_hsr1 = data_hsr1['wvl_dn_tot']
    hsr1_total = data_hsr1['f_dn_tot']
    hsr1_diff = data_hsr1['f_dn_dif']
    diff_ratio_all = hsr1_diff/hsr1_total
    hsr1_sel_wvl = (wvl_hsr1 >= 530.0) & (wvl_hsr1 <= 570.0 )
    diff_ratio_mean = np.nanmean(diff_ratio_all[:, hsr1_sel_wvl], axis=1)
    
    if len(tmhr_hsr1) != len(tmhr_ssfr):
        diff_ratio_mean = np.interp(tmhr_ssfr, tmhr_hsr1, diff_ratio_mean)

    diff_thresh = 0.18
    alt_thresh = 3000.0
    clear_times = (diff_ratio_mean < diff_thresh) & level_leg & (alt > alt_thresh)
    
    
    mean_flux_zen_clear = np.nanmean(flux_zen[clear_times, :], axis=0)
    mean_toa_zen_clear = np.nanmean(toa_flight[clear_times, :], axis=0)

    toa_ratio = mean_flux_zen_clear / mean_toa_zen_clear
    
    eff_wvl_1_start, eff_wvl_1_end = 746.0, 748.0
    eff_wvl_2_start, eff_wvl_2_end = 776.0, 884.0
    eff_wvl_3_start, eff_wvl_3_end = 996.0, 1084.0
    eff_wvl_4_start, eff_wvl_4_end = 1175.0, 1230.0
    eff_wvl_5_start, eff_wvl_5_end = 1286.0, 1290.0
    eff_wvl_6_start, eff_wvl_6_end = 1509.0, 1748.0
    
    eff_wvl_bands_start = [eff_wvl_1_start, eff_wvl_2_start, eff_wvl_3_start, eff_wvl_4_start, eff_wvl_5_start, eff_wvl_6_start]
    eff_wvl_bands_end = [eff_wvl_1_end, eff_wvl_2_end, eff_wvl_3_end, eff_wvl_4_end, eff_wvl_5_end, eff_wvl_6_end]

    
    toa_ratio_mask = toa_ratio.copy()
    toa_ratio_selct = ((wvl_zen>=eff_wvl_1_start) & (wvl_zen<=eff_wvl_1_end)) | \
                       ((wvl_zen>=eff_wvl_2_start) & (wvl_zen<=eff_wvl_2_end)) | \
                       ((wvl_zen>=eff_wvl_3_start) & (wvl_zen<=eff_wvl_3_end)) | \
                       ((wvl_zen>=eff_wvl_4_start) & (wvl_zen<=eff_wvl_4_end)) | \
                       ((wvl_zen>=eff_wvl_5_start) & (wvl_zen<=eff_wvl_5_end)) | \
                       ((wvl_zen>=eff_wvl_6_start) & (wvl_zen<=eff_wvl_6_end))
                      
    toa_ratio_selct = np.array(toa_ratio_selct)
    toa_ratio_mask[~toa_ratio_selct] = np.nan
    toa_zen_clear_mask = mean_toa_zen_clear.copy()
    toa_zen_clear_mask[~toa_ratio_selct] = np.nan
    
    toa_ratio_mask_sim_mean = np.nanmean(toa_ratio_mask)
    toa_ratio_toa_flux_mean = np.nansum(toa_ratio_mask*toa_zen_clear_mask)/np.nansum(toa_zen_clear_mask)
    toa_ratio_mask_quantile_50 = np.nanquantile(toa_ratio_mask, 0.5)
    toa_ratio_mask_quantile_90 = np.nanquantile(toa_ratio_mask, 0.9)

    


    plot_toa_check(wvl_zen, mean_flux_zen_clear, mean_toa_zen_clear,
                   toa_ratio, toa_ratio_mask_sim_mean, toa_ratio_toa_flux_mean, toa_ratio_mask_quantile_50, toa_ratio_mask_quantile_90,
                   eff_wvl_bands_start, eff_wvl_bands_end, w1, w2, w3, w4,
                   tmhr_ssfr, flux_green, flux_swir,
                   toa_green, toa_swir,
                   tmhr_hsk, alt, hed,
                   clear_times, level_leg,
                   diff_thresh, alt_thresh,
                   date_s, 
                   fig_name=f'{output_dir}/flt_toa_reflectance_check_{date_s}.png')
    
    if icing.any():
        flux_zen[icing, :] = np.nan
        flux_green = np.nanmean(flux_zen[:, green_wvl], axis=1)/np.cos(np.radians(sza))
        flux_swir = np.nanmean(flux_zen[:, swir_wvl], axis=1)/np.cos(np.radians(sza))
        
        clear_non_icing_times = (diff_ratio_mean < diff_thresh) & level_leg & (alt > alt_thresh) & (~icing)
        mean_flux_zen_clear = np.nanmean(flux_zen[clear_non_icing_times, :], axis=0)
        mean_toa_zen_clear = np.nanmean(toa_flight[clear_non_icing_times, :], axis=0)

        toa_ratio = mean_flux_zen_clear / mean_toa_zen_clear
        toa_ratio_mask = toa_ratio.copy()
        toa_ratio_selct = ((wvl_zen>=eff_wvl_1_start) & (wvl_zen<=eff_wvl_1_end)) | \
                        ((wvl_zen>=eff_wvl_2_start) & (wvl_zen<=eff_wvl_2_end)) | \
                        ((wvl_zen>=eff_wvl_3_start) & (wvl_zen<=eff_wvl_3_end)) | \
                        ((wvl_zen>=eff_wvl_4_start) & (wvl_zen<=eff_wvl_4_end)) | \
                        ((wvl_zen>=eff_wvl_5_start) & (wvl_zen<=eff_wvl_5_end)) | \
                        ((wvl_zen>=eff_wvl_6_start) & (wvl_zen<=eff_wvl_6_end))
                        
        toa_ratio_selct = np.array(toa_ratio_selct)
        toa_ratio_mask[~toa_ratio_selct] = np.nan
        toa_zen_clear_mask = mean_toa_zen_clear.copy()
        toa_zen_clear_mask[~toa_ratio_selct] = np.nan
        
        toa_ratio_mask_sim_mean = np.nanmean(toa_ratio_mask)
        toa_ratio_toa_flux_mean = np.nansum(toa_ratio_mask*toa_zen_clear_mask)/np.nansum(toa_zen_clear_mask)
        toa_ratio_mask_quantile_50 = np.nanquantile(toa_ratio_mask, 0.5)
        toa_ratio_mask_quantile_90 = np.nanquantile(toa_ratio_mask, 0.9)
        
        plot_toa_check(wvl_zen, mean_flux_zen_clear, mean_toa_zen_clear,
                   toa_ratio, toa_ratio_mask_sim_mean, toa_ratio_toa_flux_mean, toa_ratio_mask_quantile_50, toa_ratio_mask_quantile_90,
                   eff_wvl_bands_start, eff_wvl_bands_end, w1, w2, w3, w4,
                   tmhr_ssfr, flux_green, flux_swir,
                   toa_green, toa_swir,
                   tmhr_hsk, alt, hed,
                   clear_times, level_leg,
                   diff_thresh, alt_thresh,
                   date_s, 
                   fig_name=f'{output_dir}/flt_toa_reflectance_check_{date_s}_rm_icing_data.png')
        
    if date_s == '20240808' or date_s == '20240809':
        flux_zen /= scale_factor
        flux_green = np.nanmean(flux_zen[:, green_wvl], axis=1)/np.cos(np.radians(sza))
        flux_swir = np.nanmean(flux_zen[:, swir_wvl], axis=1)/np.cos(np.radians(sza))
        
        clear_times = (diff_ratio_mean < diff_thresh) & level_leg & (alt > alt_thresh)
        mean_flux_zen_clear = np.nanmean(flux_zen[clear_times, :], axis=0)
        mean_toa_zen_clear = np.nanmean(toa_flight[clear_times, :], axis=0)

        toa_ratio = mean_flux_zen_clear / mean_toa_zen_clear
        toa_ratio_mask = toa_ratio.copy()
        toa_ratio_selct = ((wvl_zen>=eff_wvl_1_start) & (wvl_zen<=eff_wvl_1_end)) | \
                        ((wvl_zen>=eff_wvl_2_start) & (wvl_zen<=eff_wvl_2_end)) | \
                        ((wvl_zen>=eff_wvl_3_start) & (wvl_zen<=eff_wvl_3_end)) | \
                        ((wvl_zen>=eff_wvl_4_start) & (wvl_zen<=eff_wvl_4_end)) | \
                        ((wvl_zen>=eff_wvl_5_start) & (wvl_zen<=eff_wvl_5_end)) | \
                        ((wvl_zen>=eff_wvl_6_start) & (wvl_zen<=eff_wvl_6_end))
                        
        toa_ratio_selct = np.array(toa_ratio_selct)
        toa_ratio_mask[~toa_ratio_selct] = np.nan
        toa_zen_clear_mask = mean_toa_zen_clear.copy()
        toa_zen_clear_mask[~toa_ratio_selct] = np.nan
        
        toa_ratio_mask_sim_mean = np.nanmean(toa_ratio_mask)
        toa_ratio_toa_flux_mean = np.nansum(toa_ratio_mask*toa_zen_clear_mask)/np.nansum(toa_zen_clear_mask)
        toa_ratio_mask_quantile_50 = np.nanquantile(toa_ratio_mask, 0.5)
        toa_ratio_mask_quantile_90 = np.nanquantile(toa_ratio_mask, 0.9)
        
        plot_toa_check(wvl_zen, mean_flux_zen_clear, mean_toa_zen_clear,
                   toa_ratio, toa_ratio_mask_sim_mean, toa_ratio_toa_flux_mean, toa_ratio_mask_quantile_50, toa_ratio_mask_quantile_90,
                   eff_wvl_bands_start, eff_wvl_bands_end, w1, w2, w3, w4,
                   tmhr_ssfr, flux_green, flux_swir,
                   toa_green, toa_swir,
                   tmhr_hsk, alt, hed,
                   clear_times, level_leg,
                   diff_thresh, alt_thresh,
                   date_s, 
                   fig_name=f'{output_dir}/flt_toa_reflectance_check_{date_s}_w_scaling.png')
        
        
        


def flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),
                     tmhr_ranges_select=[[14.10, 14.27]],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                     levels=None,
                     simulation_interval=None, # in minute
                     clear_sky=True,
                     overwrite_lrt=True,
                     manual_cloud=False,
                     manual_cloud_cer=14.4,
                     manual_cloud_cwp=0.06013,
                     manual_cloud_cth=0.945,
                     manual_cloud_cbh=0.344,
                     manual_cloud_cot=6.26,
                     iter=0
                    ):
#    
    
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
    t_ssrr = data_ssrr['tmhr']  # convert to hours
    t_marli = data_marli['time'] # in hours

    
    # atmospheric profile setting
    #/----------------------------------------------------------------------------\#
    dropsonde_file_list, dropsonde_date_list, dropsonde_tmhr_list, dropsonde_lon_list, dropsonde_lat_list = dropsonde_time_loc_list(dir_dropsonde=f'{_fdir_general_}/dropsonde')
    
    date_select = dropsonde_date_list == date.date()
    if np.sum(date_select) == 0:
        print(f"No dropsonde data found for date {date.strftime('%Y-%m-%d')}")
        log.info(f"No dropsonde data found for date {date.strftime('%Y-%m-%d')}")
        data_dropsonde = None
    else:
        dropsonde_tmhr_array = np.array(dropsonde_tmhr_list)[date_select]
        
        # find the closest dropsonde time to the flight mid times
        mid_tmhr = np.array([np.mean(rng) for rng in tmhr_ranges_select])
        dropsonde_idx = closest_indices(dropsonde_tmhr_array, mid_tmhr)
        dropsonde_file = np.array(dropsonde_file_list)[date_select][dropsonde_idx[0]]
        log.info(f"Using dropsonde file: {dropsonde_file}")
        head, data_dropsonde = read_ict_dropsonde(dropsonde_file, encoding='utf-8', na_values=[-9999999, -777, -888])


    zpt_filedir = f'{_fdir_general_}/zpt/{date_s}'
    os.makedirs(zpt_filedir, exist_ok=True)
    if levels is None:
        levels = np.concatenate((np.arange(0, 0.26, 0.05), 
                                         np.arange(0.3, 1., 0.1), 
                                         np.arange(1., 2.1, 0.2), 
                                        np.arange(2.5, 4.1, 0.5), 
                                        np.arange(5.0, 10.1, 2.5),
                                        np.array([15, 20, 30., 40., 50.])))
    

    xx = np.linspace(-12, 12, 241)
    yy_gaussian_vis = gaussian(xx, 0, 3.8251)
    yy_gaussian_nir = gaussian(xx, 0, 4.5046)
    
    import platform
    # run lower resolution on Mac for testing, higher resolution on Linux cluster
    if platform.system() == 'Darwin':
        xx_wvl_grid = np.arange(350, 2000.1, 2.5)
    elif platform.system() == 'Linux':
        xx_wvl_grid = np.arange(350, 2000.1, 1.0)
        
        
    if iter==0:
        if 1:#not os.path.exists('wvl_grid_test.dat'):
            write_2col_file('vis_0.1nm_update.dat', xx, yy_gaussian_vis,
                            header=('# SSFR Silicon slit function\n'
                                    '# wavelength (nm)      relative intensity\n'))
            write_2col_file('nir_0.1nm_update.dat', xx, yy_gaussian_nir,
                            header=('# SSFR InGaAs slit function\n'
                                    '# wavelength (nm)      relative intensity\n'))
            write_2col_file('wvl_grid_test.dat', xx_wvl_grid, np.zeros_like(xx_wvl_grid),
                            header=('# SSFR Wavelength grid test file\n'
                                    '# wavelength (nm)\n'))
    
    # write out the convolved solar flux
    #/----------------------------------------------------------------------------\#
    # Kurudz solar spectrum has a resolution of 0.5 nm
    wvl_solar_vis = np.arange(300, 950.1, 1.0)
    wvl_solar_nir = np.arange(951, 2500.1, 1.0)
    wvl_solar_coarse = np.concatenate([wvl_solar_vis, wvl_solar_nir])
    effective_wvl = wvl_solar_coarse[np.logical_and(wvl_solar_coarse >= xx_wvl_grid.min(), wvl_solar_coarse <= xx_wvl_grid.max())]
    if iter==0:
        # use Kurudz solar spectrum
        # df_solor = pd.read_csv('kurudz_0.1nm.dat', sep='\s+', header=None)
        # use CU solar spectrum
        df_solor = pd.read_csv('CU_composite_solar_processed.dat', sep='\s+', header=None)
        wvl_solar = np.array(df_solor.iloc[:, 0])
        flux_solar = np.array(df_solor.iloc[:, 1])#/1000 # convert mW/m^2/nm to W/m^2/nm
        
        # interpolate to 1 nm grid
        f_interp = interp1d(wvl_solar, flux_solar, kind='linear', bounds_error=False, fill_value=0.0)
        wvl_solar_interp = np.arange(250, 2550.1, 1.0)
        flux_solar_interp = f_interp(wvl_solar_interp)
        
        mask = wvl_solar_interp <= 2500

        wvl_solar = wvl_solar_interp[mask]
        flux_solar = flux_solar_interp[mask]
        
        assert (xx[1]-xx[0]) - (wvl_solar[1]-wvl_solar[0]) <1e-3

        flux_solar_convolved = ssfr_slit_convolve(wvl_solar, flux_solar, wvl_joint=950)
        
        write_2col_file('arcsix_ssfr_solar_flux_raw.dat', wvl_solar, flux_solar,
                        header=('# SSFR version solar flux without slit function convolution\n'
                                '# wavelength (nm)      flux (mW/m^2/nm)\n'))
        write_2col_file('arcsix_ssfr_solar_flux_slit.dat', wvl_solar, flux_solar_convolved,
                        header=('# SSFR version solar flux with slit function convolution\n'
                                '# wavelength (nm)      flux (mW/m^2/nm)\n'))
            
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
    if not os.path.exists(fname_cld_obs_info) or iter==0:      
        
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
                
                # RSP data
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

                
            if clear_sky:
                leg.update({
                    "cot": np.full_like(leg['lon'], np.nan),
                    "cer": np.full_like(leg['lon'], np.nan),
                    "cwp": np.full_like(leg['lon'], np.nan),
                    "cth": np.full_like(leg['lon'], np.nan),
                    "cgt": np.full_like(leg['lon'], np.nan),
                    "cbh": np.full_like(leg['lon'], np.nan),
                })
            elif not clear_sky and manual_cloud:
                leg.update({
                    "cot": np.full_like(leg['lon'], manual_cloud_cot),
                    "cer": np.full_like(leg['lon'], manual_cloud_cer),
                    "cwp": np.full_like(leg['lon'], manual_cloud_cwp),
                    "cth": np.full_like(leg['lon'], manual_cloud_cth),
                    "cgt": np.full_like(leg['lon'], manual_cloud_cth-manual_cloud_cbh),
                    "cbh": np.full_like(leg['lon'], manual_cloud_cbh),
                })
            else:
                raise NotImplementedError("Automatic cloud retrieval not implemented yet")
            
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
            ssfr_zen_flux[~pitch_roll_mask, :] = np.nan
            ssfr_nad_flux_interp[~pitch_roll_mask, :] = np.nan
            ssfr_zen_toa[~pitch_roll_mask, :] = np.nan
            
            hsr1_530nm_ind = np.argmin(np.abs(leg['hsr1_wvl'] - 530.0))
            hsr1_570nm_ind = np.argmin(np.abs(leg['hsr1_wvl'] - 570.0))
            hsr1_diff_ratio = data_hsr1["f_dn_dif"][sel_hsr1]/data_hsr1["f_dn_tot"][sel_hsr1]
            hsr1_diff_ratio_530_570_mean = np.nanmean(hsr1_diff_ratio[:, hsr1_530nm_ind:hsr1_570nm_ind+1], axis=1)
            hsr1_530_570_thresh = 0.18
            cloud_mask_hsr1 = hsr1_diff_ratio_530_570_mean > hsr1_530_570_thresh
            ssfr_zen_flux[cloud_mask_hsr1, :] = np.nan
            ssfr_nad_flux_interp[cloud_mask_hsr1, :] = np.nan
            
            
            leg['ssfr_zen'] = ssfr_zen_flux
            leg['ssfr_nad'] = ssfr_nad_flux_interp
            leg['ssfr_zen_wvl'] = ssfr_zen_wvl
            leg['ssfr_nad_wvl'] = ssfr_zen_wvl
            leg['ssfr_toa'] = ssfr_zen_toa
            
            
            # ssrr
            # interpolate ssrr zenith radiance to nadir wavelength grid
            f_zen_rad_interp = interp1d(data_ssrr["zen/wvl"], data_ssrr["zen/rad"][sel_ssrr, :], axis=1, bounds_error=False, fill_value=np.nan)
            ssrr_rad_zen_i = f_zen_rad_interp(ssfr_zen_wvl)
            f_nad_rad_interp = interp1d(data_ssrr["nad/wvl"], data_ssrr["nad/rad"][sel_ssrr, :], axis=1, bounds_error=False, fill_value=np.nan)
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
    
    
    solver = 'lrt'
    for ileg, _ in enumerate(leg_masks):
        
        cld_leg = vars()[f'cld_leg_{ileg}']
        time_start, time_end = cld_leg['time'][0], cld_leg['time'][-1]
        lon_avg = np.round(np.mean(cld_leg['lon']), 2)
        lat_avg = np.round(np.mean(cld_leg['lat']), 2)
        alt_avg = np.round(np.nanmean(cld_leg['alt']), 2)  # in km
        heading_avg = np.round(np.nanmean(cld_leg['heading']), 2)
        sza_avg = np.round(np.nanmean(cld_leg['sza']), 2)
        saa_avg = np.round(np.nanmean(cld_leg['saa']), 2)
        ssfr_zen_flux = cld_leg['ssfr_zen']
        ssfr_nad_flux = cld_leg['ssfr_nad']
        if np.all(np.isnan(ssfr_zen_flux)) or np.all(np.isnan(ssfr_nad_flux)):
            print(f"All SSFR zenith or nadir fluxes are NaN for leg {ileg+1}, skipping atmospheric correction")
            continue
        
        # atm profile searching setting
        boundary_from_center = 0.25 # degree
        mod_lon = np.array([lon_avg-boundary_from_center, lon_avg+boundary_from_center])
        mod_lat = np.array([lat_avg-boundary_from_center, lat_avg+boundary_from_center])
        mod_extent = [mod_lon[0], mod_lon[1], mod_lat[0], mod_lat[1]]
        
        
        if clear_sky:
            fname_h5 = '%s/%s-%s-%s-%s-time_%.2f-%.2f-alt_%.2f-clear.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag, time_start, time_end, alt_avg)
            fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_clear'
            fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
        else:
            fname_h5 = '%s/%s-%s-%s-%s-time_%.2f-%.2f-alt_%.2f-cloud.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag, time_start, time_end, alt_avg)
            fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_sat_cloud'
            fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'

        os.makedirs(fdir_tmp, exist_ok=True)
        os.makedirs(fdir, exist_ok=True)
    
        
        if not os.path.exists(fname_h5) or overwrite_lrt: 
            if iter==0:
                prepare_atmospheric_profile(_fdir_general_, date_s, case_tag, ileg, date, time_start, time_end,
                                            alt_avg, data_dropsonde,
                                            cld_leg, levels=levels,
                                            mod_extent=[np.round(np.nanmin(cld_leg['lon']), 2), 
                                                        np.round(np.nanmax(cld_leg['lon']), 2),
                                                        np.round(np.nanmin(cld_leg['lat']), 2),
                                                        np.round(np.nanmax(cld_leg['lat']), 2)],
                                            zpt_filedir=f'../data/zpt/{date_s}'
                                            )
            # =================================================================================
            
            
            # write out the surface albedo
            #/----------------------------------------------------------------------------\#
            os.makedirs(f'{_fdir_general_}/sfc_alb', exist_ok=True)
            iter_0_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_0.dat'
            if 1:#not os.path.exists(iter_0_fname) or overwrite_alb:
                
                alb_wvl = np.concatenate(([348.0], cld_leg['ssfr_zen_wvl'], [2050.]))
                alb_avg = np.nanmean(cld_leg['ssfr_nad']/cld_leg['ssfr_zen'], axis=0)
                print("cld_leg['ssfr_nad']:", np.nanmean(cld_leg['ssfr_nad'], axis=0))
                print("cld_leg['ssfr_zen']:", np.nanmean(cld_leg['ssfr_zen'], axis=0))
                
                if np.all(np.isnan(alb_avg)):
                    raise ValueError(f"All nadir/zenith ratios are NaN for leg {ileg+1}, cannot compute average albedo")
                alb_avg[alb_avg<0.0] = 0.0
                alb_avg[alb_avg>1.0] = 1.0
                alb_avg[np.isnan(alb_avg)] = 0.0
                
                alb_avg_extend = np.concatenate(([alb_avg[0]], alb_avg, [alb_avg[-1]]))

                write_2col_file(iter_0_fname, alb_wvl, alb_avg_extend,
                                header=('# SSFR derived sfc albedo\n'
                                        '# wavelength (nm)      albedo (unitless)\n'))
                
            #\----------------------------------------------------------------------------/#
            
            atm_z_grid = levels
            z_list = atm_z_grid
            atm_z_grid_str = ' '.join(['%.3f' % z for z in atm_z_grid])

          
            flux_output = np.zeros(len(data_hsk['lon'][leg_masks[ileg]]))
            
            for ix in range(1):
                flux_key_all = []
                if 0:#os.path.exists(f'{fdir}/flux_down_result_dict_sw_atm_corr.pk') and not new_compute:
                    print(f'Loading flux_down_result_dict_sw_atm_corr.pk from {fdir} ...')
                    with open(f'{fdir}/flux_down_result_dict_sw_atm_corr.pk', 'rb') as f:
                        flux_down_result_dict = pickle.load(f)
                    with open(f'{fdir}/flux_down_dir_result_dict_sw_atm_corr.pk', 'rb') as f:
                        flux_down_dir_result_dict = pickle.load(f)
                    with open(f'{fdir}/flux_down_diff_result_dict_sw_atm_corr.pk', 'rb') as f:
                        flux_down_diff_result_dict = pickle.load(f)
                    with open(f'{fdir}/flux_up_result_dict_sw_atm_corr.pk', 'rb') as f:
                        flux_up_result_dict = pickle.load(f)
                    flux_key_all.extend(flux_down_result_dict.keys())
                else:
                    flux_down_result_dict = {}
                    flux_down_dir_result_dict = {}
                    flux_down_diff_result_dict = {}
                    flux_up_result_dict = {}
                    
                    flux_down_results = []
                    flux_down_dir_results = []
                    flux_down_diff_results = []
                    flux_up_results = []
                
                flux_key = np.zeros_like(flux_output, dtype=object)
                cloudy = 0
                clear = 0
                
                # rt initialization
                #/----------------------------------------------------------------------------\#
                lrt_cfg = copy.deepcopy(er3t.rtm.lrt.get_lrt_cfg())
                
                lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km.dat')
                # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')
                lrt_cfg['solar_file'] = 'arcsix_ssfr_solar_flux_raw.dat'
                # lrt_cfg['solar_file'] = lrt_cfg['solar_file'].replace('kurudz_0.1nm.dat', 'kurudz_1.0nm.dat')
                lrt_cfg['number_of_streams'] = 4
                lrt_cfg['mol_abs_param'] = 'reptran coarse'
                # lrt_cfg['mol_abs_param'] = f'reptran medium'
                input_dict_extra_general = {
                                    'crs_model': 'rayleigh Bodhaine29',
                                    'albedo_file': f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_{iter}.dat',
                                    'mol_file': 'CH4 %s' % os.path.join(zpt_filedir, f'ch4_profiles_{date_s}_{case_tag}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km.dat'),
                                    'wavelength_grid_file': 'wvl_grid_test.dat',
                                    'atm_z_grid': atm_z_grid_str,
                                    # 'no_scattering':'mol',
                                    # 'no_absorption':'mol',
                                    }
            
                
                Nx_effective = len(effective_wvl)
                mute_list = ['albedo', 'wavelength', 'spline', 'slit_function_file']
                #/----------------------------------------------------------------------------/#

                
                inits_rad = []
                flux_key_ix = []
                output_list = []

                # cot_x = cld_leg['cot'][ix]
                # cwp_x = cld_leg['cwp'][ix]
                
                cot_x = np.nanmean(cld_leg['cot'])
                cwp_x = np.nanmean(cld_leg['cwp'])
                
                if not clear_sky:
                    input_dict_extra = copy.deepcopy(input_dict_extra_general)
                    if ((cot_x >= 0.1 and np.isfinite(cwp_x))):
                        cloudy += 1
                        
                        # cer_x = cld_leg['cer'][ix]
                        # cwp_x = cld_leg['cwp'][ix]
                        # cth_x = cld_leg['cth'][ix]
                        # cbh_x = cld_leg['cbh'][ix]
                        # cgt_x = cld_leg['cgt'][ix]

                        cer_x = np.nanmean(cld_leg['cer'])
                        cwp_x = np.nanmean(cld_leg['cwp'])
                        cth_x = np.nanmean(cld_leg['cth'])
                        cbh_x = np.nanmean(cld_leg['cbh'])
                        cgt_x = np.nanmean(cld_leg['cgt'])
    
                        cth_ind_cld = bisect.bisect_left(z_list, cth_x)
                        cbh_ind_cld = bisect.bisect_left(z_list, cbh_x)
                        
                        fname_cld = f'{fdir_tmp}/cld_{ix:04d}.txt'
                        if os.path.exists(fname_cld):
                            os.remove(fname_cld)
                        cld_cfg = er3t.rtm.lrt.get_cld_cfg()
                        cld_cfg['cloud_file'] = fname_cld
                        cld_cfg['cloud_altitude'] = z_list[cbh_ind_cld:cth_ind_cld+2]
                        cld_cfg['cloud_effective_radius']  = cer_x
                        cld_cfg['liquid_water_content'] = cwp_x*1000/(cgt_x*1000) # convert kg/m^2 to g/m^3
                        cld_cfg['cloud_optical_thickness'] = cot_x

                        dict_key_arr = np.concatenate(([cld_cfg['cloud_optical_thickness']], [cld_cfg['cloud_effective_radius']], cld_cfg['cloud_altitude'], [alt_avg]))
                        dict_key = '_'.join([f'{i:.3f}' for i in dict_key_arr])
                    else:
                        cld_cfg = None
                        dict_key = f'clear {alt_avg:.2f}'
                        clear += 1
                else:
                    cld_cfg = None
                    dict_key = f'clear {alt_avg:.2f}'
                    cot_x = 0.0
                    cwp_x = 0.0
                    cer_x = 0.0
                    cth_x = 0.0
                    cbh_x = 0.0
                    cgt_x = 0.0
                    input_dict_extra = copy.deepcopy(input_dict_extra_general)
                    clear += 1
                flux_key[ix] = dict_key
                
                if (cld_cfg is None) and (dict_key in flux_key_all):
                    flux_key_ix.append(dict_key)
                elif (cld_cfg is not None) and (dict_key in flux_key_all):
                    flux_key_ix.append(dict_key)
                else:
                    input_dict_extra_alb = copy.deepcopy(input_dict_extra)
                    init = er3t.rtm.lrt.lrt_init_mono_flx(
                            input_file  = '%s/input_%04d.txt'  % (fdir_tmp, ix),
                            output_file = '%s/output_%04d.txt' % (fdir_tmp, ix),
                            date        = date,
                            # surface_albedo=0.08,
                            solar_zenith_angle = sza_avg,
                            Nx = Nx_effective,
                            output_altitude    = [0, alt_avg, 'toa'],
                            input_dict_extra   = input_dict_extra_alb.copy(),
                            mute_list          = mute_list,
                            lrt_cfg            = lrt_cfg,
                            cld_cfg            = cld_cfg,
                            aer_cfg            = None,
                            )
                    #\----------------------------------------------------------------------------/#

                    inits_rad.append(copy.deepcopy(init))
                    output_list.append('%s/output_%04d.txt' % (fdir_tmp, ix))
                    flux_key_all.append(dict_key)
                    flux_key_ix.append(dict_key)
                    
            # # Run RT
            print(f"Start running libratran calculations for {fname_h5.replace('.h5', '')} ")
            # #/----------------------------------------------------------------------------\#
            import platform
            if platform.system() == 'Darwin':
                ##### run several libratran calculations in parallel
                if len(inits_rad) > 0:
                    print('Running libratran calculations ...')
                    # check available CPU cores
                    NCPU = os.cpu_count()
                    import platform
                    if platform.system() == 'Darwin':
                        NCPU -= 2
                    er3t.rtm.lrt.lrt_run_mp(inits_rad, Ncpu=NCPU)        
                    for i in range(len(inits_rad)):
                        data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                        flux_down_result_dict[flux_key_all[i]] = np.squeeze(data.f_down)
                        flux_down_dir_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_direct)
                        flux_down_diff_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_diffuse)
                        flux_up_result_dict[flux_key_all[i]] = np.squeeze(data.f_up)
                        
                        flux_down_results.append(np.squeeze(data.f_down))
                        flux_down_dir_results.append(np.squeeze(data.f_down_direct))
                        flux_down_diff_results.append(np.squeeze(data.f_down_diffuse))
                        flux_up_results.append(np.squeeze(data.f_up))
            ##### run several libratran calculations one by one
            
            elif platform.system() == 'Linux':
                if len(inits_rad) > 0:
                    print('Running libratran calculations ...')
                    for i in range(len(inits_rad)):
                        er3t.rtm.lrt.lrt_run(inits_rad[i])
                        data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                        flux_down_result_dict[flux_key_all[i]] = np.squeeze(data.f_down)
                        flux_down_dir_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_direct)
                        flux_down_diff_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_diffuse)
                        flux_up_result_dict[flux_key_all[i]] = np.squeeze(data.f_up)
                        
                        flux_down_results.append(np.squeeze(data.f_down))
                        flux_down_dir_results.append(np.squeeze(data.f_down_direct))
                        flux_down_diff_results.append(np.squeeze(data.f_down_diffuse))
                        flux_up_results.append(np.squeeze(data.f_up))
            # #\----------------------------------------------------------------------------/#
            ###### delete input, output, cld txt files
            for prefix in ['input', 'output', 'cld']:
                for filename in glob.glob(os.path.join(fdir_tmp, f'{prefix}_*.txt')):
                    os.remove(filename)
            ###### delete atmospheric profile files for lw

            
            
            # save dict
            # status = 'wb'
            # with open(f'{fdir}/flux_down_result_dict_sw_atm_corr.pk', status) as f:
            #     pickle.dump(flux_down_result_dict, f)
            # with open(f'{fdir}/flux_down_dir_result_dict_sw_atm_corr.pk', status) as f:
            #     pickle.dump(flux_down_dir_result_dict, f)
            # with open(f'{fdir}/flux_down_diff_result_dict_sw_atm_corr.pk', status) as f:
            #     pickle.dump(flux_down_diff_result_dict, f)
            # with open(f'{fdir}/flux_up_result_dict_sw_atm_corr.pk', status) as f:
            #     pickle.dump(flux_up_result_dict, f)

            flux_down_results = np.array(flux_down_results)
            flux_down_dir_results = np.array(flux_down_dir_results)
            flux_down_diff_results = np.array(flux_down_diff_results)
            flux_up_results = np.array(flux_up_results)
            
            for flux_dn in [flux_down_results, flux_down_dir_results, flux_down_diff_results, flux_up_results]:
                for iz in range(3):
                    for iset in range(flux_down_results.shape[0]):
                        flux_dn[iset, :, iz] = ssfr_slit_convolve(effective_wvl, flux_dn[iset, :, iz], wvl_joint=950)
            
            
            
            # simulated fluxes at p3 altitude            
            Fup_p3 = flux_up_results[:, :, 1]
            Fdn_p3 = flux_down_results[:, :, 1]
            Fdn_p3_diff_ratio = flux_down_diff_results[:, :, 1] / flux_down_results[:, :, 1]
            
            # simulated fluxes at toa
            Fup_toa = flux_up_results[:, :, -1]
            Fdn_toa = flux_down_results[:, :, -1]
            
            # interpolate the simulated fluxes to ssfr wavelength grid
            p3_up_to_dn_ratio = Fup_p3 / Fdn_p3
            p3_up_to_dn_ratio_mean = np.nanmean(p3_up_to_dn_ratio, axis=0)
            f_p3_up_to_dn_ratio_mean = interp1d(effective_wvl, p3_up_to_dn_ratio_mean, bounds_error=False, fill_value=np.nan)
            p3_up_to_dn_ratio_mean = f_p3_up_to_dn_ratio_mean(cld_leg['ssfr_zen_wvl'])
            
            f_Fup_p3_mean = interp1d(effective_wvl, np.nanmean(Fup_p3, axis=0), bounds_error=False, fill_value=np.nan)
            Fup_p3_mean_interp = f_Fup_p3_mean(cld_leg['ssfr_zen_wvl'])
            f_Fdn_p3_mean = interp1d(effective_wvl, np.nanmean(Fdn_p3, axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_p3_mean_interp = f_Fdn_p3_mean(cld_leg['ssfr_zen_wvl'])
            
            f_Fdn_p3_direct_mean = interp1d(effective_wvl, np.nanmean(flux_down_dir_results[:, :, 1], axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_p3_direct_mean_interp = f_Fdn_p3_direct_mean(cld_leg['ssfr_zen_wvl'])
            f_Fdn_p3_diff_mean = interp1d(effective_wvl, np.nanmean(flux_down_diff_results[:, :, 1], axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_p3_diff_mean_interp = f_Fdn_p3_diff_mean(cld_leg['ssfr_zen_wvl'])
            
            f_Fdn_p3_diff_ratio_mean = interp1d(effective_wvl, np.nanmean(Fdn_p3_diff_ratio, axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_p3_diff_ratio_mean_interp = f_Fdn_p3_diff_ratio_mean(cld_leg['ssfr_zen_wvl'])
            
            f_Fup_toa_mean = interp1d(effective_wvl, np.nanmean(Fup_toa, axis=0), bounds_error=False, fill_value=np.nan)
            Fup_toa_mean_interp = f_Fup_toa_mean(cld_leg['ssfr_zen_wvl'])
            f_Fdn_toa_mean = interp1d(effective_wvl, np.nanmean(Fdn_toa, axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_toa_mean_interp = f_Fdn_toa_mean(cld_leg['ssfr_zen_wvl'])

            
            # SSFR observation
            fup_mean = np.nanmean(cld_leg['ssfr_nad'], axis=0)
            fdn_mean = np.nanmean(cld_leg['ssfr_zen'], axis=0)
            fup_std = np.nanstd(cld_leg['ssfr_nad'], axis=0)
            fdn_std = np.nanstd(cld_leg['ssfr_zen'], axis=0)
            
            # hsr1
            hsr1_dn_dif_ratio_mean = np.nanmean(cld_leg['hsr1_dif']/cld_leg['hsr1_tot'], axis=0)
            
            # surface albedo correction following Odell's correction method
            corr_up = Fup_p3_mean_interp / fup_mean
            corr_dn = Fdn_p3_mean_interp / fdn_mean
            
            alb_wvl = cld_leg['ssfr_zen_wvl']
            alb_obs = np.nanmean(cld_leg['ssfr_nad']/cld_leg['ssfr_zen'], axis=0)
            alb_obs[alb_obs<0.0] = 0.0
            alb_obs[alb_obs>1.0] = 1.0
            
            
            alb_corr = alb_obs * (corr_dn/corr_up)
            alb_corr[:4] = alb_corr[4]
            alb_corr[alb_corr<0.0] = 0.0
            alb_corr[alb_corr>1.0] = 1.0
            
            alb_corr_mask = alb_masking(alb_wvl, alb_corr)
            
            alb_ice_fit = ice_alb_fitting(alb_wvl, alb_corr)
            

            
            heading_saa_diff = heading_avg - saa_avg
            if heading_saa_diff < 0:
                heading_saa_diff += 360.0
            phase_diff = 135
            fdn_mean_scale = fdn_mean * np.sin(np.radians(heading_saa_diff - phase_diff)) * 0.03 + 0.97 #+ 0.015
            
            
            pop, pcov = curve_fit(exp_decay, cld_leg['hsr1_wvl'], hsr1_dn_dif_ratio_mean, p0=[0.3, 500.0, 0.1],)# bounds=([0.0, 400.0, 0.0], [1.0, 700.0, 1.0]))
            hsr1_dn_dif_ratio_mean_interp = exp_decay(cld_leg['ssfr_zen_wvl'], *pop)
            hsr1_dn_dif_ratio_mean_interp[hsr1_dn_dif_ratio_mean_interp<0.0] = 0.0
            hsr1_dn_dif_ratio_mean_interp[hsr1_dn_dif_ratio_mean_interp>1.0] = 1
            f_dn_direct = fdn_mean * (1-hsr1_dn_dif_ratio_mean_interp)
            f_dn_diff = fdn_mean * hsr1_dn_dif_ratio_mean_interp
            f_dn_diff_scale = f_dn_diff * Fdn_p3_diff_ratio_mean_interp/hsr1_dn_dif_ratio_mean_interp
            fdn_mean_low_dif = f_dn_direct + f_dn_diff_scale
            
            fdn_sim_high_dif = Fdn_p3_direct_mean_interp +  Fdn_p3_diff_mean_interp * hsr1_dn_dif_ratio_mean_interp / Fdn_p3_diff_ratio_mean_interp
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            toa_mean = np.nanmean(cld_leg['ssfr_toa'], axis=0)
            ax.plot(cld_leg['ssfr_zen_wvl'], toa_mean, '--', color='gray', linewidth=1.0, label='TOA')
            ax.plot(cld_leg['ssfr_zen_wvl'], fup_mean, '--', linewidth=1.0, color='royalblue', label='SSFR upward')
            ax.fill_between(cld_leg['ssfr_zen_wvl'],
                            fup_mean-fup_std,
                            fup_mean+fup_std, color='paleturquoise', alpha=0.75)
            ax.plot(cld_leg['ssfr_zen_wvl'], fdn_mean, '--', linewidth=1.0, color='orange', label='SSFR downward')
            ax.fill_between(cld_leg['ssfr_zen_wvl'],
                            fdn_mean-fdn_std,
                            fdn_mean+fdn_std, color='bisque', alpha=0.75)
            ax.plot(effective_wvl, np.nanmean(Fup_p3, axis=0), color='green', linewidth=2.0, label='Simulation upward')
            ax.plot(effective_wvl, np.nanmean(Fdn_p3, axis=0), color='red', linewidth=2.0, label='Simulation downward')
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Flux (W m$^{-2}$ nm$^{-1}$)', fontsize=12)
            ax.set_xlim(cld_leg['ssfr_zen_wvl'][0], cld_leg['ssfr_zen_wvl'][-1])
            ymin, ymax = ax.get_ylim()
            ax.set_ylim([0, ymax])
            ax.legend()
            if iter == 0:
                ax.set_title(f'{date_s} {time_start:.2f}-{time_end:.2f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo = SSFR upward/downward ratio', fontsize=10)
            elif iter == 1:
                ax.set_title(f'{date_s} {time_start:.2f}-{time_end:.2f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo updated (Odell)', fontsize=10)
            elif iter == 2:
                ax.set_title(f'{date_s} {time_start:.2f}-{time_end:.2f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo updated (fit)', fontsize=10)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_time_%.2f-%.2f_alt-%.2fkm_flux_iteration_%d.png' % (date_s, date_s, case_tag, time_start, time_end, alt_avg, iter), bbox_inches='tight', dpi=150)
            # plt.show()
            
        

            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            toa_mean = np.nanmean(cld_leg['ssfr_toa'], axis=0)
            l1 = ax.plot(cld_leg['ssfr_zen_wvl'], fdn_mean/toa_mean, '--', linewidth=1.0, color='orange', label='SSFR downward/TOA')
            ax.fill_between(cld_leg['ssfr_zen_wvl'],
                            (fdn_mean-fdn_std)/toa_mean,
                            (fdn_mean+fdn_std)/toa_mean, color='bisque', alpha=0.75)
            
            l2 = ax.plot(cld_leg['ssfr_zen_wvl'], Fdn_p3_mean_interp/toa_mean, color='red', linewidth=2.0, label='Simulation downward/TOA')
            l5 = ax.plot(cld_leg['ssfr_zen_wvl'], Fdn_toa_mean_interp/toa_mean, color='green', linewidth=2.0, label='Simulation TOA dn/TOA')
            # l6 = ax.plot(cld_leg['ssfr_zen_wvl'], fdn_mean_low_dif/toa_mean, color='purple', linestyle=':', linewidth=1.0, label='SSFR downward low-diff/TOA')
            # l7 = ax.plot(cld_leg['ssfr_zen_wvl'], fdn_sim_high_dif/toa_mean, color='brown', linestyle='-.', linewidth=1.0, label='Simulation downward high-diff/TOA')
            
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Downward flux / TOA ratio ', fontsize=12)
            
            ax2 = ax.twinx()
            l3 = ax2.plot(cld_leg['hsr1_wvl'], hsr1_dn_dif_ratio_mean, color='brown', linestyle='--', linewidth=1.0, label='HSR-1 diffuse ratio')
            l4 = ax2.plot(cld_leg['ssfr_zen_wvl'], Fdn_p3_diff_ratio_mean_interp, color='magenta', linestyle='-', linewidth=1.0, label='Simulation diffuse ratio')
            
            
            ax2.set_ylabel('Downward diffuse ratio', fontsize=12)
            ax2.set_ylim([0, 0.4])
            
            ax.set_xlim(cld_leg['ssfr_zen_wvl'][0], cld_leg['ssfr_zen_wvl'][-1])
            # plot horizontal line at y=1.0
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)
            
            ax.set_ylim([0.75, 1.15])
            
            # lns = l1 + l2 + l5 + l6 + l7 + l3 + l4
            lns = l1 + l2 + l5 + l3 + l4
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, fontsize=8, loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            if iter == 0:
                ax.set_title(f'{date_s} {time_start:.2f}-{time_end:.2f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo = SSFR upward/downward ratio', fontsize=10)
            elif iter == 1:
                ax.set_title(f'{date_s} {time_start:.2f}-{time_end:.2f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo updated (Odell)', fontsize=10)
            elif iter == 2:
                ax.set_title(f'{date_s} {time_start:.2f}-{time_end:.2f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo updated (fit)', fontsize=10)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_time_%.2f-%.2f_alt-%.2fkm_toa_dnflux_toa_ratio_iteration_%d.png' % (date_s, date_s, case_tag, time_start, time_end, alt_avg, iter), bbox_inches='tight', dpi=150)

            
            if iter == 0:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
                ax.plot(alb_wvl, alb_avg, label='SSFR upward/downward ratio')
                ax.plot(alb_wvl, alb_corr, label='updated albedo (Odell)')
                ax.plot(alb_wvl, alb_ice_fit, label='updated albedo (fit)')
                # fill between wavelengths where T_total < 0.05
                ax.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(alb_corr_mask), color='gray', alpha=0.2, label='Mask Gas absorption bands')
                
                ax.set_xlabel('Wavelength (nm)', fontsize=12)
                ax.set_ylabel('Albedo', fontsize=12)
                ax.set_ylim([-0.05, 1.05])
                ax.set_xlim(cld_leg['ssfr_zen_wvl'][0], cld_leg['ssfr_zen_wvl'][-1])
                ax.legend(fontsize=10)
                ax.set_title(f'{date_s} {time_start:.2f}-{time_end:.2f} Alt {alt_avg:.2f}km')
                fig.tight_layout()
                fig.savefig('fig/%s/%s_%s_time_%.2f-%.2f_alt-%.2fkm_albedo_iteration_%d.png' % (date_s, date_s, case_tag, time_start, time_end, alt_avg, iter), bbox_inches='tight', dpi=150)
                # plt.show()
            # sys.exit()
        
            
            output_dict = {
                'wvl': cld_leg['ssfr_zen_wvl'],
                'ssfr_fup_mean': fup_mean,
                'ssfr_fdn_mean': fdn_mean,
                'ssfr_fup_std': fup_std,
                'ssfr_fdn_std': fdn_std,
                'simu_fup_mean': Fup_p3_mean_interp,
                'simu_fdn_mean': Fdn_p3_mean_interp,
                'simu_fup_toa_mean': Fup_toa_mean_interp,
                'toa_mean': toa_mean,
                'ssrr_zen_wvl': cld_leg['ssrr_zen_wvl'],
                'ssrr_nad_wvl': cld_leg['ssrr_zen_wvl'],
                'ssrr_rad_dn_mean': np.nanmean(cld_leg['ssrr_zen_rad'], axis=0),
                'ssrr_rad_up_mean': np.nanmean(cld_leg['ssrr_nad_rad'], axis=0),
                'ssrr_rad_dn_std': np.nanstd(cld_leg['ssrr_zen_rad'], axis=0),
                'ssrr_rad_up_std': np.nanstd(cld_leg['ssrr_nad_rad'], axis=0),
            }
            
            output_df = pd.DataFrame(output_dict)
            output_df.to_csv(f'{fdir}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_{iter}.csv', index=False)
            if iter == 0:
                alb_wvl_extend = np.concatenate(([348.0], alb_wvl, [2050.0]))
                alb_corr_extend = np.concatenate(([alb_corr[0]], alb_corr, [alb_corr[-1]]))
                alb_ice_fit_extend = np.concatenate(([alb_ice_fit[0]], alb_ice_fit, [alb_ice_fit[-1]]))
                # write out the new surface albedo
                #/----------------------------------------------------------------------------\#                    
                alb_avg_update3 = alb_corr_extend.copy()
                alb_avg_update3_nonnan_first_ind = np.where(~np.isnan(alb_avg_update3))[0][0]
                alb_avg_update3[:alb_avg_update3_nonnan_first_ind] = alb_avg_update3[alb_avg_update3_nonnan_first_ind]
                alb_avg_update3_nonnan_last_ind = np.where(~np.isnan(alb_avg_update3))[0][-1]
                alb_avg_update3[alb_avg_update3_nonnan_last_ind:] = alb_avg_update3[alb_avg_update3_nonnan_last_ind]
                write_2col_file(filename=os.path.join(f'{_fdir_general_}/sfc_alb', f'sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_1.dat'),
                                wvl=alb_wvl_extend,
                                val=alb_avg_update3,
                                header=(f'# SSFR atmospheric corrected sfc albedo {date_s}\n'
                                        '# wavelength (nm)      albedo (unitless)\n'
                                        )
                                )
                    
                alb_avg_update4 = alb_ice_fit_extend.copy()
                alb_avg_update4_nonnan_first_ind = np.where(~np.isnan(alb_avg_update4))[0][0]
                alb_avg_update4[:alb_avg_update4_nonnan_first_ind] = alb_avg_update4[alb_avg_update4_nonnan_first_ind]
                alb_avg_update4_nonnan_last_ind = np.where(~np.isnan(alb_avg_update4))[0][-1]
                alb_avg_update4[alb_avg_update4_nonnan_last_ind:] = alb_avg_update4[alb_avg_update4_nonnan_last_ind]
                write_2col_file(filename=os.path.join(f'{_fdir_general_}/sfc_alb', f'sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_2.dat'),
                                wvl=alb_wvl_extend,
                                val=alb_avg_update4,
                                header=(f'# SSFR atmospheric corrected sfc albedo {date_s} with smooth fitting\n'
                                        '# wavelength (nm)      albedo (unitless)\n'
                                        )
                                )
                #\----------------------------------------------------------------------------/#
                del alb_avg_update3, alb_avg_update4
            if iter > 0:
                # write out the new simulated p3 level upward to downward ratio
                #/----------------------------------------------------------------------------\#
                p3_up_to_dn_ratio_update = p3_up_to_dn_ratio_mean
                p3_up_to_dn_ratio_update_nonnan_first_ind = np.where(~np.isnan(p3_up_to_dn_ratio_update))[0][0]
                p3_up_to_dn_ratio_update[:p3_up_to_dn_ratio_update_nonnan_first_ind] = alb_avg[p3_up_to_dn_ratio_update_nonnan_first_ind]
                p3_up_to_dn_ratio_update_nonnan_last_ind = np.where(~np.isnan(p3_up_to_dn_ratio_update))[0][-1]
                p3_up_to_dn_ratio_update[p3_up_to_dn_ratio_update_nonnan_last_ind:] = p3_up_to_dn_ratio_update[p3_up_to_dn_ratio_update_nonnan_last_ind]
                write_2col_file(filename=os.path.join(f'{_fdir_general_}/sfc_alb', f'p3_up_dn_ratio_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}_{iter}.dat'),
                                wvl=alb_wvl,
                                val=p3_up_to_dn_ratio_update,
                                header=(f'# SSFR atmospheric corrected sfc albedo {date_s} iteration {iter+1}\n'
                                        '# wavelength (nm)      albedo (unitless)\n'
                                        )
                                )
                #\----------------------------------------------------------------------------/#

        del output_dict, output_df
        del cld_leg, Fup_p3, Fdn_p3
        del Fup_p3_mean_interp, Fdn_p3_mean_interp
        del Fup_toa_mean_interp, Fdn_toa_mean_interp
        del fup_mean, fdn_mean, fup_std, fdn_std
        del toa_mean
        del alb_avg, alb_ice_fit
        
        gc.collect()

    print("Finished libratran calculations.")  
    #\----------------------------------------------------------------------------/#

    return


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
    
    # for date in [
    #              datetime.datetime(2024, 5, 28),
    #              datetime.datetime(2024, 5, 30),
    #              datetime.datetime(2024, 5, 31),
    #              datetime.datetime(2024, 6, 3),
    #              datetime.datetime(2024, 6, 5),
    #              datetime.datetime(2024, 6, 6),
    #              datetime.datetime(2024, 6, 7),
    #              datetime.datetime(2024, 6, 10),
    #              datetime.datetime(2024, 6, 11),
    #              datetime.datetime(2024, 6, 13),
    #              datetime.datetime(2024, 7, 25),
    #              datetime.datetime(2024, 7, 29),
    #              datetime.datetime(2024, 7, 30),
    #              datetime.datetime(2024, 8,  1),
    #              datetime.datetime(2024, 8,  2),
    #              datetime.datetime(2024, 8,  7),
    #              datetime.datetime(2024, 8,  8),
    #              datetime.datetime(2024, 8,  9),
    #              datetime.datetime(2024, 8,  15),
    #              ]:
    #     flt_toa_check(date=date,
    #                 config=config)


    ################################################################################
    
    # need to run arcsix_gas_insitu.py first to generate gas files for each date
    
    # for iter in range(2):
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
    
    # for iter in range(2):
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
    
    # for iter in range(2):
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
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),
                        tmhr_ranges_select=[
                                            [11.29, 11.86],
                                            [11.87, 13.23],
                                            [13.23, 13.44],
                                            [16.38, 17.80],
                                            
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 6, 3),
                        tmhr_ranges_select=[
                                            [11.29, 11.86],
                                            [11.87, 13.23],
                                            [13.23, 13.44],
                                            [16.38, 17.80],
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
                        tmhr_ranges_select=[
                                            [11.33, 11.88],
                                            [12.00, 12.20],
                                            [12.33, 13.80],
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 6, 5),
                        tmhr_ranges_select=[
                                            [11.33, 11.88],
                                            [12.00, 12.20],
                                            [12.33, 13.80],
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
                        tmhr_ranges_select=[
                                            [11.29, 13.31],
                                            [17.26, 18.32],
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 6, 6),
                        tmhr_ranges_select=[
                                            [11.29, 13.31],
                                            [17.26, 18.32],
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 6, 7),
                        tmhr_ranges_select=[
                                            [13.61, 14.10],
                                            [14.17, 14.30],
                                            [14.60, 14.92],
                                            [17.67, 18.25],
                                            [18.33, 18.52]
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 6, 7),
                        tmhr_ranges_select=[
                                            [13.61, 14.10],
                                            [14.17, 14.30],
                                            [14.60, 14.92],
                                            [17.67, 18.25],
                                            [18.33, 18.52]
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    # for iter in range(2):
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
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),
                        tmhr_ranges_select=[
                                            [11.35, 12.50],
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 6, 11),
                        tmhr_ranges_select=[
                                            [11.35, 12.50],
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
                        tmhr_ranges_select=[
                                            [11.28, 13.04],
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 6, 13),
                        tmhr_ranges_select=[
                                            [11.28, 13.04],
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 7, 25),
                        tmhr_ranges_select=[
                                            [11.68, 12.40],
                                            [12.94, 13.15],
                                            [17.14, 17.31]
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 7, 25),
                        tmhr_ranges_select=[
                                            [11.68, 12.40],
                                            [12.94, 13.15],
                                            [17.14, 17.31]
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),
                        tmhr_ranges_select=[
                                            [13.05, 13.45],
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 7, 29),
                        tmhr_ranges_select=[
                                            [13.05, 13.45],
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 7, 30),
                        tmhr_ranges_select=[
                                            [11.40, 13.83],
                                            [16.30, 17.33]
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 7, 30),
                        tmhr_ranges_select=[
                                            [11.40, 13.83],
                                            [16.30, 17.33]
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 8, 1),
                        tmhr_ranges_select=[
                                            [11.52, 11.63],
                                            [12.19, 12.35],
                                            [12.61, 13.34],
                                            [16.08, 16.41],
                                            [16.71, 16.94],
                                            [17.75, 18.13],
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 8, 1),
                        tmhr_ranges_select=[
                                            [11.52, 11.63],
                                            [12.19, 12.35],
                                            [12.61, 13.34],
                                            [16.08, 16.41],
                                            [16.71, 16.94],
                                            [17.75, 18.13],
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 8, 2),
                        tmhr_ranges_select=[
                                            [11.96, 14.08],
                                            [16.74, 16.84],
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 8, 2),
                        tmhr_ranges_select=[
                                            [11.96, 14.08],
                                            [16.74, 16.84],
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    for iter in range(2):
        flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),
                        tmhr_ranges_select=[
                                            [11.96, 14.08],
                                            [16.74, 16.84],
                                            ],
                        case_tag='clear_sky_track_atm_corr',
                        config=config,
                        simulation_interval=15,
                        clear_sky=True,
                        overwrite_lrt=True,
                        manual_cloud=False,
                        iter=iter,
                        )

    atm_corr_plot(date=datetime.datetime(2024, 8, 7),
                        tmhr_ranges_select=[
                                            [11.96, 14.08],
                                            [16.74, 16.84],
                                            ],
                    case_tag='clear_sky_track_atm_corr',
                    simulation_interval=15,
                    config=config,
                    )
    
    # for iter in range(2):
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
    
    
    


        

