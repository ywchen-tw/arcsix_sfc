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
from util import *
# mpl.use('Agg')


import er3t


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

def atm_corr_processing(date=datetime.datetime(2024, 5, 31),
                        tmhr_ranges_select=[[14.10, 14.27]],
                        case_tag='default',
                        config: Optional[FlightConfig] = None,
                        simulation_interval: Optional[float] = None,  # in minutes, e.g., 10 for 10 minutes
                        clear_sky: bool = True
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
            while t_start < hi and t_start < (hi - 0.0167/6):  # 10s
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
    
    t_hsk = np.array(data_hsk["tmhr"])
    mistmatch_count = 0
    
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    initiation = True
    for i in range(len(tmhr_ranges_select)):
        time_start, time_end = tmhr_ranges_select[i][0], tmhr_ranges_select[i][-1]
        fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, time_start, time_end)
        
        with open(fname_pkl, 'rb') as f:
            cld_leg = pickle.load(f)
        alt_avg = np.nanmean(cld_leg['alt'])  # in km
        lon_avg = np.nanmean(data_hsk['lon'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])
        lat_avg = np.nanmean(data_hsk['lat'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])
        leg_lon_all = cld_leg['lon']
        leg_lat_all = cld_leg['lat']
        leg_alt_all = cld_leg['alt']
        leg_icing_all = cld_leg['ssfr_icing']
        leg_icing_pre_all = cld_leg['ssfr_icing_pre']


        
    
        ratio_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_0.dat'
        update_1_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_1.dat'
        update_2_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_2.dat'
        update_p3_1_fname = f'{_fdir_general_}/sfc_alb/p3_up_dn_ratio_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}_1.dat'
        update_p3_2_fname = f'{_fdir_general_}/sfc_alb/p3_up_dn_ratio_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}_2.dat'
        
        
        if clear_sky:
            fdir_spiral = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
        else:
            fdir_spiral = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'
        ori_csv_name = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km_iteration_0.csv'
        updated_csv_name_1 = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km_iteration_1.csv'
        updated_csv_name_2 = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km_iteration_2.csv'
        
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
        
        corr_factor = df_ori['corr_factor'].values
        
        if initiation:
            time_all = []
            fdn_550_all = []
            fup_550_all = []
            fdn_1600_all = []
            fup_1600_all = []
            fdn_all = []
            fup_all = []
            lon_all = []
            lat_all = []
            alt_all = []
            icing_all = []
            icing_pre_all = []
            fdn_up_ratio_all = []
            toa_expand_all = []
            correction_factor_all = []
            
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
            
            initiation = False
            
            
            
        
        if corr_factor.shape[0] != cld_leg['ssfr_zen'].shape[1]:
            print("Mismatch in shape between corr_factor and ssfr_zen, skipping this leg.")
            print("i:", i)
            print("alb_wvl shape:", alb_wvl.shape)
            print("flux_wvl shape:", flux_wvl.shape)
            print("corr_factor shape:", corr_factor.shape)
            print("cld_leg['ssfr_zen'] shape:", cld_leg['ssfr_zen'].shape)
            mistmatch_count += 1
            continue

        time_all.append(time)
        ssfr_wvl = cld_leg['ssfr_zen_wvl']
        ssfr_550_ind = np.argmin(np.abs(ssfr_wvl - 550))
        ssfr_1600_ind = np.argmin(np.abs(ssfr_wvl - 1600))
        fdn_550_all.append(cld_leg['ssfr_zen'][:, ssfr_550_ind])
        fup_550_all.append(cld_leg['ssfr_nad'][:, ssfr_550_ind])
        fdn_1600_all.append(cld_leg['ssfr_zen'][:, ssfr_1600_ind])
        fup_1600_all.append(cld_leg['ssfr_nad'][:, ssfr_1600_ind])
        
        lon_all.extend(cld_leg['lon'])
        lat_all.extend(cld_leg['lat'])
        alt_all.extend(cld_leg['alt'])
        fdn_all.extend(cld_leg['ssfr_zen'])
        fup_all.extend(cld_leg['ssfr_nad'])
        toa_expand_all.extend(cld_leg['ssfr_toa'])
        correction_factor_all.extend(corr_factor * np.ones_like(cld_leg['ssfr_zen']))
        fdn_up_ratio_all.extend(cld_leg['ssfr_nad']/cld_leg['ssfr_zen'])#*corr_factor)
        icing_all.extend(leg_icing_all)
        icing_pre_all.extend(leg_icing_pre_all)
        
        # print("i:", i)
        # print("alb_ratio['alb'] shape:", alb_ratio['alb'].shape)
        # print("alb_ratio_all shape:", alb_ratio_all.shape)
        alb_ratio_all[i, :] = alb_ratio['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        alb1_all[i, :] = alb_1['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        alb2_all[i, :] = alb_2['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        p3_ratio1_all[i, :] = p3_ratio_1['ratio'].values
        p3_ratio2_all[i, :] = p3_ratio_2['ratio'].values
        lon_avg_all[i] = lon_avg.copy()
        lat_avg_all[i] = lat_avg.copy()
        lon_min_all[i] = leg_lon_all.min()
        lon_max_all[i] = leg_lon_all.max()
        lat_min_all[i] = leg_lat_all.min()
        lat_max_all[i] = leg_lat_all.max()
        alt_avg_all[i] = alt_avg.copy()
        
        
        ssfr_fup_mean_all[i, :] = df_ori['ssfr_fup_mean'].values
        ssfr_fdn_mean_all[i, :] = df_ori['ssfr_fdn_mean'].values
        ssfr_fup_std_all[i, :] = df_ori['ssfr_fup_std'].values
        ssfr_fdn_std_all[i, :] = df_ori['ssfr_fdn_std'].values
        # simu_fup_mean_all_iter0[i, :] = df_ori['simu_fup_mean'].values
        # simu_fdn_mean_all_iter0[i, :] = df_ori['simu_fdn_mean'].values
        # simu_fup_mean_all_iter1[i, :] = df_upd1['simu_fup_mean'].values
        # simu_fdn_mean_all_iter1[i, :] = df_upd1['simu_fdn_mean'].values
        # simu_fup_mean_all_iter2[i, :] = df_upd2['simu_fup_mean'].values
        # simu_fdn_mean_all_iter2[i, :] = df_upd2['simu_fdn_mean'].values
        # simu_fup_toa_mean_all_iter0[i, :] = df_ori['simu_fup_toa_mean'].values
        # simu_fup_toa_mean_all_iter1[i, :] = df_upd1['simu_fup_toa_mean'].values
        # simu_fup_toa_mean_all_iter2[i, :] = df_upd2['simu_fup_toa_mean'].values
        toa_mean_all[i, :] = df_ori['toa_mean'].values
        
        
        print(f"date_s: {date_s}, time: {time_start:.3f}-{time_end:.3f}, alt_avg: {alt_avg:.2f} km")
        
        
        # if np.all(np.isnan((cld_leg['ssfr_zen']/cld_leg['ssfr_nad'])[0, :])):
        #     continue
        # alb_corr = copy.deepcopy((cld_leg['ssfr_zen']/cld_leg['ssfr_nad'])[0, :])
        # alb_corr[alb_corr < 0] = 0
        # alb_corr[alb_corr > 1] = 1
        
        # if np.any(np.isnan(alb_corr)):
        #     s = pd.Series(alb_corr)
        #     s_mask = np.isnan(alb_corr)
        #     # Fills NaN with the value immediately preceding it
        #     s_ffill = s.fillna(method='ffill', limit=2)
        #     s_ffill = s_ffill.fillna(method='bfill', limit=2)
        #     while np.any(np.isnan(s_ffill)):
        #         s_ffill = s_ffill.fillna(method='ffill', limit=2)
        #         s_ffill = s_ffill.fillna(method='bfill', limit=2)
                
        #     alb_corr[s_mask] = s_ffill[s_mask]
        
        # alb_corr = alb_corr * corr_factor
        # fdn_up_ratio_all_corr_test = alb_corr.copy()
        # fdn_up_ratio_all_corr_test[fdn_up_ratio_all_corr_test < 0] = 0
        # fdn_up_ratio_all_corr_test[fdn_up_ratio_all_corr_test > 1] = 1
                    
        # alb_corr_mask = gas_abs_masking(alb_wvl, alb_corr, alt=cld_leg['alt'][0])
        # alb_corr[np.isnan(alb_corr)] = alb_corr_mask[np.isnan(alb_corr)]
        
        # fdn_up_ratio_all_corr_fit_test = snowice_alb_fitting(alb_wvl, alb_corr, alt=alt_all[i], clear_sky=clear_sky)
        # print("fdn_up_ratio_all_corr_test shape:", fdn_up_ratio_all_corr_test.shape)
        # print("cld_leg['ssfr_toa'] shape:", cld_leg['ssfr_toa'].shape)

        # broadband_alb_iter1_test = np.sum(fdn_up_ratio_all_corr_fit_test * cld_leg['ssfr_toa'][0, :]) / np.sum(cld_leg['ssfr_toa'][0, :])
           
        # plt.close('all')
        # fig, ax = plt.subplots(figsize=(8, 6))
        # # ax.plot(alb_wvl, (cld_leg['ssfr_zen']/cld_leg['ssfr_nad'])[0, :], '-', color='r', label='Original albedo')
        # ax.plot(alb_wvl, fdn_up_ratio_all_corr_fit_test, '-', color='b', label='Corrected albedo (iter 1)')
        # ax.set_xlabel('Wavelength (nm)', fontsize=14)
        # ax.set_ylabel('Surface Albedo', fontsize=14)
        # ax.set_title(f'Surface Albedo after Atmospheric Correction {date_s} {case_tag}, broadband alb: {broadband_alb_iter1_test:.3f}', fontsize=13)
        # fig.suptitle(f'Flight leg at Z={alt_avg_all[0]:.2f} km, lon={lon_avg_all[0]:.2f}, lat={lat_avg_all[0]:.2f}', fontsize=12, y=0.98)
        # fig.tight_layout()
        # plt.show()
        # sys.exit()
    

        # find the modis location closest to the flight leg center
        if modis_alb_file is not None:
            dist = np.sqrt((modis_lon - lon_avg)**2 + (modis_lat - lat_avg)**2)
            min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            modis_alb_leg = modis_sur_alb[min_idx[0], min_idx[1], :7]
            modis_alb_legs.append(modis_alb_leg)
    
    print(f"Total mismatch count: {mistmatch_count}")
    
    time_all = np.array(time_all)
    fdn_550_all = np.array(fdn_550_all)
    fup_550_all = np.array(fup_550_all)
    fdn_1600_all = np.array(fdn_1600_all)
    fup_1600_all = np.array(fup_1600_all)
    
    lon_all = np.array(lon_all)
    lat_all = np.array(lat_all)
    alt_all = np.array(alt_all)
    fdn_all = np.array(fdn_all)
    fup_all = np.array(fup_all)
    toa_expand_all = np.array(toa_expand_all)
    fdn_up_ratio_all = np.array(fdn_up_ratio_all)
    correction_factor_all = np.array(correction_factor_all)
    icing_all = np.array(icing_all)
    icing_pre_all = np.array(icing_pre_all)

    # print("lon avg all mean:", lon_avg_all.mean())
    # print("lat avg all mean:", lat_avg_all.mean())
    # print("fdn_all shape:", fdn_all.shape)
    # print("fup_all shape:", fup_all.shape)
    # print("fdn_550_all shape:", fdn_550_all.shape)
    # print("fup_dn_ratio_all_corr shape:", fdn_up_ratio_all_corr.shape)
    # print("correction_factor_all shape:", correction_factor_all.shape)
    # sys.exit()
    
    fdn_up_ratio_all_corr = np.full(fdn_up_ratio_all.shape, np.nan)
    fdn_up_ratio_all_corr_fit = np.full(fdn_up_ratio_all.shape, np.nan)
    for i in range(fdn_up_ratio_all.shape[0]):
        if np.all(np.isnan(fdn_up_ratio_all[i, :])):
            continue
        alb_corr = fdn_up_ratio_all[i, :].copy()
        alb_corr[alb_corr < 0] = 0
        alb_corr[alb_corr > 1] = 1
        
        if np.any(np.isnan(alb_corr)):
            s = pd.Series(alb_corr)
            s_mask = np.isnan(alb_corr)
            # Fills NaN with the value immediately preceding it
            s_ffill = s.fillna(method='ffill', limit=2)
            s_ffill = s_ffill.fillna(method='bfill', limit=2)
            while np.any(np.isnan(s_ffill)):
                s_ffill = s_ffill.fillna(method='ffill', limit=2)
                s_ffill = s_ffill.fillna(method='bfill', limit=2)
                
            alb_corr[s_mask] = s_ffill[s_mask]
        
        alb_corr = alb_corr * correction_factor_all[i, :]
        alb_corr[alb_corr < 0] = 0
        alb_corr[alb_corr > 1] = 1
        fdn_up_ratio_all_corr[i, :] = alb_corr.copy()
        
        if date_s not in ['20240603', '20240807']:
            alb_corr_mask = gas_abs_masking(alb_wvl, alb_corr, alt=alt_all[i])
        else:
            alb_corr_mask = gas_abs_masking(alb_wvl, alb_corr, alt=alt_all[i], h2o_6_end=1600) # extend h2o_6 to 1600 nm for this date
        alb_corr[np.isnan(alb_corr)] = alb_corr_mask[np.isnan(alb_corr)]
        
        fdn_up_ratio_all_corr_fit[i, :] = snowice_alb_fitting(alb_wvl, alb_corr, alt=alt_all[i], clear_sky=clear_sky)

    
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
    broadband_alb_iter1_all = np.sum(fdn_up_ratio_all_corr * toa_expand_all, axis=1) / np.sum(toa_expand_all, axis=1)
    broadband_alb_iter2_all = np.sum(fdn_up_ratio_all_corr_fit * toa_expand_all, axis=1) / np.sum(toa_expand_all, axis=1)

    gas_mask = np.isfinite(gas_abs_masking(alb_wvl, np.ones_like(alb_wvl, dtype=float), alt=1))
    broadband_alb_iter0_filter = np.sum(alb_ratio_all[:, gas_mask] * toa_mean_all[:, gas_mask], axis=1) / np.sum(toa_mean_all[:, gas_mask], axis=1)
    broadband_alb_iter1_filter = np.sum(alb1_all[:, gas_mask] * toa_mean_all[:, gas_mask], axis=1) / np.sum(toa_mean_all[:, gas_mask], axis=1)
    broadband_alb_iter2_filter = np.sum(alb2_all[:, gas_mask] * toa_mean_all[:, gas_mask], axis=1) / np.sum(toa_mean_all[:, gas_mask], axis=1)
    broadband_alb_iter1_all_filter = np.sum(fdn_up_ratio_all_corr[:, gas_mask] * toa_expand_all[:, gas_mask], axis=1) / np.sum(toa_expand_all[:, gas_mask], axis=1)
    broadband_alb_iter2_all_filter = np.sum(fdn_up_ratio_all_corr_fit[:, gas_mask] * toa_expand_all[:, gas_mask], axis=1) / np.sum(toa_expand_all[:, gas_mask], axis=1)
    
    # plt.close('all')
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(alb_wvl, fdn_up_ratio_all_corr[0, :], '-', color='b', label='Corrected albedo (iter 1)')
    # ax.set_xlabel('Wavelength (nm)', fontsize=14)
    # ax.set_ylabel('Surface Albedo', fontsize=14)
    # ax.set_title(f'Surface Albedo after Atmospheric Correction {date_s} {case_tag}, broadband alb: {broadband_alb_iter1[0]:.3f}', fontsize=13)
    # fig.suptitle(f'Flight leg at Z={alt_avg_all[0]:.2f} km, lon={lon_avg_all[0]:.2f}, lat={lat_avg_all[0]:.2f}', fontsize=12, y=0.98)
    # fig.tight_layout()
    # plt.show()
    # sys.exit()
    
    fig_dir = f'fig/sfc_alb_corr_lonlat'
    os.makedirs(fig_dir, exist_ok=True)
    # set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
    try:
        plt.close('all')
        central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
        proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(1, 1, 1, projection=proj)

        # add coastlines and land features for context
        ax.coastlines(resolution='50m', linewidth=0.8)
        ax.add_feature(cartopy.feature.LAND, facecolor='#f0f0f0', zorder=0)
        ax.add_feature(cartopy.feature.OCEAN, facecolor='#e6f2ff', zorder=0)

        # gridlines with labels (only lon/lat labels make sense in PlateCarree transform)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # scatter the flight-leg centers (and all points) using PlateCarree transform
        # color by broadband surface albedo from iteration 2 if available
        color_vals = np.array(broadband_alb_iter2) if 'broadband_alb_iter2' in locals() else None
        if color_vals is None or np.all(np.isnan(color_vals)):
            sc = ax.scatter(lon_avg_all, lat_avg_all, s=40, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor='k', label='Flight legs')
        else:
            sc = ax.scatter(lon_avg_all, lat_avg_all, s=40, c=color_vals, cmap='viridis',
                            transform=ccrs.PlateCarree(), zorder=3, edgecolor='k')
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
            cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

        # also plot all sampled points along legs (optional, lighter marker)
        if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
            ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

        ax.set_title(f'Polar projection (North) - Flight legs {date_s} {case_tag}', fontsize=12)
        # set a reasonable display extent around the data if available (lon/lat box)
        if len(lon_avg_all) > 0 and len(lat_avg_all) > 0:
            lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
            lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
            # expand a bit
            pad_lon = max(0.5, (lon_max - lon_min) * 0.2)
            pad_lat = max(0.5, (lat_max - lat_min) * 0.2)
            try:
                ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
            except Exception:
                # fallback: don't set extent if projection complains
                pass
        ax.tick_params('both', labelsize=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{date_s}_{case_tag}_broadband_albedo_vs_longitude_polar_projection.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
    except Exception as e:
        # fallback to simple lon/lat scatter if cartopy fails
        print("Cartopy polar plot failed, falling back to Cartesian scatter. Error:", e)
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(lon_avg_all, lat_avg_all, c=broadband_alb_iter2 if 'broadband_alb_iter2' in locals() else 'r', cmap='viridis', s=40, edgecolor='k')
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Broadband Albedo  (atm corr + fit)', fontsize=12)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Flight legs {date_s} {case_tag}')
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{date_s}_{case_tag}_broadband_albedo_vs_longitude_cartesian.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
    

    log.info("Saved broadband albedo vs longitude plot to %s/%s_%s_broadband_albedo_vs_longitude.png", fig_dir, date_s, case_tag)
    
    # save alb1 and alb2 to a pkl file
    alb_update_dict = {
        'wvl': alb_wvl,
        'alb_iter0': alb_ratio_all,
        'alb_iter1': alb1_all,
        'alb_iter2': alb2_all,
        'p3_up_dn_ratio_1': p3_ratio1_all,
        'p3_up_dn_ratio_2': p3_ratio2_all,
        'broadband_alb_iter0': broadband_alb_iter0,
        'broadband_alb_iter1': broadband_alb_iter1,
        'broadband_alb_iter2': broadband_alb_iter2,
        'broadband_alb_iter0_filter': broadband_alb_iter0_filter,
        'broadband_alb_iter1_filter': broadband_alb_iter1_filter,
        'broadband_alb_iter2_filter': broadband_alb_iter2_filter,
        'lon_avg': lon_avg_all,
        'lat_avg': lat_avg_all,
        'lon_min': lon_min_all,
        'lon_max': lon_max_all,
        'lat_min': lat_min_all,
        'lat_max': lat_max_all,
        'alt_avg': alt_avg_all,
        'lon_all': lon_all,
        'lat_all': lat_all,
        'alt_all': alt_all,
        'fdn_all': fdn_all,
        'fup_all': fup_all,
        'toa_expand_all': toa_expand_all,
        'fdn_up_ratio_all': fdn_up_ratio_all,
        'correction_factor_all': correction_factor_all,
        'icing_all': icing_all,
        'icing_pre_all': icing_pre_all,
        'alb_iter1_all': fdn_up_ratio_all_corr,
        'alb_iter2_all': fdn_up_ratio_all_corr_fit,
        'broadband_alb_iter1_all': broadband_alb_iter1_all,
        'broadband_alb_iter2_all': broadband_alb_iter2_all,
        'broadband_alb_iter1_all_filter': broadband_alb_iter1_all_filter,
        'broadband_alb_iter2_all_filter': broadband_alb_iter2_all_filter,
        'modis_alb_legs': np.array(modis_alb_legs) if modis_alb_file is not None else None,
        'modis_bands_nm': modis_bands_nm if modis_alb_file is not None else None,
    }
    if modis_alb_file is None:
        modis_bands_nm = None
        modis_alb_legs = None
        
    output_dir = f'{_fdir_general_}/sfc_alb_combined'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{_fdir_general_}/{output_dir}/sfc_alb_update_{date_s}_{case_tag}_time_{tmhr_ranges_select[0][0]:.3f}_{tmhr_ranges_select[-1][-1]:.3f}.pkl', 'wb') as f:
        pickle.dump(alb_update_dict, f)
    log.info(f"Saved surface albedo updates to {_fdir_general_}/sfc_alb/sfc_alb_update_{date_s}_{case_tag}.pkl")

    
    print("Processing completed for date and tag:", date_s, case_tag)

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


    # atm_corr_processing(date=datetime.datetime(2024, 5, 28),
    #                 tmhr_ranges_select=[[15.610, 15.822],
    #                                     [16.905, 17.404] 
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 5, 31),
    #                 tmhr_ranges_select=[[13.839, 15.180],  # 5.6 km
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )

    

    # atm_corr_processing(date=datetime.datetime(2024, 5, 31),
    #                 tmhr_ranges_select=[
    #                                     [16.905, 17.404] 
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )

    
    # # done
    # atm_corr_processing(date=datetime.datetime(2024, 6, 3),
    #                 tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                     ],
    #                 case_tag='cloudy_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )


    # # done
    # atm_corr_processing(date=datetime.datetime(2024, 6, 3),
    #                 tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )
    

    # atm_corr_processing(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[[12.405, 13.812], # 5.7m,
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )
    


    # atm_corr_processing(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[
    #                                     [14.258, 15.036], # 100m
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[
    #                                     [15.535, 15.931], # 450m
    #                                     ],
    #                 case_tag='clear_atm_corr_3',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )
    


    # atm_corr_processing(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[
    #                                     [13.7889, 13.8010],
    #                                     [13.8350, 13.8395],
    #                                     [13.8780, 13.8885],
    #                                     [13.9240, 13.9255],
    #                                     [13.9389, 13.9403],
    #                                     [13.9540, 13.9715],
    #                                     [13.9980, 14.0153],
    #                                     [14.0417, 14.0575],
    #                                     [14.0417, 14.0475],
    #                                     [14.0560, 14.0590],
    #                                     [14.0825, 14.0975],
    #                                     [14.1264, 14.1525],
    #                                     [14.1762, 14.1975],
    #                                     [14.2194, 14.2420],
    #                                     [14.2605, 14.2810]
    #                                     ],
    #                 case_tag='clear_sky_spiral_atm_corr',
    #                 config=config,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 6, 6),
    #                 tmhr_ranges_select=[[16.250, 16.325], # 100m, 
    #                                     [16.375, 16.632], # 450m
    #                                     [16.700, 16.794], # 100m
    #                                     [16.850, 16.952], # 1.2km
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 6, 7),
    #                 tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 6, 11),
    #                 tmhr_ranges_select=[[14.5667, 14.5694],
    #                                     [14.5986, 14.6097],
    #                                     [14.6375, 14.6486], # cloud shadow
    #                                     [14.6778, 14.6903],
    #                                     [14.7208, 14.7403],
    #                                     [14.7653, 14.7875],
    #                                     [14.8125, 14.8278],
    #                                     [14.8542, 14.8736],
    #                                     [14.8986, 14.9389], # more cracks
    #                                     ],
    #                 case_tag='clear_sky_spiral_atm_corr',
    #                 config=config,
    #                 simulation_interval=None,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 6, 11),
    #                 tmhr_ranges_select=[
    #                                     [14.968, 15.229], # 100, clear, some cloud
    #                                     [14.968, 15.347],
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 6, 11),
    #                 tmhr_ranges_select=[
    #                                     [15.347, 15.813], # 100m
    #                                     [15.813, 16.115], # 100-450m, clear, some cloud
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[13.704, 13.817], # 100-450m, clear, some cloud
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[14.109, 14.140], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[15.834, 15.883], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )
    


    # atm_corr_processing(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[16.043, 16.067], # 100-200m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_3',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[16.550, 17.581], # 100-500m, clear
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 7, 25),
    #                 tmhr_ranges_select=[[15.094, 15.300], # 100m, some low clouds or fog below
    #                                     ],
    #                 case_tag='cloudy_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 7, 25),
    #                 tmhr_ranges_select=[[15.881, 15.903], # 200-500m
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )
    


    # atm_corr_processing(date=datetime.datetime(2024, 7, 29),
    #                 tmhr_ranges_select=[[13.442, 13.465],
    #                                     [13.490, 13.514],
    #                                     [13.536, 13.554],
    #                                     [13.580, 13.611],
    #                                     [13.639, 13.654],
    #                                     [13.676, 13.707],
    #                                     [13.733, 13.775],
    #                                     [13.793, 13.836],
    #                                     ],
    #                 case_tag='clear_sky_spiral_atm_corr',
    #                 config=config,
    #                 simulation_interval=None,
    #                 clear_sky=True,
    #                 )
    


    # atm_corr_processing(date=datetime.datetime(2024, 7, 29),
    #                 tmhr_ranges_select=[[13.939, 14.200], # 100m, clear
    #                                     [14.438, 14.714], # 3.7km
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 7, 29),
    #                 tmhr_ranges_select=[
    #                                     [15.214, 15.804], # 1.3km
    #                                     [16.176, 16.304], # 1.3km
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )
    


    # atm_corr_processing(date=datetime.datetime(2024, 7, 30),
    #                 tmhr_ranges_select=[[13.886, 13.908],
    #                                     [13.934, 13.950],
    #                                     [13.976, 14.000],
    #                                     [14.031, 14.051],
    #                                     [14.073, 14.096],
    #                                     [14.115, 14.134],
    #                                     [14.157, 14.179],
    #                                     [14.202, 14.219],
    #                                     [14.239, 14.254],
    #                                     [14.275, 14.294],
    #                                     ],
    #                 case_tag='clear_sky_spiral_atm_corr',
    #                 config=config,
    #                 simulation_interval=None,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 7, 30),
    #                 tmhr_ranges_select=[[14.318, 14.936], # 100-450m, clear
    #                                     [15.043, 15.140], # 1.5km
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 1),
    #                 tmhr_ranges_select=[[13.843, 14.361], # 100-450m, clear, some open ocean
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 1),
    #                 tmhr_ranges_select=[
    #                                     [14.739, 15.053], # 550m
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )
    



    # atm_corr_processing(date=datetime.datetime(2024, 8, 2),
    #                 tmhr_ranges_select=[
    #                                     [14.557, 15.100], # 100m
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 2),
    #                 tmhr_ranges_select=[
    #                                     [15.244, 16.635], # 1km
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 7),
    #                 tmhr_ranges_select=[[13.344, 13.763], # 100m, cloudy
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 7),
    #                 tmhr_ranges_select=[
    #                                     [15.472, 15.567], # 180m, cloudy
    #                                     [15.580, 15.921], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 8),
    #                 tmhr_ranges_select=[
    #                                     [12.990, 13.180], # 180m, clear
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 8),
    #                 tmhr_ranges_select=[
    #                                     [14.250, 14.373], # 180m, clear
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 8),
    #                 tmhr_ranges_select=[
    #                                     [16.471, 16.601], # 180m, clear
    #                                     ],
    #                 case_tag='clear_atm_corr_3',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 8),
    #                 tmhr_ranges_select=[
    #                                     [13.212, 13.347], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 8),
    #                 tmhr_ranges_select=[
    #                                     [15.314, 15.504], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 9),
    #                 tmhr_ranges_select=[
    #                                     [13.376, 13.600], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 9),
    #                 tmhr_ranges_select=[
    #                                     [14.750, 15.060], # 100m, clear
    #                                     [15.622, 15.887], # 100m, clear
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 9),
    #                 tmhr_ranges_select=[
    #                                     [16.029, 16.224], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 )
    


    # atm_corr_processing(date=datetime.datetime(2024, 8, 15),
    #                 tmhr_ranges_select=[
    #                                     [14.085, 14.396], # 100m, clear
    #                                     [14.550, 14.968], # 3.5km, clear
    #                                     [15.078, 15.163], # 1.7km, clear
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )