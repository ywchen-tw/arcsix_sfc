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

def combined_atm_corr():
    log = logging.getLogger("atm corr combined")

    output_dir = f'{_fdir_general_}/sfc_alb_combined'
    # glob all sfc alb files
    sfc_alb_files = sorted(glob.glob(f'{output_dir}/sfc_alb_*.pkl'))
    print(f"Found {len(sfc_alb_files)} surface albedo files for combination.")
    
    # read each file and combine data into a larger dictionary
    init_spring = True
    init_summer = True
    leg_contidions_spring = []
    leg_contidions_all_spring = []
    leg_contidions_summer = []
    leg_contidions_all_summer = []
    dates_spring = []
    dates_summer = []
    dates_spring_all = []
    dates_summer_all = []
    for sfc_alb_file in sfc_alb_files:
        print(f"Processing surface albedo file: {sfc_alb_file}")
        with open(sfc_alb_file, 'rb') as f:
            sfc_alb_data = pickle.load(f)
        
        # extract date and case tag from filename
        base_name = os.path.basename(sfc_alb_file)
        parts = base_name.replace('sfc_alb_update_', '').replace('.pkl', '').split('_')
        date_s = parts[0]
        case_tag = '_'.join(parts[1:]) if len(parts) > 1 else 'default'
        condition = 'cloudy'
        if 'clear' in case_tag.lower():
            condition = 'clear'
            
        # extract data
        
            # # save alb1 and alb2 to a pkl file
            # alb_update_dict = {
            #     'wvl': alb_wvl,
            #     'alb_iter0': alb_ratio_all,
            #     'alb_iter1': alb1_all,
            #     'alb_iter2': alb2_all,
            #     'p3_up_dn_ratio_1': p3_ratio1_all,
            #     'p3_up_dn_ratio_2': p3_ratio2_all,
            #     'broadband_alb_iter0': broadband_alb_iter0,
            #     'broadband_alb_iter1': broadband_alb_iter1,
            #     'broadband_alb_iter2': broadband_alb_iter2,
            #     'broadband_alb_iter0_filter': broadband_alb_iter0_filter,
            #     'broadband_alb_iter1_filter': broadband_alb_iter1_filter,
            #     'broadband_alb_iter2_filter': broadband_alb_iter2_filter,
            #     'lon_avg': lon_avg_all,
            #     'lat_avg': lat_avg_all,
            #     'lon_min': lon_min_all,
            #     'lon_max': lon_max_all,
            #     'lat_min': lat_min_all,
            #     'lat_max': lat_max_all,
            #     'alt_avg': alt_avg_all,
            #     'lon_all': lon_all,
            #     'lat_all': lat_all,
            #     'alt_all': alt_all,
            #     'fdn_all': fdn_all,
            #     'fup_all': fup_all,
            #     'toa_expand_all': toa_expand_all,
            #     'fdn_up_ratio_all': fdn_up_ratio_all,
            #     'correction_factor_all': correction_factor_all,
            #     'icing_all': icing_all,
            #     'icing_pre_all': icing_pre_all,
            #     'alb_iter1_all': fdn_up_ratio_all_corr,
            #     'alb_iter2_all': fdn_up_ratio_all_corr_fit,
            #     'broadband_alb_iter1_all': broadband_alb_iter1_all,
            #     'broadband_alb_iter2_all': broadband_alb_iter2_all,
            #     'broadband_alb_iter1_all_filter': broadband_alb_iter1_all_filter,
            #     'broadband_alb_iter2_all_filter': broadband_alb_iter2_all_filter,
            #     'modis_alb_legs': np.array(modis_alb_legs) if modis_alb_file is not None else None,
            #     'modis_bands_nm': modis_bands_nm if modis_alb_file is not None else None,
            # }
        
        if date_s < '20240630':
            wvl_spring = sfc_alb_data['wvl']
        else:
            wvl_summer = sfc_alb_data['wvl']
        
        
        if date_s < '20240630':
            if init_spring:
                alb_iter0_spring = sfc_alb_data['alb_iter0']
                alb_iter1_spring = sfc_alb_data['alb_iter1']
                alb_iter2_spring = sfc_alb_data['alb_iter2']
                broadband_alb_iter0_spring = sfc_alb_data['broadband_alb_iter0']
                broadband_alb_iter1_spring = sfc_alb_data['broadband_alb_iter1']
                broadband_alb_iter2_spring = sfc_alb_data['broadband_alb_iter2']
                broadband_alb_iter0_filter_spring = sfc_alb_data['broadband_alb_iter0']
                broadband_alb_iter1_filter_spring = sfc_alb_data['broadband_alb_iter1']
                broadband_alb_iter2_filter_spring = sfc_alb_data['broadband_alb_iter2']
                
                lon_avg_spring = sfc_alb_data['lon_avg']
                lat_avg_spring = sfc_alb_data['lat_avg']
                alt_avg_spring = sfc_alb_data['alt_avg']
                lon_all_spring = sfc_alb_data['lon_all']
                lat_all_spring = sfc_alb_data['lat_all']
                alt_all_spring = sfc_alb_data['alt_all']
                fdn_all_spring = sfc_alb_data['fdn_all']
                fup_all_spring = sfc_alb_data['fup_all']
                toa_expand_all_spring = sfc_alb_data['toa_expand_all']
                fdn_up_ratio_all_spring = sfc_alb_data['fdn_up_ratio_all']
                correction_factor_all_spring = sfc_alb_data['correction_factor_all']
                icing_all_spring = sfc_alb_data['icing_all']
                icing_pre_all_spring = sfc_alb_data['icing_pre_all']
                alb_iter1_all_spring = sfc_alb_data['alb_iter1_all']
                alb_iter2_all_spring = sfc_alb_data['alb_iter2_all']
                broadband_alb_iter1_all_spring = sfc_alb_data['broadband_alb_iter1_all']
                broadband_alb_iter2_all_spring = sfc_alb_data['broadband_alb_iter2_all']
                broadband_alb_iter1_all_filter_spring = sfc_alb_data['broadband_alb_iter1_all_filter']
                broadband_alb_iter2_all_filter_spring = sfc_alb_data['broadband_alb_iter2_all_filter']
                
                init_spring = False
            else:
                # concatenate all
                alb_iter0_spring = np.concatenate((alb_iter0_spring, sfc_alb_data['alb_iter0']))
                alb_iter1_spring_spring = np.concatenate((alb_iter1_spring, sfc_alb_data['alb_iter1']))
                alb_iter2_spring = np.concatenate((alb_iter2_spring, sfc_alb_data['alb_iter2']))
                broadband_alb_iter0_spring = np.concatenate((broadband_alb_iter0_spring, sfc_alb_data['broadband_alb_iter0']))
                broadband_alb_iter1_spring = np.concatenate((broadband_alb_iter1_spring, sfc_alb_data['broadband_alb_iter1']))
                broadband_alb_iter2_spring = np.concatenate((broadband_alb_iter2_spring, sfc_alb_data['broadband_alb_iter2']))
                broadband_alb_iter0_filter_spring = np.concatenate((broadband_alb_iter0_filter_spring, sfc_alb_data['broadband_alb_iter0']))
                broadband_alb_iter1_filter_spring = np.concatenate((broadband_alb_iter1_filter_spring, sfc_alb_data['broadband_alb_iter1']))
                broadband_alb_iter2_filter_spring = np.concatenate((broadband_alb_iter2_filter_spring, sfc_alb_data['broadband_alb_iter2']))
                lon_avg_spring = np.concatenate((lon_avg_spring, sfc_alb_data['lon_avg']))
                lat_avg_spring = np.concatenate((lat_avg_spring, sfc_alb_data['lat_avg']))
                alt_avg_spring = np.concatenate((alt_avg_spring, sfc_alb_data['alt_avg']))
                lon_all_spring = np.concatenate((lon_all_spring, sfc_alb_data['lon_all']))
                lat_all_spring = np.concatenate((lat_all_spring, sfc_alb_data['lat_all']))
                alt_all_spring = np.concatenate((alt_all_spring, sfc_alb_data['alt_all']))
                fdn_all_spring = np.concatenate((fdn_all_spring, sfc_alb_data['fdn_all']))
                fup_all_spring = np.concatenate((fup_all_spring, sfc_alb_data['fup_all']))
                toa_expand_all_spring = np.concatenate((toa_expand_all_spring, sfc_alb_data['toa_expand_all']))
                fdn_up_ratio_all_spring =  np.concatenate((fdn_up_ratio_all_spring, sfc_alb_data['fdn_up_ratio_all']))
                correction_factor_all_spring = np.concatenate((correction_factor_all_spring, sfc_alb_data['correction_factor_all']))
                
                icing_all_spring = np.concatenate((icing_all_spring, sfc_alb_data['icing_all']))
                icing_pre_all_spring = np.concatenate((icing_pre_all_spring, sfc_alb_data['icing_pre_all']))
                alb_iter1_all_spring = np.concatenate((alb_iter1_all_spring, sfc_alb_data['alb_iter1_all']))
                alb_iter2_all_spring = np.concatenate((alb_iter2_all_spring, sfc_alb_data['alb_iter2_all']))
                broadband_alb_iter1_all_spring = np.concatenate((broadband_alb_iter1_all_spring, sfc_alb_data['broadband_alb_iter1_all']))
                broadband_alb_iter2_all_spring = np.concatenate((broadband_alb_iter2_all_spring, sfc_alb_data['broadband_alb_iter2_all']))
                broadband_alb_iter1_all_filter_spring = np.concatenate((broadband_alb_iter1_all_filter_spring, sfc_alb_data['broadband_alb_iter1_all_filter']))
                broadband_alb_iter2_all_filter_spring = np.concatenate((broadband_alb_iter2_all_filter_spring, sfc_alb_data['broadband_alb_iter2_all_filter']))
            
            leg_contidions_spring.extend([condition]*len(sfc_alb_data['lon_avg']))
            leg_contidions_all_spring.extend([condition]*len(sfc_alb_data['lon_all']))
            dates_spring.extend([int(date_s)]*len(sfc_alb_data['lon_avg']))
            dates_spring_all.extend([int(date_s)]*len(sfc_alb_data['lon_all']))
            
        else:
            if init_summer:
                alb_iter0_summer = sfc_alb_data['alb_iter0']
                alb_iter1_summer = sfc_alb_data['alb_iter1']
                alb_iter2_summer = sfc_alb_data['alb_iter2']
                broadband_alb_iter0_summer = sfc_alb_data['broadband_alb_iter0']
                broadband_alb_iter1_summer = sfc_alb_data['broadband_alb_iter1']
                broadband_alb_iter2_summer = sfc_alb_data['broadband_alb_iter2']
                broadband_alb_iter0_filter_summer = sfc_alb_data['broadband_alb_iter0']
                broadband_alb_iter1_filter_summer = sfc_alb_data['broadband_alb_iter1']
                broadband_alb_iter2_filter_summer = sfc_alb_data['broadband_alb_iter2']
                
                lon_avg_summer = sfc_alb_data['lon_avg']
                lat_avg_summer = sfc_alb_data['lat_avg']
                alt_avg_summer = sfc_alb_data['alt_avg']
                lon_all_summer = sfc_alb_data['lon_all']
                lat_all_summer = sfc_alb_data['lat_all']
                alt_all_summer = sfc_alb_data['alt_all']
                fdn_all_summer = sfc_alb_data['fdn_all']
                fup_all_summer = sfc_alb_data['fup_all']
                toa_expand_all_summer = sfc_alb_data['toa_expand_all']
                fdn_up_ratio_all_summer = sfc_alb_data['fdn_up_ratio_all']
                correction_factor_all_summer = sfc_alb_data['correction_factor_all']
                icing_all_summer = sfc_alb_data['icing_all']
                icing_pre_all_summer = sfc_alb_data['icing_pre_all']
                alb_iter1_all_summer = sfc_alb_data['alb_iter1_all']
                alb_iter2_all_summer = sfc_alb_data['alb_iter2_all']
                broadband_alb_iter1_all_summer = sfc_alb_data['broadband_alb_iter1_all']
                broadband_alb_iter2_all_summer = sfc_alb_data['broadband_alb_iter2_all']
                broadband_alb_iter1_all_filter_summer = sfc_alb_data['broadband_alb_iter1_all_filter']
                broadband_alb_iter2_all_filter_summer = sfc_alb_data['broadband_alb_iter2_all_filter']
                
                init_summer = False
            else:
                # concatenate all
                alb_iter0_summer = np.concatenate((alb_iter0_summer, sfc_alb_data['alb_iter0']))
                alb_iter1_summer = np.concatenate((alb_iter1_summer, sfc_alb_data['alb_iter1']))
                alb_iter2_summer = np.concatenate((alb_iter2_summer, sfc_alb_data['alb_iter2']))
                broadband_alb_iter0_summer = np.concatenate((broadband_alb_iter0_summer, sfc_alb_data['broadband_alb_iter0']))
                broadband_alb_iter1_summer = np.concatenate((broadband_alb_iter1_summer, sfc_alb_data['broadband_alb_iter1']))
                broadband_alb_iter2_summer = np.concatenate((broadband_alb_iter2_summer, sfc_alb_data['broadband_alb_iter2']))
                broadband_alb_iter0_filter_summer = np.concatenate((broadband_alb_iter0_filter_summer, sfc_alb_data['broadband_alb_iter0']))
                broadband_alb_iter1_filter_summer = np.concatenate((broadband_alb_iter1_filter_summer, sfc_alb_data['broadband_alb_iter1']))
                broadband_alb_iter2_filter_summer = np.concatenate((broadband_alb_iter2_filter_summer, sfc_alb_data['broadband_alb_iter2']))
                lon_avg_summer = np.concatenate((lon_avg_summer, sfc_alb_data['lon_avg']))
                lat_avg_summer = np.concatenate((lat_avg_summer, sfc_alb_data['lat_avg']))
                alt_avg_summer = np.concatenate((alt_avg_summer, sfc_alb_data['alt_avg']))
                lon_all_summer = np.concatenate((lon_all_summer, sfc_alb_data['lon_all']))
                lat_all_summer = np.concatenate((lat_all_summer, sfc_alb_data['lat_all']))
                alt_all_summer = np.concatenate((alt_all_summer, sfc_alb_data['alt_all']))
                fdn_all_summer = np.concatenate((fdn_all_summer, sfc_alb_data['fdn_all']))
                fup_all_summer = np.concatenate((fup_all_summer, sfc_alb_data['fup_all']))
                toa_expand_all_summer = np.concatenate((toa_expand_all_summer, sfc_alb_data['toa_expand_all']))
                fdn_up_ratio_all_summer =  np.concatenate((fdn_up_ratio_all_summer, sfc_alb_data['fdn_up_ratio_all']))
                correction_factor_all_summer = np.concatenate((correction_factor_all_summer, sfc_alb_data['correction_factor_all']))
                
                icing_all_summer = np.concatenate((icing_all_summer, sfc_alb_data['icing_all']))
                icing_pre_all_summer = np.concatenate((icing_pre_all_summer, sfc_alb_data['icing_pre_all']))
                alb_iter1_all_summer = np.concatenate((alb_iter1_all_summer, sfc_alb_data['alb_iter1_all']))
                alb_iter2_all_summer = np.concatenate((alb_iter2_all_summer, sfc_alb_data['alb_iter2_all']))
                broadband_alb_iter1_all_summer = np.concatenate((broadband_alb_iter1_all_summer, sfc_alb_data['broadband_alb_iter1_all']))
                broadband_alb_iter2_all_summer = np.concatenate((broadband_alb_iter2_all_summer, sfc_alb_data['broadband_alb_iter2_all']))
                broadband_alb_iter1_all_filter_summer = np.concatenate((broadband_alb_iter1_all_filter_summer, sfc_alb_data['broadband_alb_iter1_all_filter']))
                broadband_alb_iter2_all_filter_summer = np.concatenate((broadband_alb_iter2_all_filter_summer, sfc_alb_data['broadband_alb_iter2_all_filter']))
                
            leg_contidions_summer.extend([condition]*len(sfc_alb_data['lon_avg']))
            leg_contidions_all_summer.extend([condition]*len(sfc_alb_data['lon_all']))
            dates_summer.extend([int(date_s)]*len(sfc_alb_data['lon_avg']))
            dates_summer_all.extend([int(date_s)]*len(sfc_alb_data['lon_all']))
            
        
    print(f"Combined total of {len(lon_avg_spring)} spring flight legs and {len(lon_all_spring)} total points.")
    print("alb_iter1_all_spring shape:", alb_iter1_all_spring.shape)
    print("lon_all_spring shape:", lon_all_spring.shape)
    print("lon_all_avg_spring shape:", lon_avg_spring.shape)
    print("leg_contidions_all_spring length:", len(leg_contidions_all_spring))
    
    print(f"Combined total of {len(lon_avg_summer)} summer flight legs and {len(lon_all_summer)} total points.")
    print("alb_iter1_all_summer shape:", alb_iter1_all_summer.shape)
    print("lon_all_summer shape:", lon_all_summer.shape)
    print("lon_all_avg_summer shape:", lon_avg_summer.shape)
    print("leg_contidions_all_summer length:", len(leg_contidions_all_summer))
    print("dates_summer length:", len(dates_summer))

    fdn_all_spring_nan = np.isnan(fdn_all_spring)
    alb_iter1_all_spring[fdn_all_spring_nan] = np.nan
    alb_iter2_all_spring[fdn_all_spring_nan] = np.nan
    fup_gt_fdn_all_spring = np.nansum(fup_all_spring > fdn_all_spring, axis=1)
    fup_gt_fdn_all_mask_spring = fup_gt_fdn_all_spring/fdn_all_spring.shape[1] > 0.1
    alb_iter1_all_spring[fup_gt_fdn_all_mask_spring] = np.nan
    alb_iter2_all_spring[fup_gt_fdn_all_mask_spring] = np.nan
    
    
    fdn_all_summer_nan = np.isnan(fdn_all_summer)
    alb_iter1_all_summer[fdn_all_summer_nan] = np.nan
    alb_iter2_all_summer[fdn_all_summer_nan] = np.nan
    fup_gt_fdn_all_summer = np.nansum(fup_all_summer > fdn_all_summer, axis=1)
    fup_gt_fdn_all_mask_summer = fup_gt_fdn_all_summer/fdn_all_summer.shape[1] > 0.1
    alb_iter1_all_summer[fup_gt_fdn_all_mask_summer] = np.nan
    alb_iter2_all_summer[fup_gt_fdn_all_mask_summer] = np.nan
    
    
    lon_avg_all = np.concatenate((lon_avg_spring, lon_avg_summer))
    lat_avg_all = np.concatenate((lat_avg_spring, lat_avg_summer))
    alt_avg_all = np.concatenate((alt_avg_spring, alt_avg_summer))
    lon_all = np.concatenate((lon_all_spring, lon_all_summer))
    lat_all = np.concatenate((lat_all_spring, lat_all_summer))
    alt_all = np.concatenate((alt_all_spring, alt_all_summer))
    broadband_alb_iter2_all = np.concatenate((broadband_alb_iter2_all_spring, broadband_alb_iter2_all_summer))
    
    
    
    fig_dir = f'fig/sfc_alb_corr_lonlat'
    os.makedirs(fig_dir, exist_ok=True)
    date_all = []
    date_alb = []
    date_alb_std = []
    date_alb_wvl = []
    for date in sorted((set(dates_spring_all))):
        date_mask = np.array([d == date for d in dates_spring_all])
        alt_mask = alt_all_spring <= 1.6  # only include low altitude legs
        date_mask = date_mask & alt_mask
        
        date_alb_avg = np.nanmean(alb_iter2_all_spring[date_mask], axis=0)
        if np.isnan(date_alb_avg).all():
            print(f"All NaN for date {date}, skipping.")
            continue
        date_all.append(str(date)[4:])
        date_alb_std_ = np.nanstd(alb_iter2_all_spring[date_mask], axis=0)
        date_alb.append(date_alb_avg)
        date_alb_std.append(date_alb_std_)
        date_alb_wvl.append(wvl_spring)
        
    for date in sorted(set(dates_summer_all)):
        date_mask = np.array([d == date for d in dates_summer_all])
        alt_mask = alt_all_summer <= 1.6  # only include low altitude legs
        date_mask = date_mask & alt_mask
        
        date_alb_avg = np.nanmean(alb_iter2_all_summer[date_mask], axis=0)
        if np.isnan(date_alb_avg).all():
            print(f"All NaN for date {date}, skipping.")
            continue
        date_all.append(str(date)[4:])
        date_alb_std_ = np.nanstd(alb_iter2_all_summer[date_mask], axis=0)
        date_alb.append(date_alb_avg)
        date_alb_std.append(date_alb_std_)
        date_alb_wvl.append(wvl_summer)
    
    print("date_all length:", len(date_all))
    print("date_alb length:", len(date_alb))
    print("date_alb_wvl length:", len(date_alb_wvl))
    
    plt.close('all')
    # colormap normalized to number of dates
    # colormap normalized to number of dates
    n_dates = len(date_all)
    if n_dates == 0:
        color_series = []
    else:
        cmap_name = 'jet'
        cmap = plt.cm.get_cmap(cmap_name)
        if n_dates == 1:
            color_series = [cmap(0.5)]
        else:
            color_series = [cmap(i / (n_dates - 1)) for i in range(n_dates)]

    # optional ScalarMappable if you want to add a colorbar later
    norm = mpl.colors.Normalize(vmin=0, vmax=max(1, n_dates - 1))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(date_all)):
        ax.plot(date_alb_wvl[i], date_alb[i], label=f'{date_all[i]}', color=color_series[i])
        ax.fill_between(date_alb_wvl[i], date_alb[i]-date_alb_std[i], date_alb[i]+date_alb_std[i], color=color_series[i], alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit) for All Flights', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_all_flights.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(date_all)):
        if date_all[i] not in ['0808', '0809']:
            ax.plot(date_alb_wvl[i], date_alb[i], label=f'{date_all[i]}', color=color_series[i])
            ax.fill_between(date_alb_wvl[i], date_alb[i]-date_alb_std[i], date_alb[i]+date_alb_std[i], color=color_series[i], alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit)\nexclude 0808, 0809', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_ albedo_all_flights_partial.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    
    sys.exit()
    
    
    for date in sorted((set(dates_spring_all))):
        date_mask = np.array([d == date for d in dates_spring_all])
        # set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
        plt.close('all')
        central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
        proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        fig = plt.figure(figsize=(12, 8))
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
        color_vals = np.array(broadband_alb_iter2_all_spring[date_mask]) if 'broadband_alb_iter2_spring' in locals() else None
        if color_vals is None or np.all(np.isnan(color_vals)):
            sc = ax.scatter(lon_all_spring[date_mask], lat_all_spring[date_mask], s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
        else:
            sc = ax.scatter(lon_all_spring[date_mask], lat_all_spring[date_mask], s=5, c=color_vals, cmap='jet',
                            transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
            cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

        # also plot all sampled points along legs (optional, lighter marker)
        if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
            ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

        ax.set_title(f'Polar projection (North) - Spring Flight {date}', fontsize=12)
        # set a reasonable display extent around the data if available (lon/lat box)
        if len(lon_avg_all) > 0 and len(lat_avg_all) > 0:
            lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
            lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
            # expand a bit
            pad_lon = max(0.5, (lon_max - lon_min) * 0.05)
            pad_lat = max(0.5, (lat_max - lat_min) * 0.05)
            try:
                ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
            except Exception:
                # fallback: don't set extent if projection complains
                pass
        ax.tick_params('both', labelsize=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_spring_30s_{str(date)}.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    for date in sorted(set(dates_summer_all)):
        date_mask = np.array([d == date for d in dates_summer_all])
        # set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
        plt.close('all')
        central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
        proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        fig = plt.figure(figsize=(12, 8))
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
        color_vals = np.array(broadband_alb_iter2_all_summer[date_mask]) if 'broadband_alb_iter2_summer' in locals() else None
        if color_vals is None or np.all(np.isnan(color_vals)):
            sc = ax.scatter(lon_all_summer[date_mask], lat_all_summer[date_mask], s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
        else:
            sc = ax.scatter(lon_all_summer[date_mask], lat_all_summer[date_mask], s=5, c=color_vals, cmap='jet',
                            transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
            cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

        # also plot all sampled points along legs (optional, lighter marker)
        if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
            ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

        ax.set_title(f'Polar projection (North) - Summer Flight {date}', fontsize=12)
        # set a reasonable display extent around the data if available (lon/lat box)
        if len(lon_avg_all) > 0 and len(lat_avg_all) > 0:
            lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
            lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
            # expand a bit
            pad_lon = max(0.5, (lon_max - lon_min) * 0.05)
            pad_lat = max(0.5, (lat_max - lat_min) * 0.05)
            try:
                ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
            except Exception:
                # fallback: don't set extent if projection complains
                pass
        ax.tick_params('both', labelsize=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_summer_30s_{str(date)}.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    # set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
    plt.close('all')
    central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    fig = plt.figure(figsize=(12, 8))
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
    color_vals = np.array(broadband_alb_iter2_spring) if 'broadband_alb_iter2_spring' in locals() else None
    if color_vals is None or np.all(np.isnan(color_vals)):
        sc = ax.scatter(lon_avg_spring, lat_avg_spring, s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
    else:
        sc = ax.scatter(lon_avg_spring, lat_avg_spring, s=5, c=color_vals, cmap='jet',
                        transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
        cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

    # also plot all sampled points along legs (optional, lighter marker)
    if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
        ax.scatter(lon_avg_all, lat_avg_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title(f'Polar projection (North) - Spring Flight legs', fontsize=12)
    # set a reasonable display extent around the data if available (lon/lat box)
    if len(lon_avg_all) > 0 and len(lat_avg_all) > 0:
        lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
        lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
        # expand a bit
        pad_lon = max(0.5, (lon_max - lon_min) * 0.05)
        pad_lat = max(0.5, (lat_max - lat_min) * 0.05)
        try:
            ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        except Exception:
            # fallback: don't set extent if projection complains
            pass
    ax.tick_params('both', labelsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_spring_30s.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


    plt.close('all')
    central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    fig = plt.figure(figsize=(12, 8))
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
    color_vals = np.array(broadband_alb_iter2_summer) if 'broadband_alb_iter2_summer' in locals() else None
    if color_vals is None or np.all(np.isnan(color_vals)):
        sc = ax.scatter(lon_avg_summer, lat_avg_summer, s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
    else:
        sc = ax.scatter(lon_avg_summer, lat_avg_summer, s=5, c=color_vals, cmap='jet',
                        transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
        cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

    # also plot all sampled points along legs (optional, lighter marker)
    if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
        ax.scatter(lon_avg_all, lat_avg_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title(f'Polar projection (North) - Summer Flight legs', fontsize=12)
    # set a reasonable display extent around the data if available (lon/lat box)
    if len(lon_avg_all) > 0 and len(lat_avg_all) > 0:
        lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
        lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
        # expand a bit
        pad_lon = max(0.2, (lon_max - lon_min) * 0.05)
        pad_lat = max(0.2, (lat_max - lat_min) * 0.05)
        try:
            ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        except Exception:
            # fallback: don't set extent if projection complains
            pass
    ax.tick_params('both', labelsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_summer_30s.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    # set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
    plt.close('all')
    central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    fig = plt.figure(figsize=(12, 8))
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
    color_vals = np.array(broadband_alb_iter2_all_spring) if 'broadband_alb_iter2_all_spring' in locals() else None
    if color_vals is None or np.all(np.isnan(color_vals)):
        sc = ax.scatter(lon_all_spring, lat_all_spring, s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
    else:
        sc = ax.scatter(lon_all_spring, lat_all_spring, s=5, c=color_vals, cmap='jet',
                        transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
        cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

    # also plot all sampled points along legs (optional, lighter marker)
    if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
        ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title(f'Polar projection (North) - Spring Flight legs', fontsize=12)
    # set a reasonable display extent around the data if available (lon/lat box)
    if len(lon_all) > 0 and len(lat_all) > 0:
        lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
        lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
        # expand a bit
        pad_lon = max(0.5, (lon_max - lon_min) * 0.05)
        pad_lat = max(0.5, (lat_max - lat_min) * 0.05)
        try:
            ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        except Exception:
            # fallback: don't set extent if projection complains
            pass
    ax.tick_params('both', labelsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_spring.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


    plt.close('all')
    central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    fig = plt.figure(figsize=(12, 8))
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
    color_vals = np.array(broadband_alb_iter2_all_summer) if 'broadband_alb_iter2_all_summer' in locals() else None
    if color_vals is None or np.all(np.isnan(color_vals)):
        sc = ax.scatter(lon_all_summer, lat_all_summer, s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
    else:
        sc = ax.scatter(lon_all_summer, lat_all_summer, s=5, c=color_vals, cmap='jet',
                        transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
        cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

    # also plot all sampled points along legs (optional, lighter marker)
    if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
        ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title(f'Polar projection (North) - Summer Flight legs', fontsize=12)
    # set a reasonable display extent around the data if available (lon/lat box)
    if len(lon_all) > 0 and len(lat_all) > 0:
        lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
        lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
        # expand a bit
        pad_lon = max(0.2, (lon_max - lon_min) * 0.05)
        pad_lat = max(0.2, (lat_max - lat_min) * 0.05)
        try:
            ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        except Exception:
            # fallback: don't set extent if projection complains
            pass
    ax.tick_params('both', labelsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_summer.png', bbox_inches='tight', dpi=150)
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