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
import cartopy.feature as cfeature
import bisect
import pandas as pd
import xarray as xr
from collections import defaultdict
import gc
from pyproj import Transformer, CRS
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



def combined_atm_corr():
    log = logging.getLogger("atm corr combined")

    output_dir = f'{_fdir_general_}/sfc_alb_combined_smooth_450nm'
    # glob all sfc alb files
    sfc_alb_files = sorted(glob.glob(f'{output_dir}/sfc_alb_update_*.pkl'))
    print(f"Found {len(sfc_alb_files)} surface albedo files for combination.")
    
    # read each file and combine data into a larger dictionary
    init_spring = True
    init_summer = True
    leg_contidions_spring = []
    leg_contidions_all_spring = []
    leg_contidions_summer = []
    leg_contidions_all_summer = []
    time_spring_all = []
    time_summer_all = []
    dates_spring = []
    dates_summer = []
    dates_spring_all = []
    dates_summer_all = []
    case_tags_spring = []
    case_tags_summer = []
    case_tags_spring_all = []
    case_tags_summer_all = []
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
        if 'spiral' in case_tag.lower():
            condition = 'spiral'
        elif 'clear' in case_tag.lower():
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
            time_spring_all.extend(sfc_alb_data['time_all'])
            case_tags_spring.extend([case_tag]*len(sfc_alb_data['lon_avg']))
            case_tags_spring_all.extend([case_tag]*len(sfc_alb_data['lon_all']))
            
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
            time_summer_all.extend(sfc_alb_data['time_all'])
            case_tags_summer.extend([case_tag]*len(sfc_alb_data['lon_avg']))
            case_tags_summer_all.extend([case_tag]*len(sfc_alb_data['lon_all']))
            
        
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

    # fdn_all_spring_nan = np.isnan(fdn_all_spring)
    # alb_iter1_all_spring[fdn_all_spring_nan] = np.nan
    # alb_iter2_all_spring[fdn_all_spring_nan] = np.nan
    # fup_gt_fdn_all_spring = np.nansum(fup_all_spring > fdn_all_spring, axis=1)
    # fup_gt_fdn_all_mask_spring = fup_gt_fdn_all_spring/fdn_all_spring.shape[1] > 0.1
    # alb_iter1_all_spring[fup_gt_fdn_all_mask_spring] = np.nan
    # alb_iter2_all_spring[fup_gt_fdn_all_mask_spring] = np.nan
    
    
    # fdn_all_summer_nan = np.isnan(fdn_all_summer)
    # alb_iter1_all_summer[fdn_all_summer_nan] = np.nan
    # alb_iter2_all_summer[fdn_all_summer_nan] = np.nan
    # fup_gt_fdn_all_summer = np.nansum(fup_all_summer > fdn_all_summer, axis=1)
    # fup_gt_fdn_all_mask_summer = fup_gt_fdn_all_summer/fdn_all_summer.shape[1] > 0.1
    # alb_iter1_all_summer[fup_gt_fdn_all_mask_summer] = np.nan
    # alb_iter2_all_summer[fup_gt_fdn_all_mask_summer] = np.nan
    
    broadband_alb_iter2_all_spring = np.trapz(alb_iter2_all_spring*toa_expand_all_spring, wvl_spring, axis=1) / np.trapz(toa_expand_all_spring, wvl_spring, axis=1)
    broadband_alb_iter2_all_summer = np.trapz(alb_iter2_all_summer*toa_expand_all_summer, wvl_summer, axis=1) / np.trapz(toa_expand_all_summer, wvl_summer, axis=1)
        
    
    lon_avg_all = np.concatenate((lon_avg_spring, lon_avg_summer))
    lat_avg_all = np.concatenate((lat_avg_spring, lat_avg_summer))
    alt_avg_all = np.concatenate((alt_avg_spring, alt_avg_summer))
    lon_all = np.concatenate((lon_all_spring, lon_all_summer))
    lat_all = np.concatenate((lat_all_spring, lat_all_summer))
    alt_all = np.concatenate((alt_all_spring, alt_all_summer))
    broadband_alb_iter2_all = np.concatenate((broadband_alb_iter2_all_spring, broadband_alb_iter2_all_summer))
    
    leg_contidions_all_summer = np.array(leg_contidions_all_summer)
    leg_contidions_all_spring = np.array(leg_contidions_all_spring)
    leg_contidions_all = np.concatenate((leg_contidions_all_spring, leg_contidions_all_summer))
    
    time_spring_all = np.array(time_spring_all)
    time_summer_all = np.array(time_summer_all)
    time_all = np.concatenate((time_spring_all, time_summer_all))
    
    dates_spring_all = np.array(dates_spring_all)
    dates_summer_all = np.array(dates_summer_all)
    dates_all = np.concatenate((dates_spring_all, dates_summer_all))
    
    os.makedirs('./fig/ice_age', exist_ok=True)
    # load ice fraction data
    file = f'{_fdir_general_}/ice_frac/ice_frac_all.pkl'
    with open(file, 'rb') as f:
        ice_frac_data = pickle.load(f)
    ice_frac_date = ice_frac_data['date']
    ice_frac_time = ice_frac_data['time']
    ice_frac_values = ice_frac_data['ice_frac']
    
    ice_frac_time_offset = {
        '20240528': 0,
        '20240530': 0,
        '20240531': 0,
        '20240603': -0.50/3600,
        '20240605': -0.80/3600,
        '20240606': -0.75/3600,
        '20240607': 0.35/3600,
        '20240610': 0,
        '20240611': -0.15/3600,
        '20240613': 0.55/3600,
        '20240725': 0.30/3600,
        '20240729': -0.95/3600,
        '20240730': 1.0/3600,
        '20240801': -0.50/3600,
        '20240802': -0.15/3600,
        '20240807': -0.70/3600,
        '20240808': -0.25/3600,
        '20240809': -0.05/3600,
        '20240815': -0.85/3600,
    }
    
    ice_frac_all_spring = np.zeros_like(time_spring_all)
    ice_frac_all_summer = np.zeros_like(time_summer_all)
    ice_frac_all_spring[:] = np.nan
    ice_frac_all_summer[:] = np.nan
    for date_s in sorted(set(dates_spring_all)):
        date_mask = dates_spring_all == date_s
        time_dates = time_spring_all[date_mask]
        t_offset = ice_frac_time_offset.get(str(date_s), 0)
        ice_frac_all_date = ice_frac_values[ice_frac_date == int(date_s)]
        ice_frac_time_date = ice_frac_time[ice_frac_date == int(date_s)]
        ice_frac_time_date += t_offset
        ice_frac_all_spring_tmp = np.zeros_like(time_dates)
        ice_frac_all_spring_tmp[:] = np.nan
        for i, t in enumerate(time_dates):
            # find closest time in ice_frac_time
            time_diff = np.abs(ice_frac_time_date - t)
            if i % 1000 == 0:
                print(f"  time index {i}, t: {t}, time_diff min: {np.min(time_diff)}")
            if np.min(time_diff) <= 1./60/60:  # within 1s 
                closest_index = np.argmin(time_diff)
                ice_frac_all_spring_tmp[i] = ice_frac_all_date[closest_index].copy()
        ice_frac_all_spring[date_mask] = ice_frac_all_spring_tmp.copy()
       

                
    for date_s in sorted(set(dates_summer_all)):
        date_mask = dates_summer_all == date_s
        time_dates = time_summer_all[date_mask]
        t_offset = ice_frac_time_offset.get(str(date_s), 0)
        ice_frac_all_date = ice_frac_values[ice_frac_date == int(date_s)]
        ice_frac_time_date = ice_frac_time[ice_frac_date == int(date_s)]
        ice_frac_time_date += t_offset
        ice_frac_all_summer_tmp = np.zeros_like(time_dates)
        ice_frac_all_summer_tmp[:] = np.nan
        for i, t in enumerate(time_dates):
            # find closest time in ice_frac_time
            time_diff = np.abs(ice_frac_time_date - t)
            if i % 1000 == 0:
                print(f"  time index {i}, t: {t}, time_diff min: {np.min(time_diff)}")
            if np.min(time_diff) <= 1./60/60:  # within 1s 
                closest_index = np.argmin(time_diff)
                ice_frac_all_summer_tmp[i] = ice_frac_all_date[closest_index].copy()
        ice_frac_all_summer[date_mask] = ice_frac_all_summer_tmp.copy()

    myi_age_ratio_spring_all = np.zeros_like(lon_all_spring)
    myi_age_ratio_spring_all[:] = np.nan
    myi_age_ratio_summer_all = np.zeros_like(lon_all_summer)
    myi_age_ratio_summer_all[:] = np.nan
    fyi_age_ratio_spring_all = np.zeros_like(lon_all_spring)
    fyi_age_ratio_spring_all[:] = np.nan
    fyi_age_ratio_summer_all = np.zeros_like(lon_all_summer)
    fyi_age_ratio_summer_all[:] = np.nan
    yi_age_ratio_spring_all = np.zeros_like(lon_all_spring)
    yi_age_ratio_spring_all[:] = np.nan
    yi_age_ratio_summer_all = np.zeros_like(lon_all_summer)
    yi_age_ratio_summer_all[:] = np.nan
    ice_ratio_spring_all = np.zeros_like(lon_all_spring)
    ice_ratio_spring_all[:] = np.nan
    ice_ratio_summer_all = np.zeros_like(lon_all_summer)
    ice_ratio_summer_all[:] = np.nan
    ow_ratio_spring_all = np.zeros_like(lon_all_spring)
    ow_ratio_spring_all[:] = np.nan
    ow_ratio_summer_all = np.zeros_like(lon_all_summer)
    ow_ratio_summer_all[:] = np.nan
    ice_age_spring_all = np.zeros_like(lon_all_spring)
    ice_age_spring_all[:] = np.nan
    ice_age_summer_all = np.zeros_like(lon_all_summer)
    ice_age_summer_all[:] = np.nan
    
    # iceage_nh_12.5km_20240101_20250923_ql.nc
    with Dataset(f'{_fdir_general_}/ice_age/iceage_nh_12.5km_20240101_20250923_ql.nc', 'r') as nc:
        nsidc_lon = nc.variables['longitude'][:]
        nsidc_lat = nc.variables['latitude'][:]
        time_nc = nc.variables['time'][:] # days since 1970-01-01
        nsidc_ice_age = nc.variables['age_of_sea_ice'][:]  # time, lat, lon
    time_nc_dates = np.array([datetime.datetime(1970,1,1) + datetime.timedelta(days=t) for t in time_nc])
    time_nc_dates_str = np.array([t.strftime('%Y%m%d') for t in time_nc_dates])
    nsidc_ice_age = np.array(nsidc_ice_age, dtype=np.float32)
    
    
    for date_s in sorted(set(dates_spring_all)):
        print(f"Processing ice age data for date: {date_s}")
        print(f"Ice age data file: {_fdir_general_}/ice_age/ECICE-IcetypesUncorrected-{date_s}.nc")
        # ECICE-IcetypesUncorrected-20240528.nc
        with Dataset(f'{_fdir_general_}/ice_age/ECICE-IcetypesUncorrected-{date_s}.nc', 'r') as nc:
            lon = nc.variables['LON'][:]
            lat = nc.variables['LAT'][:]
            myi_age_ratio_nc = nc.variables['MYI'][:]
            fyi_age_ratio_nc = nc.variables['FYI'][:]
            yi_age_ratio_nc = nc.variables['YI'][:]
            ice_ratio_nc = nc.variables['TOTAL_ICE'][:]
            open_water_nc = nc.variables['OW'][:]
            brt19h_nc = nc.variables['BRT19H'][:]
            brt37h_nc = nc.variables['BRT37H'][:]
            brt37v_nc = nc.variables['BRT37V'][:]
        
        lonlat_shape = lon.shape
        lon = lon.flatten()
        lat = lat.flatten()
        myi_age_ratio_nc = myi_age_ratio_nc.flatten()
        fyi_age_ratio_nc = fyi_age_ratio_nc.flatten()
        yi_age_ratio_nc = yi_age_ratio_nc.flatten()
        ice_ratio_nc = ice_ratio_nc.flatten()
        open_water_nc = open_water_nc.flatten()
            
        date_mask = dates_spring_all == date_s
        if date_mask.sum() > 0:
        
            myi_age_ratio_mesh = griddata(
                        (lon, lat), myi_age_ratio_nc,
                        (lon_all_spring[date_mask], lat_all_spring[date_mask]),
                        method='nearest'
                    )
            myi_age_ratio_spring_all[date_mask] = myi_age_ratio_mesh.copy()
            fyi_age_ratio_mesh = griddata(
                        (lon, lat), fyi_age_ratio_nc,
                        (lon_all_spring[date_mask], lat_all_spring[date_mask]),
                        method='nearest'
                    )
            fyi_age_ratio_spring_all[date_mask] = fyi_age_ratio_mesh.copy()
            yi_age_ratio_mesh = griddata(
                        (lon, lat), yi_age_ratio_nc,
                        (lon_all_spring[date_mask], lat_all_spring[date_mask]),
                        method='nearest'
                    )
            yi_age_ratio_spring_all[date_mask] = yi_age_ratio_mesh.copy()
            ice_ratio_mesh = griddata(
                        (lon, lat), ice_ratio_nc,
                        (lon_all_spring[date_mask], lat_all_spring[date_mask]),
                        method='nearest'
                    )
            ice_ratio_spring_all[date_mask] = ice_ratio_mesh.copy()
            ow_ratio_mesh = griddata(
                        (lon, lat), open_water_nc,
                        (lon_all_spring[date_mask], lat_all_spring[date_mask]),
                        method='nearest'
                    )
            ow_ratio_spring_all[date_mask] = ow_ratio_mesh.copy()
            
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), myi_age_ratio_nc.reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Multi-year Ice Conc (%)')
            fig.savefig(f'./fig/ice_age/myi_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), brt19h_nc, transform=ccrs.PlateCarree(), cmap='coolwarm',)# vmin=255, vmax=285)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='BRT19H (K)')
            fig.savefig(f'./fig/ice_age/brt19h_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), brt37h_nc, transform=ccrs.PlateCarree(), cmap='coolwarm',)# vmin=255, vmax=285)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='BRT37H (K)')
            fig.savefig(f'./fig/ice_age/brt37h_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), brt37v_nc, transform=ccrs.PlateCarree(), cmap='coolwarm',)# vmin=255, vmax=285)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='BRT37V (K)')
            fig.savefig(f'./fig/ice_age/brt37v_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), (myi_age_ratio_nc+fyi_age_ratio_nc).reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Multi-year + First-year (%)')
            fig.savefig(f'./fig/ice_age/myi_fyi_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), (myi_age_ratio_nc+fyi_age_ratio_nc+yi_age_ratio_nc).reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Multi-year + First-year + Young Ice Conc (%)')
            fig.savefig(f'./fig/ice_age/myi_fyi_yi_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), ice_ratio_nc.reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Total Ice Conc (%)')
            fig.savefig(f'./fig/ice_age/total_ice_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), open_water_nc.reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Open Water Conc (%)')
            fig.savefig(f'./fig/ice_age/ow_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), (ice_ratio_nc+open_water_nc).reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Total ice + Open water Conc (%)')
            fig.savefig(f'./fig/ice_age/total_ice_ow_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            # myi_fyi_yi_total = myi_age_ratio_nc + fyi_age_ratio_nc + yi_age_ratio_nc
            myi_fyi_yi_total = ice_ratio_nc
            myi_fyi_yi_total_flight = myi_age_ratio_spring_all[date_mask] + fyi_age_ratio_spring_all[date_mask] + yi_age_ratio_spring_all[date_mask]
            myi_to_tatal_ratio = myi_age_ratio_nc / (myi_fyi_yi_total+1e-7) * 100
            myi_to_tatal_ratio_flight = myi_age_ratio_spring_all[date_mask] / (myi_fyi_yi_total_flight+1e-7) * 100
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), myi_to_tatal_ratio.reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            c2 = ax.scatter(lon_all_spring[date_mask], lat_all_spring[date_mask], c=myi_to_tatal_ratio_flight, transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100, edgecolors='k')
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Multi-year Ice Percentage (%)')
            fig.savefig(f'./fig/ice_age/myi_percentage_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            
            # for i in range(len(lon_all_spring[date_mask])):
                # # find closest lon/lat
                # lon_diff = np.abs(lon - lon_all_spring[date_mask][i])
                # lat_diff = np.abs(lat - lat_all_spring[date_mask][i])
                # # convert diff to km
                # lon_diff_km = lon_diff * 111.32 * np.cos(np.deg2rad(lat_all_spring[i]))
                # lat_diff_km = lat_diff * 110.57
                # dist_km = np.sqrt(lon_diff_km**2 + lat_diff_km**2)
                # # find closest point
                # closest_index = np.argmin(dist_km)
                # myi_age_ratio_spring_all[date_mask][i] = myi_age_ratio_nc[closest_index].copy()
                # fyi_age_ratio_spring_all[date_mask][i] = fyi_age_ratio_nc[closest_index].copy()
                # yi_age_ratio_spring_all[date_mask][i] = yi_age_ratio_nc[closest_index].copy()
                # ice_ratio_spring_all[date_mask][i] = ice_ratio_nc[closest_index].copy()
                # ow_ratio_spring_all[date_mask][i] = open_water_nc[closest_index].copy()
                
        date_s_dt = datetime.datetime.strptime(str(date_s), '%Y%m%d')
        # find closest date in time_nc_dates
        time_diff = np.abs(np.array([(t - date_s_dt).days for t in time_nc_dates]))
        closest_index = np.argmin(time_diff)
        print(f"  Closest date in ice age data: {time_nc_dates[closest_index]}, index: {closest_index}, time diff: {time_diff[closest_index]} days")
        
        nsidc_lon_f = nsidc_lon.flatten()
        nsidc_lat_f = nsidc_lat.flatten()
        ice_age_nc = nsidc_ice_age[closest_index, :, :].flatten()
        ice_age_nc[ice_age_nc == 20] = np.nan  # land
        ice_age_nc[ice_age_nc == 21] = np.nan  # near coast
        
        ice_age_mesh = griddata(
                    (nsidc_lon_f, nsidc_lat_f), ice_age_nc,
                    (lon_all_spring[date_mask], lat_all_spring[date_mask]),
                    method='nearest'
                )
        ice_age_spring_all[date_mask] = ice_age_mesh.copy()
        ice_age_spring_all[np.isnan(ice_age_spring_all)] = 0 # set ice age to 0 if nan
        
        plt.close('all')
        central_lon = np.mean(lon_all)
        lon_min = np.min(lon_all) - 2
        lon_max = np.max(lon_all) + 2
        lat_min = np.min(lat_all) - 2
        lat_max = np.max(lat_all) + 2
        fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
        # myi_fyi_yi_total = myi_age_ratio_nc + fyi_age_ratio_nc + yi_age_ratio_nc
        c1 = ax.pcolormesh(nsidc_lon, nsidc_lat, ice_age_nc.reshape(nsidc_lon.shape), transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=5)
        c2 = ax.scatter(lon_all_spring[date_mask], lat_all_spring[date_mask], c=ice_age_spring_all[date_mask], transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=5, edgecolors='k')
        ax.coastlines()
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        cb = fig.colorbar(c1, ax=ax, label='Ice Age (years)')
        fig.savefig(f'./fig/ice_age/ice_age_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
            
    for date_s in sorted(set(dates_summer_all)):
        print(f"Processing ice age data for date: {date_s}")
        print(f"Ice age data file: {_fdir_general_}/ice_age/ECICE-IcetypesUncorrected-{date_s}.nc")
        # ECICE-IcetypesUncorrected-20240528.nc
        with Dataset(f'{_fdir_general_}/ice_age/ECICE-IcetypesUncorrected-{date_s}.nc', 'r') as nc:
            lon = nc.variables['LON'][:]
            lat = nc.variables['LAT'][:]
            myi_age_ratio_nc = nc.variables['MYI'][:]
            fyi_age_ratio_nc = nc.variables['FYI'][:]
            yi_age_ratio_nc = nc.variables['YI'][:]
            ice_ratio_nc = nc.variables['TOTAL_ICE'][:]
            open_water_nc = nc.variables['OW'][:]
            brt19h_nc = nc.variables['BRT19H'][:]
            brt37h_nc = nc.variables['BRT37H'][:]
            brt37v_nc = nc.variables['BRT37V'][:]
        
        lonlat_shape = lon.shape
        lon = lon.flatten()
        lat = lat.flatten()
        myi_age_ratio_nc = myi_age_ratio_nc.flatten()
        fyi_age_ratio_nc = fyi_age_ratio_nc.flatten()
        yi_age_ratio_nc = yi_age_ratio_nc.flatten()
        ice_ratio_nc = ice_ratio_nc.flatten()
        open_water_nc = open_water_nc.flatten()
        
        date_mask = dates_summer_all == date_s
        if date_mask.sum() > 0:
            
            myi_age_ratio_mesh = griddata(
                        (lon, lat), myi_age_ratio_nc,
                        (lon_all_summer[date_mask], lat_all_summer[date_mask]),
                        method='nearest'
                    )
            myi_age_ratio_summer_all[date_mask] = myi_age_ratio_mesh.copy()
            fyi_age_ratio_mesh = griddata(
                        (lon, lat), fyi_age_ratio_nc,
                        (lon_all_summer[date_mask], lat_all_summer[date_mask]),
                        method='nearest'
                    )
            fyi_age_ratio_summer_all[date_mask] = fyi_age_ratio_mesh.copy()
            yi_age_ratio_mesh = griddata(
                        (lon, lat), yi_age_ratio_nc,
                        (lon_all_summer[date_mask], lat_all_summer[date_mask]),
                        method='nearest'
                    )
            yi_age_ratio_summer_all[date_mask] = yi_age_ratio_mesh.copy()
            ice_ratio_mesh = griddata(
                        (lon, lat), ice_ratio_nc,
                        (lon_all_summer[date_mask], lat_all_summer[date_mask]),
                        method='nearest'
                    )
            ice_ratio_summer_all[date_mask] = ice_ratio_mesh.copy()
            ow_ratio_mesh = griddata(
                        (lon, lat), open_water_nc,
                        (lon_all_summer[date_mask], lat_all_summer[date_mask]),
                        method='nearest'
                    )
            ow_ratio_summer_all[date_mask] = ow_ratio_mesh.copy()
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), myi_age_ratio_nc.reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Multi-year Ice Conc (%)')
            fig.savefig(f'./fig/ice_age/myi_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), brt19h_nc, transform=ccrs.PlateCarree(), cmap='coolwarm',)# vmin=255, vmax=285)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='BRT19H (K)')
            fig.savefig(f'./fig/ice_age/brt19h_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), brt37h_nc, transform=ccrs.PlateCarree(), cmap='coolwarm',)# vmin=255, vmax=285)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='BRT37H (K)')
            fig.savefig(f'./fig/ice_age/brt37h_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), brt37v_nc, transform=ccrs.PlateCarree(), cmap='coolwarm',)# vmin=255, vmax=285)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='BRT37V (K)')
            fig.savefig(f'./fig/ice_age/brt37v_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), (myi_age_ratio_nc+fyi_age_ratio_nc).reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r',)# vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Multi-year + First-year (%)')
            fig.savefig(f'./fig/ice_age/myi_fyi_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), (myi_age_ratio_nc+fyi_age_ratio_nc+yi_age_ratio_nc).reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r',)# vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Multi-year + First-year + Young Ice Conc (%)')
            fig.savefig(f'./fig/ice_age/myi_fyi_yi_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), ice_ratio_nc.reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Total Ice Conc (%)')
            fig.savefig(f'./fig/ice_age/total_ice_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), open_water_nc.reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Open Water Conc (%)')
            fig.savefig(f'./fig/ice_age/ow_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), (ice_ratio_nc+open_water_nc).reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Total ice + Open water Conc (%)')
            fig.savefig(f'./fig/ice_age/total_ice_ow_conc_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plt.close('all')
            central_lon = np.mean(lon_all)
            lon_min = np.min(lon_all) - 2
            lon_max = np.max(lon_all) + 2
            lat_min = np.min(lat_all) - 2
            lat_max = np.max(lat_all) + 2
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
            # myi_fyi_yi_total = myi_age_ratio_nc + fyi_age_ratio_nc + yi_age_ratio_nc
            myi_fyi_yi_total = ice_ratio_nc
            myi_fyi_yi_total_flight = myi_age_ratio_summer_all[date_mask] + fyi_age_ratio_summer_all[date_mask] + yi_age_ratio_summer_all[date_mask]
            myi_to_tatal_ratio = myi_age_ratio_nc / (myi_fyi_yi_total+1e-7) * 100
            myi_to_tatal_ratio_flight = myi_age_ratio_summer_all[date_mask] / (myi_fyi_yi_total_flight+1e-7) * 100
            c1 = ax.pcolormesh(lon.reshape(lonlat_shape), lat.reshape(lonlat_shape), myi_to_tatal_ratio.reshape(lonlat_shape), transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100)
            c2 = ax.scatter(lon_all_summer[date_mask], lat_all_summer[date_mask], c=myi_to_tatal_ratio_flight, transform=ccrs.PlateCarree(), cmap='Blues_r', vmin=0, vmax=100, edgecolors='k')
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cb = fig.colorbar(c1, ax=ax, label='Multi-year Ice Percentage (%)')
            fig.savefig(f'./fig/ice_age/myi_percentage_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            
            
            
            # for i in range(len(lon_all_summer[date_mask])):
                
            #     # find closest lon/lat
            #     lon_diff = np.abs(lon - lon_all_summer[date_mask][i])
            #     lat_diff = np.abs(lat - lat_all_summer[date_mask][i])
            #     # convert diff to km
            #     lon_diff_km = lon_diff * 111.32 * np.cos(np.deg2rad(lat_all_summer[i]))
            #     lat_diff_km = lat_diff * 110.57
            #     dist_km = np.sqrt(lon_diff_km**2 + lat_diff_km**2)
            #     # find closest point
            #     closest_index = np.argmin(dist_km)
            #     myi_age_ratio_summer_all[date_mask][i] = myi_age_ratio_nc[closest_index].copy()
            #     fyi_age_ratio_summer_all[date_mask][i] = fyi_age_ratio_nc[closest_index].copy()
            #     yi_age_ratio_summer_all[date_mask][i] = yi_age_ratio_nc[closest_index].copy()
            #     ice_ratio_summer_all[date_mask][i] = ice_ratio_nc[closest_index].copy()
            #     ow_ratio_summer_all[date_mask][i] = open_water_nc[closest_index].copy()
        date_s_dt = datetime.datetime.strptime(str(date_s), '%Y%m%d')
        # find closest date in time_nc_dates
        time_diff = np.abs(np.array([(t - date_s_dt).days for t in time_nc_dates]))
        closest_index = np.argmin(time_diff)
        print(f"  Closest date in ice age data: {time_nc_dates[closest_index]}, index: {closest_index}, time diff: {time_diff[closest_index]} days")
        nsidc_lon_f = nsidc_lon.flatten()
        nsidc_lat_f = nsidc_lat.flatten()
        ice_age_nc = nsidc_ice_age[closest_index, :, :].flatten()
        ice_age_nc[ice_age_nc == 20] = np.nan  # land
        ice_age_nc[ice_age_nc == 21] = np.nan  # near coast
        
        ice_age_mesh = griddata(
                    (nsidc_lon_f, nsidc_lat_f), ice_age_nc,
                    (lon_all_summer[date_mask], lat_all_summer[date_mask]),
                    method='nearest'
                )
        ice_age_summer_all[date_mask] = ice_age_mesh.copy()
        ice_age_summer_all[np.isnan(ice_age_summer_all)] = 0 # set ice age to 0 if nan
        
        plt.close('all')
        central_lon = np.mean(lon_all)
        lon_min = np.min(lon_all) - 2
        lon_max = np.max(lon_all) + 2
        lat_min = np.min(lat_all) - 2
        lat_max = np.max(lat_all) + 2
        fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)})
        # myi_fyi_yi_total = myi_age_ratio_nc + fyi_age_ratio_nc + yi_age_ratio_nc
        c1 = ax.pcolormesh(nsidc_lon, nsidc_lat, ice_age_nc.reshape(nsidc_lon.shape), transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=5)
        c2 = ax.scatter(lon_all_summer[date_mask], lat_all_summer[date_mask], c=ice_age_summer_all[date_mask], transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=5, edgecolors='k')
        ax.coastlines()
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        cb = fig.colorbar(c1, ax=ax, label='Ice Age (years)')
        fig.savefig(f'./fig/ice_age/ice_age_{date_s}_collocate.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
                
    output_all_dict = {
        'wvl_spring': wvl_spring,
        'wvl_summer': wvl_summer,
        'time_spring_all': time_spring_all,
        'lon_all_spring': lon_all_spring,
        'lat_all_spring': lat_all_spring,
        'alt_all_spring': alt_all_spring,
        'dates_spring_all': dates_spring_all,
        'leg_contidions_all_spring': leg_contidions_all_spring,
        'case_tags_spring': case_tags_spring,
        'case_tags_spring_all': case_tags_spring_all,
        'fdn_all_spring': fdn_all_spring,
        'fup_all_spring': fup_all_spring,
        'toa_expand_all_spring': toa_expand_all_spring,
        'icing_all_spring': icing_all_spring,
        'icing_pre_all_spring': icing_pre_all_spring,
        'alb_iter1_all_spring': alb_iter1_all_spring,
        'alb_iter2_all_spring': alb_iter2_all_spring,
        'broadband_alb_iter1_all_spring': broadband_alb_iter1_all_spring,
        'broadband_alb_iter2_all_spring': broadband_alb_iter2_all_spring,
        'broadband_alb_iter1_all_filter_spring': broadband_alb_iter1_all_filter_spring,
        'broadband_alb_iter2_all_filter_spring': broadband_alb_iter2_all_filter_spring,
        'ice_frac_all_spring': ice_frac_all_spring,
        'myi_ratio_spring_all': myi_age_ratio_spring_all,
        'fyi_ratio_spring_all': fyi_age_ratio_spring_all,
        'yi_ratio_spring_all': yi_age_ratio_spring_all,
        'ice_ratio_spring_all': ice_ratio_spring_all,
        'ow_ratio_spring_all': ow_ratio_spring_all,
        'ice_age_spring_all': ice_age_spring_all,
        
        
        
        'time_summer_all': time_summer_all,
        'lon_all_summer': lon_all_summer,
        'lat_all_summer': lat_all_summer,
        'alt_all_summer': alt_all_summer,
        'dates_summer_all': dates_summer_all,
        'leg_contidions_all_summer': leg_contidions_all_summer,
        'case_tags_summer': case_tags_summer,
        'case_tags_summer_all': case_tags_summer_all,
        'fdn_all_summer': fdn_all_summer,
        'fup_all_summer': fup_all_summer,
        'toa_expand_all_summer': toa_expand_all_summer,
        'icing_all_summer': icing_all_summer,
        'icing_pre_all_summer': icing_pre_all_summer,
        'alb_iter1_all_summer': alb_iter1_all_summer,
        'alb_iter2_all_summer': alb_iter2_all_summer,
        'broadband_alb_iter1_all_summer': broadband_alb_iter1_all_summer,
        'broadband_alb_iter2_all_summer': broadband_alb_iter2_all_summer,
        'broadband_alb_iter1_all_filter_summer': broadband_alb_iter1_all_filter_summer,
        'broadband_alb_iter2_all_filter_summer': broadband_alb_iter2_all_filter_summer,
        'ice_frac_all_summer': ice_frac_all_summer,
        'myi_ratio_summer_all': myi_age_ratio_summer_all,
        'fyi_ratio_summer_all': fyi_age_ratio_summer_all,
        'yi_ratio_summer_all': yi_age_ratio_summer_all,
        'ice_ratio_summer_all': ice_ratio_summer_all,
        'ow_ratio_summer_all': ow_ratio_summer_all,
        'ice_age_summer_all': ice_age_summer_all,
    }
    
    combined_output_file = f'{output_dir}/sfc_alb_combined_spring_summer.pkl'
    with open(combined_output_file, 'wb') as f:
        pickle.dump(output_all_dict, f)
    print(f"Combined surface albedo data saved to {combined_output_file}")

    
    fig_dir = f'fig/sfc_alb_corr_lonlat'
    os.makedirs(fig_dir, exist_ok=True)
    date_all = []
    date_alb = []
    date_alb_std = []
    date_broadband_alb = []
    date_broadband_alb_std = []
    date_ice_frac = []
    date_ice_frac_std = []
    date_myi_ratio = []
    date_myi_ratio_std = []
    date_ice_age_avg = []
    date_ice_age_std = []
    date_ice_age_myi_ratio = []
    date_clear_all = []
    date_alb_clear = []
    date_alb_clear_std = []
    date_ice_frac_clear = []
    date_ice_frac_clear_std = []
    date_myi_ratio_clear = []
    date_cloudy_all = []
    date_alb_cloudy = []
    date_alb_cloudy_std = []
    date_ice_frac_cloudy = []
    date_ice_frac_cloudy_std = []
    date_myi_ratio_cloudy = []
    date_alb_wvl = []
    date_alb_clear_wvl = []
    date_alb_cloudy_wvl = []
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
        
        date_broadband_alb_avg = np.nanmean(broadband_alb_iter2_all_spring[date_mask])
        date_broadband_alb_std_ = np.nanstd(broadband_alb_iter2_all_spring[date_mask])
        date_broadband_alb.append(date_broadband_alb_avg)
        date_broadband_alb_std.append(date_broadband_alb_std_)
        
        date_ice_frac_avg = np.nanmean(ice_frac_all_spring[date_mask])
        date_ice_frac_std_ = np.nanstd(ice_frac_all_spring[date_mask])
        date_ice_frac.append(date_ice_frac_avg)
        date_ice_frac_std.append(date_ice_frac_std_)
        
        date_myi_ratio_avg = np.nanmean(myi_age_ratio_spring_all[date_mask]/ice_ratio_spring_all[date_mask])
        date_myi_ratio.append(date_myi_ratio_avg)
        date_myi_ratio_std_ = np.nanstd(myi_age_ratio_spring_all[date_mask]/ice_ratio_spring_all[date_mask])
        date_myi_ratio_std.append(date_myi_ratio_std_)
        
        date_ice_age_avg_ = np.nanmean(ice_age_spring_all[date_mask])
        date_ice_age_std_ = np.nanstd(ice_age_spring_all[date_mask])
        date_ice_age_avg.append(date_ice_age_avg_)
        date_ice_age_std.append(date_ice_age_std_)
        date_ice_age_myi_raio_ = (ice_age_spring_all[date_mask] >= 2).sum()/len(ice_age_spring_all[date_mask])
        date_ice_age_myi_ratio.append(date_ice_age_myi_raio_)
        
        clear_mask = date_mask & (leg_contidions_all_spring == 'clear')
        if np.any(clear_mask):
            date_clear_all.append(str(date)[4:])
            date_alb_avg_clear = np.nanmean(alb_iter2_all_spring[clear_mask], axis=0)
            date_alb_clear.append(date_alb_avg_clear)
            date_alb_clear_std_ = np.nanstd(alb_iter2_all_spring[clear_mask], axis=0)
            date_alb_clear_std.append(date_alb_clear_std_)
            date_alb_clear_wvl.append(wvl_spring)
            
            date_ice_frac_avg_clear = np.nanmean(ice_frac_all_spring[clear_mask])
            date_ice_frac_std_clear_ = np.nanstd(ice_frac_all_spring[clear_mask])
            date_ice_frac_clear.append(date_ice_frac_avg_clear)
            date_ice_frac_clear_std.append(date_ice_frac_std_clear_)
            
            date_myi_ratio_avg_clear = np.nanmean(myi_age_ratio_spring_all[clear_mask]/ice_ratio_spring_all[clear_mask])
            date_myi_ratio_clear.append(date_myi_ratio_avg_clear)
        cloudy_mask = date_mask & (leg_contidions_all_spring == 'cloudy')
        if np.any(cloudy_mask):
            date_cloudy_all.append(str(date)[4:])
            date_alb_avg_cloudy = np.nanmean(alb_iter2_all_spring[cloudy_mask], axis=0)
            date_alb_cloudy.append(date_alb_avg_cloudy)
            date_alb_cloudy_std_ = np.nanstd(alb_iter2_all_spring[cloudy_mask], axis=0)
            date_alb_cloudy_std.append(date_alb_cloudy_std_)
            date_alb_cloudy_wvl.append(wvl_spring)
            
            date_ice_frac_avg_cloudy = np.nanmean(ice_frac_all_spring[cloudy_mask])
            date_ice_frac_std_cloudy_ = np.nanstd(ice_frac_all_spring[cloudy_mask])
            date_ice_frac_cloudy.append(date_ice_frac_avg_cloudy)
            date_ice_frac_cloudy_std.append(date_ice_frac_std_cloudy_)
            
            date_myi_ratio_avg_cloudy = np.nanmean(myi_age_ratio_spring_all[cloudy_mask]/ice_ratio_spring_all[cloudy_mask])
            date_myi_ratio_cloudy.append(date_myi_ratio_avg_cloudy)
        
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
        
        date_broadband_alb_avg = np.nanmean(broadband_alb_iter2_all_summer[date_mask])
        date_broadband_alb_std_ = np.nanstd(broadband_alb_iter2_all_summer[date_mask])
        date_broadband_alb.append(date_broadband_alb_avg)
        date_broadband_alb_std.append(date_broadband_alb_std_)
        
        date_ice_frac_avg = np.nanmean(ice_frac_all_summer[date_mask])
        date_ice_frac_std_ = np.nanstd(ice_frac_all_summer[date_mask])
        date_ice_frac.append(date_ice_frac_avg)
        date_ice_frac_std.append(date_ice_frac_std_)
        
        date_myi_ratio_avg = np.nanmean(myi_age_ratio_summer_all[date_mask]/ice_ratio_summer_all[date_mask])
        date_myi_ratio.append(date_myi_ratio_avg)
        date_myi_ratio_std_ = np.nanstd(myi_age_ratio_summer_all[date_mask]/ice_ratio_summer_all[date_mask])
        date_myi_ratio_std.append(date_myi_ratio_std_)
        
        date_ice_age_avg_ = np.nanmean(ice_age_summer_all[date_mask])
        date_ice_age_std_ = np.nanstd(ice_age_summer_all[date_mask])
        date_ice_age_avg.append(date_ice_age_avg_)
        date_ice_age_std.append(date_ice_age_std_)
        date_ice_age_myi_raio_ = (ice_age_summer_all[date_mask] >= 2).sum()/len(ice_age_summer_all[date_mask])
        date_ice_age_myi_ratio.append(date_ice_age_myi_raio_)
        
        clear_mask = date_mask & (leg_contidions_all_summer == 'clear')
        if np.any(clear_mask):
            date_clear_all.append(str(date)[4:])
            date_alb_avg_clear = np.nanmean(alb_iter2_all_summer[clear_mask], axis=0)
            date_alb_clear.append(date_alb_avg_clear)
            date_alb_clear_std_ = np.nanstd(alb_iter2_all_summer[clear_mask], axis=0)
            date_alb_clear_std.append(date_alb_clear_std_)
            date_alb_clear_wvl.append(wvl_summer)
            
            date_ice_frac_avg_clear = np.nanmean(ice_frac_all_summer[clear_mask])
            date_ice_frac_std_clear_ = np.nanstd(ice_frac_all_summer[clear_mask])
            date_ice_frac_clear.append(date_ice_frac_avg_clear)
            date_ice_frac_clear_std.append(date_ice_frac_std_clear_)
            
            date_myi_ratio_avg_clear = np.nanmean(myi_age_ratio_summer_all[clear_mask]/ice_ratio_summer_all[clear_mask])
            date_myi_ratio_clear.append(date_myi_ratio_avg_clear)
            
        cloudy_mask = date_mask & (leg_contidions_all_summer == 'cloudy')
        if np.any(cloudy_mask):
            date_cloudy_all.append(str(date)[4:])
            date_alb_avg_cloudy = np.nanmean(alb_iter2_all_summer[cloudy_mask], axis=0)
            date_alb_cloudy.append(date_alb_avg_cloudy)
            date_alb_cloudy_std_ = np.nanstd(alb_iter2_all_summer[cloudy_mask], axis=0)
            date_alb_cloudy_std.append(date_alb_cloudy_std_)
            date_alb_cloudy_wvl.append(wvl_summer)
            
            date_ice_frac_avg_cloudy = np.nanmean(ice_frac_all_summer[cloudy_mask])
            date_ice_frac_std_cloudy_ = np.nanstd(ice_frac_all_summer[cloudy_mask])
            date_ice_frac_cloudy.append(date_ice_frac_avg_cloudy)
            date_ice_frac_cloudy_std.append(date_ice_frac_std_cloudy_)
            
            date_myi_ratio_avg_cloudy = np.nanmean(myi_age_ratio_summer_all[cloudy_mask]/ice_ratio_summer_all[cloudy_mask])
            date_myi_ratio_cloudy.append(date_myi_ratio_avg_cloudy)
    
    print("date_all length:", len(date_all))
    print("date_alb length:", len(date_alb))
    print("date_alb_wvl length:", len(date_alb_wvl))
    
    plt.close('all')
    # colormap normalized to number of dates
    # colormap normalized to number of dates
    n_dates = len(date_all)
    n_dates_clear = len(date_clear_all)
    n_dates_cloudy = len(date_cloudy_all)
    if n_dates == 0:
        color_series = []
        color_series_clear = []
        color_series_cloudy = []
    else:
        cmap_name = 'jet'
        cmap = plt.cm.get_cmap(cmap_name)
        if n_dates == 1:
            color_series = [cmap(0.5)]
        else:
            color_series = [cmap(i / (n_dates - 1)) for i in range(n_dates)]
            color_series_clear = [cmap(i / (n_dates_clear - 1)) for i in range(len(date_clear_all))]
            color_series_cloudy = [cmap(i / (n_dates_cloudy - 1)) for i in range(len(date_cloudy_all))]

    # optional ScalarMappable if you want to add a colorbar later
    norm = mpl.colors.Normalize(vmin=0, vmax=max(1, n_dates - 1))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(date_all)):
        ax.plot(date_alb_wvl[i], date_alb[i], label=f'{date_all[i]}, ice fraction={date_ice_frac[i]:.3f}+/-{date_ice_frac_std[i]:.3f}', color=color_series[i])
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
    # plt.show()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(date_all)):
        if date_all[i] not in ['0808', '0809']:
            ax.plot(date_alb_wvl[i], date_alb[i], label=f'{date_all[i]}, ice fraction={date_ice_frac[i]:.3f}+/-{date_ice_frac_std[i]:.3f}', color=color_series[i])
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
    
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(date_clear_all)):
        if date_clear_all[i] not in ['0808', '0809']:
            ax.plot(date_alb_clear_wvl[i], date_alb_clear[i], label=f'{date_clear_all[i]}, ice fraction={date_ice_frac_clear[i]:.3f}+/-{date_ice_frac_clear_std[i]:.3f}', color=color_series_clear[i])
            ax.fill_between(date_alb_clear_wvl[i], date_alb_clear[i]-date_alb_clear_std[i], date_alb_clear[i]+date_alb_clear_std[i], color=color_series_clear[i], alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Clear Sky Surface Albedo (atm corr + fit)\nexclude 0808, 0809', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_ albedo_all_flights_clear_partial.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(date_cloudy_all)):
        if date_cloudy_all[i] not in ['0808', '0809']:
            ax.plot(date_alb_cloudy_wvl[i], date_alb_cloudy[i], label=f'{date_cloudy_all[i]}, ice fraction={date_ice_frac_cloudy[i]:.3f}+/-{date_ice_frac_cloudy_std[i]:.3f}', color=color_series_cloudy[i])
            ax.fill_between(date_alb_cloudy_wvl[i], date_alb_cloudy[i]-date_alb_cloudy_std[i], date_alb_cloudy[i]+date_alb_cloudy_std[i], color=color_series_cloudy[i], alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Below Cloud Surface Albedo (atm corr + fit)\nexclude 0808, 0809', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_ albedo_all_flights_cloudy_partial.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(date_all)):
        ax.plot(date_alb_wvl[i], date_alb[i], label=f'{date_all[i]}, MYI={date_myi_ratio[i]*100:.1f}%', color=color_series[i])
        ax.fill_between(date_alb_wvl[i], date_alb[i]-date_alb_std[i], date_alb[i]+date_alb_std[i], color=color_series[i], alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit)', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_ albedo_all_flights_myi.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(date_all)):
        if date_all[i] not in ['0808', '0809']:
            ax.plot(date_alb_wvl[i], date_alb[i], label=f'{date_all[i]}, MYI={date_myi_ratio[i]*100:.1f}%', color=color_series[i])
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
    fig.savefig(f'{fig_dir}/arcsix_ albedo_all_flights_partial_myi.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(date_all)):
        if date_all[i] not in ['0808', '0809']:
            ax.plot(date_alb_wvl[i], date_alb[i], label=f'{date_all[i]}, Ice Age={date_ice_age_avg[i]:.1f} +/- {date_ice_age_std[i]:.1f} y', color=color_series[i])
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
    fig.savefig(f'{fig_dir}/arcsix_ albedo_all_flights_partial_ice_age.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(date_all)):
        if date_all[i] not in ['0808', '0809']:
            ax.plot(date_alb_wvl[i], date_alb[i], label=f'{date_all[i]}, MYI={date_ice_age_myi_ratio[i]*100:.1f}%', color=color_series[i])
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
    fig.savefig(f'{fig_dir}/arcsix_ albedo_all_flights_partial_ice_age_myi_ratio.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    sys.exit()
    
    """
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
    #"""
    
    """# set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
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
    #"""
    
    
    """ 
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
    #"""
    
    SF_into = {
        '0528': 'SF#01',
        '0530': 'SF#02',
        '0531': 'SF#03',
        '0603': 'SF#04',
        '0605': 'SF#05',
        '0606': 'SF#06',
        '0607': 'SF#07',
        '0610': 'SF#08',
        '0611': 'SF#09',
        '0613': 'SF#10',
        '0725': 'SF#11',
        '0729': 'SF#12',
        '0730': 'SF#13',
        '0801': 'SF#14',
        '0802': 'SF#15',
        '0807': 'SF#16',
        '0808': 'SF#17',
        '0809': 'SF#18',
        '0815': 'SF#19',
    }
    
    # glob file like NSIDC-0803_SEAICE_AMSR2_N_20240528_v2.0.nc
    patern = 'NSIDC-0803_SEAICE_AMSR2_N_2024*.nc'
    amsr2_data_dir = '../data/ice_conc/'
    amsr2_files = sorted(glob.glob(os.path.join(amsr2_data_dir, patern)))
    amsr2_dates = [os.path.basename(f).split('_')[4][0:8] for f in amsr2_files]
    amsr2_dates_int = np.array([int(d[4:8]) for d in amsr2_dates])
    
    ice_conc_all = None
    for fn in amsr2_files:
        with Dataset(fn, 'r') as ds:
            x = ds.variables['x'][:]
            y = ds.variables['y'][:]
            
            # choose variable name (e.g. 'ICECON')
            v = ds.variables['ICECON']
            print("var dims:", v.dimensions, "shape:", v.shape)
            # if variable has a time dim, pick first time index
            if 'time' in v.dimensions:
                # build index tuple: time=0, all y, all x
                idx = [0 if d == 'time' else slice(None) for d in v.dimensions]
                arr = v[tuple(idx)]
            else:
                arr = v[:]   # entire array
            
            arr = np.array(arr, dtype=np.float32)
            print("original arr min/max:", np.nanmin(arr), np.nanmax(arr))
            
            # mask fill values
            fill = getattr(v, '_FillValue', None) or getattr(v, 'missing_value', None)
            print("fill value:", fill)
            if fill is not None:
                arr = np.ma.masked_equal(arr, fill)

            arr[arr > 250] = np.nan  # set fill value to nan
            
            print("arr min/max after fill mask:", np.nanmin(arr), np.nanmax(arr))
            
            # # apply scale and offset if present
            # scale = getattr(v, 'scale_factor', 1.0)
            # offset = getattr(v, 'add_offset', 0.0)
            # print("scale_factor:", scale, "add_offset:", offset)
            # arr = arr.astype(float) * scale + offset
            
            # print("scaled arr min/max:", np.nanmin(arr), np.nanmax(arr))

            
                
            ice_conc = arr.copy()

            print("Data subset shape:", ice_conc.shape)
            
            if 'crs' in ds.variables:
                crs_var = ds.variables['crs']
                # try crs_wkt or spatial_ref or proj4 text
                wkt = getattr(crs_var, 'crs_wkt', None) or getattr(crs_var, 'spatial_ref', None)
                proj4 = getattr(crs_var, 'proj4', None) or getattr(crs_var, 'proj4text', None)
                if wkt:
                    src_crs = CRS.from_wkt(wkt)
                elif proj4:
                    src_crs = CRS.from_string(proj4)
                else:
                    # try EPSG attribute
                    epsg = getattr(crs_var, 'epsg', None)
                    src_crs = CRS.from_epsg(int(epsg)) if epsg else None
            else:
                src_crs = None

            # fallback guess if not present (your file likely has crs)
            if src_crs is None:
                src_crs = CRS.from_epsg(3411)  # North polar stereographic (example)
            
        
        # ice_conc[ice_conc > 250] = np.nan  # set fill value to nan
        print("ice_conc min/max:", np.nanmin(ice_conc), np.nanmax(ice_conc))
        ice_conc_scale = ice_conc*100  # scale factor
        # ds = xr.open_dataset(fn, decode_coords="all", mask_and_scale=False)

        # # pick variable and coords
        # var = 'ICECON' if 'ICECON' in ds.data_vars else list(ds.data_vars.keys())[0]
        # coord_x = 'x'
        # coord_y = 'y'
        # x = ds[coord_x].values
        # y = ds[coord_y].values
        # da = ds[var] # y, x
        # sel = {d: 0 for d in da.dims if d not in (coord_x, coord_y)}
        # da2 = da.isel(**sel) if sel else da
        # da2[da2 == 255] = np.nan  # set fill value to nan
        # data = da2.values*0.004

        # Build mesh and get CRS from crs variable (preferred)
        xx, yy = np.meshgrid(x, y)
        
        # crs_attrs = ds['crs'].attrs
        # # prefer explicit epsg if present, else wkt or proj4
        # if 'epsg' in crs_attrs:
        #     try:
        #         src_crs = CRS.from_user_input(int(crs_attrs['epsg']))
        #     except Exception:
        #         pass
        # if src_crs is None:
        #     # try WKT
        #     for key in ('crs_wkt', 'spatial_ref', 'spref', 'proj4', 'proj4text', 'proj4string'):
        #         if key in crs_attrs and crs_attrs[key]:
        #             try:
        #                 src_crs = CRS.from_user_input(crs_attrs[key])
        #                 break
        #             except Exception:
        #                 pass
        

        # transform to lon/lat
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
        lon_flat, lat_flat = transformer.transform(xx.ravel(), yy.ravel())
        lon = np.array(lon_flat).reshape(xx.shape)
        lat = np.array(lat_flat).reshape(xx.shape)
        print("lon/lat shape:", lon.shape, lat.shape)
        print("ice_conc_scale shape:", ice_conc_scale.shape)
        print("ice_conc_scale min/max before flatten:", np.nanmin(ice_conc_scale), np.nanmax(ice_conc_scale))
        # if ice_conc_all is None:
        #     plt.close('all')
        #     fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.NorthPolarStereo()})
        #     ax.coastlines(resolution='110m', linewidth=0.8)
        #     ax.add_feature(cartopy.feature.LAND, facecolor="#8e8e8e", zorder=0)
        #     ax.add_feature(cartopy.feature.OCEAN, facecolor="#8e8e8e", zorder=0)
        #     plt.pcolormesh(lon, lat, ice_conc_scale, cmap='Blues_r', vmin=0, vmax=100, transform=ccrs.PlateCarree())
        #     plt.colorbar(label='Ice Concentration (%)')
        #     lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
        #     lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
        #     pad_lon = max(1, (lon_max - lon_min) * 0.05)
        #     pad_lat = max(1, (lat_max - lat_min) * 0.05)
        #     ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        #     plt.title(f'Ice Concentration from AMSR2 on {os.path.basename(fn).split("_")[4]}')
        #     plt.xlabel('Longitude')
        #     plt.ylabel('Latitude')
        #     plt.show()
        #     # sys.exit()
        
        
        if ice_conc_all is None:
            # ice_conc_all = ice_conc_scale.copy().T # x, y
            ice_conc_all = np.array(ice_conc_scale.copy(), dtype=np.float32)
        else:
            ice_conc_all = np.dstack((ice_conc_all, ice_conc_scale.copy())) # x, y, time
            
    
        
        del ice_conc_scale, ds, xx, yy, lon_flat, lat_flat, transformer
    

    # ice_conc_all *= 100.0  # convert to percentage
    ice_conc_spring_avg = np.nanmean(ice_conc_all[:, :, amsr2_dates_int <= 630], axis=2)
    ice_conc_summer_avg = np.nanmean(ice_conc_all[:, :, amsr2_dates_int >= 701], axis=2)
    ice_conc_spring_std = np.nanstd(ice_conc_all[:, :, amsr2_dates_int <= 630], axis=2)
    ice_conc_summer_std = np.nanstd(ice_conc_all[:, :, amsr2_dates_int >= 701], axis=2)
    

    
    plt.close('all')
    central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    


    # Create the plot
    # fig, axd = plt.subplot_mosaic(mosaic, figsize=(18, 9), per_subplot_kw=kw, gridspec_kw={'width_ratios': [4, 3], 'wspace': 0.1})
    # ax11 = axd['map_top']
    # ax12 = axd['std_top']
    # ax21 = axd['map_bot']
    # ax22 = axd['std_bot']
    
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 2], wspace=0.1 , hspace=0.3)
    ax11 = fig.add_subplot(gs[0,0], projection=proj)  # map_top
    ax12 = fig.add_subplot(gs[0,1])                   # std_top (regular axes)
    ax21 = fig.add_subplot(gs[1,0], projection=proj)  # map_bot
    ax22 = fig.add_subplot(gs[1,1])                   # std_bot
    
    # add coastlines and land features for context
    for ax1 in [ax11, ax21]:
        # add coastlines and land features for context
        ax1.coastlines(resolution='50m', linewidth=0.8)
        ax1.add_feature(cartopy.feature.LAND, facecolor="#8e8e8e", zorder=0)
        ax1.add_feature(cartopy.feature.OCEAN, facecolor="#8e8e8e", zorder=0)

        # gridlines with labels (only lon/lat labels make sense in PlateCarree transform)
        gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        
    # plot ice concentration maps
    im11 = ax11.pcolormesh(lon, lat, ice_conc_spring_avg, cmap='Blues_r', vmin=0, vmax=100,
                        transform=ccrs.PlateCarree(), zorder=1)
    cbar11_2 = fig.colorbar(im11, ax=ax11, orientation='vertical', pad=0.01, shrink=0.9)
    cbar11_2.set_label('Ice Concentration (%)', fontsize=10)
    im21 = ax21.pcolormesh(lon, lat, ice_conc_summer_avg, cmap='Blues_r', vmin=0, vmax=100,
                        transform=ccrs.PlateCarree(), zorder=1)
    cbar21_2 = fig.colorbar(im21, ax=ax21, orientation='vertical', pad=0.01, shrink=0.9)
    cbar21_2.set_label('Ice Concentration (%)', fontsize=10)
      
    # scatter the flight-leg centers (and all points) using PlateCarree transform
    # color by broadband surface albedo from iteration 2 if available
    mask = ~np.isnan(broadband_alb_iter2_all_spring) #and broadband_alb_iter2_all_spring > 0
    sc11 = ax11.scatter(lon_all_spring[mask], lat_all_spring[mask], s=5, c=broadband_alb_iter2_all_spring[mask], cmap='jet',
                    transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=0.9)
    cbar11 = fig.colorbar(sc11, ax=ax11, orientation='vertical', pad=0.05, shrink=0.9)
    cbar11.set_label('Broadband Albedo', fontsize=10)

    # # also plot all sampled points along legs (optional, lighter marker)
    # if 'lon_all_spring' in locals() and 'lat_all_spring' in locals() and len(lon_all) > 0:
    #     ax21.scatter(lon_all_spring, lat_all_spring, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)
        
    # scatter the flight-leg centers (and all points) using PlateCarree transform
    # color by broadband surface albedo from iteration 2 if available
    mask = ~np.isnan(broadband_alb_iter2_all_summer) #  and broadband_alb_iter2_all_summer > 0
    sc21 = ax21.scatter(lon_all_summer[mask], lat_all_summer[mask], s=5, c=broadband_alb_iter2_all_summer[mask], cmap='jet',
                    transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=0.9)
    cbar21 = fig.colorbar(sc21, ax=ax21, orientation='vertical', pad=0.05 , shrink=0.9)
    cbar21.set_label('Broadband Albedo', fontsize=10)

    # # also plot all sampled points along legs (optional, lighter marker)
    # if 'lon_all_summer' in locals() and 'lat_all_summer' in locals() and len(lon_all) > 0:
    #     ax21.scatter(lon_all_summer, lat_all_summer, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)
    
    lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
    lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
    for ax in [ax11, ax21]:
        # expand a bit
        pad_lon = max(0.2, (lon_max - lon_min) * 0.05)
        pad_lat = max(0.2, (lat_max - lat_min) * 0.05)
        ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        ax.tick_params('both', labelsize=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
    ax11.set_title(f'SF#01-10 (May 28 - June 13)', fontsize=12)
    ax21.set_title(f'SF#11-19 (July 25 - Aug 15)', fontsize=12)
        
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
    
    eff_alb_ = gas_abs_masking(date_alb_wvl[0], np.ones_like(date_alb_wvl[0]), alt=5)
    for i in range(len(date_all)):
        ax12.plot(date_alb_wvl[i], date_alb[i], label=f'{SF_into[date_all[i]]} ({date_all[i]})', color=color_series[i])
        ax12.fill_between(date_alb_wvl[i], date_alb[i]-date_alb_std[i], date_alb[i]+date_alb_std[i], color=color_series[i], alpha=0.1)
    ax12.fill_between(date_alb_wvl[0], -0.05, 1.05, where=np.isnan(eff_alb_), color='gray', alpha=0.2,)# label='Mask Gas absorption bands')
    ax12.set_xlabel('Wavelength (nm)', fontsize=14)
    ax12.set_ylabel('Surface Albedo', fontsize=14)
    ax12.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.05, -0.1))
    ax12.tick_params(labelsize=12)
    ax12.set_ylim(-0.05, 1.05)
    ax12.set_xlim(350, 2000)
    
    ax22.errorbar(date_ice_frac, date_broadband_alb,
                  xerr=date_ice_frac_std,
                  yerr=date_broadband_alb_std, fmt='o', color='black', ecolor='lightgray', 
                  markersize=3, markerfacecolor='none',
                  elinewidth=1.5, capsize=1.5, zorder=2)
    ax22.scatter(date_ice_frac, date_broadband_alb, s=50, c=color_series, zorder=3)
    ax22.set_xlabel('Sea Ice Fraction', fontsize=14)
    ax22.set_ylabel('Broadband Albedo', fontsize=14)
    ax22.tick_params(labelsize=12)
    ax22.set_xlim(0.1, 1.10)
    ax22.set_ylim(0.1, 0.9)
    for ax, cap in zip([ax11, ax12, ax21, ax22], ['(a)', '(c)', '(b)', '(d)']):
        ax.text(0.0, 1.01, cap, transform=ax.transAxes, fontsize=14,
                verticalalignment='bottom', horizontalalignment='left')
    fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_spring_summer_combined.png', bbox_inches='tight', dpi=300)
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