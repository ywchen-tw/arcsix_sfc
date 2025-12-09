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
from matplotlib.gridspec import GridSpec
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
from pyproj import Transformer
from util import *
# mpl.use('Agg')
from matplotlib import rcParams

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif" # Ensure sans-serif is used as the default family



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



o2a_1_start, o2a_1_end = 748, 780
# h2o_1_start, h2o_1_end = 672, 706
# h2o_2_start, h2o_2_end = 705, 746
h2o_1_start, h2o_1_end = 650 , 706
h2o_2_start, h2o_2_end = 705, 760
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

from scipy import stats

def linear_regression_with_confidence(x, y, confidence=0.95):
    """
    Performs linear regression and calculates the confidence interval for the regression line.
    """
    # 1. Perform Linear Regression
    res = stats.linregress(x, y)
    slope = res.slope
    intercept = res.intercept
    
    # 2. Prepare data for plotting (sorting ensures the band is drawn correctly)
    # We create a sorted version of x to plot smooth lines/bands
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Predicted line
    y_pred = slope * x_sorted + intercept
    
    # 3. Calculate Statistics for Confidence Intervals
    n = len(x)
    df = n - 2  # Degrees of freedom
    
    # Residuals and Standard Error of the Estimate (SEE)
    residuals = y_sorted - y_pred
    sum_squared_residuals = np.sum(residuals**2)
    se_estimate = np.sqrt(sum_squared_residuals / df)
    
    # Critical t-value for the given confidence level (two-tailed)
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # 4. Calculate Confidence Interval (Mean Response)
    # Formula: CI = y_hat +/- t_crit * se_estimate * sqrt(1/n + (x - x_bar)^2 / Sxx)
    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean)**2)  # Sum of squared differences for x
    
    # Standard error of the fit at specific x values
    se_fit = se_estimate * np.sqrt(1/n + (x_sorted - x_mean)**2 / Sxx)
    
    # Calculate bounds
    margin_error = t_crit * se_fit
    lower_bound = y_pred - margin_error
    upper_bound = y_pred + margin_error
    
    return x_sorted, y_pred, lower_bound, upper_bound, res

def combined_atm_corr():
    log = logging.getLogger("atm corr combined")

    output_dir = f'{_fdir_general_}/sfc_alb_combined_smooth_450nm'
    

    # output_all_dict = {
    #     'wvl_spring': wvl_spring,
    #     'wvl_summer': wvl_summer,
    #     'time_all': time_all,
    #     'lon_all_spring': lon_all_spring,
    #     'lat_all_spring': lat_all_spring,
    #     'alt_all_spring': alt_all_spring,
    #     'dates_spring_all': dates_spring_all,
    #     'leg_contidions_all_spring': leg_contidions_all_spring,
    #     'case_tags_spring': case_tags_spring,
    #     'fdn_all_spring': fdn_all_spring,
    #     'fup_all_spring': fup_all_spring,
    #     'toa_expand_all_spring': toa_expand_all_spring,
    #     'icing_all_spring': icing_all_spring,
    #     'icing_pre_all_spring': icing_pre_all_spring,
    #     'alb_iter1_all_spring': alb_iter1_all_spring,
    #     'alb_iter2_all_spring': alb_iter2_all_spring,
    #     'broadband_alb_iter1_all_spring': broadband_alb_iter1_all_spring,
    #     'broadband_alb_iter2_all_spring': broadband_alb_iter2_all_spring,
    #     'broadband_alb_iter1_all_filter_spring': broadband_alb_iter1_all_filter_spring,
    #     'broadband_alb_iter2_all_filter_spring': broadband_alb_iter2_all_filter_spring,
    #     'time_summer': time_summer,
    #     'lon_all_summer': lon_all_summer,
    #     'lat_all_summer': lat_all_summer,
    #     'alt_all_summer': alt_all_summer,
    #     'dates_summer_all': dates_summer_all,
    #     'leg_contidions_all_summer': leg_contidions_all_summer,
    #     'case_tags_summer': case_tags_summer,
    #     'fdn_all_summer': fdn_all_summer,
    #     'fup_all_summer': fup_all_summer,
    #     'toa_expand_all_summer': toa_expand_all_summer,
    #     'icing_all_summer': icing_all_summer,
    #     'icing_pre_all_summer': icing_pre_all_summer,
    #     'alb_iter1_all_summer': alb_iter1_all_summer,
    #     'alb_iter2_all_summer': alb_iter2_all_summer,
    #     'broadband_alb_iter1_all_summer': broadband_alb_iter1_all_summer,
    #     'broadband_alb_iter2_all_summer': broadband_alb_iter2_all_summer,
    #     'broadband_alb_iter1_all_filter_summer': broadband_alb_iter1_all_filter_summer,
    #     'broadband_alb_iter2_all_filter_summer': broadband_alb_iter2_all_filter_summer,
    # }
    
    combined_output_file = f'{output_dir}/sfc_alb_combined_spring_summer.pkl'
    with open(combined_output_file, 'rb') as r:
        combined_data = pickle.load(r)
        
    fig_dir = f'fig/sfc_alb_corr_analysis'
    os.makedirs(fig_dir, exist_ok=True)
    
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
    
    # load ice fraction data
    file = f'{_fdir_general_}/ice_frac/ice_frac_all.pkl'
    with open(file, 'rb') as f:
        ice_frac_data = pickle.load(f)
    ice_frac_date = ice_frac_data['date']
    ice_frac_time = ice_frac_data['time']
    ice_frac_values = ice_frac_data['ice_frac']
    
    
    
    date_conditions = []
    broadband_alb_date_cond = []
    broadband_alb_date_cond_upper = []
    broadband_alb_date_cond_lower = []
    broadband_alb_date_cond_unc = []
    myi_ratio_date_cond = []
    myi_ratio_date_cond_std = []
    
    
    for date_key in (ice_frac_time_offset.keys()):
        for case_tag in ['clear', 'cloudy']:
            date_select = date_key
            date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
            case_tag_mask = np.array([case_tag in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag in ct for ct in combined_data['case_tags_summer_all']])
            final_mask = date_summer_mask & case_tag_mask
            if not np.any(final_mask):
                print(f"No data for date {date_select} and case tag {case_tag}. Skipping.")
                continue
            alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
            lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
            lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
            alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
            time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
            alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
            broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
            myi_selected_all = combined_data['myi_ratio_summer_all'][final_mask] if date_select > '20240630' else combined_data['myi_ratio_spring_all'][final_mask]
            ice_ratio_selected_all = combined_data['ice_ratio_summer_all'][final_mask] if date_select > '20240630' else combined_data['ice_ratio_spring_all'][final_mask]
            
            myi_selected_all = myi_selected_all / (ice_ratio_selected_all + 1e-6) * 100  # to percentage
            
            
            alt_mask = (alt_selected_all >= 0) & (alt_selected_all <=   1.6)
            
            
            
            if not np.any(alt_mask):
                print(f"No data with altitude < 1.6 km for date {date_select} and case tag {case_tag}. Skipping.")
                continue
            lon_selected_all = lon_selected_all[alt_mask]
            lat_selected_all = lat_selected_all[alt_mask]
            alt_selected_all = alt_selected_all[alt_mask]
            alb_selected_all = alb_selected_all[alt_mask, :]
            broadband_alb_selected_all = broadband_alb_selected_all[alt_mask]
            time_selected_all = time_selected_all[alt_mask]
            myi_selected_all = myi_selected_all[alt_mask]


            cam_time = ice_frac_time[ice_frac_date == int(date_key)]
            cam_ice_fraction = ice_frac_values[ice_frac_date == int(date_key)]
                        

            alb_selected_all_avg = np.nanmean(alb_selected_all, axis=0)
            alb_selected_all_std = np.nanstd(alb_selected_all, axis=0)
            alt_selected_all_avg = np.nanmean(alt_selected_all)
            
            """
            # find best time offset for cam ice fraction
            best_corr_coef = 0
            best_offset = 0
            cam_time_original = cam_time.copy()
            cam_ice_fraction_original = cam_ice_fraction.copy()
            offset_list = np.arange(-1, 1.05, 0.05)/3600
            offset_results = {}
            for cam_time_offset in offset_list:
                # cam_time = cam_time +  ice_frac_time_offset[date_key]
                cam_time = cam_time_original +  cam_time_offset
                
                cam_time_mask = (cam_time >= np.min(time_selected_all)) & (cam_time <= np.max(time_selected_all))
                cam_time = cam_time[cam_time_mask]
                cam_ice_fraction = cam_ice_fraction_original[cam_time_mask]
                
                
                broadband_alb_cam_time = np.zeros_like(cam_time)
                broadband_alb_cam_time[:] = np.nan
                alb_cam_time = np.zeros((len(cam_time), alb_selected_all.shape[1]))
                alb_cam_time[:] = np.nan
                for i, t in enumerate(cam_time):
                    time_diff = np.abs(time_selected_all - t)
                    if np.min(time_diff) <= 1/60/60:  # within 1s 
                        closest_idx = np.argmin(time_diff)
                        broadband_alb_cam_time[i] = broadband_alb_selected_all[closest_idx]
                        alb_cam_time[i, :] = alb_selected_all[closest_idx, :]
                
                valid_mask = ~np.isnan(cam_ice_fraction) & ~np.isnan(broadband_alb_cam_time)
                corr_coef = np.corrcoef(cam_ice_fraction[valid_mask], broadband_alb_cam_time[valid_mask])[0, 1]
                
                if corr_coef > best_corr_coef:
                    best_corr_coef = corr_coef
                    best_offset = cam_time_offset
            
            
            
                # fig, (ax, axr) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})
                # ax2 = ax.twinx()
                # l1 = ax.plot(time_selected_all, broadband_alb_selected_all, c='skyblue', label='Broadband Albedo', alpha=0.75, linewidth=2)
                # l2 = ax2.plot(cam_time, cam_ice_fraction, c='coral', label='Camera Ice Fraction', alpha=0.75, linewidth=1.5)
                # ax.set_xlabel('Time (UTC)', fontsize=14)
                # ax.set_ylabel('Broadband Albedo', fontsize=14)
                # ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
                # lns = [l1[0], l2[0]]
                # labs = [l.get_label() for l in lns]
                # ax.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
                # ax.tick_params(labelsize=12)
                # # ax.set_ylim(-0.05, 1.05)
                # ax.set_title('Surface Albedo (atm corr + fit)', fontsize=13)
                # # ax.set_xlim(350, 2000)
            
                # axr.scatter(cam_ice_fraction, broadband_alb_cam_time, c='purple', alpha=0.6)
                # # calculate slope and correlation coefficient
                # valid_mask = ~np.isnan(cam_ice_fraction) & ~np.isnan(broadband_alb_cam_time)
                # if np.sum(valid_mask) > 2:
                #     slope, intercept = np.polyfit(cam_ice_fraction[valid_mask], broadband_alb_cam_time[valid_mask], 1)
                #     corr_coef = np.corrcoef(cam_ice_fraction[valid_mask], broadband_alb_cam_time[valid_mask])[0, 1]
                #     # x_vals = np.array([0, 1])
                #     x_vals = np.array([np.min(cam_ice_fraction[valid_mask]), np.max(cam_ice_fraction[valid_mask])])
                #     y_vals = intercept + slope * x_vals
                #     axr.plot(x_vals, y_vals, color='orange', linestyle='--', label=f'Fit: y={slope:.2f}x+{intercept:.2f}\nR={corr_coef:.2f}')
                #     axr.legend(fontsize=10)
                    
                #     date_conditions.append(f'{date_key}_{case_tag}')
                #     broadband_alb_date_cond.append(intercept + slope * 1.0)  # ice fraction = 1.0
                # axr.set_xlabel('Camera Ice Fraction', fontsize=14)
                # axr.set_ylabel('Broadband Albedo', fontsize=14)
                # axr.tick_params(labelsize=12)
                # axr.set_title(f'Broadband Albedo vs. Camera Ice Fraction (time offset={cam_time_offset*3600:.2f})', fontsize=13)
                # fig.suptitle(f'Date: {date_key}, Case: {case_tag}, Alt < 1.6 km',
                #              fontsize=16)
                # fig.savefig(f'{fig_dir}/arcsix_albedo_{date_key}_{case_tag}_broadband_ice_frac_toff_{cam_time_offset*3600:.2f}s.png', bbox_inches='tight', dpi=150)
                # plt.close(fig)
                
            
            
            
            
            print(f"Date: {date_key}, Case: {case_tag}, Best Time Offset (s): {best_offset*3600:.3f}, Best Corr Coef: {best_corr_coef:.3f}")
            offset_results[f'{date_key}_{case_tag}_timeoffset'] = best_offset*3600  # in hours
            offset_results[f'{date_key}_{case_tag}_correlation'] = best_corr_coef
            """
            
            
            cam_time = cam_time +  ice_frac_time_offset[date_key]
            
            cam_time_mask = (cam_time >= np.min(time_selected_all)) & (cam_time <= np.max(time_selected_all))
            cam_time = cam_time[cam_time_mask]
            cam_ice_fraction = cam_ice_fraction[cam_time_mask]
            
            
            broadband_alb_cam_time = np.zeros_like(cam_time)
            broadband_alb_cam_time[:] = np.nan
            myi_cam_time = np.zeros_like(cam_time)
            myi_cam_time[:] = np.nan
            alb_cam_time = np.zeros((len(cam_time), alb_selected_all.shape[1]))
            alb_cam_time[:] = np.nan
            for i, t in enumerate(cam_time):
                time_diff = np.abs(time_selected_all - t)
                if np.min(time_diff) <= 1/60/60:  # within 1s 
                    closest_idx = np.argmin(time_diff)
                    broadband_alb_cam_time[i] = broadband_alb_selected_all[closest_idx]
                    alb_cam_time[i, :] = alb_selected_all[closest_idx, :]
                    myi_cam_time[i] = myi_selected_all[closest_idx]
            
            valid_mask = ~np.isnan(cam_ice_fraction) & ~np.isnan(broadband_alb_cam_time)
            corr_coef = np.corrcoef(cam_ice_fraction[valid_mask], broadband_alb_cam_time[valid_mask])[0, 1]
            
            fig, (ax, axr) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})
            ax2 = ax.twinx()
            l1 = ax.plot(time_selected_all, broadband_alb_selected_all, c='skyblue', label='Broadband Albedo', alpha=0.75, linewidth=2)
            l2 = ax2.plot(cam_time, cam_ice_fraction, c='coral', label='Camera Ice Fraction', alpha=0.75, linewidth=1.5)
            ax.set_xlabel('Time (UTC)', fontsize=14)
            ax.set_ylabel('Broadband Albedo', fontsize=14)
            ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
            lns = [l1[0], l2[0]]
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
            ax.tick_params(labelsize=12)
            # ax.set_ylim(-0.05, 1.05)
            ax.set_title('Surface Albedo (atm corr + fit)', fontsize=13)
            # ax.set_xlim(350, 2000)
        
            # axr.scatter(cam_ice_fraction, broadband_alb_cam_time, c='purple', alpha=0.6)
            cc = axr.scatter(cam_ice_fraction, broadband_alb_cam_time, c=myi_cam_time, alpha=0.6, cmap='jet', vmin=0, vmax=100)
            cbar = fig.colorbar(cc, ax=axr)
            cbar.set_label('Multi-year sea ice ratio (%)', fontsize=12)
            # calculate slope and correlation coefficient
            valid_mask = ~np.isnan(cam_ice_fraction) & ~np.isnan(broadband_alb_cam_time)
            if np.sum(valid_mask) > 2:
                x_sorted, y_pred, lower_bound, upper_bound, res = linear_regression_with_confidence(
                    cam_ice_fraction[valid_mask], broadband_alb_cam_time[valid_mask], confidence=0.95)
                # slope, intercept = np.polyfit(cam_ice_fraction[valid_mask], broadband_alb_cam_time[valid_mask], 1)
                # corr_coef = np.corrcoef(cam_ice_fraction[valid_mask], broadband_alb_cam_time[valid_mask])[0, 1]
                # # x_vals = np.array([0, 1])
                # x_vals = np.array([np.min(cam_ice_fraction[valid_mask]), np.max(cam_ice_fraction[valid_mask])])
                # y_vals = intercept + slope * x_vals
                # axr.plot(x_vals, y_vals, color='orange', linestyle='--', label=f'Fit: y={slope:.2f}x+{intercept:.2f}\nR={corr_coef:.2f}')
                axr.plot(x_sorted, y_pred, color='orange', linestyle='--', label=f'Fit: y={res.slope:.2f}x+{res.intercept:.2f}\nR={res.rvalue:.2f}')
                axr.fill_between(x_sorted, lower_bound, upper_bound, color='orange', alpha=0.2, label='95% Confidence Interval')
                intercept = res.intercept
                slope = res.slope
                axr.legend(fontsize=10)
                
                date_conditions.append(f'{date_key}_{case_tag}')
                broadband_alb_date_cond.append(intercept + slope * 1.0)  # ice fraction = 1.0
                # # calculate upper and lower bounds based on 95% confidence interval
                # # residuals
                # residuals = broadband_alb_cam_time[valid_mask] - (slope * cam_ice_fraction[valid_mask] + intercept)
                # ss_res = np.sum(residuals**2)
                # ss_tot = np.sum((broadband_alb_cam_time[valid_mask] - np.mean(broadband_alb_cam_time[valid_mask]))**2)
                # r_squared = 1 - (ss_res / ss_tot)
                # n = np.sum(valid_mask)
                # se = np.sqrt(ss_res / (n - 2))
                # t_val = 2.0  # approximate for large n
                # delta = t_val * se * np.sqrt(1/n + (1.0 - np.mean(cam_ice_fraction[valid_mask]))**2 / np.sum((cam_ice_fraction[valid_mask] - np.mean(cam_ice_fraction[valid_mask]))**2))
                # broadband_alb_date_cond_upper.append((intercept + slope * 1.0) + delta)
                # broadband_alb_date_cond_lower.append((intercept + slope * 1.0) - delta)
                # broadband_alb_date_cond_unc.append(delta)
                x_1_index = np.argmin(np.abs(x_sorted - 1.0))
                broadband_alb_at_1 = y_pred[x_1_index]
                broadband_alb_at_1_upper = upper_bound[x_1_index]
                broadband_alb_at_1_lower = lower_bound[x_1_index]
                broadband_alb_date_cond_upper.append(broadband_alb_at_1_upper)
                broadband_alb_date_cond_lower.append(broadband_alb_at_1_lower)
                broadband_alb_date_cond_unc.append((broadband_alb_at_1_upper - broadband_alb_at_1_lower)/2)
                
                myi_ratio_date_cond.append(np.nanmean(myi_cam_time))
                myi_ratio_date_cond_std.append(np.nanstd(myi_cam_time))
                
            axr.set_xlabel('Camera Ice Fraction', fontsize=14)
            axr.set_ylabel('Broadband Albedo', fontsize=14)
            axr.tick_params(labelsize=12)
            axr.set_title('Broadband Albedo vs. Camera Ice Fraction ', fontsize=13)
            fig.suptitle(f'Date: {date_key}, Case: {case_tag}, Alt < 1.6 km',
                            fontsize=16)
            fig.savefig(f'{fig_dir}/arcsix_albedo_{date_key}_{case_tag}_broadband_ice_frac.png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            

    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue' if 'clear' in dc else 'gray' for dc in date_conditions]
    ax.errorbar(np.arange(len(broadband_alb_date_cond)), broadband_alb_date_cond, 
                yerr=broadband_alb_date_cond_unc,
                fmt='o', ecolor='lightgray', capsize=5, markerfacecolor='none', zorder=1)#, label='Data Points with Uncertainty')
    ax.scatter(np.arange(len(broadband_alb_date_cond)), broadband_alb_date_cond, c=colors, alpha=0.8, s=50, zorder=2)
    ax.set_xticks(np.arange(len(broadband_alb_date_cond)))
    ax.set_xticklabels(date_conditions, rotation=45, ha='right')
    ax.legend(handles=[mpatches.Patch(color='blue', label='Clear'),
                       mpatches.Patch(color='gray', label='Cloudy')], fontsize=10)
    ax.set_xlabel('Date and Case', fontsize=14)
    ax.set_ylabel('Broadband Albedo at Ice Fraction = 1.0', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title('Broadband Albedo at Ice Fraction = 1.0 from Linear Fit', fontsize=13)
    fig.savefig(f'{fig_dir}/arcsix_albedo_broadband_ice_frac_fit_summary.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue' if 'clear' in dc else 'gray' for dc in date_conditions]
    ax.errorbar(myi_ratio_date_cond, broadband_alb_date_cond, 
                xerr=myi_ratio_date_cond_std,
                yerr=broadband_alb_date_cond_unc, 
                fmt='o', ecolor='lightgray', capsize=5, markerfacecolor='none', zorder=1)#label='Data Points with Uncertainty')
    ax.scatter(myi_ratio_date_cond, broadband_alb_date_cond, c=colors, alpha=0.8, s=50, zorder=2)
    ax.set_xlabel('Mean Multi-year Sea Ice Ratio', fontsize=14)
    ax.set_ylabel('Broadband Albedo at Ice Fraction = 1.0', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title('Broadband Albedo at Ice Fraction = 1.0 vs. Mean MYI Ratio', fontsize=13)
    fig.savefig(f'{fig_dir}/arcsix_albedo_broadband_ice_frac_vs_myi_ratio.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)

    
    

    # 8/1 clear_atm_corr 
    date_select = '20240801'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select_1 = 'clear_atm_corr'
    case_tag_mask_1 = np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_select_2 = 'clear_atm_corr_2'
    case_tag_mask_2 = np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_mask = case_tag_mask_1 | case_tag_mask_2
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    # read cam_ice_fraction_20240801_134836_140900.nc
    with Dataset(f'cam_ice_fraction_20240801_134836_140900.nc', 'r') as nc:
        cam_time = nc.variables['tmhr'][:]  
        cam_ice_fraction = nc.variables['ice_fraction'][:]  # shape (time, lat, lon)
    
    time_mask = (time_selected_all >= 13.84) & (time_selected_all <= 14.12)
    time_selected_all = time_selected_all[time_mask]
    lon_selected_all = lon_selected_all[time_mask]
    lat_selected_all = lat_selected_all[time_mask]
    alt_selected_all = alt_selected_all[time_mask]
    alb_selected_all = alb_selected_all[time_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[time_mask]
    

    alb_selected_all_avg = np.nanmean(alb_selected_all, axis=0)
    alb_selected_all_std = np.nanstd(alb_selected_all, axis=0)
    alt_selected_all_avg = np.nanmean(alt_selected_all)
    
    
    cam_time = cam_time - 0.750/60/60  # convert from hours since 00:00 to UTC time in hours 
    cam_time_mask = (cam_time >= 13.84) & (cam_time <= 14.12)
    cam_time = cam_time[cam_time_mask]
    cam_ice_fraction = cam_ice_fraction[cam_time_mask]
    
    broadband_alb_cam_time = np.zeros_like(cam_time)
    broadband_alb_cam_time[:] = np.nan
    alb_cam_time = np.zeros((len(cam_time), alb_selected_all.shape[1]))
    alb_cam_time[:] = np.nan
    for i, t in enumerate(cam_time):
        time_diff = np.abs(time_selected_all - t)
        if np.min(time_diff) <= 1/60/60:  # within 1s 
            closest_idx = np.argmin(time_diff)
            broadband_alb_cam_time[i] = broadband_alb_selected_all[closest_idx]
            alb_cam_time[i, :] = alb_selected_all[closest_idx, :]
    
    
    
    # alb_ext_wvl, alb_ext = alb_extention(alb_wvl, alb_selected_all_avg, clear_sky=True)
    # plt.close('all')
    # plt.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km')
    # plt.plot(alb_ext_wvl, alb_ext, label='Extended')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Surface Albedo')
    # plt.title('Surface Albedo Extension Check, Aug 1st')
    # plt.legend()
    # plt.show()
    # sys.exit()    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km', color='b')
    ax.fill_between(alb_wvl, alb_selected_all_avg-alb_selected_all_std, alb_selected_all_avg+alb_selected_all_std, 
                    color='b', alpha=0.1)
    ax.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km', color='r')
    ax.fill_between(alb_wvl, alb_selected_all_avg-alb_selected_all_std, alb_selected_all_avg+alb_selected_all_std, 
                    color='r', alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_avg.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lat_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_broadband_lat.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lon_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_broadband_lon.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    l1 = ax.scatter(time_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    l2 = ax2.scatter(cam_time, cam_ice_fraction, label='Camera Ice Fraction', c='r', s=5)
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
    lns = [l1, l2]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_broadband_time.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    l1 = ax.plot(time_selected_all, broadband_alb_selected_all, c='skyblue', label='Broadband Albedo', alpha=0.75, linewidth=2)
    l2 = ax2.plot(cam_time, cam_ice_fraction, c='coral', label='Camera Ice Fraction', alpha=0.75, linewidth=1.5)
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
    lns = [l1[0], l2[0]]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_broadband_time_line.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cam_ice_fraction, broadband_alb_cam_time, s=10, c='k')
    ax.set_xlabel('Camera Ice Fraction', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title('Surface Albedo vs Camera Ice Fraction, Aug 1st', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_broadband_icefraction.png', bbox_inches='tight', dpi=150)
    plt.close(fig) 
    
    wvl_450nm_idx = np.argmin(np.abs(alb_wvl - 450))
    wvl_860nm_idx = np.argmin(np.abs(alb_wvl - 860))
    wvl_1240nm_idx = np.argmin(np.abs(alb_wvl - 1240))
    wvl_1630nm_idx = np.argmin(np.abs(alb_wvl - 1630))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cam_ice_fraction, alb_cam_time[:, wvl_450nm_idx], s=10, c='b', label='450nm')
    ax.scatter(cam_ice_fraction, alb_cam_time[:, wvl_860nm_idx], s=10, c='g', label='860nm')
    ax.scatter(cam_ice_fraction, alb_cam_time[:, wvl_1240nm_idx], s=10, c='r', label='1240nm')
    ax.scatter(cam_ice_fraction, alb_cam_time[:, wvl_1630nm_idx], s=10, c='m', label='1630nm')
    ax.set_xlabel('Camera Ice Fraction', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    ax.set_title('Surface Albedo at Different Wavelengths vs Camera Ice Fraction, Aug 1st', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_wvl_icefraction.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    wvl_slope = np.zeros_like(alb_wvl)
    wvl_slope[:] = np.nan
    wvl_r2 = np.zeros_like(alb_wvl)
    wvl_r2[:] = np.nan
    from scipy.stats import linregress
    from mpl_toolkits.axes_grid1 import make_axes_locatable    

    
    for i in range(len(alb_wvl)):
        # Perform linear regression
        mask = np.isfinite(alb_cam_time[:, i]) & np.isfinite(cam_ice_fraction)
        slope, intercept, r_value, p_value, std_err = linregress(cam_ice_fraction[mask], alb_cam_time[:, i][mask])
        wvl_r2[i] = r_value**2
        wvl_slope[i] = slope
        # print(f'Wavelength: {alb_wvl[i]:.1f} nm, Slope: {slope:.4f}, R²: {r_value**2:.4f}')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()
    c1 = ax.scatter(alb_wvl, wvl_slope, c=wvl_r2, s=10, cmap='jet', vmin=0, vmax=1)
    ax2.plot(alb_wvl, alb_selected_all_avg, color='k', alpha=0.5)
    
    # Create an AxesDivider for ax1
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.3) # Create a new axes for the colorbar
    # cbar = fig.colorbar(c1, ax=ax, pad=0.55, orientation='horizontal')
    # cbar.set_label('R²', fontsize=12)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Slope', fontsize=14)
    ax2.set_ylabel('Avg Surface Albedo', fontsize=14)
    ax2.legend(['Avg Surface Albedo'], fontsize=10)
    ax.set_xlim(350, 2000)
    ax.tick_params(labelsize=12)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_title('Correlation between Surface Albedo and Camera Ice Fraction vs Wavelength, Aug 1st', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_wvl_icefraction_correlation.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    ### Plot flight tracks
    
    # cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    cartopy_proj = ccrs.NorthPolarStereo(central_longitude=np.nanmean(lon_selected_all))
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': cartopy_proj})

    # Set the extent for the main axes
    lon_min = np.nanmin(lon_selected_all)
    lon_max = np.nanmax(lon_selected_all)
    lat_min = np.nanmin(lat_selected_all)
    lat_max = np.nanmax(lat_selected_all)
    
    print("ori lon extent:", lon_min, lon_max)
    print("ori lat extent:", lat_min, lat_max)
    
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 3.
    lat_buffer = (lat_max - lat_min) * 1.
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    print("lon extent:", lon_min, lon_max)
    print("lat extent:", lat_min, lat_max)
    
    # features
    ax.coastlines(linewidth=0.5, color='black')
    ax.add_feature(
        cfeature.LAND.with_scale('50m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax.add_feature(
        cfeature.OCEAN.with_scale('50m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 15.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 5.0))
    g1.top_labels = False


    ax.scatter(lon_selected_all, lat_selected_all, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=5, zorder=3)
    
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')

    plt.savefig(f'{fig_dir}/arcsix_flight_paths_0801_clear.png', dpi=300, bbox_inches='tight')
    
    
    # 8/1 clear_atm_corr 2
    date_select = '20240801'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select_1 = 'clear_atm_corr'
    case_tag_mask_1 = np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_select_2 = 'clear_atm_corr_2'
    case_tag_mask_2 = np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_mask = case_tag_mask_1 | case_tag_mask_2
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    # read cam_ice_fraction_20240801_134836_140900.nc
    with Dataset(f'cam_ice_fraction_20240801_144200_150600.nc', 'r') as nc:
        cam_time3 = nc.variables['tmhr'][:]  
        cam_ice_fraction3 = nc.variables['ice_fraction'][:]  # shape (time, lat, lon)
    
    time_mask = (time_selected_all >= 14.739) & (time_selected_all <= 15.053)
    time_selected_all = time_selected_all[time_mask]
    lon_selected_all = lon_selected_all[time_mask]
    lat_selected_all = lat_selected_all[time_mask]
    alt_selected_all = alt_selected_all[time_mask]
    alb_selected_all = alb_selected_all[time_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[time_mask]
    

    alb_selected_all_avg = np.nanmean(alb_selected_all, axis=0)
    alb_selected_all_std = np.nanstd(alb_selected_all, axis=0)
    alt_selected_all_avg = np.nanmean(alt_selected_all)
    
    
    
    cam_time3 = cam_time3 - 0/60/60  # convert from hours since 00:00 to UTC time in hours 
    cam_time_mask3 = (cam_time3 >= 14.739) & (cam_time3 <= 15.053)
    cam_time3 = cam_time3[cam_time_mask3]
    cam_ice_fraction3 = cam_ice_fraction3[cam_time_mask3]
    
    broadband_alb_cam_time3 = np.zeros_like(cam_time3)
    broadband_alb_cam_time3[:] = np.nan
    alb_cam_time3 = np.zeros((len(cam_time3), alb_selected_all.shape[1]))
    alb_cam_time3[:] = np.nan
    for i, t in enumerate(cam_time3):
        time_diff = np.abs(time_selected_all - t)
        if np.min(time_diff) <= 1./60/60:  # within 1s 
            closest_idx = np.argmin(time_diff)
            broadband_alb_cam_time3[i] = broadband_alb_selected_all[closest_idx]
            alb_cam_time3[i, :] = alb_selected_all[closest_idx, :]

    
    # alb_ext_wvl, alb_ext = alb_extention(alb_wvl, alb_selected_all_avg, clear_sky=True)
    # plt.close('all')
    # plt.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km')
    # plt.plot(alb_ext_wvl, alb_ext, label='Extended')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Surface Albedo')
    # plt.title('Surface Albedo Extension Check, Aug 1st')
    # plt.legend()
    # plt.show()
    # sys.exit()    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km', color='b')
    ax.fill_between(alb_wvl, alb_selected_all_avg-alb_selected_all_std, alb_selected_all_avg+alb_selected_all_std, 
                    color='b', alpha=0.1)
    ax.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km', color='r')
    ax.fill_between(alb_wvl, alb_selected_all_avg-alb_selected_all_std, alb_selected_all_avg+alb_selected_all_std, 
                    color='r', alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_2_avg.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lat_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_2_broadband_lat.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lon_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_2_broadband_lon.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    l1 = ax.scatter(time_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    l2 = ax2.scatter(cam_time3, cam_ice_fraction3, label='Camera Ice Fraction', c='r', s=5)
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
    lns = [l1, l2]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_2_broadband_time.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    l1 = ax.plot(time_selected_all, broadband_alb_selected_all, c='skyblue', label='Broadband Albedo', alpha=0.75, linewidth=2)
    l2 = ax2.plot(cam_time3, cam_ice_fraction3, c='coral', label='Camera Ice Fraction', alpha=0.75, linewidth=1.5)
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
    lns = [l1[0], l2[0]]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_2_broadband_time_line.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cam_ice_fraction3, broadband_alb_cam_time3, s=10, c='k')
    ax.set_xlabel('Camera Ice Fraction', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title('Surface Albedo vs Camera Ice Fraction, Aug 1st', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_2_broadband_icefraction.png', bbox_inches='tight', dpi=150)
    plt.close(fig) 
    
    wvl_450nm_idx = np.argmin(np.abs(alb_wvl - 450))
    wvl_860nm_idx = np.argmin(np.abs(alb_wvl - 860))
    wvl_1240nm_idx = np.argmin(np.abs(alb_wvl - 1240))
    wvl_1630nm_idx = np.argmin(np.abs(alb_wvl - 1630))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cam_ice_fraction3, alb_cam_time3[:, wvl_450nm_idx], s=10, c='b', label='450nm')
    ax.scatter(cam_ice_fraction3, alb_cam_time3[:, wvl_860nm_idx], s=10, c='g', label='860nm')
    ax.scatter(cam_ice_fraction3, alb_cam_time3[:, wvl_1240nm_idx], s=10, c='r', label='1200nm')
    ax.scatter(cam_ice_fraction3, alb_cam_time3[:, wvl_1630nm_idx], s=10, c='m', label='1600nm')
    ax.set_xlabel('Camera Ice Fraction', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    ax.set_title('Surface Albedo at Different Wavelengths vs Camera Ice Fraction, Aug 1st', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_2_wvl_icefraction.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    wvl_slope = np.zeros_like(alb_wvl)
    wvl_slope[:] = np.nan
    wvl_r2 = np.zeros_like(alb_wvl)
    wvl_r2[:] = np.nan
    from scipy.stats import linregress
    from mpl_toolkits.axes_grid1 import make_axes_locatable    

    
    for i in range(len(alb_wvl)):
        # Perform linear regression
        mask = np.isfinite(alb_cam_time3[:, i]) & np.isfinite(cam_ice_fraction3)
        slope, intercept, r_value, p_value, std_err = linregress(cam_ice_fraction3[mask], alb_cam_time3[:, i][mask])
        wvl_r2[i] = r_value**2
        wvl_slope[i] = slope
        # print(f'Wavelength: {alb_wvl[i]:.1f} nm, Slope: {slope:.4f}, R²: {r_value**2:.4f}')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()
    c1 = ax.scatter(alb_wvl, wvl_slope, c=wvl_r2, s=10, cmap='jet', vmin=0, vmax=1)
    ax2.plot(alb_wvl, alb_selected_all_avg, color='k', alpha=0.5)
    
    # Create an AxesDivider for ax1
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.3) # Create a new axes for the colorbar
    # cbar = fig.colorbar(c1, ax=ax, pad=0.55, orientation='horizontal')
    # cbar.set_label('R²', fontsize=12)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Slope', fontsize=14)
    ax2.set_ylabel('Avg Surface Albedo', fontsize=14)
    ax2.legend(['Avg Surface Albedo'], fontsize=10)
    ax.set_xlim(350, 2000)
    ax.tick_params(labelsize=12)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_title('Correlation between Surface Albedo and Camera Ice Fraction vs Wavelength, Aug 1st', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_2_wvl_icefraction_correlation.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    ### Plot flight tracks
    
    # cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    cartopy_proj = ccrs.NorthPolarStereo(central_longitude=np.nanmean(lon_selected_all))
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': cartopy_proj})

    # Set the extent for the main axes
    lon_min = np.nanmin(lon_selected_all)
    lon_max = np.nanmax(lon_selected_all)
    lat_min = np.nanmin(lat_selected_all)
    lat_max = np.nanmax(lat_selected_all)
    
    print("ori lon extent:", lon_min, lon_max)
    print("ori lat extent:", lat_min, lat_max)
    
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 3.
    lat_buffer = (lat_max - lat_min) * 1.
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    print("lon extent:", lon_min, lon_max)
    print("lat extent:", lat_min, lat_max)
    
    # features
    ax.coastlines(linewidth=0.5, color='black')
    ax.add_feature(
        cfeature.LAND.with_scale('50m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax.add_feature(
        cfeature.OCEAN.with_scale('50m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 15.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 5.0))
    g1.top_labels = False


    ax.scatter(lon_selected_all, lat_selected_all, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=5, zorder=3)
    
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')

    plt.savefig(f'{fig_dir}/arcsix_flight_paths_0801_clear_2.png', dpi=300, bbox_inches='tight')
    

    
    # 8/1 clear_atm_corr + 8/2 clear_atm_corr_12
    date_select = '20240802'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select_1 = 'clear_atm_corr'
    case_tag_mask_1 = np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_select_2 = 'clear_atm_corr_2'
    case_tag_mask_2 = np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_mask = case_tag_mask_1 | case_tag_mask_2
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
        
    # read cam_ice_fraction_20240801_134836_140900.nc
    with Dataset(f'cam_ice_fraction_20240802_143112_150748.nc', 'r') as nc:
        cam_time1 = nc.variables['tmhr'][:]  
        cam_ice_fraction1 = nc.variables['ice_fraction'][:]  # shape (time)
        
    with Dataset(f'cam_ice_fraction_20240802_151312_163936.nc', 'r') as nc:
        cam_time2 = nc.variables['tmhr'][:]  
        cam_ice_fraction2 = nc.variables['ice_fraction'][:]  # shape (time)
    
    time_mask = (time_selected_all >= 14.557) & (time_selected_all <= 15.100)
    time_selected_all = time_selected_all[time_mask]
    lon_selected_all = lon_selected_all[time_mask]
    lat_selected_all = lat_selected_all[time_mask]
    alt_selected_all = alt_selected_all[time_mask]
    alb_selected_all = alb_selected_all[time_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[time_mask]
    

    alb_selected_all_avg = np.nanmean(alb_selected_all, axis=0)
    alb_selected_all_std = np.nanstd(alb_selected_all, axis=0)
    alt_selected_all_avg = np.nanmean(alt_selected_all)
    
    cam_time1 = cam_time1 + 0/60/60 
    cam_time_mask1 = (cam_time1 >= 14.557) & (cam_time1 <= 15.100)
    cam_time1 = cam_time1[cam_time_mask1]
    cam_ice_fraction1 = cam_ice_fraction1[cam_time_mask1]
    
    broadband_alb_cam_time1 = np.zeros_like(cam_time1)
    broadband_alb_cam_time1[:] = np.nan
    alb_cam_time1 = np.zeros((len(cam_time1), alb_selected_all.shape[1]))
    alb_cam_time1[:] = np.nan
    for i, t in enumerate(cam_time1):
        time_diff = np.abs(time_selected_all - t)
        if np.min(time_diff) <= 1/60/60:  # within 1s 
            closest_idx = np.argmin(time_diff)
            broadband_alb_cam_time1[i] = broadband_alb_selected_all[closest_idx]
            alb_cam_time1[i, :] = alb_selected_all[closest_idx, :]
    
    
    
    # alb_ext_wvl, alb_ext = alb_extention(alb_wvl, alb_selected_all_avg, clear_sky=True)
    # plt.close('all')
    # plt.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km')
    # plt.plot(alb_ext_wvl, alb_ext, label='Extended')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Surface Albedo')
    # plt.title('Surface Albedo Extension Check, Aug 1st')
    # plt.legend()
    # plt.show()
    # sys.exit()    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km', color='b')
    ax.fill_between(alb_wvl, alb_selected_all_avg-alb_selected_all_std, alb_selected_all_avg+alb_selected_all_std, 
                    color='b', alpha=0.1)
    ax.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km', color='r')
    ax.fill_between(alb_wvl, alb_selected_all_avg-alb_selected_all_std, alb_selected_all_avg+alb_selected_all_std, 
                    color='r', alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 2nd', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_1_avg.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lat_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 2nd', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_1_broadband_lat.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lon_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 2nd', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_1_broadband_lon.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    l1 = ax.scatter(time_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    l2 = ax2.scatter(cam_time1, cam_ice_fraction1, label='Camera Ice Fraction', c='r', s=5)
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
    lns = [l1, l2]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 2nd', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_1_broadband_time.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    ax.plot(time_selected_all, broadband_alb_selected_all, c='b', label='Broadband Albedo',)
    ax2.plot(cam_time1, cam_ice_fraction1, c='r', label='Camera Ice Fraction',)
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
    ax.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 2nd', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_1_broadband_time_line.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cam_ice_fraction1, broadband_alb_cam_time1, s=10, c='k')
    ax.set_xlabel('Camera Ice Fraction', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title('Surface Albedo vs Camera Ice Fraction, Aug 2nd', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_1_broadband_icefraction.png', bbox_inches='tight', dpi=150)
    plt.close(fig) 
    
    wvl_450nm_idx = np.argmin(np.abs(alb_wvl - 450))
    wvl_860nm_idx = np.argmin(np.abs(alb_wvl - 860))
    wvl_1200nm_idx = np.argmin(np.abs(alb_wvl - 1200))
    wvl_1600nm_idx = np.argmin(np.abs(alb_wvl - 1600))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cam_ice_fraction1, alb_cam_time1[:, wvl_450nm_idx], s=10, c='b', label='450nm')
    ax.scatter(cam_ice_fraction1, alb_cam_time1[:, wvl_860nm_idx], s=10, c='g', label='860nm')
    ax.scatter(cam_ice_fraction1, alb_cam_time1[:, wvl_1200nm_idx], s=10, c='r', label='1240nm')
    ax.scatter(cam_ice_fraction1, alb_cam_time1[:, wvl_1600nm_idx], s=10, c='m', label='1630nm')
    ax.set_xlabel('Camera Ice Fraction', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    ax.set_title('Surface Albedo at Different Wavelengths vs Camera Ice Fraction, Aug 2nd', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_1_wvl_icefraction.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    wvl_slope = np.zeros_like(alb_wvl)
    wvl_slope[:] = np.nan
    wvl_r2 = np.zeros_like(alb_wvl)
    wvl_r2[:] = np.nan
    from scipy.stats import linregress
    from mpl_toolkits.axes_grid1 import make_axes_locatable    

    
    for i in range(len(alb_wvl)):
        # Perform linear regression
        mask = np.isfinite(alb_cam_time1[:, i]) & np.isfinite(cam_ice_fraction1)
        slope, intercept, r_value, p_value, std_err = linregress(cam_ice_fraction1[mask], alb_cam_time1[:, i][mask])
        wvl_r2[i] = r_value**2
        wvl_slope[i] = slope
        # print(f'Wavelength: {alb_wvl[i]:.1f} nm, Slope: {slope:.4f}, R²: {r_value**2:.4f}')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()
    c1 = ax.scatter(alb_wvl, wvl_slope, c=wvl_r2, s=10, cmap='jet', vmin=0, vmax=1)
    ax2.plot(alb_wvl, alb_selected_all_avg, color='k', alpha=0.5)
    
    # Create an AxesDivider for ax1
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.3) # Create a new axes for the colorbar
    # cbar = fig.colorbar(c1, ax=ax, pad=0.55, orientation='horizontal')
    # cbar.set_label('R²', fontsize=12)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Slope', fontsize=14)
    ax2.set_ylabel('Avg Surface Albedo', fontsize=14)
    ax2.legend(['Avg Surface Albedo'], fontsize=10)
    ax.set_xlim(350, 2000)
    ax.tick_params(labelsize=12)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_title('Correlation between Surface Albedo and Camera Ice Fraction vs Wavelength, Aug 2nd', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_1_wvl_icefraction_correlation.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    
    # 8/1 clear_atm_corr + 8/2 clear_atm_corr_12
    date_select = '20240802'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select_1 = 'clear_atm_corr'
    case_tag_mask_1 = np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_select_2 = 'clear_atm_corr_2'
    case_tag_mask_2 = np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_mask = case_tag_mask_1 | case_tag_mask_2
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
        
    with Dataset(f'cam_ice_fraction_20240802_151312_163936.nc', 'r') as nc:
        cam_time2 = nc.variables['tmhr'][:]  
        cam_ice_fraction2 = nc.variables['ice_fraction'][:]  # shape (time)
    
    time_mask = (time_selected_all >= 15.244) & (time_selected_all <= 16.635)
    time_selected_all = time_selected_all[time_mask]
    lon_selected_all = lon_selected_all[time_mask]
    lat_selected_all = lat_selected_all[time_mask]
    alt_selected_all = alt_selected_all[time_mask]
    alb_selected_all = alb_selected_all[time_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[time_mask]
    

    alb_selected_all_avg = np.nanmean(alb_selected_all, axis=0)
    alb_selected_all_std = np.nanstd(alb_selected_all, axis=0)
    alt_selected_all_avg = np.nanmean(alt_selected_all)
    
    cam_time2 = cam_time2 + 0.1/60/60 
    cam_time_mask2= (cam_time2 >= 15.244) & (cam_time2 <= 16.635)
    cam_time2 = cam_time2[cam_time_mask2]
    cam_ice_fraction2 = cam_ice_fraction2[cam_time_mask2]
    
    broadband_alb_cam_time2 = np.zeros_like(cam_time2)
    broadband_alb_cam_time2[:] = np.nan
    alb_cam_time2 = np.zeros((len(cam_time2), alb_selected_all.shape[1]))
    alb_cam_time2[:] = np.nan
    for i, t in enumerate(cam_time2):
        time_diff = np.abs(time_selected_all - t)
        if np.min(time_diff) <= 1/60/60:  # within 1s 
            closest_idx = np.argmin(time_diff)
            broadband_alb_cam_time2[i] = broadband_alb_selected_all[closest_idx]
            alb_cam_time2[i, :] = alb_selected_all[closest_idx, :]
    
    
    
    # alb_ext_wvl, alb_ext = alb_extention(alb_wvl, alb_selected_all_avg, clear_sky=True)
    # plt.close('all')
    # plt.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km')
    # plt.plot(alb_ext_wvl, alb_ext, label='Extended')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Surface Albedo')
    # plt.title('Surface Albedo Extension Check, Aug 1st')
    # plt.legend()
    # plt.show()
    # sys.exit()    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km', color='b')
    ax.fill_between(alb_wvl, alb_selected_all_avg-alb_selected_all_std, alb_selected_all_avg+alb_selected_all_std, 
                    color='b', alpha=0.1)
    ax.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km', color='r')
    ax.fill_between(alb_wvl, alb_selected_all_avg-alb_selected_all_std, alb_selected_all_avg+alb_selected_all_std, 
                    color='r', alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 2nd', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_2_avg.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lat_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 2nd', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_2_broadband_lat.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lon_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 2nd', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_2_broadband_lon.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    l1 = ax.scatter(time_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    l2 = ax2.scatter(cam_time2, cam_ice_fraction2, label='Camera Ice Fraction', c='r', s=5)
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
    lns = [l1, l2]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 2nd', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_2_broadband_time.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    ax.plot(time_selected_all, broadband_alb_selected_all, c='b', label='Broadband Albedo',)
    ax2.plot(cam_time2, cam_ice_fraction2, c='r', label='Camera Ice Fraction',)
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
    ax.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 2nd', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_2_broadband_time_line.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cam_ice_fraction2, broadband_alb_cam_time2, s=10, c='k')
    ax.set_xlabel('Camera Ice Fraction', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title('Surface Albedo vs Camera Ice Fraction, Aug 2nd', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_2_broadband_icefraction.png', bbox_inches='tight', dpi=150)
    plt.close(fig) 
    
    wvl_450nm_idx = np.argmin(np.abs(alb_wvl - 450))
    wvl_860nm_idx = np.argmin(np.abs(alb_wvl - 860))
    wvl_1200nm_idx = np.argmin(np.abs(alb_wvl - 1200))
    wvl_1600nm_idx = np.argmin(np.abs(alb_wvl - 1600))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cam_ice_fraction2, alb_cam_time2[:, wvl_450nm_idx], s=10, c='b', label='450nm')
    ax.scatter(cam_ice_fraction2, alb_cam_time2[:, wvl_860nm_idx], s=10, c='g', label='860nm')
    ax.scatter(cam_ice_fraction2, alb_cam_time2[:, wvl_1200nm_idx], s=10, c='r', label='1200nm')
    ax.scatter(cam_ice_fraction2, alb_cam_time2[:, wvl_1600nm_idx], s=10, c='m', label='1600nm')
    ax.set_xlabel('Camera Ice Fraction', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    ax.set_title('Surface Albedo at Different Wavelengths vs Camera Ice Fraction, Aug 2nd', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_2_wvl_icefraction.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    wvl_slope = np.zeros_like(alb_wvl)
    wvl_slope[:] = np.nan
    wvl_r2 = np.zeros_like(alb_wvl)
    wvl_r2[:] = np.nan
    from scipy.stats import linregress
    from mpl_toolkits.axes_grid1 import make_axes_locatable    

    
    for i in range(len(alb_wvl)):
        # Perform linear regression
        mask = np.isfinite(alb_cam_time2[:, i]) & np.isfinite(cam_ice_fraction2)
        slope, intercept, r_value, p_value, std_err = linregress(cam_ice_fraction2[mask], alb_cam_time2[:, i][mask])
        wvl_r2[i] = r_value**2
        wvl_slope[i] = slope
        # print(f'Wavelength: {alb_wvl[i]:.1f} nm, Slope: {slope:.4f}, R²: {r_value**2:.4f}')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()
    c1 = ax.scatter(alb_wvl, wvl_slope, c=wvl_r2, s=10, cmap='jet', vmin=0, vmax=1)
    ax2.plot(alb_wvl, alb_selected_all_avg, color='k', alpha=0.5)
    
    # Create an AxesDivider for ax1
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.3) # Create a new axes for the colorbar
    # cbar = fig.colorbar(c1, ax=ax, pad=0.55, orientation='horizontal')
    # cbar.set_label('R²', fontsize=12)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Slope', fontsize=14)
    ax2.set_ylabel('Avg Surface Albedo', fontsize=14)
    ax2.legend(['Avg Surface Albedo'], fontsize=10)
    ax.set_xlim(350, 2000)
    ax.tick_params(labelsize=12)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_title('Correlation between Surface Albedo and Camera Ice Fraction vs Wavelength, Aug 2nd', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0802_clear_2_wvl_icefraction_correlation.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    alpha = 0.7
    ax.scatter(cam_ice_fraction, broadband_alb_cam_time, s=10, c='k', label='0801 13.84-14.12 (0.1km)', alpha=alpha, zorder=3)
    ax.scatter(cam_ice_fraction3, broadband_alb_cam_time3, s=10, c='g', label='0801 14.76-15.23 (0.6km)', alpha=alpha, zorder=2)
    ax.scatter(cam_ice_fraction1, broadband_alb_cam_time1, s=10, c='b', label='0802 14.56-15.10 (0.1km)', alpha=alpha, zorder=1)
    ax.scatter(cam_ice_fraction2, broadband_alb_cam_time2, s=10, c='r', label='0802 15.24-16.63 (1.0km)', alpha=alpha, zorder=0)
    ax.set_xlabel('Camera Ice Fraction', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10)
    ax.set_title('Surface Albedo vs Camera Ice Fraction, Aug 2nd', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_0802_broadband_icefraction.png', bbox_inches='tight', dpi=150)
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