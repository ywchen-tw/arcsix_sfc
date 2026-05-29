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
import matplotlib.dates as mdates
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable    
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import bisect
import pandas as pd
import xarray as xr
from scipy import stats
from collections import defaultdict
import statsmodels.api as sm
import gc
from pyproj import Transformer
from util import *
from datetime import datetime, timedelta
# mpl.use('Agg')
from matplotlib import rcParams

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif" # Ensure sans-serif is used as the default family

_mission_      = 'arcsix'

if platform.system() == 'Darwin':
    _fdir_data_ = '/Volumes/argus/field/%s/processed' % _mission_
    _fdir_data_ = '../data/processed' 
    _fdir_general_ = '../data'
    _fdir_tmp_ = './tmp'
elif platform.system() == 'Linux':
    _fdir_data_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data/processed"
    _fdir_general_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data"
    _fdir_tmp_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/tmp"

_mission_      = 'arcsix'
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


def linear_regression_with_confidence(x, y, confidence=0.95):
    """
    Performs linear regression and calculates the confidence interval for the regression line.
    """
    # 1. Perform Linear Regression
    res = linregress(x, y)
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
    file = f'{_fdir_general_}/cam_icefrac_rad/ice_frac_all.pkl'
    with open(file, 'rb') as f:
        ice_frac_data = pickle.load(f)
    ice_frac_date = ice_frac_data['date']
    ice_frac_time = ice_frac_data['time']
    ice_frac_values = ice_frac_data['ice_frac']
    nad_hdrf_values = ice_frac_data['nad_hdrf']
    nad_rad_values  = ice_frac_data['nad_rad']

    # kt19 and sza are stored directly in combined_data; loaded per-leg below
    

    date_conditions = []
    broadband_alb_date_cond = []
    broadband_alb_date_cond_upper = []
    broadband_alb_date_cond_lower = []
    broadband_alb_date_cond_unc = []
    myi_ratio_date_cond = []
    myi_ratio_date_cond_std = []
    myi_ratio_date_cond_upper = []
    myi_ratio_date_cond_lower = []
    fyi_ratio_date_cond = []
    fyi_ratio_date_cond_std = []
    yi_ratio_date_cond = []
    yi_ratio_date_cond_std = []
    fyi_yi_ratio_date_cond = []
    fyi_yi_ratio_date_cond_std = []
    myi_fyi_ratio_date_cond = []
    myi_fyi_ratio_date_cond_upper = []
    myi_fyi_ratio_date_cond_lower = []
    ice_age_date_cond = []
    ice_age_date_cond_upper = []
    ice_age_date_cond_lower = []
    ice_age_over1_date_cond = []
    ice_age_over2_date_cond = []
    kt19_date_cond = []
    kt19_date_cond_upper = []
    kt19_date_cond_lower = []
    kt19_high_hdrf_date_cond = []
    kt19_high_hdrf_date_cond_upper = []
    kt19_high_hdrf_date_cond_lower = []
    brt19h, brt37h, brt37v = [], [], []
    brt19h_upper, brt37h_upper, brt37v_upper = [], [], []
    brt19h_lower, brt37h_lower, brt37v_lower = [], [], []
    pi_37, pi_37_upper, pi_37_lower = [], [], []
    grain_size_date_cond, grain_size_date_cond_upper, grain_size_date_cond_lower = [], [], []
    grain_size_ratio_date_cond, grain_size_ratio_date_cond_upper, grain_size_ratio_date_cond_lower = [], [], []
    grain_size_ratio_1020_865_date_cond, grain_size_ratio_1020_865_date_cond_upper, grain_size_ratio_1020_865_date_cond_lower = [], [], []
    grain_size_ratio_1650_1020_date_cond, grain_size_ratio_1650_1020_date_cond_upper, grain_size_ratio_1650_1020_date_cond_lower = [], [], []
    grain_size_1240_date_cond, grain_size_1240_date_cond_upper, grain_size_1240_date_cond_lower = [], [], []
    gs_alb_1700_date_cond, gs_alb_1700_date_cond_upper, gs_alb_1700_date_cond_lower = [], [], []
    gs_alb_1240_date_cond, gs_alb_1240_date_cond_upper, gs_alb_1240_date_cond_lower = [], [], []
    alb_1700_date_cond, alb_1700_date_cond_upper, alb_1700_date_cond_lower = [], [], []
    alb_1240_date_cond, alb_1240_date_cond_upper, alb_1240_date_cond_lower = [], [], []

    # ART grain-size constants (mirrors ssfr_atm_corr_combined.py)
    # Spring (fresh/columnar/dendritic): A = 5.8 (randomly oriented hexagonal plates/columns)
    # Summer (metamorphosed, rounded by melt): A = 4.53 (Kokhanovsky & Zege 2004, spherical grains)
    _ART_A_SPRING_ = 5.8
    _ART_A_SUMMER_ = 4.53
    _ART_CHI_T_1700_ = np.array([213.0, 233.0, 253.0, 266.0, 273.0])         # K
    _ART_CHI_K_1700_ = np.array([5.8e-4, 7.8e-4, 1.04e-3, 1.38e-3, 1.61e-3]) # imaginary k at 1700 nm
    _ART_CHI_T_1240_ = np.array([213.0, 233.0, 253.0, 266.0, 273.0])          # K
    _ART_CHI_K_1240_ = np.array([4.5e-6, 5.5e-6, 6.4e-6, 7.5e-6, 8.2e-6])    # imaginary k at 1240 nm

    def _art_grain_size(alb_scalar, lam_nm, chi, K0, A):
        """ART grain size from a scalar albedo. Returns grain radius in metres."""
        alb_c = np.clip(alb_scalar, 1e-6, 1.0)
        lam   = lam_nm * 1e-9
        prefactor = lam / (4.0 * np.pi * chi)
        return prefactor * (np.log(alb_c) / (A * K0)) ** 2

    # Ratio-method AART constants — matches ssfr_atm_corr_combined.py
    _GS_RATIO_1280_CHI1_ = 1.39e-5   # imaginary k at 1280 nm
    _GS_RATIO_1280_CHI2_ = 2.89e-7   # imaginary k at 1100 nm
    _GS_RATIO_1650_CHI1_ = 1.2e-3    # imaginary k at 1650 nm
    _GS_RATIO_1650_CHI2_ = 2.0e-7    # imaginary k at 1020 nm
    _GS_RATIO_1020_CHI1_ = 2.0e-7    # imaginary k at 1020 nm
    _GS_RATIO_1020_CHI2_ = 2.4e-9    # imaginary k at 865 nm

    def _art_grain_size_ratio(alb1, alb2, wvl1_nm, chi1, wvl2_nm, chi2, K0, A):
        """ART ratio-method grain size from two scalar albedos. Returns grain radius in metres."""
        lam1 = wvl1_nm * 1e-9;  lam2 = wvl2_nm * 1e-9
        D    = np.sqrt(chi1 / lam1) - np.sqrt(chi2 / lam2)
        ln_R = np.log(np.clip(alb1, 1e-6, 1.0) / np.clip(alb2, 1e-6, 1.0))
        return (1.0 / (4.0 * np.pi)) * (ln_R / (A * K0 * D)) ** 2


    
    def _ice_weighted_mean_std(values, weights):
        """Ice-ratio weighted mean and standard deviation."""
        total_w = np.nansum(weights) + 1e-8
        mean = np.nansum(values * weights) / total_w
        std  = np.sqrt(np.nansum(((values - mean) ** 2) * weights) / total_w)
        return mean, std

    def _append_stats(lst, lst_lo, lst_hi, arr, lo=10, hi=90):
        """Append median, lo-th and hi-th percentile of finite values."""
        v = arr[np.isfinite(arr)]
        lst.append(   np.median(v)       if len(v) > 0 else np.nan)
        lst_lo.append(np.percentile(v, lo) if len(v) > 0 else np.nan)
        lst_hi.append(np.percentile(v, hi) if len(v) > 0 else np.nan)

    # best time-shift results: {(date_key, case_tag): {'shift_s': int, 'r': float}}
    kt19_hdrf_best_shift = {}

    for date_key in ice_frac_time_offset.keys():
        # camera ice fraction for this date (shared by clear/cloudy)
        cam_time_date      = ice_frac_time[ice_frac_date == int(date_key)]
        cam_ice_frac_date  = ice_frac_values[ice_frac_date == int(date_key)]

        sfx = 'summer' if date_key > '20240630' else 'spring'  # season suffix

        for case_tag in ['clear', 'cloudy']:
            date_mask     = combined_data[f'dates_{sfx}_all'] == int(date_key)
            case_tag_mask = np.array([case_tag in ct for ct in combined_data[f'case_tags_{sfx}_all']])
            final_mask    = date_mask & case_tag_mask
            if not np.any(final_mask):
                print(f"No data for date {date_key} and case tag {case_tag}. Skipping.")
                continue
            alb_wvl              = combined_data[f'wvl_{sfx}']
            lon_selected_all     = combined_data[f'lon_all_{sfx}'][final_mask]
            lat_selected_all     = combined_data[f'lat_all_{sfx}'][final_mask]
            alt_selected_all     = combined_data[f'alt_all_{sfx}'][final_mask]
            time_selected_all    = combined_data[f'time_{sfx}_all'][final_mask]
            alb_selected_all     = combined_data[f'alb_iter2_all_{sfx}'][final_mask, :]
            broadband_alb_selected_all = combined_data[f'broadband_alb_iter2_all_filter_{sfx}'][final_mask]
            myi_selected_all_ori = combined_data[f'myi_ratio_{sfx}_all'][final_mask]
            fyi_selected_all_ori = combined_data[f'fyi_ratio_{sfx}_all'][final_mask]
            yi_selected_all_ori  = combined_data[f'yi_ratio_{sfx}_all'][final_mask]
            ice_ratio_selected_all = combined_data[f'ice_ratio_{sfx}_all'][final_mask]
            ice_age_selected_all = combined_data[f'ice_age_{sfx}_all'][final_mask]
            kt19_selected_all    = combined_data[f'kt19_sfc_T_{sfx}_all'][final_mask]
            n_pts = int(np.sum(final_mask))
            nad_hdrf_selected_all = combined_data[f'nad_hdrf_{sfx}_all'][final_mask] if f'nad_hdrf_{sfx}_all' in combined_data else np.full(n_pts, np.nan)
            nad_rad_selected_all  = combined_data[f'nad_rad_{sfx}_all'][final_mask]  if f'nad_rad_{sfx}_all'  in combined_data else np.full(n_pts, np.nan)
            sza_selected_all     = combined_data[f'sza_{sfx}_all'][final_mask]
            brt19h_selected_all  = combined_data[f'brt19h_{sfx}_all'][final_mask]
            brt37h_selected_all  = combined_data[f'brt37h_{sfx}_all'][final_mask]
            brt37v_selected_all  = combined_data[f'brt37v_{sfx}_all'][final_mask]
            grain_size_selected_all = combined_data[f'grain_size_{sfx}_all'][final_mask]
            grain_size_ratio_selected_all = combined_data[f'grain_size_ratio_{sfx}_all'][final_mask]*1e6
            grain_size_ratio_1020_865_selected_all = combined_data[f'grain_size_ratio_1020_865_{sfx}_all'][final_mask] * 1e6
            grain_size_ratio_1650_1020_selected_all = combined_data[f'grain_size_ratio_1650_1020_{sfx}_all'][final_mask] * 1e6
            grain_size_1240_selected_all = combined_data[f'grain_size_1240_{sfx}_all'][final_mask] * 1e6

            ice_total = myi_selected_all_ori + fyi_selected_all_ori + yi_selected_all_ori
            myi_selected_all     = myi_selected_all_ori / (ice_total + 1e-8) * 100  # % of ice total
            fyi_selected_all     = fyi_selected_all_ori / (ice_total + 1e-8) * 100
            yi_selected_all      = yi_selected_all_ori  / (ice_total + 1e-8) * 100
            myi_fyi_selected_all = (myi_selected_all_ori + fyi_selected_all_ori) / (ice_total + 1e-8) * 100

            alt_mask = (alt_selected_all >= 0) & (alt_selected_all <= 1.6)
            if not np.any(alt_mask):
                print(f"No data with altitude < 1.6 km for date {date_key} and case tag {case_tag}. Skipping.")
                continue
            lon_selected_all       = lon_selected_all[alt_mask]
            lat_selected_all       = lat_selected_all[alt_mask]
            alt_selected_all       = alt_selected_all[alt_mask]
            alb_selected_all       = alb_selected_all[alt_mask, :]
            broadband_alb_selected_all = broadband_alb_selected_all[alt_mask]
            time_selected_all      = time_selected_all[alt_mask]
            ice_ratio_selected_all = ice_ratio_selected_all[alt_mask]
            myi_selected_all       = myi_selected_all[alt_mask]
            fyi_selected_all       = fyi_selected_all[alt_mask]
            yi_selected_all        = yi_selected_all[alt_mask]
            ice_age_selected_all   = ice_age_selected_all[alt_mask]
            myi_fyi_selected_all   = myi_fyi_selected_all[alt_mask]
            kt19_selected_all      = kt19_selected_all[alt_mask]
            nad_hdrf_selected_all  = nad_hdrf_selected_all[alt_mask]
            nad_rad_selected_all   = nad_rad_selected_all[alt_mask]
            brt19h_selected_all    = brt19h_selected_all[alt_mask]
            brt37h_selected_all    = brt37h_selected_all[alt_mask]
            brt37v_selected_all    = brt37v_selected_all[alt_mask]
            grain_size_selected_all = grain_size_selected_all[alt_mask] * 1e6  # convert to microns
            grain_size_ratio_selected_all = grain_size_ratio_selected_all[alt_mask]
            grain_size_ratio_1020_865_selected_all = grain_size_ratio_1020_865_selected_all[alt_mask]
            grain_size_ratio_1650_1020_selected_all = grain_size_ratio_1650_1020_selected_all[alt_mask]
            grain_size_1240_selected_all = grain_size_1240_selected_all[alt_mask]
            sza_selected_all             = sza_selected_all[alt_mask]

            # Interpolate spectral albedo to 1700 nm and 1240 nm
            def _interp_alb_to_wvl(alb_arr, wvl_arr, target_nm):
                idx_hi = np.clip(np.searchsorted(wvl_arr, target_nm), 1, len(wvl_arr) - 1)
                t = (target_nm - wvl_arr[idx_hi - 1]) / (wvl_arr[idx_hi] - wvl_arr[idx_hi - 1])
                return np.clip(alb_arr[:, idx_hi - 1] * (1.0 - t) + alb_arr[:, idx_hi] * t, 1e-6, 1.0)
            alb_1700_selected_all = _interp_alb_to_wvl(alb_selected_all, alb_wvl, 1700.0)
            alb_1240_selected_all = _interp_alb_to_wvl(alb_selected_all, alb_wvl, 1240.0)
            alb_1280_selected_all = _interp_alb_to_wvl(alb_selected_all, alb_wvl, 1280.0)
            alb_1100_selected_all = _interp_alb_to_wvl(alb_selected_all, alb_wvl, 1100.0)
            alb_1020_selected_all = _interp_alb_to_wvl(alb_selected_all, alb_wvl, 1020.0)
            alb_865_selected_all  = _interp_alb_to_wvl(alb_selected_all, alb_wvl,  865.0)
            alb_1650_selected_all = _interp_alb_to_wvl(alb_selected_all, alb_wvl, 1650.0)

            print(f"Date: {date_key}, Case: {case_tag}, Ice Age range: "
                  f"{np.nanmin(ice_age_selected_all):.2f} – {np.nanmax(ice_age_selected_all):.2f} yr, "
                  f"Median: {np.nanmedian(ice_age_selected_all):.2f} yr")

            # apply timing offset and clip camera time to flight window
            cam_time = cam_time_date + ice_frac_time_offset[date_key]
            cam_mask = (cam_time >= time_selected_all.min()) & (cam_time <= time_selected_all.max())
            cam_time         = cam_time[cam_mask]
            cam_ice_fraction = cam_ice_frac_date[cam_mask]
            
            
            # Vectorized nearest-neighbor time matching via searchsorted
            sort_order  = np.argsort(time_selected_all)
            t_sorted    = time_selected_all[sort_order]
            ins         = np.searchsorted(t_sorted, cam_time)
            idx_lo      = np.clip(ins - 1, 0, len(t_sorted) - 1)
            idx_hi      = np.clip(ins,     0, len(t_sorted) - 1)
            diff_lo     = np.abs(t_sorted[idx_lo] - cam_time)
            diff_hi     = np.abs(t_sorted[idx_hi] - cam_time)
            closest_idx = sort_order[np.where(diff_lo <= diff_hi, idx_lo, idx_hi)]
            within_1s   = np.minimum(diff_lo, diff_hi) <= (1 / 3600)
            m = within_1s
            broadband_alb_cam_time = np.where(m, broadband_alb_selected_all[closest_idx], np.nan)
            myi_cam_time           = np.where(m, myi_selected_all[closest_idx],            np.nan)
            fyi_cam_time           = np.where(m, fyi_selected_all[closest_idx],            np.nan)
            yi_cam_time            = np.where(m, yi_selected_all[closest_idx],             np.nan)
            myi_fyi_cam_time       = np.where(m, myi_fyi_selected_all[closest_idx],        np.nan)
            ice_ratio_cam_time     = np.where(m, ice_ratio_selected_all[closest_idx],      np.nan)
            ice_age_cam_time       = np.where(m, ice_age_selected_all[closest_idx],        np.nan)
            kt19_cam_time          = np.where(m, kt19_selected_all[closest_idx],           np.nan)
            nad_hdrf_cam_time      = np.where(m, nad_hdrf_selected_all[closest_idx],       np.nan)
            nad_rad_cam_time       = np.where(m, nad_rad_selected_all[closest_idx],        np.nan)
            brt19h_cam_time        = np.where(m, brt19h_selected_all[closest_idx],          np.nan)
            brt37h_cam_time        = np.where(m, brt37h_selected_all[closest_idx],          np.nan)
            brt37v_cam_time        = np.where(m, brt37v_selected_all[closest_idx],          np.nan)
            grain_size_cam_time       = np.where(m, grain_size_selected_all[closest_idx],       np.nan)
            grain_size_ratio_cam_time = np.where(m, grain_size_ratio_selected_all[closest_idx], np.nan)
            grain_size_ratio_1020_865_cam_time  = np.where(m, grain_size_ratio_1020_865_selected_all[closest_idx],  np.nan)
            grain_size_ratio_1650_1020_cam_time = np.where(m, grain_size_ratio_1650_1020_selected_all[closest_idx], np.nan)
            grain_size_1240_cam_time            = np.where(m, grain_size_1240_selected_all[closest_idx],            np.nan)
            sza_cam_time                        = np.where(m, sza_selected_all[closest_idx],                        np.nan)
            alb_1700_cam_time = np.where(m, alb_1700_selected_all[closest_idx], np.nan)
            alb_1240_cam_time = np.where(m, alb_1240_selected_all[closest_idx], np.nan)
            alb_1280_cam_time = np.where(m, alb_1280_selected_all[closest_idx], np.nan)
            alb_1100_cam_time = np.where(m, alb_1100_selected_all[closest_idx], np.nan)
            alb_1020_cam_time = np.where(m, alb_1020_selected_all[closest_idx], np.nan)
            alb_865_cam_time  = np.where(m, alb_865_selected_all[closest_idx],  np.nan)
            alb_1650_cam_time = np.where(m, alb_1650_selected_all[closest_idx], np.nan)
                    
            
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
        
            cc = axr.scatter(cam_ice_fraction, broadband_alb_cam_time, c=myi_cam_time, alpha=0.6, cmap='jet', vmin=0, vmax=100)
            cbar = fig.colorbar(cc, ax=axr)
            cbar.set_label('Multi-year sea ice ratio (%)', fontsize=12)
            # calculate slope and correlation coefficient
            valid_mask = ~np.isnan(cam_ice_fraction) & ~np.isnan(broadband_alb_cam_time)
            if np.sum(valid_mask) > 2:
                x_sorted, y_pred, lower_bound, upper_bound, res = linear_regression_with_confidence(
                    cam_ice_fraction[valid_mask], broadband_alb_cam_time[valid_mask], confidence=0.95)
                axr.plot(x_sorted, y_pred, color='orange', linestyle='--', label=r'Fit: y=%.2fx+%.2f\n$\mathrm{R^2}$=%.2f' %(res.slope, res.intercept, res.rvalue**2))
                intercept = res.intercept
                slope = res.slope
                axr.legend(fontsize=10)

                date_conditions.append(f'{date_key}_{case_tag}')
                broadband_alb_date_cond.append(intercept + slope * 1.0)  # ice fraction = 1.0
                
                
                X_fit = sm.add_constant(cam_ice_fraction[valid_mask])
                quantiles = [0.05, 0.5, 0.95]
                models = {}

                for q in quantiles:
                    # Initialize the model with (endog=y, exog=X_design)
                    mod = sm.QuantReg(broadband_alb_cam_time[valid_mask], X_fit)
                    res = mod.fit(q=q)
                    models[q] = res
                    
                y_fit_05 = models[0.05].predict(X_fit)
                y_fit_50 = models[0.5].predict(X_fit)
                y_fit_95 = models[0.95].predict(X_fit)
                
                # axr.plot(cam_ice_fraction[valid_mask], y_fit_50, color='orange', linestyle='--', label='Quantile Reg (50th)')
                axr.fill_between(cam_ice_fraction[valid_mask], y_fit_05, y_fit_95, color='orange', alpha=0.2, label='Quantile Regression (5th-95th)')
                axr.legend(fontsize=10)
                    
                # broadband_alb_at_1       = models[0.5].predict([1.0, 1.0])[0]
                broadband_alb_at_1       = intercept + slope * 1.0
                broadband_alb_at_1_upper = models[0.95].predict([1.0, 1.0])[0]
                broadband_alb_at_1_lower = models[0.05].predict([1.0, 1.0])[0]
                broadband_alb_date_cond_upper.append(broadband_alb_at_1_upper)
                broadband_alb_date_cond_lower.append(broadband_alb_at_1_lower)
                broadband_alb_date_cond_unc.append((broadband_alb_at_1_upper - broadband_alb_at_1_lower) / 2)

                # Extract valid samples: combine valid_mask with ratio cap (<=100) in one step
                # so all ratio arrays share the same length (safe for pairwise operations).
                ratio_valid = (valid_mask
                               & (myi_cam_time     <= 100) & (fyi_cam_time <= 100)
                               & (yi_cam_time       <= 100) & (ice_ratio_cam_time <= 100)
                               & (myi_fyi_cam_time  <= 100))
                myi_v   = myi_cam_time[ratio_valid]
                fyi_v   = fyi_cam_time[ratio_valid]
                yi_v    = yi_cam_time[ratio_valid]
                w_v     = ice_ratio_cam_time[ratio_valid]   # weights for ice-type fractions
                myi_fyi_v = myi_fyi_cam_time[ratio_valid]
                ice_age_v = ice_age_cam_time[valid_mask]
                kt19_v    = kt19_cam_time[valid_mask]
                brt19h_v      = brt19h_cam_time[valid_mask]
                brt37h_v      = brt37h_cam_time[valid_mask]
                brt37v_v      = brt37v_cam_time[valid_mask]
                grain_size_v       = grain_size_cam_time[valid_mask]
                grain_size_ratio_v         = grain_size_ratio_cam_time[valid_mask]
                grain_size_ratio_1020_865_v  = grain_size_ratio_1020_865_cam_time[valid_mask]
                grain_size_ratio_1650_1020_v = grain_size_ratio_1650_1020_cam_time[valid_mask]
                grain_size_1240_v            = grain_size_1240_cam_time[valid_mask]
                sza_v                        = sza_cam_time[valid_mask]
                alb_1700_v = alb_1700_cam_time[valid_mask]
                alb_1240_v = alb_1240_cam_time[valid_mask]
                alb_1280_v = alb_1280_cam_time[valid_mask]
                alb_1100_v = alb_1100_cam_time[valid_mask]
                alb_1020_v = alb_1020_cam_time[valid_mask]
                alb_865_v  = alb_865_cam_time[valid_mask]
                alb_1650_v = alb_1650_cam_time[valid_mask]

                # Ice-ratio weighted mean/std for ice type fractions
                for vals, mean_lst, std_lst in [
                    (myi_v,         myi_ratio_date_cond,    myi_ratio_date_cond_std),
                    (fyi_v,         fyi_ratio_date_cond,    fyi_ratio_date_cond_std),
                    (yi_v,          yi_ratio_date_cond,     yi_ratio_date_cond_std),
                    (fyi_v + yi_v,  fyi_yi_ratio_date_cond, fyi_yi_ratio_date_cond_std),
                ]:
                    mean_, std_ = _ice_weighted_mean_std(vals, w_v)
                    mean_lst.append(mean_)
                    std_lst.append(std_)
                v_myi = myi_v[np.isfinite(myi_v)]
                myi_ratio_date_cond_lower.append(np.percentile(v_myi, 10) if len(v_myi) > 0 else np.nan)
                myi_ratio_date_cond_upper.append(np.percentile(v_myi, 90) if len(v_myi) > 0 else np.nan)

                # Percentile stats (median, 10th, 90th) for remaining variables
                _append_stats(myi_fyi_ratio_date_cond, myi_fyi_ratio_date_cond_lower,
                              myi_fyi_ratio_date_cond_upper, myi_fyi_v)
                _append_stats(ice_age_date_cond, ice_age_date_cond_lower,
                              ice_age_date_cond_upper, ice_age_v)
                ice_age_over1_date_cond.append(np.nansum(ice_age_v >= 1) / len(ice_age_v))
                ice_age_over2_date_cond.append(np.nansum(ice_age_v >= 2) / len(ice_age_v))
                
                # plot kt19 hist
                plt.close('all')
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(kt19_v, bins=101, color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_xlabel('KT19 Surface Temperature (°C)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title(f'KT19 Surface Temperature Distribution\nDate: {date_key}, Case: {case_tag}', fontsize=13)
                plt.tight_layout()
                plt.savefig(f'{fig_dir}/kt19_hist_{date_key}_{case_tag}.png', dpi=300)
                
                # plot scatter
                plt.close('all')
                fig, ax = plt.subplots(figsize=(6, 4))
                sc = ax.scatter(cam_ice_fraction, kt19_cam_time, c='steelblue', alpha=0.6)
                ax.set_xlabel('Camera Ice Fraction', fontsize=12)
                ax.set_ylabel('KT19 Surface Temperature (°C)', fontsize=12)
                ax.set_title(f'KT19 Surface Temperature vs Camera Ice Fraction\nDate: {date_key}, Case: {case_tag}', fontsize=13)
                plt.tight_layout()
                plt.savefig(f'{fig_dir}/kt19_cam_time_vs_ice_fraction_scatter_{date_key}_{case_tag}.png', dpi=300)

                # plot KT-19 vs nadir radiance for this flight
                kt19_nad_rad_valid = np.isfinite(kt19_cam_time) & np.isfinite(nad_rad_cam_time)
                if np.sum(kt19_nad_rad_valid) > 2:
                    x_kr = kt19_cam_time[kt19_nad_rad_valid]
                    y_kr = nad_rad_cam_time[kt19_nad_rad_valid]
                    res_kr = linregress(x_kr, y_kr)
                    x_line = np.linspace(np.nanmin(x_kr) * 0.97, np.nanmax(x_kr) * 1.03, 100)
                    y_line = res_kr.intercept + res_kr.slope * x_line

                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(x_kr, y_kr, c='steelblue', alpha=0.6)
                    ax.plot(x_line, y_line, color='orange', linestyle='--',
                            label=r'$\mathrm{R^2}$=%.3f, p=%.2e'
                                  % (res_kr.rvalue**2, res_kr.pvalue))
                    ax.legend(fontsize=10)
                    ax.set_xlabel('KT19 Surface Temperature (°C)', fontsize=12)
                    ax.set_ylabel('Nadir Radiance', fontsize=12)
                    ax.set_title(f'KT19 Surface Temperature vs Nadir Radiance\nDate: {date_key}, Case: {case_tag}', fontsize=13)
                    plt.tight_layout()
                    plt.savefig(f'{fig_dir}/kt19_vs_nad_rad_scatter_{date_key}_{case_tag}.png', dpi=300)
                
                # plot kt19 time series ve nad_hdrf
                plt.close('all')
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax2 = ax1.twinx()
                ax1.plot(time_selected_all, nad_hdrf_selected_all, color='skyblue', label='NAD HDRF', alpha=0.75, linewidth=2)
                ax2.plot(time_selected_all, kt19_selected_all, color='coral', label='KT19 Surface Temperature', alpha=0.75, linewidth=1.5)
                ax1.set_xlabel('Time (UTC)', fontsize=14)
                ax1.set_ylabel('NAD HDRF', fontsize=14)
                ax2.set_ylabel('KT19 Surface Temperature (°C)', fontsize=14)
                lns = [ax1.lines[0], ax2.lines[0]]
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
                ax1.tick_params(labelsize=12)
                ax1.set_title('KT19 Surface Temperature and NAD HDRF Time Series', fontsize=13)
                plt.tight_layout()
                plt.savefig(f'{fig_dir}/kt19_time_series_{date_key}_{case_tag}.png', dpi=300)

                # --- find best time shift maximising KT19 vs NAD-HDRF Pearson correlation ---
                valid_both = ~np.isnan(nad_hdrf_selected_all) & ~np.isnan(kt19_selected_all)
                if np.sum(valid_both) > 10:
                    t_v      = time_selected_all[valid_both]
                    hdrf_v   = nad_hdrf_selected_all[valid_both]
                    kt19_v2  = kt19_selected_all[valid_both]

                    hdrf_interp = interp1d(t_v, hdrf_v, bounds_error=False, fill_value=np.nan)

                    shifts_s = np.arange(-3, 3.0001, 0.01)              # -5 … +5 s in 1-s steps
                    shifts_h = shifts_s / 3600.0

                    corrs = np.full(len(shifts_s), np.nan)
                    for si, dt in enumerate(shifts_h):
                        hdrf_shifted = hdrf_interp(t_v - dt)    # shift hdrf by dt hours
                        ok = ~np.isnan(hdrf_shifted)
                        if np.sum(ok) > 10:
                            corrs[si] = np.corrcoef(kt19_v2[ok], hdrf_shifted[ok])[0, 1]

                    best_idx     = np.nanargmin(np.array(corrs))   # high HDRF ↔ low KT19 → negative r
                    best_shift_s = float(shifts_s[best_idx])
                    best_corr    = corrs[best_idx]
                    kt19_hdrf_best_shift[(date_key, case_tag)] = {'shift_s': best_shift_s, 'r': float(best_corr)}
                    print(f"  [{date_key} {case_tag}] best shift: {best_shift_s:+.2f}s, r={best_corr:.3f}")

                    # Re-derive kt19_cam_time using the corrected time offset.
                    # best_shift_s was applied to HDRF (hdrf sampled at t - dt), so KT19
                    # leads HDRF by best_shift_s seconds.  To look up KT19 values at the
                    # camera time grid we query at  cam_time - best_shift_s/3600  (reverse sign).
                    cam_time_kt19_corr = cam_time - best_shift_s / 3600.0
                    ins_corr    = np.searchsorted(t_sorted, cam_time_kt19_corr)
                    idx_lo_corr = np.clip(ins_corr - 1, 0, len(t_sorted) - 1)
                    idx_hi_corr = np.clip(ins_corr,     0, len(t_sorted) - 1)
                    diff_lo_corr = np.abs(t_sorted[idx_lo_corr] - cam_time_kt19_corr)
                    diff_hi_corr = np.abs(t_sorted[idx_hi_corr] - cam_time_kt19_corr)
                    closest_idx_corr = sort_order[np.where(diff_lo_corr <= diff_hi_corr, idx_lo_corr, idx_hi_corr)]
                    within_1s_corr   = np.minimum(diff_lo_corr, diff_hi_corr) <= (1 / 3600)
                    kt19_cam_time    = np.where(within_1s_corr, kt19_selected_all[closest_idx_corr], np.nan)
                    kt19_v           = kt19_cam_time[valid_mask]
                    print(f"  [{date_key} {case_tag}] kt19_cam_time re-derived with shift {-best_shift_s:+.2f}s applied to KT19 lookup")

                    # plot correlation vs shift
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(shifts_s, corrs, color='steelblue', linewidth=1.5)
                    ax.axvline(best_shift_s, color='red', linestyle='--',
                               label=f'Best: {best_shift_s:+.2f}s  (r={best_corr:.3f})')
                    ax.set_xlabel('Time shift applied to NAD HDRF (s)', fontsize=13)
                    ax.set_ylabel('Pearson r  (KT19 vs NAD HDRF)', fontsize=13)
                    ax.set_title(f'Cross-correlation: KT19 vs NAD HDRF\n{date_key} — {case_tag}', fontsize=13)
                    ax.legend(fontsize=11)
                    plt.tight_layout()
                    plt.savefig(f'{fig_dir}/kt19_hdrf_timeshift_{date_key}_{case_tag}.png', dpi=300)
                    plt.close('all')

                    # plot aligned time series with best shift applied
                    hdrf_aligned = hdrf_interp(t_v - best_shift_s / 3600.0)
                    plt.close('all')
                    fig, ax1 = plt.subplots(figsize=(10, 5))
                    ax2 = ax1.twinx()
                    ax1.plot(t_v, hdrf_aligned, color='skyblue',
                             label=f'NAD HDRF (shifted {best_shift_s:+.2f}s)', alpha=0.75, linewidth=2)
                    ax2.plot(t_v, kt19_v2, color='coral',
                             label='KT19 Surface Temperature', alpha=0.75, linewidth=1.5)
                    ax1.set_xlabel('Time (UTC)', fontsize=14)
                    ax1.set_ylabel('NAD HDRF', fontsize=14)
                    ax2.set_ylabel('KT19 Surface Temperature (°C)', fontsize=14)
                    lns2 = [ax1.lines[0], ax2.lines[0]]
                    ax1.legend(lns2, [l.get_label() for l in lns2], fontsize=10)
                    ax1.set_title(
                        f'KT19 & NAD HDRF — best shift = {best_shift_s:+.2f}s  (r={best_corr:.3f})\n{date_key} — {case_tag}',
                        fontsize=13)
                    plt.tight_layout()
                    plt.savefig(f'{fig_dir}/kt19_hdrf_aligned_{date_key}_{case_tag}.png', dpi=300)
                    plt.close('all')

                # continue
                
                nad_hdrf_thres = 0.4
                kt19_v_thres = kt19_cam_time[valid_mask & (nad_hdrf_cam_time >= nad_hdrf_thres)]
                
                # _append_stats
                _append_stats(kt19_high_hdrf_date_cond,  kt19_high_hdrf_date_cond_lower,  kt19_high_hdrf_date_cond_upper,  kt19_v_thres)
                _append_stats(kt19_date_cond,  kt19_date_cond_lower,  kt19_date_cond_upper,  kt19_v)
                _append_stats(brt19h, brt19h_lower, brt19h_upper, brt19h_v)
                _append_stats(brt37h, brt37h_lower, brt37h_upper, brt37h_v)
                _append_stats(brt37v, brt37v_lower, brt37v_upper, brt37v_v)
                denom = brt37v_v + brt37h_v
                pi_v = np.where(denom != 0, (brt37v_v - brt37h_v) / denom, np.nan)
                _append_stats(pi_37, pi_37_lower, pi_37_upper, pi_v)
                # grain_size_date_cond, grain_size_ratio_*, grain_size_1240_date_cond
                # are computed from albedo at SIF=1 below and appended there

                # --- grain size from regressed spectral albedo at ice fraction = 1 ---
                # Linear regression at 1700 nm for median; quantile regression for bounds
                valid_1700 = ~np.isnan(cam_ice_fraction) & ~np.isnan(alb_1700_cam_time)
                if np.sum(valid_1700) > 2:
                    _, _, _, _, res_1700 = linear_regression_with_confidence(
                        cam_ice_fraction[valid_1700], alb_1700_cam_time[valid_1700], confidence=0.95)
                    alb_1700_at_1 = np.clip(res_1700.intercept + res_1700.slope * 1.0, 1e-6, 1.0)
                    X_fit_1700 = sm.add_constant(cam_ice_fraction[valid_1700])
                    models_1700 = {}
                    for q in [0.05, 0.95]:
                        models_1700[q] = sm.QuantReg(alb_1700_cam_time[valid_1700], X_fit_1700).fit(q=q)
                    alb_1700_at_1_lo = np.clip(models_1700[0.05].predict([1.0, 1.0])[0], 1e-6, 1.0)
                    alb_1700_at_1_hi = np.clip(models_1700[0.95].predict([1.0, 1.0])[0], 1e-6, 1.0)
                else:
                    alb_1700_at_1 = alb_1700_at_1_lo = alb_1700_at_1_hi = np.nan

                # Linear regression at 1240 nm for median; quantile regression for bounds
                valid_1240 = ~np.isnan(cam_ice_fraction) & ~np.isnan(alb_1240_cam_time)
                if np.sum(valid_1240) > 2:
                    _, _, _, _, res_1240 = linear_regression_with_confidence(
                        cam_ice_fraction[valid_1240], alb_1240_cam_time[valid_1240], confidence=0.95)
                    alb_1240_at_1 = np.clip(res_1240.intercept + res_1240.slope * 1.0, 1e-6, 1.0)
                    X_fit_1240 = sm.add_constant(cam_ice_fraction[valid_1240])
                    models_1240 = {}
                    for q in [0.05, 0.95]:
                        models_1240[q] = sm.QuantReg(alb_1240_cam_time[valid_1240], X_fit_1240).fit(q=q)
                    alb_1240_at_1_lo = np.clip(models_1240[0.05].predict([1.0, 1.0])[0], 1e-6, 1.0)
                    alb_1240_at_1_hi = np.clip(models_1240[0.95].predict([1.0, 1.0])[0], 1e-6, 1.0)
                else:
                    alb_1240_at_1 = alb_1240_at_1_lo = alb_1240_at_1_hi = np.nan

                # χ(T) at retrieval wavelength using median KT19 temperature
                T_K_med = np.clip(np.nanmedian(kt19_v) + 273.15 if np.any(np.isfinite(kt19_v)) else 266.0,
                                  _ART_CHI_T_1700_[0], _ART_CHI_T_1700_[-1])
                chi_1700 = np.interp(T_K_med, _ART_CHI_T_1700_, _ART_CHI_K_1700_)
                T_K_med_1240 = np.clip(T_K_med, _ART_CHI_T_1240_[0], _ART_CHI_T_1240_[-1])
                chi_1240 = np.interp(T_K_med_1240, _ART_CHI_T_1240_, _ART_CHI_K_1240_)

                # K0: 1.0 for cloudy (white-sky); SZA-dependent for clear (black-sky)
                if 'clear' in case_tag and np.any(np.isfinite(sza_v)):
                    sza_med = np.nanmedian(sza_v)
                    K0 = (3.0 / 7.0) * (1.0 + 2.0 * np.cos(np.deg2rad(sza_med)))
                    K0 = K0 if K0 > 0 else np.nan
                else:
                    K0 = 1.0

                # Derive grain size (m → µm). Lower albedo → larger grain, upper albedo → smaller grain.
                A_season = _ART_A_SUMMER_ if sfx == 'summer' else _ART_A_SPRING_

                def _gs_um(alb_scalar, lam_nm, chi):
                    return _art_grain_size(alb_scalar, lam_nm, chi, K0, A_season) * 1e6

                gs_1700_med   = _gs_um(alb_1700_at_1,    1700.0, chi_1700)
                gs_1700_lower = _gs_um(alb_1700_at_1_hi, 1700.0, chi_1700)  # brighter → smaller grain
                gs_1700_upper = _gs_um(alb_1700_at_1_lo, 1700.0, chi_1700)  # darker  → larger grain
                gs_1240_med   = _gs_um(alb_1240_at_1,    1240.0, chi_1240)
                gs_1240_lower = _gs_um(alb_1240_at_1_hi, 1240.0, chi_1240)
                gs_1240_upper = _gs_um(alb_1240_at_1_lo, 1240.0, chi_1240)

                gs_alb_1700_date_cond.append(gs_1700_med)
                gs_alb_1700_date_cond_upper.append(gs_1700_upper)
                gs_alb_1700_date_cond_lower.append(gs_1700_lower)
                gs_alb_1240_date_cond.append(gs_1240_med)
                gs_alb_1240_date_cond_upper.append(gs_1240_upper)
                gs_alb_1240_date_cond_lower.append(gs_1240_lower)
                alb_1700_date_cond.append(alb_1700_at_1)
                alb_1700_date_cond_upper.append(alb_1700_at_1_hi)
                alb_1700_date_cond_lower.append(alb_1700_at_1_lo)
                alb_1240_date_cond.append(alb_1240_at_1)
                alb_1240_date_cond_upper.append(alb_1240_at_1_hi)
                alb_1240_date_cond_lower.append(alb_1240_at_1_lo)

                # grain_size_date_cond (1700 nm single-band) from SIF=1
                grain_size_date_cond.append(gs_1700_med)
                grain_size_date_cond_lower.append(gs_1700_lower)
                grain_size_date_cond_upper.append(gs_1700_upper)

                # grain_size_1240_date_cond from SIF=1
                grain_size_1240_date_cond.append(gs_1240_med)
                grain_size_1240_date_cond_lower.append(gs_1240_lower)
                grain_size_1240_date_cond_upper.append(gs_1240_upper)

                # --- ratio-method grain sizes from regressed albedo at SIF=1 ---
                def _linreg_alb_at_1(ice_frac, alb_arr):
                    """Return (med, lo, hi) albedo at ice_fraction=1 via lin+quantile reg."""
                    vmask = ~np.isnan(ice_frac) & ~np.isnan(alb_arr)
                    if np.sum(vmask) < 3:
                        return np.nan, np.nan, np.nan
                    _, _, _, _, r = linear_regression_with_confidence(
                        ice_frac[vmask], alb_arr[vmask], confidence=0.95)
                    med = np.clip(r.intercept + r.slope * 1.0, 1e-6, 1.0)
                    X   = sm.add_constant(ice_frac[vmask])
                    lo  = np.clip(sm.QuantReg(alb_arr[vmask], X).fit(q=0.05).predict([1.0, 1.0])[0], 1e-6, 1.0)
                    hi  = np.clip(sm.QuantReg(alb_arr[vmask], X).fit(q=0.95).predict([1.0, 1.0])[0], 1e-6, 1.0)
                    return med, lo, hi

                alb1280_med, alb1280_lo, alb1280_hi = _linreg_alb_at_1(cam_ice_fraction, alb_1280_cam_time)
                alb1100_med, alb1100_lo, alb1100_hi = _linreg_alb_at_1(cam_ice_fraction, alb_1100_cam_time)
                alb1020_med, alb1020_lo, alb1020_hi = _linreg_alb_at_1(cam_ice_fraction, alb_1020_cam_time)
                alb865_med,  alb865_lo,  alb865_hi  = _linreg_alb_at_1(cam_ice_fraction, alb_865_cam_time)
                alb1650_med, alb1650_lo, alb1650_hi = _linreg_alb_at_1(cam_ice_fraction, alb_1650_cam_time)

                def _gs_ratio_um(a1, a2, w1, chi1, w2, chi2):
                    return _art_grain_size_ratio(a1, a2, w1, chi1, w2, chi2, K0, A_season) * 1e6

                # 1280/1100 nm ratio
                gs_ratio_1280_med   = _gs_ratio_um(alb1280_med, alb1100_med, 1280.0, _GS_RATIO_1280_CHI1_, 1100.0, _GS_RATIO_1280_CHI2_)
                gs_ratio_1280_lower = _gs_ratio_um(alb1280_hi,  alb1100_lo,  1280.0, _GS_RATIO_1280_CHI1_, 1100.0, _GS_RATIO_1280_CHI2_)  # brighter λ1 / darker λ2 → smallest grain
                gs_ratio_1280_upper = _gs_ratio_um(alb1280_lo,  alb1100_hi,  1280.0, _GS_RATIO_1280_CHI1_, 1100.0, _GS_RATIO_1280_CHI2_)  # darker λ1 / brighter λ2 → largest grain
                grain_size_ratio_date_cond.append(gs_ratio_1280_med)
                grain_size_ratio_date_cond_lower.append(gs_ratio_1280_lower)
                grain_size_ratio_date_cond_upper.append(gs_ratio_1280_upper)

                # 1020/865 nm ratio
                gs_ratio_1020_med   = _gs_ratio_um(alb1020_med, alb865_med,  1020.0, _GS_RATIO_1020_CHI1_,  865.0, _GS_RATIO_1020_CHI2_)
                gs_ratio_1020_lower = _gs_ratio_um(alb1020_hi,  alb865_lo,   1020.0, _GS_RATIO_1020_CHI1_,  865.0, _GS_RATIO_1020_CHI2_)
                gs_ratio_1020_upper = _gs_ratio_um(alb1020_lo,  alb865_hi,   1020.0, _GS_RATIO_1020_CHI1_,  865.0, _GS_RATIO_1020_CHI2_)
                grain_size_ratio_1020_865_date_cond.append(gs_ratio_1020_med)
                grain_size_ratio_1020_865_date_cond_lower.append(gs_ratio_1020_lower)
                grain_size_ratio_1020_865_date_cond_upper.append(gs_ratio_1020_upper)

                # 1650/1020 nm ratio
                gs_ratio_1650_med   = _gs_ratio_um(alb1650_med, alb1020_med, 1650.0, _GS_RATIO_1650_CHI1_, 1020.0, _GS_RATIO_1650_CHI2_)
                gs_ratio_1650_lower = _gs_ratio_um(alb1650_hi,  alb1020_lo,  1650.0, _GS_RATIO_1650_CHI1_, 1020.0, _GS_RATIO_1650_CHI2_)
                gs_ratio_1650_upper = _gs_ratio_um(alb1650_lo,  alb1020_hi,  1650.0, _GS_RATIO_1650_CHI1_, 1020.0, _GS_RATIO_1650_CHI2_)
                grain_size_ratio_1650_1020_date_cond.append(gs_ratio_1650_med)
                grain_size_ratio_1650_1020_date_cond_lower.append(gs_ratio_1650_lower)
                grain_size_ratio_1650_1020_date_cond_upper.append(gs_ratio_1650_upper)

                print(f"Date: {date_key}, Case: {case_tag}, Broadband Albedo at Ice Fraction=1.0: {broadband_alb_at_1:.3f} (+{broadband_alb_at_1_upper - broadband_alb_at_1:.3f}/-{broadband_alb_at_1 - broadband_alb_at_1_lower:.3f})")
                print(f"    Mean MYI Ratio: {myi_ratio_date_cond[-1]:.2f} +/- {myi_ratio_date_cond_std[-1]:.2f} %")
                print(f"    Mean FYI Ratio: {fyi_ratio_date_cond[-1]:.2f} +/- {fyi_ratio_date_cond_std[-1]:.2f} %")
                print(f"    Mean YI Ratio: {yi_ratio_date_cond[-1]:.2f} +/- {yi_ratio_date_cond_std[-1]:.2f} %")
                print(f"    Mean FYI+YI Ratio: {fyi_yi_ratio_date_cond[-1]:.2f} +/- {fyi_yi_ratio_date_cond_std[-1]:.2f} %")
                      
            axr.set_xlabel('Camera Ice Fraction', fontsize=14)
            axr.set_ylabel('Broadband Albedo', fontsize=14)
            axr.tick_params(labelsize=12)
            axr.set_title('Broadband Albedo vs. Camera Ice Fraction ', fontsize=13)
            fig.suptitle(f'Date: {date_key}, Case: {case_tag}, Alt < 1.6 km',
                            fontsize=16)
            fig.savefig(f'{fig_dir}/arcsix_albedo_{date_key}_{case_tag}_broadband_ice_frac.png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            

    
    # --- print best KT19 vs NAD-HDRF time-shift summary (like ice_frac_time_offset) ---
    print("\nkt19_hdrf_best_shift = {")
    for (dk, ct), v in kt19_hdrf_best_shift.items():
        shift_s = v['shift_s']
        r       = v['r']
        sign    = '+' if shift_s >= 0 else '-'
        print(f"    ('{dk}', '{ct}'): {sign}{abs(shift_s):.2f}/3600,  # r={r:.3f}")
    print("}")

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
    # ax.set_title('Broadband Albedo at Ice Fraction = 1.0 from Linear Fit', fontsize=13)
    fig.savefig(f'{fig_dir}/arcsix_albedo_broadband_ice_frac_fit_summary.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)

    for alb_vals, alb_upper, alb_lower, wvl_label, savename in [
        (alb_1700_date_cond, alb_1700_date_cond_upper, alb_1700_date_cond_lower,
         '1700 nm', 'arcsix_albedo_1700nm_ice_frac_fit_summary.png'),
        (alb_1240_date_cond, alb_1240_date_cond_upper, alb_1240_date_cond_lower,
         '1240 nm', 'arcsix_albedo_1240nm_ice_frac_fit_summary.png'),
    ]:
        if len(alb_vals) == 0:
            continue
        alb_unc = [(u - l) / 2 for u, l in zip(alb_upper, alb_lower)]
        alb_colors = colors[:len(alb_vals)]
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(np.arange(len(alb_vals)), alb_vals,
                    yerr=alb_unc,
                    fmt='o', ecolor='lightgray', capsize=5, markerfacecolor='none', zorder=1)
        ax.scatter(np.arange(len(alb_vals)), alb_vals, c=alb_colors, alpha=0.8, s=50, zorder=2)
        ax.set_xticks(np.arange(len(alb_vals)))
        ax.set_xticklabels(date_conditions[:len(alb_vals)], rotation=45, ha='right')
        ax.legend(handles=[mpatches.Patch(color='blue', label='Clear'),
                           mpatches.Patch(color='gray', label='Cloudy')], fontsize=10)
        ax.set_xlabel('Date and Case', fontsize=14)
        ax.set_ylabel(f'{wvl_label} Albedo at Ice Fraction = 1.0', fontsize=14)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{savename}', bbox_inches='tight', dpi=150)
        plt.close(fig)

    # Scatter: broadband albedo at SIF=1 vs spectral albedo at SIF=1
    bb_arr   = np.array(broadband_alb_date_cond)
    bb_unc   = np.array(broadband_alb_date_cond_unc)
    for alb_vals, alb_upper, alb_lower, wvl_label, savename in [
        (alb_1240_date_cond, alb_1240_date_cond_upper, alb_1240_date_cond_lower,
         '1240 nm', 'arcsix_albedo_broadband_vs_1240nm_sif1.png'),
        (alb_1700_date_cond, alb_1700_date_cond_upper, alb_1700_date_cond_lower,
         '1700 nm', 'arcsix_albedo_broadband_vs_1700nm_sif1.png'),
    ]:
        if len(alb_vals) == 0:
            continue
        x_arr = np.array(alb_vals)
        x_lo  = np.array(alb_lower)
        x_hi  = np.array(alb_upper)
        finite = np.isfinite(x_arr) & np.isfinite(bb_arr)
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.errorbar(x_arr, bb_arr,
                    xerr=[x_arr - x_lo, x_hi - x_arr],
                    yerr=bb_unc,
                    fmt='o', ecolor='lightgray', capsize=5, markerfacecolor='none', zorder=1)
        ax.scatter(x_arr, bb_arr, c=colors[:len(x_arr)], alpha=0.8, s=60, zorder=2)
        legend_handles = [mpatches.Patch(color='blue', label='Clear'),
                          mpatches.Patch(color='gray', label='Cloudy')]
        if finite.sum() >= 3:
            res = sm.WLS(bb_arr[finite], sm.add_constant(x_arr[finite])).fit()
            x_line = np.linspace(np.nanmin(x_arr[finite]) * 0.97,
                                 np.nanmax(x_arr[finite]) * 1.03, 100)
            fit_line, = ax.plot(x_line, res.params[0] + res.params[1] * x_line,
                                color='orange', linestyle='--',
                                label=r'$\mathrm{R^2}$=%.3f, p=%.2e' % (res.rsquared, res.pvalues[1]))
            legend_handles.append(fit_line)
        ax.set_xlabel(f'{wvl_label} Albedo at Sea Ice Fraction = 1.0', fontsize=14)
        ax.set_ylabel('Broadband Albedo at Sea Ice Fraction = 1.0', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(handles=legend_handles, fontsize=10)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{savename}', bbox_inches='tight', dpi=150)
        plt.close(fig)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue' if 'clear' in dc else 'gray' for dc in date_conditions]
    ax.errorbar(myi_ratio_date_cond, broadband_alb_date_cond,
                xerr=myi_ratio_date_cond_std,
                yerr=broadband_alb_date_cond_unc, 
                fmt='o', ecolor='lightgray', capsize=5, markerfacecolor='none', zorder=1)#label='Data Points with Uncertainty')
    ax.scatter(myi_ratio_date_cond, broadband_alb_date_cond, c=colors, alpha=0.8, s=50, zorder=2)
    X = sm.add_constant(myi_ratio_date_cond)
    y = np.array(broadband_alb_date_cond)
    mod_wls = sm.WLS(y, X)
    res_wls = mod_wls.fit()
    slope = res_wls.params[1]
    intercept = res_wls.params[0]
    r_value = res_wls.rsquared
    x_fit = np.linspace(0, np.max(myi_ratio_date_cond)*1.05, 100)
    y_fit = intercept + slope * x_fit
    p_values = res_wls.pvalues
    ax.plot(x_fit, y_fit, color='orange', linestyle='--', label=r'Fit: y=%.3fx+%.3f, $\mathrm{R}$=%.3f, p-value=%.3f'%(slope*100, intercept, r_value, p_values[1]))

    ax.legend(fontsize=12)
    ax.set_xlabel('Mean Multi-year Sea Ice Ratio', fontsize=14)
    ax.set_ylabel('Broadband Albedo at Ice Fraction = 1.0', fontsize=14)
    ax.tick_params(labelsize=12)
    # ax.set_title('Broadband Albedo at Ice Fraction = 1.0 vs. Mean MYI Ratio', fontsize=13)
    fig.savefig(f'{fig_dir}/arcsix_albedo_broadband_ice_frac_vs_myi_ratio.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    print("broadband_alb_date_cond min, max:", np.min(broadband_alb_date_cond), np.max(broadband_alb_date_cond))
    
    date_conditions_simplified = [dc.replace('2024', '').replace('_', '-') for dc in date_conditions]

    def _plot_summary_panel(x_var, x_xerr, x_label, savename):
        """Two-panel summary: (a) date-index timeline, (b) scatter + WLS fit vs x_var."""
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5),
                                       gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2})
        n = len(broadband_alb_date_cond)
        # (a) date-index timeline
        ax1.errorbar(np.arange(n), broadband_alb_date_cond, yerr=broadband_alb_date_cond_unc,
                     fmt='o', ecolor='lightgray', capsize=5, markerfacecolor='none', zorder=1)
        ax1.scatter(np.arange(n), broadband_alb_date_cond, c=colors, alpha=0.8, s=50, zorder=2)
        ax1.set_xticks(np.arange(n))
        ax1.set_xticklabels(date_conditions_simplified, rotation=45, ha='right')
        ax1.legend(handles=[mpatches.Patch(color='blue', label='Clear'),
                             mpatches.Patch(color='gray', label='Cloudy')], fontsize=10)
        ax1.set_xlabel('Date and condition', fontsize=14)
        ax1.set_ylabel('Broadband Albedo at Sea Ice Fraction = 1.0', fontsize=14)
        # (b) scatter + WLS fit
        res_wls = sm.WLS(np.array(broadband_alb_date_cond), sm.add_constant(x_var)).fit()
        slope, intercept = res_wls.params[1], res_wls.params[0]
        x_fit = np.linspace(0, np.max(x_var) * 1.05, 100)
        ax2.errorbar(x_var, broadband_alb_date_cond, xerr=x_xerr, yerr=broadband_alb_date_cond_unc,
                     fmt='o', ecolor='lightgray', capsize=5, markerfacecolor='none', zorder=1)
        ax2.scatter(x_var, broadband_alb_date_cond, c=colors, alpha=0.8, s=50, zorder=2)
        ax2.plot(x_fit, intercept + slope * x_fit, color='orange', linestyle='--',
                 label=r'Fit: y=%.3fx+%.3f, $\mathrm{R}$=%.3f, p-value=%.3f'
                       % (slope * 100, intercept, res_wls.rsquared, res_wls.pvalues[1]))
        ax2.legend(fontsize=12)
        ax2.set_xlabel(x_label, fontsize=14)
        ax2.set_ylabel('Broadband Albedo at Sea Ice Fraction = 1.0', fontsize=14)
        for ax, cap in zip([ax1, ax2], ['(a)', '(b)']):
            ax.tick_params(labelsize=12)
            ax.text(0, 1.07, cap, transform=ax.transAxes, fontsize=16, va='top', ha='left')
        fig.savefig(f'{fig_dir}/{savename}', bbox_inches='tight', dpi=150)
        plt.close(fig)

    _plot_summary_panel(
        myi_ratio_date_cond, myi_ratio_date_cond_std,
        'Mean Multi-year Sea Ice Coverage (%)', 'arcsix_albedo_broadband_ice_frac_summary.png')
    _plot_summary_panel(
        myi_fyi_ratio_date_cond, (myi_fyi_ratio_date_cond_lower, myi_fyi_ratio_date_cond_upper),
        'Mean Multi-year Sea Ice Coverage (%)', 'arcsix_albedo_broadband_myi_fyi_ice_frac_summary.png')
    _plot_summary_panel(
        ice_age_date_cond, (ice_age_date_cond_lower, ice_age_date_cond_upper),
        'Median Sea Ice Age (year)', 'arcsix_albedo_broadband_ice_age_summary.png')
    _plot_summary_panel(
        ice_age_over1_date_cond, None,
        'Sea Ice Age >= 1 year ratio', 'arcsix_albedo_broadband_ice_age_over1_summary.png')
    _plot_summary_panel(
        ice_age_over2_date_cond, None,
        'Sea Ice Age >= 2 year ratio', 'arcsix_albedo_broadband_ice_age_over2_summary.png')
    
    # multi variables regression
    X_multi = np.column_stack((myi_ratio_date_cond, fyi_ratio_date_cond, yi_ratio_date_cond))
    X_multi = sm.add_constant(X_multi)
    y_multi = np.array(broadband_alb_date_cond)
    mod_multi = sm.OLS(y_multi, X_multi)
    res_multi = mod_multi.fit()
    print(res_multi.summary())
    print("Fit: y = %.3f + %.3f*MYI + %.3f*FYI + %.3f*YI" % (res_multi.params[0], res_multi.params[1]*100, res_multi.params[2]*100, res_multi.params[3]*100))
    print("R-squared:", res_multi.rsquared)
    print("p-values:", res_multi.pvalues)
    #

    # ------------------------------------------------------------------
    # Plots: broadband albedo vs kt19, brt19h, brt37h, brt37v
    # same style as arcsix_albedo_broadband_ice_frac_vs_myi_ratio.png
    # ------------------------------------------------------------------
    def _plot_alb_vs_var(x_vals, x_lower, x_upper, y_vals, y_unc, colors,
                         xlabel, savename, fig_dir):
        """WLS fit + scatter of broadband albedo vs one surface variable."""
        x_arr = np.array(x_vals)
        x_lo  = np.array(x_lower)
        x_hi  = np.array(x_upper)
        y_arr = np.array(y_vals)
        y_unc_arr = np.array(y_unc)

        if len(x_arr) == 0 or len(y_arr) == 0:
            print(f"No data to plot {xlabel}. Skipping.")
            return

        # keep only finite pairs
        finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr[:len(x_arr)])
        if finite_mask.sum() < 3:
            print(f"Not enough valid points to plot {xlabel}. Skipping.")
            return

        X_fit = sm.add_constant(x_arr[finite_mask])
        mod   = sm.WLS(y_arr[finite_mask], X_fit)
        res   = mod.fit()
        slope     = res.params[1]
        intercept = res.params[0]
        r_value   = res.rsquared
        p_value   = res.pvalues[1]
        x_line = np.linspace(np.nanmin(x_arr[finite_mask]) * 0.97,
                             np.nanmax(x_arr[finite_mask]) * 1.03, 100)
        y_line = intercept + slope * x_line

        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(x_arr, y_arr,
                    xerr=[x_arr - x_lo, x_hi - x_arr],
                    yerr=y_unc_arr,
                    fmt='o', ecolor='lightgray', capsize=5,
                    markerfacecolor='none', zorder=1)
        ax.scatter(x_arr, y_arr, c=colors, alpha=0.8, s=60, zorder=2)
        # ax.plot(x_line, y_line, color='orange', linestyle='--',
        #         label=r'Fit: y=%.4fx+%.3f, $\mathrm{R^2}$=%.3f, p=%.3f'
        #               % (slope, intercept, r_value, p_value))
        ax.plot(x_line, y_line, color='orange', linestyle='--',
                label=r'$\mathrm{R^2}$=%.3f, p=%.2e'
                      % (r_value, p_value))
        ax.legend(fontsize=11)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel('Broadband Albedo at Sea Ice Fraction = 1.0', fontsize=14)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{savename}', bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    def _plot_alb_vs_var1var2(x1_vals, x1_lower, x1_upper,
                              x2_vals, x2_lower, x2_upper,
                              y_vals, y_unc, 
                              colors,
                              x1label, x2label,
                              savename, fig_dir):
        """WLS fit + scatter of broadband albedo vs one surface variable."""
        x1_arr = np.array(x1_vals)
        x1_lo  = np.array(x1_lower)
        x1_hi  = np.array(x1_upper)
        x2_arr = np.array(x2_vals)
        x2_lo  = np.array(x2_lower)
        x2_hi  = np.array(x2_upper)
        y_arr = np.array(y_vals)
        y_unc_arr = np.array(y_unc)

        # keep only finite pairs
        finite_mask = np.isfinite(x1_arr) & np.isfinite(x2_arr) & np.isfinite(y_arr)
        if finite_mask.sum() < 3:
            print(f"Not enough valid points to plot {x1label} and {x2label}. Skipping.")
            return

        X1_fit = sm.add_constant(x1_arr[finite_mask])
        mod   = sm.WLS(y_arr[finite_mask], X1_fit)
        res   = mod.fit()
        slope1     = res.params[1]
        intercept1 = res.params[0]
        r_value1   = res.rsquared
        p_value1   = res.pvalues[1]
        x_line1 = np.linspace(np.nanmin(x1_arr[finite_mask]) * 0.97,
                             np.nanmax(x1_arr[finite_mask]) * 1.03, 100)
        y_line1 = intercept1 + slope1 * x_line1
        
        X2_fit = sm.add_constant(x2_arr[finite_mask])
        mod   = sm.WLS(y_arr[finite_mask], X2_fit)
        res   = mod.fit()
        slope2     = res.params[1]
        intercept2 = res.params[0]
        r_value2   = res.rsquared
        p_value2   = res.pvalues[1]
        x_line2 = np.linspace(np.nanmin(x2_arr[finite_mask]) * 0.97, 
                              np.nanmax(x2_arr[finite_mask]) * 1.03, 100)
        y_line2 = intercept2 + slope2 * x_line2

        plt.close('all')
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={ 'wspace': 0.2})
        ax.errorbar(x1_arr, y_arr,
                    xerr=[x1_arr - x1_lo, x1_hi - x1_arr],
                    yerr=y_unc_arr,
                    fmt='o', ecolor='lightgray', capsize=5,
                    markerfacecolor='none', zorder=1)
        ax.scatter(x1_arr, y_arr, c=colors, alpha=0.8, s=60, zorder=2)
        # ax.plot(x_line, y_line, color='orange', linestyle='--',
        #         label=r'Fit: y=%.4fx+%.3f, $\mathrm{R^2}$=%.3f, p=%.3f'
        #               % (slope, intercept, r_value, p_value))
        ax.plot(x_line1, y_line1, color='orange', linestyle='--',
                label=r'$\mathrm{R^2}$=%.3f, p=%.2e'
                      % (r_value1, p_value1))
        ax.legend(fontsize=11)
        ax.set_xlabel(x1label, fontsize=14)
        
        ax2.errorbar(x2_arr, y_arr,
                    xerr=[x2_arr - x2_lo, x2_hi - x2_arr],
                    yerr=y_unc_arr,
                    fmt='o', ecolor='lightgray', capsize=5,
                    markerfacecolor='none', zorder=1)
        ax2.scatter(x2_arr, y_arr, c=colors, alpha=0.8, s=60, zorder=2)
        ax2.plot(x_line2, y_line2, color='orange', linestyle='--',
                label=r'$\mathrm{R^2}$=%.3f, p=%.2e'
                      % (r_value2, p_value2))
        ax2.legend(fontsize=11)
        ax2.set_xlabel(x2label, fontsize=14)
        
        for ax_, cap in zip([ax, ax2], ['(a)', '(b)']):
            ax_.tick_params(labelsize=12)
            ax_.text(0,  1.07, cap, transform=ax_.transAxes, fontsize=16, va='top', ha='left')
            ax_.set_ylabel('Broadband Albedo at Sea Ice Fraction = 1.0', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{savename}', bbox_inches='tight', dpi=150)
        plt.close(fig)

    myi_upper = myi_ratio_date_cond_upper
    myi_lower = myi_ratio_date_cond_lower
    _plot_alb_vs_var(
        myi_ratio_date_cond, myi_lower, myi_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Mean MYI Ratio (%)',
        savename='arcsix_albedo_broadband_vs_myi_ratio.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        kt19_date_cond, kt19_date_cond_lower, kt19_date_cond_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Median KT-19 Surface Temperature ($^o$C)',
        savename='arcsix_albedo_broadband_vs_kt19.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        kt19_high_hdrf_date_cond, kt19_high_hdrf_date_cond_lower, kt19_high_hdrf_date_cond_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Median KT-19 Surface Temperature over ice ($^o$C)',
        savename='arcsix_albedo_broadband_vs_kt19_thres.png',
        fig_dir=fig_dir,
    )

    _plot_alb_vs_var1var2(kt19_date_cond, kt19_date_cond_lower, kt19_date_cond_upper,
                          myi_ratio_date_cond, myi_lower, myi_upper,
                          broadband_alb_date_cond, broadband_alb_date_cond_unc,
                          colors,
                          x1label='Median KT-19 Surface Temperature ($^o$C)', 
                          x2label='Mean MYI Ratio (%)',
                          savename='arcsix_albedo_broadband_vs_kt19_myi_ratio.png', 
                          fig_dir=fig_dir)
    _plot_alb_vs_var(
        brt19h, brt19h_lower, brt19h_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Median BRT19H Brightness Temperature (K)',
        savename='arcsix_albedo_broadband_vs_brt19h.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        brt37h, brt37h_lower, brt37h_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Median BRT37H Brightness Temperature (K)',
        savename='arcsix_albedo_broadband_vs_brt37h.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        brt37v, brt37v_lower, brt37v_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Median BRT37V Brightness Temperature (K)',
        savename='arcsix_albedo_broadband_vs_brt37v.png',
        fig_dir=fig_dir,
    )
    pi_arr  = np.array(pi_37)
    pi_lo   = np.array(pi_37_lower)
    pi_hi   = np.array(pi_37_upper)
    bb_arr  = np.array(broadband_alb_date_cond)
    bb_unc  = np.array(broadband_alb_date_cond_unc)
    clr_arr = np.array(colors)
    spring_mask = np.array([dc[:8] < '20240630' for dc in date_conditions])
    summer_mask = ~spring_mask
    _xlabel_pi = 'Median Polarization Index (BRT37V$-$BRT37H)/(BRT37V$+$BRT37H)'
    _plot_alb_vs_var(
        pi_37, pi_37_lower, pi_37_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel=_xlabel_pi,
        savename='arcsix_albedo_broadband_vs_pi37.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        pi_arr[spring_mask], pi_lo[spring_mask], pi_hi[spring_mask],
        bb_arr[spring_mask], bb_unc[spring_mask], clr_arr[spring_mask].tolist(),
        xlabel=_xlabel_pi,
        savename='arcsix_albedo_broadband_vs_pi37_spring.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        pi_arr[summer_mask], pi_lo[summer_mask], pi_hi[summer_mask],
        bb_arr[summer_mask], bb_unc[summer_mask], clr_arr[summer_mask].tolist(),
        xlabel=_xlabel_pi,
        savename='arcsix_albedo_broadband_vs_pi37_summer.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        grain_size_date_cond, grain_size_date_cond_lower, grain_size_date_cond_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Median Optical-Equivalent Grain Radius via 1700 nm ($\mathrm{\mu}$m)',
        savename='arcsix_albedo_broadband_vs_grain_size.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        grain_size_ratio_date_cond, grain_size_ratio_date_cond_lower, grain_size_ratio_date_cond_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Median Optical-Equivalent Grain Radius via 1280/1100 nm Ratio ($\mathrm{\mu}$m)',
        savename='arcsix_albedo_broadband_vs_grain_size_ratio.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        grain_size_ratio_1020_865_date_cond, grain_size_ratio_1020_865_date_cond_lower, grain_size_ratio_1020_865_date_cond_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Median Optical-Equivalent Grain Radius via 1020/865 nm Ratio ($\mathrm{\mu}$m)',
        savename='arcsix_albedo_broadband_vs_grain_size_ratio_1020_865.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        grain_size_ratio_1650_1020_date_cond, grain_size_ratio_1650_1020_date_cond_lower, grain_size_ratio_1650_1020_date_cond_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Median Optical-Equivalent Grain Radius via 1650/1020 nm Ratio ($\mathrm{\mu}$m)',
        savename='arcsix_albedo_broadband_vs_grain_size_ratio_1650_1020.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        grain_size_1240_date_cond, grain_size_1240_date_cond_lower, grain_size_1240_date_cond_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Median Optical-Equivalent Grain Radius via Single-Band AART at 1240 nm ($\mathrm{\mu}$m)',
        savename='arcsix_albedo_broadband_vs_grain_size_1240.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        gs_alb_1700_date_cond, gs_alb_1700_date_cond_lower, gs_alb_1700_date_cond_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Grain Radius from Regressed Albedo at 1700 nm (ART, $\mathrm{\mu}$m)',
        savename='arcsix_albedo_broadband_vs_gs_alb_1700.png',
        fig_dir=fig_dir,
    )
    _plot_alb_vs_var(
        gs_alb_1240_date_cond, gs_alb_1240_date_cond_lower, gs_alb_1240_date_cond_upper,
        broadband_alb_date_cond, broadband_alb_date_cond_unc, colors,
        xlabel='Grain Radius from Regressed Albedo at 1240 nm (ART, $\mathrm{\mu}$m)',
        savename='arcsix_albedo_broadband_vs_gs_alb_1240.png',
        fig_dir=fig_dir,
    )

    # MYI ratio vs KT-19 scatter plot
    myi_arr   = np.array(myi_ratio_date_cond)
    kt19_arr  = np.array(kt19_date_cond)
    kt19_lo   = np.array(kt19_date_cond_lower)
    kt19_hi   = np.array(kt19_date_cond_upper)
    finite_mk = np.isfinite(myi_arr) & np.isfinite(kt19_arr)
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(myi_arr, kt19_arr,
                xerr=[myi_arr - np.array(myi_lower), np.array(myi_upper) - myi_arr],
                yerr=[kt19_arr - kt19_lo, kt19_hi - kt19_arr],
                fmt='o', ecolor='lightgray', capsize=5, markerfacecolor='none', zorder=1)
    ax.scatter(myi_arr, kt19_arr, c=colors, alpha=0.8, s=60, zorder=2)
    if finite_mk.sum() >= 3:
        res_mk = sm.WLS(kt19_arr[finite_mk], sm.add_constant(myi_arr[finite_mk])).fit()
        x_line = np.linspace(np.nanmin(myi_arr[finite_mk]) * 0.97,
                             np.nanmax(myi_arr[finite_mk]) * 1.03, 100)
        fit_line, = ax.plot(x_line, res_mk.params[0] + res_mk.params[1] * x_line,
                            color='orange', linestyle='--',
                            label=r'$\mathrm{R^2}$=%.3f, p=%.2e' % (res_mk.rsquared, res_mk.pvalues[1]))
        ax.legend(handles=[mpatches.Patch(color='blue', label='Clear'),
                           mpatches.Patch(color='gray', label='Cloudy'),
                           fit_line], fontsize=10)
    else:
        ax.legend(handles=[mpatches.Patch(color='blue', label='Clear'),
                           mpatches.Patch(color='gray', label='Cloudy')], fontsize=10)
    ax.set_xlabel('Mean Multi-year Sea Ice Coverage (%)', fontsize=14)
    ax.set_ylabel('Median KT-19 Surface Temperature ($^o$C)', fontsize=14)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_myi_ratio_vs_kt19.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    # sys.exit()

    def _sfx(date_str):
        """Return 'summer' or 'spring' suffix for the given date string."""
        return 'summer' if date_str > '20240630' else 'spring'

    def _load_leg_data(combined_data, date_str, case_tags):
        """Load arrays from combined_data for a given date and case tag substrings.

        Returns (alb_wvl, lon, lat, alt, time, alb_spectral, broadband_alb).
        """
        sfx = _sfx(date_str)
        date_mask = combined_data[f'dates_{sfx}_all'] == int(date_str)
        case_mask = np.zeros(len(date_mask), dtype=bool)
        for ct in case_tags:
            case_mask |= np.array([ct in tag for tag in combined_data[f'case_tags_{sfx}_all']])
        m = date_mask & case_mask
        return (
            combined_data[f'wvl_{sfx}'],
            combined_data[f'lon_all_{sfx}'][m],
            combined_data[f'lat_all_{sfx}'][m],
            combined_data[f'alt_all_{sfx}'][m],
            combined_data[f'time_{sfx}_all'][m],
            combined_data[f'alb_iter2_all_{sfx}'][m, :],
            combined_data[f'broadband_alb_iter2_all_filter_{sfx}'][m],
        )

    def _match_cam_to_ssfr(cam_time, time_ssfr, broadband_alb, alb_spectral, threshold=1/3600):
        """Vectorized nearest-neighbor match of camera timestamps to SSFR timestamps.

        Returns (broadband_alb_matched, alb_spectral_matched) with NaN where no
        SSFR measurement falls within *threshold* hours of the camera timestamp.
        """
        sort_order = np.argsort(time_ssfr)
        t_sorted   = time_ssfr[sort_order]
        ins        = np.searchsorted(t_sorted, cam_time)
        idx_lo     = np.clip(ins - 1, 0, len(t_sorted) - 1)
        idx_hi     = np.clip(ins,     0, len(t_sorted) - 1)
        diff_lo    = np.abs(t_sorted[idx_lo] - cam_time)
        diff_hi    = np.abs(t_sorted[idx_hi] - cam_time)
        closest    = sort_order[np.where(diff_lo <= diff_hi, idx_lo, idx_hi)]
        within     = np.minimum(diff_lo, diff_hi) <= threshold
        bb_matched  = np.where(within, broadband_alb[closest], np.nan)
        alb_matched = np.where(within[:, np.newaxis], alb_spectral[closest], np.nan)
        return bb_matched, alb_matched

    def _plot_leg(time_ssfr, broadband_alb, lat, lon, alt, alb_spectral, alb_wvl,
                  cam_time, cam_ice_frac, broadband_alb_cam, alb_cam,
                  date_label, fig_dir, prefix):
        """Generate all diagnostic plots for a single camera/SSFR flight leg."""
        alb_avg = np.nanmean(alb_spectral, axis=0)
        alb_std = np.nanstd(alb_spectral, axis=0)
        alt_avg = np.nanmean(alt)

        # 1. Average spectral albedo with std shading
        plt.close('all')
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(alb_wvl, alb_avg, label=f'Alt: {alt_avg:.1f}km', color='b')
        ax.fill_between(alb_wvl, alb_avg - alb_std, alb_avg + alb_std, color='b', alpha=0.1)
        for band in gas_bands:
            ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Surface Albedo', fontsize=14)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'Surface Albedo (atm corr + fit), {date_label}', fontsize=13)
        ax.set_xlim(350, 2000)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{prefix}_avg.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

        # 2. Latitude vs broadband albedo
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(lat, broadband_alb, label=f'Alt: {alt_avg:.1f}km', c='b', s=10)
        ax.set_xlabel('Latitude', fontsize=14)
        ax.set_ylabel('Broadband Albedo', fontsize=14)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=12)
        ax.set_title(f'Surface Albedo (atm corr + fit), {date_label}', fontsize=13)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{prefix}_broadband_lat.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

        # 3. Longitude vs broadband albedo
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(lon, broadband_alb, label=f'Alt: {alt_avg:.1f}km', c='b', s=10)
        ax.set_xlabel('Longitude', fontsize=14)
        ax.set_ylabel('Broadband Albedo', fontsize=14)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=12)
        ax.set_title(f'Surface Albedo (atm corr + fit), {date_label}', fontsize=13)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{prefix}_broadband_lon.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

        # 4. Time scatter: broadband albedo + camera ice fraction (twin y)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax2 = ax.twinx()
        l1 = ax.scatter(time_ssfr, broadband_alb, label=f'Alt: {alt_avg:.1f}km', c='b', s=10)
        l2 = ax2.scatter(cam_time, cam_ice_frac, label='Camera Ice Fraction', c='r', s=5)
        ax.set_xlabel('Time (UTC)', fontsize=14)
        ax.set_ylabel('Broadband Albedo', fontsize=14)
        ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
        ax.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=10)
        ax.tick_params(labelsize=12)
        ax.set_title(f'Surface Albedo (atm corr + fit), {date_label}', fontsize=13)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{prefix}_broadband_time.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

        # 5. Time line: broadband albedo + camera ice fraction (twin y)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax2 = ax.twinx()
        l1 = ax.plot(time_ssfr, broadband_alb, c='skyblue', label='Broadband Albedo', alpha=0.75, linewidth=2)
        l2 = ax2.plot(cam_time, cam_ice_frac, c='coral', label='Camera Ice Fraction', alpha=0.75, linewidth=1.5)
        ax.set_xlabel('Time (UTC)', fontsize=14)
        ax.set_ylabel('Broadband Albedo', fontsize=14)
        ax2.set_ylabel('Camera Ice Fraction', fontsize=14)
        ax.legend([l1[0], l2[0]], [l1[0].get_label(), l2[0].get_label()], fontsize=10)
        ax.tick_params(labelsize=12)
        ax.set_title(f'Surface Albedo (atm corr + fit), {date_label}', fontsize=13)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{prefix}_broadband_time_line.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

        # 6. Camera ice fraction vs broadband albedo scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(cam_ice_frac, broadband_alb_cam, s=10, c='k')
        ax.set_xlabel('Camera Ice Fraction', fontsize=14)
        ax.set_ylabel('Broadband Albedo', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_title(f'Surface Albedo vs Camera Ice Fraction, {date_label}', fontsize=13)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{prefix}_broadband_icefraction.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

        # 7. Camera ice fraction vs albedo at selected wavelengths
        wvl_targets = [450, 860, 1240, 1630]
        wvl_colors  = ['b', 'g', 'r', 'm']
        wvl_idxs    = [np.argmin(np.abs(alb_wvl - w)) for w in wvl_targets]
        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, wvl, col in zip(wvl_idxs, wvl_targets, wvl_colors):
            ax.scatter(cam_ice_frac, alb_cam[:, idx], s=10, c=col, label=f'{wvl}nm')
        ax.set_xlabel('Camera Ice Fraction', fontsize=14)
        ax.set_ylabel('Surface Albedo', fontsize=14)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=12)
        ax.set_title(f'Surface Albedo at Different Wavelengths vs Camera Ice Fraction, {date_label}', fontsize=13)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{prefix}_wvl_icefraction.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

        # 8. Slope and R² vs wavelength
        wvl_slope = np.full_like(alb_wvl, np.nan)
        wvl_r2    = np.full_like(alb_wvl, np.nan)
        for i in range(len(alb_wvl)):
            mask = np.isfinite(alb_cam[:, i]) & np.isfinite(cam_ice_frac)
            if mask.sum() > 2:
                res_i        = linregress(cam_ice_frac[mask], alb_cam[:, i][mask])
                wvl_slope[i] = res_i.slope
                wvl_r2[i]    = res_i.rvalue ** 2
        fig, ax = plt.subplots(figsize=(8, 6))
        ax2 = ax.twinx()
        ax.scatter(alb_wvl, wvl_slope, c=wvl_r2, s=10, cmap='jet', vmin=0, vmax=1)
        ax2.plot(alb_wvl, alb_avg, color='k', alpha=0.5)
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Slope', fontsize=14)
        ax2.set_ylabel('Avg Surface Albedo', fontsize=14)
        ax2.legend(['Avg Surface Albedo'], fontsize=10)
        ax.set_xlim(350, 2000)
        ax.tick_params(labelsize=12)
        for band in gas_bands:
            ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
        ax.set_title(f'Correlation between Surface Albedo and Camera Ice Fraction vs Wavelength, {date_label}', fontsize=13)
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{prefix}_wvl_icefraction_correlation.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

        return alb_avg

    # ---------------------------------------------------------------
    # Case study: individual low-altitude legs with camera ice fraction
    # Each leg: (date_str, cam_nc_file, cam_offset_hr, t1, t2, date_label, prefix)
    # ---------------------------------------------------------------
    case_study_legs = [
        ('20240801', 'cam_ice_fraction_20240801_134836_140900.nc', -0.750/3600,
         13.84,  14.12,  'Aug 1st', 'arcsix_albedo_0801_clear'),
        ('20240801', 'cam_ice_fraction_20240801_144200_150600.nc',  0.0,
         14.739, 15.053, 'Aug 1st', 'arcsix_albedo_0801_clear_2'),
        ('20240802', 'cam_ice_fraction_20240802_143112_150748.nc',  0.0,
         14.557, 15.100, 'Aug 2nd', 'arcsix_albedo_0802_clear_1'),
        ('20240802', 'cam_ice_fraction_20240802_151312_163936.nc',  0.1/3600,
         15.244, 16.635, 'Aug 2nd', 'arcsix_albedo_0802_clear_2'),
    ]

    leg_results = {}  # prefix -> (cam_t, cam_ice, bb_cam) for comparison plot

    for date_str, cam_nc, cam_offset, t1, t2, date_label, prefix in case_study_legs:
        alb_wvl, lon, lat, alt, time_ssfr, alb, bb_alb = _load_leg_data(
            combined_data, date_str, ['clear_atm_corr', 'clear_atm_corr_2'])

        # Time filter
        tmask     = (time_ssfr >= t1) & (time_ssfr <= t2)
        lon       = lon[tmask];  lat      = lat[tmask];    alt    = alt[tmask]
        time_ssfr = time_ssfr[tmask];  alb = alb[tmask, :];  bb_alb = bb_alb[tmask]

        # Load and filter camera data
        with Dataset(cam_nc, 'r') as nc:
            cam_t_raw   = nc.variables['tmhr'][:]
            cam_ice_raw = nc.variables['ice_fraction'][:]
        cam_t = cam_t_raw + cam_offset
        cmask = (cam_t >= t1) & (cam_t <= t2)
        cam_t = cam_t[cmask];  cam_ice = cam_ice_raw[cmask]

        # Vectorized time matching
        bb_cam, alb_cam = _match_cam_to_ssfr(cam_t, time_ssfr, bb_alb, alb)

        # Standard diagnostic plots
        _plot_leg(time_ssfr, bb_alb, lat, lon, alt, alb, alb_wvl,
                  cam_t, cam_ice, bb_cam, alb_cam, date_label, fig_dir, prefix)

        leg_results[prefix] = (cam_t, cam_ice, bb_cam)

    # ---------------------------------------------------------------
    # Aug 1 leg 1: special combined figure with annotated camera images
    # ---------------------------------------------------------------
    alb_wvl, lon, lat, alt, time_ssfr, alb, bb_alb = _load_leg_data(
        combined_data, '20240801', ['clear_atm_corr', 'clear_atm_corr_2'])
    tmask     = (time_ssfr >= 13.84) & (time_ssfr <= 14.12)
    time_ssfr = time_ssfr[tmask];  bb_alb = bb_alb[tmask];  alb = alb[tmask, :]
    cam_t, cam_ice, bb_cam = leg_results['arcsix_albedo_0801_clear']

    mask            = np.isfinite(bb_cam) & np.isfinite(cam_ice)
    X_fit           = sm.add_constant(cam_ice[mask])
    models_fig      = {q: sm.QuantReg(bb_cam[mask], X_fit).fit(q=q) for q in [0.05, 0.5, 0.95]}
    slope_fig, intercept_fig, r_value_fig, _, _ = linregress(cam_ice[mask], bb_cam[mask])
    sorted_cam_ice  = np.sort(cam_ice[mask])

    base_date      = datetime(2024, 8, 1)
    time_ssfr_dt   = [base_date + timedelta(hours=t) for t in time_ssfr]
    cam_t_dt       = [base_date + timedelta(hours=t) for t in cam_t]

    fig = plt.figure(figsize=(18, 9.5))
    gs_fig = GridSpec(3, 16, height_ratios=[1, 1, 1.5], hspace=0.5, wspace=0.6, figure=fig)
    ax11   = fig.add_subplot(gs_fig[:2, :8])
    ax11_2 = ax11.twinx()
    ax12   = fig.add_subplot(gs_fig[:2, 9:])
    ax21   = fig.add_subplot(gs_fig[2, :4])
    ax22   = fig.add_subplot(gs_fig[2, 4:8])
    ax23   = fig.add_subplot(gs_fig[2, 8:12])
    ax24   = fig.add_subplot(gs_fig[2, 12:])

    l1 = ax11.scatter(time_ssfr_dt, bb_alb, label='Broadband albedo', c='b', s=10)
    l2 = ax11_2.scatter(cam_t_dt, cam_ice, label='camera SIC', c='r', s=5)
    ax11.set_xlabel('Time (UTC)', fontsize=14)
    ax11.set_ylabel('Broadband Albedo', fontsize=14, color='b')
    ax11_2.set_ylabel('Camera Sea Ice Fraction', fontsize=14, color='r')
    ax11_2.set_ylim(-0.05, 1.05)
    ax11.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=10,
                loc='center left', bbox_to_anchor=(0.15, 0.1))
    ax11.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax11.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    image_times = [13+53/60+55/3600, 13+56/60+47/3600, 13+59/60+11/3600, 14+4/60+24/3600]
    img_colors  = ['purple', 'green', 'orange', 'brown']
    directions  = ['down', 'up', 'leftup', 'rightup']
    for it, direction, color in zip(image_times, directions, img_colors):
        y_v = bb_alb[np.argmin(np.abs(time_ssfr - it))]
        ap  = dict(facecolor=color, edgecolor=color, shrink=0.05, width=2, headwidth=8)
        t_dt = base_date + timedelta(hours=it)
        if direction == 'down':
            ax11.annotate('', xy=(t_dt, y_v+0.01), xytext=(t_dt, y_v+0.09), arrowprops=ap)
        elif direction == 'up':
            ax11.annotate('', xy=(t_dt, y_v-0.01), xytext=(t_dt, y_v-0.09), arrowprops=ap)
        elif direction == 'leftup':
            ax11.annotate('', xy=(t_dt+timedelta(minutes=0.5), y_v-0.01),
                          xytext=(t_dt+timedelta(minutes=1.5), y_v-0.09), arrowprops=ap)
        elif direction == 'rightup':
            ax11.annotate('', xy=(t_dt-timedelta(minutes=0.5), y_v-0.01),
                          xytext=(t_dt-timedelta(minutes=1.5), y_v-0.09), arrowprops=ap)
    ax11.tick_params(labelsize=12)

    ax12.scatter(cam_ice[mask], bb_cam[mask], s=10, c='k')
    ax12.plot(sorted_cam_ice, slope_fig*sorted_cam_ice + intercept_fig, color='red', linestyle='--',
              label=r'Linear Fit: y=%.2fx+%.2f, $\mathrm{R^2}$=%.2f' % (slope_fig, intercept_fig, r_value_fig**2))
    ax12.fill_between(sorted_cam_ice,
                      models_fig[0.05].predict(sm.add_constant(sorted_cam_ice)),
                      models_fig[0.95].predict(sm.add_constant(sorted_cam_ice)),
                      color='coral', alpha=0.3, label='Quantile Regression (5th-95th)')
    ax12.set_xlabel('Camera Sea Ice Fraction', fontsize=14)
    ax12.set_ylabel('Broadband Albedo', fontsize=14)
    ax12.legend(fontsize=10)
    ax12.tick_params(labelsize=12)

    img_crop = (slice(0, 1980), slice(500, 2500))
    for ax_img, fn, title, col in [
        (ax21, f'{_fdir_general_}/camera/20240801/Capture_02436_13_53_54Z.jpg', '13:53:54 UTC', img_colors[0]),
        (ax22, f'{_fdir_general_}/camera/20240801/Capture_02584_13_56_47Z.jpg', '13:56:47 UTC', img_colors[1]),
        (ax23, f'{_fdir_general_}/camera/20240801/Capture_02712_13_59_11Z.jpg', '13:59:11 UTC', img_colors[2]),
        (ax24, f'{_fdir_general_}/camera/20240801/Capture_02991_14_04_24Z.jpg', '14:04:24 UTC', img_colors[3]),
    ]:
        img = plt.imread(fn)
        ax_img.imshow(img[img_crop[0], img_crop[1], :])
        ax_img.set_title(title, color=col, fontsize=14)
        ax_img.axis('off')

    for ax_cap, cap in zip([ax11, ax12, ax21, ax22, ax23, ax24],
                            ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
        ax_cap.text(0, 1.025, cap, transform=ax_cap.transAxes, fontsize=16, va='bottom', ha='left')
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_broadband_icefraction_combined.png',
                bbox_inches='tight', dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Combined comparison: all 4 legs on one scatter plot
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    alpha = 0.7
    for prefix, label, color, zorder in [
        ('arcsix_albedo_0801_clear',   '0801 13.84-14.12 (0.1km)', 'k', 3),
        ('arcsix_albedo_0801_clear_2', '0801 14.74-15.05 (0.6km)', 'g', 2),
        ('arcsix_albedo_0802_clear_1', '0802 14.56-15.10 (0.1km)', 'b', 1),
        ('arcsix_albedo_0802_clear_2', '0802 15.24-16.63 (1.0km)', 'r', 0),
    ]:
        cam_t_leg, cam_ice_leg, bb_leg = leg_results[prefix]
        ax.scatter(cam_ice_leg, bb_leg, s=10, c=color, label=label, alpha=alpha, zorder=zorder)
    ax.set_xlabel('Camera Ice Fraction', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10)
    ax.set_title('Surface Albedo vs Camera Ice Fraction', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_0802_broadband_icefraction.png', bbox_inches='tight', dpi=150)
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
