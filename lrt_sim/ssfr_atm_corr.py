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
    

def ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_ranges_select, date_s, case_tag, pitch_roll_thres=3.0):
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
    
    pitch_roll_mask = np.sqrt(pitch_ang**2 + roll_ang**2) < pitch_roll_thres  # pitch and roll greater < pitch_roll_thres
    pitch_ang_valid = pitch_ang.copy()
    roll_ang_valid = roll_ang.copy()
    pitch_ang_valid[~pitch_roll_mask] = np.nan
    roll_ang_valid[~pitch_roll_mask] = np.nan
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
    
    # hsr1_530_570_thresh = 0.18
    # cloud_mask_hsr1 = hsr1_diff_ratio_530_570_mean > hsr1_530_570_thresh
    # ssfr_fup_tmhr[cloud_mask_hsr1] = np.nan
    # ssfr_fdn_tmhr[cloud_mask_hsr1] = np.nan
    
    icing = (data_ssfr['flag'] & ssfr_flags.camera_icing) != 0
    pitch_roll_exceed = (data_ssfr['flag'] & ssfr_flags.pitcth_roll_exceed_threshold) != 0
    alp_ang_pit_rol_issue = (data_ssfr['flag'] & ssfr_flags.alp_ang_pit_rol_issue) != 0
    
    alp_ang_pit_rol_issue_tmhr = alp_ang_pit_rol_issue[t_ssfr_tmhr_mask]
    ssfr_fup_tmhr[alp_ang_pit_rol_issue_tmhr] = np.nan
    ssfr_fdn_tmhr[alp_ang_pit_rol_issue_tmhr] = np.nan
    
    
    ssfr_zen_550_ind = np.argmin(np.abs(ssfr_zen_wvl - 550))
    ssfr_nad_550_ind = np.argmin(np.abs(ssfr_nad_wvl - 550))
    
    time_start, time_end = t_ssfr_tmhr[0], t_ssfr_tmhr[-1]
    
    fig, (ax10, ax20) = plt.subplots(2, 1, figsize=(16, 12))
    ax11 = ax10.twinx()
    
    
    l1 = ax10.plot(t_ssfr, ssfr_fdn[:, ssfr_zen_550_ind], '--', color='k', alpha=0.85)
    l2 = ax10.plot(t_ssfr, ssfr_fup[:, ssfr_nad_550_ind], '--', color='k', alpha=0.85)
    
    ax10.plot(t_ssfr_tmhr, ssfr_fdn_tmhr[:, ssfr_zen_550_ind], 'r-', label='SSFR Down 550nm', linewidth=3)
    ax10.plot(t_ssfr_tmhr, ssfr_fup_tmhr[:, ssfr_nad_550_ind], 'b-', label='SSFR Up 550nm', linewidth=3)
    
    l3 = ax11.plot(t_hsr1_tmhr, hsr_dif_ratio[:, hsr_550_ind], 'm-', label='HSR1 Diff Ratio 550nm')
    ax11.set_ylabel('HSR1 Diff Ratio 550nm', fontsize=14)
    
    # ax2.plot(t_hsk, data_hsk['ang_hed'], 'r-', label='HSK Heading')
    ax20.plot(t_hsk, pitch_ang, 'g-', label='HSK Pitch', linewidth=1.0)
    ax20.plot(t_hsk, roll_ang, 'b-', label='HSK Roll', linewidth=1.0)
    ax20.plot(t_hsk, pitch_ang_valid, 'g-', linewidth=3.0)
    ax20.plot(t_hsk, roll_ang_valid, 'b-', linewidth=3.0)
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
    fig.savefig('fig/%s/%s_%s_ssfr_pitch_roll_heading_550nm_time_%.2f-%.2f.png' % (date_s, date_s, case_tag, time_start, time_end), bbox_inches='tight', dpi=150)

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
    

def ice_alb_fitting(alb_wvl, alb_corr, alt):
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
    
    alb_corr_mask_to_ice_alb_ratio = gas_abs_masking(fit_wvl, alb_corr_mask_to_ice_alb_ratio, alt=alt)
    
                
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
    data_hsr1 = load_h5(config.hsr1(date_s))
    
    if date_s != '20240603':
        # MARLI netCDF
        with Dataset(str(config.marli(date_s))) as ds:
            data_marli = {var: ds.variables[var][:] for var in ("time","Alt","H","T","LSR","WVMR")}
    else:
        data_marli = {'time': np.array([]), 'Alt': np.array([]), 'H': np.array([]), 'T': np.array([]), 'LSR': np.array([]), 'WVMR': np.array([])}
    
    log.info("ssfr filename:", config.ssfr(date_s))
    
    # plot ssfr time series for checking sable legs selection
    ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_ranges_select, date_s, case_tag, pitch_roll_thres=3.0)

    # Build leg masks
    t_hsk = np.array(data_hsk["tmhr"])
    leg_masks = [(t_hsk>=lo)&(t_hsk<=hi) for lo,hi in tmhr_ranges_select]
    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_hsr1 = data_hsr1['time']/3600.0  # convert to hours
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
        if clear_sky:
            xx_wvl_grid = np.arange(350, 2000.1, 2.5)
        else:
            xx_wvl_grid = np.arange(350, 2000.1, 10.0)
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

    # read satellite granule
    #/----------------------------------------------------------------------------\#
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)
    fname_cld_obs_info = '%s/%s_cld_obs_info_%s_%s_%s_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag)
    if iter==0:      
        
        # Loop legs: load raw NC, apply cloud logic, interpolate, plot
        for i, mask in enumerate(leg_masks):
            
            # find index arrays in one go
            times_leg = t_hsk[mask]
            print(f"Leg {i+1}: time range {times_leg.min()}-{times_leg.max()}h")
            
            sel_ssfr, sel_hsr1 = (
                nearest_indices(t_hsk, mask, arr)
                for arr in (t_ssfr, t_hsr1)
            )
            
            if len(t_marli) > 0:
                sel_marli = nearest_indices(t_hsk, mask, t_marli)
            

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
        
            pitch_roll_mask = np.sqrt(data_hsk["ang_pit"][mask]**2 + data_hsk["ang_rol"][mask]**2) < 3.0
            ssfr_zen_flux[~pitch_roll_mask, :] = np.nan
            ssfr_nad_flux_interp[~pitch_roll_mask, :] = np.nan
            ssfr_zen_toa[~pitch_roll_mask, :] = np.nan
            
            # hsr1_530nm_ind = np.argmin(np.abs(leg['hsr1_wvl'] - 530.0))
            # hsr1_570nm_ind = np.argmin(np.abs(leg['hsr1_wvl'] - 570.0))
            # hsr1_diff_ratio = data_hsr1["f_dn_dif"][sel_hsr1]/data_hsr1["f_dn_tot"][sel_hsr1]
            # hsr1_diff_ratio_530_570_mean = np.nanmean(hsr1_diff_ratio[:, hsr1_530nm_ind:hsr1_570nm_ind+1], axis=1)
            # hsr1_530_570_thresh = 0.18
            # cloud_mask_hsr1 = hsr1_diff_ratio_530_570_mean > hsr1_530_570_thresh
            # ssfr_zen_flux[cloud_mask_hsr1, :] = np.nan
            # ssfr_nad_flux_interp[cloud_mask_hsr1, :] = np.nan
            
            icing = (data_ssfr['flag'] & ssfr_flags.camera_icing) != 0
            icing_pre = (data_ssfr['flag'] & ssfr_flags.camera_icing_pre) != 0
            pitch_roll_exceed = (data_ssfr['flag'] & ssfr_flags.pitcth_roll_exceed_threshold) != 0
            alp_ang_pit_rol_issue = (data_ssfr['flag'] & ssfr_flags.alp_ang_pit_rol_issue) != 0
            
            alp_ang_pit_rol_issue_tmhr = alp_ang_pit_rol_issue[sel_ssfr]
            ssfr_zen_flux[alp_ang_pit_rol_issue_tmhr] = np.nan
            ssfr_nad_flux_interp[alp_ang_pit_rol_issue_tmhr] = np.nan
            
            
            leg['ssfr_zen'] = ssfr_zen_flux
            leg['ssfr_nad'] = ssfr_nad_flux_interp
            leg['ssfr_zen_wvl'] = ssfr_zen_wvl
            leg['ssfr_nad_wvl'] = ssfr_zen_wvl
            leg['ssfr_toa'] = ssfr_zen_toa
            
            leg['ssfr_icing'] = icing
            leg['ssfr_icing_pre'] = icing_pre
            

            vars()["cld_leg_%d" % i] = leg
            
            time_start, time_end = leg['time'][0], leg['time'][-1]
        
            # save the cloud observation information to a pickle file
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, time_start, time_end)
            with open(fname_pkl, 'wb') as f:
                pickle.dump(vars()["cld_leg_%d" % i], f, protocol=pickle.HIGHEST_PROTOCOL)

            del leg  # free memory
            del sel_ssfr, sel_hsr1
            gc.collect()
        
    else:
        print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
        for i in range(len(tmhr_ranges_select)):
            time_start, time_end = tmhr_ranges_select[i][0], tmhr_ranges_select[i][-1]
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, time_start, time_end)
            with open(fname_pkl, 'rb') as f:
                vars()[f"cld_leg_{i}"] = pickle.load(f)  
                
    # return None 
    
    
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

        mod_extent=[np.round(np.nanmin(cld_leg['lon']), 2), 
                                                        np.round(np.nanmax(cld_leg['lon']), 2),
                                                        np.round(np.nanmin(cld_leg['lat']), 2),
                                                        np.round(np.nanmax(cld_leg['lat']), 2)]
        print("mod_extent:", mod_extent)
        
        if not os.path.exists(fname_h5) or overwrite_lrt: 
            if iter==0:
                prepare_atmospheric_profile(_fdir_general_, date_s, case_tag, ileg, date, time_start, time_end,
                                            alt_avg, data_dropsonde,
                                            cld_leg, levels=levels,
                                            mod_extent=[np.round(np.nanmin(cld_leg['lon']), 2), 
                                                        np.round(np.nanmax(cld_leg['lon']), 2),
                                                        np.round(np.nanmin(cld_leg['lat']), 2),
                                                        np.round(np.nanmax(cld_leg['lat']), 2)],
                                            zpt_filedir=f'{_fdir_general_}/zpt/{date_s}'
                                            )
            # =================================================================================
            
            
            # write out the surface albedo
            #/----------------------------------------------------------------------------\#
            os.makedirs(f'{_fdir_general_}/sfc_alb', exist_ok=True)
            iter_0_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_0.dat'
            if 1:#not os.path.exists(iter_0_fname):
                
                alb_wvl = np.concatenate(([348.0], cld_leg['ssfr_zen_wvl'], [2050.]))
                alb_avg = np.nanmean(cld_leg['ssfr_nad']/cld_leg['ssfr_zen'], axis=0)
                # print("cld_leg['ssfr_nad']:", np.nanmean(cld_leg['ssfr_nad'], axis=0))
                # print("cld_leg['ssfr_zen']:", np.nanmean(cld_leg['ssfr_zen'], axis=0))
                
                if np.all(np.isnan(alb_avg)):
                    raise ValueError(f"All nadir/zenith ratios are NaN for leg {ileg+1}, cannot compute average albedo")
                alb_avg[alb_avg<0.0] = 0.0
                alb_avg[alb_avg>1.0] = 1.0
                # alb_avg[np.isnan(alb_avg)] = 0.0
                
                if np.any(np.isnan(alb_avg)):
                    s = pd.Series(alb_avg)
                    s_mask = np.isnan(alb_avg)
                    # Fills NaN with the value immediately preceding it
                    s_ffill = s.fillna(method='ffill', limit=2)
                    s_ffill = s_ffill.fillna(method='bfill', limit=2)
                    while np.any(np.isnan(s_ffill)):
                        s_ffill = s_ffill.fillna(method='ffill', limit=2)
                        s_ffill = s_ffill.fillna(method='bfill', limit=2)
                        
                    alb_avg[s_mask] = s_ffill[s_mask]
                        
                alb_avg_extend = np.concatenate(([alb_avg[0]], alb_avg, [alb_avg[-1]]))

                write_2col_file(iter_0_fname, alb_wvl, alb_avg_extend,
                                header=('# SSFR derived sfc albedo\n'
                                        '# wavelength (nm)      albedo (unitless)\n'))
                
                # plt.close('all')
                # plt.figure(figsize=(8, 5))
                # plt.plot(alb_wvl, np.nanmean(cld_leg['ssfr_nad'], axis=0), label='Nadir Radiance')
                # plt.plot(alb_wvl, np.nanmean(cld_leg['ssfr_zen'], axis=0), label='Zenith Radiance')
                # plt.plot(alb_wvl, np.nanmean(cld_leg['ssfr_toa'], axis=0), label='Top-of-Atmosphere')
                # plt.xlabel('Wavelength (nm)')
                # plt.ylabel('Radiance (W/m$^2$/sr/nm)')
                # plt.show()
                
                # plt.close('all')
                # plt.figure(figsize=(8, 5))
                # plt.plot(alb_wvl, alb_avg, label='Derived Surface Albedo')
                # plt.xlabel('Wavelength (nm)')
                # plt.ylabel('Albedo (unitless)')
                # sys.exit()
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
                import platform
                # run less streams on Mac for testing, higher resolution on Linux cluster
                if platform.system() == 'Darwin':
                    lrt_cfg['number_of_streams'] = 4
                elif platform.system() == 'Linux':
                    lrt_cfg['number_of_streams'] = 8
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
                        cld_cfg['cloud_altitude'] = z_list[cbh_ind_cld:cth_ind_cld+1]
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
            # for prefix in ['input', 'output', 'cld']:
            #     for filename in glob.glob(os.path.join(fdir_tmp, f'{prefix}_*.txt')):
            #         os.remove(filename)
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
            
            if iter == 0:
                alb_obs = np.nanmean(cld_leg['ssfr_nad']/cld_leg['ssfr_zen'], axis=0)
                alb_obs[alb_obs<0.0] = 0.0
                alb_obs[alb_obs>1.0] = 1.0
            else:
                alb_file = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_{iter}.dat'
                alb_data = np.loadtxt(alb_file, comments='#', skiprows=2)
                alb_obs = alb_data[:, 1][1:-1]
            
            alb_corr = alb_obs * (corr_dn/corr_up)
            alb_corr[:4] = alb_corr[4]
            alb_corr[alb_corr<0.0] = 0.0
            alb_corr[alb_corr>1.0] = 1.0
            
            alb_corr_mask = gas_abs_masking(alb_wvl, alb_corr, alt=alt_avg)
            
            alb_corr[np.isnan(alb_corr)] = alb_corr_mask[np.isnan(alb_corr)]
            
            # alb_ice_fit = ice_alb_fitting(alb_wvl, alb_corr, alt=alt_avg)
            alb_ice_fit = snowice_alb_fitting(alb_wvl, alb_corr, alt=alt_avg, clear_sky=clear_sky)
            

            
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

            
            if 1:#iter == 0:
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
                'corr_factor': (corr_dn/corr_up),
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
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
    #                 config=config,
    #                 )
    
    # atm_corr_plot(date=datetime.datetime(2024, 6, 6),
    #                 tmhr_ranges_select=[[13.99, 14.18], [14.26, 14.46]],
    #                 case_tag='cloudy_track_4_atm_corr_after',
    #                 simulation_interval=0.5,
    #                 config=config,
    #                 )
    
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.3400, 15.7583], [15.8403, 16.2653]],
    #                     case_tag='cloudy_track_2_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),
    #                     tmhr_ranges_select=[[16.076, 16.109],
    #                                         [16.123, 16.255]],
    #                     case_tag='cloudy_track_1_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[15.85, 15.882], [16.057, 16.060]],
    #                     case_tag='cloudy_track_1_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
    #                 config=config,
    #                 )
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[15.85, 15.882], [16.057, 16.060]],
    #                     case_tag='cloudy_track_2_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
    #                 config=config,
    #                 )
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    #                     tmhr_ranges_select=[[16.0555, 16.0585], [16.207, 16.213]],
    #                     case_tag='cloudy_track_3_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
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
    #                 simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                 simulation_interval=0.5, # in minute
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
    #                 simulation_interval=0.5, # in minute
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    # --------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------

    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 5, 28),
    #                     tmhr_ranges_select=[[15.610, 15.822],
    #                                         [16.905, 17.404] 
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
    #                     tmhr_ranges_select=[
    #                                         [13.7889, 13.8010],
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
    #                     case_tag='clear_sky_spiral_atm_corr',
    #                     config=config,
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
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
    #                     tmhr_ranges_select=[[16.250, 16.325], # 100m, 
    #                                         [16.375, 16.632], # 450m
    #                                         [16.700, 16.794], # 100m
    #                                         [16.850, 16.952], # 1.2km
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
        flt_trk_atm_corr(date=datetime.datetime(2024, 6, 7),
                        tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
                                            ],
                        case_tag='cloudy_atm_corr',
                        config=config,
                        levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                               np.arange(5.0, 10.1, 2.5),
                                               np.array([15, 20, 30., 40., 45.]))),
                        simulation_interval=0.5,
                        clear_sky=False,
                        overwrite_lrt=True,
                        manual_cloud=True,
                        manual_cloud_cer=6.7,
                        manual_cloud_cwp=26.96,
                        manual_cloud_cth=0.43,
                        manual_cloud_cbh=0.15,
                        manual_cloud_cot=6.02,
                        iter=iter,
                        )
        
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 2),
    #                     tmhr_ranges_select=[
    #                                         [14.557, 15.100], # 100m
    #                                         [15.244, 16.635], # 1km
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),
    #                     tmhr_ranges_select=[[13.344, 13.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.65, 0.69, 0.78, 1.0,]),
    #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
    #                     tmhr_ranges_select=[
    #                                         [12.990, 13.180], # 180m, clear
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
    #                     tmhr_ranges_select=[
    #                                         [14.250, 14.373], # 180m, clear
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
    #                     tmhr_ranges_select=[
    #                                         [16.471, 16.601], # 180m, clear
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
        
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),
    #                     tmhr_ranges_select=[
    #                                         [14.750, 15.060], # 100m, clear
    #                                         [15.622, 15.887], # 100m, clear
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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
    #                     simulation_interval=0.5,
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
        
    
    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 15),
    #                     tmhr_ranges_select=[
    #                                         [14.085, 14.396], # 100m, clear
    #                                         [14.550, 14.968], # 3.5km, clear
    #                                         [15.078, 15.163], # 1.7km, clear
    #                                         ],
    #                     case_tag='clear_atm_corr',
    #                     config=config,
    #                     simulation_interval=0.5,
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