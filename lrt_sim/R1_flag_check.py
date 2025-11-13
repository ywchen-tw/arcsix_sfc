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





from enum import IntFlag, auto
class ssfr_flags(IntFlag):
    pitcth_roll_exceed_threshold = auto()  #condition when the pitch or roll angle exceeded a certain threshold
    camera_icing = auto()  #condition when the camera experienced icing issues at the moment of measurement
    camera_icing_pre = auto()  #condition when the camera experienced icing issues within 1 hour prior to the moment of measurement
    zen_toa_over_threshold = auto()  #condition when the zenith TOA irradiance is over a certain threshold 
    alp_ang_pit_issue = auto()  #condition when the leveling platform angle is over a certain threshold
    alp_ang_rol_issue = auto()  #condition when the leveling platform angle is over a certain threshold


def find_continuous_time_ranges(time_array, max_gap=1/3600):
    """Find continuous time ranges in a time array given a maximum gap"""
    if len(time_array) == 0:
        return []
    
    time_array = np.sort(time_array)
    ranges = []
    start_time = time_array[0]
    prev_time = time_array[0]
    
    for current_time in time_array[1:]:
        if current_time - prev_time > max_gap:
            end_time = prev_time
            ranges.append((start_time, end_time))
            start_time = current_time
        prev_time = current_time
    
    # Add the last range
    ranges.append((start_time, prev_time))
    
    return ranges

def flt_flat_check(date=datetime.datetime(2024, 5, 31),
                  config: Optional[FlightConfig] = None):
    """Flight TOA reflectance check plot"""
    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    
    output_dir = f'fig/zen_flux_ang_issue_check'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1) Load all instrument & satellite metadata
    data_ssfr = load_h5(config.ssfr(date_s))
    
    
    # Evaluation
    tmhr_ssfr = data_ssfr['time']/3600.0  # convert to hours

    wvl_zen = data_ssfr['wvl_dn']
    flux_zen = data_ssfr['f_dn']
    
    w1, w2, w3 = 550, 860, 1600
    w1_ind = np.argmin(np.abs(wvl_zen - w1))
    w2_ind = np.argmin(np.abs(wvl_zen - w2))
    w3_ind = np.argmin(np.abs(wvl_zen - w3))
    w1_flux = flux_zen[:, w1_ind]
    w2_flux = flux_zen[:, w2_ind]
    w3_flux = flux_zen[:, w3_ind]
    
    d550nmflux_dt = np.gradient(w1_flux, tmhr_ssfr*3600.0)  # W/m2/nm/s
    
    
    dpitch_diff_dt = data_ssfr['dpitch_diff_dt']
    droll_diff_dt = data_ssfr['droll_diff_dt']
    
    
    icing = (data_ssfr['flag'] & ssfr_flags.camera_icing) != 0
    pitch_roll_exceed = (data_ssfr['flag'] & ssfr_flags.pitcth_roll_exceed_threshold) != 0
    alp_ang_pit_issue = (data_ssfr['flag'] & ssfr_flags.alp_ang_pit_issue) != 0
    alp_ang_rol_issue = (data_ssfr['flag'] & ssfr_flags.alp_ang_rol_issue) != 0
    
    alp_ang_pit_issue_start, alp_ang_pit_issue_end = [], []
    alp_ang_rol_issue_start, alp_ang_rol_issue_end = [], []
    
    alp_ang_pit_issue_start_, alp_ang_pit_issue_end_ = [], []
    alp_ang_rol_issue_start_, alp_ang_rol_issue_end_ = [], []
    
    
    ang_change_threshold = 0.5  # degrees per second
    
    
    alp_ang_pit_issue = np.abs(dpitch_diff_dt) >= ang_change_threshold  # degrees per second
    alp_ang_rol_issue = np.abs(droll_diff_dt) >= ang_change_threshold
    alp_hsk_ang_pit_issue_ind = np.where(alp_ang_pit_issue)[0]
    alp_hsk_ang_rol_issue_ind = np.where(alp_ang_rol_issue)[0]

    
    # surrounding std check
    extend_surrounding_seconds = 60  # seconds
    flux_550_rel_std_rate_threshold = 0.01  
    for ind in alp_hsk_ang_pit_issue_ind:
        start_ind = max(0, ind - extend_surrounding_seconds)
        end_ind = min(ind + extend_surrounding_seconds, len(tmhr_ssfr) - 1)
        
        t_mask = (tmhr_ssfr >= tmhr_ssfr[start_ind]) & (tmhr_ssfr <= tmhr_ssfr[end_ind])
        if np.abs(dpitch_diff_dt)[ind] < np.nanstd(dpitch_diff_dt[t_mask])*2:
            alp_hsk_ang_pit_issue_ind = np.array(alp_hsk_ang_pit_issue_ind[alp_hsk_ang_pit_issue_ind != ind])
        if (np.nanstd(w1_flux[t_mask])/np.nanmean(w1_flux[t_mask])) < flux_550_rel_std_rate_threshold:
            alp_hsk_ang_pit_issue_ind = alp_hsk_ang_pit_issue_ind[alp_hsk_ang_pit_issue_ind != ind]
    for ind in alp_hsk_ang_rol_issue_ind:
        start_ind = max(0, ind - extend_surrounding_seconds)
        end_ind = min(ind + extend_surrounding_seconds, len(tmhr_ssfr) - 1)
        t_mask = (tmhr_ssfr >= tmhr_ssfr[start_ind]) & (tmhr_ssfr <= tmhr_ssfr[end_ind])
        if np.abs(droll_diff_dt)[ind] < np.nanstd(dpitch_diff_dt[t_mask])*3:
            alp_hsk_ang_rol_issue_ind = np.array(alp_hsk_ang_rol_issue_ind[alp_hsk_ang_rol_issue_ind != ind])
        if (np.nanstd(w1_flux[t_mask])/np.nanmean(w1_flux[t_mask])) < flux_550_rel_std_rate_threshold:
            alp_hsk_ang_rol_issue_ind = alp_hsk_ang_rol_issue_ind[alp_hsk_ang_rol_issue_ind != ind]
         
            
    extend_indices = 10  # extend seconds before or after
    dflux_550_std_rate_range = 0.005  # W/m2/nm/s
    dflux_550_mean_rate_thres = 0.005  # W/m2/nm/s

    for ind in alp_hsk_ang_pit_issue_ind:
        start_ind = max(0, ind - extend_indices)
        end_ind = min(ind + extend_indices, len(tmhr_ssfr) - 1)
        
        t_mask = (tmhr_ssfr >= tmhr_ssfr[start_ind]) & (tmhr_ssfr <= tmhr_ssfr[end_ind])
        condition_1 = np.nanstd(d550nmflux_dt[t_mask]) < dflux_550_std_rate_range
        condition_2 = np.all(np.isnan(d550nmflux_dt[t_mask]))
        if np.any([condition_1, condition_2]):
            # remove ind from the array
            alp_hsk_ang_pit_issue_ind = np.array(alp_hsk_ang_pit_issue_ind[alp_hsk_ang_pit_issue_ind != ind])
            alp_ang_pit_issue[ind] = False
            start_time, end_time = tmhr_ssfr[start_ind], tmhr_ssfr[end_ind]
            alp_ang_pit_issue_start_.append(start_time)
            alp_ang_pit_issue_end_.append(end_time)
    for ind in alp_hsk_ang_rol_issue_ind:
        start_ind = max(0, ind - extend_indices)
        end_ind = min(ind + extend_indices, len(tmhr_ssfr) - 1)
        t_mask = (tmhr_ssfr >= tmhr_ssfr[start_ind]) & (tmhr_ssfr <= tmhr_ssfr[end_ind])
        condition_1 = np.nanstd(d550nmflux_dt[t_mask]) < dflux_550_std_rate_range
        condition_2 = np.all(np.isnan(d550nmflux_dt[t_mask]))        
        if np.any([condition_1, condition_2]):
            alp_hsk_ang_rol_issue_ind = np.array(alp_hsk_ang_rol_issue_ind[alp_hsk_ang_rol_issue_ind != ind])
            alp_ang_rol_issue[ind] = False
            start_time, end_time = tmhr_ssfr[start_ind], tmhr_ssfr[end_ind]
            alp_ang_rol_issue_start_.append(start_time)
            alp_ang_rol_issue_end_.append(end_time)
                

    # reset the flags
    alp_ang_pit_issue[:] = False
    alp_ang_rol_issue[:] = False
    for ind in alp_hsk_ang_pit_issue_ind:
        start_ind = max(0, ind - extend_indices)
        end_ind = min(ind + extend_indices, len(tmhr_ssfr) - 1)
        alp_ang_pit_issue[start_ind:end_ind+1] = True
    for ind in alp_hsk_ang_rol_issue_ind:
        start_ind = max(0, ind - extend_indices)
        end_ind = min(ind + extend_indices, len(tmhr_ssfr) - 1)
        alp_ang_rol_issue[start_ind:end_ind+1] = True
            
    # (flags & COND_A) != 0
    
    
    
    
    if np.any(alp_ang_pit_issue):
        alp_ang_pit_issue_times = tmhr_ssfr[alp_ang_pit_issue]
        for start_time, end_time in find_continuous_time_ranges(alp_ang_pit_issue_times, max_gap=1.5/3600):                
            alp_ang_pit_issue_start.append(start_time)
            alp_ang_pit_issue_end.append(end_time)
    if np.any(alp_ang_rol_issue):
        alp_ang_rol_issue_times = tmhr_ssfr[alp_ang_rol_issue]
        for start_time, end_time in find_continuous_time_ranges(alp_ang_rol_issue_times, max_gap=1.5/3600):
            alp_ang_rol_issue_start.append(start_time)
            alp_ang_rol_issue_end.append(end_time)


    
    

    # slice tmhr_ssfr into every 1 hour for plotting
    time_bins = np.arange(tmhr_ssfr[0], tmhr_ssfr[-1]+0.01, 1.0)
    for j in range(len(time_bins)-1):
        time_mask = (tmhr_ssfr >= time_bins[j]) & (tmhr_ssfr < time_bins[j+1])
        if np.sum(time_mask) == 0 or (not np.any(alp_ang_pit_issue[time_mask]) and not np.any(alp_ang_rol_issue[time_mask])):
            continue
            
    
        # Plotting SSFR zenith mean clear-sky fluxes and TOA comparison
        plt.close('all')
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 12), sharex=True)


        ax1.plot(tmhr_ssfr[time_mask], w1_flux[time_mask], label=f'SSFR Zenith {w1}nm (all)', color='red', linewidth=0.25)
        ax1.plot(tmhr_ssfr[time_mask], w2_flux[time_mask], label=f'SSFR Zenith {w2}nm (all)', color='green', linewidth=0.25)
        ax1.plot(tmhr_ssfr[time_mask], w3_flux[time_mask], label=f'SSFR Zenith {w3}nm (all)', color='blue', linewidth=0.25)
        
        w1_flux_pitch_roll_exceed = copy.deepcopy(w1_flux)
        w2_flux_pitch_roll_exceed = copy.deepcopy(w2_flux)
        w3_flux_pitch_roll_exceed = copy.deepcopy(w3_flux)
        
        w1_flux_pitch_roll_exceed[pitch_roll_exceed] = np.nan
        w2_flux_pitch_roll_exceed[pitch_roll_exceed] = np.nan
        w3_flux_pitch_roll_exceed[pitch_roll_exceed] = np.nan
        ax1.plot(tmhr_ssfr[time_mask], w1_flux_pitch_roll_exceed[time_mask], label=f'SSFR Zenith {w1}nm (exclude large pitch/roll)', color='red', linewidth=1.5)
        ax1.plot(tmhr_ssfr[time_mask], w2_flux_pitch_roll_exceed[time_mask], label=f'SSFR Zenith {w2}nm (exclude large pitch/roll)', color='green', linewidth=1.5)
        ax1.plot(tmhr_ssfr[time_mask], w3_flux_pitch_roll_exceed[time_mask], label=f'SSFR Zenith {w3}nm (exclude large pitch/roll)',  color='blue', linewidth=1.5)
        ax1.set_ylabel('Downwelling Flux (W m$^{-2}$ nm$^{-1}$)')
        

        ax2.plot(tmhr_ssfr[time_mask], dpitch_diff_dt[time_mask], label='Pitch Angle Rate of Change', color='brown')
        ax2.set_xlabel('Time (hr)')
        ax2.set_ylabel('Pitch Angle Rate of Change (deg/s)')
        ang_range = ang_change_threshold
        ax2.axhspan(-ang_range, ang_range, color='gray', alpha=0.2, label='Normal Range')
        ax2.set_ylim(-3, 3)
        
        ax3.plot(tmhr_ssfr[time_mask], droll_diff_dt[time_mask], label='Roll Angle Rate of Change', color='blue')
        # ax3.set_xlabel('Time (hr)')
        ax3.set_ylabel('Roll Angle Rate of Change (deg/s)')
        ang_range = ang_change_threshold
        ax3.axhspan(-ang_range, ang_range, color='gray', alpha=0.2, label='Normal Range')
        ax3.set_ylim(-3, 3)
        
        ax4.plot(tmhr_ssfr[time_mask], d550nmflux_dt[time_mask], label='dF550nm/dt', color='black')
        ax4.set_xlabel('Time (hr)')
        ax4.set_ylabel('dF550nm/dt (W m$^{-2}$ nm$^{-1}$ s$^{-1}$)')
        flux_550_std_rate_range = dflux_550_std_rate_range
        ax4.axhspan(-flux_550_std_rate_range, flux_550_std_rate_range, color='gray', alpha=0.2, label='Normal Range')
        ax4.set_ylim(-0.1, 0.1)
        
        for i in range(len(alp_ang_pit_issue_start)):
            for ax in [ax1, ax2, ax4]:
                ax.axvspan(alp_ang_pit_issue_start[i], alp_ang_pit_issue_end[i], color='orange', alpha=0.4, label='Leveling Platform Pitch Issue' if i==0 else "")
        for i in range(len(alp_ang_pit_issue_start_)):
            for ax in [ax1, ax2, ax4]:
                ax.axvspan(alp_ang_pit_issue_start_[i], alp_ang_pit_issue_end_[i], color='pink', alpha=0.05 , label='Leveling Platform Pitch Issue (not reported)' if i==0 else "")

            
        for i in range(len(alp_ang_rol_issue_start)):
            for ax in [ax1, ax3, ax4]:
                ax.axvspan(alp_ang_rol_issue_start[i], alp_ang_rol_issue_end[i], color='purple', alpha=0.25 , label='Leveling Platform Roll Issue' if i==0 else "")
        for i in range(len(alp_ang_rol_issue_start_)):
            for ax in [ax1, ax3, ax4]:
                ax.axvspan(alp_ang_rol_issue_start_[i], alp_ang_rol_issue_end_[i], color='lightblue', alpha=0.05, label='Leveling Platform Roll Issue (not reported)' if i==0 else "")
        
                
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(time_bins[j], time_bins[j+1])
            ax.grid()
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        
        fig.suptitle(f'{date_s} SSFR Zenith', fontsize=16)# y=0.98)
        fig.tight_layout()
        fig.savefig('{}/flt_flat_check_{}_ang_issue_time_{:.2f}_{:.2f}.png'.format(output_dir, date_s, time_bins[j], time_bins[j+1]), dpi=300)
        

        
        


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
    
    for date in [
                 datetime.datetime(2024, 5, 28),
                 datetime.datetime(2024, 5, 30),
                 datetime.datetime(2024, 5, 31),
                 datetime.datetime(2024, 6, 3),
                 datetime.datetime(2024, 6, 5),
                 datetime.datetime(2024, 6, 6),
                 datetime.datetime(2024, 6, 7),
                 datetime.datetime(2024, 6, 10),
                 datetime.datetime(2024, 6, 11),
                 datetime.datetime(2024, 6, 13),
                 datetime.datetime(2024, 7, 25),
                 datetime.datetime(2024, 7, 29),
                 datetime.datetime(2024, 7, 30),
                 datetime.datetime(2024, 8,  1),
                 datetime.datetime(2024, 8,  2),
                 datetime.datetime(2024, 8,  7),
                 datetime.datetime(2024, 8,  8),
                 datetime.datetime(2024, 8,  9),
                 datetime.datetime(2024, 8,  15),
                 ]:
        flt_flat_check(date=date,
                    config=config)


    