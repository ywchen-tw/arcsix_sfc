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

# from util.util import *
# from util.arcsix_atm import prepare_atmospheric_profile
from util import *

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


def cre_sim(date=datetime.datetime(2024, 5, 31),
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
                     lw=False,
                     manual_alb=None,
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


    log.info("ssfr filename:", config.ssfr(date_s))
    

    # atmospheric profile setting
    #/----------------------------------------------------------------------------\#
    dropsonde_file_list, dropsonde_date_list, dropsonde_tmhr_list, _, _ = dropsonde_time_loc_list(dir_dropsonde=f'{_fdir_general_}/dropsonde')
    
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

        del dropsonde_file_list, dropsonde_date_list, dropsonde_tmhr_list, dropsonde_tmhr_array, dropsonde_idx
        gc.collect()

    zpt_filedir = f'{_fdir_general_}/zpt/{date_s}'
    os.makedirs(zpt_filedir, exist_ok=True)
    if levels is None:
        levels = np.concatenate((np.arange(0, 0.26, 0.05), 
                                         np.arange(0.3, 1., 0.1), 
                                         np.arange(1., 2.1, 0.2), 
                                        np.arange(2.5, 4.1, 0.5), 
                                        np.arange(5.0, 10.1, 2.5),
                                        np.array([15, 20, 30., 40., 50.])))
    
    
    import platform
    # run lower resolution on Mac for testing, higher resolution on Linux cluster
    if platform.system() == 'Darwin':
        xx_wvl_grid_sw = np.arange(300, 4000.1, 600.0)
        xx_wvl_grid_lw = np.arange(5000, 100000.1, 5000.0)
    elif platform.system() == 'Linux':
        xx_wvl_grid_sw = np.arange(300, 4000.1, 2.5)
        xx_wvl_grid_lw = np.arange(5000, 100000.1, 10.0)
        

    if not os.path.exists('wvl_grid_test_cre_sw.dat') or not os.path.exists('wvl_grid_test_cre_lw.dat'):
        write_2col_file('wvl_grid_test_cre_sw.dat', xx_wvl_grid_sw, np.zeros_like(xx_wvl_grid_sw),
                        header=('# SSFR Wavelength grid test file\n'
                                '# wavelength (nm)\n'))
        write_2col_file('wvl_grid_test_cre_lw.dat', xx_wvl_grid_lw, np.zeros_like(xx_wvl_grid_lw),
                        header=('# SSFR Wavelength grid test file\n'
                                '# wavelength (nm)\n'))
    
    # write out the convolved solar flux
    #/----------------------------------------------------------------------------\#
    # Kurudz solar spectrum has a resolution of 0.5 nm
    if 1:#not os.path.exists('arcsix_ssfr_solar_flux_raw_cre.dat'):
        # use Kurudz solar spectrum
        # df_solor = pd.read_csv('kurudz_0.1nm.dat', sep='\s+', header=None)
        # use CU solar spectrum
        df_solor = pd.read_csv('CU_composite_solar_processed.dat', sep='\s+', header=None)
        wvl_solar = np.array(df_solor.iloc[:, 0])
        flux_solar = np.array(df_solor.iloc[:, 1])#/1000 # convert mW/m^2/nm to W/m^2/nm
        
        # interpolate to 1 nm grid
        f_interp = interp1d(wvl_solar, flux_solar, kind='linear', bounds_error=False, fill_value=0.0)
        wvl_solar_interp = np.arange(250, 4550.1, 1.0)
        flux_solar_interp = f_interp(wvl_solar_interp)
        
        mask = wvl_solar_interp <= 4500

        wvl_solar = wvl_solar_interp[mask]
        flux_solar = flux_solar_interp[mask]
        
        
        write_2col_file('arcsix_ssfr_solar_flux_raw_cre.dat', wvl_solar, flux_solar,
                        header=('# SSFR version solar flux without slit function convolution\n'
                                '# wavelength (nm)      flux (mW/m^2/nm)\n'))

    del xx_wvl_grid_sw, xx_wvl_grid_lw
    del df_solor, wvl_solar_interp, flux_solar_interp, f_interp
    gc.collect()
    
    # read satellite granule
    #/----------------------------------------------------------------------------\#
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)
        
    processed_dir = f'{_fdir_general_}/sfc_alb_combined_smooth_450nm'
    os.makedirs(processed_dir, exist_ok=True)
    
    lon_all = np.array([])
    lat_all = np.array([])
    alt_all = np.array([])
    sza_all = np.array([])
    saa_all = np.array([])
    sfc_T = np.array([])
    
    time_all = np.array([])
    marli_all_h = np.array([])
    marli_all_wvmr = np.array([])
    
    init = True
    alb_iter2_all = None
    
    for i in range(len(tmhr_ranges_select)):
        time_start, time_end = tmhr_ranges_select[i][0], tmhr_ranges_select[i][-1]
        
        fname_cld_obs_info = '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, time_start, time_end)
        # print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
        
        processed_file = f'{processed_dir}/sfc_alb_update_{date_s}_{case_tag}_time_{tmhr_ranges_select[0][0]:.3f}_{tmhr_ranges_select[-1][-1]:.3f}.pkl'
        
        if os.path.exists(processed_file):
        
            with open(fname_cld_obs_info, 'rb') as f:
                vars()[f"cld_leg_{i}"] = pickle.load(f)  
                
            
            with open(processed_file, 'rb') as f:
                vars()[f"processed_leg_{i}"] = pickle.load(f)
                
    
    
            alb_wvl = vars()[f"processed_leg_{i}"]['wvl']
            
            time_all = np.concatenate((time_all, vars()[f"processed_leg_{i}"]['time_all']))
            lon_all = np.concatenate((lon_all, vars()[f"processed_leg_{i}"]['lon_all']))
            lat_all = np.concatenate((lat_all, vars()[f"processed_leg_{i}"]['lat_all']))
            alt_all = np.concatenate((alt_all, vars()[f"processed_leg_{i}"]['alt_all']))
            sza_all = np.concatenate((sza_all, vars()[f"cld_leg_{i}"]['sza']))
            saa_all = np.concatenate((saa_all, vars()[f"cld_leg_{i}"]['saa']))
            sfc_T = np.concatenate((sfc_T, vars()[f"cld_leg_{i}"]['kt19_sfc_T']))
            
            if vars()[f"cld_leg_{i}"]['marli_h'] is not None:
                marli_all_h = np.concatenate((marli_all_h, vars()[f"cld_leg_{i}"]['marli_h']))
                marli_all_wvmr = np.concatenate((marli_all_wvmr, vars()[f"cld_leg_{i}"]['marli_wvmr']))
            
            if init:
                alb_iter2_all = vars()[f"processed_leg_{i}"]['alb_iter2_all']
                init = False
            else:
                alb_iter2_all = np.vstack((alb_iter2_all, vars()[f"processed_leg_{i}"]['alb_iter2_all']))
            
        else:
            print(f"Processed file {processed_file} not found. Skipping leg {i}.")
        
    
    
    
    sfc_T += 273.15  # convert to K
    
    lon_avg = np.round(np.mean(lon_all), 2)
    lat_avg = np.round(np.mean(lat_all), 2)
    lon_min, lon_max = np.round(np.min(lon_all), 2), np.round(np.max(lon_all), 2)
    lat_min, lat_max = np.round(np.min(lat_all), 2), np.round(np.max(lat_all), 2)
    alt_avg = np.round(np.nanmean(alt_all), 2)  # in km
    sza_avg = np.round(np.nanmean(sza_all), 2)
    saa_avg = np.round(np.nanmean(saa_all), 2)
    sfc_T_avg = np.round(np.nanmean(sfc_T), 2)
    sfc_T_std = np.round(np.nanstd(sfc_T), 2)
    
    # print(f"{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}")
    # print(f"sfc T (degC): {sfc_T_avg - 273.15:.2f} +/- {sfc_T_std:.2f}")
    # return None


    
    if marli_all_h.size == 0:
        cld_marli = {'marli_h': None,
                     'marli_wvmr': None}
    else:
        marli_h_set_sorted = np.sort(list(set(marli_all_h)))
        marli_wvmr_avg = []
        for h in marli_h_set_sorted:
            marli_wvmr_avg.append(np.nanmean(marli_all_wvmr[marli_all_h == h]))
        marli_wvmr_avg = np.array(marli_wvmr_avg)
        cld_marli = {'marli_h': marli_h_set_sorted,
                    'marli_wvmr': marli_wvmr_avg}
           
        
    if clear_sky:
        fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_clear'
        fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
    else:
        fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_sat_cloud'
        fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'
        
    
    del lon_all, lat_all, alt_all, sza_all, saa_all, sfc_T, marli_all_h, marli_all_wvmr
    gc.collect()
    
    
    mode = 'lw' if lw else 'sw'
    # sza_arr = np.array([50, 52.5, 55, 57.5, 60, 62.5, 65, 67.5, 70, 71.5, 72.5, 73, 73.5, 75, 77.5, 80, 82.5, 85, 87, sza_avg], dtype=np.float32)
    # sza_arr = np.array([50, 52.5, 55, 57.5, 60, 62.5, 65, 67.5, 70, 71.5, 72.5, 73, 73.5, 75, sza_avg], dtype=np.float32)
    # sza_arr = np.array([50, 52.5, 55, 57.5, 60, 62.5, 65, 67.5], dtype=np.float32)
    # sza_arr = np.array([70, 71.5, 72.5, 73, 73.5, 75, sza_avg], dtype=np.float32)
    # sza_arr = np.array([62.5, 65, 67.5, 75, sza_avg], dtype=np.float32)
    # sza_arr = np.array([50, 52.5, 55, 57.5,], dtype=np.float32)
    # sza_arr = np.array([60, 62.5, 65, 67.5], dtype=np.float32)
    # sza_arr = np.array([70, 71.5,], dtype=np.float32)
    # sza_arr = np.array([72.5, 73], dtype=np.float32)
    # sza_arr = np.array([73.5, 75, sza_avg], dtype=np.float32)
    sza_arr = np.array([50, 52.5,], dtype=np.float32)
    # sza_arr = np.array([55, 57.5,], dtype=np.float32)
    # sza_arr = np.array([60, 62.5], dtype=np.float32)
    # sza_arr = np.array([65, 67.5], dtype=np.float32)
    # sza_arr = np.array([70, 71.5,], dtype=np.float32)
    # sza_arr = np.array([72.5, 73], dtype=np.float32)
    # sza_arr = np.array([73.5, 75], dtype=np.float32)
    # sza_arr = np.array([sza_avg], dtype=np.float32)
    
    for sza_sim in sza_arr:
    
        if manual_alb is None:
            output_csv_name = f'{fdir}/ssfr_simu_flux_{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}_alt-{alt_avg:.2f}km_cre_{mode}_sza_{sza_sim:.2f}_0.99.csv'
        else:
            output_csv_name = f'{fdir}/ssfr_simu_flux_{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}_alt-{alt_avg:.2f}km_cre_{mode}_sza_{sza_sim:.2f}_alb-manual-{manual_alb.replace(".dat", "")}_0.99.csv'
        

        os.makedirs(fdir_tmp, exist_ok=True)
        os.makedirs(fdir, exist_ok=True)
        

        alb_corr_fit_avg = np.nanmean(alb_iter2_all, axis=0)

        
        
        if not os.path.exists(output_csv_name) or overwrite_lrt:
            print('Preparing atmospheric profile ...')
            # =================================================================================
            prepare_atmospheric_profile(_fdir_general_, date_s, case_tag, 0, date, 
                                        time_start=time_all[0], time_end=time_all[-1],
                                        alt_avg=alt_avg, data_dropsonde=data_dropsonde,
                                        cld_leg=cld_marli, levels=levels,
                                        mod_extent=[lon_min, lon_max, lat_min, lat_max],
                                        zpt_filedir=f'{_fdir_general_}/zpt/{date_s}',
                                        sfc_T=sfc_T_avg,
                                        )
            
            # =================================================================================
            
            
            # write out the surface albedo
            #/----------------------------------------------------------------------------\#
            sfc_alb_dir = f'{_fdir_general_}/sfc_alb_cre'
            os.makedirs(sfc_alb_dir, exist_ok=True)
            
            if manual_alb is None:
                alb_fname = f'{sfc_alb_dir}/sfc_alb_{date_s}_{time_all[0]:.3f}_{time_all[-1]:.3f}_{alt_avg:.2f}km_cre_alb.dat'
            
                if 1:#not os.path.exists(alb_fname):
                
                    ext_wvl, ext_alb = alb_extention(alb_wvl, alb_corr_fit_avg, clear_sky=clear_sky)
                
                    # ext_alb *= 1.012  # apply scaling factor
            
                    ext_alb = np.clip(ext_alb, 0.0, 1.0)  # ensure albedo is between 0 and 1

                    write_2col_file(alb_fname, ext_wvl, ext_alb,
                                    header=('# SSFR derived sfc albedo\n'
                                            '# wavelength (nm)      albedo (unitless)\n'))
                
                else:
                    ext_alb = pd.read_csv(alb_fname, delim_whitespace=True, comment='#', header=None).iloc[:, 1].values
                    ext_wvl = pd.read_csv(alb_fname, delim_whitespace=True, comment='#', header=None).iloc[:, 0].values
            else:
                alb_fname = f'{sfc_alb_dir}/{manual_alb}'
                if not os.path.exists(alb_fname):
                    raise FileNotFoundError(f"Manual albedo file {alb_fname} not found.")
                else:
                    with open(alb_fname, 'r') as f:
                        header = f.readline()

                    ext_alb = pd.read_csv(alb_fname, delim_whitespace=True, comment='#', header=None).iloc[:, 1].values
                    ext_wvl = pd.read_csv(alb_fname, delim_whitespace=True, comment='#', header=None).iloc[:, 0].values
            
            
            
            alb_mean = np.round(np.nanmean(ext_alb[(ext_wvl >= 400) & (ext_wvl <= 2500)]), 5)
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ext_wvl, ext_alb, label='Extended Surface Albedo')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Albedo')
            ax.set_xlim(300, 4000)
            fig.tight_layout()
            if manual_alb is None:
                fig.savefig(f'fig/{date_s}/{date_s}_{case_tag}_{time_all[0]:.3f}_{time_all[-1]:.3f}_{alt_avg:.2f}km_cre_alb.png', bbox_inches='tight', dpi=150)
            else:
                fig.savefig(f'fig/{date_s}/{date_s}_{case_tag}_manual_{manual_alb.replace(".dat", "")}_{time_all[0]:.3f}_{time_all[-1]:.3f}_{alt_avg:.2f}km_cre_alb.png', bbox_inches='tight', dpi=150)
            # plt.show()
            #\----------------------------------------------------------------------------/#
            # # use CU solar spectrum
            # df_solor = pd.read_csv('CU_composite_solar_processed.dat', sep='\s+', header=None)
            # wvl_solar = np.array(df_solor.iloc[:, 0])
            # flux_solar = np.array(df_solor.iloc[:, 1])#/1000 # convert mW/m^2/nm to W/m^2/nm
            
            # # interpolate to 1 nm grid
            # f_interp = interp1d(wvl_solar, flux_solar, kind='linear', bounds_error=False, fill_value=0.0)
            # flux_solar_interp = f_interp(ext_wvl)
            # broadband_alb = np.trapz(ext_alb * flux_solar_interp, ext_wvl) / np.trapz(flux_solar_interp, ext_wvl)
            # print(f"alb file name: {alb_fname}, mean alb (400-2500nm): {alb_mean:.3f}, broadband alb: {broadband_alb:.5f}")
            # sys.exit()
            
            
            atm_z_grid = levels
            z_list = atm_z_grid
            atm_z_grid_str = ' '.join(['%.3f' % z for z in atm_z_grid])

            if platform.system() == 'Darwin':
                cwp_list = [0, 5, 10, 30, 50, 100, 200]  # g/m^2
            elif platform.system() == 'Linux':
                cwp_list = [0, 1, 2, 3, 5, 7.5, 10, 15, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 600]  # g/m^2
            
            cwp_list.append(manual_cloud_cwp*1000)  # convert kg/m^2 to g/m^2
            cwp_list = np.array(cwp_list)/1000  # convert to kg/m^2
            rho_liquid_water = 1000  # kg/m^3
            cot_list = 3/(2 * manual_cloud_cer * 1e-6 * rho_liquid_water) * cwp_list  # from cwp to cot
            
            flux_output = np.zeros((len(cot_list), 2)) * np.nan  # down, up 
            
            
            inits_rad = []
            flux_key_ix = []
            output_list = []
            for ix in range(len(cot_list)):
                flux_key_all = []
                
                flux_down_results = []
                flux_down_dir_results = []
                flux_down_diff_results = []
                flux_up_results = []
                
                flux_key = np.zeros_like(flux_output, dtype=object)
                
                # rt initialization
                #/----------------------------------------------------------------------------\#
                lrt_cfg = copy.deepcopy(er3t.rtm.lrt.get_lrt_cfg())
                fname_zpt = f'atm_profiles_{date_s}_{case_tag}_{time_all[0]:.3f}_{time_all[-1]:.3f}_{alt_avg:.2f}km.dat'
                fname_ch4 = f'ch4_profiles_{date_s}_{case_tag}_{time_all[0]:.3f}_{time_all[-1]:.3f}_{alt_avg:.2f}km.dat'
                atm_file = os.path.join(zpt_filedir, fname_zpt)
                ch4_file = os.path.join(zpt_filedir, fname_ch4)
                
                lrt_cfg['atmosphere_file'] = atm_file
                lrt_cfg['mol_abs_param'] = 'reptran coarse'
                
                import platform
                # run less streams on Mac for testing, higher resolution on Linux cluster
                Nstreams = 4
                if platform.system() == 'Darwin':
                    Nstreams = 4
                elif platform.system() == 'Linux':
                    Nstreams = 8
                lrt_cfg['number_of_streams'] = Nstreams
                if not lw:
                    # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')
                    lrt_cfg['solar_file'] = 'arcsix_ssfr_solar_flux_raw_cre.dat'
                    # lrt_cfg['solar_file'] = lrt_cfg['solar_file'].replace('kurudz_0.1nm.dat', 'kurudz_1.0nm.dat')
                    
                    input_dict_extra_general = {
                                        'crs_model': 'rayleigh Bodhaine29',
                                        'albedo_file': alb_fname,
                                        'mol_file': 'CH4 %s' % ch4_file,
                                        'wavelength_grid_file': 'wvl_grid_test_cre_sw.dat',
                                        # 'wavelength_add' : '300 4000',
                                        'atm_z_grid': atm_z_grid_str,
                                        'output_process': 'integrate',
                                        }
                    Nx_effective = 1
                    mute_list = ['albedo', 'wavelength', 'spline', 'slit_function_file']
                else:

                    input_dict_extra_general = {
                                        'source': 'thermal',
                                        'albedo_add': '0.01',  # emissivity = 0.99
                                        'atm_z_grid': atm_z_grid_str,
                                        'mol_file': f'CH4 {ch4_file}',
                                        # 'wavelength_grid_file': 'wvl_grid_test_cre_lw.dat',
                                        'wavelength_add' : '4000 100000',
                                        'output_process': 'integrate',
                                        }
                    Nx_effective = 1 # integrate over all wavelengths
                    mute_list = ['albedo', 'wavelength', 'spline', 'source solar', 'slit_function_file']
                #/----------------------------------------------------------------------------/#

                

                

                input_dict_extra = copy.deepcopy(input_dict_extra_general)

                cot_x = cot_list[ix]
                cwp_x = cwp_list[ix]
                if cot_x > 0:
                    cer_x = manual_cloud_cer
                    cth_x = manual_cloud_cth
                    cbh_x = manual_cloud_cbh
                    cgt_x = cth_x - cbh_x

                    cth_ind_cld = bisect.bisect_left(z_list, cth_x)
                    cbh_ind_cld = bisect.bisect_left(z_list, cbh_x)
                    
                    fname_cld = f'{fdir_tmp}/cld_{ix:04d}_{date_s}_{case_tag}_{time_all[0]:.3f}_{time_all[-1]:.3f}_{alt_avg:.2f}km_sza_{sza_sim}_alb_{alb_mean}.txt'
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
                
                    flux_key[ix] = dict_key
                else:
                    cld_cfg = None
                    dict_key_arr = np.array([0.0, 0.0, alt_avg])
                    dict_key = '_'.join([f'{i:.3f}' for i in dict_key_arr])
                    flux_key[ix] = dict_key
                
                if (cld_cfg is None) and (dict_key in flux_key_all):
                    flux_key_ix.append(dict_key)
                elif (cld_cfg is not None) and (dict_key in flux_key_all):
                    flux_key_ix.append(dict_key)
                else:
                    input_dict_extra_alb = copy.deepcopy(input_dict_extra)
                    init = er3t.rtm.lrt.lrt_init_mono_flx(
                            input_file  = f'{fdir_tmp}/input_{ix:04d}_{date_s}_{case_tag}_{time_all[0]:.3f}_{time_all[-1]:.3f}_{alt_avg:.2f}km_sza_{sza_sim}_alb_{alb_mean}_{mode}.txt',
                            output_file = f'{fdir_tmp}/output_{ix:04d}_{date_s}_{case_tag}_{time_all[0]:.3f}_{time_all[-1]:.3f}_{alt_avg:.2f}km_sza_{sza_sim}_alb_{alb_mean}_{mode}.txt',
                            date        = date,
                            # surface_albedo=0.08,
                            solar_zenith_angle = sza_sim,
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
                    output_list.append(f'{fdir_tmp}/output_{ix:04d}_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.txt')
                    flux_key_all.append(dict_key)
                    flux_key_ix.append(dict_key)
                    
            # # Run RT
            print(f"Start running libratran calculations for {output_csv_name.replace('.csv', '')}, mode: {mode}. alb mean: {alb_mean}")
            # #/----------------------------------------------------------------------------\#
            # check output file size
            output_file_check = f'{fdir_tmp}/output_0000_{date_s}_{case_tag}_{time_all[0]:.3f}_{time_all[-1]:.3f}_{alt_avg:.2f}km_sza_{sza_sim}_alb_{alb_mean}_{mode}.txt'
            run = True
            if (not overwrite_lrt) and os.path.exists(output_file_check):
                if os.path.getsize(output_file_check) > 100:
                    run = False
                
            
            import platform
            if platform.system() == 'Darwin':
                ##### run several libratran calculations in parallel
                if len(inits_rad) > 0:
                    print('Running libratran calculations ...')
                    if run:
                        # check available CPU cores
                        NCPU = np.max([os.cpu_count() - 2, 1])
                        er3t.rtm.lrt.lrt_run_mp(inits_rad, Ncpu=NCPU) 
                    for i in range(len(inits_rad)):
                        data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                        
                        flux_down_results.append(np.squeeze(data.f_down))
                        flux_down_dir_results.append(np.squeeze(data.f_down_direct))
                        flux_down_diff_results.append(np.squeeze(data.f_down_diffuse))
                        flux_up_results.append(np.squeeze(data.f_up))
            ##### run several libratran calculations one by one
            
            elif platform.system() == 'Linux':
                if len(inits_rad) > 0:
                    print('Running libratran calculations ...')
                    for i in range(len(inits_rad)):
                        if run:
                            er3t.rtm.lrt.lrt_run(inits_rad[i])
                            
                        try:
                            data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                        
                        except Exception as e:
                            print(f"Error reading output for index {i}, retrying once. Error: {e}")
                            # retry once
                            er3t.rtm.lrt.lrt_run(inits_rad[i])
                            data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                        
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


            flux_down_results = np.array(flux_down_results)
            flux_down_dir_results = np.array(flux_down_dir_results)
            flux_down_diff_results = np.array(flux_down_diff_results)
            flux_up_results = np.array(flux_up_results)
            

            
            # simulated fluxes at p3 altitude            
            Fup_p3 = flux_up_results[:, 1]
            Fdn_p3 = flux_down_results[:, 1]
            
            # simulated fluxes at sfc
            Fup_sfc = flux_up_results[:, 0]
            Fdn_sfc = flux_down_results[:, 0]
            

            
            mode = 'sw' if not lw else 'lw'
            print(f"Saving simulated fluxes to {output_csv_name} in {mode} mode")
            output_dict = {
                'sza': [sza_sim]*len(cot_list),
                'cot': cot_list,
                'cwp': cwp_list,
                'cer': [manual_cloud_cer]*len(cot_list),
                'cth': [manual_cloud_cth]*len(cot_list),
                'cbh': [manual_cloud_cbh]*len(cot_list),
                'Fup_sfc': Fup_sfc,
                'Fdn_sfc': Fdn_sfc,
            }
            
            output_df = pd.DataFrame(output_dict)
            output_df.to_csv(output_csv_name, index=False)
            

            del output_dict, output_df
            del Fup_p3, Fdn_p3
            del Fup_sfc, Fdn_sfc
            del ext_alb, ext_wvl
            del flux_down_results, flux_down_dir_results, flux_down_diff_results, flux_up_results
            
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


    # surface albedo derivation
    # --------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------

    atm_corr_overwrite_lrt = False
    lw = False  # shortwave
    


    # cre_sim(date=datetime.datetime(2024, 6, 3),
    #                 tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                     ],
    #                 case_tag='cloudy_atm_corr_1',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                         np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=13.0,
    #                 manual_cloud_cwp=77.82/1000,
    #                 manual_cloud_cth=1.93,
    #                 manual_cloud_cbh=1.41,
    #                 manual_cloud_cot=21.27,
    #                 lw=lw,
    #                 )
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240606_16.250_16.950_0.50km_cre_alb.dat',
    #                     )
        
    for lw in [False, True]:
        cre_sim(date=datetime.datetime(2024, 6, 3),
                        tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
                                            ],
                        case_tag='cloudy_atm_corr_1',
                        config=config,
                        levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
                                                np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
                                                np.arange(5.0, 10.1, 2.5),
                                                np.array([15, 20, 30., 40., 45.]))),
                        simulation_interval=0.5,
                        clear_sky=False,
                        overwrite_lrt=atm_corr_overwrite_lrt,
                        manual_cloud=True,
                        manual_cloud_cer=13.0,
                        manual_cloud_cwp=77.82/1000,
                        manual_cloud_cth=1.93,
                        manual_cloud_cbh=1.41,
                        manual_cloud_cot=21.27,
                        lw=lw,
                        manual_alb='sfc_alb_20240607_15.336_15.761_0.12km_cre_alb.dat',
                        )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240603_13.620_13.750_0.32km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240613_16.550_17.581_0.22km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240725_15.094_15.300_0.11km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240807_13.344_13.761_0.13km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240613_14.109_14.140_0.11km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240725_15.881_15.903_0.33km_cre_alb.dat',
    #                     )
        
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240605_12.422_13.812_5.80km_cre_alb.dat',
    #                     )
        
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240528_15.610_17.404_0.22km_cre_alb_ori.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240528_15.610_17.404_0.22km_cre_alb_scale_0.99X.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_ori.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_scale_0.97X.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                             np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=13.0,
    #                     manual_cloud_cwp=77.82/1000,
    #                     manual_cloud_cth=1.93,
    #                     manual_cloud_cbh=1.41,
    #                     manual_cloud_cot=21.27,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_scale_1.012X.dat',
    #                     )
    
    
    
    
    
    
    


    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240606_16.250_16.950_0.50km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb=None,
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240603_13.620_13.750_0.32km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240613_16.550_17.581_0.22km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240725_15.094_15.300_0.11km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240807_13.344_13.761_0.13km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240613_14.109_14.140_0.11km_cre_alb.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240725_15.881_15.903_0.33km_cre_alb.dat',
    #                     )
        
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240605_12.422_13.812_5.80km_cre_alb.dat',
    #                     )
        
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240528_15.610_17.404_0.22km_cre_alb_ori.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240528_15.610_17.404_0.22km_cre_alb_scale_0.99X.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_ori.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_scale_0.97X.dat',
    #                     )
        
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_scale_1.012X.dat',
    #                     )
        










    
    # cre_sim(date=datetime.datetime(2024, 6, 6),
    #                 tmhr_ranges_select=[[16.250, 16.325], # 100m, 
    #                                     [16.375, 16.632], # 450m
    #                                     [16.700, 16.794], # 100m
    #                                     [16.850, 16.952], # 1.2km
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 lw=lw,
    #                 )
    

    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb=None,
    #                     )
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240725_15.094_15.300_0.11km_cre_alb.dat',
    #                     )
    
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240613_16.550_17.581_0.22km_cre_alb.dat',
    #                     )
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240606_16.250_16.950_0.50km_cre_alb.dat',
    #                     )
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240603_13.620_13.750_0.32km_cre_alb.dat',
    #                     )
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240807_13.344_13.761_0.13km_cre_alb.dat',
    #                     )
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240613_14.109_14.140_0.11km_cre_alb.dat',
    #                     )
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240725_15.881_15.903_0.33km_cre_alb.dat',
    #                     )
    
    # new    
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240605_12.422_13.812_5.80km_cre_alb.dat',
    #                     )
    
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240528_15.610_17.404_0.22km_cre_alb_ori.dat',
    #                     )
    
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240528_15.610_17.404_0.22km_cre_alb_scale_0.99X.dat',
    #                     )
    
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_ori.dat',
    #                     )
    
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_scale_0.97X.dat',
    #                     )
    
    
    # for lw in [False, True]:
    #     cre_sim(date=datetime.datetime(2024, 6, 7),
    #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    #                                             np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                             np.arange(5.0, 10.1, 2.5),
    #                                             np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=6.7,
    #                     manual_cloud_cwp=26.96/1000,
    #                     manual_cloud_cth=0.43,
    #                     manual_cloud_cbh=0.15,
    #                     manual_cloud_cot=6.02,
    #                     lw=lw,
    #                     manual_alb='sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_scale_1.012X.dat',
    #                     )
    






    # cre_sim(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[14.109, 14.140], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_1',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.4, 0.52, 0.6, 0.8, 1.0,]),
    #                                         np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=17.4,
    #                 manual_cloud_cwp=90.51,
    #                 manual_cloud_cth=0.52,
    #                 manual_cloud_cbh=0.15,
    #                 manual_cloud_cot=7.82,
    #                 lw=lw,
    #                 manual_alb=None,
    #                 )
    
    # cre_sim(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[15.834, 15.883], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.28, 0.3, 0.5, 0.58, 0.8, 1.0,]),
    #                                     np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                     np.arange(5.0, 10.1, 2.5),
    #                                     np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=22.4,
    #                 manual_cloud_cwp=35.6 ,
    #                 manual_cloud_cth=0.58,
    #                 manual_cloud_cbh=0.28,
    #                 manual_cloud_cot=2.39,
    #                 lw=lw,
    #                 manual_alb=None,
    #                 )
        

    # cre_sim(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[16.043, 16.067], # 100-200m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_3',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.38, 0.5, 0.68, 0.8, 1.0,]),
    #                                         np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=8.9,
    #                 manual_cloud_cwp=21.29,
    #                 manual_cloud_cth=0.68,
    #                 manual_cloud_cbh=0.38,
    #                 manual_cloud_cot=3.59,
    #                 lw=lw,
    #                 manual_alb=None,
    #                 )
    

    # cre_sim(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[16.550, 17.581], # 100-500m, clear
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 lw=lw,
    #                 manual_alb=None,
    #                 )

    # cre_sim(date=datetime.datetime(2024, 7, 25),
    #                 tmhr_ranges_select=[[15.094, 15.300], # 100m, some low clouds or fog below
    #                                     ],
    #                 case_tag='cloudy_atm_corr',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,]),
    #                                         np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=11.4,
    #                 manual_cloud_cwp=9.94,
    #                 manual_cloud_cth=0.30,
    #                 manual_cloud_cbh=0.16,
    #                 manual_cloud_cot=1.31,
    #                 )
    

    # cre_sim(date=datetime.datetime(2024, 7, 25),
    #                 tmhr_ranges_select=[[15.881, 15.903], # 200-500m
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,]),
    #                                         np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=11.4,
    #                 manual_cloud_cwp=9.94,
    #                 manual_cloud_cth=0.30,
    #                 manual_cloud_cbh=0.16,
    #                 manual_cloud_cot=1.31,
    #                 lw=lw,
    #                 )
        
    # cre_sim(date=datetime.datetime(2024, 8, 1),
    #                 tmhr_ranges_select=[[13.843, 14.361], # 100-450m, clear, some open ocean
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 lw=lw,
    #                 )



    # cre_sim(date=datetime.datetime(2024, 8, 1),
    #                 tmhr_ranges_select=[
    #                                     [14.739, 15.053], # 550m
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 lw=lw,
    #                 )
    



    # cre_sim(date=datetime.datetime(2024, 8, 2),
    #                 tmhr_ranges_select=[
    #                                     [14.557, 15.100], # 100m
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 lw=lw,
    #                 )



    # cre_sim(date=datetime.datetime(2024, 8, 2),
    #                 tmhr_ranges_select=[
    #                                     [15.244, 16.635], # 1km
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 lw=lw,
    #                 )

    # cre_sim(date=datetime.datetime(2024, 8, 7),
    #                 tmhr_ranges_select=[[13.344, 13.763], # 100m, cloudy
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.65, 0.69, 0.78, 1.0,]),
    #                                         np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=10.7,
    #                 manual_cloud_cwp=11.28,
    #                 manual_cloud_cth=0.78,
    #                 manual_cloud_cbh=0.69,
    #                 manual_cloud_cot=1.59,
    #                 lw=lw,
    #                 manual_alb=None,
    #                 )
    
 
    # cre_sim(date=datetime.datetime(2024, 8, 7),
    #                 tmhr_ranges_select=[
    #                                     [15.472, 15.567], # 180m, cloudy
    #                                     [15.580, 15.921], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.62, 0.8, 0.96,]),
    #                                        np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                        np.arange(5.0, 10.1, 2.5),
    #                                        np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 lw=lw,
    #                 )
    


    # cre_sim(date=datetime.datetime(2024, 8, 8),
    #                 tmhr_ranges_select=[
    #                                     [13.212, 13.347], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_1',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.67, 0.8, 1.0,]),
    #                                         np.array([1.5, 1.98, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=False,
    #                 manual_cloud_cer=15.3,
    #                 manual_cloud_cwp=143.94,
    #                 manual_cloud_cth=1.98,
    #                 manual_cloud_cbh=0.67,
    #                 manual_cloud_cot=14.12,
    #                 lw=lw,
    #                 )
    

    # cre_sim(date=datetime.datetime(2024, 8, 8),
    #                 tmhr_ranges_select=[
    #                                     [15.314, 15.504], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.78, 1.0,]),
    #                                         np.array([1.5, 1.81, 2.21, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=False,
    #                 manual_cloud_cer=7.8,
    #                 manual_cloud_cwp=64.18,
    #                 manual_cloud_cth=2.21,
    #                 manual_cloud_cbh=1.81,
    #                 manual_cloud_cot=12.41,
    #                 lw=lw,
    #                 )
    

    # cre_sim(date=datetime.datetime(2024, 8, 9),
    #                 tmhr_ranges_select=[
    #                                     [13.376, 13.600], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_1',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.34, 0.4, 0.6, 0.77, 1.0,]),
    #                                         np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=9.0,
    #                 manual_cloud_cwp=83.49,
    #                 manual_cloud_cth=0.77,
    #                 manual_cloud_cbh=0.34,
    #                 manual_cloud_cot=13.93,
    #                 lw=lw,
    #                 )
    


    # cre_sim(date=datetime.datetime(2024, 8, 9),
    #                 tmhr_ranges_select=[
    #                                     [16.029, 16.224], # 100m, cloudy
    #                                     ],
    #                 case_tag='cloudy_atm_corr_2',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.29, 0.4, 0.62, 0.8, 1.0,]),
    #                                         np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=8.3,
    #                 manual_cloud_cwp=49.10,
    #                 manual_cloud_cth=0.62,
    #                 manual_cloud_cbh=0.29,
    #                 manual_cloud_cot=8.93,
    #                 lw=lw,
    #                 )
        
    

    # cre_sim(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[[12.405, 13.812], # 5.7m,
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )
    


    # cre_sim(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[
    #                                     [14.258, 15.036], # 100m
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # cre_sim(date=datetime.datetime(2024, 6, 5),
    #                 tmhr_ranges_select=[
    #                                     [15.535, 15.931], # 450m
    #                                     ],
    #                 case_tag='clear_atm_corr_3',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )
    
    
    
    # cre_sim(date=datetime.datetime(2024, 6, 11),
    #                 tmhr_ranges_select=[
    #                                     [14.968, 15.229], # 100, clear, some cloud
    #                                     [14.968, 15.347],
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # cre_sim(date=datetime.datetime(2024, 6, 11),
    #                 tmhr_ranges_select=[
    #                                     [15.347, 15.813], # 100m
    #                                     [15.813, 16.115], # 100-450m, clear, some cloud
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # cre_sim(date=datetime.datetime(2024, 6, 13),
    #                 tmhr_ranges_select=[[13.704, 13.817], # 100-450m, clear, some cloud
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )
    
    # cre_sim(date=datetime.datetime(2024, 5, 28),
    #                 tmhr_ranges_select=[[15.610, 15.822],
    #                                     [16.905, 17.404] 
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # cre_sim(date=datetime.datetime(2024, 5, 31),
    #                 tmhr_ranges_select=[[13.839, 15.180],  # 5.6 km
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )

    

    # cre_sim(date=datetime.datetime(2024, 5, 31),
    #                 tmhr_ranges_select=[
    #                                     [16.905, 17.404] 
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )