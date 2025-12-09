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


def write_2col_file(filename, wvl, val, header):
    """Write two-column data to a file with a header"""
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(val)):
            f.write(f'{wvl[i]:11.3f} {val[i]:12.3e}\n')


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
    
    print("Loading KT19 data from:", config.kt19(date_s))
    _, data_kt19 = read_ict_kt19(config.kt19(date_s))
    
    log.info("ssfr filename:", config.ssfr(date_s))
    
    # plot ssfr time series for checking sable legs selection
    ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_ranges_select, date_s, case_tag, pitch_roll_thres=3.0)

    # Build leg masks
    t_hsk = np.array(data_hsk["tmhr"])
    leg_masks = [(t_hsk>=lo)&(t_hsk<=hi) for lo,hi in tmhr_ranges_select]
    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_hsr1 = data_hsr1['time']/3600.0  # convert to hours
    t_marli = data_marli['time'] # in hours
    t_kt19 = data_kt19['tmhr'] # in hours

    

    
 
    # write out the convolved solar flux
    #/----------------------------------------------------------------------------\#
    # Kurudz solar spectrum has a resolution of 0.5 nm
    wvl_solar_vis = np.arange(300, 950.1, 1.0)
    wvl_solar_nir = np.arange(951, 2500.1, 1.0)
    if not os.path.exists('arcsix_ssfr_solar_flux_slit.dat'):
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
        
        write_2col_file('arcsix_ssfr_solar_flux_slit.dat', wvl_solar, flux_solar_convolved,
                        header=('# SSFR version solar flux with slit function convolution\n'
                                '# wavelength (nm)      flux (mW/m^2/nm)\n'))
            
    # Solar spectrum interpolation function
    flux_solar_interp = solar_interpolation_func(solar_flux_file='arcsix_ssfr_solar_flux_slit.dat', date=date)

    # read satellite granule
    #/----------------------------------------------------------------------------\#
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)

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
        
        if len(t_kt19) > 0:
            sel_kt19 = nearest_indices(t_hsk, mask, t_kt19)
        

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
            
        if len(data_kt19['tmhr']) > 0:
            sfc_T = data_kt19['ir_sfc_T'][sel_kt19]
            leg.update({
                "kt19_sfc_T": sfc_T,
            })
        else:
            leg.update({
                "kt19_sfc_T": None,
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
        
        time_start, time_end = tmhr_ranges_select[i][0], tmhr_ranges_select[i][-1]
    
        # save the cloud observation information to a pickle file
        fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, time_start, time_end)
        with open(fname_pkl, 'wb') as f:
            pickle.dump(vars()["cld_leg_%d" % i], f, protocol=pickle.HIGHEST_PROTOCOL)

        del leg  # free memory
        del sel_ssfr, sel_hsr1
        gc.collect()
        

     
    return None 
    
    
    
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

    atm_corr_overwrite_lrt = True

    
    flt_trk_atm_corr(date=datetime.datetime(2024, 5, 28),
                    tmhr_ranges_select=[[15.610, 15.822],
                                        [16.905, 17.404] 
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),
                    tmhr_ranges_select=[[13.839, 15.180],  # 5.6 km
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),
                    tmhr_ranges_select=[
                                        [16.905, 17.404] 
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),
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
                    manual_cloud_cer=13.0 ,
                    manual_cloud_cwp=77.82,
                    manual_cloud_cth=1.93,
                    manual_cloud_cbh=1.41,
                    manual_cloud_cot=21.27,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),
                    tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
                                            np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=7.0,
                    manual_cloud_cwp=113.65,
                    manual_cloud_cth=1.91,
                    manual_cloud_cbh=0.50,
                    manual_cloud_cot=24.31,
                    )
 

    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
                    tmhr_ranges_select=[[12.405, 13.812], # 5.7m,
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
                    tmhr_ranges_select=[
                                        [14.258, 15.036], # 100m
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
                    tmhr_ranges_select=[
                                        [15.535, 15.931], # 450m
                                        ],
                    case_tag='clear_atm_corr_3',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
                    tmhr_ranges_select=[
                                        [13.7889, 13.8010],
                                        [13.8350, 13.8395],
                                        [13.8780, 13.8885],
                                        [13.9240, 13.9255],
                                        [13.9389, 13.9403],
                                        [13.9540, 13.9715],
                                        [13.9980, 14.0153],
                                        [14.0417, 14.0575],
                                        [14.0417, 14.0475],
                                        [14.0560, 14.0590],
                                        [14.0825, 14.0975],
                                        [14.1264, 14.1525],
                                        [14.1762, 14.1975],
                                        [14.2194, 14.2420],
                                        [14.2605, 14.2810]
                                        ],
                    case_tag='clear_sky_spiral_atm_corr',
                    config=config,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
                    tmhr_ranges_select=[[16.250, 16.325], # 100m, 
                                        [16.375, 16.632], # 450m
                                        [16.700, 16.794], # 100m
                                        [16.850, 16.952], # 1.2km
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )
 

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
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=6.7,
                    manual_cloud_cwp=26.96,
                    manual_cloud_cth=0.43,
                    manual_cloud_cbh=0.15,
                    manual_cloud_cot=6.02,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),
                    tmhr_ranges_select=[[14.5667, 14.5694],
                                        [14.5986, 14.6097],
                                        [14.6375, 14.6486], # cloud shadow
                                        [14.6778, 14.6903],
                                        [14.7208, 14.7403],
                                        [14.7653, 14.7875],
                                        [14.8125, 14.8278],
                                        [14.8542, 14.8736],
                                        [14.8986, 14.9389], # more cracks
                                        ],
                    case_tag='clear_sky_spiral_atm_corr',
                    config=config,
                    simulation_interval=None,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),
                    tmhr_ranges_select=[
                                        [14.968, 15.229], # 100, clear, some cloud
                                        [14.968, 15.347],
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),
                    tmhr_ranges_select=[
                                        [15.347, 15.813], # 100m
                                        [15.813, 16.115], # 100-450m, clear, some cloud
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
                    tmhr_ranges_select=[[13.704, 13.817], # 100-450m, clear, some cloud
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
                    tmhr_ranges_select=[[14.109, 14.140], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_1',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.4, 0.52, 0.6, 0.8, 1.0,]),
                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=17.4,
                    manual_cloud_cwp=90.51,
                    manual_cloud_cth=0.52,
                    manual_cloud_cbh=0.15,
                    manual_cloud_cot=7.82,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
                    tmhr_ranges_select=[[15.834, 15.883], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.28, 0.3, 0.5, 0.58, 0.8, 1.0,]),
                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=22.4,
                    manual_cloud_cwp=35.6 ,
                    manual_cloud_cth=0.58,
                    manual_cloud_cbh=0.28,
                    manual_cloud_cot=2.39,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
                    tmhr_ranges_select=[[16.043, 16.067], # 100-200m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_3',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.38, 0.5, 0.68, 0.8, 1.0,]),
                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=8.9,
                    manual_cloud_cwp=21.29,
                    manual_cloud_cth=0.68,
                    manual_cloud_cbh=0.38,
                    manual_cloud_cot=3.59,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
                    tmhr_ranges_select=[[16.550, 17.581], # 100-500m, clear
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 7, 25),
                    tmhr_ranges_select=[[15.094, 15.300], # 100m, some low clouds or fog below
                                        ],
                    case_tag='cloudy_atm_corr',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,]),
                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=11.4,
                    manual_cloud_cwp=9.94,
                    manual_cloud_cth=0.30,
                    manual_cloud_cbh=0.16,
                    manual_cloud_cot=1.31,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 7, 25),
                    tmhr_ranges_select=[[15.881, 15.903], # 200-500m
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,]),
                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=11.4,
                    manual_cloud_cwp=9.94,
                    manual_cloud_cth=0.30,
                    manual_cloud_cbh=0.16,
                    manual_cloud_cot=1.31,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),
                    tmhr_ranges_select=[[13.442, 13.465],
                                        [13.490, 13.514],
                                        [13.536, 13.554],
                                        [13.580, 13.611],
                                        [13.639, 13.654],
                                        [13.676, 13.707],
                                        [13.733, 13.775],
                                        [13.793, 13.836],
                                        ],
                    case_tag='clear_sky_spiral_atm_corr',
                    config=config,
                    simulation_interval=None,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),
                    tmhr_ranges_select=[[13.939, 14.200], # 100m, clear
                                        [14.438, 14.714], # 3.7km
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),
                    tmhr_ranges_select=[
                                        [15.214, 15.804], # 1.3km
                                        [16.176, 16.304], # 1.3km
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 7, 30),
                    tmhr_ranges_select=[[13.886, 13.908],
                                        [13.934, 13.950],
                                        [13.976, 14.000],
                                        [14.031, 14.051],
                                        [14.073, 14.096],
                                        [14.115, 14.134],
                                        [14.157, 14.179],
                                        [14.202, 14.219],
                                        [14.239, 14.254],
                                        [14.275, 14.294],
                                        ],
                    case_tag='clear_sky_spiral_atm_corr',
                    config=config,
                    simulation_interval=None,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 7, 30),
                    tmhr_ranges_select=[[14.318, 14.936], # 100-450m, clear
                                        [15.043, 15.140], # 1.5km
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 1),
                    tmhr_ranges_select=[[13.843, 14.361], # 100-450m, clear, some open ocean
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 1),
                    tmhr_ranges_select=[
                                        [14.739, 15.053], # 550m
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )
    


    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 2),
                    tmhr_ranges_select=[
                                        [14.557, 15.100], # 100m
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 2),
                    tmhr_ranges_select=[
                                        [15.244, 16.635], # 1km
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),
                    tmhr_ranges_select=[[13.344, 13.763], # 100m, cloudy
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.65, 0.69, 0.78, 1.0,]),
                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=10.7,
                    manual_cloud_cwp=11.28,
                    manual_cloud_cth=0.78,
                    manual_cloud_cbh=0.69,
                    manual_cloud_cot=1.59,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),
                    tmhr_ranges_select=[
                                        [15.472, 15.567], # 180m, cloudy
                                        [15.580, 15.921], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.62, 0.8, 0.96,]),
                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=7.2,
                    manual_cloud_cwp=77.5,
                    manual_cloud_cth=0.96,
                    manual_cloud_cbh=0.62,
                    manual_cloud_cot=16.21,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
                    tmhr_ranges_select=[
                                        [12.990, 13.180], # 180m, clear
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
                    tmhr_ranges_select=[
                                        [14.250, 14.373], # 180m, clear
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
                    tmhr_ranges_select=[
                                        [16.471, 16.601], # 180m, clear
                                        ],
                    case_tag='clear_atm_corr_3',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
                    tmhr_ranges_select=[
                                        [13.212, 13.347], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_1',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.67, 0.8, 1.0,]),
                                            np.array([1.5, 1.98, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=15.3,
                    manual_cloud_cwp=143.94,
                    manual_cloud_cth=1.98,
                    manual_cloud_cbh=0.67,
                    manual_cloud_cot=14.12,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
                    tmhr_ranges_select=[
                                        [15.314, 15.504], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.78, 1.0,]),
                                            np.array([1.5, 1.81, 2.21, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=7.8,
                    manual_cloud_cwp=64.18,
                    manual_cloud_cth=2.21,
                    manual_cloud_cbh=1.81,
                    manual_cloud_cot=12.41,
                    )


    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),
                    tmhr_ranges_select=[
                                        [13.376, 13.600], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_1',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.34, 0.4, 0.6, 0.77, 1.0,]),
                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=9.0,
                    manual_cloud_cwp=83.49,
                    manual_cloud_cth=0.77,
                    manual_cloud_cbh=0.34,
                    manual_cloud_cot=13.93,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),
                    tmhr_ranges_select=[
                                        [14.750, 15.060], # 100m, clear
                                        [15.622, 15.887], # 100m, clear
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )



    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),
                    tmhr_ranges_select=[
                                        [16.029, 16.224], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.29, 0.4, 0.62, 0.8, 1.0,]),
                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=8.3,
                    manual_cloud_cwp=49.10,
                    manual_cloud_cth=0.62,
                    manual_cloud_cbh=0.29,
                    manual_cloud_cot=8.93,
                    )
    


    flt_trk_atm_corr(date=datetime.datetime(2024, 8, 15),
                    tmhr_ranges_select=[
                                        [14.085, 14.396], # 100m, clear
                                        [14.550, 14.968], # 3.5km, clear
                                        [15.078, 15.163], # 1.7km, clear
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    )