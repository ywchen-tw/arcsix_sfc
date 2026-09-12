"""
by Hong Chen (hong.chen@lasp.colorado.edu)

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
from tqdm import tqdm
import h5py
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata, NearestNDInterpolator
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.axes as maxes
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
import bisect
import pandas as pd
from scipy.signal import convolve
import netCDF4 as nc
import xarray as xr
from collections import defaultdict
import platform
# mpl.use('Agg')


import er3t

from util import gaussian, read_ict_radiosonde, read_ict_dropsonde, read_ict_lwc, read_ict_cloud_micro_2DGRAY50, read_ict_cloud_micro_FCDP, read_ict_bbr, read_ict_kt19


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
    _fdir_data_ = 'data/processed' 
    _fdir_general_ = 'data'
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


def flt_trk_lwc(date=datetime.datetime(2024, 5, 31),
                tmhr_ranges_select=[[14.10, 14.27]],
                output_lwp_alt=[False, True],
                fname_LWC='data/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict',
                fname_cloud_micro='data/cloud_prob/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
                ):

    date_s = date.strftime('%Y%m%d')
    
    dir_fig = f'./fig/{date_s}'
    os.makedirs(dir_fig, exist_ok=True)

    # read aircraft housekeeping data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#
    
    # read ssfr data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#
    
    # read HSR1 data
    #ARCSIX-HSR1_P3B_20240531_RA.h5
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSR1_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsr1 = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#

    # read LWC data
    #/----------------------------------------------------------------------------\#
    head, data_lwc = read_ict_lwc(fname_LWC, encoding='utf-8', na_values=[-9999999, -777, -888])
    #/----------------------------------------------------------------------------/#
    
    # read cloud microphysical data
    #/----------------------------------------------------------------------------\#
    if '2DGRAY50' in fname_cloud_micro:
        head, data_cloud_micro = read_ict_cloud_micro_2DGRAY50(fname_cloud_micro, encoding='utf-8', na_values=[-9999999, -777, -888])
    elif 'FCDP' in fname_cloud_micro:
        head, data_cloud_micro = read_ict_cloud_micro_FCDP(fname_cloud_micro, encoding='utf-8', na_values=[-9999999, -777, -888])
    #/----------------------------------------------------------------------------/#

    # selected stacked legs
    #/----------------------------------------------------------------------------\#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0])&(data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #\----------------------------------------------------------------------------/#

        
    # save the observation information
    data_ssfr_time = data_ssfr['tmhr']
    data_hsr1_time = data_hsr1['tmhr']
    data_lwc_time = np.array(data_lwc['tmhr'])
    data_cloud_micro_time = np.array(data_cloud_micro['tmhr'])
    
    # plt.hlines(1, data_hsk['tmhr'][0], data_hsk['tmhr'][-1], color='k', lw=0.5, ls='--', label='p3')
    # plt.hlines(2, data_ssfr['tmhr'][0], data_ssfr['tmhr'][-1], color='r', lw=0.5, ls='--', label='ssfr')
    # plt.hlines(3, data_hsr1['tmhr'][0], data_hsr1['tmhr'][-1], color='g', lw=0.5, ls='--', label='hsr1')
    # plt.hlines(4, data_lwc_time[0], data_lwc_time[-1], color='b', lw=0.5, ls='--', label='lwc')
    # plt.hlines(5, data_cloud_micro_time[0], data_cloud_micro_time[-1], color='c', lw=0.5, ls='--', label='cloud micro') 
    # plt.xlabel('Time (UTC)')
    # plt.ylabel('Data source')
    # plt.title('Data source time ranges')
    # plt.legend()
    # plt.show()
    # sys.exit()
    
    for i in range(len(tmhr_ranges_select)):
        vars()[f"ssfr_logics_select_{i}_ind"] = []
        vars()[f"hsr1_logics_select_{i}_ind"] = []
        vars()[f"lwc_logics_select_{i}_ind"] = []
        vars()[f"cloud_micro_logics_select_{i}_ind"] = []
        for time_hsk in data_hsk['tmhr'][logics_select[i]]:            
            vars()[f"ssfr_logics_select_{i}_ind"].append(np.argmin(np.abs(data_ssfr_time - time_hsk)))
            vars()[f"hsr1_logics_select_{i}_ind"].append(np.argmin(np.abs(data_hsr1_time - time_hsk)))
            vars()[f"lwc_logics_select_{i}_ind"].append(np.argmin(np.abs(data_lwc_time - time_hsk)))
            vars()[f"cloud_micro_logics_select_{i}_ind"].append(np.argmin(np.abs(data_cloud_micro_time - time_hsk)))
            
            
            
        vars()[f"ssfr_logics_select_{i}_ind"] = np.array(vars()[f"ssfr_logics_select_{i}_ind"])
        vars()[f"hsr1_logics_select_{i}_ind"] = np.array(vars()[f"hsr1_logics_select_{i}_ind"])
        vars()[f"lwc_logics_select_{i}_ind"] = np.array(vars()[f"lwc_logics_select_{i}_ind"])
        vars()[f"cloud_micro_logics_select_{i}_ind"] = np.array(vars()[f"cloud_micro_logics_select_{i}_ind"])
        
        vars()["cld_leg_%d" % i] = {}
        vars()["cld_leg_%d" % i]['lon'] = data_hsk['lon'][logics_select[i]]
        vars()["cld_leg_%d" % i]['lat'] = data_hsk['lat'][logics_select[i]]
        vars()["cld_leg_%d" % i]['sza'] = data_hsk['sza'][logics_select[i]]
        vars()["cld_leg_%d" % i]['saa'] = data_hsk['saa'][logics_select[i]]
        vars()["cld_leg_%d" % i]['ssfr_nad'] = data_ssfr['nad/flux'][vars()[f'ssfr_logics_select_{i}_ind'], :]
        vars()["cld_leg_%d" % i]['ssfr_zen'] = data_ssfr['zen/flux'][vars()[f'ssfr_logics_select_{i}_ind'], :]
        vars()["cld_leg_%d" % i]['hsr1_total'] = data_hsr1['tot/flux'][vars()[f'hsr1_logics_select_{i}_ind']]
        vars()["cld_leg_%d" % i]['hsr1_dif'] = data_hsr1['dif/flux'][vars()[f'hsr1_logics_select_{i}_ind']]
        vars()["cld_leg_%d" % i]['p3_alt'] = data_hsk['alt'][logics_select[i]]/1000 # m to km
        vars()["cld_leg_%d" % i]['twc'] = np.array(data_lwc['TWC'])[vars()[f"lwc_logics_select_{i}_ind"]]
        vars()["cld_leg_%d" % i]['lwc_1'] = np.array(data_lwc['LWC_1'])[vars()[f"lwc_logics_select_{i}_ind"]]
        vars()["cld_leg_%d" % i]['lwc_2'] = np.array(data_lwc['LWC_2'])[vars()[f"lwc_logics_select_{i}_ind"]]
        vars()["cld_leg_%d" % i]['cld_conc'] = np.array(data_cloud_micro['conc'])[vars()[f"cloud_micro_logics_select_{i}_ind"]]
        vars()["cld_leg_%d" % i]['cld_ext'] = np.array(data_cloud_micro['ext'])[vars()[f"cloud_micro_logics_select_{i}_ind"]]
        if '2DGRAY50' in fname_cloud_micro:
            vars()["cld_leg_%d" % i]['cld_iwc'] = np.array(data_cloud_micro['iwc'])[vars()[f"cloud_micro_logics_select_{i}_ind"]]
            vars()["cld_leg_%d" % i]['cld_cer'] = np.array(data_cloud_micro['effectiveDiam'])[vars()[f"cloud_micro_logics_select_{i}_ind"]]
        elif 'FCDP' in fname_cloud_micro:
            vars()["cld_leg_%d" % i]['cld_lwc'] = np.array(data_cloud_micro['lwc'])[vars()[f"cloud_micro_logics_select_{i}_ind"]]
        
        # find the time points that lwc variable crosses threshold
        lwc = vars()["cld_leg_%d" % i]['lwc_1']
        twc_threshold = 0.008 # g/m^3
        twc_crossing = np.where(np.diff((lwc > twc_threshold).astype(int)))[0]
        
        if twc_crossing.size > 0:
            twc_crossing_start_end = [twc_crossing[0], twc_crossing[-1]]
            print("crossing altitude:", vars()["cld_leg_%d" % i]['p3_alt'][twc_crossing_start_end])
        
        # plot alt-time, lwc_-time, hsr1-time, ssfr-time
        plt.close('all')
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)
        
        time_series = np.array(data_hsk['tmhr'][logics_select[i]])
        print("time_series 0, -1:", time_series[0], time_series[-1])
        
        ax1.plot(time_series, data_hsk['alt'][logics_select[i]]/1000, 'o-', color='k', lw=1.0, markersize=2.0)
        ax1.set_xlabel('Time (UTC)')
        ax1.set_ylabel('Altitude (km)')
        ax1.set_title('P3B altitude', fontsize=16)
        
        ax2.plot(time_series, vars()["cld_leg_%d" % i]['twc'].flatten(), 'o-', color='b', lw=1.0, markersize=2.0, label='TWC')
        ax2.plot(time_series, vars()["cld_leg_%d" % i]['lwc_1'].flatten(), 'o-', color='k', lw=1.0, markersize=2.0, label='LWC_1')
        ax2.plot(time_series, vars()["cld_leg_%d" % i]['lwc_2'].flatten(), 'o-', color='r', lw=1.0, markersize=2.0, label='LWC_2')
        ax2.set_xlabel('Time (UTC)')
        ax2.set_ylabel('LWC (g/m^3)')
        ax2.set_title('P3B LWC', fontsize=16)
        ax2.legend()
        
        if twc_crossing.size > 0:
            for ax in [ax1, ax2]:
                for crossing in [twc_crossing[0], twc_crossing[-1]]:
                    ax.axvline(time_series[crossing], color='gray', linestyle='--', lw=0.5, alpha=0.5)
        
        hsr1_diff_ratio = vars()["cld_leg_%d" % i]['hsr1_dif'] / vars()["cld_leg_%d" % i]['hsr1_total']
        
        hsr1_wvl = data_hsr1['tot/wvl']
        wvl_plot = 550.0
        wvl_ind_hsr1 = np.argmin(np.abs(hsr1_wvl - wvl_plot))
        
        ax3.plot(time_series, hsr1_diff_ratio[:, wvl_ind_hsr1], 'o-', color='k', lw=1.0, markersize=2.0, label='HSR1 diffussion ratio')
        ax3.set_xlabel('Time (UTC)')
        ax3.set_ylabel('HSR1 diffussion ratio')
        ax3.set_title(f'HSR1 diffussion ratio at {wvl_plot:.0f} nm', fontsize=16)
        
        wvl_ind_ssfr_zen = np.argmin(np.abs(data_ssfr['zen/wvl'] - wvl_plot))
        
        ax4.plot(time_series, vars()["cld_leg_%d" % i]['ssfr_zen'][:, wvl_ind_ssfr_zen], 'o-', color='k', lw=1.0, markersize=2.0, label='SSFR zen flux')
        # hsr1 total flux
        ax4.plot(time_series, vars()["cld_leg_%d" % i]['hsr1_total'][:, wvl_ind_hsr1], 'o-', color='r', lw=1.0, markersize=2.0, label='HSR1 total flux')
        ax4.set_xlabel('Time (UTC)')
        ax4.set_ylabel('Flux (W/m^2)')
        ax4.set_title(f'SSFR zen flux and HSR1 total flux at {wvl_plot:.0f} nm', fontsize=16)
        ax4.legend()
        
        ax5.plot(time_series, vars()["cld_leg_%d" % i]['cld_ext'], 'o-', color='k', lw=1.0, markersize=2.0, label='cloud extinction')
        ax5.set_xlabel('Time (UTC)')
        ax5.set_ylabel('Cloud extinction (km^-1)')
        ax5.set_title('Cloud extinction', fontsize=16)
        
        if '2DGRAY50' in fname_cloud_micro:
            ax6.plot(time_series, vars()["cld_leg_%d" % i]['cld_iwc'], 'o-', color='k', lw=1.0, markersize=2.0, label='ice water content')
            ax6.set_ylabel('Ice water content (g/m^3)')
            ax6.set_title('2DGRAY50 ice water content', fontsize=16)
        elif 'FCDP' in fname_cloud_micro:
            ax6.plot(time_series, vars()["cld_leg_%d" % i]['cld_lwc'], 'o-', color='k', lw=1.0, markersize=2.0, label='liquid water content')
            ax6.set_ylabel('Liquid water content (g/m^3)')
            ax6.set_title('FCDP liquid water content', fontsize=16)
        ax6.set_xlabel('Time (UTC)')
       
        
        
        fig.suptitle(f'P3B LWC and HSR1/SSFR data for {date_s} - {time_series[0]:.2f} to {time_series[-1]:.2f}', fontsize=20)
        fig.tight_layout()
        fig.savefig(f'fig/{date_s}/P3B_LWC_HSR1_SSFR_{date_s}_{time_series[0]:.2f}_{time_series[-1]:.2f}.png', bbox_inches='tight')
        plt.close(fig)
        
        if output_lwp_alt[i] and twc_crossing.size > 0:
            alt_array = (data_hsk['alt'][logics_select[i]]/1000)[twc_crossing[0]:twc_crossing[-1]+1]
            
            lwc_array = lwc[twc_crossing[0]:twc_crossing[-1]+1]
            ext_array = vars()["cld_leg_%d" % i]['cld_ext'][twc_crossing[0]:twc_crossing[-1]+1] # in km^-1
            
            alt_sorting = np.argsort(alt_array)
            alt_array = alt_array[alt_sorting]
            lwc_array = lwc_array[alt_sorting]
            ext_array = ext_array[alt_sorting]
            
            
            
            dz = 0.001 # km
            alt_avg = np.arange(np.round(alt_array.min(), decimals=3), np.round(alt_array.max(), decimals=3)+1e-5, dz)
            from scipy.interpolate import interp1d
            lwc_interp = interp1d(alt_array, lwc_array, bounds_error=False, fill_value=np.nan)
            lwc_fit = lwc_interp(alt_avg)
            lwp = np.nansum(lwc_fit * dz * 1000)/1000 # kg/m^2
            
            ext_interp = interp1d(alt_array, ext_array, bounds_error=False, fill_value=np.nan)
            ext_fit = ext_interp(alt_avg)
            
            cot = np.nansum(ext_fit * dz) # unitless
            rho_water = 1000.0 # kg/m^3
            if lwp > 0. and cot > 0.0:
                cer_layer = 3/2 * lwp / (cot * rho_water) * 1e6 # um
            else:
                cer_layer = np.nan
            
            if 'FCDP' in fname_cloud_micro:
                lwc_FCDP_array = vars()["cld_leg_%d" % i]['cld_lwc'][twc_crossing[0]:twc_crossing[-1]+1]
                lwc_FCDP_array = lwc_FCDP_array[alt_sorting]
                lwc_FCDP_interp = interp1d(alt_array, lwc_FCDP_array, bounds_error=False, fill_value=np.nan)
                lwc_FCDP_fit = lwc_FCDP_interp(alt_avg)
                lwp_FCDP = np.nansum(lwc_FCDP_fit * dz * 1000)/1000 # kg/m^2
                
                if lwp_FCDP > 0. and cot > 0.0:
                    cer_layer_FCDP = 3/2 * lwp_FCDP / (cot * rho_water) * 1e6 # um
                else:
                    cer_layer_FCDP = np.nan
            
            plt.close('all')
            fig = plt.figure(figsize=(16, 6))
            ax = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            ax.plot(lwc_array, alt_array, 'o-', color='k', lw=1.0, markersize=2.0, label='LWC data (all)')
            ax.plot(lwc_fit, alt_avg, 'r-', lw=2.5, label='Interpolated LWC')
            if 'FCDP' in fname_cloud_micro:
                ax.plot(lwc_FCDP_fit, alt_avg, 'g-', lw=2.0, label='Interpolated LWC (FCDP)')
                ax.plot(lwc_FCDP_array, alt_array, 'o-', color='orange', lw=1.0, markersize=2.0, label='LWC data (FCDP)')
                ax.text(0.65, 0.3, f'FCDP LWP: {lwp_FCDP*1000:.3f} g/m^2', transform=ax.transAxes, fontsize=12, va='top', ha='left')
            ax.set_xlabel('LWC (g/m^3)')
            ax.set_ylabel('Altitude (km)')
            ax.legend(ncol=2, loc='best')
            ax.text(0.65, 0.25, f'LWP: {lwp*1000:.4f} g/m^2', transform=ax.transAxes, fontsize=12, va='top', ha='left')
            # cloud altitude
            ax.text(0.65, 0.2, f'Alt: {alt_array.min():.3f} to {alt_array.max():.3f} km', transform=ax.transAxes, fontsize=12, va='top', ha='left')
            ax.set_title(f'LWC vs Altitude', fontsize=16)
            
            
            ax2.plot(ext_array, alt_array, 'o-', color='k', lw=1.0, markersize=2.0, label='Extinction data (all)')
            ax2.plot(ext_fit, alt_avg, 'b-', lw=2.5, label='Interpolated Extinction')
            ax2.set_xlabel('Extinction (km^-1)')
            ax2.set_ylabel('Altitude (km)')
            ax2.legend()
            ax2.text(0.7, 0.25, f'COT: {cot:.4f}', transform=ax2.transAxes, fontsize=12, va='top', ha='left')
            ax2.text(0.7, 0.2, f'CER: {cer_layer:.1f} um', transform=ax2.transAxes, fontsize=12, va='top', ha='left')
            if 'FCDP' in fname_cloud_micro:
                ax2.text(0.7, 0.15, f'FCDP CER: {cer_layer_FCDP:.1f} um', transform=ax2.transAxes, fontsize=12, va='top', ha='left')
            
            fig.suptitle(f'P3B LWC and Cloud Microphysics for {date_s} - {time_series[0]:.2f} to {time_series[-1]:.2f}', fontsize=20)
            
            fig.tight_layout()
            fig.savefig(f'fig/{date_s}/P3B_LWP_vs_Altitude_{date_s}_{time_series[0]:.2f}_{time_series[-1]:.2f}.png', bbox_inches='tight')
            plt.close(fig)
            
            
            fig = plt.figure(figsize=(9, 5.4))
            ax = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            # ax.plot(lwc_array, alt_array, 'o-', color='k', lw=1.0, markersize=2.0, label='LWC data (all)')
            # ax.plot(lwc_fit, alt_avg, 'r-', lw=2.5, label='Interpolated LWC')
            if 'FCDP' in fname_cloud_micro:
                # ax.plot(lwc_FCDP_fit, alt_avg, 'g-', lw=2.0, label='Interpolated LWC (FCDP)')
                ax.plot(lwc_FCDP_array*1000, alt_array, '-', color='orange', lw=2.0, markersize=2.0, label='LWC data (FCDP)')
                ax.text(0.5, 0.2, f'LWP: {lwp_FCDP*1000:.1f} '+r'g $m^{-2}$', transform=ax.transAxes, fontsize=12, va='top', ha='left')
            ax.set_xlabel('LWC (g m$^{-3}$)', fontsize=12)
            ax.set_ylabel('Altitude (km)', fontsize=12)
            # ax.legend(ncol=2, loc='best')
            # ax.text(0.65, 0.25, f'LWP: {lwp*1000:.1f} g/m^2', transform=ax.transAxes, fontsize=12, va='top', ha='left')
            # cloud altitude
            ax.text(0.5, 0.15, f'Alt: {alt_array.min():.2f} to {alt_array.max():.2f} km', transform=ax.transAxes, fontsize=12, va='top', ha='left')
            # ax.set_title(f'LWC vs Altitude', fontsize=16)
            
            
            ax2.plot(ext_array, alt_array, 'o-', color='k', lw=2.0, markersize=2.0, label='Extinction data (all)')
            # ax2.plot(ext_fit, alt_avg, 'b-', lw=2.5, label='Interpolated Extinction')
            ax2.set_xlabel('Extinction (km$^{-1}$)', fontsize=12)
            ax2.set_ylabel('Altitude (km)', fontsize=12)
            # ax2.legend()
            ax2.text(0.65, 0.225, f'COT: {cot:.1f}', transform=ax2.transAxes, fontsize=12, va='top', ha='left')
            ax2.text(0.65, 0.175, f'CER: {cer_layer:.1f} $\mu$m', transform=ax2.transAxes, fontsize=12, va='top', ha='left')
            # if 'FCDP' in fname_cloud_micro:
            #     ax2.text(0.7, 0.15, f'FCDP CER: {cer_layer_FCDP:.1f} um', transform=ax2.transAxes, fontsize=12, va='top', ha='left')
            
            # fig.suptitle(f'P3B LWC and Cloud Microphysics for {date_s} - {time_series[0]:.2f} to {time_series[-1]:.2f}', fontsize=20)
            
            fig.tight_layout()
            fig.savefig(f'fig/{date_s}/P3B_LWP_vs_Altitude_{date_s}_{time_series[0]:.2f}_{time_series[-1]:.2f}_poster.png', bbox_inches='tight', dpi=300)
            plt.close(fig)

def marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 5),
                     extent=[-44, -58, 83.4, 84.1],
                     sizes1 = [50, 20, 4],
                     tmhr_ranges_select=[[15.36, 15.60], [16.32, 16.60]],
                     fname_marli='data/marli/ARCSIX-MARLi_P3B_20240605_R0.cdf',
                     fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240605_R0.ict',
                     case_tag='marli_test',):
    date_s = date.strftime('%Y%m%d')
    
    # read aircraft housekeeping data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#
    
    # read ssfr data
    #/----------------------------------------------------------------------------\#
    # fname_ssfr = '%s/R0_submitted-on-20250228/%s-SSFR_%s_%s_R0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    #\----------------------------------------------------------------------------/#
    
    # read HSR1 data
    #ARCSIX-HSR1_P3B_20240531_RA.h5
    #/----------------------------------------------------------------------------\#
    # fname_hsr1 = '%s/R0_submitted-on-20250228/%s-HSR1_%s_%s_R0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    fname_hsr1 = '%s/%s-HSR1_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsr1 = er3t.util.load_h5(fname_hsr1)
    #\----------------------------------------------------------------------------/#


    # read in all logic data
    #/----------------------------------------------------------------------------\#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #\----------------------------------------------------------------------------/#

    # read kt19 data
    #/----------------------------------------------------------------------------\#
    head, data_kt19 = read_ict_kt19(fname_kt19, encoding='utf-8', na_values=[-9999999, -777, -888])
    #/----------------------------------------------------------------------------\#

    # read collocated satellite data
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    sat_name_list = []
    sat_time_list = []
    for i, sat_name in enumerate(fnames):
        sat_select_text = os.path.basename(sat_name).replace('.nc', '').replace('CLDPROP_L2_', '')
        sat_name_list.append(sat_select_text.split('.')[0].replace('_', ' '))
        sat_utc = sat_select_text.split('.')[2]
        sat_utc_hh = int(sat_utc[:2])
        sat_utc_mm = int(sat_utc[2:4])
        tmhr = sat_utc_hh + sat_utc_mm / 60.0
        sat_time_list.append(tmhr)
    #\----------------------------------------------------------------------------/#
    # print("tmhr_ranges_select:", tmhr_ranges_select)
    sat_select = []
    for i, name in enumerate(fnames):
        print(f"Satellite data {i}: {name}")
    for i in range(len(tmhr_ranges_select)):
        mean_time = np.mean(tmhr_ranges_select[i])
        closest_tmhr_ind = np.argmin(np.abs(sat_time_list - mean_time))
        print(f"Satellite {closest_tmhr_ind} ({sat_name_list[closest_tmhr_ind]}, timm {sat_time_list[closest_tmhr_ind]:.2f}) closest to P3B time {mean_time:.2f} UTC")
        sat_select.append(closest_tmhr_ind)
    
    # read MARLI netcdf data
    #/----------------------------------------------------------------------------\#
    with nc.Dataset(fname_marli) as ds:
        marli_data = {'time': ds.variables['time'][:],
                      'alt': ds.variables['Alt'][:],
                      'H': ds.variables['H'][:],
                      'T': ds.variables['T'][:],
                      'LSR': ds.variables['LSR'][:],
                      'WVMR': ds.variables['WVMR'][:],
                        }
        # marli_data['tmhr'] = marli_data['time'].dt.hour + marli_data['time'].dt.minute / 60.0
    #     print("marli time:",  marli_data['time'])
    #     print("marli_data[T] shape:", marli_data['T'].shape)
    #     print("np.array(marli_data['T']).shape :", np.array(marli_data['T']).shape)
    #     print("marli_data[LSR] shape:", marli_data['LSR'].shape)
    #     print("marli_data[H] shape:", marli_data['H'].shape)
    #     print("marli_data[WVMR] shape:", marli_data['WVMR'].shape)
    # sys.exit()
    #/----------------------------------------------------------------------------/#
    
    # selected stacked legs
    #/----------------------------------------------------------------------------\#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0])&(data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #\----------------------------------------------------------------------------/#
    
    import platform
    
    # fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    # os.makedirs(fdir_cld_obs_info, exist_ok=True)
    # fname_cld_obs_info = '%s/%s_cld_obs_info_%s_%s_%s_0.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag)
    # if not os.path.exists(fname_cld_obs_info):
    #     # save the cloud observation information
    #     data_ssfr_time = data_ssfr['tmhr']
    #     data_hsr1_time = data_hsr1['tmhr']
    #     data_kt19_time = np.array(data_kt19['tmhr'])
        
    #     for i in range(len(tmhr_ranges_select)):
    #         if platform.system() == 'Darwin':
    #             fname = '/Volumes/argus/field/arcsix/sat-data/%s/%s' % (date_s, fnames[sat_select[i]])
    #         elif platform.system() == 'Linux':
    #             fname = '/pl/active/vikas-arcsix/yuch8913/arcsix/data/sat-data/%s/%s' % (date_s, fnames[sat_select[i]])
                
    #         # read the cloud observation information
    #         print('Reading cloud observation information from %s ...' % fname)
    #         f = Dataset(fname, 'r')
    #         lon_s = f.groups['geolocation_data'].variables['longitude'][...].data
    #         lat_s = f.groups['geolocation_data'].variables['latitude'][...].data

    #         cot_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness'][...].data
    #         cot_pcl_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_PCL'][...].data

    #         cer_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius'][...].data
    #         cer_pcl_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_PCL'][...].data
            
    #         cwp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Water_Path'][...].data)
    #         cwp_pcl_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Water_Path_PCL'][...].data)
            
    #         cwp_s /= 1000 # convert g/m^2 to kg/m^2
    #         cwp_pcl_s /= 1000 # convert g/m^2 to kg/m

    #         cth_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Height'][...].data)
            
    #         cth_s /= 1000 # m to km
            
    #         ctt_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Temperature'][...].data)

    #         ctp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Phase_Optical_Properties'][...].data)

    #         scan_utc_s = f.groups['scan_line_attributes'].variables['scan_start_time'][...].data
    #         jday_s0 = np.array([er3t.util.dtime_to_jday(datetime.datetime(1993, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=scan_utc_s[i])) for i in range(scan_utc_s.size)])
    #         f.close()

    #         logic_pcl = (cot_s<0.0) & (cot_pcl_s>0.0) & (cer_s<0.0) & (cer_pcl_s>0.0)
    #         cot_s[logic_pcl] = cot_pcl_s[logic_pcl]
    #         cer_s[logic_pcl] = cer_pcl_s[logic_pcl]
    #         cwp_s[logic_pcl] = cwp_pcl_s[logic_pcl]
    #         cwp_s[cwp_s<0.0] = np.nan

    #         logic_0 = (ctp_s==0.0)&((cot_s<0.0)|(cer_s<0.0))
    #         logic_1 = (ctp_s==1.0)&((cot_s<0.0)|(cer_s<0.0))
    #         logic_2 = (ctp_s==2.0)&((cot_s<0.0)|(cer_s<0.0))
    #         logic_3 = (ctp_s==3.0)&((cot_s<0.0)|(cer_s<0.0))
    #         logic_4 = (ctp_s==4.0)&((cot_s<0.0)|(cer_s<0.0))

    #         cot_s[logic_0] = np.nan
    #         cer_s[logic_0] = np.nan

    #         cot_s[logic_1] = 0.0
    #         cer_s[logic_1] = 1.0

    #         cot_s[logic_2] = -2.0
    #         cer_s[logic_2] = 1.0

    #         cot_s[logic_3] = -3.0
    #         cer_s[logic_3] = 1.0

    #         cot_s[logic_4] = -4.0
    #         cer_s[logic_4] = 1.0

    #         cot_s[cot_s<-10.0] = np.nan
    #         cer_s[cot_s<-10.0] = np.nan
            
    #         cgt_s = np.zeros_like(cth_s)
    #         cgt_s[...] = np.nan
            
    #         # read CGT_Data.csv
    #         cgt_table = pd.read_csv('CGT_Data.csv')
    #         for j in range(len(cgt_table)):
    #             pixel_select_cwp_low = np.logical_and(np.logical_and(cth_s > cgt_table['cth_low'][j], cth_s <= cgt_table['cth_high'][j]),
    #                                                 cwp_s < cgt_table['cwp_threshold'][j]/1000)
    #             pixel_select_cwp_high = np.logical_and(np.logical_and(cth_s > cgt_table['cth_low'][j], cth_s <= cgt_table['cth_high'][j]),
    #                                                 cwp_s >= cgt_table['cwp_threshold'][j]/1000)
    #             cgt_s[pixel_select_cwp_low] = cgt_table['a'][j]*cwp_s[pixel_select_cwp_low] + cgt_table['b'][j]
    #             cgt_s[pixel_select_cwp_high] = cgt_table['a_extra'][j]*cwp_s[pixel_select_cwp_high] + cgt_table['b_extra'][j]
                
            
    #         ext = [0.13, 0.25, 0.39, 0.55, 0.67] # in km-1
    #         ctt_thresh = [0, 200, 220, 240, 260, 400]
    #         # thin cirrus cloud
    #         for j in range(len(ctt_thresh)-1):
    #             pixel_select_thin = np.logical_and(cot_s>0, np.logical_and(cot_s < 1., np.logical_and(ctt_s > ctt_thresh[j], ctt_s <= ctt_thresh[j+1])))
    #             cgt_s[pixel_select_thin] = cot_s[pixel_select_thin]/ext[j]
                

    #         # sys.exit()
    #         cgt_s[cgt_s>cth_s] = cth_s[cgt_s>cth_s]
    #         cbh_s = cth_s - cgt_s
            
    #         cot_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cot_s.flatten())
    #         cer_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cer_s.flatten())
    #         cwp_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cwp_s.flatten())
    #         cth_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cth_s.flatten())
    #         cgt_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cgt_s.flatten())
    #         cbh_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cbh_s.flatten())
        
        
        
        
        
    #         vars()[f"ssfr_logics_select_{i}_ind"] = []
    #         vars()[f"hsr1_logics_select_{i}_ind"] = []
    #         vars()[f"kt19_logics_select_{i}_ind"] = []
                
    #         for time_hsk in data_hsk['tmhr'][logics_select[i]]:
                
    #             vars()[f"ssfr_logics_select_{i}_ind"].append(np.argmin(np.abs(data_ssfr_time - time_hsk)))
    #             vars()[f"hsr1_logics_select_{i}_ind"].append(np.argmin(np.abs(data_hsr1_time - time_hsk)))
    #             vars()[f"kt19_logics_select_{i}_ind"].append(np.argmin(np.abs(data_kt19_time - time_hsk)))
                
                
    #         vars()[f"ssfr_logics_select_{i}_ind"] = np.array(vars()[f"ssfr_logics_select_{i}_ind"])
    #         vars()[f"hsr1_logics_select_{i}_ind"] = np.array(vars()[f"hsr1_logics_select_{i}_ind"])
    #         vars()[f"kt19_logics_select_{i}_ind"] = np.array(vars()[f"kt19_logics_select_{i}_ind"])
            
    #         vars()["cld_leg_%d" % i] = {}
    #         vars()["cld_leg_%d" % i]['tmhr'] = data_hsk['tmhr'][logics_select[i]]
    #         vars()["cld_leg_%d" % i]['lon'] = data_hsk['lon'][logics_select[i]]
    #         vars()["cld_leg_%d" % i]['lat'] = data_hsk['lat'][logics_select[i]]
    #         vars()["cld_leg_%d" % i]['sza'] = data_hsk['sza'][logics_select[i]]
    #         vars()["cld_leg_%d" % i]['saa'] = data_hsk['saa'][logics_select[i]]
    #         vars()["cld_leg_%d" % i]['ssfr_nad'] = data_ssfr['nad/flux'][vars()[f'ssfr_logics_select_{i}_ind'], :]
    #         vars()["cld_leg_%d" % i]['ssfr_zen'] = data_ssfr['zen/flux'][vars()[f'ssfr_logics_select_{i}_ind'], :]
    #         vars()["cld_leg_%d" % i]['hsr1_total'] = data_hsr1['tot/flux'][vars()[f'hsr1_logics_select_{i}_ind']]
    #         vars()["cld_leg_%d" % i]['hsr1_dif'] = data_hsr1['dif/flux'][vars()[f'hsr1_logics_select_{i}_ind']]
    #         vars()["cld_leg_%d" % i]['p3_alt'] = data_hsk['alt'][logics_select[i]]/1000 # m to km
    #         vars()["cld_leg_%d" % i]['cot'] = cot_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
    #         vars()["cld_leg_%d" % i]['cer'] = cer_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
    #         vars()["cld_leg_%d" % i]['cwp'] = cwp_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
    #         vars()["cld_leg_%d" % i]['cth'] = cth_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
    #         vars()["cld_leg_%d" % i]['cgt'] = cgt_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
    #         vars()["cld_leg_%d" % i]['cbh'] = cbh_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
    #         vars()["cld_leg_%d" % i]['ir_sfc_T'] = np.array(data_kt19['ir_sfc_T'])[vars()[f"kt19_logics_select_{i}_ind"]]
        
    #         # save the cloud observation information to a pickle file
    #         fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
    #         with open(fname_pkl, 'wb') as f:
    #             pickle.dump(vars()[f"cld_leg_{i}"], f, protocol=pickle.HIGHEST_PROTOCOL)
                

            
    # else:
    #     print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
    #     for i in range(len(tmhr_ranges_select)):
    #         fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
    #         with open(fname_pkl, 'rb') as f:
    #             vars()[f"cld_leg_{i}"] = pickle.load(f)  
        
    if platform.system() == 'Darwin':
        for i in range(len(tmhr_ranges_select)):
            if platform.system() == 'Darwin':
                fname = '/Volumes/argus/field/arcsix/sat-data/%s/%s' % (date_s, fnames[sat_select[i]])
            elif platform.system() == 'Linux':
                fname = '/pl/active/vikas-arcsix/yuch8913/arcsix/data/sat-data/%s/%s' % (date_s, fnames[sat_select[i]])
                
            # read the cloud observation information
            print('Reading cloud observation information from %s ...' % fname)
            f = Dataset(fname, 'r')
            lon_s = f.groups['geolocation_data'].variables['longitude'][...].data
            lat_s = f.groups['geolocation_data'].variables['latitude'][...].data

            cot_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness'][...].data
            cot_pcl_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_PCL'][...].data

            cer_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius'][...].data
            cer_pcl_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_PCL'][...].data
            
            cwp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Water_Path'][...].data)
            cwp_pcl_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Water_Path_PCL'][...].data)
            
            cwp_s /= 1000 # convert g/m^2 to kg/m^2
            cwp_pcl_s /= 1000 # convert g/m^2 to kg/m

            cth_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Height'][...].data)
            
            cth_s /= 1000 # m to km
            
            ctt_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Temperature'][...].data)

            ctp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Phase_Optical_Properties'][...].data)

            scan_utc_s = f.groups['scan_line_attributes'].variables['scan_start_time'][...].data
            jday_s0 = np.array([er3t.util.dtime_to_jday(datetime.datetime(1993, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=scan_utc_s[i])) for i in range(scan_utc_s.size)])
            f.close()

            logic_pcl = (cot_s<0.0) & (cot_pcl_s>0.0) & (cer_s<0.0) & (cer_pcl_s>0.0)
            cot_s[logic_pcl] = cot_pcl_s[logic_pcl]
            cer_s[logic_pcl] = cer_pcl_s[logic_pcl]
            cwp_s[logic_pcl] = cwp_pcl_s[logic_pcl]
            cwp_s[cwp_s<0.0] = np.nan

            logic_0 = (ctp_s==0.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_1 = (ctp_s==1.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_2 = (ctp_s==2.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_3 = (ctp_s==3.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_4 = (ctp_s==4.0)&((cot_s<0.0)|(cer_s<0.0))

            cot_s[logic_0] = np.nan
            cer_s[logic_0] = np.nan

            cot_s[logic_1] = 0.0
            cer_s[logic_1] = 1.0

            cot_s[logic_2] = -2.0
            cer_s[logic_2] = 1.0

            cot_s[logic_3] = -3.0
            cer_s[logic_3] = 1.0

            cot_s[logic_4] = -4.0
            cer_s[logic_4] = 1.0

            cot_s[cot_s<-10.0] = np.nan
            cer_s[cot_s<-10.0] = np.nan
            
            cgt_s = np.zeros_like(cth_s)
            cgt_s[...] = np.nan
            
            # read CGT_Data.csv
            cgt_table = pd.read_csv('CGT_Data.csv')
            for j in range(len(cgt_table)):
                pixel_select_cwp_low = np.logical_and(np.logical_and(cth_s > cgt_table['cth_low'][j], cth_s <= cgt_table['cth_high'][j]),
                                                    cwp_s < cgt_table['cwp_threshold'][j]/1000)
                pixel_select_cwp_high = np.logical_and(np.logical_and(cth_s > cgt_table['cth_low'][j], cth_s <= cgt_table['cth_high'][j]),
                                                    cwp_s >= cgt_table['cwp_threshold'][j]/1000)
                cgt_s[pixel_select_cwp_low] = cgt_table['a'][j]*cwp_s[pixel_select_cwp_low] + cgt_table['b'][j]
                cgt_s[pixel_select_cwp_high] = cgt_table['a_extra'][j]*cwp_s[pixel_select_cwp_high] + cgt_table['b_extra'][j]
                
            
            ext = [0.13, 0.25, 0.39, 0.55, 0.67] # in km-1
            ctt_thresh = [0, 200, 220, 240, 260, 400]
            # thin cirrus cloud
            for j in range(len(ctt_thresh)-1):
                pixel_select_thin = np.logical_and(cot_s>0, np.logical_and(cot_s < 1., np.logical_and(ctt_s > ctt_thresh[j], ctt_s <= ctt_thresh[j+1])))
                cgt_s[pixel_select_thin] = cot_s[pixel_select_thin]/ext[j]
                

            # sys.exit()
            cgt_s[cgt_s>cth_s] = cth_s[cgt_s>cth_s]
            cbh_s = cth_s - cgt_s
            
            cot_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cot_s.flatten())
            cer_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cer_s.flatten())
            cwp_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cwp_s.flatten())
            cth_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cth_s.flatten())
            cgt_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cgt_s.flatten())
            cbh_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cbh_s.flatten())
            
            cot_s0 = cot_s.copy()
            cot_s0[...] = 255.0
            # cs = ax1.pcolormesh(lon_s, lat_s, cot_s0, cmap='gray', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=1.0)
            cot_s[cot_s<=0.0] = np.nan
            ctp_s[ctp_s==1.0] = np.nan
            cth_s[cth_s<=0.0] = np.nan
            
            proj0 = ccrs.Orthographic(
                    central_longitude=((extent[0]+extent[1])/2.0),
                    central_latitude=((extent[2]+extent[3])/2.0),
                    )
            plt.close('all')
            fig = plt.figure(figsize=(18, 12))
            ax1 = fig.add_subplot(211, projection=proj0)
            ax2 = fig.add_subplot(212, projection=proj0)
            cs_ctp = ax1.pcolormesh(lon_s, lat_s,  ctp_s, cmap='viridis', vmin=0.0, vmax=5.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
            cs_cot = ax1.pcolormesh(lon_s, lat_s,  cot_s, cmap='jet', vmin=0.0, vmax=10.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)

            ax1.plot(data_hsk['lon'], data_hsk['lat'], lw=2.5, color='k', transform=ccrs.PlateCarree(), zorder=1)

            colors1 = ['r', 'g', 'b', 'brown']
            color = colors1[i]

            text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
            text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
            ax1.scatter(data_hsk['lon'][logics_select[i]], data_hsk['lat'][logics_select[i]], color=color, s=sizes1[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
            ax1.text(data_hsk['lon'][logics_select[i]][0], data_hsk['lat'][logics_select[i]][0]+0.03, text1, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
            ax1.text(data_hsk['lon'][logics_select[i]][-1], data_hsk['lat'][logics_select[i]][-1]+0.03, text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
            # ax1.scatter(f_lon[3], f_lat[3], s=5, marker='^', color='orange')
            # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
            # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
            # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)


            
            cbar1 = fig.colorbar(cs_cot, ax=ax1, orientation='vertical', pad=0.05, shrink=1)
            cbar1.set_label('Cloud Optical Thickness', fontsize=16)
            cbar1.ax.tick_params(labelsize=14)
            
            
            
            # ax2 plot the cloud top height
            cs_cth = ax2.pcolormesh(lon_s, lat_s, cth_s, cmap='jet', vmin=0.0, vmax=8.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
            ax2.plot(data_hsk['lon'], data_hsk['lat'], lw=2.5, color='k', transform=ccrs.PlateCarree(), zorder=1)


            text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
            text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
            ax2.scatter(data_hsk['lon'][logics_select[i]], data_hsk['lat'][logics_select[i]], color=color, s=sizes1[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
            ax2.text(data_hsk['lon'][logics_select[i]][0], data_hsk['lat'][logics_select[i]][0]+0.03, text1, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
            ax2.text(data_hsk['lon'][logics_select[i]][-1], data_hsk['lat'][logics_select[i]][-1]+0.03, text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
            # ax1.scatter(f_lon[3], f_lat[3], s=5, marker='^', color='orange')
            # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
            # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
            # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)

            
            
            cbar2 = fig.colorbar(cs_cth, ax=ax2, orientation='vertical', pad=0.05, shrink=1, extend='max')
            cbar2.set_label('Cloud Top Height (km)', fontsize=16)
            cbar2.ax.tick_params(labelsize=14)
            
            
            for ax, label in zip([ax1, ax2], ['Cloud Optical Thickness', 'Cloud Top Height']):
                ax.coastlines(resolution='10m', color='gray', lw=0.5)
                g_lines = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
                g_lines.xlocator = FixedLocator(np.arange(-180, 181, 5.0))
                g_lines.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2))
                g_lines.top_labels = False
                g_lines.right_labels = False
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.set_title(label, fontsize=16)
            
            
            sat_select_text = os.path.basename(fname).replace('.nc', '').replace('CLDPROP_L2_', '')
            sat_ = sat_select_text.split('.')[0].replace('_', ' ')
            sat_utc = sat_select_text.split('.')[2]
            title_text = f'{sat_} {sat_utc}\n' + \
                f'Flight track {text1} - {text2} UTC'
            fig.suptitle(title_text, fontsize=20, y=0.95)
            

            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            fig.savefig('fig/%s/%s_%s_sat_%d_leg_%d.png' % (date_s, date_s, case_tag, sat_select[i], i), bbox_inches='tight', metadata=_metadata)
            #\--------------------------------------------------------------/# 
            
            # print("Marli time:", marli_data['tmhr'].min(), marli_data['tmhr'].max())

            # plot vertical marli data
            #/--------------------------------------------------------------/\#
            plt.close('all')
            
            data_marli_time = np.array(marli_data['time'])
            data_ssfr_time = np.array(data_ssfr['tmhr'])
            data_hsr1_time = np.array(data_hsr1['tmhr'])
            marli_select = []
            ssfr_select = []
            hsr1_select = []
            for time_hsk in data_hsk['tmhr'][logics_select[i]]:
                marli_select.append(np.argmin(np.abs(data_marli_time - time_hsk)))
                ssfr_select.append(np.argmin(np.abs(data_ssfr_time - time_hsk)))
                hsr1_select.append(np.argmin(np.abs(data_hsr1_time - time_hsk)))
            
            marli_select = np.array(marli_select)
            ssfr_select = np.array(ssfr_select)
            hsr1_select = np.array(hsr1_select)
            
            marli_time = marli_data['time'][marli_select]
            marli_alt = marli_data['alt'][marli_select]
            marli_H = marli_data['H']
            marli_T = np.array(marli_data['T'])[marli_select, :]
            marli_T[marli_T == 9999] = np.nan  # remove unrealistic temperatures
            marli_T[marli_T > 100] = np.nan  # remove unrealistic temperatures
            marli_T[marli_T < -100] = np.nan  # remove unrealistic temperatures
            marli_LSR = np.array(marli_data['LSR'])[marli_select, :]
            marli_LSR[marli_LSR == 9999] = np.nan  # remove unrealistic liquid water content
            marli_WVMR = np.array(marli_data['WVMR'])[marli_select, :]
            marli_WVMR[marli_WVMR == 9999] = np.nan  # remove unrealistic water vapor mixing ratios
            
            marli_time_repeat = np.repeat(marli_time[:, np.newaxis], marli_H.size, axis=1)
            marli_H_repeat = np.repeat(marli_H[np.newaxis, :], marli_time.size, axis=0)
            
            print("marli_H shape:", marli_H.shape)
            print("marli_time_repeat shape:", marli_time_repeat.shape)
            print("marli_H_repeat shape:", marli_H_repeat.shape)
            print("marli_T shape:", marli_T.shape)
            print("marli_LSR shape:", marli_LSR.shape)
            print("marli_WVMR shape:", marli_WVMR.shape)
            
            ssfr_zen_flux = data_ssfr['zen/flux'][ssfr_select, :]
            ssfr_nad_flux = data_ssfr['nad/flux'][ssfr_select, :]
            ssfr_zen_toa = data_ssfr['zen/toa0']
            ssfr_zen_wvl = data_ssfr['zen/wvl']
            ssfr_nad_wvl = data_ssfr['nad/wvl']
            
            ssfr_nad_flux_interp = ssfr_zen_flux.copy()
            from scipy.interpolate import interp1d
            for j in range(ssfr_nad_flux.shape[0]):
                f_nad_flux_interp = interp1d(ssfr_nad_wvl, ssfr_nad_flux[j, :], axis=0, bounds_error=False, fill_value=np.nan)
                ssfr_nad_flux_interp[j, :] = f_nad_flux_interp(ssfr_zen_wvl)
            
            zen_795_left = np.argmin(np.abs(ssfr_zen_wvl - (795.0-4)))
            zen_795_right = np.argmin(np.abs(ssfr_zen_wvl - (795.0+4)))
            zen_1050_left = np.argmin(np.abs(ssfr_zen_wvl - (1050.0-4)))
            zen_1050_right = np.argmin(np.abs(ssfr_zen_wvl - (1050.0+4)))
            
            toa_795_avg = np.nanmean(ssfr_zen_toa[zen_795_left:zen_795_right+1])
            toa_1050_avg = np.nanmean(ssfr_zen_toa[zen_1050_left:zen_1050_right+1])
            
            zen_795_avg = np.nanmean(ssfr_zen_flux[zen_795_left:zen_795_right+1])
            zen_1050_avg = np.nanmean(ssfr_zen_flux[zen_1050_left:zen_1050_right+1])
            
            nad_795_avg = np.nanmean(ssfr_nad_flux_interp[zen_795_left:zen_795_right+1])
            nad_1050_avg = np.nanmean(ssfr_nad_flux_interp[zen_1050_left:zen_1050_right+1])
            
            zen_scaling = (zen_795_avg/zen_1050_avg) / (toa_795_avg/toa_1050_avg)
            nad_scaling = (nad_795_avg/nad_1050_avg) / (toa_795_avg/toa_1050_avg)
            
            zen_950_ind = np.argmin(np.abs(ssfr_zen_wvl - 950.0))
            
            ssfr_zen_flux[:, :zen_950_ind+1] /= zen_scaling
            ssfr_nad_flux_interp[:, :zen_950_ind+1] /= nad_scaling
            
            ssfr_nad_zen_ratio = ssfr_nad_flux_interp / ssfr_zen_flux
            
            wvl_550_ind = np.argmin(np.abs(ssfr_zen_wvl - 550.0))
            wvl_860_ind = np.argmin(np.abs(ssfr_zen_wvl - 860.0))
            wvl_1640_ind = np.argmin(np.abs(ssfr_zen_wvl - 1640.0))
            
            hsr1_wvl = data_hsr1['tot/wvl']
            hsr1_total_flux = data_hsr1['tot/flux'][hsr1_select]
            hsr1_dif_flux = data_hsr1['dif/flux'][hsr1_select]
            hsr1_dif_ratio = hsr1_dif_flux / hsr1_total_flux
            hsr1_wvl_790_ind = np.argmin(np.abs(hsr1_wvl - 790.0))
            
            fig = plt.figure(figsize=(18, 12))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            
            cc1 = ax1.contourf(marli_time_repeat, marli_H_repeat, marli_T, cmap='coolwarm', extend='both')
            ax1.set_xlabel('Time (UTC)', fontsize=16)
            ax1.set_ylabel('Height (km)', fontsize=16)
            cbar1 = fig.colorbar(cc1, ax=ax1, orientation='vertical', pad=0.05, shrink=1)
            cbar1.set_label('Temperature ($^o$C)', fontsize=16)
            cbar1.ax.tick_params(labelsize=14)
            ax1.set_title('MARLI Temperature', fontsize=20, y=1.01, color='k')
            
            cc2 = ax2.contourf(marli_time_repeat, marli_H_repeat, marli_LSR, cmap='jet', extend='both')
            ax2.set_xlabel('Time (UTC)', fontsize=16)
            ax2.set_ylabel('Height (km)', fontsize=16)
            cbar2 = fig.colorbar(cc2, ax=ax2, orientation='vertical', pad=0.05, shrink=1)
            cbar2.set_label('lidar scattering ratio', fontsize= 16)
            ax2.set_title('MARLI Lidar Scattering Ratio', fontsize=20, y=1.01, color='k')
            
            cc3 = ax3.contourf(marli_time_repeat, marli_H_repeat, marli_WVMR, cmap='Blues', extend='both')
            ax3.set_xlabel('Time (UTC)', fontsize=16)
            ax3.set_ylabel('Height (km)', fontsize=16)
            cbar3 = fig.colorbar(cc3, ax=ax3, orientation='vertical', pad=0.05, shrink=1)
            cbar3.set_label('Water Vapor Mixing Ratio (g/kg)', fontsize=16)
            ax3.set_title('MARLI Water Vapor Mixing Ratio', fontsize=20, y=1.01, color='k')
            
            ymax = np.round(np.nanmax(marli_alt)*1.1, 1)
            for ax in [ax1, ax2, ax3]:
                ax.plot(marli_time, marli_alt, color='k', lw=2.0)
                ax.set_ylim(0, ymax)
            
            color_list = ssfr_zen_flux[:, wvl_550_ind]/ np.nanmax(ssfr_zen_flux[:, wvl_550_ind])
            color_list[np.isnan(color_list)] = 0
            
            cc4 = ax4.scatter(data_ssfr_time[ssfr_select], ssfr_nad_zen_ratio[:, wvl_550_ind], c=color_list, s=0.5)
            ax4.plot(data_ssfr_time[ssfr_select], ssfr_nad_zen_ratio[:, wvl_550_ind], label='SSFR 550 nm', lw=2.0)
            ax4.plot(data_ssfr_time[ssfr_select], ssfr_nad_zen_ratio[:, wvl_860_ind], label='SSFR 860 nm', lw=2.0)
            ax4.plot(data_ssfr_time[ssfr_select], ssfr_nad_zen_ratio[:, wvl_1640_ind], label='SSFR 1640 nm', lw=2.0)
            ax4.plot(data_hsr1_time[hsr1_select], hsr1_dif_ratio[:, hsr1_wvl_790_ind], label='HSR1 790 nm diffuse', lw=2.0)
            ax4.legend(fontsize=16)
            ax4.set_xlabel('Time (UTC)', fontsize=16)
            ax4.set_ylabel('Nadir to Zenith Ratio', fontsize=16)
            ax4.set_title('SSFR upward/downward flux ratio\nHSR1 Diffuse ratio', fontsize=20, y=1.01, color='k')
            
            cbar4 = fig.colorbar(cc4, ax=ax4, orientation='vertical', pad=0.05, shrink=1)
            
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(labelsize=14)
                time_interval = len(data_ssfr_time[ssfr_select])//5
                ax.set_xticks(data_ssfr_time[ssfr_select][::time_interval])
                ax.set_xticklabels([datetime.datetime.utcfromtimestamp(t*60*60).strftime('%H:%M') for t in data_ssfr_time[ssfr_select][::time_interval]])
            
            sat_select_text = os.path.basename(fname).replace('.nc', '').replace('CLDPROP_L2_', '')
            sat_ = sat_select_text.split('.')[0].replace('_', ' ')
            sat_utc = sat_select_text.split('.')[2]
            title_text = f'{sat_} {sat_utc}\n' + \
                f'Flight track {text1} - {text2} UTC'
                
            fig.suptitle(title_text, fontsize=24, color='k')
            
            
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_marli_ssfr_%d_leg_%d.png' % (date_s, date_s, case_tag, sat_select[i], i), bbox_inches='tight')
            
            
        
    

def flt_trk_lrt_para(date=datetime.datetime(2024, 5, 31),
                     extent=[-60, -80, 82.4, 84.6],
                     sizes1 = [50, 20, 4],
                     tmhr_ranges_select=[[14.10, 14.27]],
                     sat_select=[1],
                     fname_radiosonde='data/radiosonde/arcsix-THAAO-RSxx_SONDE_20240531183300_RA.ict',
                     fname_dropsonde='data/dropsonde/ARCSIX-AVAPS_G3_20240531_R0/ARCSIX-AVAPS_G3_20240531142150_R0.ict',
                     fname_bbr='data/bbr/ARCSIX-BBR_P3B_20240611_R0.ict',
                     fname_kt19='data/kt19/ARCSIX-KT19_P3B_20240611_R0.ict',
                     fname_LWC='data/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict',
                     modis_07_file=['./data/sat-data/20240531/MOD07_L2.A2024152.1525.061.2024153011814.hdf'],
                     case_tag='default',
                     levels=None,
                     simulation_interval=3,
                     clear_sky=True,
                     lw=False,
                     manual_cloud=False,
                     manual_cloud_cth=0.0,
                     manual_cloud_cbh=0.0,
                     manual_cloud_cot=0.0,
                     manual_cloud_cwp=0.0,
                     manual_cloud_cer=0.0,
                     overwrite_atm=False,
                     overwrite_alb=False,
                     overwrite_cld=False,
                     overwrite_lrt=True,
                     new_compute=False,
                            ):
    print('Running flt_trk_lrt_para with the following parameters:')
    print(f'  date: {date}, case_tag: {case_tag}, simulation_interval: {simulation_interval},')
    print(f'clear_sky: {clear_sky}, lw: {lw}, manual_cloud: {manual_cloud}')
    print('.......')
    # case specification
    #/----------------------------------------------------------------------------\#
    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#

    date_s = date.strftime('%Y%m%d')
    
    # read read_ict_radiosonde data
    #/----------------------------------------------------------------------------\#
    head, data_radiosonde = read_ict_radiosonde(fname_radiosonde, encoding='utf-8', na_values=[-9999999, -777, -888])
    data_radiosonde['lon'] = np.mean(extent[:2])
    data_radiosonde['lat'] = np.mean(extent[2:])
    #\----------------------------------------------------------------------------/#
    
    # read read_ict_radiosonde data
    #/----------------------------------------------------------------------------\#
    head, data_dropsonde = read_ict_dropsonde(fname_dropsonde, encoding='utf-8', na_values=[-9999999, -777, -888])
    data_dropsonde['lon'] = np.mean(data_dropsonde['lon_all'])
    data_dropsonde['lat'] = np.mean(data_dropsonde['lat_all'])
    #\----------------------------------------------------------------------------/#
    
    # create atmospheric profile
    #/----------------------------------------------------------------------------\#
    zpt_filedir = f'{_fdir_general_}/zpt/{date_s}'
    os.makedirs(zpt_filedir, exist_ok=True)
    if levels is None:
        levels = np.concatenate((np.arange(0, 2.1, 0.2), 
                                np.arange(2.5, 4.1, 0.5), 
                                np.arange(5.0, 10.1, 2.5),
                                np.array([15, 20, 30., 40., 50.])))
    if not os.path.exists(os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')) or overwrite_atm:
        lon = np.mean(extent[:2])
        lat = np.mean(extent[2:])
        boundary_from_center = 0.25 # degree
        mod_lon = np.array([lon-boundary_from_center, lon+boundary_from_center])
        mod_lat = np.array([lat-boundary_from_center, lat+boundary_from_center])
        mod_extent = [mod_lon[0], mod_lon[1], mod_lat[0], mod_lat[1]]
        
        zpt_filename = f'zpt_{date_s}_{case_tag}.h5'
        
        fname_atm = f'modis_dropsonde_atm_{date_s}_{case_tag}.pk'
        
        status, ws10m = er3t.pre.atm.create_modis_dropsonde_atm(o2mix=0.20935, output_dir=zpt_filedir, output=zpt_filename, 
                                                fname_mod07=modis_07_file, dropsonde_df=data_dropsonde,
                                                levels=levels,
                                                extent=mod_extent, new_h_edge=None,sfc_T_set=None, sfc_h_to_zero=True,)
        
        atm0      = er3t.pre.atm.modis_dropsonde_atmmod(zpt_file=f'{zpt_filedir}/{zpt_filename}',
                            fname=fname_atm, 
                            fname_co2_clim=f'{_fdir_general_}/climatology/cams73_latest_co2_conc_surface_inst_2020.nc',
                            fname_o3_clim=f'{_fdir_general_}/climatology/ozone_merra2_202405_202408.h5',
                            date=date, extent=mod_extent,
                            overwrite=True)
    
        # write out the atmospheric profile in ascii format
        with open(os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat'), 'w') as f:
            header = ('# Adjusted MODIS 07 atmospheric profile\n'
                    '#      z(km)      p(mb)        T(K)    air(cm-3)    o3(cm-3)     o2(cm-3)    h2o(cm-3)    co2(cm-3)     no2(cm-3)\n'
                    )
            # Build all profile lines in one go.
            lines = [
                    f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["pressure"]["data"][i]:11.5f} {atm0.lev["temperature"]["data"][i]:11.3f} '
                    f'{atm0.lev["air"]["data"][i]:12.6e} {atm0.lev["o3"]["data"][i]:12.6e} {atm0.lev["o2"]["data"][i]:12.6e} '
                    f'{atm0.lev["h2o"]["data"][i]:12.6e} {atm0.lev["co2"]["data"][i]:12.6e} {atm0.lev["no2"]["data"][i]:12.6e}'
                    for i in range(len(atm0.lev['altitude']['data']))[::-1]
                    ]
            f.write(header + "\n".join(lines))
        
        with open(f'{zpt_filedir}/ch4_profiles_{date_s}_{case_tag}.dat', 'w') as f:  
            header = ('# Adjusted MODIS 07 atmospheric profile for ch4 only\n'
                    '#      z(km)      ch4(cm-3)\n'
                    )
            lines = [
                    f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["ch4"]["data"][i]:12.6e}'
                    for i in range(len(atm0.lev['altitude']['data']))[::-1]
                    ]
            f.write(header + "\n".join(lines))
    # =================================================================================

    # read aircraft housekeeping data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#
    
    # read ssfr data
    #/----------------------------------------------------------------------------\#
    # fname_ssfr = '%s/R0_submitted-on-20250228/%s-SSFR_%s_%s_R0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    #\----------------------------------------------------------------------------/#
    
    # read HSR1 data
    #ARCSIX-HSR1_P3B_20240531_RA.h5
    #/----------------------------------------------------------------------------\#
    # fname_hsr1 = '%s/R0_submitted-on-20250228/%s-HSR1_%s_%s_R0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    fname_hsr1 = '%s/%s-HSR1_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsr1 = er3t.util.load_h5(fname_hsr1)
    #\----------------------------------------------------------------------------/#


    # read in all logic data
    #/----------------------------------------------------------------------------\#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #\----------------------------------------------------------------------------/#


    # read collocated satellite data
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    
    sat_name_list = []
    sat_time_list = []
    for i, sat_name in enumerate(fnames):
        sat_select_text = os.path.basename(sat_name).replace('.nc', '').replace('CLDPROP_L2_', '')
        sat_name_list.append(sat_select_text.split('.')[0].replace('_', ' '))
        sat_utc = sat_select_text.split('.')[2]
        sat_utc_hh = int(sat_utc[:2])
        sat_utc_mm = int(sat_utc[2:4])
        tmhr = sat_utc_hh + sat_utc_mm / 60.0
        sat_time_list.append(tmhr)
    #\----------------------------------------------------------------------------/#
    sat_select = []
    for i, name in enumerate(fnames):
        print(f"Satellite data {i}: {name}")
    for i in range(len(tmhr_ranges_select)):
        mean_time = np.mean(tmhr_ranges_select[i])
        closest_tmhr_ind = np.argmin(np.abs(sat_time_list - mean_time))
        print(f"Satellite {closest_tmhr_ind} ({sat_name_list[closest_tmhr_ind]}, timm {sat_time_list[closest_tmhr_ind]:.2f}) closest to P3B time {mean_time:.2f} UTC")
        sat_select.append(closest_tmhr_ind)


    # read LWC data
    #/----------------------------------------------------------------------------\#
    if fname_LWC is not None:
        head, data_lwc = read_ict_lwc(fname_LWC, encoding='utf-8', na_values=[-9999999, -777, -888])
    #/----------------------------------------------------------------------------/#
    
    # read bbr data
    #/----------------------------------------------------------------------------\#
    if fname_bbr is not None:
        head, data_bbr = read_ict_bbr(fname_bbr, encoding='utf-8', na_values=[-9999999, -777, -888])
    #/----------------------------------------------------------------------------\#
    
    # read kt19 data
    #/----------------------------------------------------------------------------\#
    if fname_kt19 is not None:
        head, data_kt19 = read_ict_kt19(fname_kt19, encoding='utf-8', na_values=[-9999999, -777, -888])
    #/----------------------------------------------------------------------------\#
        
    # selected stacked legs
    #/----------------------------------------------------------------------------\#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0])&(data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #\----------------------------------------------------------------------------/#


    # write out the surface albedo
    #/----------------------------------------------------------------------------\#
    os.makedirs(f'{_fdir_general_}/sfc_alb', exist_ok=True)
    if not os.path.exists(f'{_fdir_general_}/sfc_alb/sfc_alb_%s.dat' % date_s) or overwrite_alb:
    
        alb_file = 'data_albedo_20240607_low.h5'
        
        with h5py.File(alb_file, 'r') as f:
            # print(f['wvl'])
            alb_wvl = f['wvl'][...]
            alb_inter = f['albedo_interp'][...]
            
        alb_avg = np.nanmean(alb_inter, axis=0)
        
        
        with open(os.path.join(f'{_fdir_general_}/sfc_alb', f'sfc_alb_{date_s}.dat'), 'w') as f:
            header = ('# SSFR derived sfc albedo on 6/7\n'
                    '# wavelength (nm)      albedo (unitless)\n'
                    )
            # Build all profile lines in one go.
            lines = [
                    f'{alb_wvl[i]:11.3f} '
                    f'{alb_avg[i]:12.3e}'
                    for i in range(len(alb_avg))
                    ]
            f.write(header + "\n".join(lines))
    #\----------------------------------------------------------------------------/#
    
    xx = np.linspace(-12, 12, 241)
    yy_gaussian_vis = gaussian(xx, 0, 3.82)
    yy_gaussian_nir = gaussian(xx, 0, 5.10)
    with open(os.path.join('.', 'vis_0.1nm_0710.dat'), 'w') as f_slit:
        header = ('# SSFR Silicon slit function\n'
                    '# wavelength (nm)      relative intensity\n'
                    )
        # Build all profile lines in one go.
        lines = [
                 f'{xx[i]:11.1f} '
                 f'{yy_gaussian_vis[i]:12.5e}'
                 for i in range(len(xx))
                ]
        f_slit.write(header + "\n".join(lines))
        
    xx_wvl_grid = np.arange(360, 1990.1, 5)
    with open(os.path.join('.', 'wvl_grid_test.dat'), 'w') as f_grid:
        # Build all profile lines in one go.
        lines = [
                 f'{xx_wvl_grid[i]:11.1f} '
                 for i in range(len(xx_wvl_grid))
                ]
        f_grid.write(header + "\n".join(lines))
    
    # write out the convolved solar flux
    #/----------------------------------------------------------------------------\#
    wvl_solar_vis = np.arange(300, 950.1, 1)
    wvl_solar_nir = np.arange(951, 2500.1, 1)
    wvl_solar_coarse = np.concatenate([wvl_solar_vis, wvl_solar_nir])
    effective_wvl = wvl_solar_coarse[np.logical_and(wvl_solar_coarse >= xx_wvl_grid.min(), wvl_solar_coarse <= xx_wvl_grid.max())]
    if 1:#not os.path.exists('kurudz_ssfr.dat') or overwrite_lrt:
        df_solor = pd.read_csv('kurudz_0.1nm.dat', sep='\s+', header=None)
        wvl_solar = np.array(df_solor.iloc[:, 0])
        flux_solar = np.array(df_solor.iloc[:, 1])#/1000 # convert mW/m^2/nm to W/m^2/nm
        mask = wvl_solar <= 2500

        wvl_solar = wvl_solar[mask]
        flux_solar = flux_solar[mask]
        flux_solar_convolved = flux_solar.copy()
        
        assert (xx[1]-xx[0]) - (wvl_solar[1]-wvl_solar[0]) <1e-3

        flux_solar_convolved_vis = convolve(flux_solar, yy_gaussian_vis, mode='same') / np.sum(yy_gaussian_vis)
        flux_solar_convolved_nir = convolve(flux_solar, yy_gaussian_nir, mode='same') / np.sum(yy_gaussian_nir)
        flux_solar_convolved[wvl_solar<=950] = flux_solar_convolved_vis[wvl_solar<=950]
        flux_solar_convolved[wvl_solar>950] = flux_solar_convolved_nir[wvl_solar>950]
        
        flux_solar_convolved_coarse = np.zeros_like(wvl_solar_coarse)
        for vis_wvl_i in range(len(wvl_solar_vis)):
            ind = wvl_solar == wvl_solar_vis[vis_wvl_i]
            flux_solar_convolved_coarse[vis_wvl_i] = flux_solar_convolved_vis[ind]
        for nir_wvl_i in range(len(wvl_solar_nir)):
            ind = wvl_solar == wvl_solar_nir[nir_wvl_i]
            flux_solar_convolved_coarse[nir_wvl_i+len(wvl_solar_vis)] = flux_solar_convolved_nir[ind]
        
        with open('kurudz_ssfr.dat', 'w') as f_solar:
            header = ('# SSFR version solar flux\n'
                    '# wavelength (nm)      flux (mW/m^2/nm)\n'
                    )
            # Build all profile lines in one go.
            lines = [
                    f'{wvl_solar_coarse[i]:11.1f} '
                    f'{flux_solar_convolved_coarse[i]:12.5e}'
                    for i in range(len(wvl_solar_coarse))
                    ]
            f_solar.write(header + "\n".join(lines))
        

    

    # read satellite granule
    #/----------------------------------------------------------------------------\#
    import platform
    
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)
    fname_cld_obs_info = '%s/%s_cld_obs_info_%s_%s_%s_0.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag)
    if not os.path.exists(fname_cld_obs_info) or overwrite_cld:
        # save the cloud observation information
        data_ssfr_time = data_ssfr['tmhr']
        data_hsr1_time = data_hsr1['tmhr']
        if fname_LWC is not None:
            data_lwc_time = np.array(data_lwc['tmhr'])
        if fname_bbr is not None:
            data_bbr_time = np.array(data_bbr['tmhr'])
        if fname_kt19 is not None:
            data_kt19_time = np.array(data_kt19['tmhr'])
        
        for i in range(len(tmhr_ranges_select)):
            if platform.system() == 'Darwin':
                fname = '/Volumes/argus/field/arcsix/sat-data/%s/%s' % (date_s, fnames[sat_select[i]])
            elif platform.system() == 'Linux':
                fname = '/pl/active/vikas-arcsix/yuch8913/arcsix/data/sat-data/%s/%s' % (date_s, fnames[sat_select[i]])
                
            # read the cloud observation information
            print('Reading cloud observation information from %s ...' % fname)
            f = Dataset(fname, 'r')
            lon_s = f.groups['geolocation_data'].variables['longitude'][...].data
            lat_s = f.groups['geolocation_data'].variables['latitude'][...].data

            cot_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness'][...].data
            cot_pcl_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_PCL'][...].data

            cer_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius'][...].data
            cer_pcl_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_PCL'][...].data
            
            cwp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Water_Path'][...].data)
            cwp_pcl_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Water_Path_PCL'][...].data)
            
            cwp_s /= 1000 # convert g/m^2 to kg/m^2
            cwp_pcl_s /= 1000 # convert g/m^2 to kg/m

            cth_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Height'][...].data)
            
            cth_s /= 1000 # m to km
            
            ctt_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Temperature'][...].data)

            ctp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Phase_Optical_Properties'][...].data)

            scan_utc_s = f.groups['scan_line_attributes'].variables['scan_start_time'][...].data
            jday_s0 = np.array([er3t.util.dtime_to_jday(datetime.datetime(1993, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=scan_utc_s[i])) for i in range(scan_utc_s.size)])
            f.close()

            logic_pcl = (cot_s<0.0) & (cot_pcl_s>0.0) & (cer_s<0.0) & (cer_pcl_s>0.0)
            cot_s[logic_pcl] = cot_pcl_s[logic_pcl]
            cer_s[logic_pcl] = cer_pcl_s[logic_pcl]
            cwp_s[logic_pcl] = cwp_pcl_s[logic_pcl]
            cwp_s[cwp_s<0.0] = np.nan

            logic_0 = (ctp_s==0.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_1 = (ctp_s==1.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_2 = (ctp_s==2.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_3 = (ctp_s==3.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_4 = (ctp_s==4.0)&((cot_s<0.0)|(cer_s<0.0))

            cot_s[logic_0] = np.nan
            cer_s[logic_0] = np.nan

            cot_s[logic_1] = 0.0
            cer_s[logic_1] = 1.0

            cot_s[logic_2] = -2.0
            cer_s[logic_2] = 1.0

            cot_s[logic_3] = -3.0
            cer_s[logic_3] = 1.0

            cot_s[logic_4] = -4.0
            cer_s[logic_4] = 1.0

            cot_s[cot_s<-10.0] = np.nan
            cer_s[cot_s<-10.0] = np.nan
            
            cgt_s = np.zeros_like(cth_s)
            cgt_s[...] = np.nan
            
            # read CGT_Data.csv
            cgt_table = pd.read_csv('CGT_Data.csv')
            for j in range(len(cgt_table)):
                pixel_select_cwp_low = np.logical_and(np.logical_and(cth_s > cgt_table['cth_low'][j], cth_s <= cgt_table['cth_high'][j]),
                                                    cwp_s < cgt_table['cwp_threshold'][j]/1000)
                pixel_select_cwp_high = np.logical_and(np.logical_and(cth_s > cgt_table['cth_low'][j], cth_s <= cgt_table['cth_high'][j]),
                                                    cwp_s >= cgt_table['cwp_threshold'][j]/1000)
                cgt_s[pixel_select_cwp_low] = cgt_table['a'][j]*cwp_s[pixel_select_cwp_low] + cgt_table['b'][j]
                cgt_s[pixel_select_cwp_high] = cgt_table['a_extra'][j]*cwp_s[pixel_select_cwp_high] + cgt_table['b_extra'][j]
                
            
            ext = [0.13, 0.25, 0.39, 0.55, 0.67] # in km-1
            ctt_thresh = [0, 200, 220, 240, 260, 400]
            # thin cirrus cloud
            for j in range(len(ctt_thresh)-1):
                pixel_select_thin = np.logical_and(cot_s>0, np.logical_and(cot_s < 1., np.logical_and(ctt_s > ctt_thresh[j], ctt_s <= ctt_thresh[j+1])))
                cgt_s[pixel_select_thin] = cot_s[pixel_select_thin]/ext[j]
                

            # sys.exit()
            cgt_s[cgt_s>cth_s] = cth_s[cgt_s>cth_s]
            cbh_s = cth_s - cgt_s
            
            cot_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cot_s.flatten())
            cer_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cer_s.flatten())
            cwp_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cwp_s.flatten())
            cth_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cth_s.flatten())
            cgt_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cgt_s.flatten())
            cbh_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cbh_s.flatten())
        
        
        
        
        
            vars()[f"ssfr_logics_select_{i}_ind"] = []
            vars()[f"hsr1_logics_select_{i}_ind"] = []
            if fname_LWC is not None:
                vars()[f"lwc_logics_select_{i}_ind"] = []
            if fname_bbr is not None:
                vars()[f"bbr_logics_select_{i}_ind"] = []
            if fname_kt19 is not None:
                vars()[f"kt19_logics_select_{i}_ind"] = []
                
            for time_hsk in data_hsk['tmhr'][logics_select[i]]:
                
                vars()[f"ssfr_logics_select_{i}_ind"].append(np.argmin(np.abs(data_ssfr_time - time_hsk)))
                vars()[f"hsr1_logics_select_{i}_ind"].append(np.argmin(np.abs(data_hsr1_time - time_hsk)))
                if fname_LWC is not None:
                    vars()[f"lwc_logics_select_{i}_ind"].append(np.argmin(np.abs(data_lwc_time - time_hsk)))
                if fname_bbr is not None:
                    vars()[f"bbr_logics_select_{i}_ind"].append(np.argmin(np.abs(data_bbr_time - time_hsk)))
                if fname_kt19 is not None:
                    vars()[f"kt19_logics_select_{i}_ind"].append(np.argmin(np.abs(data_kt19_time - time_hsk)))
                
                
            vars()[f"ssfr_logics_select_{i}_ind"] = np.array(vars()[f"ssfr_logics_select_{i}_ind"])
            vars()[f"hsr1_logics_select_{i}_ind"] = np.array(vars()[f"hsr1_logics_select_{i}_ind"])
            if fname_LWC is not None:
                vars()[f"lwc_logics_select_{i}_ind"] = np.array(vars()[f"lwc_logics_select_{i}_ind"])
            if fname_bbr is not None:
                vars()[f"bbr_logics_select_{i}_ind"] = np.array(vars()[f"bbr_logics_select_{i}_ind"])
            if fname_kt19 is not None:
                vars()[f"kt19_logics_select_{i}_ind"] = np.array(vars()[f"kt19_logics_select_{i}_ind"])
            
            vars()["cld_leg_%d" % i] = {}
            vars()["cld_leg_%d" % i]['tmhr'] = data_hsk['tmhr'][logics_select[i]]
            vars()["cld_leg_%d" % i]['lon'] = data_hsk['lon'][logics_select[i]]
            vars()["cld_leg_%d" % i]['lat'] = data_hsk['lat'][logics_select[i]]
            vars()["cld_leg_%d" % i]['sza'] = data_hsk['sza'][logics_select[i]]
            vars()["cld_leg_%d" % i]['saa'] = data_hsk['saa'][logics_select[i]]
            vars()["cld_leg_%d" % i]['ssfr_nad'] = data_ssfr['nad/flux'][vars()[f'ssfr_logics_select_{i}_ind'], :]
            vars()["cld_leg_%d" % i]['ssfr_zen'] = data_ssfr['zen/flux'][vars()[f'ssfr_logics_select_{i}_ind'], :]
            vars()["cld_leg_%d" % i]['hsr1_total'] = data_hsr1['tot/flux'][vars()[f'hsr1_logics_select_{i}_ind']]
            vars()["cld_leg_%d" % i]['hsr1_dif'] = data_hsr1['dif/flux'][vars()[f'hsr1_logics_select_{i}_ind']]
            vars()["cld_leg_%d" % i]['p3_alt'] = data_hsk['alt'][logics_select[i]]/1000 # m to km
            vars()["cld_leg_%d" % i]['cot'] = cot_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
            vars()["cld_leg_%d" % i]['cer'] = cer_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
            vars()["cld_leg_%d" % i]['cwp'] = cwp_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
            vars()["cld_leg_%d" % i]['cth'] = cth_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
            vars()["cld_leg_%d" % i]['cgt'] = cgt_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
            vars()["cld_leg_%d" % i]['cbh'] = cbh_interp(vars()[f"cld_leg_{i}"]['lon'], vars()[f"cld_leg_{i}"]['lat'])
            

            if fname_LWC is not None:
                vars()["cld_leg_%d" % i]['twc'] = np.array(data_lwc['TWC'])[vars()[f"lwc_logics_select_{i}_ind"]]
                vars()["cld_leg_%d" % i]['lwc_1'] = np.array(data_lwc['LWC_1'])[vars()[f"lwc_logics_select_{i}_ind"]]
                vars()["cld_leg_%d" % i]['lwc_2'] = np.array(data_lwc['LWC_2'])[vars()[f"lwc_logics_select_{i}_ind"]]
            if fname_bbr is not None:
                vars()["cld_leg_%d" % i]['down_ir_flux'] = np.array(data_bbr['DN_IR_Irrad'])[vars()[f"bbr_logics_select_{i}_ind"]]
                vars()["cld_leg_%d" % i]['up_ir_flux'] = np.array(data_bbr['UP_IR_Irrad'])[vars()[f"bbr_logics_select_{i}_ind"]]
                vars()["cld_leg_%d" % i]['ir_sky_T'] = np.array(data_bbr['IR_Sky_Temp'])[vars()[f"bbr_logics_select_{i}_ind"]]
            if fname_kt19 is not None:
                vars()["cld_leg_%d" % i]['ir_sfc_T'] = np.array(data_kt19['ir_sfc_T'])[vars()[f"kt19_logics_select_{i}_ind"]]
        
            # save the cloud observation information to a pickle file
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
            with open(fname_pkl, 'wb') as f:
                pickle.dump(vars()[f"cld_leg_{i}"], f, protocol=pickle.HIGHEST_PROTOCOL)
                
                
            # # plot clouds cot, cth
            # label_size = 12
            # plt.close('all')
            # fig = plt.figure(figsize=(16, 6))
            # ax1 = fig.add_subplot(121)
            # ax2 = fig.add_subplot(122)
            # cc1 = ax1.scatter(vars()["cld_leg_%d" % i]['lon'], vars()["cld_leg_%d" % i]['lat'], c=vars()["cld_leg_%d" % i]['cot'], s=3, cmap='viridis')
            # ax1.set_title("Cloud Optical Thickness (COT)", fontsize=label_size+2)
            # ax1.set_xlabel("Longitude", fontsize=label_size)
            # ax1.set_ylabel("Latitude", fontsize=label_size)
            # # ax1.set_xlim(mod_lon.min(), mod_lon.max())
            # # ax1.set_ylim(mod_lat.min(), mod_lat.max())
            # cbar1 = fig.colorbar(cc1, ax=ax1, orientation='vertical', pad=0.02)
            # cbar1.set_label("COT", fontsize=label_size) 
            # cc2 = ax2.scatter(vars()["cld_leg_%d" % i]['lon'], vars()["cld_leg_%d" % i]['lat'], c=vars()["cld_leg_%d" % i]['cth'], s=3, cmap='viridis')
            # ax2.set_title("Cloud Top Height (CTH)", fontsize=label_size+2)
            # ax2.set_xlabel("Longitude", fontsize=label_size)
            # ax2.set_ylabel("Latitude", fontsize=label_size)
            # # ax2.set_xlim(mod_lon.min(), mod_lon.max())
            # # ax2.set_ylim(mod_lat.min(), mod_lat.max())
            # cbar2 = fig.colorbar(cc2, ax=ax2, orientation='vertical', pad=0.02)
            # cbar2.set_label("CTH (km)", fontsize=label_size) 
            # fig.suptitle(f"Clouds on {date}", fontsize=label_size+2, y=0.98)
            # fig.tight_layout()
            # plt.show()
            if platform.system() == 'Darwin':



                
                cot_s0 = cot_s.copy()
                cot_s0[...] = 255.0
                # cs = ax1.pcolormesh(lon_s, lat_s, cot_s0, cmap='gray', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=1.0)
                cot_s[cot_s<=0.0] = np.nan
                ctp_s[ctp_s==1.0] = np.nan
                
                proj0 = ccrs.Orthographic(
                        central_longitude=((extent[0]+extent[1])/2.0),
                        central_latitude=((extent[2]+extent[3])/2.0),
                        )
                plt.close('all')
                fig = plt.figure(figsize=(18, 12))
                ax1 = fig.add_subplot(111, projection=proj0)
                cs_ctp = ax1.pcolormesh(lon_s, lat_s,  ctp_s, cmap='viridis', vmin=0.0, vmax=5.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
                cs_cot = ax1.pcolormesh(lon_s, lat_s,  cot_s, cmap='jet', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)

                ax1.plot(data_hsk['lon'], data_hsk['lat'], lw=2.5, color='k', transform=ccrs.PlateCarree(), zorder=1)

                color = colors1[i]

                text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
                text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
                ax1.scatter(data_hsk['lon'][logics_select[i]], data_hsk['lat'][logics_select[i]], color=color, s=sizes1[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
                # ax1.text(data_hsk['lon'][logics_select[i]][0], data_hsk['lat'][logics_select[i]][0], text1, color=color, fontsize=12, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
                # ax1.text(data_hsk['lon'][logics_select[i]][-1], data_hsk['lat'][logics_select[i]][-1], text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
                # ax1.scatter(f_lon[3], f_lat[3], s=5, marker='^', color='orange')
                # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
                # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
                # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)

                sat_select_text = os.path.basename(fname).replace('.nc', '').replace('CLDPROP_L2_', '')
                sat_ = sat_select_text.split('.')[0].replace('_', ' ')
                sat_utc = sat_select_text.split('.')[2]
                title_text = f'{sat_} {sat_utc}\n' + \
                    f'Flight track {text1} - {text2} UTC'
                ax1.set_title(title_text, fontsize=24, y=1.01, color='k')
                ax1.coastlines(resolution='10m', color='gray', lw=0.5)
                g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
                g1.xlocator = FixedLocator(np.arange(-180, 181, 5.0))
                g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2))
                g1.top_labels = False
                g1.right_labels = False
                

                ax1.set_extent(extent, crs=ccrs.PlateCarree())
                
                cbar1 = fig.colorbar(cs_cot, ax=ax1, orientation='vertical', pad=0.05, aspect=50, shrink=0.8)
                cbar1.set_label('Cloud Optical Thickness', fontsize=16)
                cbar1.ax.tick_params(labelsize=14)
                
                # cbar2 = fig.colorbar(cs_ctp, ax=ax1, orientation='vertical', pad=0.05, aspect=50, shrink=0.8)
                # cbar2.set_label('Cloud Phase', fontsize=16)
                # cbar2.ax.tick_params(labelsize=14)
                #\--------------------------------------------------------------/#

                # save figure
                #/--------------------------------------------------------------\#
                fig.subplots_adjust(hspace=0.3, wspace=0.3)
                _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                fig.savefig('fig/%s/%s_%s_sat_%d_leg_%d.png' % (date_s, date_s, case_tag, sat_select[i], i), bbox_inches='tight', metadata=_metadata)
                #\--------------------------------------------------------------/#
        # sys.exit()
    else:
        print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
        for i in range(len(tmhr_ranges_select)):
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
            with open(fname_pkl, 'rb') as f:
                vars()[f"cld_leg_{i}"] = pickle.load(f)   
    
    solver = 'lrt'

    if not lw:
        if clear_sky:
            fname_h5 = '%s/%s-%s-%s-%s-clear.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
            fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_clear'
            fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
        else:
            if manual_cloud:
                fname_h5 = '%s/%s-%s-%s-%s-manual_cloud.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
                fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_manual_cloud'
                fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_manual_cloud'
            else:
                fname_h5 = '%s/%s-%s-%s-%s-sat_cloud.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
                fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_sat_cloud'
                fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'
    else:
        fdir_lw_zpt_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_lw_zpt'
        os.makedirs(fdir_lw_zpt_tmp, exist_ok=True)
        if clear_sky:
            fname_h5 = '%s/%s-%s-%s-%s-clear-lw.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
            fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_clear-lw'
            fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear-lw'
        else:
            if manual_cloud:
                fname_h5 = '%s/%s-%s-%s-%s-manual_cloud-lw.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
                fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_manual_cloud-lw'
                fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_manual_cloud-lw'
            else:
                fname_h5 = '%s/%s-%s-%s-%s-sat_cloud-lw.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
                fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_sat_cloud-lw'
                fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud-lw'
  
    os.makedirs(fdir_tmp, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    
    if not os.path.exists(fname_h5) or overwrite_lrt:
         
        atm_z_grid = levels
        z_list = atm_z_grid
        atm_z_grid_str = ' '.join(['%.2f' % z for z in atm_z_grid])
        

        flux_output = np.zeros(np.sum([len(data_hsk['lon'][logics_select[k]]) for k in range(len(tmhr_ranges_select))]))
        

        flux_key_all = []
        
        if os.path.exists(f'{fdir}/flux_down_result_dict_sw.pk') and not new_compute:
            print(f'Loading flux_down_result_dict_sw.pk from {fdir} ...')
            with open(f'{fdir}/flux_down_result_dict_sw.pk', 'rb') as f:
                flux_down_result_dict = pickle.load(f)
            with open(f'{fdir}/flux_down_dir_result_dict_sw.pk', 'rb') as f:
                flux_down_dir_result_dict = pickle.load(f)
            with open(f'{fdir}/flux_down_diff_result_dict_sw.pk', 'rb') as f:
                flux_down_diff_result_dict = pickle.load(f)
            with open(f'{fdir}/flux_up_result_dict_sw.pk', 'rb') as f:
                flux_up_result_dict = pickle.load(f)
                
            flux_key_all.extend(flux_down_result_dict.keys())
            print("flux_down_result_dict keys: ", flux_down_result_dict.keys())
            
        else:
            flux_down_result_dict = {}
            flux_down_dir_result_dict = {}
            flux_down_diff_result_dict = {}
            flux_up_result_dict = {}
            
        
        flux_key = np.zeros_like(flux_output, dtype=object)
        cloudy = 0
        clear = 0
        
        # rt initialization
        #/----------------------------------------------------------------------------\#
        lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
        
        if not lw:
            lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')
            # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')
            # lrt_cfg['solar_file'] = None
            lrt_cfg['solar_file'] = 'kurudz_ssfr.dat'
            # lrt_cfg['solar_file'] = lrt_cfg['solar_file'].replace('kurudz_0.1nm.dat', 'kurudz_1.0nm.dat')
            lrt_cfg['number_of_streams'] = 4
            lrt_cfg['mol_abs_param'] = 'reptran coarse'
            # lrt_cfg['mol_abs_param'] = f'reptran medium'
            input_dict_extra_general = {
                                'crs_model': 'rayleigh Bodhaine29',
                                # 'crs_model': 'rayleigh Nicolet',
                                # 'crs_model': 'o3 Bogumil',
                                'albedo_file': f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}.dat',
                                'atm_z_grid': atm_z_grid_str,
                                'wavelength_grid_file': 'wvl_grid_test.dat',
                                # 'no_scattering':'mol',
                                # 'no_absorption':'mol',
                                }
            Nx_effective = len(effective_wvl)
            mute_list = ['albedo', 'wavelength', 'spline']
        else:
            # lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')
            lrt_cfg['number_of_streams'] = 4
            lrt_cfg['mol_abs_param'] = 'reptran coarse'
            # ch4_file = os.path.join(zpt_filedir, f'ch4_profiles_{date_s}_{case_tag}.dat')
            input_dict_extra_general = {
                                'source': 'thermal',
                                'albedo_add': '0',
                                'atm_z_grid': atm_z_grid_str,
                                # 'mol_file': f'CH4 {ch4_file}',
                                # 'wavelength_grid_file': 'wvl_grid_thermal.dat',
                                'wavelength_add' : '4500 42000',
                                'output_process': 'integrate',
                                }
            Nx_effective = 1 # integrate over all wavelengths
            mute_list = ['albedo', 'wavelength', 'spline', 'source solar', 'atmosphere_file']
        #/----------------------------------------------------------------------------/#

        
        inits_rad = []
        flux_key_ix = []
        output_list = []
        tmhr_ranges_length = [len(data_hsk['lon'][logics_select[k]]) for k in range(len(tmhr_ranges_select))]
        length_range = [0] + [np.sum(tmhr_ranges_length[:k+1]) for k in range(len(tmhr_ranges_length))]
        print("len(flux_output): ", len(flux_output))
        print("length_range: ", length_range)
        for ix in range(len(flux_output))[::simulation_interval]:
            length_range_ind = bisect.bisect_left(length_range, ix)
            if length_range_ind > 0 and (ix not in length_range[1:-1]):
                length_range_ind -= 1
            
            cld_leg = vars()[f'cld_leg_{length_range_ind}']
                    
            ind = ix - length_range[length_range_ind]
                    
            # print(f"ix {ix}, index {ind}")
            # print(f"altitude {np.round(cld_leg['p3_alt'][ind], decimals=2)} ...")
                
            cot_x = cld_leg['cot'][ind]
            cwp_x = cld_leg['cwp'][ind]
            sza_x = cld_leg['sza'][ind]
            saa_x = cld_leg['saa'][ind]
            p3_alt_x = cld_leg['p3_alt'][ind]
            p3_alt_x = np.round(p3_alt_x, decimals=2)
            if fname_kt19 is not None:
                sfc_T_kt19 = cld_leg['ir_sfc_T'][ind] + +273.15 # convert to Kelvin
                sfc_T_kt19 = np.round(sfc_T_kt19, decimals=1)
            else:
                sfc_T_kt19 = None
            if not clear_sky:
                input_dict_extra = copy.deepcopy(input_dict_extra_general)
                if ((cot_x >= 0.1 and np.isfinite(cwp_x))) or manual_cloud:
                    cloudy += 1
                    
                    if not manual_cloud:
                        cer_x = cld_leg['cer'][ind]
                        cwp_x = cld_leg['cwp'][ind]
                        cth_x = cld_leg['cth'][ind]
                        cbh_x = cld_leg['cbh'][ind]
                        cgt_x = cld_leg['cgt'][ind]
                    else:
                        # manual cloud properties
                        cer_x = manual_cloud_cer
                        cwp_x = manual_cloud_cwp
                        cth_x = manual_cloud_cth
                        cbh_x = manual_cloud_cbh
                        cot_x = manual_cloud_cot
                        cgt_x = cth_x-cbh_x
                    
                    cth_ind_cld = bisect.bisect_left(z_list, cth_x)
                    cbh_ind_cld = bisect.bisect_left(z_list, cbh_x)
                    cth_ind_atm = bisect.bisect_left(atm_z_grid, cth_x)
                    cbh_ind_atm = bisect.bisect_left(atm_z_grid, cbh_x)
                    
                    fname_cld = f'{fdir_tmp}/cld_{ix:04d}.txt'
                    if os.path.exists(fname_cld):
                        os.remove(fname_cld)
                        
                    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
                    cld_cfg['cloud_file'] = fname_cld
                    cld_cfg['cloud_altitude'] = z_list[cbh_ind_cld:cth_ind_cld+2]#-0.2
                    # cld_cfg['cloud_altitude'] = atm_z_grid[cbh_ind_atm:cth_ind_atm+2]#-0.2
                    cld_cfg['cloud_effective_radius']  = cer_x
                    cld_cfg['liquid_water_content'] = cwp_x*1000/(cgt_x*1000) # convert kg/m^2 to g/m^3
                    cld_cfg['cloud_optical_thickness'] = cot_x
                    # print(cer_x, cwp_x, cot_x, cld_cfg['cloud_altitude'])
                    
                    if not lw:
                        dict_key_arr = np.concatenate(([cld_cfg['cloud_optical_thickness']], [cld_cfg['cloud_effective_radius']], cld_cfg['cloud_altitude'], [p3_alt_x]))
                    else:
                        dict_key_arr = np.concatenate(([cld_cfg['cloud_optical_thickness']], [cld_cfg['cloud_effective_radius']], cld_cfg['cloud_altitude'], [p3_alt_x], [sfc_T_kt19]))
                    dict_key = '_'.join([f'{i:.3f}' for i in dict_key_arr])
                    # input_dict_extra['wc_properties'] = 'mie interpolate'
                else:
                    cld_cfg = None
                    if not lw:
                        dict_key = f'clear {p3_alt_x:.2f}'
                    else:
                        dict_key = f'clear {p3_alt_x:.2f} {sfc_T_kt19:.1f}'
                    clear += 1
            else:
                cld_cfg = None
                if not lw:
                    dict_key = f'clear {p3_alt_x:.2f}'
                else:
                    dict_key = f'clear {p3_alt_x:.2f} {sfc_T_kt19:.1f}'
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
                
                if lw:
                    # generate the new atmospheric profile with kt19 surface temperature for each simulation
                    if sfc_T_kt19 is None:
                        raise ValueError("sfc_T_kt19 must be provided for longwave simulations.")
                    if levels is None:
                        levels = np.concatenate((np.arange(0, 2.1, 0.2), 
                                                np.arange(2.5, 4.1, 0.5), 
                                                np.arange(5.0, 10.1, 2.5),
                                                np.array([15, 20, 30., 40., 50.])))
                    if not os.path.exists(os.path.join(fdir_lw_zpt_tmp, f'atm_profiles_{date_s}_{case_tag}_sfcT_{sfc_T_kt19:.1f}K.dat')) or overwrite_atm:
                        lon = np.mean(extent[:2])
                        lat = np.mean(extent[2:])
                        boundary_from_center = 0.25 # degree
                        mod_lon = np.array([lon-boundary_from_center, lon+boundary_from_center])
                        mod_lat = np.array([lat-boundary_from_center, lat+boundary_from_center])
                        mod_extent = [mod_lon[0], mod_lon[1], mod_lat[0], mod_lat[1]]
                        
                        zpt_filename = f'zpt_{date_s}_{case_tag}_sfcT_{sfc_T_kt19:.1f}K.h5'
                        
                        fname_atm = f'modis_dropsonde_atm_{date_s}_{case_tag}_sfcT_{sfc_T_kt19:.1f}K.pk'
                        
                        status, ws10m = er3t.pre.atm.create_modis_dropsonde_atm(o2mix=0.20935, output_dir=fdir_lw_zpt_tmp, output=zpt_filename, 
                                                                fname_mod07=modis_07_file, dropsonde_df=data_dropsonde,
                                                                levels=levels,
                                                                extent=mod_extent, new_h_edge=None,sfc_T_set=sfc_T_kt19, sfc_h_to_zero=True, plot=False,)
                        
                        atm0      = er3t.pre.atm.modis_dropsonde_atmmod(zpt_file=f'{fdir_lw_zpt_tmp}/{zpt_filename}',
                                            fname=f'{fdir_lw_zpt_tmp}/{fname_atm}', 
                                            fname_co2_clim=f'{_fdir_general_}/climatology/cams73_latest_co2_conc_surface_inst_2020.nc',
                                            fname_o3_clim=f'{_fdir_general_}/climatology/ozone_merra2_202405_202408.h5',
                                            date=date, extent=mod_extent,
                                            overwrite=True, plot=False,)
                    
                        # write out the atmospheric profile in ascii format
                        with open(os.path.join(fdir_lw_zpt_tmp, f'atm_profiles_{date_s}_{case_tag}_{sfc_T_kt19:.1f}K.dat'), 'w') as f:
                            header = ('# Adjusted MODIS 07 atmospheric profile\n'
                                    '#      z(km)      p(mb)        T(K)    air(cm-3)    o3(cm-3)     o2(cm-3)    h2o(cm-3)    co2(cm-3)     no2(cm-3)\n'
                                    )
                            # Build all profile lines in one go.
                            lines = [
                                    f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["pressure"]["data"][i]:11.5f} {atm0.lev["temperature"]["data"][i]:11.3f} '
                                    f'{atm0.lev["air"]["data"][i]:12.6e} {atm0.lev["o3"]["data"][i]:12.6e} {atm0.lev["o2"]["data"][i]:12.6e} '
                                    f'{atm0.lev["h2o"]["data"][i]:12.6e} {atm0.lev["co2"]["data"][i]:12.6e} {atm0.lev["no2"]["data"][i]:12.6e}'
                                    for i in range(len(atm0.lev['altitude']['data']))[::-1]
                                    ]
                            f.write(header + "\n".join(lines))
                        
                        with open(f'{fdir_lw_zpt_tmp}/ch4_profiles_{date_s}_{case_tag}_{sfc_T_kt19:.1f}K.dat', 'w') as f:  
                            header = ('# Adjusted MODIS 07 atmospheric profile for ch4 only\n'
                                    '#      z(km)      ch4(cm-3)\n'
                                    )
                            lines = [
                                    f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["ch4"]["data"][i]:12.6e}'
                                    for i in range(len(atm0.lev['altitude']['data']))[::-1]
                                    ]
                            f.write(header + "\n".join(lines))
                    input_dict_extra['atmosphere_file_add'] = os.path.join(fdir_lw_zpt_tmp, f'atm_profiles_{date_s}_{case_tag}_{sfc_T_kt19:.1f}K.dat')
                    ch4_file = os.path.join(fdir_lw_zpt_tmp, f'ch4_profiles_{date_s}_{case_tag}_{sfc_T_kt19:.1f}K.dat')
                    input_dict_extra['mol_file'] = f'CH4 {ch4_file}'
                
                
                
                # rt setup
                #/----------------------------------------------------------------------------\#
                
                init = er3t.rtm.lrt.lrt_init_mono_flx(
                        input_file  = '%s/input_%04d.txt'  % (fdir_tmp, ix),
                        output_file = '%s/output_%04d.txt' % (fdir_tmp, ix),
                        date        = date,
                        # surface_albedo=0.08,
                        solar_zenith_angle = sza_x,
                        # wavelength         = wavelength,
                        Nx = Nx_effective,
                        output_altitude    = [0, 'toa'],
                        input_dict_extra   = input_dict_extra.copy(),
                        mute_list          = mute_list,
                        lrt_cfg            = lrt_cfg,
                        cld_cfg            = cld_cfg,
                        aer_cfg            = None,
                        # output_format     = 'lambda uu edir edn',
                        )
                #\----------------------------------------------------------------------------/#
                inits_rad.append(copy.deepcopy(init))
                output_list.append('%s/output_%04d.txt' % (fdir_tmp, ix))
                flux_key_all.append(dict_key)
                flux_key_ix.append(dict_key)
                    
        print('len(inits_rad): ', len(inits_rad))
        print("flux_key_all: ", flux_key_all)
        print("flux_key_ix set: ", set(flux_key_ix))
        print("flux_key_all length: ", len(flux_key_all))
        print("flux_key_ix length: ", len(flux_key_ix))
        print("len set(flux_key_ix): ", len(set(flux_key_ix)))
        print("set(flux_key_ix) == set(flux_key_all): ", set(flux_key_ix) == set(flux_key_all))
        # sys.exit()
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
        ##### run several libratran calculations one by one
        
        elif platform.system() == 'Linux':
            if len(inits_rad) > 0:
                print('Running libratran calculations ...')
                for i in range(len(inits_rad)):
                    if not os.path.exists(output_list[i]):
                        er3t.rtm.lrt.lrt_run(inits_rad[i])
                    else:
                        if os.path.getsize(output_list[i]) == 0:
                            er3t.rtm.lrt.lrt_run(inits_rad[i])
                    data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                    flux_down_result_dict[flux_key_all[i]] = np.squeeze(data.f_down)
                    flux_down_dir_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_direct)
                    flux_down_diff_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_diffuse)
                    flux_up_result_dict[flux_key_all[i]] = np.squeeze(data.f_up)
        # #\----------------------------------------------------------------------------/#
        ###### delete input, output, cld txt files
        # for prefix in ['input', 'output', 'cld']:
        #     for filename in glob.glob(os.path.join(fdir_tmp, f'{prefix}_*.txt')):
        #         os.remove(filename)
        ###### delete atmospheric profile files for lw
        if lw:
            if platform.system() == 'Darwin':
                for filename in glob.glob(os.path.join(fdir_lw_zpt_tmp, f'atm_profiles*.dat')):
                    os.remove(filename)
                for filename in glob.glob(os.path.join(fdir_lw_zpt_tmp, f'ch4_profiles*.dat')):
                    os.remove(filename)
            for filename in glob.glob(os.path.join(fdir_lw_zpt_tmp, f'modis_dropsonde_atm_*.pk')):
                os.remove(filename)
            for filename in glob.glob(os.path.join(fdir_lw_zpt_tmp, f'zpt_*.h5')):
                os.remove(filename)
            
        
        # save dict
        status = 'wb'
        with open(f'{fdir}/flux_down_result_dict_sw.pk', status) as f:
            pickle.dump(flux_down_result_dict, f)
        with open(f'{fdir}/flux_down_dir_result_dict_sw.pk', status) as f:
            pickle.dump(flux_down_dir_result_dict, f)
        with open(f'{fdir}/flux_down_diff_result_dict_sw.pk', status) as f:
            pickle.dump(flux_down_diff_result_dict, f)
        with open(f'{fdir}/flux_up_result_dict_sw.pk', status) as f:
            pickle.dump(flux_up_result_dict, f)


        
        flux_output_t = np.zeros(len(range(len(flux_output))))
        f_tmhr = np.zeros(len(flux_output))
        f_lon = np.zeros(len(flux_output_t))
        f_lat = np.zeros(len(flux_output_t))
        f_alt = np.zeros(len(flux_output_t))
        f_sza = np.zeros(len(flux_output_t))
        f_cth = np.zeros(len(flux_output_t))
        f_cbh = np.zeros(len(flux_output_t))
        f_cot = np.zeros(len(flux_output_t))
        f_cwp = np.zeros(len(flux_output_t))
        f_cer = np.zeros(len(flux_output_t))
        f_cgt = np.zeros(len(flux_output_t))
        f_ssfr_zen = np.zeros((len(flux_output_t), len(data_ssfr['zen/wvl'])))
        f_ssfr_nad = np.zeros((len(flux_output_t), len(data_ssfr['nad/wvl'])))
        f_hsr1_total = np.zeros((len(flux_output_t), 401))
        f_hsr1_diff = np.zeros((len(flux_output_t), 401))
        f_twp = np.zeros(len(flux_output_t))
        f_lwc_1 = np.zeros(len(flux_output_t))
        f_lwc_2 = np.zeros(len(flux_output_t))
        f_bbr_up = np.zeros(len(flux_output_t))
        f_bbr_down = np.zeros(len(flux_output_t))
        f_bbr_sky_T = np.zeros(len(flux_output_t))
        f_kt19 = np.zeros(len(flux_output_t))
        
        f_down_1d = np.zeros((len(flux_output_t), Nx_effective, 2))
        f_down_dir_1d = np.zeros((len(flux_output_t), Nx_effective, 2))
        f_down_diff_1d = np.zeros((len(flux_output_t), Nx_effective, 2))
        f_up_1d = np.zeros((len(flux_output_t), Nx_effective, 2))
        
        for f_array in [f_tmhr, f_lon, f_lat, f_alt, f_sza,
                        f_cth, f_cbh, f_cot, f_cwp, f_cer, f_cgt, f_ssfr_zen, f_ssfr_nad, 
                        f_hsr1_total, f_hsr1_diff,
                        f_twp, f_lwc_1, f_lwc_2, 
                        f_bbr_up, f_bbr_down, f_bbr_sky_T, f_kt19,
                        f_down_1d, f_down_dir_1d, f_down_diff_1d, f_up_1d]:
            f_array[...] = np.nan

        
        for ix in range(len(flux_output))[::simulation_interval]:
            length_range_ind = bisect.bisect_left(length_range, ix)
            if length_range_ind > 0 and (ix not in length_range[1:-1]):
                length_range_ind -= 1
            
            cld_leg = vars()[f'cld_leg_{length_range_ind}']
                    
            ind = ix - length_range[length_range_ind]
                    
            f_down_1d[ix] = flux_down_result_dict[flux_key[ix]]
            f_down_dir_1d[ix] = flux_down_dir_result_dict[flux_key[ix]]
            f_down_diff_1d[ix] = flux_down_diff_result_dict[flux_key[ix]]
            f_up_1d[ix] = flux_up_result_dict[flux_key[ix]]
                
                
        for ix in range(len(flux_output)):
            length_range_ind = bisect.bisect_left(length_range, ix)
            if length_range_ind > 0 and (ix not in length_range[1:-1]):
                length_range_ind -= 1
            
            cld_leg = vars()[f'cld_leg_{length_range_ind}']
                    
            ind = ix - length_range[length_range_ind]
            
            f_tmhr[ix] = cld_leg['tmhr'][ind]
            f_lon[ix] = cld_leg['lon'][ind]
            f_lat[ix] = cld_leg['lat'][ind]
            f_alt[ix] = cld_leg['p3_alt'][ind]
            f_sza[ix] = cld_leg['sza'][ind]
            f_ssfr_zen[ix] = cld_leg['ssfr_zen'][ind]
            f_ssfr_nad[ix] = cld_leg['ssfr_nad'][ind]
            f_hsr1_total[ix] = cld_leg['hsr1_total'][ind]
            f_hsr1_diff[ix] = cld_leg['hsr1_dif'][ind]
            if fname_LWC is not None:
                f_twp[ix] = cld_leg['twc'][ind]
                f_lwc_1[ix] = cld_leg['lwc_1'][ind]
                f_lwc_2[ix] = cld_leg['lwc_2'][ind]
            if fname_bbr is not None:
                f_bbr_up[ix] = cld_leg['up_ir_flux'][ind]
                f_bbr_down[ix] = cld_leg['down_ir_flux'][ind]
                f_bbr_sky_T[ix] = cld_leg['ir_sky_T'][ind]
            if fname_kt19 is not None:
                f_kt19[ix] = cld_leg['ir_sfc_T'][ind]
                
            if not clear_sky:
                if not manual_cloud:
                        f_cer[ix] = cld_leg['cer'][ind]
                        f_cwp[ix] = cld_leg['cwp'][ind]
                        f_cth[ix] = cld_leg['cth'][ind]
                        f_cbh[ix] = cld_leg['cbh'][ind]
                        f_cgt[ix] = cld_leg['cgt'][ind]
                        f_cot[ix] = cld_leg['cot'][ind]
                else:
                    # manual cloud properties
                    f_cer[ix] = manual_cloud_cer
                    f_cwp[ix] = manual_cloud_cwp
                    f_cth[ix] = manual_cloud_cth
                    f_cbh[ix] = manual_cloud_cbh
                    f_cot[ix] = manual_cloud_cot
                    f_cgt[ix] = manual_cloud_cth-manual_cloud_cbh
            else:
                f_cth[ix] = 0.0
                f_cbh[ix] = 0.0
                f_cot[ix] = 0.0
                f_cwp[ix] = 0.0
                f_cgt[ix] = 0.0
                f_cer[ix] = 0.0
            
                
        # save rad_2d results
        with h5py.File(fname_h5, 'w') as f:
            f.create_dataset('tmhr', data=f_tmhr)
            f.create_dataset('lon', data=f_lon)
            f.create_dataset('lat', data=f_lat)
            f.create_dataset('alt', data=f_alt)
            f.create_dataset('sza', data=f_sza)
            f.create_dataset('f_down', data=f_down_1d)
            f.create_dataset('f_down_dir', data=f_down_dir_1d)
            f.create_dataset('f_down_diff', data=f_down_diff_1d)
            f.create_dataset('f_up', data=f_up_1d)
            f.create_dataset('cth', data=f_cth)
            f.create_dataset('cbh', data=f_cbh)
            f.create_dataset('cot', data=f_cot)
            f.create_dataset('cwp', data=f_cwp)
            f.create_dataset('cgt', data=f_cgt)
            f.create_dataset('cer', data=f_cer)
            f.create_dataset('ssfr_zen', data=f_ssfr_zen)
            f.create_dataset('ssfr_nad', data=f_ssfr_nad)
            f.create_dataset('ssfr_nad_wvl', data=data_ssfr['nad/wvl'])
            f.create_dataset('ssfr_zen_wvl', data=data_ssfr['zen/wvl'])
            f.create_dataset('ssfr_toa0', data=data_ssfr['zen/toa0'])
            f.create_dataset('hsr1_toa0', data=data_hsr1['tot/toa0'])
            f.create_dataset('hsr1_total', data=f_hsr1_total)
            f.create_dataset('hsr1_diff', data=f_hsr1_diff)
            f.create_dataset('hsr1_wvl', data=data_hsr1['tot/wvl'])
            f.create_dataset('twc', data=f_twp)
            f.create_dataset('lwc_1', data=f_lwc_1)
            f.create_dataset('lwc_2', data=f_lwc_2)
            f.create_dataset('bbr_up', data=f_bbr_up)
            f.create_dataset('bbr_down', data=f_bbr_down)
            f.create_dataset('bbr_sky_T', data=f_bbr_sky_T)
            f.create_dataset('kt19', data=f_kt19)
            
    else:
        print('Loading existing libratran results ...')
        with h5py.File(fname_h5, 'r') as f:
            f_tmhr = f['tmhr'][...]
            f_lon = f['lon'][...]
            f_lat = f['lat'][...]
            f_alt = f['alt'][...]
            f_sza = f['sza'][...]
            f_down_1d = f['f_down'][...]
            f_down_dir_1d = f['f_down_dir'][...]
            f_down_diff_1d = f['f_down_diff'][...]
            f_up_1d = f['f_up'][...]
            f_cth = f['cth'][...]
            f_cbh = f['cbh'][...]
            f_cot = f['cot'][...]
            f_cwp = f['cwp'][...]
            f_cgt = f['cgt'][...]
            f_ssfr_zen = f['ssfr_zen'][...]
            f_ssfr_nad = f['ssfr_nad'][...]
            data_ssfr_wvl_zen = np.array(f['ssfr_zen_wvl'])
            data_ssfr_wvl_nad = np.array(f['ssfr_nad_wvl'])
            data_ssfr_toa0 = np.array(f['ssfr_toa0'])
            data_hsr1_toa0 = np.array(f['hsr1_toa0'])
            data_hsr1_total = np.array(f['hsr1_total'])
            data_hsr1_diff = np.array(f['hsr1_diff'])
            data_hsr1_wvl = np.array(f['hsr1_wvl'])
            f_twp = f['twc'][...]
            f_lwc_1 = f['lwc_1'][...]
            f_lwc_2 = f['lwc_2'][...]
            f_bbr_up = f['bbr_up'][...]
            f_bbr_down = f['bbr_down'][...]
            f_bbr_sky_T = f['bbr_sky_T'][...]
            f_kt19 = f['kt19'][...]
        
        #############

    print("Finished libratran calculations.")  
    #\----------------------------------------------------------------------------/#

    return

if __name__ == '__main__':

    # figure_01_arcsix1_flight_tracks_all()
    # figure_cop_wall_20240605()
    # figure_cop_wall_20240607()
    # figure_cop_wall_20240611()
    # figure_cop_flt_trk_20240605()
    # figure_cop_flt_trk_20240607()
    # figure_cop_flt_trk_20240611()
    # sys.exit()
    
    dir_fig = './fig'
    os.makedirs(dir_fig, exist_ok=True)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 5),
    #                  extent=[-44, -58, 83.3, 84.1],
    #                  sizes1 = [50, 20, 4],
    #                  tmhr_ranges_select=[[15.36, 15.60], [16.32, 16.60], [16.78, 16.85]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240605_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240605_R0.ict',
    #                  case_tag='marli_test',)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 13),
    #                  extent=[-10, -20, 81.9, 82.7],
    #                  sizes1 = [50, 20, 4],
    #                  tmhr_ranges_select=[[13.5, 13.6], [13.67, 13.78]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240613_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240613_R0.ict',
    #                  case_tag='marli_0613_1',)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 11),
    #                  extent=[-60, -70, 85.2, 86.0],
    #                  sizes1 = [50, 20, 4],
    #                  tmhr_ranges_select=[[13.08, 13.21], [13.33, 13.43]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240611_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240611_R0.ict',
    #                  case_tag='marli_test_0611_1',)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 7),
    #                  extent=[-60, -70, 78.0, 79.6],
    #                  sizes1 = [50, 20, 4],
    #                  tmhr_ranges_select=[[13.8, 13.87], [13.92, 14.0]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240607_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240607_R0.ict',
    #                  case_tag='marli_test_0607_1',)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 5, 28),
    #                  extent=[-42, -52, 84.2, 85.9],
    #                  sizes1 = [50, 20, 4],
    #                  tmhr_ranges_select=[[15.66, 15.96]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240528_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240528_R0.ict',
    #                  case_tag='marli_test_0528_1',)
    
    
    # sys.exit()
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 5),
    #             tmhr_ranges_select=[[13, 17], [14.5, 16.1], [15.04, 15.28]],
    #             output_lwp_alt=[False, False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240605_R1.ict',
    #             # fname_cloud_micro=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240605_R1.ict'
    #             )
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 11),
    #             tmhr_ranges_select=[[13.9111, 15.7139], [14.03, 14.075]],
    #             output_lwp_alt=[False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict',
    #             # fname_cloud_micro=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240611_R1.ict'
    #             )
    
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 7),
    #             tmhr_ranges_select=[[15.34, 16.27], [15.765, 15.795]],
    #             output_lwp_alt=[False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240607_R1.ict',
    #             # fname_cloud_micro=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240607104243_R1.ict',
    #             fname_cloud_micro=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240607_R1.ict'
    #             )
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 13),
    #             tmhr_ranges_select=[[14.0, 16.5], [14.92, 15.28], [15.03, 15.11], [15.8, 16.1], [15.88, 15.94]],
    #             output_lwp_alt=[False, False, True, False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240613_R1.ict',
    #             # fname_cloud_micro=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240613_R1.ict'
    #             )
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 3),
    #             tmhr_ranges_select=[[14.51, 15.76], [14.84, 14.97], [13.23, 13.95], [13.39, 13.60], [13.73, 13.83]],
    #             output_lwp_alt=[False, True, False, True, False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240603_R1.ict',
    #             # fname_cloud_micro=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240603_R1.ict'
    #             )
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 6),
    #             tmhr_ranges_select=[[13.48, 14.44], [13.63, 13.71], [13.76, 13.92]],
    #             output_lwp_alt=[True, True, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240606_R1.ict',
    #             # fname_cloud_micro=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240606_R1.ict'
    #             )
    
    
    # sys.exit()
    
    if False:
        for lw in [False, True]:
            for clear_sky in [False, True]:
                for manual_cloud in [False, True]:
                    flt_trk_lrt_para(date=datetime.datetime(2024, 6, 7),
                                    extent = [-55, -40, 83.4, 85.2],
                                    sizes1 = [50, 20, 4],
                                    tmhr_ranges_select=[[15.3400, 15.7583], [15.8403, 16.2653]],
                                    sat_select=[1],
                                    fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240607191800_RA.ict',
                                    fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240607_R0/ARCSIX-AVAPS_G3_20240607160915_R0.ict',
                                    fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240607_R0.ict',
                                    fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240607_R0.ict',
                                    fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240607_R1.ict',
                                    modis_07_file=[f'{_fdir_general_}/sat-data/20240607/MYD07_L2.A2024159.1520.061.2024160161210.hdf'],
                                    simulation_interval=2,
                                    levels=np.concatenate((np.array([0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]),
                                                            np.array([1.25, 1.5, 2.0, 2.5, 3.0, 4.0]), 
                                                            np.arange(5.0, 10.1, 2.5),
                                                            np.array([15, 20, 30., 40., 45.]))),
                                    case_tag='cloudy_track_2_cre',
                                    lw=lw,
                                    clear_sky=clear_sky,
                                    manual_cloud=manual_cloud,
                                    manual_cloud_cer=8.0,
                                    manual_cloud_cwp=0.0229,
                                    manual_cloud_cth=0.47,
                                    manual_cloud_cbh=0.25,
                                    manual_cloud_cot=4.3,
                                    overwrite_atm=False,
                                    overwrite_alb=False,
                                    overwrite_cld=False,
                                    overwrite_lrt=True,
                                    new_compute=True,)
                    
    if True:
        for lw in [False, True]:
            for clear_sky in [False, True]:
                for manual_cloud in [False, True]:    
                    flt_trk_lrt_para(date=datetime.datetime(2024, 6, 13),
                                    extent=[-39, -47, 83.3, 84.1],
                                    sizes1 = [50, 20, 4],
                                    tmhr_ranges_select=[[15.84, 15.88], [15.94, 15.98]],
                                    sat_select=[1],
                                    fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240613183800_RA.ict',
                                    fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240613_R0/ARCSIX-AVAPS_G3_20240613151255_R0.ict',
                                    fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240613_R0.ict',
                                    fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240613_R0.ict',
                                    fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240613_R1.ict',
                                    modis_07_file=[f'{_fdir_general_}/sat-data/20240613/MYD07_L2.A2024165.1610.061.2024166155733.hdf'],
                                    simulation_interval=1,
                                    case_tag='cloudy_track_1_cre',
                                    levels=np.concatenate((np.arange(0.0, 1.01, 0.1),
                                                            np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0]), 
                                                            np.arange(5.0, 10.1, 2.5),
                                                            np.array([15, 20, 30., 40., 45.]))),
                                    lw=lw,
                                    clear_sky=clear_sky,
                                    manual_cloud=manual_cloud,
                                    manual_cloud_cer=14.4,
                                    manual_cloud_cwp=0.06013,
                                    manual_cloud_cth=0.945,
                                    manual_cloud_cbh=0.344,
                                    manual_cloud_cot=6.26,
                                    overwrite_atm=False,
                                    overwrite_alb=False,
                                    overwrite_cld=False,
                                    overwrite_lrt=True,
                                    new_compute=True,
                                    )
    
    if False:
        for lw in [False, True]:
            for clear_sky in [False, True]:
                for manual_cloud in [False, True]:    
                    flt_trk_lrt_para(date=datetime.datetime(2024, 6, 3),
                                                extent=[-42, -48, 83.5, 84.5],
                                            sizes1 = [50, 20, 4],
                                            tmhr_ranges_select=[[14.72, 14.86], [14.95, 15.09]],
                                            sat_select=[1],
                                            fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240603180200_RA.ict',
                                            fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240603_R0/ARCSIX-AVAPS_G3_20240603142310_R0.ict',
                                            fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240603_R0.ict',
                                            fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240603_R0.ict',
                                            fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240603_R1.ict',
                                            modis_07_file=[f'{_fdir_general_}/sat-data/20240603/MYD07_L2.A2024155.1555.061.2024156171946.hdf'],
                                            case_tag='cloudy_track_1_cre',
                                            simulation_interval=1,
                                    levels=np.concatenate((np.arange(0.0, 1.1, 0.1),
                                                        np.arange(1.05, 2.51, 0.05),
                                                            np.array([3.0, 3.5, 4.0]), 
                                                            np.arange(5.0, 10.1, 2.5),
                                                            np.array([15, 20, 30., 40., 45.]))),
                                    lw=lw,
                                    clear_sky=clear_sky,
                                    manual_cloud=manual_cloud,
                                    manual_cloud_cer=5.9,
                                    manual_cloud_cwp=0.1012,
                                    manual_cloud_cth=2.23,
                                    manual_cloud_cbh=0.33,
                                    manual_cloud_cot=25.78,
                                    overwrite_atm=False,
                                    overwrite_alb=False,
                                    overwrite_cld=False,
                                    overwrite_lrt=True,
                                    new_compute=True,
                                    )
    
    if False:
        for lw in [False, True]:
            for clear_sky in [False, True]:
                for manual_cloud in [False, True]:
                    flt_trk_lrt_para(date=datetime.datetime(2024, 6, 6),
                                    extent=[-9, -17, 82.7, 83.7],
                                    sizes1 = [50, 20, 4],
                                    tmhr_ranges_select=[[13.99, 14.18], [14.26, 14.46]],
                                    sat_select=[6, 7],
                                    fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240606234800_RA.ict',
                                    fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240606_R0/ARCSIX-AVAPS_G3_20240606161914_R0.ict',
                                    fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240606_R0.ict',
                                    fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240606_R0.ict',
                                    fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240606_R1.ict',
                                    modis_07_file=[f'{_fdir_general_}/sat-data/20240606/MYD07_L2.A2024158.1620.061.2024159154912.hdf'],
                                    case_tag='cloudy_track_2_cre',
                                    simulation_interval=2,
                                    levels=np.concatenate((np.arange(0.0, 1.61, 0.1),
                                                            np.array([1.8, 2.0, 2.5, 3.0, 4.0]), 
                                                            np.arange(5.0, 10.1, 2.5),
                                                            np.array([15, 20, 30., 40., 45.]))),
                                    lw=lw,
                                    clear_sky=clear_sky,
                                    manual_cloud=manual_cloud,
                                    manual_cloud_cer=6.9,
                                    manual_cloud_cwp=0.0231,
                                    manual_cloud_cth=0.3,
                                    manual_cloud_cbh=0.101,
                                    manual_cloud_cot=5.01,
                                    overwrite_atm=False,
                                    overwrite_alb=False,
                                    overwrite_cld=False,
                                    overwrite_lrt=True,
                                    new_compute=True,
                                )
    
    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 11),
    #                 extent = [-72, -50, 83.4, 84.4],
    #                 sizes1 = [50, 20, 4],
    #                 tmhr_ranges_select=[[13.9111, 14.3417], [15.3528, 15.7139]],
    #                 sat_select=[8],
    #                 fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240611190300_RA.ict',
    #                 fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240611_R0/ARCSIX-AVAPS_G3_20240611143225_R0.ict',
    #                 fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict',
    #                 modis_07_file=[f'{_fdir_general_}/sat-data/20240611/MYD07_L2.A2024163.1450.061.2024164151334.hdf'],
    #                 simulation_interval=3,
    #                 clear_sky=True,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=4.3,
    #                 manual_cloud_cwp=0.477,
    #                 manual_cloud_cth=3.01,
    #                 manual_cloud_cbh=3.00,
    #                 manual_cloud_cot=0.17,
    #                 overwrite_atm=True,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True)
    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 5, 31),
    #                             extent=[-60, -80, 82.4, 84.6],
    #                           sizes1 = [50, 20, 4],
    #                           tmhr_ranges_select=[[14.10, 14.27], [16.49, 16.72]],
    #                           sat_select=[1],
    #                           fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240531183300_RA.ict',
    #                           fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240531_R0/ARCSIX-AVAPS_G3_20240531142150_R0.ict',
    #                           fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240531_R0.ict',
    #                           fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240531_R0.ict',
    #                           fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict',
    #                           modis_07_file=[f'{_fdir_general_}/sat-data/20240531/MOD07_L2.A2024152.1525.061.2024153011814.hdf'],
    #                           case_tag='clear_sky_track_1',
    #                           simulation_interval=1,
    #                           clear_sky=True,
    #                           lw=True,
    #                 overwrite_atm=False,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True,
    #                 new_compute=True,
    #                 )

    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 13),
    #                             extent=[-61, -68, 81.2, 81.8],
    #                           sizes1 = [50, 20, 4],
    #                           tmhr_ranges_select=[[16.78, 16.85], [16.91, 17.00]],
    #                           sat_select=[3],
    #                           fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240613183800_RA.ict',
    #                           fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240613_R0/ARCSIX-AVAPS_G3_20240613151255_R0.ict',
    #                           fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240613_R0.ict',
    #                           fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240613_R0.ict',
    #                           fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240613_R1.ict',
    #                           modis_07_file=[f'{_fdir_general_}/sat-data/20240613/MYD07_L2.A2024165.1610.061.2024166155733.hdf'],
    #                           simulation_interval=1,
    #                           case_tag='clear_sky_track_1',
    #                         #   levels=np.concatenate((np.array([0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]),
    #                         #                 np.array([1.5, 2.0, 3.0, 4.0]), 
    #                         #                 np.arange(5.0, 10.1, 2.5),
    #                         #                 np.array([15, 20, 30., 40., 45.]))),
    #                           clear_sky=True,
    #                           lw=True,
    #                 overwrite_atm=True,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=False,
    #                 new_compute=False,
    #                 )

    
    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 5),
    #                             extent=[-68, -42, 83.0, 84.1],
    #                           sizes1 = [50, 20, 4],
    #                           tmhr_ranges_select=[[15.55, 15.9292], [16.0431, 16.32]],
    #                           sat_select=[1],
    #                           fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240605184800_RA.ict',
    #                           fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240605_R0/ARCSIX-AVAPS_G3_20240605152820_R0.ict',
    #                           fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240605_R0.ict',
    #                           fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240605_R0.ict',
    #                           fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240605_R1.ict',
    #                           modis_07_file=[f'{_fdir_general_}/sat-data/20240605/MYD07_L2.A2024157.1540.061.2024158183620.hdf'],
    #                           case_tag='clear_sky_track_1',
    #                           simulation_interval=1,
    #                           clear_sky=True,
    #                           lw=True,
    #                 overwrite_atm=False,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True,
    #                 new_compute=True,
    #                 )

    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 6),
    #                             extent=[-58, -50, 83.2, 84.0],
    #                           sizes1 = [50, 20, 4],
    #                           tmhr_ranges_select=[[16.54, 16.62], [16.85, 16.94]],
    #                           sat_select=[1],
    #                           fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240606234800_RA.ict',
    #                           fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240606_R0/ARCSIX-AVAPS_G3_20240606161914_R0.ict',
    #                           fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240606_R0.ict',
    #                           fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240606_R0.ict',
    #                           fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240606_R1.ict',
    #                           modis_07_file=[f'{_fdir_general_}/sat-data/20240606/MYD07_L2.A2024158.1620.061.2024159154912.hdf'],
    #                           case_tag='clear_sky_track_1',
    #                           simulation_interval=1,
    #                           clear_sky=True,
    #                           lw=True,
    #                 overwrite_atm=False,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True,
    #                 new_compute=False,
    #                 )

    pass
