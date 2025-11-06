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
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Optional
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
from scipy.interpolate import interp1d
import netCDF4 as nc
from netCDF4 import Dataset as NcDataset
import xarray as xr
from collections import defaultdict
import platform
# mpl.use('Agg')


import er3t

from util.util import gaussian, read_ict_radiosonde, read_ict_dropsonde, read_ict_lwc, read_ict_cloud_micro_2DGRAY50, read_ict_cloud_micro_FCDP, read_ict_bbr, read_ict_kt19
from util.util import read_ict_dlh_h2o, read_ict_co, read_ict_ch4, read_ict_co2, read_ict_o3

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


# --- Configuration ----------------------------------------------------------

@dataclass(frozen=True)
class FlightConfig:
    mission: str
    platform: str
    data_root: Path
    sat_root_mac: Path
    sat_root_linux: Path

    def hsk(self, date_s):    return f"{self.data_root}/{self.mission}-HSK_{self.platform}_{date_s}_v0.h5"
    def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R0.h5"
    def hsr1(self, date_s):   return f"{self.data_root}/{self.mission}-HSR1_{self.platform}_{date_s}_R0.h5"
    def logic(self, date_s):  return f"{self.data_root}/{self.mission}-LOGIC_{self.platform}_{date_s}_RA.h5"
    def sat_coll(self, date_s): return f"{self.data_root}/{self.mission}-SAT-CLD_{self.platform}_{date_s}_v0.h5"
    def marli(self, fname):   return Path(fname)
    def kt19(self, fname):    return Path(fname)
    def sat_nc(self, date_s, raw):  # choose root by platform
        root = self.sat_root_mac if sys.platform=="darwin" else self.sat_root_linux
        return f"{root}/{date_s}/{raw}"

# --- Helpers ----------------------------------------------------------------

def load_h5(path):
    if not os.path.exists(path): raise FileNotFoundError(path)
    return er3t.util.load_h5(str(path))

def parse_sat(path):
    with h5py.File(str(path),"r") as f:
        desc = f["sat/jday"].attrs["description"].split("\n")
    names, tmhrs, files = [], [], []
    for raw in desc:
        base = Path(raw).stem.replace("CLDPROP_L2_","")
        hh, mm = float(base.split(".")[2][:4].rjust(4,"0")[:2]), float(base.split(".")[2][2:4])
        print("hh, mm:", hh, mm)
        tm = hh + mm/60.0
        names.append(base.split(".")[0].replace("_"," "))
        tmhrs.append(tm)
        files.append(Path(raw).name)
    return names, np.array(tmhrs), files

def nearest_indices(t_hsk, mask, times):
    # vectorized nearest‐index lookup per leg
    return np.argmin(np.abs(times[:,None] - t_hsk[mask][None,:]), axis=0)

def closest_indices(available: np.ndarray, targets: np.ndarray):
    # vectorized closest-index
    return np.argmin(np.abs(available[:,None] - targets[None,:]), axis=0)

def gases_insitu(date, gas_dir, 
                         config: Optional[FlightConfig] = None,
                         plot=False,
                         time_select_range=None,):
    
    log = logging.getLogger("Gases Profile")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")

    # 1) Load hsk and gases data
    data_hsk  = load_h5(config.hsk(date_s))
    h2o_filename = f"{gas_dir}/ARCSIX-DLH-H2O_P3B_{date_s}_R0.ict"
    ch4_filename = f"{gas_dir}/ARCSIX-TraceGas-CH4_P3B_{date_s}_R0.ict"
    co2_filename = f"{gas_dir}/ARCSIX-TraceGas-CO2_P3B_{date_s}_R0.ict"
    co_filename = f"{gas_dir}/ARCSIX-TraceGas-CO_P3B_{date_s}_R0.ict"
    o3_filename = f"{gas_dir}/ARCSIX-TraceGas-O3_P3B_{date_s}_R0.ict"
    head, data_dlh_h2o = read_ict_dlh_h2o(h2o_filename, encoding="utf-8", na_values=[-9999,-777,-888])
    head, data_ch4 = read_ict_ch4(ch4_filename, encoding="utf-8", na_values=[-9999,-777,-888])
    head, data_co2 = read_ict_co2(co2_filename, encoding="utf-8", na_values=[-9999,-777,-888])
    head, data_co = read_ict_co(co_filename, encoding="utf-8", na_values=[-9999,-777,-888])
    head, data_o3 = read_ict_o3(o3_filename, encoding="utf-8", na_values=[-9999,-777,-888])
    
    dropsonde_file_list, dropsonde_date_list, dropsonde_tmhr_list, dropsonde_lon_list, dropsonde_lat_list = dropsonde_time_loc_list()
    date_select = dropsonde_date_list == date.date()
    if np.sum(date_select) == 0:
        print(f"No dropsonde data found for date {date.strftime('%Y-%m-%d')}")
        data_dropsonde_alt = [np.nan]
        data_dropsonde_h2o = [np.nan]
    else:
        data_dropsonde_alt = []
        data_dropsonde_h2o = []
        for dropsonde_file in np.array(dropsonde_file_list)[date_select]:
            head, data_dropsonde_tmp = read_ict_dropsonde(dropsonde_file, encoding='utf-8', na_values=[-9999999, -777, -888])
            data_dropsonde_alt.extend(data_dropsonde_tmp['alt']/1000)  # convert to km
            data_dropsonde_h2o.extend(data_dropsonde_tmp['h2o_mr'])  # in g/kg
        data_dropsonde_alt = np.array(data_dropsonde_alt)
        data_dropsonde_h2o = np.array(data_dropsonde_h2o)
        data_dropsonde_h2o = data_dropsonde_h2o / 18.01528 / 1000 * 28.97 * 1e6  # convert to ppmv
    
    
    # 2) Get the altitude from hsk
    hsk_time = np.array(data_hsk['tmhr'])
    hsk_alt = np.array(data_hsk['alt'])/1000  # convert to km
    hsk_alt_interp = interp1d(hsk_time, hsk_alt, bounds_error=False, fill_value=np.nan)
    
    # 3) Interpolate altitude to gases data time
    dlh_h2o_time = np.array(data_dlh_h2o['tmhr'])
    ch4_time = np.array(data_ch4['tmhr'])
    co2_time = np.array(data_co2['tmhr'])
    co_time = np.array(data_co['tmhr'])
    o3_time = np.array(data_o3['tmhr'])
    
    dlh_h2o_alt = hsk_alt_interp(dlh_h2o_time)
    ch4_alt = hsk_alt_interp(ch4_time)
    co2_alt = hsk_alt_interp(co2_time)
    co_alt = hsk_alt_interp(co_time)
    o3_alt = hsk_alt_interp(o3_time)
    
    # 4) calculate the gases profiles based on altitude bins
    alt_bins = np.arange(0, 10, 0.1)  # 0 to 10 km, 0.1 km interval
    alt_bin_centers = (alt_bins[:-1] + alt_bins[1:]) / 2
    h2o_profile = np.full(alt_bin_centers.shape, np.nan)
    ch4_profile = np.full(alt_bin_centers.shape, np.nan)
    co2_profile = np.full(alt_bin_centers.shape, np.nan)
    co_profile = np.full(alt_bin_centers.shape, np.nan)
    o3_profile = np.full(alt_bin_centers.shape, np.nan)
    
    h2o_dropsonde_profile = np.full(alt_bin_centers.shape, np.nan)
    
    h2o_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    ch4_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    co2_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    co_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    o3_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    
    h2o_dropsonde_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    
    for i in range(len(alt_bin_centers)):
        alt_min = alt_bins[i]
        alt_max = alt_bins[i+1]
        
        # H2O
        mask_h2o = (dlh_h2o_alt >= alt_min) & (dlh_h2o_alt < alt_max) & (~np.isnan(data_dlh_h2o['h2o_vmr']))
        if np.any(mask_h2o):
            h2o_profile[i] = np.nanmean(data_dlh_h2o['h2o_vmr'][mask_h2o])
            h2o_profile_unc[i] = np.nanstd(data_dlh_h2o['h2o_vmr'][mask_h2o]) / np.sqrt(np.sum(mask_h2o))
            
        # Dropsonde H2O
        mask_h2o_dropsonde = (data_dropsonde_alt >= alt_min) & (data_dropsonde_alt < alt_max)
        if np.any(mask_h2o_dropsonde):
            h2o_dropsonde_profile[i] = np.nanmean(data_dropsonde_h2o[mask_h2o_dropsonde])
            h2o_dropsonde_profile_unc[i] = np.nanstd(data_dropsonde_h2o[mask_h2o_dropsonde]) / np.sqrt(np.sum(mask_h2o_dropsonde))
        
        # CH4
        mask_ch4 = (ch4_alt >= alt_min) & (ch4_alt < alt_max) & (~np.isnan(data_ch4['ch4']))
        if np.any(mask_ch4):
            ch4_profile[i] = np.nanmean(data_ch4['ch4'][mask_ch4])
            ch4_profile_unc[i] = np.nanstd(data_ch4['ch4'][mask_ch4]) / np.sqrt(np.sum(mask_ch4))
        
        # CO2
        mask_co2 = (co2_alt >= alt_min) & (co2_alt < alt_max) & (~np.isnan(data_co2['co2']))
        if np.any(mask_co2):
            co2_profile[i] = np.nanmean(data_co2['co2'][mask_co2])
            co2_profile_unc[i] = np.nanstd(data_co2['co2'][mask_co2]) / np.sqrt(np.sum(mask_co2))
        
        # CO
        mask_co = (co_alt >= alt_min) & (co_alt < alt_max) & (~np.isnan(data_co['co']))
        if np.any(mask_co):
            co_profile[i] = np.nanmean(data_co['co'][mask_co])
            co_profile_unc[i] = np.nanstd(data_co['co'][mask_co]) / np.sqrt(np.sum(mask_co))
        
        # O3
        mask_o3 = (o3_alt >= alt_min) & (o3_alt < alt_max) & (~np.isnan(data_o3['o3']))
        if np.any(mask_o3):
            o3_profile[i] = np.nanmean(data_o3['o3'][mask_o3])
            o3_profile_unc[i] = np.nanstd(data_o3['o3'][mask_o3]) / np.sqrt(np.sum(mask_o3))
    
    # 5) Save the profiles to a csv file
    df_profiles = pd.DataFrame({
        'Altitude_km': alt_bin_centers,
        'H2O_VMR_ppm': h2o_profile,
        'H2O_VMR_unc_ppm': h2o_profile_unc,
        'H2O_Dropsonde_VMR_ppm': h2o_dropsonde_profile,
        'H2O_Dropsonde_VMR_unc_ppm': h2o_dropsonde_profile_unc,
        'CH4_VMR_ppm': ch4_profile,
        'CH4_VMR_unc_ppm': ch4_profile_unc,
        'CO2_VMR_ppm': co2_profile,
        'CO2_VMR_unc_ppm': co2_profile_unc,
        'CO_VMR_ppb': co_profile,
        'CO_VMR_unc_ppb': co_profile_unc,
        'O3_VMR_ppb': o3_profile,
        'O3_VMR_unc_ppb': o3_profile_unc,
    })
    os.makedirs(f'{_fdir_general_}/zpt/{date_s}', exist_ok=True)
    df_profiles.to_csv(f'{_fdir_general_}/zpt/{date_s}/{date_s}_gases_profiles.csv', index=False)
    log.info(f"Saved gases profiles to {_fdir_general_}/zpt/{date_s}/{date_s}_gases_profiles.csv")
    
    
    if plot:
        # plot the gases profiles
        plt.close('all')
        fig, axs = plt.subplots(1, 5, figsize=(20, 6))
        ax1, ax2, ax3, ax4, ax5 = axs
        ax1.errorbar(h2o_profile, alt_bin_centers, xerr=h2o_profile_unc, fmt='b.-', alpha=0.7)
        ax1.plot(h2o_profile, alt_bin_centers, 'b.-', label='DLH in-situ')
        ax1.errorbar(h2o_dropsonde_profile, alt_bin_centers, xerr=h2o_dropsonde_profile_unc, fmt='k.--', alpha=0.7)
        ax1.plot(h2o_dropsonde_profile, alt_bin_centers, 'k.--', label='Dropsonde')
        ax1.legend(fontsize=12)
        ax1.set_xlabel('H$_2$O VMR (ppm)', fontsize=14)
        ax1.set_ylabel('Altitude (km)', fontsize=14)
        ax1.set_title('H$_2$O Profile', fontsize=16)
        
        ax2.errorbar(ch4_profile, alt_bin_centers, xerr=ch4_profile_unc, fmt='g.-', alpha=0.7)
        ax2.plot(ch4_profile, alt_bin_centers, 'g.-')
        ax2.set_xlabel('CH$_4$ VMR (ppb)', fontsize=14)
        ax2.set_title('CH$_4$ Profile', fontsize=16)
        ax2.yaxis.set_visible(False)
        
        ax3.errorbar(co2_profile, alt_bin_centers, xerr=co2_profile_unc, fmt='r.-', alpha=0.7)
        ax3.plot(co2_profile, alt_bin_centers, 'r.-')
        ax3.set_xlabel('CO$_2$ VMR (ppm)', fontsize=14)
        ax3.set_title('CO$_2$ Profile', fontsize=16)
        ax3.yaxis.set_visible(False)
        
        ax4.errorbar(co_profile, alt_bin_centers, xerr=co_profile_unc, fmt='m.-', alpha=0.7)
        ax4.plot(co_profile, alt_bin_centers, 'm.-')
        ax4.set_xlabel('CO VMR (ppb)', fontsize=14)
        ax4.set_title('CO Profile', fontsize=16)
        ax4.yaxis.set_visible(False)
        
        ax5.errorbar(o3_profile, alt_bin_centers, xerr=o3_profile_unc, fmt='c.-', alpha=0.7)        
        ax5.plot(o3_profile, alt_bin_centers, 'c.-')
        ax5.set_xlabel('O$_3$ VMR (ppb)', fontsize=14)
        ax5.set_title('O$_3$ Profile', fontsize=16)
        ax5.yaxis.set_visible(False)
        for ax in axs:
            ax.tick_params(labelsize=12)
            ax.grid(True)
            ax.set_ylim(0, 10)
        fig.tight_layout(pad=2.0)
        fig.suptitle(f'Gases Vertical Profiles {date_s}', fontsize=18, y=1.02)
        fig.savefig('%s/zpt/%s/%s_gases_profiles.png' % (_fdir_general_, date_s, date_s), bbox_inches='tight')
        # plt.show()
        
        if time_select_range is not None:
            for time_start, time_end in time_select_range:
                time_mask_hsk = (data_hsk['tmhr']>=time_start) & (data_hsk['tmhr']<=time_end)
                time_mask_dlh_h2o = (data_dlh_h2o['tmhr']>=time_start) & (data_dlh_h2o['tmhr']<=time_end)
                time_mask_ch4 = (data_ch4['tmhr']>=time_start) & (data_ch4['tmhr']<=time_end)
                time_mask_co2 = (data_co2['tmhr']>=time_start) & (data_co2['tmhr']<=time_end)
                time_mask_co = (data_co['tmhr']>=time_start) & (data_co['tmhr']<=time_end)
                time_mask_o3 = (data_o3['tmhr']>=time_start) & (data_o3['tmhr']<=time_end)
                
                
                # plot time series
                plt.close('all')
                fig, axs = plt.subplots(2, 2, figsize=(14, 8))
                ax1 = axs[0, 0]
                ax2 = axs[0, 1]
                ax3 = axs[1, 0]
                ax4 = axs[1, 1]
                
                ax1.plot(dlh_h2o_time[time_mask_dlh_h2o], data_dlh_h2o['h2o_vmr'][time_mask_dlh_h2o], 'b.-', label='DLH H2O VMR')
                ax1.set_ylabel('H2O VMR (ppm)', fontsize=14)
                ax1_1 = ax1.twinx()
                ax1_1.plot(co2_time[time_mask_co2], data_co2['co2'][time_mask_co2], 'r.-', label='CO2 VMR')
                ax1_1.set_ylabel('CO2 VMR (ppm)', fontsize=14, color='r')
                ax1.set_xlabel('Time (hours)', fontsize=14)
                
                ax2.plot(ch4_time[time_mask_ch4], data_ch4['ch4'][time_mask_ch4], 'g.-', label='CH4 VMR')
                ax2.set_ylabel('CH4 VMR (ppb)', fontsize=14)
                ax2_1 = ax2.twinx()
                ax2_1.plot(co2_time[time_mask_co2], data_co2['co2'][time_mask_co2], 'r.-', label='CO2 VMR')
                ax2.set_xlabel('Time (hours)', fontsize=14)
                
                ax3.plot(co_time[time_mask_co], data_co['co'][time_mask_co], 'm.-', label='CO VMR')
                ax3.set_ylabel('CO VMR (ppb)', fontsize=14)
                ax3_1 = ax3.twinx()
                ax3_1.plot(co2_time[time_mask_co2], data_co2['co2'][time_mask_co2], 'r.-', label='CO2 VMR')
                ax3_1.set_ylabel('CO2 VMR (ppm)', fontsize=14, color='r')
                ax3.set_xlabel('Time (hours)', fontsize=14)
                
                ax4.plot(o3_time[time_mask_o3], data_o3['o3'][time_mask_o3], 'c.-', label='O3 VMR')
                ax4.set_ylabel('O3 VMR (ppb)', fontsize=14)
                ax4_1 = ax4.twinx()
                ax4_1.plot(co2_time[time_mask_co2], data_co2['co2'][time_mask_co2], 'r.-', label='CO2 VMR')
                ax4_1.set_ylabel('CO2 VMR (ppm)', fontsize=14, color='r')
                ax4.set_xlabel('Time (hours)', fontsize=14) 
                
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.tick_params(labelsize=12)
                    # ax.grid(True)
                    ax.legend(fontsize=12)
                fig.suptitle(f'Gases Time Series {date_s} {time_start:.2f}-{time_end:.2f} UTC', fontsize=16)
                fig.tight_layout(pad=2.0)
                plt.show()

def dropsonde_time_loc_list(dir_dropsonde='data/dropsonde'):
    """
    Get the dropsonde time list from the dropsonde directory.
    
    Parameters
    ----------
    dir_dropsonde : str
        The directory of dropsonde files.
        
    Returns
    -------
    time_list : list of datetime
        The list of dropsonde times.
    file_list : list of str
        The list of dropsonde file names.
    """
    file_list = sorted(glob.glob(os.path.join(dir_dropsonde, '*.ict')))
    date_list = []
    tmhr_list = []
    lon_list = []
    lat_list = []
    for f in file_list:
        fname = os.path.basename(f)
        # Example filename: ARCSIX-AVAPS_G3_20240531_R0/ARCSIX-AVAPS_G3_20240531142150_R0.ict
        date_str = fname.split('_')[2]  # '20240531142150'
        date_time = datetime.datetime.strptime(date_str, '%Y%m%d%H%M%S')
        date_list.append(date_time.date())
        
        tmhr_list.append(date_time.hour + date_time.minute/60 + date_time.second/3600)
        
        head, data_dropsonde = read_ict_dropsonde(f, encoding='utf-8', na_values=[-9999999, -777, -888])
        lon_list.append(np.mean(data_dropsonde['lon_all']))
        lat_list.append(np.mean(data_dropsonde['lat_all']))
    return np.array(file_list), np.array(date_list), np.array(tmhr_list), np.array(lon_list), np.array(lat_list)

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
    
    config = FlightConfig(mission='ARCSIX',
                            platform='P3B',
                            data_root=_fdir_data_,
                            sat_root_mac='/Volumes/argus/field/arcsix/sat-data',
                            sat_root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data/sat-data',)
    
    for date in [
                 datetime.datetime(2024, 5, 28),
                 datetime.datetime(2024, 5, 30),
                 datetime.datetime(2024, 5, 31),
                 datetime.datetime(2024, 6, 3),
                 datetime.datetime(2024, 6, 5),
                 datetime.datetime(2024, 6, 6),
                 datetime.datetime(2024, 6, 7),
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

        gases_insitu(date=date, 
                    gas_dir=f'{_fdir_general_}/gases', 
                    config=config,
                    plot=True,
                    )
    
 
    


    pass
