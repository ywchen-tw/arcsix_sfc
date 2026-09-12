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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
import bisect
import pandas as pd
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import netCDF4 as nc
from netCDF4 import Dataset as NcDataset
import xarray as xr
from collections import defaultdict
import platform
import gc
import netCDF4
from pyproj import Transformer
# mpl.use('Agg')


import er3t
from er3t.pre.abs import abs_rep

from util import gaussian, read_ict_radiosonde, read_ict_dropsonde, read_ict_lwc, read_ict_cloud_micro_2DGRAY50, read_ict_cloud_micro_FCDP, read_ict_bbr, read_ict_kt19
from util import read_ict_dlh_h2o, read_ict_co, read_ict_ch4, read_ict_co2, read_ict_o3

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


# --- Configuration ----------------------------------------------------------

@dataclass(frozen=True)
class FlightConfig:
    mission: str
    platform: str
    data_root: Path
    root_mac: Path
    root_linux: Path

    def hsk(self, date_s):    return f"{self.data_root}/{self.mission}-HSK_{self.platform}_{date_s}_v0.h5"
    # def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R0.h5"
    # def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R1_test.h5"
    # def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R2_test_before_corr.h5"
    # def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_V1_test_before_corr.h5"
    # def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_V1_test_after_corr.h5"
    # def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_V2_test_before_corr.h5"
    # def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_V2_test_after_corr.h5"
    # def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R3_V2_test_after_corr.h5"
    def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R4_V2_test_after_corr.h5"
    def ssrr(self, date_s):  return f"{self.data_root}/{self.mission}-SSRR_{self.platform}_{date_s}_RA.h5"
    def hsr1(self, date_s):   return f"{self.data_root}/{self.mission}-HSR1_{self.platform}_{date_s}_R0.h5"
    def logic(self, date_s):  return f"{self.data_root}/{self.mission}-LOGIC_{self.platform}_{date_s}_RA.h5"
    def sat_coll(self, date_s): return f"{self.data_root}/{self.mission}-SAT-CLD_{self.platform}_{date_s}_v0.h5"
    def marli(self, date_s):   
        root = self.root_mac if sys.platform=="darwin" else self.root_linux
        return f"{root}/marli/ARCSIX-MARLi_P3B_{date_s}_R0.cdf"
    def kt19(self, fname):    return Path(fname)
    def sat_nc(self, date_s, raw):  # choose root by platform
        root = self.root_mac if sys.platform=="darwin" else self.root_linux
        return f"{root}/sat-data/{date_s}/{raw}"

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


def alb_combine():
    
    #
    
    """ 
    #save
    alb_update_dict = {
        'wvl': alb_wvl,
        'alb_iter0': alb_ratio_all,
        'alb_iter1': alb1_all,
        'alb_iter2': alb2_all,
        'p3_up_dn_ratio_1': p3_ratio1_all,
        'p3_up_dn_ratio_2': p3_ratio2_all,
        'lon_avg': lon_avg_all,
        'lat_avg': lat_avg_all,
        'lon_min': lon_min_all,
        'lon_max': lon_max_all,
        'lat_min': lat_min_all,
        'lat_max': lat_max_all,
        'alt_avg': alt_avg_all,
        'modis_alb_legs': np.array(modis_alb_legs) if modis_alb_file is not None else None,
        'modis_bands_nm': modis_bands_nm if modis_alb_file is not None else None,
    }
    with open(f'{_fdir_general_}/sfc_alb/sfc_alb_update_{date_s}_{case_tag}.pkl', 'wb') as f:
        pickle.dump(alb_update_dict, f)
    """
    
    # list all sfc_alb files in _fdir_general_/sfc_alb
    fdir_cld_obs_info = f'{_fdir_general_}/sfc_alb'
    sfc_alb_files = sorted(glob.glob(os.path.join(fdir_cld_obs_info, f'sfc_alb_update_*.pkl')))
    
    # combine all files
    alb_wvl = None
    alb_ratio_all = []
    alb1_all = []
    alb2_all = []
    p3_ratio1_all = []
    p3_ratio2_all = []
    lon_avg_all = []
    lat_avg_all = []
    lon_min_all = []
    lon_max_all = []
    lat_min_all = []
    lat_max_all = []
    alt_avg_all = []
    modis_alb_legs_all = []
    modis_bands_nm = None
    date_all = []
    
    for i, fname in enumerate(sfc_alb_files):
        print("Loading file:", fname)
        with open(fname, 'rb') as f:
            alb_update_dict = pickle.load(f)
        date_s = os.path.basename(fname).split('_')[3]
        date_all.extend([date_s]*len(alb_update_dict['alb_iter0']))
        if alb_wvl is None:
            alb_wvl= alb_update_dict['wvl']
        
        # alb_ratio_all.append(alb_update_dict['alb_iter0'])
        # alb1_all.append(alb_update_dict['alb_iter1'])
        # # print("alb_update_dict['alb_iter1'] shape:", np.array(alb_update_dict['alb_iter1']).shape)
        # alb2_all.append(alb_update_dict['alb_iter2'])
        # p3_ratio1_all.append(alb_update_dict['p3_up_dn_ratio_1'])
        # p3_ratio2_all.append(alb_update_dict['p3_up_dn_ratio_2'])
        # lon_avg_all.append(alb_update_dict['lon_avg'])
        # lat_avg_all.append(alb_update_dict['lat_avg'])
        # lon_min_all.append(alb_update_dict['lon_min'])
        # lon_max_all.append(alb_update_dict['lon_max'])
        # lat_min_all.append(alb_update_dict['lat_min'])
        # lat_max_all.append(alb_update_dict['lat_max'])
        # alt_avg_all.append(alb_update_dict['alt_avg'])
        
        if i == 0:
            alb_ratio_all = np.array(alb_update_dict['alb_iter0'])
            alb1_all = np.array(alb_update_dict['alb_iter1'])
            alb2_all = np.array(alb_update_dict['alb_iter2'])
            p3_ratio1_all = np.array(alb_update_dict['p3_up_dn_ratio_1'])
            p3_ratio2_all = np.array(alb_update_dict['p3_up_dn_ratio_2'])
            lon_avg_all = np.array(alb_update_dict['lon_avg'])
            lat_avg_all = np.array(alb_update_dict['lat_avg'])
            lon_min_all = np.array(alb_update_dict['lon_min'])
            lon_max_all = np.array(alb_update_dict['lon_max'])
            lat_min_all = np.array(alb_update_dict['lat_min'])
            lat_max_all = np.array(alb_update_dict['lat_max'])
            alt_avg_all = np.array(alb_update_dict['alt_avg'])
            if 'modis_alb_legs' in alb_update_dict and alb_update_dict['modis_alb_legs'] is not None:
                modis_alb_legs_all = np.array(alb_update_dict['modis_alb_legs'])
                if alb_update_dict['modis_bands_nm'] is not None and len(modis_alb_legs_all)>0 and modis_bands_nm is None:
                    modis_bands_nm = alb_update_dict['modis_bands_nm']
            else:
                modis_alb_legs_all = np.full((len(alb_ratio_all), 7), np.nan)  # assuming 7 MODIS bands
            
                
        else:
            alb_ratio_all = np.vstack((alb_ratio_all, np.array(alb_update_dict['alb_iter0'])))
            alb1_all = np.vstack((alb1_all, np.array(alb_update_dict['alb_iter1'])))
            alb2_all = np.vstack((alb2_all, np.array(alb_update_dict['alb_iter2'])))
            p3_ratio1_all = np.vstack((p3_ratio1_all, np.array(alb_update_dict['p3_up_dn_ratio_1'])))
            p3_ratio2_all = np.vstack((p3_ratio2_all, np.array(alb_update_dict['p3_up_dn_ratio_2'])))
            lon_avg_all = np.hstack((lon_avg_all, np.array(alb_update_dict['lon_avg'])))
            lat_avg_all = np.hstack((lat_avg_all, np.array(alb_update_dict['lat_avg'])))
            lon_min_all = np.hstack((lon_min_all, np.array(alb_update_dict['lon_min'])))
            lon_max_all = np.hstack((lon_max_all, np.array(alb_update_dict['lon_max'])))
            lat_min_all = np.hstack((lat_min_all, np.array(alb_update_dict['lat_min'])))
            lat_max_all = np.hstack((lat_max_all, np.array(alb_update_dict['lat_max'])))
            alt_avg_all = np.hstack((alt_avg_all, np.array(alb_update_dict['alt_avg'])))
            if 'modis_alb_legs' in alb_update_dict and alb_update_dict['modis_alb_legs'] is not None:
                modis_alb_legs_all = np.vstack((modis_alb_legs_all, np.array(alb_update_dict['modis_alb_legs'])))
            else:
                modis_alb_legs_all = np.vstack((modis_alb_legs_all, np.full((len(alb_update_dict['alb_iter0']), 7), np.nan)))
            
    # alb_ratio_all = np.array(alb_ratio_all)
    # alb1_all = np.array(alb1_all)
    # alb2_all = np.array(alb2_all)
    # p3_ratio1_all = np.array(p3_ratio1_all)
    # p3_ratio2_all = np.array(p3_ratio2_all)
    # lon_avg_all = np.array(lon_avg_all)
    # lat_avg_all = np.array(lat_avg_all)
    # lon_min_all = np.array(lon_min_all)
    # lon_max_all = np.array(lon_max_all)
    # lat_min_all = np.array(lat_min_all)
    # lat_max_all = np.array(lat_max_all)
    # alt_avg_all = np.array(alt_avg_all)
    # modis_alb_legs_all = np.array(modis_alb_legs_all)
    date_all = np.array(date_all)   
    
    print("lon_min_all shape:", lon_min_all.shape)
    print("alb1_all shape:", alb1_all.shape)
            
    # 1. Define your numerical data range
    data_min, data_max = 0, 1

    # 2. Create a Normalize instance
    norm = mcolors.Normalize(vmin=data_min, vmax=data_max)

    # 3. Choose a Colormap
    cmap = cm.jet # Or any other built-in colormap like cm.viridis

    # 4. Create a ScalarMappable
    s_m = cm.ScalarMappable(norm=norm, cmap=cmap)

    color_series = s_m.to_rgba(np.arange(len(alt_avg_all))/(len(alt_avg_all)-1))
    
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
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    date_set = sorted(set(date_all))
    color_series = s_m.to_rgba(np.arange(len(date_set))/(len(date_set)-1))
    alb1_date_mean = np.full((len(date_set), len(alb_wvl)), np.nan)
    alb1_date_std = np.full((len(date_set), len(alb_wvl)), np.nan)
    modis_date_mean = np.full((len(date_set), len(modis_bands_nm)), np.nan)
    modis_date_std = np.full((len(date_set), len(modis_bands_nm)), np.nan)
    for i in range(len(date_set)):
        date_i = date_set[i]
        idxs = [j for j in range(len(date_all)) if date_all[j]==date_i]
        alb1_date_mean[i, :] = np.nanmean(np.array(alb1_all)[idxs, :], axis=0)
        alb1_date_std[i, :] = np.nanstd(np.array(alb1_all)[idxs, :], axis=0)
        modis_date_mean[i, :] = np.nanmean(np.array(modis_alb_legs_all)[idxs, :], axis=0)
        modis_date_std[i, :] = np.nanstd(np.array(modis_alb_legs_all)[idxs, :], axis=0)
        ax.fill_between(alb_wvl, alb1_date_mean[i, :]-alb1_date_std[i, :], alb1_date_mean[i, :]+alb1_date_std[i, :], color=color_series[i], alpha=0.3)
        ax.plot(alb_wvl, alb1_date_mean[i, :], '-', color=color_series[i], label=f'{date_i}, N={len(idxs)}')
        
        ax.scatter(modis_bands_nm, modis_date_mean[i, :], s=50, color=color_series[i], marker='*', edgecolors='k') 
    
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.9, zorder=10)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(360, 2100)
    fig.suptitle(f'Atmospheric Corrected Surface Albedo Comparison', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/updated_alb_comparison_spring.png', bbox_inches='tight', dpi=150)
    # plt.show()
    
def atm_corr_plot(date=datetime.datetime(2024, 5, 31),
                     extent=[-60, -80, 82.4, 84.6],
                     tmhr_ranges_select=[[14.10, 14.27]],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                     simulation_interval: Optional[float] = None,  # in minutes, e.g., 10 for 10 minutes
                     rsp_plot=False,
                            ):
    log = logging.getLogger("atm corr spiral plot")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    doy_s = date.timetuple().tm_yday
    print(f"Processing date: {date_s}, DOY: {doy_s}")
    
    if simulation_interval is not None:
        # split tmhr_ranges_select into smaller intervals
        tmhr_ranges_select_new = []
        for lo, hi in tmhr_ranges_select:
            t_start = lo
            while t_start < hi and t_start < (hi - 0.0167/6):  # 1 minute
                t_end = min(t_start + simulation_interval/60.0, hi)
                tmhr_ranges_select_new.append([t_start, t_end])
                t_start = t_end
    tmhr_ranges_select = tmhr_ranges_select_new

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    
    modis_alb_dir = f'{_fdir_general_}/modis_albedo'
    # list all modis albedo files
    modis_alb_files = sorted(glob.glob(os.path.join(modis_alb_dir, f'M*.nc')))
    for fname in modis_alb_files:
        print("Checking modis file:", os.path.basename(fname).split('.')[1])
        if str(doy_s) in os.path.basename(fname).split('.')[1]:
            modis_alb_file = fname
            break
    else:
        modis_alb_file = None

        
    if modis_alb_file is not None:
        with nc.Dataset(modis_alb_file, 'r') as ds:
            modis_lon = ds.variables['Longitude'][:]
            modis_lat = ds.variables['Latitude'][:]
            modis_bands = ds.variables['Bands'][:]
            modis_sur_alb = ds.variables['Albedo_1km'][:]
        
        modis_alb_legs = []
        modis_bands_nm = np.array([float(i) for i in modis_bands[:7]])*1000  # in nm   
            
            
    print("modis_alb_file:", modis_alb_file)
    
    
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    for i in range(len(tmhr_ranges_select)):
        print(f"Processing time range {i+1}/{len(tmhr_ranges_select)}: {tmhr_ranges_select[i][0]:.2f} - {tmhr_ranges_select[i][1]:.2f} UTC")
        fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
        with open(fname_pkl, 'rb') as f:
            cld_leg = pickle.load(f)
        time = cld_leg['time']
        time_start = tmhr_ranges_select[i][0]
        time_end = tmhr_ranges_select[i][1]
        alt_avg = np.nanmean(data_hsk['alt'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])/1000  # in km
        lon_avg = np.nanmean(data_hsk['lon'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])
        lat_avg = np.nanmean(data_hsk['lat'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])
        lon_all = data_hsk['lon'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)]
        lat_all = data_hsk['lat'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)]
        
    
        ratio_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_0.dat'
        update_1_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_1.dat'
        update_2_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_2.dat'
        update_p3_1_fname = f'{_fdir_general_}/sfc_alb/p3_up_dn_ratio_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}_1.dat'
        update_p3_2_fname = f'{_fdir_general_}/sfc_alb/p3_up_dn_ratio_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}_2.dat'
        
        if "cloudy" in case_tag:
            fdir_spiral = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'
        else:
            fdir_spiral = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
        ori_csv_name = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_0.csv'
        updated_csv_name_1 = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_1.csv'
        updated_csv_name_2 = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_2.csv'
        
        
        # p3_up_dn_ratio_20240605_13.79_13.81_5.80_1.dat
        
        alb_ratio = pd.read_csv(ratio_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        alb_1 = pd.read_csv(update_1_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        alb_2 = pd.read_csv(update_2_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        p3_ratio_1 = pd.read_csv(update_p3_1_fname, delim_whitespace=True, comment='#', names=['wvl', 'ratio'])
        p3_ratio_2 = pd.read_csv(update_p3_2_fname, delim_whitespace=True, comment='#', names=['wvl', 'ratio'])
        
        df_ori = pd.read_csv(ori_csv_name)
        df_upd1 = pd.read_csv(updated_csv_name_1)
        df_upd2 = pd.read_csv(updated_csv_name_2)
        
        if i == 0:
            time_all = []
            fdn_550_all = []
            fup_550_all = []
            fdn_1600_all = []
            fup_1600_all = []
            alb_wvl = alb_ratio['wvl'].values
            alb_ratio_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            alb1_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            alb2_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            p3_ratio1_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            p3_ratio2_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            lon_avg_all = np.zeros(len(tmhr_ranges_select))
            lat_avg_all = np.zeros(len(tmhr_ranges_select))
            lon_min_all = np.zeros(len(tmhr_ranges_select))
            lon_max_all = np.zeros(len(tmhr_ranges_select))
            lat_min_all = np.zeros(len(tmhr_ranges_select))
            lat_max_all = np.zeros(len(tmhr_ranges_select))
            alt_avg_all = np.zeros(len(tmhr_ranges_select))

            
            flux_wvl = df_ori['wvl'].values
            ssfr_fup_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fup_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            toa_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_up_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_dn_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_up_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssrr_rad_dn_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            
            rsp_rad_mean_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_rad_std_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_ref_mean_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_ref_std_all = np.zeros((len(tmhr_ranges_select), 9))
            rsp_mu0_mean_all = np.zeros(len(tmhr_ranges_select))
            rsp_sd_mean_all = np.zeros(len(tmhr_ranges_select))

            

        time_all.append(time)
        ssfr_wvl = cld_leg['ssfr_zen_wvl']
        ssfr_550_ind = np.argmin(np.abs(ssfr_wvl - 550))
        ssfr_1600_ind = np.argmin(np.abs(ssfr_wvl - 1600))
        fdn_550_all.append(cld_leg['ssfr_zen'][:, ssfr_550_ind])
        fup_550_all.append(cld_leg['ssfr_nad'][:, ssfr_550_ind])
        fdn_1600_all.append(cld_leg['ssfr_zen'][:, ssfr_1600_ind])
        fup_1600_all.append(cld_leg['ssfr_nad'][:, ssfr_1600_ind])
        
        
        alb_ratio_all[i, :] = alb_ratio['alb'].values
        alb1_all[i, :] = alb_1['alb'].values
        alb2_all[i, :] = alb_2['alb'].values
        p3_ratio1_all[i, :] = p3_ratio_1['ratio'].values
        p3_ratio2_all[i, :] = p3_ratio_2['ratio'].values
        lon_avg_all[i] = lon_avg.copy()
        lat_avg_all[i] = lat_avg.copy()
        lon_min_all[i] = lon_all.min()
        lon_max_all[i] = lon_all.max()
        lat_min_all[i] = lat_all.min()
        lat_max_all[i] = lat_all.max()
        alt_avg_all[i] = alt_avg.copy()
        
        
        ssfr_fup_mean_all[i, :] = df_ori['ssfr_fup_mean'].values
        ssfr_fdn_mean_all[i, :] = df_ori['ssfr_fdn_mean'].values
        ssfr_fup_std_all[i, :] = df_ori['ssfr_fup_std'].values
        ssfr_fdn_std_all[i, :] = df_ori['ssfr_fdn_std'].values
        simu_fup_mean_all_iter0[i, :] = df_ori['simu_fup_mean'].values
        simu_fdn_mean_all_iter0[i, :] = df_ori['simu_fdn_mean'].values
        simu_fup_mean_all_iter1[i, :] = df_upd1['simu_fup_mean'].values
        simu_fdn_mean_all_iter1[i, :] = df_upd1['simu_fdn_mean'].values
        simu_fup_mean_all_iter2[i, :] = df_upd2['simu_fup_mean'].values
        simu_fdn_mean_all_iter2[i, :] = df_upd2['simu_fdn_mean'].values
        toa_mean_all[i, :] = df_ori['toa_mean'].values
        
        ssrr_rad_up_mean_all[i, :] = df_ori['ssrr_rad_up_mean'].values
        ssrr_rad_dn_mean_all[i, :] = df_ori['ssrr_rad_dn_mean'].values
        ssrr_rad_up_std_all[i, :] = df_ori['ssrr_rad_up_std'].values
        ssrr_rad_dn_std_all[i, :] = df_ori['ssrr_rad_dn_std'].values
        ssrr_rad_up_wvl = df_ori['ssrr_nad_wvl'].values
        ssrr_rad_dn_wvl = df_ori['ssrr_zen_wvl'].values
        
        rsp_wvl = cld_leg['rsp_wvl']
        rsp_mu0 = cld_leg['rsp_mu0']
        rsp_sd = cld_leg['rsp_sd']
        print("cld_leg['rsp_rad'] shape:", cld_leg['rsp_rad'].shape)
        
        rsp_rad_mean_all[i, :] = np.nanmean(cld_leg['rsp_rad'], axis=0)/1000  # in W m-2 sr-1 nm-1
        rsp_rad_std_all[i, :] = np.nanstd(cld_leg['rsp_rad'], axis=0)/1000  # in W m-2 sr-1 nm-1
        
        rsp_ref_mean_all[i, :] = np.nanmean(cld_leg['rsp_ref'], axis=0)
        rsp_ref_std_all[i, :] = np.nanstd(cld_leg['rsp_ref'], axis=0)
        
        rsp_mu0_mean_all[i] = np.nanmean(cld_leg['rsp_mu0'])
        rsp_sd_mean_all[i] = np.nanmean(cld_leg['rsp_sd'])
        
        # plt.close('all')
        # plt.plot(rsp_wvl, rsp_rad_mean_all[i, :], '.-', label=f'Leg {i} mean')
        # plt.plot(cld_leg['ssrr_nad_wvl'], np.nanmean(cld_leg['ssrr_nad_rad'], axis=0), 'x--', label=f'Leg {i} SSFR nadir')
        # plt.xlabel('Wavelength (nm)', fontsize=14)
        # plt.ylabel('Radiance (mW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=14)
        # plt.show()
        
        # plt.close('all')
        # plt.scatter(cld_leg['rsp_lon'], cld_leg['rsp_lat'], c='b', s=20, marker='o')
        # plt.scatter(cld_leg['lon'], cld_leg['lat'], c='r', s=20, marker='x')
        # plt.show()
        # sys.exit()
        
        
        print(f"date_s: {date_s}, time: {time_start:.2f}-{time_end:.2f}, alt_avg: {alt_avg:.2f} km")
    
        # find the modis location closest to the flight leg center
        if modis_alb_file is not None:
            dist = np.sqrt((modis_lon - lon_avg)**2 + (modis_lat - lat_avg)**2)
            min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            modis_alb_leg = modis_sur_alb[min_idx[0], min_idx[1], :7]
            modis_alb_legs.append(modis_alb_leg)
    
    # save alb1 and alb2 to a pkl file
    alb_update_dict = {
        'wvl': alb_wvl,
        'alb_iter0': alb_ratio_all,
        'alb_iter1': alb1_all,
        'alb_iter2': alb2_all,
        'p3_up_dn_ratio_1': p3_ratio1_all,
        'p3_up_dn_ratio_2': p3_ratio2_all,
        'lon_avg': lon_avg_all,
        'lat_avg': lat_avg_all,
        'lon_min': lon_min_all,
        'lon_max': lon_max_all,
        'lat_min': lat_min_all,
        'lat_max': lat_max_all,
        'alt_avg': alt_avg_all,
        'modis_alb_legs': np.array(modis_alb_legs) if modis_alb_file is not None else None,
        'modis_bands_nm': modis_bands_nm if modis_alb_file is not None else None,
    }
    with open(f'{_fdir_general_}/sfc_alb/sfc_alb_update_{date_s}_{case_tag}.pkl', 'wb') as f:
        pickle.dump(alb_update_dict, f)
    log.info(f"Saved surface albedo updates to {_fdir_general_}/sfc_alb/sfc_alb_update_{date_s}_{case_tag}.pkl")

    ssrr_ref = ssrr_rad_up_mean_all * np.pi / toa_mean_all / rsp_mu0_mean_all[:, np.newaxis] * (rsp_sd_mean_all[:, np.newaxis]**2)
    ssfr_fup_ref = ssfr_fup_mean_all / toa_mean_all /  rsp_mu0_mean_all[:, np.newaxis] * (rsp_sd_mean_all[:, np.newaxis]**2)
       
    print("rsp_wvl:", rsp_wvl)
    
    # print("ssfr_fup_ref shape:", ssfr_fup_ref.shape)
    # sys.exit()
        
    print("lon avg all mean:", lon_avg_all.mean())
    print("lat avg all mean:", lat_avg_all.mean())
        
    

    
        
    
    

    
    
    

    
    
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(tmhr_ranges_select)):
        ax.plot(alb_ratio['wvl'], alb_ratio_all[i, :], '-', color=color_series[i], label='Z=%.2fkm' % (alt_avg_all[i]))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('SSFR upward/downward ratio', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # ax.vlines(950, -0.05, 1.05, color='gray', linestyle='--')
    # ax.vlines(1050, -0.05, 1.05, color='gray', linestyle='--')
    # ax.vlines(1250, -0.05, 1.05, color='gray', linestyle='--')
    # ax.vlines(1300, -0.05, 1.05, color='skyblue', linestyle='--')
    # ax.vlines(1350, -0.05, 1.05, color='cyan', linestyle='--')
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'albedo used: R0 SSFR upward/downward ratio', fontsize=13)
    fig.suptitle(f'SSFR upward/downward ratio Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_up_dn_ratio_comparison.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(tmhr_ranges_select)):
        ax.plot(alb_ratio['wvl'], p3_ratio1_all[i, :], '-', color=color_series[i], label='Z=%.2fkm, lon:%.2f, lat: %.2f' % (alt_avg_all[i], lon_avg_all[i], lat_avg_all[i]))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('P3 level upward/downward ratio', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(360, 2100)
    ax.set_title(f'albedo used: updated surface albedo (Odell)', fontsize=13)
    fig.suptitle(f'P3 level upward/downward ratio Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_up_dn_ratio_comparison_updated_1.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(tmhr_ranges_select)):
        ax.plot(alb_ratio['wvl'], p3_ratio2_all[i, :], '-', color=color_series[i], label='Z=%.2fkm, lon:%.2f, lat: %.2f' % (alt_avg_all[i], lon_avg_all[i], lat_avg_all[i]))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('P3 level upward/downward ratio', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(360, 2100)
    ax.set_title(f'albedo used: updated surface albedo (fit)', fontsize=13)
    fig.suptitle(f'P3 level upward/downward ratio Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_up_dn_ratio_comparison_updated_2.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    # plt.show()
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(tmhr_ranges_select)):
        ax.plot(alb_ratio['wvl'], alb_ratio_all[i, :], '-', color=color_series[i], label='Z=%.2fkm, lon:%.2f, lat: %.2f' % (alt_avg_all[i], lon_avg_all[i], lat_avg_all[i]))
        ax.scatter(modis_bands_nm, modis_alb_legs[i], s=50, color=color_series[i], marker='*', edgecolors='k') if modis_alb_file is not None else None
    # ax.plot(alb_ratio['wvl'], alb_sfc_extrap_all, '--', color='gray', label='Ratio Extrapolated to Surface')
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('SSFR upward/downward ratio', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    
    # ax.set_title(f'albedo used: updated surface albedo (1)', fontsize=13)
    ax.set_title(f'SSFR measurement', fontsize=13)
    fig.suptitle(f'SSFR upward/downward ratio Comparison', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_SSFR_ratio_comparison.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i in range(len(tmhr_ranges_select)):
        ax.plot(alb_ratio['wvl'], alb1_all[i, :], '-', color=color_series[i], label='Z=%.2fkm, lon:%.2f, lat: %.2f' % (alt_avg_all[i], lon_avg_all[i], lat_avg_all[i]))
        ax.scatter(modis_bands_nm, modis_alb_legs[i], s=50, color=color_series[i], marker='*', edgecolors='k') if modis_alb_file is not None else None
    
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Updated Surface Albedo (Odell)', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(360, 2100)
    ax.set_title(f'albedo used: updated surface albedo (Odell)', fontsize=13)
    fig.suptitle(f'Updated Surface Albedo (Odell) Comparison', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_updated_alb_1_comparison.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    # plt.show()
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    for i in range(len(tmhr_ranges_select)):
        ax.plot(alb_ratio['wvl'], alb2_all[i, :], '-', color=color_series[i], label='Z=%.2fkm, lon:%.2f, lat: %.2f' % (alt_avg_all[i], lon_avg_all[i], lat_avg_all[i]))
        ax.scatter(modis_bands_nm, modis_alb_legs[i], s=50, color=color_series[i], marker='*', edgecolors='k') if modis_alb_file is not None else None
    
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Updated Surface Albedo (fit)', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(360, 2100)
    ax.set_title(f'albedo used: updated surface albedo (fit)', fontsize=13)
    fig.suptitle(f'Updated Surface Albedo (fit) Comparison', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_updated_alb_2_comparison.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    # plt.show()
    
    
    
    # output_dict = {
    #             'wvl': cld_leg['ssfr_zen_wvl'],
    #             'ssfr_fup_mean': fup_mean,
    #             'ssfr_fdn_mean': fdn_mean,
    #             'ssfr_fup_std': fup_std,
    #             'ssfr_fdn_std': fdn_std,
    #             'simu_fup_mean': Fup_p3_mean_interp,
    #             'simu_fdn_mean': Fdn_p3_mean_interp,
    #             'toa_mean':
    #         }
            
    #         output_df = pd.DataFrame(output_dict)
    #         output_df.to_csv(f'{fdir}/ssfr_simu_flux_{date_s}_{time_start:.2f}-{time_end:.2f}_alt-{alt_avg:.2f}km_iteration_{iter}.csv', index=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), sharex=True, sharey='row')
    ax1.plot(flux_wvl, toa_mean_all[0, :], '--', color='gray', label='TOA')
        
    for alt_ind in range(len(tmhr_ranges_select)):
        ax1.tick_params(labelsize=12)
        ax1.plot(flux_wvl, ssfr_fdn_mean_all[alt_ind, :], color=color_series[alt_ind], 
                 label='Z=%.2fkm, lon:%.2f to %.2f' % (alt_avg_all[alt_ind], lon_min_all[alt_ind], lon_max_all[alt_ind]))

        ax2.tick_params(labelsize=12)
        ax2.plot(flux_wvl, ssfr_fup_mean_all[alt_ind, :], color=color_series[alt_ind], 
                 label='Z=%.2fkm, lon:%.2f to %.2f' % (alt_avg_all[alt_ind], lon_min_all[alt_ind], lon_max_all[alt_ind]))

        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    # ax1.vlines(950, -0.05, 1.05, color='gray', linestyle='--')
    # ax1.vlines(1050, -0.05, 1.05, color='gray', linestyle='--')
    # ax1.vlines(1200, -0.05, 1.05, color='g', linestyle='--')
    # ax1.vlines(1225, -0.05, 1.05, color='cyan', linestyle='--')
    # ax1.vlines(1250, -0.05, 1.05, color='gray', linestyle='--')
    # ax1.vlines(1300, -0.05, 1.05, color='skyblue', linestyle='--')
    # ax1.vlines(1350, -0.05, 1.05, color='cyan', linestyle='--')
    
    
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    
    ax1.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    
    ax1.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    # ax4.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    
    ax2.legend(fontsize=10,)
    # ax6.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_obs_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    
    wvl1 = 550
    wvl2 = 650
    wvl3 = 760
    wvl4 = 940
    wvl5 = 1050
    wvl6 = 1250
    wvl7 = 1600
    wvl1_idx = np.argmin(np.abs(flux_wvl - wvl1))
    wvl2_idx = np.argmin(np.abs(flux_wvl - wvl2))
    wvl3_idx = np.argmin(np.abs(flux_wvl - wvl3))
    wvl4_idx = np.argmin(np.abs(flux_wvl - wvl4))
    wvl5_idx = np.argmin(np.abs(flux_wvl - wvl5))
    wvl6_idx = np.argmin(np.abs(flux_wvl - wvl6))
    wvl7_idx = np.argmin(np.abs(flux_wvl - wvl7))
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'orange']
    wvl_list = [wvl1, wvl2, wvl3, wvl4, wvl5, wvl6, wvl7]
    wvl_idx_list = [wvl1_idx, wvl2_idx, wvl3_idx, wvl4_idx, wvl5_idx, wvl6_idx, wvl7_idx]
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.5))
    for ind in range(len(tmhr_ranges_select)):
        ax1.plot(time_all[ind], fdn_550_all[ind], '-', color='b', label='550 nm downward' if ind==0 else None)
        ax1.plot(time_all[ind], fup_550_all[ind], '-', color='r', label='550 nm upward' if ind==0 else None)
        ax2.plot(time_all[ind], fdn_1600_all[ind], '-', color='b', label='1600 nm downward' if ind==0 else None)
        ax2.plot(time_all[ind], fup_1600_all[ind], '-', color='r', label='1600 nm upward' if ind==0 else None)
    ax1.set_xlabel('Time (UTC)', fontsize=14)
    ax1.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax1.set_title('SSFR 550 nm Flux', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=12)
    ax2.set_xlabel('Time (UTC)', fontsize=14)
    ax2.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_title('SSFR 1600 nm Flux', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=12)
    fig.suptitle(f'SSFR Flux Time Series {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_time_series_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '--x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '--x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax1.plot(simu_fdn_mean_all_iter0[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter0 {wvl} nm', color=color, alpha=0.7)
        
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax2.plot(simu_fup_mean_all_iter0[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter0 {wvl} nm', color=color, alpha=0.7)
        
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax3.plot(simu_fup_mean_all_iter0[:, wvl_idx]/simu_fdn_mean_all_iter0[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter0 {wvl} nm', color=color, alpha=0.7)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # reverse y axis for pressure
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_in_alt_all_iter0.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '--x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax1.plot(simu_fdn_mean_all_iter1[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter1 {wvl} nm', color=color, alpha=0.7)
        
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax2.plot(simu_fup_mean_all_iter1[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter1 {wvl} nm', color=color, alpha=0.7)
        
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax3.plot(simu_fup_mean_all_iter1[:, wvl_idx]/simu_fdn_mean_all_iter1[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter1 {wvl} nm', color=color, alpha=0.7)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # reverse y axis for pressure
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_in_alt_all_iter1.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6), sharex=True, sharey='row')
    for i in range(len(wvl_list)):
        wvl = wvl_list[i]
        wvl_idx = wvl_idx_list[i]
        color = color_list[i]
        # ax1.plot(toa_mean_all[0, wvl_idx], alt_avg_all, '--x', label='TOA', color=color)
        ax1.plot(ssfr_fdn_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax1.plot(simu_fdn_mean_all_iter2[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter2 {wvl} nm', color=color, alpha=0.7)
        
        ax2.plot(ssfr_fup_mean_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax2.plot(simu_fup_mean_all_iter2[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter12 {wvl} nm', color=color, alpha=0.7)
        
        ax3.plot(alb_ratio_all[:, wvl_idx], alt_avg_all, '-o', label=f'{wvl} nm', color=color)
        
        ax3.plot(simu_fup_mean_all_iter2[:, wvl_idx]/simu_fdn_mean_all_iter2[:, wvl_idx],
                 alt_avg_all, '--x', label=f'Sim Iter2 {wvl} nm', color=color, alpha=0.7)
        
    ax1.set_title('SSFR Obs Downward Flux', fontsize=12)
    ax2.set_title('SSFR Obs Upward Flux', fontsize=12)
    ax3.set_title('SSFR Up/Down Flux Ratio', fontsize=12)
    ax1.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_xlabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax3.set_xlabel('Up/Down Flux Ratio', fontsize=14)   
    ax1.set_ylabel('Altitude (km)', fontsize=14)
    ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # reverse y axis for pressure
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    fig.suptitle(f'SSFR Flux Spiral Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_alt_profile_in_alt_all_iter2.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sim0_obs_dn_diff_perc = (simu_fdn_mean_all_iter0 - ssfr_fdn_mean_all)/ssfr_fdn_mean_all * 100
    for alt_ind in range(len(tmhr_ranges_select)):
        ax.plot(flux_wvl, sim0_obs_dn_diff_perc[alt_ind, :], '-', color=color_series[alt_ind], 
                label='Z=%.2fkm, lon:%.2f to %.2f' % (alt_avg_all[alt_ind], lon_min_all[alt_ind], lon_max_all[alt_ind]))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Sim-Obs Down/Obs (%)', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.axhline(0, color='k', linestyle='--', alpha=0.7)
    ax.set_ylim(-50, 50)
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    # ax.set_title(f'Simulated - Observed Downward Flux Difference Percentage\nalb=SSFR up/down flux ratio', fontsize=13)
    fig.suptitle(f'SSFR Downward Flux Difference Percentage Comparison {date_s}', fontsize=16, y=0.98)
    fig.suptitle(f'Downward Flux Sim vs Obs {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_flux_dn_diff_perc_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
    
    if rsp_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax2 = ax.twinx()
        for alt_ind in range(len(tmhr_ranges_select)):
            ax.plot(flux_wvl, ssrr_ref[alt_ind, :], '-', color=color_series[alt_ind], 
                    label='Z=%.2fkm, lon:%.2f to %.2f' % (alt_avg_all[alt_ind], lon_min_all[alt_ind], lon_max_all[alt_ind]))
            ax.scatter(rsp_wvl, rsp_ref_mean_all[alt_ind, :], marker='o', color=color_series[alt_ind], edgecolors='k', s=50, alpha=0.7)
                
            ax2.plot(flux_wvl, ssfr_fup_ref[alt_ind, :], '--', color=color_series[alt_ind], alpha=0.7)
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('SSRR radiance x $\pi$ / TOA', fontsize=14)
        ax2.set_ylabel('SSFR Upward flux / TOA', fontsize=14)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.16, 0.5))
        # plt.grid(True)
        ax.set_ylim(-0.05, 1.2)
        ax2.set_ylim(-0.05, 1.2)
        ax.tick_params(labelsize=12)
        # ax.set_title(f'Simulated - Observed Downward Flux Difference Percentage\nalb=SSFR up/down flux ratio', fontsize=13)
        fig.suptitle(f'Upward Radiance and Flux Ref Comparison {date_s}', fontsize=16, y=0.98)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssfr_ssrr_ref_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
        
        for alt_ind in range(len(tmhr_ranges_select)):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax2 = ax.twinx()
            l1 = ax.plot(flux_wvl, ssrr_ref[alt_ind, :], '-', color=color_series[alt_ind], label='SSRR')
            
            l2 = ax.scatter(rsp_wvl, rsp_ref_mean_all[alt_ind, :], marker='o', color=color_series[alt_ind], edgecolors='k', s=50, alpha=0.7, label='RSP')
            l5 = ax2.plot(flux_wvl, ssfr_fup_ref[alt_ind, :], '--', color=color_series[alt_ind], alpha=0.7, label='SSFR')
            
            ll = l1 + [l2] + l5
            labs = [l.get_label() for l in ll]
            ax.legend(ll, labs, fontsize=10, loc='upper right', ) 
            ax.set_xlabel('Wavelength (nm)', fontsize=14)
            ax.set_ylabel('SSRR radiance x $\pi$ / TOA', fontsize=14)
            ax2.set_ylabel('SSFR Upward flux / TOA', fontsize=14)
            ax.set_ylim(-0.05, 1.2)
            ax2.set_ylim(-0.05, 1.2)
            ax.tick_params(labelsize=12)
            fig.suptitle(f'Upward Radiance and Flux Ref Comparison {date_s}\nZ={alt_avg_all[alt_ind]:.2f}km, lon: {lon_min_all[alt_ind]:.3f}-{lon_max_all[alt_ind]:.3f}', 
                        fontsize=16, y=0.98)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_ssfr_ssrr_ref_comparison_Z%.2fkm.png' % (date_s, date_s, case_tag, alt_avg_all[alt_ind]), bbox_inches='tight', dpi=150)
            plt.close('all')
    
   
    
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for alt_ind in range(len(tmhr_ranges_select)):
            # plot ssrr
            ax.tick_params(labelsize=12)
            # ax.errorbar(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
            #             yerr=ssrr_rad_up_std_all[alt_ind, :], color=color_series[alt_ind], markersize=4, label='SSRR Z=%.2fkm' % (alt_avg_all[alt_ind]))
            ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
                    color=color_series[alt_ind], label='Z=%.2fkm' % (alt_avg_all[alt_ind]))
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        # ax.set_ylim(-0.05, 1.05)
        fig.suptitle(f'SSRR Upward Radiance Comparison {date_s}', fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssrr_rad_up_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
        plt.close('all')
        
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax2 = ax.twinx()
        for alt_ind in range(len(tmhr_ranges_select)):
            # plot ssrr
            ax.tick_params(labelsize=12)
            ax.plot(ssrr_rad_up_wvl, ssrr_rad_up_mean_all[alt_ind, :],
                    color=color_series[alt_ind], label='SSRR Z=%.2fkm' % (alt_avg_all[alt_ind]))

            ax2.plot(flux_wvl, ssfr_fup_mean_all[alt_ind, :],
                    linestyle='--', label='SSRR Ref Z=%.2fkm' % (alt_avg_all[alt_ind]))
            
            for rsp_wvl_ind in range(len(rsp_wvl)):
                rsp_wvl_val = rsp_wvl[rsp_wvl_ind]
                # find the closest wavelength in flux_wvl
                rsp_rad_mean = rsp_rad_mean_all[alt_ind, rsp_wvl_ind]
                rsp_rad_std = rsp_rad_std_all[alt_ind, rsp_wvl_ind]
                ax.errorbar(rsp_wvl_val, rsp_rad_mean, yerr=rsp_rad_std, fmt='s', color=color_series[alt_ind], markersize=6,)# label='SSRR RSP Z=%.2fkm' % (alt_avg_all[alt_ind]))
        
        
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=14)
        ax2.set_ylabel('SSFR Ref Upward Flux (W/m$^2$/nm)', fontsize=14)
        ax.legend(fontsize=10, loc='upper right')
        # plt.grid(True)
        ax.tick_params(labelsize=12)
        # ax.set_ylim(-0.05, 1.05)
        fig.suptitle(f'SSRR Upward Radiance Comparison {date_s}', fontsize=13)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssrr_ssfr_rad_up_comparison_all.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)
        plt.close('all')
    
    
   

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
                            root_mac=_fdir_general_,
                            root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data',)
    
    
    alb_combine()


