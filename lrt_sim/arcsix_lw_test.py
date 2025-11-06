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
import matplotlib.ticker as mtick
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

from util import gaussian, read_ict_radiosonde, read_ict_dropsonde, read_ict_lwc, read_ict_cloud_micro_2DGRAY50, read_ict_cloud_micro_FCDP, read_ict_bbr, read_ict_kt19
from util import read_ict_dlh_h2o, read_ict_co, read_ict_ch4, read_ict_co2, read_ict_o3, ssfr_slit_convolve

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
    # def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R4_V2_test_after_corr.h5"
    # def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R5_V2_test_before_corr.h5"
    def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R5_V2_test_after_corr.h5"
    def ssrr(self, date_s):  return f"{self.data_root}/{self.mission}-SSRR_{self.platform}_{date_s}_RA.h5"
    def hsr1(self, date_s):   return f"{self.data_root}/{self.mission}-HSR1_{self.platform}_{date_s}_R0.h5"
    def logic(self, date_s):  return f"{self.data_root}/{self.mission}-LOGIC_{self.platform}_{date_s}_RA.h5"
    # def sat_coll(self, date_s): return f"{self.data_root}/{self.mission}-SAT-CLD_{self.platform}_{date_s}_v0.h5"
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


def nearest_indices(t_hsk, mask, times):
    # vectorized nearest‐index lookup per leg
    return np.argmin(np.abs(times[:,None] - t_hsk[mask][None,:]), axis=0)

def closest_indices(available: np.ndarray, targets: np.ndarray):
    # vectorized closest-index
    return np.argmin(np.abs(available[:,None] - targets[None,:]), axis=0)

def dropsonde_time_loc_list(dir_dropsonde=f'{_fdir_general_}/dropsonde'):
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
    # glob daughter directories 
    dir_list = sorted([d for d in glob.glob(os.path.join(dir_dropsonde, '*')) if os.path.isdir(d)])
    for d in dir_list:
        file_list.extend(sorted(glob.glob(os.path.join(d, '*.ict'))))
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


def ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_ranges_select, date_s, case_tag):
    t_hsk = np.array(data_hsk["tmhr"])    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_hsr1 = data_hsr1['time']/3600.0  # convert to hours
    
    pitch_ang = data_hsk['ang_pit']
    roll_ang = data_hsk['ang_rol']
    
    tmhr_hsk_mask = (t_hsk >= tmhr_ranges_select[0][0]) & (t_hsk <= tmhr_ranges_select[-1][1])
    
    pitch_roll_mask = np.sqrt(pitch_ang**2 + roll_ang**2) < 2.5  # pitch and roll greater < 2.5 deg
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
    
    print("ssfr_fdn_tmhr shape:", ssfr_fdn_tmhr.shape)
    print("pitch_roll_mask shape:", pitch_roll_mask.shape)
    print("ssfr_fdn_tmhr[pitch_roll_mask, ssfr_zen_550_ind] shape:", ssfr_fdn_tmhr[pitch_roll_mask, :].shape)
    
    ssfr_zen_550_ind = np.argmin(np.abs(ssfr_zen_wvl - 550))
    ssfr_nad_550_ind = np.argmin(np.abs(ssfr_nad_wvl - 550))
    
    hsr_wvl = data_hsr1['wvl_dn_tot']
    hsr_ftot = data_hsr1['f_dn_tot']
    hsr_fdif = data_hsr1['f_dn_dif']
    hsr_dif_ratio = hsr_fdif/hsr_ftot
    hsr_550_ind = np.argmin(np.abs(hsr_wvl - 550))
    
    fig, (ax10, ax20) = plt.subplots(2, 1, figsize=(16, 12))
    ax11 = ax10.twinx()
    
    
    l1 = ax10.plot(t_ssfr, ssfr_fdn[:, ssfr_zen_550_ind], '--', color='k', alpha=0.85)
    l2 = ax10.plot(t_ssfr, ssfr_fup[:, ssfr_nad_550_ind], '--', color='k', alpha=0.85)
    
    ax10.plot(t_ssfr_tmhr, ssfr_fdn_tmhr[:, ssfr_zen_550_ind], 'r-', label='SSFR Down 550nm', linewidth=3)
    ax10.plot(t_ssfr_tmhr, ssfr_fup_tmhr[:, ssfr_nad_550_ind], 'b-', label='SSFR Up 550nm', linewidth=3)
    
    l3 = ax11.plot(t_hsr1, hsr_dif_ratio[:, hsr_550_ind], 'm-', label='HSR1 Diff Ratio 550nm')
    ax11.set_ylabel('HSR1 Diff Ratio 550nm', fontsize=14)
    
    # ax2.plot(t_hsk, data_hsk['ang_hed'], 'r-', label='HSK Heading')
    ax20.plot(t_hsk, pitch_ang, 'g-', label='HSK Pitch')
    ax20.plot(t_hsk, roll_ang, 'b-', label='HSK Roll')
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
    fig.savefig('fig/%s/%s_%s_ssfr_pitch_roll_heading_550nm.png' % (date_s, date_s, case_tag), bbox_inches='tight', dpi=150)

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

def prepare_atmospheric_profile(date_s, case_tag, ileg, date, time_start, time_end,
                                alt_avg, data_dropsonde,
                                cld_leg, levels=None, sfc_T=None,
                                mod_extent=[-60.0, -80.0, 82.4, 84.6],
                                zpt_filedir='./data/atmospheric_profiles'
                                ):
    
    from er3t.util.modis import get_filename_tag
    from modis07_download import modis_download
    
    
    mod_extent = [np.float64(i) for i in mod_extent]
    
    aqua_time_tag = get_filename_tag(date, np.array(mod_extent[:2]), np.array(mod_extent[2:]), satID='aqua')
    terra_time_tag = get_filename_tag(date, np.array(mod_extent[:2]), np.array(mod_extent[2:]), satID='terra')
    
    sat_time_tags, sat_time_tags_int, sat_id_list = [], [], []
    if aqua_time_tag is not None:
        sat_time_tags.extend(aqua_time_tag)
        sat_time_tags_int.extend([int(t.split('.')[1]) for t in aqua_time_tag])
        sat_id_list.extend(['aqua']*len(aqua_time_tag))
    if terra_time_tag is not None:
        sat_time_tags.extend(terra_time_tag)
        sat_time_tags_int.extend([int(t.split('.')[1]) for t in terra_time_tag])
        sat_id_list.extend(['terra']*len(terra_time_tag))
    
    time_s = f'{np.floor(time_start):2.0f}{int((time_start - np.floor(time_start))*60):02d}'
    
    # find the closest overpass time
    time_diffs = [abs(int(t) - int(time_s)) for t in sat_time_tags_int]
    if len(time_diffs) == 0:
        raise ValueError("No MODIS overpass found for the given date and extent")
    min_ind = np.argmin(time_diffs)
    satID = sat_id_list[min_ind]
    sat_time_tag_final = sat_time_tags[min_ind]
    
    # print("Available MODIS overpass times and IDs:", sat_time_tags, sat_id_list)
    # print(f"Selected MODIS satellite: {satID} {sat_time_tag_final} for atmospheric profile generation")
    
    sat0 = modis_download(date=date, 
                          satID=satID,
                              fdir_out='../data/sat-data', 
                              fdir_pre_data='../data/sat-data',
                              extent=mod_extent,
                              extent_analysis=mod_extent,
                              filename_tag=sat_time_tag_final,
                              fname=f'modis_{date_s}_{time_s}.pk', overwrite=False)
    
    modis_07_file = sat0.fnames['mod_07']
    
    # print("Using MODIS 07 file:", modis_07_file)
    
    zpt_filename = f'zpt_{date_s}_{case_tag}_leg_{ileg}.h5'
                
    fname_atm = f'modis_dropsonde_atm_{date_s}_{case_tag}_leg_{ileg}.pk'
    
    status, ws10m = er3t.pre.atm.create_modis_dropsonde_atm(o2mix=0.20935, output_dir=zpt_filedir, output=zpt_filename, 
                                            fname_mod07=modis_07_file, dropsonde_df=data_dropsonde,
                                            levels=levels,
                                            extent=mod_extent, new_h_edge=None, sfc_T_set=sfc_T, sfc_h_to_zero=True,)
    
    fname_insitu = f'{_fdir_general_}/zpt/{date_s}/{date_s}_gases_profiles.csv'
    if not os.path.exists(fname_insitu):
        fname_insitu = None
        
    atm0      = er3t.pre.atm.modis_dropsonde_arcsix_atmmod(zpt_file=f'{zpt_filedir}/{zpt_filename}',
                        fname=fname_atm, 
                        fname_co2_clim=f'{_fdir_general_}/climatology/cams73_latest_co2_conc_surface_inst_2020.nc',
                        fname_ch4_clim=f'{_fdir_general_}/climatology/cams_ch4_202005-202008.nc',
                        fname_o3_clim=f'{_fdir_general_}/climatology/ozone_merra2_202405_202408.h5',
                        fname_insitu=fname_insitu,
                        marli_h=cld_leg['marli_h'], marli_wvmr=cld_leg['marli_wvmr'],
                        date=date, extent=mod_extent,
                        overwrite=True)
    
    if np.any(np.isnan(atm0.lev["pressure"]["data"])):
        raise ValueError("NaN values found in pressure profile, please check the dropsonde data and MODIS 07 data coverage")

    # write out the atmospheric profile in ascii format
    with open(os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km.dat'), 'w') as f:
        header = ('# Combined atmospheric profile\n'
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
    
    with open(f'{zpt_filedir}/ch4_profiles_{date_s}_{case_tag}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km.dat', 'w') as f:  
        header = ('# Combined atmospheric profile for ch4 only\n'
                '#      z(km)      ch4(cm-3)\n'
                )
        lines = [
                f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["ch4"]["data"][i]:12.6e}'
                for i in range(len(atm0.lev['altitude']['data']))[::-1]
                ]
        f.write(header + "\n".join(lines))
        
    return None

def alb_masking(wvl, alb):
    o2a_1_start, o2a_1_end = 748, 776
    h2o_1_start, h2o_1_end = 672, 706
    h2o_2_start, h2o_2_end = 705, 746
    h2o_3_start, h2o_3_end = 884, 996
    h2o_4_start, h2o_4_end = 1084, 1175
    h2o_5_start, h2o_5_end = 1230, 1286
    h2o_6_start, h2o_6_end = 1290, 1509
    h2o_7_start, h2o_7_end = 1748, 2050
    final_start, final_end = 2110, 2200
    
    alb_mask = alb.copy()
    alb_mask[((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
             ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
             ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
             ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
             ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
             ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
             ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
             ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
             ((wvl>=final_start) & (wvl<=final_end))
            ] = np.nan
    
    return alb_mask
    

def ice_alb_fitting(alb_wvl, alb_corr):
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
    
    alb_corr_mask_to_ice_alb_ratio = alb_masking(fit_wvl, alb_corr_mask_to_ice_alb_ratio)
    
                
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
                     wvl=10, # wavelength in micron
                     wvl_std=0.2, # std for spectral response function
                     iter=0,
                     purturb_cot=0.0,
                     band=0,
                    ):

    # case specification
    #/----------------------------------------------------------------------------\#
    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#
    
    
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
    data_ssrr = load_h5(config.ssrr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    
    if date_s != '20240603':
        # MARLI netCDF
        with Dataset(str(config.marli(date_s))) as ds:
            data_marli = {var: ds.variables[var][:] for var in ("time","Alt","H","T","LSR","WVMR")}
    else:
        data_marli = {'time': np.array([]), 'Alt': np.array([]), 'H': np.array([]), 'T': np.array([]), 'LSR': np.array([]), 'WVMR': np.array([])}
    
    log.info("ssfr filename:", config.ssfr(date_s))
    
    # plot ssfr time series for checking sable legs selection
    ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_ranges_select, date_s, case_tag)

    # Build leg masks
    t_hsk = np.array(data_hsk["tmhr"])
    leg_masks = [(t_hsk>=lo)&(t_hsk<=hi) for lo,hi in tmhr_ranges_select]
    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_hsr1 = data_hsr1['time']/3600.0  # convert to hours
    t_ssrr = data_ssrr['tmhr']  # convert to hours
    t_marli = data_marli['time'] # in hours

    
    # atmospheric profile setting
    #/----------------------------------------------------------------------------\#
    dropsonde_file_list, dropsonde_date_list, dropsonde_tmhr_list, dropsonde_lon_list, dropsonde_lat_list = dropsonde_time_loc_list()
    
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
        xx_wvl_grid = np.arange(350, 2100.1, 2.5)
    elif platform.system() == 'Linux':
        xx_wvl_grid = np.arange(350, 2100.1, 1.0)
        
        
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
    flux_solar_interp = solar_interpolation_func(solar_flux_file='arcsix_ssfr_solar_flux.dat', date=date)

    # check rsp l1b folder
    rsp_l1b_dir = f'{_fdir_general_}/rsp/ARCSIX-RSP-L1B_P3B_{date_s}_R01'
    if os.path.exists(rsp_l1b_dir):
        rsp_l1b_files = sorted(glob.glob(f'{rsp_l1b_dir}/ARCSIX-RSP-L1B_P3B_{date_s}*.h5'))
        print("rsp_l1b_files:", rsp_l1b_files)
        if len(rsp_l1b_files) == 0:
            print(f"No RSP L1B files found in {rsp_l1b_dir}")
            rsp_1lb_avail = False
        else:
            rsp_l1b_files = np.array(rsp_l1b_files)
            log.info(f"Found {len(rsp_l1b_files)} RSP L1B files in {rsp_l1b_dir}")
            # ARCSIX-RSP-L1B_P3B_20240605111213_R01.h5
            print([os.path.basename(f).split('_')[2] for f in rsp_l1b_files])
            rsp_l1b_times = np.array([int(os.path.basename(f).split('_')[2][8:10]) + int(os.path.basename(f).split('_')[2][10:12])/60.0 + int(os.path.basename(f).split('_')[2][12:14])/3600.0 for f in rsp_l1b_files])
            rsp_1lb_avail = True
    else:
        print(f"RSP L1B directory {rsp_l1b_dir} does not exist")
        rsp_1lb_avail = False

    # read satellite granule
    #/----------------------------------------------------------------------------\#
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)
    fname_cld_obs_info = '%s/%s_cld_obs_info_%s_%s_%s_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag)
    if not os.path.exists(fname_cld_obs_info):# or iter==0:      
        
        # Loop legs: load raw NC, apply cloud logic, interpolate, plot
        for i, mask in enumerate(leg_masks):
            
            # find index arrays in one go
            times_leg = t_hsk[mask]
            print(f"Leg {i+1}: time range {times_leg.min()}-{times_leg.max()}h")
            
            sel_ssfr, sel_hsr1, sel_ssrr = (
                nearest_indices(t_hsk, mask, arr)
                for arr in (t_ssfr, t_hsr1, t_ssrr)
            )
            
            if len(t_marli) > 0:
                sel_marli = nearest_indices(t_hsk, mask, t_marli)
            
            # choose the rsp l1b file for this leg
            if rsp_1lb_avail:

                time_leg_start = times_leg.min()
                time_leg_end = times_leg.max()
                rsp_file_start = np.where(rsp_l1b_times==rsp_l1b_times[((rsp_l1b_times - time_leg_start)<0)][-1])[0][0]
                rsp_file_end = np.where(rsp_l1b_times==rsp_l1b_times[((rsp_l1b_times - time_leg_end)>0)][0])[0][0]
                rsp_l1b_sel = np.zeros(rsp_l1b_times.shape, dtype=bool)
                rsp_l1b_sel[rsp_file_start:rsp_file_end] = True
                if len(rsp_l1b_sel) == 0:
                    print(f"No RSP L1B files found for leg {i+1} time range {time_leg_start}-{time_leg_end}h")
                    rsp_l1b_files_leg = None
                rsp_l1b_files_leg = rsp_l1b_files[rsp_l1b_sel]
                
                rsp_time_all = []
                rsp_rad_all = []
                rsp_ref_all = []
                rsp_rad_norm_all = []
                rsp_lon_all = []
                rsp_lat_all = []
                rsp_mu0_all = []
                rsp_sd_all = []
                for rsp_file_name in rsp_l1b_files_leg:
                    log.info(f"Reading RSP L1B file: {rsp_file_name}")
                    data_rsp = load_h5(rsp_file_name)
                    t_rsp = data_rsp['Platform/Fraction_of_Day']*24.0  # in hours, (dim_Scans)
                    rsp_vza = data_rsp['Geometry/Viewing_Zenith']  # in degrees, (dim_Scans, dim_Scene_Sectors)
                    rsp_ground_lat = data_rsp['Geometry/Ground_Latitude']  # (dim_Scans, dim_Scene_Sectors)
                    rsp_ground_lon = data_rsp['Geometry/Ground_Longitude']  # (dim_Scans, dim_Scene_Sectors)
                    rsp_sza = data_rsp['Platform/Platform_Solar_Zenith']  # (dim_Scans), in radians
                    rsp_sd = data_rsp['Platform/Solar_Distance']  # (dim_Scans), in AU
                    
                    intensity_1 = data_rsp['Data/Intensity_1']  # (dim_Scans, dim_Scene_Sectors, bands)
                    intensity_2 = data_rsp['Data/Intensity_2']  # (dim_Scans, dim_Scene_Sectors, bands)
                    rsp_wvl = data_rsp['Data/Wavelength']  # in nm, (bands,)
                    rsp_solar_const = data_rsp['Calibration/Solar_Constant']  # in W/m^2/nm, (bands,)
                    
                    
                    nadir_select = np.argmax(rsp_vza, axis=1)  # select the nadir sector
                    sel_rsp = nearest_indices(t_hsk, mask, t_rsp)
                    
                    rsp_time_sel = t_rsp[sel_rsp]
                    rsp_int_1_sel = intensity_1[sel_rsp, nadir_select[sel_rsp], :]  # (time, bands)
                    rsp_int_2_sel = intensity_2[sel_rsp, nadir_select[sel_rsp], :]  # (time, bands)
                    rsp_lon_sel = rsp_ground_lon[sel_rsp, nadir_select[sel_rsp]]
                    rsp_lat_sel = rsp_ground_lat[sel_rsp, nadir_select[sel_rsp]]
                    rsp_sza_sel = rsp_sza[sel_rsp]
                    rsp_sd_sel = rsp_sd[sel_rsp]
                    
                    rsp_rad_norm = (rsp_int_1_sel + rsp_int_2_sel) / 2  # (time, bands), in counts
                    rsp_rad = rsp_rad_norm * rsp_solar_const[np.newaxis, :] / np.pi
                    rsp_ref_cal = rsp_rad_norm * rsp_sd_sel[:, np.newaxis]**2 / np.cos(rsp_sza_sel[:, np.newaxis])  # (time, bands), in W/m^2/sr/nm
                    
                    rsp_time_all.extend(rsp_time_sel)
                    rsp_rad_all.extend(rsp_rad)
                    rsp_rad_norm_all.extend(rsp_rad_norm)
                    rsp_ref_all.extend(rsp_ref_cal)
                    rsp_lon_all.extend(rsp_lon_sel)
                    rsp_lat_all.extend(rsp_lat_sel)
                    rsp_mu0_all.extend(np.cos(np.deg2rad(rsp_sza_sel)))
                    rsp_sd_all.extend(rsp_sd_sel)
                    


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
                "p3_alt":  data_hsk["alt"][mask] / 1000.0,
                
                "rsp_lon": np.array(rsp_lon_all) if rsp_1lb_avail else None,
                "rsp_lat": np.array(rsp_lat_all) if rsp_1lb_avail else None,
                "rsp_rad": np.array(rsp_rad_all) if rsp_1lb_avail else None,
                "rsp_rad_norm": np.array(rsp_rad_norm_all) if rsp_1lb_avail else None,
                "rsp_ref": np.array(rsp_ref_all) if rsp_1lb_avail else None,
                "rsp_wvl": rsp_wvl if rsp_1lb_avail else None,
                "rsp_mu0": np.array(rsp_mu0_all) if rsp_1lb_avail else None,
                "rsp_sd":  np.array(rsp_sd_all) if rsp_1lb_avail else None,   
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
        
            pitch_roll_mask = np.sqrt(data_hsk["ang_pit"][mask]**2 + data_hsk["ang_rol"][mask]**2) < 2.5
            ssfr_zen_flux[~pitch_roll_mask, :] = np.nan
            ssfr_nad_flux_interp[~pitch_roll_mask, :] = np.nan
            ssfr_zen_toa[~pitch_roll_mask, :] = np.nan
            
            
            leg['ssfr_zen'] = ssfr_zen_flux
            leg['ssfr_nad'] = ssfr_nad_flux_interp
            leg['ssfr_zen_wvl'] = ssfr_zen_wvl
            leg['ssfr_nad_wvl'] = ssfr_zen_wvl
            leg['ssfr_toa'] = ssfr_zen_toa
            
            
            # ssrr
            # interpolate ssrr zenith radiance to nadir wavelength grid
            f_zen_rad_interp = interp1d(data_ssrr["zen/wvl"], data_ssrr["zen/rad"][sel_ssrr, :], axis=1, bounds_error=False, fill_value=np.nan)
            ssrr_rad_zen_i = f_zen_rad_interp(ssfr_zen_wvl)
            f_nad_rad_interp = interp1d(data_ssrr["nad/wvl"], data_ssrr["nad/rad"][sel_ssrr, :], axis=1, bounds_error=False, fill_value=np.nan)
            ssrr_rad_nad_i = f_nad_rad_interp(ssfr_zen_wvl)
            
            leg['ssrr_zen_rad'] = ssrr_rad_zen_i
            leg['ssrr_nad_rad'] = ssrr_rad_nad_i
            leg['ssrr_zen_wvl'] = ssfr_zen_wvl
            leg['ssrr_nad_wvl'] = ssfr_zen_wvl

            vars()["cld_leg_%d" % i] = leg
        
            # save the cloud observation information to a pickle file
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
            with open(fname_pkl, 'wb') as f:
                pickle.dump(vars()["cld_leg_%d" % i], f, protocol=pickle.HIGHEST_PROTOCOL)

            del leg  # free memory
            del sel_ssfr, sel_ssrr, sel_hsr1
            if rsp_1lb_avail:
                del rsp_time_all, rsp_rad_all, rsp_ref_all, rsp_lon_all, rsp_lat_all
                del rsp_mu0_all, rsp_sd_all
                del data_rsp
                del t_rsp, intensity_1, intensity_2, rsp_solar_const
                del rsp_ground_lon, rsp_ground_lat, rsp_sza, rsp_vza
            gc.collect()
        
    else:
        print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
        for i in range(len(tmhr_ranges_select)):
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
            with open(fname_pkl, 'rb') as f:
                vars()[f"cld_leg_{i}"] = pickle.load(f)   
    
    
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
    
        
        if not os.path.exists(fname_h5) or overwrite_lrt: 
            if not os.path.exists(os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km.dat')):
                prepare_atmospheric_profile(date_s, case_tag, ileg, date, time_start, time_end,
                                            alt_avg, data_dropsonde,
                                            cld_leg, levels=levels, sfc_T=294,
                                            mod_extent=[np.round(np.nanmin(cld_leg['lon']), 2), 
                                                        np.round(np.nanmax(cld_leg['lon']), 2),
                                                        np.round(np.nanmin(cld_leg['lat']), 2),
                                                        np.round(np.nanmax(cld_leg['lat']), 2)],
                                            zpt_filedir=f'../data/zpt/{date_s}'
                                            )
            # =================================================================================
            
            
            # write out the surface albedo
            #/----------------------------------------------------------------------------\#
            os.makedirs(f'{_fdir_general_}/sfc_alb', exist_ok=True)
            iter_0_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_0.dat'
            if 1:#not os.path.exists(iter_0_fname) or overwrite_alb:
                
                alb_wvl = np.concatenate(([348.0], cld_leg['ssfr_zen_wvl']))
                alb_avg = np.nanmean(cld_leg['ssfr_nad']/cld_leg['ssfr_zen'], axis=0)
                
                if np.all(np.isnan(alb_avg)):
                    raise ValueError(f"All nadir/zenith ratios are NaN for leg {ileg+1}, cannot compute average albedo")
                alb_avg[alb_avg<0.0] = 0.0
                alb_avg[alb_avg>1.0] = 1.0
                alb_avg[np.isnan(alb_avg)] = 0.0
                
                alb_avg_extend = np.concatenate(([alb_avg[0]], alb_avg))

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
            inits_rad = []
            flux_key_ix = []
            output_list = []
                
            for ix in range(len(manual_cloud_cot)):
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
                    rad_result_dict = {}
                    
                    rad_results = []
                
                flux_key = np.zeros_like(flux_output, dtype=object)
                cloudy = 0
                clear = 0
                
                # rt initialization
                #/----------------------------------------------------------------------------\#
                lrt_cfg = copy.deepcopy(er3t.rtm.lrt.get_lrt_cfg())
                
                lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km.dat')
                # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')
                # lrt_cfg['solar_file'] = 'CU_composite_solar.dat'
                # lrt_cfg['solar_file'] = lrt_cfg['solar_file'].replace('kurudz_0.1nm.dat', 'kurudz_1.0nm.dat')

                # lw
                # lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')
                lrt_cfg['number_of_streams'] = 32
                lrt_cfg['mol_abs_param'] = 'reptran fine'
                lrt_cfg['output_process'] = 'integrate'
                # ch4_file = os.path.join(zpt_filedir, f'ch4_profiles_{date_s}_{case_tag}.dat')
                input_dict_extra_general = {
                                    'source': 'thermal',
                                    # 'mol_file': 'CH4 %s' % os.path.join(zpt_filedir, f'ch4_profiles_{date_s}_{case_tag}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km.dat'),
                                    # 'wavelength_grid_file': 'wvl_grid_thermal.dat',
                                    'wavelength_add' : f'{((wvl-wvl_std)*1000):.0f} {((wvl+wvl_std)*1000):.0f}', # 9820-10180 nm
                                    'output_quantity': 'brightness',
                                    
                                    }
                Nx_effective = 1 # integrate over all wavelengths
                mute_list = ['wavelength', 'spline', 'source solar', 'slit_function_file', 'wc_properties', 'ic_modify tau set']
                #/----------------------------------------------------------------------------/#

                
                

                # cot_x = cld_leg['cot'][ix]
                # cwp_x = cld_leg['cwp'][ix]
                
                cot_x = manual_cloud_cot[ix]
                cwp_x = manual_cloud_cwp[ix]
                
                if not clear_sky:
                    input_dict_extra = copy.deepcopy(input_dict_extra_general)
                    if ((cot_x >= 0.01 and np.isfinite(cwp_x))):
                        cloudy += 1

                        cer_x = manual_cloud_cer
                        cth_x = manual_cloud_cth
                        cbh_x = manual_cloud_cbh
                        cgt_x = cth_x - cbh_x
    
                        cth_ind_cld = bisect.bisect_left(z_list, cth_x)-1
                        cbh_ind_cld = bisect.bisect_left(z_list, cbh_x)-1
                        
                        fname_cld = f'{fdir_tmp}/cld_{ix:04d}.txt'
                        if os.path.exists(fname_cld):
                            os.remove(fname_cld)
                        cld_cfg = er3t.rtm.lrt.get_cld_cfg()
                        cld_cfg['cloud_file'] = fname_cld
                        cld_cfg['cloud_altitude'] = z_list[cbh_ind_cld:cth_ind_cld+1]
                        cld_cfg['cloud_effective_radius']  = cer_x
                        # cld_cfg['liquid_water_content'] = cwp_x*1000/(cgt_x*1000) # convert kg/m^2 to g/m^3
                        cld_cfg['liquid_water_content'] = cwp_x*1000/(cgt_x*1000) # convert kg/m^2 to g/m^3
                        cld_cfg['ic_properties'] = 'yang2013'
                        # cld_cfg['ic_properties'] = 'baum_v36'
                        cld_cfg['cloud_type'] = 'ice'  # 'water' or 'ice'
                        cld_cfg['cloud_optical_thickness'] = cot_x
                        
                        dict_key_arr = np.concatenate(([cot_x], [cld_cfg['cloud_effective_radius']], cld_cfg['cloud_altitude'], [alt_avg]))
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
                    if not clear_sky:
                        input_dict_extra_alb['ic_habit_yang2013']  = 'column_8elements severe'
                        # input_dict_extra_alb['ic_habit']  = 'rough-aggregate'
                        # input_dict_extra_alb['interpret_as_level'] = 'ic'
                    init = er3t.rtm.lrt.lrt_init_mono_rad(
                            input_file  = '%s/input_%04d.txt'  % (fdir_tmp, ix),
                            output_file = '%s/output_%04d.txt' % (fdir_tmp, ix),
                            date        = date,
                            surface_albedo = 0.02,  # emissivity = 0.98
                            solar_zenith_angle = sza_avg,
                            solar_azimuth_angle = saa_avg,
                            sensor_zenith_angle = 5.0,
                            sensor_azimuth_angle = 0.0,
                            # Nx = Nx_effective,
                            output_altitude    = 'toa',
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
            print("inits_rad length:", len(inits_rad))
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
                        data = er3t.rtm.lrt.lrt_read_uvspec_rad([inits_rad[i]])
                        
                        rad_results.append(np.squeeze(data.rad))
            ##### run several libratran calculations one by one
            
            elif platform.system() == 'Linux':
                if len(inits_rad) > 0:
                    print('Running libratran calculations ...')
                    for i in range(len(inits_rad)):
                        er3t.rtm.lrt.lrt_run(inits_rad[i])
                        data = er3t.rtm.lrt.lrt_read_uvspec_rad([inits_rad[i]])
                        rad_result_dict[flux_key_all[i]] = np.squeeze(data.rad)
                        
                        rad_results.append(np.squeeze(data.rad))
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

            rad_results = np.array(rad_results)*1000 # original code use rad / 1000
            
            # rad_results = np.array(rad_results) # original code use rad / 1000
            
            # for flux_dn in [flux_down_results, flux_down_dir_results, flux_down_diff_results, flux_up_results]:
            #     for iz in range(3):
            #         for iset in range(flux_down_results.shape[0]):
            #             flux_dn[iset, :, iz] = ssfr_slit_convolve(effective_wvl, flux_dn[iset, :, iz], wvl_joint=950)
            
            print(rad_results)
            print(rad_results.shape)
            
            # save output to pkl
            if purturb_cot > 0:
                pkl_name = 'bT_pert_results_cer_%02dum_wvl_%.2fum.pkl' % (manual_cloud_cer, wvl)
            else:
                pkl_name = 'bT_results_cer_%02dum_wvl_%.2fum.pkl' % (manual_cloud_cer, wvl)

            with open(pkl_name, 'wb') as f:
                pickle.dump({
                    'band': band,
                    'wvl': wvl,
                    'wvl_std': wvl_std,
                    'cloud_effective_radius': manual_cloud_cer,
                    'cloud_top_height': manual_cloud_cth,
                    'cloud_base_height': manual_cloud_cbh,
                    'cloud_optical_thickness': manual_cloud_cot,
                    'brightness_temperature_TOA': rad_results,
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # simulated brightness T at TOA         
            TOA_bT = rad_results
            cot_arr = manual_cloud_cot
            
            # plt.close('all')
            # fig, ax = plt.subplots(figsize=(8, 5))
            # ax.plot(cot_arr, TOA_bT, marker='o', label='wvl=%.1f$\mum' % (wvl))
            # ax.set_xlabel('Cloud Optical Thickness')
            # ax.set_ylabel('Simulated Brightness Temperature at TOA (K)')
            # ax.set_xscale('log')
            # ax.xticksformatter = plt.FormatStrFormatter('%.1f')
            # ax.set_title(f'CER={manual_cloud_cer:.1f} $\mu$m, CTH={manual_cloud_cth:.2f} km, CBH={manual_cloud_cbh:.2f} km')
            # fig.tight_layout()
            # fig.savefig(f'{dir_fig}/{date_s}/flt_trk_atm_corr_TOA_bT_{date_s}_leg{ileg+1}_{case_tag}_wvl{wvl:.1f}um_iter{iter}.png', dpi=300)
            # plt.show()
        
        gc.collect()

    print("Finished libratran calculations.")  
    #\----------------------------------------------------------------------------/#

    return



def plot_bt():
    pk_files = glob.glob('bT_results_cer_20um_wvl_*.pkl')
    wv_list = []
    bT_list = []
    for pk_file in pk_files:
        with open(pk_file, 'rb') as f:
            data = pickle.load(f)
        if (data['brightness_temperature_TOA']==0).all():
            continue
        wv_list.append(data['wvl'])
        bT_list.append(data['brightness_temperature_TOA'])
        cot_list = data['cloud_optical_thickness']
        
    wv_arr = np.array(wv_list)
    wn_arr = 10000.0 / wv_arr
    
    
    bT_arr = np.array(bT_list)  # (n_wvl, n_cot)
    cot_arr = np.array(cot_list)
    
    dbT = bT_arr[:, 1:] - bT_arr[:, :-1]
    dcot = cot_arr[1:] - cot_arr[:-1]
    dbTdcot = dbT / dcot[np.newaxis, :]  # (n_wvl, n_cot-1)
    cot_avg = 0.5 * (cot_arr[1:] + cot_arr[:-1])
    
    # sort by wavelength
    sort_ix = np.argsort(wv_arr)
    wv_arr = wv_arr[sort_ix]
    wn_arr = wn_arr[sort_ix]
    bT_arr = bT_arr[sort_ix, :]
    dbTdcot = dbTdcot[sort_ix, :]
    # plot
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for iwv in range(len(wv_arr)):
        ax1.plot(cot_arr, bT_arr[iwv, :], label=r'wvl=%.1f $\mu m$ (%.0f $cm^{-1}$)' % (wv_arr[iwv], wn_arr[iwv]))
        ax2.plot(cot_avg, dbTdcot[iwv, :], label=r'wvl=%.1f $\mu m$ (%.0f $cm^{-1}$)' % (wv_arr[iwv], wn_arr[iwv]))
    ax1.set_xlabel('Cloud Optical Thickness')
    ax1.set_ylabel('Simulated TOA BT (K)')
    
    
    
    ax2.set_xlabel('Cloud Optical Thickness')
    ax2.set_ylabel(r'$\Delta$BT / $\Delta$COT (K)')
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    
    
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_xlim(0.08, 11)
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.1f}'))
    fig.suptitle(f'CER=10.0 $\mu$m, CTH=5.50 km, CBH=5.45 km')
    fig.tight_layout()
    fig.savefig(f'./fig/simulated_TOA_brightness_temperature_vs_COT_CER20um_CTH5.5km_CBH5.45km.png', dpi=300)
    plt.show()
    
def plot_bt2():
    pk_files = glob.glob('bT_results_wvl_*.pkl')
    pert_pk_files = glob.glob('bT_pert_results_wvl_*.pkl')
    wv_list = []
    bT_list = []
    bT_pert_list = []
    for pk_file in pk_files:
        with open(pk_file, 'rb') as f:
            data = pickle.load(f)
        if (data['brightness_temperature_TOA']==0).all():
            continue
        wv_list.append(data['wvl'])
        bT_list.append(data['brightness_temperature_TOA'])
        cot_list = data['cloud_optical_thickness']
        
    for pk_file in pert_pk_files:
        with open(pk_file, 'rb') as f:
            data = pickle.load(f)
        if (data['brightness_temperature_TOA']==0).all():
            continue
        bT_pert_list.append(data['brightness_temperature_TOA'])
        cot_pert_list = data['cloud_optical_thickness']
        
    wv_arr = np.array(wv_list)
    wn_arr = 10000.0 / wv_arr
    
    
    bT_arr = np.array(bT_list)  # (n_wvl, n_cot)
    cot_arr = np.array(cot_list)
    
    bT_pert_arr = np.array(bT_pert_list)  # (n_wvl, n_cot)
    cot_pert_arr = np.array(cot_pert_list)
    
    dbT = bT_arr[:, 1:] - bT_arr[:, :-1]
    dcot = cot_arr[1:] - cot_arr[:-1]
    dbTdcot = dbT / dcot[np.newaxis, :]  # (n_wvl, n_cot-1)
    cot_avg = 0.5 * (cot_arr[1:] + cot_arr[:-1])
    
    
    
    # sort by wavelength
    sort_ix = np.argsort(wv_arr)
    wv_arr = wv_arr[sort_ix]
    wn_arr = wn_arr[sort_ix]
    bT_arr = bT_arr[sort_ix, :]
    dbTdcot = dbTdcot[sort_ix, :]
    
    bT_pert_arr = bT_pert_arr[sort_ix, :]
    
    
    jacobian_list = numerical_jacobian(bT_arr, cot_arr, bT_pert_arr, cot_pert_arr)
    jacobian_arr = np.array(jacobian_list)
    
    # plot
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for iwv in range(len(wv_arr)):
        ax1.plot(cot_arr, bT_arr[iwv, :], label=r'wvl=%.1f $\mu m$ (%.0f $cm^{-1}$)' % (wv_arr[iwv], wn_arr[iwv]))
        # ax2.plot(cot_avg, dbTdcot[iwv, :], label=r'wvl=%.1f $\mu m$ (%.0f $cm^{-1}$)' % (wv_arr[iwv], wn_arr[iwv]))
        ax2.plot(cot_arr, jacobian_arr[iwv, :], label=r'wvl=%.1f $\mu m$ (%.0f $cm^{-1}$)' % (wv_arr[iwv], wn_arr[iwv]))
    ax1.set_xlabel('Cloud Optical Thickness')
    ax1.set_ylabel('Simulated TOA BT (K)')
    
    ax1.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.1f}'))
    
    ax2.set_xlabel('Cloud Optical Thickness')
    ax2.set_ylabel(r'$\Delta$BT / $\Delta$COT (K)')
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    
    
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_xlim(0.1, 11)
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.1f}'))
    fig.suptitle(f'CER=10.0 $\mu$m, CTH=5.50 km, CBH=5.45 km')
    fig.tight_layout()
    fig.savefig(f'./fig/simulated_TOA_brightness_temperature_vs_COT_CER20um_CTH5.5km_CBH5.45km.png', dpi=300)
    plt.show()
    
def plot_bt3():
    pk_files = glob.glob('bT_results_cer_*_wvl_*.pkl')
    # pert_pk_files = glob.glob('bT_pert_results_wvl_*.pkl')
    band_list = []
    wv_list = []
    bT_list = []
    bT_pert_list = []
    cer_list = []
    for pk_file in pk_files:
        with open(pk_file, 'rb') as f:
            data = pickle.load(f)
        if (data['brightness_temperature_TOA']==0).all():
            continue
        band_list.append(data['band'])
        wv_list.append(data['wvl'])
        bT_list.append(data['brightness_temperature_TOA'].copy())
        cot_list = data['cloud_optical_thickness']
        cer_list.append(data['cloud_effective_radius'])
        
    
    wv_arr = np.zeros((len(set(wv_list)), len(set(cer_list)), len(cot_list)))
    bT_arr = np.zeros((len(set(wv_list)), len(set(cer_list)), len(cot_list)))
    cer_arr = np.zeros((len(set(wv_list)), len(set(cer_list)), len(cot_list)))
    cot_arr = np.zeros((len(set(wv_list)), len(set(cer_list)), len(cot_list)))
    
    wv_list = np.array(wv_list)
    cer_list = np.array(cer_list)
    for iwv, wv in enumerate(sorted(set(wv_list))):
        for icer, cer in enumerate(sorted(set(cer_list))):
            list_ind = np.where(np.logical_and(wv_list==wv,cer_list==cer))[0][0]
            bT_arr[iwv, icer, :] = bT_list[list_ind].copy()
            # print("iw, icer,  bT:", iwv, icer, bT_list[list_ind].copy())
            cer_arr[:, icer, :] = cer
            cot_arr[:, icer, :] = cot_list

    
    
    
    
    print("bT_arr shape:", bT_arr.shape)
    print("cot_arr shape:", cot_arr.shape)
    
    cer_sort_set = np.array(sorted(set(cer_list)))
    cot_sort_set = np.array(sorted(set(cot_list)))
    wvl_sort_set = np.array(sorted(set(wv_list)))
    band_sort_set = np.array(sorted(set(band_list)))
    
    wn_sort_set = 10000.0 / wvl_sort_set
    
    
    plt.close('all')
    icer = np.where(cer_sort_set==20.0)[0][0]
    plt.contourf(cot_sort_set, wvl_sort_set, bT_arr[:, icer, :], levels=20, cmap='RdBu_r')
    plt.colorbar(label='BT (K)')
    plt.xlabel('Cloud Optical Thickness')
    plt.ylabel('Wavelength ($\mu$m)')
    plt.xscale('log')
    plt.show()
    
    # dbT = bT_arr[:, 1:] - bT_arr[:, :-1]
    # dcot = cot_arr[1:] - cot_arr[:-1]
    # dbTdcot = dbT / dcot[np.newaxis, :]  # (n_wvl, n_cot-1)
    # cot_avg = 0.5 * (cot_arr[1:] + cot_arr[:-1])
    dcer_wvl = np.zeros((len(set(wv_list)), len(set(cer_list)), len(set(cot_list))))
    dcot_wvl = np.zeros((len(set(wv_list)), len(set(cer_list)), len(set(cot_list))))
    
    for iwv in range(len(set(wv_list))):
        dcer, dcot  = np.gradient(bT_arr[iwv, ...], cer_sort_set, cot_sort_set, edge_order=2)

        dcer_wvl[iwv, :] = dcer
        dcot_wvl[iwv, :] = dcot
    

    

    jacobian_arr = dcot_wvl
    
    plt.close('all')
    icer = np.where(cer_sort_set==20.0)[0][0]
    plt.contourf(cot_sort_set, wvl_sort_set, jacobian_arr[:, icer, :], levels=20, cmap='RdBu_r')
    plt.colorbar(label=r'$\Delta$BT / $\Delta$COT (K)')
    plt.xlabel('Cloud Optical Thickness')
    plt.ylabel('Wavelength ($\mu$m)')
    plt.xscale('log')
    plt.show()
    
    
    # plot
    icer = np.where(cer_sort_set==20.0)[0][0]
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for iwv in range(len(wvl_sort_set)):
        ax1.plot(cot_sort_set, bT_arr[iwv, icer, :], label=r'band%d: %.1f $\mu m$ (%.0f $cm^{-1}$)' % (band_sort_set[iwv], wvl_sort_set[iwv], wn_sort_set[iwv]))
        # ax2.plot(cot_avg, dbTdcot[iwv, :], label=r'wvl=%.1f $\mu m$ (%.0f $cm^{-1}$)' % (wv_arr[iwv], wn_arr[iwv]))
        ax2.plot(cot_sort_set, jacobian_arr[iwv, icer, :], label=r'band%d: %.1f $\mu m$ (%.0f $cm^{-1}$)' % (band_sort_set[iwv], wvl_sort_set[iwv], wn_sort_set[iwv]))
    ax1.set_xlabel('Cloud Optical Thickness')
    ax1.set_ylabel('Simulated TOA BT (K)')
    
    ax1.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.1f}'))
    
    ax2.set_xlabel('Cloud Optical Thickness')
    ax2.set_ylabel(r'dBT/d$\tau$ (K)')
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    
    
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_xlim(0.08, 11)
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.1f}'))
    fig.suptitle(f'CER=20.0 $\mu$m, CTH=10.0 km, CBH=9.8 km')
    fig.tight_layout()
    fig.savefig(f'./fig/simulated_TOA_brightness_temperature_vs_COT_CER20um_CTH10.0km_CBH9.8km.png', dpi=300)
    plt.show()
    
    
    # plot
    icer = np.where(cer_sort_set==20.0)[0][0]
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 4))
    for iwv in range(len(wvl_sort_set)):
        ax1.plot(cot_sort_set, bT_arr[iwv, icer, :], label=r'band%d: %.1f $\mu m$ (%.0f $cm^{-1}$)' % (band_sort_set[iwv], wvl_sort_set[iwv], wn_sort_set[iwv]))
        # ax2.plot(cot_avg, dbTdcot[iwv, :], label=r'wvl=%.1f $\mu m$ (%.0f $cm^{-1}$)' % (wv_arr[iwv], wn_arr[iwv]))
        ax2.plot(cot_sort_set, jacobian_arr[iwv, icer, :], label=r'band%d: %.1f $\mu m$ (%.0f $cm^{-1}$)' % (band_sort_set[iwv], wvl_sort_set[iwv], wn_sort_set[iwv]))
        ax3.plot(cot_sort_set, dcer_wvl[iwv, icer, :], label=r'band%d: %.1f $\mu m$ (%.0f $cm^{-1}$)' % (band_sort_set[iwv], wvl_sort_set[iwv], wn_sort_set[iwv]))
    ax1.set_ylabel('Simulated TOA BT (K)')

    ax2.set_ylabel(r'dBT/d$\tau$ (K)')
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)

    ax3.set_ylabel(r'dBT/d$r_{eff}$ (K/$\mu m$)')
    ax3.axhline(0, color='k', linestyle='--', linewidth=1)
    
    
    
    ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Cloud Optical Thickness')
        ax.set_xscale('log')
        ax.set_xlim(0.08, 11)
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.1f}'))
    fig.suptitle(f'CER=20.0 $\mu$m, CTH=10.0 km, CBH=9.8 km')
    fig.tight_layout()
    fig.savefig(f'./fig/simulated_TOA_brightness_temperature_vs_COT_CER20um_CTH10.0km_CBH9.8km_2.png', dpi=300)
    plt.show()

def numerical_jacobian(f, x, fp, xp):
    """Computes the Jacobian of a function using finite differences."""
    x = np.asarray(x, dtype=float)
    n_outputs = len(f)
    n_inputs = len(x)
    jacobian_matrix = np.zeros((n_outputs, n_inputs))
    print("n_outputs:", n_outputs)
    print("n_inputs:", n_inputs)
    print("f shape:", f.shape)

    for i in range(n_inputs):
        # # Perturb the input vector at index i
        # x_plus_epsilon = x.copy()
        # x_plus_epsilon[i] += epsilon
        
        # Calculate the partial derivative for each output
        jacobian_matrix[:, i] = (fp[:, i] - f[:, i]) / (xp[i] - x[i])

    return jacobian_matrix


if __name__ == '__main__':

    
    dir_fig = './fig'
    os.makedirs(dir_fig, exist_ok=True)
    
    config = FlightConfig(mission='ARCSIX',
                            platform='P3B',
                            data_root=_fdir_data_,
                            root_mac=_fdir_general_,
                            root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data',)
    

    # # plot_bt()
    # # # # plot_bt2()
    plot_bt3()
    sys.exit()
    
    tirs1_mean = [
        None, None, None, 3.92, 4.26, 4.76, 5.51, None, None, 8.40, 8.90, 9.76, 10.59,
        11.41, 12.23, 13.04, None, None, 15.96, 16.51, 17.33, 18.18, 19.02, 19.86,
        20.71, 21.53, 22.39, 23.22, 24.05, 24.89, 25.71, 26.54, 27.40, 28.23,
        None, None, 31.16, 31.84, 32.55, 33.36, 34.23, 35.09, 35.91, 36.68, 37.51,
        38.39, 39.25, 40.12, 40.93, 41.81, 42.55, 43.47, 44.19, 45.19, 45.88,
        46.94, 47.61, 48.58, 49.43, 50.09, 51.12, 51.90, 52.66
    ]

    tirs2_mean = [
        None, None, None, 4.47, 5.10, 5.89, 6.31, None, None, 9.29, 10.16, 10.98, 11.80,
        12.62, 13.38, 13.87, None, None, 16.89, 17.74, 18.59, 19.43, 20.28, 21.18,
        21.96, 22.79, 23.62, 24.46, 25.29, 26.11, 26.96, 27.81, 28.62, 29.45,
        None, None, 32.17, 32.92, 33.78, 34.67, 35.49, 36.28, 37.09, 37.93, 38.80,
        39.67, 40.53, 41.35, 42.17, 42.99, 43.83, 44.65, 45.56, 46.34, 47.34,
        48.08, 49.10, 49.70, 50.59, 51.51, 52.26, 53.09, 54.11
    ]
    
    # wvl_list =     [5.51, 9.76, 15.96, 20.71, 25.71, 31.16, 35.91, 40.93, 45.88, 49.43, 52.66]  # um
    # wvl_std_list = [0.05, 0.18,  0.17,  0.21,  0.15,  0.16,  0.20,  0.28,  0.35,  0.42,  0.46]  # um
    band_list = [10, 13, 16, 21, 26, 31]
    wvl_list = [8.40, 10.59, 13.04, 17.33, 21.54, 25.71]  # um
    wvl_std_list = [0.16, 0.18, 0.16, 0.22, 0.20, 0.15]  # um
    
    # wvl_list_wavenumber = [10000/float(wvl) for wvl in wvl_list]
    # for wlvl, wn in zip(wvl_list, wvl_list_wavenumber):
    #     print(f"Wavelength: {wlvl} um, Wavenumber: {wn} cm^-1")
    # sys.exit()
    ice_denity = 917  # kg/m^3
    
    for wvl, wvl_std, band in zip(wvl_list, wvl_std_list, band_list):
        cer = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
        cer = [10, 15, 18, 20, 22, 25, 30,]
        # cot = np.array([0.01, 0.025, 0.05, 0.075, 0.08, 0.09, 0.095, 0.1, 0.105, 0.11, 0.12, 0.13, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.9, 0.95, 1.0, 1.1, 1.22, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11])
        cot = np.array([0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0, 12.0, 15.0, ])#20.0, 25.0, 30.0, 50.0])
        # cot = np.array([0.01, 0.03, 0.05, 0.10, 0.125, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0, 12.0, 15.0, ])#20.0, 25.0, 30.0, 50.0])

        for cer_val in cer:
            lwp = 2/3 * ice_denity * cot * cer_val * 1e-6 # in g/m^2
            for iter in range(1):
                flt_trk_atm_corr(date=datetime.datetime(2024, 5, 30),
                                tmhr_ranges_select=[
                                                    [11.37, 11.43],
                                                    ],
                                case_tag='clear_sky_track_atm_corr',
                                config=config,
                                simulation_interval=10,
                                levels=np.concatenate((np.arange(0., 1., 0.5), 
                                            np.arange(1., 4.1, 1.0),
                                            np.arange(5.0, 9.6, 1.0),
                                            np.arange(9.4, 10.2, 0.1),
                                            np.array([12, 15, 20, 30., 40., 45.]))),
                                clear_sky=False,
                                overwrite_lrt=True,
                                manual_cloud=True,
                                manual_cloud_cer=cer_val,
                                manual_cloud_cwp=lwp,
                                manual_cloud_cth=10.0,
                                manual_cloud_cbh=9.8,
                                manual_cloud_cot=cot,
                                wvl=wvl, # um
                                wvl_std=wvl_std, # um
                                iter=iter,
                                band=band,
                                )
                
    sys.exit()

    for wvl, wvl_std in zip(wvl_list, wvl_std_list):
        cer = 20.0
        # cot = np.array([0.01, 0.025, 0.05, 0.075, 0.08, 0.09, 0.095, 0.1, 0.105, 0.11, 0.12, 0.13, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.9, 0.95, 1.0, 1.1, 1.22, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11])
        cot = np.array([0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0, 12.0, 15.0, ])#20.0, 25.0, 30.0, 50.0])
        lwp = 2/3 * ice_denity * cot * cer * 1e-6 # in kg/m^2
        
        for iter in range(1):
            flt_trk_atm_corr(date=datetime.datetime(2024, 5, 30),
                            tmhr_ranges_select=[
                                                [11.37, 11.43],
                                                ],
                            case_tag='clear_sky_track_atm_corr',
                            config=config,
                            simulation_interval=10,
                            levels=np.concatenate((np.arange(0., 1., 0.5), 
                                            np.arange(1., 4.1, 1.0),
                                            np.arange(5.0, 9.6, 1.0),
                                            np.arange(9.4, 10.2, 0.1),
                                            np.array([12, 15, 20, 30., 40., 45.]))),
                            clear_sky=False,
                            overwrite_lrt=True,
                            manual_cloud=True,
                            manual_cloud_cer=cer,
                            manual_cloud_cwp=lwp,
                            manual_cloud_cth=10.0,
                            manual_cloud_cbh=9.8,
                            manual_cloud_cot=cot,
                            wvl=wvl, # um
                            wvl_std=wvl_std, # um
                            iter=iter,
                            )
            
        # cot = np.array([0.01, 0.025, 0.05, 0.075, 0.08, 0.09, 0.095, 0.1, 0.105, 0.11, 0.12, 0.13, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.9, 0.95, 1.0, 1.1, 1.22, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11])
        # purturb_cot = 0.01
        # cot += purturb_cot
        # # cot = np.array([0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0, 12.0, 15.0, ])#20.0, 25.0, 30.0, 50.0])
        # lwp = 2/3 * ice_denity * cot * cer # in g/m^2
        
        # for iter in range(1):
        #     flt_trk_atm_corr(date=datetime.datetime(2024, 5, 30),
        #                     tmhr_ranges_select=[
        #                                         [11.37, 11.43],
        #                                         ],
        #                     case_tag='clear_sky_track_atm_corr',
        #                     config=config,
        #                     simulation_interval=10,
        #                     clear_sky=False,
        #                     overwrite_lrt=True,
        #                     manual_cloud=True,
        #                     manual_cloud_cer=cer,
        #                     manual_cloud_cwp=lwp,
        #                     manual_cloud_cth=10.0,
        #                     manual_cloud_cbh=9.9,
        #                     manual_cloud_cot=cot,
        #                     wvl=wvl, # um
        #                     wvl_std=wvl_std, # um
        #                     iter=iter,
        #                     purturb_cot=purturb_cot,
        #                     )
    
    # atm_corr_plot(date=datetime.datetime(2024, 5, 30),
    #                     tmhr_ranges_select=[
    #                                         [11.37, 11.43],
    #                                         ],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 simulation_interval=10,
    #                 config=config,
    #                 )
    
    

