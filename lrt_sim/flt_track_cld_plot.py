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

# change font to Arial
rcParams['font.family'] = 'Arial'


import er3t

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
    def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R1.h5"
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


def marli_flt_trk_lrt_para(
        date: datetime.datetime = datetime.datetime(2024,6,5),
        extent = (-44,-58,83.4,84.1),
        sizes = (50,20,4),
        tmhr_ranges_select = ((15.36,15.6),(16.32,16.6)),
        sat_id_offset=0,
        fname_marli: str = "../data/marli/…cdf",
        fname_kt19: str = "../data/kt19/…ict",
        case_tag: str = "marli_test",
        config: Optional[FlightConfig] = None,
    ):
    log = logging.getLogger("marli")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    data_logic= load_h5(config.logic(date_s))
    data_sat_coll = load_h5(config.sat_coll(date_s))
    head, data_kt19 = read_ict_kt19(str(config.kt19(fname_kt19)), encoding="utf-8",
                                    na_values=[-9999999,-777,-888])
    # MARLI netCDF
    with NcDataset(str(config.marli(fname_marli))) as ds:
        data_marli = {var: ds.variables[var][:] for var in ("time","Alt","H","T","LSR","WVMR")}

    # Satellite metadata
    sat_names, sat_tmhrs, sat_files = parse_sat(config.sat_coll(date_s))
    mid_times = np.array([np.mean(rng) for rng in tmhr_ranges_select])
    idx = closest_indices(sat_tmhrs, mid_times) + sat_id_offset
    
    
    # Solar spectrum
    f_solar = pd.read_csv('kurudz_ssfr.dat', delim_whitespace=True, comment='#', names=['wvl', 'flux'])
    wvl_solar = f_solar['wvl'].values
    flux_solar = f_solar['flux'].values/1000 # in W/m^2/nm
    flux_solar_interp = interp1d(wvl_solar, flux_solar, bounds_error=False, fill_value=0.0)


    # Build leg masks
    hsk_tm = np.array(data_hsk["tmhr"])
    leg_masks = [(hsk_tm>=lo)&(hsk_tm<=hi) for lo,hi in tmhr_ranges_select]

    # Loop legs: load raw NC, apply cloud logic, interpolate, plot
    for i, mask in enumerate(leg_masks):
        sat_nc = config.sat_nc(date_s, sat_files[idx[i]])
        # … load geolocation_data + geophysical_data …
        # … apply PCL fill, CGT table, thin cirrus conditioners …
        # … build interpolators (NearestNDInterpolator) …
        # … call two plotting helpers: one for map overlay, one for MARLI panels …
        
        
            
        # read the cloud observation information
        print('Reading cloud observation information from %s ...' % sat_nc)
        f = Dataset(sat_nc, 'r')
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
        fig = plt.figure(figsize=(16, 10))
        ax1 = fig.add_subplot(211, projection=proj0)
        cs_cot = ax1.pcolormesh(lon_s, lat_s,  cot_s, cmap='jet', vmin=0.0, vmax=8.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
        ax1.plot(data_hsk['lon'], data_hsk['lat'], lw=1.5, color='grey', transform=ccrs.PlateCarree(), zorder=1, alpha=0.6)

        colors1 = ['r', 'g', 'b', 'brown']
        # colors1 = ['b', 'brown']
        color = colors1[i]

        text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
        text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
        ax1.scatter(data_hsk['lon'][mask], data_hsk['lat'][mask], color=color, s=35, 
                    lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
        text_arg = dict(color=color, fontsize=12, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree(), fontweight='bold')
        ax1.text(data_hsk['lon'][mask][0], data_hsk['lat'][mask][0]-0.06, text1, **text_arg)
        ax1.text(data_hsk['lon'][mask][-1], data_hsk['lat'][mask][-1]+0.03 , text2, **text_arg)
        # ax1.scatter(f_lon[3], f_lat[3], s=5, marker='^', color='orange')
        # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)
        hsk_time_1530_idx = np.argmin(np.abs(data_hsk['tmhr'] - (15+30/60+56/3600.0)))
        ax1.scatter(data_hsk['lon'][hsk_time_1530_idx], data_hsk['lat'][hsk_time_1530_idx], s=450, marker='*', 
                    color='orange', transform=ccrs.PlateCarree(), zorder=2, alpha=1.0)


        cbar1 = fig.colorbar(cs_cot, ax=ax1, orientation='horizontal',
                             pad=0.1, shrink=0.39, aspect=30,# extend='max'
                             )
        cbar1.set_label('Cloud Optical Thickness', fontsize=10)
        cbar1.ax.tick_params(labelsize=10)
        
    
        
        for ax, label in zip([ax1], ['Cloud Optical Thickness',]):
            ax.coastlines(resolution='10m', color='gray', lw=0.5)
            g_lines = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
            g_lines.xlocator = FixedLocator(np.arange(-53, 181, 3))
            g_lines.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2))
            g_lines.top_labels = False
            g_lines.right_labels = False
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.tick_params(axis='both', which='major', labelsize=12)
            # ax.set_title(label, fontsize=12)
        
        
        sat_select_text = os.path.basename(sat_nc).replace('.nc', '').replace('CLDPROP_L2_', '')
        sat_ = sat_select_text.split('.')[0].replace('_', ' ')
        sat_utc = sat_select_text.split('.')[2]
        title_text = f'{sat_} {sat_utc}\n' + \
            f'Flight track {text1} - {text2} UTC'
        # fig.suptitle(title_text, fontsize=12 , y=0.95)

        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('fig/%s/%s_%s_sat_%s_%s_leg_%d.png' % (date_s, date_s, case_tag, sat_, sat_utc, i), bbox_inches='tight', metadata=_metadata, dpi=150)
        #\--------------------------------------------------------------/# 
        
        # print("Marli time:", data_marli['tmhr'].min(), data_marli['tmhr'].max())

        # plot vertical marli data
        #/--------------------------------------------------------------/\#
        plt.close('all')
        
        


    log.info("Finished.")                


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
    
    
  
    
    
    marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 5),
                     extent=[-46.5 , -53.5 , 83.65, 84.05],
                     sizes = [50, 20, 4],
                     tmhr_ranges_select=[[15.42  , 15.55]],
                     sat_id_offset=0,
                     fname_marli='../data/marli/ARCSIX-MARLi_P3B_20240605_R0.cdf',
                     fname_kt19='../data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240605_R0.ict',
                     case_tag='flt_track',
                     config=config)
    
    marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 5),
                     extent=[-46.5 , -53.5 , 83.65, 84.05],
                     sizes = [50, 20, 4],
                     tmhr_ranges_select=[[15.42  , 15.55]],
                     sat_id_offset=-1,
                     fname_marli='../data/marli/ARCSIX-MARLi_P3B_20240605_R0.cdf',
                     fname_kt19='../data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240605_R0.ict',
                     case_tag='flt_track',
                     config=config)
    
    
    
    
    sys.exit()
    
    
    

    pass
