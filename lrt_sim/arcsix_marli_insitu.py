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

from util.util import *

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


def marli_flt_trk_lrt_para(
        date: datetime.datetime = datetime.datetime(2024,6,5),
        extent = (-44,-58,83.4,84.1),
        sizes = (50,20,4),
        tmhr_ranges_select = ((15.36,15.6),(16.32,16.6)),
        fname_marli: str = "data/marli/…cdf",
        fname_kt19: str = "data/kt19/…ict",
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
    idx = closest_indices(sat_tmhrs, mid_times)
    
    
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
        ax1.scatter(data_hsk['lon'][mask], data_hsk['lat'][mask], color=color, s=sizes[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
        ax1.text(data_hsk['lon'][mask][0], data_hsk['lat'][mask][0]+0.03, text1, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
        ax1.text(data_hsk['lon'][mask][-1], data_hsk['lat'][mask][-1]+0.03, text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
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
        ax2.scatter(data_hsk['lon'][mask], data_hsk['lat'][mask], color=color, s=sizes[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
        ax2.text(data_hsk['lon'][mask][0], data_hsk['lat'][mask][0]+0.03, text1, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
        ax2.text(data_hsk['lon'][mask][-1], data_hsk['lat'][mask][-1]+0.03, text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
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
        
        
        sat_select_text = os.path.basename(sat_nc).replace('.nc', '').replace('CLDPROP_L2_', '')
        sat_ = sat_select_text.split('.')[0].replace('_', ' ')
        sat_utc = sat_select_text.split('.')[2]
        title_text = f'{sat_} {sat_utc}\n' + \
            f'Flight track {text1} - {text2} UTC'
        fig.suptitle(title_text, fontsize=20, y=0.95)
        

        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('fig/%s/%s_%s_sat_%d_leg_%d.png' % (date_s, date_s, case_tag, idx[i], i), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/# 
        
        # print("Marli time:", data_marli['tmhr'].min(), data_marli['tmhr'].max())

        # plot vertical marli data
        #/--------------------------------------------------------------/\#
        plt.close('all')
        
        data_marli_time = np.array(data_marli['time'])
        data_ssfr_time = np.array(data_ssfr['time'])/3600  # convert to hours
        data_hsr1_time = np.array(data_hsr1['time'])/3600  # convert to hours
        marli_select = []
        ssfr_select = []
        hsr1_select = []
        for time_hsk in data_hsk['tmhr'][mask]:
            marli_select.append(np.argmin(np.abs(data_marli_time - time_hsk)))
            ssfr_select.append(np.argmin(np.abs(data_ssfr_time - time_hsk)))
            hsr1_select.append(np.argmin(np.abs(data_hsr1_time - time_hsk)))
        
        marli_select = np.array(marli_select)
        ssfr_select = np.array(ssfr_select)
        hsr1_select = np.array(hsr1_select)
        
        marli_time = data_marli['time'][marli_select]
        marli_alt = data_marli['Alt'][marli_select]
        marli_H = data_marli['H']
        marli_T = np.array(data_marli['T'])[marli_select, :]
        marli_T[marli_T == 9999] = np.nan  # remove unrealistic temperatures
        marli_T[marli_T > 100] = np.nan  # remove unrealistic temperatures
        marli_T[marli_T < -100] = np.nan  # remove unrealistic temperatures
        marli_LSR = np.array(data_marli['LSR'])[marli_select, :]
        marli_LSR[marli_LSR == 9999] = np.nan  # remove unrealistic liquid water content
        marli_WVMR = np.array(data_marli['WVMR'])[marli_select, :]
        marli_WVMR[marli_WVMR == 9999] = np.nan  # remove unrealistic water vapor mixing ratios
        
        marli_time_repeat = np.repeat(marli_time[:, np.newaxis], marli_H.size, axis=1)
        marli_H_repeat = np.repeat(marli_H[np.newaxis, :], marli_time.size, axis=0)
        
        print("marli_H shape:", marli_H.shape)
        print("marli_time_repeat shape:", marli_time_repeat.shape)
        print("marli_H_repeat shape:", marli_H_repeat.shape)
        print("marli_T shape:", marli_T.shape)
        print("marli_LSR shape:", marli_LSR.shape)
        print("marli_WVMR shape:", marli_WVMR.shape)
        
        sza_hsk = data_hsk['sza'][mask]
        ssfr_zen_flux = data_ssfr['f_dn'][ssfr_select, :]
        ssfr_nad_flux = data_ssfr['f_up'][ssfr_select, :]
        ssfr_zen_toa = flux_solar_interp(data_ssfr['wvl_dn']) * np.cos(np.deg2rad(sza_hsk))[:, np.newaxis]  # W/m^2/nm
        ssfr_zen_wvl = data_ssfr['wvl_dn']
        ssfr_nad_wvl = data_ssfr['wvl_up']
        
        ssfr_nad_flux_interp = ssfr_zen_flux.copy()
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
        
        hsr1_wvl = data_hsr1['wvl_dn_tot']
        hsr1_total_flux = data_hsr1['f_dn_tot'][hsr1_select]
        hsr1_dif_flux = data_hsr1['f_dn_dif'][hsr1_select]
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
        
        sat_select_text = os.path.basename(sat_nc).replace('.nc', '').replace('CLDPROP_L2_', '')
        sat_ = sat_select_text.split('.')[0].replace('_', ' ')
        sat_utc = sat_select_text.split('.')[2]
        title_text = f'{sat_} {sat_utc}\n' + \
            f'Flight track {text1} - {text2} UTC'
            
        fig.suptitle(title_text, fontsize=24, color='k')
        
        
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_marli_ssfr_%d_leg_%d.png' % (date_s, date_s, case_tag, idx[i], i), bbox_inches='tight')
        
        cot_ssfr = cot_interp(data_ssfr['gps_lon'][ssfr_select], data_ssfr['gps_lat'][ssfr_select])
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1_1 = ax1.twinx()
        l1 = ax1.plot(data_ssfr_time[ssfr_select], ssfr_nad_zen_ratio[:, wvl_860_ind], label='SSFR 860 nm', color='blue', lw=2.0)
        l2 = ax1.plot(data_hsr1_time[hsr1_select], hsr1_dif_ratio[:, hsr1_wvl_790_ind], label='HSR1 790 nm diffuse', color='green', lw=2.0)
        l3 = ax1_1.plot(data_ssfr_time[ssfr_select], cot_ssfr, label='Satellite COT', color='orange', lw=2.0)
        ax1.set_xlabel('Time (UTC)', fontsize=16)
        ax1.set_ylabel('Ratio', fontsize=16)
        ax1_1.set_ylabel('Satellite COT', fontsize=16, color='orange')
        line_lebels = [l.get_label() for l in l1+l2+l3]
        ax1.legend(line_lebels, fontsize=16, loc='best')
        sat_select_text = os.path.basename(sat_nc).replace('.nc', '').replace('CLDPROP_L2_', '')
        sat_ = sat_select_text.split('.')[0].replace('_', ' ')
        sat_utc = sat_select_text.split('.')[2]
        title_text = f'{sat_} {sat_utc}\n' + \
            f'Flight track {text1} - {text2} UTC'
            
        fig.suptitle(title_text, fontsize=24, color='k')
        
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssfr_hsr1_cot_%d_leg_%d.png' % (date_s, date_s, case_tag, idx[i], i), bbox_inches='tight')


    log.info("Finished.")                
    

if __name__ == '__main__':
    
    dir_fig = './fig'
    os.makedirs(dir_fig, exist_ok=True)
    
    config = FlightConfig(mission='ARCSIX',
                            platform='P3B',
                            data_root=_fdir_data_,
                            sat_root_mac='/Volumes/argus/field/arcsix/sat-data',
                            sat_root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data/sat-data',)    
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 5),
    #                  extent=[-44, -58, 83.3, 84.1],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[15.36, 15.60], [16.32, 16.60], [16.78, 16.85]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240605_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240605_R0.ict',
    #                  case_tag='marli_test',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 5),
    #                  extent=[-44, -58, 83.3, 84.1],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[15.36, 15.60], [16.32, 16.60], [16.78, 16.85]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240605_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240605_R0.ict',
    #                  case_tag='marli_test',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 13),
    #                  extent=[-10, -20, 81.9, 82.7],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[13.5, 13.6], [13.67, 13.78]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240613_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240613_R0.ict',
    #                  case_tag='marli_0613_1',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 13),
    #                  extent=[-50, -66, 78.0, 80.0],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[11.52, 11.94]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240613_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240613_R0.ict',
    #                  case_tag='marli_0613_1',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 11),
    #                  extent=[-60, -70, 85.2, 86.0],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[13.08, 13.21], [13.33, 13.43]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240611_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240611_R0.ict',
    #                  case_tag='marli_test_0611_1',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 7),
    #                  extent=[-60, -70, 78.0, 79.6],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[13.8, 13.87], [13.92, 14.0]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240607_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240607_R0.ict',
    #                  case_tag='marli_test_0607_1',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 5, 28),
    #                  extent=[-42, -52, 84.2, 85.9],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[15.66, 15.96]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240528_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240528_R0.ict',
    #                  case_tag='marli_test_0528_1',
    #                  config=config)
    
    marli_flt_trk_lrt_para(date=datetime.datetime(2024, 5, 28),
                     extent=[-40, -52, 84.2, 85.9],
                     sizes = [50, 20, 4],
                     tmhr_ranges_select=[[15.05, 15.12]],
                     fname_marli='data/marli/ARCSIX-MARLi_P3B_20240528_R0.cdf',
                     fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240528_R0.ict',
                     case_tag='marli_test_0528_1',
                     config=config)
    

    pass
