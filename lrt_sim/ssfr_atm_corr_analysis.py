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
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import bisect
import pandas as pd
import xarray as xr
from collections import defaultdict
import gc
from pyproj import Transformer
from util import *
# mpl.use('Agg')
from matplotlib import rcParams

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif" # Ensure sans-serif is used as the default family



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



o2a_1_start, o2a_1_end = 748, 780
# h2o_1_start, h2o_1_end = 672, 706
# h2o_2_start, h2o_2_end = 705, 746
h2o_1_start, h2o_1_end = 650 , 706
h2o_2_start, h2o_2_end = 705, 760
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



def combined_atm_corr():
    log = logging.getLogger("atm corr combined")

    output_dir = f'{_fdir_general_}/sfc_alb_combined_smooth_450nm'
    
    combined_output_file = f'{output_dir}/sfc_alb_combined_spring_summer.pkl'
    with open(combined_output_file, 'rb') as r:
        combined_data = pickle.load(r)
        
    fig_dir = f'fig/sfc_alb_corr_analysis'
    os.makedirs(fig_dir, exist_ok=True)
    
    #""" 
    # 7/29 clear_atm_corr_1
    date_select = '20240729'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select)
    case_tag_select = 'clear_atm_corr_1'
    case_tag_mask = np.array([case_tag_select in ct for ct in combined_data['case_tags_summer_all']])
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    lat_mask = (lat_selected_all >= 83.9) & (lat_selected_all <=   85.0)
    lon_selected_all = lon_selected_all[lat_mask]
    lat_selected_all = lat_selected_all[lat_mask]
    alt_selected_all = alt_selected_all[lat_mask]
    alb_selected_all = alb_selected_all[lat_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[lat_mask]
    
    # print("combined_data['case_tags_summer_all'] set:", set(combined_data['case_tags_summer_all']))
    # print("date_summer_mask sum:", np.sum(date_summer_mask))
    # print("case_tag_mask sum:", np.sum(case_tag_mask))
    # print("alb_selected_all.shape:", alb_selected_all.shape)
    # sys.exit()
    
    alb_selected_high_alt_mask = alt_selected_all >= 3.0  # select alt >= 3 km
    alb_selected_low_alt_mask = alt_selected_all < 3.0  # select alt < 3 km
    alb_selected_high_alt = alb_selected_all[alb_selected_high_alt_mask, :]
    alb_selected_low_alt = alb_selected_all[alb_selected_low_alt_mask, :]
    alb_selected_high_alt_avg = np.nanmean(alb_selected_high_alt, axis=0)
    alb_selected_low_alt_avg = np.nanmean(alb_selected_low_alt, axis=0)
    alb_selected_high_alt_std = np.nanstd(alb_selected_high_alt, axis=0)
    alb_selected_low_alt_std = np.nanstd(alb_selected_low_alt, axis=0)
    alt_selected_high_alt_avg = np.nanmean(alt_selected_all[alb_selected_high_alt_mask])
    alt_selected_low_alt_avg = np.nanmean(alt_selected_all[alb_selected_low_alt_mask])
    lon_selected_high_alt = lon_selected_all[alb_selected_high_alt_mask]
    lon_selected_low_alt = lon_selected_all[alb_selected_low_alt_mask]
    lat_selected_high_alt = lat_selected_all[alb_selected_high_alt_mask]
    lat_selected_low_alt = lat_selected_all[alb_selected_low_alt_mask]
    broadband_alb_selected_all_high_alt = broadband_alb_selected_all[alb_selected_high_alt_mask]
    broadband_alb_selected_all_low_alt = broadband_alb_selected_all[alb_selected_low_alt_mask]
    
    
    
    ### Plot flight tracks
    
    cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    
    eff_alb_ = gas_abs_masking(alb_wvl, np.ones_like(alb_wvl), alt=np.nanmean(alt_selected_all))
    
    fig = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(2, 7, left=0.05, right=0.95, wspace=0.8, hspace=0.3)
    ax1 = fig.add_subplot(gs1[:, :3], projection=cartopy_proj)
    ax2 = fig.add_subplot(gs1[0, 3:])
    ax3 = fig.add_subplot(gs1[1, 3:])
    
    # Set the extent for the main axes
    lon_min = np.min(lon_selected_all)
    lon_max = np.max(lon_selected_all)
    lat_min = np.min(lat_selected_all)
    lat_max = np.max(lat_selected_all)
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 4.5
    lat_buffer = (lat_max - lat_min) * 1
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # features
    ax1.coastlines(linewidth=0.5, color='black')
    ax1.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax1.add_feature(
        cfeature.OCEAN.with_scale('10m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 5.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 1.0))
    g1.top_labels = False


    ax1.scatter(lon_selected_high_alt, lat_selected_high_alt, transform=ccrs.PlateCarree(),
               label=f'{alt_selected_high_alt_avg:.1f} km', c='b', s=5, zorder=3)
    
    ax1.scatter(lon_selected_low_alt, lat_selected_low_alt, transform=ccrs.PlateCarree(),
               label=f'{alt_selected_low_alt_avg:.1f} km', c='r', s=7.5, zorder=2)
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax1.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')
    ax1.set_xlabel('Longitude', fontsize=14)
    ax1.set_ylabel('Latitude', fontsize=14)
    
    
    ax2.plot(alb_wvl, alb_selected_high_alt_avg, label=f'{alt_selected_high_alt_avg:.1f} km', color='b')
    ax2.fill_between(alb_wvl, alb_selected_high_alt_avg-alb_selected_high_alt_std, alb_selected_high_alt_avg+alb_selected_high_alt_std, 
                    color='b', alpha=0.1)
    ax2.plot(alb_wvl, alb_selected_low_alt_avg, label=f'{alt_selected_low_alt_avg:.1f} km', color='r')
    ax2.fill_between(alb_wvl, alb_selected_low_alt_avg-alb_selected_low_alt_std, alb_selected_low_alt_avg+alb_selected_low_alt_std, 
                    color='r', alpha=0.1)
    ax2.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(eff_alb_), color='gray', alpha=0.2, label='Mask Gas absorption bands')
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    ax2.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.hlines(0, 350, 2000, colors='k', linestyles='dashed', linewidth=1)
    ax2.set_xlim(350, 2000)
    
    ax3.scatter(lat_selected_high_alt, broadband_alb_selected_all_high_alt, label=f'{alt_selected_high_alt_avg:.1f} km', c='b', s=10)
    ax3.scatter(lat_selected_low_alt, broadband_alb_selected_all_low_alt, label=f'{alt_selected_low_alt_avg:.1f} km', c='r', s=10)
    ax3.set_xlabel('Latitude ($\mathrm{^o}$N)', fontsize=14)
    ax3.set_ylabel('Broadband Albedo', fontsize=14)
    ax3.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax3.tick_params(labelsize=12)
    
    for ax, ax_num in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
        ax.text(0.01, 1.05, f'({ax_num})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='left')
        ax.tick_params(labelsize=12, which='both')
    
    
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0729_clear_1_summary.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Broadband albedo avg and std high alt:", np.nanmean(broadband_alb_selected_all_high_alt), np.nanstd(broadband_alb_selected_all_high_alt))
    print("Broadband albedo avg and std low alt:", np.nanmean(broadband_alb_selected_all_low_alt), np.nanstd(broadband_alb_selected_all_low_alt))
    
    sys.exit()
    
    #"""
    
    """
    # 6/5 clear_atm_corr_2, 3
    date_select = '20240605'
    date_summer_mask = combined_data['dates_spring_all'] == int(date_select)
    case_tag_select_1 = 'clear_atm_corr_2'
    case_tag_mask_1 = np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_spring_all']])
    case_tag_select_2 = 'clear_atm_corr_3'
    case_tag_mask_2 = np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_spring_all']])
    case_tag_mask = case_tag_mask_1 | case_tag_mask_2
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    # lat_mask = (lat_selected_all >= 83.9) & (lat_selected_all <=   85.0)
    # lon_selected_all = lon_selected_all[lat_mask]
    # lat_selected_all = lat_selected_all[lat_mask]
    # alt_selected_all = alt_selected_all[lat_mask]
    # alb_selected_all = alb_selected_all[lat_mask, :]
    # broadband_alb_selected_all = broadband_alb_selected_all[lat_mask]
    
    time_mask = (time_selected_all >= 14.594)  
    lon_selected_all = lon_selected_all[time_mask]
    lat_selected_all = lat_selected_all[time_mask]
    alt_selected_all = alt_selected_all[time_mask]
    alb_selected_all = alb_selected_all[time_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[time_mask]
    
    # print("combined_data['case_tags_summer_all'] set:", set(combined_data['case_tags_summer_all']))
    # print("date_summer_mask sum:", np.sum(date_summer_mask))
    # print("case_tag_mask sum:", np.sum(case_tag_mask))
    # print("alb_selected_all.shape:", alb_selected_all.shape)
    # sys.exit()
    
    alb_selected_high_alt_mask = alt_selected_all >= 0.3  # select alt >= 300 m
    alb_selected_low_alt_mask = alt_selected_all < 0.3  # select alt < 300 m
    alb_selected_high_alt = alb_selected_all[alb_selected_high_alt_mask, :]
    alb_selected_low_alt = alb_selected_all[alb_selected_low_alt_mask, :]
    alb_selected_high_alt_avg = np.nanmean(alb_selected_high_alt, axis=0)
    alb_selected_low_alt_avg = np.nanmean(alb_selected_low_alt, axis=0)
    alb_selected_high_alt_std = np.nanstd(alb_selected_high_alt, axis=0)
    alb_selected_low_alt_std = np.nanstd(alb_selected_low_alt, axis=0)
    alt_selected_high_alt_avg = np.nanmean(alt_selected_all[alb_selected_high_alt_mask])
    alt_selected_low_alt_avg = np.nanmean(alt_selected_all[alb_selected_low_alt_mask])
    lon_selected_high_alt = lon_selected_all[alb_selected_high_alt_mask]
    lon_selected_low_alt = lon_selected_all[alb_selected_low_alt_mask]
    lat_selected_high_alt = lat_selected_all[alb_selected_high_alt_mask]
    lat_selected_low_alt = lat_selected_all[alb_selected_low_alt_mask]
    broadband_alb_selected_all_high_alt = broadband_alb_selected_all[alb_selected_high_alt_mask]
    broadband_alb_selected_all_low_alt = broadband_alb_selected_all[alb_selected_low_alt_mask]
    
    ### Plot flight tracks
    cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    
    eff_alb_ = gas_abs_masking(alb_wvl, np.ones_like(alb_wvl), alt=np.nanmean(alt_selected_all))
    
    fig = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(2, 7, left=0.05, right=0.95, wspace=0.8, hspace=0.3)
    ax1 = fig.add_subplot(gs1[:, :3], projection=cartopy_proj)
    ax2 = fig.add_subplot(gs1[0, 3:])
    ax3 = fig.add_subplot(gs1[1, 3:])
    
    # Set the extent for the main axes
    lon_min = np.min(lon_selected_all)
    lon_max = np.max(lon_selected_all)
    lat_min = np.min(lat_selected_all)
    lat_max = np.max(lat_selected_all)
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 0.75
    lat_buffer = (lat_max - lat_min) * 3.
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # features
    ax1.coastlines(linewidth=0.5, color='black')
    ax1.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax1.add_feature(
        cfeature.OCEAN.with_scale('10m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 15.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 5.0))
    g1.top_labels = False


    ax1.scatter(lon_selected_high_alt, lat_selected_high_alt, transform=ccrs.PlateCarree(),
               label=f'{alt_selected_high_alt_avg:.1f} km', c='b', s=5, zorder=3)
    
    ax1.scatter(lon_selected_low_alt, lat_selected_low_alt, transform=ccrs.PlateCarree(),
               label=f'{alt_selected_low_alt_avg:.1f} km', c='r', s=7.5, zorder=2)
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax1.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')
    ax1.set_xlabel('Longitude', fontsize=14)
    ax1.set_ylabel('Latitude', fontsize=14)
    
    
    ax2.plot(alb_wvl, alb_selected_high_alt_avg, label=f'{alt_selected_high_alt_avg:.1f} km', color='b')
    ax2.fill_between(alb_wvl, alb_selected_high_alt_avg-alb_selected_high_alt_std, alb_selected_high_alt_avg+alb_selected_high_alt_std, 
                    color='b', alpha=0.1)
    ax2.plot(alb_wvl, alb_selected_low_alt_avg, label=f'{alt_selected_low_alt_avg:.1f} km', color='r')
    ax2.fill_between(alb_wvl, alb_selected_low_alt_avg-alb_selected_low_alt_std, alb_selected_low_alt_avg+alb_selected_low_alt_std, 
                    color='r', alpha=0.1)
    ax2.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(eff_alb_), color='gray', alpha=0.2, label='Mask Gas absorption bands')
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    ax2.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.hlines(0, 350, 2000, colors='k', linestyles='dashed', linewidth=1)
    ax2.set_xlim(350, 2000)
    
    ax3.scatter(lat_selected_high_alt, broadband_alb_selected_all_high_alt, label=f'{alt_selected_high_alt_avg:.1f} km', c='b', s=10)
    ax3.scatter(lat_selected_low_alt, broadband_alb_selected_all_low_alt, label=f'{alt_selected_low_alt_avg:.1f} km', c='r', s=10)
    ax3.set_xlabel('Latitude ($\mathrm{^o}$N)', fontsize=14)
    ax3.set_ylabel('Broadband Albedo', fontsize=14)
    ax3.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax3.tick_params(labelsize=12)
    
    for ax, ax_num in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
        ax.text(0.01, 1.05, f'({ax_num})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='left')
        ax.tick_params(labelsize=12, which='both')
    
    
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0605_clear_23_summary.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    sys.exit()
    
    #"""
    
    """
    # 8/15 clear_atm_corr 100 & 3500 m
    date_select = '20240815'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select_1 = 'clear_atm_corr'
    case_tag_mask_1 = np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_select_2 = 'clear_atm_corr_2'
    case_tag_mask_2 = np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_mask = case_tag_mask_1 | case_tag_mask_2
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    # lat_mask = (lat_selected_all >= 83.9) & (lat_selected_all <=   85.0)
    # lon_selected_all = lon_selected_all[lat_mask]
    # lat_selected_all = lat_selected_all[lat_mask]
    # alt_selected_all = alt_selected_all[lat_mask]
    # alb_selected_all = alb_selected_all[lat_mask, :]
    # broadband_alb_selected_all = broadband_alb_selected_all[lat_mask]
    
    time_mask = (time_selected_all >= 14.15) & (time_selected_all <= 14.73)
    lon_selected_all = lon_selected_all[time_mask]
    lat_selected_all = lat_selected_all[time_mask]
    alt_selected_all = alt_selected_all[time_mask]
    alb_selected_all = alb_selected_all[time_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[time_mask]
    
    # print("combined_data['case_tags_summer_all'] set:", set(combined_data['case_tags_summer_all']))
    # print("date_summer_mask sum:", np.sum(date_summer_mask))
    # print("case_tag_mask sum:", np.sum(case_tag_mask))
    # print("alb_selected_all.shape:", alb_selected_all.shape)
    # sys.exit()
    
    alb_selected_high_alt_mask = alt_selected_all >= 0.3  # select alt >= 300 m
    alb_selected_low_alt_mask = alt_selected_all < 0.3  # select alt < 300 m
    alb_selected_high_alt = alb_selected_all[alb_selected_high_alt_mask, :]
    alb_selected_low_alt = alb_selected_all[alb_selected_low_alt_mask, :]
    alb_selected_high_alt_avg = np.nanmean(alb_selected_high_alt, axis=0)
    alb_selected_low_alt_avg = np.nanmean(alb_selected_low_alt, axis=0)
    alb_selected_high_alt_std = np.nanstd(alb_selected_high_alt, axis=0)
    alb_selected_low_alt_std = np.nanstd(alb_selected_low_alt, axis=0)
    alt_selected_high_alt_avg = np.nanmean(alt_selected_all[alb_selected_high_alt_mask])
    alt_selected_low_alt_avg = np.nanmean(alt_selected_all[alb_selected_low_alt_mask])
    lon_selected_high_alt = lon_selected_all[alb_selected_high_alt_mask]
    lon_selected_low_alt = lon_selected_all[alb_selected_low_alt_mask]
    lat_selected_high_alt = lat_selected_all[alb_selected_high_alt_mask]
    lat_selected_low_alt = lat_selected_all[alb_selected_low_alt_mask]
    broadband_alb_selected_all_high_alt = broadband_alb_selected_all[alb_selected_high_alt_mask]
    broadband_alb_selected_all_low_alt = broadband_alb_selected_all[alb_selected_low_alt_mask]
    
    
    
    ### Plot flight tracks
    cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    
    eff_alb_ = gas_abs_masking(alb_wvl, np.ones_like(alb_wvl), alt=np.nanmean(alt_selected_all))
    
    fig = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(2, 7, left=0.05, right=0.95, wspace=0.8, hspace=0.3)
    ax1 = fig.add_subplot(gs1[:, :3], projection=cartopy_proj)
    ax2 = fig.add_subplot(gs1[0, 3:])
    ax3 = fig.add_subplot(gs1[1, 3:])
    
    # Set the extent for the main axes
    lon_min = np.min(lon_selected_all)
    lon_max = np.max(lon_selected_all)
    lat_min = np.min(lat_selected_all)
    lat_max = np.max(lat_selected_all)
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 0.65
    lat_buffer = (lat_max - lat_min) * 3.9
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # features
    ax1.coastlines(linewidth=0.5, color='black')
    ax1.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax1.add_feature(
        cfeature.OCEAN.with_scale('10m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 15.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 5.0))
    g1.top_labels = False


    ax1.scatter(lon_selected_high_alt, lat_selected_high_alt, transform=ccrs.PlateCarree(),
               label=f'{alt_selected_high_alt_avg:.1f} km', c='b', s=5, zorder=3)
    
    ax1.scatter(lon_selected_low_alt, lat_selected_low_alt, transform=ccrs.PlateCarree(),
               label=f'{alt_selected_low_alt_avg:.1f} km', c='r', s=7.5, zorder=2)
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax1.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')
    ax1.set_xlabel('Longitude', fontsize=14)
    ax1.set_ylabel('Latitude', fontsize=14)
    
    
    ax2.plot(alb_wvl, alb_selected_high_alt_avg, label=f'{alt_selected_high_alt_avg:.1f} km', color='b')
    ax2.fill_between(alb_wvl, alb_selected_high_alt_avg-alb_selected_high_alt_std, alb_selected_high_alt_avg+alb_selected_high_alt_std, 
                    color='b', alpha=0.1)
    ax2.plot(alb_wvl, alb_selected_low_alt_avg, label=f'{alt_selected_low_alt_avg:.1f} km', color='r')
    ax2.fill_between(alb_wvl, alb_selected_low_alt_avg-alb_selected_low_alt_std, alb_selected_low_alt_avg+alb_selected_low_alt_std, 
                    color='r', alpha=0.1)
    ax2.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(eff_alb_), color='gray', alpha=0.2, label='Mask Gas absorption bands')
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    ax2.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.hlines(0, 350, 2000, colors='k', linestyles='dashed', linewidth=1)
    ax2.set_xlim(350, 2000)
    
    ax3.scatter(lat_selected_high_alt, broadband_alb_selected_all_high_alt, label=f'{alt_selected_high_alt_avg:.1f} km', c='b', s=10)
    ax3.scatter(lat_selected_low_alt, broadband_alb_selected_all_low_alt, label=f'{alt_selected_low_alt_avg:.1f} km', c='r', s=10)
    ax3.set_xlabel('Latitude ($\mathrm{^o}$N)', fontsize=14)
    ax3.set_ylabel('Broadband Albedo', fontsize=14)
    ax3.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax3.tick_params(labelsize=12)
    
    for ax, ax_num in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
        ax.text(0.01, 1.05, f'({ax_num})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='left')
        ax.tick_params(labelsize=12, which='both')
    
    
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0815_clear_12_summary.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    # sys.exit()
    
    #"""
    
    """
    # 8/15 clear_atm_corr 1.7 & 3.5 km
    date_select = '20240815'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select_1 = 'clear_atm_corr'
    case_tag_mask_1 = np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_select_2 = 'clear_atm_corr_2'
    case_tag_mask_2 = np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_mask = case_tag_mask_1 | case_tag_mask_2
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    # lat_mask = (lat_selected_all >= 83.9) & (lat_selected_all <=   85.0)
    # time_selected_all = time_selected_all[lat_mask]
    # lon_selected_all = lon_selected_all[lat_mask]
    # lat_selected_all = lat_selected_all[lat_mask]
    # alt_selected_all = alt_selected_all[lat_mask]
    # alb_selected_all = alb_selected_all[lat_mask, :]
    # broadband_alb_selected_all = broadband_alb_selected_all[lat_mask]
    
    lon_mask = (lon_selected_all >= -22.0) & (lon_selected_all <= -17.95)
    time_selected_all = time_selected_all[lon_mask]
    lon_selected_all = lon_selected_all[lon_mask]
    lat_selected_all = lat_selected_all[lon_mask]
    alt_selected_all = alt_selected_all[lon_mask]
    alb_selected_all = alb_selected_all[lon_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[lon_mask]
    
    time_mask = (time_selected_all >= 14.68 ) & (time_selected_all <= 15.16)
    lon_selected_all = lon_selected_all[time_mask]
    lat_selected_all = lat_selected_all[time_mask]
    alt_selected_all = alt_selected_all[time_mask]
    alb_selected_all = alb_selected_all[time_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[time_mask]
    
    # print("combined_data['case_tags_summer_all'] set:", set(combined_data['case_tags_summer_all']))
    # print("date_summer_mask sum:", np.sum(date_summer_mask))
    # print("case_tag_mask sum:", np.sum(case_tag_mask))
    # print("alb_selected_all.shape:", alb_selected_all.shape)
    # sys.exit()
    
    alb_selected_high_alt_mask = alt_selected_all >= 2. 
    alb_selected_low_alt_mask = alt_selected_all < 2.  
    alb_selected_high_alt = alb_selected_all[alb_selected_high_alt_mask, :]
    alb_selected_low_alt = alb_selected_all[alb_selected_low_alt_mask, :]
    alb_selected_high_alt_avg = np.nanmean(alb_selected_high_alt, axis=0)
    alb_selected_low_alt_avg = np.nanmean(alb_selected_low_alt, axis=0)
    alb_selected_high_alt_std = np.nanstd(alb_selected_high_alt, axis=0)
    alb_selected_low_alt_std = np.nanstd(alb_selected_low_alt, axis=0)
    alt_selected_high_alt_avg = np.nanmean(alt_selected_all[alb_selected_high_alt_mask])
    alt_selected_low_alt_avg = np.nanmean(alt_selected_all[alb_selected_low_alt_mask])
    lon_selected_high_alt = lon_selected_all[alb_selected_high_alt_mask]
    lon_selected_low_alt = lon_selected_all[alb_selected_low_alt_mask]
    lat_selected_high_alt = lat_selected_all[alb_selected_high_alt_mask]
    lat_selected_low_alt = lat_selected_all[alb_selected_low_alt_mask]
    broadband_alb_selected_all_high_alt = broadband_alb_selected_all[alb_selected_high_alt_mask]
    broadband_alb_selected_all_low_alt = broadband_alb_selected_all[alb_selected_low_alt_mask]
    


    
    ### Plot flight tracks
    cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    
    eff_alb_ = gas_abs_masking(alb_wvl, np.ones_like(alb_wvl), alt=np.nanmean(alt_selected_all))
    
    fig = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(2, 7, left=0.05, right=0.95, wspace=1.0, hspace=0.3)
    ax1 = fig.add_subplot(gs1[:, :3], projection=cartopy_proj)
    ax2 = fig.add_subplot(gs1[0, 3:])
    ax3 = fig.add_subplot(gs1[1, 3:])
    
    # Set the extent for the main axes
    lon_min = np.min(lon_selected_all)
    lon_max = np.max(lon_selected_all)
    lat_min = np.min(lat_selected_all)
    lat_max = np.max(lat_selected_all)
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 3.6
    lat_buffer = (lat_max - lat_min) * 14.4
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # features
    ax1.coastlines(linewidth=0.5, color='black')
    ax1.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax1.add_feature(
        cfeature.OCEAN.with_scale('10m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 15.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 5.0))
    g1.top_labels = False


    ax1.scatter(lon_selected_high_alt, lat_selected_high_alt, transform=ccrs.PlateCarree(),
               label=f'{alt_selected_high_alt_avg:.1f} km', c='b', s=5, zorder=3)
    
    ax1.scatter(lon_selected_low_alt, lat_selected_low_alt, transform=ccrs.PlateCarree(),
               label=f'{alt_selected_low_alt_avg:.1f} km', c='r', s=7.5, zorder=2)
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax1.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')
    ax1.set_xlabel('Longitude', fontsize=14)
    ax1.set_ylabel('Latitude', fontsize=14)
    
    
    ax2.plot(alb_wvl, alb_selected_high_alt_avg, label=f'{alt_selected_high_alt_avg:.1f} km', color='b')
    ax2.fill_between(alb_wvl, alb_selected_high_alt_avg-alb_selected_high_alt_std, alb_selected_high_alt_avg+alb_selected_high_alt_std, 
                    color='b', alpha=0.1)
    ax2.plot(alb_wvl, alb_selected_low_alt_avg, label=f'{alt_selected_low_alt_avg:.1f} km', color='r')
    ax2.fill_between(alb_wvl, alb_selected_low_alt_avg-alb_selected_low_alt_std, alb_selected_low_alt_avg+alb_selected_low_alt_std, 
                    color='r', alpha=0.1)
    ax2.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(eff_alb_), color='gray', alpha=0.2, label='Mask Gas absorption bands')
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    ax2.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.hlines(0, 350, 2000, colors='k', linestyles='dashed', linewidth=1)
    ax2.set_xlim(350, 2000)
    
    lon_selected_high_alt[lon_selected_high_alt<0] *= -1 # convert to degrees W
    lon_selected_low_alt[lon_selected_low_alt<0] *= -1   # convert
    ax3.scatter(lon_selected_high_alt, broadband_alb_selected_all_high_alt, label=f'{alt_selected_high_alt_avg:.1f} km', c='b', s=10)
    ax3.scatter(lon_selected_low_alt, broadband_alb_selected_all_low_alt, label=f'{alt_selected_low_alt_avg:.1f} km', c='r', s=10)
    # revert x-axis for degrees W
    ax3.set_xlim(np.max(lon_selected_all)*-0.95, np.min(lon_selected_all)*-1.05)
    ax3.set_xlabel('Longitude ($\mathrm{^o}$W)', fontsize=14)
    ax3.set_ylabel('Broadband Albedo', fontsize=14)
    ax3.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax3.tick_params(labelsize=12)
    
    for ax, ax_num in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
        ax.text(0.01, 1.05, f'({ax_num})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='left')
        ax.tick_params(labelsize=12, which='both')
    
    
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0815_clear_23_summary.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    sys.exit()
    
    #"""
    
    """
    # 0729 clear_sky_spiral_atm_corr
    date_select = '20240729'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select = 'clear_sky_spiral_atm_corr'
    case_tag_mask = np.array([case_tag_select in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select in ct for ct in combined_data['case_tags_summer_all']])
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    lon_select = []
    lat_select = []
    alt_select = []
    time_select = []
    alb_select = []
    broadband_alb_select = []
    lon_select_avg = []
    lat_select_avg = []
    alt_select_avg = []
    time_select_avg = []
    alb_select_avg = []
    alb_select_std = []
    broadband_alb_select_avg = []
    broadband_alb_select_std = []
    
    # find contiguous segments in time_selected_all
    time_diff = np.diff(time_selected_all*60*60)  # convert to seconds
    gap_indices = np.where(time_diff > 1.1)[0]  # indices where gaps occur
    segment_indices = np.split(np.arange(len(time_selected_all)), gap_indices + 1)
    for segment in segment_indices:
        # if len(segment) < 5:
        #     continue  # skip segments with less than 5 points
        print("time start-end:", time_selected_all[segment[0]], time_selected_all[segment[-1]])
        lon_select.append(lon_selected_all[segment])
        lat_select.append(lat_selected_all[segment])
        alt_select.append(alt_selected_all[segment])
        time_select.append(time_selected_all[segment])
        alb_select.append(alb_selected_all[segment, :])
        broadband_alb_select.append(broadband_alb_selected_all[segment])
        lon_select_avg.append(np.nanmean(lon_selected_all[segment]))
        lat_select_avg.append(np.nanmean(lat_selected_all[segment]))
        alt_select_avg.append(np.nanmean(alt_selected_all[segment]))
        time_select_avg.append(np.nanmean(time_selected_all[segment]))
        alb_select_avg.append(np.nanmean(alb_selected_all[segment, :], axis=0))
        alb_select_std.append(np.nanstd(alb_selected_all[segment, :], axis=0))
        broadband_alb_select_avg.append(np.nanmean(broadband_alb_selected_all[segment]))
        broadband_alb_select_std.append(np.nanstd(broadband_alb_selected_all[segment]))
    
    lon_select_avg = np.array(lon_select_avg)
    lat_select_avg = np.array(lat_select_avg)
    alt_select_avg = np.array(alt_select_avg)
    time_select_avg = np.array(time_select_avg)
    alb_select_avg = np.array(alb_select_avg)
    alb_select_std = np.array(alb_select_std)
    broadband_alb_select_avg = np.array(broadband_alb_select_avg)
    broadband_alb_select_std = np.array(broadband_alb_select_std)
    
    ### Plot flight tracks
    cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    
    eff_alb_ = gas_abs_masking(alb_wvl, np.ones_like(alb_wvl), alt=np.nanmean(alt_selected_all))
    
    fig = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(2, 7, left=0.05, right=0.95, wspace=0.8, hspace=0.3)
    ax1 = fig.add_subplot(gs1[:, :3], projection=cartopy_proj)
    ax2 = fig.add_subplot(gs1[0, 3:])
    ax3 = fig.add_subplot(gs1[1, 3:])
    
    # Set the extent for the main axes
    lon_min = np.min(lon_selected_all)
    lon_max = np.max(lon_selected_all)
    lat_min = np.min(lat_selected_all)
    lat_max = np.max(lat_selected_all)
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 3.5
    lat_buffer = (lat_max - lat_min) * 3.5
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # features
    ax1.coastlines(linewidth=0.5, color='black')
    ax1.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax1.add_feature(
        cfeature.OCEAN.with_scale('10m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 5.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 1.0))
    g1.top_labels = False
    
    # Create a ScalarMappable
    data_min, data_max = np.arange(len(alt_select_avg)).min(), np.arange(len(alt_select_avg)).max()
    norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
    cmap = cm.jet # Or any other built-in colormap like cm.viridis
    s_m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_series = s_m.to_rgba(np.arange(len(alt_select_avg)))
    
    

    for i in range(len(lon_select)):
        alt_avg = alt_select_avg[i]
        ax1.plot(lon_select[i], lat_select[i], transform=ccrs.PlateCarree(),
                 label=f'{alt_avg:.1f} km', c=color_series[i], linewidth=2, zorder=2)

    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax1.legend(fontsize=14)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')
    ax1.set_xlabel('Longitude', fontsize=14)
    ax1.set_ylabel('Latitude', fontsize=14)
    
    for i in range(len(alb_select_avg)):
        alt_avg = alt_select_avg[i]
        ax2.plot(alb_wvl, alb_select_avg[i, :], label=f'{alt_avg:.1f} km', color=color_series[i])
        ax2.fill_between(alb_wvl, alb_select_avg[i, :]-alb_select_std[i, :], alb_select_avg[i, :]+alb_select_std[i, :], 
                        color=color_series[i], alpha=0.1)
        
        ax3.scatter(broadband_alb_select[i], alt_select[i], c=color_series[i], s=15, alpha=0.8)
        
        
    
    ax2.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(eff_alb_), color='gray', alpha=0.2,)#  label='Mask Gas absorption bands')
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    # ax2.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, -0.1))
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.hlines(0, 350, 2000, colors='k', linestyles='dashed', linewidth=1)
    ax2.set_xlim(350, 2000)
    

    ax3.set_ylabel('Altitude (km)', fontsize=14)
    ax3.set_xlabel('Broadband Albedo', fontsize=14)
    # ax3.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax3.tick_params(labelsize=12)
    
    for ax, ax_num in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
        ax.text(0.01, 1.05, f'({ax_num})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='left')
        ax.tick_params(labelsize=12, which='both')
    
    
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0729_clear_spiral_summary.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    # sys.exit()
    
    #"""
    
    """
    # 0605 clear_sky_spiral_atm_corr
    date_select = '20240605'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select = 'clear_sky_spiral_atm_corr'
    case_tag_mask = np.array([case_tag_select in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select in ct for ct in combined_data['case_tags_summer_all']])
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    lon_select = []
    lat_select = []
    alt_select = []
    time_select = []
    alb_select = []
    broadband_alb_select = []
    lon_select_avg = []
    lat_select_avg = []
    alt_select_avg = []
    time_select_avg = []
    alb_select_avg = []
    alb_select_std = []
    broadband_alb_select_avg = []
    broadband_alb_select_std = []
    
    # find contiguous segments in time_selected_all
    time_diff = np.diff(time_selected_all*60*60)  # convert to seconds
    gap_indices = np.where(time_diff > 1.1)[0]  # indices where gaps occur
    segment_indices = np.split(np.arange(len(time_selected_all)), gap_indices + 1)
    for segment in segment_indices:
        # if len(segment) < 5:
        #     continue  # skip segments with less than 5 points
        print("time start-end:", time_selected_all[segment[0]], time_selected_all[segment[-1]])
        lon_select.append(lon_selected_all[segment])
        lat_select.append(lat_selected_all[segment])
        alt_select.append(alt_selected_all[segment])
        time_select.append(time_selected_all[segment])
        alb_select.append(alb_selected_all[segment, :])
        broadband_alb_select.append(broadband_alb_selected_all[segment])
        lon_select_avg.append(np.nanmean(lon_selected_all[segment]))
        lat_select_avg.append(np.nanmean(lat_selected_all[segment]))
        alt_select_avg.append(np.nanmean(alt_selected_all[segment]))
        time_select_avg.append(np.nanmean(time_selected_all[segment]))
        alb_select_avg.append(np.nanmean(alb_selected_all[segment, :], axis=0))
        alb_select_std.append(np.nanstd(alb_selected_all[segment, :], axis=0))
        broadband_alb_select_avg.append(np.nanmean(broadband_alb_selected_all[segment]))
        broadband_alb_select_std.append(np.nanstd(broadband_alb_selected_all[segment]))
    
    lon_select_avg = np.array(lon_select_avg)
    lat_select_avg = np.array(lat_select_avg)
    alt_select_avg = np.array(alt_select_avg)
    time_select_avg = np.array(time_select_avg)
    alb_select_avg = np.array(alb_select_avg)
    alb_select_std = np.array(alb_select_std)
    broadband_alb_select_avg = np.array(broadband_alb_select_avg)
    broadband_alb_select_std = np.array(broadband_alb_select_std)
    
    ### Plot flight tracks
    cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    
    eff_alb_ = gas_abs_masking(alb_wvl, np.ones_like(alb_wvl), alt=np.nanmean(alt_selected_all))
    
    fig = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(2, 7, left=0.05, right=0.95, wspace=0.8, hspace=0.3)
    ax1 = fig.add_subplot(gs1[:, :3], projection=cartopy_proj)
    ax2 = fig.add_subplot(gs1[0, 3:])
    ax3 = fig.add_subplot(gs1[1, 3:])
    
    # Set the extent for the main axes
    lon_min = np.min(lon_selected_all)
    lon_max = np.max(lon_selected_all)
    lat_min = np.min(lat_selected_all)
    lat_max = np.max(lat_selected_all)
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 3.5
    lat_buffer = (lat_max - lat_min) * 3.5
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # features
    ax1.coastlines(linewidth=0.5, color='black')
    ax1.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax1.add_feature(
        cfeature.OCEAN.with_scale('10m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 5.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 1.0))
    g1.top_labels = False
    
    # Create a ScalarMappable
    data_min, data_max = np.arange(len(alt_select_avg)).min(), np.arange(len(alt_select_avg)).max()
    norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
    cmap = cm.jet # Or any other built-in colormap like cm.viridis
    s_m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_series = s_m.to_rgba(np.arange(len(alt_select_avg)))
    
    

    for i in range(len(lon_select)):
        alt_avg = alt_select_avg[i]
        ax1.plot(lon_select[i], lat_select[i], transform=ccrs.PlateCarree(),
                 label=f'{alt_avg:.1f} km', c=color_series[i], linewidth=2, zorder=2)

    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax1.legend(fontsize=14)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')
    ax1.set_xlabel('Longitude', fontsize=14)
    ax1.set_ylabel('Latitude', fontsize=14)
    
    for i in range(len(alb_select_avg)):
        alt_avg = alt_select_avg[i]
        ax2.plot(alb_wvl, alb_select_avg[i, :], label=f'{alt_avg:.1f} km', color=color_series[i])
        ax2.fill_between(alb_wvl, alb_select_avg[i, :]-alb_select_std[i, :], alb_select_avg[i, :]+alb_select_std[i, :], 
                        color=color_series[i], alpha=0.1)
        
        ax3.scatter(broadband_alb_select[i], alt_select[i], c=color_series[i], s=15, alpha=0.8)
        
        
    
    ax2.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(eff_alb_), color='gray', alpha=0.2,)#  label='Mask Gas absorption bands')
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    # ax2.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, -0.1))
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.hlines(0, 350, 2000, colors='k', linestyles='dashed', linewidth=1)
    ax2.set_xlim(350, 2000)
    

    ax3.set_ylabel('Altitude (km)', fontsize=14)
    ax3.set_xlabel('Broadband Albedo', fontsize=14)
    # ax3.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax3.tick_params(labelsize=12)
    
    for ax, ax_num in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
        ax.text(0.01, 1.05, f'({ax_num})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='left')
        ax.tick_params(labelsize=12, which='both')
    
    
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0605_clear_spiral_summary.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    # sys.exit()
    
    #"""
    
    """
    # 0611 clear_sky_spiral_atm_corr
    date_select = '20240611'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select = 'clear_sky_spiral_atm_corr'
    case_tag_mask = np.array([case_tag_select in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select in ct for ct in combined_data['case_tags_summer_all']])
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    lon_select = []
    lat_select = []
    alt_select = []
    time_select = []
    alb_select = []
    broadband_alb_select = []
    lon_select_avg = []
    lat_select_avg = []
    alt_select_avg = []
    time_select_avg = []
    alb_select_avg = []
    alb_select_std = []
    broadband_alb_select_avg = []
    broadband_alb_select_std = []
    
    # find contiguous segments in time_selected_all
    time_diff = np.diff(time_selected_all*60*60)  # convert to seconds
    gap_indices = np.where(time_diff > 1.1)[0]  # indices where gaps occur
    segment_indices = np.split(np.arange(len(time_selected_all)), gap_indices + 1)
    for segment in segment_indices:
        # if len(segment) < 5:
        #     continue  # skip segments with less than 5 points
        print("time start-end:", time_selected_all[segment[0]], time_selected_all[segment[-1]])
        lon_select.append(lon_selected_all[segment])
        lat_select.append(lat_selected_all[segment])
        alt_select.append(alt_selected_all[segment])
        time_select.append(time_selected_all[segment])
        alb_select.append(alb_selected_all[segment, :])
        broadband_alb_select.append(broadband_alb_selected_all[segment])
        lon_select_avg.append(np.nanmean(lon_selected_all[segment]))
        lat_select_avg.append(np.nanmean(lat_selected_all[segment]))
        alt_select_avg.append(np.nanmean(alt_selected_all[segment]))
        time_select_avg.append(np.nanmean(time_selected_all[segment]))
        alb_select_avg.append(np.nanmean(alb_selected_all[segment, :], axis=0))
        alb_select_std.append(np.nanstd(alb_selected_all[segment, :], axis=0))
        broadband_alb_select_avg.append(np.nanmean(broadband_alb_selected_all[segment]))
        broadband_alb_select_std.append(np.nanstd(broadband_alb_selected_all[segment]))
    
    lon_select_avg = np.array(lon_select_avg)
    lat_select_avg = np.array(lat_select_avg)
    alt_select_avg = np.array(alt_select_avg)
    time_select_avg = np.array(time_select_avg)
    alb_select_avg = np.array(alb_select_avg)
    alb_select_std = np.array(alb_select_std)
    broadband_alb_select_avg = np.array(broadband_alb_select_avg)
    broadband_alb_select_std = np.array(broadband_alb_select_std)
    
    ### Plot flight tracks
    cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    
    eff_alb_ = gas_abs_masking(alb_wvl, np.ones_like(alb_wvl), alt=np.nanmean(alt_selected_all))
    
    fig = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(2, 7, left=0.05, right=0.95, wspace=0.8, hspace=0.3)
    ax1 = fig.add_subplot(gs1[:, :3], projection=cartopy_proj)
    ax2 = fig.add_subplot(gs1[0, 3:])
    ax3 = fig.add_subplot(gs1[1, 3:])
    
    # Set the extent for the main axes
    lon_min = np.min(lon_selected_all)
    lon_max = np.max(lon_selected_all)
    lat_min = np.min(lat_selected_all)
    lat_max = np.max(lat_selected_all)
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 3.5
    lat_buffer = (lat_max - lat_min) * 3.5
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # features
    ax1.coastlines(linewidth=0.5, color='black')
    ax1.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax1.add_feature(
        cfeature.OCEAN.with_scale('10m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 5.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 1.0))
    g1.top_labels = False
    
    # Create a ScalarMappable
    data_min, data_max = np.arange(len(alt_select_avg)).min(), np.arange(len(alt_select_avg)).max()
    norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
    cmap = cm.jet # Or any other built-in colormap like cm.viridis
    s_m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_series = s_m.to_rgba(np.arange(len(alt_select_avg)))
    
    

    for i in range(len(lon_select)):
        alt_avg = alt_select_avg[i]
        ax1.plot(lon_select[i], lat_select[i], transform=ccrs.PlateCarree(),
                 label=f'{alt_avg:.1f} km', c=color_series[i], linewidth=2, zorder=2)

    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax1.legend(fontsize=14)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')
    ax1.set_xlabel('Longitude', fontsize=14)
    ax1.set_ylabel('Latitude', fontsize=14)
    
    for i in range(len(alb_select_avg)):
        alt_avg = alt_select_avg[i]
        ax2.plot(alb_wvl, alb_select_avg[i, :], label=f'{alt_avg:.1f} km', color=color_series[i])
        ax2.fill_between(alb_wvl, alb_select_avg[i, :]-alb_select_std[i, :], alb_select_avg[i, :]+alb_select_std[i, :], 
                        color=color_series[i], alpha=0.1)
        
        ax3.scatter(broadband_alb_select[i], alt_select[i], c=color_series[i], s=15, alpha=0.8)
        
        
    
    ax2.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(eff_alb_), color='gray', alpha=0.2,)#  label='Mask Gas absorption bands')
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    # ax2.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, -0.1))
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.hlines(0, 350, 2000, colors='k', linestyles='dashed', linewidth=1)
    ax2.set_xlim(350, 2000)
    

    ax3.set_ylabel('Altitude (km)', fontsize=14)
    ax3.set_xlabel('Broadband Albedo', fontsize=14)
    # ax3.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax3.tick_params(labelsize=12)
    
    for ax, ax_num in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
        ax.text(0.01, 1.05, f'({ax_num})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='left')
        ax.tick_params(labelsize=12, which='both')
    
    
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0611_clear_spiral_summary.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    # sys.exit()
    
    #"""
    
    """
    # 0730 clear_sky_spiral_atm_corr
    date_select = '20240730'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select = 'clear_sky_spiral_atm_corr'
    case_tag_mask = np.array([case_tag_select in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select in ct for ct in combined_data['case_tags_summer_all']])
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    lon_select = []
    lat_select = []
    alt_select = []
    time_select = []
    alb_select = []
    broadband_alb_select = []
    lon_select_avg = []
    lat_select_avg = []
    alt_select_avg = []
    time_select_avg = []
    alb_select_avg = []
    alb_select_std = []
    broadband_alb_select_avg = []
    broadband_alb_select_std = []
    
    # find contiguous segments in time_selected_all
    time_diff = np.diff(time_selected_all*60*60)  # convert to seconds
    gap_indices = np.where(time_diff > 1.1)[0]  # indices where gaps occur
    segment_indices = np.split(np.arange(len(time_selected_all)), gap_indices + 1)
    for segment in segment_indices:
        # if len(segment) < 5:
        #     continue  # skip segments with less than 5 points
        print("time start-end:", time_selected_all[segment[0]], time_selected_all[segment[-1]])
        lon_select.append(lon_selected_all[segment])
        lat_select.append(lat_selected_all[segment])
        alt_select.append(alt_selected_all[segment])
        time_select.append(time_selected_all[segment])
        alb_select.append(alb_selected_all[segment, :])
        broadband_alb_select.append(broadband_alb_selected_all[segment])
        lon_select_avg.append(np.nanmean(lon_selected_all[segment]))
        lat_select_avg.append(np.nanmean(lat_selected_all[segment]))
        alt_select_avg.append(np.nanmean(alt_selected_all[segment]))
        time_select_avg.append(np.nanmean(time_selected_all[segment]))
        alb_select_avg.append(np.nanmean(alb_selected_all[segment, :], axis=0))
        alb_select_std.append(np.nanstd(alb_selected_all[segment, :], axis=0))
        broadband_alb_select_avg.append(np.nanmean(broadband_alb_selected_all[segment]))
        broadband_alb_select_std.append(np.nanstd(broadband_alb_selected_all[segment]))
    
    lon_select_avg = np.array(lon_select_avg)
    lat_select_avg = np.array(lat_select_avg)
    alt_select_avg = np.array(alt_select_avg)
    time_select_avg = np.array(time_select_avg)
    alb_select_avg = np.array(alb_select_avg)
    alb_select_std = np.array(alb_select_std)
    broadband_alb_select_avg = np.array(broadband_alb_select_avg)
    broadband_alb_select_std = np.array(broadband_alb_select_std)
    
    ### Plot flight tracks
    cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    
    eff_alb_ = gas_abs_masking(alb_wvl, np.ones_like(alb_wvl), alt=np.nanmean(alt_selected_all))
    
    fig = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(2, 7, left=0.05, right=0.95, wspace=0.8, hspace=0.3)
    ax1 = fig.add_subplot(gs1[:, :3], projection=cartopy_proj)
    ax2 = fig.add_subplot(gs1[0, 3:])
    ax3 = fig.add_subplot(gs1[1, 3:])
    
    # Set the extent for the main axes
    lon_min = np.min(lon_selected_all)
    lon_max = np.max(lon_selected_all)
    lat_min = np.min(lat_selected_all)
    lat_max = np.max(lat_selected_all)
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 3.5
    lat_buffer = (lat_max - lat_min) * 3.5
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # features
    ax1.coastlines(linewidth=0.5, color='black')
    ax1.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax1.add_feature(
        cfeature.OCEAN.with_scale('10m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 5.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 1.0))
    g1.top_labels = False
    
    # Create a ScalarMappable
    data_min, data_max = np.arange(len(alt_select_avg)).min(), np.arange(len(alt_select_avg)).max()
    norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
    cmap = cm.jet # Or any other built-in colormap like cm.viridis
    s_m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_series = s_m.to_rgba(np.arange(len(alt_select_avg)))
    
    

    for i in range(len(lon_select)):
        alt_avg = alt_select_avg[i]
        ax1.plot(lon_select[i], lat_select[i], transform=ccrs.PlateCarree(),
                 label=f'{alt_avg:.1f} km', c=color_series[i], linewidth=2, zorder=2)

    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax1.legend(fontsize=14)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')
    ax1.set_xlabel('Longitude', fontsize=14)
    ax1.set_ylabel('Latitude', fontsize=14)
    
    for i in range(len(alb_select_avg)):
        alt_avg = alt_select_avg[i]
        ax2.plot(alb_wvl, alb_select_avg[i, :], label=f'{alt_avg:.1f} km', color=color_series[i])
        ax2.fill_between(alb_wvl, alb_select_avg[i, :]-alb_select_std[i, :], alb_select_avg[i, :]+alb_select_std[i, :], 
                        color=color_series[i], alpha=0.1)
        
        ax3.scatter(broadband_alb_select[i], alt_select[i], c=color_series[i], s=15, alpha=0.8)
        
        
    
    ax2.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(eff_alb_), color='gray', alpha=0.2,)#  label='Mask Gas absorption bands')
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    # ax2.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, -0.1))
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.hlines(0, 350, 2000, colors='k', linestyles='dashed', linewidth=1)
    ax2.set_xlim(350, 2000)
    

    ax3.set_ylabel('Altitude (km)', fontsize=14)
    ax3.set_xlabel('Broadband Albedo', fontsize=14)
    # ax3.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax3.tick_params(labelsize=12)
    
    for ax, ax_num in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
        ax.text(0.01, 1.05, f'({ax_num})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='left')
        ax.tick_params(labelsize=12, which='both')
    
    
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0730_clear_spiral_summary.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    sys.exit()
    
    #"""
    
    """
    # 7/30 clear_atm_corr 
    date_select = '20240730'
    date_summer_mask = combined_data['dates_summer_all'] == int(date_select) if date_select > '20240630' else combined_data['dates_spring_all'] == int(date_select)
    case_tag_select_1 = 'clear_atm_corr'
    case_tag_mask_1 = np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_1 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_select_2 = 'clear_atm_corr_2'
    case_tag_mask_2 = np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_spring_all']]) if date_select <= '20240630' else np.array([case_tag_select_2 in ct for ct in combined_data['case_tags_summer_all']])
    case_tag_mask = case_tag_mask_1 | case_tag_mask_2
    final_mask = date_summer_mask & case_tag_mask
    alb_wvl = combined_data['wvl_summer'] if date_select > '20240630' else combined_data['wvl_spring']
    lon_selected_all = combined_data['lon_all_summer'][final_mask] if date_select > '20240630' else combined_data['lon_all_spring'][final_mask]
    lat_selected_all = combined_data['lat_all_summer'][final_mask] if date_select > '20240630' else combined_data['lat_all_spring'][final_mask]
    alt_selected_all = combined_data['alt_all_summer'][final_mask] if date_select > '20240630' else combined_data['alt_all_spring'][final_mask]
    time_selected_all = combined_data['time_summer_all'][final_mask] if date_select > '20240630' else combined_data['time_spring_all'][final_mask]
    alb_selected_all = combined_data['alb_iter2_all_summer'][final_mask, :] if date_select > '20240630' else combined_data['alb_iter2_all_spring'][final_mask, :]
    broadband_alb_selected_all = combined_data['broadband_alb_iter2_all_filter_summer'][final_mask] if date_select > '20240630' else combined_data['broadband_alb_iter2_all_filter_spring'][final_mask]
    
    
    time_mask = (time_selected_all >= 14.726) & (time_selected_all <= 14.936)
    lon_selected_all = lon_selected_all[time_mask]
    lat_selected_all = lat_selected_all[time_mask]
    alt_selected_all = alt_selected_all[time_mask]
    alb_selected_all = alb_selected_all[time_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[time_mask]
    

    alb_selected_all_avg = np.nanmean(alb_selected_all, axis=0)
    alb_selected_all_std = np.nanstd(alb_selected_all, axis=0)
    alt_selected_all_avg = np.nanmean(alt_selected_all)
    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km', color='b')
    ax.fill_between(alb_wvl, alb_selected_all_avg-alb_selected_all_std, alb_selected_all_avg+alb_selected_all_std, 
                    color='b', alpha=0.1)
    ax.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km', color='r')
    ax.fill_between(alb_wvl, alb_selected_all_avg-alb_selected_all_std, alb_selected_all_avg+alb_selected_all_std, 
                    color='r', alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), July 30', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0730_clear_avg.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lat_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), July 30', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0730_clear_broadband_lat.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lon_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), July 30', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0730_clear_broadband_lon.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    

    
    ### Plot flight tracks
    
    # cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    cartopy_proj = ccrs.NorthPolarStereo(central_longitude=np.nanmean(lon_selected_all))
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': cartopy_proj})

    # Set the extent for the main axes
    lon_min = np.nanmin(lon_selected_all)
    lon_max = np.nanmax(lon_selected_all)
    lat_min = np.nanmin(lat_selected_all)
    lat_max = np.nanmax(lat_selected_all)
    
    print("ori lon extent:", lon_min, lon_max)
    print("ori lat extent:", lat_min, lat_max)
    
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 2.
    lat_buffer = (lat_max - lat_min) * 15.
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    print("lon extent:", lon_min, lon_max)
    print("lat extent:", lat_min, lat_max)
    
    # features
    ax.coastlines(linewidth=0.5, color='black')
    ax.add_feature(
        cfeature.LAND.with_scale('50m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax.add_feature(
        cfeature.OCEAN.with_scale('50m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 15.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 5.0))
    g1.top_labels = False


    ax.scatter(lon_selected_all, lat_selected_all, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=5, zorder=3)
    
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')

    plt.savefig(f'{fig_dir}/arcsix_flight_paths_0730_clear.png', dpi=300, bbox_inches='tight')
    #"""
    

    
    
    for date in sorted((set(dates_spring_all))):
        date_mask = np.array([d == date for d in dates_spring_all])
        # set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
        plt.close('all')
        central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
        proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(1, 1, 1, projection=proj)

        # add coastlines and land features for context
        ax.coastlines(resolution='50m', linewidth=0.8)
        ax.add_feature(cartopy.feature.LAND, facecolor='#f0f0f0', zorder=0)
        ax.add_feature(cartopy.feature.OCEAN, facecolor='#e6f2ff', zorder=0)

        # gridlines with labels (only lon/lat labels make sense in PlateCarree transform)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # scatter the flight-leg centers (and all points) using PlateCarree transform
        # color by broadband surface albedo from iteration 2 if available
        color_vals = np.array(broadband_alb_iter2_all_spring[date_mask]) if 'broadband_alb_iter2_spring' in locals() else None
        if color_vals is None or np.all(np.isnan(color_vals)):
            sc = ax.scatter(lon_all_spring[date_mask], lat_all_spring[date_mask], s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
        else:
            sc = ax.scatter(lon_all_spring[date_mask], lat_all_spring[date_mask], s=5, c=color_vals, cmap='jet',
                            transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
            cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

        # also plot all sampled points along legs (optional, lighter marker)
        if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
            ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

        ax.set_title(f'Polar projection (North) - Spring Flight {date}', fontsize=12)
        # set a reasonable display extent around the data if available (lon/lat box)
        if len(lon_avg_all) > 0 and len(lat_avg_all) > 0:
            lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
            lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
            # expand a bit
            pad_lon = max(0.5, (lon_max - lon_min) * 0.05)
            pad_lat = max(0.5, (lat_max - lat_min) * 0.05)
            try:
                ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
            except Exception:
                # fallback: don't set extent if projection complains
                pass
        ax.tick_params('both', labelsize=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_spring_30s_{str(date)}.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    for date in sorted(set(dates_summer_all)):
        date_mask = np.array([d == date for d in dates_summer_all])
        # set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
        plt.close('all')
        central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
        proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(1, 1, 1, projection=proj)

        # add coastlines and land features for context
        ax.coastlines(resolution='50m', linewidth=0.8)
        ax.add_feature(cartopy.feature.LAND, facecolor='#f0f0f0', zorder=0)
        ax.add_feature(cartopy.feature.OCEAN, facecolor='#e6f2ff', zorder=0)

        # gridlines with labels (only lon/lat labels make sense in PlateCarree transform)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # scatter the flight-leg centers (and all points) using PlateCarree transform
        # color by broadband surface albedo from iteration 2 if available
        color_vals = np.array(broadband_alb_iter2_all_summer[date_mask]) if 'broadband_alb_iter2_summer' in locals() else None
        if color_vals is None or np.all(np.isnan(color_vals)):
            sc = ax.scatter(lon_all_summer[date_mask], lat_all_summer[date_mask], s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
        else:
            sc = ax.scatter(lon_all_summer[date_mask], lat_all_summer[date_mask], s=5, c=color_vals, cmap='jet',
                            transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
            cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

        # also plot all sampled points along legs (optional, lighter marker)
        if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
            ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

        ax.set_title(f'Polar projection (North) - Summer Flight {date}', fontsize=12)
        # set a reasonable display extent around the data if available (lon/lat box)
        if len(lon_avg_all) > 0 and len(lat_avg_all) > 0:
            lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
            lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
            # expand a bit
            pad_lon = max(0.5, (lon_max - lon_min) * 0.05)
            pad_lat = max(0.5, (lat_max - lat_min) * 0.05)
            try:
                ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
            except Exception:
                # fallback: don't set extent if projection complains
                pass
        ax.tick_params('both', labelsize=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_summer_30s_{str(date)}.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    # set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
    plt.close('all')
    central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=proj)

    # add coastlines and land features for context
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='#e6f2ff', zorder=0)

    # gridlines with labels (only lon/lat labels make sense in PlateCarree transform)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # scatter the flight-leg centers (and all points) using PlateCarree transform
    # color by broadband surface albedo from iteration 2 if available
    color_vals = np.array(broadband_alb_iter2_spring) if 'broadband_alb_iter2_spring' in locals() else None
    if color_vals is None or np.all(np.isnan(color_vals)):
        sc = ax.scatter(lon_avg_spring, lat_avg_spring, s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
    else:
        sc = ax.scatter(lon_avg_spring, lat_avg_spring, s=5, c=color_vals, cmap='jet',
                        transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
        cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

    # also plot all sampled points along legs (optional, lighter marker)
    if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
        ax.scatter(lon_avg_all, lat_avg_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title(f'Polar projection (North) - Spring Flight legs', fontsize=12)
    # set a reasonable display extent around the data if available (lon/lat box)
    if len(lon_avg_all) > 0 and len(lat_avg_all) > 0:
        lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
        lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
        # expand a bit
        pad_lon = max(0.5, (lon_max - lon_min) * 0.05)
        pad_lat = max(0.5, (lat_max - lat_min) * 0.05)
        try:
            ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        except Exception:
            # fallback: don't set extent if projection complains
            pass
    ax.tick_params('both', labelsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_spring_30s.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


    plt.close('all')
    central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=proj)

    # add coastlines and land features for context
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='#e6f2ff', zorder=0)

    # gridlines with labels (only lon/lat labels make sense in PlateCarree transform)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # scatter the flight-leg centers (and all points) using PlateCarree transform
    # color by broadband surface albedo from iteration 2 if available
    color_vals = np.array(broadband_alb_iter2_summer) if 'broadband_alb_iter2_summer' in locals() else None
    if color_vals is None or np.all(np.isnan(color_vals)):
        sc = ax.scatter(lon_avg_summer, lat_avg_summer, s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
    else:
        sc = ax.scatter(lon_avg_summer, lat_avg_summer, s=5, c=color_vals, cmap='jet',
                        transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
        cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

    # also plot all sampled points along legs (optional, lighter marker)
    if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
        ax.scatter(lon_avg_all, lat_avg_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title(f'Polar projection (North) - Summer Flight legs', fontsize=12)
    # set a reasonable display extent around the data if available (lon/lat box)
    if len(lon_avg_all) > 0 and len(lat_avg_all) > 0:
        lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
        lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
        # expand a bit
        pad_lon = max(0.2, (lon_max - lon_min) * 0.05)
        pad_lat = max(0.2, (lat_max - lat_min) * 0.05)
        try:
            ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        except Exception:
            # fallback: don't set extent if projection complains
            pass
    ax.tick_params('both', labelsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_summer_30s.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    # set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
    plt.close('all')
    central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=proj)

    # add coastlines and land features for context
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='#e6f2ff', zorder=0)

    # gridlines with labels (only lon/lat labels make sense in PlateCarree transform)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # scatter the flight-leg centers (and all points) using PlateCarree transform
    # color by broadband surface albedo from iteration 2 if available
    color_vals = np.array(broadband_alb_iter2_all_spring) if 'broadband_alb_iter2_all_spring' in locals() else None
    if color_vals is None or np.all(np.isnan(color_vals)):
        sc = ax.scatter(lon_all_spring, lat_all_spring, s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
    else:
        sc = ax.scatter(lon_all_spring, lat_all_spring, s=5, c=color_vals, cmap='jet',
                        transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
        cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

    # also plot all sampled points along legs (optional, lighter marker)
    if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
        ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title(f'Polar projection (North) - Spring Flight legs', fontsize=12)
    # set a reasonable display extent around the data if available (lon/lat box)
    if len(lon_all) > 0 and len(lat_all) > 0:
        lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
        lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
        # expand a bit
        pad_lon = max(0.5, (lon_max - lon_min) * 0.05)
        pad_lat = max(0.5, (lat_max - lat_min) * 0.05)
        try:
            ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        except Exception:
            # fallback: don't set extent if projection complains
            pass
    ax.tick_params('both', labelsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_spring.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


    plt.close('all')
    central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=proj)

    # add coastlines and land features for context
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='#e6f2ff', zorder=0)

    # gridlines with labels (only lon/lat labels make sense in PlateCarree transform)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # scatter the flight-leg centers (and all points) using PlateCarree transform
    # color by broadband surface albedo from iteration 2 if available
    color_vals = np.array(broadband_alb_iter2_all_summer) if 'broadband_alb_iter2_all_summer' in locals() else None
    if color_vals is None or np.all(np.isnan(color_vals)):
        sc = ax.scatter(lon_all_summer, lat_all_summer, s=5, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1, label='Flight legs')
    else:
        sc = ax.scatter(lon_all_summer, lat_all_summer, s=5, c=color_vals, cmap='jet',
                        transform=ccrs.PlateCarree(), zorder=3, edgecolor=None, vmin=0.1, vmax=1)
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
        cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

    # also plot all sampled points along legs (optional, lighter marker)
    if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
        ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title(f'Polar projection (North) - Summer Flight legs', fontsize=12)
    # set a reasonable display extent around the data if available (lon/lat box)
    if len(lon_all) > 0 and len(lat_all) > 0:
        lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
        lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
        # expand a bit
        pad_lon = max(0.2, (lon_max - lon_min) * 0.05)
        pad_lat = max(0.2, (lat_max - lat_min) * 0.05)
        try:
            ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        except Exception:
            # fallback: don't set extent if projection complains
            pass
    ax.tick_params('both', labelsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_broadband_albedo_vs_longitude_polar_projection_summer.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

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
    # ------------------------------------------


    combined_atm_corr()