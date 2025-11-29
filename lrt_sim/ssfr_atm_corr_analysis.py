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
import cartopy.feature as cfeature
import bisect
import pandas as pd
import xarray as xr
from collections import defaultdict
import gc
from pyproj import Transformer
from util import *
# mpl.use('Agg')


import er3t


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

 
def ssfr_alb_plot(date_s, tmhr_ranges_select, wvl, alb, color_series, 
                   alt_avg_all,
                   modis_bands_nm, modis_alb_legs, modis_alb_file,
                   case_tag,
                   ylabel='SSFR upward/downward ratio',
                   title='SSFR measurement',
                   suptitle='SSFR upward/downward ratio Comparison',
                   file_description='', 
                   lon_avg_all=None,
                   lat_avg_all=None,
                   aviris_file=None,
                   aviris_closest=False,
                   aviris_reflectance_wvl=None,
                   aviris_reflectance_spectrum=None,
                   aviris_reflectance_spectrum_unc=None
                   ):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if aviris_file is not None:
        aviris_label = 'AVIRIS Reflectance' if ((aviris_file is not None) and (aviris_closest)) else 'All AVIRIS mean'
        ax.scatter(aviris_reflectance_wvl, aviris_reflectance_spectrum, s=5, c='m', label=aviris_label, alpha=0.7) if ((aviris_file is not None)) else None
        ax.fill_between(aviris_reflectance_wvl, aviris_reflectance_spectrum-aviris_reflectance_spectrum_unc, aviris_reflectance_spectrum+aviris_reflectance_spectrum_unc, color='m', alpha=0.3) if aviris_file is not None else None
        if modis_alb_file is not None:
            ax.scatter(modis_bands_nm, modis_alb_legs, s=50, c='g', marker='*', label='MODIS Albedo', edgecolors='k')

    for i in range(len(tmhr_ranges_select)):
        if lon_avg_all is not None and lat_avg_all is not None:
            ax.plot(wvl, alb[i, :], '-', color=color_series[i], label='Z=%.2fkm, lon:%.2f, lat: %.2f' % (alt_avg_all[i], lon_avg_all[i], lat_avg_all[i]))
        else:
            ax.plot(wvl, alb[i, :], '-', color=color_series[i], label='Z=%.2fkm' % (alt_avg_all[i]))
        if aviris_file is None and modis_alb_file is not None:
            ax.scatter(modis_bands_nm, modis_alb_legs[i], s=50, color=color_series[i], marker='*', edgecolors='k')
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=13)
    fig.suptitle(suptitle, fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_%s_comparison.png' % (date_s, date_s, case_tag, file_description), bbox_inches='tight', dpi=150)

 

def ssfr_up_dn_ratio_plot(date_s, tmhr_ranges_select, wvl, up_dn_ratio, color_series, alt_avg_all, case_tag,
                          albedo_used='albedo used: SSFR upward/downward ratio',
                          file_suffix=''):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(tmhr_ranges_select)):
        ax.plot(wvl, up_dn_ratio[i, :], '-', color=color_series[i], label='Z=%.2fkm' % (alt_avg_all[i]))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('SSFR upward/downward ratio', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(albedo_used, fontsize=13)
    fig.suptitle(f'P3 level upward/downward ratio Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_up_dn_ratio_comparison%s.png' % (date_s, date_s, case_tag, file_suffix), bbox_inches='tight', dpi=150)

def combined_atm_corr():
    log = logging.getLogger("atm corr combined")

    output_dir = f'{_fdir_general_}/sfc_alb_combined'
    

    # output_all_dict = {
    #     'wvl_spring': wvl_spring,
    #     'wvl_summer': wvl_summer,
    #     'time_all': time_all,
    #     'lon_all_spring': lon_all_spring,
    #     'lat_all_spring': lat_all_spring,
    #     'alt_all_spring': alt_all_spring,
    #     'dates_spring_all': dates_spring_all,
    #     'leg_contidions_all_spring': leg_contidions_all_spring,
    #     'case_tags_spring': case_tags_spring,
    #     'fdn_all_spring': fdn_all_spring,
    #     'fup_all_spring': fup_all_spring,
    #     'toa_expand_all_spring': toa_expand_all_spring,
    #     'icing_all_spring': icing_all_spring,
    #     'icing_pre_all_spring': icing_pre_all_spring,
    #     'alb_iter1_all_spring': alb_iter1_all_spring,
    #     'alb_iter2_all_spring': alb_iter2_all_spring,
    #     'broadband_alb_iter1_all_spring': broadband_alb_iter1_all_spring,
    #     'broadband_alb_iter2_all_spring': broadband_alb_iter2_all_spring,
    #     'broadband_alb_iter1_all_filter_spring': broadband_alb_iter1_all_filter_spring,
    #     'broadband_alb_iter2_all_filter_spring': broadband_alb_iter2_all_filter_spring,
    #     'time_summer': time_summer,
    #     'lon_all_summer': lon_all_summer,
    #     'lat_all_summer': lat_all_summer,
    #     'alt_all_summer': alt_all_summer,
    #     'dates_summer_all': dates_summer_all,
    #     'leg_contidions_all_summer': leg_contidions_all_summer,
    #     'case_tags_summer': case_tags_summer,
    #     'fdn_all_summer': fdn_all_summer,
    #     'fup_all_summer': fup_all_summer,
    #     'toa_expand_all_summer': toa_expand_all_summer,
    #     'icing_all_summer': icing_all_summer,
    #     'icing_pre_all_summer': icing_pre_all_summer,
    #     'alb_iter1_all_summer': alb_iter1_all_summer,
    #     'alb_iter2_all_summer': alb_iter2_all_summer,
    #     'broadband_alb_iter1_all_summer': broadband_alb_iter1_all_summer,
    #     'broadband_alb_iter2_all_summer': broadband_alb_iter2_all_summer,
    #     'broadband_alb_iter1_all_filter_summer': broadband_alb_iter1_all_filter_summer,
    #     'broadband_alb_iter2_all_filter_summer': broadband_alb_iter2_all_filter_summer,
    # }
    
    combined_output_file = f'{output_dir}/sfc_alb_combined_spring_summer.pkl'
    with open(combined_output_file, 'rb') as r:
        combined_data = pickle.load(r)
        
    fig_dir = f'fig/sfc_alb_corr_analysis'
    os.makedirs(fig_dir, exist_ok=True)
    
    """ 
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
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alb_wvl, alb_selected_high_alt_avg, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', color='b')
    ax.fill_between(alb_wvl, alb_selected_high_alt_avg-alb_selected_high_alt_std, alb_selected_high_alt_avg+alb_selected_high_alt_std, 
                    color='b', alpha=0.1)
    ax.plot(alb_wvl, alb_selected_low_alt_avg, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', color='r')
    ax.fill_between(alb_wvl, alb_selected_low_alt_avg-alb_selected_low_alt_std, alb_selected_low_alt_avg+alb_selected_low_alt_std, 
                    color='r', alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), July 29', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0729_clear_1_avg.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lat_selected_high_alt, broadband_alb_selected_all_high_alt, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=10)
    ax.scatter(lat_selected_low_alt, broadband_alb_selected_all_low_alt, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), July 29', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0729_clear_1_broadband_lat.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    ### Plot flight tracks
    
    cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': cartopy_proj})

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
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # features
    ax.coastlines(linewidth=0.5, color='black')
    ax.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax.add_feature(
        cfeature.OCEAN.with_scale('10m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 15.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 5.0))
    g1.top_labels = False


    ax.scatter(lon_selected_high_alt, lat_selected_high_alt, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=5, zorder=3)
    
    ax.scatter(lon_selected_low_alt, lat_selected_low_alt, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=7.5, zorder=2)
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')

    plt.savefig(f'{fig_dir}/arcsix_flight_paths_0729_clear_1.png', dpi=300, bbox_inches='tight')
    #"""
    
    """# 6/5 clear_atm_corr_2, 3
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
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alb_wvl, alb_selected_high_alt_avg, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', color='b')
    ax.fill_between(alb_wvl, alb_selected_high_alt_avg-alb_selected_high_alt_std, alb_selected_high_alt_avg+alb_selected_high_alt_std, 
                    color='b', alpha=0.1)
    ax.plot(alb_wvl, alb_selected_low_alt_avg, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', color='r')
    ax.fill_between(alb_wvl, alb_selected_low_alt_avg-alb_selected_low_alt_std, alb_selected_low_alt_avg+alb_selected_low_alt_std, 
                    color='r', alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), June 5th', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0605_clear_23_avg.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lat_selected_high_alt, broadband_alb_selected_all_high_alt, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=10)
    ax.scatter(lat_selected_low_alt, broadband_alb_selected_all_low_alt, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), June 5th', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0605_clear_23_broadband_lat.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    ### Plot flight tracks
    
    # cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_selected_all), central_latitude=np.nanmean(lat_selected_all))
    cartopy_proj = ccrs.NorthPolarStereo(central_longitude=np.nanmean(lon_selected_all))
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': cartopy_proj})

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
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

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


    ax.scatter(lon_selected_high_alt, lat_selected_high_alt, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=5, zorder=3)
    
    ax.scatter(lon_selected_low_alt, lat_selected_low_alt, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=7.5, zorder=2)
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')

    plt.savefig(f'{fig_dir}/arcsix_flight_paths_0605_clear_23.png', dpi=300, bbox_inches='tight')
    """
    
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
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alb_wvl, alb_selected_high_alt_avg, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', color='b')
    ax.fill_between(alb_wvl, alb_selected_high_alt_avg-alb_selected_high_alt_std, alb_selected_high_alt_avg+alb_selected_high_alt_std, 
                    color='b', alpha=0.1)
    ax.plot(alb_wvl, alb_selected_low_alt_avg, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', color='r')
    ax.fill_between(alb_wvl, alb_selected_low_alt_avg-alb_selected_low_alt_std, alb_selected_low_alt_avg+alb_selected_low_alt_std, 
                    color='r', alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 15', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0815_clear_12_avg.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lat_selected_high_alt, broadband_alb_selected_all_high_alt, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=10)
    ax.scatter(lat_selected_low_alt, broadband_alb_selected_all_low_alt, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 15', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0815_clear_12_broadband_lat.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lon_selected_high_alt, lat_selected_high_alt, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=10)
    ax.scatter(lon_selected_low_alt, lat_selected_low_alt, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=10)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # # ax.set_ylim(-0.05, 1.05)
    # ax.set_title('Surface Albedo (atm corr + fit), Aug 15', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0815_clear_12_lonlat.png', bbox_inches='tight', dpi=150)
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
    lon_buffer = (lon_max - lon_min) * 0.5
    lat_buffer = (lat_max - lat_min) * 3
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


    ax.scatter(lon_selected_high_alt, lat_selected_high_alt, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=5, zorder=3)
    
    ax.scatter(lon_selected_low_alt, lat_selected_low_alt, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=7.5, zorder=2)
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')

    plt.savefig(f'{fig_dir}/arcsix_flight_paths_0815_clear_12.png', dpi=300, bbox_inches='tight')
    """
    
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
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alb_wvl, alb_selected_high_alt_avg, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', color='b')
    ax.fill_between(alb_wvl, alb_selected_high_alt_avg-alb_selected_high_alt_std, alb_selected_high_alt_avg+alb_selected_high_alt_std, 
                    color='b', alpha=0.1)
    ax.plot(alb_wvl, alb_selected_low_alt_avg, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', color='r')
    ax.fill_between(alb_wvl, alb_selected_low_alt_avg-alb_selected_low_alt_std, alb_selected_low_alt_avg+alb_selected_low_alt_std, 
                    color='r', alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.legend(fontsize=10,)#loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 15', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0815_clear_23_avg.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lat_selected_high_alt, broadband_alb_selected_all_high_alt, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=10)
    ax.scatter(lat_selected_low_alt, broadband_alb_selected_all_low_alt, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 15', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0815_clear_23_broadband_lat.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lon_selected_high_alt, broadband_alb_selected_all_high_alt, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=10)
    ax.scatter(lon_selected_low_alt, broadband_alb_selected_all_low_alt, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 15', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0815_clear_23_broadband_lon.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lon_selected_high_alt, lat_selected_high_alt, label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=10)
    ax.scatter(lon_selected_low_alt, lat_selected_low_alt, label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=10)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # # ax.set_ylim(-0.05, 1.05)
    # ax.set_title('Surface Albedo (atm corr + fit), Aug 15', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0815_clear_23_lonlat.png', bbox_inches='tight', dpi=150)
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


    ax.scatter(lon_selected_high_alt, lat_selected_high_alt, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_high_alt_avg:.1f}km', c='b', s=5, zorder=3)
    
    ax.scatter(lon_selected_low_alt, lat_selected_low_alt, transform=ccrs.PlateCarree(),
               label=f'Alt: {alt_selected_low_alt_avg:.1f}km', c='r', s=7.5, zorder=2)
    
    # leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg = ax.legend(fontsize=10)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')

    plt.savefig(f'{fig_dir}/arcsix_flight_paths_0815_clear_23.png', dpi=300, bbox_inches='tight')
    #"""
    
    
    """# 7/30 clear_atm_corr 
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
    

    # 8/1 clear_atm_corr 
    date_select = '20240801'
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
    
    
    time_mask = (time_selected_all >= 13.84) & (time_selected_all <= 14.12)
    time_selected_all = time_selected_all[time_mask]
    lon_selected_all = lon_selected_all[time_mask]
    lat_selected_all = lat_selected_all[time_mask]
    alt_selected_all = alt_selected_all[time_mask]
    alb_selected_all = alb_selected_all[time_mask, :]
    broadband_alb_selected_all = broadband_alb_selected_all[time_mask]
    

    alb_selected_all_avg = np.nanmean(alb_selected_all, axis=0)
    alb_selected_all_std = np.nanstd(alb_selected_all, axis=0)
    alt_selected_all_avg = np.nanmean(alt_selected_all)
    
    
    alb_ext_wvl, alb_ext = alb_extention(alb_wvl, alb_selected_all_avg, clear_sky=True)
    plt.close('all')
    plt.plot(alb_wvl, alb_selected_all_avg, label=f'Alt: {alt_selected_all_avg:.1f}km')
    plt.plot(alb_ext_wvl, alb_ext, label='Extended')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Surface Albedo')
    plt.title('Surface Albedo Extension Check, Aug 1st')
    plt.legend()
    plt.show()
    sys.exit()    
    
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
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_avg.png', bbox_inches='tight', dpi=150)
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
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_broadband_lat.png', bbox_inches='tight', dpi=150)
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
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_broadband_lon.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(time_selected_all, broadband_alb_selected_all, label=f'Alt: {alt_selected_all_avg:.1f}km', c='b', s=10)
    # for band in gas_bands:
    #     ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Broadband Albedo', fontsize=14)
    ax.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title('Surface Albedo (atm corr + fit), Aug 1st', fontsize=13)
    # ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_0801_clear_broadband_time.png', bbox_inches='tight', dpi=150)
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
    lon_buffer = (lon_max - lon_min) * 3.
    lat_buffer = (lat_max - lat_min) * 1.
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

    plt.savefig(f'{fig_dir}/arcsix_flight_paths_0801_clear.png', dpi=300, bbox_inches='tight')
    
    
    sys.exit()
    
    
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