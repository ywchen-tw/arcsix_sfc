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
from scipy import interpolate
# mpl.use('Agg')


import er3t


_mission_      = 'arcsix'


legend_size = 12
label_size = 14


_h = 6.626e-34
_c = 3.0e+8
_k = 1.38e-23

def planck(wvl_nm, T):
    """Calculate the Planck function for a given wavelength and temperature.
    Args:
        wav (float or float array): Wavelength in nm.
        T (float or float array): Temperature in Kelvin.
    Returns:
        float or float array: Irradiance in W/m^2/m.
    """
    wvl_m = wvl_nm * 1e-9  # convert nm to m
    a = 2.0*np.pi*_h*_c**2
    b = _h*_c/(wvl_m*_k*T)
    irradiance_per_m = a/ ( (wvl_m**5) * (np.exp(b) - 1.0) )
    irradiance_per_nm = irradiance_per_m * 1e-9  # convert from m to nm
    return irradiance_per_nm



def flt_trk_lrt_para_2lay(date='20240611',
                          separate_height=1000.,
                          manual_cloud=False,
                          plot_interval=100,
                          case_tag='default',
                          overwrite=False
                            ):

    # read lrt results
    if manual_cloud:
        cloud_tag = 'manual_cloud'
    else:
        cloud_tag = 'sat_cloud'
    
    lrt_fname = f'flt_trk_lrt_para-lrt-{date}-{case_tag}-{cloud_tag}.h5'
    
    with h5py.File(lrt_fname, 'r') as f:
        lon = f['lon'][...]
        lat = f['lon'][...]
        alt = f['alt'][...]*1000  # convert km to m
        sza = f['sza'][...]
        cth = f['cth'][...]*1000  # convert km to m
        cbh = f['cbh'][...]*1000  # convert km to m
        cot = f['cot'][...]
        f_down = f['f_down'][...]
        f_down_dir = f['f_down_dir'][...]
        f_down_diff = f['f_down_diff'][...]
        f_up = f['f_up'][...]
    
        ssfr_zen = f['ssfr_zen'][...]
        ssfr_nad = f['ssfr_nad'][...]
        ssfr_zen_wvl = f['ssfr_zen_wvl'][...]
        ssfr_nad_wvl = f['ssfr_nad_wvl'][...]
        ssfr_toa0 = f['ssfr_toa0'][...]
        
        hsr1_wvl = f['hsr1_wvl'][...]
        hsr1_toa0 = f['hsr1_toa0'][...]
        hsr1_total = f['hsr1_total'][...]
        hsr1_diff = f['hsr1_diff'][...]
        
    hsr1_direct_ratio = np.ones_like(hsr1_total) - (hsr1_diff / hsr1_total)
        
    with h5py.File(f'flt_trk_lrt_para-lrt-{date}-{case_tag}-clear.h5', 'r') as f:
        # lon = f['lon'][...]
        # lat = f['lon'][...]
        # alt = f['alt'][...]*1000  # convert km to m
        f_down_clear = f['f_down'][...]
        f_down_clear_dir = f['f_down_dir'][...]
        f_down_clear_diff = f['f_down_diff'][...]
        f_up_clear = f['f_up'][...]
        
    with open('kurudz_ssfr_1nm_convolution.dat', 'r') as f_solar:
        # skip 2 lines with #
        f_solar_lines = f_solar.readlines()
        f_solar_wvl = np.array([float(line.split()[0]) for line in f_solar_lines[2:]])
        f_solar_flux = np.array([float(line.split()[1]) for line in f_solar_lines[2:]])    
    
    # plot clouds cot, cth
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    cc1 = ax1.scatter(lon, lat, c=cot, s=1, cmap='viridis')
    ax1.set_title("Cloud Optical Thickness (COT)", fontsize=label_size+2)
    ax1.set_xlabel("Longitude", fontsize=label_size)
    ax1.set_ylabel("Latitude", fontsize=label_size)
    ax1.set_xlim(lon.min(), lon.max())
    ax1.set_ylim(lat.min(), lat.max())
    cbar1 = fig.colorbar(cc1, ax=ax1, orientation='vertical', pad=0.02)
    cbar1.set_label("COT", fontsize=label_size) 
    cc2 = ax2.scatter(lon, lat, c=cth/1000., s=1, cmap='viridis')
    ax2.set_title("Cloud Top Height (CTH)", fontsize=label_size+2)
    ax2.set_xlabel("Longitude", fontsize=label_size)
    ax2.set_ylabel("Latitude", fontsize=label_size)
    ax2.set_xlim(lon.min(), lon.max())
    ax2.set_ylim(lat.min(), lat.max())
    cbar2 = fig.colorbar(cc2, ax=ax2, orientation='vertical', pad=0.02)
    cbar2.set_label("CTH (km)", fontsize=label_size) 
    fig.suptitle(f"Clouds on {date}", fontsize=label_size+2, y=0.98)
    fig.tight_layout()
    fig.savefig(f"fig/{date}/clouds_{case_tag}_{cloud_tag}_{date}.png")
    
    
    data_len = f_down.shape[0]
        
    wvl_si = 795
    wvl_in = 1050
    wvl_width = 8
    hsr1_wvl_si_ind_1 = np.argmin(np.abs(hsr1_wvl - (wvl_si-wvl_width/2)))
    hsr1_wvl_si_ind_2 = np.argmin(np.abs(hsr1_wvl - (wvl_si+wvl_width/2)))
    
    hsr1_wvl_in_ind_1 = np.argmin(np.abs(hsr1_wvl - (wvl_in-wvl_width/2)))
    hsr1_wvl_in_ind_2 = np.argmin(np.abs(hsr1_wvl - (wvl_in+wvl_width/2)))
    hsr1_toa0_si = hsr1_toa0[hsr1_wvl_si_ind_1:hsr1_wvl_si_ind_2+1]
    hsr1_toa0_in = hsr1_toa0[hsr1_wvl_in_ind_1:hsr1_wvl_in_ind_2+1]
    hsr1_toa0_ratio = np.nanmean(hsr1_toa0_si) / np.nanmean(hsr1_toa0_in)
    
    f_solar_si_ind_1 = np.argmin(np.abs(f_solar_wvl - (wvl_si-wvl_width/2)))
    f_solar_si_ind_2 = np.argmin(np.abs(f_solar_wvl - (wvl_si+wvl_width/2)))
    f_solar_in_ind_1 = np.argmin(np.abs(f_solar_wvl - (wvl_in-wvl_width/2)))
    f_solar_in_ind_2 = np.argmin(np.abs(f_solar_wvl - (wvl_in+wvl_width/2)))
    f_solar_si = f_solar_flux[f_solar_si_ind_1:f_solar_si_ind_2+1]
    f_solar_in = f_solar_flux[f_solar_in_ind_1:f_solar_in_ind_2+1]
    f_solar_ratio = np.nanmean(f_solar_si) / np.nanmean(f_solar_in)

    
    ssfr_zen_wvl_si_ind_1 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_si-wvl_width/2)))
    ssfr_zen_wvl_si_ind_2 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_si+wvl_width/2)))
    ssfr_zen_wvl_in_ind_1 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_in-wvl_width/2)))
    ssfr_zen_wvl_in_ind_2 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_in+wvl_width/2)))
    
    ssfr_toa0_si = ssfr_toa0[ssfr_zen_wvl_si_ind_1:ssfr_zen_wvl_si_ind_2+1]
    ssfr_toa0_in = ssfr_toa0[ssfr_zen_wvl_in_ind_1:ssfr_zen_wvl_in_ind_2+1]
    
    ssfr_zen_flux_si = ssfr_zen[:, ssfr_zen_wvl_si_ind_1:ssfr_zen_wvl_si_ind_2+1]
    ssfr_zen_flux_in = ssfr_zen[:, ssfr_zen_wvl_in_ind_1:ssfr_zen_wvl_in_ind_2+1]
    ssfr_toa0_ratio = np.nanmean(ssfr_toa0_si) / np.nanmean(ssfr_toa0_in)
    
    
    
    toa0_ratio = f_solar_ratio
    

    ssfr_zen_flux_ratio = np.nanmean(ssfr_zen_flux_si, axis=1) / np.nanmean(ssfr_zen_flux_in, axis=1)
    
    
    ssfr_nad_wvl_si_ind_1 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_si-wvl_width/2)))
    ssfr_nad_wvl_si_ind_2 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_si+wvl_width/2)))
    ssfr_nad_wvl_in_ind_1 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_in-wvl_width/2)))
    ssfr_nad_wvl_in_ind_2 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_in+wvl_width/2)))
    ssfr_nad_flux_si = ssfr_nad[:, ssfr_nad_wvl_si_ind_1:ssfr_nad_wvl_si_ind_2+1]
    ssfr_nad_flux_in = ssfr_nad[:, ssfr_nad_wvl_in_ind_1:ssfr_nad_wvl_in_ind_2+1]
    ssfr_nad_flux_ratio = np.nanmean(ssfr_nad_flux_si, axis=1) / np.nanmean(ssfr_nad_flux_in, axis=1)
    
    ssfr_zen_scaling_factor = ssfr_zen_flux_ratio / toa0_ratio
    ssfr_nad_scaling_factor = ssfr_nad_flux_ratio / toa0_ratio
    
    
    ssfr_nad_avg = np.nanmean(ssfr_nad, axis=0)
    ssfr_zen_avg = np.nanmean(ssfr_zen, axis=0)
    f_down_avg_p3 = np.nanmean(f_down[:, :, 0], axis=0)
    f_up_avg_p3 = np.nanmean(f_up[:, :, 0], axis=0)
    
    ssfr_nad_std = np.nanstd(ssfr_nad, axis=0)
    ssfr_zen_std = np.nanstd(ssfr_zen, axis=0)
    f_down_std_p3 = np.std(f_down[:, :, 0], axis=0)
    f_up_std_p3 = np.std(f_up[:, :, 0], axis=0)

    # repeat ssfr_toa0 for the same shape as ssfr_zen
    ssfr_toa0_expand = np.repeat(ssfr_toa0[np.newaxis, :], ssfr_zen.shape[0], axis=0)
    print("ssfr_toa0_expand.shape:", ssfr_toa0_expand.shape)
    print("ssfr_zen.shape:", ssfr_zen.shape)
    mu_sza = np.cos(np.deg2rad(sza))
    ssfr_toa_cos = ssfr_toa0_expand * mu_sza[:, np.newaxis]
    ssfr_toa_cos_avg = np.nanmean(ssfr_toa_cos, axis=0)
    ssfr_toa_cos_std = np.nanstd(ssfr_toa_cos, axis=0)
    
    f_solar_expand = np.repeat(f_solar_flux[np.newaxis, :], ssfr_zen.shape[0], axis=0)
    f_solar_cos = f_solar_expand * mu_sza[:, np.newaxis]
    f_solar_cos_avg = np.nanmean(f_solar_cos, axis=0)
    f_solar_cos_std = np.nanstd(f_solar_cos, axis=0)
    
    ssfr_zen_interp = np.zeros((ssfr_zen.shape[0], ssfr_nad_wvl.shape[0]))
    ssfr_zen_interp[:, :] = np.nan
    
    ssfr_zen_adjust = np.zeros_like(ssfr_zen_interp)
    ssfr_zen_adjust[:, :] = np.nan
    ssfr_nad_adjust = np.zeros_like(ssfr_zen_interp)
    ssfr_nad_adjust[:, :] = np.nan
    wvl_950_index_left = np.where(ssfr_nad_wvl < 950)[0][-1]
    wvl_950_index_right = np.where(ssfr_nad_wvl >= 950)[0][0]
    

     
    for i in range(ssfr_zen.shape[0]):
        if np.isnan(ssfr_zen[i, :]).all():
            # print(f"Warning: NaN found in ssfr_zen at index {i}, skipping interpolation.")
            continue
        f = interpolate.interp1d(ssfr_zen_wvl, ssfr_zen[i, :], fill_value="extrapolate")
        ssfr_zen_interp[i, :] = f(ssfr_nad_wvl)
    
    ssfr_zen_adjust[:, :wvl_950_index_right] = ssfr_zen_interp[:, :wvl_950_index_right] / ssfr_zen_scaling_factor[:, np.newaxis]
    ssfr_zen_adjust[:, wvl_950_index_right:] = ssfr_zen_interp[:, wvl_950_index_right:]
    
    ssfr_nad_adjust[:, :wvl_950_index_right] = ssfr_nad[:, :wvl_950_index_right] / ssfr_nad_scaling_factor[:, np.newaxis]
    ssfr_nad_adjust[:, wvl_950_index_right:] = ssfr_nad[:, wvl_950_index_right:]
    
    ssfr_zen_wvl_adjust = ssfr_nad_wvl.copy()
        
    
        
    low_alt_mask = np.logical_and(alt < separate_height, cot > 0)
    high_alt_mask = np.logical_and(alt >= separate_height, cot > 0)
    
    low_alt_mask = alt < separate_height
    high_alt_mask = alt >= separate_height
    
    # ssfr_nad_low_avg = np.nanmean(ssfr_nad[low_alt_mask, :], axis=0)
    # ssfr_zen_low_avg = np.nanmean(ssfr_zen[low_alt_mask, :], axis=0)
    # f_down_low_avg_p3 = np.nanmean(f_down[low_alt_mask, :, 0], axis=0)
    # f_up_low_avg_p3 = np.nanmean(f_up[low_alt_mask, :, 0], axis=0)
    # f_down_low_avg_toa = np.nanmean(f_down[low_alt_mask, :, 1], axis=0)
    # f_up_low_avg_toa = np.nanmean(f_up[low_alt_mask, :, 1], axis=0)
    # f_down_clear_low_avg_toa = np.nanmean(f_down_clear[low_alt_mask, :, 1], axis=0)
    # f_up_clear_low_avg_toa = np.nanmean(f_up_clear[low_alt_mask, :, 1], axis=0)
    
    # ssfr_nad_high_avg = np.nanmean(ssfr_nad[high_alt_mask, :], axis=0)
    # ssfr_zen_high_avg = np.nanmean(ssfr_zen[high_alt_mask, :], axis=0)
    # f_down_high_avg_p3 = np.nanmean(f_down[high_alt_mask, :, 0], axis=0)
    # f_up_high_avg_p3 = np.nanmean(f_up[high_alt_mask, :, 0], axis=0)
    # f_down_high_avg_toa = np.nanmean(f_down[high_alt_mask, :, 1], axis=0)
    # f_up_high_avg_toa = np.nanmean(f_up[high_alt_mask, :, 1], axis=0)
    # f_down_clear_high_avg_toa = np.nanmean(f_down_clear[high_alt_mask, :, 1], axis=0)
    # f_up_clear_high_avg_toa = np.nanmean(f_up_clear[high_alt_mask, :, 1], axis=0)
    
    # ssfr_nad_low_std = np.nanstd(ssfr_nad[low_alt_mask, :], axis=0)
    # ssfr_zen_low_std = np.nanstd(ssfr_zen[low_alt_mask, :], axis=0)
    # f_down_low_std_p3 = np.std(f_down[low_alt_mask, :, 0], axis=0)
    # f_up_low_std_p3 = np.std(f_up[low_alt_mask, :, 0], axis=0)
    # f_down_low_std_toa = np.std(f_down[low_alt_mask, :, 1], axis=0)
    # f_up_low_std_toa = np.std(f_up[low_alt_mask, :, 1], axis=0)
    # f_down_clear_low_std_toa = np.std(f_down_clear[low_alt_mask, :, 1], axis=0)
    # f_up_clear_low_std_toa = np.std(f_up_clear[low_alt_mask, :, 1], axis=0)
    
    # ssfr_nad_high_std = np.nanstd(ssfr_nad[high_alt_mask, :], axis=0)
    # ssfr_zen_high_std = np.nanstd(ssfr_zen[high_alt_mask, :], axis=0)
    # f_down_high_std_p3 = np.std(f_down[high_alt_mask, :, 0], axis=0)
    # f_up_high_std_p3 = np.std(f_up[high_alt_mask, :, 0], axis=0)
    # f_down_high_std_toa = np.std(f_down[high_alt_mask, :, 1], axis=0)
    # f_up_high_std_toa = np.std(f_up[high_alt_mask, :, 1], axis=0)
    # f_down_clear_high_std_toa = np.std(f_down_clear[high_alt_mask, :, 1], axis=0)
    # f_up_clear_high_std_toa = np.std(f_up_clear[high_alt_mask, :, 1], axis=0)
    
    ssfr_nad_low_avg = np.nanmean(ssfr_nad_adjust[low_alt_mask, :], axis=0)
    ssfr_zen_low_avg = np.nanmean(ssfr_zen_adjust[low_alt_mask, :], axis=0)
    f_down_low_avg_p3 = np.nanmean(f_down[low_alt_mask, :, 0], axis=0)
    f_up_low_avg_p3 = np.nanmean(f_up[low_alt_mask, :, 0], axis=0)
    f_down_low_avg_toa = np.nanmean(f_down[low_alt_mask, :, 1], axis=0)
    f_up_low_avg_toa = np.nanmean(f_up[low_alt_mask, :, 1], axis=0)
    f_down_clear_low_avg_toa = np.nanmean(f_down_clear[low_alt_mask, :, 1], axis=0)
    f_up_clear_low_avg_toa = np.nanmean(f_up_clear[low_alt_mask, :, 1], axis=0)
    
    ssfr_nad_high_avg = np.nanmean(ssfr_nad_adjust[high_alt_mask, :], axis=0)
    ssfr_zen_high_avg = np.nanmean(ssfr_zen_adjust[high_alt_mask, :], axis=0)
    f_down_high_avg_p3 = np.nanmean(f_down[high_alt_mask, :, 0], axis=0)
    f_up_high_avg_p3 = np.nanmean(f_up[high_alt_mask, :, 0], axis=0)
    f_down_high_avg_toa = np.nanmean(f_down[high_alt_mask, :, 1], axis=0)
    f_up_high_avg_toa = np.nanmean(f_up[high_alt_mask, :, 1], axis=0)
    f_down_clear_high_avg_toa = np.nanmean(f_down_clear[high_alt_mask, :, 1], axis=0)
    f_up_clear_high_avg_toa = np.nanmean(f_up_clear[high_alt_mask, :, 1], axis=0)
    f_down_clear_high_avg_p3 = np.nanmean(f_down_clear[high_alt_mask, :, 0], axis=0)
    
    ssfr_nad_low_std = np.nanstd(ssfr_nad_adjust[low_alt_mask, :], axis=0)
    ssfr_zen_low_std = np.nanstd(ssfr_zen_adjust[low_alt_mask, :], axis=0)
    f_down_low_std_p3 = np.std(f_down[low_alt_mask, :, 0], axis=0)
    f_up_low_std_p3 = np.std(f_up[low_alt_mask, :, 0], axis=0)
    f_down_low_std_toa = np.std(f_down[low_alt_mask, :, 1], axis=0)
    f_up_low_std_toa = np.std(f_up[low_alt_mask, :, 1], axis=0)
    f_down_clear_low_std_toa = np.std(f_down_clear[low_alt_mask, :, 1], axis=0)
    f_up_clear_low_std_toa = np.std(f_up_clear[low_alt_mask, :, 1], axis=0)
    
    ssfr_nad_high_std = np.nanstd(ssfr_nad_adjust[high_alt_mask, :], axis=0)
    ssfr_zen_high_std = np.nanstd(ssfr_zen_adjust[high_alt_mask, :], axis=0)
    f_down_high_std_p3 = np.std(f_down[high_alt_mask, :, 0], axis=0)
    f_up_high_std_p3 = np.std(f_up[high_alt_mask, :, 0], axis=0)
    f_down_high_std_toa = np.std(f_down[high_alt_mask, :, 1], axis=0)
    f_up_high_std_toa = np.std(f_up[high_alt_mask, :, 1], axis=0)
    f_down_clear_high_std_toa = np.std(f_down_clear[high_alt_mask, :, 1], axis=0)
    f_up_clear_high_std_toa = np.std(f_up_clear[high_alt_mask, :, 1], axis=0)
    f_down_clear_high_std_p3 = np.std(f_down_clear[high_alt_mask, :, 0], axis=0)

    
    # calculate the CRE at TOA
    dwvl = 1.0
    F_cloud_low = f_down_low_avg_toa - f_up_low_avg_toa
    F_clear_low = f_down_clear_low_avg_toa - f_up_clear_low_avg_toa
    F_cloud_low_std = np.sqrt(f_up_low_std_toa**2 + f_down_low_std_toa**2)
    F_clear_low_std = np.sqrt(f_up_clear_low_std_toa**2 + f_down_clear_low_std_toa**2)
    F_cre_low = F_cloud_low - F_clear_low
    F_cre_low_std = np.sqrt(F_cloud_low_std**2 + F_clear_low_std**2)
    cre_low = np.sum(F_cre_low * dwvl)
    
    cre_low_std = np.sum(F_cre_low_std * dwvl)
    F_cloud_high = f_down_high_avg_toa - f_up_high_avg_toa
    F_clear_high = f_down_clear_high_avg_toa - f_up_clear_high_avg_toa
    F_cloud_high_std = np.sqrt(f_up_high_std_toa**2 + f_down_high_std_toa**2)
    F_clear_high_std = np.sqrt(f_up_clear_high_std_toa**2 + f_down_clear_high_std_toa**2)
    F_cre_high = F_cloud_high - F_clear_high
    F_cre_high_std = np.sqrt(F_cloud_high_std**2 + F_clear_high_std**2)
    cre_high = np.sum(F_cre_high * dwvl)
    cre_high_std = np.sum(F_cre_high_std * dwvl)

    lrt_sim_wvl = np.arange(360, 1990.1, 1)
    for i in range(data_len)[::plot_interval]:
        fig = plt.figure(figsize=(18, 4.5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.plot(ssfr_nad_wvl, ssfr_nad[i, :], label='SSFR nadir flux')
        ax1.plot(lrt_sim_wvl, f_up[i, :, 0], label='LRT upward flux')
        ax1.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
        ax1.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax1.legend(fontsize=legend_size)
        
        ax2.plot(ssfr_zen_wvl, ssfr_zen[i, :], label='SSFR zen flux')
        ax2.plot(lrt_sim_wvl, f_down[i, :, 0], label='LRT downward flux')
        ax2.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
        ax2.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax2.legend(fontsize=legend_size)
        

        f = interpolate.interp1d(ssfr_zen_wvl, ssfr_zen[i, :], fill_value="extrapolate")
        xnew = ssfr_nad_wvl
        ynew = f(xnew)   # use interpolation function returned by `interp1d`
        
        ax3.plot(ssfr_nad_wvl, ssfr_nad[i, :]/ynew, label='SSFR nadir flux / zenith flux')
        ax3.plot(lrt_sim_wvl, f_up[i, :, 0]/f_down[i, :, 0], label='LRT upward flux / downward flux')
        # ax3.set_ylabel("W m^-2 nm^-1")
        ax3.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax3.legend(fontsize=legend_size)
        ax3.set_ylim(-0.05, 1.35)
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(350, 2000)
        
        fig.suptitle(f"Alt: {alt[i]:.0f} m COT: {cot[i]:.2f} CTH: {cth[i]:.0f} m CBH: {cbh[i]:.0f} m", fontsize=label_size)#, y=0.98)
        fig.tight_layout()    
        fig.savefig(f"fig/{date}/ssfr_{case_tag}_{cloud_tag}_{date}_{i}.png")
        
        # plt.show()
        fig.clear()
        plt.close(fig)
        
    fig = plt.figure(figsize=(4.8, 7.2))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(ssfr_nad_wvl, ssfr_nad_low_avg, color='tab:blue', label='SSFR upward flux avg')
    ax1.plot(lrt_sim_wvl, f_up_low_avg_p3, color='tab:orange', label='LRT upward flux avg')
    ax1.fill_between(ssfr_nad_wvl, ssfr_nad_low_avg-ssfr_nad_low_std, ssfr_nad_low_avg+ssfr_nad_low_std, alpha=0.2, color='tab:blue')
    ax1.fill_between(lrt_sim_wvl, f_up_low_avg_p3-f_up_low_std_p3, f_up_low_avg_p3+f_up_low_std_p3, alpha=0.2, color='tab:orange')
    ax1.set_title("Upward Flux", fontsize=label_size+2)
    
    ax2.plot(ssfr_zen_wvl_adjust, ssfr_zen_low_avg, color='tab:blue', label='SSFR downward flux')
    ax2.plot(lrt_sim_wvl, f_down_low_avg_p3, color='tab:orange', label='LRT downward flux')
    ax2.plot(f_solar_wvl, f_solar_cos_avg, color='tab:gray', label='Solar TOA flux')
    ax2.fill_between(ssfr_zen_wvl_adjust, ssfr_zen_low_avg-ssfr_zen_low_std, ssfr_zen_low_avg+ssfr_zen_low_std, alpha=0.2, color='tab:blue')
    ax2.fill_between(lrt_sim_wvl, f_down_low_avg_p3-f_down_low_std_p3, f_down_low_avg_p3+f_down_low_std_p3, alpha=0.2, color='tab:orange')
    for ax in [ax1, ax2]:
        ax.set_xlim(350, 2000)
        ax.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax.set_ylabel(r"Flux (W m$^{-2}$ nm$^{-1}$)", fontsize=label_size)
        ax.legend(fontsize=legend_size)
    for ax in [ax1, ax2]:
        ax.set_ylim(-0.05, 1.1)
        
    fig.suptitle(f"{date}", fontsize=label_size+2)#, y=0.98
    fig.tight_layout()
    fig.savefig(f"fig/{date}/ssfr_{cloud_tag}_{date}_{case_tag}_avg_poster.png", dpi=300)
    
    # plt.show()
    fig.clear()
    plt.close(fig)
    sys.exit()
    
    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    ax1.plot(ssfr_nad_wvl, ssfr_nad_low_avg, color='tab:blue', label='SSFR upward flux avg')
    ax1.plot(lrt_sim_wvl, f_up_low_avg_p3, color='tab:orange', label='LRT upward flux avg')
    ax1.fill_between(ssfr_nad_wvl, ssfr_nad_low_avg-ssfr_nad_low_std, ssfr_nad_low_avg+ssfr_nad_low_std, alpha=0.2, color='tab:blue')
    ax1.fill_between(lrt_sim_wvl, f_up_low_avg_p3-f_up_low_std_p3, f_up_low_avg_p3+f_up_low_std_p3, alpha=0.2, color='tab:orange')
    ax1.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax1.legend(fontsize=legend_size)
    
    ax2.plot(ssfr_zen_wvl_adjust, ssfr_zen_low_avg, color='tab:blue', label='SSFR downward flux')
    ax2.plot(lrt_sim_wvl, f_down_low_avg_p3, color='tab:orange', label='LRT downward flux')
    ax2.plot(f_solar_wvl, f_solar_cos_avg, color='tab:gray', label='Solar TOA flux')
    ax2.fill_between(ssfr_zen_wvl_adjust, ssfr_zen_low_avg-ssfr_zen_low_std, ssfr_zen_low_avg+ssfr_zen_low_std, alpha=0.2, color='tab:blue')
    ax2.fill_between(lrt_sim_wvl, f_down_low_avg_p3-f_down_low_std_p3, f_down_low_avg_p3+f_down_low_std_p3, alpha=0.2, color='tab:orange')
    ax2.fill_between(f_solar_wvl, f_solar_cos_avg-f_solar_cos_std, f_solar_cos_avg+f_solar_cos_std, alpha=0.2, color='tab:gray')
    # ax2.set
    ax2.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax2.legend(fontsize=legend_size)
    ax2.set_title(f"Low leg ({np.round(np.nanmean(alt[low_alt_mask]), -2):.0f} m)", fontsize=label_size+2)
    

    f = interpolate.interp1d(ssfr_zen_wvl_adjust, ssfr_zen_low_avg, fill_value="extrapolate")
    xnew = ssfr_nad_wvl
    ynew = f(xnew)   # use interpolation function returned by `interp1d`
    
    ax3.plot(ssfr_nad_wvl, ssfr_nad_low_avg/ynew, color='tab:blue', label='SSFR upward / downward flux')
    ax3.plot(lrt_sim_wvl, f_up_low_avg_p3/f_down_low_avg_p3, color='tab:orange', label='LRT upward / downward flux')
    ax3.plot(hsr1_wvl, np.nanmean(hsr1_direct_ratio[low_alt_mask], axis=0), color='tab:green', label='HSR1 direct ratio')
    ax3.legend(fontsize=legend_size)
    ax3.set_ylim(-0.05, 1.35)
    
    ax4.plot(ssfr_nad_wvl, ssfr_nad_high_avg, color='tab:blue', label='SSFR upward flux avg')
    ax4.plot(lrt_sim_wvl, f_up_high_avg_p3, color='tab:orange', label='LRT upward flux avg')
    ax4.fill_between(ssfr_nad_wvl, ssfr_nad_high_avg-ssfr_nad_high_std, ssfr_nad_high_avg+ssfr_nad_high_std, alpha=0.2, color='tab:blue')
    ax4.fill_between(lrt_sim_wvl, f_up_high_avg_p3-f_up_high_std_p3, f_up_high_avg_p3+f_up_high_std_p3, alpha=0.2, color='tab:orange')
    ax4.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax4.legend(fontsize=legend_size)
    
    ax5.plot(ssfr_zen_wvl_adjust, ssfr_zen_high_avg, color='tab:blue', label='SSFR downward flux')
    ax5.plot(lrt_sim_wvl, f_down_high_avg_p3, color='tab:orange', label='LRT downward flux')
    ax5.plot(f_solar_wvl, f_solar_cos_avg, color='tab:gray', label='Solar TOA flux')
    ax5.fill_between(ssfr_zen_wvl_adjust, ssfr_zen_high_avg-ssfr_zen_high_std, ssfr_zen_high_avg+ssfr_zen_high_std, alpha=0.2, color='tab:blue')
    ax5.fill_between(lrt_sim_wvl, f_down_high_avg_p3-f_down_high_std_p3, f_down_high_avg_p3+f_down_high_std_p3, alpha=0.2, color='tab:orange')
    ax5.fill_between(f_solar_wvl, f_solar_cos_avg-f_solar_cos_std, f_solar_cos_avg+f_solar_cos_std, alpha=0.2, color='tab:gray')
    # ax2.set
    ax5.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax5.legend(fontsize=legend_size)
    ax5.set_title(f"High leg ({np.round(np.nanmean(alt[high_alt_mask]), -2):.0f} m)", fontsize=label_size+2)
    

    f = interpolate.interp1d(ssfr_zen_wvl_adjust, ssfr_zen_high_avg, fill_value="extrapolate")
    xnew = ssfr_nad_wvl
    ynew = f(xnew)   # use interpolation function returned by `interp1d`
    
    ax6.plot(ssfr_nad_wvl, ssfr_nad_high_avg/ynew, color='tab:blue', label='SSFR upward / downward flux')
    ax6.plot(lrt_sim_wvl, f_up_high_avg_p3/f_down_high_avg_p3, color='tab:orange', label='LRT upward / downward flux')
    ax6.plot(hsr1_wvl, np.nanmean(hsr1_direct_ratio[high_alt_mask], axis=0), color='tab:green', label='HSR1 direct ratio')
    ax6.legend(fontsize=legend_size)
    ax6.set_ylim(-0.05, 1.35)
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim(350, 2000)
        ax.set_xlabel("Wavelength (nm)", fontsize=label_size)
    for ax in [ax1, ax2, ax4, ax5]:
        ax.set_ylim(-0.05, 1.2)
        
    fig.suptitle(f"{date} average ({cloud_tag})", fontsize=label_size)#, y=0.98
    fig.tight_layout()
    fig.savefig(f"fig/{date}/ssfr_{case_tag}_{cloud_tag}_{date}_avg.png")
    
    # plt.show()
    fig.clear()
    plt.close(fig)
    
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax4 = fig.add_subplot(223)
    ax5 = fig.add_subplot(224)
    
    ax1.plot(lrt_sim_wvl, f_up_low_avg_toa, color='tab:orange', label='with clouds')
    ax1.plot(lrt_sim_wvl, f_up_clear_low_avg_toa, color='tab:red', label='clear sky')
    ax1.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax1.legend(fontsize=legend_size)
    ax1.set_title("Upward (Low leg)", fontsize=label_size+2)
    
    ax2.plot(lrt_sim_wvl, f_down_low_avg_toa, color='tab:orange', label='with clouds')
    ax2.plot(lrt_sim_wvl, f_down_clear_low_avg_toa, color='tab:red', label='clear sky')
    ax2.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax2.legend(fontsize=legend_size)
    ax2.set_title("Downward (Low leg)", fontsize=label_size+2)
    
    ax4.plot(lrt_sim_wvl, f_up_high_avg_toa, color='tab:orange', label='with clouds')
    ax4.plot(lrt_sim_wvl, f_up_clear_high_avg_toa, color='tab:red', label='clear sky')
    ax4.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax4.legend(fontsize=legend_size)
    ax4.set_title("Upward (High leg)", fontsize=label_size+2)
    
    ax5.plot(lrt_sim_wvl, f_down_high_avg_toa, color='tab:orange', label='with clouds')
    ax5.plot(lrt_sim_wvl, f_down_clear_high_avg_toa, color='tab:red', label='clear sky')
    ax5.plot(lrt_sim_wvl, f_down_clear_high_avg_p3, color='tab:blue', label='clear sky (p3 level)')
    ax5.plot(lrt_sim_wvl, f_down_high_avg_p3, color='tab:green', label='with clouds (p3 level)')
    ax5.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax5.legend(fontsize=legend_size)
    ax5.set_title("Downward (High leg)", fontsize=label_size+2)
    
    
    for ax in [ax1, ax2, ax4, ax5]:
        ax.set_xlim(350, 2000)
        ax.set_xlabel("Wavelength (nm)", fontsize=label_size)
    for ax in [ax1, ax2, ax4, ax5]:
        ax.set_ylim(-0.05, 1.2)
        
    fig.suptitle(f"{date} TOA average ({cloud_tag})\nCRE (low leg): {cre_low:.1f} +/- {cre_low_std:.1f} W m-2; CRE (high leg): {cre_high:.1f} +/- {cre_high_std:.1f} W m-2", fontsize=label_size)#, y=0.98
    fig.tight_layout()
    fig.savefig(f"fig/{date}/ssfr_{case_tag}_{cloud_tag}_{date}_cre.png")
    
    # plt.show()
    fig.clear()
    plt.close(fig)
    #\----------------------------------------------------------------------------/#

    return



def flt_trk_lrt_para_clear_2lay(date='20240611',
                          separate_height=1000.,
                          plot_interval=300,
                          case_tag='clear_sky_track_1',
                          overwrite=False
                            ):

    cloud_tag = 'clear'
    
    # read lrt results    
    lrt_fname = f'flt_trk_lrt_para-lrt-{date}-{case_tag}-clear.h5'
    
    with h5py.File(lrt_fname, 'r') as f:
        lon = f['lon'][...]
        lat = f['lon'][...]
        alt = f['alt'][...]*1000  # convert km to m
        sza = f['sza'][...]
        cth = f['cth'][...]*1000  # convert km to m
        cbh = f['cbh'][...]*1000  # convert km to m
        cot = f['cot'][...]
        f_down_clear = f['f_down'][...]
        f_down_clear_dir = f['f_down_dir'][...]
        f_down_clear_diff = f['f_down_diff'][...]
        f_up_clear = f['f_up'][...]
    
        ssfr_zen = f['ssfr_zen'][...]
        ssfr_nad = f['ssfr_nad'][...]
        ssfr_zen_wvl = f['ssfr_zen_wvl'][...]
        ssfr_nad_wvl = f['ssfr_nad_wvl'][...]
        ssfr_toa0 = f['ssfr_toa0'][...]
        
        hsr1_wvl = f['hsr1_wvl'][...]
        hsr1_toa0 = f['hsr1_toa0'][...]
        hsr1_total = f['hsr1_total'][...]
        hsr1_diff = f['hsr1_diff'][...]
    
    hsr1_diff_ratio =  hsr1_diff / hsr1_total
    hsr1_direct_ratio = np.ones_like(hsr1_diff_ratio) - hsr1_diff_ratio
        
    with open('kurudz_ssfr_1nm_convolution.dat', 'r') as f_solar:
        # skip 2 lines with #
        f_solar_lines = f_solar.readlines()
        f_solar_wvl = np.array([float(line.split()[0]) for line in f_solar_lines[2:]])
        f_solar_flux = np.array([float(line.split()[1]) for line in f_solar_lines[2:]])    
        
    data_len = f_down_clear.shape[0]
        
    wvl_si = 795
    wvl_in = 1050
    wvl_width = 8
    hsr1_wvl_si_ind_1 = np.argmin(np.abs(hsr1_wvl - (wvl_si-wvl_width/2)))
    hsr1_wvl_si_ind_2 = np.argmin(np.abs(hsr1_wvl - (wvl_si+wvl_width/2)))
    
    hsr1_wvl_in_ind_1 = np.argmin(np.abs(hsr1_wvl - (wvl_in-wvl_width/2)))
    hsr1_wvl_in_ind_2 = np.argmin(np.abs(hsr1_wvl - (wvl_in+wvl_width/2)))
    hsr1_toa0_si = hsr1_toa0[hsr1_wvl_si_ind_1:hsr1_wvl_si_ind_2+1]
    hsr1_toa0_in = hsr1_toa0[hsr1_wvl_in_ind_1:hsr1_wvl_in_ind_2+1]
    hsr1_toa0_ratio = np.nanmean(hsr1_toa0_si) / np.nanmean(hsr1_toa0_in)
    
    f_solar_si_ind_1 = np.argmin(np.abs(f_solar_wvl - (wvl_si-wvl_width/2)))
    f_solar_si_ind_2 = np.argmin(np.abs(f_solar_wvl - (wvl_si+wvl_width/2)))
    f_solar_in_ind_1 = np.argmin(np.abs(f_solar_wvl - (wvl_in-wvl_width/2)))
    f_solar_in_ind_2 = np.argmin(np.abs(f_solar_wvl - (wvl_in+wvl_width/2)))
    f_solar_si = f_solar_flux[f_solar_si_ind_1:f_solar_si_ind_2+1]
    f_solar_in = f_solar_flux[f_solar_in_ind_1:f_solar_in_ind_2+1]
    f_solar_ratio = np.nanmean(f_solar_si) / np.nanmean(f_solar_in)

    
    ssfr_zen_wvl_si_ind_1 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_si-wvl_width/2)))
    ssfr_zen_wvl_si_ind_2 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_si+wvl_width/2)))
    ssfr_zen_wvl_in_ind_1 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_in-wvl_width/2)))
    ssfr_zen_wvl_in_ind_2 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_in+wvl_width/2)))
    
    ssfr_toa0_si = ssfr_toa0[ssfr_zen_wvl_si_ind_1:ssfr_zen_wvl_si_ind_2+1]
    ssfr_toa0_in = ssfr_toa0[ssfr_zen_wvl_in_ind_1:ssfr_zen_wvl_in_ind_2+1]
    
    ssfr_zen_flux_si = ssfr_zen[:, ssfr_zen_wvl_si_ind_1:ssfr_zen_wvl_si_ind_2+1]
    ssfr_zen_flux_in = ssfr_zen[:, ssfr_zen_wvl_in_ind_1:ssfr_zen_wvl_in_ind_2+1]
    ssfr_toa0_ratio = np.nanmean(ssfr_toa0_si) / np.nanmean(ssfr_toa0_in)
    
    
    
    toa0_ratio = f_solar_ratio
    

    ssfr_zen_flux_ratio = np.nanmean(ssfr_zen_flux_si, axis=1) / np.nanmean(ssfr_zen_flux_in, axis=1)
    
    
    ssfr_nad_wvl_si_ind_1 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_si-wvl_width/2)))
    ssfr_nad_wvl_si_ind_2 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_si+wvl_width/2)))
    ssfr_nad_wvl_in_ind_1 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_in-wvl_width/2)))
    ssfr_nad_wvl_in_ind_2 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_in+wvl_width/2)))
    ssfr_nad_flux_si = ssfr_nad[:, ssfr_nad_wvl_si_ind_1:ssfr_nad_wvl_si_ind_2+1]
    ssfr_nad_flux_in = ssfr_nad[:, ssfr_nad_wvl_in_ind_1:ssfr_nad_wvl_in_ind_2+1]
    ssfr_nad_flux_ratio = np.nanmean(ssfr_nad_flux_si, axis=1) / np.nanmean(ssfr_nad_flux_in, axis=1)
    
    ssfr_zen_scaling_factor = ssfr_zen_flux_ratio / toa0_ratio
    ssfr_nad_scaling_factor = ssfr_nad_flux_ratio / toa0_ratio
    
    
    ssfr_nad_avg = np.nanmean(ssfr_nad, axis=0)
    ssfr_zen_avg = np.nanmean(ssfr_zen, axis=0)
    
    ssfr_nad_std = np.nanstd(ssfr_nad, axis=0)
    ssfr_zen_std = np.nanstd(ssfr_zen, axis=0)

    # repeat ssfr_toa0 for the same shape as ssfr_zen
    ssfr_toa0_expand = np.repeat(ssfr_toa0[np.newaxis, :], ssfr_zen.shape[0], axis=0)
    print("ssfr_toa0_expand.shape:", ssfr_toa0_expand.shape)
    print("ssfr_zen.shape:", ssfr_zen.shape)
    mu_sza = np.cos(np.deg2rad(sza))
    ssfr_toa_cos = ssfr_toa0_expand * mu_sza[:, np.newaxis]
    ssfr_toa_cos_avg = np.nanmean(ssfr_toa_cos, axis=0)
    ssfr_toa_cos_std = np.nanstd(ssfr_toa_cos, axis=0)
    
    f_solar_expand = np.repeat(f_solar_flux[np.newaxis, :], ssfr_zen.shape[0], axis=0)
    f_solar_cos = f_solar_expand * mu_sza[:, np.newaxis]
    f_solar_cos_avg = np.nanmean(f_solar_cos, axis=0)
    f_solar_cos_std = np.nanstd(f_solar_cos, axis=0)
    
    ssfr_zen_interp = np.zeros((ssfr_zen.shape[0], ssfr_nad_wvl.shape[0]))
    ssfr_zen_interp[:, :] = np.nan
    
    ssfr_zen_adjust = np.zeros_like(ssfr_zen_interp)
    ssfr_zen_adjust[:, :] = np.nan
    ssfr_nad_adjust = np.zeros_like(ssfr_zen_interp)
    ssfr_nad_adjust[:, :] = np.nan
    wvl_950_index_left = np.where(ssfr_nad_wvl < 950)[0][-1]
    wvl_950_index_right = np.where(ssfr_nad_wvl >= 950)[0][0]
    

     
    for i in range(ssfr_zen.shape[0]):
        if np.isnan(ssfr_zen[i, :]).all():
            # print(f"Warning: NaN found in ssfr_zen at index {i}, skipping interpolation.")
            continue
        f = interpolate.interp1d(ssfr_zen_wvl, ssfr_zen[i, :], fill_value="extrapolate")
        ssfr_zen_interp[i, :] = f(ssfr_nad_wvl)
    
    ssfr_zen_adjust[:, :wvl_950_index_right] = ssfr_zen_interp[:, :wvl_950_index_right] / ssfr_zen_scaling_factor[:, np.newaxis]
    ssfr_zen_adjust[:, wvl_950_index_right:] = ssfr_zen_interp[:, wvl_950_index_right:]
    
    ssfr_nad_adjust[:, :wvl_950_index_right] = ssfr_nad[:, :wvl_950_index_right] / ssfr_nad_scaling_factor[:, np.newaxis]
    ssfr_nad_adjust[:, wvl_950_index_right:] = ssfr_nad[:, wvl_950_index_right:]
    
    ssfr_zen_wvl_adjust = ssfr_nad_wvl.copy()
        
    adjusted_albedo = np.nanmean(ssfr_nad_adjust / ssfr_zen_adjust, axis=0)
    # neglect the derived ajdusted albedo from  1350 to 1415 nm, do the interpolation instead
    wvl_1350_index_right = np.where(ssfr_nad_wvl >= 1350)[0][0]
    wvl_1415_index_left = np.where(ssfr_nad_wvl < 1415)[0][-1]
    adjusted_albedo_remove_1350_1415 = adjusted_albedo.copy()
    adjusted_albedo_remove_1350_1415[wvl_1350_index_right:wvl_1415_index_left+1] = np.nan
    mask_1350_1415 = np.isnan(adjusted_albedo_remove_1350_1415)
    f_interp = interpolate.interp1d(ssfr_nad_wvl[~mask_1350_1415],
                                    adjusted_albedo_remove_1350_1415[~mask_1350_1415],
                                    fill_value="extrapolate")
    adjusted_albedo[mask_1350_1415] = f_interp(ssfr_nad_wvl[mask_1350_1415])
    # neglect the derived ajdusted albedo aftr 1800 nm, use the last value
    wvl_1800_index_left = np.where(ssfr_nad_wvl < 1800)[0][-1]
    adjusted_albedo[wvl_1800_index_left+1:] = adjusted_albedo[wvl_1800_index_left]
    
    # plt.plot(ssfr_nad_wvl, np.nanmean(ssfr_nad_adjust / ssfr_zen_adjust, axis=0), label='Adjusted albedo before interpolation')
    # plt.plot(ssfr_nad_wvl, adjusted_albedo, label='Adjusted albedo after interpolation')
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Albedo (unitless)")
    # plt.xlim(1200, 2000)
    # plt.legend()
    # plt.title("Adjusted albedo on 20240531")
    # plt.show()
    # sys.exit()
    
    os.makedirs('data/sfc_alb', exist_ok=True)
    with open(os.path.join('data/sfc_alb', f'sfc_alb_{date}.dat'), 'w') as f:
        header = (f'# SSFR derived sfc albedo (adjusted) on {date}\n'
                '# wavelength (nm)      albedo (unitless)\n'
                )
        # Build all profile lines in one go.
        lines = [
                f'{ssfr_nad_wvl[i]:11.3f} '
                f'{adjusted_albedo[i]:12.3e}'
                for i in range(len(adjusted_albedo))
                ]
        f.write(header + "\n".join(lines))
        
    low_alt_mask = alt < separate_height
    high_alt_mask = alt >= separate_height
    
    # ssfr_nad_low_avg = np.nanmean(ssfr_nad[low_alt_mask, :], axis=0)
    # ssfr_zen_low_avg = np.nanmean(ssfr_zen[low_alt_mask, :], axis=0)
    # f_down_low_avg_p3 = np.nanmean(f_down[low_alt_mask, :, 0], axis=0)
    # f_up_low_avg_p3 = np.nanmean(f_up[low_alt_mask, :, 0], axis=0)
    # f_down_low_avg_toa = np.nanmean(f_down[low_alt_mask, :, 1], axis=0)
    # f_up_low_avg_toa = np.nanmean(f_up[low_alt_mask, :, 1], axis=0)
    # f_down_clear_low_avg_toa = np.nanmean(f_down_clear[low_alt_mask, :, 1], axis=0)
    # f_up_clear_low_avg_toa = np.nanmean(f_up_clear[low_alt_mask, :, 1], axis=0)
    
    # ssfr_nad_high_avg = np.nanmean(ssfr_nad[high_alt_mask, :], axis=0)
    # ssfr_zen_high_avg = np.nanmean(ssfr_zen[high_alt_mask, :], axis=0)
    # f_down_high_avg_p3 = np.nanmean(f_down[high_alt_mask, :, 0], axis=0)
    # f_up_high_avg_p3 = np.nanmean(f_up[high_alt_mask, :, 0], axis=0)
    # f_down_high_avg_toa = np.nanmean(f_down[high_alt_mask, :, 1], axis=0)
    # f_up_high_avg_toa = np.nanmean(f_up[high_alt_mask, :, 1], axis=0)
    # f_down_clear_high_avg_toa = np.nanmean(f_down_clear[high_alt_mask, :, 1], axis=0)
    # f_up_clear_high_avg_toa = np.nanmean(f_up_clear[high_alt_mask, :, 1], axis=0)
    
    # ssfr_nad_low_std = np.nanstd(ssfr_nad[low_alt_mask, :], axis=0)
    # ssfr_zen_low_std = np.nanstd(ssfr_zen[low_alt_mask, :], axis=0)
    # f_down_low_std_p3 = np.std(f_down[low_alt_mask, :, 0], axis=0)
    # f_up_low_std_p3 = np.std(f_up[low_alt_mask, :, 0], axis=0)
    # f_down_low_std_toa = np.std(f_down[low_alt_mask, :, 1], axis=0)
    # f_up_low_std_toa = np.std(f_up[low_alt_mask, :, 1], axis=0)
    # f_down_clear_low_std_toa = np.std(f_down_clear[low_alt_mask, :, 1], axis=0)
    # f_up_clear_low_std_toa = np.std(f_up_clear[low_alt_mask, :, 1], axis=0)
    
    # ssfr_nad_high_std = np.nanstd(ssfr_nad[high_alt_mask, :], axis=0)
    # ssfr_zen_high_std = np.nanstd(ssfr_zen[high_alt_mask, :], axis=0)
    # f_down_high_std_p3 = np.std(f_down[high_alt_mask, :, 0], axis=0)
    # f_up_high_std_p3 = np.std(f_up[high_alt_mask, :, 0], axis=0)
    # f_down_high_std_toa = np.std(f_down[high_alt_mask, :, 1], axis=0)
    # f_up_high_std_toa = np.std(f_up[high_alt_mask, :, 1], axis=0)
    # f_down_clear_high_std_toa = np.std(f_down_clear[high_alt_mask, :, 1], axis=0)
    # f_up_clear_high_std_toa = np.std(f_up_clear[high_alt_mask, :, 1], axis=0)
    
    ssfr_nad_low_avg = np.nanmean(ssfr_nad_adjust[low_alt_mask, :], axis=0)
    ssfr_zen_low_avg = np.nanmean(ssfr_zen_adjust[low_alt_mask, :], axis=0)
    f_down_clear_low_avg_toa = np.nanmean(f_down_clear[low_alt_mask, :, 1], axis=0)
    f_up_clear_low_avg_toa = np.nanmean(f_up_clear[low_alt_mask, :, 1], axis=0)
    f_down_clear_low_avg_p3 = np.nanmean(f_down_clear[low_alt_mask, :, 0], axis=0)
    f_up_clear_low_avg_p3 = np.nanmean(f_up_clear[low_alt_mask, :, 0], axis=0)
    
    
    ssfr_nad_high_avg = np.nanmean(ssfr_nad_adjust[high_alt_mask, :], axis=0)
    ssfr_zen_high_avg = np.nanmean(ssfr_zen_adjust[high_alt_mask, :], axis=0)
    f_down_clear_high_avg_toa = np.nanmean(f_down_clear[high_alt_mask, :, 1], axis=0)
    f_up_clear_high_avg_toa = np.nanmean(f_up_clear[high_alt_mask, :, 1], axis=0)
    f_down_clear_high_avg_p3 = np.nanmean(f_down_clear[high_alt_mask, :, 0], axis=0)
    f_up_clear_high_avg_p3 = np.nanmean(f_up_clear[high_alt_mask, :, 0], axis=0)
    
    ssfr_nad_low_std = np.nanstd(ssfr_nad_adjust[low_alt_mask, :], axis=0)
    ssfr_zen_low_std = np.nanstd(ssfr_zen_adjust[low_alt_mask, :], axis=0)
    f_down_clear_low_std_toa = np.std(f_down_clear[low_alt_mask, :, 1], axis=0)
    f_up_clear_low_std_toa = np.std(f_up_clear[low_alt_mask, :, 1], axis=0)
    f_down_clear_low_std_p3 = np.std(f_down_clear[low_alt_mask, :, 0], axis=0)
    f_up_clear_low_std_p3 = np.std(f_up_clear[low_alt_mask, :, 0], axis=0)
    
    ssfr_nad_high_std = np.nanstd(ssfr_nad_adjust[high_alt_mask, :], axis=0)
    ssfr_zen_high_std = np.nanstd(ssfr_zen_adjust[high_alt_mask, :], axis=0)
    f_down_clear_high_std_toa = np.std(f_down_clear[high_alt_mask, :, 1], axis=0)
    f_up_clear_high_std_toa = np.std(f_up_clear[high_alt_mask, :, 1], axis=0)
    f_down_clear_high_std_p3 = np.std(f_down_clear[high_alt_mask, :, 0], axis=0)
    f_up_clear_high_std_p3 = np.std(f_up_clear[high_alt_mask, :, 0], axis=0)

    lrt_sim_wvl = np.arange(360, 1990.1, 1)
    for i in range(data_len)[::plot_interval]:
        fig = plt.figure(figsize=(18, 4.5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.plot(ssfr_nad_wvl, ssfr_nad[i, :], lw=2, label='SSFR upward flux')
        ax1.plot(ssfr_nad_wvl, ssfr_nad_adjust[i, :], label='SSFR upward flux (adjusted)')
        ax1.plot(lrt_sim_wvl, f_up_clear[i, :, 0], label='LRT upward flux')
        ax1.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
        ax1.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax1.legend(fontsize=legend_size)
        
        ax2.plot(ssfr_zen_wvl, ssfr_zen[i, :], lw=2, label='SSFR downward flux')
        ax2.plot(ssfr_zen_wvl_adjust, ssfr_zen_adjust[i, :], label='SSFR downward flux (adjusted)')
        ax2.plot(lrt_sim_wvl, f_down_clear[i, :, 0], label='LRT downward flux')
        ax2.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
        ax2.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax2.legend(fontsize=legend_size)
        

        f = interpolate.interp1d(ssfr_zen_wvl, ssfr_zen[i, :], fill_value="extrapolate")
        xnew = ssfr_nad_wvl
        ynew = f(xnew)   # use interpolation function returned by `interp1d`
        
        ax3.plot(ssfr_nad_wvl, ssfr_nad[i, :]/ynew, lw=2, label='SSFR upward / downward flux')
        ax3.plot(ssfr_nad_wvl, ssfr_nad_adjust[i, :]/ynew, label='SSFR upward / downward flux (adjusted)')
        ax3.plot(lrt_sim_wvl, f_up_clear[i, :, 0]/f_down_clear[i, :, 0], label='LRT upward / downward flux')
        # ax3.set_ylabel("W m^-2 nm^-1")
        ax3.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax3.legend(fontsize=legend_size)
        ax3.set_ylim(-0.05, 1.35)
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(350, 2000)
        
        fig.suptitle(f"Alt: {alt[i]:.0f} m COT: {cot[i]:.2f} CTH: {cth[i]:.0f} m CBH: {cbh[i]:.0f} m", fontsize=label_size)#, y=0.98)
        fig.tight_layout()    
        fig.savefig(f"fig/{date}/ssfr_{cloud_tag}_{date}_{case_tag}_{i}.png")
        
        # plt.show()
        fig.clear()
        plt.close(fig)
    
    
    fig = plt.figure(figsize=(4.8, 7.2))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(ssfr_nad_wvl, ssfr_nad_high_avg, color='tab:blue', label='SSFR upward flux avg')
    ax1.plot(lrt_sim_wvl, f_up_clear_high_avg_p3, color='tab:orange', label='LRT upward flux avg')
    ax1.fill_between(ssfr_nad_wvl, ssfr_nad_high_avg-ssfr_nad_high_std, ssfr_nad_high_avg+ssfr_nad_high_std, alpha=0.2, color='tab:blue')
    ax1.fill_between(lrt_sim_wvl, f_up_clear_high_avg_p3-f_up_clear_high_std_p3, f_up_clear_high_avg_p3+f_up_clear_high_std_p3, alpha=0.2, color='tab:orange')
    ax1.set_title("Upward Flux", fontsize=label_size+2)
    
    ax2.plot(ssfr_zen_wvl_adjust, ssfr_zen_high_avg, color='tab:blue', label='SSFR downward flux')
    ax2.plot(lrt_sim_wvl, f_down_clear_high_avg_p3, color='tab:orange', label='LRT downward flux')
    ax2.plot(f_solar_wvl, f_solar_cos_avg, color='tab:gray', label='Solar TOA flux')
    ax2.fill_between(ssfr_zen_wvl_adjust, ssfr_zen_high_avg-ssfr_zen_high_std, ssfr_zen_high_avg+ssfr_zen_high_std, alpha=0.2, color='tab:blue')
    ax2.fill_between(lrt_sim_wvl, f_down_clear_high_avg_p3-f_down_clear_high_std_p3, f_down_clear_high_avg_p3+f_down_clear_high_std_p3, alpha=0.2, color='tab:orange')
    ax2.set_title(f"Downward Flux", fontsize=label_size+2)
    for ax in [ax1, ax2]:
        ax.set_xlim(350, 2000)
        ax.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax.set_ylabel(r"Flux (W m$^{-2}$ nm$^{-1}$)", fontsize=label_size)
        ax.legend(fontsize=legend_size)
    for ax in [ax1, ax2]:
        ax.set_ylim(-0.05, 1.1)
        
    fig.suptitle(f"{date}", fontsize=label_size+2)#, y=0.98
    fig.tight_layout()
    fig.savefig(f"fig/{date}/ssfr_{cloud_tag}_{date}_{case_tag}_avg_poster.png", dpi=300)
    
    # plt.show()
    fig.clear()
    plt.close(fig)
    # sys.exit()
    
    
    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    ax1.plot(ssfr_nad_wvl, ssfr_nad_low_avg, color='tab:blue', label='SSFR upward flux avg')
    ax1.plot(lrt_sim_wvl, f_up_clear_low_avg_p3, color='tab:orange', label='LRT upward flux avg')
    ax1.fill_between(ssfr_nad_wvl, ssfr_nad_low_avg-ssfr_nad_low_std, ssfr_nad_low_avg+ssfr_nad_low_std, alpha=0.2, color='tab:blue')
    ax1.fill_between(lrt_sim_wvl, f_up_clear_low_avg_p3-f_up_clear_low_std_p3, f_up_clear_low_avg_p3+f_up_clear_low_std_p3, alpha=0.2, color='tab:orange')
    ax1.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax1.legend(fontsize=legend_size)
    
    ax2.plot(ssfr_zen_wvl_adjust, ssfr_zen_low_avg, color='tab:blue', label='SSFR downward flux')
    ax2.plot(lrt_sim_wvl, f_down_clear_low_avg_p3, color='tab:orange', label='LRT downward flux')
    ax2.plot(f_solar_wvl, f_solar_cos_avg, color='tab:gray', label='Solar TOA flux')
    ax2.fill_between(ssfr_zen_wvl_adjust, ssfr_zen_low_avg-ssfr_zen_low_std, ssfr_zen_low_avg+ssfr_zen_low_std, alpha=0.2, color='tab:blue')
    ax2.fill_between(lrt_sim_wvl, f_down_clear_low_avg_p3-f_down_clear_low_std_p3, f_down_clear_low_avg_p3+f_down_clear_low_std_p3, alpha=0.2, color='tab:orange')
    ax2.fill_between(f_solar_wvl, f_solar_cos_avg-f_solar_cos_std, f_solar_cos_avg+f_solar_cos_std, alpha=0.2, color='tab:gray')
    # ax2.set
    ax2.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax2.legend(fontsize=legend_size)
    ax2.set_title(f"Low leg ({np.round(np.nanmean(alt[low_alt_mask]), -2):.0f} m)", fontsize=label_size+2)
    

    f = interpolate.interp1d(ssfr_zen_wvl_adjust, ssfr_zen_low_avg, fill_value="extrapolate")
    xnew = ssfr_nad_wvl
    ynew = f(xnew)   # use interpolation function returned by `interp1d`
    
    ax3.plot(ssfr_nad_wvl, ssfr_nad_low_avg/ynew, color='tab:blue', label='SSFR upward / downward flux')
    ax3.plot(lrt_sim_wvl, f_up_clear_low_avg_p3/f_down_clear_low_avg_p3, color='tab:orange', label='LRT upward / downward flux')
    ax3.plot(hsr1_wvl, np.nanmean(hsr1_direct_ratio[low_alt_mask], axis=0), color='tab:green', label='HSR1 direct ratio')
    ax3.legend(fontsize=legend_size)
    ax3.set_ylim(-0.05, 1.35)
    
    ax4.plot(ssfr_nad_wvl, ssfr_nad_high_avg, color='tab:blue', label='SSFR upward flux avg')
    ax4.plot(lrt_sim_wvl, f_up_clear_high_avg_p3, color='tab:orange', label='LRT upward flux avg')
    ax4.fill_between(ssfr_nad_wvl, ssfr_nad_high_avg-ssfr_nad_high_std, ssfr_nad_high_avg+ssfr_nad_high_std, alpha=0.2, color='tab:blue')
    ax4.fill_between(lrt_sim_wvl, f_up_clear_high_avg_p3-f_up_clear_high_std_p3, f_up_clear_high_avg_p3+f_up_clear_high_std_p3, alpha=0.2, color='tab:orange')
    ax4.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax4.legend(fontsize=legend_size)
    
    ax5.plot(ssfr_zen_wvl_adjust, ssfr_zen_high_avg, color='tab:blue', label='SSFR downward flux')
    ax5.plot(lrt_sim_wvl, f_down_clear_high_avg_p3, color='tab:orange', label='LRT downward flux')
    ax5.plot(f_solar_wvl, f_solar_cos_avg, color='tab:gray', label='Solar TOA flux')
    ax5.fill_between(ssfr_zen_wvl_adjust, ssfr_zen_high_avg-ssfr_zen_high_std, ssfr_zen_high_avg+ssfr_zen_high_std, alpha=0.2, color='tab:blue')
    ax5.fill_between(lrt_sim_wvl, f_down_clear_high_avg_p3-f_down_clear_high_std_p3, f_down_clear_high_avg_p3+f_down_clear_high_std_p3, alpha=0.2, color='tab:orange')
    ax5.fill_between(f_solar_wvl, f_solar_cos_avg-f_solar_cos_std, f_solar_cos_avg+f_solar_cos_std, alpha=0.2, color='tab:gray')
    # ax2.set
    ax5.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax5.legend(fontsize=legend_size)
    ax5.set_title(f"High leg ({np.round(np.nanmean(alt[high_alt_mask]), -2):.0f} m)", fontsize=label_size+2)
    

    f = interpolate.interp1d(ssfr_zen_wvl_adjust, ssfr_zen_high_avg, fill_value="extrapolate")
    xnew = ssfr_nad_wvl
    ynew = f(xnew)   # use interpolation function returned by `interp1d`
    
    ax6.plot(ssfr_nad_wvl, ssfr_nad_high_avg/ynew, color='tab:blue', label='SSFR upward / downward flux')
    ax6.plot(lrt_sim_wvl, f_up_clear_high_avg_p3/f_down_clear_high_avg_p3, color='tab:orange', label='LRT upward / downward flux')
    ax6.plot(hsr1_wvl, np.nanmean(hsr1_direct_ratio[high_alt_mask], axis=0), color='tab:green', label='HSR1 direct ratio')
    ax6.legend(fontsize=legend_size)
    ax6.set_ylim(-0.05, 1.35)
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim(350, 2000)
        ax.set_xlabel("Wavelength (nm)", fontsize=label_size)
    for ax in [ax1, ax2, ax4, ax5]:
        ax.set_ylim(-0.05, 1.2)
        
    fig.suptitle(f"{date} average ({cloud_tag})", fontsize=label_size)#, y=0.98
    fig.tight_layout()
    fig.savefig(f"fig/{date}/ssfr_{cloud_tag}_{date}_{case_tag}_avg.png")
    
    # plt.show()
    fig.clear()
    plt.close(fig)
    
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax4 = fig.add_subplot(223)
    ax5 = fig.add_subplot(224)
    
    ax1.plot(lrt_sim_wvl, f_up_clear_low_avg_toa, color='tab:red', label='clear sky')
    ax1.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax1.legend(fontsize=legend_size)
    ax1.set_title("Upward (Low leg)", fontsize=label_size+2)
    
    ax2.plot(lrt_sim_wvl, f_down_clear_low_avg_toa, color='tab:red', label='clear sky')
    ax2.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax2.legend(fontsize=legend_size)
    ax2.set_title("Downward (Low leg)", fontsize=label_size+2)
    
    ax4.plot(lrt_sim_wvl, f_up_clear_high_avg_toa, color='tab:red', label='clear sky')
    ax4.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax4.legend(fontsize=legend_size)
    ax4.set_title("Upward (High leg)", fontsize=label_size+2)
    
    ax5.plot(lrt_sim_wvl, f_down_clear_high_avg_toa, color='tab:red', label='clear sky')
    ax5.plot(lrt_sim_wvl, f_down_clear_high_avg_p3, color='tab:blue', label='clear sky (p3 level)')
    ax5.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax5.legend(fontsize=legend_size)
    ax5.set_title("Downward (High leg)", fontsize=label_size+2)
    
    
    for ax in [ax1, ax2, ax4, ax5]:
        ax.set_xlim(350, 2000)
        ax.set_xlabel("Wavelength (nm)", fontsize=label_size)
    for ax in [ax1, ax2, ax4, ax5]:
        ax.set_ylim(-0.05, 1.2)
        
    fig.suptitle(f"{date} TOA average ({cloud_tag})", fontsize=label_size)#, y=0.98
    fig.tight_layout()
    fig.savefig(f"fig/{date}/ssfr_{cloud_tag}_{date}_{case_tag}_cre.png")
    
    # plt.show()
    fig.clear()
    plt.close(fig)
    #\----------------------------------------------------------------------------/#

    return





def flt_trk_lrt_para_clear_2lay_lw(date='20240611',
                          separate_height=1000.,
                          plot_interval=300,
                          case_tag='clear_sky_track_1',
                          overwrite=False
                            ):

    cloud_tag = 'clear'
    
    # read lrt results    
    lrt_fname = f'flt_trk_lrt_para-lrt-{date}-{case_tag}-clear-lw.h5'
    
    with h5py.File(lrt_fname, 'r') as f:
        tmhr = f['tmhr'][...]
        lon = f['lon'][...]
        lat = f['lon'][...]
        alt = f['alt'][...]*1000  # convert km to m
        sza = f['sza'][...]
        f_down_clear = f['f_down'][...]*1000
        f_down_clear_dir = f['f_down_dir'][...]*1000
        f_down_clear_diff = f['f_down_diff'][...]*1000
        f_up_clear = f['f_up'][...]*1000
    
        bbr_up = f['bbr_up'][...]
        bbr_down = f['bbr_down'][...]
        bbr_sky_T = f['bbr_sky_T'][...] + 273.15  # convert to Kelvin
        kt19_T = f['kt19'][...] + 273.15  # convert to Kelvin

    data_len = f_down_clear.shape[0]
    
    f_down_clear_p3 = f_down_clear[:, 0, 0]  # p3 level
    f_up_clear_p3 = f_up_clear[:, 0, 0]  # p3
    f_down_clear_toa = f_down_clear[:, 0, 1]  # toa level
    f_up_clear_toa = f_up_clear[:, 0, 1]  # toa level
    
    # print(np.nanmean(f_down_clear_p3))
    # print(f_down_clear_p3[:10])
    # sys.exit()
    
    dwvl_nm = 1
    planck_wvl_nm = np.arange(4500, 42000.1, dwvl_nm)
    planck_irradiance = np.zeros((len(planck_wvl_nm), len(kt19_T)))
    planck_irradiance_sky = np.zeros((len(planck_wvl_nm), len(kt19_T)))
    for i in range(len(kt19_T)):
        planck_irradiance[:, i] = planck(planck_wvl_nm, kt19_T[i])
        planck_irradiance_sky[:, i] = planck(planck_wvl_nm, bbr_sky_T[i])
    
    planck_irradiance_sum = np.sum(planck_irradiance*dwvl_nm, axis=0)
    planck_irradiance_sky_sum = np.sum(planck_irradiance_sky*dwvl_nm, axis=0)
    
    low_alt_mask = alt < separate_height
    high_alt_mask = alt >= separate_height
    
    
    # f_down_clear_low_avg_toa = np.nanmean(f_down_clear[low_alt_mask, :, 1], axis=0)
    # f_up_clear_low_avg_toa = np.nanmean(f_up_clear[low_alt_mask, :, 1], axis=0)
    # f_down_clear_low_avg_p3 = np.nanmean(f_down_clear[low_alt_mask, :, 0], axis=0)
    # f_up_clear_low_avg_p3 = np.nanmean(f_up_clear[low_alt_mask, :, 0], axis=0)
    # bbr_down_low_avg = np.nanmean(bbr_down[low_alt_mask, :], axis=0)
    # bbr_up_low_avg = np.nanmean(bbr_up[low_alt_mask, :], axis=0)
    
    # f_down_clear_high_avg_toa = np.nanmean(f_down_clear[high_alt_mask, :, 1], axis=0)
    # f_up_clear_high_avg_toa = np.nanmean(f_up_clear[high_alt_mask, :, 1], axis=0)
    # f_down_clear_high_avg_p3 = np.nanmean(f_down_clear[high_alt_mask, :, 0], axis=0)
    # f_up_clear_high_avg_p3 = np.nanmean(f_up_clear[high_alt_mask, :, 0], axis=0)
    

    # f_down_clear_low_std_toa = np.std(f_down_clear[low_alt_mask, :, 1], axis=0)
    # f_up_clear_low_std_toa = np.std(f_up_clear[low_alt_mask, :, 1], axis=0)
    # f_down_clear_low_std_p3 = np.std(f_down_clear[low_alt_mask, :, 0], axis=0)
    # f_up_clear_low_std_p3 = np.std(f_up_clear[low_alt_mask, :, 0], axis=0)
    

    # f_down_clear_high_std_toa = np.std(f_down_clear[high_alt_mask, :, 1], axis=0)
    # f_up_clear_high_std_toa = np.std(f_up_clear[high_alt_mask, :, 1], axis=0)
    # f_down_clear_high_std_p3 = np.std(f_down_clear[high_alt_mask, :, 0], axis=0)
    # f_up_clear_high_std_p3 = np.std(f_up_clear[high_alt_mask, :, 0], axis=0)


    fig = plt.figure(figsize=(5, 4))
    ax3 = fig.add_subplot(111)
    ax3.hist(bbr_down[low_alt_mask], bins=50, color='tab:blue', label='BBR downward', density=True)
    ax3.axvline(np.nanmean(f_down_clear_p3[low_alt_mask]), color='tab:green', linestyle='--', label='LRT downward')
    xmin = min(np.nanmin(bbr_down[low_alt_mask]), np.nanmin(f_down_clear_p3[low_alt_mask]))
    xmax = max(np.nanmax(bbr_down[low_alt_mask]), np.nanmax(f_down_clear_p3[low_alt_mask]))
    ax3.set_xlim(xmin*0.975, xmax*1.025)
    ax3.set_xlabel(r"Broadband Flux (W m$^{-2}$)", fontsize=label_size+1)
    ax3.set_ylabel("Density", fontsize=label_size+1)
    ax3.legend(fontsize=legend_size+1)
    fig.tight_layout()
    fig.savefig(f"fig/{date}/bbr_down_{cloud_tag}_{date}_{case_tag}_hist_poster.png", dpi=300)

    
    
    fig = plt.figure(figsize=(24, 10))
    ax1 = fig.add_subplot(241)
    ax4 = fig.add_subplot(242)
    ax2 = fig.add_subplot(243)
    ax3 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245)
    ax8 = fig.add_subplot(246)
    ax6 = fig.add_subplot(247)
    ax7 = fig.add_subplot(248)
    
    ax1_1 = ax1.twinx()
    line_1_1 = ax1.plot(tmhr[low_alt_mask], bbr_up[low_alt_mask], lw=2, color='tab:blue', label='BBR upward flux (p3 level)')
    line_1_2 = ax1.plot(tmhr[low_alt_mask], planck_irradiance_sum[low_alt_mask], lw=2, color='tab:orange', label='KT19 sfc blackbody upward flux')
    line_1_3 = ax1.plot(tmhr[low_alt_mask], f_up_clear_p3[low_alt_mask], lw=2, color='tab:green', label='LRT upward flux (p3 level)')
    line_1_4 = ax1.plot(tmhr[low_alt_mask], f_up_clear_toa[low_alt_mask], lw=2, color='grey', label='LRT upward flux (TOA level)')
    line_1_5 = ax1_1.plot(tmhr[low_alt_mask], kt19_T[low_alt_mask], color='tab:red', label='KT19 T')
    ax1.set_ylabel(r"W m$^{-2}$", fontsize=label_size)
    ax1.set_xlabel("Time (UTC)", fontsize=label_size)
    ax1_1.set_ylabel("KT19 T (K)", fontsize=label_size, color='tab:red')
    ax1_1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(handles=line_1_1+line_1_2+line_1_3+line_1_4+line_1_5, fontsize=legend_size-1, loc=(0.05, 1.05))
    
    ax2.plot(tmhr[low_alt_mask], bbr_down[low_alt_mask], lw=2, color='tab:blue', label='BBR downward flux (p3 level)')
    ax2.plot(tmhr[low_alt_mask], f_down_clear_p3[low_alt_mask], lw=2, color='tab:green', label='LRT downward flux (p3 level)')
    # ax2.plot(tmhr[low_alt_mask], f_down_clear_toa[low_alt_mask], lw=2, color='grey', label='LRT downward flux (TOA level)')
    # ax2.plot(tmhr[low_alt_mask], planck_irradiance_sky_sum[low_alt_mask], lw=2, color='tab:orange', label='BBR sky T blackbody downward flux')
    ax2.set_ylabel(r"W m$^{-2}$", fontsize=label_size)
    ax2.legend(fontsize=legend_size)
    ax2.set_title(f"Low leg ({np.round(np.nanmean(alt[low_alt_mask]), -2):.0f} m)", fontsize=label_size+2)

    ax3.hist(bbr_down[low_alt_mask], bins=50, color='tab:blue', label='BBR downward', density=True)
    # ax3.hist(f_down_clear_p3[low_alt_mask], color='tab:green', label='LRT downward (p3 level)', alpha=0.5)
    ax3.axvline(np.nanmean(f_down_clear_p3[low_alt_mask]), color='tab:green', linestyle='--', label='LRT downward mean (p3 level)')
    # ax3.axvline(np.nanmean(planck_irradiance_sky_sum[low_alt_mask]), color='tab:orange', linestyle='--', label='BBR sky T blackbody flux mean')
    # xmin = min(np.nanmin(bbr_down[low_alt_mask]), np.nanmin(f_down_clear_p3[low_alt_mask]), np.nanmean(planck_irradiance_sky_sum[low_alt_mask]))
    # xmax = max(np.nanmax(bbr_down[low_alt_mask]), np.nanmax(f_down_clear_p3[low_alt_mask]), np.nanmean(planck_irradiance_sky_sum[low_alt_mask]))
    xmin = min(np.nanmin(bbr_down[low_alt_mask]), np.nanmin(f_down_clear_p3[low_alt_mask]))
    xmax = max(np.nanmax(bbr_down[low_alt_mask]), np.nanmax(f_down_clear_p3[low_alt_mask]))
    ax3.set_xlim(xmin*0.975, xmax*1.025)


    ax4.hist(bbr_up[low_alt_mask], bins=50, color='tab:blue', label='BBR upward')
    # ax4.hist(f_up_clear_p3[low_alt_mask], color='tab:green', label='LRT upward (p3 level)', alpha=0.5)
    ax4.axvline(np.nanmean(f_up_clear_p3[low_alt_mask]), color='tab:green', linestyle='--', label='LRT upward mean (p3 level)')
    ax4.axvline(np.nanmean(planck_irradiance_sum[low_alt_mask]), color='tab:orange', linestyle='--', label='KT19 blackbody flux mean')
    xmin = min(np.nanmin(bbr_up[low_alt_mask]), np.nanmin(f_up_clear_p3[low_alt_mask]), np.nanmean(planck_irradiance_sum[low_alt_mask]))
    xmax = max(np.nanmax(bbr_up[low_alt_mask]), np.nanmax(f_up_clear_p3[low_alt_mask]), np.nanmean(planck_irradiance_sum[low_alt_mask]))
    ax4.set_xlim(xmin*0.975, xmax*1.025)

    
    line_5_1 = ax5.plot(tmhr[high_alt_mask], bbr_up[high_alt_mask], lw=2, color='tab:blue', label='BBR upward flux (p3 level)')
    line_5_2 = ax5.plot(tmhr[high_alt_mask], planck_irradiance_sum[high_alt_mask], lw=2, color='tab:orange', label='KT19 sfc blackbody upward flux')
    line_5_3 = ax5.plot(tmhr[high_alt_mask], f_up_clear_p3[high_alt_mask], lw=2, color='tab:green', label='LRT upward flux (p3 level)')
    line_5_4 = ax5.plot(tmhr[high_alt_mask], f_up_clear_toa[high_alt_mask], lw=2, color='grey', label='LRT upward flux (TOA level)')
    ax5_1 = ax5.twinx()
    line_5_5 = ax5_1.plot(tmhr[high_alt_mask], kt19_T[high_alt_mask], color='tab:red', label='KT19 T')
    ax5.set_ylabel(r"W m$^{-2}$", fontsize=label_size)
    ax5.set_xlabel("Time (UTC)", fontsize=label_size)
    ax5_1.set_ylabel("KT19 T (K)", fontsize=label_size, color='tab:red')
    ax5_1.tick_params(axis='y', labelcolor='tab:red')
    # ax5.legend(handles=line_5_1+line_5_2+line_5_3+line_5_4+line_5_5, fontsize=legend_size)
    
    ax6.plot(tmhr[high_alt_mask], bbr_down[high_alt_mask], lw=2, color='tab:blue', label='BBR downward flux (p3 level)')
    ax6.plot(tmhr[high_alt_mask], f_down_clear_p3[high_alt_mask], lw=2, color='tab:green', label='LRT downward flux (p3 level)')
    # ax6.plot(tmhr[high_alt_mask], planck_irradiance_sky_sum[high_alt_mask], lw=2, color='tab:orange', label='BBR sky T blackbody downward flux')
    # ax6.plot(tmhr[high_alt_mask], f_down_clear_toa[high_alt_mask], lw=2, color='grey', label='LRT downward flux (TOA level)')
    ax6.set_ylabel(r"W m$^{-2}$", fontsize=label_size)
    ax6.legend(fontsize=legend_size)
    ax6.set_title(f"High leg ({np.round(np.nanmean(alt[high_alt_mask]), -2):.0f} m)", fontsize=label_size+2)    
    
    ax7.hist(bbr_down[high_alt_mask], bins=50, color='tab:blue', label='BBR downward')
    # ax7.hist(f_down_clear_p3[high_alt_mask], color='tab:green', label='LRT downward (p3 level)', alpha=0.5)
    ax7.axvline(np.nanmean(f_down_clear_p3[high_alt_mask]), color='tab:green', linestyle='--', label='LRT downward mean (p3 level)')
    # ax7.axvline(np.nanmean(planck_irradiance_sky_sum[high_alt_mask]), color='tab:orange', linestyle='--', label='BBR sky T blackbody flux mean')
    # xmin = min(np.nanmin(bbr_down[high_alt_mask]), np.nanmin(f_down_clear_p3[high_alt_mask]), np.nanmean(planck_irradiance_sky_sum[high_alt_mask]))
    # xmax = max(np.nanmax(bbr_down[high_alt_mask]), np.nanmax(f_down_clear_p3[high_alt_mask]), np.nanmean(planck_irradiance_sky_sum[high_alt_mask]))
    xmin = min(np.nanmin(bbr_down[high_alt_mask]), np.nanmin(f_down_clear_p3[high_alt_mask]))
    xmax = max(np.nanmax(bbr_down[high_alt_mask]), np.nanmax(f_down_clear_p3[high_alt_mask]))
    ax7.set_xlim(xmin*0.975, xmax*1.025)
    
    ax8.hist(bbr_up[high_alt_mask], bins=50, color='tab:blue', label='BBR upward')
    # ax8.hist(f_up_clear_p3[high_alt_mask], color='tab:green', label='LRT upward (p3 level)', alpha=0.5)
    ax8.axvline(np.nanmean(f_up_clear_p3[high_alt_mask]), color='tab:green', linestyle='--', label='LRT upward mean (p3 level)')
    ax8.axvline(np.nanmean(planck_irradiance_sum[high_alt_mask]), color='tab:orange', linestyle='--', label='KT19 blackbody flux mean')
    xmin = min(np.nanmin(bbr_up[high_alt_mask]), np.nanmin(f_up_clear_p3[high_alt_mask]), np.nanmean(planck_irradiance_sum[high_alt_mask]))
    xmax = max(np.nanmax(bbr_up[high_alt_mask]), np.nanmax(f_up_clear_p3[high_alt_mask]), np.nanmean(planck_irradiance_sum[high_alt_mask]))
    ax8.set_xlim(xmin*0.975, xmax*1.025)

    
    
    for ax in [ax3, ax4, ax7, ax8]:
        ax.legend(fontsize=legend_size, loc=(0.05, 1.025), ncol=1)
        ax.set_xlabel(r"W m$^{-2}$", fontsize=label_size)
        ax.set_ylabel("Count", fontsize=label_size)
        
    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    #     ax.set_xlim(350, 2000)
    #     ax.set_xlabel("Wavelength (nm)", fontsize=label_size)
    # for ax in [ax1, ax2, ax4, ax5]:
    #     ax.set_ylim(-0.05, 1.2)
        
    fig.suptitle(f"{date} LW ({cloud_tag})", fontsize=label_size+6)#, y=0.98
    fig.tight_layout(pad=0.5)
    fig.savefig(f"fig/{date}/bbr_{cloud_tag}_{date}_{case_tag}.png")

    #\----------------------------------------------------------------------------/#

    return




def flt_trk_lrt_para_cloudy_2lay_lw(date='20240611',
                          separate_height=1000.,
                          manual_cloud=True,
                          plot_interval=300,
                          case_tag='cloud_track_1',
                          overwrite=False
                            ):

    cloud_tag = 'clear'
    
    # read lrt results    
    lrt_fname_clear = f'flt_trk_lrt_para-lrt-{date}-{case_tag}-clear-lw.h5'
    if manual_cloud:
        lrt_fname_cloud = f'flt_trk_lrt_para-lrt-{date}-{case_tag}-manual_cloud-lw.h5'
    else:
        lrt_fname_cloud = f'flt_trk_lrt_para-lrt-{date}-{case_tag}-sat_cloud-lw.h5'
    
    with h5py.File(lrt_fname_cloud, 'r') as f:
        tmhr = f['tmhr'][...]
        lon = f['lon'][...]
        lat = f['lon'][...]
        alt = f['alt'][...]*1000  # convert km to m
        sza = f['sza'][...]
        f_down_clear = f['f_down'][...]*1000
        f_down_clear_dir = f['f_down_dir'][...]*1000
        f_down_clear_diff = f['f_down_diff'][...]*1000
        f_up_clear = f['f_up'][...]*1000
    
        bbr_up = f['bbr_up'][...]
        bbr_down = f['bbr_down'][...]
        kt19_T = f['kt19'][...] + 273.15  # convert to Kelvin

    data_len = f_down_clear.shape[0]
    
    f_down_clear_p3 = f_down_clear[:, 0, 0]  # p3 level
    f_up_clear_p3 = f_up_clear[:, 0, 0]  # p3
    f_down_clear_toa = f_down_clear[:, 0, 1]  # toa level
    f_up_clear_toa = f_up_clear[:, 0, 1]  # toa level
    
    # print(np.nanmean(f_down_clear_p3))
    # print(f_down_clear_p3[:10])
    # sys.exit()
    
    dwvl_nm = 1
    planck_wvl_nm = np.arange(4500, 42000.1, dwvl_nm)
    planck_irradiance = np.zeros((len(planck_wvl_nm), len(kt19_T)))
    for i in range(len(kt19_T)):
        planck_irradiance[:, i] = planck(planck_wvl_nm, kt19_T[i])
    
    planck_irradiance_sum = np.sum(planck_irradiance*dwvl_nm, axis=0)
    
    low_alt_mask = alt < separate_height
    high_alt_mask = alt >= separate_height
    
    
    # f_down_clear_low_avg_toa = np.nanmean(f_down_clear[low_alt_mask, :, 1], axis=0)
    # f_up_clear_low_avg_toa = np.nanmean(f_up_clear[low_alt_mask, :, 1], axis=0)
    # f_down_clear_low_avg_p3 = np.nanmean(f_down_clear[low_alt_mask, :, 0], axis=0)
    # f_up_clear_low_avg_p3 = np.nanmean(f_up_clear[low_alt_mask, :, 0], axis=0)
    # bbr_down_low_avg = np.nanmean(bbr_down[low_alt_mask, :], axis=0)
    # bbr_up_low_avg = np.nanmean(bbr_up[low_alt_mask, :], axis=0)
    
    # f_down_clear_high_avg_toa = np.nanmean(f_down_clear[high_alt_mask, :, 1], axis=0)
    # f_up_clear_high_avg_toa = np.nanmean(f_up_clear[high_alt_mask, :, 1], axis=0)
    # f_down_clear_high_avg_p3 = np.nanmean(f_down_clear[high_alt_mask, :, 0], axis=0)
    # f_up_clear_high_avg_p3 = np.nanmean(f_up_clear[high_alt_mask, :, 0], axis=0)
    

    # f_down_clear_low_std_toa = np.std(f_down_clear[low_alt_mask, :, 1], axis=0)
    # f_up_clear_low_std_toa = np.std(f_up_clear[low_alt_mask, :, 1], axis=0)
    # f_down_clear_low_std_p3 = np.std(f_down_clear[low_alt_mask, :, 0], axis=0)
    # f_up_clear_low_std_p3 = np.std(f_up_clear[low_alt_mask, :, 0], axis=0)
    

    # f_down_clear_high_std_toa = np.std(f_down_clear[high_alt_mask, :, 1], axis=0)
    # f_up_clear_high_std_toa = np.std(f_up_clear[high_alt_mask, :, 1], axis=0)
    # f_down_clear_high_std_p3 = np.std(f_down_clear[high_alt_mask, :, 0], axis=0)
    # f_up_clear_high_std_p3 = np.std(f_up_clear[high_alt_mask, :, 0], axis=0)

    fig = plt.figure(figsize=(5, 4))
    ax3 = fig.add_subplot(111)
    ax3.hist(bbr_down[low_alt_mask], bins=50, color='tab:blue', label='BBR downward', density=True)
    ax3.axvline(np.nanmean(f_down_clear_p3[low_alt_mask]), color='tab:green', linestyle='--', label='LRT downward')
    xmin = min(np.nanmin(bbr_down[low_alt_mask]), np.nanmin(f_down_clear_p3[low_alt_mask]))
    xmax = max(np.nanmax(bbr_down[low_alt_mask]), np.nanmax(f_down_clear_p3[low_alt_mask]))
    ax3.set_xlim(xmin*0.975, xmax*1.025)
    ax3.set_xlabel(r"Broadband Flux (W m$^{-2}$)", fontsize=label_size+1)
    ax3.set_ylabel("Density", fontsize=label_size+1)
    ax3.legend(fontsize=legend_size+1)
    fig.tight_layout()
    fig.savefig(f"fig/{date}/bbr_down_{cloud_tag}_{date}_{case_tag}_cloudy_hist_poster.png", dpi=300)
    
    
    fig = plt.figure(figsize=(5, 4))
    ax3 = fig.add_subplot(111)
    ax3.hist(bbr_up[low_alt_mask], bins=50, color='tab:blue', label='BBR downward', density=True)
    ax3.axvline(np.nanmean(f_up_clear_p3[low_alt_mask]), color='tab:green', linestyle='--', label='LRT downward')
    xmin = min(np.nanmin(bbr_up[low_alt_mask]), np.nanmin(f_up_clear_p3[low_alt_mask]))
    xmax = max(np.nanmax(bbr_up[low_alt_mask]), np.nanmax(f_up_clear_p3[low_alt_mask]))
    ax3.set_xlim(xmin*0.975, xmax*1.025)
    ax3.set_xlabel(r"Broadband Flux (W m$^{-2}$)", fontsize=label_size+1)
    ax3.set_ylabel("Density", fontsize=label_size+1)
    ax3.legend(fontsize=legend_size+1)
    fig.tight_layout()
    fig.savefig(f"fig/{date}/bbr_up_{cloud_tag}_{date}_{case_tag}_cloudy_hist_poster.png", dpi=300)
    
    
    fig = plt.figure(figsize=(24, 10))
    ax1 = fig.add_subplot(241)
    ax4 = fig.add_subplot(242)
    ax2 = fig.add_subplot(243)
    ax3 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245)
    ax8 = fig.add_subplot(246)
    ax6 = fig.add_subplot(247)
    ax7 = fig.add_subplot(248)
    
    ax1_1 = ax1.twinx()
    line_1_1 = ax1.plot(tmhr[low_alt_mask], bbr_up[low_alt_mask], lw=2, color='tab:blue', label='BBR upward flux (p3 level)')
    line_1_2 = ax1.plot(tmhr[low_alt_mask], planck_irradiance_sum[low_alt_mask], lw=2, color='tab:orange', label='KT19 sfc blackbody upward flux')
    line_1_3 = ax1.plot(tmhr[low_alt_mask], f_up_clear_p3[low_alt_mask], lw=2, color='tab:green', label='LRT upward flux (p3 level)')
    line_1_4 = ax1.plot(tmhr[low_alt_mask], f_up_clear_toa[low_alt_mask], lw=2, color='grey', label='LRT upward flux (TOA level)')
    line_1_5 = ax1_1.plot(tmhr[low_alt_mask], kt19_T[low_alt_mask], color='tab:red', label='KT19 T')
    ax1.set_ylabel(r"W m$^{-2}$", fontsize=label_size)
    ax1.set_xlabel("Time (UTC)", fontsize=label_size)
    ax1_1.set_ylabel("KT19 T (K)", fontsize=label_size, color='tab:red')
    ax1_1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(handles=line_1_1+line_1_2+line_1_3+line_1_4+line_1_5, fontsize=legend_size-1, loc=(0.05, 1.05))
    
    ax2.plot(tmhr[low_alt_mask], bbr_down[low_alt_mask], lw=2, color='tab:blue', label='BBR downward flux (p3 level)')
    ax2.plot(tmhr[low_alt_mask], f_down_clear_p3[low_alt_mask], lw=2, color='tab:green', label='LRT downward flux (p3 level)')
    ax2.plot(tmhr[low_alt_mask], f_down_clear_toa[low_alt_mask], lw=2, color='grey', label='LRT downward flux (TOA level)')
    ax2.set_ylabel(r"W m$^{-2}$", fontsize=label_size)
    ax2.legend(fontsize=legend_size)
    ax2.set_title(f"Low leg ({np.round(np.nanmean(alt[low_alt_mask]), -2):.0f} m)", fontsize=label_size+2)

    ax3.hist(bbr_down[low_alt_mask], bins=50, color='tab:blue', label='BBR downward')
    # ax3.hist(f_down_clear_p3[low_alt_mask], color='tab:green', label='LRT downward (p3 level)', alpha=0.5)
    ax3.axvline(np.nanmean(f_down_clear_p3[low_alt_mask]), color='tab:green', linestyle='--', label='LRT downward mean (p3 level)')
    xmin = min(np.nanmin(bbr_down[low_alt_mask]), np.nanmin(f_down_clear_p3[low_alt_mask]))
    xmax = max(np.nanmax(bbr_down[low_alt_mask]), np.nanmax(f_down_clear_p3[low_alt_mask]))
    ax3.set_xlim(xmin*0.975, xmax*1.025)


    ax4.hist(bbr_up[low_alt_mask], bins=50, color='tab:blue', label='BBR upward')
    # ax4.hist(f_up_clear_p3[low_alt_mask], color='tab:green', label='LRT upward (p3 level)', alpha=0.5)
    ax4.axvline(np.nanmean(f_up_clear_p3[low_alt_mask]), color='tab:green', linestyle='--', label='LRT upward mean (p3 level)')
    xmin = min(np.nanmin(bbr_up[low_alt_mask]), np.nanmin(f_up_clear_p3[low_alt_mask]))
    xmax = max(np.nanmax(bbr_up[low_alt_mask]), np.nanmax(f_up_clear_p3[low_alt_mask]))
    ax4.set_xlim(xmin*0.975, xmax*1.025)

    
    line_5_1 = ax5.plot(tmhr[high_alt_mask], bbr_up[high_alt_mask], lw=2, color='tab:blue', label='BBR upward flux (p3 level)')
    line_5_2 = ax5.plot(tmhr[high_alt_mask], planck_irradiance_sum[high_alt_mask], lw=2, color='tab:orange', label='KT19 sfc blackbody upward flux')
    line_5_3 = ax5.plot(tmhr[high_alt_mask], f_up_clear_p3[high_alt_mask], lw=2, color='tab:green', label='LRT upward flux (p3 level)')
    line_5_4 = ax5.plot(tmhr[high_alt_mask], f_up_clear_toa[high_alt_mask], lw=2, color='grey', label='LRT upward flux (TOA level)')
    ax5_1 = ax5.twinx()
    line_5_5 = ax5_1.plot(tmhr[high_alt_mask], kt19_T[high_alt_mask], color='tab:red', label='KT19 T')
    ax5.set_ylabel(r"W m$^{-2}$", fontsize=label_size)
    ax5.set_xlabel("Time (UTC)", fontsize=label_size)
    ax5_1.set_ylabel("KT19 T (K)", fontsize=label_size, color='tab:red')
    ax5_1.tick_params(axis='y', labelcolor='tab:red')
    # ax5.legend(handles=line_5_1+line_5_2+line_5_3+line_5_4+line_5_5, fontsize=legend_size)
    
    ax6.plot(tmhr[high_alt_mask], bbr_down[high_alt_mask], lw=2, color='tab:blue', label='BBR downward flux (p3 level)')
    ax6.plot(tmhr[high_alt_mask], f_down_clear_p3[high_alt_mask], lw=2, color='tab:green', label='LRT downward flux (p3 level)')
    ax6.plot(tmhr[high_alt_mask], f_down_clear_toa[high_alt_mask], lw=2, color='grey', label='LRT downward flux (TOA level)')
    ax6.set_ylabel(r"W m$^{-2}$", fontsize=label_size)
    ax6.legend(fontsize=legend_size)
    ax6.set_title(f"High leg ({np.round(np.nanmean(alt[high_alt_mask]), -2):.0f} m)", fontsize=label_size+2)    
    
    ax7.hist(bbr_down[high_alt_mask], bins=50, color='tab:blue', label='BBR downward')
    # ax7.hist(f_down_clear_p3[high_alt_mask], color='tab:green', label='LRT downward (p3 level)', alpha=0.5)
    ax7.axvline(np.nanmean(f_down_clear_p3[high_alt_mask]), color='tab:green', linestyle='--', label='LRT downward mean (p3 level)')
    xmin = min(np.nanmin(bbr_down[high_alt_mask]), np.nanmin(f_down_clear_p3[high_alt_mask]))
    xmax = max(np.nanmax(bbr_down[high_alt_mask]), np.nanmax(f_down_clear_p3[high_alt_mask]))
    ax7.set_xlim(xmin*0.975, xmax*1.025)
    
    ax8.hist(bbr_up[high_alt_mask], bins=50, color='tab:blue', label='BBR upward')
    # ax8.hist(f_up_clear_p3[high_alt_mask], color='tab:green', label='LRT upward (p3 level)', alpha=0.5)
    ax8.axvline(np.nanmean(f_up_clear_p3[high_alt_mask]), color='tab:green', linestyle='--', label='LRT upward mean (p3 level)')
    xmin = min(np.nanmin(bbr_up[high_alt_mask]), np.nanmin(f_up_clear_p3[high_alt_mask]))
    xmax = max(np.nanmax(bbr_up[high_alt_mask]), np.nanmax(f_up_clear_p3[high_alt_mask]))
    ax8.set_xlim(xmin*0.975, xmax*1.025)

    
    
    for ax in [ax3, ax4, ax7, ax8]:
        ax.legend(fontsize=legend_size, loc=(0.05, 1.025), ncol=1)
        ax.set_xlabel(r"W m$^{-2}$", fontsize=label_size)
        ax.set_ylabel("Count", fontsize=label_size)
        
    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    #     ax.set_xlim(350, 2000)
    #     ax.set_xlabel("Wavelength (nm)", fontsize=label_size)
    # for ax in [ax1, ax2, ax4, ax5]:
    #     ax.set_ylim(-0.05, 1.2)
        
    fig.suptitle(f"{date} LW ({cloud_tag})", fontsize=label_size+6)#, y=0.98
    fig.tight_layout(pad=0.5)
    if manual_cloud:
        fig.savefig(f"fig/{date}/bbr_{cloud_tag}_{date}_{case_tag}_manual_cloud.png")
    else:
        fig.savefig(f"fig/{date}/bbr_{cloud_tag}_{date}_{case_tag}_sat_cloud.png")

    #\----------------------------------------------------------------------------/#

    return




def flt_trk_20240531_lrt_para(
                              overwrite=False
                            ):

    # save rad_2d results
    with h5py.File('flt_trk_lrt_para-lrt-20240531-clear.h5', 'r') as f:
        lon = f['lon'][...]
        lat = f['lon'][...]
        alt = f['alt'][...]*1000  # convert km to m
        sza = f['sza'][...]
        cth = f['cth'][...]*1000  # convert km to m
        cbh = f['cbh'][...]*1000  # convert km to m
        cot = f['cot'][...]
        f_down = f['f_down'][...]
        f_down_dir = f['f_down_dir'][...]
        f_down_diff = f['f_down_diff'][...]
        f_up = f['f_up'][...]
    
        ssfr_zen = f['ssfr_zen'][...]
        ssfr_nad = f['ssfr_nad'][...]
        ssfr_zen_wvl = f['ssfr_zen_wvl'][...]
        ssfr_nad_wvl = f['ssfr_nad_wvl'][...]
        ssfr_toa0 = f['ssfr_toa0'][...]
        
        hsr1_wvl = f['hsr1_wvl'][...]
        hsr1_toa0 = f['hsr1_toa0'][...]
        hsr1_total = f['hsr1_total'][...]
        hsr1_diff = f['hsr1_diff'][...]
        
    data_len = f_down.shape[0]
        
    wvl_si = 795
    wvl_in = 1050
    wvl_width = 8
    hsr1_wvl_si_ind_1 = np.argmin(np.abs(hsr1_wvl - (wvl_si-wvl_width/2)))
    hsr1_wvl_si_ind_2 = np.argmin(np.abs(hsr1_wvl - (wvl_si+wvl_width/2)))
    
    hsr1_wvl_in_ind_1 = np.argmin(np.abs(hsr1_wvl - (wvl_in-wvl_width/2)))
    hsr1_wvl_in_ind_2 = np.argmin(np.abs(hsr1_wvl - (wvl_in+wvl_width/2)))
    hsr1_toa0_si = hsr1_toa0[hsr1_wvl_si_ind_1:hsr1_wvl_si_ind_2+1]
    hsr1_toa0_in = hsr1_toa0[hsr1_wvl_in_ind_1:hsr1_wvl_in_ind_2+1]
    toa0_ratio = np.nanmean(hsr1_toa0_si) / np.nanmean(hsr1_toa0_in)

    
    ssfr_zen_wvl_si_ind_1 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_si-wvl_width/2)))
    ssfr_zen_wvl_si_ind_2 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_si+wvl_width/2)))
    ssfr_zen_wvl_in_ind_1 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_in-wvl_width/2)))
    ssfr_zen_wvl_in_ind_2 = np.argmin(np.abs(ssfr_zen_wvl - (wvl_in+wvl_width/2)))
    
    ssfr_toa0_si = ssfr_toa0[ssfr_zen_wvl_si_ind_1:ssfr_zen_wvl_si_ind_2+1]
    ssfr_toa0_in = ssfr_toa0[ssfr_zen_wvl_in_ind_1:ssfr_zen_wvl_in_ind_2+1]
    
    ssfr_zen_flux_si = ssfr_zen[:, ssfr_zen_wvl_si_ind_1:ssfr_zen_wvl_si_ind_2+1]
    ssfr_zen_flux_in = ssfr_zen[:, ssfr_zen_wvl_in_ind_1:ssfr_zen_wvl_in_ind_2+1]
    ssfr_toa0_ratio = np.nanmean(ssfr_toa0_si) / np.nanmean(ssfr_toa0_in)
    
    toa0_ratio = ssfr_toa0_ratio
    
    ssfr_zen_flux_ratio = np.nanmean(ssfr_zen_flux_si, axis=1) / np.nanmean(ssfr_zen_flux_in, axis=1)
    
    ssfr_nad_wvl_si_ind_1 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_si-wvl_width/2)))
    ssfr_nad_wvl_si_ind_2 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_si+wvl_width/2)))
    ssfr_nad_wvl_in_ind_1 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_in-wvl_width/2)))
    ssfr_nad_wvl_in_ind_2 = np.argmin(np.abs(ssfr_nad_wvl - (wvl_in+wvl_width/2)))
    ssfr_nad_flux_si = ssfr_nad[:, ssfr_nad_wvl_si_ind_1:ssfr_nad_wvl_si_ind_2+1]
    ssfr_nad_flux_in = ssfr_nad[:, ssfr_nad_wvl_in_ind_1:ssfr_nad_wvl_in_ind_2+1]
    ssfr_nad_flux_ratio = np.nanmean(ssfr_nad_flux_si, axis=1) / np.nanmean(ssfr_nad_flux_in, axis=1)
    
    ssfr_zen_flux_ratio_to_hsr1 = ssfr_zen_flux_ratio / toa0_ratio
    ssfr_nad_flux_ratio_to_hsr1 = ssfr_nad_flux_ratio / toa0_ratio
    
    # print("ssfr_zen_flux_ratio_to_hsr1[:3]:", ssfr_zen_flux_ratio_to_hsr1[:3])
    # print("ssfr_nad_flux_ratio_to_hsr1[:3]:", ssfr_nad_flux_ratio_to_hsr1[:3])
    
    # plt.hist(ssfr_zen_flux_ratio_to_hsr1, bins=100, range=(0.5, 1.5), label='SSFR zen flux ratio to HSR1')
    # plt.hist(ssfr_nad_flux_ratio_to_hsr1, bins=100, range=(0.5, 1.5), label='SSFR nad flux ratio to HSR1')
    # plt.xlabel("Flux ratio")
    # plt.ylabel("Count")
    # plt.legend()
    # plt.show()
    # sys.exit()
    
    # ssfr_zen_flux_ratio_to_hsr1 = toa0_ratio / ssfr_zen_flux_ratio
    # ssfr_nad_flux_ratio_to_hsr1 = toa0_ratio / ssfr_nad_flux_ratio
    
    ssfr_nad_avg = np.nanmean(ssfr_nad, axis=0)
    ssfr_zen_avg = np.nanmean(ssfr_zen, axis=0)
    f_down_avg_p3 = np.nanmean(f_down[:, :, 0], axis=0)
    f_up_avg_p3 = np.nanmean(f_up[:, :, 0], axis=0)
    
    ssfr_nad_std = np.nanstd(ssfr_nad, axis=0)
    ssfr_zen_std = np.nanstd(ssfr_zen, axis=0)
    f_down_std_p3 = np.std(f_down[:, :, 0], axis=0)
    f_up_std_p3 = np.std(f_up[:, :, 0], axis=0)

    # repeat ssfr_toa0 for the same shape as ssfr_zen
    ssfr_toa0_expand = np.repeat(ssfr_toa0[np.newaxis, :], ssfr_zen.shape[0], axis=0)
    print("ssfr_toa0_expand.shape:", ssfr_toa0_expand.shape)
    print("ssfr_zen.shape:", ssfr_zen.shape)
    mu_sza = np.cos(np.deg2rad(sza))
    ssfr_toa_cos = ssfr_toa0_expand * mu_sza[:, np.newaxis]
    ssfr_toa_cos_avg = np.nanmean(ssfr_toa_cos, axis=0)
    ssfr_toa_cos_std = np.nanstd(ssfr_toa_cos, axis=0)
    
    ssfr_zen_interp = np.zeros((ssfr_zen.shape[0], ssfr_nad_wvl.shape[0]))
    ssfr_zen_interp[:, :] = np.nan
    
    ssfr_zen_adjust = np.zeros_like(ssfr_zen_interp)
    ssfr_zen_adjust[:, :] = np.nan
    ssfr_nad_adjust = np.zeros_like(ssfr_zen_interp)
    ssfr_nad_adjust[:, :] = np.nan
    wvl_950_index_left = np.where(ssfr_nad_wvl < 950)[0][-1]
    wvl_950_index_right = np.where(ssfr_nad_wvl >= 950)[0][0]
     
    for i in range(ssfr_zen.shape[0]):
        if np.isnan(ssfr_zen[i, :]).all():
            # print(f"Warning: NaN found in ssfr_zen at index {i}, skipping interpolation.")
            continue
        f = interpolate.interp1d(ssfr_zen_wvl, ssfr_zen[i, :], fill_value="extrapolate")
        ssfr_zen_interp[i, :] = f(ssfr_nad_wvl)
    
    ssfr_zen_scaling_factor = ssfr_zen_interp[:, wvl_950_index_right] / ssfr_zen_interp[:, wvl_950_index_left]
    # ssfr_zen_adjust[:, :wvl_950_index_right] = ssfr_zen_interp[:, :wvl_950_index_right] / ssfr_zen_scaling_factor[:, np.newaxis]
    ssfr_zen_adjust[:, :wvl_950_index_right] = ssfr_zen_interp[:, :wvl_950_index_right] / ssfr_zen_flux_ratio_to_hsr1[:, np.newaxis]
    ssfr_zen_adjust[:, wvl_950_index_right:] = ssfr_zen_interp[:, wvl_950_index_right:]
    
    ssfr_nad_scaling_factor = ssfr_nad[:, wvl_950_index_right] / ssfr_nad[:, wvl_950_index_left]
    # ssfr_nad_adjust[:, :wvl_950_index_right] = ssfr_nad[:, :wvl_950_index_right] / ssfr_nad_scaling_factor[:, np.newaxis]
    ssfr_nad_adjust[:, :wvl_950_index_right] = ssfr_nad[:, :wvl_950_index_right] / ssfr_nad_flux_ratio_to_hsr1[:, np.newaxis]
    ssfr_nad_adjust[:, wvl_950_index_right:] = ssfr_nad[:, wvl_950_index_right:]
    
    
    
    adjusted_albedo = np.nanmean(ssfr_nad_adjust / ssfr_zen_adjust, axis=0)
    # neglect the derived ajdusted albedo from  1350 to 1415 nm, do the interpolation instead
    wvl_1350_index_right = np.where(ssfr_nad_wvl >= 1350)[0][0]
    wvl_1415_index_left = np.where(ssfr_nad_wvl < 1415)[0][-1]
    adjusted_albedo_remove_1350_1415 = adjusted_albedo.copy()
    adjusted_albedo_remove_1350_1415[wvl_1350_index_right:wvl_1415_index_left+1] = np.nan
    mask_1350_1415 = np.isnan(adjusted_albedo_remove_1350_1415)
    f_interp = interpolate.interp1d(ssfr_nad_wvl[~mask_1350_1415],
                                    adjusted_albedo_remove_1350_1415[~mask_1350_1415],
                                    fill_value="extrapolate")
    adjusted_albedo[mask_1350_1415] = f_interp(ssfr_nad_wvl[mask_1350_1415])
    # neglect the derived ajdusted albedo aftr 1800 nm, use the last value
    wvl_1800_index_left = np.where(ssfr_nad_wvl < 1800)[0][-1]
    adjusted_albedo[wvl_1800_index_left+1:] = adjusted_albedo[wvl_1800_index_left]
    
    # plt.plot(ssfr_nad_wvl, np.nanmean(ssfr_nad_adjust / ssfr_zen_adjust, axis=0), label='Adjusted albedo before interpolation')
    # plt.plot(ssfr_nad_wvl, adjusted_albedo, label='Adjusted albedo after interpolation')
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Albedo (unitless)")
    # plt.xlim(1200, 2000)
    # plt.legend()
    # plt.title("Adjusted albedo on 20240531")
    # plt.show()
    # sys.exit()
    
    os.makedirs('data/sfc_alb', exist_ok=True)
    with open(os.path.join('data/sfc_alb', f'sfc_alb_20240531.dat'), 'w') as f:
        header = (f'# SSFR derived sfc albedo (adjusted) on 20240531\n'
                '# wavelength (nm)      albedo (unitless)\n'
                )
        # Build all profile lines in one go.
        lines = [
                f'{ssfr_nad_wvl[i]:11.3f} '
                f'{adjusted_albedo[i]:12.3e}'
                for i in range(len(adjusted_albedo))
                ]
        f.write(header + "\n".join(lines))
    
    
    ssfr_zen_interp_avg = np.nanmean(ssfr_zen_interp, axis=0)
    ssfr_zen_interp_std = np.nanstd(ssfr_zen_interp, axis=0)
    
    ssfr_zen_adjust_avg = np.nanmean(ssfr_zen_adjust, axis=0)
    ssfr_nad_adjust_avg = np.nanmean(ssfr_nad_adjust, axis=0)
    ssfr_zen_adjust_std = np.nanstd(ssfr_zen_adjust, axis=0)
    ssfr_nad_adjust_std = np.nanstd(ssfr_nad_adjust, axis=0)
    
    # plt.plot(ssfr_nad_wvl, ssfr_zen_interp_avg, color='tab:blue', label='SSFR zen flux adjusted avg')
    # plt.plot(ssfr_zen_wvl, ssfr_zen_avg, color='tab:orange', label='SSFR zen flux avg')
    # plt.legend()
    # plt.show()
    # sys.exit()
    
    lrt_sim_wvl = np.arange(360, 1990.1, 1)
    for i in range(data_len)[::500]:
        fig = plt.figure(figsize=(18, 4.5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.plot(ssfr_nad_wvl, ssfr_nad[i, :], label='SSFR nadir flux')
        ax1.plot(lrt_sim_wvl, f_up[i, :, 0], label='LRT upward flux')
        ax1.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
        ax1.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax1.legend(fontsize=legend_size)
        
        ax2.plot(ssfr_zen_wvl, ssfr_zen[i, :], label='SSFR zen flux')
        ax2.plot(lrt_sim_wvl, f_down[i, :, 0], label='LRT downward flux')
        ax2.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
        ax2.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax2.legend(fontsize=legend_size)
        

        f = interpolate.interp1d(ssfr_zen_wvl, ssfr_zen[i, :], fill_value="extrapolate")
        xnew = ssfr_nad_wvl
        ynew = f(xnew)   # use interpolation function returned by `interp1d`
        
        ax3.plot(ssfr_nad_wvl, ssfr_nad[i, :]/ynew, label='SSFR nadir flux / zenith flux')
        ax3.plot(lrt_sim_wvl, f_up[i, :, 0]/f_down[i, :, 0], label='LRT upward flux / downward flux')
        # ax3.set_ylabel("W m^-2 nm^-1")
        ax3.set_xlabel("Wavelength (nm)", fontsize=label_size)
        ax3.legend(fontsize=legend_size)
        ax3.set_ylim(-0.05, 1.35)
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(350, 2000)
        
        fig.suptitle(f"Alt: {alt[i]:.0f} m COT: {cot[i]:.2f} CTH: {cth[i]:.0f} m CBH: {cbh[i]:.0f} m", fontsize=label_size)#, y=0.98)
        fig.tight_layout()
        fig.savefig(f"fig/20240531/ssfr_20240531_{i}.png")
        
        # plt.show()
        fig.clear()
        plt.close(fig)
    
    fig = plt.figure(figsize=(18, 4.5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.plot(ssfr_nad_wvl, ssfr_nad_avg, color='tab:blue', label='SSFR nadir flux avg')
    ax1.plot(lrt_sim_wvl, f_up_avg_p3, color='tab:orange', label='LRT upward flux avg')
    ax1.fill_between(ssfr_nad_wvl, ssfr_nad_avg-ssfr_nad_std, ssfr_nad_avg+ssfr_nad_std, alpha=0.2, color='tab:blue')
    ax1.fill_between(lrt_sim_wvl, f_up_avg_p3-f_up_std_p3, f_up_avg_p3+f_up_std_p3, alpha=0.2, color='tab:orange')
    ax1.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax1.set_xlabel("Wavelength (nm)", fontsize=label_size)
    ax1.legend(fontsize=legend_size)
    
    ax2.plot(ssfr_zen_wvl, ssfr_zen_avg, color='tab:blue', label='SSFR zen flux')
    ax2.plot(lrt_sim_wvl, f_down_avg_p3, color='tab:orange', label='LRT downward flux')
    ax2.fill_between(ssfr_zen_wvl, ssfr_zen_avg-ssfr_zen_std, ssfr_zen_avg+ssfr_zen_std, alpha=0.2, color='tab:blue')
    ax2.fill_between(lrt_sim_wvl, f_down_avg_p3-f_down_std_p3, f_down_avg_p3+f_down_std_p3, alpha=0.2, color='tab:orange')
    # ax2.set
    ax2.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax2.set_xlabel("Wavelength (nm)", fontsize=label_size)
    ax2.legend(fontsize=legend_size)
    

    
    ax3.plot(ssfr_nad_wvl, ssfr_nad_avg/ssfr_zen_interp_avg, color='tab:blue', label='SSFR nadir flux / zenith flux')
    ax3.plot(lrt_sim_wvl, f_up_avg_p3/f_down_avg_p3, color='tab:orange', label='LRT upward flux / downward flux')
    # ax3.set_ylabel("W m^-2 nm^-1")
    ax3.set_xlabel("Wavelength (nm)", fontsize=label_size)
    ax3.legend(fontsize=legend_size)
    ax3.set_ylim(-0.05, 1.35)
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(350, 2000)
    for ax in [ax1, ax2]:
        ax.set_ylim(-0.05, 1.2)
        ax.vlines(950, -0.05, 1.2, color='tab:green', linestyle='--', label='950 nm')
    
    fig.suptitle(f"20240531 clear sky average", fontsize=label_size)#, y=0.98
    fig.tight_layout()
    fig.savefig(f"fig/20240531/ssfr_20240531_avg.png")
    
    # plt.show()
    fig.clear()
    plt.close(fig)
    
    
    fig = plt.figure(figsize=(18, 4.5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.plot(ssfr_nad_wvl, ssfr_nad_adjust_avg, color='tab:blue', label='SSFR nadir flux avg')
    ax1.plot(lrt_sim_wvl, f_up_avg_p3, color='tab:orange', label='LRT upward flux avg')
    ax1.fill_between(ssfr_nad_wvl, ssfr_nad_adjust_avg-ssfr_nad_adjust_std, ssfr_nad_adjust_avg+ssfr_nad_adjust_std, alpha=0.2, color='tab:blue')
    ax1.fill_between(lrt_sim_wvl, f_up_avg_p3-f_up_std_p3, f_up_avg_p3+f_up_std_p3, alpha=0.2, color='tab:orange')
    ax1.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax1.set_xlabel("Wavelength (nm)", fontsize=label_size)
    ax1.legend(fontsize=legend_size)
    
    ax2.plot(ssfr_nad_wvl, ssfr_zen_adjust_avg, color='tab:blue', label='SSFR zen flux')
    ax2.plot(lrt_sim_wvl, f_down_avg_p3, color='tab:orange', label='LRT downward flux')
    ax2.plot(ssfr_zen_wvl, ssfr_toa_cos_avg, color='grey', label='SSFR toa0')
    ax2.fill_between(ssfr_nad_wvl, ssfr_zen_adjust_avg-ssfr_zen_adjust_std, ssfr_zen_adjust_avg+ssfr_zen_adjust_std, alpha=0.2, color='tab:blue')
    ax2.fill_between(lrt_sim_wvl, f_down_avg_p3-f_down_std_p3, f_down_avg_p3+f_down_std_p3, alpha=0.2, color='tab:orange')
    ax2.fill_between(ssfr_zen_wvl, ssfr_toa_cos_avg-ssfr_toa_cos_std, ssfr_toa_cos_avg+ssfr_toa_cos_std, alpha=0.2, color='grey')
    # ax2.set
    ax2.set_ylabel(r"W m$^{-2}$ nm$^{-1}$", fontsize=label_size)
    ax2.set_xlabel("Wavelength (nm)", fontsize=label_size)
    ax2.legend(fontsize=legend_size)
    
    
    ax3.plot(ssfr_nad_wvl, ssfr_nad_adjust_avg/ssfr_zen_adjust_avg, color='tab:blue', label='SSFR nadir flux / zenith flux')
    ax3.plot(lrt_sim_wvl, f_up_avg_p3/f_down_avg_p3, color='tab:orange', label='LRT upward flux / downward flux')
    # ax3.set_ylabel("W m^-2 nm^-1")
    ax3.set_xlabel("Wavelength (nm)", fontsize=label_size)
    ax3.legend(fontsize=legend_size)
    ax3.set_ylim(-0.05, 1.35)
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(350, 2000)
    for ax in [ax1, ax2]:
        ax.set_ylim(-0.05, 1.2)
        ax.vlines(950, -0.05, 1.2, color='tab:green', linestyle='--', label='950 nm')
        ax.vlines(wvl_in, -0.05, 1.2, color='r', linestyle='--', label=f'{wvl_in:.0f} nm')
        ax.fill_betweenx([-0.05, 1.2], wvl_in-wvl_width/2, wvl_in+wvl_width/2, color='r', alpha=0.3)
        ax.vlines(wvl_si, -0.05, 1.2, color='b', linestyle='--', label=f'{wvl_si:.0f} nm')
        ax.fill_betweenx([-0.05, 1.2], wvl_si-wvl_width/2, wvl_si+wvl_width/2, color='b', alpha=0.3)
    
    fig.suptitle(f"20240531 clear sky average (adjusted)", fontsize=label_size)#, y=0.98
    fig.tight_layout()
    fig.savefig(f"fig/20240531/ssfr_20240531_adjust_avg.png")
    
    # plt.show()
    fig.clear()
    plt.close(fig)
    #\----------------------------------------------------------------------------/#

    return

def f_solar_convolved(overwrite=False):
    from util import gaussian
    from scipy.signal import convolve
    
    
    xx = np.linspace(-12, 12, 241)
    yy_gaussian_vis = gaussian(xx, 0, 3.82)
    yy_gaussian_nir = gaussian(xx, 0, 5.10)
    
    
    # write out the convolved solar flux
    #/----------------------------------------------------------------------------\#
    if not os.path.exists('kurudz_ssfr_1nm_convolution.dat') or overwrite:
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
        wvl_solar_vis = np.arange(300, 950.1, 1)
        wvl_solar_nir = np.arange(951, 2500.1, 1)
        wvl_solar_coarse = np.concatenate([wvl_solar_vis, wvl_solar_nir])
        flux_solar_convolved_coarse = np.zeros_like(wvl_solar_coarse)
        for vis_wvl_i in range(len(wvl_solar_vis)):
            ind = wvl_solar == wvl_solar_vis[vis_wvl_i]
            flux_solar_convolved_coarse[vis_wvl_i] = flux_solar_convolved_vis[ind]
        for nir_wvl_i in range(len(wvl_solar_nir)):
            ind = wvl_solar == wvl_solar_nir[nir_wvl_i]
            flux_solar_convolved_coarse[nir_wvl_i+len(wvl_solar_vis)] = flux_solar_convolved_nir[ind]
        
        flux_solar_convolved_coarse /= 1000  # convert mW/m^2/nm to W/m^2/nm
        
        with open('kurudz_ssfr_1nm_convolution.dat', 'w') as f_solar:
            header = ('# SSFR version solar flux\n'
                    '# wavelength (nm)      flux (W/m^2/nm)\n'
                    )
            # Build all profile lines in one go.
            lines = [
                    f'{wvl_solar_coarse[i]:11.1f} '
                    f'{flux_solar_convolved_coarse[i]:12.7e}'
                    for i in range(len(wvl_solar_coarse))
                    ]
            f_solar.write(header + "\n".join(lines))

if __name__ == '__main__':

    f_solar_convolved(overwrite=False)
   
    # flt_trk_20240531_lrt_para()
    
    ##### SW analysis #####
    
    ### clear sky SW analysis ###
    #/----------------------------------------------------------------------------\#
    # flt_trk_lrt_para_clear_2lay(date='20240531',
    #                       separate_height=1500,
    #                       plot_interval=300,
    #                       case_tag='clear_sky_track_1',
    #                       overwrite=True
    #                         )

    # flt_trk_lrt_para_clear_2lay(date='20240605',
    #                       separate_height=1500,
    #                       plot_interval=300,
    #                       case_tag='clear_sky_track_1',
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_clear_2lay(date='20240613',
    #                       separate_height=300,
    #                       plot_interval=100,
    #                       case_tag='clear_sky_track_1',
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_clear_2lay(date='20240606',
    #                       separate_height=800,
    #                       plot_interval=100,
    #                       case_tag='clear_sky_track_1',
    #                       overwrite=True
    #                         )
    #/----------------------------------------------------------------------------\#
    

    ### cloudy sky SW analysis ###
    #/----------------------------------------------------------------------------\#
    # flt_trk_lrt_para_2lay(date='20240607',
    #                       separate_height=300.,
    #                       manual_cloud=True,
    #                       case_tag='cloudy_track_1',
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_2lay(date='20240611',
    #                       separate_height=1000.,
    #                       manual_cloud=True,
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_2lay(date='20240613',
    #                       separate_height=300.,
    #                       manual_cloud=True,
    #                       plot_interval=500,
    #                       case_tag='cloudy_track_1',
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_2lay(date='20240603',
    #                       separate_height=600.,
    #                       manual_cloud=True,
    #                       plot_interval=100,
    #                       case_tag='cloudy_track_1',
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_2lay(date='20240606',
    #                       separate_height=300.,
    #                       manual_cloud=True,
    #                       plot_interval=500,
    #                       case_tag='cloudy_track_1',
    #                       overwrite=True
    #                         )
    
    
    # flt_trk_lrt_para_2lay(date='20240613',
    #                       separate_height=300.,
    #                       manual_cloud=False,
    #                       plot_interval=500,
    #                       case_tag='cloudy_track_1',
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_2lay(date='20240603',
    #                       separate_height=600.,
    #                       manual_cloud=False,
    #                       plot_interval=500,
    #                       case_tag='cloudy_track_1',
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_2lay(date='20240606',
    #                       separate_height=300.,
    #                       manual_cloud=False,
    #                       plot_interval=500,
    #                       case_tag='cloudy_track_2',
    #                       overwrite=True
    #                         )
    
    #/----------------------------------------------------------------------------\#
    
    
    ##### LW analysis #####
    #/----------------------------------------------------------------------------\#
    ### clear sky LW analysis ###
    # flt_trk_lrt_para_clear_2lay_lw(date='20240531',
    #                       separate_height=1500,
    #                       plot_interval=300,
    #                       case_tag='clear_sky_track_1',
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_clear_2lay_lw(date='20240605',
    #                       separate_height=1500,
    #                       plot_interval=300,
    #                       case_tag='clear_sky_track_1',
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_clear_2lay_lw(date='20240613',
    #                       separate_height=300,
    #                       plot_interval=100,
    #                       case_tag='clear_sky_track_1',
    #                       overwrite=True
    #                         )
    
    # flt_trk_lrt_para_clear_2lay_lw(date='20240606',
    #                       separate_height=800,
    #                       plot_interval=100,
    #                       case_tag='clear_sky_track_1',
    #                       overwrite=True
    #                         ) 
    #/----------------------------------------------------------------------------\#
    
    flt_trk_lrt_para_cloudy_2lay_lw(date='20240603',
                          separate_height=600.,
                          manual_cloud=True,
                          plot_interval=100,
                          case_tag='cloudy_track_1',
                          overwrite=True
                            )