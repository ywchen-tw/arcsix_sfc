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


def plot_cld_var(date_list, sat_list, manual_list, 
                 xlabel="COT", ylabel="Density",
                 title="Cloud Optical Thickness (COT)",
                 var_name='cot', fig_tag='test',
                 verbose=False,
                 label_size=12, legend_size=8):
    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(111)
    
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    for ind, j in enumerate(sorted(set(date_list))):
        mask = np.array(date_list) == j
        print(f"Processing date: {j}, mask length: {np.sum(mask)}")
            
        date_label = f'{int(j[4:6])}/{int(j[6:])}'
        
        ax1.hist(np.array(sat_list)[mask], bins=25, alpha=0.35, label=f'satellite ({date_label})', density=True, color=color_list[ind],)
        ax1.axvline(np.nanmean(np.array(manual_list)[mask]), linestyle='--', linewidth=2, alpha=0.8,
                    label=f'In situ ({date_label})', 
                    color=color_list[ind])
        if verbose:
            print(f"Date: {j}, Mean: {np.nanmean(np.array(manual_list)[mask])}, Std: {np.nanstd(np.array(manual_list)[mask])}")
    xmin, xmax = ax1.get_xlim()
    ax1.set_xlim(0, xmax)
    ax1.set_title(title, fontsize=label_size+2)
    ax1.set_xlabel(xlabel, fontsize=label_size)
    ax1.set_ylabel(ylabel, fontsize=label_size)
    ax1.legend(fontsize=legend_size)
    ax1.tick_params('both', labelsize=label_size-2)
    # fig.suptitle(f"Clouds on {date}", fontsize=label_size+2, y=0.98)
    fig.tight_layout()
    fig.savefig(f"fig/clouds/clouds_{var_name}_compare_{fig_tag}.png", dpi=300, bbox_inches='tight')


def flt_trk_cld_2lay(dates=['20240611', '20240613'],
                          case_tag=['default', 'default'],
                          separate_heights=[1000, 2000],
                          fig_tag='test',
                          overwrite=False
                            ):


    manual_cloud_tag = 'manual_cloud'
    sat_cloud_tag = 'sat_cloud'
    clear_tag = 'clear'
    
    lrt_fname = 'flt_trk_lrt_para-lrt-{}-{}-{}.h5'
    
    seg_list = []
    alt_list = []
    date_list = []
    sza_list = []
    sat_cot_list = []
    sat_cth_list = []
    sat_cwp_list = []
    sat_cer_list = []
    manual_cot_list = []
    manual_cth_list = []
    manual_cwp_list = []
    manual_cer_list = []
    
    for i, (date, tag, sep_H) in enumerate(zip(dates, case_tag, separate_heights)):
        print(f"Processing date: {date}, case_tag: {tag}")
        
        # process sat cloud data
        lrt_fname_full = lrt_fname.format(date, tag, sat_cloud_tag)
        
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
            # Process the data and create the file

        
        with h5py.File(lrt_fname_full, 'r') as f:
            alt = f['alt'][...]
            sza = f['sza'][...]
            cth = f['cth'][...]
            cot = f['cot'][...]
            cwp = f['cwp'][...]*1000  # convert to g m^-2
            cer = f['cer'][...]
        
        date_repeat = np.repeat(date, alt.shape[0])
        date_list.extend(date_repeat)
        alt_list.extend(alt)
        sza_list.extend(sza)
        
        
        seg = np.zeros_like(alt, dtype=int)
        seg[alt < sep_H] = i+1  # low altitude segment
        seg[alt >= sep_H] = (i+1)*1000+1  # high altitude segment
        seg_list.extend(seg)
        
        cth = np.where(cth < 0, np.nan, cth)  # set negative values to NaN
        sat_cot_list.extend(cot)
        sat_cth_list.extend(cth)
        sat_cwp_list.extend(cwp)
        sat_cer_list.extend(cer)
        
        # process manual cloud data
        lrt_fname_full = lrt_fname.format(date, tag, manual_cloud_tag)
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
        
        with h5py.File(lrt_fname_full, 'r') as f:
            cth = f['cth'][...]  # convert km
            cot = f['cot'][...]
            cwp = f['cwp'][...]*1000  # convert to g m^-2
            cer = f['cer'][...]
        
        cth = np.where(cth < 0, np.nan, cth)  # set negative values to NaN
        manual_cot_list.extend(cot)
        manual_cth_list.extend(cth)
        manual_cwp_list.extend(cwp)
        manual_cer_list.extend(cer)
    
    sat_cot_nan_mask = np.isnan(sat_cot_list)
    
    for cloud_list in [sat_cot_list, sat_cth_list, sat_cwp_list, sat_cer_list,
                       manual_cot_list, manual_cth_list, manual_cwp_list, manual_cer_list]:
        cloud_list = np.array(cloud_list)
        cloud_list[cloud_list<0] = np.nan  # set negative values to NaN
    
    for cloud_list in [date_list, alt_list, sza_list, seg_list,
                       sat_cot_list, sat_cth_list, sat_cwp_list, sat_cer_list,
                       manual_cot_list, manual_cth_list, manual_cwp_list, manual_cer_list]:
        print(f"Processing cloud list with length: {len(cloud_list)}")
        cloud_list = np.array(cloud_list)
        cloud_list = cloud_list[~sat_cot_nan_mask]
        

    
    plot_cld_var(date_list, sat_cot_list, manual_cot_list, 
                 xlabel="COT", ylabel="Density",
                 title="Cloud Optical Thickness (COT)",
                 var_name='cot', fig_tag='test',)
    
    plot_cld_var(date_list, sat_cwp_list, manual_cwp_list, 
                 xlabel="LWP (g m$^{-2}$)", ylabel="Density",
                 title="Cloud Liquid Water Path (LWP)",
                 var_name='lwp', fig_tag='test', verbose=True,)
    
    plot_cld_var(date_list, sat_cth_list, manual_cth_list, 
                 xlabel="CTH (km)", ylabel="Density",
                 title="Cloud Top Height (CTH)",
                 var_name='cth', fig_tag='test',)
    
    plot_cld_var(date_list, sat_cer_list, manual_cer_list, 
                 xlabel="CER ($/mu$m)", ylabel="Density",
                 title="Cloud Effective Radius (CER)",
                 var_name='cer', fig_tag='test',)
    
 
    

if __name__ == '__main__':

   
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
    #                       case_tag='cloudy_track_1',
    #                       overwrite=True
    #                         )
    
    #/----------------------------------------------------------------------------\#
    
    # flt_trk_cld_2lay(dates=['20240603', '20240607', '20240606', '20240613'],
    #                       case_tag=['cloudy_track_1', 'cloudy_track_2', 'cloudy_track_2', 'cloudy_track_1'],
    #                       separate_heights=[600, 300, 300, 300],
    #                       fig_tag='test',
    #                       overwrite=False
    #                         )
    
    flt_trk_cld_2lay(dates=['20240603', '20240607', '20240606', '20240613'],
                          case_tag=['cloudy_track_1', 'cloudy_track_2', 'cloudy_track_2', 'cloudy_track_1'],
                          separate_heights=[600, 300, 300, 300],
                          fig_tag='test',
                          overwrite=False
                            )