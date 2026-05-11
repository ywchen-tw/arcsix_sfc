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


def pri_response_ori(fig_dir):
    
    nad_file = '../data/ssfr_cal/2024-03-29_lamp-1324|2024-03-29_lamp-150c_after-pri|2025-02-18_lamp-150c_post|2025-10-27_processed-for-arcsix|rad-resp|lasp|ssfr-a|nad|si-080|in-250|lamp-adjust.h5'
    zen_file = '../data/ssfr_cal/2024-03-29_lamp-1324|2024-03-29_lamp-150c_after-pri|2025-02-18_lamp-150c_post|2025-10-27_processed-for-arcsix|rad-resp|lasp|ssfr-a|zen|si-080|in-250|lamp-adjust.h5'
    
    with h5py.File(nad_file, 'r') as f:
        nad_si_wvl = f['raw/si/wvl'][:]
        nad_si_pri_resp = f['raw/si/pri_resp'][:]
        nad_in_wvl = f['raw/in/wvl'][:]
        nad_in_pri_resp = f['raw/in/pri_resp'][:]
        
    with h5py.File(zen_file, 'r') as f:
        zen_si_wvl = f['raw/si/wvl'][:]
        zen_si_pri_resp = f['raw/si/pri_resp'][:]
        zen_in_wvl = f['raw/in/wvl'][:]
        zen_in_pri_resp = f['raw/in/pri_resp'][:]
    
    labelsize = 12
    legendsize = 10
    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 4))
    linewidth = 1.0
    lin_args = {'linewidth': linewidth}
    ax.plot(nad_si_wvl, nad_si_pri_resp, label='nad-Si-pri-0', color='blue', **lin_args)
    ax.plot(zen_si_wvl, zen_si_pri_resp, label='zen-Si-pri-0', color='orange', **lin_args)
    ax.plot(nad_in_wvl, nad_in_pri_resp, label='nad-InGaAs-pri-0', color='green', **lin_args)
    ax.plot(zen_in_wvl, zen_in_pri_resp, label='zen-InGaAs-pri-0', color='red', **lin_args)
    ax.legend(fontsize=legendsize, ncol=2, loc='center left', bbox_to_anchor=(0.125, -0.25))
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)
    xmin = min(nad_si_wvl.min(), zen_si_wvl.min())
    xmax = max(nad_in_wvl.max(), zen_in_wvl.max())
    ax.set_xlim(xmin, xmax)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=labelsize)
    ax.set_ylabel('Primary Response\n(counts / ($\mathrm{W\,m^{-2}\,sr^{-1}\,nm^{-1}}$))', fontsize=labelsize)
    fig.savefig(os.path.join(fig_dir, 'pri_response_ori.png'), dpi=300, bbox_inches='tight')
    
    
def pri_response_ori2(fig_dir):
    
    nad_file = '../data/ssfr_cal/2024-03-29_lamp-1324|2024-03-29_lamp-150c_after-pri|2025-02-18_lamp-150c_post|2025-10-27_processed-for-arcsix|rad-resp|lasp|ssfr-a|nad|si-080|in-250|lamp-adjust.h5'
    zen_file = '../data/ssfr_cal/2024-03-29_lamp-1324|2024-03-29_lamp-150c_after-pri|2025-02-18_lamp-150c_post|2025-10-27_processed-for-arcsix|rad-resp|lasp|ssfr-a|zen|si-080|in-250|lamp-adjust.h5'
    
    with h5py.File(nad_file, 'r') as f:
        nad_si_wvl = f['raw/si/wvl'][:]
        nad_si_pri_resp = f['raw/si/pri_resp'][:]
        nad_in_wvl = f['raw/in/wvl'][:]
        nad_in_pri_resp = f['raw/in/pri_resp'][:]
        
    with h5py.File(zen_file, 'r') as f:
        zen_si_wvl = f['raw/si/wvl'][:]
        zen_si_pri_resp = f['raw/si/pri_resp'][:]
        zen_in_wvl = f['raw/in/wvl'][:]
        zen_in_pri_resp = f['raw/in/pri_resp'][:]
    
    labelsize = 12
    legendsize = 10
    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 4))
    linewidth = 1.0
    lin_args = {'linewidth': linewidth}
    ax.plot(nad_si_wvl, nad_si_pri_resp, label=r'$R_{nad, Si}^{pri\,(ori)}$', color='blue', **lin_args)
    ax.plot(zen_si_wvl, zen_si_pri_resp, label=r'$R_{zen, Si}^{pri\,(ori)}$', color='orange', **lin_args)
    ax.plot(nad_in_wvl, nad_in_pri_resp, label=r'$R_{nad, IR}^{pri\,(ori)}$', color='green', **lin_args)
    ax.plot(zen_in_wvl, zen_in_pri_resp, label=r'$R_{zen, IR}^{pri\,(ori)}$', color='red', **lin_args)
    ax.legend(fontsize=legendsize, ncol=4, loc='center left', bbox_to_anchor=(0.0, -0.225))
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)
    xmin = min(nad_si_wvl.min(), zen_si_wvl.min())
    xmax = max(nad_in_wvl.max(), zen_in_wvl.max())
    ax.set_xlim(xmin, xmax)
    
    ax.set_xlabel('Wavelength [nm]', fontsize=labelsize)
    ax.set_ylabel('Primary Response\n[counts / ($\mathrm{W\,m^{-2}\,sr^{-1}\,nm^{-1}}$)]', fontsize=labelsize)
    fig.savefig(os.path.join(fig_dir, 'pri_response_ori_2.png'), dpi=300, bbox_inches='tight')      





if __name__ == '__main__':

    
    dir_fig = './fig/SI'
    os.makedirs(dir_fig, exist_ok=True)
    
    # pri_response_ori(dir_fig)
    pri_response_ori2(dir_fig)