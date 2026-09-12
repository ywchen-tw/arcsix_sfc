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

def sfc_alb_plot(alb_dir, fnames_alb):
    
    """
    Plot surface albedo from the given file names.
    
    Parameters:
    -----------
    fnames_alb : list of str
        List of file names containing surface albedo data.
    """
    
    
    
    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    
    for fname in fnames_alb:
        basename = os.path.basename(fname)
        date = basename.split('_')[2].split('.')[0]
        mm = int(date[4:6])
        dd = int(date[6:8])
        date_str = f"{mm:d}/{dd:d}"
        
        df_alb = pd.read_csv(f'{alb_dir}/{fname}', sep='\s+', header=None)
        df_alb.drop(df_alb.index[0:2], inplace=True)  # Remove the first two row if it is a header
        # only keep the first two columns
        df_alb = df_alb.iloc[:, :2]
        df_alb.reset_index(drop=True, inplace=True)
        # Convert the first column to numeric values
        df_alb.iloc[:, 0] = pd.to_numeric(df_alb.iloc[:, 0], errors='coerce')
        df_alb.iloc[:, 1] = pd.to_numeric(df_alb.iloc[:, 1], errors='coerce')
        wvl = np.array(df_alb.iloc[:, 0])
        alb = np.array(df_alb.iloc[:, 1])#/1000 # convert mW/m^2/nm to W/m^2/nm
        
        
        wvl_mask_1 = (wvl >= 900) & (wvl <= 1020)
        alb[wvl_mask_1] = np.nan  # Mask out the 1.2-0.89 um region
        wvl_mask_2 = (wvl >= 1320) & (wvl <= 1420)
        alb[wvl_mask_2] = np.nan  # Mask out the 1.2-0.89 um region
        wvl_mask_3 = (wvl >= 1800) & (wvl <= 2000)
        alb[wvl_mask_3] = np.nan
        
        ax.plot(wvl, alb, label=date_str, linewidth=2.5)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=label_size)
    ax.set_ylabel('Albedo', fontsize=label_size)
    # ax.set_title('Surface Albedo', fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=label_size-2)
    ax.set_xlim(400, 1800)
    ax.legend(fontsize=legend_size)
    # ax.grid(True)
    
    fig.tight_layout()
    fig.savefig('sfc_alb_plot.png', dpi=300)
    plt.close(fig)
    

if __name__ == '__main__':

   sfc_alb_plot(alb_dir='data/sfc_alb', 
                fnames_alb=["sfc_alb_20240531.dat",
                            "sfc_alb_20240605.dat",
                            "sfc_alb_20240606.dat",
                            "sfc_alb_20240607.dat",
                            "sfc_alb_20240613.dat",])