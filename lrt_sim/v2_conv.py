import os
import sys
import glob
import datetime
import h5py
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.crs as ccrs
# mpl.use('Agg')



def main():

    #####
    # convert final v2 data to R0 format
    #####
    ssfr_v02_fname = '/Users/yuch8913/programming/ssfr_arcsix/ssfr/v2_conv/ARCSIX-SSFR_P3B_20240605_v2.h5'
    with h5py.File(ssfr_v02_fname, 'r') as f:
        alt = f['alt'][:]
        lon = f['lon'][:]
        lat = f['lat'][:]
        time = f['tmhr'][:] * 3600.0 # convert to seconds
        wvl_dn = f['zen/wvl'][:]
        wvl_up = f['nad/wvl'][:]
        f_dn = f['zen/flux'][:,:]
        f_up = f['nad/flux'][:,:]
    
    with h5py.File(ssfr_v02_fname.replace('_v2', '_R5_V2_test_after_corr'), 'w') as f:
        f.create_dataset('gps_alt', data=alt)
        f.create_dataset('gps_lon', data=lon)
        f.create_dataset('gps_lat', data=lat)
        f.create_dataset('time', data=time)
        f.create_dataset('wvl_dn', data=wvl_dn)
        f.create_dataset('wvl_up', data=wvl_up)
        f.create_dataset('f_dn', data=f_dn)
        f.create_dataset('f_up', data=f_up)


if __name__ == '__main__':
    main()