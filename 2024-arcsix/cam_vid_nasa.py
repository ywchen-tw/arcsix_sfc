"""
by Hong Chen (hong.chen@lasp.colorado.edu)

Flight video
"""

import os
import sys
import glob
import copy
import time
from collections import OrderedDict
import warnings
import datetime
import pickle
from tqdm import tqdm
import subprocess
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import h5py
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
from PIL import ImageFile
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.style as mplstyle
import cartopy
import cartopy.crs as ccrs


import ssfr

_MISSION_      = 'arcsix'
_FDIR_CAM_IMG_ = '/argus/field/%s/2024/p3' % _MISSION_
_CAM_          = 'nac'

_DATE_SPECS_ = {
        '20240517': {
            'tmhr_range': [19.20, 23.00],
           'description': 'ARCSIX Test Flight #1',
       'cam_time_offset': 0.0,
            },

        '20240521': {
            'tmhr_range': [14.80, 17.50],
           'description': 'ARCSIX Test Flight #2',
       'cam_time_offset': 0.0,
            },

        '20240524': {
            'tmhr_range': [ 9.90, 17.90],
           'description': 'ARCSIX Transit Flight #1',
       'cam_time_offset': 0.0,
            },

        '20240528': {
            'tmhr_range': [11.80, 18.70],
           'description': 'ARCSIX Science Flight #1',
       'cam_time_offset': 2.0,
            },

        '20240530': {
            'tmhr_range': [10.80, 18.40],
           'description': 'ARCSIX Science Flight #2',
       'cam_time_offset': 2.0,
            },

        '20240531': {
            'tmhr_range': [12.40, 19.50],
           'description': 'ARCSIX Science Flight #3',
       'cam_time_offset': 6.0,
            },

        '20240603': {
            'tmhr_range': [10.90, 18.10],
           'description': 'ARCSIX Science Flight #4',
       'cam_time_offset': 0.0,
            },

        '20240605': {
            'tmhr_range': [11.00, 18.90],
           'description': 'ARCSIX Science Flight #5',
       'cam_time_offset': 0.0,
            },

        '20240606': {
            'tmhr_range': [10.90, 19.90],
           'description': 'ARCSIX Science Flight #6',
       'cam_time_offset': 0.0,
            },

        '20240607': {
            'tmhr_range': [13.20, 19.00],
           'description': 'ARCSIX Science Flight #7',
       'cam_time_offset': 0.0,
            },

        '20240610': {
            'tmhr_range': [10.90, 19.00],
           'description': 'ARCSIX Science Flight #8',
       'cam_time_offset': 0.0,
            },

        '20240611': {
            'tmhr_range': [10.90, 18.90],
           'description': 'ARCSIX Science Flight #9',
       'cam_time_offset': 0.0,
            },

        '20240613': {
            'tmhr_range': [10.90, 19.90],
           'description': 'ARCSIX Science Flight #10',
       'cam_time_offset': 0.0,
            },

        '20240722': {
            'tmhr_range': [10.00, 19.00],
           'description': 'ARCSIX Transit Flight #3',
       'cam_time_offset': 0.0,
            },

        '20240725': {
            'tmhr_range': [11.40, 19.00],
           'description': 'ARCSIX Science Flight #11',
       'cam_time_offset': 0.0,
            },

        '20240729': {
            'tmhr_range': [11.00, 20.00],
           'description': 'ARCSIX Science Flight #12',
       'cam_time_offset': 0.0,
            },

        '20240730': {
            'tmhr_range': [11.00, 20.00],
           'description': 'ARCSIX Science Flight #13',
       'cam_time_offset': 0.0,
            },

        '20240801': {
            'tmhr_range': [11.00, 20.00],
           'description': 'ARCSIX Science Flight #14',
       'cam_time_offset': 0.0,
            },

        '20240802': {
            'tmhr_range': [11.00, 20.00],
           'description': 'ARCSIX Science Flight #15',
       'cam_time_offset': 0.0,
            },

        '20240807': {
            'tmhr_range': [11.00, 20.00],
           'description': 'ARCSIX Science Flight #16',
       'cam_time_offset': 0.0,
            },

        '20240808': {
            'tmhr_range': [11.00, 20.00],
           'description': 'ARCSIX Science Flight #17',
       'cam_time_offset': 0.0,
            },

        '20240809': {
            'tmhr_range': [11.00, 20.00],
           'description': 'ARCSIX Science Flight #18',
       'cam_time_offset': 0.0,
            },

        '20240815': {
            'tmhr_range': [11.00, 20.00],
           'description': 'ARCSIX Science Flight #19',
       'cam_time_offset': 0.0,
            },

        '20240816': {
            'tmhr_range': [11.00, 20.00],
           'description': 'ARCSIX Transit Flight #4',
       'cam_time_offset': 0.0,
            },
        }


def get_jday_cam_img(date, fnames):

    """
    Get UTC time in hour from the camera file name

    Input:
        fnames: Python list, file paths of all the camera jpg data

    Output:
        jday: numpy array, julian day
    """

    jday = []
    for fname in fnames:
        filename = os.path.basename(fname).split('.')[0]
        dtime_s_ = filename[:23].split(' ')[-1]
        dtime_s = '%s_%s' % (date.strftime('%Y_%m_%d'), dtime_s_)
        dtime0 = datetime.datetime.strptime(dtime_s, '%Y_%m_%d_%H_%M_%SZ')
        jday0 = ssfr.util.dtime_to_jday(dtime0)
        jday.append(jday0)

    return np.array(jday)

def convert_cam_img(fname_cam, jday_cam):

    # ax_nad_img
    #╭────────────────────────────────────────────────────────────────────────────╮#
    ang_cam_offset = -53.0 # for ARCSIX
    cam_x_s = 5.0
    cam_x_e = 255.0*4.0
    cam_y_s = 0.0
    cam_y_e = 0.12

    img = mpl_img.imread(fname_cam)[:-200, 540:-640, :]

    img = ndimage.rotate(img, ang_cam_offset, reshape=False)
    # img_plot = img.copy()
    # img_plot[img_plot>255] = 255
    # img_plot = np.int_(img_plot)

    dtime_cam = ssfr.util.jday_to_dtime(jday_cam)
    fname_jpg = 'cam-rot_%s.jpg' % dtime_cam.strftime('%Y-%m-%d_%H:%M:%S')
    plt.imsave(fname_jpg, img, format='jpg', dpi=300)
    #╰────────────────────────────────────────────────────────────────────────────╯#

def main():

    dates = [
            # datetime.datetime(2024, 5, 28), # [✓]
            # datetime.datetime(2024, 6, 5), # [✓]
            datetime.datetime(2024, 6, 6), # [✓]
            # datetime.datetime(2024, 7, 29), # [✓]
            datetime.datetime(2024, 7, 30), # [✓]
        ]

    tmhr_ranges= {
            '20240528': [15.6139, 15.8153],
            '20240605': [14.6139, 14.8375],
            '20240606': [16.7181, 16.7833],
            '20240729': [13.9472, 14.1778],
            '20240730': [14.8723, 14.9375],
            }

    for date in dates:
        date_s = date.strftime('%Y%m%d')
        tmhr_range = tmhr_ranges[date_s]

        # process camera imagery
        #╭────────────────────────────────────────────────────────────────────────────╮#
        pattern = '*%4.4d*%2.2d*%2.2d*%s*jpg*' % (date.year, date.month, date.day, _CAM_)
        fdirs = ssfr.util.get_all_folders(_FDIR_CAM_IMG_, pattern='*%4.4d*%2.2d*%2.2d*%s*jpg*' % (date.year, date.month, date.day, _CAM_))
        if len(fdirs) > 0:
            fdir_cam0 = sorted(fdirs, key=os.path.getmtime)[-1]
            fnames_cam0 = sorted(glob.glob('%s/*.jpg' % (fdir_cam0)))
            if len(fnames_cam0) > 0:
                has_cam = True
                jday_cam0 = get_jday_cam_img(date, fnames_cam0) + _DATE_SPECS_[date_s]['cam_time_offset']/86400.0
                tmhr_cam0 = (jday_cam0 - ssfr.util.dtime_to_jday(date)) * 24.0
                indices = np.where((tmhr_cam0>=tmhr_range[0]) & (tmhr_cam0<=tmhr_range[1]))[0]
                for index in indices:
                    fname_cam = fnames_cam0[index]
                    convert_cam_img(fname_cam, jday_cam0[index])
            else:
                has_cam = False
        else:
            has_cam = False
        #╰────────────────────────────────────────────────────────────────────────────╯#


if __name__ == '__main__':

    main()
