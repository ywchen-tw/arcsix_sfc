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


# _test_ = True
_test_ = False
_time_interval_ = 60
if not _test_:
    mplstyle.use('fast')
    mpl.use('Agg')


ImageFile.LOAD_TRUNCATED_IMAGES = True
_Ncpu_ = 12

_mission_      = 'arcsix'
_platform_     = 'p3b'

_hsk_          = 'hsk'
_alp_          = 'alp'
_spns_         = 'spns-b'
_ssfr_         = 'ssfr-b'
_ssrr_         = 'ssfr-a'
_cam_          = 'nac'

_fdir_main_       = 'data/%s/flt-vid' % _mission_
_fdir_sat_img_    = 'data/%s/sat-img' % _mission_
_fdir_cam_img_    = 'data/%s/2024/p3' % _mission_
_fdir_lid_img_    = 'data/%s/lid-img' % _mission_
_wavelength_      = 555.0

_fdir_data_ = 'data/%s/processed' % _mission_

_fdir_sat_img_vn_ = 'data/%s/sat-img-vn' % _mission_
_fdir_sat_img_hc_ = 'data/%s/sat-img-hc' % _mission_

_fdir_tmp_ = './tmp-data'
_delete_tmp_ = True
_delete_ask_ = False


_date_specs_ = {
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
        }

_extent_ll_study_region_ = [-80.0000, -30.0000, 71.0000, 88.00]
_extent_xy_study_region_ = [-877574.55,877574.55,-751452.90,963254.75]

def cal_proj_xy_extent(extent, closed=True):

    """
    Calculate globe map projection <ccrs.Orthographic> centered at the center of the granule defined by corner points

    Input:
        line_data: Python dictionary (details see <read_geo_meta>) that contains basic information of a satellite granule
        closed=True: if True, return five corner points with the last point repeating the first point;
                     if False, return four corner points

    Output:
        proj_xy: globe map projection <ccrs.Orthographic> centered at the center of the granule defined by corner points
        xy: dimension of (5, 2) if <closed=True> and (4, 2) if <closed=False>
    """

    import cartopy.crs as ccrs

    line_data = {
            'GRingLongitude1': extent[0],
            'GRingLongitude2': extent[1],
            'GRingLongitude3': extent[1],
            'GRingLongitude4': extent[0],
             'GRingLatitude1': extent[2],
             'GRingLatitude2': extent[2],
             'GRingLatitude3': extent[3],
             'GRingLatitude4': extent[3],
            }

    # get corner points
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon_  = np.array([
        float(line_data['GRingLongitude1']),
        float(line_data['GRingLongitude2']),
        float(line_data['GRingLongitude3']),
        float(line_data['GRingLongitude4']),
        float(line_data['GRingLongitude1'])
        ])

    lat_  = np.array([
        float(line_data['GRingLatitude1']),
        float(line_data['GRingLatitude2']),
        float(line_data['GRingLatitude3']),
        float(line_data['GRingLatitude4']),
        float(line_data['GRingLatitude1'])
        ])

    if (abs(lon_[0]-lon_[1])>180.0) | (abs(lon_[0]-lon_[2])>180.0) | \
       (abs(lon_[0]-lon_[3])>180.0) | (abs(lon_[1]-lon_[2])>180.0) | \
       (abs(lon_[1]-lon_[3])>180.0) | (abs(lon_[2]-lon_[3])>180.0):

        lon_[lon_<0.0] += 360.0
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # roughly determine the center of granule
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon = lon_[:-1]
    lat = lat_[:-1]
    center_lon_ = lon.mean()
    center_lat_ = lat.mean()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # find the true center
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj_lonlat = ccrs.PlateCarree()

    proj_xy_ = ccrs.Orthographic(central_longitude=center_lon_, central_latitude=center_lat_)
    xy_ = proj_xy_.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]

    center_x  = xy_[:, 0].mean()
    center_y  = xy_[:, 1].mean()
    center_lon, center_lat = proj_lonlat.transform_point(center_x, center_y, proj_xy_)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # convert lon/lat corner points into xy
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj_xy = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    xy_  = proj_xy.transform_points(proj_lonlat, lon_, lat_)[:, [0, 1]]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    if closed:
        return proj_xy, xy_
    else:
        return proj_xy, xy_[:-1, :]

def contain_lonlat_check(
             lon,
             lat,
             extent,
             ):

    # check cartopy and matplotlib
    #╭────────────────────────────────────────────────────────────────────────────╮#
    import cartopy.crs as ccrs
    import matplotlib.path as mpl_path
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # convert longitude in [-180, 180] range
    # since the longitude in GeoMeta dataset is in the range of [-180, 180]
    # or check overlap within region of interest
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon[lon>180.0] -= 360.0
    logic = (lon>=-180.0)&(lon<=180.0) & (lat>=-90.0)&(lat<=90.0)
    lon   = lon[logic]
    lat   = lat[logic]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # loop through all the satellite "granules" constructed through four corner points
    # and find which granules contain the input data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj_lonlat = ccrs.PlateCarree()

    # get bounds of the satellite overpass/granule
    proj_xy, xy_granule = cal_proj_xy_extent(extent, closed=True)
    sat_granule  = mpl_path.Path(xy_granule, closed=True)

    # check if the overpass/granule overlaps with region of interest
    xy_in      = proj_xy.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]
    points_in  = sat_granule.contains_points(xy_in)

    Npoint_in  = points_in.sum()

    if (Npoint_in>0):
        contain = True
    else:
        contain = False
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return contain

def check_continuity(data, threshold=0.1):

    data = np.append(data[0], data)

    return (np.abs(data[1:]-data[:-1]) < threshold)

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

def get_jday_sat_img_vn(fnames):

    """
    Get UTC time in hour from the satellite file name

    Input:
        fnames: Python list, file paths of all the satellite data

    Output:
        jday: numpy array, julian day
    """

    jday = []
    for fname in fnames:
        filename = os.path.basename(fname)
        strings  = filename.split('_')
        dtime_s  = strings[2]

        # dtime0 = datetime.datetime.strptime(dtime_s, '%Y-%m-%dT%H:%M:%SZ')
        dtime0 = datetime.datetime.strptime(dtime_s, '%Y-%m-%d-%H%M%SZ')
        jday0 = ssfr.util.dtime_to_jday(dtime0)
        jday.append(jday0)

    return np.array(jday)

def process_sat_img_overlay(fnames_sat_, max_overlay=12):

    """
    lincoln_sea/VIIRS-NOAA-21_TrueColor_2024-05-31-094200Z_(-120.00,36.69,77.94,88.88).png
    """

    jday_sat_ = get_jday_sat_img_vn(fnames_sat_)
    jday_sat_unique = np.sort(np.unique(jday_sat_))

    fnames_sat = []
    jday_sat = []

    # plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    extent_plot = [-80.00, -30.00, 71.00, 88.00]
    extent_plot_xy = [-877574.55, 877574.55, -751452.90, 963254.75]

    lon_c = (extent_plot[0]+extent_plot[1])/2.0
    lat_c = (extent_plot[2]+extent_plot[3])/2.0
    proj_plot = ccrs.Orthographic(
            central_longitude=lon_c,
            central_latitude=lat_c,
            )
    plt.close('all')
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(111, projection=proj_plot)

    ax1.coastlines(resolution='10m', color='gray', lw=0.5, zorder=500)
    ax1.set_extent(extent_plot, crs=ccrs.PlateCarree())

    ax1.axis('off')
    #╰────────────────────────────────────────────────────────────────────────────╯#

    imshows0 = []
    imshows1 = []
    for i, jday_sat0 in enumerate(tqdm(jday_sat_unique)):

        indices = np.where(jday_sat_==jday_sat0)[0]
        fname0 = sorted([fnames_sat_[index] for index in indices])[-1]

        filename = os.path.basename(fname0)
        info = filename.replace('.png', '').split('_')
        extent = [float(item) for item in info[-1].replace('(', '').replace(')', '').split(',')]
        extent_xy = [float(item) for item in info[-2].replace('(', '').replace(')', '').split(',')]

        dtime_s = ssfr.util.jday_to_dtime(jday_sat0).strftime('%Y-%m-%d_%H:%M:%S')
        sat_tag = info[0].replace('TERRA', 'Terra').replace('AQUA', 'Aqua').replace('SUOMI', 'Suomi').replace('MODIS-', 'MODIS_').replace('VIIRS-', 'VIIRS_')
        img_tag = info[1]
        fname0_out = '%s/%s_%s_%s' % (_fdir_sat_img_hc_, img_tag, dtime_s, sat_tag)

        try:
            img = mpl_img.imread(fname0)

            logic_tran = (img[:, :, 0]==1.0) & (img[:, :, 1]==1.0) & (img[:, :, 2]==1.0)

            img_bkg = np.ones_like(img[:, :, 0], dtype=np.float32)
            img[logic_tran, 3] = 0.0
            img_bkg[logic_tran] = np.nan

            imshow0 = ax1.imshow(img_bkg, cmap='Greys_r', vmin=0.0, vmax=1.5, extent=extent_xy, interpolation='nearest', zorder=i+1, alpha=0.0)
            imshow1 = ax1.imshow(img, extent=extent_xy, interpolation='nearest', zorder=i)

            imshows0.append(imshow0)
            imshows1.append(imshow1)

            if len(imshows0) > 1:
                imshows0[-2].set_alpha(0.5)
            if len(imshows0) > max_overlay:
                temp0 = imshows0.pop(0)
                temp0.remove()
            if len(imshows1) > max_overlay:
                temp1 = imshows1.pop(0)
                temp1.remove()

            # save figure
            #╭──────────────────────────────────────────────────────────────╮#
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            fname0_out = '%s_(%.2f,%.2f,%.2f,%.2f)_(%.4f,%.4f,%.4f,%.4f).jpg' % (fname0_out, *ax1.get_xlim(), *ax1.get_ylim(), *extent_plot)
            plt.savefig(fname0_out, bbox_inches='tight', pad_inches=0)
            #╰──────────────────────────────────────────────────────────────╯#

            fnames_sat.append(fname0_out)

            print(fname0_out)

        except Exception as error:
            print(fname0)
            warnings.warn(error)

    return np.array(jday_sat), fnames_sat

def process_marli(date, run=True, test=False):

    date_s = date.strftime('%Y%m%d')

    try:
    # if True:
        fname = sorted(ssfr.util.get_all_files('data/arcsix/2024/p3/aux/marli', pattern='*%s*.cdf' % (date_s)))[-1]

        fname_hsk = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)
        f_hsk = h5py.File(fname_hsk, 'r')
        tmhr = f_hsk['tmhr'][...]
        alt  = f_hsk['alt'][...]
        f_hsk.close()

        # read marli
        #╭────────────────────────────────────────────────────────────────────────────╮#
        f = Dataset(fname, 'r')
        tmhr_1d = np.array(f.variables['time'][...].data, dtype=np.float64)
        h_1d = np.array(f.variables['H'][...].data*1000.0, dtype=np.float64)
        data_2d = np.array(f.variables['LSR'][...].data, dtype=np.float64)
        f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        data_2d[data_2d<=0.0] = np.nan

        tmhr_2d, h_2d = np.meshgrid(tmhr_1d, h_1d, indexing='ij')

        h_2d_new = np.zeros_like(h_2d)
        h_2d_new[...] = np.nan
        data_2d_new = np.zeros_like(data_2d)
        data_2d_new[...] = np.nan
        for i in range(tmhr_1d.size):
            h_1d_new = h_2d[i, :] + np.interp(tmhr_1d[i], tmhr, alt)
            logic = h_1d_new >= 0.0

            h_2d_new[i, 0:logic.sum()] = h_1d_new[logic]
            data_2d_new[i, 0:logic.sum()] = data_2d[i, logic]

        h_nan = np.sum(np.isnan(h_2d_new), axis=-1)
        h_1d_new = h_2d_new[np.argmin(h_nan), :]

        indices_nan = np.where(np.isnan(h_1d_new))[0]
        if indices_nan.size > 0:
            dh = h_1d_new[indices_nan[0]-1]-h_1d_new[indices_nan[0]-2]
            h_1d_new[indices_nan] = h_1d_new[indices_nan[0]-1] + dh*(indices_nan-indices_nan[0]+1)

        tmhr_2d_new, h_2d_new = np.meshgrid(tmhr_1d, h_1d_new, indexing='ij')
        h_2d_new /= 1000.0

        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if run:
            plt.close('all')
            fig = plt.figure(figsize=(24, 4))
            # plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax1 = fig.add_subplot(111)
            cs = ax1.pcolormesh(tmhr_2d_new, h_2d_new, data_2d_new, cmap='viridis', zorder=0, vmin=0.0, vmax=100.0) #, extent=extent, vmin=0.0, vmax=0.5)
            ax1.axis('off')
            ax1.set_xlim((np.nanmin(tmhr_2d_new), np.nanmax(tmhr_2d_new)))
            ax1.set_ylim((np.nanmin(h_2d_new), np.nanmax(h_2d_new)))
            #╰──────────────────────────────────────────────────────────────╯#

            if not test:
                extent_tag = '(%.4f,%.4f,%.4f,%.4f)' % (np.nanmin(tmhr_2d_new), np.nanmax(tmhr_2d_new), np.nanmin(h_2d_new), np.nanmax(h_2d_new))
                fname_png = '%s/MARLI-P3B_LSR_%s_%s.png' % (_fdir_lid_img_, date.strftime('%Y-%m-%d'), extent_tag)
                # save figure
                #╭──────────────────────────────────────────────────────────────╮#
                _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                plt.savefig(fname_png, bbox_inches='tight', metadata=_metadata, pad_inches=0.0, dpi=200)
                #╰──────────────────────────────────────────────────────────────╯#
                plt.close(fig)
                plt.clf()
            else:
                plt.show()
                sys.exit()
        #╰────────────────────────────────────────────────────────────────────────────╯#
    except Exception as error:
        # warnings.warn(error)
        print(error)
        fname_png = None

    return fname_png




# for science flights in the Arctic
#╭────────────────────────────────────────────────────────────────────────────╮#
def plot_video_frame(statements, test=False):

    if test:
        time_s = time.time()

    # extract arguments
    #╭────────────────────────────────────────────────────────────────────────────╮#
    shared_data, unshared_data, index_pnt = statements

    data_plt = {}
    shm_tmp = {}
    for key in shared_data.keys():
        shm_tmp[key] = SharedMemory(name=shared_data[key]['shm_name'])
        data_plt[key] = np.ndarray(shared_data[key]['shape'], dtype=shared_data[key]['dtype'], buffer=shm_tmp[key].buf)
    for key in unshared_data.keys():
        data_plt[key] = unshared_data[key]

    vnames_plt = data_plt.keys()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # param settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tmhr_current = data_plt['tmhr'][index_pnt]
    jday_current = data_plt['jday'][index_pnt]
    lon_current  = data_plt['lon'][index_pnt]
    lat_current  = data_plt['lat'][index_pnt]
    alt_current  = data_plt['alt'][index_pnt]
    sza_current  = data_plt['sza'][index_pnt]
    dtime_current = ssfr.util.jday_to_dtime(jday_current)

    tmhr_length  = 0.5 # half an hour
    tmhr_past    = tmhr_current-tmhr_length
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # general plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vars_plot = OrderedDict()

    vars_plot['SSFR↑']   = {
            'vname':'f-up_ssfr',
            'color':'red',
            'vname_wvl':'wvl_ssfr_nad',
            'zorder': 18,
            }
    vars_plot['SSFR↓']   = {
            'vname':'f-down_ssfr',
            'color':'blue',
            'vname_wvl':'wvl_ssfr_zen',
            'zorder': 17,
            }
    vars_plot['Albedo']   = {
            'vname':'alb_ssfr',
            'vname_mean':'alb-mean_ssfr',
            'vname_std':'alb-std_ssfr',
            'vname_min':'alb-min_ssfr',
            'vname_max':'alb-max_ssfr',
            'color':'deeppink',
            'vname_wvl':'wvl_ssfr_alb',
            'zorder': 16,
            }
    vars_plot['SPNS$_{Total}$↓']   = {
            'vname':'f-down-total_spns',
            'color':'green',
            'vname_wvl':'wvl_spns',
            'zorder': 15,
            }
    vars_plot['SPNS$_{Diffuse}$↓']   = {
            'vname':'f-down-diffuse_spns',
            'color':'springgreen',
            'vname_wvl':'wvl_spns',
            'zorder': 14,
            }
    vars_plot['TOA↓']   = {
            'vname':'f-down_toa',
            'color':'dimgray',
            'vname_wvl':'wvl_ssfr_zen',
            'zorder': 12,
            }
    vars_plot['KT19 T↑']   = {
            'vname':'t-up_kt19',
            'color':'none',
            'zorder': 11,
            }
    vars_plot['Altitude']   = {
            'vname':'alt',
            'color':'orange',
            'zorder': 10,
            }

    for vname in vars_plot.keys():

        vname_ori = vars_plot[vname]['vname']
        if vname_ori in vnames_plt:
            vars_plot[vname]['plot?'] = True
        else:
            vars_plot[vname]['plot?'] = False

        if 'vname_wvl' in vars_plot[vname].keys():
            vars_plot[vname]['spectra?'] = True
        else:
            vars_plot[vname]['spectra?'] = False
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    _aspect_    = 'auto'
    _alt_cmap_  = 'gist_ncar'
    _temp_cmap_ = 'seismic'
    _dpi_       = 150

    _alt_base_ = 0.0
    _alt_ceil_ = 8.0
    _temp_base_ = -20.0
    _temp_ceil_ = 20.0

    _flux_base_ = 0.0
    _flux_ceil_ = 1.5
    _title_extra_ = _date_specs_[dtime_current.strftime('%Y%m%d')]['description']

    hist_bins = np.linspace(0.0, 2.0, 81)
    hist_x = (hist_bins[1:]+hist_bins[:-1])/2.0
    hist_bin_w = hist_bins[1]-hist_bins[0]
    hist_bottoms = {key:0.0 for key in vars_plot.keys()}

    alt_cmap = mpl.colormaps[_alt_cmap_]
    alt_norm = mpl.colors.Normalize(vmin=_alt_base_, vmax=_alt_ceil_)

    temp_cmap = mpl.colormaps[_temp_cmap_]
    temp_norm = mpl.colors.Normalize(vmin=_temp_base_, vmax=_temp_ceil_)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # check data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    has_trk = True

    if ('ang_pit' in vnames_plt) and ('ang_rol' in vnames_plt):
        has_att = True
    else:
        has_att = False

    if ('ang_pit_m' in vnames_plt) and ('ang_rol_m' in vnames_plt):
        has_att_corr = True
    else:
        has_att_corr = False

    if ('t-up_kt19' in  vnames_plt):
        has_kt19 = True
        kt19_current = data_plt['t-up_kt19'][index_pnt]
    else:
        has_kt19 = False

    if ('fnames_sat0' in vnames_plt):
        has_sat0 = True
    else:
        has_sat0 = False

    if ('fnames_sat1' in vnames_plt):
        has_sat1 = True
    else:
        has_sat1 = False

    if ('fnames_cam0' in vnames_plt):
        has_cam0 = True
    else:
        has_cam0 = False

    if ('fnames_lid0' in vnames_plt):
        has_lid0 = True
    else:
        has_lid0 = False

    has_spectra = any([vars_plot[key]['plot?'] for key in vars_plot.keys() if vars_plot[key]['spectra?']])
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # figure setup
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fig = plt.figure(figsize=(16, 9))

    gs = gridspec.GridSpec(12, 20)

    # ax of all
    ax_all = fig.add_subplot(gs[:, :])

    # map of flight track overlay satellite imagery
    proj0 = ccrs.Orthographic(
            central_longitude=(_extent_ll_study_region_[0]+_extent_ll_study_region_[1])/2.0,
            central_latitude=(_extent_ll_study_region_[2]+_extent_ll_study_region_[3])/2.0,
            )
    ax_map = fig.add_subplot(gs[:8, :9], projection=proj0, aspect=_aspect_)

    # altitude colorbar next to the map
    divider_map = make_axes_locatable(ax_map)
    ax_alt_cbar = divider_map.append_axes('right', size='4%', pad=0.0, axes_class=maxes.Axes)

    # profile (shared y axis) next to the map
    if has_spectra:
        ax_alt_prof = divider_map.append_axes('right', size='32%', pad=0.0, axes_class=maxes.Axes)
    else:
        ax_alt_prof = divider_map.append_axes('right', size='0.001%', pad=0.0, axes_class=maxes.Axes)

    # data histogram (shared x axis) next to the map
    ax_alt_hist = ax_alt_prof.twinx()

    # a secondary map
    ax_map0 = fig.add_subplot(gs[:5, 10:15], projection=proj0, aspect=_aspect_)

    # camera imagery
    ax_img  = fig.add_subplot(gs[:5, 15:])
    ax_img_hist = ax_img.twinx()

    # spetral irradiance
    ax_wvl  = fig.add_subplot(gs[5:8, 10:])

    # albedo
    ax_wvl0 = ax_wvl.twinx()

    # aircraft and platform attitude status
    ax_nav  = inset_axes(ax_wvl, width=1.0, height=0.7, loc='upper right')

    # time series
    ax_tms = fig.add_subplot(gs[9:, :])
    if has_att:
        ax_tms_ = ax_tms.twinx()
    ax_tms_alt  = ax_tms.twinx()

    fig.subplots_adjust(hspace=10.0, wspace=10.0)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # base plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    ax_map.set_extent(_extent_ll_study_region_, crs=ccrs.PlateCarree())
    if has_sat0:

        fname_sat0 = data_plt['fnames_sat0'][index_pnt]
        img_sat0 = mpl_img.imread(fname_sat0).copy()

        if has_sat1:

            fname_sat1 = data_plt['fnames_sat1'][index_pnt]
            img_sat1 = mpl_img.imread(fname_sat1)
            Ny, Nx, Nc = img_sat1.shape
            x_current, y_current = proj0.transform_point(lon_current, lat_current, ccrs.PlateCarree())
            x_1d_ = np.linspace(data_plt['extent_sat1'][0], data_plt['extent_sat1'][1], Nx+1)
            y_1d_ = np.linspace(data_plt['extent_sat1'][3], data_plt['extent_sat1'][2], Ny+1)
            x_1d = (x_1d_[1:]+x_1d_[:-1])/2.0
            y_1d = (y_1d_[1:]+y_1d_[:-1])/2.0

            dx = x_1d[1]-x_1d[0]
            dy = y_1d[1]-y_1d[0]

            extend_Nx = 200
            extend_Ny = 150

            index_x = int((x_current-(x_1d[0]-dx/2.0))//dx)
            index_y = int((y_current-(y_1d[0]+dy/2.0))//dy)

            index_xs = max(index_x-extend_Nx, 0)
            index_xe = min(index_x+extend_Nx, Nx-1)
            index_ys = max(index_y-extend_Ny, 0)
            index_ye = min(index_y+extend_Ny, Ny-1)

            extent_sat1 = [x_1d[index_xs]-dx/2.0, x_1d[index_xe]+dx/2.0, y_1d[index_ye]+dy/2.0, y_1d[index_ys]-dy/2.0]

            img_sat1 = img_sat1[index_ys:index_ye, index_xs:index_xe, :]
            ax_map0.imshow(img_sat1, extent=extent_sat1, aspect='auto', zorder=1)

            # plus minus one to avoid white padding from imshow
            ax_map0.set_xlim([x_current-dx*(extend_Nx-1), x_current+dx*(extend_Nx-1)])
            ax_map0.set_ylim([y_current+dy*(extend_Ny-1), y_current-dy*(extend_Ny-1)])

            img_sat0[index_ys+1:index_ye-1, index_xs+1:index_xe-1, :] = img_sat1[1:-1, 1:-1, :]

        ax_map.imshow(img_sat0, extent=data_plt['extent_sat0'], aspect='auto', zorder=0)

    if has_cam0:
        ang_cam_offset = -53.0 # for ARCSIX
        cam_x_s = 5.0
        cam_x_e = 255.0*4.0
        cam_y_s = 0.0
        cam_y_e = 0.12
        cam_hist_x_s = 0.0
        cam_hist_x_e = 255.0
        cam_hist_bins = np.linspace(cam_hist_x_s, cam_hist_x_e, 31)

        fname_cam = data_plt['fnames_cam0'][index_pnt]
        img = mpl_img.imread(fname_cam)[:-200, 540:-640, :]

        img = ndimage.rotate(img, ang_cam_offset, reshape=False)
        img_plot = img.copy()
        img_plot[img_plot>255] = 255
        img_plot = np.int_(img_plot)
        ax_img.imshow(img_plot, origin='upper', aspect='auto', zorder=0, extent=[cam_x_s, cam_x_e, cam_y_s, cam_y_e])

        ax_img_hist.hist(img[:, :, 0].ravel(), bins=cam_hist_bins, histtype='step', lw=0.5, alpha=0.9, density=True, color='r')
        ax_img_hist.hist(img[:, :, 1].ravel(), bins=cam_hist_bins, histtype='step', lw=0.5, alpha=0.9, density=True, color='g')
        ax_img_hist.hist(img[:, :, 2].ravel(), bins=cam_hist_bins, histtype='step', lw=0.5, alpha=0.9, density=True, color='b')
        ax_img.plot([255, 255], [0, 0.005], color='white', lw=1.0, ls='-')

    if has_att:

        ax_nav.axhspan(-10.0, 0.0, lw=0.0, color='blanchedalmond', zorder=5, alpha=1.0)
        ax_nav.axhspan(0.0,  10.0, lw=0.0, color='aliceblue', zorder=5, alpha=1.0)

        ax_nav.plot([0.0, 0.0], [-6.5, 10.0], lw=0.5, color='gray', zorder=6)
        ax_nav.axhline(0.0, lw=0.5, color='gray', zorder=6)

        ang_pit0 = data_plt['ang_pit'][index_pnt]
        ang_rol0 = data_plt['ang_rol'][index_pnt]

        x  = np.linspace(-10.0, 10.0, 101)

        slope0  = -np.tan(np.deg2rad(ang_rol0))
        offset0 = ang_pit0
        y0 = slope0*x + offset0

        ax_nav.plot(x[15:-15], y0[15:-15], lw=1.0, color='red', zorder=6, alpha=0.6)
        ax_nav.scatter(x[50], y0[50], lw=0.0, s=40, c='red', zorder=6, alpha=0.6)
        ax_nav.text(-9.5, 9.2, 'P:%+4.1f$^\\circ$' % ang_pit0, ha='left', va='top', color='red', zorder=7, fontsize=6)
        ax_nav.text( 3.7, 9.2, 'R:%+4.1f$^\\circ$' % ang_rol0, ha='left', va='top', color='red', zorder=7, fontsize=6)

        if has_att_corr:

            ang_pit_offset = 0.0
            ang_rol_offset = 0.0

            ang_pit0 = data_plt['ang_pit_s'][index_pnt]
            ang_rol0 = data_plt['ang_rol_s'][index_pnt]
            ang_pit_m0 = data_plt['ang_pit_m'][index_pnt]
            ang_rol_m0 = data_plt['ang_rol_m'][index_pnt]
            rel_hum0   = data_plt['rel_hum'][index_pnt]

            ang_pit_stage0 = ang_pit0-ang_pit_m0+ang_pit_offset
            ang_rol_stage0 = ang_rol0-ang_rol_m0+ang_rol_offset
            slope1  = -np.tan(np.deg2rad(ang_rol_stage0))
            offset1 = (ang_pit_stage0)
            y1 = slope1*x + offset1

            ax_nav.plot(x[25:-25], y1[25:-25], lw=2.0, color='green', zorder=7, alpha=0.7)
            if ~np.isnan(ang_pit_stage0) and ~np.isnan(ang_rol_stage0):
                ax_nav.text(-9.5, -9.8, 'P:%+4.1f$^\\circ$' % ang_pit_stage0, ha='left', va='bottom', color='green', zorder=7, fontsize=6)
                ax_nav.text( 3.7, -9.8, 'R:%+4.1f$^\\circ$' % ang_rol_stage0, ha='left', va='bottom', color='green', zorder=7, fontsize=6)

            if ~np.isnan(rel_hum0):
                if rel_hum0 < 60.0:
                    ax_nav.text(0.0, -9.8, '%4.1f%%' % rel_hum0, ha='center', va='bottom', color='black', zorder=7, fontsize=6)
                else:
                    ax_nav.text(0.0, -9.8, '%4.1f%%' % rel_hum0, ha='center', va='bottom', color='red', zorder=7, fontsize=6)

    for vname in vars_plot.keys():

        var_plot = vars_plot[vname]
        if var_plot['plot?']:
            if 'vname_wvl' in var_plot.keys():
                wvl_x  = data_plt[var_plot['vname_wvl']]
                if 'toa' in var_plot['vname']:
                    spec_y = data_plt[var_plot['vname']] * np.cos(np.deg2rad(sza_current))
                else:
                    spec_y = data_plt[var_plot['vname']][index_pnt, :]

                if vname != 'Albedo':
                    ax_wvl.plot(wvl_x, spec_y,
                            color=var_plot['color'], marker='o', markersize=1.2, lw=0.5, markeredgewidth=0.0, alpha=0.85, zorder=var_plot['zorder'])

                    wvl_index = np.argmin(np.abs(wvl_x-_wavelength_))
                    ax_wvl.axvline(wvl_x[wvl_index], color=var_plot['color'], ls='-', lw=1.0, alpha=0.5, zorder=var_plot['zorder'])

                    if ('f-down-diffuse_spns' == var_plot['vname'].lower()) and ('f-down-total_spns' in vnames_plt):
                        ax_wvl0.plot(wvl_x, data_plt['f-down-diffuse_spns'][index_pnt, :]/data_plt['f-down-total_spns'][index_pnt, :],
                                     color=var_plot['color'], lw=4.5, alpha=0.45, zorder=0)
                        ax_wvl0.plot(wvl_x, data_plt['f-down-diffuse_spns'][index_pnt, :]/data_plt['f-down-total_spns'][index_pnt, :],
                                     color='white', lw=0.5, alpha=1.0, zorder=1)
                        ax_wvl0.axhline(1.0, color=var_plot['color'], ls=':', lw=1.0, alpha=0.5, zorder=0)
                        ax_wvl0.axhline(0.5, color=var_plot['color'], ls=':', lw=1.0, alpha=0.5, zorder=0)

                else:
                    ax_wvl0.plot(wvl_x, spec_y,
                            color=var_plot['color'], marker='o', markersize=1.2, lw=0.5, markeredgewidth=0.0, alpha=0.85, zorder=var_plot['zorder'])
                    ax_wvl0.plot(wvl_x, data_plt[var_plot['vname_mean']][index_pnt, :],
                                 color='white', lw=0.5, alpha=1.0, zorder=var_plot['zorder']-0.5)
                    ax_wvl0.fill_between(wvl_x, data_plt[var_plot['vname_mean']][index_pnt, :]-data_plt[var_plot['vname_std']][index_pnt, :], data_plt[var_plot['vname_mean']][index_pnt, :]+data_plt[var_plot['vname_std']][index_pnt, :],
                                         fc=var_plot['color'], lw=0.0, alpha=0.3, zorder=var_plot['zorder']-1)
                    ax_wvl0.fill_between(wvl_x, data_plt[var_plot['vname_min']][index_pnt, :], data_plt[var_plot['vname_max']][index_pnt, :],
                                         fc=var_plot['color'], lw=0.0, alpha=0.2, zorder=var_plot['zorder']-2)

    if has_lid0:
        fname_lid0 = data_plt['fnames_lid0'][index_pnt]
        img = mpl_img.imread(fname_lid0)
        extent_lid0 = [float(x) for x in os.path.basename(fname_lid0).replace('.png', '').split('_')[-1].replace('(', '').replace(')', '').split(',')]
        ax_tms_alt.imshow(img, extent=extent_lid0, aspect='auto', alpha=0.5, zorder=0)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # time series
    #╭────────────────────────────────────────────────────────────────────────────╮#
    step_trans = 6
    step_solid = 3

    logic_solid = (data_plt['tmhr']>=tmhr_past) & (data_plt['tmhr']<=tmhr_current)
    logic_trans = np.logical_not(logic_solid) & (data_plt['tmhr']<=tmhr_current)

    alpha_trans = 0.30

    ax_map.scatter(         data_plt['lon'][logic_trans][::step_trans], data_plt['lat'][logic_trans][::step_trans], c=data_plt['alt'][logic_trans][::step_trans], s=0.5, lw=0.0, zorder=2, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, alpha=alpha_trans, transform=ccrs.PlateCarree())
    cs_alt = ax_map.scatter(data_plt['lon'][logic_solid][::step_solid], data_plt['lat'][logic_solid][::step_solid], c=data_plt['alt'][logic_solid][::step_solid], s=2  , lw=0.0, zorder=3, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, transform=ccrs.PlateCarree())

    ax_map.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6, transform=ccrs.PlateCarree())
    ax_map.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, transform=ccrs.PlateCarree())

    ax_map0.scatter(data_plt['lon'][logic_trans], data_plt['lat'][logic_trans], c=data_plt['alt'][logic_trans], s=1.5, lw=0.0, zorder=2, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, alpha=alpha_trans/3.0, transform=ccrs.PlateCarree())
    ax_map0.scatter(data_plt['lon'][logic_solid], data_plt['lat'][logic_solid], c=data_plt['alt'][logic_solid], s=5.0, lw=0.0, zorder=3, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, transform=ccrs.PlateCarree())
    ax_map0.plot(data_plt['lon'][:index_pnt][::10], data_plt['lat'][:index_pnt][::10], color='white', lw=0.25, zorder=3.5, alpha=1.0, transform=ccrs.PlateCarree())

    ax_map0.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6, transform=ccrs.PlateCarree())
    ax_map0.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, transform=ccrs.PlateCarree())

    if logic_solid.sum() > 0:

        for vname in vars_plot.keys():

            var_plot = vars_plot[vname]

            if var_plot['plot?']:

                if 'vname_wvl' in var_plot.keys():
                    wvl_x  = data_plt[var_plot['vname_wvl']]
                    index_wvl = np.argmin(np.abs(wvl_x-_wavelength_))
                    if 'toa' in var_plot['vname']:
                        tms_y = data_plt[var_plot['vname']][index_wvl] * np.cos(np.deg2rad(data_plt['sza']))
                    else:
                        tms_y = data_plt[var_plot['vname']][:, index_wvl]
                else:
                    tms_y = data_plt[var_plot['vname']]

                if vname == 'Altitude':

                    if has_lid0:
                        ax_tms_alt.plot(data_plt['tmhr'][logic_solid], tms_y[logic_solid], color=var_plot['color'], lw=1.0, zorder=var_plot['zorder'])
                    else:
                        ax_tms_alt.fill_between(data_plt['tmhr'][logic_solid], tms_y[logic_solid], facecolor=var_plot['color'], alpha=0.25, lw=0.0, zorder=var_plot['zorder'])

                elif vname == 'Albedo':

                    pass

                else:

                    if vname not in ['TOA↓', 'KT19 T↑']:
                        if has_att:
                            ang_pit_solid = data_plt['ang_pit'][logic_solid]
                            ang_rol_solid = data_plt['ang_rol'][logic_solid]
                            logic_steady = (np.abs(ang_pit_solid)<=5.0) & (np.abs(ang_rol_solid)<=2.5)
                            ax_alt_prof.scatter(tms_y[logic_solid][~logic_steady], data_plt['alt'][logic_solid][~logic_steady], c=var_plot['color'], s=1, lw=0.0, zorder=var_plot['zorder'], alpha=0.15)
                            ax_alt_prof.scatter(tms_y[logic_solid][logic_steady] , data_plt['alt'][logic_solid][logic_steady] , c=var_plot['color'], s=2, lw=0.0, zorder=var_plot['zorder']*2)

                            hist_y, _ = np.histogram(tms_y[logic_solid][logic_steady], bins=hist_bins)
                            ax_alt_hist.bar(hist_x, hist_y, width=hist_bin_w, bottom=hist_bottoms[vname], color=var_plot['color'], alpha=0.5, lw=0.0, zorder=var_plot['zorder']-1)
                            hist_bottoms[vname] += hist_y

                            ax_tms_.scatter(data_plt['tmhr'][logic_solid][~logic_steady], tms_y[logic_solid][~logic_steady], c=var_plot['color'], s=1, lw=0.0, zorder=var_plot['zorder'], alpha=0.4)
                            ax_tms.scatter(data_plt['tmhr'][logic_solid][logic_steady], tms_y[logic_solid][logic_steady], c=var_plot['color'], s=2, lw=0.0, zorder=var_plot['zorder']*2)
                        else:
                            ax_alt_prof.scatter(tms_y[logic_solid], data_plt['alt'][logic_solid], c=var_plot['color'], s=1, lw=0.0, zorder=var_plot['zorder'])

                            hist_y = np.histogram(tms_y[logic_solid], bins=hist_bins)
                            ax_alt_hist.bar(hist_x, hist_y, width=hist_bin_w, bottom=hist_bottoms[vname], color=var_plot['color'], alpha=0.5, lw=0.0, zorder=var_plot['zorder']-1)
                            hist_bottoms[vname] += hist_y

                            ax_tms.scatter(data_plt['tmhr'][logic_solid], tms_y[logic_solid], c=var_plot['color'], s=2, lw=0.0, zorder=var_plot['zorder'])
                    elif vname in ['KT19 T↑']:
                        ax_tms_alt.bar(data_plt['tmhr'][logic_solid], np.repeat(-10.0, logic_solid.sum()), width=1.0/3600.0, bottom=0.0, color=temp_cmap(temp_norm(tms_y[logic_solid])), lw=0.0, zorder=var_plot['zorder'])
                    else:
                        ax_tms.scatter(data_plt['tmhr'][logic_solid], tms_y[logic_solid], c=var_plot['color'], s=2, lw=0.0, zorder=var_plot['zorder'])
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # figure settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    title_fig = '%s UTC' % (dtime_current.strftime('%Y-%m-%d %H:%M:%S'))
    if (_title_extra_ is not None) and (_title_extra_!=''):
        title_fig = '%s\n%s' % (_title_extra_, title_fig)
    fig.suptitle(title_fig, y=0.98, fontsize=20)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # map plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if has_sat0:

        title_map = '%s at %s UTC' % (data_plt['id_sat0'][index_pnt], ssfr.util.jday_to_dtime(data_plt['jday_sat0'][index_pnt]).strftime('%H:%M'))
        time_diff = np.abs(data_plt['jday_sat0'][index_pnt]-jday_current)*86400.0
        if time_diff > 300.0:
            ax_map.set_title(title_map, color='gray')
        else:
            ax_map.set_title(title_map)

        g1 = ax_map.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
        g1.xlocator = FixedLocator(np.arange(-180, 181, 10.0))
        g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 2.0))
        g1.top_labels = False
        g1.right_labels = False
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # navigation plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    ax_nav.set_xlim((-10.0, 10.0))
    ax_nav.set_ylim((-10.0, 10.0))
    ax_nav.axis('off')
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # map0 plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if has_sat1:
        title_map0 = 'False Color 367'
        time_diff = np.abs(data_plt['jday_sat1'][index_pnt]-jday_current)*86400.0
        if time_diff > 300.0:
            ax_map0.set_title(title_map0, color='gray')
        else:
            ax_map0.set_title(title_map0)

        g2 = ax_map0.gridlines(lw=0.5, color='gray', ls='-')
        g2.xlocator = FixedLocator(np.arange(-180.0, 180.1, 5.0))
        g2.ylocator = FixedLocator(np.arange(-89.0, 89.1, 0.5))
    ax_map0.axis('off')
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # camera image plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if has_cam0:
        jday_cam  = data_plt['jday_cam0'][index_pnt]
        dtime_cam = ssfr.util.jday_to_dtime(jday_cam)

        title_img = 'Camera at %s UTC' % (dtime_cam.strftime('%H:%M:%S'))
        time_diff = np.abs(jday_current-jday_cam)*86400.0
        if time_diff > 300.0:
            ax_img.set_title(title_img, color='gray')
        else:
            ax_img.set_title(title_img)

        ax_img_hist.set_xlim((cam_x_s, cam_x_e))
        ax_img_hist.set_ylim((cam_y_s, cam_y_e))

    ax_img.axis('off')
    ax_img_hist.axis('off')
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # time series plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    ax_tms.grid()
    ax_tms.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    xticks = np.linspace(tmhr_past, tmhr_current, 7)
    ax_tms.xaxis.set_major_locator(FixedLocator(xticks))
    ax_tms.xaxis.set_minor_locator(FixedLocator(np.arange(tmhr_past, tmhr_current+0.001, 1.0/60.0)))
    xtick_labels = ['' for i in range(xticks.size)]
    xtick_labels[0]  = '%.4f' % tmhr_past
    xtick_labels[-1] = '%.4f' % tmhr_current
    index_center = int(xticks.size//2)
    xtick_labels[index_center] = '%.4f' % xticks[index_center]
    ax_tms.set_xticklabels(xtick_labels)
    ax_tms.set_xlabel('Time [hour]')

    text_left = ' ← %d minutes ago' % (tmhr_length*60.0)
    ax_tms.annotate(text_left, xy=(0.03, -0.15), fontsize=12, color='gray', xycoords='axes fraction', ha='left', va='center')
    text_right = 'Current → '
    ax_tms.annotate(text_right, xy=(0.97, -0.15), fontsize=12, color='gray', xycoords='axes fraction', ha='right', va='center')

    if has_spectra:
        ax_tms.set_ylim(bottom=_flux_base_, top=min([_flux_ceil_, ax_tms.get_ylim()[-1]+0.15]))
        ax_tms.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 10.1, 0.5)))
        ax_tms.yaxis.set_minor_locator(FixedLocator(np.arange(0.0, 10.1, 0.1)))
        ax_tms.set_ylabel('Flux [$\\mathrm{W m^{-2} nm^{-1}}$]')
    else:
        ax_tms.yaxis.set_ticks([])

    if alt_current < 1.0:
        if has_kt19:
            title_all = 'Longitude %9.4f$^\\circ$, Latitude %8.4f$^\\circ$, Altitude %4d m, Solar Zenith %6.2f$^\\circ$, Nadir IR Temperature %5.1f$^\\circ C$' % (lon_current, lat_current, alt_current*1000.0, sza_current, kt19_current)
        else:
            title_all = 'Longitude %9.4f$^\\circ$, Latitude %8.4f$^\\circ$, Altitude %4d m, Solar Zenith %6.2f$^\\circ$' % (lon_current, lat_current, alt_current*1000.0, sza_current)
    else:
        if has_kt19:
            title_all = 'Longitude %9.4f$^\\circ$, Latitude %8.4f$^\\circ$, Altitude %6.3f km, Solar Zenith %6.2f$^\\circ$, Nadir IR Temperature %5.1f$^\\circ C$' % (lon_current, lat_current, alt_current, sza_current, kt19_current)
        else:
            title_all = 'Longitude %9.4f$^\\circ$, Latitude %8.4f$^\\circ$, Altitude %6.3f km, Solar Zenith %6.2f$^\\circ$' % (lon_current, lat_current, alt_current, sza_current)

    # coloring text (future option)
    #╭──────────────────────────────────────────────────────────────╮#
    ax_tms.set_title(title_all)

    # reserved for future
    #╭────────────────────────────────────────────────╮#
    # lon_text = 'Longitude %9.4f$^\\circ$' % (lon_current)
    # lat_text = 'Latitude %9.4f$^\\circ$' % (lat_current)

    # if alt_current < 1.0:
    #     alt_text = 'Altitude %4d m' % (alt_current*1000.0)
    # else:
    #     alt_text = 'Altitude %6.3f km' % (alt_current)

    # if has_kt19:
    #     kt19_text = 'Nadir IR Temperature %5.1f$^\\circ C$' % (kt19_current)
    # else:
    #     kt19_text = ''

    # sza_text = 'Solar Zenith %6.2f$^\\circ$' % (sza_current)

    # text_list = [lon_text, lat_text, alt_text, sza_text, kt19_text]

    # text_all = ', '.join(text_list)
    #╰────────────────────────────────────────────────╯#
    #╰──────────────────────────────────────────────────────────────╯#

    ax_tms.spines['right'].set_visible(False)
    ax_tms.set_zorder(ax_tms_alt.get_zorder()+1)
    ax_tms.patch.set_visible(False)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # spectra plot setting
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if has_spectra:
        ax_wvl.set_xlim((200, 2200))
        ax_wvl.set_ylim(copy.deepcopy(ax_tms.get_ylim()))
        ax_wvl.xaxis.set_major_locator(FixedLocator(np.arange(0, 2401, 400)))
        ax_wvl.xaxis.set_minor_locator(FixedLocator(np.arange(0, 2401, 100)))
        ax_wvl.set_xlabel('Wavelength [nm]')
        ax_wvl.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 10.1, 0.5)))
        ax_wvl.yaxis.set_minor_locator(FixedLocator(np.arange(0.0, 10.1, 0.1)))

        ax_wvl0.set_ylim((0.0, 1.2))
        ax_wvl0.yaxis.set_major_locator(FixedLocator(np.arange(0, 2.1, 1.0)))
        ax_wvl0.yaxis.set_minor_locator(FixedLocator(np.arange(0, 2.1, 0.1)))
        ax_wvl0.set_ylabel('Albedo', rotation=270.0, labelpad=18, color=vars_plot['Albedo']['color'])
        ax_wvl0.set_frame_on(True)
        for spine in ax_wvl0.spines.values():
            spine.set_visible(False)
        ax_wvl0.spines['right'].set_visible(True)
        ax_wvl0.spines['right'].set_color(vars_plot['Albedo']['color'])
        ax_wvl0.tick_params(axis='y', which='both', colors=vars_plot['Albedo']['color'])
    else:
        ax_wvl.axis('off')
    #╰────────────────────────────────────────────────────────────────────────────╯#



    # profile plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    ax_alt_prof.axhline(alt_current, lw=1.0, color=vars_plot['Altitude']['color'], zorder=1, alpha=0.9)
    ax_alt_prof.grid()

    if has_spectra:
        ax_alt_prof.set_xlim(copy.deepcopy(ax_tms.get_ylim()))
        ax_alt_prof.xaxis.set_major_locator(FixedLocator(np.arange(0.5, 10.1, 0.5)))
        ax_alt_prof.xaxis.set_minor_locator(FixedLocator(np.arange(0.0, 10.1, 0.1)))
        ax_alt_prof.set_ylim(
                bottom=max([_alt_base_, ax_alt_prof.get_ylim()[0]-0.5]),
                top=min([ax_alt_prof.get_ylim()[-1]+0.5, _alt_ceil_]),
                )
    else:
        ax_alt_prof.xaxis.set_ticks([])
        ax_alt_prof.set_ylim([_alt_base_, _alt_ceil_])

    ax_alt_prof.yaxis.set_label_position('right')
    ax_alt_prof.yaxis.tick_right()
    ax_alt_prof.yaxis.set_major_locator(FixedLocator(np.arange(_alt_base_, _alt_ceil_+0.1, 1.0)))
    ax_alt_prof.yaxis.set_minor_locator(FixedLocator(np.arange(_alt_base_, _alt_ceil_+0.1, 0.1)))

    ax_alt_prof.set_ylabel('Altitude [km]', rotation=270.0, labelpad=18, color=vars_plot['Altitude']['color'])
    ax_alt_prof.spines['right'].set_visible(True)
    ax_alt_prof.spines['right'].set_color(vars_plot['Altitude']['color'])
    ax_alt_prof.tick_params(axis='y', which='both', colors=vars_plot['Altitude']['color'])
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # histogram plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    ax_alt_hist.set_xlim(copy.deepcopy(ax_tms.get_ylim()))
    ax_alt_hist.set_ylim((0, 5000))
    ax_alt_hist.axis('off')
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # altitude/sza plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    cbar = fig.colorbar(cs_alt, cax=ax_alt_cbar)
    ax_alt_cbar.axhline(alt_current, lw=3.0, color='white', zorder=1, alpha=0.6)
    ax_alt_cbar.axhline(alt_current, lw=1.0, color=vars_plot['Altitude']['color'], zorder=2, alpha=1.0)
    ax_alt_cbar.set_ylim(ax_alt_prof.get_ylim())
    ax_alt_cbar.xaxis.set_ticks([])
    ax_alt_cbar.yaxis.set_ticks([])
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # altitude (time series) plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if has_kt19:
        ax_tms_alt.set_ylim(bottom=-(ax_alt_prof.get_ylim()[-1])*0.05, top=ax_alt_prof.get_ylim()[-1])
        ax_tms.set_ylim(bottom=-(ax_tms.get_ylim()[-1]-_flux_base_) * 0.05)
    else:
        ax_tms_alt.set_ylim(bottom=0.0, top=ax_alt_prof.get_ylim()[-1])
    ax_tms_.axis('off')
    ax_tms_.set_ylim(ax_tms.get_ylim())

    if ax_alt_prof.get_ylim()[-1] > 2.0:
        ax_tms_alt.yaxis.set_major_locator(FixedLocator(np.arange(_alt_base_, _alt_ceil_+0.1, 2.0)))
    else:
        ax_tms_alt.yaxis.set_major_locator(FixedLocator(np.arange(_alt_base_, _alt_ceil_+0.1, 1.0)))
    ax_tms_alt.yaxis.set_minor_locator(FixedLocator(np.arange(_alt_base_, _alt_ceil_+0.1, 0.5)))
    ax_tms_alt.yaxis.tick_right()
    ax_tms_alt.yaxis.set_label_position('right')
    ax_tms_alt.set_ylabel('Altitude [km]', rotation=270.0, labelpad=18, color=vars_plot['Altitude']['color'])

    ax_tms_alt.set_frame_on(True)
    for spine in ax_tms_alt.spines.values():
        spine.set_visible(False)
    ax_tms_alt.spines['right'].set_visible(True)
    ax_tms_alt.spines['right'].set_color(vars_plot['Altitude']['color'])
    ax_tms_alt.tick_params(axis='y', which='both', colors=vars_plot['Altitude']['color'])
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # acknowledgements
    #╭────────────────────────────────────────────────────────────────────────────╮#
    text1 = '\
presented by ARCSIX SSFR Team\
'
    if has_kt19:
        text1 = '%s | MetNav Team' % text1
    if has_lid0:
        text1 = '%s | MARLi Team' % text1

    text1 = '%s\n' % text1

    ax_all.annotate(text1, xy=(0.5, 0.26), fontsize=8, color='gray', xycoords='axes fraction', ha='center', va='center')

    text2 = '\
IN-FIELD USE ONLY\n\
'
    ax_all.annotate(text2, xy=(0.5, 0.23), fontsize=10, color='red', xycoords='axes fraction', ha='center', va='center')
    ax_all.axis('off')
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # legend plot settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    patches_legend = []
    for vname in vars_plot.keys():
        var_plot = vars_plot[vname]
        if (vname not in ['Altitude', 'KT19 T↑']) and var_plot['plot?']:
            patches_legend.append(mpatches.Patch(color=var_plot['color'], label=vname))
    if len(patches_legend) > 0:
        ax_wvl.legend(handles=patches_legend, bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=len(patches_legend), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=10)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if test:
        time_e = time.time()
        print('Elapsed time: %.1f seconds.' % (time_e-time_s))
        plt.show()
        sys.exit()
    else:
        if _time_interval_ >= 20:
            plt.savefig('%s/%5.5d.jpg' % (data_plt['fdir_tmp'], index_pnt), bbox_inches='tight', dpi=_dpi_, pil_kwargs={'quality':70})
        else:
            plt.savefig('%s/%5.5d.jpg' % (data_plt['fdir_tmp'], index_pnt), bbox_inches='tight', dpi=_dpi_, pil_kwargs={'quality':85})
        plt.close(fig)
        plt.clf()

def post_process_sat_img_vn(
        date,
        ):

    # date time stamp
    #╭────────────────────────────────────────────────────────────────────────────╮#
    date_sat_s = date.strftime('%Y-%m-%d')
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # process satellite imagery
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # for layername in ['TrueColor', 'FalseColor721', 'FalseColor367']:
    for layername in ['TrueColor', 'FalseColor367']:
        fnames_sat = ssfr.util.get_all_files(_fdir_sat_img_vn_, pattern='*%s*%s*Z*(-877574.55,877574.55,-751452.90,963254.75)_(-80.0000,-30.0000,71.0000,88.0000).png' % (layername, date_sat_s))
        jday_sat, fnames_sat = process_sat_img_overlay(fnames_sat)
    #╰────────────────────────────────────────────────────────────────────────────╯#

def main_pre(
        date,
        time_step=1,
        wvl_step_spns=1,
        wvl_step_ssfr=1,
        ):

    # date time stamp
    #╭────────────────────────────────────────────────────────────────────────────╮#
    date_s = date.strftime('%Y%m%d')
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read in aircraft hsk data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)

    f_hsk = h5py.File(fname_hsk, 'r')
    jday   = f_hsk['jday'][...]
    tmhr   = f_hsk['tmhr'][...]
    sza    = f_hsk['sza'][...]
    lon    = f_hsk['lon'][...]
    lat    = f_hsk['lat'][...]
    alt    = f_hsk['alt'][...]
    lon[lon>=180.0] -= 360.0

    hsk_keys = [key for key in f_hsk.keys()]

    # filter out invalid data
    #╭──────────────────────────────────────────────────────────────╮#
    logic0 = (~np.isnan(jday) & ~np.isnan(sza) & ~np.isinf(sza))  & \
             (alt>=0.0) & (alt<=12000.0) & \
             (lat>=71.5) &\
             check_continuity(lon, threshold=1.0) & \
             check_continuity(lat, threshold=1.0) & \
             (tmhr>=_date_specs_[date_s]['tmhr_range'][0]) & (tmhr<=_date_specs_[date_s]['tmhr_range'][1])
    #╰──────────────────────────────────────────────────────────────╯#

    jday = jday[logic0][::time_step]

    tmhr = tmhr[logic0][::time_step]
    sza  = sza[logic0][::time_step]
    lon  = lon[logic0][::time_step]
    lat  = lat[logic0][::time_step]
    alt    = alt[logic0][::time_step]

    ang_pit = f_hsk['ang_pit'][...][logic0][::time_step]
    ang_rol = f_hsk['ang_rol'][...][logic0][::time_step]

    f_hsk.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # marli
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_marli = process_marli(date)
    if fname_marli is not None:
        has_marli = True
    else:
        has_marli = False
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read kt19
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if 'ir_surf_temp' in hsk_keys:
        has_kt19 = True
    else:
        has_kt19 = False

    if has_kt19:
        f_hsk = h5py.File(fname_hsk, 'r')
        kt19_nad_temp = f_hsk['ir_surf_temp'][...][logic0][::time_step]
        f_hsk.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read in nav data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_alp = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _alp_.upper(), _platform_.upper(), date_s)
    if os.path.exists(fname_alp):
        has_alp = True
    else:
        has_alp = False

    if has_alp:
        f_alp = h5py.File(fname_alp, 'r')
        ang_pit_s = f_alp['ang_pit_s'][...][logic0][::time_step]
        ang_rol_s = f_alp['ang_rol_s'][...][logic0][::time_step]
        ang_pit_m = f_alp['ang_pit_m'][...][logic0][::time_step]
        ang_rol_m = f_alp['ang_rol_m'][...][logic0][::time_step]
        rel_hum   = f_alp['rh'][...][logic0][::time_step]
        f_alp.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read in spns data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_spns = '%s/%s-SPNS_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    # fname_spns = '%s/%s-%s_%s_%s_v2.h5' % (_fdir_data_, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)
    # fname_spns = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)
    if os.path.exists(fname_spns):
        has_spns = True
    else:
        has_spns = False

    if has_spns:
        f_spns = h5py.File(fname_spns, 'r')
        spns_tot_flux = f_spns['tot/flux'][...][logic0, :][::time_step, ::wvl_step_spns]
        spns_tot_wvl  = f_spns['tot/wvl'][...][::wvl_step_spns]
        spns_dif_flux = f_spns['dif/flux'][...][logic0, :][::time_step, ::wvl_step_spns]
        spns_dif_wvl  = f_spns['dif/wvl'][...][::wvl_step_spns]
        f_spns.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read in ssfr-a data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    # fname_ssfr = '%s/%s-%s_%s_%s_v2.h5' % (_fdir_data_, _mission_.upper(), _ssfr_.upper(), _platform_.upper(), date_s)
    # fname_ssfr = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _ssfr_.upper(), _platform_.upper(), date_s)
    if os.path.exists(fname_ssfr):
        has_ssfr = True
    else:
        has_ssfr = False

    if has_ssfr:
        f_ssfr = h5py.File(fname_ssfr, 'r')
        ssfr_zen_flux = f_ssfr['zen/flux'][...][logic0, :][::time_step, ::wvl_step_ssfr]
        ssfr_zen_wvl  = f_ssfr['zen/wvl'][...][::wvl_step_ssfr]
        ssfr_nad_flux = f_ssfr['nad/flux'][...][logic0, :][::time_step, ::wvl_step_ssfr]
        ssfr_nad_wvl  = f_ssfr['nad/wvl'][...][::wvl_step_ssfr]
        ssfr_zen_toa  = f_ssfr['zen/toa0'][...][::wvl_step_ssfr]
        f_ssfr.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # pre-process the aircraft and satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # create a filter to remove invalid data, e.g., out of available satellite data time range,
    # invalid solar zenith angles etc.
    tmhr_interval = 10.0/60.0
    half_interval = tmhr_interval/48.0

    jday_s = ((jday[0]  * 86400.0) // (60.0) + 1) * (60.0) / 86400.0
    jday_e = ((jday[-1] * 86400.0) // (60.0)    ) * (60.0) / 86400.0

    jday_edges = np.arange(jday_s, jday_e+half_interval, half_interval*2.0)

    logic = (jday>=jday_s) & (jday<=jday_e)

    # create python dictionary to store valid flight data
    _data_ = {}
    _data_['jday'] = jday[logic]
    _data_['lon']  = lon[logic]
    _data_['lat']  = lat[logic]
    _data_['sza']  = sza[logic]
    _data_['tmhr'] = tmhr[logic]
    _data_['alt']  = alt[logic]/1000.0

    _data_['ang_pit'] = ang_pit[logic]
    _data_['ang_rol'] = ang_rol[logic]

    if has_kt19:
        _data_['t-up_kt19'] = kt19_nad_temp[logic]

    if has_alp:
        _data_['ang_pit_s'] = ang_pit_s[logic]
        _data_['ang_rol_s'] = ang_rol_s[logic]
        _data_['ang_pit_m'] = ang_pit_m[logic]
        _data_['ang_rol_m'] = ang_rol_m[logic]
        _data_['rel_hum']   = rel_hum[logic]

    if has_spns:
        _data_['f-down-total_spns']   = spns_tot_flux[logic, :]
        _data_['f-down-diffuse_spns'] = spns_dif_flux[logic, :]
        _data_['f-down-direct_spns']  = _data_['f-down-total_spns'] - _data_['f-down-diffuse_spns']
        _data_['wvl_spns'] = spns_tot_wvl

    if has_ssfr:
        # _data_['f-down_ssfr']   = ssfr_zen_flux[logic, :]
        print('haha')
        _data_['f-down_ssfr']   = ssfr_zen_flux[logic, :] * 2.5

        _data_['f-up_ssfr']     = ssfr_nad_flux[logic, :]
        _data_['wvl_ssfr_zen'] = ssfr_zen_wvl
        _data_['wvl_ssfr_nad'] = ssfr_nad_wvl
        _data_['f-down_toa']    = ssfr_zen_toa

        # calculate albedo
        #╭──────────────────────────────────────────────────────────────╮#
        _data_['alb_ssfr'] = np.zeros_like(_data_['f-down_ssfr'])
        _data_['alb-mean_ssfr'] = np.zeros_like(_data_['f-down_ssfr'])
        _data_['alb-std_ssfr'] = np.zeros_like(_data_['f-down_ssfr'])
        _data_['alb-min_ssfr'] = np.zeros_like(_data_['f-down_ssfr'])
        _data_['alb-max_ssfr'] = np.zeros_like(_data_['f-down_ssfr'])

        logic_good = (np.abs(_data_['ang_pit'])<=5.0) & (np.abs(_data_['ang_rol'])<=2.5) & \
                     (~np.isnan(_data_['f-down_ssfr'][:, np.argmin(np.abs(_wavelength_-_data_['wvl_ssfr_zen']))])) & \
                     (  ~np.isnan(_data_['f-up_ssfr'][:, np.argmin(np.abs(_wavelength_-_data_['wvl_ssfr_nad']))])) & \
                     (_data_['f-down_ssfr'][:, np.argmin(np.abs(_wavelength_-_data_['wvl_ssfr_zen']))] > 0.0)

        for i in tqdm(range(_data_['jday'].size)):

            f_dn  = _data_['f-down_ssfr'][i, :]
            f_up_ = _data_['f-up_ssfr'][i, :]
            f_up = np.interp(_data_['wvl_ssfr_zen'], _data_['wvl_ssfr_nad'], f_up_)
            _data_['alb_ssfr'][i, :] = f_up/f_dn

            logic = logic_good & (_data_['jday']>=(_data_['jday'][i]-30.0/86400.0)) & (_data_['jday']<=(_data_['jday'][i]))

            if logic.sum() > 0:

                f_dn  = _data_['f-down_ssfr'][logic, :]
                f_up_ = _data_['f-up_ssfr'][logic, :]

                f_up = np.zeros_like(f_dn)
                for ii_ in range(f_dn.shape[0]):
                    f_up[ii_, :] = np.interp(_data_['wvl_ssfr_zen'], _data_['wvl_ssfr_nad'], f_up_[ii_, :])

                alb_all = f_up/f_dn

                _data_['alb-mean_ssfr'][i, :] = np.nanmean(alb_all, axis=0)
                _data_['alb-std_ssfr'][i, :]  = np.nanstd(alb_all, axis=0)
                _data_['alb-min_ssfr'][i, :]  = np.nanmin(alb_all, axis=0)
                _data_['alb-max_ssfr'][i, :]  = np.nanmax(alb_all, axis=0)

            else:

                _data_['alb-mean_ssfr'][i, :] = np.nan
                _data_['alb-std_ssfr'][i, :]  = np.nan
                _data_['alb-min_ssfr'][i, :]  = np.nan
                _data_['alb-max_ssfr'][i, :]  = np.nan

        absorb_bands = [[649.0, 662.0], [676.0, 697.0], [707.0, 729.0], [743.0, 779.0], [796.0, 830.0], [911.0, 976.0], [1108.0, 1193.0], [1262.0, 1298.0], [1324.0, 1470.0], [1750.0, 1960.0]]
        logic_wvl = np.repeat(True, _data_['wvl_ssfr_zen'].size)
        for absorb_band in absorb_bands:
            wvl_l, wvl_r = absorb_band
            logic0 = (_data_['wvl_ssfr_zen']>=wvl_l)&(_data_['wvl_ssfr_zen']<=wvl_r)
            logic_wvl[logic0] = False

        wvl_alb = _data_['wvl_ssfr_zen'].copy()
        wvl_alb[~logic_wvl] = np.nan
        _data_['wvl_ssfr_alb'] = wvl_alb
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # process camera imagery
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fdirs = ssfr.util.get_all_folders(_fdir_cam_img_, pattern='*%4.4d*%2.2d*%2.2d*nac*jpg*' % (date.year, date.month, date.day))
    if len(fdirs) > 0:
        fdir_cam0 = sorted(fdirs, key=os.path.getmtime)[-1]
        fnames_cam0 = sorted(glob.glob('%s/*.jpg' % (fdir_cam0)))
        if len(fnames_cam0) > 0:
            has_cam = True
            jday_cam0 = get_jday_cam_img(date, fnames_cam0) + _date_specs_[date_s]['cam_time_offset']/86400.0
        else:
            has_cam = False
    else:
        has_cam = False
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # process satellite imagery
    #╭────────────────────────────────────────────────────────────────────────────╮#
    date_sat_s  = date.strftime('%Y-%m-%d')

    fnames_sat0 = {}
    fnames_sat1 = {}

    fnames_sat00 = sorted(ssfr.util.get_all_files(_fdir_sat_img_hc_, pattern='TrueColor*%s*(-877574.55,877574.55,-751452.90,963254.75)_(-80.0000,-30.0000,71.0000,88.0000).jpg' % date_sat_s))
    jday_sat00 = np.zeros(len(fnames_sat00), dtype=np.float64)
    for i, fname_sat00 in enumerate(fnames_sat00):
        dtime00_s = '_'.join(os.path.basename(fname_sat00).split('_')[1:3])
        dtime00 = datetime.datetime.strptime(dtime00_s, '%Y-%m-%d_%H:%M:%S')
        jday_sat00[i] = ssfr.util.dtime_to_jday(dtime00)

    fnames_sat11 = sorted(ssfr.util.get_all_files(_fdir_sat_img_hc_, pattern='FalseColor367*%s*(-877574.55,877574.55,-751452.90,963254.75)_(-80.0000,-30.0000,71.0000,88.0000).jpg' % date_sat_s))
    # fnames_sat11 = sorted(ssfr.util.get_all_files(_fdir_sat_img_hc_, pattern='FalseColor721*%s*(-877574.55,877574.55,-751452.90,963254.75)_(-80.0000,-30.0000,71.0000,88.0000).jpg' % date_sat_s))
    jday_sat11 = np.zeros(len(fnames_sat11), dtype=np.float64)
    for i, fname_sat11 in enumerate(fnames_sat11):
        dtime11_s = '_'.join(os.path.basename(fname_sat11).split('_')[1:3])
        dtime11 = datetime.datetime.strptime(dtime11_s, '%Y-%m-%d_%H:%M:%S')
        jday_sat11[i] = ssfr.util.dtime_to_jday(dtime11)

    fnames_sat0['jday']    = jday_sat00
    fnames_sat0['fnames']  = fnames_sat00

    fnames_sat1['jday']   = jday_sat11
    fnames_sat1['fnames'] = fnames_sat11
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # create python dictionary to store corresponding satellite imagery data info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    _data_['id_sat0'] = []
    _data_['fnames_sat0'] = []
    _data_['extent_sat0'] = _extent_xy_study_region_
    _data_['jday_sat0']   = np.array([], dtype=np.float64)

    _data_['fnames_sat1'] = []
    _data_['extent_sat1'] = _extent_xy_study_region_
    _data_['jday_sat1']   = np.array([], dtype=np.float64)

    if has_cam:
        _data_['fnames_cam0']  = []
        _data_['jday_cam0'] = np.array([], dtype=np.float64)

    if has_marli:
        _data_['fnames_lid0']  = []

    for i in range(_data_['jday'].size):

        jday_sat0_   = fnames_sat0['jday']
        fnames_sat0_ = fnames_sat0['fnames']
        index_sat0   = np.argmin(np.abs(jday_sat0_-_data_['jday'][i]))
        _data_['id_sat0'].append(' '.join(os.path.basename(fnames_sat0_[index_sat0]).split('_')[3:5][::-1]))
        _data_['fnames_sat0'].append(fnames_sat0_[index_sat0])
        _data_['jday_sat0'] = np.append(_data_['jday_sat0'], jday_sat0_[index_sat0])

        jday_sat1_ = fnames_sat1['jday']
        fnames_sat1_ = fnames_sat1['fnames']
        index_sat1 = np.argmin(np.abs(jday_sat1_-_data_['jday'][i]))
        _data_['fnames_sat1'].append(fnames_sat1_[index_sat1])
        _data_['jday_sat1'] = np.append(_data_['jday_sat1'], jday_sat1_[index_sat1])

        if has_cam:
            index_cam = np.argmin(np.abs(jday_cam0-_data_['jday'][i]))
            _data_['fnames_cam0'].append(fnames_cam0[index_cam])
            _data_['jday_cam0'] = np.append(_data_['jday_cam0'], jday_cam0[index_cam])

        if has_marli:
            _data_['fnames_lid0'].append(fname_marli)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # reduce memory size
    #╭────────────────────────────────────────────────────────────────────────────╮#
    _data_['date_s'] = date_s

    for key in _data_.keys():
        if ('jday' not in key.lower()) and (hasattr(_data_[key], 'dtype')):
            if _data_[key].dtype in ['float64']:
                if key[:2] == 'f-':
                    _data_[key] = _data_[key].astype(np.float16)
                else:
                    _data_[key] = _data_[key].astype(np.float32)
            elif _data_[key].dtype in ['int64','int32', 'int16']:
                _data_[key] = _data_[key].astype(np.int8)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # save data to pickle file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = '%s/%s-FLT-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _platform_.upper(), date_s)
    with open(fname, 'wb') as f:
        print('Message [flt_sim]: Saving object into <%s> ...' % fname)
        pickle.dump(_data_, f)
    #╰────────────────────────────────────────────────────────────────────────────╯#

def main_vid(
        date,
        fdir_tmp=_fdir_tmp_,
        Ncpu=_Ncpu_,
        time_interval=_time_interval_,
        ):

    # date time stamp
    #╭────────────────────────────────────────────────────────────────────────────╮#
    date_s = date.strftime('%Y%m%d')
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # load pickle file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = '%s/%s-FLT-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _platform_.upper(), date_s)
    with open(fname, 'rb') as f:
        _data_ = pickle.load(f)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # create temporary directory to store images
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # job_tag = '%s-FLT-VID-%2.2d_%s_%s' % (_mission_.upper(), time_interval, _platform_.upper(), _data_['date_s'])
    job_tag = '%s_SSFR_Flight-Video-%2.2d' % (_data_['date_s'], time_interval)
    fdir_tmp = os.path.join(fdir_tmp, job_tag)
    if os.path.exists(fdir_tmp):
        os.system('rm -rf %s' % fdir_tmp)
    os.makedirs(fdir_tmp)
    _data_['fdir_tmp'] = fdir_tmp
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # prepare statements for parallel processing
    #╭────────────────────────────────────────────────────────────────────────────╮#
    Ndata = _data_['jday'].size // time_interval
    indices = np.arange(Ndata)*time_interval
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # create images
    #╭────────────────────────────────────────────────────────────────────────────╮#
    with SharedMemoryManager() as smm:

        shared_data = {}
        unshared_data = {}

        for key in _data_.keys():
            data0 = _data_[key]
            if isinstance(data0, np.ndarray):
                shm = smm.SharedMemory(data0.nbytes)
                shm_np_array = np.ndarray(data0.shape, dtype=data0.dtype, buffer=shm.buf)
                shm_np_array[...] = data0[...]
                shared_data[key] = {
                        'name': key,
                        'shm_name': shm.name,
                        'shape': data0.shape,
                        'dtype': data0.dtype,
                        }
            else:
                unshared_data[key] = data0

        if not _test_:
            statements = zip([shared_data]*Ndata, [unshared_data]*Ndata, indices)
            with mp.Pool(processes=Ncpu) as pool:
                r = list(tqdm(pool.imap(plot_video_frame, statements), total=Ndata))
        else:
            statements = (shared_data, unshared_data, int(Ndata//2))
            plot_video_frame(statements, test=_test_)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # make video
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_mp4 = '%s.mp4' % (job_tag)
    if _time_interval_ >= 20:
        os.system('ffmpeg -y -framerate 30 -pattern_type glob -i "%s/*.jpg" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -crf 20 -pix_fmt yuvj420p %s' % (fdir_tmp, fname_mp4))
    else:
        os.system('ffmpeg -y -framerate 30 -pattern_type glob -i "%s/*.jpg" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -crf 15 -pix_fmt yuvj420p %s' % (fdir_tmp, fname_mp4))
    if _delete_tmp_:
        if _delete_ask_:
            print('\nMessage [flt_vid]: Press <y/n> to delete/keep temporary files (total of %s):' % (subprocess.check_output('du -hs %s' % fdir_tmp, shell=True, text=True).split('\t')[0]))
            os.system('rm -rf -I %s' % fdir_tmp)
        else:
            os.system('rm -rf %s' % fdir_tmp)
    #╰────────────────────────────────────────────────────────────────────────────╯#
#╰────────────────────────────────────────────────────────────────────────────╯#


if __name__ == '__main__':

    dates = [
            # datetime.datetime(2024, 5, 28), # [✓] ARCSIX science flight #1; clear-sky spiral
            # datetime.datetime(2024, 5, 30), # [✓] ARCSIX science flight #2; cloud wall
            # datetime.datetime(2024, 5, 31), # [✓] ARCSIX science flight #3; bowling alley, surface BRDF
            # datetime.datetime(2024, 6,  3), # ARCSIX science flight #4; cloud wall, (no MARLi)
            # datetime.datetime(2024, 6,  5), # [✓] ARCSIX science flight #5; bowling alley, surface BRDF
            # datetime.datetime(2024, 6,  6), # [✓] ARCSIX science flight #6; cloud wall
            # datetime.datetime(2024, 6,  7), # [✓] ARCSIX science flight #7; cloud wall
            # datetime.datetime(2024, 6, 10), # [✓] ARCSIX science flight #8; cloud wall
            # datetime.datetime(2024, 6, 11), # [✓] ARCSIX science flight #9; cloud wall
            # datetime.datetime(2024, 6, 13), # [✓] ARCSIX science flight #10
            # datetime.datetime(2024, 7, 25), # [✓] ARCSIX science flight # 11; cloud wall
            # datetime.datetime(2024, 7, 29), # [✓] ARCSIX science flight # 12; clear-sky BRDF
            # datetime.datetime(2024, 7, 30), # [✓] ARCSIX science flight # 13; clear-sky BRDF
            # datetime.datetime(2024, 8, 1), # [✓] ARCSIX science flight # 14; cloud walls
            # datetime.datetime(2024, 8, 2), # [✓] ARCSIX science flight # 15; cloud walls
            # datetime.datetime(2024, 8, 7), # ARCSIX science flight # 16; cloud walls, (missing MetNav)
            # datetime.datetime(2024, 8, 8), # [✓] ARCSIX science flight # 17; cloud walls, (missing MARLi)
            datetime.datetime(2024, 8, 9), # [✓] ARCSIX science flight # 18; cloud walls, (missing MARLi)
        ]

    for date in dates[::-1]:

        # process satellite imageries
        #╭────────────────────────────────────────────────────────────────────────────╮#
        post_process_sat_img_vn(date)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # main processes for generating video
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # if not _test_:
        #     main_pre(date)
        # main_vid(date)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        pass

    pass
