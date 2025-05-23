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
from scipy.interpolate import RegularGridInterpolator
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
import matplotlib.axes as maxes
import cartopy
import cartopy.crs as ccrs
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

_fdir_main_       = 'data/%s/flt-vid' % _mission_
_fdir_sat_img_    = 'data/%s/sat-img' % _mission_
_fdir_sat_data_   = 'data/%s/sat-data' % _mission_
_fdir_cam_img_    = 'data/%s/2024-Spring/p3' % _mission_
_wavelength_      = 555.0

_fdir_sat_img_vn_ = 'data/%s/sat-img-vn' % _mission_

_preferred_region_ = 'ca_archipelago'
_aspect_ = 'equal'

_fdir_data_ = '/argus/field/%s/processed' % _mission_
_fdir_tmp_graph_ = 'tmp-graph_flt-vid'

_title_extra_ = 'ARCSIX RF#1'

_tmhr_range_ = {
        '20240517': [19.20, 23.00],
        '20240521': [14.80, 17.50],
        }

# science flight numbering
#╭────────────────────────────────────────────────────────────────────────────╮#
_sf_number_ = {
        '20240528': 1,
        '20240530': 2,
        '20240531': 3,
        '20240603': 4,
        '20240605': 5,
        '20240606': 6,
        '20240607': 7,
        '20240610': 8,
        '20240611': 9,
        '20240613': 10,
        '20240725': 11,
        '20240729': 12,
        '20240730': 13,
        '20240801': 14,
        '20240802': 15,
        '20240807': 16,
        '20240808': 17,
        '20240809': 18,
        '20240815': 19,
        }
#╰────────────────────────────────────────────────────────────────────────────╯#

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



def figure_agu_2024_sfc_alb():

    #╭────────────────────────────────────────────────────────────────────────────╮#
    dates = [
            datetime.datetime(2024, 6, 6), # [✓]
            datetime.datetime(2024, 7, 30), # [✓]
        ]

    colors = ['blue', 'red']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    patches_legend = []

    tmhr_ranges = {
            '20240606': [16.7181, 16.7833],
            '20240730': [14.8723, 14.9375],
            }

    rcParams['font.size'] = 20
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(111)

    # absorption bands
    #╭────────────────────────────────────────────────────────────────────────────╮#
    absorb_bands = [[649.0, 662.0], [676.0, 697.0], [707.0, 729.0], [743.0, 779.0], [796.0, 830.0], [911.0, 976.0], [1108.0, 1193.0], [1262.0, 1298.0], [1324.0, 1470.0], [1800.0, 1960.0]]
    for absorb_band in absorb_bands:
        wvl_L, wvl_R = absorb_band
        ax1.axvspan(wvl_L, wvl_R, color='k', lw=0.0, alpha=0.1, zorder=0)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    for i, date in enumerate(dates):

        date_s     = date.strftime('%Y%m%d')

        # read hsk data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
        data_hsk = er3t.util.load_h5(fname_hsk)
        alt = data_hsk['alt']/1000.0
        kt19 = data_hsk['ir_surf_temp']
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # read SSFR
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
        data_ssfr = er3t.util.load_h5(fname_ssfr)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        wvl = data_ssfr['zen/wvl'].copy()
        logic = np.repeat(True, wvl.size)

        for absorb_band in absorb_bands:
            wvl_L, wvl_R = absorb_band
            logic0 = (wvl>=wvl_L)&(wvl<=wvl_R)
            logic[logic0] = False
        wvl[~logic] = np.nan

        tmhr_range = tmhr_ranges[date_s]
        tmhr_range_legs = []
        logic_select = (data_hsk['tmhr']>tmhr_range[0]) & (data_hsk['tmhr']<tmhr_range[1]) & \
                       (np.abs(data_hsk['ang_pit'])<=2.0) & (np.abs(data_hsk['ang_rol'])<=2.0)

        color = colors[i]
        zorder = 10
        alpha = 0.6
        if date < datetime.datetime(2024, 7, 1):
            patches_legend.append(mpatches.Patch(color=color, alpha=1.0, label='ARCSIX-1 on 2024 %s' % (date.strftime('%B-%d').replace('-0', ' ').replace('-', ' '))))
        else:
            patches_legend.append(mpatches.Patch(color=color, alpha=1.0, label='ARCSIX-2 on 2024 %s' % (date.strftime('%B-%d').replace('-0', ' ').replace('-', ' '))))

        logic = (data_hsk['tmhr']>tmhr_range[0]) & (data_hsk['tmhr']<tmhr_range[1]) & \
                (np.abs(data_hsk['ang_pit'])<=2.0) & (np.abs(data_hsk['ang_rol'])<=2.0) & \
                (~np.isnan(data_ssfr['zen/flux'][:, np.argmin(np.abs(532.0-data_ssfr['zen/wvl']))])) & \
                (~np.isnan(data_ssfr['nad/flux'][:, np.argmin(np.abs(532.0-data_ssfr['nad/wvl']))])) & \
                ((data_ssfr['zen/flux'][:, np.argmin(np.abs(532.0-data_ssfr['zen/wvl']))])>0.0)

        if logic.sum() > 0:
            print(logic.sum())

            f_dn = data_ssfr['zen/flux'][logic, :]

            f_up_ = data_ssfr['nad/flux'][logic, :]
            f_up = np.zeros_like(f_dn)
            for ii_ in range(f_dn.shape[0]):
                f_up[ii_, :] = np.interp(data_ssfr['zen/wvl'], data_ssfr['nad/wvl'], f_up_[ii_, :])

            alb_all = f_up/f_dn
            alb_mean = np.nanmean(alb_all, axis=0)
            alb_std  = np.nanstd(alb_all, axis=0)
            alb_max  = np.nanmax(alb_all, axis=0)
            alb_min  = np.nanmin(alb_all, axis=0)

            alt_mean = np.mean(alt[logic])

            ax1.fill_between(wvl, alb_min, alb_max, fc=color, lw=0.0, alpha=max(alpha-0.4, 0.0), zorder=zorder+100)
            ax1.errorbar(wvl, alb_mean, yerr=alb_std, color=color, alpha=max(alpha-0.2, 0.0), zorder=zorder+100)
            ax1.plot(wvl, alb_mean, lw=2.0, color=color, alpha=min(alpha+1.0, 1.0), zorder=zorder+100)

            if True:
                # save h5 file
                #╭────────────────────────────────────────────────────────────────────────────╮#
                fname = 'data_albedo_fake_%s.h5' % date_s
                h5f = h5py.File(fname, 'w')
                h5d_alb_mean = h5f.create_dataset('alb_mean', data=alb_mean, compression='gzip', compression_opts=9, chunks=True)
                h5d_wvl = h5f.create_dataset('wvl', data=wvl, compression='gzip', compression_opts=9, chunks=True)
                h5f.close()
                #╰────────────────────────────────────────────────────────────────────────────╯#


    fname_alb = 'data/Albedo_All.h5'

    # brandt and warren 2005
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f0 = h5py.File(fname_alb, 'r')
    brandt_wvl = f0['wvl_ice_thick_snow_2005'][...]
    brandt_alb = f0['albedo_ice_thick_snow_2005'][...]
    f0.close()
    ax1.plot(brandt_wvl, brandt_alb, color='purple', label="Ice+deep snow (Brandt et al. 2005)", marker='o', markeredgecolor='none', markersize=10, lw=1.5, zorder=1, alpha=0.7)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # Lyapustin albedo
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f0 = h5py.File(fname_alb, 'r')
    lya_wvl = f0['wvl_fresh_snow_2010'][...]
    lya_alb = f0['albedo_fresh_snow_2010'][...]
    f0.close()
    ax1.plot(lya_wvl, lya_alb, color='orange', label="Fresh snow (Lyapustin et al. 2010)", marker='o', markeredgecolor='none', markersize=10, lw=1.5, zorder=2, alpha=0.7)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # kay albedo
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f0 = h5py.File(fname_alb, 'r')
    jen_wvl = f0['wvl_dry_season_2013'][...]
    jen_wvl = f0['wvl_wet_season_2013'][...]
    a_dry = f0['albedo_dry_season_2013'][...]
    a_wet = f0['albedo_wet_season_2013'][...]
    f0.close()
    ax1.plot(jen_wvl, a_dry, color='k', label="Dry season (Kay and L'Ecuyer 2013)", marker='o', markeredgecolor='none', markersize=10, lw=1.5, zorder=3, alpha=0.7)
    ax1.plot(jen_wvl, a_wet, color='k', label="Wet season (Kay and L'Ecuyer 2013)", marker='s', markeredgecolor='none', markersize=10, lw=1.5, zorder=4, alpha=0.7)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # arise albedo
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f0 = h5py.File(fname_alb, 'r')
    arise_wvl = f0['wvl_arise'][...]
    arise_slope = f0['slope_arise'][...]
    arise_intercept = f0['intercept_arise'][...]
    arise_albedo = arise_slope*1.0 + arise_intercept
    f0.close()

    logic = np.repeat(True, arise_wvl.size)
    for absorb_band in absorb_bands:
        wvl_L, wvl_R = absorb_band
        logic0 = (arise_wvl>=wvl_L)&(arise_wvl<=wvl_R)
        logic[logic0] = False
    arise_wvl[~logic] = np.nan

    ax1.scatter(arise_wvl, arise_albedo, color='green', label="ARISE on 2014 September 11 (Chen et al., 2021)", marker='o', ec='none', s=10, zorder=0, alpha=0.8)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    ax1.xaxis.set_major_locator(FixedLocator(np.arange(200, 2201, 200)))
    ax1.set_xlim((200, 2200))
    ax1.set_ylim((0.0, 1.0))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Albedo')

    legend1 = ax1.legend(handles=patches_legend, loc='upper right', fontsize=18, numpoints=1, scatterpoints=3, markerscale=1)
    ax1.add_artist(legend1)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, loc='lower left', fontsize=14, numpoints=1, scatterpoints=3, markerscale=1.0)

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fig.savefig('%s.png' % (_metadata['Function']), bbox_inches='tight', metadata=_metadata, transparent=True)
    plt.show()
    #\----------------------------------------------------------------------------/#

def figure_agu_2024_sfc_alb_flt_trk_xy(
        fname,
        ):

    #╭────────────────────────────────────────────────────────────────────────────╮#
    dates = [
            datetime.datetime(2024, 6, 6), # [✓]
            datetime.datetime(2024, 7, 30), # [✓]
        ]

    colors = ['blue', 'red']

    tmhr_ranges = {
            '20240606': [16.7181, 16.7833],
            '20240730': [14.8723, 14.9375],
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # extract date and region information from filename
    #╭────────────────────────────────────────────────────────────────────────────╮#
    filename = os.path.basename(fname)
    info = filename.replace('.jpg', '').split('_')

    dtime_sat_s = '_'.join(info[1:3])
    dtime_sat = datetime.datetime.strptime(dtime_sat_s, '%Y-%m-%d_%H:%M:%S')
    jday_sat = er3t.util.dtime_to_jday(dtime_sat)
    date_sat_s = dtime_sat.strftime('%Y%m%d')

    inset_region = [-0.85e5, 1.65e5, 3.3e5, 5.8e5]

    extent = [float(item) for item in info[-1].replace('(', '').replace(')', '').split(',')]
    extent_xy = [float(item) for item in info[-2].replace('(', '').replace(')', '').split(',')]
    extent_plot = extent.copy()

    img = mpl_img.imread(fname)

    lon_c = (extent[0]+extent[1])/2.0
    lat_c = (extent[2]+extent[3])/2.0

    proj0 = ccrs.Orthographic(
            central_longitude=lon_c,
            central_latitude=lat_c,
            )
    #╰────────────────────────────────────────────────────────────────────────────╯#

    plt.close('all')
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(111, projection=proj0)

    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=False, ls='--', zorder=2)
    xlocators = np.arange(extent_plot[0], extent_plot[1]+1, 10.0)
    ylocators = np.arange(76.0, 86.9, 2.0)
    g1.xlocator = FixedLocator(np.arange(-180, 181, 10.0))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 2.0))
    for xlocator in xlocators:
        ax1.text(xlocator,  74.0, '$%d ^\\circ E$' % xlocator, ha='center', va='center', fontsize=12, color='k', transform=ccrs.PlateCarree(), bbox={'alpha':0.5, 'facecolor':'white', 'boxstyle':'round, pad=0.1'}, zorder=99)
    for jj, ylocator in enumerate(ylocators):
        ax1.text(-80.0, ylocator, '$%d ^\\circ N$' % ylocator, ha='center', va='center', fontsize=12, color='k', transform=ccrs.PlateCarree(), bbox={'alpha':0.5, 'facecolor':'white', 'boxstyle':'round, pad=0.1'}, zorder=99)

    if date_sat_s == '20240606':
        text = ax1.text(-130,  82.0, '', ha='left', va='top', fontsize=24, color=colors[0], transform=ccrs.PlateCarree(), bbox={'alpha':0.5, 'facecolor':'white', 'boxstyle':'round, pad=0.1'}, zorder=1000)
    else:
        text = ax1.text(-130,  82.0, '', ha='left', va='top', fontsize=24, color=colors[1], transform=ccrs.PlateCarree(), bbox={'alpha':0.5, 'facecolor':'white', 'boxstyle':'round, pad=0.1'}, zorder=1000)

    ax1.imshow(img, extent=extent_xy, zorder=0)
    text.set_text('%s (%s %s)' % (dtime_sat.strftime('%Y-%m-%d %H:%M'), *info[3:5][::-1]))

    ax_inset = ax1.inset_axes([0.58, 0.12, 0.40, 0.40], xlim=(inset_region[0], inset_region[1]), ylim=(inset_region[2], inset_region[3]), xticks=[], yticks=[], projection=proj0)
    for axis in ['top','bottom','left','right']:
        ax_inset.spines[axis].set_linewidth(1.0)
        ax_inset.spines[axis].set_zorder(200)
    ax_inset.imshow(img, extent=extent_xy, zorder=100, clip_on=True)
    tmp1, tmp2 = ax1.indicate_inset_zoom(ax_inset, edgecolor='black', lw=1.0, alpha=1.0)
    for tmp_ in tmp2:
        tmp_.set_linewidth(1.0)

    patches_legend = []
    for i, date in enumerate(dates):

        color = colors[i]
        date_s = date.strftime('%Y%m%d')
        if date_s == date_sat_s:
            alpha=1.0
        else:
            alpha=0.5

        fname_hsk = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)
        data_hsk = er3t.util.load_h5(fname_hsk)
        lon    = data_hsk['lon']
        lat    = data_hsk['lat']

        tmhr_range = tmhr_ranges[date_s]
        logic_select = (data_hsk['tmhr']>tmhr_range[0]) & (data_hsk['tmhr']<tmhr_range[1]) & \
                       (np.abs(data_hsk['ang_pit'])<=2.0) & (np.abs(data_hsk['ang_rol'])<=2.0)

        ax1.plot(lon, lat, lw=0.5, color=color, zorder=0, alpha=alpha, transform=ccrs.PlateCarree())
        ax1.plot(lon[logic_select], lat[logic_select], lw=4.0, color=color, zorder=1, alpha=alpha, transform=ccrs.PlateCarree())
        ax_inset.plot(lon, lat, lw=1.0, color=color, zorder=101, alpha=alpha, transform=ccrs.PlateCarree())
        ax_inset.plot(lon[logic_select], lat[logic_select], lw=6.0, color=color, zorder=102, alpha=alpha, transform=ccrs.PlateCarree())

    ax1.set_extent(extent, crs=ccrs.PlateCarree())

    # ax1.legend(handles=patches_legend, loc=(0.01, 0.68), fontsize=12, framealpha=0.5).set_zorder(300)
    ax1.axis('off')

    # save figure
    #╭──────────────────────────────────────────────────────────────╮#
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    #╰──────────────────────────────────────────────────────────────╯#
    print(filename)



def figure_agu_2024_cloud_wall_20240607_flt_trk_z(
        date=datetime.datetime(2024, 6, 7),
        ):

    # case specification
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tmhr_ranges_select = [[15.3694, 15.7473], [15.8556, 16.2338]]
    sat_select = [4, 5]
    xlim = [-55.0, -40.0]
    ylim1 = [0.0, 1.0]
    ylim2 = [0.0, 20.0]
    dy1 = 0.2
    dy2 = 4

    vname_x = 'lon'
    colors1 = ['r', 'b', 'g', 'brown']
    colors2 = ['magenta', 'cyan']
    # colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read in all logic data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # selected stacked legs
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0])&(data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        rcParams['font.size'] = 24
        plt.close('all')
        fig = plt.figure(figsize=(15, 5))
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax2_ = ax1.twinx()

        ax1.plot(data_hsk[vname_x], data_hsk['alt']/1000.0, color='k', lw=0.5, alpha=1.0)
        ax1.axhspan(0.3, 0.5, color='gray', alpha=0.5, zorder=0)

        for i in range(len(tmhr_ranges_select)):
            color = colors1[i]

            text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
            text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
            ax1.scatter(data_hsk[vname_x][logics_select[i]], data_hsk['alt'][logics_select[i]]/1000.0, color=color, s=20, lw=0.0, alpha=1.0)
            ax1.text(data_hsk[vname_x][logics_select[i]][0], data_hsk['alt'][logics_select[i]][0]/1000.0, text1, color=color, fontsize=20, alpha=1.0, va='bottom', ha='center', zorder=2)
            ax1.text(data_hsk[vname_x][logics_select[i]][-1], data_hsk['alt'][logics_select[i]][-1]/1000.0, text2, color=color, fontsize=20, alpha=1.0, va='bottom', ha='center', zorder=2)

        for i, index_sat in enumerate(sat_select):
            color = colors2[i]
            index0 = np.argmin(np.abs(data_sat['sat/jday'][index_sat]-data_hsk['jday']))
            ax1.scatter(data_hsk[vname_x][index0], data_hsk['alt'][index0]/1000.0, c=color, s=400, marker='*', lw=0.8, ec=color, alpha=1.0, zorder=1000)
            ax1.scatter(data_hsk[vname_x][index0], data_hsk['alt'][index0]/1000.0, fc=color, s=100, marker='*', lw=0.8, ec='k', alpha=1.0, zorder=1001)

            img_tag, sat_tag = fnames[index_sat].split('.')[0].split('_')[-2:]
            text0 = '%s (%s-%s)' % (er3t.util.jday_to_dtime(data_sat['sat/jday'][index_sat]).strftime('%H:%M'), sat_tag, img_tag)
            ax1.text(data_hsk[vname_x][index0], ylim1[1]*0.9, text0, color='k', fontsize=16, alpha=1.0, va='top', ha='left', rotation=270, zorder=100)
            ax1.axvline(data_hsk[vname_x][index0], color=color, ls='--', lw=1.0)

            logic0 = logics_select[0] & (data_sat['sat/cot'][:, index_sat]>0.0)
            ax2.scatter(data_hsk[vname_x][logic0], data_sat['sat/cot'][logic0, index_sat], s=40, fc='none', lw=0.8, ec=color, marker='D', alpha=0.5, zorder=0)

            # logic1 = logics_select[0] & (data_sat['sat/ctp'][:, index_sat]==1.0)
            # width = np.nanmin(np.abs((data_hsk[vname_x][logic1][1:])-(data_hsk[vname_x][logic1][:-1])))
            # bottom = np.repeat(ylim2[1]*0.98-0.02*i*ylim2[1], logic1.sum())
            # ax2_.bar(data_hsk[vname_x][logic1], np.repeat(0.02*ylim2[1], logic1.sum()), width=width, bottom=bottom, color=color, lw=0.0, zorder=10)

            # logic1 = logics_select[0] & (data_sat['sat/ctp'][:, index_sat]!=1.0) & (data_sat['sat/cot'][:, index_sat]<0.0)
            # width = np.nanmin(np.abs((data_hsk[vname_x][logic1][1:])-(data_hsk[vname_x][logic1][:-1])))
            # bottom = np.repeat(ylim2[1]*0.98-0.02*i*ylim2[1], logic1.sum())
            # ax2_.bar(data_hsk[vname_x][logic1], np.repeat(0.02*ylim2[1], logic1.sum()), width=width, bottom=bottom, color=color, lw=0.0, zorder=10, alpha=0.5)

            # ax2_.axhline(bottom[0], lw=0.5, color='k')


        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        ax2.set_zorder(0)
        ax2_.set_zorder(0)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Longitude [$^\\circ$]')
        ax1.set_ylabel('Altitude [km]')
        ax1.set_title('"Cloud Wall" on %s' % date.strftime('%Y-%m-%d'), fontsize=24)
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(-70.0, -39.0, 5.0)))
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(ylim1[0], ylim1[1]+0.1, dy1)))

        ax2.set_ylim(ylim2)
        ax2.yaxis.set_major_locator(FixedLocator(np.arange(ylim2[0], ylim2[1]+0.1, dy2)))
        ax2.set_ylabel('Cloud Optical Thickness', rotation=270, labelpad=24)

        ax2_.axis('off')
        ax2_.set_ylim(ylim2)
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s.png' % (_metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=True)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    return

def figure_agu_2024_cloud_wall_20240607_flt_trk_xy(
        fname,
        ):

    #╭────────────────────────────────────────────────────────────────────────────╮#
    dates = [
            datetime.datetime(2024, 6, 7), # [✓]
            datetime.datetime(2024, 6, 7), # [✓]
        ]

    colors = ['red', 'blue']

    tmhr_ranges = {
            '20240607-0': [15.3694, 15.7473],
            '20240607-1': [15.8556, 16.2338],
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # extract date and region information from filename
    #╭────────────────────────────────────────────────────────────────────────────╮#
    filename = os.path.basename(fname)
    info = filename.replace('.jpg', '').split('_')

    dtime_sat_s = '_'.join(info[1:3])
    dtime_sat = datetime.datetime.strptime(dtime_sat_s, '%Y-%m-%d_%H:%M:%S')
    jday_sat = er3t.util.dtime_to_jday(dtime_sat)
    date_sat_s = dtime_sat.strftime('%Y%m%d')

    inset_region = [-0.55e5, 1.95e5, 4.2e5, 6.7e5]

    extent = [float(item) for item in info[-1].replace('(', '').replace(')', '').split(',')]
    extent_xy = [float(item) for item in info[-2].replace('(', '').replace(')', '').split(',')]
    extent_plot = extent.copy()

    img = mpl_img.imread(fname)

    lon_c = (extent[0]+extent[1])/2.0
    lat_c = (extent[2]+extent[3])/2.0

    proj0 = ccrs.Orthographic(
            central_longitude=lon_c,
            central_latitude=lat_c,
            )
    #╰────────────────────────────────────────────────────────────────────────────╯#

    plt.close('all')
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(111, projection=proj0)

    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=False, ls='--', zorder=2)
    xlocators = np.arange(extent_plot[0], extent_plot[1]+1, 10.0)
    ylocators = np.arange(76.0, 86.9, 2.0)
    g1.xlocator = FixedLocator(np.arange(-180, 181, 10.0))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 2.0))
    for xlocator in xlocators:
        ax1.text(xlocator,  74.0, '$%d ^\\circ E$' % xlocator, ha='center', va='center', fontsize=12, color='k', transform=ccrs.PlateCarree(), bbox={'alpha':0.5, 'facecolor':'white', 'boxstyle':'round, pad=0.1'}, zorder=99)
    for jj, ylocator in enumerate(ylocators):
        ax1.text(-80.0, ylocator, '$%d ^\\circ N$' % ylocator, ha='center', va='center', fontsize=12, color='k', transform=ccrs.PlateCarree(), bbox={'alpha':0.5, 'facecolor':'white', 'boxstyle':'round, pad=0.1'}, zorder=99)

    if dtime_sat_s == '2024-06-07_15:20:00':
        text = ax1.text(-130,  82.0, '', ha='left', va='top', fontsize=24, color='k', transform=ccrs.PlateCarree(), bbox={'alpha':0.5, 'facecolor':'magenta', 'boxstyle':'round, pad=0.1'}, zorder=1000)
    else:
        text = ax1.text(-130,  82.0, '', ha='left', va='top', fontsize=24, color='k', transform=ccrs.PlateCarree(), bbox={'alpha':0.5, 'facecolor':'cyan', 'boxstyle':'round, pad=0.1'}, zorder=1000)

    ax1.imshow(img, extent=extent_xy, zorder=0)
    text.set_text('%s (%s %s)' % (dtime_sat.strftime('%Y-%m-%d %H:%M'), *info[3:5][::-1]))

    ax_inset = ax1.inset_axes([0.46, 0.24, 0.40, 0.40], xlim=(inset_region[0], inset_region[1]), ylim=(inset_region[2], inset_region[3]), xticks=[], yticks=[], projection=proj0)
    for axis in ['top','bottom','left','right']:
        ax_inset.spines[axis].set_linewidth(1.0)
        ax_inset.spines[axis].set_zorder(200)

    # read satellite granule
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_l2 = sorted(glob.glob('/argus/field/arcsix/sat-data/%s/*.%2.2d%2.2d.*' % (date_sat_s, dtime_sat.hour, dtime_sat.minute)))[0]
    if os.path.exists(fname_l2):
        f = Dataset(fname_l2, 'r')
        lon_s = f.groups['geolocation_data'].variables['longitude'][...].data
        lat_s = f.groups['geolocation_data'].variables['latitude'][...].data

        cot_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness'][...].data
        cot_pcl_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_PCL'][...].data

        cer_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius'][...].data
        cer_pcl_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_PCL'][...].data

        cth_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Height'][...].data)

        ctp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Phase_Optical_Properties'][...].data)

        scan_utc_s = f.groups['scan_line_attributes'].variables['scan_start_time'][...].data
        jday_s0 = np.array([er3t.util.dtime_to_jday(datetime.datetime(1993, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=scan_utc_s[i])) for i in range(scan_utc_s.size)])
        f.close()

        logic_pcl = (cot_s<0.0) & (cot_pcl_s>0.0) & (cer_s<0.0) & (cer_pcl_s>0.0)
        cot_s[logic_pcl] = cot_pcl_s[logic_pcl]
        cer_s[logic_pcl] = cer_pcl_s[logic_pcl]

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

        Nx, Ny = lon_s.shape
        jday_s = np.zeros((Nx, Ny), dtype=np.float64)
        jday_s[...] = np.repeat(jday_s0, int(Nx/jday_s0.size))[:, None]

        cot_s0 = cot_s.copy()
        cot_s0[...] = 255.0
        cot_s[cot_s<=0.0] = np.nan
        ctp_s[ctp_s==1.0] = np.nan

        # cs = ax_inset.pcolormesh(lon_s, lat_s,  ctp_s, cmap='viridis', vmin=0.0, vmax=5.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
        cs = ax_inset.pcolormesh(lon_s, lat_s,  cot_s, cmap='jet', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=1.0)
    else:
        ax_inset.imshow(img, extent=extent_xy, zorder=100, clip_on=True)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    tmp1, tmp2 = ax1.indicate_inset_zoom(ax_inset, edgecolor='black', lw=1.0, alpha=1.0)
    for tmp_ in tmp2:
        tmp_.set_linewidth(1.0)

    patches_legend = []
    for i, date in enumerate(dates):

        color = colors[i]
        date_s = date.strftime('%Y%m%d')
        alpha=1.0
        if i == 0:
            lw=6
        else:
            lw=2

        fname_hsk = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)
        data_hsk = er3t.util.load_h5(fname_hsk)
        lon    = data_hsk['lon']
        lat    = data_hsk['lat']

        tmhr_range = tmhr_ranges['%s-%d' % (date_s, i)]
        logic_select = (data_hsk['tmhr']>tmhr_range[0]) & (data_hsk['tmhr']<tmhr_range[1])

        if i == 0:
            ax1.plot(lon, lat, lw=0.8, color='gray', zorder=0, alpha=alpha, transform=ccrs.PlateCarree())
            ax_inset.plot(lon, lat, lw=0.8*1.5, color='gray', zorder=101, alpha=alpha, transform=ccrs.PlateCarree())
        ax1.plot(lon[logic_select], lat[logic_select], lw=lw, color=color, zorder=1, alpha=alpha, transform=ccrs.PlateCarree())
        ax_inset.plot(lon[logic_select], lat[logic_select], lw=lw*1.5, color=color, zorder=102, alpha=alpha, transform=ccrs.PlateCarree())

    ax1.set_extent(extent, crs=ccrs.PlateCarree())

    # ax1.legend(handles=patches_legend, loc=(0.01, 0.68), fontsize=12, framealpha=0.5).set_zorder(300)
    ax1.axis('off')

    divider = make_axes_locatable(ax_inset)
    cax = divider.append_axes('bottom', '5%', pad='2%', axes_class=maxes.Axes)
    cbar = fig.colorbar(cs, cax=cax, orientation='horizontal')
    cbar.set_label('Cloud Optical Thickness', labelpad=0.0)

    # save figure
    #╭──────────────────────────────────────────────────────────────╮#
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0, transparent=False)
    #╰──────────────────────────────────────────────────────────────╯#
    print(filename)





def old(
        date=datetime.datetime(2024, 6, 7),
        ):

    # case specification
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tmhr_ranges_select = [[15.3694, 15.7473], [15.8556, 16.2338]]
    sat_select = [4, 5]
    xlim = [-53.0, -43.0]
    ylim1 = [200.0, 600.0]
    ylim2 = [0.0, 20.0]
    dy1 = 100
    dy2 = 4

    vname_x = 'lon'
    colors1 = ['r', 'b']
    leg_tags = ['low', 'high']
    colors2 = ['magenta', 'cyan']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    date_s     = date.strftime('%Y%m%d')

    # read hsk data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    patches_legend = []

    rcParams['font.size'] = 24
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    # absorption bands
    #╭────────────────────────────────────────────────────────────────────────────╮#
    absorb_bands = [[649.0, 662.0], [676.0, 697.0], [707.0, 729.0], [743.0, 779.0], [796.0, 830.0], [911.0, 976.0], [1108.0, 1193.0], [1262.0, 1298.0], [1324.0, 1470.0], [1800.0, 1960.0]]
    for absorb_band in absorb_bands:
        wvl_L, wvl_R = absorb_band
        ax1.axvspan(wvl_L, wvl_R, color='k', lw=0.0, alpha=0.3, zorder=0)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    for i, tmhr_range in enumerate(tmhr_ranges_select):


        wvl = data_ssfr['zen/wvl'].copy()
        logic = np.repeat(True, wvl.size)

        for absorb_band in absorb_bands:
            wvl_L, wvl_R = absorb_band
            logic0 = (wvl>=wvl_L)&(wvl<=wvl_R)
            logic[logic0] = False
        wvl[~logic] = np.nan

        tmhr_range = tmhr_ranges[date_s]
        tmhr_range_legs = []
        logic_select = (data_hsk['tmhr']>tmhr_range[0]) & (data_hsk['tmhr']<tmhr_range[1]) & \
                       (np.abs(data_hsk['ang_pit'])<=2.0) & (np.abs(data_hsk['ang_rol'])<=2.0)

        color = colors[i]
        zorder = 10
        alpha = 0.6
        if date < datetime.datetime(2024, 7, 1):
            patches_legend.append(mpatches.Patch(color=color, alpha=1.0, label='ARCSIX-1 on 2024 %s' % (date.strftime('%B-%d').replace('-0', ' ').replace('-', ' '))))
        else:
            patches_legend.append(mpatches.Patch(color=color, alpha=1.0, label='ARCSIX-2 on 2024 %s' % (date.strftime('%B-%d').replace('-0', ' ').replace('-', ' '))))

        logic = (data_hsk['tmhr']>tmhr_range[0]) & (data_hsk['tmhr']<tmhr_range[1]) & \
                (np.abs(data_hsk['ang_pit'])<=2.0) & (np.abs(data_hsk['ang_rol'])<=2.0) & \
                (~np.isnan(data_ssfr['zen/flux'][:, np.argmin(np.abs(532.0-data_ssfr['zen/wvl']))])) & \
                (~np.isnan(data_ssfr['nad/flux'][:, np.argmin(np.abs(532.0-data_ssfr['nad/wvl']))])) & \
                ((data_ssfr['zen/flux'][:, np.argmin(np.abs(532.0-data_ssfr['zen/wvl']))])>0.0)

        if logic.sum() > 0:
            print(logic.sum())

            f_dn = data_ssfr['zen/flux'][logic, :]

            f_up_ = data_ssfr['nad/flux'][logic, :]
            f_up = np.zeros_like(f_dn)
            for ii_ in range(f_dn.shape[0]):
                f_up[ii_, :] = np.interp(data_ssfr['zen/wvl'], data_ssfr['nad/wvl'], f_up_[ii_, :])

            alb_all = f_up/f_dn
            alb_mean = np.nanmean(alb_all, axis=0)
            alb_std  = np.nanstd(alb_all, axis=0)
            alb_max  = np.nanmax(alb_all, axis=0)
            alb_min  = np.nanmin(alb_all, axis=0)

            alt_mean = np.mean(alt[logic])

            ax1.fill_between(wvl, alb_min, alb_max, fc=color, lw=0.0, alpha=max(alpha-0.4, 0.0), zorder=zorder+100)
            ax1.errorbar(wvl, alb_mean, yerr=alb_std, color=color, alpha=max(alpha-0.2, 0.0), zorder=zorder+100)
            ax1.plot(wvl, alb_mean, lw=2.0, color=color, alpha=min(alpha+1.0, 1.0), zorder=zorder+100)


    ax1.xaxis.set_major_locator(FixedLocator(np.arange(200, 2201, 200)))
    ax1.set_xlim((200, 2200))
    ax1.set_ylim((0.0, 1.0))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Albedo')

    legend1 = ax1.legend(handles=patches_legend, loc='upper right', fontsize=18, numpoints=1, scatterpoints=3, markerscale=1)
    ax1.add_artist(legend1)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, loc='lower left', fontsize=14, numpoints=1, scatterpoints=3, markerscale=1)

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fig.savefig('%s.png' % (_metadata['Function']), bbox_inches='tight', metadata=_metadata, transparent=True)
    plt.show()
    #\----------------------------------------------------------------------------/#

def figure_agu_2024_cloud_wall_20240607_flux_low_leg_time_series(
        date=datetime.datetime(2024, 6, 7),
        scale_up=1.0,
        scale_dn=1.0,
        ):

    # case specification
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tmhr_ranges_select = [[15.3694, 15.7473], [15.8556, 16.2338]]
    sat_select = [4, 5]
    xlim = [-53.0, -43.0]
    ylim1 = [0.0, 600.0]
    ylim2 = [0.0, 20.0]
    dy1 = 200
    dy2 = 4

    vname_x = 'lon'
    colors1 = ['r', 'b']
    leg_tags = ['low', 'high']
    colors2 = ['magenta', 'cyan']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    wvl = data_ssfr['zen/wvl']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read in all logic data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # selected stacked legs
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0]) & (data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        rcParams['font.size'] = 24
        plt.close('all')
        fig = plt.figure(figsize=(8, 5))
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)

        for i in [0]:
            color = colors1[i]

            if i == 0:
                alpha_up = 0.3
                alpha_dn = 0.5
                color_dn1 = colors1[i]
                color_up1 = 'k'
                color_dn2 = colors2[i]
                color_up2 = 'gray'
            elif i == 1:
                alpha_up = 0.5
                alpha_dn = 0.3
                color_up1 = colors1[i]
                color_dn1 = 'k'
                color_up2 = colors2[i]
                color_dn2 = 'gray'

            # satellite parsing
            #╭────────────────────────────────────────────────────────────────────────────╮#
            index_sat = sat_select[i]
            img_tag, sat_tag = fnames[index_sat].split('.')[0].split('_')[-2:]
            text0 = '%s-%s at %s' % (sat_tag, img_tag, er3t.util.jday_to_dtime(data_sat['sat/jday'][index_sat]).strftime('%H:%M'))
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # read rt calculations
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fname_alb = 'data_albedo_20240607_%s.h5' % (leg_tags[i])
            data_alb = er3t.util.load_h5(fname_alb)
            fname_rtm = 'data_rtm_lrt_cloud_20240607_%s.h5' % (leg_tags[i])
            data_rtm = er3t.util.load_h5(fname_rtm)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # logic_select
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rtm_indices = data_alb['indices']
            logic_rtm = np.repeat(False, data_ssfr['jday'].size)
            logic_alb = (np.sum(np.isnan(data_alb['albedo_interp']), axis=-1)<80) & (data_rtm['f_dn'][:, 100]>0.0) & (data_rtm['f_up'][:, 100]>0.0)
            logic_rtm[rtm_indices[logic_alb]] = True

            logic_select = logics_select[i] &\
                          (data_ssfr['nad/flux'][:, 100]>0.0) &\
                          (data_ssfr['zen/flux'][:, 100]>0.0) &\
                          logic_rtm

            logic_select_rtm = logic_select[rtm_indices]
            #╰────────────────────────────────────────────────────────────────────────────╯#


            # ssfr upwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_ssfr['nad/wvl']
            ssfr_f_up_ = data_ssfr['nad/flux'][logic_select, :]

            Nselect = logic_select.sum()
            ssfr_f_up = np.zeros(Nselect, dtype=np.float32)
            for iselect in range(Nselect):
                logic_wvl = ~np.isnan(ssfr_f_up_[iselect, :])
                wvl = wvl_.copy()[logic_wvl]
                ssfr_f_up[iselect] = np.trapz(ssfr_f_up_[iselect, logic_wvl], x=wvl)

            ax1.scatter(data_hsk[vname_x][logic_select], ssfr_f_up/scale_up, marker='^', color=color_up1, s=40, alpha=alpha_up, label='SSFR$_\\uparrow$')
            print(Nselect)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # ssfr downwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_ssfr['zen/wvl']
            ssfr_f_dn_ = data_ssfr['zen/flux'][logic_select, :]

            Nselect = logic_select.sum()
            ssfr_f_dn = np.zeros(Nselect, dtype=np.float32)
            for iselect in range(Nselect):
                logic_wvl = ~np.isnan(ssfr_f_dn_[iselect, :])
                wvl = wvl_.copy()[logic_wvl]
                ssfr_f_dn[iselect] = np.trapz(ssfr_f_dn_[iselect, logic_wvl], x=wvl)

            ax1.scatter(data_hsk[vname_x][logic_select], ssfr_f_dn/scale_dn, marker='v', color=color_dn1, s=40, alpha=alpha_dn, label='SSFR$_\\downarrow$')
            print(Nselect)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm upwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_alb['wvl']
            rtm_f_up_ = data_rtm['f_up'][logic_select_rtm, :]

            Nselect = logic_select_rtm.sum()
            rtm_f_up = np.zeros(Nselect, dtype=np.float32)
            for iselect in range(Nselect):
                logic_wvl = ~np.isnan(data_alb['albedo_interp'][logic_select_rtm, :][iselect, :])
                wvl = wvl_.copy()[logic_wvl]
                rtm_f_up[iselect] = np.trapz(rtm_f_up_[iselect, logic_wvl], x=wvl)

            ax1.scatter(data_hsk[vname_x][logic_select], rtm_f_up, marker='^', color=color_up2, s=40, alpha=alpha_up, label='RTM$_\\uparrow$')
            print(Nselect)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm downwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_alb['wvl']
            rtm_f_dn_ = data_rtm['f_dn'][logic_select_rtm, :]

            Nselect = logic_select_rtm.sum()
            rtm_f_dn = np.zeros(Nselect, dtype=np.float32)
            for iselect in range(Nselect):
                logic_wvl = ~np.isnan(data_alb['albedo_interp'][logic_select_rtm, :][iselect, :])
                wvl = wvl_.copy()[logic_wvl]
                rtm_f_dn[iselect] = np.trapz(rtm_f_dn_[iselect, logic_wvl], x=wvl)

            ax1.scatter(data_hsk[vname_x][logic_select], rtm_f_dn, marker='v', color=color_dn2, s=40, alpha=alpha_dn, label='RTM$_\\downarrow$')
            print(Nselect)
            #╰────────────────────────────────────────────────────────────────────────────╯#

        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        # ax2.set_zorder(0)
        # ax2_.set_zorder(0)
        ax1.set_xlabel('Longitude [$^\\circ$]')
        ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2}}$]')
        ax1.set_xlim(xlim)
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(-70.0, -39.0, 5.0)))
        ax1.set_ylim(ylim1)
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(ylim1[0], ylim1[1]+0.1, dy1)))
        ax1.set_title('Low Leg (%s)' % text0, color=colors2[i], y=1.01)

        patches_legend = [
                          mpatches.Patch(color=color_dn1, label='SSFR$_\\downarrow$'), \
                          mpatches.Patch(color=color_dn2, label='RTM$_\\downarrow$'), \
                          mpatches.Patch(color=color_up1, label='SSFR$_\\uparrow$'), \
                          mpatches.Patch(color=color_up2, label='RTM$_\\uparrow$'), \
                         ]
        ax1.legend(handles=patches_legend, loc='lower right', fontsize=18)

        # ax2.set_ylim(ylim2)
        # ax2.yaxis.set_major_locator(FixedLocator(np.arange(ylim2[0], ylim2[1]+0.1, dy2)))
        # ax2.set_ylabel('Cloud Optical Thickness', rotation=270, labelpad=24)

        # ax2_.axis('off')
        # ax2_.set_ylim(ylim2)
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s.png' % (_metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=True)
        #╰──────────────────────────────────────────────────────────────╯#
        # plt.show()
        # sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    return

def figure_agu_2024_cloud_wall_20240607_flux_high_leg_time_series(
        date=datetime.datetime(2024, 6, 7),
        scale_up=1.0,
        scale_dn=1.0,
        ):

    # case specification
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tmhr_ranges_select = [[15.3694, 15.7473], [15.8556, 16.2338]]
    sat_select = [4, 5]
    xlim = [-53.0, -43.0]
    ylim1 = [0.0, 600.0]
    ylim2 = [0.0, 20.0]
    dy1 = 200
    dy2 = 4

    vname_x = 'lon'
    colors1 = ['r', 'b']
    leg_tags = ['low', 'high']
    colors2 = ['magenta', 'cyan']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    wvl = data_ssfr['zen/wvl']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read in all logic data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # selected stacked legs
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0]) & (data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        rcParams['font.size'] = 24
        plt.close('all')
        fig = plt.figure(figsize=(8, 5))
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)

        for i in [1]:
            color = colors1[i]

            if i == 0:
                alpha_up = 0.3
                alpha_dn = 0.5
                color_dn1 = colors1[i]
                color_up1 = 'k'
                color_dn2 = colors2[i]
                color_up2 = 'gray'
            elif i == 1:
                alpha_up = 0.5
                alpha_dn = 0.3
                color_up1 = colors1[i]
                color_dn1 = 'k'
                color_up2 = colors2[i]
                color_dn2 = 'gray'

            # satellite parsing
            #╭────────────────────────────────────────────────────────────────────────────╮#
            index_sat = sat_select[i]
            img_tag, sat_tag = fnames[index_sat].split('.')[0].split('_')[-2:]
            text0 = '%s-%s at %s' % (sat_tag, img_tag, er3t.util.jday_to_dtime(data_sat['sat/jday'][index_sat]).strftime('%H:%M'))
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # read rt calculations
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fname_alb = 'data_albedo_20240607_%s.h5' % (leg_tags[i])
            data_alb = er3t.util.load_h5(fname_alb)
            fname_rtm = 'data_rtm_lrt_cloud_20240607_%s.h5' % (leg_tags[i])
            data_rtm = er3t.util.load_h5(fname_rtm)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # logic_select
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rtm_indices = data_alb['indices']
            logic_rtm = np.repeat(False, data_ssfr['jday'].size)
            logic_alb = (np.sum(np.isnan(data_alb['albedo_interp']), axis=-1)<80) & (data_rtm['f_dn'][:, 100]>0.0) & (data_rtm['f_up'][:, 100]>0.0)
            logic_rtm[rtm_indices[logic_alb]] = True

            logic_select = logics_select[i] &\
                          (data_ssfr['nad/flux'][:, 100]>0.0) &\
                          (data_ssfr['zen/flux'][:, 100]>0.0) &\
                          logic_rtm

            logic_select_rtm = logic_select[rtm_indices]
            #╰────────────────────────────────────────────────────────────────────────────╯#


            # ssfr upwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_ssfr['nad/wvl']
            ssfr_f_up_ = data_ssfr['nad/flux'][logic_select, :]

            Nselect = logic_select.sum()
            ssfr_f_up = np.zeros(Nselect, dtype=np.float32)
            for iselect in range(Nselect):
                logic_wvl = ~np.isnan(ssfr_f_up_[iselect, :])
                wvl = wvl_.copy()[logic_wvl]
                ssfr_f_up[iselect] = np.trapz(ssfr_f_up_[iselect, logic_wvl], x=wvl)

            ax1.scatter(data_hsk[vname_x][logic_select], ssfr_f_up/scale_up, marker='^', color=color_up1, s=40, alpha=alpha_up, label='SSFR$_\\uparrow$')
            print(Nselect)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # ssfr downwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_ssfr['zen/wvl']
            ssfr_f_dn_ = data_ssfr['zen/flux'][logic_select, :]

            Nselect = logic_select.sum()
            ssfr_f_dn = np.zeros(Nselect, dtype=np.float32)
            for iselect in range(Nselect):
                logic_wvl = ~np.isnan(ssfr_f_dn_[iselect, :])
                wvl = wvl_.copy()[logic_wvl]
                ssfr_f_dn[iselect] = np.trapz(ssfr_f_dn_[iselect, logic_wvl], x=wvl)

            ax1.scatter(data_hsk[vname_x][logic_select], ssfr_f_dn/scale_dn, marker='v', color=color_dn1, s=40, alpha=alpha_dn, label='SSFR$_\\downarrow$')
            print(Nselect)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm upwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_alb['wvl']
            rtm_f_up_ = data_rtm['f_up'][logic_select_rtm, :]

            Nselect = logic_select_rtm.sum()
            rtm_f_up = np.zeros(Nselect, dtype=np.float32)
            for iselect in range(Nselect):
                logic_wvl = ~np.isnan(data_alb['albedo_interp'][logic_select_rtm, :][iselect, :])
                wvl = wvl_.copy()[logic_wvl]
                rtm_f_up[iselect] = np.trapz(rtm_f_up_[iselect, logic_wvl], x=wvl)

            ax1.scatter(data_hsk[vname_x][logic_select], rtm_f_up, marker='^', color=color_up2, s=40, alpha=alpha_up, label='RTM$_\\uparrow$')
            print(Nselect)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm downwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_alb['wvl']
            rtm_f_dn_ = data_rtm['f_dn'][logic_select_rtm, :]

            Nselect = logic_select_rtm.sum()
            rtm_f_dn = np.zeros(Nselect, dtype=np.float32)
            for iselect in range(Nselect):
                logic_wvl = ~np.isnan(data_alb['albedo_interp'][logic_select_rtm, :][iselect, :])
                wvl = wvl_.copy()[logic_wvl]
                rtm_f_dn[iselect] = np.trapz(rtm_f_dn_[iselect, logic_wvl], x=wvl)

            ax1.scatter(data_hsk[vname_x][logic_select], rtm_f_dn, marker='v', color=color_dn2, s=40, alpha=alpha_dn, label='RTM$_\\downarrow$')
            print(Nselect)
            #╰────────────────────────────────────────────────────────────────────────────╯#

        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        # ax2.set_zorder(0)
        # ax2_.set_zorder(0)
        ax1.set_xlabel('Longitude [$^\\circ$]')
        ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2}}$]')
        ax1.set_xlim(xlim)
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(-70.0, -39.0, 5.0)))
        ax1.set_ylim(ylim1)
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(ylim1[0], ylim1[1]+0.1, dy1)))
        ax1.set_title('High Leg (%s)' % text0, color=colors2[i], y=1.01)

        patches_legend = [
                          mpatches.Patch(color=color_dn1, label='SSFR$_\\downarrow$'), \
                          mpatches.Patch(color=color_dn2, label='RTM$_\\downarrow$'), \
                          mpatches.Patch(color=color_up1, label='SSFR$_\\uparrow$'), \
                          mpatches.Patch(color=color_up2, label='RTM$_\\uparrow$'), \
                         ]
        ax1.legend(handles=patches_legend, loc='lower right', fontsize=18)

        # ax2.set_ylim(ylim2)
        # ax2.yaxis.set_major_locator(FixedLocator(np.arange(ylim2[0], ylim2[1]+0.1, dy2)))
        # ax2.set_ylabel('Cloud Optical Thickness', rotation=270, labelpad=24)

        # ax2_.axis('off')
        # ax2_.set_ylim(ylim2)
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s.png' % (_metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=True)
        #╰──────────────────────────────────────────────────────────────╯#
        # plt.show()
        # sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    return




def figure_agu_2024_cloud_wall_20240607_flux_low_leg_spectral(
        date=datetime.datetime(2024, 6, 7),
        scale_up=1.0,
        scale_dn=1.0,
        ):

    # case specification
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tmhr_ranges_select = [[15.3694, 15.7473], [15.8556, 16.2338]]
    sat_select = [4, 5]
    xlim = [200, 2200]
    ylim1 = [0, 1.2]
    ylim2 = [0.0, 20.0]
    dy1 = 0.4
    dy2 = 4

    vname_x = 'lon'
    colors1 = ['r', 'b']
    leg_tags = ['low', 'high']
    colors2 = ['magenta', 'cyan']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    wvl = data_ssfr['zen/wvl']
    data_ssfr['zen/flux'][:, wvl<=1150.0] *= 0.97
    data_ssfr['zen/flux'][:, wvl<=750.0] *= 0.97
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read in all logic data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # selected stacked legs
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0]) & (data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        rcParams['font.size'] = 24
        plt.close('all')
        fig = plt.figure(figsize=(8, 5))
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)

        for i in [0]:
            color = colors1[i]

            if i == 0:
                alpha_up = 0.3
                alpha_dn = 0.5
                color_dn1 = colors1[i]
                color_up1 = 'k'
                color_dn2 = colors2[i]
                color_up2 = 'gray'
            elif i == 1:
                alpha_up = 0.5
                alpha_dn = 0.3
                color_up1 = colors1[i]
                color_dn1 = 'k'
                color_up2 = colors2[i]
                color_dn2 = 'gray'

            # satellite parsing
            #╭────────────────────────────────────────────────────────────────────────────╮#
            index_sat = sat_select[i]
            img_tag, sat_tag = fnames[index_sat].split('.')[0].split('_')[-2:]
            text0 = '%s-%s at %s' % (sat_tag, img_tag, er3t.util.jday_to_dtime(data_sat['sat/jday'][index_sat]).strftime('%H:%M'))
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # read rt calculations
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fname_alb = 'data_albedo_20240607_%s.h5' % (leg_tags[i])
            data_alb = er3t.util.load_h5(fname_alb)
            fname_rtm = 'data_rtm_lrt_cloud_20240607_%s.h5' % (leg_tags[i])
            data_rtm = er3t.util.load_h5(fname_rtm)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # logic_select
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rtm_indices = data_alb['indices']
            logic_rtm = np.repeat(False, data_ssfr['jday'].size)
            logic_alb = (np.sum(np.isnan(data_alb['albedo_interp']), axis=-1)<80) & (data_rtm['f_dn'][:, 100]>0.0) & (data_rtm['f_up'][:, 100]>0.0)
            logic_rtm[rtm_indices[logic_alb]] = True

            logic_select = logics_select[i] &\
                          (data_ssfr['nad/flux'][:, 100]>0.0) &\
                          (data_ssfr['zen/flux'][:, 100]>0.0) &\
                          logic_rtm

            logic_select_rtm = logic_select[rtm_indices]
            #╰────────────────────────────────────────────────────────────────────────────╯#


            # ssfr upwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_ssfr['nad/wvl']
            ssfr_f_up_ = data_ssfr['nad/flux'][logic_select, :]

            ssfr_f_up_mean = np.nanmean(ssfr_f_up_, axis=0)
            ssfr_f_up_std  = np.nanstd(ssfr_f_up_, axis=0)
            ssfr_f_up_max  = np.nanmax(ssfr_f_up_, axis=0)
            ssfr_f_up_min  = np.nanmin(ssfr_f_up_, axis=0)

            # ax1.fill_between(wvl_, ssfr_f_up_min, ssfr_f_up_max, fc=color_up1, lw=0.0, alpha=alpha_up, zorder=100)
            ax1.fill_between(wvl_, ssfr_f_up_mean-ssfr_f_up_std, ssfr_f_up_mean+ssfr_f_up_std, fc=color_up1, lw=0.0, alpha=alpha_up, zorder=100)
            # ax1.errorbar(wvl_, ssfr_f_up_mean, yerr=ssfr_f_up_std, color=color_up1, alpha=alpha_up, zorder=200, lw=0.5)
            ax1.plot(wvl_, ssfr_f_up_mean, lw=1.0, color=color_up1, alpha=1.0, zorder=300, marker='^', markersize=0)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # ssfr downwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_ssfr['zen/wvl']
            ssfr_f_dn_ = data_ssfr['zen/flux'][logic_select, :]

            ssfr_f_dn_mean = np.nanmean(ssfr_f_dn_, axis=0)
            ssfr_f_dn_std  = np.nanstd(ssfr_f_dn_, axis=0)
            ssfr_f_dn_max  = np.nanmax(ssfr_f_dn_, axis=0)
            ssfr_f_dn_min  = np.nanmin(ssfr_f_dn_, axis=0)

            # ax1.fill_between(wvl_, ssfr_f_dn_min, ssfr_f_dn_max, fc=color_dn1, lw=0.0, alpha=alpha_dn, zorder=100)
            ax1.fill_between(wvl_, ssfr_f_dn_mean-ssfr_f_dn_std, ssfr_f_dn_mean+ssfr_f_dn_std, fc=color_dn1, lw=0.0, alpha=alpha_dn, zorder=100)
            # ax1.errorbar(wvl_, ssfr_f_dn_mean, yerr=ssfr_f_dn_std, color=color_dn1, alpha=alpha_dn, zorder=200, lw=0.5)
            ax1.plot(wvl_, ssfr_f_dn_mean, lw=1.0, color=color_dn1, alpha=1.0, zorder=300, marker='v', markersize=0)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm upwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_alb['wvl']
            rtm_f_up_ = data_rtm['f_up'][logic_select_rtm, :]

            rtm_f_up_mean = np.nanmean(rtm_f_up_, axis=0)
            rtm_f_up_std  = np.nanstd(rtm_f_up_, axis=0)
            rtm_f_up_max  = np.nanmax(rtm_f_up_, axis=0)
            rtm_f_up_min  = np.nanmin(rtm_f_up_, axis=0)

            # ax1.fill_between(wvl_, rtm_f_up_min, rtm_f_up_max, fc=color_up2, lw=0.0, alpha=alpha_up, zorder=100)
            ax1.fill_between(wvl_, rtm_f_up_mean-rtm_f_up_std, rtm_f_up_mean+rtm_f_up_std, fc=color_up2, lw=0.0, alpha=alpha_up, zorder=100)
            # ax1.errorbar(wvl_, rtm_f_up_mean, yerr=rtm_f_up_std, color=color_up2, alpha=alpha_up, zorder=200, lw=0.5)
            ax1.plot(wvl_, rtm_f_up_mean, lw=1.0, color=color_up2, alpha=1.0, zorder=300, marker='^', markersize=0)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm downwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_alb['wvl']
            rtm_f_dn_ = data_rtm['f_dn'][logic_select_rtm, :]

            rtm_f_dn_mean = np.nanmean(rtm_f_dn_, axis=0)
            rtm_f_dn_std  = np.nanstd(rtm_f_dn_, axis=0)
            rtm_f_dn_max  = np.nanmax(rtm_f_dn_, axis=0)
            rtm_f_dn_min  = np.nanmin(rtm_f_dn_, axis=0)

            # ax1.fill_between(wvl_, rtm_f_dn_min, rtm_f_dn_max, fc=color_dn2, lw=0.0, alpha=alpha_dn, zorder=100)
            ax1.fill_between(wvl_, rtm_f_dn_mean-rtm_f_dn_std, rtm_f_dn_mean+rtm_f_dn_std, fc=color_dn2, lw=0.0, alpha=alpha_dn, zorder=100)
            # ax1.errorbar(wvl_, rtm_f_dn_mean, yerr=rtm_f_dn_std, color=color_dn2, alpha=alpha_dn, zorder=200, lw=0.5)
            ax1.plot(wvl_, rtm_f_dn_mean, lw=1.0, color=color_dn2, alpha=1.0, zorder=300, marker='v', markersize=0)
            #╰────────────────────────────────────────────────────────────────────────────╯#

        ax1.axhline(0.0, ls='--', lw=1.0)

        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_xlim(xlim)
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 2401, 400)))
        ax1.set_ylim(ylim1)
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(ylim1[0], ylim1[1]+0.1, dy1)))
        ax1.set_title('Low Leg (%s)' % text0, color=colors2[i], y=1.01)

        patches_legend = [
                          mpatches.Patch(color=color_dn1, label='SSFR$_\\downarrow$'), \
                          mpatches.Patch(color=color_dn2, label='RTM$_\\downarrow$'), \
                          mpatches.Patch(color=color_up1, label='SSFR$_\\uparrow$'), \
                          mpatches.Patch(color=color_up2, label='RTM$_\\uparrow$'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=18)

        # ax2.set_ylim(ylim2)
        # ax2.yaxis.set_major_locator(FixedLocator(np.arange(ylim2[0], ylim2[1]+0.1, dy2)))
        # ax2.set_ylabel('Cloud Optical Thickness', rotation=270, labelpad=24)

        # ax2_.axis('off')
        # ax2_.set_ylim(ylim2)
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s.png' % (_metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=True)
        #╰──────────────────────────────────────────────────────────────╯#
        # plt.show()
        # sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    return

def figure_agu_2024_cloud_wall_20240607_flux_high_leg_spectral(
        date=datetime.datetime(2024, 6, 7),
        scale_up=1.0,
        scale_dn=1.0,
        ):

    # case specification
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tmhr_ranges_select = [[15.3694, 15.7473], [15.8556, 16.2338]]
    sat_select = [4, 5]
    xlim = [200, 2200]
    ylim1 = [0, 1.2]
    ylim2 = [0.0, 20.0]
    dy1 = 0.4
    dy2 = 4

    vname_x = 'lon'
    colors1 = ['r', 'b']
    leg_tags = ['low', 'high']
    colors2 = ['magenta', 'cyan']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    wvl = data_ssfr['zen/wvl']
    data_ssfr['zen/flux'][:, wvl<=1150.0] *= 0.97
    data_ssfr['zen/flux'][:, wvl<=750.0] *= 0.97
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read in all logic data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # selected stacked legs
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0]) & (data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        rcParams['font.size'] = 24
        plt.close('all')
        fig = plt.figure(figsize=(8, 5))
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        for i in [1]:
            color = colors1[i]

            if i == 0:
                alpha_up = 0.3
                alpha_dn = 0.5
                color_dn1 = colors1[i]
                color_up1 = 'k'
                color_dn2 = colors2[i]
                color_up2 = 'gray'
            elif i == 1:
                alpha_up = 0.5
                alpha_dn = 0.3
                color_up1 = colors1[i]
                color_dn1 = 'k'
                color_up2 = colors2[i]
                color_dn2 = 'gray'

            # satellite parsing
            #╭────────────────────────────────────────────────────────────────────────────╮#
            index_sat = sat_select[i]
            img_tag, sat_tag = fnames[index_sat].split('.')[0].split('_')[-2:]
            text0 = '%s-%s at %s' % (sat_tag, img_tag, er3t.util.jday_to_dtime(data_sat['sat/jday'][index_sat]).strftime('%H:%M'))
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # read rt calculations
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fname_alb = 'data_albedo_20240607_%s.h5' % (leg_tags[i])
            data_alb = er3t.util.load_h5(fname_alb)
            fname_rtm = 'data_rtm_lrt_cloud_20240607_%s.h5' % (leg_tags[i])
            data_rtm = er3t.util.load_h5(fname_rtm)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # logic_select
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rtm_indices = data_alb['indices']
            logic_rtm = np.repeat(False, data_ssfr['jday'].size)
            logic_alb = (np.sum(np.isnan(data_alb['albedo_interp']), axis=-1)<80) & (data_rtm['f_dn'][:, 100]>0.0) & (data_rtm['f_up'][:, 100]>0.0)
            logic_rtm[rtm_indices[logic_alb]] = True

            logic_select = logics_select[i] &\
                          (data_ssfr['nad/flux'][:, 100]>0.0) &\
                          (data_ssfr['zen/flux'][:, 100]>0.0) &\
                          logic_rtm

            logic_select_rtm = logic_select[rtm_indices]
            #╰────────────────────────────────────────────────────────────────────────────╯#


            # ssfr upwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_ssfr['nad/wvl']
            ssfr_f_up_ = data_ssfr['nad/flux'][logic_select, :]

            ssfr_f_up_mean = np.nanmean(ssfr_f_up_, axis=0)
            ssfr_f_up_std  = np.nanstd(ssfr_f_up_, axis=0)
            ssfr_f_up_max  = np.nanmax(ssfr_f_up_, axis=0)
            ssfr_f_up_min  = np.nanmin(ssfr_f_up_, axis=0)

            # ax1.fill_between(wvl_, ssfr_f_up_min, ssfr_f_up_max, fc=color_up1, lw=0.0, alpha=alpha_up, zorder=100)
            ax1.fill_between(wvl_, ssfr_f_up_mean-ssfr_f_up_std, ssfr_f_up_mean+ssfr_f_up_std, fc=color_up1, lw=0.0, alpha=alpha_up, zorder=100)
            # ax1.errorbar(wvl_, ssfr_f_up_mean, yerr=ssfr_f_up_std, color=color_up1, alpha=alpha_up, zorder=200, lw=0.5)
            ax1.plot(wvl_, ssfr_f_up_mean, lw=1.0, color=color_up1, alpha=1.0, zorder=300, marker='^', markersize=0)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # ssfr downwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_ssfr['zen/wvl']
            ssfr_f_dn_ = data_ssfr['zen/flux'][logic_select, :]

            ssfr_f_dn_mean = np.nanmean(ssfr_f_dn_, axis=0)
            ssfr_f_dn_std  = np.nanstd(ssfr_f_dn_, axis=0)
            ssfr_f_dn_max  = np.nanmax(ssfr_f_dn_, axis=0)
            ssfr_f_dn_min  = np.nanmin(ssfr_f_dn_, axis=0)

            # ax1.fill_between(wvl_, ssfr_f_dn_min, ssfr_f_dn_max, fc=color_dn1, lw=0.0, alpha=alpha_dn, zorder=100)
            ax1.fill_between(wvl_, ssfr_f_dn_mean-ssfr_f_dn_std, ssfr_f_dn_mean+ssfr_f_dn_std, fc=color_dn1, lw=0.0, alpha=alpha_dn, zorder=100)
            # ax1.errorbar(wvl_, ssfr_f_dn_mean, yerr=ssfr_f_dn_std, color=color_dn1, alpha=alpha_dn, zorder=200, lw=0.5)
            ax1.plot(wvl_, ssfr_f_dn_mean, lw=1.0, color=color_dn1, alpha=1.0, zorder=300, marker='v', markersize=0)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm upwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_alb['wvl']
            rtm_f_up_ = data_rtm['f_up'][logic_select_rtm, :]

            rtm_f_up_mean = np.nanmean(rtm_f_up_, axis=0)
            rtm_f_up_std  = np.nanstd(rtm_f_up_, axis=0)
            rtm_f_up_max  = np.nanmax(rtm_f_up_, axis=0)
            rtm_f_up_min  = np.nanmin(rtm_f_up_, axis=0)

            # ax1.fill_between(wvl_, rtm_f_up_min, rtm_f_up_max, fc=color_up2, lw=0.0, alpha=alpha_up, zorder=100)
            ax1.fill_between(wvl_, rtm_f_up_mean-rtm_f_up_std, rtm_f_up_mean+rtm_f_up_std, fc=color_up2, lw=0.0, alpha=alpha_up, zorder=100)
            # ax1.errorbar(wvl_, rtm_f_up_mean, yerr=rtm_f_up_std, color=color_up2, alpha=alpha_up, zorder=200, lw=0.5)
            ax1.plot(wvl_, rtm_f_up_mean, lw=1.0, color=color_up2, alpha=1.0, zorder=300, marker='^', markersize=0)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm downwelling
            #╭────────────────────────────────────────────────────────────────────────────╮#
            wvl_ = data_alb['wvl']
            rtm_f_dn_ = data_rtm['f_dn'][logic_select_rtm, :]

            rtm_f_dn_mean = np.nanmean(rtm_f_dn_, axis=0)
            rtm_f_dn_std  = np.nanstd(rtm_f_dn_, axis=0)
            rtm_f_dn_max  = np.nanmax(rtm_f_dn_, axis=0)
            rtm_f_dn_min  = np.nanmin(rtm_f_dn_, axis=0)

            # ax1.fill_between(wvl_, rtm_f_dn_min, rtm_f_dn_max, fc=color_dn2, lw=0.0, alpha=alpha_dn, zorder=100)
            ax1.fill_between(wvl_, rtm_f_dn_mean-rtm_f_dn_std, rtm_f_dn_mean+rtm_f_dn_std, fc=color_dn2, lw=0.0, alpha=alpha_dn, zorder=100)
            # ax1.errorbar(wvl_, rtm_f_dn_mean, yerr=rtm_f_dn_std, color=color_dn2, alpha=alpha_dn, zorder=200, lw=0.5)
            ax1.plot(wvl_, rtm_f_dn_mean, lw=1.0, color=color_dn2, alpha=1.0, zorder=300, marker='v', markersize=0)
            #╰────────────────────────────────────────────────────────────────────────────╯#

        # ax2.scatter(wvl_, rtm_f_dn_mean/ssfr_f_dn_mean, s=20, c='purple')
        ax2.scatter(wvl_, ssfr_f_dn_mean/rtm_f_dn_mean, s=20, c='purple')
        ax2.set_ylim(0.5, 1.5)
        ax2.axhline(1.0, color='k')

        ax1.axhline(0.0, ls='--', lw=1.0)

        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_xlim(xlim)
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 2401, 400)))
        ax1.set_ylim(ylim1)
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(ylim1[0], ylim1[1]+0.1, dy1)))
        ax1.set_title('High Leg (%s)' % text0, color=colors2[i], y=1.01)

        patches_legend = [
                          mpatches.Patch(color=color_dn1, label='SSFR$_\\downarrow$'), \
                          mpatches.Patch(color=color_dn2, label='RTM$_\\downarrow$'), \
                          mpatches.Patch(color=color_up1, label='SSFR$_\\uparrow$'), \
                          mpatches.Patch(color=color_up2, label='RTM$_\\uparrow$'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=18)

        # ax2.set_ylim(ylim2)
        # ax2.yaxis.set_major_locator(FixedLocator(np.arange(ylim2[0], ylim2[1]+0.1, dy2)))
        # ax2.set_ylabel('Cloud Optical Thickness', rotation=270, labelpad=24)

        # ax2_.axis('off')
        # ax2_.set_ylim(ylim2)
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s.png' % (_metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=True)
        #╰──────────────────────────────────────────────────────────────╯#
        # plt.show()
        # sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    return



def get_flux_broadband(wvl, flux):

    Nselect = flux.shape[0]
    flux_broadband = np.zeros(Nselect, dtype=np.float32)
    for iselect in range(Nselect):
        logic_wvl = ~(np.isnan(flux[iselect, :]) | (flux[iselect, :]<=0.0))
        wvl_new = wvl.copy()[logic_wvl]
        flux_broadband[iselect] = np.trapz(flux[iselect, logic_wvl], x=wvl_new)

    return flux_broadband

def get_flux_spectral(wvl, flux, wvl_new):

    Nselect = flux.shape[0]
    flux_spectral = np.zeros((Nselect, wvl_new.size), dtype=np.float32)
    for iselect in range(Nselect):
        logic_wvl = ~(np.isnan(flux[iselect, :]) | (flux[iselect, :]<=0.0))
        flux_spectral[iselect, :] = np.interp(wvl_new, wvl[logic_wvl], flux[iselect, logic_wvl])

    return flux_spectral

def figure_agu_2024_cloud_wall_20240607_flux_cre_broadband(
        date=datetime.datetime(2024, 6, 7),
        scale_up=1.0,
        scale_dn=1.0,
        ):

    # case specification
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tmhr_ranges_select = [[15.3694, 15.7473], [15.8556, 16.2338]]
    sat_select = [4, 5]
    xlim = [-100.0, 20.0]
    ylim1 = [0.0, 0.08]
    ylim2 = [0.0, 20.0]
    dy1 = 200
    dy2 = 4

    vname_x = 'lon'
    colors1 = ['r', 'b']
    leg_tags = ['low', 'high']
    colors2 = ['magenta', 'cyan']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    wvl = data_ssfr['zen/wvl']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read in all logic data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # selected stacked legs
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0]) & (data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        rcParams['font.size'] = 24
        plt.close('all')
        fig = plt.figure(figsize=(8, 5))
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        # ax2 = fig.add_subplot(122, aspect='equal')

        # for i in [0, 1]:
        for i in [0]:
            color = colors1[i]

            if i == 0:
                alpha_up = 0.3
                alpha_dn = 0.5
                color_dn1 = colors1[i]
                color_up1 = 'k'
                color_dn2 = colors2[i]
                color_up2 = 'gray'
            elif i == 1:
                alpha_up = 0.5
                alpha_dn = 0.3
                color_up1 = colors1[i]
                color_dn1 = 'k'
                color_up2 = colors2[i]
                color_dn2 = 'gray'

            # satellite parsing
            #╭────────────────────────────────────────────────────────────────────────────╮#
            index_sat = sat_select[i]
            img_tag, sat_tag = fnames[index_sat].split('.')[0].split('_')[-2:]
            text0 = '%s-%s at %s' % (sat_tag, img_tag, er3t.util.jday_to_dtime(data_sat['sat/jday'][index_sat]).strftime('%H:%M'))
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # read rt calculations
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fname_alb = 'data/data_albedo_20240607_%s.h5' % (leg_tags[i])
            data_alb = er3t.util.load_h5(fname_alb)
            fname_rtm = 'data/data_rtm_lrt_cloud_20240607_%s.h5' % (leg_tags[i])
            data_rtm = er3t.util.load_h5(fname_rtm)
            fname_rtm0 = 'data/data_rtm_lrt_clear_20240607_%s.h5' % (leg_tags[i])
            data_rtm0 = er3t.util.load_h5(fname_rtm0)

            fname_fake = 'data/data_rtm_lrt_cloud_fake_20240607_%s.h5' % (leg_tags[i])
            data_fake = er3t.util.load_h5(fname_fake)
            fname_fake0 = 'data/data_rtm_lrt_clear_fake_20240607_%s.h5' % (leg_tags[i])
            data_fake0 = er3t.util.load_h5(fname_fake0)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # logic_select
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rtm_indices = data_alb['indices']
            logic_rtm = np.repeat(False, data_ssfr['jday'].size)
            logic_alb = (np.sum(np.isnan(data_alb['albedo_interp']), axis=-1)<80) & (data_rtm['f_dn'][:, 100]>0.0) & (data_rtm['f_up'][:, 100]>0.0)
            logic_rtm[rtm_indices[logic_alb]] = True

            logic_select = logics_select[i] &\
                          (data_ssfr['nad/flux'][:, 100]>0.0) &\
                          (data_ssfr['zen/flux'][:, 100]>0.0) &\
                          logic_rtm

            logic_select_rtm = logic_select[rtm_indices]
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # ssfr (cloud)
            #╭────────────────────────────────────────────────────────────────────────────╮#
            ssfr_f_up = get_flux_broadband(data_ssfr['nad/wvl'], data_ssfr['nad/flux'][logic_select, :])
            ssfr_f_dn = get_flux_broadband(data_ssfr['zen/wvl'], data_ssfr['zen/flux'][logic_select, :])
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm (cloud)
            #╭────────────────────────────────────────────────────────────────────────────╮#
            print(data_alb['wvl'])
            rtm_f_up = get_flux_broadband(data_alb['wvl'], data_rtm['f_up'][logic_select_rtm, :])
            rtm_f_dn = get_flux_broadband(data_alb['wvl'], data_rtm['f_dn'][logic_select_rtm, :])
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm (clear)
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rtm_f_up0 = get_flux_broadband(data_alb['wvl'], data_rtm0['f_up'][logic_select_rtm, :])
            rtm_f_dn0 = get_flux_broadband(data_alb['wvl'], data_rtm0['f_dn'][logic_select_rtm, :])
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # fake (cloud)
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fake_f_up = get_flux_broadband(data_alb['wvl'], data_fake['f_up'][logic_select_rtm, :])
            fake_f_dn = get_flux_broadband(data_alb['wvl'], data_fake['f_dn'][logic_select_rtm, :])
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # fake (clear)
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fake_f_up0 = get_flux_broadband(data_alb['wvl'], data_fake0['f_up'][logic_select_rtm, :])
            fake_f_dn0 = get_flux_broadband(data_alb['wvl'], data_fake0['f_dn'][logic_select_rtm, :])
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # SSFR CRE
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fake_f_net_clear  = fake_f_dn0 - fake_f_up0
            fake_f_net_cloud  = fake_f_dn - fake_f_up

            rtm_f_net_clear  = rtm_f_dn0 - rtm_f_up0
            ssfr_f_net_cloud = ssfr_f_dn - ssfr_f_up
            rtm_f_net_cloud  = rtm_f_dn - rtm_f_up

            ssfr_cre = ssfr_f_net_cloud - rtm_f_net_clear
            rtm_cre  = rtm_f_net_cloud - rtm_f_net_clear
            fake_cre = fake_f_net_cloud - fake_f_net_clear


            bins = np.arange(xlim[0]-25, xlim[1]+26, 2)

            ax1.hist(ssfr_cre.ravel(), bins=bins, density=True, histtype='stepfilled', lw=0.0, color=colors1[i], alpha=0.3)
            ax1.hist(rtm_cre.ravel(), bins=bins, density=True, histtype='stepfilled', lw=0.0, color=colors2[i], alpha=0.3)
            ax1.hist(ssfr_cre.ravel(), bins=bins, density=True, histtype='step', lw=0.5, color=colors1[i], alpha=0.8)
            ax1.hist(rtm_cre.ravel(), bins=bins, density=True, histtype='step', lw=0.5, color=colors2[i], alpha=0.8)

            ax1.hist(fake_cre.ravel(), bins=bins, density=True, histtype='step', lw=1.5, ls='-', color=colors2[i], alpha=0.8)

            ax1.axvline(ssfr_cre.mean(), lw=1.5, color=colors1[i], ls='--')
            ax1.axvline(rtm_cre.mean(), lw=1.5, color=colors2[i], ls='--')
            ax1.axvline(fake_cre.mean(), lw=1.5, color=colors2[i], ls=':')


            # ax2.scatter(ssfr_cre, rtm_cre, edgecolor=colors1[i], alpha=0.3, facecolor='none')
            # ax2.scatter(ssfr_cre, fake_cre, edgecolor=colors2[i], alpha=0.3, facecolor='none')

            # ax2.scatter(ssfr_cre, rtm_cre, facecolor=colors1[i], alpha=1.0, edgecolor='none', s=2)
            # ax2.scatter(ssfr_cre, fake_cre, facecolor=colors2[i], alpha=1.0, edgecolor='none', s=2)
            # ax2.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], lw=1.0, ls='--', color='k')
            # ax2.plot(rtm_f_dn)
            #╰────────────────────────────────────────────────────────────────────────────╯#

        ax1.axvline(0.0, lw=1.5, ls='-', color='k')

        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        # ax1.set_xlabel('Cloud Radiative Effects [$\\mathrm{W m^{-2}}$]')
        ax1.set_xlabel('CRE [$\\mathrm{W m^{-2}}$]')
        ax1.set_ylabel('Probability Density', labelpad=4)
        ax1.set_xlim(xlim)
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(xlim[0], xlim[1]+1, 20.0)))
        ax1.xaxis.set_minor_locator(FixedLocator(np.arange(xlim[0], xlim[1]+1, 10.0)))
        ax1.set_ylim(ylim1)
        ax1.yaxis.set_major_locator(FixedLocator(np.array([ylim1[0]-0.1, ylim1[1]+0.1])))
        # ax1.set_title('High Leg (%s)' % text0, color=colors2[i], y=1.01)

        # ax2.set_zorder(0)
        # ax2_.set_zorder(0)
        # ax2.set_xlim([xlim[0]-100, xlim[1]+100])
        # ax2.set_ylim([xlim[0]-100, xlim[1]+100])

        patches_legend = [
                          # mpatches.Patch(color=colors1[1], label='High Leg SSFR'), \
                          # mpatches.Patch(color=colors2[1], label='High Leg RTM (VIIRS at 16:24)'), \
                          mpatches.Patch(color=colors1[0], label='Low Leg SSFR'), \
                          mpatches.Patch(color=colors2[0], label='Low Leg RTM (MODIS at 15:20)'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        # ax2.set_ylim(ylim2)
        # ax2.yaxis.set_major_locator(FixedLocator(np.arange(ylim2[0], ylim2[1]+0.1, dy2)))
        # ax2.set_ylabel('Cloud Optical Thickness', rotation=270, labelpad=24)

        # ax2_.axis('off')
        # ax2_.set_ylim(ylim2)
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s.png' % (_metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=True)
        #╰──────────────────────────────────────────────────────────────╯#
        # plt.show()
        # sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    return

def figure_agu_2024_cloud_wall_20240607_flux_cre_spectral(
        date=datetime.datetime(2024, 6, 7),
        scale_up=1.0,
        scale_dn=1.0,
        ):

    # case specification
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tmhr_ranges_select = [[15.3694, 15.7473], [15.8556, 16.2338]]
    sat_select = [4, 5]
    xlim = [200.0, 2200.0]
    ylim1 = [-0.1, 0.05]
    ylim2 = [0.0, 20.0]
    dy1 = 200
    dy2 = 4

    vname_x = 'lon'
    colors1 = ['r', 'b']
    leg_tags = ['low', 'high']
    colors2 = ['magenta', 'cyan']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    wvl = data_ssfr['zen/wvl']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read in all logic data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # selected stacked legs
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0]) & (data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        rcParams['font.size'] = 24
        plt.close('all')
        fig = plt.figure(figsize=(8, 5))
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        # ax2 = fig.add_subplot(122, aspect='equal')

        # for i in [0, 1]:
        for i in [0]:
            color = colors1[i]

            if i == 0:
                alpha_up = 0.3
                alpha_dn = 0.5
                color_dn1 = colors1[i]
                color_up1 = 'k'
                color_dn2 = colors2[i]
                color_up2 = 'gray'
            elif i == 1:
                alpha_up = 0.5
                alpha_dn = 0.3
                color_up1 = colors1[i]
                color_dn1 = 'k'
                color_up2 = colors2[i]
                color_dn2 = 'gray'

            # satellite parsing
            #╭────────────────────────────────────────────────────────────────────────────╮#
            index_sat = sat_select[i]
            img_tag, sat_tag = fnames[index_sat].split('.')[0].split('_')[-2:]
            text0 = '%s-%s at %s' % (sat_tag, img_tag, er3t.util.jday_to_dtime(data_sat['sat/jday'][index_sat]).strftime('%H:%M'))
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # read rt calculations
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fname_alb = 'data/data_albedo_20240607_%s.h5' % (leg_tags[i])
            data_alb = er3t.util.load_h5(fname_alb)
            fname_rtm = 'data/data_rtm_lrt_cloud_20240607_%s.h5' % (leg_tags[i])
            data_rtm = er3t.util.load_h5(fname_rtm)
            fname_rtm0 = 'data/data_rtm_lrt_clear_20240607_%s.h5' % (leg_tags[i])
            data_rtm0 = er3t.util.load_h5(fname_rtm0)

            fname_fake = 'data/data_rtm_lrt_cloud_fake_20240607_%s.h5' % (leg_tags[i])
            data_fake = er3t.util.load_h5(fname_fake)
            fname_fake0 = 'data/data_rtm_lrt_clear_fake_20240607_%s.h5' % (leg_tags[i])
            data_fake0 = er3t.util.load_h5(fname_fake0)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # logic_select
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rtm_indices = data_alb['indices']
            logic_rtm = np.repeat(False, data_ssfr['jday'].size)
            logic_alb = (np.sum(np.isnan(data_alb['albedo_interp']), axis=-1)<80) & (data_rtm['f_dn'][:, 100]>0.0) & (data_rtm['f_up'][:, 100]>0.0)
            logic_rtm[rtm_indices[logic_alb]] = True

            logic_select = logics_select[i] &\
                          (data_ssfr['nad/flux'][:, 100]>0.0) &\
                          (data_ssfr['zen/flux'][:, 100]>0.0) &\
                          logic_rtm

            logic_select_rtm = logic_select[rtm_indices]
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # ssfr (cloud)
            #╭────────────────────────────────────────────────────────────────────────────╮#
            ssfr_f_up = get_flux_spectral(data_ssfr['nad/wvl'], data_ssfr['nad/flux'][logic_select, :], data_alb['wvl'])
            ssfr_f_dn = get_flux_spectral(data_ssfr['zen/wvl'], data_ssfr['zen/flux'][logic_select, :], data_alb['wvl'])
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm (cloud)
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rtm_f_up = get_flux_spectral(data_alb['wvl'], data_rtm['f_up'][logic_select_rtm, :], data_alb['wvl'])
            rtm_f_dn = get_flux_spectral(data_alb['wvl'], data_rtm['f_dn'][logic_select_rtm, :], data_alb['wvl'])
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # rtm (clear)
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rtm_f_up0 = get_flux_spectral(data_alb['wvl'], data_rtm0['f_up'][logic_select_rtm, :], data_alb['wvl'])
            rtm_f_dn0 = get_flux_spectral(data_alb['wvl'], data_rtm0['f_dn'][logic_select_rtm, :], data_alb['wvl'])
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # fake (cloud)
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fake_f_up = get_flux_spectral(data_alb['wvl'], data_fake['f_up'][logic_select_rtm, :], data_alb['wvl'])
            fake_f_dn = get_flux_spectral(data_alb['wvl'], data_fake['f_dn'][logic_select_rtm, :], data_alb['wvl'])
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # fake (clear)
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fake_f_up0 = get_flux_spectral(data_alb['wvl'], data_fake0['f_up'][logic_select_rtm, :], data_alb['wvl'])
            fake_f_dn0 = get_flux_spectral(data_alb['wvl'], data_fake0['f_dn'][logic_select_rtm, :], data_alb['wvl'])
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # SSFR CRE
            #╭────────────────────────────────────────────────────────────────────────────╮#
            fake_f_net_clear  = fake_f_dn0 - fake_f_up0
            fake_f_net_cloud  = fake_f_dn - fake_f_up

            rtm_f_net_clear  = rtm_f_dn0 - rtm_f_up0
            ssfr_f_net_cloud = ssfr_f_dn - ssfr_f_up
            rtm_f_net_cloud  = rtm_f_dn - rtm_f_up

            ssfr_cre = ssfr_f_net_cloud - rtm_f_net_clear
            rtm_cre  = rtm_f_net_cloud - rtm_f_net_clear
            fake_cre = fake_f_net_cloud - fake_f_net_clear

            wvl = data_alb['wvl']
            ssfr_cre_mean = np.nanmean(ssfr_cre, axis=0)
            rtm_cre_mean = np.nanmean(rtm_cre, axis=0)
            fake_cre_mean = np.nanmean(fake_cre, axis=0)
            ssfr_cre_std = np.nanstd(ssfr_cre, axis=0)
            rtm_cre_std = np.nanstd(rtm_cre, axis=0)
            fake_cre_std = np.nanstd(fake_cre, axis=0)


            ax1.fill_between(wvl, ssfr_cre_mean-ssfr_cre_std, ssfr_cre_mean+ssfr_cre_std, lw=0.1, facecolor='none', edgecolor=colors1[i], alpha=0.8, ls='-', zorder=1)
            ax1.fill_between(wvl, ssfr_cre_mean-ssfr_cre_std, ssfr_cre_mean+ssfr_cre_std, lw=0.0, facecolor=colors1[i], edgecolor='none', alpha=0.3, ls='-', zorder=0)
            ax1.fill_between(wvl, rtm_cre_mean-rtm_cre_std, rtm_cre_mean+rtm_cre_std, lw=0.1, facecolor='none', edgecolor=colors2[i], alpha=0.8, ls='-', zorder=1)
            ax1.fill_between(wvl, rtm_cre_mean-rtm_cre_std, rtm_cre_mean+rtm_cre_std, lw=0.0, facecolor=colors2[i], edgecolor='none', alpha=0.3, ls='-', zorder=0)

            ax1.plot(wvl, ssfr_cre_mean, lw=1.5, color=colors1[i], alpha=0.8, zorder=4)
            ax1.plot(wvl, rtm_cre_mean, lw=1.5, color=colors2[i], alpha=0.8, zorder=3)
            ax1.plot(wvl, fake_cre_mean, lw=1.5, color=colors2[i], alpha=0.8, ls='--', zorder=2)

            # ax1.hist(ssfr_cre.ravel(), bins=bins, density=True, histtype='stepfilled', lw=0.0, color=colors1[i], alpha=0.3)
            # ax1.hist(rtm_cre.ravel(), bins=bins, density=True, histtype='stepfilled', lw=0.0, color=colors2[i], alpha=0.3)
            # ax1.hist(ssfr_cre.ravel(), bins=bins, density=True, histtype='step', lw=0.5, color=colors1[i], alpha=0.8)
            # ax1.hist(rtm_cre.ravel(), bins=bins, density=True, histtype='step', lw=0.5, color=colors2[i], alpha=0.8)

            # ax1.hist(fake_cre.ravel(), bins=bins, density=True, histtype='step', lw=1.5, ls='-', color=colors2[i], alpha=0.8)

            # ax1.axvline(ssfr_cre.mean(), lw=1.5, color=colors1[i], ls='--')
            # ax1.axvline(rtm_cre.mean(), lw=1.5, color=colors2[i], ls='--')
            # ax1.axvline(fake_cre.mean(), lw=1.5, color=colors2[i], ls=':')


            # ax2.scatter(ssfr_cre, rtm_cre, edgecolor=colors1[i], alpha=0.3, facecolor='none')
            # ax2.scatter(ssfr_cre, fake_cre, edgecolor=colors2[i], alpha=0.3, facecolor='none')

            # ax2.scatter(ssfr_cre, rtm_cre, facecolor=colors1[i], alpha=1.0, edgecolor='none', s=2)
            # ax2.scatter(ssfr_cre, fake_cre, facecolor=colors2[i], alpha=1.0, edgecolor='none', s=2)
            # ax2.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], lw=1.0, ls='--', color='k')
            # ax2.plot(rtm_f_dn)
            #╰────────────────────────────────────────────────────────────────────────────╯#

        ax1.axhline(0.0, lw=1.0, ls='-', color='k')
        # absorption bands
        #╭────────────────────────────────────────────────────────────────────────────╮#
        absorb_bands = [[649.0, 662.0], [676.0, 697.0], [707.0, 729.0], [743.0, 779.0], [796.0, 830.0], [911.0, 976.0], [1108.0, 1193.0], [1262.0, 1298.0], [1324.0, 1470.0], [1800.0, 1960.0]]
        for absorb_band in absorb_bands:
            wvl_L, wvl_R = absorb_band
            ax1.axvspan(wvl_L, wvl_R, color='k', lw=0.0, alpha=0.1, zorder=0)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('CRE [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_xlim(xlim)
        ax1.xaxis.set_minor_locator(FixedLocator(np.arange(xlim[0], xlim[1]+1.0, 100.0)))
        ax1.set_ylim(ylim1)
        # ax1.yaxis.set_major_locator(FixedLocator(np.array([ylim1[0]-0.1, ylim1[1]+0.1])))
        # ax1.set_title('High Leg (%s)' % text0, color=colors2[i], y=1.01)

        # ax2.set_zorder(0)
        # ax2_.set_zorder(0)
        # ax2.set_xlim([xlim[0]-100, xlim[1]+100])
        # ax2.set_ylim([xlim[0]-100, xlim[1]+100])

        patches_legend = [
                          # mpatches.Patch(color=colors1[1], label='High Leg SSFR'), \
                          # mpatches.Patch(color=colors2[1], label='High Leg RTM (VIIRS at 16:24)'), \
                          mpatches.Patch(color=colors1[0], label='Low Leg SSFR'), \
                          mpatches.Patch(color=colors2[0], label='Low Leg RTM (MODIS at 15:20)'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        # ax2.set_ylim(ylim2)
        # ax2.yaxis.set_major_locator(FixedLocator(np.arange(ylim2[0], ylim2[1]+0.1, dy2)))
        # ax2.set_ylabel('Cloud Optical Thickness', rotation=270, labelpad=24)

        # ax2_.axis('off')
        # ax2_.set_ylim(ylim2)
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s.png' % (_metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=True)
        #╰──────────────────────────────────────────────────────────────╯#
        # plt.show()
        # sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    return


if __name__ == '__main__':

    # plot albedo from two deployments
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # figure_agu_2024_sfc_alb()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot flight track/satellite imagery overlay for the selected albedos
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # fname = sorted(glob.glob('/argus/field/arcsix/sat-img-hc/TrueColor*2024-06-06_16:42*.jpg'))[0]
    # figure_agu_2024_sfc_alb_flt_trk_xy(fname)
    # fname = sorted(glob.glob('/argus/field/arcsix/sat-img-hc/TrueColor*2024-07-30_14:48*.jpg'))[0]
    # figure_agu_2024_sfc_alb_flt_trk_xy(fname)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot cloud wall for 2024-06-07
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # figure_agu_2024_cloud_wall_20240607_flt_trk_z()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot flight track/satellite imagery overlay for the cloud wall
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # fname = sorted(glob.glob('/argus/field/arcsix/sat-img-hc/TrueColor*2024-06-07_15:20*.jpg'))[0]
    # figure_agu_2024_cloud_wall_20240607_flt_trk_xy(fname)
    # fname = sorted(glob.glob('/argus/field/arcsix/sat-img-hc/TrueColor*2024-06-07_16:24*.jpg'))[0]
    # figure_agu_2024_cloud_wall_20240607_flt_trk_xy(fname)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot time series of radiative fluxes (ssfr and calculations)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # figure_agu_2024_cloud_wall_20240607_flux_low_leg_time_series(scale_up=1.0, scale_dn=1.0)
    # figure_agu_2024_cloud_wall_20240607_flux_high_leg_time_series(scale_up=1.0, scale_dn=1.0)
    # figure_agu_2024_cloud_wall_20240607_flux_low_leg_time_series(scale_up=1.15, scale_dn=1.08)
    # figure_agu_2024_cloud_wall_20240607_flux_high_leg_time_series(scale_up=1.15, scale_dn=1.08)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot spectral radiative fluxes (ssfr and calculations)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # figure_agu_2024_cloud_wall_20240607_flux_low_leg_spectral(scale_up=1.0, scale_dn=1.0)
    # figure_agu_2024_cloud_wall_20240607_flux_high_leg_spectral(scale_up=1.0, scale_dn=1.0)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # figure_agu_2024_cloud_wall_20240607_flux_cre_broadband()
    figure_agu_2024_cloud_wall_20240607_flux_cre_spectral()
    pass
