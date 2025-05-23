"""
Code for processing data collected by "radiation instruments" during NASA ARCSIX 2024.


Acknowledgements:
    Instrument engineering:
        Jeffery Drouet, Sebastian Schmidt
    Pre-mission calibration and analysis:
        Hong Chen, Yu-Wen Chen, Ken Hirata, Sebastian Schmidt, Bruce Kindel
    In-field calibration and on-flight operation:
        Arabella Chamberlain, Vikas Nataraja, Ken Hirata, Sebastian Schmidt
"""

import os
import sys
import glob
import datetime
import warnings
from tqdm import tqdm
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav
from scipy.optimize import curve_fit
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

# import ssfr
import er3t

# parameters
#╭────────────────────────────────────────────────────────────────────────────╮#
_mission_     = 'arcsix'
_platform_    = 'p3b'

_hsk_         = 'hsk'
_alp_         = 'alp'
_spns_        = 'spns-a'
_ssfr1_       = 'ssfr-a'
_ssfr2_       = 'ssfr-b'
_cam_         = 'nac'

_fdir_hsk_   = '/argus/field/arcsix/2024-Spring/p3/aux/hsk'
_fdir_cal_   = '/argus/field/%s/cal' % _mission_

_fdir_data_  = '/argus/field/%s/processed' % _mission_
_fdir_out_   = '%s/processed' % _fdir_data_

_verbose_   = True
#╰────────────────────────────────────────────────────────────────────────────╯#

def cdata_sfc_alb_low(
        date,
        tmhr_range,
        exclude_tmhr_ranges=[],
        absorb_bands=[
            [649.0, 662.0],
            [676.0, 697.0],
            [707.0, 729.0],
            [743.0, 779.0],
            [796.0, 830.0],
            [911.0, 976.0],
            [1108.0, 1193.0],
            [1262.0, 1298.0],
            [1324.0, 1470.0],
            [1800.0, 1960.0]],
        tmhr_range_low=None,
        ):

    date_s = date.strftime('%Y%m%d')

    # read hsk file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read logic file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic= er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames_sat = f['sat/jday'].attrs['description'].split('\n')
    f.close()

    sat_select = [3, 4, 5, 6]
    #╰────────────────────────────────────────────────────────────────────────────╯#
    # for i, fname_sat in enumerate(fnames_sat):
    #     print(i, fname_sat)
    # sys.exit()

    # generate identify clear-sky
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_tmhr = (data_hsk['tmhr']>=tmhr_range[0]) & (data_hsk['tmhr']<=tmhr_range[1])

    for tmhr_range_ in exclude_tmhr_ranges:
        logic0 = (data_hsk['tmhr']>=tmhr_range_[0]) & (data_hsk['tmhr']<=tmhr_range_[1])
        logic_tmhr[logic0] = False

    # logic_select = logic_tmhr & data_logic['logic_clear'] & data_logic['logic_steady'] & data_logic['logic_ocean']
    logic_select = logic_tmhr & data_logic['logic_steady'] & data_logic['logic_ocean']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # absorption bands
    #╭────────────────────────────────────────────────────────────────────────────╮#
    wvl = data_ssfr['zen/wvl'].copy()
    logic_wvl = np.repeat(True, wvl.size)
    for absorb_band in absorb_bands:
        wvl_l, wvl_r = absorb_band
        logic0 = (wvl>=wvl_l) & (wvl<=wvl_r)
        logic_wvl[logic0] = False
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if tmhr_range_low is None:
        tmhr_range_low = tmhr_range
    logic_tmhr_low = (data_hsk['tmhr']>=tmhr_range_low[0]) & (data_hsk['tmhr']<=tmhr_range_low[1])
    logic_select_low = logic_tmhr_low & data_logic['logic_steady'] & data_logic['logic_ocean']
    indices_low = np.where(logic_select_low)[0]

    indices = np.where(logic_select)[0]
    albedo_ori = np.zeros((indices.size, wvl.size), dtype=np.float32)
    albedo_new = np.zeros((indices.size, wvl.size), dtype=np.float32)

    if indices.size > 0:
        for i, index in enumerate(indices):
            lon0 = data_hsk['lon'][index]
            lat0 = data_hsk['lat'][index]
            index_ = indices_low[np.argmin(np.abs((lon0-data_hsk['lon'][logic_select_low])**2+(lat0-data_hsk['lat'][logic_select_low])**2))]

            f_dn  = data_ssfr['zen/flux'][index_, :]

            f_up_ = data_ssfr['nad/flux'][index_, :]
            f_up  = np.interp(data_ssfr['zen/wvl'], data_ssfr['nad/wvl'], f_up_)

            alb0 = f_up/f_dn

            albedo_ori[i, :] = alb0
            albedo_new[i, :] = np.interp(wvl, wvl[logic_wvl], alb0[logic_wvl])

    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = 'data_albedo_%s_low.h5' % date_s
    h5f = h5py.File(fname, 'w')

    h5d_jday = h5f.create_dataset('jday', data=data_hsk['jday'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_tmhr = h5f.create_dataset('tmhr', data=data_hsk['tmhr'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_lon = h5f.create_dataset('lon', data=data_hsk['lon'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_lat = h5f.create_dataset('lat', data=data_hsk['lat'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_alt = h5f.create_dataset('alt', data=data_hsk['alt'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_sza = h5f.create_dataset('sza', data=data_hsk['sza'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_indices = h5f.create_dataset('indices', data=indices, compression='gzip', compression_opts=9, chunks=True)

    for index in sat_select:
        h5g = h5f.create_group(fnames_sat[index])
        h5g.create_dataset('cot', data=data_sat['sat/cot'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('cer', data=data_sat['sat/cer'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('cth', data=data_sat['sat/cth'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('ctp', data=data_sat['sat/ctp'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)

    h5d_wvl = h5f.create_dataset('wvl', data=wvl, compression='gzip', compression_opts=9, chunks=True)
    h5d_logic_wvl = h5f.create_dataset('logic_wvl', data=logic_wvl, compression='gzip', compression_opts=9, chunks=True)

    h5d_albedo_ori = h5f.create_dataset('albedo_ori', data=albedo_ori, compression='gzip', compression_opts=9, chunks=True)
    h5d_albedo_interp = h5f.create_dataset('albedo_interp', data=albedo_new, compression='gzip', compression_opts=9, chunks=True)

    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def cdata_sfc_alb_high(
        date,
        tmhr_range,
        exclude_tmhr_ranges=[],
        absorb_bands=[
            [649.0, 662.0],
            [676.0, 697.0],
            [707.0, 729.0],
            [743.0, 779.0],
            [796.0, 830.0],
            [911.0, 976.0],
            [1108.0, 1193.0],
            [1262.0, 1298.0],
            [1324.0, 1470.0],
            [1800.0, 1960.0]],
        tmhr_range_low=None,
        ):

    date_s = date.strftime('%Y%m%d')

    # read hsk file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read logic file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic= er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames_sat = f['sat/jday'].attrs['description'].split('\n')
    f.close()

    sat_select = [3, 4, 5, 6]
    #╰────────────────────────────────────────────────────────────────────────────╯#
    # for i, fname_sat in enumerate(fnames_sat):
    #     print(i, fname_sat)
    # sys.exit()

    # generate identify clear-sky
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_tmhr = (data_hsk['tmhr']>=tmhr_range[0]) & (data_hsk['tmhr']<=tmhr_range[1])

    for tmhr_range_ in exclude_tmhr_ranges:
        logic0 = (data_hsk['tmhr']>=tmhr_range_[0]) & (data_hsk['tmhr']<=tmhr_range_[1])
        logic_tmhr[logic0] = False

    # logic_select = logic_tmhr & data_logic['logic_clear'] & data_logic['logic_steady'] & data_logic['logic_ocean']
    logic_select = logic_tmhr & data_logic['logic_steady'] & data_logic['logic_ocean']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # absorption bands
    #╭────────────────────────────────────────────────────────────────────────────╮#
    wvl = data_ssfr['zen/wvl'].copy()
    logic_wvl = np.repeat(True, wvl.size)
    for absorb_band in absorb_bands:
        wvl_l, wvl_r = absorb_band
        logic0 = (wvl>=wvl_l) & (wvl<=wvl_r)
        logic_wvl[logic0] = False
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if tmhr_range_low is None:
        tmhr_range_low = tmhr_range
    logic_tmhr_low = (data_hsk['tmhr']>=tmhr_range_low[0]) & (data_hsk['tmhr']<=tmhr_range_low[1])
    logic_select_low = logic_tmhr_low & data_logic['logic_steady'] & data_logic['logic_ocean']
    indices_low = np.where(logic_select_low)[0]

    indices = np.where(logic_select)[0]
    albedo_ori = np.zeros((indices.size, wvl.size), dtype=np.float32)
    albedo_new = np.zeros((indices.size, wvl.size), dtype=np.float32)

    if indices.size > 0:
        for i, index in enumerate(indices):
            lon0 = data_hsk['lon'][index]
            lat0 = data_hsk['lat'][index]
            index_ = indices_low[np.argmin(np.abs((lon0-data_hsk['lon'][logic_select_low])**2+(lat0-data_hsk['lat'][logic_select_low])**2))]

            f_dn  = data_ssfr['zen/flux'][index_, :]

            f_up_ = data_ssfr['nad/flux'][index_, :]
            f_up  = np.interp(data_ssfr['zen/wvl'], data_ssfr['nad/wvl'], f_up_)

            alb0 = f_up/f_dn

            albedo_ori[i, :] = alb0
            albedo_new[i, :] = np.interp(wvl, wvl[logic_wvl], alb0[logic_wvl])

    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = 'data_albedo_%s_high.h5' % date_s
    h5f = h5py.File(fname, 'w')

    h5d_jday = h5f.create_dataset('jday', data=data_hsk['jday'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_tmhr = h5f.create_dataset('tmhr', data=data_hsk['tmhr'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_lon = h5f.create_dataset('lon', data=data_hsk['lon'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_lat = h5f.create_dataset('lat', data=data_hsk['lat'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_alt = h5f.create_dataset('alt', data=data_hsk['alt'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_sza = h5f.create_dataset('sza', data=data_hsk['sza'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_indices = h5f.create_dataset('indices', data=indices, compression='gzip', compression_opts=9, chunks=True)

    for index in sat_select:
        h5g = h5f.create_group(fnames_sat[index])
        h5g.create_dataset('cot', data=data_sat['sat/cot'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('cer', data=data_sat['sat/cer'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('cth', data=data_sat['sat/cth'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('ctp', data=data_sat['sat/ctp'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)

    h5d_wvl = h5f.create_dataset('wvl', data=wvl, compression='gzip', compression_opts=9, chunks=True)
    h5d_logic_wvl = h5f.create_dataset('logic_wvl', data=logic_wvl, compression='gzip', compression_opts=9, chunks=True)

    h5d_albedo_ori = h5f.create_dataset('albedo_ori', data=albedo_ori, compression='gzip', compression_opts=9, chunks=True)
    h5d_albedo_interp = h5f.create_dataset('albedo_interp', data=albedo_new, compression='gzip', compression_opts=9, chunks=True)

    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def test():
    fname1 = 'data_albedo_20240607.h5'
    fname2 = 'data_albedo_20240607_high.h5'

    data1 = er3t.util.load_h5(fname1)
    data2 = er3t.util.load_h5(fname2)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        # ax1.scatter(data1['lon'], data1['albedo_ori'][:, 100], s=6, c='k', lw=0.0)
        # ax1.scatter(data2['lon'], data2['albedo_ori'][:, 100], s=6, c='r', lw=0.0)
        ax1.scatter(data1['lon'], data1['albedo_ori'][:, 300], s=6, c='k', lw=0.0)
        ax1.scatter(data2['lon'], data2['albedo_ori'][:, 300], s=6, c='r', lw=0.0)
        # ax1.set_xlim((0, 1))
        # ax1.set_ylim((0, 1))
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_title('Plot1')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s_%s.png' % (_metadata_['Date'], _metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#
    pass



def cdata_sfc_alb_low_fake(
        date,
        tmhr_range,
        exclude_tmhr_ranges=[],
        absorb_bands=[
            [649.0, 662.0],
            [676.0, 697.0],
            [707.0, 729.0],
            [743.0, 779.0],
            [796.0, 830.0],
            [911.0, 976.0],
            [1108.0, 1193.0],
            [1262.0, 1298.0],
            [1324.0, 1470.0],
            [1800.0, 1960.0]],
        tmhr_range_low=None,
        ):

    date_s = date.strftime('%Y%m%d')

    # read hsk file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read logic file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic= er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames_sat = f['sat/jday'].attrs['description'].split('\n')
    f.close()

    sat_select = [3, 4, 5, 6]
    #╰────────────────────────────────────────────────────────────────────────────╯#
    # for i, fname_sat in enumerate(fnames_sat):
    #     print(i, fname_sat)
    # sys.exit()

    # generate identify clear-sky
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_tmhr = (data_hsk['tmhr']>=tmhr_range[0]) & (data_hsk['tmhr']<=tmhr_range[1])

    for tmhr_range_ in exclude_tmhr_ranges:
        logic0 = (data_hsk['tmhr']>=tmhr_range_[0]) & (data_hsk['tmhr']<=tmhr_range_[1])
        logic_tmhr[logic0] = False

    # logic_select = logic_tmhr & data_logic['logic_clear'] & data_logic['logic_steady'] & data_logic['logic_ocean']
    logic_select = logic_tmhr & data_logic['logic_steady'] & data_logic['logic_ocean']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # absorption bands
    #╭────────────────────────────────────────────────────────────────────────────╮#
    wvl = data_ssfr['zen/wvl'].copy()
    logic_wvl = np.repeat(True, wvl.size)
    for absorb_band in absorb_bands:
        wvl_l, wvl_r = absorb_band
        logic0 = (wvl>=wvl_l) & (wvl<=wvl_r)
        logic_wvl[logic0] = False
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if tmhr_range_low is None:
        tmhr_range_low = tmhr_range
    logic_tmhr_low = (data_hsk['tmhr']>=tmhr_range_low[0]) & (data_hsk['tmhr']<=tmhr_range_low[1])
    logic_select_low = logic_tmhr_low & data_logic['logic_steady'] & data_logic['logic_ocean']
    indices_low = np.where(logic_select_low)[0]

    indices = np.where(logic_select)[0]
    albedo_ori = np.zeros((indices.size, wvl.size), dtype=np.float32)
    albedo_new = np.zeros((indices.size, wvl.size), dtype=np.float32)


    # read fake albedo
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = 'data_albedo_fake_20240730.h5'
    h5f = h5py.File(fname, 'r')
    alb0 = h5f['alb_mean'][...]
    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if indices.size > 0:
        for i, index in enumerate(indices):
            lon0 = data_hsk['lon'][index]
            lat0 = data_hsk['lat'][index]

            # index_ = indices_low[np.argmin(np.abs((lon0-data_hsk['lon'][logic_select_low])**2+(lat0-data_hsk['lat'][logic_select_low])**2))]
            # f_dn  = data_ssfr['zen/flux'][index_, :]
            # f_up_ = data_ssfr['nad/flux'][index_, :]
            # f_up  = np.interp(data_ssfr['zen/wvl'], data_ssfr['nad/wvl'], f_up_)
            # alb0 = f_up/f_dn

            albedo_ori[i, :] = alb0
            albedo_new[i, :] = np.interp(wvl, wvl[logic_wvl], alb0[logic_wvl])

    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = 'data_albedo_fake_%s_low.h5' % date_s
    h5f = h5py.File(fname, 'w')

    h5d_jday = h5f.create_dataset('jday', data=data_hsk['jday'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_tmhr = h5f.create_dataset('tmhr', data=data_hsk['tmhr'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_lon = h5f.create_dataset('lon', data=data_hsk['lon'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_lat = h5f.create_dataset('lat', data=data_hsk['lat'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_alt = h5f.create_dataset('alt', data=data_hsk['alt'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_sza = h5f.create_dataset('sza', data=data_hsk['sza'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_indices = h5f.create_dataset('indices', data=indices, compression='gzip', compression_opts=9, chunks=True)

    for index in sat_select:
        h5g = h5f.create_group(fnames_sat[index])
        h5g.create_dataset('cot', data=data_sat['sat/cot'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('cer', data=data_sat['sat/cer'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('cth', data=data_sat['sat/cth'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('ctp', data=data_sat['sat/ctp'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)

    h5d_wvl = h5f.create_dataset('wvl', data=wvl, compression='gzip', compression_opts=9, chunks=True)
    h5d_logic_wvl = h5f.create_dataset('logic_wvl', data=logic_wvl, compression='gzip', compression_opts=9, chunks=True)

    h5d_albedo_ori = h5f.create_dataset('albedo_ori', data=albedo_ori, compression='gzip', compression_opts=9, chunks=True)
    h5d_albedo_interp = h5f.create_dataset('albedo_interp', data=albedo_new, compression='gzip', compression_opts=9, chunks=True)

    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def cdata_sfc_alb_high_fake(
        date,
        tmhr_range,
        exclude_tmhr_ranges=[],
        absorb_bands=[
            [649.0, 662.0],
            [676.0, 697.0],
            [707.0, 729.0],
            [743.0, 779.0],
            [796.0, 830.0],
            [911.0, 976.0],
            [1108.0, 1193.0],
            [1262.0, 1298.0],
            [1324.0, 1470.0],
            [1800.0, 1960.0]],
        tmhr_range_low=None,
        ):

    date_s = date.strftime('%Y%m%d')

    # read hsk file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read logic file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_logic = '%s/%s-LOGIC_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_logic= er3t.util.load_h5(fname_logic)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read collocated satellite data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames_sat = f['sat/jday'].attrs['description'].split('\n')
    f.close()

    sat_select = [3, 4, 5, 6]
    #╰────────────────────────────────────────────────────────────────────────────╯#
    # for i, fname_sat in enumerate(fnames_sat):
    #     print(i, fname_sat)
    # sys.exit()

    # generate identify clear-sky
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic_tmhr = (data_hsk['tmhr']>=tmhr_range[0]) & (data_hsk['tmhr']<=tmhr_range[1])

    for tmhr_range_ in exclude_tmhr_ranges:
        logic0 = (data_hsk['tmhr']>=tmhr_range_[0]) & (data_hsk['tmhr']<=tmhr_range_[1])
        logic_tmhr[logic0] = False

    # logic_select = logic_tmhr & data_logic['logic_clear'] & data_logic['logic_steady'] & data_logic['logic_ocean']
    logic_select = logic_tmhr & data_logic['logic_steady'] & data_logic['logic_ocean']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # absorption bands
    #╭────────────────────────────────────────────────────────────────────────────╮#
    wvl = data_ssfr['zen/wvl'].copy()
    logic_wvl = np.repeat(True, wvl.size)
    for absorb_band in absorb_bands:
        wvl_l, wvl_r = absorb_band
        logic0 = (wvl>=wvl_l) & (wvl<=wvl_r)
        logic_wvl[logic0] = False
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if tmhr_range_low is None:
        tmhr_range_low = tmhr_range
    logic_tmhr_low = (data_hsk['tmhr']>=tmhr_range_low[0]) & (data_hsk['tmhr']<=tmhr_range_low[1])
    logic_select_low = logic_tmhr_low & data_logic['logic_steady'] & data_logic['logic_ocean']
    indices_low = np.where(logic_select_low)[0]

    indices = np.where(logic_select)[0]
    albedo_ori = np.zeros((indices.size, wvl.size), dtype=np.float32)
    albedo_new = np.zeros((indices.size, wvl.size), dtype=np.float32)

    # read fake albedo
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = 'data_albedo_fake_20240730.h5'
    h5f = h5py.File(fname, 'r')
    alb0 = h5f['alb_mean'][...]
    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if indices.size > 0:
        for i, index in enumerate(indices):
            lon0 = data_hsk['lon'][index]
            lat0 = data_hsk['lat'][index]
            # index_ = indices_low[np.argmin(np.abs((lon0-data_hsk['lon'][logic_select_low])**2+(lat0-data_hsk['lat'][logic_select_low])**2))]

            # f_dn  = data_ssfr['zen/flux'][index_, :]

            # f_up_ = data_ssfr['nad/flux'][index_, :]
            # f_up  = np.interp(data_ssfr['zen/wvl'], data_ssfr['nad/wvl'], f_up_)

            # alb0 = f_up/f_dn

            albedo_ori[i, :] = alb0
            albedo_new[i, :] = np.interp(wvl, wvl[logic_wvl], alb0[logic_wvl])

    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = 'data_albedo_fake_%s_high.h5' % date_s
    h5f = h5py.File(fname, 'w')

    h5d_jday = h5f.create_dataset('jday', data=data_hsk['jday'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_tmhr = h5f.create_dataset('tmhr', data=data_hsk['tmhr'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_lon = h5f.create_dataset('lon', data=data_hsk['lon'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_lat = h5f.create_dataset('lat', data=data_hsk['lat'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_alt = h5f.create_dataset('alt', data=data_hsk['alt'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_sza = h5f.create_dataset('sza', data=data_hsk['sza'][logic_select], compression='gzip', compression_opts=9, chunks=True)
    h5d_indices = h5f.create_dataset('indices', data=indices, compression='gzip', compression_opts=9, chunks=True)

    for index in sat_select:
        h5g = h5f.create_group(fnames_sat[index])
        h5g.create_dataset('cot', data=data_sat['sat/cot'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('cer', data=data_sat['sat/cer'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('cth', data=data_sat['sat/cth'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)
        h5g.create_dataset('ctp', data=data_sat['sat/ctp'][logic_select, index], compression='gzip', compression_opts=9, chunks=True)

    h5d_wvl = h5f.create_dataset('wvl', data=wvl, compression='gzip', compression_opts=9, chunks=True)
    h5d_logic_wvl = h5f.create_dataset('logic_wvl', data=logic_wvl, compression='gzip', compression_opts=9, chunks=True)

    h5d_albedo_ori = h5f.create_dataset('albedo_ori', data=albedo_ori, compression='gzip', compression_opts=9, chunks=True)
    h5d_albedo_interp = h5f.create_dataset('albedo_interp', data=albedo_new, compression='gzip', compression_opts=9, chunks=True)

    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

if __name__ == '__main__':

    # create surface albedo data for 2024-06-11
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # date = datetime.datetime(2024, 6, 11)
    # tmhr_range = [15.3528, 15.7139]
    # cdata_sfc_alb(date, tmhr_range)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # create surface albedo data for 2024-06-07
    #╭────────────────────────────────────────────────────────────────────────────╮#
    date = datetime.datetime(2024, 6, 7)
    # tmhr_range = [15.3694, 15.7473]
    # cdata_sfc_alb_low(date, tmhr_range)
    # cdata_sfc_alb_low_fake(date, tmhr_range)

    tmhr_range = [15.8556, 16.2338]
    # cdata_sfc_alb_high(date, tmhr_range, tmhr_range_low=[15.3694, 15.7473])
    cdata_sfc_alb_high_fake(date, tmhr_range, tmhr_range_low=[15.3694, 15.7473])
    #╰────────────────────────────────────────────────────────────────────────────╯#

    pass
