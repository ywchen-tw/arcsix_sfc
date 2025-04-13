import os
import sys
import glob
import copy
import warnings
from collections import OrderedDict
import h5py
import numpy as np
import datetime
import time
import pickle
import multiprocessing as mp
from scipy.io import readsav
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg

import er3t
from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_les, cld_sat
from er3t.pre.pha import pha_mie_wc as pha_mie

from er3t.rtm.mca import mca_atm_1d, mca_atm_3d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.rtm.mca import mca_sca




def save_h5(date,
        wavelength=532.0,
        vnames=['jday', 'lon', 'lat', 'sza', \
            'tmhr', 'alt',\
            'ang_pit', 'ang_rol',\
            # 'f-up_ssfr', 'f-down_ssfr',\
            'f-down-diffuse_hsr1', 'f-down_hsr1', \
            'cot', 'cer', 'cth', \
            'f-down_mca-3d', 'f-down-diffuse_mca-3d', 'f-down-direct_mca-3d', 'f-up_mca-3d',\
            'f-down_mca-3d-alt-all', 'f-down-diffuse_mca-3d-alt-all', 'f-down-direct_mca-3d-alt-all', 'f-up_mca-3d-alt-all',\
            'f-down_mca-ipa', 'f-down-diffuse_mca-ipa', 'f-down-direct_mca-ipa', 'f-up_mca-ipa']):

    date_s = date.strftime('%Y%m%d')

    fname      = 'data/flt_sim_%09.4fnm_%s.pk' % (wavelength, date_s)
    flt_sim0   = flt_sim(fname=fname)

    fname_h5   = fname.replace('.pk', '.h5')
    f = h5py.File(fname_h5, 'w')

    for vname in vnames:

        if 'alt-all' in vname:


            for i in range(len(flt_sim0.flt_trks)):

                flt_trk = flt_sim0.flt_trks[i]

                if i == 0:
                    if vname in flt_trk.keys():
                        data0 = flt_trk[vname]
                    else:
                        data0 = np.repeat(np.nan, 21*flt_trk['jday'].size).reshape((-1, 21))
                else:
                    if vname in flt_trk.keys():
                        data0 = np.vstack((data0, flt_trk[vname]))
                    else:
                        data0 = np.vstack((data0, np.repeat(np.nan, 21*flt_trk['jday'].size).reshape((-1, 21))))

        else:
            data0 = np.array([], dtype=np.float64)
            for flt_trk in flt_sim0.flt_trks:
                if vname in flt_trk.keys():
                    data0 = np.append(data0, flt_trk[vname])
                else:
                    data0 = np.append(data0, np.repeat(np.nan, flt_trk['jday'].size))

        f[vname] = data0

    f.close()

    # fname_des = '/data/hong/share/%s' % os.path.basename(fname_h5)
    # os.system('cp %s %s' % (fname_h5, fname_des))
    # os.system('chmod 777 %s' % fname_des)

def check_continuity(data, threshold=0.1):

    data = np.append(data[0], data)

    return (np.abs(data[1:]-data[:-1]) < threshold)

def partition_flight_track(flt_trk, tmhr_interval=0.1, margin_px=25.0, margin_py=25.0):

    """
    Input:
        flt_trk: Python dictionary that contains
            ['jday']: numpy array, UTC time in hour
            ['tmhr']: numpy array, UTC time in hour
            ['lon'] : numpy array, longitude
            ['lat'] : numpy array, latitude
            ['alt'] : numpy array, altitude
            ['sza'] : numpy array, solar zenith angle
            [...]   : numpy array, other data variables, e.g., 'f_up_0600'

        tmhr_interval=: float, time interval of legs to be partitioned, default=0.1
        margin_x=     : float, margin in x (longitude) direction that to be used to
                        define the rectangular box to contain cloud field, default=1.0
        margin_y=     : float, margin in y (latitude) direction that to be used to
                        define the rectangular box to contain cloud field, default=1.0


    Output:
        flt_trk_segments: Python list that contains data for each partitioned leg in Python dictionary, e.g., legs[i] contains
            [i]['jday'] : numpy array, UTC time in hour
            [i]['tmhr'] : numpy array, UTC time in hour
            [i]['lon']  : numpy array, longitude
            [i]['lat']  : numpy array, latitude
            [i]['alt']  : numpy array, altitude
            [i]['sza']  : numpy array, solar zenith angle
            [i]['jday0']: mean value
            [i]['tmhr0']: mean value
            [i]['lon0'] : mean value
            [i]['lat0'] : mean value
            [i]['alt0'] : mean value
            [i]['sza0'] : mean value
            [i][...]    : numpy array, other data variables
    """

    jday_interval = tmhr_interval/24.0

    jday_start = jday_interval * (flt_trk['jday'][0]//jday_interval)
    jday_end   = jday_interval * (flt_trk['jday'][-1]//jday_interval + 1)

    jday_edges = np.arange(jday_start, jday_end+jday_interval, jday_interval)

    flt_trk_segments = []

    for i in range(jday_edges.size-1):

        logic      = (flt_trk['jday']>=jday_edges[i]) & (flt_trk['jday']<=jday_edges[i+1]) & (np.logical_not(np.isnan(flt_trk['sza'])))
        if logic.sum() > 0:

            flt_trk_segment = {}

            for key in flt_trk.keys():
                flt_trk_segment[key]     = flt_trk[key][logic]
                if key in ['jday', 'tmhr', 'lon', 'lat', 'alt', 'sza']:
                    flt_trk_segment[key+'0'] = np.nanmean(flt_trk_segment[key])

            margin_x = (np.nanmax(flt_trk_segment['lon'])-np.nanmin(flt_trk_segment['lon'])) * margin_px/100.0
            margin_y = (np.nanmax(flt_trk_segment['lat'])-np.nanmin(flt_trk_segment['lat'])) * margin_py/100.0
            margin_x = max(0.2, margin_x)
            margin_y = max(0.2, margin_y)

            flt_trk_segment['extent'] = np.array([np.nanmin(flt_trk_segment['lon'])-margin_x, \
                                                  np.nanmax(flt_trk_segment['lon'])+margin_x, \
                                                  np.nanmin(flt_trk_segment['lat'])-margin_y, \
                                                  np.nanmax(flt_trk_segment['lat'])+margin_y])

            flt_trk_segments.append(flt_trk_segment)

    return flt_trk_segments

def run_mcarats_one(
        index,
        fname_sat,
        extent,
        solar_zenith_angle,
        cloud_top_height=None,
        fdir='tmp-data',
        wavelength=532.0,
        date=datetime.datetime.now(),
        target='flux',
        solver='3D',
        photons=5e6,
        Ncpu=14,
        overwrite=True,
        quiet=False
        ):

    """
    Run MCARaTS with specified inputs (a general function from 04_pre_mca.py)
    """

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    levels    = np.append(np.arange(0.0, 2.0, 0.2), np.arange(2.0, 20.1, 2.0))
    fname_atm = '%s/atm_%3.3d.pk' % (fdir, index)
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_abs = '%s/abs_%3.3d.pk' % (fdir, index)
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld_sat_%3.3d.pk' % (fdir, index)

    if overwrite:
        sat0 = er3t.util.abi_l2(fnames=[fname_sat], extent=extent, vnames=['cld_height_acha'])
        lon_2d, lat_2d, cot_2d = er3t.util.grid_by_dxdy(sat0.data['lon']['data'], sat0.data['lat']['data'], sat0.data['cot']['data'], extent=extent, dx=3000.0, dy=3000.0)
        lon_2d, lat_2d, cer_2d = er3t.util.grid_by_dxdy(sat0.data['lon']['data'], sat0.data['lat']['data'], sat0.data['cer']['data'], extent=extent, dx=3000.0, dy=3000.0)
        cot_2d[cot_2d>100.0] = 100.0
        cer_2d[cer_2d==0.0] = 1.0
        sat0.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=lon_2d)
        sat0.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=lat_2d)
        sat0.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d)
        sat0.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d)

        if cloud_top_height is None:
            try:
                lon_2d, lat_2d, cth_2d = er3t.util.grid_by_dxdy(sat0.data['lon']['data'], sat0.data['lat']['data'], sat0.data['cld_height_acha']['data'], extent=extent, dx=3000.0, dy=3000.0)
                cth_2d[cth_2d<0.0]  = 0.0; cth_2d /= 1000.0
                sat0.data['cth_2d'] = dict(name='Gridded cloud top height', units='km', data=cth_2d)
                cloud_top_height = sat0.data['cth_2d']['data']
            except Exception as err:
                print('Warning [run_mcarats_one]: Cannot generate 2D CTH field, falling back to 1km for CTH <%s>...' % err)
                cld_top_height = np.ones_like(cot_2d)
        cld0 = cld_sat(sat_obj=sat0, fname=fname_cld, cth=cloud_top_height, cgt=0.4, dz=(levels[1]-levels[0]), overwrite=overwrite)
    else:
        cld0 = cld_sat(fname=fname_cld, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define phase object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    pha0 = pha_mie(wavelength=wavelength)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    #╭────────────────────────────────────────────────────────────────────────────╮#
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]

    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir, quiet=quiet, overwrite=overwrite)
    # atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir, quiet=quiet, overwrite=False)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define mcarats object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    mca0 = mcarats_ng(
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sca=sca,
            date=date,
            weights=abs0.coef['weight']['data'],
            solar_zenith_angle=solar_zenith_angle,
            fdir='%s/%.2fnm/sat/%s/%3.3d' % (fdir, wavelength, solver.lower(), index),
            Nrun=3,
            photons=photons,
            solver=solver,
            target=target,
            Ncpu=Ncpu,
            mp_mode='py',
            quiet=quiet,
            overwrite=overwrite
            # overwrite=False
            )
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define mcarats output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    out0 = mca_out_ng(fname='%s/mca-out-%s-%s_sat_%3.3d.h5' % (fdir, target.lower(), solver.lower(), index), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, quiet=quiet, overwrite=overwrite)
    # out0 = mca_out_ng(fname='%s/mca-out-%s-%s_sat_%3.3d.h5' % (fdir, target.lower(), solver.lower(), index), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, quiet=quiet, overwrite=False)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return atm0, cld0, out0

def interpolate_3d_to_flight_track(flt_trk, data_3d):

    """
    Extract radiative properties along flight track from MCARaTS outputs

    Input:
        flt_trk: Python dictionary
            ['tmhr']: UTC time in hour
            ['lon'] : longitude
            ['lat'] : latitude
            ['alt'] : altitude

        data_3d: Python dictionary
            ['lon']: longitude
            ['lat']: latitude
            ['alt']: altitude
            [...]  : other variables that contain 3D data field

    Output:
        flt_trk:
            [...]: add interpolated data from data_3d[...]
    """

    points = np.transpose(np.vstack((flt_trk['lon'], flt_trk['lat'], flt_trk['alt'])))

    lon_field = data_3d['lon']
    lat_field = data_3d['lat']
    dlon    = lon_field[1]-lon_field[0]
    dlat    = lat_field[1]-lat_field[0]
    lon_trk = flt_trk['lon']
    lat_trk = flt_trk['lat']
    indices_lon = np.int_(np.round((lon_trk-lon_field[0])/dlon, decimals=0))
    indices_lat = np.int_(np.round((lat_trk-lat_field[0])/dlat, decimals=0))

    for key in data_3d.keys():
        if key not in ['tmhr', 'lon', 'lat', 'alt']:
            f_interp     = RegularGridInterpolator((data_3d['lon'], data_3d['lat'], data_3d['alt']), data_3d[key])
            flt_trk[key] = f_interp(points)

            flt_trk['%s-alt-all' % key] = data_3d[key][indices_lon, indices_lat, :]

    return flt_trk

def get_jday_geos_east(fnames):

    """
    Get UTC time in hour from the satellite (GOES-East) file name

    Input:
        fnames: Python list, file paths of all the satellite data

    Output:
        jday: numpy array, julian day
    """

    jday = []
    for fname in fnames:
        filename = os.path.basename(fname)
        strings  = filename.split('_')

        string_t = strings[4]
        dtime0 = datetime.datetime.strptime(string_t, 's%Y%j%H%M%f')
        jday0 = er3t.util.dtime_to_jday(dtime0)
        jday.append(jday0)

    return np.array(jday)




class flt_sim:

    def __init__(
            self,
            date=datetime.datetime.now(),
            photons=2e8,
            Ncpu=16,
            wavelength=None,
            flt_trks=None,
            sat_imgs=None,
            fname=None,
            overwrite=False,
            overwrite_rtm=False,
            quiet=False,
            verbose=False
            ):

        self.date      = date
        self.photons   = photons
        self.Ncpu      = Ncpu
        self.wvl       = wavelength
        self.flt_trks  = flt_trks
        self.sat_imgs  = sat_imgs
        self.overwrite = overwrite
        self.quiet     = quiet
        self.verbose   = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif (((flt_trks is not None) and (sat_imgs is not None) and (wavelength is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((flt_trks is not None) and (sat_imgs is not None) and (wavelength is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run(overwrite=overwrite_rtm)
            self.dump(fname)

        elif (((flt_trks is not None) and (sat_imgs is not None) and (wavelength is not None)) and (fname is None)):

            self.run()

        else:

            sys.exit('Error [flt_sim]: Please check if \'%s\' exists or provide \'wavelength\', \'flt_trks\', and \'sat_imgs\' to proceed.' % fname)

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'flt_trks') and hasattr(obj, 'sat_imgs'):
                if self.verbose:
                    print('Message [flt_sim]: Loading %s ...' % fname)
                self.wvl      = obj.wvl
                self.fname    = obj.fname
                self.flt_trks = obj.flt_trks
                self.sat_imgs = obj.sat_imgs
            else:
                sys.exit('Error [flt_sim]: File \'%s\' is not the correct pickle file to load.' % fname)

    def run(self, overwrite=True):

        N = len(self.flt_trks)

        for i in range(N):

            print('%3.3d/%3.3d' % (i, N))

            flt_trk = self.flt_trks[i]
            sat_img = self.sat_imgs[i]

            try:
                atm0, cld_sat0, mca_out_ipa0 = run_mcarats_one(i, sat_img['fname'], sat_img['extent'], flt_trk['sza0'], date=self.date, wavelength=self.wvl, solver='IPA', fdir='tmp-data/%s/%09.4fnm' % (self.date.strftime('%Y%m%d'), self.wvl), photons=self.photons, Ncpu=self.Ncpu, overwrite=overwrite, quiet=self.quiet)
                atm0, cld_sat0, mca_out_3d0  = run_mcarats_one(i, sat_img['fname'], sat_img['extent'], flt_trk['sza0'], date=self.date, wavelength=self.wvl, solver='3D' , fdir='tmp-data/%s/%09.4fnm' % (self.date.strftime('%Y%m%d'), self.wvl), photons=self.photons, Ncpu=self.Ncpu, overwrite=overwrite, quiet=self.quiet)

                self.sat_imgs[i]['lon'] = cld_sat0.lay['lon']['data']
                self.sat_imgs[i]['lat'] = cld_sat0.lay['lat']['data']
                self.sat_imgs[i]['cot'] = cld_sat0.lay['cot']['data']
                self.sat_imgs[i]['cer'] = cld_sat0.lay['cer']['data'][:, :, -1]

                lon_sat = self.sat_imgs[i]['lon'][:, 0]
                lat_sat = self.sat_imgs[i]['lat'][0, :]
                dlon    = lon_sat[1]-lon_sat[0]
                dlat    = lat_sat[1]-lat_sat[0]
                lon_trk = self.flt_trks[i]['lon']
                lat_trk = self.flt_trks[i]['lat']
                indices_lon = np.int_(np.round((lon_trk-lon_sat[0])/dlon, decimals=0))
                indices_lat = np.int_(np.round((lat_trk-lat_sat[0])/dlat, decimals=0))
                self.flt_trks[i]['cot'] = self.sat_imgs[i]['cot'][indices_lon, indices_lat]
                self.flt_trks[i]['cer'] = self.sat_imgs[i]['cer'][indices_lon, indices_lat]

                if 'cth' in cld_sat0.lay.keys():
                    self.sat_imgs[i]['cth'] = cld_sat0.lay['cth']['data']
                    self.flt_trks[i]['cth'] = self.sat_imgs[i]['cth'][indices_lon, indices_lat]

                data_3d_mca = {
                    'lon'         : cld_sat0.lay['lon']['data'][:, 0],
                    'lat'         : cld_sat0.lay['lat']['data'][0, :],
                    'alt'         : atm0.lev['altitude']['data'],
                    }

                index_h = np.argmin(np.abs(atm0.lev['altitude']['data']-flt_trk['alt0']))

                if atm0.lev['altitude']['data'][index_h] > flt_trk['alt0']:
                    index_h -= 1
                if index_h < 0:
                    index_h = 0

                for key in mca_out_3d0.data.keys():
                    if key in ['f_down', 'f_down_diffuse', 'f_down_direct', 'f_up', 'toa']:
                        if 'toa' not in key:
                            vname = key.replace('_', '-') + '_mca-3d'
                            self.sat_imgs[i][vname] = mca_out_3d0.data[key]['data'][..., index_h]
                            data_3d_mca[vname] = mca_out_3d0.data[key]['data']

                for key in mca_out_ipa0.data.keys():
                    if key in ['f_down', 'f_down_diffuse', 'f_down_direct', 'f_up', 'toa']:
                        if 'toa' not in key:
                            vname = key.replace('_', '-') + '_mca-ipa'
                            self.sat_imgs[i][vname] = mca_out_ipa0.data[key]['data'][..., index_h]
                            data_3d_mca[vname] = mca_out_ipa0.data[key]['data']

                self.flt_trks[i] = interpolate_3d_to_flight_track(flt_trk, data_3d_mca)

                # figure
                #╭────────────────────────────────────────────────────────────────────────────╮#
                plot = False
                if plot:
                    rcParams['font.size'] = 12
                    plt.close('all')
                    fig = plt.figure(figsize=(12, 4))
                    fig.suptitle('%4.4d %s\n(%s)' % (i, str(er3t.util.jday_to_dtime(self.flt_trks[i]['jday0'])), os.path.basename(self.sat_imgs[i]['fname'])), y=1.03)

                    # plot1
                    #╭──────────────────────────────────────────────────────────────╮#
                    ax1 = fig.add_subplot(131)
                    cot = self.sat_imgs[i]['cot'].copy()
                    cot[cot==0.0] = np.nan
                    cs = ax1.imshow(cot.T, origin='lower', cmap='jet', zorder=0, extent=self.sat_imgs[i]['extent'], vmin=0.0, vmax=20.0)
                    ax1.scatter(self.flt_trks[i]['lon'], self.flt_trks[i]['lat'], s=2, c='k', lw=0.0)
                    ax1.set_xlabel('Longitude')
                    ax1.set_ylabel('Latitude')
                    divider = make_axes_locatable(ax1)
                    cax = divider.append_axes('right', '5%', pad='3%')
                    cbar = fig.colorbar(cs, cax=cax)
                    #╰──────────────────────────────────────────────────────────────╯#

                    # plot2
                    #╭──────────────────────────────────────────────────────────────╮#
                    ax2 = fig.add_subplot(132)
                    cer = self.sat_imgs[i]['cer'].copy()
                    cer[np.isnan(cot)] = np.nan
                    cs = ax2.imshow(cer.T, origin='lower', cmap='jet', zorder=0, extent=self.sat_imgs[i]['extent'], vmin=0.0, vmax=20.0)
                    ax2.scatter(self.flt_trks[i]['lon'], self.flt_trks[i]['lat'], s=2, c='k', lw=0.0)
                    ax2.set_xlabel('Longitude')
                    ax2.set_ylabel('Latitude')
                    # ax2.set_title('%4.4d %s\n(%s)' % (i, str(er3t.util.jday_to_dtime(self.flt_trks[i]['jday0'])), os.path.basename(self.sat_imgs[i]['fname'])), y=1.03)
                    divider = make_axes_locatable(ax2)
                    cax = divider.append_axes('right', '5%', pad='3%')
                    cbar = fig.colorbar(cs, cax=cax)
                    #╰──────────────────────────────────────────────────────────────╯#

                    # plot3
                    #╭──────────────────────────────────────────────────────────────╮#
                    ax3 = fig.add_subplot(133)
                    cth = self.sat_imgs[i]['cth'].copy()
                    cth[np.isnan(cot)] = np.nan
                    cs = ax3.imshow(cth.T, origin='lower', cmap='jet', zorder=0, extent=self.sat_imgs[i]['extent'], vmin=0.0, vmax=10.0)
                    ax3.scatter(self.flt_trks[i]['lon'], self.flt_trks[i]['lat'], s=2, c='k', lw=0.0)
                    ax3.set_xlabel('Longitude')
                    ax3.set_ylabel('Latitude')
                    # ax3.set_title('%4.4d %s\n(%s)' % (i, str(er3t.util.jday_to_dtime(self.flt_trks[i]['jday0'])), os.path.basename(self.sat_imgs[i]['fname'])), y=1.03)
                    divider = make_axes_locatable(ax3)
                    cax = divider.append_axes('right', '5%', pad='3%')
                    cbar = fig.colorbar(cs, cax=cax)
                    #╰──────────────────────────────────────────────────────────────╯#


                    # save figure
                    #╭──────────────────────────────────────────────────────────────╮#
                    fig.subplots_adjust(hspace=0.35, wspace=0.35)
                    _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
                    fname_fig = '%4.4d_%s.png' % (i, _metadata_['Function'],)
                    plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
                    #╰──────────────────────────────────────────────────────────────╯#
                    # plt.show()
                    # sys.exit()
                    plt.close(fig)
                    plt.clf()
                #╰────────────────────────────────────────────────────────────────────────────╯#

            except Exception as error:
                msg = 'Error [flt_sim]: Error <%s> encountered.' % error
                warnings.warn(msg)

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [flt_sim]: Saving object into %s ...' % fname)
            pickle.dump(self, f)

def first_run(
        date,
        wavelength=532.0,
        hsr1=True,
        ssfr=False,
        run_rtm=True,
        run_plt=True,
        fdir_sat='data/magpie/2023/sat',
        fdir_data='data/magpie/processed'
        ):

    date_s = date.strftime('%Y%m%d')
    date_s_ = date.strftime('%Y-%m-%d')

    # tmp directory
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fdir = os.path.abspath('tmp-data/%s/%09.4fnm' % (date_s, wavelength))
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # satellite data
    # get all the avaiable satellite data (GOES-East) and calculate jday for each file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    date_prev = (date-datetime.timedelta(days=1)).strftime('%m_%d_%j')
    date_this = (date).strftime('%m_%d_%j')
    date_next = (date+datetime.timedelta(days=1)).strftime('%m_%d_%j')

    fnames_sat_all = sorted(glob.glob('%s/%s/*.nc' % (fdir_sat, date_prev))) +\
                     sorted(glob.glob('%s/%s/*.nc' % (fdir_sat, date_this))) +\
                     sorted(glob.glob('%s/%s/*.nc' % (fdir_sat, date_next)))

    jday_sat_all   = get_jday_geos_east(fnames_sat_all)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # flight nav data from house-keeping file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = '%s/MAGPIE-HSK_DHC6_%s_v0.h5' % (fdir_data, date_s)
    if not os.path.exists(fname_hsk):
        sys.exit('Error [first_run]: Cannot find HSK data.')

    f_hsk = h5py.File(fname_hsk, 'r')
    jday   = f_hsk['jday'][...]
    sza    = f_hsk['sza'][...]
    lon    = f_hsk['lon'][...]
    lat    = f_hsk['lat'][...]
    pit    = f_hsk['ang_pit'][...]
    rol    = f_hsk['ang_rol'][...]
    alt    = f_hsk['alt'][...]
    tmhr   = f_hsk['tmhr'][...]
    f_hsk.close()

    logic = (jday>=jday_sat_all[0]) & (jday<=jday_sat_all[-1]) & \
            (np.logical_not(np.isnan(jday))) & (np.logical_not(np.isnan(sza)))
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # put flight nav data into flt_trk (Python dict)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    flt_trk = {}
    flt_trk['jday'] = jday[logic]
    flt_trk['lon']  = lon[logic]
    flt_trk['lat']  = lat[logic]
    flt_trk['ang_pit']  = pit[logic]
    flt_trk['ang_rol']  = rol[logic]
    flt_trk['sza']  = sza[logic]
    flt_trk['tmhr'] = tmhr[logic]
    flt_trk['alt']  = alt[logic]/1000.0
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # SSFR data
    # we didn't fly SSFR for MAGPIE but code is presented for an example
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if ssfr:

        # fname_hsk = '%s/MAGPIE-SSFR_DHC6_%s_v2.h5' % (fdir_data, date_s)
        # f_ssfr = h5py.File(fname_ssfr, 'r')
        # flt_trk['f-up_ssfr']   = f_ssfr['nad_flux'][...][:, np.argmin(np.abs(f_ssfr['nad_wvl'][...]-wavelength))][logic]
        # flt_trk['f-down_ssfr'] = f_ssfr['zen_flux'][...][:, np.argmin(np.abs(f_ssfr['zen_wvl'][...]-wavelength))][logic]
        # logic_turn = (np.abs(pit[logic])>5.0) | (np.abs(rol[logic])>5.0)
        # flt_trk['f-up_ssfr'][logic_turn]   = np.nan
        # flt_trk['f-down_ssfr'][logic_turn] = np.nan
        # f_ssfr.close()

        pass
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # HSR1 (used to be called SPNS) data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if hsr1:

        fname_hsr1 = '%s/MAGPIE_SPN-S_%s_v2.h5' % (fdir_data, date_s_)
        f_hsr1 = h5py.File(fname_hsr1, 'r')
        flt_trk['f-down-diffuse_hsr1']= f_hsr1['dif/flux'][:, np.argmin(np.abs(f_hsr1['dif/wvl'][...]-wavelength))][logic]
        flt_trk['f-down_hsr1']= f_hsr1['tot/flux'][:, np.argmin(np.abs(f_hsr1['tot/wvl'][...]-wavelength))][logic]
        f_hsr1.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # re-select satellite data based on flight duration
    #╭────────────────────────────────────────────────────────────────────────────╮#
    indices = np.where((jday_sat_all>=(flt_trk['jday'].min()-20.0/1440.0)) & (jday_sat_all<=(flt_trk['jday'].max()+20.0/1440.0)))[0]
    fnames_sat = [fnames_sat_all[i] for i in indices]
    jday_sat   = jday_sat_all[indices]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # partition flight track
    #╭────────────────────────────────────────────────────────────────────────────╮#
    flt_trks = partition_flight_track(flt_trk, tmhr_interval=0.05, margin_px=25.0, margin_py=25.0)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate sat_imgs (Python list) to store satellite data related info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sat_imgs = []
    for i in range(len(flt_trks)):
        sat_img = {}

        index0  = np.argmin(np.abs(jday_sat-flt_trks[i]['jday0']))
        sat_img['fname']  = fnames_sat[index0]
        sat_img['extent'] = flt_trks[i]['extent']

        sat_imgs.append(sat_img)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    sim0 = flt_sim(date=date, wavelength=wavelength, flt_trks=flt_trks, sat_imgs=sat_imgs, fname='data/flt_sim_%09.4fnm_%s.pk' % (wavelength, date_s), overwrite=True, overwrite_rtm=run_rtm)

    # os.system('rm -rf %s' % fdir)



if __name__ == '__main__':

    run_rtm=True
    run_plt=False

    dates = [
            datetime.datetime(2023, 8, 15), # heaviest aerosol condition
            # datetime.datetime(2023, 8, 2),
            # datetime.datetime(2023, 8, 3),
            # datetime.datetime(2023, 8, 5),
            # datetime.datetime(2023, 8, 13),
            # datetime.datetime(2023, 8, 14), # heavy aerosol condition
            # datetime.datetime(2023, 8, 16),
            # datetime.datetime(2023, 8, 18),
            # datetime.datetime(2023, 8, 20),
            # datetime.datetime(2023, 8, 21),
            # datetime.datetime(2023, 8, 22),
            # datetime.datetime(2023, 8, 23),
            # datetime.datetime(2023, 8, 25),
            # datetime.datetime(2023, 8, 26),
            # datetime.datetime(2023, 8, 27),
            # datetime.datetime(2023, 8, 28), # bad dewpoint temperature (thus RH) data at the end of the flight
        ]

    wavelength = 745.0

    for date in dates:
        first_run(date, run_rtm=run_rtm, run_plt=run_plt, wavelength=wavelength, hsr1=True)
        save_h5(date, wavelength=wavelength)
