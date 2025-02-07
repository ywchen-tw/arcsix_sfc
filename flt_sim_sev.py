import os
import sys
import glob
import copy
import h5py
import numpy as np
import datetime
import time
import pickle
import multiprocessing as mp
from scipy.io import readsav
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_les, cld_sev

from er3t.rtm.mca import mca_atm_1d, mca_atm_3d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.util import send_email



def get_tmhr_seviri(fnames):

    """
    Get UTC time in hour from the satellite (SEVIRI) file name

    Input:
        fnames: Python list, file paths of all the satellite data

    Output:
        tmhr: numpy array, UTC time in hour
    """

    tmhr = []
    for fname in fnames:
        filename = os.path.basename(fname)
        time_s   = filename.split('.')[1]
        tmhr.append(float(time_s[:2]) + float(time_s[2:])/60.0)

    return np.array(tmhr)



def partition_flight_track(flt_trk, tmhr_interval=0.1, margin_x=1.0, margin_y=1.0):

    """
    Input:
        flt_trk: Python dictionary that contains
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
            [i]['tmhr'] : numpy array, UTC time in hour
            [i]['lon']  : numpy array, longitude
            [i]['lat']  : numpy array, latitude
            [i]['alt']  : numpy array, altitude
            [i]['sza']  : numpy array, solar zenith angle
            [i]['tmhr0']: mean value
            [i]['lon0'] : mean value
            [i]['lat0'] : mean value
            [i]['alt0'] : mean value
            [i]['sza0'] : mean value
            [i][...]    : numpy array, other data variables
    """

    tmhr_start = tmhr_interval * (flt_trk['tmhr'][0] //tmhr_interval)
    tmhr_end   = tmhr_interval * (flt_trk['tmhr'][-1]//tmhr_interval + 1)

    tmhr_edges = np.arange(tmhr_start, tmhr_end+tmhr_interval, tmhr_interval)

    flt_trk_segments = []

    for i in range(tmhr_edges.size-1):

        logic = (flt_trk['tmhr']>=tmhr_edges[i]) & (flt_trk['tmhr']<=tmhr_edges[i+1])

        if logic.sum() > 0:

            flt_trk_segment = {}
            for key in flt_trk.keys():
                flt_trk_segment[key]     = flt_trk[key][logic]
                if key in ['tmhr', 'lon', 'lat', 'alt', 'sza']:
                    flt_trk_segment[key+'0'] = flt_trk_segment[key].mean()

            flt_trk_segment['extent'] = np.array([flt_trk_segment['lon'].min()-margin_x, \
                                                  flt_trk_segment['lon'].max()+margin_x, \
                                                  flt_trk_segment['lat'].min()-margin_y, \
                                                  flt_trk_segment['lat'].max()+margin_y])

            flt_trk_segments.append(flt_trk_segment)

    return flt_trk_segments



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

    for key in data_3d.keys():
        if key not in ['tmhr', 'lon', 'lat', 'alt']:
            f_interp     = RegularGridInterpolator((data_3d['lon'], data_3d['lat'], data_3d['alt']), data_3d[key])
            flt_trk[key] = f_interp(points)

    return flt_trk



def run_mcarats_single(
        index,
        fname_sat,
        extent,
        solar_zenith_angle,
        fdir='tmp-data/01',
        wavelength=600.0,
        date=datetime.datetime(2017, 8, 13),
        target='flux',
        solver='3D',
        photons=1e6,
        Ncpu=10,
        overwrite=True,
        quiet=False
        ):

    """
    Run MCARaTS with specified inputs (a general function from 04_pre_mca.py)
    """


    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    levels    = np.linspace(0.0, 20.0, 201)
    fname_atm = '%s/atm_%3.3d.pk' % (fdir, index)
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # ------------------------------------------------------------------------------------------------------


    # define an absorption object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_abs = '%s/abs_%3.3d.pk' % (fdir, index)
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    # ------------------------------------------------------------------------------------------------------

    # define an cloud object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_cld = '%s/cld_sev_%3.3d.pk' % (fdir, index)
    cld0      = cld_sev(fname_h4=fname_sat, fname=fname_cld, extent=extent, overwrite=overwrite)
    # ----------------------------------------------------------------------------------------------------


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]

    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir, quiet=quiet, overwrite=overwrite)
    atm_3ds = [atm3d0]
    # ------------------------------------------------------------------------------------------------------


    # define mcarats object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    mca0 = mcarats_ng(
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            date=date,
            solar_zenith_angle=solar_zenith_angle,
            fdir='%s/%.2fnm/seviri/%s/%3.3d' % (fdir, wavelength, solver.lower(), index),
            Nrun=3,
            photons=photons,
            solver=solver,
            target=target,
            Ncpu=Ncpu,
            mp_mode='py',
            quiet=quiet,
            overwrite=overwrite
            )
    # ------------------------------------------------------------------------------------------------------


    # define mcarats output object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    out0 = mca_out_ng(fname='%s/mca-out-%s-%s_seviri_%3.3d.h5' % (fdir, target.lower(), solver.lower(), index), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, quiet=quiet, overwrite=overwrite)
    # ------------------------------------------------------------------------------------------------------

    return atm0, cld0, out0



class flt_sim:


    def __init__(
            self,
            wavelength=None,
            flt_trks=None,
            sat_imgs=None,
            fname=None,
            overwrite=True,
            quiet=False,
            verbose=False
            ):

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

            self.run()
            self.dump(fname)

        elif (((flt_trks is not None) and (sat_imgs is not None) and (wavelength is not None)) and (fname is None)):

            self.run()

        else:

            sys.exit('Error   [flt_sim]: Please check if \'%s\' exists or provide \'wavelength\', \'flt_trks\', and \'sat_imgs\' to proceed.' % fname)


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
                sys.exit('Error   [flt_sim]: File \'%s\' is not the correct pickle file to load.' % fname)


    def run(self):

        N = len(self.flt_trks)

        for i in range(N):

            flt_trk = self.flt_trks[i]
            sat_img = self.sat_imgs[i]

            atm0, cld_sev0, mca_out_ipa0 = run_mcarats_single(i, sat_img['fname'], sat_img['extent'], flt_trk['sza0'], wavelength=self.wvl, solver='IPA', overwrite=self.overwrite, quiet=self.quiet)
            atm0, cld_sev0, mca_out_3d0  = run_mcarats_single(i, sat_img['fname'], sat_img['extent'], flt_trk['sza0'], wavelength=self.wvl, solver='3D' , overwrite=self.overwrite, quiet=self.quiet)

            self.sat_imgs[i]['lon'] = cld_sev0.lay['lon']['data']
            self.sat_imgs[i]['lat'] = cld_sev0.lay['lat']['data']
            self.sat_imgs[i]['cot'] = cld_sev0.lay['cot']['data']
            self.sat_imgs[i]['cer'] = cld_sev0.lay['cer']['data']

            data_3d_mca = {
                'lon'         : cld_sev0.lay['lon']['data'][:, 0],
                'lat'         : cld_sev0.lay['lat']['data'][0, :],
                'alt'         : atm0.lev['altitude']['data'],
                }

            index_h = np.argmin(np.abs(atm0.lev['altitude']['data']-flt_trk['alt0']))
            if atm0.lev['altitude']['data'][index_h] > flt_trk['alt0']:
                index_h -= 1
            if index_h < 0:
                index_h = 0

            for key in mca_out_3d0.data.keys():
                vname = key.replace('_', '-') + '_mca-3d'
                self.sat_imgs[i][vname] = mca_out_3d0.data[key]['data'][..., index_h]
                data_3d_mca[vname] = mca_out_3d0.data[key]['data']
            for key in mca_out_ipa0.data.keys():
                vname = key.replace('_', '-') + '_mca-ipa'
                self.sat_imgs[i][vname] = mca_out_ipa0.data[key]['data'][..., index_h]
                data_3d_mca[vname] = mca_out_ipa0.data[key]['data']

            self.flt_trks[i] = interpolate_3d_to_flight_track(flt_trk, data_3d_mca)


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [flt_sim]: Saving object into %s ...' % fname)
            pickle.dump(self, f)



def first_run(wavelength=600.0):

    # create test-data/04 directory if it does not exist
    fdir = os.path.abspath('tmp-data/01')
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # +
    # prepare the data
    # this is the main part to change for your own use

    # get all the avaiable satellite data (SEVIRI) and calculate the time in hour for each file
    fnames_seviri = sorted(glob.glob('data/flt_sim/sat/*.hdf'))
    tmhr_seviri   = get_tmhr_seviri(fnames_seviri)

    # flt_trks
    p3    = readsav('data/flt_sim/flt.out')
    ssfr  = readsav('data/flt_sim/ssfr.out')
    logic = (p3.utc>=tmhr_seviri[0]) & (p3.utc<=tmhr_seviri[-1])

    flt_trk = {}
    flt_trk['tmhr'] = p3.utc[logic]
    flt_trk['lon']  = p3.lon[logic]
    flt_trk['lat']  = p3.lat[logic]
    flt_trk['alt']  = p3.alt[logic]/1000.0
    flt_trk['sza']  = p3.sza[logic]
    flt_trk['f-up_ssfr']   = ssfr.nadspectra[:, np.argmin(np.abs(ssfr.nadlambda-wavelength))][logic]
    flt_trk['f-down_ssfr'] = ssfr.zenspectra[:, np.argmin(np.abs(ssfr.zenlambda-wavelength))][logic]

    # partition flight track
    flt_trks = partition_flight_track(flt_trk, tmhr_interval=0.05, margin_x=1.0, margin_y=1.0)


    # sat_imgs
    sat_imgs = []
    for i in range(len(flt_trks)):
        sat_img = {}

        index0  = np.argmin(np.abs(tmhr_seviri-flt_trks[i]['tmhr0']))
        sat_img['fname']  = fnames_seviri[index0]
        sat_img['extent'] = flt_trks[i]['extent']

        sat_imgs.append(sat_img)
    # -


    sim0 = flt_sim(wavelength=wavelength, flt_trks=flt_trks, sat_imgs=sat_imgs, fname='flt_sim_%.2fnm.pk' % wavelength, overwrite=True)

    # os.system('rm -rf %s' % fdir)



def plot_flux_comp_time_series(flt_sim0):

    fig = plt.figure(figsize=(12, 10))
    # fig = plt.figure(figsize=(6, 5))
    ax1 = fig.add_subplot(211)
    ax2 = ax1.twinx()

    ax3 = fig.add_subplot(212)
    ax4 = ax3.twinx()

    for flt_trk in flt_sim0.flt_trks:

        ax1.plot(flt_trk['tmhr'], flt_trk['f-up_mca-ipa'], c='b', lw=2.5)
        ax1.plot(flt_trk['tmhr'], flt_trk['f-up_mca-3d'] ,  c='r', lw=1.5)
        ax1.scatter(flt_trk['tmhr'], flt_trk['f-up_ssfr'],  c='k', s=1)
        ax2.plot(flt_trk['tmhr'], flt_trk['alt'], c='orange', lw=1.5, alpha=0.7)

        ax3.plot(flt_trk['tmhr'], flt_trk['f-down_mca-ipa'], c='b', lw=2.5)
        ax3.plot(flt_trk['tmhr'], flt_trk['f-down_mca-3d'] ,  c='r', lw=1.5)
        ax3.scatter(flt_trk['tmhr'], flt_trk['f-down_ssfr'],  c='k', s=1)
        ax4.plot(flt_trk['tmhr'], flt_trk['alt'], c='orange', lw=1.5, alpha=0.7)


    ax1.set_ylim((0.0, 1.5))
    ax1.set_xlabel('UTC [hour]')
    ax1.set_ylabel('$F_\\uparrow$ [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_title('Flux Comparison at %.2f nm' % sim0.wvl)
    patches_legend = [
                mpatches.Patch(color='black' , label='SSFR'),
                mpatches.Patch(color='red'   , label='MCARaTS 3D'),
                mpatches.Patch(color='blue'  , label='MCARaTS IPA'),
                mpatches.Patch(color='orange', label='Altitude')
                ]
    ax2.set_ylabel('Altitude [km]', color='orange', rotation=270, labelpad=20)
    ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)
    ax2.set_ylim((0.0, 8.0))

    ax3.set_ylabel('$F_\downarrow$ [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax3.set_xlabel('UTC [hour]')
    patches_legend = [
                mpatches.Patch(color='black' , label='SSFR'),
                mpatches.Patch(color='red'   , label='MCARaTS 3D'),
                mpatches.Patch(color='blue'  , label='MCARaTS IPA'),
                mpatches.Patch(color='orange', label='Altitude')
                ]
    ax4.set_ylabel('Altitude [km]', color='orange', rotation=270, labelpad=20)
    ax3.legend(handles=patches_legend, loc='upper right', fontsize=12)
    ax3.set_ylim((0.0, 2.0))
    ax4.set_ylim((0.0, 8.0))

    plt.savefig('flux_comp_%.2fnm.png' % sim0.wvl, bbox_inches='tight')
    plt.show()
    # plt.close(fig)
    # ---------------------------------------------------------------------



def plot_flux_2d(flt_sim0):

    N = len(flt_sim0.flt_trks)

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    for i in range(N):

        flt_trk      = flt_sim0.flt_trks[i]
        sat_img      = flt_sim0.sat_imgs[i]

        ax1.imshow(sat_img['cot']           , extent=sat_img['extent'], cmap='Greys_r', vmin=0.0, vmax=50.0, alpha=0.1, zorder=0)
        ax2.imshow(sat_img['f-up_mca-3d']   , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax3.imshow(sat_img['f-up_mca-ipa']  , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax4.imshow(sat_img['cer']           , extent=sat_img['extent'], cmap='Greys_r', vmin=0.0, vmax=24.0, alpha=0.1, zorder=0)
        ax5.imshow(sat_img['f-down_mca-3d'] , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax6.imshow(sat_img['f-down_mca-ipa'], extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)

        ax1.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax2.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax3.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax4.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax5.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax6.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)

    ax1.set_title('COT')
    ax2.set_title('$F_\\uparrow$ 3D')
    ax3.set_title('$F_\\uparrow$ IPA')
    ax4.set_title('CER')
    ax5.set_title('$F_\downarrow$ 3D')
    ax6.set_title('$F_\downarrow$ IPA')

    plt.show()



def plot_flux_comp_scatter(flt_sim0):


    fig = plt.figure(figsize=(6.5, 6))
    ax1 = fig.add_subplot(111)

    for flt_trk in flt_sim0.flt_trks:

        if flt_trk['alt0'] > 5.0:
            ax1.scatter(flt_trk['f-up_ssfr'], flt_trk['f-up_mca-ipa'],  c='b', s=5.0, alpha=0.5)
            ax1.scatter(flt_trk['f-up_ssfr'], flt_trk['f-up_mca-3d'] ,  c='r', s=2.5)

    ax1.plot([0, 2], [0, 2], lw=1.0, ls='--', c='gray')
    ax1.set_xlabel('SSFR $F_\\uparrow$ [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_ylabel('MCARaTS $F_\\uparrow$ [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_title('Flux at %.2f nm' % flt_sim0.wvl)
    ax1.set_xlim((0.0, 1.2))
    ax1.set_ylim((0.0, 1.2))

    patches_legend = [
                mpatches.Patch(color='red'   , label='3D'),
                mpatches.Patch(color='blue'  , label='IPA')
                ]

    ax1.legend(handles=patches_legend, loc='upper left', fontsize=12)
    plt.savefig('flux_comp_scatter_%.2fnm.png' % flt_sim0.wvl, bbox_inches='tight')
    plt.show()



def plot_flux_2d(flt_sim0):

    N = len(flt_sim0.flt_trks)

    # fig = plt.figure(figsize=(12, 10))
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[0, 2]
    ax4 = axes[1, 0]
    ax5 = axes[1, 1]
    ax6 = axes[1, 2]

    for i in range(N):

        flt_trk      = flt_sim0.flt_trks[i]
        sat_img      = flt_sim0.sat_imgs[i]

        ax1.imshow(sat_img['cot']           , extent=sat_img['extent'], cmap='Greys_r', vmin=0.0, vmax=50.0, alpha=0.1, zorder=0)
        ax2.imshow(sat_img['f-up_mca-3d']   , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax3.imshow(sat_img['f-up_mca-ipa']  , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax4.imshow(sat_img['cer']           , extent=sat_img['extent'], cmap='Greys_r', vmin=0.0, vmax=24.0, alpha=0.1, zorder=0)
        ax5.imshow(sat_img['f-down_mca-3d'] , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax6.imshow(sat_img['f-down_mca-ipa'], extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)

        ax1.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax2.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax3.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax4.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax5.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax6.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)

    ax1.set_title('COT')
    ax2.set_title('$F_\\uparrow$ 3D')
    ax3.set_title('$F_\\uparrow$ IPA')
    ax4.set_title('CER')
    ax5.set_title('$F_\downarrow$ 3D')
    ax6.set_title('$F_\downarrow$ IPA')

    plt.show()



def plot_video_frame(statements):

    flt_sim0, index_trk, index_pnt, n = statements

    fig = plt.figure(figsize=(15, 5))

    gs = gridspec.GridSpec(2, 8)

    ax1 = fig.add_subplot(gs[:, :2])
    divider = make_axes_locatable(ax1)
    ax0 = divider.append_axes('right', size='5%', pad=0.0)
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, 2:])

    fig.subplots_adjust(hspace=0.0, wspace=1.0)

    for itrk in range(index_trk+1):

        flt_trk      = flt_sim0.flt_trks[itrk]
        sat_img      = flt_sim0.sat_imgs[itrk]

        if itrk == index_trk:
            alpha = 0.9
            ax1.scatter(flt_trk['lon'][:index_pnt], flt_trk['lat'][:index_pnt], c='g', s=1)
            ax1.scatter(flt_trk['lon'][index_pnt] , flt_trk['lat'][index_pnt] , c='r', s=30)

            ax2.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-up_mca-ipa'][:index_pnt+1], c='b', s=4)
            ax2.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-up_mca-3d'][:index_pnt+1] , c='r', s=4)
            ax2.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-up_ssfr'][:index_pnt+1]   , c='k', s=2)

            ax3.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down_mca-ipa'][:index_pnt+1], c='b', s=4)
            ax3.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down_mca-3d'][:index_pnt+1] , c='r', s=4)
            ax3.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down_ssfr'][:index_pnt+1]   , c='k', s=2)

        else:
            alpha = 0.4
            ax1.scatter(flt_trk['lon'], flt_trk['lat'], c='g', s=1, alpha=1.0)

            ax2.scatter(flt_trk['tmhr'], flt_trk['f-up_mca-ipa'], c='b', s=4)
            ax2.scatter(flt_trk['tmhr'], flt_trk['f-up_mca-3d'] , c='r', s=4)
            ax2.scatter(flt_trk['tmhr'], flt_trk['f-up_ssfr']   , c='k', s=2)

            ax3.scatter(flt_trk['tmhr'], flt_trk['f-down_mca-ipa'], c='b', s=4)
            ax3.scatter(flt_trk['tmhr'], flt_trk['f-down_mca-3d'] , c='r', s=4)
            ax3.scatter(flt_trk['tmhr'], flt_trk['f-down_ssfr']   , c='k', s=2)

        ax1.imshow(sat_img['cot'].T, extent=sat_img['extent'], cmap='Greys_r', origin='lower', vmin=0.0, vmax=30.0, alpha=alpha, aspect='auto', zorder=0)

    ax2.axvline(flt_trk['tmhr'][index_pnt], lw=1.0, color='gray')
    ax3.axvline(flt_trk['tmhr'][index_pnt], lw=1.0, color='gray')

    ax0.set_ylim((0.0, 8.0))
    ax0.axhline(flt_trk['alt'][index_pnt], lw=2.0, color='r')
    ax0.xaxis.set_ticks([])
    ax0.yaxis.tick_right()
    ax0.yaxis.set_label_position('right')
    ax0.set_ylabel('Altitude [km]', rotation=270.0, labelpad=20)


    ax1.set_xlim((3,  6.2))
    ax1.set_ylim((-11, -4))
    ax1.set_xlabel('Longitude [$^\circ$]')
    ax1.set_ylabel('Latitude [$^\circ$]')

    ax2.set_xlim((11.0, 15.5))
    ax2.set_ylim((0.0, 1.5))
    ax2.xaxis.set_ticks([])
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel('$F_\\uparrow [\mathrm{W m^{-2} nm^{-1}}]$', rotation=270.0, labelpad=20)

    patches_legend = [
                mpatches.Patch(color='black' , label='SSFR'),
                mpatches.Patch(color='red'   , label='RTM 3D'),
                mpatches.Patch(color='blue'  , label='RTM IPA')
                ]
    ax2.legend(handles=patches_legend, bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=len(patches_legend), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=16)

    ax3.set_xlim((11.0, 15.5))
    ax3.set_ylim((0.0, 2.0))
    ax3.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 1.6, 0.5)))
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    ax3.set_xlabel('UTC [hour]')
    ax3.set_ylabel('$F_\downarrow [\mathrm{W m^{-2} nm^{-1}}]$', rotation=270.0, labelpad=20)


    plt.savefig('%5.5d.png' % n, bbox_inches='tight')
    plt.close(fig)



def create_video_frames(flt_sim0):

    Ntrk        = len(flt_sim0.flt_trks)
    indices_trk = np.array([], dtype=np.int32)
    indices_pnt = np.array([], dtype=np.int32)
    for itrk in range(Ntrk):
        indices_trk = np.append(indices_trk, np.repeat(itrk, flt_sim0.flt_trks[itrk]['tmhr'].size))
        indices_pnt = np.append(indices_pnt, np.arange(flt_sim0.flt_trks[itrk]['tmhr'].size))

    Npnt        = indices_trk.size
    indices     = np.arange(Npnt)

    interval = 10
    indices_trk = indices_trk[::interval]
    indices_pnt = indices_pnt[::interval]
    indices     = indices[::interval]

    statements = zip([flt_sim0]*indices_trk.size, indices_trk, indices_pnt, indices)

    with mp.Pool(processes=12) as pool:
        r = list(tqdm(pool.imap(plot_video_frame, statements), total=indices_trk.size))

    # make video
    # ffmpeg -y -framerate 30 -pattern_type glob -i '*.png' -vf scale=2000:-1 -c:v libx264 -pix_fmt yuv420p oracles.mp4



if __name__ == '__main__':

    # for wavelength in np.array([355, 380, 452, 470, 501, 520, 530, 532, 550, 606, 620, 660, 675, 700, 781, 865, 1020, 1040, 1064, 1236, 1250, 1559, 1627, 1650]):
    # for wavelength in np.array([532.0]):
    #     first_run(wavelength=wavelength)


    sim0 = flt_sim(fname='flt_sim_532.00nm.pk', overwrite=False)
    # plot_flux_comp_time_series(sim0)
    # plot_flux_2d(sim0)
    # plot_flux_comp_scatter(sim0)

    create_video_frames(sim0)
