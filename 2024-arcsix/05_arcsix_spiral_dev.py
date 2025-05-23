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
# mpl.use('TkAgg')


import ssfr



# parameters
#/----------------------------------------------------------------------------\#
_mission_     = 'arcsix'
_platform_    = 'p3b'

_hsk_         = 'hsk'
_alp_         = 'alp'
_spns_        = 'spns-a'
_ssfr1_       = 'ssfr-a'
_ssfr2_       = 'ssfr-b'
_cam_         = 'nac'

_fdir_hsk_   = 'data/arcsix/2024-Spring/p3/aux/hsk'
_fdir_cal_   = 'data/%s/cal' % _mission_

_fdir_data_  = 'data/%s/processed' % _mission_
_fdir_out_   = '%s/processed' % _fdir_data_


_verbose_   = True

_fnames_ = {}

_alp_time_offset_ = {
        '20240517':   5.55,
        '20240521': -17.94,
        '20240524': -18.39,
        '20240528': -17.19,
        }
_spns_time_offset_ = {
        '20240517': 0.0,
        '20240521': 0.0,
        '20240524': 86400.0,
        '20240528': 0.0,
        }
_ssfr1_time_offset_ = {
        '20240517': 185.0,
        '20240521': 182.0,
        '20240524': -145.75,
        '20240528': -156.26,
        }
_ssfr2_time_offset_ = {
        '20240517': 115.0,
        '20240521': -6.0,
        '20240524': -208.22,
        '20240528': -222.66,
        }
#\----------------------------------------------------------------------------/#







# functions for ssfr calibrations
#/----------------------------------------------------------------------------\#
def wvl_cal(ssfr_tag, lc_tag, lamp_tag, Nchan=256):

    fdir_data = '/argus/field/arcsix/cal/wvl-cal'

    indices_spec = {
            'zen': [0, 1],
            'nad': [2, 3]
            }

    fdir =  sorted(glob.glob('%s/*%s*%s*%s*' % (fdir_data, ssfr_tag, lc_tag, lamp_tag)))[0]
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))

    ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp')

    xchan = np.arange(Nchan)

    spectra0 = np.nanmean(ssfr0.dset0['spectra_dark-corr'][:, :, indices_spec[lc_tag]], axis=0)
    spectra1 = np.nanmean(ssfr0.dset1['spectra_dark-corr'][:, :, indices_spec[lc_tag]], axis=0)

    # spectra_inp = {lamp_tag.lower(): spectra0[:, 0]}
    # ssfr.cal.cal_wvl_coef(spectra_inp, which_spec='lasp|%s|%s|si' % (ssfr_tag.lower(), lc_tag.lower()))

    spectra_inp = {lamp_tag.lower(): spectra0[:, 1]}
    ssfr.cal.cal_wvl_coef(spectra_inp, which_spec='lasp|%s|%s|in' % (ssfr_tag.lower(), lc_tag.lower()))
    sys.exit()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('%s %s (illuminated by %s Lamp)' % (ssfr_tag.upper(), lc_tag.title(), lamp_tag.upper()))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(121)
        ax1.plot(xchan, spectra0[:, 0], lw=1, c='r')
        ax1.plot(xchan, spectra1[:, 0], lw=1, c='b')
        ax1.set_xlabel('Channel #')
        ax1.set_ylabel('Counts')
        ax1.set_ylim(bottom=0)
        ax1.set_title('Silicon')

        ax2 = fig.add_subplot(122)
        ax2.plot(xchan, spectra0[:, 1], lw=1, c='r')
        ax2.plot(xchan, spectra1[:, 1], lw=1, c='b')
        ax2.set_xlabel('Channel #')
        ax2.set_ylabel('Counts')
        ax2.set_ylim(bottom=0)
        ax2.set_title('InGaAs')
        #\--------------------------------------------------------------/#

        patches_legend = [
                          mpatches.Patch(color='red' , label='IntTime set 1'), \
                          mpatches.Patch(color='blue', label='IntTime set 2'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s_%s_%s.png' % (_metadata['Function'], ssfr_tag.lower(), lc_tag.lower(), lamp_tag.lower()), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

def rad_cal(ssfr_tag, lc_tag, lamp_tag, Nchan=256):

    fdir_data = 'data/arcsix/cal/rad-cal'

    indices_spec = {
            'zen': [0, 1],
            'nad': [2, 3]
            }

    fdir =  sorted(glob.glob('%s/*%s*%s*%s*' % (fdir_data, ssfr_tag, lc_tag, lamp_tag)))[0]
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))


    date_cal_s   = '2023-11-16'
    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr(fnames)
    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        fname = '%s/cal/%s|cal-rad-pri|lasp|%s|%s|%s-si%3.3d-in%3.3d|%s.h5' % (ssfr.common.fdir_data, date_cal_s, ssfr_tag.lower(), lc_tag.lower(), dset_tag.lower(), int_time['%s|si' % lc_tag], int_time['%s|in' % lc_tag], date_today_s)
        f = h5py.File(fname, 'w')

        resp_pri = ssfr.cal.cal_rad_resp(fnames, which_ssfr='lasp|%s' % ssfr_tag.lower(), which_lc=lc_tag.lower(), int_time=int_time, which_lamp=lamp_tag.lower())

        for key in resp_pri.keys():
            f[key] = resp_pri[key]

        f.close()

def ang_cal(fdir):

    """

    Notes:
        angular calibration is done for three different azimuth angles (reference to the vaccum port)
        60, 180, 300

        angles
    """

    tags = os.path.basename(fdir).split('_')
    ssfr_tag = tags[1]
    lc_tag   = tags[2]

    # get angles
    #/----------------------------------------------------------------------------\#
    angles_pos = np.concatenate((np.arange(0.0, 30.0, 3.0), np.arange(30.0, 50.0, 5.0), np.arange(50.0, 91.0, 10.0)))
    angles_neg = -angles_pos
    angles = np.concatenate((angles_pos, angles_neg, np.array([0.0])))
    #\----------------------------------------------------------------------------/#

    # make fnames, a dictionary <key:value> with file name as key, angle as value
    #/----------------------------------------------------------------------------\#
    fnames_ = sorted(glob.glob('%s/*.SKS' % fdir))
    fnames  = {
            fnames_[i]: angles[i] for i in range(angles.size)
            }
    #\----------------------------------------------------------------------------/#

    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr([fnames_[0]])
    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        filename_tag = '%s|%s|%s|%s' % (tags[0], tags[4], date_today_s, dset_tag)

        ssfr.cal.cdata_ang_resp(fnames, filename_tag=filename_tag, which_ssfr='lasp|%s' % ssfr_tag, which_lc=lc_tag, int_time=int_time)
#\----------------------------------------------------------------------------/#



# instrument calibrations
#/----------------------------------------------------------------------------\#
def main_calibration():

    """
    Notes:
        irradiance setup:
            SSFR-A (Alvin)
              - nadir : LC6 + stainless steel cased fiber
              - zenith: LC4 + black plastic cased fiber
    """

    # wavelength calibration
    #/----------------------------------------------------------------------------\#
    # for ssfr_tag in ['SSFR-A', 'SSFR-B']:
    #     for lc_tag in ['zen', 'nad']:
    #         for lamp_tag in ['kr', 'hg']:
    #             wvl_cal(ssfr_tag, lc_tag, lamp_tag)
    #\----------------------------------------------------------------------------/#

    # radiometric calibration
    #/----------------------------------------------------------------------------\#
    # for ssfr_tag in ['SSFR-A', 'SSFR-B']:
    #     for lc_tag in ['zen', 'nad']:
    #         for lamp_tag in ['1324']:
    #             rad_cal(ssfr_tag, lc_tag, lamp_tag)
    #\----------------------------------------------------------------------------/#

    # angular calibration
    #/----------------------------------------------------------------------------\#
    fdirs = [
            '/argus/field/arcsix/cal/ang-cal/2024-03-19_SSFR-A_zen-lc4_ang-cal_vaa-060_lamp-507_si-080-120_in-250-350',
            '/argus/field/arcsix/cal/ang-cal/2024-03-15_SSFR-A_zen-lc4_ang-cal_vaa-180_lamp-507_si-080-120_in-250-350',
            '/argus/field/arcsix/cal/ang-cal/2024-03-19_SSFR-A_zen-lc4_ang-cal_vaa-300_lamp-507_si-080-120_in-250-350',
            '/argus/field/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad-lc6_ang-cal_vaa-060_lamp-507_si-080-120_in-250-350',
            '/argus/field/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad-lc6_ang-cal_vaa-180_lamp-507_si-080-120_in-250-350',
            '/argus/field/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad-lc6_ang-cal_vaa-300_lamp-507_si-080-120_in-250-350',
            ]
    for fdir in fdirs:
        ang_cal(fdir)
    #\----------------------------------------------------------------------------/#

def test_data_a(ssfr_tag, lc_tag, lamp_tag, Nchan=256):

    fdir_data = '/argus/field/arcsix/cal/rad-cal'

    indices_spec = {
            'zen': [0, 1],
            'nad': [2, 3]
            }

    fdir =  sorted(glob.glob('%s/*%s*%s*%s*' % (fdir_data, ssfr_tag, lc_tag, lamp_tag)))[0]
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))


    date_cal_s   = '2023-11-16'
    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr(fnames)
    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        fname = '%s/cal/%s|RAD-CAL-PRI|LASP|%s|%s|%s-SI%3.3d-IN%3.3d|%s.h5' % (ssfr.common.fdir_data, date_cal_s, ssfr_tag.upper(), lc_tag.upper(), dset_tag.upper(), int_time['%s|si' % lc_tag], int_time['%s|in' % lc_tag], date_today_s)
        f = h5py.File(fname, 'w')

        resp_pri = ssfr.cal.cal_rad_resp(fnames, which_ssfr='lasp|%s' % ssfr_tag.lower(), which_lc=lc_tag.lower(), int_time=int_time, which_lamp=lamp_tag.lower())

        for key in resp_pri.keys():
            f[key] = resp_pri[key]

        f.close()

def test_data_b(
        date,
        fdir_data=_fdir_out_,
        fdir_out=_fdir_out_,
        pitch_angle=0.0,
        roll_angle=0.0,
        ):

    date_s = date.strftime('%Y-%m-%d')

    fname_h5 = '%s/%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), _ssfr_.upper(), date_s)
    f = h5py.File(fname_h5, 'w')

    fname_h5 = '%s/%s_%s_%s_v1.h5' % (fdir_data, _mission_.upper(), _ssfr_.upper(), date_s)
    f_ = h5py.File(fname_h5, 'r')
    tmhr = f_['tmhr'][...]
    for dset_s in f_.keys():

        if 'dset' in dset_s:

            # primary calibration (from pre-mission arcsix in lab on 2023-11-16)
            #/----------------------------------------------------------------------------\#
            wvls = ssfr.lasp_ssfr.get_ssfr_wavelength()
            wvl_start = 350.0
            wvl_end   = 2200.0
            wvl_join  = 950.0

            # zenith wavelength
            #/----------------------------------------------------------------------------\#
            logic_zen_si = (wvls['zen|si'] >= wvl_start) & (wvls['zen|si'] <= wvl_join)
            logic_zen_in = (wvls['zen|in'] >  wvl_join)  & (wvls['zen|in'] <= wvl_end)

            wvl_zen = np.concatenate((wvls['zen|si'][logic_zen_si], wvls['zen|in'][logic_zen_in]))

            indices_sort_zen = np.argsort(wvl_zen)
            wvl_zen = wvl_zen[indices_sort_zen]
            #\----------------------------------------------------------------------------/#

            # nadir wavelength
            #/----------------------------------------------------------------------------\#
            logic_nad_si = (wvls['nad|si'] >= wvl_start) & (wvls['nad|si'] <= wvl_join)
            logic_nad_in = (wvls['nad|in'] >  wvl_join)  & (wvls['nad|in'] <= wvl_end)

            wvl_nad = np.concatenate((wvls['nad|si'][logic_nad_si], wvls['nad|in'][logic_nad_in]))

            indices_sort_nad = np.argsort(wvl_nad)
            wvl_nad = wvl_nad[indices_sort_nad]
            #\----------------------------------------------------------------------------/#

            fnames_zen = sorted(glob.glob('%s/cal/*RAD-CAL-PRI|LASP|%s|ZEN|%s*.h5' % (ssfr.common.fdir_data, _ssfr_.upper(), dset_s.upper())))
            fnames_nad = sorted(glob.glob('%s/cal/*RAD-CAL-PRI|LASP|%s|NAD|%s*.h5' % (ssfr.common.fdir_data, _ssfr_.upper(), dset_s.upper())))
            if len(fnames_zen) == 1 and len(fnames_nad) == 1:
                fname_zen = fnames_zen[0]
                fname_nad = fnames_nad[0]

                f_zen = h5py.File(fname_zen, 'r')
                sec_resp_zen_si = f_zen['zen|si'][...]
                sec_resp_zen_in = f_zen['zen|in'][...]
                f_zen.close()

                f_nad = h5py.File(fname_nad, 'r')
                sec_resp_nad_si = f_nad['nad|si'][...]
                sec_resp_nad_in = f_nad['nad|in'][...]
                f_nad.close()

                sec_resp_zen = np.concatenate((sec_resp_zen_si[logic_zen_si], sec_resp_zen_in[logic_zen_in]))[indices_sort_zen]
                sec_resp_nad = np.concatenate((sec_resp_nad_si[logic_nad_si], sec_resp_nad_in[logic_nad_in]))[indices_sort_nad]
            #\----------------------------------------------------------------------------/#

            # zenith
            #/--------------------------------------------------------------\#
            cnt_zen = f_['%s/cnt_zen' % dset_s][...]
            wvl_zen = f_['%s/wvl_zen' % dset_s][...]

            # sec_resp_zen = np.interp(wvl_zen, wvl_resp_zen_, sec_resp_zen_)

            flux_zen = cnt_zen.copy()
            for i in range(tmhr.size):
                if np.isnan(cnt_zen[i, :]).sum() == 0:
                    flux_zen[i, :] = cnt_zen[i, :] / sec_resp_zen
            #\--------------------------------------------------------------/#

            # nadir
            #/--------------------------------------------------------------\#
            cnt_nad = f_['%s/cnt_nad' % dset_s][...]
            wvl_nad = f_['%s/wvl_nad' % dset_s][...]

            # sec_resp_nad = np.interp(wvl_nad, wvl_resp_nad_, sec_resp_nad_)

            flux_nad = cnt_nad.copy()
            for i in range(tmhr.size):
                if np.isnan(cnt_nad[i, :]).sum() == 0:
                    flux_nad[i, :] = cnt_nad[i, :] / sec_resp_nad
            #\--------------------------------------------------------------/#

            g = f.create_group(dset_s)
            g['flux_zen'] = flux_zen
            g['flux_nad'] = flux_nad
            g['wvl_zen']  = wvl_zen
            g['wvl_nad']  = wvl_nad

        else:

            f[dset_s] = f_[dset_s][...]

    f_.close()

    f.close()

    return
#\----------------------------------------------------------------------------/#

# functions for processing HSK and ALP
#/----------------------------------------------------------------------------\#
def cdata_arcsix_hsk_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    For processing aricraft housekeeping file

    Notes:
        The housekeeping data would require some corrections before its release by the
        data system team, we usually request the raw IWG file (similar data but with a
        slightly different data formatting) from the team right after each flight to
        facilitate our data processing in a timely manner.
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)
    if run:

        # this would change if we are processing IWG file
        #/--------------------------------------------------------------\#
        # fname = ssfr.util.get_all_files(fdir_data, pattern='*%4.4d*%2.2d*%2.2d*.ict' % (date.year, date.month, date.day))[0]
        # data_hsk = ssfr.util.read_ict(fname)
        # var_dict = {
        #         'lon': 'longitude',
        #         'lat': 'latitude',
        #         'alt': 'gps_altitude',
        #         'tmhr': 'tmhr',
        #         'ang_pit': 'pitch_angle',
        #         'ang_rol': 'roll_angle',
        #         'ang_hed': 'true_heading',
        #         }
        try:
            fname = ssfr.util.get_all_files(fdir_data, pattern='*%4.4d*%2.2d*%2.2d*.iwg' % (date.year, date.month, date.day))[0]
            data_hsk = ssfr.util.read_iwg_nsrc(fname)
            var_dict = {
                    'tmhr': 'tmhr',
                    'lon': 'longitude',
                    'lat': 'latitude',
                    'alt': 'gps_alt_msl',
                    'ang_pit': 'pitch_angle',
                    'ang_rol': 'roll_angle',
                    'ang_hed': 'true_heading',
                    }
        except Exception as error:
            print(error)
            fname = ssfr.util.get_all_files(fdir_data, pattern='*%4.4d*%2.2d*%2.2d*.mts' % (date.year, date.month, date.day))[0]
            data_hsk = ssfr.util.read_iwg_mts(fname)
            var_dict = {
                    'tmhr': 'tmhr',
                    'lon': 'longitude',
                    'lat': 'latitude',
                    'alt': 'gps_msl_altitude',
                    'ang_pit': 'pitch',
                    'ang_rol': 'roll',
                    'ang_hed': 'true_heading',
                    }
        print()
        print('Processing HSK file:', fname)
        print()
        #\--------------------------------------------------------------/#

        # fake hsk for skywatch
        #/----------------------------------------------------------------------------\#
        # tmhr = np.arange(tmhr_range[0]*3600.0, tmhr_range[-1]*3600.0, 1.0)/3600.0
        # lon0 = -105.24227862207863 # skywatch longitude
        # lat0 =  40.01097849056196  # skywatch latitude
        # alt0 =  4.0                # skywatch altitude
        # pit0 = 0.0
        # rol0 = 0.0
        # hed0 = 0.0
        # data_hsk = {
        #         'tmhr': {'data': tmhr, 'units': 'hour'},
        #         'long': {'data': np.repeat(lon0, tmhr.size), 'units': 'degree'},
        #         'lat' : {'data': np.repeat(lat0, tmhr.size), 'units': 'degree'},
        #         'palt': {'data': np.repeat(alt0, tmhr.size), 'units': 'meter'},
        #         'pitch'   : {'data': np.repeat(pit0, tmhr.size), 'units': 'degree'},
        #         'roll'    : {'data': np.repeat(rol0, tmhr.size), 'units': 'degree'},
        #         'heading' : {'data': np.repeat(hed0, tmhr.size), 'units': 'degree'},
        #         }
        #\----------------------------------------------------------------------------/#


        # solar geometries
        #/----------------------------------------------------------------------------\#
        jday0 = ssfr.util.dtime_to_jday(date)
        jday  = jday0 + data_hsk[var_dict['tmhr']]['data']/24.0
        sza, saa = ssfr.util.cal_solar_angles(jday, data_hsk[var_dict['lon']]['data'], data_hsk[var_dict['lat']]['data'], data_hsk[var_dict['alt']]['data'])
        #\----------------------------------------------------------------------------/#

        # save processed data
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')
        for var in var_dict.keys():
            f[var] = data_hsk[var_dict[var]]['data']
        f['jday'] = jday
        f['sza']  = sza
        f['saa']  = saa
        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_arcsix_alp_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    v0: directly read raw ALP (Active Leveling Platform) data

    Notes:
        ALP raw data has a finer temporal resolution than 1Hz and a higher measurement
        precision (or sensitivity) of the aircraft attitude.
    """

    date_s = date.strftime('%Y%m%d')

    # read ALP raw data
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _alp_.upper(), _platform_.upper(), date_s)
    if run:
        fnames_alp = ssfr.util.get_all_files(fdir_data, pattern='*.plt3')
        alp0 = ssfr.lasp_alp.read_alp(fnames_alp, date=date)
        alp0.save_h5(fname_h5)
    #\----------------------------------------------------------------------------/#

    return os.path.abspath(fname_h5)

def cdata_arcsix_alp_v1(
        date,
        fname_v0,
        fname_hsk,
        fdir_out=_fdir_out_,
        run=True
        ):

    """
    v1:
    1) calculate time offset (seconds) between aircraft housekeeping data and ALP raw data
       (referencing to aircraft housekeeping)
    2) interpolate raw alp data to aircraft housekeeping time

    Notes:
        ALP raw data has a finer temporal resolution than 1Hz and a higher measurement
        precision (or sensitivity) of the aircraft attitude.
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _alp_.upper(), _platform_.upper(), date_s)

    if run:

        data_hsk = ssfr.util.load_h5(fname_hsk)
        data_alp = ssfr.util.load_h5(fname_v0)

        time_offset = _alp_time_offset_[date_s]


        f = h5py.File(fname_h5, 'w')
        f.attrs['description'] = 'v1:\n  1) raw data interpolated to HSK time frame;\n  2) time offset (seconds) was calculated and applied.'

        f['tmhr']        = data_hsk['tmhr']
        f['jday']        = data_hsk['jday']
        f['tmhr_ori']    = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori']    = data_hsk['jday'] - time_offset/86400.0
        f['time_offset'] = time_offset
        f['sza']         = data_hsk['sza']
        f['saa']         = data_hsk['saa']

        jday_corr        = data_alp['jday'] + time_offset/86400.0
        for vname in data_alp.keys():
            if vname not in ['tmhr', 'jday']:
                f[vname] = ssfr.util.interp(data_hsk['jday'], jday_corr, data_alp[vname], mode='linear')
        f.close()

    return fname_h5
#\----------------------------------------------------------------------------/#

# functions for processing SPNS
#/----------------------------------------------------------------------------\#
def cdata_arcsix_spns_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    Process raw SPN-S data
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)

    if run:

        # read spn-s raw data
        #/----------------------------------------------------------------------------\#
        fname_dif = ssfr.util.get_all_files(fdir_data, pattern='*Diffuse.txt')[-1]
        data0_dif = ssfr.lasp_spn.read_spns(fname=fname_dif)

        fname_tot = ssfr.util.get_all_files(fdir_data, pattern='*Total.txt')[-1]
        data0_tot = ssfr.lasp_spn.read_spns(fname=fname_tot)

        msg = 'Processing %s data:\n%s\n%s\n' % (_spns_.upper(), fname_dif, fname_tot)
        print(msg)
        #/----------------------------------------------------------------------------\#

        # read wavelengths and calculate toa downwelling solar flux
        #/----------------------------------------------------------------------------\#
        flux_toa = ssfr.util.get_solar_kurudz()

        wvl_tot = data0_tot.data['wvl']
        f_dn_sol_tot = np.zeros_like(wvl_tot)
        for i, wvl0 in enumerate(wvl_tot):
            f_dn_sol_tot[i] = ssfr.util.cal_weighted_flux(wvl0, flux_toa[:, 0], flux_toa[:, 1])*ssfr.util.cal_solar_factor(date)
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        g1 = f.create_group('dif')
        for key in data0_dif.data.keys():
            if key in ['tmhr', 'jday', 'wvl', 'flux']:
                dset0 = g1.create_dataset(key, data=data0_dif.data[key], compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        for key in data0_tot.data.keys():
            if key in ['tmhr', 'jday', 'wvl', 'flux']:
                dset0 = g2.create_dataset(key, data=data0_tot.data[key], compression='gzip', compression_opts=9, chunks=True)
        g2['toa0'] = f_dn_sol_tot

        f.close()

    return fname_h5

def cdata_arcsix_spns_v1(
        date,
        fname_spns_v0,
        fname_hsk,
        fdir_out=_fdir_out_,
        time_offset=0.0,
        run=True,
        ):

    """
    Check for time offset and merge SPN-S data with aircraft data
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)

    if run:
        # read spn-s v0
        #/----------------------------------------------------------------------------\#
        data_spns_v0 = ssfr.util.load_h5(fname_spns_v0)
        #/----------------------------------------------------------------------------\#

        # read hsk v0
        #/----------------------------------------------------------------------------\#
        data_hsk= ssfr.util.load_h5(fname_hsk)
        #\----------------------------------------------------------------------------/#

        time_offset = _spns_time_offset_[date_s]

        # interpolate spn-s data to hsk time frame
        #/----------------------------------------------------------------------------\#
        flux_dif = np.zeros((data_hsk['jday'].size, data_spns_v0['dif/wvl'].size), dtype=np.float64)
        for i in range(flux_dif.shape[-1]):
            flux_dif[:, i] = ssfr.util.interp(data_hsk['jday'], data_spns_v0['dif/jday']+time_offset/86400.0, data_spns_v0['dif/flux'][:, i], mode='nearest')

        flux_tot = np.zeros((data_hsk['jday'].size, data_spns_v0['tot/wvl'].size), dtype=np.float64)
        for i in range(flux_tot.shape[-1]):
            flux_tot[:, i] = ssfr.util.interp(data_hsk['jday'], data_spns_v0['tot/jday']+time_offset/86400.0, data_spns_v0['tot/flux'][:, i], mode='nearest')
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        f['time_offset'] = time_offset
        f['tmhr_ori'] = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori'] = data_hsk['jday'] - time_offset/86400.0

        g1 = f.create_group('dif')
        g1['wvl']   = data_spns_v0['dif/wvl']
        dset0 = g1.create_dataset('flux', data=flux_dif, compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        g2['wvl']   = data_spns_v0['tot/wvl']
        g2['toa0']  = data_spns_v0['tot/toa0']
        dset0 = g2.create_dataset('flux', data=flux_tot, compression='gzip', compression_opts=9, chunks=True)

        f.close()

    return fname_h5

def cdata_arcsix_spns_v2(
        date,
        fname_spns_v1,
        fname_hsk, # interchangable with fname_alp_v1
        wvl_range=[350.0, 900.0],
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    Apply attitude correction to account for aircraft attitude (pitch, roll, heading)
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)

    if run:

        # read spn-s v1
        #/----------------------------------------------------------------------------\#
        data_spns_v1 = ssfr.util.load_h5(fname_spns_v1)
        #/----------------------------------------------------------------------------\#

        # read hsk v0
        #/----------------------------------------------------------------------------\#
        data_hsk = ssfr.util.load_h5(fname_hsk)
        #/----------------------------------------------------------------------------\#

        # correction factor
        #/----------------------------------------------------------------------------\#
        mu = np.cos(np.deg2rad(data_hsk['sza']))

        try:
            iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit']+ang_pit_offset, data_hsk['ang_rol']+ang_rol_offset, data_hsk['ang_hed'])
        except Exception as error:
            print(error)
            iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit_s']+ang_pit_offset, data_hsk['ang_rol_s']+ang_rol_offset, data_hsk['ang_hed'])
        dc = ssfr.util.muslope(data_hsk['sza'], data_hsk['saa'], iza, iaa)

        factors = mu / dc
        #\----------------------------------------------------------------------------/#

        # attitude correction
        #/----------------------------------------------------------------------------\#
        f_dn_dir = data_spns_v1['tot/flux'] - data_spns_v1['dif/flux']
        f_dn_dir_corr = np.zeros_like(f_dn_dir)
        f_dn_tot_corr = np.zeros_like(f_dn_dir)
        for iwvl in range(data_spns_v1['tot/wvl'].size):
            f_dn_dir_corr[..., iwvl] = f_dn_dir[..., iwvl]*factors
            f_dn_tot_corr[..., iwvl] = f_dn_dir_corr[..., iwvl] + data_spns_v1['dif/flux'][..., iwvl]
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        g0 = f.create_group('att_corr')
        g0['mu'] = mu
        g0['dc'] = dc
        g0['factors'] = factors

        if wvl_range is None:
            wvl_range = [0.0, 2200.0]

        logic_wvl_dif = (data_spns_v1['dif/wvl']>=wvl_range[0]) & (data_spns_v1['dif/wvl']<=wvl_range[1])
        logic_wvl_tot = (data_spns_v1['tot/wvl']>=wvl_range[0]) & (data_spns_v1['tot/wvl']<=wvl_range[1])

        g1 = f.create_group('dif')
        g1['wvl']   = data_spns_v1['dif/wvl'][logic_wvl_dif]
        dset0 = g1.create_dataset('flux', data=data_spns_v1['dif/flux'][:, logic_wvl_dif], compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        g2['wvl']   = data_spns_v1['tot/wvl'][logic_wvl_tot]
        g2['toa0']  = data_spns_v1['tot/toa0'][logic_wvl_tot]
        dset0 = g2.create_dataset('flux', data=f_dn_tot_corr[:, logic_wvl_tot], compression='gzip', compression_opts=9, chunks=True)

        f.close()

    return fname_h5
#\----------------------------------------------------------------------------/#

# functions for processing SSFR
#/----------------------------------------------------------------------------\#
def cdata_arcsix_ssfr_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        which_ssfr='ssfr-a',
        run=True,
        ):

    """
    version 0: counts after dark correction
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), _platform_.upper(), date_s)

    if run:
        fnames_ssfr = ssfr.util.get_all_files(fdir_data, pattern='*.SKS')

        ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames_ssfr, dark_corr_mode='interp', which_ssfr='lasp|%s' % which_ssfr.lower())

        # data that are useful
        #   wvl_zen [nm]
        #   cnt_zen [counts/ms]
        #   sat_zen: saturation tag, 1 means saturation
        #   wvl_nad [nm]
        #   cnt_nad [counts/ms]
        #   sat_nad: saturation tag, 1 means saturation
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')

        g = f.create_group('raw')
        for key in ssfr0.data_raw.keys():
            if isinstance(ssfr0.data_raw[key], np.ndarray):
                g.create_dataset(key, data=ssfr0.data_raw[key], compression='gzip', compression_opts=9, chunks=True)

        g = f.create_group('spec')
        for key in ssfr0.data_spec.keys():
            if isinstance(ssfr0.data_spec[key], np.ndarray):
                g.create_dataset(key, data=ssfr0.data_spec[key], compression='gzip', compression_opts=9, chunks=True)

        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_arcsix_ssfr_v1(
        date,
        fname_ssfr_v0,
        fname_hsk,
        fdir_out=_fdir_out_,
        time_offset=0.0,
        which_ssfr='ssfr-a',
        run=True,
        ):

    """
    version 1: 1) time adjustment          : check for time offset and merge SSFR data with aircraft housekeeping data
               2) time synchronization     : interpolate raw SSFR data into the time frame of the housekeeping data
               3) counts-to-flux conversion: apply primary and secondary calibration to convert counts to fluxes
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), _platform_.upper(), date_s)

    if run:

        # load ssfr v0 data
        #/----------------------------------------------------------------------------\#
        data_ssfr_v0 = ssfr.util.load_h5(fname_ssfr_v0)
        #\----------------------------------------------------------------------------/#


        # load hsk
        #/----------------------------------------------------------------------------\#
        data_hsk = ssfr.util.load_h5(fname_hsk)
        #\----------------------------------------------------------------------------/#


        # read wavelengths and calculate toa downwelling solar flux
        #/----------------------------------------------------------------------------\#
        flux_toa = ssfr.util.get_solar_kurudz()

        wvl_zen = data_ssfr_v0['spec/wvl_zen']
        f_dn_sol_zen = np.zeros_like(wvl_zen)
        for i, wvl0 in enumerate(wvl_zen):
            f_dn_sol_zen[i] = ssfr.util.cal_weighted_flux(wvl0, flux_toa[:, 0], flux_toa[:, 1])*ssfr.util.cal_solar_factor(date)
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        # processing data - since we have dual integration times, SSFR data with different
        # integration time will be processed seperately
        #/----------------------------------------------------------------------------\#
        jday     = data_ssfr_v0['raw/jday']
        dset_num = data_ssfr_v0['raw/dset_num']

        wvl_zen  = data_ssfr_v0['spec/wvl_zen']
        wvl_nad  = data_ssfr_v0['spec/wvl_nad']

        cnt_zen  = data_ssfr_v0['spec/cnt_zen']
        spec_zen = np.zeros_like(cnt_zen)
        cnt_nad  = data_ssfr_v0['spec/cnt_nad']
        spec_nad = np.zeros_like(cnt_nad)

        for idset in np.unique(dset_num):

            if which_ssfr == 'ssfr-a':
                # select calibration file (can later be adjusted for different integration time sets)
                #/----------------------------------------------------------------------------\#
                fdir_cal = '%s/rad-cal' % _fdir_cal_
                # fname_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*%s*%s*zen*' % (dset_s, which_ssfr.lower())), key=os.path.getmtime)[-1]
                fname_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*%s*zen*' % (which_ssfr.lower())), key=os.path.getmtime)[-1]
                data_cal_zen = ssfr.util.load_h5(fname_cal_zen)

                # fname_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*%s*%s*nad*' % (dset_s, which_ssfr.lower())), key=os.path.getmtime)[-1]
                fname_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*%s*nad*' % (which_ssfr.lower())), key=os.path.getmtime)[-1]
                data_cal_nad = ssfr.util.load_h5(fname_cal_nad)
                #\----------------------------------------------------------------------------/#
            elif which_ssfr == 'ssfr-b':
                factor_zen = (np.nanmax(cnt_zen)-np.nanmin(cnt_zen)) / 2.0
                data_cal_zen = {
                        'sec_resp': np.repeat(factor_zen, wvl_zen.size)
                        }
                factor_nad = (np.nanmax(cnt_nad)-np.nanmin(cnt_nad)) / 2.0
                data_cal_nad = {
                        'sec_resp': np.repeat(factor_nad, wvl_nad.size)
                        }

            logic_dset = (dset_num == idset)

            # convert counts to flux
            #/----------------------------------------------------------------------------\#
            for i in range(wvl_zen.size):
                spec_zen[logic_dset, i] = cnt_zen[logic_dset, i] / data_cal_zen['sec_resp'][i]

            for i in range(wvl_nad.size):
                spec_nad[logic_dset, i] = cnt_nad[logic_dset, i] / data_cal_nad['sec_resp'][i]
            #\----------------------------------------------------------------------------/#

            # set saturation to 0
            #/--------------------------------------------------------------\#
            spec_zen[data_ssfr_v0['spec/sat_zen']==1] = -0.05
            spec_nad[data_ssfr_v0['spec/sat_nad']==1] = -0.05
            #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#


        # check time offset
        #/----------------------------------------------------------------------------\#
        if which_ssfr == 'ssfr-a':
            time_offset = _ssfr1_time_offset_[date_s]
        elif which_ssfr == 'ssfr-b':
            time_offset = _ssfr2_time_offset_[date_s]
        #\----------------------------------------------------------------------------/#


        # interpolate ssfr data to hsk time frame
        # and convert counts to flux
        #/----------------------------------------------------------------------------\#
        cnt_zen_hsk  = np.zeros((data_hsk['jday'].size, wvl_zen.size), dtype=np.float64)
        spec_zen_hsk = np.zeros_like(cnt_zen_hsk)

        for i in range(wvl_zen.size):
            cnt_zen_hsk[:, i]  = ssfr.util.interp(data_hsk['jday'], jday+time_offset/86400.0, cnt_zen[:, i], mode='nearest')
            spec_zen_hsk[:, i] = ssfr.util.interp(data_hsk['jday'], jday+time_offset/86400.0, spec_zen[:, i], mode='nearest')

        cnt_nad_hsk  = np.zeros((data_hsk['jday'].size, wvl_nad.size), dtype=np.float64)
        spec_nad_hsk = np.zeros_like(cnt_nad_hsk)
        for i in range(wvl_nad.size):
            cnt_nad_hsk[:, i]  = ssfr.util.interp(data_hsk['jday'], jday+time_offset/86400.0, cnt_nad[:, i], mode='nearest')
            spec_nad_hsk[:, i] = ssfr.util.interp(data_hsk['jday'], jday+time_offset/86400.0, spec_nad[:, i], mode='nearest')
        #\----------------------------------------------------------------------------/#


        # save processed data
        #/----------------------------------------------------------------------------\#
        g0 = f.create_group('v0')
        g0.create_dataset('jday', data=jday+time_offset/86400.0, compression='gzip', compression_opts=9, chunks=True)
        g0.create_dataset('wvl_zen', data=wvl_zen, compression='gzip', compression_opts=9, chunks=True)
        g0.create_dataset('wvl_nad', data=wvl_nad, compression='gzip', compression_opts=9, chunks=True)
        g0.create_dataset('spec_zen', data=spec_zen, compression='gzip', compression_opts=9, chunks=True)
        g0.create_dataset('spec_nad', data=spec_nad, compression='gzip', compression_opts=9, chunks=True)

        g1 = f.create_group('zen')
        g1.create_dataset('wvl' , data=wvl_zen     , compression='gzip', compression_opts=9, chunks=True)
        g1.create_dataset('cnt' , data=cnt_zen_hsk , compression='gzip', compression_opts=9, chunks=True)
        if which_ssfr == 'ssfr-a':
            g1.create_dataset('flux', data=spec_zen_hsk, compression='gzip', compression_opts=9, chunks=True)
            g1.create_dataset('toa0', data=f_dn_sol_zen, compression='gzip', compression_opts=9, chunks=True)
        elif which_ssfr == 'ssfr-b':
            g1.create_dataset('rad', data=spec_zen_hsk, compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('nad')
        g2.create_dataset('wvl' , data=wvl_nad     , compression='gzip', compression_opts=9, chunks=True)
        g2.create_dataset('cnt' , data=cnt_nad_hsk , compression='gzip', compression_opts=9, chunks=True)
        if which_ssfr == 'ssfr-a':
            g2.create_dataset('flux', data=spec_nad_hsk, compression='gzip', compression_opts=9, chunks=True)
        elif which_ssfr == 'ssfr-b':
            g2.create_dataset('rad', data=spec_nad_hsk, compression='gzip', compression_opts=9, chunks=True)
        #\----------------------------------------------------------------------------/#


        # save processed data
        #/----------------------------------------------------------------------------\#
        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        f['time_offset'] = time_offset
        f['tmhr_ori'] = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori'] = data_hsk['jday'] - time_offset/86400.0

        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_arcsix_ssfr_v2(
        date,
        fname_ssfr_v1,
        fname_alp_v1,
        fname_spns_v2,
        which_ssfr='ssfr-a',
        fdir_out=_fdir_out_,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        run=True,
        run_aux=True,
        ):

    """
    version 2: apply cosine correction to correct for non-linear angular resposne
               diffuse radiation: use cosine response integrated over the entire angular space
               direct radiation: use cosine response over the entire angular space measured in the lab

               diffuse and direct seperation is guided by the diffuse ratio measured by SPNS
    """

    def func_diff_ratio(x, a, b, c):

        return a * (x/500.0)**(b) + c

    def fit_diff_ratio(wavelength, ratio):

        popt, pcov = curve_fit(func_diff_ratio, wavelength, ratio, maxfev=1000000, bounds=(np.array([0.0, -np.inf, 0.0]), np.array([np.inf, 0.0, np.inf])))

        return popt, pcov

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), _platform_.upper(), date_s)

    if run:

        data_ssfr_v1 = ssfr.util.load_h5(fname_ssfr_v1)

        fname_aux = '%s/%s-%s_%s_%s_v2-aux.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), _platform_.upper(), date_s)

        if run_aux:

            # calculate diffuse/global ratio from SPNS data
            #/----------------------------------------------------------------------------\#
            data_spns_v2 = ssfr.util.load_h5(fname_spns_v2)

            f_ = h5py.File(fname_aux, 'w')

            wvl_ssfr_zen = data_ssfr_v1['zen/wvl']
            wvl_spns     = data_spns_v2['tot/wvl']

            Nt, Nwvl = data_ssfr_v1['zen/flux'].shape

            diff_ratio = np.zeros((Nt, Nwvl), dtype=np.float64)
            diff_ratio[...] = np.nan

            poly_coefs = np.zeros((Nt, 3), dtype=np.float64)
            poly_coefs[...] = np.nan

            qual_flag = np.repeat(0, Nt)

            # do spectral fit based on 400 nm - 750 nm observations
            #/--------------------------------------------------------------\#
            for i in tqdm(range(Nt)):

                diff_ratio0_spns = data_spns_v2['dif/flux'][i, :] / data_spns_v2['tot/flux'][i, :]
                logic_valid = (~np.isnan(diff_ratio0_spns)) & (diff_ratio0_spns>=0.0) & (diff_ratio0_spns<=1.0) & (wvl_spns>=400.0) & (wvl_spns<=750.0)
                if logic_valid.sum() > 20:

                    x = data_spns_v2['tot/wvl'][logic_valid]
                    y = diff_ratio0_spns[logic_valid]
                    popt, pcov = fit_diff_ratio(x, y)

                    diff_ratio[i, :] = func_diff_ratio(wvl_ssfr_zen, *popt)
                    poly_coefs[i, :] = popt

                    qual_flag[i] = 1

            diff_ratio[diff_ratio<0.0] = 0.0
            diff_ratio[diff_ratio>1.0] = 1.0
            #\--------------------------------------------------------------/#

            # fill in nan values in time space
            #/--------------------------------------------------------------\#
            for i in range(Nwvl):

                logic_nan   = np.isnan(diff_ratio[:, i])
                logic_valid = ~logic_nan
                f_interp = interpolate.interp1d(data_ssfr_v1['tmhr'][logic_valid], diff_ratio[:, i][logic_valid], bounds_error=None, fill_value='extrapolate')
                diff_ratio[logic_nan, i] = f_interp(data_ssfr_v1['tmhr'][logic_nan])

            diff_ratio[diff_ratio<0.0] = 0.0
            diff_ratio[diff_ratio>1.0] = 1.0
            #\--------------------------------------------------------------/#

            # save data
            #/--------------------------------------------------------------\#
            f_.create_dataset('diff_ratio', data=diff_ratio  , compression='gzip', compression_opts=9, chunks=True)
            g_ = f_.create_group('diff_ratio_aux')
            g_.create_dataset('wvl'       , data=wvl_ssfr_zen, compression='gzip', compression_opts=9, chunks=True)
            g_.create_dataset('coef'      , data=poly_coefs  , compression='gzip', compression_opts=9, chunks=True)
            g_.create_dataset('qual_flag' , data=qual_flag   , compression='gzip', compression_opts=9, chunks=True)
            #\--------------------------------------------------------------/#
            #\----------------------------------------------------------------------------/#


            # alp
            #/----------------------------------------------------------------------------\#
            data_alp_v1  = ssfr.util.load_h5(fname_alp_v1)
            for key in data_alp_v1.keys():
                try:
                    f_.create_dataset(key, data=data_alp_v1[key], compression='gzip', compression_opts=9, chunks=True)
                except TypeError as error:
                    print(error)
                    f_[key] = data_alp_v1[key]
            f_.close()
            #\----------------------------------------------------------------------------/#

        # aux data processing ends here

        data_aux = ssfr.util.load_h5(fname_aux)

        # diffuse ratio
        #/----------------------------------------------------------------------------\#
        diff_ratio = data_aux['diff_ratio']
        #\----------------------------------------------------------------------------/#


        # angles
        #/----------------------------------------------------------------------------\#
        angles = {}
        angles['sza'] = data_aux['sza']
        angles['saa'] = data_aux['saa']
        angles['ang_pit']   = data_aux['ang_pit_s'] # pitch angle from SPAN-CPT
        angles['ang_rol']   = data_aux['ang_rol_s'] # roll angle from SPAN-CPT
        angles['ang_hed']   = data_aux['ang_hed']
        angles['ang_pit_m'] = data_aux['ang_pit_m']
        angles['ang_rol_m'] = data_aux['ang_rol_m']
        angles['ang_pit_offset'] = ang_pit_offset
        angles['ang_rol_offset'] = ang_rol_offset
        #\----------------------------------------------------------------------------/#


        # select calibration file for attitude correction
        #/----------------------------------------------------------------------------\#
        dset_s = 'dset1'
        fdir_cal = '%s/ang-cal' % _fdir_cal_
        fname_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*vaa-180|*%s*%s*zen*' % (dset_s, which_ssfr.lower())), key=os.path.getmtime)[-1]
        fname_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*vaa-180|*%s*%s*nad*' % (dset_s, which_ssfr.lower())), key=os.path.getmtime)[-1]
        #\----------------------------------------------------------------------------/#


        # calculate attitude correction factors
        #/----------------------------------------------------------------------------\#
        fnames_cal = {
                'zen': fname_cal_zen,
                'nad': fname_cal_nad,
                }
        factors = ssfr.corr.att_corr(fnames_cal, angles, diff_ratio=diff_ratio)
        #\----------------------------------------------------------------------------/#

        # save data
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')
        for key in ['tmhr', 'jday', 'lon', 'lat', 'alt']:
            f.create_dataset(key, data=data_aux[key], compression='gzip', compression_opts=9, chunks=True)

        g1 = f.create_group('att_corr')
        g1.create_dataset('factors_zen', data=factors['zen'], compression='gzip', compression_opts=9, chunks=True)
        g1.create_dataset('factors_nad', data=factors['nad'], compression='gzip', compression_opts=9, chunks=True)
        for key in ['sza', 'saa', 'ang_pit_s', 'ang_rol_s', 'ang_hed', 'ang_pit_m', 'ang_rol_m']:
            g1.create_dataset(key, data=data_aux[key], compression='gzip', compression_opts=9, chunks=True)

        # apply attitude correction
        #/----------------------------------------------------------------------------\#
        g2 = f.create_group('zen')
        g2.create_dataset('flux', data=data_ssfr_v1['zen/flux']*factors['zen'], compression='gzip', compression_opts=9, chunks=True)
        g2.create_dataset('wvl' , data=data_ssfr_v1['zen/wvl']                , compression='gzip', compression_opts=9, chunks=True)
        g2.create_dataset('toa0', data=data_ssfr_v1['zen/toa0']               , compression='gzip', compression_opts=9, chunks=True)

        g3 = f.create_group('nad')
        g3.create_dataset('flux', data=data_ssfr_v1['nad/flux']*factors['nad'], compression='gzip', compression_opts=9, chunks=True)
        g3.create_dataset('wvl' , data=data_ssfr_v1['nad/wvl']                , compression='gzip', compression_opts=9, chunks=True)
        #\----------------------------------------------------------------------------/#

        f.close()

    return fname_h5

def cdata_arcsix_ssfr_archive():

    # header
    #/----------------------------------------------------------------------------\#
    # comments_list = []
    # comments_list.append('Bandwidth of Silicon channels (wavelength < 950nm) as defined by the FWHM: 6 nm')
    # comments_list.append('Bandwidth of InGaAs channels (wavelength > 950nm) as defined by the FWHM: 12 nm')
    # comments_list.append('Pitch angle offset: %.1f degree' % pitch_angle)
    # comments_list.append('Roll angle offset: %.1f degree' % roll_angle)

    # for key in fnames_rad_cal.keys():
    #     comments_list.append('Radiometric calibration file (%s): %s' % (key, os.path.basename(fnames_rad_cal[key])))
    # for key in fnames_ang_cal.keys():
    #     comments_list.append('Angular calibration file (%s): %s' % (key, os.path.basename(fnames_ang_cal[key])))
    # comments = '\n'.join(comments_list)

    # print(date_s)
    # print(comments)
    # print()
    #\----------------------------------------------------------------------------/#

    # create hsk file for ssfr (nasa data archive)
    #/----------------------------------------------------------------------------\#
    fname_ssfr = '%s/ssfr_%s_hsk.h5' % (fdir_processed, date_s)
    # f = h5py.File(fname_ssfr, 'w')

    # dset = f.create_dataset('comments', data=comments)
    # dset.attrs['description'] = 'comments on the data'

    # dset = f.create_dataset('info', data=version_info)
    # dset.attrs['description'] = 'information on the version'

    # dset = f.create_dataset('utc', data=data_hsk['tmhr'])
    # dset.attrs['description'] = 'universal time (numbers above 24 are for the next day)'
    # dset.attrs['unit'] = 'decimal hour'

    # dset = f.create_dataset('altitude', data=data_hsk['gps_altitude'])
    # dset.attrs['description'] = 'altitude above sea level (GPS altitude)'
    # dset.attrs['unit'] = 'meter'

    # dset = f.create_dataset('longitude', data=data_hsk['longitude'])
    # dset.attrs['description'] = 'longitude'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('latitude', data=data_hsk['latitude'])
    # dset.attrs['description'] = 'latitude'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('zen_wvl', data=ssfr_v0.zen_wvl)
    # dset.attrs['description'] = 'center wavelengths of zenith channels (bandwidth see info)'
    # dset.attrs['unit'] = 'nm'

    # dset = f.create_dataset('nad_wvl', data=ssfr_v0.nad_wvl)
    # dset.attrs['description'] = 'center wavelengths of nadir channels (bandwidth see info)'
    # dset.attrs['unit'] = 'nm'

    # dset = f.create_dataset('zen_flux', data=zen_flux)
    # dset.attrs['description'] = 'downwelling shortwave spectral irradiance'
    # dset.attrs['unit'] = 'W / m2 / nm'

    # dset = f.create_dataset('nad_flux', data=nad_flux)
    # dset.attrs['description'] = 'upwelling shortwave spectral irradiance'
    # dset.attrs['unit'] = 'W / m2 / nm'

    # dset = f.create_dataset('pitch', data=pitch)
    # dset.attrs['description'] = 'aircraft pitch angle (positive values indicate nose up)'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('roll', data=roll)
    # dset.attrs['description'] = 'aircraft roll angle (positive values indicate right wing down)'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('heading', data=heading)
    # dset.attrs['description'] = 'aircraft heading angle (positive values clockwise, w.r.t north)'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('sza', data=sza)
    # dset.attrs['description'] = 'solar zenith angle'
    # dset.attrs['unit'] = 'degree'

    # f.close()
    #\----------------------------------------------------------------------------/#

    return fname_ssfr
#\----------------------------------------------------------------------------/#


def run_time_offset_check(date):

    date_s = date.strftime('%Y%m%d')
    data_hsk = ssfr.util.load_h5(_fnames_['%s_hsk_v0' % date_s])
    data_alp = ssfr.util.load_h5(_fnames_['%s_alp_v0' % date_s])
    data_spns_v0 = ssfr.util.load_h5(_fnames_['%s_spns_v0' % date_s])
    data_ssfr1_v0 = ssfr.util.load_h5(_fnames_['%s_ssfr1_v0' % date_s])
    data_ssfr2_v0 = ssfr.util.load_h5(_fnames_['%s_ssfr2_v0' % date_s])

    # data_spns_v0['tot/jday'] += 1.0
    # data_spns_v0['dif/jday'] += 1.0

    # _offset_x_range_ = [-6000.0, 6000.0]
    _offset_x_range_ = [-300.0, 300.0]

    # ALP pitch vs HSK pitch
    #/----------------------------------------------------------------------------\#
    data_offset = {
            'x0': data_hsk['jday']*86400.0,
            'y0': data_hsk['ang_pit'],
            'x1': data_alp['jday'][::10]*86400.0,
            'y1': data_alp['ang_pit_s'][::10],
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=False,
            description='ALP Pitch vs. HSK Pitch',
            fname_html='alp-pit_offset_check_%s.html' % date_s)
    #\----------------------------------------------------------------------------/#

    # ALP roll vs HSK roll
    #/----------------------------------------------------------------------------\#
    data_offset = {
            'x0': data_hsk['jday']*86400.0,
            'y0': data_hsk['ang_rol'],
            'x1': data_alp['jday'][::10]*86400.0,
            'y1': data_alp['ang_rol_s'][::10],
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=False,
            description='ALP Roll vs. HSK Roll',
            fname_html='alp-rol_offset_check_%s.html' % date_s)
    #\----------------------------------------------------------------------------/#

    # ALP altitude vs HSK altitude
    #/----------------------------------------------------------------------------\#
    data_offset = {
            'x0': data_hsk['jday']*86400.0,
            'y0': data_hsk['alt'],
            'x1': data_alp['jday'][::10]*86400.0,
            'y1': data_alp['alt'][::10],
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='ALP Altitude vs. HSK Altitude',
            fname_html='alp-alt_offset_check_%s.html' % date_s)
    #\----------------------------------------------------------------------------/#


    # SPNS vs TOA
    #/----------------------------------------------------------------------------\#
    index_wvl = np.argmin(np.abs(745.0-data_spns_v0['tot/wvl']))
    data_y1   = data_spns_v0['tot/flux'][:, index_wvl]

    mu = np.cos(np.deg2rad(data_hsk['sza']))
    iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit'], data_hsk['ang_rol'], data_hsk['ang_hed'])
    dc = ssfr.util.muslope(data_hsk['sza'], data_hsk['saa'], iza, iaa)
    factors = mu/dc
    data_y0   = data_spns_v0['tot/toa0'][index_wvl]*np.cos(np.deg2rad(data_hsk['sza']))/factors

    data_offset = {
            'x0': data_hsk['jday']*86400.0,
            'y0': data_y0,
            'x1': data_spns_v0['tot/jday']*86400.0,
            'y1': data_y1,
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='SPNS Total vs. TOA (745 nm)',
            fname_html='spns-toa_offset_check_%s.html' % date_s)
    #\----------------------------------------------------------------------------/#


    # SSFR-A vs SPNS
    #/----------------------------------------------------------------------------\#
    index_wvl_spns = np.argmin(np.abs(745.0-data_spns_v0['tot/wvl']))
    data_y0 = data_spns_v0['tot/flux'][:, index_wvl_spns]

    index_wvl_ssfr = np.argmin(np.abs(745.0-data_ssfr1_v0['spec/wvl_zen']))
    data_y1 = data_ssfr1_v0['spec/cnt_zen'][:, index_wvl_ssfr]
    data_offset = {
            'x0': data_spns_v0['tot/jday']*86400.0,
            'y0': data_y0,
            'x1': data_ssfr1_v0['raw/jday']*86400.0,
            'y1': data_y1,
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='SSFR-A Zenith Count vs. SPNS Total (745nm)',
            fname_html='ssfr-a_offset_check_%s.html' % (date_s))
    #\----------------------------------------------------------------------------/#

    # SSFR-B vs SSFR-A
    #/----------------------------------------------------------------------------\#
    index_wvl_ssfr = np.argmin(np.abs(745.0-data_ssfr1_v0['spec/wvl_nad']))
    data_y0 = data_ssfr1_v0['spec/cnt_nad'][:, index_wvl_ssfr]

    index_wvl_ssfr = np.argmin(np.abs(745.0-data_ssfr2_v0['spec/wvl_nad']))
    data_y1 = data_ssfr2_v0['spec/cnt_nad'][:, index_wvl_ssfr]
    data_offset = {
            'x0': data_ssfr1_v0['raw/jday']*86400.0,
            'y0': data_y0,
            'x1': data_ssfr2_v0['raw/jday']*86400.0,
            'y1': data_y1,
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='SSFR-B Nadir Count vs. SSFR-A Nadir (745nm)',
            fname_html='ssfr-b_offset_check_%s.html' % (date_s))
    #\----------------------------------------------------------------------------/#


def run_angle_offset_check(
        date,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        wvl0=745.0,
        ):

    date_s = date.strftime('%Y%m%d')
    data_hsk = ssfr.util.load_h5(_fnames_['%s_hsk_v0' % date_s])


    # SPNS v1
    #/----------------------------------------------------------------------------\#
    data_spns_v1 = ssfr.util.load_h5(_fnames_['%s_spns_v1' % date_s])
    index_wvl_spns = np.argmin(np.abs(wvl0-data_spns_v1['tot/wvl']))
    data_y1 = data_spns_v1['tot/flux'][:, index_wvl_spns]
    #\----------------------------------------------------------------------------/#


    # SPNS v2
    #/----------------------------------------------------------------------------\#
    fname_spns_v2 = cdata_arcsix_spns_v2(date, _fnames_['%s_spns_v1' % date_s], _fnames_['%s_hsk_v0' % date_s],
            fdir_out=_fdir_out_,
            run=True,
            ang_pit_offset=ang_pit_offset,
            ang_rol_offset=ang_rol_offset,
            )
    data_spns_v2 = ssfr.util.load_h5(_fnames_['%s_spns_v2' % date_s])
    data_y2 = data_spns_v2['tot/flux'][:, index_wvl_spns]
    #\----------------------------------------------------------------------------/#


    # SSFR-A v2
    #/----------------------------------------------------------------------------\#
    data_ssfr1_v2 = ssfr.util.load_h5(_fnames_['%s_ssfr1_v2' % date_s])
    index_wvl_ssfr = np.argmin(np.abs(wvl0-data_ssfr1_v2['zen/wvl']))
    data_y0 = data_ssfr1_v2['zen/flux'][:, index_wvl_ssfr]
    #\----------------------------------------------------------------------------/#

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(15, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        ax1.scatter(data_hsk['tmhr'], data_y1, s=3, c='r', lw=0.0)
        ax1.scatter(data_hsk['tmhr'], data_y0, s=3, c='k', lw=0.0)
        ax1.scatter(data_hsk['tmhr'], data_y2, s=3, c='g', lw=0.0)
        # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
        # ax1.plot([0, 1], [0, 1], color='k', ls='--')
        # ax1.set_xlim(())
        # ax1.set_ylim(())
        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#





# main program
#/----------------------------------------------------------------------------\#
def main_process_data_v0(date, run=True):

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    date_s = date.strftime('%Y%m%d')

    # HSK v0: raw data
    #/----------------------------------------------------------------------------\#
    fname_hsk_v0 = cdata_arcsix_hsk_v0(date, fdir_data=_fdir_hsk_,
            fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#

    # ALP v0: raw data
    #/----------------------------------------------------------------------------\#
    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*%s' % (date.year, date.month, date.day, _alp_))
    fdir_data_alp = sorted(fdirs, key=os.path.getmtime)[-1]
    fname_alp_v0 = cdata_arcsix_alp_v0(date, fdir_data=fdir_data_alp,
            fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#

    # SPNS v0: raw data
    #/----------------------------------------------------------------------------\#
    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*%s' % (date.year, date.month, date.day, _spns_))
    fdir_data_spns = sorted(fdirs, key=os.path.getmtime)[-1]
    fname_spns_v0 = cdata_arcsix_spns_v0(date, fdir_data=fdir_data_spns,
            fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#

    # SSFR-A v0: raw data
    #/----------------------------------------------------------------------------\#
    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*%s' % (date.year, date.month, date.day, _ssfr1_))
    fdir_data_ssfr1 = sorted(fdirs, key=os.path.getmtime)[-1]
    fname_ssfr1_v0 = cdata_arcsix_ssfr_v0(date, fdir_data=fdir_data_ssfr1,
            which_ssfr='ssfr-a', fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#

    # SSFR-B v0: raw data
    #/----------------------------------------------------------------------------\#
    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*%s' % (date.year, date.month, date.day, _ssfr2_))
    fdir_data_ssfr2 = sorted(fdirs, key=os.path.getmtime)[-1]
    fname_ssfr2_v0 = cdata_arcsix_ssfr_v0(date, fdir_data=fdir_data_ssfr2,
            which_ssfr='ssfr-b', fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#

    _fnames_['%s_hsk_v0' % date_s]   = fname_hsk_v0
    _fnames_['%s_alp_v0' % date_s]   = fname_alp_v0
    _fnames_['%s_spns_v0' % date_s]  = fname_spns_v0
    _fnames_['%s_ssfr1_v0' % date_s] = fname_ssfr1_v0
    _fnames_['%s_ssfr2_v0' % date_s] = fname_ssfr2_v0

def main_process_data_v1(date, run=True):

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    date_s = date.strftime('%Y%m%d')

    # ALP v1: time synced with hsk time with time offset applied
    #/----------------------------------------------------------------------------\#
    fname_alp_v1 = cdata_arcsix_alp_v1(date, _fnames_['%s_alp_v0' % date_s], _fnames_['%s_hsk_v0' % date_s],
            fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#

    # SPNS v1: time synced with hsk time with time offset applied
    #/----------------------------------------------------------------------------\#
    fname_spns_v1 = cdata_arcsix_spns_v1(date, _fnames_['%s_spns_v0' % date_s], _fnames_['%s_hsk_v0' % date_s],
            fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#

    # SSFR-A v1: time synced with hsk time with time offset applied
    #/----------------------------------------------------------------------------\#
    fname_ssfr1_v1 = cdata_arcsix_ssfr_v1(date, _fnames_['%s_ssfr1_v0' % date_s], _fnames_['%s_hsk_v0' % date_s],
            which_ssfr='ssfr-a', fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#

    # SSFR-B v1: time synced with hsk time with time offset applied
    #/----------------------------------------------------------------------------\#
    fname_ssfr2_v1 = cdata_arcsix_ssfr_v1(date, _fnames_['%s_ssfr2_v0' % date_s], _fnames_['%s_hsk_v0' % date_s],
            which_ssfr='ssfr-b', fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#

    _fnames_['%s_alp_v1'   % date_s] = fname_alp_v1
    _fnames_['%s_spns_v1'  % date_s] = fname_spns_v1
    _fnames_['%s_ssfr1_v1' % date_s] = fname_ssfr1_v1
    _fnames_['%s_ssfr2_v1' % date_s] = fname_ssfr2_v1

def main_process_data_v2(date, run=True):

    """
    v0: raw data directly read out from the data files
    v1: data collocated/synced to aircraft nav
    v2: attitude corrected data
    """

    date_s = date.strftime('%Y%m%d')

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # SPNS v2
    #/----------------------------------------------------------------------------\#
    # based on ALP v1
    # fname_spns_v2 = cdata_arcsix_spns_v2(date, _fnames_['%s_spns_v1' % date_s], _fnames_['%s_alp_v1' % date_s],
    #         fdir_out=fdir_out, run=run)
    # fname_spns_v2 = cdata_arcsix_spns_v2(date, _fnames_['%s_spns_v1' % date_s], _fnames_['%s_alp_v1' % date_s],
    #         fdir_out=fdir_out, run=True)

    # based on HSK v0
    fname_spns_v2 = cdata_arcsix_spns_v2(date, _fnames_['%s_spns_v1' % date_s], _fnames_['%s_hsk_v0' % date_s],
            fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#
    _fnames_['%s_spns_v2' % date_s] = fname_spns_v2


    # SSFR v2
    #/----------------------------------------------------------------------------\#
    fname_ssfr1_v2 = cdata_arcsix_ssfr_v2(date, _fnames_['%s_ssfr1_v1' % date_s], _fnames_['%s_alp_v1' % date_s], _fnames_['%s_spns_v2' % date_s],
            which_ssfr='ssfr-a', fdir_out=fdir_out, run=run, run_aux=True)
    #\----------------------------------------------------------------------------/#
    _fnames_['%s_ssfr1_v2' % date_s] = fname_ssfr1_v2
#\----------------------------------------------------------------------------/#




def plot_spiral_old(
        date,
        tmhr_range=[15.15, 15.54],
        wvl0=550.0
        ):

    tmhr_range=[14.5861, 14.8833]

    date_s     = date.strftime('%Y%m%d')

    tmhr_range_legs = [[14.5806, 14.6236], [14.6236, 14.6639], [14.6639, 14.7014], [14.7014, 14.7528], [14.7528, 14.7944], [14.7944, 14.8417], [14.8417, 14.8861]]

    # read hsk data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = ssfr.util.load_h5(fname_hsk)
    logic = (data_hsk['tmhr']>tmhr_range[0]) & (data_hsk['tmhr']<tmhr_range[1]) & \
            (np.abs(data_hsk['ang_pit'])<=3.0) & (np.abs(data_hsk['ang_rol'])<=3.0)

    alt = data_hsk['alt'][logic]/1000.0
    #\----------------------------------------------------------------------------/#


    # read SSFR v2
    #/----------------------------------------------------------------------------\#
    fname_ssfr1_ra = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr1_v2 = ssfr.util.load_h5(fname_ssfr1_ra)
    index_wvl_ssfr = np.argmin(np.abs(wvl0-data_ssfr1_v2['zen/wvl']))
    f_dn = data_ssfr1_v2['zen/flux'][logic, index_wvl_ssfr]
    index_wvl_ssfr = np.argmin(np.abs(wvl0-data_ssfr1_v2['nad/wvl']))
    f_up = data_ssfr1_v2['nad/flux'][logic, index_wvl_ssfr]

    # f_net = f_dn - f_up
    #\----------------------------------------------------------------------------/#


    # SPNS v2
    #/----------------------------------------------------------------------------\#
    # data_spns_v2 = ssfr.util.load_h5(_fnames_['%s_spns_v2' % date_s])
    # index_wvl_spns = np.argmin(np.abs(wvl0-data_spns_v2['tot/wvl']))
    # f_dn_tot = data_spns_v2['tot/flux'][logic, index_wvl_spns]
    # f_dn_dif = data_spns_v2['dif/flux'][logic, index_wvl_spns]
    #\----------------------------------------------------------------------------/#


    #/----------------------------------------------------------------------------\#
    rcParams['font.size'] = 20
    fig = plt.figure(figsize=(6, 12))
    ax1 = fig.add_subplot(111)
    ax1.scatter(f_up, alt, s=5, c='r')
    ax1.scatter(f_dn, alt, s=5, c='b')
    # ax1.scatter(f_net, alt, s=5, c='k')
    # ax1.scatter(f_dn_dif, alt, s=5, c='g')
    # ax1.scatter(f_dn_tot, alt, s=5, c='cyan')
    ax1.axvspan(-0.1, 0.0, color='r', lw=0.0, alpha=0.1)
    ax1.set_xlim((-0.1, 1.0))
    ax1.set_ylim((0.0, 6.0))
    ax1.set_xlabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_ylabel('Altitude [km]')
    ax1.set_title('Spiral on %s (%.1f nm)' % (date.strftime('%Y-%m-%d'), wvl0))

    ax1.grid()

    patches_legend = [
                mpatches.Patch(color='red'   , label='SSFR $F_\\uparrow$'),
                mpatches.Patch(color='blue'  , label='SSFR $F_\\downarrow$'),
                # mpatches.Patch(color='black' , label='SSFR $F_{net}$'),
                # mpatches.Patch(color='green' , label='SPN-S $F_\\downarrow^{dif}$'),
                # mpatches.Patch(color='cyan'  , label='SPN-S $F_\\downarrow^{tot}$')
                ]
    ax1.legend(handles=patches_legend, loc='upper center', fontsize=14)

    plt.savefig('%s_spiral.png' % date_s, bbox_inches='tight')
    plt.show()
    #\----------------------------------------------------------------------------/#




def plot_spiral(
        date,
        tmhr_range=[15.15, 15.54],
        wvl0=550.0
        ):

    tmhr_range=[14.5861, 14.8833]

    date_s     = date.strftime('%Y%m%d')

    tmhr_range_legs = [[14.5806, 14.6236], [14.6236, 14.6639], [14.6639, 14.7014], [14.7014, 14.7528], [14.7528, 14.7944], [14.7944, 14.8417], [14.8417, 14.8861]]

    # read hsk data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = ssfr.util.load_h5(fname_hsk)
    alt = data_hsk['alt']/1000.0

    alt_cmap = mpl.colormaps['jet']
    alt_norm = mpl.colors.Normalize(vmin=0.0, vmax=5.0)
    #\----------------------------------------------------------------------------/#

    # read SSFR
    #/----------------------------------------------------------------------------\#
    fname_ssfr1_ra = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr1_ra = ssfr.util.load_h5(fname_ssfr1_ra)
    #\----------------------------------------------------------------------------/#

    wvl = data_ssfr1_ra['zen/wvl'].copy()
    absorb_bands = [[649.0, 662.0], [676.0, 697.0], [707.0, 729.0], [743.0, 779.0], [796.0, 830.0], [911.0, 976.0], [1108.0, 1193.0], [1262.0, 1298.0], [1324.0, 1470.0], [1800.0, 1960.0]]
    logic = np.repeat(True, wvl.size)

    for absorb_band in absorb_bands:
        wvl_L, wvl_R = absorb_band
        logic0 = (wvl>=wvl_L)&(wvl<=wvl_R)
        logic[logic0] = False
    wvl[~logic] = np.nan

    rcParams['font.size'] = 24
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(111)

    for absorb_band in absorb_bands:
        wvl_L, wvl_R = absorb_band
        ax1.axvspan(wvl_L, wvl_R, color='red', lw=0.0, alpha=0.1, zorder=0)

    for tmhr_range in tmhr_range_legs:
        logic = (data_hsk['tmhr']>tmhr_range[0]) & (data_hsk['tmhr']<tmhr_range[1]) & \
                (np.abs(data_hsk['ang_pit'])<=2.0) & (np.abs(data_hsk['ang_rol'])<=2.0)

        f_dn = data_ssfr1_ra['zen/flux'][logic, :]
        f_up = data_ssfr1_ra['nad/flux'][logic, 1:]

        alb_all = f_up/f_dn
        alb_mean = np.nanmean(alb_all, axis=0)
        alb_std  = np.nanstd(alb_all, axis=0)

        color=alt_cmap(alt_norm(np.mean(alt[logic])))
        ax1.scatter(data_ssfr1_ra['zen/wvl'], alb_mean, alpha=0.3, color=color, s=3, lw=0.0, zorder=0)
        ax1.errorbar(wvl, alb_mean, yerr=alb_std, color=color, zorder=1)


    # arise albedo
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File('Albedo_All.h5', 'r')
    arise_wvl = f0['wvl_arise'][...]
    arise_slope = f0['slope_arise'][...]
    arise_intercept = f0['intercept_arise'][...]
    arise_albedo = arise_slope*1.0 + arise_intercept
    f0.close()
    ax1.scatter(arise_wvl, arise_albedo, color='k', label="ARISE SSFR (2014-09-11)", marker='o', ec='none', s=8, zorder=0, alpha=0.6)
    #\----------------------------------------------------------------------------/#


    # Lyapustin albedo
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File('Albedo_All.h5', 'r')
    lya_wvl = f0['wvl_fresh_snow_2010'][...]
    lya_alb = f0['albedo_fresh_snow_2010'][...]
    f0.close()
    ax1.plot(lya_wvl, lya_alb, color='orange', label="Fresh snow (Lyapustin et al. 2010)", marker='o', markeredgecolor='none', markersize=10, lw=0.5, zorder=4, alpha=0.8)
    #\----------------------------------------------------------------------------/#


    # brandt and warren 2005
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File('Albedo_All.h5', 'r')
    brandt_wvl = f0['wvl_ice_thick_snow_2005'][...]
    brandt_alb = f0['albedo_ice_thick_snow_2005'][...]
    f0.close()
    ax1.plot(brandt_wvl, brandt_alb, color='lime', label="Ice+deep snow (Brandt et al. 2005)", marker='o', markeredgecolor='none', markersize=10, lw=0.5, zorder=4, alpha=0.8)
    #\----------------------------------------------------------------------------/#


    # kay albedo
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File('Albedo_All.h5', 'r')
    jen_wvl = f0['wvl_dry_season_2013'][...]
    jen_wvl = f0['wvl_wet_season_2013'][...]
    a_dry = f0['albedo_dry_season_2013'][...]
    a_wet = f0['albedo_wet_season_2013'][...]
    f0.close()
    ax1.plot(jen_wvl, a_dry, color='gray', label="Dry season (Kay and L'Ecuyer 2013)", marker='o', markeredgecolor='none', markersize=10, lw=0.5, zorder=4, alpha=0.8)
    ax1.plot(jen_wvl, a_wet, color='gray', label="Wet season (Kay and L'Ecuyer 2013)", marker='s', markeredgecolor='none', markersize=10, lw=0.5, zorder=5, alpha=0.8)
    #\----------------------------------------------------------------------------/#


    cs = ax1.scatter(-1, -1, c=1, cmap='jet', vmin=0.0, vmax=5.0)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', '3%', pad='3%')
    cbar = fig.colorbar(cs, cax=cax)
    cbar.set_label('Altitude [km]', rotation=270, labelpad=22.0)

    ax1.xaxis.set_major_locator(FixedLocator(np.arange(200, 2201, 200)))
    ax1.set_xlim((200, 2200))
    ax1.set_ylim((0.0, 1.0))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Albedo')
    ax1.set_title('Square Spiral on %s' % (date.strftime('%Y-%m-%d')), y=1.02)

    # patches_legend = [
    #             mpatches.Patch(color='red'   , label='SSFR $F_\\uparrow$'),
    #             mpatches.Patch(color='blue'  , label='SSFR $F_\\downarrow$'),
    #             # mpatches.Patch(color='black' , label='SSFR $F_{net}$'),
    #             # mpatches.Patch(color='green' , label='SPN-S $F_\\downarrow^{dif}$'),
    #             # mpatches.Patch(color='cyan'  , label='SPN-S $F_\\downarrow^{tot}$')
    #             ]
    # ax1.legend(handles=patches_legend, loc='upper center', fontsize=14)

    ax1.legend(fontsize=16, framealpha=0.1, loc='upper right')

    plt.savefig('%s_spiral.png' % date_s, bbox_inches='tight')
    plt.show()
    #\----------------------------------------------------------------------------/#




def PLOT_ALBEDO():


    f = h5py.File('data/frac_info_20140913.h5', 'r')
    frac = f['frac'][...]
    logic_frac = f['logic'][...]
    f.close()

    f = h5py.File('data/ssfr_info_02_20140913.h5', 'r')
    tmhr = f['tmhr'][...]
    logic_ssfr = f['logic'][...]
    f_up = f['f_up'][...]
    f_dn = f['f_dn'][...]
    wvl  = f['wvl'][...]
    f.close()

    f = h5py.File('data/alp_info_20140913.h5', 'r')
    alt  = f['alt'][...]
    f.close()

    f0 = h5py.File('Albedo_All.h5', 'w')


    albedo = f_up/f_dn

    logic = logic_frac & logic_ssfr

    slope = np.zeros(wvl.size)
    intercept = np.zeros(wvl.size)

    frac_all   = np.zeros((frac.size, wvl.size))
    albedo_all = albedo.copy()
    logic_all  = np.zeros((frac.size, wvl.size))

    for i in range(wvl.size):

        logic_new = logic & np.logical_not(np.isnan(frac)) & np.logical_not(np.isnan(albedo[:, i]))
        logic_all[:, i] = logic_new
        frac_all[:, i]  = frac
        if logic_new.sum() > 0:
            slope[i], intercept[i], r_value, p_value, std_err = stats.linregress(frac[logic], albedo[logic, i])
        else:
            slope[i], intercept[i] = np.nan, np.nan

    wvl_arise0    = wvl.copy()
    albedo_arise0 = slope*1.0 + intercept
    wvl_arise, slope_arise     = ABSORB_FILTER(wvl_arise0, slope)
    wvl_arise, intercept_arise = ABSORB_FILTER(wvl_arise0, intercept)
    f0['wvl_arise']       = wvl_arise
    f0['slope_arise']     = slope_arise
    f0['intercept_arise'] = intercept_arise

    f0['frac_all']   = frac_all
    f0['wvl_all']    = wvl_arise0
    f0['albedo_all'] = albedo_all
    f0['logic_all']  = logic_all
    f0['alt_all']    = alt


    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(111)

    #  brandt and warren 2005
    data = np.loadtxt('/argus/home/chen/work/02_arise/10_poster/ice_only.txt', delimiter=',')
    ax1.plot(data[:, 0], data[:, 1], color='lime', label='ice only (Brandt and Warren 2005)', marker='o', markeredgecolor='none', zorder=1)
    f0['wvl_ice_only_2005'] = data[:, 0]
    f0['albedo_ice_only_2005'] = data[:, 1]
    data = np.loadtxt('/argus/home/chen/work/02_arise/10_poster/ice_thin_snow.txt', delimiter=',')
    ax1.plot(data[:, 0], data[:, 1], color='limegreen', label='ice+thin snow (Brandt and Warren 2005)', marker='o', markeredgecolor='none', zorder=2)
    f0['wvl_ice_thin_snow_2005'] = data[:, 0]
    f0['albedo_ice_thin_snow_2005'] = data[:, 1]
    data = np.loadtxt('/argus/home/chen/work/02_arise/10_poster/ice_deep_snow.txt', delimiter=',')
    ax1.plot(data[:, 0], data[:, 1], color='forestgreen', label='ice+deep snow (Brandt and Warren 2005)', marker='o', markeredgecolor='none', zorder=3)
    f0['wvl_ice_thick_snow_2005'] = data[:, 0]
    f0['albedo_ice_thick_snow_2005'] = data[:, 1]

    wvl_ref = np.arange(300.0, 2001.0, 50.0)

    # perovich et al 2002
    data = np.loadtxt('/argus/home/chen/work/02_arise/10_poster/albedo_perovich.csv', delimiter=',')
    wvl_per    = data[:, 0]
    albedo_per = data[:, 1]
    ax1.plot(wvl_per, albedo_per, color='royalblue', label='melting bare white ice (Perovich et al. 2002)', marker='o', markeredgecolor='none', zorder=0)
    f0['wvl_ice_melt_2002'] = data[:, 0]
    f0['albedo_ice_melt_2002'] = data[:, 1]

    jen_wvl = np.array([659.0, 858.0, 1240.0, 1640.0, 2130.0, 3740.0])
    a_dry   = np.array([90.0, 85.0, 45.0, 15.0, 15.0, 5.0])/100.0
    a_wet   = np.array([85.0, 75.0, 25.0,  5.0,  5.0, 5.0])/100.0
    ax1.plot(jen_wvl, a_dry, color='indianred', label="dry season (Kay and L'Ecuyer 2013)", marker='s', markeredgecolor='none', zorder=4)
    ax1.plot(jen_wvl, a_wet, color='orangered', label="wet season (Kay and L'Ecuyer 2013)", marker='s', markeredgecolor='none', zorder=5)
    f0['wvl_dry_season_2013'] = jen_wvl
    f0['wvl_wet_season_2013'] = jen_wvl
    f0['albedo_dry_season_2013'] = a_dry
    f0['albedo_wet_season_2013'] = a_wet


    data = np.genfromtxt('/argus/home/chen/work/02_arise/10_poster/20080415ELalbedo.dat', skip_header=1)
    wvl_arctas    = data[:, 0]
    albedo_arctas = data[:, 1]
    wvl_arctas, albedo_arctas = ABSORB_FILTER(wvl_arctas, albedo_arctas)
    ax1.plot(wvl_arctas, albedo_arctas, color='violet', label='ARCTAS SSFR (site EL)', marker='o', markeredgecolor='none', zorder=6)
    f0['wvl_arctas'] = wvl_arctas
    f0['albedo_arctas'] = albedo_arctas


    wvl_arise, albedo_arise = ABSORB_FILTER(wvl_arise0, albedo_arise0)
    # wvl_arise, std_arise    = ABSORB_FILTER(wvl_arise0, std_arise0)
    ax1.plot(wvl_arise, albedo_arise, color='k', label='ARISE SSFR (IF=100%)', marker='*', markeredgecolor='k', markersize=8, zorder=9, lw=2.5)
    ax1.plot(wvl_arise[wvl_arise<350.0], albedo_arise[wvl_arise<350.0], color='gray', marker='*', markeredgecolor='gray', markersize=8, zorder=9, lw=2.5)

    # ax1.fill_between(wvl_arise, albedo_arise*(1.0-std_arise), albedo_arise*(1.0+std_arise), linewidth=0.0, facecolor='k', alpha=0.7, zorder=8)
    # ax1.plot(wvl_arise[4], albedo_arise[4], markerfacecolor='k', marker='*', markeredgecolor='red', markersize=18, markeredgewidth=2.0, zorder=10)

    f0.close()

    ax1.set_xlim([300, 1800])
    ax1.xaxis.set_major_locator(FixedLocator(np.arange(300, 1801, 300)))
    ax1.xaxis.set_minor_locator(FixedLocator(np.arange(300, 1801, 100)))
    ax1.set_ylim([0, 1.0])
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Albedo')
    ax1.legend(fontsize=10, framealpha=0.1, loc='upper right')
    plt.savefig('albedo_all.png', transparent=False)
    plt.show()
    exit()


if __name__ == '__main__':

    warnings.warn('!!!!!!!! Under development !!!!!!!!')

    # data procesing
    #/----------------------------------------------------------------------------\#
    dates = [
             # datetime.datetime(2024, 5, 17), # ARCSIX test flight #1 at NASA WFF
             # datetime.datetime(2024, 5, 21), # ARCSIX test flight #2 at NASA WFF
             # datetime.datetime(2024, 5, 24), # ARCSIX transit flight #1 from NASA WFF to Pituffik Space Base
             # datetime.datetime(2024, 5, 28), # ARCSIX research flight #1 over Lincoln Sea
             datetime.datetime(2024, 6, 11), # ARCSIX research flight #1 over Lincoln Sea
            ]
    for date in dates[::-1]:
        # main_process_data_v0(date, run=False)
        # main_process_data_v1(date, run=False)
        # main_process_data_v2(date, run=False)

        plot_spiral(date)
    #\----------------------------------------------------------------------------/#

    pass
