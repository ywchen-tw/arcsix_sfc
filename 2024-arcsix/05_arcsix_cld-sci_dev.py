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
_fdir_sat_data_   = 'data/%s/sat' % _mission_
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

class flt_sim:

    def __init__(
            self,
            date=datetime.datetime.now(),
            fdir='./',
            extent=None,
            wavelength=None,
            flt_trks=None,
            flt_imgs=None,
            fname=None,
            overwrite=False,
            quiet=False,
            verbose=False
            ):

        self.date      = date
        self.wvl0      = wavelength
        self.fdir      = os.path.abspath(fdir)
        self.extent    = extent
        self.flt_trks  = flt_trks
        self.flt_imgs  = flt_imgs
        self.overwrite = overwrite
        self.quiet     = quiet
        self.verbose   = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif (((flt_trks is not None) and (flt_imgs is not None) and (wavelength is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((flt_trks is not None) and (flt_imgs is not None) and (wavelength is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run_rtm()
            self.dump(fname)

        elif (((flt_trks is not None) and (flt_imgs is not None) and (wavelength is not None)) and (fname is None)):

            self.run()

        else:

            sys.exit('Error   [flt_sim]: Please check if \'%s\' exists or provide \'wavelength\', \'flt_trks\', and \'flt_imgs\' to proceed.' % fname)

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'flt_trks') and hasattr(obj, 'flt_imgs'):
                if self.verbose:
                    print('Message [flt_sim]: Loading %s ...' % fname)
                self.date     = obj.date
                self.fdir     = obj.fdir
                self.extent   = obj.extent
                self.wvl0     = obj.wvl0
                self.fname    = obj.fname
                self.flt_trks = obj.flt_trks
                self.flt_imgs = obj.flt_imgs
            else:
                sys.exit('Error   [flt_sim]: File \'%s\' is not the correct pickle file to load.' % fname)

    def run_rtm(self, overwrite=True):

        N = len(self.flt_trks)

        for i in range(N):

            flt_trk = self.flt_trks[i]
            flt_img = self.flt_imgs[i]

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [flt_sim]: Saving object into %s ...' % fname)
            pickle.dump(self, f)



class satellite_download:

    def __init__(
            self,
            date=None,
            extent=None,
            fname=None,
            satellite='aqua',
            fdir_out='data/arcsix/sat-data',
            overwrite=False,
            quiet=False,
            verbose=False):

        self.date      = date
        self.extent    = extent
        self.satellite = satellite
        self.fdir_out  = fdir_out
        self.quiet     = quiet
        self.verbose   = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif (((date is not None) and (extent is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((date is not None) and (extent is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run()
            self.dump(fname)

        elif (((date is not None) and (extent is not None)) and (fname is None)):

            self.run()

        else:

            msg = '\nError [satellite_download]: Please check if <%s> exists or provide <date> and <extent> to proceed.' % fname
            raise OSError(msg)

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'fnames') and hasattr(obj, 'extent') and hasattr(obj, 'fdir_out') and hasattr(obj, 'date'):
                if self.verbose:
                    print('Message [satellite_download]: Loading %s ...' % fname)
                self.date     = obj.date
                self.extent   = obj.extent
                self.fnames   = obj.fnames
                self.fdir_out = obj.fdir_out
            else:
                msg = '\nError [satellite_download]: File <%s> is not the correct pickle file to load.' % fname
                raise OSError(msg)

    def run(self, run=True):

        lon0 = np.linspace(self.extent[0], self.extent[1], 800)
        lat0 = np.linspace(self.extent[2], self.extent[3], 800)
        lon, lat = np.meshgrid(lon0, lat0, indexing='ij')

        # create prefixes for the satellite products
        if self.satellite.lower() == 'aqua':
            dataset_tags = ['61/MYD03', '61/MYD06_L2', '61/MYD02QKM']
        elif self.satellite.lower() == 'terra':
            dataset_tags = ['61/MOD03', '61/MOD06_L2', '61/MOD02QKM']
        else:
            msg = '\nError [satellite_download]: Satellite must be either \'Aqua\' or \'Terra\'. %s is currently not supported' % self.satellite
            raise ValueError(msg)

        self.fnames = {}

        # MODIS RGB imagery
        self.fnames['mod_rgb'] = [er3t.util.download_worldview_image(self.date, self.extent, fdir_out=self.fdir_out, satellite=self.satellite, instrument='modis', coastline=True)]

        # MODIS Level 2 Cloud Product and MODIS 03 geo file
        self.fnames['mod_l2'] = []
        self.fnames['mod_02'] = []
        self.fnames['mod_03'] = []

        filename_tags_03 = er3t.util.get_satfile_tag(self.date, lon, lat, satellite=self.satellite, instrument='modis')
        if self.verbose:
           print('Message [satellite_download]: Found %s %s overpasses' % (len(filename_tags_03), self.satellite))

        for filename_tag in filename_tags_03:
            fnames_03     = er3t.util.download_laads_https(self.date, dataset_tags[0], filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_l2     = er3t.util.download_laads_https(self.date, dataset_tags[1], filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_02     = er3t.util.download_laads_https(self.date, dataset_tags[2], filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)

            self.fnames['mod_l2'] += fnames_l2
            self.fnames['mod_02'] += fnames_02
            self.fnames['mod_03'] += fnames_03

        # MODIS surface product
        self.fnames['mod_43a1'] = []
        self.fnames['mod_43a3'] = []
        filename_tags_43 = er3t.util.modis.get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_43:
            fnames_43a1 = er3t.util.download_laads_https(self.date, '61/MCD43A1', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_43a1'] += fnames_43a1
            fnames_43a3 = er3t.util.download_laads_https(self.date, '61/MCD43A3', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_43a3'] += fnames_43a3

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [satellite_download]: Saving object into %s ...' % fname)
            pickle.dump(self, f)



def main_pre_download_satellite_data(
        date,
        ):

    date_s = date.strftime('%Y%m%d')

    # read flt obj
    #/----------------------------------------------------------------------------\#
    fname = '%s/%s-FLT-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _platform_.upper(), date_s)
    flt_sim0 = flt_sim(
            fname=fname,
            overwrite=False,
            )
    #\----------------------------------------------------------------------------/#


    # retrieve longitude, latitude from flt_sim object
    # why not use hsk:
    #   hsk will bring more calculations that won't be used in the video
    #/----------------------------------------------------------------------------\#
    fdir_out = 'data/arcsix/sat-data/%s' % date_s
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    if date_s in ['20240530', '20240531', '20240603']:
        satellites   = ['aqua', 'noaa20']
        instruments  = ['modis', 'viirs']
        dataset_tags = ['5111/CLDPROP_L2_MODIS_Aqua', '5111/CLDPROP_L2_VIIRS_NOAA20']
    else:
        satellites   = ['aqua', 'snpp', 'noaa20']
        instruments  = ['modis', 'viirs', 'viirs']
        dataset_tags = ['5111/CLDPROP_L2_MODIS_Aqua', '5111/CLDPROP_L2_VIIRS_SNPP', '5111/CLDPROP_L2_VIIRS_NOAA20']

    for i, flt_trk0 in enumerate(flt_sim0.flt_trks):

        jday0 = flt_trk0['jday']
        lon0  = flt_trk0['lon']
        lat0  = flt_trk0['lat']

        for i, satellite0 in enumerate(satellites):
            instrument0 = instruments[i]
            dataset_tag0 = dataset_tags[i]

            filename_tags = er3t.util.get_satfile_tag(date, lon0, lat0, satellite=satellite0, instrument=instrument0)
            for filename_tag0 in filename_tags:
                fnames_l2 = er3t.util.download_laads_https(date, dataset_tag0, filename_tag0, day_interval=1, fdir_out=fdir_out, run=True, verbose=True)
    #\----------------------------------------------------------------------------/#



def main_pre_retrieve_satellite_cloud(
        date,
        ):

    date_s = date.strftime('%Y%m%d')
    print(date_s)

    # read flt obj
    #/----------------------------------------------------------------------------\#
    fname = '%s/%s-FLT-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _platform_.upper(), date_s)
    flt_sim0 = flt_sim(
            fname=fname,
            overwrite=False,
            )
    #\----------------------------------------------------------------------------/#


    # derive time stamp for each satellite file and sort all the satellite files
    # according to their time stamp
    #/----------------------------------------------------------------------------\#
    fdir_sat_data = 'data/arcsix/sat-data/%s' % date_s

    fnames_sat = sorted(glob.glob('%s/*.nc' % fdir_sat_data))
    Nsat = len(fnames_sat)

    jday_sat = np.zeros(Nsat, dtype=np.float64)
    for i, fname in enumerate(fnames_sat):
        dtime0_s = '.'.join(os.path.basename(fname).split('.')[1:3])[1:]
        dtime0 = datetime.datetime.strptime(dtime0_s, '%Y%j.%H%M')
        jday0 = er3t.util.dtime_to_jday(dtime0)
        jday_sat[i] = jday0

    indices_sort = np.argsort(jday_sat)
    jday_sat = jday_sat[indices_sort]
    fnames_sat = [fnames_sat[i] for i in indices_sort]
    #\----------------------------------------------------------------------------/#


    # roughly select satellite products that can be used to derive cloud retrievals
    # for the flight track
    #/----------------------------------------------------------------------------\#
    valid_sat_time_window = 60.0 * 60.0 / 86400.0 # 1 hour
    fnames_sat_select = []
    jday_sat_select = []

    for i, flt_trk0 in enumerate(flt_sim0.flt_trks):

        jday0 = flt_trk0['jday']
        lon0  = flt_trk0['lon']
        lat0  = flt_trk0['lat']

        cot0 = np.repeat(np.nan, jday0.size)
        index = np.argmin(np.abs(np.nanmean(jday0)-jday_sat))
        if (np.nanmin(jday0) >= (jday_sat[index]-valid_sat_time_window)) and (np.nanmax(jday0) <= (jday_sat[index]+valid_sat_time_window)):
            if fnames_sat[index] not in fnames_sat_select:
                fnames_sat_select.append(fnames_sat[index])
                jday_sat_select.append(jday_sat[index])
    #\----------------------------------------------------------------------------/#


    # interpolate
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)
    f_hsk = h5py.File(fname_hsk, 'r')
    jday   = f_hsk['jday'][...]
    tmhr   = f_hsk['tmhr'][...]
    sza    = f_hsk['sza'][...]
    lon    = f_hsk['lon'][...]
    lat    = f_hsk['lat'][...]
    alt    = f_hsk['alt'][...]
    f_hsk.close()

    Nselect = len(fnames_sat_select)

    cot  = np.zeros((jday.size, Nselect), dtype=np.float64)
    cer  = np.zeros((jday.size, Nselect), dtype=np.float64)
    cth  = np.zeros((jday.size, Nselect), dtype=np.float64)
    ctp  = np.zeros((jday.size, Nselect), dtype=np.float64)
    jdiff = np.zeros((jday.size, Nselect), dtype=np.float64)

    for i, fname in enumerate(fnames_sat_select):

        f = Dataset(fname, 'r')
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

        logic_pcl = ((cot_s<0.0)&(cer_s<0.0)) & ((cot_pcl_s>0.0)&(cer_pcl_s>0.0))
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
        jday_s0 = np.repeat(jday_s0, int(Nx/jday_s0.size))

        jday_s = np.zeros((Nx, Ny), dtype=np.float64)
        jday_s[...] = jday_s0[:, None]

        cot[:, i] = er3t.util.find_nearest(lon_s, lat_s, cot_s, lon, lat, Ngrid_limit=4, fill_value=np.nan)
        cer[:, i] = er3t.util.find_nearest(lon_s, lat_s, cer_s, lon, lat, Ngrid_limit=4, fill_value=np.nan)
        cth[:, i] = er3t.util.find_nearest(lon_s, lat_s, cth_s, lon, lat, Ngrid_limit=4, fill_value=np.nan)
        ctp[:, i] = er3t.util.find_nearest(lon_s, lat_s, ctp_s, lon, lat, Ngrid_limit=4, fill_value=np.nan)

        jdiff[:, i] = er3t.util.find_nearest(lon_s, lat_s, jday_s, lon, lat, Ngrid_limit=4, fill_value=np.nan) - jday
        print(i+1, Nselect)

    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    f = h5py.File(fname_sat, 'w')
    f['sat/jday'] = jday_sat_select
    f['sat/jday'].attrs['description'] = '\n'.join([os.path.basename(fname) for fname in fnames_sat_select])
    f['sat/cot'] = cot
    f['sat/cer'] = cer
    f['sat/cth'] = cth
    f['sat/ctp'] = ctp
    f['sat/time_diff'] = jdiff*86400.0

    f['jday'] = jday
    f['tmhr'] = tmhr
    f['lon'] = lon
    f['lat'] = lat
    f['alt'] = alt
    f['sza'] = sza
    f.close()



def main_pre_rt_sim_flux(
        date,
        tmhr_offset_threshold=15.0/60.0,
        ):

    date_s = date.strftime('%Y%m%d')
    print(date_s)

    # pick collocated satellite data that's closest in time with aircraft
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    tmhr_offset = np.abs(data_sat['sat/time_diff']/3600.0)
    tmhr_offset[np.isnan(tmhr_offset)] = 999.0
    tmhr_offset[tmhr_offset>tmhr_offset_threshold] = 999.0

    indices = np.argmin(tmhr_offset, axis=-1)
    cot = np.zeros_like(data_sat['jday'])
    cot[...] = np.nan
    cer = np.zeros_like(data_sat['jday'])
    cer[...] = np.nan

    ctp = np.zeros(data_sat['jday'].size, dtype=np.int32)
    ctp[...] = -1
    for i, index in enumerate(indices):
        if (tmhr_offset[i, :] == 999.0).sum() < data_sat['sat/jday'].size:
            cot[i] = data_sat['sat/cot'][i, index]
            cer[i] = data_sat['sat/cer'][i, index]
            ctp[i] = data_sat['sat/ctp'][i, index]

    logic_ic = (ctp==3)
    logic_wc = (ctp==2) | (ctp==4)

    cot[cot<0.0] = np.nan
    cer[cer<0.0] = np.nan
    cot[(ctp==1)&np.isnan(cot)] = 0.0
    cer[(ctp==1)&np.isnan(cer)] = 1.0

    logic_sat = ~np.isnan(cot)

    print('Valid satellite retrieval:')
    print('%.1f%%' % (logic_sat.sum()/logic_sat.size*100.0))
    #\----------------------------------------------------------------------------/#


    # selected for rt calculation
    #/----------------------------------------------------------------------------\#
    # logic_rt = (ctp==1) | (ctp==2) | (ctp==4)
    # print(logic_rt.sum()/logic_rt.size * 100.0)
    #\----------------------------------------------------------------------------/#


    # steady leg
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    logic_steady = (np.abs(data_hsk['ang_rol'])<2.0) & (np.abs(data_hsk['ang_pit'])<2.0)
    print('Steady leg:')
    print('%.1f%%' % (logic_steady.sum()/logic_steady.size*100.0))
    #\----------------------------------------------------------------------------/#


    # cloud detected by spns
    #/----------------------------------------------------------------------------\#
    fname_spns = '%s/%s-SPNS_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_spns = er3t.util.load_h5(fname_spns)
    index_wvl = np.argmin(np.abs(data_spns['tot/wvl']-532.0))
    diff_ratio = data_spns['dif/flux'][:, index_wvl]/data_spns['tot/flux'][:, index_wvl]
    logic_cld_spns = (diff_ratio > 0.9) & logic_steady
    print('Cloud detected by SPNS:')
    print('%.1f%%' % (logic_cld_spns.sum()/logic_cld_spns.size*100.0))
    #\----------------------------------------------------------------------------/#


    # cloud detected by marli
    #/----------------------------------------------------------------------------\#
    #\----------------------------------------------------------------------------/#


    logic_cld     = logic_cld_spns & (~np.isnan(cot))

    logic_detect  = logic_cld & (ctp!=1)
    logic_miss    = logic_cld & (~logic_detect)

    # print(logic_cld.sum())
    # print(logic_correct.sum())
    # print(logic_miss.sum())

    print('Cloud detected by SPNS but missed by satellite:')
    print('%.1f%%' % (logic_miss.sum()/logic_cld.sum()*100.0))
    print()



    # lower leg
    #/----------------------------------------------------------------------------\#
    # logic_low = (data_spns['alt'] < 500.0)
    # print(logic_low.sum()/logic_low.size * 100.0)
    #\----------------------------------------------------------------------------/#


    # logic = (logic_low|logic_high) & logic_steady & logic_rt
    # print(logic.sum()/logic.size * 100.0)


    # rt simulation setup
    #/----------------------------------------------------------------------------\#
    # sza = data_hsk['sza']

    # lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    # cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    # cld_cfg['cloud_file'] = '%s/cloud.txt' % fdir_tmp
    # cld_cfg['cloud_optical_thickness'] = 10.0
    # cld_cfg['cloud_effective_radius']  = 12.0
    # cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    # inits = []
    # for i, sza0 in enumerate(sza):

    #     init = er3t.rtm.lrt.lrt_init_mono_flx(
    #             input_file  = '%s/input%2.2d.txt'  % (fdir_tmp, i),
    #             output_file = '%s/output%2.2d.txt' % (fdir_tmp, i),
    #             date        = datetime.datetime(2014, 9, 11),
    #             surface_albedo     = 0.8,
    #             solar_zenith_angle = sza0,
    #             wavelength         = 532.31281,
    #             output_altitude    = 5.0,
    #             lrt_cfg            = lrt_cfg,
    #             cld_cfg            = cld_cfg
    #             )
    #     inits.append(init)

    # # run with multi cores
    # er3t.rtm.lrt.lrt_run_mp(inits, Ncpu=6)

    # data = er3t.rtm.lrt.lrt_read_uvspec_flx(inits)

    # # the flux calculated can be accessed through
    # print('Results for <%s>:' % _metadata['Function'])
    # print('  Upwelling flux: ', np.squeeze(data.f_up))
    # print('  Downwelling flux: ', np.squeeze(data.f_down))
    # print('  Down-diffuse flux: ', np.squeeze(data.f_down_diffuse))
    # print('  Down-direct flux: ', np.squeeze(data.f_down_direct))
    # print()
    #\----------------------------------------------------------------------------/#

    # figure
    #/----------------------------------------------------------------------------\#
    if False:
        plt.close('all')
        fig = plt.figure(figsize=(20, 5))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        # cs = ax1.imshow(time_offset.T, origin='lower', cmap='jet', zorder=0, aspect='auto', vmin=0.0, vmax=7.0)
        # ax1.scatter(data_sat['tmhr'], cot, c=indices, s=10, cmap='jet', lw=0.0, alpha=0.01)
        # ax1.scatter(data_sat['tmhr'][logic], cot[logic], c=indices[logic], s=10, cmap='jet', lw=0.0)
        ax1.scatter(data_sat['tmhr'], data_sat['alt'], c=indices, s=10, cmap='jet', lw=0.0, alpha=0.01)
        ax1.scatter(data_sat['tmhr'][logic], data_sat['alt'][logic], c=indices[logic], s=10, cmap='jet', lw=0.0)
        # ax1.scatter(data_sat['tmhr'][logic_ic], cot[logic_ic], c=indices[logic_ic], s=10, cmap='jet', lw=0.0, marker='^')
        # ax2.scatter(data_sat['tmhr'], ctp, c=indices, s=20, cmap='viridis', lw=0.0)
        # ax1.plot([0, 1], [0, 1], color='k', ls='--')
        # ax1.set_xlim(())
        ax1.set_ylim(bottom=0.0)
        ax1.set_xlabel('Time [Hour]')
        ax1.set_ylabel('COT')
        ax1.set_title(date_s)
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





def plot_cop(
        date,
        tmhr_offset_threshold=1.0,
        ):

    date_s = date.strftime('%Y%m%d')

    tmhr_range_leg = [15.3486, 15.7222]

    # steady leg
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    logic_steady = (data_hsk['tmhr']>=tmhr_range_leg[0]) & (data_hsk['tmhr']<=tmhr_range_leg[1]) & \
                   (np.abs(data_hsk['ang_rol'])<2.5) & (np.abs(data_hsk['ang_pit'])<2.5)
    #\----------------------------------------------------------------------------/#

    # pick collocated satellite data that's closest in time with aircraft
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    tmhr_offset = np.abs(data_sat['sat/time_diff']/3600.0)
    tmhr_offset[np.isnan(tmhr_offset)] = 999.0
    tmhr_offset[tmhr_offset>tmhr_offset_threshold] = 999.0
    tmhr_offset[tmhr_offset==999.0] = np.nan
    tmhr_offset[data_sat['sat/cot']<0.0] = np.nan

    cot = np.zeros_like(data_sat['jday'])
    cot[...] = np.nan
    cer = np.zeros_like(data_sat['jday'])
    cer[...] = np.nan

    ctp = np.zeros(data_sat['jday'].size, dtype=np.int32)
    ctp[...] = -1

    t_diff = np.zeros_like(data_sat['jday'])
    t_diff[...] = np.nan

    for i in range(cot.size):
        if np.isnan(tmhr_offset[i, :]).sum() < data_sat['sat/jday'].size:
            index = np.nanargmin(tmhr_offset[i, :])
            cot[i] = data_sat['sat/cot'][i, index]
            cer[i] = data_sat['sat/cer'][i, index]
            ctp[i] = data_sat['sat/ctp'][i, index]
            t_diff[i] = tmhr_offset[i, index]

    logic_ic = (ctp==3)
    logic_wc = (ctp==2) | (ctp==4)

    cot[cot<0.0] = np.nan
    cer[cer<0.0] = np.nan
    cot[((ctp==1))&np.isnan(cot)] = 0.0
    cer[((ctp==1))&np.isnan(cer)] = 1.0

    # upper leg (cirrus free)
    #/----------------------------------------------------------------------------\#
    fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_ssfr = er3t.util.load_h5(fname_ssfr)
    index_wvl = np.argmin(np.abs(data_ssfr['zen/wvl']-532.0))
    f_dn = data_ssfr['zen/flux'][:, index_wvl]
    #\----------------------------------------------------------------------------/#

    # figure
    #/----------------------------------------------------------------------------\#
    if False:
        rcParams['font.size'] = 24
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(data_hsk['lon'], data_hsk['alt']/1000.0, s=2, c=data_hsk['tmhr'], alpha=0.02, cmap='jet')
        ax1.scatter(data_hsk['lon'][logic_steady], data_hsk['alt'][logic_steady]/1000.0, s=20, c=data_hsk['tmhr'][logic_steady], alpha=1.0, cmap='jet')

        ax1.set_xlim((-70.0, -55.0))
        ax1.set_ylim((0.0, 4.0))
        ax1.set_xlabel('Longitude [$^\\circ$]')
        ax1.set_ylabel('Altitude [km]')
        ax1.set_title('Low Leg on 2024-06-11')
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

    print(np.nanmin(t_diff[logic_steady]))
    print(np.nanmax(t_diff[logic_steady]))
    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        rcParams['font.size'] = 24
        plt.close('all')
        fig = plt.figure(figsize=(15, 5))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.scatter(data_hsk['tmhr'][logic_steady], f_dn[logic_steady], c='k', s=10)
        # ax2.bar(data_hsk['tmhr'][logic_steady], cot[logic_steady], width=1.0/3600.0, color='green', alpha=1.0)
        ax2.scatter(data_hsk['tmhr'][logic_steady], cot[logic_steady], c=t_diff[logic_steady], alpha=1.0, lw=0.0, vmin=0.0, vmax=1.0, cmap='jet')

        # for index in range(data_sat['sat/cot'].shape[-1]):
        #     ax2.scatter(data_hsk['tmhr'][logic_steady], data_sat['sat/cot'][logic_steady, index], alpha=1.0, lw=0.0, s=2)

        # ax1.set_xlim((-70.0, -55.0))
        ax1.set_ylim((0.0, 1.5))
        ax2.set_ylim((0.0, 15.0))
        ax1.set_xlabel('Time [Hour]')
        ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_title('Low Leg on 2024-06-11')
        ax2.set_ylabel('MODIS and VIIRS COT', color='green', rotation=270, labelpad=20)
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



    # selected for rt calculation
    #/----------------------------------------------------------------------------\#
    logic_rt = (ctp==1) | (ctp==2) | (ctp==4)
    print(logic_rt.sum()/logic_rt.size * 100.0)
    #\----------------------------------------------------------------------------/#






    # lower leg
    #/----------------------------------------------------------------------------\#
    logic_low = (data_spns['alt'] < 500.0)
    print(logic_low.sum()/logic_low.size * 100.0)
    #\----------------------------------------------------------------------------/#


    logic = (logic_low|logic_high) & logic_steady & logic_rt
    print(logic.sum()/logic.size * 100.0)


    # rt simulation setup
    #/----------------------------------------------------------------------------\#
    sza = data_hsk['sza']

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    cld_cfg['cloud_file'] = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = 10.0
    cld_cfg['cloud_effective_radius']  = 12.0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    inits = []
    for i, sza0 in enumerate(sza):

        init = er3t.rtm.lrt.lrt_init_mono_flx(
                input_file  = '%s/input%2.2d.txt'  % (fdir_tmp, i),
                output_file = '%s/output%2.2d.txt' % (fdir_tmp, i),
                date        = datetime.datetime(2014, 9, 11),
                surface_albedo     = 0.8,
                solar_zenith_angle = sza0,
                wavelength         = 532.31281,
                output_altitude    = 5.0,
                lrt_cfg            = lrt_cfg,
                cld_cfg            = cld_cfg
                )
        inits.append(init)

    # run with multi cores
    er3t.rtm.lrt.lrt_run_mp(inits, Ncpu=6)

    data = er3t.rtm.lrt.lrt_read_uvspec_flx(inits)
    #\----------------------------------------------------------------------------/#

    # the flux calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Upwelling flux: ', np.squeeze(data.f_up))
    print('  Downwelling flux: ', np.squeeze(data.f_down))
    print('  Down-diffuse flux: ', np.squeeze(data.f_down_diffuse))
    print('  Down-direct flux: ', np.squeeze(data.f_down_direct))
    print()

    # figure
    #/----------------------------------------------------------------------------\#
    if False:
        plt.close('all')
        fig = plt.figure(figsize=(20, 5))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        # cs = ax1.imshow(time_offset.T, origin='lower', cmap='jet', zorder=0, aspect='auto', vmin=0.0, vmax=7.0)
        # ax1.scatter(data_sat['tmhr'], cot, c=indices, s=10, cmap='jet', lw=0.0, alpha=0.01)
        # ax1.scatter(data_sat['tmhr'][logic], cot[logic], c=indices[logic], s=10, cmap='jet', lw=0.0)
        ax1.scatter(data_sat['tmhr'], data_sat['alt'], c=indices, s=10, cmap='jet', lw=0.0, alpha=0.01)
        ax1.scatter(data_sat['tmhr'][logic], data_sat['alt'][logic], c=indices[logic], s=10, cmap='jet', lw=0.0)
        # ax1.scatter(data_sat['tmhr'][logic_ic], cot[logic_ic], c=indices[logic_ic], s=10, cmap='jet', lw=0.0, marker='^')
        # ax2.scatter(data_sat['tmhr'], ctp, c=indices, s=20, cmap='viridis', lw=0.0)
        # ax1.plot([0, 1], [0, 1], color='k', ls='--')
        # ax1.set_xlim(())
        ax1.set_ylim(bottom=0.0)
        ax1.set_xlabel('Time [Hour]')
        ax1.set_ylabel('COT')
        ax1.set_title(date_s)
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





def process_marli(date):

    date_s = date.strftime('%Y%m%d')

    try:
    # if True:
        fname = sorted(er3t.util.get_all_files('data/arcsix/2024-Spring/p3/aux/marli', pattern='*%s*.cdf' % (date_s)))[-1]

        fname_hsk = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)
        f_hsk = h5py.File(fname_hsk, 'r')
        tmhr = f_hsk['tmhr'][...]
        alt  = f_hsk['alt'][...]
        f_hsk.close()

        # read marli
        #/----------------------------------------------------------------------------\#
        f = Dataset(fname, 'r')
        tmhr_1d = np.array(f.variables['time'][...].data, dtype=np.float64)
        h_1d = np.array(f.variables['H'][...].data*1000.0, dtype=np.float64)
        data_2d = np.array(f.variables['LSR'][...].data, dtype=np.float64)
        f.close()
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

        data_return = {}
        data_return['tmhr_2d'] = tmhr_2d_new
        data_return['h_2d']    = h_2d_new
        data_return['lsr_2d']  = data_2d_new


        return data_return
        #\----------------------------------------------------------------------------/#
    except Exception as error:
        print(error)
        data_return = None

    return data_return

def is_over_land(lon0, lat0):

    import cartopy.io.shapereader as shapereader
    import shapely.geometry as sgeom

    MAP_RES  = '110m'
    MAP_TYPE = 'physical'
    MAP_NAME = 'land'

    shape_data = shapereader.natural_earth(resolution=MAP_RES, category=MAP_TYPE, name=MAP_NAME)
    lands = shapereader.Reader(shape_data).geometries()

    # Check if a point is over land.
    for land in lands:
        if land.contains(sgeom.Point(lon0, lat0)):
            return True
    # If it wasn't found, return False.
    return False

def cdata_logic_ocean(
        date,
        ):

    date_s = date.strftime('%Y%m%d')
    print(date_s)

    # read hsk file
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#

    # over ocean/snow ice
    #/----------------------------------------------------------------------------\#
    logic_ocean = np.repeat(False, data_hsk['tmhr'].size)
    for i in tqdm(range(logic_ocean.size)):
        logic_ocean[i] = (not is_over_land(data_hsk['lon'][i], data_hsk['lat'][i]))
    #\----------------------------------------------------------------------------/#

    f = h5py.File(fname_hsk, 'r+')
    try:
        f['logic_ocean'] = logic_ocean
    except Exception as error:
        print(error)
        del f['logic_ocean']
        f['logic_ocean'] = logic_ocean
    f.close()

def cdata_logic_all(
        date,
        ):

    date_s = date.strftime('%Y%m%d')
    print(date_s)

    # read aircraft housekeeping data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    N_total = data_hsk['ang_rol'].size
    #\----------------------------------------------------------------------------/#


    # steady leg
    #/----------------------------------------------------------------------------\#
    logic_steady = (np.abs(data_hsk['ang_rol'])<2.5) & (np.abs(data_hsk['ang_pit'])<2.5)

    N_steady = logic_steady.sum()
    print('Steady leg:')
    print('%d of %d, %.1f%%' % (N_steady, N_total, N_steady/N_total*100.0))
    #\----------------------------------------------------------------------------/#


    # over ocean/snow ice
    #/----------------------------------------------------------------------------\#
    logic_ocean = data_hsk['logic_ocean']

    N_ocean = logic_ocean.sum()
    print('Over-ocean leg:')
    print('%d of %d, %.1f%%' % (N_ocean, N_total, N_ocean/N_total*100.0))
    #\----------------------------------------------------------------------------/#


    # cloud and clear-sky detected by ssfr zenith (since it has platform, more reliable than spns zenith total)
    #/----------------------------------------------------------------------------\#
    # fname_ssfr = '%s/%s-SSFR_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    # data_ssfr = er3t.util.load_h5(fname_ssfr)
    # index_wvl = np.argmin(np.abs(data_ssfr['zen/wvl']-532.0))
    # diff_toa = np.abs(data_ssfr['zen/flux'][:, index_wvl]-(data_ssfr['zen/toa0'][index_wvl]*np.cos(np.deg2rad(data_hsk['sza']))))/(data_ssfr['zen/toa0'][index_wvl]*np.cos(np.deg2rad(data_hsk['sza'])))
    #\----------------------------------------------------------------------------/#


    # cloud and clear-sky detected by spns
    #/----------------------------------------------------------------------------\#
    fname_spns = '%s/%s-SPNS_%s_%s_RA.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_spns = er3t.util.load_h5(fname_spns)
    index_wvl = np.argmin(np.abs(data_spns['tot/wvl']-532.0))
    diff_ratio = data_spns['dif/flux'][:, index_wvl]/data_spns['tot/flux'][:, index_wvl]

    logic_cloud_spns = (diff_ratio > 0.9) # more on strict side
    logic_clear_spns = (diff_ratio < 0.3) # more on loose side

    # N_cloud_spns = logic_cloud_spns.sum()
    # print('Cloud detected by SPNS:')
    # print('%d of %d, %.1f%%' % (N_cloud_spns, N_total, N_cloud_spns/N_total*100.0))
    #\----------------------------------------------------------------------------/#


    # cloud and clear-sky detected by marli
    #/----------------------------------------------------------------------------\#
    data_marli = process_marli(date)

    dates_bad  = []
    if data_marli is not None:

        tmhr_m = data_marli['tmhr_2d'][:, 0]
        h_m    = data_marli['h_2d'][0, :]*1000.0


        h_max = np.zeros_like(tmhr_m)

        for i in range(tmhr_m.size):

            lsr0 = data_marli['lsr_2d'][i, :].copy()
            logic0 = lsr0 > 100.0
            if logic0.sum() > 0:
                h_max[i] = np.nanmax(h_m[logic0])

        h = np.interp(data_hsk['tmhr'], tmhr_m, h_max)

    else:
        dates_bad.append(date)

    if date in dates_bad:
        logic_cloud_marli = logic_cloud_spns.copy()
        logic_clear_marli = (data_hsk['alt']>4000.0)
    else:
        logic_cloud_marli = (h > 50.0) & logic_ocean # assume iceberg is lower than 50m, more strict
        logic_clear_marli = (h < 100.0) & logic_ocean # assume iceber is lower than 100m, more loose

    # N_cloud_marli = logic_cloud_marli.sum()
    # print('Cloud detected by MARLi:')
    # print('%d of %d, %.1f%%' % (N_cloud_marli, N_total, N_cloud_marli/N_total*100.0))
    #\----------------------------------------------------------------------------/#


    # cloud combined
    #/----------------------------------------------------------------------------\#
    logic_cloud = (logic_cloud_spns|logic_cloud_marli) & logic_steady & logic_ocean

    N_cloud = logic_cloud.sum()
    print('Cloud detected by SPNS/MARLi:')
    print('%d of %d, %.1f%%' % (N_cloud, N_total, N_cloud/N_total*100.0))
    #\----------------------------------------------------------------------------/#


    # clear-sky combined
    #/----------------------------------------------------------------------------\#
    logic_clear = (logic_clear_spns&logic_clear_marli) & logic_steady & logic_ocean

    N_clear = logic_clear.sum()
    print('Clear-sky detected by SPNS/MARLi:')
    print('%d of %d, %.1f%%' % (N_clear, N_total, N_clear/N_total*100.0))
    #\----------------------------------------------------------------------------/#


    fname_cld = 'data/%s-LOGIC_%s_%s_RA.h5' % (_mission_.upper(), _platform_.upper(), date_s)
    f = h5py.File(fname_cld, 'w')
    f['logic_steady'] = logic_steady
    f['logic_ocean'] = logic_ocean
    f['logic_cloud_spns']  = logic_cloud_spns
    f['logic_cloud_marli'] = logic_cloud_marli
    f['logic_clear_spns']  = logic_clear_spns
    f['logic_clear_marli'] = logic_clear_marli
    f['logic_cloud']  = logic_cloud
    f['logic_clear']  = logic_clear
    f.close()

    return






def figure_01_arcsix1_flight_tracks_all(
        ):

    dates = [
            datetime.datetime(2024, 5, 28), # [✓]
            datetime.datetime(2024, 5, 30), #
            datetime.datetime(2024, 5, 31), #
            datetime.datetime(2024, 6,  3), #
            datetime.datetime(2024, 6,  5), # [✓]
            datetime.datetime(2024, 6,  6), # [✓]
            datetime.datetime(2024, 6,  7), # [✓]
            datetime.datetime(2024, 6, 10), # [✓]
            datetime.datetime(2024, 6, 11), # [✓]
            datetime.datetime(2024, 6, 13), # [✓]
        ]

    extent = [-80, -30, 71, 88]

    proj0 = ccrs.Orthographic(
            central_longitude=((extent[0]+extent[1])/2.0),
            central_latitude=((extent[2]+extent[3])/2.0),
            )

    plt.close('all')
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(111, projection=proj0)

    colors = mpl.colormaps['jet'](np.linspace(0.0, 1.0, len(dates)))
    patches_legend = []

    for i, date in enumerate(dates):

        date_s = date.strftime('%Y%m%d')

        fname_hsk = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)
        f_hsk = h5py.File(fname_hsk, 'r')
        lon    = f_hsk['lon'][...]
        lat    = f_hsk['lat'][...]
        f_hsk.close()

        ax1.plot(lon, lat, lw=2, color=colors[i, ...], zorder=1, transform=ccrs.PlateCarree())
        patches_legend.append(mpatches.Patch(color=colors[i, ...], label='SF#%d on %s ' % (i+1, date.strftime('%B-%d').replace('-0', ' ').replace('-', ' '))))

    ax1.coastlines(resolution='10m', color='black', lw=0.5)
    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    g1.xlocator = FixedLocator(np.arange(-180, 181, 10.0))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 2.0))
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    ax1.stock_img()

    ax1.legend(handles=patches_legend, loc='lower right', fontsize=16)

    fig.suptitle('2024 ARCSIX-1 (Spring Deployment)', fontsize=24, y=0.95)
    # save figure
    #/--------------------------------------------------------------\#
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fig.savefig('%s.png' % (_metadata['Function']), bbox_inches='tight', metadata=_metadata)
    #\--------------------------------------------------------------/#
    plt.show()






def figure_cop_wall_20240605(
        ):

    # case specification
    #/----------------------------------------------------------------------------\#
    date = datetime.datetime(2024, 6, 5)
    tmhr_ranges_select = [[14.5806, 15.2472], [15.3417, 15.9292], [16.0431, 16.6306]]
    sat_select = [7, 8, 9]
    xlim = [-68.0, -42.0]
    ylim1 = [0.0, 3.0]
    ylim2 = [0.0, 8.0]
    dy1 = 0.5
    dy2 = 2

    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#


    # read in all logic data
    #/----------------------------------------------------------------------------\#
    fname_logic = 'data/%s-LOGIC_%s_%s_RA.h5' % (_mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #\----------------------------------------------------------------------------/#


    # read collocated satellite data
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #\----------------------------------------------------------------------------/#


    # selected stacked legs
    #/----------------------------------------------------------------------------\#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0])&(data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #\----------------------------------------------------------------------------/#


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(18, 5))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.plot(data_hsk[vname_x], data_hsk['alt']/1000.0, color='gray', lw=0.5, alpha=1.0)

        for i in range(len(tmhr_ranges_select)):
            color = colors1[i]

            text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
            text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
            ax1.scatter(data_hsk[vname_x][logics_select[i]], data_hsk['alt'][logics_select[i]]/1000.0, color=color, s=20, lw=0.0, alpha=1.0)
            ax1.text(data_hsk[vname_x][logics_select[i]][0], data_hsk['alt'][logics_select[i]][0]/1000.0, text1, color=color, fontsize=12, alpha=1.0, va='bottom', ha='center')
            ax1.text(data_hsk[vname_x][logics_select[i]][-1], data_hsk['alt'][logics_select[i]][-1]/1000.0, text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center')

        # ax1.axvspan(-52.3248, -40.0, color='b', lw=0.0, alpha=0.5, zorder=0)
        # ax1.axvspan(-51.7540, -40.0, color='g', lw=0.0, alpha=0.5, zorder=0)
        # ax1.axvspan(-51.3029, -40.0, color='r', lw=0.0, alpha=0.5, zorder=0)
        ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
        ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
        ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)

        for i, index_sat in enumerate(sat_select):
            color = colors2[i]
            index0 = np.argmin(np.abs(data_sat['sat/jday'][index_sat]-data_hsk['jday']))
            ax1.scatter(data_hsk[vname_x][index0], data_hsk['alt'][index0]/1000.0, fc=color, s=400, marker='*', lw=0.8, ec=color, alpha=1.0, zorder=1000)
            ax1.scatter(data_hsk[vname_x][index0], data_hsk['alt'][index0]/1000.0, fc='white', s=100, marker='*', lw=0.8, ec='k', alpha=1.0, zorder=1001)

            img_tag, sat_tag = fnames[index_sat].split('.')[0].split('_')[-2:]
            text0 = '%s-%s at %s' % (sat_tag, img_tag, er3t.util.jday_to_dtime(data_sat['sat/jday'][index_sat]).strftime('%H:%M'))
            ax1.text(data_hsk[vname_x][index0], ylim1[1]*0.8, text0, color=color, fontsize=9, alpha=1.0, va='top', ha='center')
            ax1.axvline(data_hsk[vname_x][index0], color=color, ls='--', lw=1.0)

            logic0 = logics_select[1] &  (data_sat['sat/cot'][:, index_sat]>0.0)
            ax2.scatter(data_hsk[vname_x][logic0], data_sat['sat/cot'][logic0, index_sat], s=40, fc='none', lw=0.8, ec=color, marker='D', alpha=0.5)

            logic1 = logics_select[1] &  (data_sat['sat/ctp'][:, index_sat]==1.0)
            ax2.scatter(data_hsk[vname_x][logic1], np.repeat(ylim2[1]*0.98-0.01*i*ylim2[1], logic1.sum()), s=2, fc=color, lw=0.0, ec=color, marker='D')

        # logic_cloud = logic_select&data_logic['logic_cloud']
        # y_cloud = np.ones_like(data_hsk['tmhr'])
        # y_cloud[~logic_cloud] = np.nan

        # logic_clear = logic_select&data_logic['logic_clear']
        # y_clear = np.zeros_like(data_hsk['tmhr'])
        # y_clear[~logic_clear] = np.nan

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Longitude [$^\\circ$]')
        ax1.set_ylabel('Altitude [km]')
        ax1.set_title('Stacked Walls on %s' % date.strftime('%Y-%m-%d'), fontsize=24)
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(-70.0, -39.0, 5.0)))
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(ylim1[0], ylim1[1]+0.1, dy1)))

        ax2.set_ylim(ylim2)
        ax2.yaxis.set_major_locator(FixedLocator(np.arange(ylim2[0], ylim2[1]+0.1, dy2)))
        ax2.set_ylabel('Cloud Optical Thickness', rotation=270, labelpad=18)
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#


    return

def figure_cop_wall_20240607(
        ):

    # case specification
    #/----------------------------------------------------------------------------\#
    date = datetime.datetime(2024, 6, 7)
    tmhr_ranges_select = [[15.3400, 15.7583], [15.8403, 16.2653]]
    sat_select = [4, 5]
    xlim = [-55.0, -40.0]
    ylim1 = [0.0, 1.0]
    ylim2 = [0.0, 20.0]
    dy1 = 0.2
    dy2 = 5

    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#


    # read in all logic data
    #/----------------------------------------------------------------------------\#
    fname_logic = 'data/%s-LOGIC_%s_%s_RA.h5' % (_mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #\----------------------------------------------------------------------------/#


    # read collocated satellite data
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #\----------------------------------------------------------------------------/#


    # selected stacked legs
    #/----------------------------------------------------------------------------\#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0])&(data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #\----------------------------------------------------------------------------/#


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(18, 5))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.plot(data_hsk[vname_x], data_hsk['alt']/1000.0, color='gray', lw=0.5, alpha=1.0)

        for i in range(len(tmhr_ranges_select)):
            color = colors1[i]

            text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
            text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
            ax1.scatter(data_hsk[vname_x][logics_select[i]], data_hsk['alt'][logics_select[i]]/1000.0, color=color, s=20, lw=0.0, alpha=1.0)
            ax1.text(data_hsk[vname_x][logics_select[i]][0], data_hsk['alt'][logics_select[i]][0]/1000.0, text1, color=color, fontsize=12, alpha=1.0, va='bottom', ha='center')
            ax1.text(data_hsk[vname_x][logics_select[i]][-1], data_hsk['alt'][logics_select[i]][-1]/1000.0, text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center')

        # ax1.axvspan(-52.3248, -40.0, color='b', lw=0.0, alpha=0.5, zorder=0)
        # ax1.axvspan(-51.7540, -40.0, color='g', lw=0.0, alpha=0.5, zorder=0)
        # ax1.axvspan(-51.3029, -40.0, color='r', lw=0.0, alpha=0.5, zorder=0)
        # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)

        for i, index_sat in enumerate(sat_select):
            color = colors2[i]
            index0 = np.argmin(np.abs(data_sat['sat/jday'][index_sat]-data_hsk['jday']))
            ax1.scatter(data_hsk[vname_x][index0], data_hsk['alt'][index0]/1000.0, fc=color, s=400, marker='*', lw=0.8, ec=color, alpha=1.0, zorder=1000)
            ax1.scatter(data_hsk[vname_x][index0], data_hsk['alt'][index0]/1000.0, fc='white', s=100, marker='*', lw=0.8, ec='k', alpha=1.0, zorder=1001)

            img_tag, sat_tag = fnames[index_sat].split('.')[0].split('_')[-2:]
            text0 = '%s-%s at %s' % (sat_tag, img_tag, er3t.util.jday_to_dtime(data_sat['sat/jday'][index_sat]).strftime('%H:%M'))
            ax1.text(data_hsk[vname_x][index0], ylim1[1]*0.8, text0, color=color, fontsize=9, alpha=1.0, va='top', ha='center')
            ax1.axvline(data_hsk[vname_x][index0], color=color, ls='--', lw=1.0)

            logic0 = logics_select[1] &  (data_sat['sat/cot'][:, index_sat]>0.0)
            ax2.scatter(data_hsk[vname_x][logic0], data_sat['sat/cot'][logic0, index_sat], s=40, fc='none', lw=0.8, ec=color, marker='D', alpha=0.5)

            logic1 = logics_select[1] &  (data_sat['sat/ctp'][:, index_sat]==1.0)
            ax2.scatter(data_hsk[vname_x][logic1], np.repeat(ylim2[1]*0.98-0.01*i*ylim2[1], logic1.sum()), s=2, fc=color, lw=0.0, ec=color, marker='D')

        # logic_cloud = logic_select&data_logic['logic_cloud']
        # y_cloud = np.ones_like(data_hsk['tmhr'])
        # y_cloud[~logic_cloud] = np.nan

        # logic_clear = logic_select&data_logic['logic_clear']
        # y_clear = np.zeros_like(data_hsk['tmhr'])
        # y_clear[~logic_clear] = np.nan

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Longitude [$^\\circ$]')
        ax1.set_ylabel('Altitude [km]')
        ax1.set_title('Stacked Walls on %s' % date.strftime('%Y-%m-%d'), fontsize=24)
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(-70.0, -39.0, 5.0)))
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(ylim1[0], ylim1[1]+0.1, dy1)))

        ax2.set_ylim(ylim2)
        ax2.yaxis.set_major_locator(FixedLocator(np.arange(ylim2[0], ylim2[1]+0.1, dy2)))
        ax2.set_ylabel('Cloud Optical Thickness', rotation=270, labelpad=18)
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#


    return

def figure_cop_wall_20240611(
        ):

    # case specification
    #/----------------------------------------------------------------------------\#
    date = datetime.datetime(2024, 6, 11)
    tmhr_ranges_select = [[13.9111, 14.3417], [15.3528, 15.7139]]
    sat_select = [7, 8, 9, 10]
    xlim = [-72.0, -50.0]
    ylim1 = [0.0, 5.0]
    ylim2 = [0.0, 50.0]
    dy1 = 1.0
    dy2 = 10

    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#


    # read in all logic data
    #/----------------------------------------------------------------------------\#
    fname_logic = 'data/%s-LOGIC_%s_%s_RA.h5' % (_mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #\----------------------------------------------------------------------------/#


    # read collocated satellite data
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #\----------------------------------------------------------------------------/#


    # selected stacked legs
    #/----------------------------------------------------------------------------\#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0])&(data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #\----------------------------------------------------------------------------/#


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(18, 5))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.plot(data_hsk[vname_x], data_hsk['alt']/1000.0, color='gray', lw=0.5, alpha=1.0)

        for i in range(len(tmhr_ranges_select)):
            color = colors1[i]

            text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
            text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
            ax1.scatter(data_hsk[vname_x][logics_select[i]], data_hsk['alt'][logics_select[i]]/1000.0, color=color, s=20, lw=0.0, alpha=1.0)
            ax1.text(data_hsk[vname_x][logics_select[i]][0], data_hsk['alt'][logics_select[i]][0]/1000.0, text1, color=color, fontsize=12, alpha=1.0, va='bottom', ha='center')
            ax1.text(data_hsk[vname_x][logics_select[i]][-1], data_hsk['alt'][logics_select[i]][-1]/1000.0, text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center')

        # ax1.axvspan(-52.3248, -40.0, color='b', lw=0.0, alpha=0.5, zorder=0)
        # ax1.axvspan(-51.7540, -40.0, color='g', lw=0.0, alpha=0.5, zorder=0)
        # ax1.axvspan(-51.3029, -40.0, color='r', lw=0.0, alpha=0.5, zorder=0)
        # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)

        for i, index_sat in enumerate(sat_select):
            color = colors2[i]
            index0 = np.argmin(np.abs(data_sat['sat/jday'][index_sat]-data_hsk['jday']))
            ax1.scatter(data_hsk[vname_x][index0], data_hsk['alt'][index0]/1000.0, fc=color, s=400, marker='*', lw=0.8, ec=color, alpha=1.0, zorder=1000)
            ax1.scatter(data_hsk[vname_x][index0], data_hsk['alt'][index0]/1000.0, fc='white', s=100, marker='*', lw=0.8, ec='k', alpha=1.0, zorder=1001)

            img_tag, sat_tag = fnames[index_sat].split('.')[0].split('_')[-2:]
            text0 = '%s-%s at %s' % (sat_tag, img_tag, er3t.util.jday_to_dtime(data_sat['sat/jday'][index_sat]).strftime('%H:%M'))
            ax1.text(data_hsk[vname_x][index0], ylim1[1]*0.8, text0, color=color, fontsize=9, alpha=1.0, va='top', ha='center')
            ax1.axvline(data_hsk[vname_x][index0], color=color, ls='--', lw=1.0)

            logic0 = logics_select[1] &  (data_sat['sat/cot'][:, index_sat]>0.0)
            ax2.scatter(data_hsk[vname_x][logic0], data_sat['sat/cot'][logic0, index_sat], s=40, fc='none', lw=0.8, ec=color, marker='D', alpha=0.5)

            logic1 = logics_select[1] &  (data_sat['sat/ctp'][:, index_sat]==1.0)
            ax2.scatter(data_hsk[vname_x][logic1], np.repeat(ylim2[1]*0.98-0.01*i*ylim2[1], logic1.sum()), s=2, fc=color, lw=0.0, ec=color, marker='D')

        # logic_cloud = logic_select&data_logic['logic_cloud']
        # y_cloud = np.ones_like(data_hsk['tmhr'])
        # y_cloud[~logic_cloud] = np.nan

        # logic_clear = logic_select&data_logic['logic_clear']
        # y_clear = np.zeros_like(data_hsk['tmhr'])
        # y_clear[~logic_clear] = np.nan

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Longitude [$^\\circ$]')
        ax1.set_ylabel('Altitude [km]')
        ax1.set_title('Stacked Walls on %s' % date.strftime('%Y-%m-%d'), fontsize=24)
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(-70.0, -39.0, 5.0)))
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(ylim1[0], ylim1[1]+0.1, dy1)))

        ax2.set_ylim(ylim2)
        ax2.yaxis.set_major_locator(FixedLocator(np.arange(ylim2[0], ylim2[1]+0.1, dy2)))
        ax2.set_ylabel('Cloud Optical Thickness', rotation=270, labelpad=18)
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#


    return



def figure_cop_flt_trk_20240605(
        ):

    # case specification
    #/----------------------------------------------------------------------------\#
    extent = [-68, -42, 83.0, 84.1]
    sizes1 = [50, 20, 4]

    date = datetime.datetime(2024, 6, 5)

    tmhr_ranges_select = [[14.5806, 15.2472], [15.3417, 15.9292], [16.0431, 16.6306]]
    sat_select = [7, 8, 9]
    xlim = [-68.0, -42.0]
    ylim1 = [0.0, 3.0]
    ylim2 = [0.0, 8.0]
    dy1 = 0.5
    dy2 = 2

    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#


    # read in all logic data
    #/----------------------------------------------------------------------------\#
    fname_logic = 'data/%s-LOGIC_%s_%s_RA.h5' % (_mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #\----------------------------------------------------------------------------/#


    # read collocated satellite data
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #\----------------------------------------------------------------------------/#


    # selected stacked legs
    #/----------------------------------------------------------------------------\#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0])&(data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #\----------------------------------------------------------------------------/#


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        proj0 = ccrs.Orthographic(
                central_longitude=((extent[0]+extent[1])/2.0),
                central_latitude=((extent[2]+extent[3])/2.0),
                )
        plt.close('all')
        fig = plt.figure(figsize=(18, 12))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111, projection=proj0)

        # read satellite granule
        #/----------------------------------------------------------------------------\#
        fname = 'data/arcsix/sat-data/%s/%s' % (date_s, fnames[sat_select[1]])
        f = Dataset(fname, 'r')
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
        #\----------------------------------------------------------------------------/#

        cot_s0 = cot_s.copy()
        cot_s0[...] = 255.0
        # cs = ax1.pcolormesh(lon_s, lat_s, cot_s0, cmap='gray', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=1.0)
        cot_s[cot_s<=0.0] = np.nan
        ctp_s[ctp_s==1.0] = np.nan
        cs = ax1.pcolormesh(lon_s, lat_s,  ctp_s, cmap='viridis', vmin=0.0, vmax=5.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
        cs = ax1.pcolormesh(lon_s, lat_s,  cot_s, cmap='jet', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)

        ax1.plot(data_hsk['lon'], data_hsk['lat'], lw=1.0, color='k', transform=ccrs.PlateCarree(), zorder=1)
        for i in range(len(tmhr_ranges_select)):
            color = colors1[i]

            text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
            text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
            ax1.scatter(data_hsk['lon'][logics_select[i]], data_hsk['lat'][logics_select[i]], color=color, s=sizes1[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
            # ax1.text(data_hsk['lon'][logics_select[i]][0], data_hsk['lat'][logics_select[i]][0], text1, color=color, fontsize=12, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
            # ax1.text(data_hsk['lon'][logics_select[i]][-1], data_hsk['lat'][logics_select[i]][-1], text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())

        ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
        ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
        ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)

        ax1.set_title(os.path.basename(fname).replace('.nc', '').replace('CLDPROP_L2_', ''), fontsize=24, y=1.01, color=colors2[1])
        ax1.coastlines(resolution='10m', color='gray', lw=0.5)
        g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
        g1.xlocator = FixedLocator(np.arange(-180, 181, 5.0))
        g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2))
        g1.top_labels = False
        g1.right_labels = False

        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

    return

def figure_cop_flt_trk_20240607(
        ):

    # case specification
    #/----------------------------------------------------------------------------\#
    extent = [-55, -40, 83.4, 85.2]
    sizes1 = [50, 20, 4]

    date = datetime.datetime(2024, 6, 7)
    tmhr_ranges_select = [[15.3400, 15.7583], [15.8403, 16.2653]]
    sat_select = [4, 5]
    xlim = [-55.0, -40.0]
    ylim1 = [0.0, 1.0]
    ylim2 = [0.0, 20.0]
    dy1 = 0.2
    dy2 = 5

    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#


    # read in all logic data
    #/----------------------------------------------------------------------------\#
    fname_logic = 'data/%s-LOGIC_%s_%s_RA.h5' % (_mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #\----------------------------------------------------------------------------/#


    # read collocated satellite data
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #\----------------------------------------------------------------------------/#


    # selected stacked legs
    #/----------------------------------------------------------------------------\#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0])&(data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #\----------------------------------------------------------------------------/#


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        proj0 = ccrs.Orthographic(
                central_longitude=((extent[0]+extent[1])/2.0),
                central_latitude=((extent[2]+extent[3])/2.0),
                )
        plt.close('all')
        fig = plt.figure(figsize=(18, 12))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111, projection=proj0)

        # read satellite granule
        #/----------------------------------------------------------------------------\#
        fname = 'data/arcsix/sat-data/%s/%s' % (date_s, fnames[sat_select[1]])
        f = Dataset(fname, 'r')
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
        #\----------------------------------------------------------------------------/#

        cot_s0 = cot_s.copy()
        cot_s0[...] = 255.0
        # cs = ax1.pcolormesh(lon_s, lat_s, cot_s0, cmap='gray', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=1.0)
        cot_s[cot_s<=0.0] = np.nan
        ctp_s[ctp_s==1.0] = np.nan
        cs = ax1.pcolormesh(lon_s, lat_s,  ctp_s, cmap='viridis', vmin=0.0, vmax=5.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
        cs = ax1.pcolormesh(lon_s, lat_s,  cot_s, cmap='jet', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)

        ax1.plot(data_hsk['lon'], data_hsk['lat'], lw=1.0, color='k', transform=ccrs.PlateCarree(), zorder=1)
        for i in range(len(tmhr_ranges_select)):
            color = colors1[i]

            text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
            text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
            ax1.scatter(data_hsk['lon'][logics_select[i]], data_hsk['lat'][logics_select[i]], color=color, s=sizes1[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
            # ax1.text(data_hsk['lon'][logics_select[i]][0], data_hsk['lat'][logics_select[i]][0], text1, color=color, fontsize=12, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
            # ax1.text(data_hsk['lon'][logics_select[i]][-1], data_hsk['lat'][logics_select[i]][-1], text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())

        # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)

        ax1.set_title(os.path.basename(fname).replace('.nc', '').replace('CLDPROP_L2_', ''), fontsize=24, y=1.01, color=colors2[1])
        ax1.coastlines(resolution='10m', color='gray', lw=0.5)
        g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
        g1.xlocator = FixedLocator(np.arange(-180, 181, 5.0))
        g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2))
        g1.top_labels = False
        g1.right_labels = False

        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

    return

def figure_cop_flt_trk_20240611(
        ):

    # case specification
    #/----------------------------------------------------------------------------\#
    extent = [-72, -50, 83.4, 84.4]
    sizes1 = [50, 20, 4]

    date = datetime.datetime(2024, 6, 11)
    tmhr_ranges_select = [[13.9111, 14.3417], [15.3528, 15.7139]]
    sat_select = [7, 8, 9, 10]
    xlim = [-72.0, -50.0]
    ylim1 = [0.0, 5.0]
    ylim2 = [0.0, 50.0]
    dy1 = 1.0
    dy2 = 10

    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#

    date_s = date.strftime('%Y%m%d')

    # read aircraft housekeeping data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-HSK_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_hsk = er3t.util.load_h5(fname_hsk)
    #\----------------------------------------------------------------------------/#


    # read in all logic data
    #/----------------------------------------------------------------------------\#
    fname_logic = '/data/hong/mygit/ssfr/examples/data/%s-LOGIC_%s_%s_RA.h5' % (_mission_.upper(), _platform_.upper(), date_s)
    data_logic = er3t.util.load_h5(fname_logic)
    #\----------------------------------------------------------------------------/#


    # read collocated satellite data
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/%s-SAT-CLD_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _platform_.upper(), date_s)
    data_sat = er3t.util.load_h5(fname_sat)

    f = h5py.File(fname_sat, 'r')
    fnames = f['sat/jday'].attrs['description'].split('\n')
    f.close()
    #\----------------------------------------------------------------------------/#


    # selected stacked legs
    #/----------------------------------------------------------------------------\#
    logic_select = np.repeat(False, data_hsk['tmhr'].size)
    logics_select = []
    for tmhr_range in tmhr_ranges_select:
        logic_select0 = (data_hsk['tmhr']>=tmhr_range[0])&(data_hsk['tmhr']<=tmhr_range[1])
        logics_select.append(logic_select0)
        logic_select[logic_select0] = True
    #\----------------------------------------------------------------------------/#


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        proj0 = ccrs.Orthographic(
                central_longitude=((extent[0]+extent[1])/2.0),
                central_latitude=((extent[2]+extent[3])/2.0),
                )
        plt.close('all')
        fig = plt.figure(figsize=(18, 12))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111, projection=proj0)

        # read satellite granule
        #/----------------------------------------------------------------------------\#
        fname = '/data/hong/mygit/ssfr/examples/data/arcsix/sat-data/%s/%s' % (date_s, fnames[sat_select[1]])
        f = Dataset(fname, 'r')
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
        #\----------------------------------------------------------------------------/#

        cot_s0 = cot_s.copy()
        cot_s0[...] = 255.0
        # cs = ax1.pcolormesh(lon_s, lat_s, cot_s0, cmap='gray', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=1.0)
        cot_s[cot_s<=0.0] = np.nan
        ctp_s[ctp_s==1.0] = np.nan
        cs = ax1.pcolormesh(lon_s, lat_s,  ctp_s, cmap='viridis', vmin=0.0, vmax=5.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
        cs = ax1.pcolormesh(lon_s, lat_s,  cot_s, cmap='jet', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)

        ax1.plot(data_hsk['lon'], data_hsk['lat'], lw=1.0, color='k', transform=ccrs.PlateCarree(), zorder=1)
        for i in range(len(tmhr_ranges_select)):
            color = colors1[i]

            text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
            text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
            ax1.scatter(data_hsk['lon'][logics_select[i]], data_hsk['lat'][logics_select[i]], color=color, s=sizes1[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
            # ax1.text(data_hsk['lon'][logics_select[i]][0], data_hsk['lat'][logics_select[i]][0], text1, color=color, fontsize=12, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
            # ax1.text(data_hsk['lon'][logics_select[i]][-1], data_hsk['lat'][logics_select[i]][-1], text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())

        # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)

        ax1.set_title(os.path.basename(fname).replace('.nc', '').replace('CLDPROP_L2_', ''), fontsize=24, y=1.01, color=colors2[1])
        ax1.coastlines(resolution='10m', color='gray', lw=0.5)
        g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
        g1.xlocator = FixedLocator(np.arange(-180, 181, 5.0))
        g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2))
        g1.top_labels = False
        g1.right_labels = False

        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

    return



if __name__ == '__main__':

    # figure_01_arcsix1_flight_tracks_all()
    # figure_cop_wall_20240605()
    # figure_cop_wall_20240607()
    # figure_cop_wall_20240611()
    # figure_cop_flt_trk_20240605()
    # figure_cop_flt_trk_20240607()
    figure_cop_flt_trk_20240611()
    sys.exit()

    dates = [
            datetime.datetime(2024, 5, 28), # [✓]
            datetime.datetime(2024, 5, 30), #
            datetime.datetime(2024, 5, 31), #
            datetime.datetime(2024, 6,  3), #
            datetime.datetime(2024, 6,  5), # [✓]
            datetime.datetime(2024, 6,  6), # [✓]
            datetime.datetime(2024, 6,  7), # [✓]
            datetime.datetime(2024, 6, 10), # [✓]
            datetime.datetime(2024, 6, 11), # [✓]
            datetime.datetime(2024, 6, 13), # [✓]
        ]

    # dates = [datetime.datetime(2024, 6, 11)]
    # dates = [datetime.datetime(2024, 6, 5)]

    for date in dates[::-1]:

        # research flights in the Arctic
        #/----------------------------------------------------------------------------\#
        # main_pre_download_satellite_data(date)
        # main_pre_retrieve_satellite_cloud(date)

        # cdata_logic_ocean(date)
        # cdata_logic_all(date)
        sat_test(date)
        print()

        # plot_cop(date)
        # main_pre_rt_sim_flux(date)
        #\----------------------------------------------------------------------------/#

        pass

    sys.exit()

    pass
