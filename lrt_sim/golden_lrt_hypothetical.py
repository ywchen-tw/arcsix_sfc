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
from scipy.signal import convolve
import netCDF4 as nc
import xarray as xr
from collections import defaultdict
import platform
# mpl.use('Agg')


import er3t

from util import gaussian

_mission_      = 'arcsix'
_platform_     = 'p3b'

_hsk_          = 'hsk'
_alp_          = 'alp'
_spns_         = 'spns-a'
_ssfr1_        = 'ssfr-a'
_ssfr2_        = 'ssfr-b'
_cam_          = 'nac'

# _fdir_main_       = 'data/%s/flt-vid' % _mission_
_fdir_main_       = 'data/flt-vid'
_fdir_sat_img_    = 'data/%s/sat-img' % _mission_
_fdir_sat_data_   = 'data/%s/sat' % _mission_
_fdir_cam_img_    = 'data/%s/2024-Spring/p3' % _mission_
_wavelength_      = 555.0

_fdir_sat_img_vn_ = 'data/%s/sat-img-vn' % _mission_

_preferred_region_ = 'ca_archipelago'
_aspect_ = 'equal'

if platform.system() == 'Darwin':
    _fdir_data_ = '/Volumes/argus/field/%s/processed' % _mission_
    _fdir_data_ = 'data/processed' 
    _fdir_general_ = 'data'
    _fdir_tmp_ = './tmp'
elif platform.system() == 'Linux':
    _fdir_data_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data/processed"
    _fdir_general_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data"
    _fdir_tmp_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/tmp"
_fdir_tmp_graph_ = 'tmp-graph_flt-vid'

_title_extra_ = 'ARCSIX RF#1'

   
    

def flt_trk_lrt_para(date=datetime.datetime(2024, 5, 31),
                     zpt_filedir='data/zpt',
                     fnames_zpt=['data/radiosonde/arcsix-THAAO-RSxx_SONDE_20240531183300_RA.ict'],
                     albedo_filedir='data/sfc_alb',
                     fnames_albedo=['data/albedo/data_albedo_20240607_low.h5'],
                     sza=60,
                     Nstreams=4,
                     case_tag='default',
                     levels=None,
                     lw=False,
                     manual_cloud_cbh_cth=[[0.0, 0.0]],
                    #  manual_cloud_cot=[0.0],
                     manual_cloud_cwp=[0.0],
                     manual_cloud_cer=[0.0],
                     overwrite_lrt=True,
                     new_compute=False,
                            ):

    # case specification
    #/----------------------------------------------------------------------------\#
    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#

    date_s = date.strftime('%Y%m%d')
    

    
    xx = np.linspace(-12, 12, 241)
    yy_gaussian_vis = gaussian(xx, 0, 3.82)
    yy_gaussian_nir = gaussian(xx, 0, 5.10)

    
    solver = 'lrt'

    if not lw:
        fname_h5 = '%s/%s-%s-%s-%s-hypothetical-sw.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
        fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_hypothetical'
        fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_hypothetical'
    else:
        fname_h5 = '%s/%s-%s-%s-%s-hypothetical-lw.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
        fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_lw_hypothetical'
        fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_lw_hypothetical'
  
    os.makedirs(fdir_tmp, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    
    if levels is None:
        levels = np.concatenate((np.arange(0.0, 1.61, 0.1),
                                            np.array([1.8, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.])))
    wvl_solar_vis = np.arange(300, 950.1, 1)
    wvl_solar_nir = np.arange(951, 2500.1, 1)
    wvl_solar_coarse = np.concatenate([wvl_solar_vis, wvl_solar_nir])
    effective_wvl = wvl_solar_coarse[np.logical_and(wvl_solar_coarse >= 360, wvl_solar_coarse <= 1990)]
    
    if not os.path.exists(fname_h5) or overwrite_lrt:
         
        atm_z_grid = levels
        z_list = atm_z_grid
        atm_z_grid_str = ' '.join(['%.2f' % z for z in atm_z_grid])
        
        # output_len_cloud = len(manual_cloud_cot) * len(manual_cloud_cbh_cth) * len(manual_cloud_cwp) * len(manual_cloud_cer) * len(fnames_zpt) * len(fnames_albedo)
        output_len_cloud = len(manual_cloud_cbh_cth) * len(manual_cloud_cwp) * len(manual_cloud_cer) * len(fnames_zpt) * len(fnames_albedo)

        output_len_clear = len(fnames_zpt) * len(fnames_albedo)
        
        flux_output = np.zeros(output_len_cloud+output_len_clear)
        

        flux_key_all = []
        
        if os.path.exists(f'{fdir}/flux_down_result_dict.pk') and not new_compute:
            print(f'Loading flux_down_result_dict.pk from {fdir} ...')
            with open(f'{fdir}/flux_down_result_dict.pk', 'rb') as f:
                flux_down_result_dict = pickle.load(f)
            with open(f'{fdir}/flux_down_dir_result_dict.pk', 'rb') as f:
                flux_down_dir_result_dict = pickle.load(f)
            with open(f'{fdir}/flux_down_diff_result_dict.pk', 'rb') as f:
                flux_down_diff_result_dict = pickle.load(f)
            with open(f'{fdir}/flux_up_result_dict.pk', 'rb') as f:
                flux_up_result_dict = pickle.load(f)
                
            flux_key_all.extend(flux_down_result_dict.keys())
            print("flux_down_result_dict keys: ", flux_down_result_dict.keys())
            
        else:
            flux_down_result_dict = {}
            flux_down_dir_result_dict = {}
            flux_down_diff_result_dict = {}
            flux_up_result_dict = {}
            
        
        flux_key = np.zeros_like(flux_output, dtype=object)
        cloudy = 0
        clear = 0
        
        

        
        inits_rad = []
        flux_key_ix = []
        output_list = []
        
        ix = 0
        
        zpt_list = []
        alb_list = []
        cot_list = []
        cwp_list = []
        cer_list = []
        cth_list = []
        cbh_list = []
            
        # start with clear sky
        for fname_zpt in fnames_zpt:
            for fname_albedo in fnames_albedo:
                clear += 1
                # rt initialization
                #/----------------------------------------------------------------------------\#
                lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
                lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, fname_zpt)
                if not lw:
                    
                    lrt_cfg['solar_file'] = 'kurudz_ssfr.dat'
                    lrt_cfg['number_of_streams'] = Nstreams
                    lrt_cfg['mol_abs_param'] = 'reptran coarse'
                    input_dict_extra_general = {
                                        'crs_model': 'rayleigh Bodhaine29',
                                        'albedo_file': f'{albedo_filedir}/{fname_albedo}',
                                        'atm_z_grid': atm_z_grid_str,
                                        'wavelength_grid_file': 'wvl_grid_test.dat',
                                        }
                    Nx_effective = len(effective_wvl)
                    mute_list = ['albedo', 'wavelength', 'spline']
                else:
                    lrt_cfg['number_of_streams'] = Nstreams
                    lrt_cfg['mol_abs_param'] = 'reptran coarse'
                    ch4_file = os.path.join(zpt_filedir, fname_zpt.replace('atm', 'ch4'))
                    input_dict_extra_general = {
                                        'source': 'thermal',
                                        'albedo_add': '0',
                                        'atm_z_grid': atm_z_grid_str,
                                        'mol_file': f'CH4 {ch4_file}',
                                        # 'wavelength_grid_file': 'wvl_grid_thermal.dat',
                                        'wavelength_add' : '4500 42000',
                                        'output_process': 'integrate',
                                        }
                    Nx_effective = 1 # integrate over all wavelengths
                    mute_list = ['albedo', 'wavelength', 'spline', 'source solar', 'slit_function_file']
                #/----------------------------------------------------------------------------/#

                if not lw:
                    dict_key = f'clear {fname_zpt} {fname_albedo} sw'
                else:
                    dict_key = f'clear {fname_zpt} {fname_albedo} lw'
                    
                input_dict_extra = copy.deepcopy(input_dict_extra_general)
                
                
                flux_key[ix] = dict_key
                
                
                # rt setup
                #/----------------------------------------------------------------------------\#
                
                init = er3t.rtm.lrt.lrt_init_mono_flx(
                        input_file  = '%s/input_%04d.txt'  % (fdir_tmp, ix),
                        output_file = '%s/output_%04d.txt' % (fdir_tmp, ix),
                        date        = date,
                        # surface_albedo=0.08,
                        solar_zenith_angle = sza,
                        Nx = Nx_effective,
                        output_altitude    = [0, 'toa'],
                        input_dict_extra   = input_dict_extra.copy(),
                        mute_list          = mute_list,
                        lrt_cfg            = lrt_cfg,
                        cld_cfg            = None,
                        aer_cfg            = None,
                        )
                #\----------------------------------------------------------------------------/#
                inits_rad.append(copy.deepcopy(init))
                output_list.append('%s/output_%04d.txt' % (fdir_tmp, ix))
                flux_key_all.append(dict_key)
                
                
        
                zpt_list.append(fname_zpt)
                alb_list.append(fname_albedo)
                cot_list.append(0.0)
                cwp_list.append(0.0)
                cer_list.append(0.0)
                cth_list.append(0.0)
                cbh_list.append(0.0)
                ix += 1
                
        rho_water = 1000 # kg/m^3
        Q_sca = 2
        # now with clouds
        for fname_zpt in fnames_zpt:
            for fname_albedo in fnames_albedo:
                # for cot_x in manual_cloud_cot:
                for cwp_x in manual_cloud_cwp:
                    for cer_x in manual_cloud_cer:
                        for cbh_x, cth_x in manual_cloud_cbh_cth:
                                cloudy += 1
                                cgt_x = cth_x - cbh_x
                                cot_x = Q_sca * 3 * cwp_x / (4 * rho_water * cer_x * 1e-6)
                                # rt initialization
                                #/----------------------------------------------------------------------------\#
                                lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
                                lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, fname_zpt)
                                if not lw:
                                    
                                    lrt_cfg['solar_file'] = 'kurudz_ssfr.dat'
                                    lrt_cfg['number_of_streams'] = Nstreams
                                    lrt_cfg['mol_abs_param'] = 'reptran coarse'
                                    input_dict_extra_general = {
                                                        'crs_model': 'rayleigh Bodhaine29',
                                                        'albedo_file': f'{albedo_filedir}/{fname_albedo}',
                                                        'atm_z_grid': atm_z_grid_str,
                                                        'wavelength_grid_file': 'wvl_grid_test.dat',
                                                        }
                                    Nx_effective = len(effective_wvl)
                                    mute_list = ['albedo', 'wavelength', 'spline']
                                else:
                                    lrt_cfg['number_of_streams'] = Nstreams
                                    lrt_cfg['mol_abs_param'] = 'reptran coarse'
                                    ch4_file = os.path.join(zpt_filedir, fname_zpt.replace('atm', 'ch4'))
                                    input_dict_extra_general = {
                                                        'source': 'thermal',
                                                        'albedo_add': '0',
                                                        'atm_z_grid': atm_z_grid_str,
                                                        'mol_file': f'CH4 {ch4_file}',
                                                        # 'wavelength_grid_file': 'wvl_grid_thermal.dat',
                                                        'wavelength_add' : '4500 42000',
                                                        'output_process': 'integrate',
                                                        }
                                    Nx_effective = 1 # integrate over all wavelengths
                                    mute_list = ['albedo', 'wavelength', 'spline', 'source solar', 'slit_function_file']
                                #/----------------------------------------------------------------------------/#
            
                
                                cth_ind_cld = bisect.bisect_left(z_list, cth_x)
                                cbh_ind_cld = bisect.bisect_left(z_list, cbh_x)
                
                                fname_cld = f'{fdir_tmp}/cld_{ix:04d}.txt'
                                if os.path.exists(fname_cld):
                                    os.remove(fname_cld)
                    
                                cld_cfg = er3t.rtm.lrt.get_cld_cfg()
                                cld_cfg['cloud_file'] = fname_cld
                                cld_cfg['cloud_altitude'] = z_list[cbh_ind_cld:cth_ind_cld+2]
                                cld_cfg['cloud_effective_radius']  = cer_x
                                cld_cfg['liquid_water_content'] = cwp_x*1000/(cgt_x*1000) # convert kg/m^2 to g/m^3
                                cld_cfg['cloud_optical_thickness'] = cot_x
                
                                if not lw:
                                    dict_key = f'cloud {fname_zpt} {fname_albedo} {cot_x:.3f} {cer_x:.3f} {cwp_x:.3f} {cth_x:.3f} {cbh_x:.3f} sw'
                                else:
                                    dict_key = f'cloud {fname_zpt} {fname_albedo} {cot_x:.3f} {cer_x:.3f} {cwp_x:.3f} {cth_x:.3f} {cbh_x:.3f} lw'


                                input_dict_extra = copy.deepcopy(input_dict_extra_general)
                                flux_key[ix] = dict_key


                                # rt setup
                                #/----------------------------------------------------------------------------\#
                                
                                init = er3t.rtm.lrt.lrt_init_mono_flx(
                                        input_file  = '%s/input_%04d.txt'  % (fdir_tmp, ix),
                                        output_file = '%s/output_%04d.txt' % (fdir_tmp, ix),
                                        date        = date,
                                        solar_zenith_angle = sza,
                                        Nx = Nx_effective,
                                        output_altitude    = [0, 'toa'],
                                        input_dict_extra   = input_dict_extra.copy(),
                                        mute_list          = mute_list,
                                        lrt_cfg            = lrt_cfg,
                                        cld_cfg            = cld_cfg,
                                        aer_cfg            = None,
                                        )
                                #\----------------------------------------------------------------------------/#
                                inits_rad.append(copy.deepcopy(init))
                                output_list.append('%s/output_%04d.txt' % (fdir_tmp, ix))
                                flux_key_all.append(dict_key)
                                
                                zpt_list.append(fname_zpt)
                                alb_list.append(fname_albedo)
                                cot_list.append(cot_x)
                                cwp_list.append(cwp_x)
                                cer_list.append(cer_x)
                                cth_list.append(cth_x)
                                cbh_list.append(cbh_x)
                                ix += 1
                    
        # print('len(inits_rad): ', len(inits_rad))
        # print("flux_key_all: ", flux_key_all)
        # print("flux_key_ix set: ", set(flux_key_ix))
        # print("flux_key_all length: ", len(flux_key_all))
        # print("flux_key_ix length: ", len(flux_key_ix))
        # print("len set(flux_key_ix): ", len(set(flux_key_ix)))
        # print("set(flux_key_ix) == set(flux_key_all): ", set(flux_key_ix) == set(flux_key_all))
        # sys.exit()
        # # Run RT
        print(f"Start running libratran calculations for {fname_h5.replace('.h5', '')} ")
        # #/----------------------------------------------------------------------------\#
        import platform
        if platform.system() == 'Darwin':
            ##### run several libratran calculations in parallel
            if len(inits_rad) > 0:
                print('Running libratran calculations ...')
                # check available CPU cores
                NCPU = os.cpu_count()
                import platform
                if platform.system() == 'Darwin':
                    NCPU -= 2
                er3t.rtm.lrt.lrt_run_mp(inits_rad, Ncpu=NCPU)        
                for i in range(len(inits_rad)):
                    data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                    flux_down_result_dict[flux_key_all[i]] = np.squeeze(data.f_down)
                    flux_down_dir_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_direct)
                    flux_down_diff_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_diffuse)
                    flux_up_result_dict[flux_key_all[i]] = np.squeeze(data.f_up)
        ##### run several libratran calculations one by one
        
        elif platform.system() == 'Linux':
            if len(inits_rad) > 0:
                print('Running libratran calculations ...')
                for i in range(len(inits_rad)):
                    if not os.path.exists(output_list[i]):
                        er3t.rtm.lrt.lrt_run(inits_rad[i])
                    else:
                        if os.path.getsize(output_list[i]) == 0:
                            er3t.rtm.lrt.lrt_run(inits_rad[i])
                    data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                    flux_down_result_dict[flux_key_all[i]] = np.squeeze(data.f_down)
                    flux_down_dir_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_direct)
                    flux_down_diff_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_diffuse)
                    flux_up_result_dict[flux_key_all[i]] = np.squeeze(data.f_up)
        # #\----------------------------------------------------------------------------/#
        ###### delete input, output, cld txt files
        # for prefix in ['input', 'output', 'cld']:
        #     for filename in glob.glob(os.path.join(fdir_tmp, f'{prefix}_*.txt')):
        #         os.remove(filename)

            
        
        # save dict
        status = 'wb'
        with open(f'{fdir}/flux_down_result_dict.pk', status) as f:
            pickle.dump(flux_down_result_dict, f)
        with open(f'{fdir}/flux_down_dir_result_dict.pk', status) as f:
            pickle.dump(flux_down_dir_result_dict, f)
        with open(f'{fdir}/flux_down_diff_result_dict.pk', status) as f:
            pickle.dump(flux_down_diff_result_dict, f)
        with open(f'{fdir}/flux_up_result_dict.pk', status) as f:
            pickle.dump(flux_up_result_dict, f)


        
        flux_output_t = np.zeros(len(range(len(flux_output))))
        
        
        f_down_1d = np.zeros((len(flux_output_t), Nx_effective, 2))
        f_down_dir_1d = np.zeros((len(flux_output_t), Nx_effective, 2))
        f_down_diff_1d = np.zeros((len(flux_output_t), Nx_effective, 2))
        f_up_1d = np.zeros((len(flux_output_t), Nx_effective, 2))
        
        for f_array in [f_down_1d, f_down_dir_1d, f_down_diff_1d, f_up_1d]:
            f_array[...] = np.nan

        
        for ix in range(len(flux_output)):               
            f_down_1d[ix] = flux_down_result_dict[flux_key[ix]]
            f_down_dir_1d[ix] = flux_down_dir_result_dict[flux_key[ix]]
            f_down_diff_1d[ix] = flux_down_diff_result_dict[flux_key[ix]]
            f_up_1d[ix] = flux_up_result_dict[flux_key[ix]]
                

                
        # save rad_2d results
        with h5py.File(fname_h5, 'w') as f:
            f.create_dataset('zpt', data=zpt_list)
            f.create_dataset('alb', data=alb_list)
            f.create_dataset('sza', data=sza)
            f.create_dataset('f_down', data=f_down_1d)
            f.create_dataset('f_down_dir', data=f_down_dir_1d)
            f.create_dataset('f_down_diff', data=f_down_diff_1d)
            f.create_dataset('f_up', data=f_up_1d)
            f.create_dataset('cth', data=cth_list)
            f.create_dataset('cbh', data=cbh_list)
            f.create_dataset('cot', data=cot_list)
            f.create_dataset('cwp', data=cwp_list)
            f.create_dataset('cer', data=cer_list)
    else:
        print('Loading existing libratran results ...')
        with h5py.File(fname_h5, 'r') as f:
            f_zpt = f['zpt'][...]
            f_alb = f['alb'][...]
            f_sza = f['sza'][...]
            f_down_1d = f['f_down'][...]
            f_down_dir_1d = f['f_down_dir'][...]
            f_down_diff_1d = f['f_down_diff'][...]
            f_up_1d = f['f_up'][...]
            f_cth = f['cth'][...]
            f_cbh = f['cbh'][...]
            f_cot = f['cot'][...]
            f_cwp = f['cwp'][...]
            f_cer = f['cer'][...]
            
        #############

    print("Finished libratran calculations.")  
    #\----------------------------------------------------------------------------/#

    return

if __name__ == '__main__':
   
    
    flt_trk_lrt_para(date=datetime.datetime(2024, 5, 31),
                     zpt_filedir=f'{_fdir_general_}/zpt',
                     fnames_zpt=['20240603/atm_profiles_2024063.dat'],
                     albedo_filedir=f'{_fdir_general_}/sfc_alb',
                     fnames_albedo=['sfc_alb_20240607.dat', 'sfc_alb_20240531.dat'],
                     sza=60,
                     Nstreams=4,
                     case_tag='test_cld_alb',
                     levels=np.concatenate((np.arange(0.0, 1.1, 0.1),
                                            np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.2, 3.4, 3.7, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                     lw=False,
                     manual_cloud_cbh_cth=[[0.1, 0.3], [0.1, 0.2], [0., 0.1], [1.0, 1.2], [3.0, 3.2]], # in km
                    #  manual_cloud_cot=[5.0],
                     manual_cloud_cwp=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2], # in kg/m^2
                     manual_cloud_cer=[8.0],
                     overwrite_lrt=True,
                     new_compute=True,
                            )

    pass
