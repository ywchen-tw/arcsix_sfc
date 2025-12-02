"""
Inheritated from Hong Chen (hong.chen@lasp.colorado.edu)
Edited by Yu-Wen Chen (yu-wen.chen@colorado.edu)

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
import platform

if platform.system() == 'Linux':
    # Define the path to your module directory
    # Use os.path.abspath and os.path.join for platform independence
    # module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'util'))

    # Add the directory to the Python search path
    # sys.path.insert(0, module_path)
    sys.path.append("/projects/yuch8913/arcsix_sfc/lrt_sim/")  

import glob
import copy
import time
from collections import OrderedDict
import datetime
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Optional
import h5py
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata, NearestNDInterpolator
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.axes as maxes
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
import bisect
import pandas as pd
import xarray as xr
from collections import defaultdict
import gc
from pyproj import Transformer
# mpl.use('Agg')


import er3t

# from util.util import *
# from util.arcsix_atm import prepare_atmospheric_profile
from util import *

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
    _fdir_data_ = '../data/processed' 
    _fdir_general_ = '../data'
    _fdir_tmp_ = './tmp'
elif platform.system() == 'Linux':
    _fdir_data_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data/processed"
    _fdir_general_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data"
    _fdir_tmp_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/tmp"
_fdir_tmp_graph_ = 'tmp-graph_flt-vid'

_title_extra_ = 'ARCSIX RF#1'

_tmhr_range_ = {
        '20240517': [19.20, 23.00],
        '20240521': [14.80, 17.50],
        }

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

_dates_ = _dates1_


o2a_1_start, o2a_1_end = 748, 780
h2o_1_start, h2o_1_end = 672, 706
h2o_2_start, h2o_2_end = 705, 746
h2o_3_start, h2o_3_end = 884, 996
h2o_4_start, h2o_4_end = 1084, 1175
h2o_5_start, h2o_5_end = 1230, 1286
h2o_6_start, h2o_6_end = 1290, 1509
h2o_7_start, h2o_7_end = 1748, 2050
h2o_8_start, h2o_8_end = 801, 843
final_start, final_end = 2110, 2200

gas_bands = [(o2a_1_start, o2a_1_end), (h2o_1_start, h2o_1_end), (h2o_2_start, h2o_2_end),
                (h2o_3_start, h2o_3_end), (h2o_4_start, h2o_4_end), (h2o_5_start, h2o_5_end),
                (h2o_6_start, h2o_6_end), (h2o_7_start, h2o_7_end), (h2o_8_start, h2o_8_end),
                (final_start, final_end)]


def cre_sim_plot(date=datetime.datetime(2024, 5, 31),
                     tmhr_ranges_select=[[14.10, 14.27]],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                     levels=None,
                     simulation_interval=None, # in minute
                     clear_sky=True,
                     overwrite_lrt=True,
                     manual_cloud=False,
                     manual_cloud_cer=14.4,
                     manual_cloud_cwp=0.06013,
                     manual_cloud_cth=0.945,
                     manual_cloud_cbh=0.344,
                     manual_cloud_cot=6.26,
                    ):
    
    log = logging.getLogger("lrt")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    
    os.makedirs(f'fig/{date_s}', exist_ok=True)
    
    if simulation_interval is not None:
        # split tmhr_ranges_select into smaller intervals
        tmhr_ranges_select_new = []
        for lo, hi in tmhr_ranges_select:
            t_start = lo
            while t_start < hi and t_start < (hi - 0.0167/6):  # 10s
                t_end = min(t_start + simulation_interval/60.0, hi)
                tmhr_ranges_select_new.append([t_start, t_end])
                t_start = t_end
        tmhr_ranges_select = tmhr_ranges_select_new

    
    # read satellite granule
    #/----------------------------------------------------------------------------\#
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)
        
    processed_dir = f'{_fdir_general_}/sfc_alb_combined_smooth_450nm'
    os.makedirs(processed_dir, exist_ok=True)
    
    lon_all = np.array([])
    lat_all = np.array([])
    alt_all = np.array([])
    sza_all = np.array([])
    saa_all = np.array([])
    sfc_T = np.array([])
    
    time_all = np.array([])
    marli_all_h = np.array([])
    marli_all_wvmr = np.array([])
    
    init = True
    alb_iter2_all = []
    
    for i in range(len(tmhr_ranges_select)):
        time_start, time_end = tmhr_ranges_select[i][0], tmhr_ranges_select[i][-1]
        
        fname_cld_obs_info = '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, time_start, time_end)
        print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
        
        processed_file = f'{processed_dir}/sfc_alb_update_{date_s}_{case_tag}_time_{tmhr_ranges_select[0][0]:.3f}_{tmhr_ranges_select[-1][-1]:.3f}.pkl'
        
        if os.path.exists(processed_file):
        
            with open(fname_cld_obs_info, 'rb') as f:
                vars()[f"cld_leg_{i}"] = pickle.load(f)  
                
            
            with open(processed_file, 'rb') as f:
                vars()[f"processed_leg_{i}"] = pickle.load(f)
                
    
    
            alb_wvl = vars()[f"processed_leg_{i}"]['wvl']
            
            time_all = np.concatenate((time_all, vars()[f"processed_leg_{i}"]['time_all']))
            lon_all = np.concatenate((lon_all, vars()[f"processed_leg_{i}"]['lon_all']))
            lat_all = np.concatenate((lat_all, vars()[f"processed_leg_{i}"]['lat_all']))
            alt_all = np.concatenate((alt_all, vars()[f"processed_leg_{i}"]['alt_all']))
            sza_all = np.concatenate((sza_all, vars()[f"cld_leg_{i}"]['sza']))
            saa_all = np.concatenate((saa_all, vars()[f"cld_leg_{i}"]['saa']))
            sfc_T = np.concatenate((sfc_T, vars()[f"cld_leg_{i}"]['kt19_sfc_T']))
            
            if vars()[f"cld_leg_{i}"]['marli_h'] is not None:
                marli_all_h = np.concatenate((marli_all_h, vars()[f"cld_leg_{i}"]['marli_h']))
                marli_all_wvmr = np.concatenate((marli_all_wvmr, vars()[f"cld_leg_{i}"]['marli_wvmr']))
            
            if init:
                alb_iter2_all = vars()[f"processed_leg_{i}"]['alb_iter2_all']
                init = False
            else:
                alb_iter2_all = np.vstack((alb_iter2_all, vars()[f"processed_leg_{i}"]['alb_iter2_all']))
            
        else:
            print(f"Processed file {processed_file} not found. Skipping leg {i}.")
        
    
    
       
    lon_avg = np.round(np.mean(lon_all), 2)
    lat_avg = np.round(np.mean(lat_all), 2)
    lon_min, lon_max = np.round(np.min(lon_all), 2), np.round(np.max(lon_all), 2)
    lat_min, lat_max = np.round(np.min(lat_all), 2), np.round(np.max(lat_all), 2)
    alt_avg = np.round(np.nanmean(alt_all), 2)  # in km
    sza_avg = np.round(np.nanmean(sza_all), 2)
    saa_avg = np.round(np.nanmean(saa_all), 2)
    sfc_T_avg = np.round(np.nanmean(sfc_T), 2)

        
   
    
    if marli_all_h.size == 0:
        cld_marli = {'marli_h': None,
                     'marli_wvmr': None}
    else:
        marli_h_set_sorted = np.sort(set(marli_all_h))
        marli_wvmr_avg = []
        for h in marli_h_set_sorted:
            marli_wvmr_avg.append(np.nanmean(marli_all_wvmr[marli_all_h == h]))
        marli_wvmr_avg = np.array(marli_wvmr_avg)
        cld_marli = {'marli_h': marli_h_set_sorted,
                    'marli_wvmr': marli_wvmr_avg}
        
        
    if clear_sky:
        fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_clear'
        fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
    else:
        fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_sat_cloud'
        fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'

    output_csv_name_sw = f'{fdir}/ssfr_simu_flux_{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}_alt-{alt_avg:.2f}km_cre_sw.csv'
    output_csv_name_lw = f'{fdir}/ssfr_simu_flux_{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}_alt-{alt_avg:.2f}km_cre_lw.csv'

    os.makedirs(fdir_tmp, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    

    # read csv and extract simulated fluxes
    with open(output_csv_name_sw, 'r') as f:
        df_sw = pd.read_csv(f)
        
    with open(output_csv_name_lw, 'r') as f:
        df_lw = pd.read_csv(f)
    
    cot_list = df_sw['cot'].values
    cwp_list = df_sw['cwp'].values
    cer_list = df_sw['cer'].values
    cth_list = df_sw['cth'].values
    cbh_list = df_sw['cbh'].values
    Fup_sfc_sw = df_sw['Fup_sfc'].values
    Fdn_sfc_sw = df_sw['Fdn_sfc'].values
    Fup_sfc_lw = df_lw['Fup_sfc'].values
    Fdn_sfc_lw = df_lw['Fdn_sfc'].values
    
    Fup_sfc_lw *= 1000  # convert kW/m2 to W/m2
    Fdn_sfc_lw *= 1000  # convert kW/m2 to W/m2
    
    cot0_ind = cot_list == 0.0
    
    F_sfc_sw = Fdn_sfc_sw - Fup_sfc_sw
    F_sfc_lw = Fdn_sfc_lw - Fup_sfc_lw
    F_sfc_sw_clear = F_sfc_sw[cot0_ind]
    F_sfc_lw_clear = F_sfc_lw[cot0_ind]
    F_sfc_sw_cre = F_sfc_sw - F_sfc_sw_clear
    F_sfc_lw_cre = F_sfc_lw - F_sfc_lw_clear
    cot_cre = cot_list
    cwp_cre = cwp_list
    F_sfc_net_cre = F_sfc_sw_cre + F_sfc_lw_cre
    
    print("cwp_cre:", cwp_cre)
    print("F_sfc_sw_cre:", F_sfc_sw_cre)
    print("F_sfc_lw_cre:", F_sfc_lw_cre)
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cwp_cre, F_sfc_sw_cre, '-o', label='SW CRE')
    ax.plot(cwp_cre, F_sfc_lw_cre, '-o', label='LW CRE')
    ax.plot(cwp_cre, F_sfc_net_cre, '-o', label='Net CRE')
    ax.set_xlabel('Cloud Liquid Water Path (g/m2)', fontsize=14)
    ax.set_ylabel('Surface CRE (W/m2)', fontsize=14)
    ax.set_title(f'Surface CRE vs. LWP on {date_s}', fontsize=16)
    ax.hlines(0, xmin=0, xmax=np.max(cwp_cre), colors='gray', linestyles='dashed')
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(f'fig/{date_s}/surface_cre_vs_lwp_{date_s}_{case_tag}.png', dpi=300)
    
    
    
    

    print("Finished libratran calculations.")  
    #\----------------------------------------------------------------------------/#

    return

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


if __name__ == '__main__':

    
    dir_fig = './fig'
    os.makedirs(dir_fig, exist_ok=True)
    
    config = FlightConfig(mission='ARCSIX',
                            platform='P3B',
                            data_root=_fdir_data_,
                            root_mac=_fdir_general_,
                            root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data',)
    
    """
    # IMPORTANT
    # need to run arcsix_gas_insitu.py first to generate gas files for each date
    """


    # surface albedo derivation
    # --------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------

    atm_corr_overwrite_lrt = True
    lw = True  # shortwave
    


    cre_sim_plot(date=datetime.datetime(2024, 6, 3),
                    tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
                                        ],
                    case_tag='cloudy_atm_corr_1',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
                                            np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_cloud_cer=13.0 ,
                    manual_cloud_cwp=77.82,
                    manual_cloud_cth=1.93,
                    manual_cloud_cbh=1.41,
                    manual_cloud_cot=21.27,
                    )


    # for iter in range(3):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),
    #                     tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
    #                                            np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=False,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=True,
    #                     manual_cloud_cer=7.0,
    #                     manual_cloud_cwp=113.65,
    #                     manual_cloud_cth=1.91,
    #                     manual_cloud_cbh=0.50,
    #                     manual_cloud_cot=24.31,
    #                     iter=iter,
    #                     )
        
    
    
    
    # done   
    # # for iter in range(3):
    # #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 7),
    # #                     tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
    # #                                         ],
    # #                     case_tag='cloudy_atm_corr',
    # #                     config=config,
    # #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
    # #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    # #                                            np.arange(5.0, 10.1, 2.5),
    # #                                            np.array([15, 20, 30., 40., 45.]))),
    # #                     simulation_interval=0.5,
    # #                     clear_sky=False,
    # #                     overwrite_lrt=atm_corr_overwrite_lrt,
    # #                     manual_cloud=True,
    # #                     manual_cloud_cer=6.7,
    # #                     manual_cloud_cwp=26.96,
    # #                     manual_cloud_cth=0.43,
    # #                     manual_cloud_cbh=0.15,
    # #                     manual_cloud_cot=6.02,
    # #                     iter=iter,
    # #                     )
    
    
    # done
    # # for iter in range(3):
    # #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    # #                     tmhr_ranges_select=[[14.109, 14.140], # 100m, cloudy
    # #                                         ],
    # #                     case_tag='cloudy_atm_corr_1',
    # #                     config=config,
    # #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.4, 0.52, 0.6, 0.8, 1.0,]),
    # #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    # #                                            np.arange(5.0, 10.1, 2.5),
    # #                                            np.array([15, 20, 30., 40., 45.]))),
    # #                     simulation_interval=0.5,
    # #                     clear_sky=False,
    # #                     overwrite_lrt=atm_corr_overwrite_lrt,
    # #                     manual_cloud=True,
    # #                     manual_cloud_cer=17.4,
    # #                     manual_cloud_cwp=90.51,
    # #                     manual_cloud_cth=0.52,
    # #                     manual_cloud_cbh=0.15,
    # #                     manual_cloud_cot=7.82,
    # #                     iter=iter,
    # #                     )
    
    # done
    # # for iter in range(3):
    # #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    # #                     tmhr_ranges_select=[[15.834, 15.883], # 100m, cloudy
    # #                                         ],
    # #                     case_tag='cloudy_atm_corr_2',
    # #                     config=config,
    # #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.28, 0.3, 0.5, 0.58, 0.8, 1.0,]),
    # #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    # #                                            np.arange(5.0, 10.1, 2.5),
    # #                                            np.array([15, 20, 30., 40., 45.]))),
    # #                     simulation_interval=0.5,
    # #                     clear_sky=False,
    # #                     overwrite_lrt=atm_corr_overwrite_lrt,
    # #                     manual_cloud=True,
    # #                     manual_cloud_cer=22.4,
    # #                     manual_cloud_cwp=35.6 ,
    # #                     manual_cloud_cth=0.58,
    # #                     manual_cloud_cbh=0.28,
    # #                     manual_cloud_cot=2.39,
    # #                     iter=iter,
    # #                     )
        
    # done
    # # for iter in range(3):
    # #     flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),
    # #                     tmhr_ranges_select=[[16.043, 16.067], # 100-200m, cloudy
    # #                                         ],
    # #                     case_tag='cloudy_atm_corr_3',
    # #                     config=config,
    # #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.38, 0.5, 0.68, 0.8, 1.0,]),
    # #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    # #                                            np.arange(5.0, 10.1, 2.5),
    # #                                            np.array([15, 20, 30., 40., 45.]))),
    # #                     simulation_interval=0.5,
    # #                     clear_sky=False,
    # #                     overwrite_lrt=atm_corr_overwrite_lrt,
    # #                     manual_cloud=True,
    # #                     manual_cloud_cer=8.9,
    # #                     manual_cloud_cwp=21.29,
    # #                     manual_cloud_cth=0.68,
    # #                     manual_cloud_cbh=0.38,
    # #                     manual_cloud_cot=3.59,
    # #                     iter=iter,
    # #                     )
    

    # done
    # # for iter in range(3):
    # #     flt_trk_atm_corr(date=datetime.datetime(2024, 7, 25),
    # #                     tmhr_ranges_select=[[15.094, 15.300], # 100m, some low clouds or fog below
    # #                                         ],
    # #                     case_tag='cloudy_atm_corr',
    # #                     config=config,
    # #                     levels=np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,]),
    # #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    # #                                            np.arange(5.0, 10.1, 2.5),
    # #                                            np.array([15, 20, 30., 40., 45.]))),
    # #                     simulation_interval=0.5,
    # #                     clear_sky=False,
    # #                     overwrite_lrt=atm_corr_overwrite_lrt,
    # #                     manual_cloud=True,
    # #                     manual_cloud_cer=11.4,
    # #                     manual_cloud_cwp=9.94,
    # #                     manual_cloud_cth=0.30,
    # #                     manual_cloud_cbh=0.16,
    # #                     manual_cloud_cot=1.31,
    # #                     iter=iter,
    # #                     )
    
    # done
    # # for iter in range(3):
    # #     flt_trk_atm_corr(date=datetime.datetime(2024, 7, 25),
    # #                     tmhr_ranges_select=[[15.881, 15.903], # 200-500m
    # #                                         ],
    # #                     case_tag='cloudy_atm_corr_2',
    # #                     config=config,
    # #                     levels=np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,]),
    # #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    # #                                            np.arange(5.0, 10.1, 2.5),
    # #                                            np.array([15, 20, 30., 40., 45.]))),
    # #                     simulation_interval=0.5,
    # #                     clear_sky=False,
    # #                     overwrite_lrt=atm_corr_overwrite_lrt,
    # #                     manual_cloud=True,
    # #                     manual_cloud_cer=11.4,
    # #                     manual_cloud_cwp=9.94,
    # #                     manual_cloud_cth=0.30,
    # #                     manual_cloud_cbh=0.16,
    # #                     manual_cloud_cot=1.31,
    # #                     iter=iter,
    # #                     )
        

    # done
    # # for iter in range(3):
    # #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),
    # #                     tmhr_ranges_select=[[13.344, 13.763], # 100m, cloudy
    # #                                         ],
    # #                     case_tag='clear_atm_corr_1',
    # #                     config=config,
    # #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.65, 0.69, 0.78, 1.0,]),
    # #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    # #                                            np.arange(5.0, 10.1, 2.5),
    # #                                            np.array([15, 20, 30., 40., 45.]))),
    # #                     simulation_interval=0.5,
    # #                     clear_sky=False,
    # #                     overwrite_lrt=atm_corr_overwrite_lrt,
    # #                     manual_cloud=True,
    # #                     manual_cloud_cer=10.7,
    # #                     manual_cloud_cwp=11.28,
    # #                     manual_cloud_cth=0.78,
    # #                     manual_cloud_cbh=0.69,
    # #                     manual_cloud_cot=1.59,
    # #                     iter=iter,
    # #                     )
    
    # done
    # # for iter in range(3):
    # #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),
    # #                     tmhr_ranges_select=[
    # #                                         [15.472, 15.567], # 180m, cloudy
    # #                                         [15.580, 15.921], # 100m, cloudy
    # #                                         ],
    # #                     case_tag='cloudy_atm_corr_2',
    # #                     config=config,
    # #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.62, 0.8, 0.96,]),
    # #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    # #                                            np.arange(5.0, 10.1, 2.5),
    # #                                            np.array([15, 20, 30., 40., 45.]))),
    # #                     simulation_interval=0.5,
    # #                     clear_sky=False,
    # #                     overwrite_lrt=atm_corr_overwrite_lrt,
    # #                     manual_cloud=True,
    # #                     manual_cloud_cer=7.2,
    # #                     manual_cloud_cwp=77.5,
    # #                     manual_cloud_cth=0.96,
    # #                     manual_cloud_cbh=0.62,
    # #                     manual_cloud_cot=16.21,
    # #                     iter=iter,
    # #                     )
    

    # for iter in range(1):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
    #                     tmhr_ranges_select=[
    #                                         [13.212, 13.347], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr_1',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.67, 0.8, 1.0,]),
    #                                            np.array([1.5, 1.98, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=True,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=15.3,
    #                     manual_cloud_cwp=143.94,
    #                     manual_cloud_cth=1.98,
    #                     manual_cloud_cbh=0.67,
    #                     manual_cloud_cot=14.12,
    #                     iter=iter,
    #                     )
    

    # for iter in range(1):
    #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),
    #                     tmhr_ranges_select=[
    #                                         [15.314, 15.504], # 100m, cloudy
    #                                         ],
    #                     case_tag='cloudy_atm_corr_2',
    #                     config=config,
    #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.78, 1.0,]),
    #                                            np.array([1.5, 1.81, 2.21, 2.5, 3.0, 4.0]), 
    #                                            np.arange(5.0, 10.1, 2.5),
    #                                            np.array([15, 20, 30., 40., 45.]))),
    #                     simulation_interval=0.5,
    #                     clear_sky=True,
    #                     overwrite_lrt=atm_corr_overwrite_lrt,
    #                     manual_cloud=False,
    #                     manual_cloud_cer=7.8,
    #                     manual_cloud_cwp=64.18,
    #                     manual_cloud_cth=2.21,
    #                     manual_cloud_cbh=1.81,
    #                     manual_cloud_cot=12.41,
    #                     iter=iter,
    #                     )
    
    # done
    # # for iter in range(3):
    # #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),
    # #                     tmhr_ranges_select=[
    # #                                         [13.376, 13.600], # 100m, cloudy
    # #                                         ],
    # #                     case_tag='cloudy_atm_corr_1',
    # #                     config=config,
    # #                     levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.34, 0.4, 0.6, 0.77, 1.0,]),
    # #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    # #                                            np.arange(5.0, 10.1, 2.5),
    # #                                            np.array([15, 20, 30., 40., 45.]))),
    # #                     simulation_interval=0.5,
    # #                     clear_sky=False,
    # #                     overwrite_lrt=atm_corr_overwrite_lrt,
    # #                     manual_cloud=True,
    # #                     manual_cloud_cer=9.0,
    # #                     manual_cloud_cwp=83.49,
    # #                     manual_cloud_cth=0.77,
    # #                     manual_cloud_cbh=0.34,
    # #                     manual_cloud_cot=13.93,
    # #                     iter=iter,
    # #                     )
    

    # done
    # # for iter in range(3):
    # #     flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),
    # #                     tmhr_ranges_select=[
    # #                                         [16.029, 16.224], # 100m, cloudy
    # #                                         ],
    # #                     case_tag='cloudy_atm_corr_2',
    # #                     config=config,
    # #                     levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.29, 0.4, 0.62, 0.8, 1.0,]),
    # #                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
    # #                                            np.arange(5.0, 10.1, 2.5),
    # #                                            np.array([15, 20, 30., 40., 45.]))),
    # #                     simulation_interval=0.5,
    # #                     clear_sky=False,
    # #                     overwrite_lrt=atm_corr_overwrite_lrt,
    # #                     manual_cloud=True,
    # #                     manual_cloud_cer=8.3,
    # #                     manual_cloud_cwp=49.10,
    # #                     manual_cloud_cth=0.62,
    # #                     manual_cloud_cbh=0.29,
    # #                     manual_cloud_cot=8.93,
    # #                     iter=iter,
    # #                     )
        
