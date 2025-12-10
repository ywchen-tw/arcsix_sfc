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

from matplotlib import rcParams

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif" # Ensure sans-serif is used as the default family


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
                     manual_alb=None,
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
        # print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
        
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
            # print(f"Processed file {processed_file} not found. Skipping leg {i}.")
            None
        
    
    
    
    lon_avg = np.round(np.mean(lon_all), 2)
    lat_avg = np.round(np.mean(lat_all), 2)
    lon_min, lon_max = np.round(np.min(lon_all), 2), np.round(np.max(lon_all), 2)
    lat_min, lat_max = np.round(np.min(lat_all), 2), np.round(np.max(lat_all), 2)
    alt_avg = np.round(np.nanmean(alt_all), 2)  # in km
    sza_avg = np.round(np.nanmean(sza_all), 2)
    saa_avg = np.round(np.nanmean(saa_all), 2)
    sfc_T_avg = np.round(np.nanmean(sfc_T), 2)

        
    sza_arr = np.array([50, 55, 60, 65, 70, 75, 77.5, 80, 82.5, 85, 87, np.round(sza_avg, 2)], dtype=np.float32)
        
        
    if clear_sky:
        fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_clear'
        fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
    else:
        fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_sat_cloud'
        fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'
    
    fdir_alb = f'{_fdir_general_}/sfc_alb_cre'
    
    if not os.path.exists(f'{fdir}/{date_s}_{case_tag}_cre_simulations_all_alb.csv'):

        if manual_alb is None:
            manual_alb = [None]
        elif isinstance(manual_alb, str):
            manual_alb = [manual_alb]
        else:
            assert isinstance(manual_alb, list) #"manual_alb should be None, str, or list of str"
            
        cot_list_all = []
        cwp_list_all = []
        cer_list_all = []
        cth_list_all = []
        cbh_list_all = []
        sza_list_all = []
        Fup_sfc_sw_all = []
        Fdn_sfc_sw_all = []
        F_sfc_sw_cre_all = []
        F_sfc_lw_cre_all = []
        F_sfc_net_cre_all = []
        broadband_alb_list_all = []
        broadband_alb_ori_list_all = []
        
        cot_real_list_all = []
        cwp_real_list_all = []
        cer_real_list_all = []
        cth_real_list_all = []
        cbh_real_list_all = []
        sza_real_list_all = []
        Fup_real_sfc_sw_all = []
        Fdn_real_sfc_sw_all = []
        F_sfc_sw_cre_real_all = []
        F_sfc_lw_cre_real_all = []
        F_sfc_net_cre_real_all = []
        broadband_alb_list_real_all = []
        broadband_alb_ori_list_real_all = []
        
        alb_wvl_all = []
        alb_all = []
        broadband_alb_all = []
        broadband_alb_ori_all = []
        
        
        # use CU solar spectrum
        df_solor = pd.read_csv('CU_composite_solar_processed.dat', sep='\s+', header=None)
        wvl_solar = np.array(df_solor.iloc[:, 0])
        flux_solar = np.array(df_solor.iloc[:, 1])#/1000 # convert mW/m^2/nm to W/m^2/nm
        
        # interpolate to 1 nm grid
        f_interp = interp1d(wvl_solar, flux_solar, kind='linear', bounds_error=False, fill_value=0.0)
        
        
        
        
        
        # if manual_alb is None:
        #     output_csv_name = f'{fdir}/ssfr_simu_flux_{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}_alt-{alt_avg:.2f}km_cre_{mode}_sza_{sza_sim:.2f}.csv'
        # else:
        #     output_csv_name = f'{fdir}/ssfr_simu_flux_{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}_alt-{alt_avg:.2f}km_cre_{mode}_sza_{sza_sim:.2f}_alb-manual-{manual_alb.replace(".dat", "")}.csv'

        for i in range(len(manual_alb)):
            print(f"Processing manual_alb {i+1}/{len(manual_alb)} ...")
            
            for sza_sim in sza_arr:
                
                manual_alb_i = manual_alb[i]
                if manual_alb_i is None:
                    output_csv_name_sw = f'{fdir}/ssfr_simu_flux_{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}_alt-{alt_avg:.2f}km_cre_sw_sza_{sza_sim:.2f}_0.99.csv'
                    output_csv_name_lw = f'{fdir}/ssfr_simu_flux_{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}_alt-{alt_avg:.2f}km_cre_lw_sza_{sza_sim:.2f}_0.99.csv'
                else:
                    output_csv_name_sw = f'{fdir}/ssfr_simu_flux_{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}_alt-{alt_avg:.2f}km_cre_sw_sza_{sza_sim:.2f}_alb-manual-{manual_alb_i.replace(".dat", "")}_0.99.csv'
                    output_csv_name_lw = f'{fdir}/ssfr_simu_flux_{date_s}_{time_all[0]:.3f}-{time_all[-1]:.3f}_alt-{alt_avg:.2f}km_cre_lw_sza_{sza_sim:.2f}_alb-manual-{manual_alb_i.replace(".dat", "")}_0.99.csv'

                os.makedirs(fdir_tmp, exist_ok=True)
                os.makedirs(fdir, exist_ok=True)
                    

                # read csv and extract simulated fluxes
                with open(output_csv_name_sw, 'r') as f:
                    df_sw = pd.read_csv(f)
                    
                with open(output_csv_name_lw, 'r') as f:
                    df_lw = pd.read_csv(f)
                    
                # with open(output_csv_name_sw.replace('.csv', '_2.csv'), 'r') as f:
                #     df_sw_2 = pd.read_csv(f)
                    
                # with open(output_csv_name_lw.replace('.csv', '_2.csv'), 'r') as f:
                #     df_lw_2 = pd.read_csv(f)
                    
                cot_list = df_sw['cot'].values 
                cwp_list = df_sw['cwp'].values
                cer_list = df_sw['cer'].values
                cth_list = df_sw['cth'].values
                cbh_list = df_sw['cbh'].values
                sza_list = df_sw['sza'].values
                Fup_sfc_sw = df_sw['Fup_sfc'].values
                Fdn_sfc_sw = df_sw['Fdn_sfc'].values
                Fup_sfc_lw = df_lw['Fup_sfc'].values
                Fdn_sfc_lw = df_lw['Fdn_sfc'].values
                
                # cot_list = np.concatenate((df_sw['cot'].values, df_sw_2['cot'].values))
                # cwp_list = np.concatenate((df_sw['cwp'].values, df_sw_2['cwp'].values))
                # cer_list = np.concatenate((df_sw['cer'].values, df_sw_2['cer'].values))
                # cth_list = np.concatenate((df_sw['cth'].values, df_sw_2['cth'].values))
                # cbh_list = np.concatenate((df_sw['cbh'].values, df_sw_2['cbh'].values))
                # sza_list = np.concatenate((df_sw['sza'].values, df_sw_2['sza'].values))
                # Fup_sfc_sw = np.concatenate((df_sw['Fup_sfc'].values, df_sw_2['Fup_sfc'].values))
                # Fdn_sfc_sw = np.concatenate((df_sw['Fdn_sfc'].values, df_sw_2['Fdn_sfc'].values))
                # Fup_sfc_lw = np.concatenate((df_lw['Fup_sfc'].values, df_lw_2['Fup_sfc'].values))
                # Fdn_sfc_lw = np.concatenate((df_lw['Fdn_sfc'].values, df_lw_2['Fdn_sfc'].values))
                
                
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
                cwp_cre = np.array(cwp_list) * 1000
                F_sfc_net_cre = F_sfc_sw_cre + F_sfc_lw_cre
                
                # print("cwp_cre:", cwp_cre)
                # print("F_sfc_sw_cre:", F_sfc_sw_cre)
                # print("F_sfc_lw_cre:", F_sfc_lw_cre)
                
                select = np.array([cwp%0.5==0 for cwp in cwp_cre])
                case_sel = ~select
                
                select = cwp >= 0
                case_sel = ~np.array([cwp%0.5==0 for cwp in cwp_cre])
                
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(cwp_cre[select], F_sfc_sw_cre[select], '-', label='SW CRE')
                ax.plot(cwp_cre[select], F_sfc_lw_cre[select], '-', label='LW CRE')
                ax.plot(cwp_cre[select], F_sfc_net_cre[select], '-', label='Net CRE')
                ax.scatter(cwp_cre[case_sel], F_sfc_net_cre[case_sel], c='C0', marker='o', s=50, label='Flight case')
                ax.set_xlabel('Cloud Liquid Water Path (g/m2)', fontsize=14)
                ax.set_ylabel('Surface CRE (W/m2)', fontsize=14)
                ax.set_title(f'Surface CRE vs. LWP on {date_s}', fontsize=16)
                ax.hlines(0, xmin=0, xmax=np.max(cwp_cre), colors='gray', linestyles='dashed')
                ax.legend(fontsize=12)
                fig.tight_layout()
                if manual_alb_i is not None:
                    fig.savefig(f'fig/{date_s}/surface_cre_vs_lwp_{date_s}_{case_tag}_alb-manual-{manual_alb_i.replace(".dat", "")}.png', dpi=300)
                else:
                    fig.savefig(f'fig/{date_s}/surface_cre_vs_lwp_{date_s}_{case_tag}.png', dpi=300)
            
                plt.close(fig)
                
                cot_list_all.extend(cot_list[select])
                cwp_list_all.extend(cwp_cre[select])
                cer_list_all.extend(cer_list[select])
                cth_list_all.extend(cth_list[select])
                cbh_list_all.extend(cbh_list[select])
                sza_list_all.extend(sza_list[select])
                Fup_sfc_sw_all.extend(Fup_sfc_sw[select])
                Fdn_sfc_sw_all.extend(Fdn_sfc_sw[select])
                F_sfc_sw_cre_all.extend(F_sfc_sw_cre[select])
                F_sfc_lw_cre_all.extend(F_sfc_lw_cre[select])
                F_sfc_net_cre_all.extend(F_sfc_net_cre[select])
                
                cot_real_list_all.extend(cot_list[case_sel])
                cwp_real_list_all.extend(cwp_cre[case_sel])
                cer_real_list_all.extend(cer_list[case_sel])
                cth_real_list_all.extend(cth_list[case_sel])
                cbh_real_list_all.extend(cbh_list[case_sel])
                sza_real_list_all.extend(sza_list[case_sel])
                Fup_real_sfc_sw_all.extend(Fup_sfc_sw[case_sel])
                Fdn_real_sfc_sw_all.extend(Fdn_sfc_sw[case_sel])
                F_sfc_sw_cre_real_all.extend(F_sfc_sw_cre[case_sel])
                F_sfc_lw_cre_real_all.extend(F_sfc_lw_cre[case_sel])
                F_sfc_net_cre_real_all.extend(F_sfc_net_cre[case_sel])
                
                if manual_alb_i is None:
                    f_alb = f'{fdir_alb}/sfc_alb_{date_s}_{time_all[0]:.3f}_{time_all[-1]:.3f}_{alt_avg:.2f}km_cre_alb.dat'
                else:
                    f_alb = f'{fdir_alb}/{manual_alb_i}'
                alb_data = np.loadtxt(f_alb)
                ext_wvl = alb_data[:, 0]
                ext_alb = alb_data[:, 1]
                
                
                
                flux_solar_interp = f_interp(ext_wvl)
                broadband_alb = np.trapz(ext_alb * flux_solar_interp, ext_wvl) / np.trapz(flux_solar_interp, ext_wvl)
                # broadband_alb = np.sum(ext_alb * flux_solar_interp) / np.sum(flux_solar_interp)
                broadband_alb = np.round(broadband_alb, 3)
                
                broadband_alb_list_all.extend([broadband_alb]*len(cot_list[select]))
                broadband_alb_list_real_all.extend([broadband_alb]*len(cot_list[case_sel]))
                
                flux_solar_interp_ori = f_interp(alb_wvl)
                alb_ori = ext_alb[np.logical_and(ext_wvl >=alb_wvl[0], ext_wvl <= alb_wvl[-1])]
                broadband_alb_ori = np.trapz(alb_ori * flux_solar_interp_ori, alb_wvl) / np.trapz(flux_solar_interp_ori, alb_wvl)
                # broadband_alb_ori = np.sum(alb_ori * flux_solar_interp_ori) / np.sum(flux_solar_interp_ori)
                broadband_alb_ori = np.round(broadband_alb_ori, 3)
                
                broadband_alb_ori_list_all.extend([broadband_alb_ori]*len(cot_list[select]))
                broadband_alb_ori_list_real_all.extend([broadband_alb_ori]*len(cot_list[case_sel]))
                
            alb_wvl_all.append(ext_wvl)
            alb_all.append(ext_alb)
            broadband_alb_all.append(broadband_alb)
            broadband_alb_ori_all.append(broadband_alb_ori)
        
        
        cot_list_all = np.array(cot_list_all).flatten()
        cwp_list_all = np.array(cwp_list_all).flatten()
        cer_list_all = np.array(cer_list_all).flatten()
        cth_list_all = np.array(cth_list_all).flatten()
        cbh_list_all = np.array(cbh_list_all).flatten()
        sza_list_all = np.array(sza_list_all).flatten()
        Fup_sfc_sw_all = np.array(Fup_sfc_sw_all).flatten()
        Fdn_sfc_sw_all = np.array(Fdn_sfc_sw_all).flatten()
        F_sfc_sw_cre_all = np.array(F_sfc_sw_cre_all).flatten()
        F_sfc_lw_cre_all = np.array(F_sfc_lw_cre_all).flatten()
        F_sfc_net_cre_all = np.array(F_sfc_net_cre_all).flatten()
        broadband_alb_list_all = np.array(broadband_alb_list_all).flatten()
        broadband_alb_ori_list_all = np.array(broadband_alb_ori_list_all).flatten()
        
        cot_real_list_all = np.array(cot_real_list_all).flatten()
        cwp_real_list_all = np.array(cwp_real_list_all).flatten()
        cer_real_list_all = np.array(cer_real_list_all).flatten()
        cth_real_list_all = np.array(cth_real_list_all).flatten()
        cbh_real_list_all = np.array(cbh_real_list_all).flatten()
        sza_real_list_all = np.array(sza_real_list_all).flatten()
        Fup_real_sfc_sw_all = np.array(Fup_real_sfc_sw_all).flatten()
        Fdn_real_sfc_sw_all = np.array(Fdn_real_sfc_sw_all).flatten()
        F_sfc_sw_cre_real_all = np.array(F_sfc_sw_cre_real_all).flatten()
        F_sfc_lw_cre_real_all = np.array(F_sfc_lw_cre_real_all).flatten()
        F_sfc_net_cre_real_all = np.array(F_sfc_net_cre_real_all).flatten()
        broadband_alb_list_real_all = np.array(broadband_alb_list_real_all).flatten()
        broadband_alb_ori_list_real_all = np.array(broadband_alb_ori_list_real_all).flatten()
        
        
        df_all = pd.DataFrame({"cot": cot_list_all,
                            "cwp": cwp_list_all,
                            "cer": cer_list_all,
                            "cth": cth_list_all,
                            "cbh": cbh_list_all,
                            "sza": sza_list_all,
                            "Fup_sfc_sw": Fup_sfc_sw_all,
                            "Fdn_sfc_sw": Fdn_sfc_sw_all,
                            "F_sfc_sw_cre": F_sfc_sw_cre_all,
                            "F_sfc_lw_cre": F_sfc_lw_cre_all,
                            "F_sfc_net_cre": F_sfc_net_cre_all,
                            "broadband_alb": broadband_alb_list_all,
                            "broadband_alb_ori": broadband_alb_ori_list_all,
                            })
        df_all.to_csv(f'{fdir}/{date_s}_{case_tag}_cre_simulations_all_alb.csv', index=False)
        
        df_real_all = pd.DataFrame({"cot": cot_real_list_all,
                            "cwp": cwp_real_list_all,
                            "cer": cer_real_list_all,
                            "cth": cth_real_list_all,
                            "cbh": cbh_real_list_all,
                            "sza": sza_real_list_all,
                            "Fup_sfc_sw": Fup_real_sfc_sw_all,
                            "Fdn_sfc_sw": Fdn_real_sfc_sw_all,
                            "F_sfc_sw_cre": F_sfc_sw_cre_real_all,
                            "F_sfc_lw_cre": F_sfc_lw_cre_real_all,
                            "F_sfc_net_cre": F_sfc_net_cre_real_all,
                            "broadband_alb": broadband_alb_list_real_all,
                            "broadband_alb_ori": broadband_alb_ori_list_real_all,
                            })
        df_real_all.to_csv(f'{fdir}/{date_s}_{case_tag}_cre_simulations_real_cases_all_alb.csv', index=False)
        
        
        alb_spectra_all = {'alb_wvl_all': alb_wvl_all,
                           'alb_all': alb_all,
                           'broadband_alb_all': broadband_alb_all,
                           'broadband_alb_ori_all': broadband_alb_ori_all,
                           }
        with open(f'{fdir}/{date_s}_{case_tag}_cre_simulations_alb_spectra.pkl', 'wb') as f:
            pickle.dump(alb_spectra_all, f)
        
        print("unique broadband_alb_all:", np.unique(broadband_alb_all))
        # 0.561 0.655 0.735 0.766
        print("unique broadband_alb_ori_all:", np.unique(broadband_alb_ori_all))
    
    else:
        with open(f'{fdir}/{date_s}_{case_tag}_cre_simulations_all_alb.csv', 'r') as f:
            df_all = pd.read_csv(f)
        
        with open(f'{fdir}/{date_s}_{case_tag}_cre_simulations_real_cases_all_alb.csv', 'r') as f:
            df_real_all = pd.read_csv(f)
            
        with open(f'{fdir}/{date_s}_{case_tag}_cre_simulations_alb_spectra.pkl', 'rb') as f:
            alb_spectra_all = pickle.load(f)
            alb_wvl_all = alb_spectra_all['alb_wvl_all']
            alb_all = alb_spectra_all['alb_all']
            broadband_alb_all = alb_spectra_all['broadband_alb_all']
            broadband_alb_ori_all = alb_spectra_all['broadband_alb_ori_all']
    
    
    df_real_all = df_real_all.loc[np.logical_and(df_real_all['sza'].values%5 > 0, df_real_all['cwp'].values%0.5 > 0), :]
    print(df_real_all)
    for col in df_real_all.columns:
        print(f"{col} data:", df_real_all[col].values)

    
    broadband_alb_all_unique = sorted(list(set(df_all['broadband_alb'].values)))
    print("broadband_alb_all_unique:", broadband_alb_all_unique)
    
    sza_mesh, broadband_alb_mesh = np.meshgrid(sza_arr, broadband_alb_all_unique, indexing='ij')
    cwp_zero_arr = np.zeros_like(sza_mesh, dtype=np.float32) * np.nan
    for i in range(sza_mesh.shape[0]):
        for j in range(sza_mesh.shape[1]):
            sza_sim = sza_arr[i]
            broadband_alb = broadband_alb_all_unique[j]
            df_select_mask = np.logical_and((df_all['sza'].values==sza_sim), (df_all['broadband_alb'].values==broadband_alb))
            df_sub = df_all.loc[df_select_mask, :]
            cwp_arr = df_sub['cwp'].values
            F_cre_net_arr = df_sub['F_sfc_net_cre'].values
        
            
            # find zero crossing
            zero_crossings = np.where(np.diff(np.sign(F_cre_net_arr)))[0]
            print(f'  Zero crossings indices: {zero_crossings}')
            zero_crossings_tmp = []
            for zero_crossing in zero_crossings[1:]:
                cwp1 = cwp_arr[zero_crossing]
                cwp2 = cwp_arr[zero_crossing + 1]
                F1 = F_cre_net_arr[zero_crossing]
                F2 = F_cre_net_arr[zero_crossing + 1]
                # linear interpolation to find the root
                cwp_zero = cwp1 - F1 * (cwp2 - cwp1) / (F2 - F1)
                print(f' sza: {sza_sim}, broadband_alb: {broadband_alb}')
                print(f'  Zero crossing at CWP: {cwp_zero:.2f} g/m2 between {cwp1:.2f} and {cwp2:.2f} g/m2')
                zero_crossings_tmp.append(cwp_zero)
            if len(zero_crossings_tmp) > 0:
                # sza_sim_list.append(sza_sim)
                # broadband_alb_sim_list.append(broadband_alb)
                # cwp_zero_list.append(np.nanmean(zero_crossings_tmp))
                cwp_zero_arr[i, j] = zero_crossings_tmp[0]  # take the first zero crossing
                
                if sza_sim >= 75 or len(zero_crossings_tmp) > 1:
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(cwp_arr, F_cre_net_arr, '-', label='Net CRE')
                    ax.scatter(zero_crossings_tmp[0], 0, c='C0', marker='o', s=50, label='Zero Crossing' if len(zero_crossings_tmp)>0 else '')
                    ax.hlines(0, xmin=0, xmax=np.max(cwp_arr), colors='gray', linestyles='dashed')
                    ax.set_xlabel('Cloud Liquid Water Path (g/m2)', fontsize=14)
                    ax.set_ylabel('Surface Net CRE (W/m2)', fontsize=14)
                    ax.set_title(f'Surface Net CRE vs. LWP on {date_s}, SZA: {sza_sim}, Broadband Albedo: {broadband_alb}', fontsize=16)
                    ax.legend(fontsize=12)
                    fig.tight_layout()
                    plt.show()
                    plt.close(fig)
        
    
    # Create a ScalarMappable
    color_series = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',]
    
    plt.close('all')
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(11, 11, figure=fig, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[:5, :6])
    ax2 = fig.add_subplot(gs[6:, :6])
    ax3 = fig.add_subplot(gs[2:10, 7:])
    sza_real_df_all = df_all.loc[df_all['sza']==61.46, :]
    sza_real_df_real_all = df_real_all.loc[df_real_all['sza']==61.46, :]
    for i in range(len(broadband_alb_all_unique)):
        broadband_alb_i = broadband_alb_all_unique[i]
        df_select_mask = sza_real_df_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_all_i = sza_real_df_all.loc[df_select_mask, :]
        df_real_mask = sza_real_df_real_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_real_all_i = sza_real_df_real_all.loc[df_real_mask, :]
        
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_sw_cre'].values, '--', color=color_series[i], alpha=0.5)
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_lw_cre'].values, '-.', color=color_series[i], alpha=0.5)
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, '-', color=color_series[i], label=f'Albedo-{i+1}')
        ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k')
                
        # ax2.plot(alb_wvl_all[i], alb_all[i], '-', color=color_series[i], label=f'Extended Broadband Albedo: {broadband_alb_all[i]:.3f} (Original: {broadband_alb_ori_all[i]:.3f})')
        ax2.plot(alb_wvl_all[i], alb_all[i], '-', color=color_series[i], label=f'Extended Broadband Albedo: {broadband_alb_all[i]:.3f}')

    from matplotlib.lines import Line2D
    linestyles = ['-', '--', '-.']
    # Create proxy artists (one per linestyle). Use a neutral color so legend doesn't reflect line colors.
    proxy_handles = [Line2D([0], [0], color='black', lw=1, linestyle=ls) for ls in linestyles]
    proxy_labels  = ['Net', 'SW', 'LW']

    ax1.legend(proxy_handles, proxy_labels, fontsize=10)
    
    ax1.set_xlabel('Cloud Liquid Water Path $\mathrm{(g/m^2)}$',
                   fontsize=14)
    ax1.set_ylabel('Surface CRE $\mathrm{(W/m^2)}$', 
                   fontsize=14)
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    

    
    ax2.legend(fontsize=10,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.hlines(0, xmin=0, xmax=np.max(sza_real_df_all['cwp'].values), colors='gray', linestyles='dashed')
    ax1.set_xlim(0, 200)
    ax2.set_xlim(300, 4000)
    ax2.set_ylim(-0.05, 1.05)
    ax2.hlines(0, xmin=300, xmax=4000, colors='gray', linestyles='dashed')
    
    level_labels = [20, 30, 40, 50, 60, 70, 80, 100, 125, 150, 175, 200, 250, 300]
    
    cc1 = ax3.scatter(sza_mesh.flatten(), broadband_alb_mesh.flatten(), c=cwp_zero_arr, s=50, alpha=0.5, cmap='jet', vmin=20, vmax=300, zorder=3)
    
    
    
    # cc = ax3.contour(sza_mesh, broadband_alb_mesh, cwp_zero_arr, levels=level_labels, cmap='jet')
    # # ax3.clabel(cc, cc.levels, fontsize=12, colors='k')
    # ax3.clabel(cc, level_labels, fontsize=12, colors='k')
    
    cc = ax3.contourf(sza_mesh, broadband_alb_mesh, cwp_zero_arr, cmap='jet', vmin=20, vmax=300, zorder=1)
    ax3.set_xlabel('Solar Zenith Angle (degrees)', fontsize=14)
    ax3.set_ylabel('Broadband Albedo', fontsize=14)

    
    
    cbar = fig.colorbar(cc1, ax=ax3, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Critical LWP ($\mathrm{g/m^2}$)',
                   fontsize=14)
    ax3.scatter(61.46, 0.735, color='red', marker='^', s=100, label='Flight Case SZA and Albedo')
    ax3.set_xlim(50, 80)
    
    for ax, subcase in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.text(0.0, 1.03, subcase, transform=ax.transAxes, fontsize=16, va='bottom', ha='left')
    fig.savefig(f'fig/{date_s}/surface_cre_vs_lwp_all_alb_{date_s}_{case_tag}_combined.png', dpi=300, bbox_inches='tight')
    
    sys.exit()
    
    
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'hspace': 0.3})
    for i in range(len(manual_alb)):
        ax1.plot(cwp_list_all[i], F_sfc_sw_cre_all[i], '--', color=color_series[i], alpha=0.5)
        ax1.plot(cwp_list_all[i], F_sfc_lw_cre_all[i], '-.', color=color_series[i], alpha=0.5)
        ax1.plot(cwp_list_all[i], F_sfc_net_cre_all[i], '-', color=color_series[i], label=f'Albedo-{i+1}')
        ax1.scatter(cwp_real_list_all[i], F_sfc_net_cre_real_all[i], color=color_series[i], marker='o', s=50, edgecolors='k')
        
        ax2.plot(alb_wvl_all[i], alb_all[i], '-', color=color_series[i], label=f'Extended Broadband Albedo: {broadband_alb_all[i]:.3f} (Original: {broadband_alb_ori_all[i]:.3f})')
    
    ax1.set_xlabel('Cloud Liquid Water Path $\mathrm{(g/m^2)}$',
                   fontsize=14)
    ax1.set_ylabel('Surface CRE $\mathrm{(W/m^2)}$', 
                   fontsize=14)
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    
    ax2.legend(fontsize=12,)# loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.hlines(0, xmin=0, xmax=np.max(cwp_list_all), colors='gray', linestyles='dashed')
    ax1.set_xlim(0, 250)
    ax2.set_xlim(300, 4000)
    ax2.set_ylim(-0.05, 1.05)
    ax2.hlines(0, xmin=300, xmax=4000, colors='gray', linestyles='dashed')
    for ax, subcase in zip([ax1, ax2], ['(a)', '(b)']):
        ax.text(0.0, 1.03, subcase, transform=ax.transAxes, fontsize=16, va='bottom', ha='left')
    fig.savefig(f'fig/{date_s}/surface_cre_vs_lwp_all_alb_{date_s}_{case_tag}_all.png', dpi=300, bbox_inches='tight')
    

 
    #\----------------------------------------------------------------------------/#

    return

def albedo_plot(albedo_file_list, date_s):
    
    
    fdir_alb = f'{_fdir_general_}/sfc_alb_cre'
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5), gridspec_kw={'hspace': 0.3})
    for alb_file in albedo_file_list:
        alb_data = np.loadtxt(f'{fdir_alb}/{alb_file}')
        wvl = alb_data[:, 0]
        alb = alb_data[:, 1]
        

        
        ax1.plot(wvl, alb, '-', label=alb_file.split('_')[2])
    

    ax1.set_xlabel('Wavelength (nm)', fontsize=14)
    ax1.set_ylabel('Surface Albedo', fontsize=14)
    
    ax1.legend(fontsize=12,)# loc='center left', bbox_to_anchor=(1.02, 0.5))

    ax1.set_xlim(350, 2000)
    ax1.set_ylim(-0.05, 1.05)
    ax1.hlines(0, xmin=300, xmax=4000, colors='gray', linestyles='dashed')
    fig.savefig(f'fig/{date_s}/surface_alb_test.png', dpi=300, bbox_inches='tight')

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
    


    # cre_sim_plot(date=datetime.datetime(2024, 6, 3),
    #                 tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
    #                                     ],
    #                 case_tag='cloudy_atm_corr_1',
    #                 config=config,
    #                 levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
    #                                         np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 simulation_interval=0.5,
    #                 clear_sky=False,
    #                 overwrite_lrt=atm_corr_overwrite_lrt,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=13.0 ,
    #                 manual_cloud_cwp=77.82,
    #                 manual_cloud_cth=1.93,
    #                 manual_cloud_cbh=1.41,
    #                 manual_cloud_cot=21.27,
    #                 )


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
    
    # albedo_plot([
    #             'sfc_alb_20240603_13.620_13.750_0.32km_cre_alb.dat',
    #             'sfc_alb_20240606_16.250_16.950_0.50km_cre_alb.dat',
    #             'sfc_alb_20240607_15.336_15.761_0.12km_cre_alb.dat',
    #             'sfc_alb_20240613_16.550_17.581_0.22km_cre_alb.dat',
    #             'sfc_alb_20240725_15.094_15.300_0.11km_cre_alb.dat',
    #             'sfc_alb_20240807_13.344_13.761_0.13km_cre_alb.dat',
    #             ], 
    #             '20240607')
    
    # sys.exit()
    
    
    
    # done   
    cre_sim_plot(date=datetime.datetime(2024, 6, 7),
                    tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr',
                    config=config,
                    levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
                                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                            np.arange(5.0, 10.1, 2.5),
                                            np.array([15, 20, 30., 40., 45.]))),
                    simulation_interval=0.5,
                    clear_sky=False,
                    overwrite_lrt=atm_corr_overwrite_lrt,
                    manual_cloud=True,
                    manual_alb=[
                                'sfc_alb_20240606_16.250_16.950_0.50km_cre_alb.dat',
                                None,
                                'sfc_alb_20240603_13.620_13.750_0.32km_cre_alb.dat',
                                
                                'sfc_alb_20240613_16.550_17.581_0.22km_cre_alb.dat',
                                'sfc_alb_20240725_15.094_15.300_0.11km_cre_alb.dat',
                                ]
                    )
    

    
    

    

    
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
        
