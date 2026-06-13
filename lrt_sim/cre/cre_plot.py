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
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_LRT_SIM_ROOT = str(_THIS_FILE.parents[1])
_REPO_ROOT = str(_THIS_FILE.parents[2])
for _path in (_REPO_ROOT, _LRT_SIM_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

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

# Reuse the atmospheric-correction settings (single source of truth for paths
# and mission constants) instead of redefining them here.
try:
    from ssfr_atm_corr.settings import (
        _mission_, _platform_, _hsk_, _alp_, _spns_, _ssfr1_, _ssfr2_, _cam_,
        _fdir_main_, _fdir_sat_img_, _fdir_sat_data_, _fdir_cam_img_, _wavelength_,
        _fdir_sat_img_vn_, _preferred_region_, _aspect_,
        _fdir_data_, _fdir_general_, _fdir_tmp_, _fdir_tmp_graph_, _title_extra_,
    )
except ImportError:
    from lrt_sim.ssfr_atm_corr.settings import (
        _mission_, _platform_, _hsk_, _alp_, _spns_, _ssfr1_, _ssfr2_, _cam_,
        _fdir_main_, _fdir_sat_img_, _fdir_sat_data_, _fdir_cam_img_, _wavelength_,
        _fdir_sat_img_vn_, _preferred_region_, _aspect_,
        _fdir_data_, _fdir_general_, _fdir_tmp_, _fdir_tmp_graph_, _title_extra_,
    )

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


def combined_product_file():
    """Return the path to the combined atmospheric-correction product."""
    return f'{_fdir_general_}/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl'


def load_case_from_combined(date_s, case_tag, combined_file=None):
    """Select one case's albedo, broadband albedo, sza and coordinates from the
    combined atmospheric-correction product, matching :func:`cre.cre_sim`.

    Returns ``None`` when the combined file is missing or has no rows for the
    requested case so the caller can fall back to the per-leg processed pickles.
    """
    if combined_file is None:
        combined_file = combined_product_file()
    if not os.path.exists(combined_file):
        return None

    with open(combined_file, 'rb') as f:
        d = pickle.load(f)

    sfx = 'summer' if str(date_s) > '20240630' else 'spring'
    date_mask = np.asarray(d[f'dates_{sfx}_all']) == int(date_s)
    case_mask = np.array([str(ct) == str(case_tag) for ct in d[f'case_tags_{sfx}_all']])
    mask = date_mask & case_mask
    if not np.any(mask):
        return None

    n_pts = int(np.sum(mask))
    bb_key = f'broadband_alb_iter2_{sfx}_all'
    broadband = (
        np.asarray(d[bb_key])[mask] if bb_key in d else np.full(n_pts, np.nan)
    )
    return {
        'alb_wvl':       np.asarray(d[f'wvl_{sfx}']),
        'alb':           np.asarray(d[f'alb_iter2_all_{sfx}'])[mask, :],
        'broadband_alb': broadband,
        'time':          np.asarray(d[f'time_{sfx}_all'])[mask],
        'lon':           np.asarray(d[f'lon_all_{sfx}'])[mask],
        'lat':           np.asarray(d[f'lat_all_{sfx}'])[mask],
        'alt':           np.asarray(d[f'alt_all_{sfx}'])[mask],
        'sza':           np.asarray(d[f'sza_{sfx}_all'])[mask],
    }


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
        
    processed_dir = f'{_fdir_general_}/sfc_alb_combined'
    os.makedirs(processed_dir, exist_ok=True)
    
    lon_all = np.array([])
    lat_all = np.array([])
    alt_all = np.array([])
    sza_all = np.array([])
    saa_all = np.array([])
    sfc_T = np.array([])
    broadband_alb_all = np.array([])

    time_all = np.array([])
    marli_all_h = np.array([])
    marli_all_wvmr = np.array([])

    init = True
    alb_iter2_all = []

    # Albedo / broadband / sza / coordinates come from the combined product so
    # that file naming (time window + alt) matches what cre_sim wrote.
    combined_case = load_case_from_combined(date_s, case_tag)

    if combined_case is not None:
        alb_wvl           = combined_case['alb_wvl']
        alb_iter2_all     = combined_case['alb']
        broadband_alb_all = combined_case['broadband_alb']
        time_all          = combined_case['time']
        lon_all           = combined_case['lon']
        lat_all           = combined_case['lat']
        alt_all           = combined_case['alt']
        sza_all           = combined_case['sza']
    else:
        for i in range(len(tmhr_ranges_select)):
            time_start, time_end = tmhr_ranges_select[i][0], tmhr_ranges_select[i][-1]

            fname_cld_obs_info = '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, time_start, time_end)

            processed_file = f'{processed_dir}/sfc_alb_update_{date_s}_{case_tag}_time_{tmhr_ranges_select[0][0]:.3f}_{tmhr_ranges_select[-1][-1]:.3f}.pkl'

            if not os.path.exists(processed_file):
                continue

            with open(fname_cld_obs_info, 'rb') as f:
                cld_leg = pickle.load(f)
            with open(processed_file, 'rb') as f:
                processed_leg = pickle.load(f)

            alb_wvl = processed_leg['wvl']
            time_all = np.concatenate((time_all, processed_leg['time_all']))
            lon_all = np.concatenate((lon_all, processed_leg['lon_all']))
            lat_all = np.concatenate((lat_all, processed_leg['lat_all']))
            alt_all = np.concatenate((alt_all, processed_leg['alt_all']))
            sza_all = np.concatenate((sza_all, cld_leg['sza']))
            saa_all = np.concatenate((saa_all, cld_leg['saa']))
            sfc_T = np.concatenate((sfc_T, cld_leg['kt19_sfc_T']))
            broadband_alb_all = np.concatenate((broadband_alb_all, processed_leg['broadband_alb_iter2_all']))

            if cld_leg['marli_h'] is not None:
                marli_all_h = np.concatenate((marli_all_h, cld_leg['marli_h']))
                marli_all_wvmr = np.concatenate((marli_all_wvmr, cld_leg['marli_wvmr']))

            if init:
                alb_iter2_all = processed_leg['alb_iter2_all']
                init = False
            else:
                alb_iter2_all = np.vstack((alb_iter2_all, processed_leg['alb_iter2_all']))

    lon_avg = np.round(np.mean(lon_all), 2)
    lat_avg = np.round(np.mean(lat_all), 2)
    lon_min, lon_max = np.round(np.min(lon_all), 2), np.round(np.max(lon_all), 2)
    lat_min, lat_max = np.round(np.min(lat_all), 2), np.round(np.max(lat_all), 2)
    alt_avg = np.round(np.nanmean(alt_all), 2)  # in km
    sza_avg = np.round(np.nanmean(sza_all), 2)
    saa_avg = np.round(np.nanmean(saa_all), 2)
    sfc_T_avg = np.round(np.nanmean(sfc_T), 2)
    
    alb_iter2_all_avg = np.nanmean(alb_iter2_all, axis=0)
    
    # iceage_nh_12.5km_20240101_20250923_ql.nc
    with Dataset(f'{_fdir_general_}/era5/forecast_albedo_0_daily-mean.nc', 'r') as nc:
        era5_lon = nc.variables['longitude'][:]
        era5_lat = nc.variables['latitude'][:]
        era5_time = nc.variables['valid_time'][:] # days since 2024-05-01
        era5_alb = nc.variables['fal'][:]  # time, lat, lon
    era5_time_dates = np.array([datetime.datetime(2024,5,1) + datetime.timedelta(days=int(t)) for t in era5_time])
    era5_time_dates_str = np.array([t.strftime('%Y%m%d') for t in era5_time_dates])
    era5_alb = np.array(era5_alb, dtype=np.float32)
    era5_lat_mesh, era5_lon_mesh = np.meshgrid(era5_lat, era5_lon, indexing='ij')
    
    era5_alb_date = era5_alb[era5_time_dates_str == str(date_s)]
    era5_alb_interp = griddata(
                (era5_lon_mesh.flatten(), era5_lat_mesh.flatten()), era5_alb_date[0].flatten(),
                (lon_all, lat_all),
                method='nearest'
            )
    
    # df_solor = pd.read_csv('CU_composite_solar_processed.dat', sep='\s+', header=None)
    # wvl_solar = np.array(df_solor.iloc[:, 0])
    # flux_solar = np.array(df_solor.iloc[:, 1])#/1000 # convert mW/m^2/nm to W/m^2/nm
    # # interpolate to 1 nm grid
    # f_interp = interp1d(wvl_solar, flux_solar, kind='linear', bounds_error=False, fill_value=0.0)
    # ext_broadband_alb_all = np.zeros_like(era5_alb_interp)
    # ext_wvl, _ = alb_extention(alb_wvl, alb_iter2_all_avg, clear_sky=clear_sky)
    # flux_solar_interp = f_interp(ext_wvl)
    # for i in range(len(broadband_alb_all)):
    #     alb_i = np.clip(alb_iter2_all[i, :], 0.0, 1.0)
    #     if np.all(np.isnan(alb_i)):
    #         ext_broadband_alb_all[i] = np.nan
    #         continue
    #     ext_wvl_i, ext_alb_i = alb_extention(alb_wvl, alb_i, clear_sky=clear_sky)
    #     ext_broadband_alb_all[i] = np.trapz(ext_alb_i * flux_solar_interp, ext_wvl_i) / np.trapz(flux_solar_interp, ext_wvl_i)

    # plt.close('all')
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.scatter(ext_broadband_alb_all, era5_alb_interp, c='blue', s=10, alpha=0.5)
    # ax.scatter(np.nanmean(ext_broadband_alb_all), np.nanmean(era5_alb_interp), c='red', s=100, label='Mean value')
    # ax.plot([0, 1], [0, 1], 'k--')
    # ax.set_xlim(0.2, 0.9)
    # ax.set_ylim(0.2, 0.9)
    # ax.set_xlabel('Surface Broadband Albedo from SSFR', fontsize=14)
    # ax.set_ylabel('Surface Broadband Albedo from ERA5', fontsize=14)
    # fig.savefig(f'fig/{date_s}/{date_s}_{case_tag}_sfc_alb_ssfr_vs_era5.png', dpi=300)
    
    # print("Average surface broadband albedo from SSFR:", np.nanmean(ext_broadband_alb_all)) # 0.7040066
    # print("Average surface broadband albedo from ERA5:", np.nanmean(era5_alb_interp)) # 0.6548484
    
    
        
    # sza_arr = np.array([50, 52.5, 55, 57.5, 60, np.round(sza_avg, 2), 62.5, 65, 67.5, 70, 71.5, 72.5, 73, 73.5, 75, 77.5, 80, 82.5, 85, 87], dtype=np.float32)

    # sza_arr = np.array([50, 52.5, 55, 57.5, 60, np.round(sza_avg, 2), 62.5, 65, 67.5, 70, 71.5, 72.5, 73, 73.5, 75, 77.5, ], dtype=np.float32)
    # sza_arr = np.array([50, 55, 60, np.round(sza_avg, 2), 65, 70, 75, 77.5, 80, 82.5, 85, 87], dtype=np.float32)
    
    sza_arr = np.array([50, 52.5, 55, 57.5, 60, np.round(sza_avg, 2), 62.5, 65, 67.5, 70, 71.5, 72.5, 73, 73.5, 75,], dtype=np.float32)

        
        
    # CRE outputs live in the dedicated ``..._cre`` folder written by cre_sim.
    sky_tag = 'clear' if clear_sky else 'sat_cloud'
    fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_{sky_tag}_cre'
    fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_{sky_tag}_cre'
    
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
                    
                # if not os.path.exists(output_csv_name_sw):
                #     print(f"File {output_csv_name_sw} not found. Skipping ...")

                # if not os.path.exists(output_csv_name_lw):
                #     print(f"File {output_csv_name_lw} not found. Skipping ...")
                    
                # continue

                # read csv and extract simulated fluxes
                with open(output_csv_name_sw, 'r') as f:
                    df_sw = pd.read_csv(f)
                    
                with open(output_csv_name_lw, 'r') as f:
                    df_lw = pd.read_csv(f)
                    
                # with open(output_csv_name_sw.replace('.csv', '_2.csv'), 'r') as f:
                #     df_sw_2 = pd.read_csv(f)
                    
                # with open(output_csv_name_lw.replace('.csv', '_2.csv'), 'r') as f:
                #     df_lw_2 = pd.read_csv(f)
                
                    
                cot_list = df_sw['cot'].values[:20]
                cwp_list = df_sw['cwp'].values[:20]
                cer_list = df_sw['cer'].values[:20]
                cth_list = df_sw['cth'].values[:20]
                cbh_list = df_sw['cbh'].values[:20]
                sza_list = df_sw['sza'].values[:20]
                Fup_sfc_sw = df_sw['Fup_sfc'].values[:20]
                Fdn_sfc_sw = df_sw['Fdn_sfc'].values[:20]
                Fup_sfc_lw = df_lw['Fup_sfc'].values[:20]
                Fdn_sfc_lw = df_lw['Fdn_sfc'].values[:20]
                
                # print("cwp_list shape:", cwp_list.shape)
                # print("cwp:", cwp_list)
                # print("cot_list shape:", cot_list.shape)
                # print("Fup_sfc_lw shape:", Fup_sfc_lw.shape)
                
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
                
                cwp_sort_ind = np.argsort(cwp_list)
                cot_list = cot_list[cwp_sort_ind]
                cwp_list = cwp_list[cwp_sort_ind]
                cer_list = cer_list[cwp_sort_ind]
                cth_list = cth_list[cwp_sort_ind]
                cbh_list = cbh_list[cwp_sort_ind]
                sza_list = sza_list[cwp_sort_ind]
                Fup_sfc_sw = Fup_sfc_sw[cwp_sort_ind]
                Fdn_sfc_sw = Fdn_sfc_sw[cwp_sort_ind]
                Fup_sfc_lw = Fup_sfc_lw[cwp_sort_ind]
                Fdn_sfc_lw = Fdn_sfc_lw[cwp_sort_ind]
                
                sza_list = np.round(sza_list, 2)
                
                
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
                
                # select = np.array([cwp%0.5==0 for cwp in cwp_cre])
                # case_sel = ~select
                
                select = np.array([cwp>=0 for cwp in cwp_cre])
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
                
                # if manual_alb_i is None and sza_sim % 0.5 > 0:
                #     plt.figure()
                #     plt.plot(cwp_cre, F_sfc_net_cre, '-o', label='NET CRE')
                #     plt.xlabel('CWP (g/m2)')
                #     plt.ylabel('Surface NET CRE (W/m2)')
                #     plt.title(f'Surface NET CRE vs. CWP on {date_s} (sza={sza_sim:.2f}°, alb broadband={broadband_alb})')
                #     plt.hlines(0, xmin=0, xmax=np.max(cwp_cre), colors='gray', linestyles='dashed')
                #     plt.legend()
                #     plt.grid()
                #     plt.show()
                #     return
                
            alb_wvl_all.append(ext_wvl)
            alb_all.append(ext_alb)
            broadband_alb_all.append(broadband_alb)
            broadband_alb_ori_all.append(broadband_alb_ori)
            
            if manual_alb_i is None:
                print(f"Processed default alb: broadband_alb={broadband_alb}, broadband_alb_ori={broadband_alb_ori}")
            else:
                print(f"Processed manual alb {manual_alb_i}: broadband_alb={broadband_alb}, broadband_alb_ori={broadband_alb_ori}")
        
        
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
    
    
    print("set df_real_all['sza']:", sorted(list(set(df_real_all['sza'].values))))
    print("set df_real_all['cwp']:", sorted(list(set(df_real_all['cwp'].values))))
    df_real_all = df_real_all.loc[np.logical_and(df_real_all['sza'].values%5 > 0, df_real_all['cwp'].values%0.5 > 0), :]
    print(df_real_all)
    for col in df_real_all.columns:
        print(f"{col} data:", df_real_all[col].values)

    
    broadband_alb_all_unique = sorted(list(set(df_all['broadband_alb'].values)), reverse=True) # sort largest to smallest
    print("broadband_alb_all_unique:", broadband_alb_all_unique)
    print("broadband_alb_all:", broadband_alb_all)
    
    sza_mesh, broadband_alb_mesh = np.meshgrid(sza_arr, broadband_alb_all_unique, indexing='ij')
    cos_sza_arr = np.cos(np.deg2rad(sza_arr.copy()))
    print("cos_sza_arr:", cos_sza_arr)
    cos_sza_mesh, _ = np.meshgrid(cos_sza_arr, broadband_alb_all_unique, indexing='ij')
    cwp_zero_arr = np.zeros_like(cos_sza_mesh, dtype=np.float32) * np.nan
    for i in range(sza_mesh.shape[0]):
        for j in range(sza_mesh.shape[1]):
            sza_sim = sza_arr[i]
            broadband_alb = broadband_alb_all_unique[j]
            sza_value_unique = np.unique(df_all['sza'].values)
            sza_sim_select = sza_value_unique[np.argmin(np.abs(sza_value_unique - sza_sim))]
            if np.abs(sza_sim - sza_sim_select) > 1e-2:
                print(f"  Skip sza_sim: {sza_sim}, broadband_alb: {broadband_alb} due to no matching sza in simulations.")
                continue
            df_select_mask = np.logical_and((df_all['sza'].values==sza_sim_select), (df_all['broadband_alb'].values==broadband_alb))
            df_sub = df_all.loc[df_select_mask, :]
            cwp_arr = df_sub['cwp'].values
            F_cre_net_arr = df_sub['F_sfc_net_cre'].values
        
            
            # find zero crossing
            zero_crossings = np.where(np.diff(np.sign(F_cre_net_arr)))[0]
            # print(f'  Zero crossings indices: {zero_crossings}')
            zero_crossings_tmp = []
            for zero_crossing in zero_crossings[1:]:
                cwp1 = cwp_arr[zero_crossing]
                cwp2 = cwp_arr[zero_crossing + 1]
                F1 = F_cre_net_arr[zero_crossing]
                F2 = F_cre_net_arr[zero_crossing + 1]
                # linear interpolation to find the root
                cwp_zero = cwp1 - F1 * (cwp2 - cwp1) / (F2 - F1)
                # print(f' sza: {sza_sim}, broadband_alb: {broadband_alb}')
                # print(f'  Zero crossing at CWP: {cwp_zero:.2f} g/m2 between {cwp1:.2f} and {cwp2:.2f} g/m2')
                zero_crossings_tmp.append(cwp_zero)
            if len(zero_crossings_tmp) > 0:
                # sza_sim_list.append(sza_sim)
                # broadband_alb_sim_list.append(broadband_alb)
                # cwp_zero_list.append(np.nanmean(zero_crossings_tmp))
                cwp_zero_arr[i, j] = zero_crossings_tmp[0]  # take the first zero crossing
                
            #     # if sza_sim >= 70 or sza_sim%0.5>0:
            #     if sza_sim >= 70  and np.abs(broadband_alb - 0.666) < 1e-3:
            #         plt.close('all')
            #         fig, ax = plt.subplots(figsize=(8, 6))
            #         ax.plot(cwp_arr, F_cre_net_arr, '-', label='Net CRE')
            #         ax.scatter(zero_crossings_tmp[0], 0, c='C0', marker='o', s=50, label='Zero Crossing' if len(zero_crossings_tmp)>0 else '')
            #         ax.hlines(0, xmin=0, xmax=np.max(cwp_arr), colors='gray', linestyles='dashed')
            #         ax.set_xlabel('Cloud Liquid Water Path (g/m2)', fontsize=14)
            #         ax.set_ylabel('Surface Net CRE (W/m2)', fontsize=14)
            #         ax.set_title(f'Surface Net CRE vs. LWP on {date_s}, SZA: {sza_sim:.2f}, Broadband Albedo: {broadband_alb}', fontsize=16)
            #         ax.text(0.05, 0.9, f'Zero Crossing at CWP: {zero_crossings_tmp[0]:.2f} g/m2', transform=ax.transAxes, fontsize=12, verticalalignment='top')
            #         ax.legend(fontsize=12)
            #         fig.tight_layout()
            #         plt.show()
            #         plt.close(fig)
                    
            # elif sza_sim >= 71 and (np.abs(broadband_alb - 0.666) < 1e-3 or np.abs(broadband_alb - 0.704) < 1e-3):
            #     print(f'  No zero crossing found for sza: {sza_sim}, broadband_alb: {broadband_alb}')
            #     plt.close('all')
            #     fig, ax = plt.subplots(figsize=(8, 6))
            #     ax.plot(cwp_arr, F_cre_net_arr, '-', label='Net CRE')
            #     # ax.scatter(zero_crossings_tmp[0], 0, c='C0', marker='o', s=50, label='Zero Crossing' if len(zero_crossings_tmp)>0 else '')
            #     ax.hlines(0, xmin=0, xmax=np.max(cwp_arr), colors='gray', linestyles='dashed')
            #     ax.set_xlabel('Cloud Liquid Water Path (g/m2)', fontsize=14)
            #     ax.set_ylabel('Surface Net CRE (W/m2)', fontsize=14)
            #     ax.set_title(f'Surface Net CRE vs. LWP on {date_s}, SZA: {sza_sim:.2f}, Broadband Albedo: {broadband_alb}', fontsize=16)
            #     # ax.text(0.05, 0.9, f'Zero Crossing at CWP: {zero_crossings_tmp[0]:.2f} g/m2', transform=ax.transAxes, fontsize=12, verticalalignment='top')
            #     ax.legend(fontsize=12)
            #     fig.tight_layout()
            #     plt.show()
            #     plt.close(fig)
    
    plt.close('all')
    # sza_select = 61.46
    sza_select = 61.93
    sza_select_ind = np.argmin(np.abs(sza_arr - sza_select))
    broadband_alb_select = 0.704
    broadband_alb_delect_ind = np.argmin(np.abs(np.array(broadband_alb_all_unique) - broadband_alb_select))
    df_alb_sel = df_all.loc[(df_all['broadband_alb'].values==broadband_alb_all_unique[broadband_alb_delect_ind]), :]

    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # for sza_val in sza_arr:
    #     sza_select_ind = np.argmin(np.abs(sza_arr - sza_val))
    #     if cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind] is np.nan:
    #         continue
    #     df_sza_sel = df_all.loc[(df_all['broadband_alb'].values==broadband_alb_all_unique[broadband_alb_delect_ind]) & (df_all['sza'].values==sza_val), :]
    #     ax.plot(df_sza_sel['cwp'].values, df_sza_sel['F_sfc_net_cre'].values, label=f'SZA: {sza_val:.2f}')  
    #     print(f"SZA: {sza_val:.2f}, Zero Crossing at CWP: {cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind]:.2f} g/m2")  
    #     ax.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind], 0, marker='x', s=100, label=f"SZA: {sza_val:.2f}")
    # ax.hlines(0, xmin=0, xmax=np.max(df_alb_sel['cwp'].values), colors='gray', linestyles='dashed')
    # ax.set_xlabel('Cloud Liquid Water Path (g/m2)', fontsize=14)
    # ax.set_ylabel('Surface Net CRE (W/m2)', fontsize=14)
    # ax.set_title(f'Surface Net CRE vs. LWP on {date_s}, SZA: {sza_select}, Broadband Albedo: {broadband_alb_select}', fontsize=16)
    # ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # fig.tight_layout()
    # plt.show()
    # plt.close()

    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.plot([50, 54, 60, 65, 68, 70, 72, 73, 75], [0.653, 0.639, 0.614, 0.583, 0.562, 0.545, 0.517, 0.500, 0.464], linewidth=1.5, color='r')
    ax.set_xlim(30, 90)
    ax.set_ylim(0, 1)
    # Set the background color of the Figure to transparent
    fig.patch.set_facecolor('none')
    # Alternatively: fig.patch.set_alpha(0.0)

    # Set the background color of the Axes (plot area) to transparent
    ax.patch.set_facecolor('none')
    # Alternatively: ax.patch.set_alpha(0.0)
    fig.savefig(f'fig/{date_s}/shupe_paper_lwp_30_test.png', dpi=300, bbox_inches='tight')

    
    
    # Create a ScalarMappable
    color_series = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',]
    
    plt.close('all')
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(11, 11, figure=fig, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[:5, :6])
    ax2 = fig.add_subplot(gs[6:, :6])
    ax3 = fig.add_subplot(gs[2:10, 7:])
    # sza_select = 61.46
    sza_select = 61.93
    sza_unique_sorted = np.array(sorted(list(set(df_all['sza'].values)), reverse=False))
    cos_sza_unique_sorted = np.cos(np.deg2rad(sza_unique_sorted))
    sza_select_ind = np.argmin(np.abs(cos_sza_unique_sorted - np.cos(np.deg2rad(sza_select))))
    sza_real_df_all = df_all.loc[df_all['sza']==sza_unique_sorted[sza_select_ind], :]
    sza_real_df_real_all = df_real_all.loc[df_real_all['sza']==sza_unique_sorted[sza_select_ind], :]
    print("sum df_all['sza']==sza_unique_sorted[sza_select_ind]:", np.sum(df_all['sza']==sza_unique_sorted[sza_select_ind]))
    print("sza_unique_sorted[sza_select_ind]:", sza_unique_sorted[sza_select_ind])
    print("sza_unique_sorted:", sza_unique_sorted)
    print("sza_real_df_all length:", len(sza_real_df_all))
    print("sza_real_df_real_all length:", len(sza_real_df_real_all))
    for i in range(5):
        broadband_alb_i = broadband_alb_all[i]
        df_select_mask = sza_real_df_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_all_i = sza_real_df_all.loc[df_select_mask, :]
        df_real_mask = sza_real_df_real_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_real_all_i = sza_real_df_real_all.loc[df_real_mask, :]
        
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_sw_cre'].values, '--', color=color_series[i], alpha=0.5)
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_lw_cre'].values, '-.', color=color_series[i], alpha=0.5)
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, '-', color=color_series[i], label=f'Albedo-{i+1}')
        if i == 2:
            ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k')
            ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind], 0, color=color_series[i], marker='^', s=100, label=f'Zero Crossing Albedo-{i+1}')
            real_cond_color = color_series[i]
            print(f"real net CRE:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind])
                
        # ax2.plot(alb_wvl_all[i], alb_all[i], '-', color=color_series[i], label=f'Extended Broadband Albedo: {broadband_alb_all[i]:.3f} (Original: {broadband_alb_ori_all[i]:.3f})')
        broadband_alb_ind = np.argmin(np.abs(np.array(broadband_alb_all) - broadband_alb_i))
        ax2.plot(alb_wvl_all[broadband_alb_ind], alb_all[broadband_alb_ind], '-', color=color_series[i], label=f'Extended Broadband Albedo: {broadband_alb_all[i]:.3f}')

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
    
    level_labels = [20, 25, 30, 40, 50, 60, 70, 80, 100, 125, 150, 175, 200, 300,  ]
    
    # cc1 = ax3.scatter(cos_sza_mesh.flatten(), broadband_alb_mesh.flatten(), c=cwp_zero_arr, s=50, alpha=0.5, cmap='jet', vmin=20, vmax=300, zorder=3)
    
    
    print("cwp_zero_arr min and max:", np.nanmin(cwp_zero_arr), np.nanmax(cwp_zero_arr))

    cc = ax3.contour(cos_sza_mesh, broadband_alb_mesh, cwp_zero_arr, levels=level_labels, cmap='jet', vmin=10, vmax=350 , zorder=2)
    ax3.clabel(cc, level_labels, fontsize=12, colors='k')
    # cc = ax3.contour(sza_mesh, broadband_alb_mesh, cwp_zero_arr, levels=20, cmap='jet')
    # ax3.clabel(cc, cc.levels, fontsize=12, colors='k')
    
    shupe_sza = np.array([50, 54, 60, 65, 68, 70, 72, 73, 75])
    shupe_alb = np.array([0.653, 0.639, 0.614, 0.583, 0.562, 0.545, 0.517, 0.500, 0.464])
    shupe_cos_sza = np.cos(np.deg2rad(shupe_sza.copy()))
    shupe_lwp = np.ones_like(shupe_alb, dtype=np.float32)*30
    ax3.plot(shupe_cos_sza, shupe_alb, linewidth=1.5, color='orange', label='LWP=30 in Shupe and Intrieri (2003)')
    # cc = ax3.contourf(sza_mesh, broadband_alb_mesh, cwp_zero_arr, cmap='jet', vmin=20, vmax=300, zorder=1)
    ax3.set_xlabel('cos[Solar Zenith Angle]', fontsize=14)
    ax3.set_ylabel('Broadband Albedo', fontsize=14)
    ax3.legend(fontsize=12, loc='center', bbox_to_anchor=(0.5, -0.15), )#frame=None)
    ax3.text(0.28,  0.74, 'Contour levels:\n LWP in $\mathrm{g/m^2}$', fontsize=12)

    
    
    # cbar = fig.colorbar(cc , ax=ax3, orientation='vertical', pad=0.02, shrink=0.8)
    # cbar.set_label('Critical LWP ($\mathrm{g/m^2}$)',
    #                fontsize=14)
    cos_sza_real = np.cos(np.deg2rad(sza_unique_sorted[sza_select_ind]))
    ax3.scatter(cos_sza_real, 0.704, color=real_cond_color, marker='^', s=100, label='Flight Case SZA and Albedo', zorder=4 )
    # ax3.set_xlim(50, 80)
    ax3.set_xlim(np.cos(np.deg2rad(75)), np.cos(np.deg2rad(50)))
    ax3.set_ylim(np.nanmin(broadband_alb_mesh), np.nanmax(broadband_alb_mesh))
    
    # print("cos_sza_real:", cos_sza_real)
    
    # ax3.set_xticks([np.cos(np.deg2rad(angle)) for angle in range(75, 45, -5)],  labels=[f'{angle}°' for angle in range(75, 45, -5)])
    
    ax3.tick_params(
                    axis='x',         # Apply to the x-axis
                    bottom=True,      # Show ticks on the bottom
                    top=False,         # Show ticks on the top
                    labelbottom=True, # Show labels on the bottom
                    labeltop=False,    # Show labels on the top
                        # Optional: adjust label rotation if needed
                        # labelrotation=45,
                    )   
    # set ax3 with both top and bottom x-axis
    ax3_top = ax3.secondary_xaxis('top')
    ax3_top.set_xlabel('Solar Zenith Angle (degrees)', fontsize=14, labelpad=15)
    ax3_top.set_xticks([np.cos(np.deg2rad(angle)) for angle in range(75, 45, -5)], labels=[f'{angle}°' for angle in range(75, 45, -5)])
    ax3_top.tick_params(
                    axis='x',         # Apply to the x-axis
                    labelsize=12,)
    # ax3_top.set_xticklabels(ax3_top.get_xticklabels(), rotation=0)
    
    for ax, subcase in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        ax.tick_params(axis='both', which='major', labelsize=12)
        if ax == ax3:
            ax.text(0.0, 1.08 , subcase, transform=ax.transAxes, fontsize=16, va='bottom', ha='left')
        else:
            ax.text(0.0, 1.03, subcase, transform=ax.transAxes, fontsize=16, va='bottom', ha='left')
    fig.savefig(f'fig/{date_s}/surface_cre_vs_lwp_all_alb_{date_s}_{case_tag}_combined.png', dpi=300, bbox_inches='tight')
    
    
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    for i in range(5):
        broadband_alb_i = broadband_alb_all[i]
        df_select_mask = sza_real_df_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_all_i = sza_real_df_all.loc[df_select_mask, :]
        df_real_mask = sza_real_df_real_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_real_all_i = sza_real_df_real_all.loc[df_real_mask, :]
        
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, '-', color=color_series[i],)# label=f'Albedo-{i+1}')
        if i == 2:
            ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k', label=f'Real Case net CRE @ SSFR extended albedo={broadband_alb_all[i]:.3f}')
            ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind], 0, color=color_series[i], marker='*', s=100, label=f'Zero-crossing @ SSFR extended albedo={broadband_alb_all[i]:.3f}')
            start_cwp = sza_real_df_real_all_i['cwp'].values[0]
            start_Fnet = sza_real_df_real_all_i['F_sfc_net_cre'].values[0]-3
            real_cond_color = color_series[i] 
            print(f"real net CRE:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind])
            
        if i == 3:
            broadband_alb_delect_ind_0655 = np.argmin(np.abs(np.array(broadband_alb_all_unique) - 0.655))
            # sza_real_df_all_i_real_finterp = interp1d(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, kind='linear', fill_value='extrapolate')
            # F_sfc_net_cre_real_at_cwp_zero_0655 = sza_real_df_all_i_real_finterp(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind
            ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k', label=f'net CRE @ ERA5 albedo={broadband_alb_all[i]:.3f}')
            ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655], 0, color=color_series[i], marker='^', s=100, label=f'Zero-crossing @ ERA5 albedo={broadband_alb_all[i]:.3f}')

            end_cwp = sza_real_df_real_all_i['cwp'].values[0]
            end_Fnet = sza_real_df_real_all_i['F_sfc_net_cre'].values[0]+4
            print(f"real net CRE for albedo 0.655:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp for albedo 0.655:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655])
            ax1.arrow(start_cwp, start_Fnet, 
                      end_cwp - start_cwp, end_Fnet - start_Fnet,
                      head_width=2, head_length=2, 
                      lw=1.5,
                      fc='k', ec='k', linestyle='-')
            
    ax1.set_xlabel('Cloud Liquid Water Path $\mathrm{(g/m^2)}$',
                   fontsize=14)
    ax1.set_ylabel('Surface Net CRE $\mathrm{(W/m^2)}$', 
                   fontsize=14)
    ax1.hlines(0, xmin=0, xmax=np.max(sza_real_df_all['cwp'].values), colors='gray', linestyles='dashed')
    ax1.set_xlim(0, 200)
    ax1.set_ylim(-65, 30)
    ax1.legend(fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    for i in range(5):
        broadband_alb_i = broadband_alb_all[i]
        broadband_alb_ind = np.argmin(np.abs(np.array(broadband_alb_all) - broadband_alb_i))
        ax2.plot(alb_wvl_all[broadband_alb_ind], alb_all[broadband_alb_ind], '-', color=color_series[i], label=f'Extended Broadband Albedo: {broadband_alb_all[i]:.3f}')
    ax2.set_xlabel('Wavelength (nm)', fontsize=14)
    ax2.set_ylabel('Surface Albedo', fontsize=14)
    ax2.hlines(0, xmin=300, xmax=4000, colors='gray', linestyles='dashed')
    ax2.set_xlim(300, 4000)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.tick_params(axis='both', which='major', labelsize=12)
    for ax, subcase in zip([ax1, ax2], ['(a)', '(b)']):
        ax.text(0.0, 1.03, subcase, transform=ax.transAxes, fontsize=16, va='bottom', ha='left')
    
    fig.savefig(f'fig/{date_s}/surface_net_cre_vs_lwp_5_alb_{date_s}_{case_tag}_combined.png', dpi=300, bbox_inches='tight')
    
    
    
    plt.close('all')
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    for i in range(5):
        broadband_alb_i = broadband_alb_all[i]
        df_select_mask = sza_real_df_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_all_i = sza_real_df_all.loc[df_select_mask, :]
        df_real_mask = sza_real_df_real_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_real_all_i = sza_real_df_real_all.loc[df_real_mask, :]
        
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, '-', color=color_series[i],)# label=f'Albedo-{i+1}')
        if i == 2:
            ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k', label=f'Real Case net CRE @ SSFR extended albedo={broadband_alb_all[i]:.3f}')
            ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind], 0, color=color_series[i], marker='*', s=100, label=f'Zero-crossing @ SSFR extended albedo={broadband_alb_all[i]:.3f}')
            start_cwp = sza_real_df_real_all_i['cwp'].values[0]
            start_Fnet = sza_real_df_real_all_i['F_sfc_net_cre'].values[0]-3
            real_cond_color = color_series[i] 
            print(f"real net CRE:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind])
            
        if i == 3:
            broadband_alb_delect_ind_0655 = np.argmin(np.abs(np.array(broadband_alb_all_unique) - 0.655))
            # sza_real_df_all_i_real_finterp = interp1d(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, kind='linear', fill_value='extrapolate')
            # F_sfc_net_cre_real_at_cwp_zero_0655 = sza_real_df_all_i_real_finterp(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind
            ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k', label=f'net CRE @ ERA5 albedo={broadband_alb_all[i]:.3f}')
            ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655], 0, color=color_series[i], marker='^', s=100, label=f'Zero-crossing @ ERA5 albedo={broadband_alb_all[i]:.3f}')

            end_cwp = sza_real_df_real_all_i['cwp'].values[0]
            end_Fnet = sza_real_df_real_all_i['F_sfc_net_cre'].values[0]+4
            print(f"real net CRE for albedo 0.655:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp for albedo 0.655:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655])
            ax1.arrow(start_cwp, start_Fnet, 
                      end_cwp - start_cwp, end_Fnet - start_Fnet,
                      head_width=2, head_length=2, 
                      lw=1.5,
                      fc='k', ec='k', linestyle='-')
            
    ax1.set_xlabel('Cloud Liquid Water Path $\mathrm{(g/m^2)}$',
                   fontsize=14)
    ax1.set_ylabel('Surface Net CRE $\mathrm{(W/m^2)}$', 
                   fontsize=14)
    ax1.hlines(0, xmin=0, xmax=np.max(sza_real_df_all['cwp'].values), colors='gray', linestyles='dashed')
    xmin, xmax = 0, 200
    ymin, ymax = -65, 30
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.legend(fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # fill the y>0 with light red color, and y<0 with light blue color, make the color transparent and zorder to be 0
    ax1.fill_between([xmin, xmax], 0, ymax, color='lightcoral', alpha=0.5, zorder=0)
    ax1.fill_between([xmin, xmax], ymin, 0, color='lightblue', alpha=0.5, zorder=0)
    
    
    fig.savefig(f'fig/{date_s}/surface_net_cre_vs_lwp_5_alb_{date_s}_{case_tag}_upper.png', dpi=300, bbox_inches='tight')
    
    
    plt.close('all')
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    for i in range(5):
        broadband_alb_i = broadband_alb_all[i]
        df_select_mask = sza_real_df_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_all_i = sza_real_df_all.loc[df_select_mask, :]
        df_real_mask = sza_real_df_real_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_real_all_i = sza_real_df_real_all.loc[df_real_mask, :]
        
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, '-', color=color_series[i],)# label=f'Albedo-{i+1}')
        if i == 2:
            ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k', label=f'Real Case net CRE @ SSFR extended albedo={broadband_alb_all[i]:.3f}')
            ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind], 0, color=color_series[i], marker='*', s=100, label=f'Zero-crossing @ SSFR extended albedo={broadband_alb_all[i]:.3f}')
            start_cwp = sza_real_df_real_all_i['cwp'].values[0]
            start_Fnet = sza_real_df_real_all_i['F_sfc_net_cre'].values[0]-3 +3 -4
            real_cond_color = color_series[i] 
            print(f"real net CRE:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind])
            
        if i == 3:
            broadband_alb_delect_ind_0655 = np.argmin(np.abs(np.array(broadband_alb_all_unique) - 0.655))
            # sza_real_df_all_i_real_finterp = interp1d(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, kind='linear', fill_value='extrapolate')
            # F_sfc_net_cre_real_at_cwp_zero_0655 = sza_real_df_all_i_real_finterp(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind
            ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k', label=f'net CRE @ ERA5 albedo={broadband_alb_all[i]:.3f}')
            ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655], 0, color=color_series[i], marker='^', s=100, label=f'Zero-crossing @ ERA5 albedo={broadband_alb_all[i]:.3f}')

            end_cwp = sza_real_df_real_all_i['cwp'].values[0]
            end_Fnet = sza_real_df_real_all_i['F_sfc_net_cre'].values[0]+4 -4 +2
            print(f"real net CRE for albedo 0.655:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp for albedo 0.655:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655])
            # ax1.arrow(start_cwp, start_Fnet, 
            #           end_cwp - start_cwp, end_Fnet - start_Fnet,
            #           head_width=2, head_length=2, 
            #           lw=1.5,
            #           fc='k', ec='k', linestyle='-')
            ax1.arrow(end_cwp, end_Fnet, 
                      - end_cwp + start_cwp, - end_Fnet + start_Fnet,
                      head_width=2, head_length=2, 
                      lw=2,
                      fc='k', ec='k', linestyle='-')
            
    ax1.set_xlabel('Cloud Liquid Water Path $\mathrm{(g/m^2)}$',
                   fontsize=14)
    ax1.set_ylabel('Surface Net CRE $\mathrm{(W/m^2)}$', 
                   fontsize=14)
    ax1.hlines(0, xmin=0, xmax=np.max(sza_real_df_all['cwp'].values), colors='gray', linestyles='dashed')
    xmin, xmax = 0, 200
    ymin, ymax = -65, 30
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    # ax1.legend(fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # fill the y>0 with light red color, and y<0 with light blue color, make the color transparent and zorder to be 0
    ax1.fill_between([xmin, xmax], 0, ymax, color='lightcoral', alpha=0.5, zorder=0)
    ax1.fill_between([xmin, xmax], ymin, 0, color='lightblue', alpha=0.5, zorder=0)
    
    
    fig.savefig(f'fig/{date_s}/surface_net_cre_vs_lwp_5_alb_{date_s}_{case_tag}_upper_1.png', dpi=300, bbox_inches='tight')
    
    
    plt.close('all')
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    for i in range(5):
        broadband_alb_i = broadband_alb_all[i]
        df_select_mask = sza_real_df_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_all_i = sza_real_df_all.loc[df_select_mask, :]
        df_real_mask = sza_real_df_real_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_real_all_i = sza_real_df_real_all.loc[df_real_mask, :]
        
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, '-', color=color_series[i],)# label=f'Albedo-{i+1}')
        if i == 2:
            # ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k', label=f'Real Case net CRE @ SSFR extended albedo={broadband_alb_all[i]:.3f}')
            # ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind], 0, color=color_series[i], marker='*', s=100, label=f'Zero-crossing @ SSFR extended albedo={broadband_alb_all[i]:.3f}')
            start_cwp = sza_real_df_real_all_i['cwp'].values[0]
            start_Fnet = sza_real_df_real_all_i['F_sfc_net_cre'].values[0]-3
            real_cond_color = color_series[i] 
            print(f"real net CRE:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind])
            
        if i == 3:
            broadband_alb_delect_ind_0655 = np.argmin(np.abs(np.array(broadband_alb_all_unique) - 0.655))
            # sza_real_df_all_i_real_finterp = interp1d(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, kind='linear', fill_value='extrapolate')
            # F_sfc_net_cre_real_at_cwp_zero_0655 = sza_real_df_all_i_real_finterp(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind
            # ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k', label=f'net CRE @ ERA5 albedo={broadband_alb_all[i]:.3f}')
            # ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655], 0, color=color_series[i], marker='^', s=100, label=f'Zero-crossing @ ERA5 albedo={broadband_alb_all[i]:.3f}')

            end_cwp = sza_real_df_real_all_i['cwp'].values[0]
            end_Fnet = sza_real_df_real_all_i['F_sfc_net_cre'].values[0]+4
            print(f"real net CRE for albedo 0.655:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp for albedo 0.655:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655])
            # ax1.arrow(start_cwp, start_Fnet, 
            #           end_cwp - start_cwp, end_Fnet - start_Fnet,
            #           head_width=2, head_length=2, 
            #           lw=1.5,
            #           fc='k', ec='k', linestyle='-')
            
    ax1.set_xlabel('Cloud Liquid Water Path $\mathrm{(g/m^2)}$',
                   fontsize=14)
    ax1.set_ylabel('Surface Net CRE $\mathrm{(W/m^2)}$', 
                   fontsize=14)
    ax1.hlines(0, xmin=0, xmax=np.max(sza_real_df_all['cwp'].values), colors='gray', linestyles='dashed')
    xmin, xmax = 0, 200
    ymin, ymax = -65, 30
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    # ax1.legend(fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # fill the y>0 with light red color, and y<0 with light blue color, make the color transparent and zorder to be 0
    ax1.fill_between([xmin, xmax], 0, ymax, color='lightcoral', alpha=0.5, zorder=0)
    ax1.fill_between([xmin, xmax], ymin, 0, color='lightblue', alpha=0.5, zorder=0)
    
    
    fig.savefig(f'fig/{date_s}/surface_net_cre_vs_lwp_5_alb_{date_s}_{case_tag}_upper_2.png', dpi=300, bbox_inches='tight')
    
    plt.close('all')
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    for i in [3]:
        broadband_alb_i = broadband_alb_all[i]
        df_select_mask = sza_real_df_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_all_i = sza_real_df_all.loc[df_select_mask, :]
        df_real_mask = sza_real_df_real_all['broadband_alb'].values==broadband_alb_i
        sza_real_df_real_all_i = sza_real_df_real_all.loc[df_real_mask, :]
        
        ax1.plot(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, '-', color=color_series[i],)# label=f'Albedo-{i+1}')
        if i == 2:
            # ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k', label=f'Real Case net CRE @ SSFR extended albedo={broadband_alb_all[i]:.3f}')
            # ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind], 0, color=color_series[i], marker='*', s=100, label=f'Zero-crossing @ SSFR extended albedo={broadband_alb_all[i]:.3f}')
            start_cwp = sza_real_df_real_all_i['cwp'].values[0]
            start_Fnet = sza_real_df_real_all_i['F_sfc_net_cre'].values[0]-3
            real_cond_color = color_series[i] 
            print(f"real net CRE:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind])
            
        if i == 3:
            broadband_alb_delect_ind_0655 = np.argmin(np.abs(np.array(broadband_alb_all_unique) - 0.655))
            # sza_real_df_all_i_real_finterp = interp1d(sza_real_df_all_i['cwp'].values, sza_real_df_all_i['F_sfc_net_cre'].values, kind='linear', fill_value='extrapolate')
            # F_sfc_net_cre_real_at_cwp_zero_0655 = sza_real_df_all_i_real_finterp(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind
            # ax1.scatter(sza_real_df_real_all_i['cwp'].values, sza_real_df_real_all_i['F_sfc_net_cre'].values, color=color_series[i], marker='o', s=50, edgecolors='k', label=f'net CRE @ ERA5 albedo={broadband_alb_all[i]:.3f}')
            # ax1.scatter(cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655], 0, color=color_series[i], marker='^', s=100, label=f'Zero-crossing @ ERA5 albedo={broadband_alb_all[i]:.3f}')

            end_cwp = sza_real_df_real_all_i['cwp'].values[0]
            end_Fnet = sza_real_df_real_all_i['F_sfc_net_cre'].values[0]+4
            print(f"real net CRE for albedo 0.655:", sza_real_df_real_all_i['F_sfc_net_cre'].values)
            print(f"zero crossing cwp for albedo 0.655:", cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655])
            # ax1.arrow(start_cwp, start_Fnet, 
            #           end_cwp - start_cwp, end_Fnet - start_Fnet,
            #           head_width=2, head_length=2, 
            #           lw=1.5,
            #           fc='k', ec='k', linestyle='-')
            
    ax1.set_xlabel('Cloud Liquid Water Path $\mathrm{(g/m^2)}$',
                   fontsize=14)
    ax1.set_ylabel('Surface Net CRE $\mathrm{(W/m^2)}$', 
                   fontsize=14)
    ax1.hlines(0, xmin=0, xmax=np.max(sza_real_df_all['cwp'].values), colors='gray', linestyles='dashed')
    xmin, xmax = 0, 200
    ymin, ymax = -65, 30
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    # ax1.legend(fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # fill the y>0 with light red color, and y<0 with light blue color, make the color transparent and zorder to be 0
    ax1.fill_between([xmin, cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655]], 0, ymax, color='lightcoral', alpha=0.5, zorder=0)
    ax1.fill_between([cwp_zero_arr[sza_select_ind, broadband_alb_delect_ind_0655], xmax], ymin, 0, color='lightblue', alpha=0.5, zorder=0)
    
    
    fig.savefig(f'fig/{date_s}/surface_net_cre_vs_lwp_5_alb_{date_s}_{case_tag}_upper_3.png', dpi=300, bbox_inches='tight')
    
    plt.close('all')
    fig, ax3 = plt.subplots(figsize=(7, 8))
    # sza_select = 61.46
    sza_select = 61.93
    sza_unique_sorted = np.array(sorted(list(set(df_all['sza'].values)), reverse=False))
    cos_sza_unique_sorted = np.cos(np.deg2rad(sza_unique_sorted))
    sza_select_ind = np.argmin(np.abs(cos_sza_unique_sorted - np.cos(np.deg2rad(sza_select))))
    sza_real_df_all = df_all.loc[df_all['sza']==sza_unique_sorted[sza_select_ind], :]
    sza_real_df_real_all = df_real_all.loc[df_real_all['sza']==sza_unique_sorted[sza_select_ind], :]
    print("sum df_all['sza']==sza_unique_sorted[sza_select_ind]:", np.sum(df_all['sza']==sza_unique_sorted[sza_select_ind]))
    print("sza_unique_sorted[sza_select_ind]:", sza_unique_sorted[sza_select_ind])
    print("sza_unique_sorted:", sza_unique_sorted)
    print("sza_real_df_all length:", len(sza_real_df_all))
    print("sza_real_df_real_all length:", len(sza_real_df_real_all))
    
    cc = ax3.contour(cos_sza_mesh, broadband_alb_mesh, cwp_zero_arr, levels=level_labels, cmap='jet', vmin=10, vmax=350 , zorder=2)
    ax3.clabel(cc, level_labels, fontsize=12, colors='k')
    # cc = ax3.contour(sza_mesh, broadband_alb_mesh, cwp_zero_arr, levels=20, cmap='jet')
    # ax3.clabel(cc, cc.levels, fontsize=12, colors='k')
    
    shupe_sza = np.array([50, 54, 60, 65, 68, 70, 72, 73, 75])
    shupe_alb = np.array([0.653, 0.639, 0.614, 0.583, 0.562, 0.545, 0.517, 0.500, 0.464])
    shupe_cos_sza = np.cos(np.deg2rad(shupe_sza.copy()))
    shupe_lwp = np.ones_like(shupe_alb, dtype=np.float32)*30
    ax3.plot(shupe_cos_sza, shupe_alb, linewidth=1.5, color='orange', label='LWP=30 in Shupe and Intrieri (2003)')
    # cc = ax3.contourf(sza_mesh, broadband_alb_mesh, cwp_zero_arr, cmap='jet', vmin=20, vmax=300, zorder=1)
    ax3.set_xlabel('cos[Solar Zenith Angle]', fontsize=14)
    ax3.set_ylabel('Broadband Albedo', fontsize=14)
    ax3.legend(fontsize=12, loc='center', bbox_to_anchor=(0.5, -0.15), )#frame=None)
    ax3.text(0.28,  0.74, 'Contour levels:\n LWP in $\mathrm{g/m^2}$', fontsize=12)

    
    
    # cbar = fig.colorbar(cc , ax=ax3, orientation='vertical', pad=0.02, shrink=0.8)
    # cbar.set_label('Critical LWP ($\mathrm{g/m^2}$)',
    #                fontsize=14)
    cos_sza_real = np.cos(np.deg2rad(sza_unique_sorted[sza_select_ind]))
    ax3.scatter(cos_sza_real, 0.7040066, color='orange', marker='*', s=150, label='SSFR Albedo', zorder=4, alpha=0.7)
    ax3.scatter(cos_sza_real, 0.6548484, color='orange', marker='^', s=150, label='ERA5 Albedo', zorder=4, alpha=0.7)
    ax3.text(cos_sza_real+0.01, 0.7040066-0.002, 'ARCSIX', color='orange', fontsize=12)
    ax3.text(cos_sza_real+0.01, 0.6548484-0.002, 'ERA5', color='orange', fontsize=12)
    # plot arrow from SSFR Albedo to ERA5 Albedo
    ax3.annotate('', xy=(cos_sza_real, 0.701), xytext=(cos_sza_real, 0.657),
                 arrowprops=dict(facecolor='purple', arrowstyle='->', 
                                 edgecolor='purple',
                                 lw=2.5),
                 )
    # ax3.set_xlim(50, 80)
    ax3.set_xlim(np.cos(np.deg2rad(75)), np.cos(np.deg2rad(50)))
    ax3.set_ylim(np.nanmin(broadband_alb_mesh), np.nanmax(broadband_alb_mesh))
    
    # print("cos_sza_real:", cos_sza_real)
    
    # ax3.set_xticks([np.cos(np.deg2rad(angle)) for angle in range(75, 45, -5)],  labels=[f'{angle}°' for angle in range(75, 45, -5)])
    
    ax3.tick_params(
                    axis='x',         # Apply to the x-axis
                    bottom=True,      # Show ticks on the bottom
                    top=False,         # Show ticks on the top
                    labelbottom=True, # Show labels on the bottom
                    labeltop=False,    # Show labels on the top
                        # Optional: adjust label rotation if needed
                        # labelrotation=45,
                    )   
    # set ax3 with both top and bottom x-axis
    ax3_top = ax3.secondary_xaxis('top')
    ax3_top.set_xlabel('Solar Zenith Angle (degrees)', fontsize=14, labelpad=15)
    ax3_top.set_xticks([np.cos(np.deg2rad(angle)) for angle in range(75, 45, -5)], labels=[f'{angle}°' for angle in range(75, 45, -5)])
    ax3_top.tick_params(
                    axis='x',         # Apply to the x-axis
                    labelsize=12,)
    # ax3_top.set_xticklabels(ax3_top.get_xticklabels(), rotation=0)
    

    fig.savefig(f'fig/{date_s}/surface_cre_vs_lwp_all_alb_{date_s}_{case_tag}_combined_contour_only.png', dpi=300, bbox_inches='tight')
    
    
    plt.close('all')
    fig, ax3 = plt.subplots(figsize=(7, 8))
    # sza_select = 61.46
    sza_select = 61.93
    sza_unique_sorted = np.array(sorted(list(set(df_all['sza'].values)), reverse=False))
    cos_sza_unique_sorted = np.cos(np.deg2rad(sza_unique_sorted))
    sza_select_ind = np.argmin(np.abs(cos_sza_unique_sorted - np.cos(np.deg2rad(sza_select))))
    sza_real_df_all = df_all.loc[df_all['sza']==sza_unique_sorted[sza_select_ind], :]
    sza_real_df_real_all = df_real_all.loc[df_real_all['sza']==sza_unique_sorted[sza_select_ind], :]
    print("sum df_all['sza']==sza_unique_sorted[sza_select_ind]:", np.sum(df_all['sza']==sza_unique_sorted[sza_select_ind]))
    print("sza_unique_sorted[sza_select_ind]:", sza_unique_sorted[sza_select_ind])
    print("sza_unique_sorted:", sza_unique_sorted)
    print("sza_real_df_all length:", len(sza_real_df_all))
    print("sza_real_df_real_all length:", len(sza_real_df_real_all))
    
    cc = ax3.contour(cos_sza_mesh, broadband_alb_mesh, cwp_zero_arr, levels=level_labels, cmap='jet', vmin=10, vmax=350 , zorder=2)
    ax3.clabel(cc, level_labels, fontsize=12, colors='k')
    # cc = ax3.contour(sza_mesh, broadband_alb_mesh, cwp_zero_arr, levels=20, cmap='jet')
    # ax3.clabel(cc, cc.levels, fontsize=12, colors='k')
    
    shupe_sza = np.array([50, 54, 60, 65, 68, 70, 72, 73, 75])
    shupe_alb = np.array([0.653, 0.639, 0.614, 0.583, 0.562, 0.545, 0.517, 0.500, 0.464])
    shupe_cos_sza = np.cos(np.deg2rad(shupe_sza.copy()))
    shupe_lwp = np.ones_like(shupe_alb, dtype=np.float32)*30
    ax3.plot(shupe_cos_sza, shupe_alb, linewidth=1.5, color='orange', label='LWP=30 in Shupe and Intrieri (2003)')
    # cc = ax3.contourf(sza_mesh, broadband_alb_mesh, cwp_zero_arr, cmap='jet', vmin=20, vmax=300, zorder=1)
    ax3.set_xlabel('cos[Solar Zenith Angle]', fontsize=14)
    ax3.set_ylabel('Broadband Albedo', fontsize=14)
    ax3.legend(fontsize=12, loc='center', bbox_to_anchor=(0.5, -0.15), )#frame=None)
    ax3.text(0.28,  0.74, 'Contour levels:\n LWP in $\mathrm{g/m^2}$', fontsize=12)

    
    
    # cbar = fig.colorbar(cc , ax=ax3, orientation='vertical', pad=0.02, shrink=0.8)
    # cbar.set_label('Critical LWP ($\mathrm{g/m^2}$)',
    #                fontsize=14)
    cos_sza_real = np.cos(np.deg2rad(sza_unique_sorted[sza_select_ind]))
    # plot arrow from SSFR Albedo to ERA5 Albedo
    # ax3.annotate('', xy=(cos_sza_real, 0.657), xytext=(cos_sza_real, 0.701),
    #              arrowprops=dict(facecolor='purple', arrowstyle='->', 
    #                              edgecolor='purple',
    #                              lw=2.5),
    #              )
    # ax3.set_xlim(50, 80)
    ax3.set_xlim(np.cos(np.deg2rad(75)), np.cos(np.deg2rad(50)))
    ax3.set_ylim(np.nanmin(broadband_alb_mesh), np.nanmax(broadband_alb_mesh))
    
    # print("cos_sza_real:", cos_sza_real)
    
    # ax3.set_xticks([np.cos(np.deg2rad(angle)) for angle in range(75, 45, -5)],  labels=[f'{angle}°' for angle in range(75, 45, -5)])
    
    ax3.tick_params(
                    axis='x',         # Apply to the x-axis
                    bottom=True,      # Show ticks on the bottom
                    top=False,         # Show ticks on the top
                    labelbottom=True, # Show labels on the bottom
                    labeltop=False,    # Show labels on the top
                        # Optional: adjust label rotation if needed
                        # labelrotation=45,
                    )   
    # set ax3 with both top and bottom x-axis
    ax3_top = ax3.secondary_xaxis('top')
    ax3_top.set_xlabel('Solar Zenith Angle (degrees)', fontsize=14, labelpad=15)
    ax3_top.set_xticks([np.cos(np.deg2rad(angle)) for angle in range(75, 45, -5)], labels=[f'{angle}°' for angle in range(75, 45, -5)])
    ax3_top.tick_params(
                    axis='x',         # Apply to the x-axis
                    labelsize=12,)
    # ax3_top.set_xticklabels(ax3_top.get_xticklabels(), rotation=0)
    

    fig.savefig(f'fig/{date_s}/surface_cre_vs_lwp_all_alb_{date_s}_{case_tag}_combined_contour_only_no_symbol.png', dpi=300, bbox_inches='tight')
    
    sys.exit()
    
    
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'hspace': 0.3})
    for i in range(5):
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


def make_default_config():
    """Return the default ARCSIX P-3B FlightConfig."""
    return FlightConfig(
        mission='ARCSIX',
        platform='P3B',
        data_root=_fdir_data_,
        root_mac=_fdir_general_,
        root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data',
    )


def find_catalog_case(case_id):
    """Return one case from the atmospheric-correction catalog (good or bad list)."""
    try:
        from ssfr_atm_corr.case_catalog import ALL_CASE_CATALOG, BAD_CASE_CATALOG
    except ImportError:
        from lrt_sim.ssfr_atm_corr.case_catalog import ALL_CASE_CATALOG, BAD_CASE_CATALOG
    for case in list(ALL_CASE_CATALOG) + list(BAD_CASE_CATALOG):
        if case['id'] == case_id:
            return case
    raise KeyError(f'Unknown CRE case id: {case_id}')


def plot_cre_case(config, case_id, manual_alb=None, overwrite_lrt=False):
    """Plot the CRE simulation results for one atmospheric-correction catalog case."""
    case = find_catalog_case(case_id)
    year, month, day = [int(part) for part in case['date'].split('-')]

    cloud_kwargs = {}
    if 'manual_cloud' in case:
        cloud_kwargs['manual_cloud'] = case['manual_cloud']

    return cre_sim_plot(
        date=datetime.datetime(year, month, day),
        tmhr_ranges_select=case['tmhr_ranges_select'],
        case_tag=case['case_tag'],
        config=config,
        levels=case.get('levels'),
        simulation_interval=case.get('simulation_interval'),
        clear_sky=case.get('clear_sky', True),
        overwrite_lrt=overwrite_lrt,
        manual_alb=manual_alb,
        **cloud_kwargs,
    )


if __name__ == '__main__':

    os.makedirs('./fig', exist_ok=True)
    config = make_default_config()

    # Example: post-process / plot the 2024-06-03 cloudy_atm_corr_2 case
    # (case_004), sweeping a set of manual albedo spectra.
    plot_cre_case(
        config,
        'case_004',
        manual_alb=[
            'sfc_alb_20240606_16.250_16.950_0.50km_cre_alb.dat',
            'sfc_alb_20240607_15.336_15.761_0.12km_cre_alb.dat',
            'sfc_alb_20240603_13.620_13.750_0.32km_cre_alb.dat',
            'sfc_alb_20240613_16.550_17.581_0.22km_cre_alb.dat',
            'sfc_alb_20240725_15.094_15.300_0.11km_cre_alb.dat',
        ],
        overwrite_lrt=True,
    )
