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
from util import *
# mpl.use('Agg')

try:
    from .ssfr_atm_corr.settings import *
    from .ssfr_atm_corr_product_helpers import (
        fill_nan_ffill_bfill,
        ssfr_alb_plot,
        ssfr_up_dn_ratio_plot,
        weighted_broadband_alb,
    )
    from .ssfr_atm_corr.setup import split_tmhr_ranges
except ImportError:
    from ssfr_atm_corr.settings import *
    from ssfr_atm_corr_product_helpers import (
        fill_nan_ffill_bfill,
        ssfr_alb_plot,
        ssfr_up_dn_ratio_plot,
        weighted_broadband_alb,
    )
    from ssfr_atm_corr.setup import split_tmhr_ranges

 
# Per-date h2o_6_end overrides for gas_abs_masking / snowice_alb_fitting
_DATE_H2O_6_END = {
    '20240603': {'mask': 1650, 'fit': 1650},
    '20240807': {'mask': 1550, 'fit': 1570},
}


def atm_corr_processing(date=datetime.datetime(2024, 5, 31),
                        tmhr_ranges_select=[[14.10, 14.27]],
                        case_tag='default',
                        config: Optional[FlightConfig] = None,
                        simulation_interval: Optional[float] = None,  # in minutes, e.g., 10 for 10 minutes
                        clear_sky: bool = True
                        ):
    log = logging.getLogger("atm corr spiral plot")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    doy_s = date.timetuple().tm_yday
    print(f"Processing date: {date_s}, DOY: {doy_s}")
    
    tmhr_ranges_select = split_tmhr_ranges(tmhr_ranges_select, simulation_interval)

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    
    modis_alb_dir = f'{_fdir_general_}/modis_albedo'
    # list all modis albedo files
    modis_alb_files = sorted(glob.glob(os.path.join(modis_alb_dir, f'M*.nc')))
    for fname in modis_alb_files:
        print("Checking modis file:", os.path.basename(fname).split('.')[1])
        if str(doy_s) in os.path.basename(fname).split('.')[1]:
            modis_alb_file = fname
            break
    else:
        modis_alb_file = None

        
    if modis_alb_file is not None:
        with Dataset(modis_alb_file, 'r') as ds:
            modis_lon = ds.variables['Longitude'][:]
            modis_lat = ds.variables['Latitude'][:]
            modis_bands = ds.variables['Bands'][:]
            modis_sur_alb = ds.variables['Albedo_1km'][:]
        
        modis_alb_legs = []
        modis_bands_nm = np.array([float(i) for i in modis_bands[:7]])*1000  # in nm   
            
            
    print("modis_alb_file:", modis_alb_file)
    
    df_solor = pd.read_csv('CU_composite_solar_processed.dat', sep='\s+', header=None)
    wvl_solar = np.array(df_solor.iloc[:, 0])
    flux_solar = np.array(df_solor.iloc[:, 1])#/1000 # convert mW/m^2/nm to W/m^2/nm
    # interpolate to 1 nm grid
    solar_flux_interp = interp1d(wvl_solar, flux_solar, kind='linear', bounds_error=False, fill_value=0.0)
    
    t_hsk = np.array(data_hsk["tmhr"])
    mistmatch_count = 0
    
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    alb_wvl: np.ndarray  # assigned in first loop iteration via initiation block
    initiation = True
    for i in range(len(tmhr_ranges_select)):
        time_start, time_end = tmhr_ranges_select[i][0], tmhr_ranges_select[i][-1]
        fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, time_start, time_end)
        
        with open(fname_pkl, 'rb') as f:
            cld_leg = pickle.load(f)
        alt_avg = np.nanmean(cld_leg['alt'])  # in km
        lon_avg = np.nanmean(data_hsk['lon'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])
        lat_avg = np.nanmean(data_hsk['lat'][(data_hsk['tmhr']>=time_start)&(data_hsk['tmhr']<=time_end)])
        leg_lon_all = cld_leg['lon']
        leg_lat_all = cld_leg['lat']
        leg_alt_all = cld_leg['alt']
        leg_icing_all = cld_leg['ssfr_icing']
        leg_icing_pre_all = cld_leg['ssfr_icing_pre']


        
    
        ratio_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_0.dat'
        update_1_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_1.dat'
        update_2_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_2.dat'
        update_p3_1_fname = f'{_fdir_general_}/sfc_alb/p3_up_dn_ratio_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}_1.dat'
        update_p3_2_fname = f'{_fdir_general_}/sfc_alb/p3_up_dn_ratio_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}_2.dat'
        
        
        if clear_sky:
            fdir_spiral = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
        else:
            fdir_spiral = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'
        ori_csv_name = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km_iteration_0.csv'
        updated_csv_name_1 = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km_iteration_1.csv'
        updated_csv_name_2 = f'{fdir_spiral}/ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km_iteration_2.csv'
        
        if not os.path.exists(ori_csv_name):
            log.error("Original CSV file not found: %s", ori_csv_name)
            continue
        
        
        # p3_up_dn_ratio_20240605_13.79_13.81_5.80_1.dat
        
        alb_ratio = pd.read_csv(ratio_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        alb_1 = pd.read_csv(update_1_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        alb_2 = pd.read_csv(update_2_fname, delim_whitespace=True, comment='#', names=['wvl', 'alb'])
        p3_ratio_1 = pd.read_csv(update_p3_1_fname, delim_whitespace=True, comment='#', names=['wvl', 'ratio'])
        p3_ratio_2 = pd.read_csv(update_p3_2_fname, delim_whitespace=True, comment='#', names=['wvl', 'ratio'])
        
        df_ori = pd.read_csv(ori_csv_name)
        df_upd1 = pd.read_csv(updated_csv_name_1)
        df_upd2 = pd.read_csv(updated_csv_name_2)
        
        corr_factor = df_ori['corr_factor'].values
        
        if initiation:
            time_all = []
            fdn_all = []
            fup_all = []
            lon_all = []
            lat_all = []
            alt_all = []
            sza_all = []
            kt19_sfc_T_all = []
            icing_all = []
            icing_pre_all = []
            fdn_up_ratio_all = []
            toa_expand_all = []
            correction_factor_all = []
            
            alb_wvl = alb_ratio['wvl'].values[1:-1] # skip the first value at 348 nm and last value at 2050 nm
            alb_ratio_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            alb1_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            alb2_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            p3_ratio1_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            p3_ratio2_all = np.zeros((len(tmhr_ranges_select), len(alb_wvl)))
            lon_avg_all = np.zeros(len(tmhr_ranges_select))
            lat_avg_all = np.zeros(len(tmhr_ranges_select))
            lon_min_all = np.zeros(len(tmhr_ranges_select))
            lon_max_all = np.zeros(len(tmhr_ranges_select))
            lat_min_all = np.zeros(len(tmhr_ranges_select))
            lat_max_all = np.zeros(len(tmhr_ranges_select))
            alt_avg_all = np.zeros(len(tmhr_ranges_select))
            cos_sza_avg_all = np.zeros(len(tmhr_ranges_select))

            
            flux_wvl = df_ori['wvl'].values
            ssfr_fup_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fup_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            ssfr_fdn_std_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fdn_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_toa_mean_all_iter0 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_toa_mean_all_iter1 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            simu_fup_toa_mean_all_iter2 = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            toa_mean_all = np.zeros((len(tmhr_ranges_select), len(flux_wvl)))
            
            initiation = False

        if corr_factor.shape[0] != cld_leg['ssfr_zen'].shape[1]:
            print("Mismatch in shape between corr_factor and ssfr_zen, skipping this leg.")
            print("i:", i)
            print("alb_wvl shape:", alb_wvl.shape)
            print("flux_wvl shape:", flux_wvl.shape)
            print("corr_factor shape:", corr_factor.shape)
            print("cld_leg['ssfr_zen'] shape:", cld_leg['ssfr_zen'].shape)
            mistmatch_count += 1
            continue

        time_all.extend(cld_leg['time'])
        ssfr_wvl = cld_leg['ssfr_zen_wvl']
        
        lon_all.extend(cld_leg['lon'])
        lat_all.extend(cld_leg['lat'])
        alt_all.extend(cld_leg['alt'])
        sza_all.extend(cld_leg['sza'])
        fdn_all.extend(cld_leg['ssfr_zen'])
        fup_all.extend(cld_leg['ssfr_nad'])
        kt19_sfc_T_all.extend(cld_leg['kt19_sfc_T'])
        toa_expand_all.extend(cld_leg['ssfr_toa'])
        correction_factor_all.extend(corr_factor * np.ones_like(cld_leg['ssfr_zen']))
        fdn_up_ratio_all.extend(cld_leg['ssfr_nad']/cld_leg['ssfr_zen'])#*corr_factor)
        icing_all.extend(leg_icing_all)
        icing_pre_all.extend(leg_icing_pre_all)
        
        alb_ratio_all[i, :] = alb_ratio['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        alb1_all[i, :] = alb_1['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        alb2_all[i, :] = alb_2['alb'].values[1:-1]  # skip the first value at 348 nm and last value at 2050 nm
        p3_ratio1_all[i, :] = p3_ratio_1['ratio'].values
        p3_ratio2_all[i, :] = p3_ratio_2['ratio'].values
        lon_avg_all[i] = lon_avg.copy()
        lat_avg_all[i] = lat_avg.copy()
        lon_min_all[i] = leg_lon_all.min()
        lon_max_all[i] = leg_lon_all.max()
        lat_min_all[i] = leg_lat_all.min()
        lat_max_all[i] = leg_lat_all.max()
        alt_avg_all[i] = alt_avg.copy()
        cos_sza_avg_all[i] = np.cos(np.deg2rad(np.nanmean(cld_leg['sza'])))
        
        ssfr_fup_mean_all[i, :] = df_ori['ssfr_fup_mean'].values
        ssfr_fdn_mean_all[i, :] = df_ori['ssfr_fdn_mean'].values
        ssfr_fup_std_all[i, :] = df_ori['ssfr_fup_std'].values
        ssfr_fdn_std_all[i, :] = df_ori['ssfr_fdn_std'].values
        toa_mean_all[i, :] = df_ori['toa_mean'].values
        
        
        print(f"date_s: {date_s}, time: {time_start:.3f}-{time_end:.3f}, alt_avg: {alt_avg:.2f} km")

        # find the modis location closest to the flight leg center
        if modis_alb_file is not None:
            dist = np.sqrt((modis_lon - lon_avg)**2 + (modis_lat - lat_avg)**2)
            min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            modis_alb_leg = modis_sur_alb[min_idx[0], min_idx[1], :7]
            modis_alb_legs.append(modis_alb_leg)
    
    print(f"Total mismatch count: {mistmatch_count}")
    
    time_all = np.array(time_all)    
    lon_all = np.array(lon_all)
    lat_all = np.array(lat_all)
    alt_all = np.array(alt_all)
    sza_all = np.array(sza_all)
    fdn_all = np.array(fdn_all)
    fup_all = np.array(fup_all)
    toa_expand_all = np.array(toa_expand_all)
    fdn_up_ratio_all = np.array(fdn_up_ratio_all)
    correction_factor_all = np.array(correction_factor_all)
    icing_all = np.array(icing_all)
    icing_pre_all = np.array(icing_pre_all)
    kt19_sfc_T_all = np.array(kt19_sfc_T_all)

    
    fdn_up_ratio_all_corr = np.full(fdn_up_ratio_all.shape, np.nan)
    fdn_up_ratio_all_corr_fit = np.full(fdn_up_ratio_all.shape, np.nan)
    
    wvl450_ind = np.argmin(np.abs(alb_wvl - 450))
    wvl650_ind = np.argmin(np.abs(alb_wvl - 650))
    wvl1450_ind = np.argmin(np.abs(alb_wvl - 1450))
    wvl1520_ind = np.argmin(np.abs(alb_wvl - 1520)) 
    wvl1650_ind = np.argmin(np.abs(alb_wvl - 1650))
    wvl1710_ind = np.argmin(np.abs(alb_wvl - 1710))
    wvl1800_ind = np.argmin(np.abs(alb_wvl - 1800))
    
    for i in range(fdn_up_ratio_all.shape[0]):
        print("Processing leg %d/%d" % (i+1, fdn_up_ratio_all.shape[0])) if i%100==0 else None
        if np.all(np.isnan(fdn_up_ratio_all[i, :])):
            continue
        alb_corr_mask = gas_abs_masking(alb_wvl, fdn_up_ratio_all[i, :], alt=alt_all[i], interp_nan=False)
        
        # dalb_corr_mask_dwl = np.gradient(alb_corr_mask, alb_wvl)
        # plt.close('all')
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax2 = ax.twinx()
        # l1 = ax.plot(alb_wvl, fdn_up_ratio_all[i, :], '-k', label='Original ratio')
        # l2 = ax2.plot(alb_wvl, dalb_corr_mask_dwl, '-b', label='dAlbedo/dWavelength')
        # ax.plot(alb_wvl[alb_corr_mask >= 1.0], alb_corr_mask[alb_corr_mask >= 1.0], '--r', linewidth=2)
        # l3 = ax.plot(alb_wvl, alb_corr_mask, '--r', label='Gas absorption masked albedo')
        # ax.set_xlabel('Wavelength (nm)', fontsize=14)
        # ax.set_ylabel('Surface Albedo', fontsize=14)
        # ax2.set_ylabel('dAlbedo/dWavelength', fontsize=14)
        # ax.set_title(f'Derivative of Gas Absorption Masked Albedo {date_s} {case_tag}, Z={alt_all[i]:.2f} km', fontsize=13)
        # ll = l1 + l2 + l3
        # labels = [l.get_label() for l in ll]
        # ax.legend(ll, labels, fontsize=10)
        # plt.show()
        
        if np.all(np.isnan(alb_corr_mask)):
            continue
        eff_alb_ = gas_abs_masking(alb_wvl, np.ones_like(fdn_up_ratio_all[i, :]), alt=alt_all[i])
        
        if np.nansum(alb_corr_mask >= 1.0)/np.nansum(eff_alb_) >= 0.1:
            print("Skipping leg %d due to high albedo >= 1.0 fraction" % (i+1))
            continue
        
        if np.nansum(alb_corr_mask[:wvl650_ind] >= 1.0)/np.nansum(eff_alb_[:wvl650_ind]) >= 0.1:
            
            print("Skipping leg %d due to high albedo >= 1.0 fraction in <650nm" % (i+1))
            continue
        
        if np.isnan(alb_corr_mask[eff_alb_==1]).sum()/np.nansum(eff_alb_) >= 0.1:
            print("Skipping leg %d due to high NaN fraction" % (i+1))
            continue
        
        if np.isnan(alb_corr_mask[:wvl650_ind][eff_alb_[:wvl650_ind]==1]).sum()/np.nansum(eff_alb_[:wvl650_ind]) >= 0.1:
            print("Skipping leg %d due to high NaN fraction in <650nm" % (i+1))
            continue
        
        alb_corr = fdn_up_ratio_all[i, :].copy()
        alb_corr = np.clip(alb_corr, 0, 1)
        
        dalb_corr_mask_dwl = np.gradient(alb_corr_mask, alb_wvl)

        
        if np.any(dalb_corr_mask_dwl[wvl450_ind:wvl650_ind] > 0.05):
            continue
        
        if np.any(np.isnan(alb_corr)):
            s_mask = np.isnan(alb_corr)
            alb_corr[s_mask] = fill_nan_ffill_bfill(alb_corr)[s_mask]
        
        alb_corr = alb_corr * correction_factor_all[i, :]
        alb_corr = np.clip(alb_corr, 0, 1)
        
        # plt.close('all')
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax2 = ax.twinx()
        # ax.plot(alb_wvl, fdn_up_ratio_all[i, :], '-', color='r', label='Original albedo')
        # ax.plot(alb_wvl, alb_corr, '-', color='b', label='Corrected albedo before fitting')
        # ax.plot(alb_wvl, alb_corr_mask, '--', color='k', label='Gas absorption masked albedo')
        # ax2.plot(alb_wvl, correction_factor_all[i, :], '-', color='g', label='Correction Factor')
        # ax.set_xlabel('Wavelength (nm)', fontsize=14)
        # ax.set_ylabel('Surface Albedo', fontsize=14)
        # ax2.set_ylabel('Correction Factor', fontsize=14)
        # ax.set_title(f'Surface Albedo before Atmospheric Correction {date_s} {case_tag}, Z={alt_all[i]:.2f} km', fontsize=13)
        # ax.legend(fontsize=10)
        # plt.show()
        
        fdn_up_ratio_all_corr[i, :] = alb_corr.copy()
        
        h2o_override = _DATE_H2O_6_END.get(date_s, {})
        mask_kwargs = {'h2o_6_end': h2o_override['mask']} if 'mask' in h2o_override else {}
        fit_kwargs  = {'h2o_6_end': h2o_override['fit']}  if 'fit'  in h2o_override else {}
        alb_corr_mask = gas_abs_masking(alb_wvl, alb_corr, alt=alt_all[i], **mask_kwargs)
        fdn_up_ratio_all_corr_fit[i, :] = snowice_alb_fitting(alb_wvl, alb_corr, alt=alt_all[i], clear_sky=clear_sky, **fit_kwargs)
        start_wvl = alb_wvl[wvl1450_ind:wvl1520_ind][np.argmin(fdn_up_ratio_all_corr_fit[i, wvl1450_ind:wvl1520_ind])]
        start_ind = np.argmin(np.abs(alb_wvl - start_wvl))+1
        end_wvl = alb_wvl[wvl1800_ind:][np.argmax(fdn_up_ratio_all_corr_fit[i, wvl1800_ind:])]
        end_ind = np.argmin(np.abs(alb_wvl - end_wvl))-1
        alb_corr[np.isnan(alb_corr)] = alb_corr_mask[np.isnan(alb_corr)]
        
        
        fdn_up_ratio_all_corr_fit_before = fdn_up_ratio_all_corr_fit[i, :].copy()
        
        # fit_1D = np.poly1d(np.polyfit(alb_wvl[start_ind:end_ind+1], fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1], 1))
        fit_1D = interp1d([start_wvl, end_wvl], 
                          [fdn_up_ratio_all_corr_fit[i, start_ind], fdn_up_ratio_all_corr_fit[i, end_ind]], 
                          kind='linear', fill_value='extrapolate')
        interp_wvl = alb_wvl[wvl1650_ind:wvl1710_ind+1:]
        interp_alb = fit_1D(interp_wvl)
        interp_alb = np.clip(interp_alb, 0, 1)
        compare_alb = fdn_up_ratio_all_corr_fit[i, wvl1650_ind:wvl1710_ind+1:].copy()
        diff_alb = np.abs((compare_alb - interp_alb)/interp_alb)
        if np.any(diff_alb > 0.05):
            fdn_up_ratio_all_corr_fit[i, wvl1650_ind:wvl1710_ind+1:] = interp_alb.copy()
            # print(f"Applied linear fit between 1650-1710 nm for index {i} due to large deviation.")

            
        min_val =  fdn_up_ratio_all_corr_fit[i, start_ind].copy()
        fdn_up_ratio_all_corr_fit_lt_min = fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1] < min_val
        fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1][fdn_up_ratio_all_corr_fit_lt_min] = min_val
        
        compare_alb_2 = fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1].copy()
        diff_alb_2 = np.abs((compare_alb_2 - fit_1D(alb_wvl[start_ind:end_ind+1]))/fit_1D(alb_wvl[start_ind:end_ind+1]))
        if np.nanstd(diff_alb_2) > 0.1:
            fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1] = (fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1] + fit_1D(alb_wvl[start_ind:end_ind+1])*7 ) / 8.0 
            # print(f"Applied extra 1st order polynomial fit smooth weight between {start_wvl:.1f}-{end_wvl:.1f} nm for index {i} due to high std deviation.")
        elif np.nanstd(diff_alb_2) > 0.05 and np.nanstd(diff_alb_2) <= 0.1:
            fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1] = (fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1] + fit_1D(alb_wvl[start_ind:end_ind+1])*3 ) / 4.0 
            # print(f"Applied extra 1st order polynomial fit smooth weight between {start_wvl:.1f}-{end_wvl:.1f} nm for index {i} due to medium std deviation.")
        elif np.nanstd(diff_alb_2) > 0.02 and np.nanstd(diff_alb_2) <= 0.05:
            fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1] = (fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1] + fit_1D(alb_wvl[start_ind:end_ind+1])*1 ) / 2.0 
            # print(f"Applied extra 1st order polynomial fit smooth weight between {start_wvl:.1f}-{end_wvl:.1f} nm for index {i} due to low std deviation.")
        # else:
        #     fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1] = (fdn_up_ratio_all_corr_fit[i, start_ind:end_ind+1] + fit_1D(alb_wvl[start_ind:end_ind+1]) ) / 2 .0 
            
        # smooth with window size of 11
        window_size = 5
        alb_corr_fit_smooth = fdn_up_ratio_all_corr_fit[i, start_ind:].copy()
        alb_corr_fit_smooth = uniform_filter1d(alb_corr_fit_smooth, size=window_size, mode='reflect')
        alb_corr_fit_smooth = np.clip(alb_corr_fit_smooth, 0, 1)
        
        fdn_up_ratio_all_corr_fit[i, start_ind:] = alb_corr_fit_smooth
            
        if i % 100 == 0:
            fig_dir = f'fig/{date_s}'
            os.makedirs(fig_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(9, 5))
            
            ax.plot(alb_wvl, fdn_up_ratio_all_corr[i, :], '-', color='b', label='Corrected albedo (atm corr)')
            ax.plot(alb_wvl, fdn_up_ratio_all_corr_fit_before, '-', color='g', label='Corrected albedo (atm corr + fit) before')
            # ax.plot(alb_wvl[start_ind:end_ind+1], fit_1D(alb_wvl[start_ind:end_ind+1]), '--', color='orange', label='1st order polynomial fit', linewidth=2.5)
            ax.plot(alb_wvl, fdn_up_ratio_all_corr_fit[i, :], '-', color='r', label='Corrected albedo (atm corr + fit) after')
            ax.plot(alb_wvl, fdn_up_ratio_all[i, :], '-', color='k', label='Original ratio')
            ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
            ax.hlines(min_val, alb_wvl[start_ind], alb_wvl[end_ind], colors='gray', linestyles='dashed', label='Minimum albedo level')
            for band in gas_bands:
                ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
            ax.set_xlabel('Wavelength (nm)', fontsize=14)
            ax.set_ylabel('Surface Albedo', fontsize=14)
            ax.tick_params(labelsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'Surface Albedo (atm corr + fit) for {date_s} {case_tag} time {time_start:.3f}-{time_end:.3f}, alt {alt_avg:.2f} km ({i})', 
                        fontsize=13)
            ax.set_xlim(350, 2000)
            fig.tight_layout()
            fig.savefig(f'{fig_dir}/arcsix_albedo_{date_s}_{case_tag}_{time_start:.3f}-{time_end:.3f}_{alt_avg:.2f}km_{i}.png',
                        bbox_inches='tight', dpi=150)
            plt.close(fig)
        
        del alb_corr_mask, eff_alb_, alb_corr, fit_1D, interp_wvl, interp_alb, compare_alb, diff_alb,
        del min_val, fdn_up_ratio_all_corr_fit_before, alb_corr_fit_smooth
        gc.collect()

    
    select = np.full(len(tmhr_ranges_select), True)
    # remove invalid time
    for i in range(len(tmhr_ranges_select)):
        if np.all(alb_ratio_all[i, :] == 0):
            select[i] = False
            
    alb_ratio_all = alb_ratio_all[select]
    alb1_all = alb1_all[select]
    alb2_all = alb2_all[select]
    p3_ratio1_all = p3_ratio1_all[select]
    p3_ratio2_all = p3_ratio2_all[select]
    lon_avg_all = lon_avg_all[select]
    lat_avg_all = lat_avg_all[select]
    lon_min_all = lon_min_all[select]
    lon_max_all = lon_max_all[select]
    lat_min_all = lat_min_all[select]
    lat_max_all = lat_max_all[select]
    alt_avg_all = alt_avg_all[select]
    ssfr_fup_mean_all = ssfr_fup_mean_all[select]
    ssfr_fdn_mean_all = ssfr_fdn_mean_all[select]
    ssfr_fup_std_all = ssfr_fup_std_all[select]
    ssfr_fdn_std_all = ssfr_fdn_std_all[select]
    simu_fup_mean_all_iter0 = simu_fup_mean_all_iter0[select]
    simu_fdn_mean_all_iter0 = simu_fdn_mean_all_iter0[select]
    simu_fup_mean_all_iter1 = simu_fup_mean_all_iter1[select]
    simu_fdn_mean_all_iter1 = simu_fdn_mean_all_iter1[select]
    simu_fup_mean_all_iter2 = simu_fup_mean_all_iter2[select]
    simu_fdn_mean_all_iter2 = simu_fdn_mean_all_iter2[select]
    simu_fup_toa_mean_all_iter0 = simu_fup_toa_mean_all_iter0[select]
    simu_fup_toa_mean_all_iter1 = simu_fup_toa_mean_all_iter1[select]
    simu_fup_toa_mean_all_iter2 = simu_fup_toa_mean_all_iter2[select]
    toa_mean_all = toa_mean_all[select]

    broadband_alb_iter0     = weighted_broadband_alb(alb_ratio_all,          toa_mean_all,   alb_wvl)
    broadband_alb_iter1     = weighted_broadband_alb(alb1_all,                toa_mean_all,   alb_wvl)
    broadband_alb_iter2     = weighted_broadband_alb(alb2_all,                toa_mean_all,   alb_wvl)
    broadband_alb_iter1_all = weighted_broadband_alb(fdn_up_ratio_all_corr,   toa_expand_all, alb_wvl)
    broadband_alb_iter2_all = weighted_broadband_alb(fdn_up_ratio_all_corr_fit, toa_expand_all, alb_wvl)

    gas_mask = np.isfinite(gas_abs_masking(alb_wvl, np.ones_like(alb_wvl, dtype=float), alt=1))
    wvl_g = alb_wvl[gas_mask]
    broadband_alb_iter0_filter     = weighted_broadband_alb(alb_ratio_all[:, gas_mask],            toa_mean_all[:, gas_mask],   wvl_g)
    broadband_alb_iter1_filter     = weighted_broadband_alb(alb1_all[:, gas_mask],                 toa_mean_all[:, gas_mask],   wvl_g)
    broadband_alb_iter2_filter     = weighted_broadband_alb(alb2_all[:, gas_mask],                 toa_mean_all[:, gas_mask],   wvl_g)
    broadband_alb_iter1_all_filter = weighted_broadband_alb(fdn_up_ratio_all_corr[:, gas_mask],    toa_expand_all[:, gas_mask], wvl_g)
    broadband_alb_iter2_all_filter = weighted_broadband_alb(fdn_up_ratio_all_corr_fit[:, gas_mask], toa_expand_all[:, gas_mask], wvl_g)
    
    
    
    j_ind = 0
    while j_ind < len(fdn_up_ratio_all_corr_fit):
        if not np.all(np.isnan(fdn_up_ratio_all_corr_fit[j_ind, :])):
            ext_wvl, _ = alb_extention(alb_wvl, fdn_up_ratio_all_corr_fit[j_ind, :], clear_sky=clear_sky)
            break
        j_ind += 1
    toa_ext_expand_all = np.zeros((len(fdn_up_ratio_all_corr_fit), len(ext_wvl)))
    # print("ext_wvl shape:", ext_wvl.shape)
    toa_ext_wvl = solar_flux_interp(ext_wvl)
    # print("toa_ext_wvl shape:", toa_ext_wvl.shape)
    alb_iter2_ext_all = np.zeros((len(fdn_up_ratio_all_corr_fit), len(ext_wvl)))
    broadband_alb_iter2_ext_all = np.zeros(len(fdn_up_ratio_all_corr_fit))
    for j in range(len(fdn_up_ratio_all_corr_fit)):
        alb_j = np.clip(fdn_up_ratio_all_corr_fit[j, :], 0.0, 1.0)
        toa_cos_sza_j = toa_ext_wvl * np.cos(np.deg2rad(sza_all[j]))
        toa_ext_expand_all[j, :] = toa_cos_sza_j.copy()
        if np.all(np.isnan(alb_j)):
            broadband_alb_iter2_ext_all[j] = np.nan
        else:    
            _, ext_alb_j = alb_extention(alb_wvl, alb_j, clear_sky=clear_sky)
            alb_iter2_ext_all[j, :] = ext_alb_j.copy()
            
            broadband_alb_iter2_ext_all[j] = np.trapz(ext_alb_j * toa_cos_sza_j, x=ext_wvl) / np.trapz(toa_cos_sza_j, x=ext_wvl)
     
        
    
    
    # plt.close('all')
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(alb_wvl, fdn_up_ratio_all_corr[0, :], '-', color='b', label='Corrected albedo (iter 1)')
    # ax.set_xlabel('Wavelength (nm)', fontsize=14)
    # ax.set_ylabel('Surface Albedo', fontsize=14)
    # ax.set_title(f'Surface Albedo after Atmospheric Correction {date_s} {case_tag}, broadband alb: {broadband_alb_iter1[0]:.3f}', fontsize=13)
    # fig.suptitle(f'Flight leg at Z={alt_avg_all[0]:.2f} km, lon={lon_avg_all[0]:.2f}, lat={lat_avg_all[0]:.2f}', fontsize=12, y=0.98)
    # fig.tight_layout()
    # plt.show()
    # sys.exit()
    
    fig_dir = f'fig/sfc_alb_corr_lonlat'
    os.makedirs(fig_dir, exist_ok=True)
    
    alb_avg = np.nanmean(fdn_up_ratio_all_corr_fit, axis=0)
    alb_std = np.nanstd(fdn_up_ratio_all_corr_fit, axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alb_wvl, alb_avg, '-', color='blue', label='Mean albedo (atm corr + fit)')
    ax.fill_between(alb_wvl, alb_avg-alb_std, alb_avg+alb_std, color='blue', alpha=0.1)
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'Surface Albedo (atm corr + fit) for {date_s} {case_tag}', fontsize=13)
    ax.set_xlim(350, 2000)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/arcsix_albedo_{date_s}_{case_tag}.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    
    # set projection to polar (North Polar Stereographic) and plot lon/lat in that projection
    try:
        plt.close('all')
        central_lon = float(np.nanmean(lon_avg_all)) if len(lon_avg_all) > 0 else 0.0
        proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(1, 1, 1, projection=proj)

        # add coastlines and land features for context
        ax.coastlines(resolution='50m', linewidth=0.8)
        ax.add_feature(cartopy.feature.LAND, facecolor='#f0f0f0', zorder=0)
        ax.add_feature(cartopy.feature.OCEAN, facecolor='#e6f2ff', zorder=0)

        # gridlines with labels (only lon/lat labels make sense in PlateCarree transform)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # scatter the flight-leg centers (and all points) using PlateCarree transform
        # color by broadband surface albedo from iteration 2 if available
        color_vals = np.array(broadband_alb_iter2) if 'broadband_alb_iter2' in locals() else None
        if color_vals is None or np.all(np.isnan(color_vals)):
            sc = ax.scatter(lon_avg_all, lat_avg_all, s=40, c='red', transform=ccrs.PlateCarree(), zorder=3, edgecolor='k', label='Flight legs')
        else:
            sc = ax.scatter(lon_avg_all, lat_avg_all, s=40, c=color_vals, cmap='viridis',
                            transform=ccrs.PlateCarree(), zorder=3, edgecolor='k')
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
            cbar.set_label('Broadband Albedo (atm corr + fit)', fontsize=10)

        # also plot all sampled points along legs (optional, lighter marker)
        if 'lon_all' in locals() and 'lat_all' in locals() and len(lon_all) > 0:
            ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

        ax.set_title(f'Polar projection (North) - Flight legs {date_s} {case_tag}', fontsize=12)
        # set a reasonable display extent around the data if available (lon/lat box)
        if len(lon_avg_all) > 0 and len(lat_avg_all) > 0:
            lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
            lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
            # expand a bit
            pad_lon = max(0.5, (lon_max - lon_min) * 0.2)
            pad_lat = max(0.5, (lat_max - lat_min) * 0.2)
            try:
                ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
            except Exception:
                # fallback: don't set extent if projection complains
                pass
        ax.tick_params('both', labelsize=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{date_s}_{case_tag}_broadband_albedo_vs_longitude_polar_projection.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
    except Exception as e:
        # fallback to simple lon/lat scatter if cartopy fails
        print("Cartopy polar plot failed, falling back to Cartesian scatter. Error:", e)
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(lon_avg_all, lat_avg_all, c=broadband_alb_iter2 if 'broadband_alb_iter2' in locals() else 'r', cmap='viridis', s=40, edgecolor='k')
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Broadband Albedo  (atm corr + fit)', fontsize=12)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Flight legs {date_s} {case_tag}')
        fig.tight_layout()
        fig.savefig(f'{fig_dir}/{date_s}_{case_tag}_broadband_albedo_vs_longitude_cartesian.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
    

    log.info("Saved broadband albedo vs longitude plot to %s/%s_%s_broadband_albedo_vs_longitude.png", fig_dir, date_s, case_tag)
    
    # save alb1 and alb2 to a pkl file
    alb_update_dict = {
        'wvl': alb_wvl,
        'alb_iter0': alb_ratio_all,
        'alb_iter1': alb1_all,
        'alb_iter2': alb2_all,
        'p3_up_dn_ratio_1': p3_ratio1_all,
        'p3_up_dn_ratio_2': p3_ratio2_all,
        'broadband_alb_iter0': broadband_alb_iter0,
        'broadband_alb_iter1': broadband_alb_iter1,
        'broadband_alb_iter2': broadband_alb_iter2,
        'broadband_alb_iter0_filter': broadband_alb_iter0_filter,
        'broadband_alb_iter1_filter': broadband_alb_iter1_filter,
        'broadband_alb_iter2_filter': broadband_alb_iter2_filter,
        'lon_avg': lon_avg_all,
        'lat_avg': lat_avg_all,
        'lon_min': lon_min_all,
        'lon_max': lon_max_all,
        'lat_min': lat_min_all,
        'lat_max': lat_max_all,
        'alt_avg': alt_avg_all,
        'time_all': time_all,
        'lon_all': lon_all,
        'lat_all': lat_all,
        'alt_all': alt_all,
        'sza_all': sza_all,
        'kt19_sfc_T_all': kt19_sfc_T_all,
        'fdn_all': fdn_all,
        'fup_all': fup_all,
        'toa_expand_all': toa_expand_all,
        'fdn_up_ratio_all': fdn_up_ratio_all,
        'correction_factor_all': correction_factor_all,
        'icing_all': icing_all,
        'icing_pre_all': icing_pre_all,
        'alb_iter1_all': fdn_up_ratio_all_corr,
        'alb_iter2_all': fdn_up_ratio_all_corr_fit,
        'broadband_alb_iter1_all': broadband_alb_iter1_all,
        'broadband_alb_iter2_all': broadband_alb_iter2_all,
        'broadband_alb_iter1_all_filter': broadband_alb_iter1_all_filter,
        'broadband_alb_iter2_all_filter': broadband_alb_iter2_all_filter,
        'ext_wvl': ext_wvl,
        'alb_iter2_ext_all': alb_iter2_ext_all,
        'broadband_alb_iter2_ext_all': broadband_alb_iter2_ext_all,
        'modis_alb_legs': np.array(modis_alb_legs) if modis_alb_file is not None else None,
        'modis_bands_nm': modis_bands_nm if modis_alb_file is not None else None,
    }
    if modis_alb_file is None:
        modis_bands_nm = None
        modis_alb_legs = None
        
    output_dir = f'{_fdir_general_}/sfc_alb_combined_smooth_450nm'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{_fdir_general_}/{output_dir}/sfc_alb_update_{date_s}_{case_tag}_time_{tmhr_ranges_select[0][0]:.3f}_{tmhr_ranges_select[-1][-1]:.3f}.pkl', 'wb') as f:
        pickle.dump(alb_update_dict, f)
    log.info(f"Saved surface albedo updates to {_fdir_general_}/sfc_alb/sfc_alb_update_{date_s}_{case_tag}.pkl")

    
    print("Processing completed for date and tag:", date_s, case_tag)

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
    # ------------------------------------------


    atm_corr_processing(date=datetime.datetime(2024, 5, 28),
                    tmhr_ranges_select=[[15.610, 15.822],
                                        [16.905, 17.404] 
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 5, 31),
                    tmhr_ranges_select=[[13.839, 15.180],  # 5.6 km
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )

    

    atm_corr_processing(date=datetime.datetime(2024, 5, 31),
                    tmhr_ranges_select=[
                                        [16.905, 17.404] 
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )

    
    # done
    atm_corr_processing(date=datetime.datetime(2024, 6, 3),
                    tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing
                                        ],
                    case_tag='cloudy_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )
    



    # done
    atm_corr_processing(date=datetime.datetime(2024, 6, 3),
                    tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )
    

    atm_corr_processing(date=datetime.datetime(2024, 6, 5),
                    tmhr_ranges_select=[[12.405, 13.812], # 5.7m,
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )
    


    atm_corr_processing(date=datetime.datetime(2024, 6, 5),
                    tmhr_ranges_select=[
                                        [14.258, 15.036], # 100m
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 6, 5),
                    tmhr_ranges_select=[
                                        [15.535, 15.931], # 450m
                                        ],
                    case_tag='clear_atm_corr_3',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )
    


    atm_corr_processing(date=datetime.datetime(2024, 6, 5),
                    tmhr_ranges_select=[
                                        [13.7889, 13.8010],
                                        [13.8350, 13.8395],
                                        [13.8780, 13.8885],
                                        [13.9240, 13.9255],
                                        [13.9389, 13.9403],
                                        [13.9540, 13.9715],
                                        [13.9980, 14.0153],
                                        [14.0417, 14.0575],
                                        [14.0417, 14.0475],
                                        [14.0560, 14.0590],
                                        [14.0825, 14.0975],
                                        [14.1264, 14.1525],
                                        [14.1762, 14.1975],
                                        [14.2194, 14.2420],
                                        [14.2605, 14.2810]
                                        ],
                    case_tag='clear_sky_spiral_atm_corr',
                    config=config,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 6, 6),
                    tmhr_ranges_select=[[16.250, 16.325], # 100m, 
                                        [16.375, 16.632], # 450m
                                        [16.700, 16.794], # 100m
                                        [16.850, 16.952], # 1.2km
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 6, 7),
                    tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 6, 11),
                    tmhr_ranges_select=[[14.5667, 14.5694],
                                        [14.5986, 14.6097],
                                        [14.6375, 14.6486], # cloud shadow
                                        [14.6778, 14.6903],
                                        [14.7208, 14.7403],
                                        [14.7653, 14.7875],
                                        [14.8125, 14.8278],
                                        [14.8542, 14.8736],
                                        [14.8986, 14.9389], # more cracks
                                        ],
                    case_tag='clear_sky_spiral_atm_corr',
                    config=config,
                    simulation_interval=None,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 6, 11),
                    tmhr_ranges_select=[
                                        [14.968, 15.229], # 100, clear, some cloud
                                        [14.968, 15.347],
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 6, 11),
                    tmhr_ranges_select=[
                                        [15.347, 15.813], # 100m
                                        [15.813, 16.115], # 100-450m, clear, some cloud
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 6, 13),
                    tmhr_ranges_select=[[13.704, 13.817], # 100-450m, clear, some cloud
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 6, 13),
                    tmhr_ranges_select=[[14.109, 14.140], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 6, 13),
                    tmhr_ranges_select=[[15.834, 15.883], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )
    


    atm_corr_processing(date=datetime.datetime(2024, 6, 13),
                    tmhr_ranges_select=[[16.043, 16.067], # 100-200m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_3',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 6, 13),
                    tmhr_ranges_select=[[16.550, 17.581], # 100-500m, clear
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 7, 25),
                    tmhr_ranges_select=[[15.094, 15.300], # 100m, some low clouds or fog below
                                        ],
                    case_tag='cloudy_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 7, 25),
                    tmhr_ranges_select=[[15.881, 15.903], # 200-500m
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )
    


    atm_corr_processing(date=datetime.datetime(2024, 7, 29),
                    tmhr_ranges_select=[[13.442, 13.465],
                                        [13.490, 13.514],
                                        [13.536, 13.554],
                                        [13.580, 13.611],
                                        [13.639, 13.654],
                                        [13.676, 13.707],
                                        [13.733, 13.775],
                                        [13.793, 13.836],
                                        ],
                    case_tag='clear_sky_spiral_atm_corr',
                    config=config,
                    simulation_interval=None,
                    clear_sky=True,
                    )
    


    atm_corr_processing(date=datetime.datetime(2024, 7, 29),
                    tmhr_ranges_select=[[13.939, 14.200], # 100m, clear
                                        [14.438, 14.714], # 3.7km
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 7, 29),
                    tmhr_ranges_select=[
                                        [15.214, 15.804], # 1.3km
                                        [16.176, 16.304], # 1.3km
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )
    


    atm_corr_processing(date=datetime.datetime(2024, 7, 30),
                    tmhr_ranges_select=[[13.886, 13.908],
                                        [13.934, 13.950],
                                        [13.976, 14.000],
                                        [14.031, 14.051],
                                        [14.073, 14.096],
                                        [14.115, 14.134],
                                        [14.157, 14.179],
                                        [14.202, 14.219],
                                        [14.239, 14.254],
                                        [14.275, 14.294],
                                        ],
                    case_tag='clear_sky_spiral_atm_corr',
                    config=config,
                    simulation_interval=None,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 7, 30),
                    tmhr_ranges_select=[[14.318, 14.936], # 100-450m, clear
                                        [15.043, 15.140], # 1.5km
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 8, 1),
                    tmhr_ranges_select=[[13.843, 14.361], # 100-450m, clear, some open ocean
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 8, 1),
                    tmhr_ranges_select=[
                                        [14.739, 15.053], # 550m
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )
    



    atm_corr_processing(date=datetime.datetime(2024, 8, 2),
                    tmhr_ranges_select=[
                                        [14.557, 15.100], # 100m
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 8, 2),
                    tmhr_ranges_select=[
                                        [15.244, 16.635], # 1km
                                        ],
                    case_tag='clear_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 8, 7),
                    tmhr_ranges_select=[[13.344, 13.763], # 100m, cloudy
                                        ],
                    case_tag='clear_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 8, 7),
                    tmhr_ranges_select=[
                                        [15.472, 15.567], # 180m, cloudy
                                        [15.580, 15.921], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 8),
    #                 tmhr_ranges_select=[
    #                                     [12.990, 13.180], # 180m, clear
    #                                     ],
    #                 case_tag='clear_atm_corr_1',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 8),
    #                 tmhr_ranges_select=[
    #                                     [14.250, 14.373], # 180m, clear
    #                                     ],
    #                 case_tag='clear_atm_corr_2',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 8),
    #                 tmhr_ranges_select=[
    #                                     [16.471, 16.601], # 180m, clear
    #                                     ],
    #                 case_tag='clear_atm_corr_3',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    atm_corr_processing(date=datetime.datetime(2024, 8, 8),
                    tmhr_ranges_select=[
                                        [13.212, 13.347], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 8, 8),
                    tmhr_ranges_select=[
                                        [15.314, 15.504], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )



    atm_corr_processing(date=datetime.datetime(2024, 8, 9),
                    tmhr_ranges_select=[
                                        [13.376, 13.600], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_1',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )



    # atm_corr_processing(date=datetime.datetime(2024, 8, 9),
    #                 tmhr_ranges_select=[
    #                                     [14.750, 15.060], # 100m, clear
    #                                     [15.622, 15.887], # 100m, clear
    #                                     ],
    #                 case_tag='clear_atm_corr',
    #                 config=config,
    #                 simulation_interval=0.5,
    #                 clear_sky=True,
    #                 )



    atm_corr_processing(date=datetime.datetime(2024, 8, 9),
                    tmhr_ranges_select=[
                                        [16.029, 16.224], # 100m, cloudy
                                        ],
                    case_tag='cloudy_atm_corr_2',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=False,
                    )
    


    atm_corr_processing(date=datetime.datetime(2024, 8, 15),
                    tmhr_ranges_select=[
                                        [14.085, 14.396], # 100m, clear
                                        [14.550, 14.968], # 3.5km, clear
                                        [15.078, 15.163], # 1.7km, clear
                                        ],
                    case_tag='clear_atm_corr',
                    config=config,
                    simulation_interval=0.5,
                    clear_sky=True,
                    )
