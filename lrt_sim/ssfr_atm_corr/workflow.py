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

_LRT_SIM_ROOT = str(Path(__file__).resolve().parents[1])
if _LRT_SIM_ROOT not in sys.path:
    sys.path.insert(0, _LRT_SIM_ROOT)

if platform.system() == 'Linux':
    # Define the path to your module directory
    # Use os.path.abspath and os.path.join for platform independence
    # module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'util'))

    # Add the directory to the Python search path
    # sys.path.insert(0, module_path)
    sys.path.append("/projects/yuch8913/arcsix_sfc/lrt_sim/")  

import glob
import copy
import shutil
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

try:
    from .settings import *
    from .helpers import fit_1d_poly, gas_abs_masking, ssfr_flags, write_2col_file
    from .qc_plotting import ssfr_time_series_plot
    from .setup import (
        default_atm_levels,
        load_cloud_observation_legs,
        load_nearest_dropsonde,
        run_uvspec_inits,
        split_tmhr_ranges,
        write_final_sw_support_files,
        write_ssfr_support_files,
    )
except ImportError:
    from settings import *
    from helpers import fit_1d_poly, gas_abs_masking, ssfr_flags, write_2col_file
    from qc_plotting import ssfr_time_series_plot
    from setup import (
        default_atm_levels,
        load_cloud_observation_legs,
        load_nearest_dropsonde,
        run_uvspec_inits,
        split_tmhr_ranges,
        write_final_sw_support_files,
        write_ssfr_support_files,
    )


def read_uvspec_flux_output(init):
    """Read a uvspec flux output file, inferring the actual wavelength grid."""
    data = np.loadtxt(init.output_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f'{init.output_file} has {data.shape[1]} columns; expected at least 4 flux columns.')

    n_altitude = init.Ny
    if data.shape[0] % n_altitude != 0:
        raise ValueError(
            f'{init.output_file} has {data.shape[0]} rows, which is not divisible by '
            f'{n_altitude} output altitude(s).'
        )

    n_wavelength = data.shape[0] // n_altitude
    data = data.reshape((n_wavelength, n_altitude, data.shape[1]))
    wavelength = data[:, 0, 0]
    f_down_direct = data[:, :, 1] / 1000.0
    f_down_diffuse = data[:, :, 2] / 1000.0
    f_up = data[:, :, 3] / 1000.0
    return wavelength, f_down_direct + f_down_diffuse, f_down_direct, f_down_diffuse, f_up


def flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),
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
                     iter=0,
                     final_sim=False,
                     final_status='closure_passed',
                    ):
    
    log = logging.getLogger("lrt")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    
    os.makedirs(f'fig/{date_s}', exist_ok=True)
    
    tmhr_ranges_select = split_tmhr_ranges(tmhr_ranges_select, simulation_interval)

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    
 
    log.info("ssfr filename:", config.ssfr(date_s))
    
    # plot ssfr time series for checking sable legs selection
    ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_ranges_select, date_s, case_tag, pitch_roll_thres=3.0)

    # Build leg masks
    t_hsk = np.array(data_hsk["tmhr"])
    leg_masks = [(t_hsk>=lo)&(t_hsk<=hi) for lo,hi in tmhr_ranges_select]
    del data_hsk, data_ssfr, data_hsr1, t_hsk
    gc.collect()

    
    data_dropsonde_legs = load_nearest_dropsonde(_fdir_general_, date, tmhr_ranges_select, log)

    zpt_filedir = f'{_fdir_general_}/zpt/{date_s}'
    os.makedirs(zpt_filedir, exist_ok=True)
    using_custom_levels = levels is not None
    if levels is None:
        levels = default_atm_levels()
    levels = np.asarray(levels, dtype=float)
    levels = np.unique(levels[np.isfinite(levels)])
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError('Atmospheric levels must be a non-empty 1-D array.')
    level_source = 'custom' if using_custom_levels else 'default'
    print(
        f'Using {level_source} atmospheric levels for {date_s}_{case_tag}: '
        f'{levels.size} levels from {levels[0]:.3f} to {levels[-1]:.3f} km'
    )

    if final_sim:
        effective_wvl = write_final_sw_support_files()
        wavelength_grid_file = os.path.abspath('wvl_grid_final_sw.dat')
        solar_file = os.path.abspath('arcsix_ssfr_solar_flux_raw_final.dat')
    else:
        effective_wvl = write_ssfr_support_files(iter=iter, clear_sky=clear_sky)
        wavelength_grid_file = os.path.abspath('wvl_grid_test.dat')
        solar_file = os.path.abspath('arcsix_ssfr_solar_flux_raw.dat')
            

    # read satellite granule
    #/----------------------------------------------------------------------------\#
    cld_legs = load_cloud_observation_legs(
        _fdir_general_,
        _mission_,
        _platform_,
        date_s,
        case_tag,
        tmhr_ranges_select,
    )
     
    # return None 
    
    
    solver = 'lrt'
    for ileg, _ in enumerate(leg_masks):
        
        cld_leg = cld_legs[ileg]
        time_start, time_end = cld_leg['time'][0], cld_leg['time'][-1]
        lon_avg = np.round(np.mean(cld_leg['lon']), 2)
        lat_avg = np.round(np.mean(cld_leg['lat']), 2)
        alt_avg = np.round(np.nanmean(cld_leg['alt']), 2)  # in km
        heading_avg = np.round(np.nanmean(cld_leg['heading']), 2)
        sza_avg = np.round(np.nanmean(cld_leg['sza']), 2)
        saa_avg = np.round(np.nanmean(cld_leg['saa']), 2)
        ssfr_zen_flux = cld_leg['ssfr_zen']
        ssfr_nad_flux = cld_leg['ssfr_nad']
        if np.all(np.isnan(ssfr_zen_flux)) or np.all(np.isnan(ssfr_nad_flux)):
            print(f"All SSFR zenith or nadir fluxes are NaN for leg {ileg+1}, skipping atmospheric correction")
            cld_legs[ileg] = None
            del cld_leg, ssfr_zen_flux, ssfr_nad_flux
            gc.collect()
            continue
        
        # atm profile searching setting
        boundary_from_center = 0.25 # degree
        mod_lon = np.array([lon_avg-boundary_from_center, lon_avg+boundary_from_center])
        mod_lat = np.array([lat_avg-boundary_from_center, lat_avg+boundary_from_center])
        mod_extent = [mod_lon[0], mod_lon[1], mod_lat[0], mod_lat[1]]
        
        
        if clear_sky:
            fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_clear'
            fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
        else:
            fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_sat_cloud'
            fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'
            
        if final_sim:
            output_csv_name = f'{fdir}/ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km_final_extension.csv'
            native_iteration_csv_name = f'{fdir}/ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km_iteration_{iter}.csv'
            native_final_csv_name = f'{fdir}/ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km_final.csv'
            native_iteration_albedo_name = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_{iter}.dat'
            native_final_albedo_name = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_final.dat'
            final_extension_albedo_name = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_final_extension.dat'
        else:
            output_csv_name = f'{fdir}/ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km_iteration_{iter}.csv'

        os.makedirs(fdir_tmp, exist_ok=True)
        os.makedirs(fdir, exist_ok=True)

        if final_sim:
            if os.path.exists(native_iteration_csv_name):
                if not os.path.exists(native_final_csv_name) or overwrite_lrt:
                    shutil.copy2(native_iteration_csv_name, native_final_csv_name)
                    print(f"Saving SSFR-grid final simulation to {native_final_csv_name}")
            else:
                print(
                    f"Warning: cannot create SSFR-grid final CSV because "
                    f"{native_iteration_csv_name} does not exist."
                )
            if os.path.exists(native_iteration_albedo_name):
                if not os.path.exists(native_final_albedo_name) or overwrite_lrt:
                    shutil.copy2(native_iteration_albedo_name, native_final_albedo_name)
                    print(f"Saving SSFR-grid final albedo to {native_final_albedo_name}")
            else:
                print(
                    f"Warning: cannot create SSFR-grid final albedo because "
                    f"{native_iteration_albedo_name} does not exist."
                )
            if (
                os.path.exists(output_csv_name)
                and not overwrite_lrt
                and os.path.exists(native_iteration_albedo_name)
                and not os.path.exists(final_extension_albedo_name)
            ):
                alb_data = np.loadtxt(native_iteration_albedo_name, comments='#')
                final_alb_wvl, final_alb = alb_extention(alb_data[:, 0], alb_data[:, 1], clear_sky=clear_sky)
                final_alb = np.clip(final_alb, 0.0, 1.0)
                write_2col_file(
                    final_extension_albedo_name,
                    final_alb_wvl,
                    final_alb,
                    header=(f'# SSFR final extended sfc albedo {date_s} iteration {iter}\n'
                            '# wavelength (nm)      albedo (unitless)\n'),
                )
                print(f"Saving final extended albedo to {final_extension_albedo_name}")

        
        if not os.path.exists(output_csv_name) or overwrite_lrt:
            print("Start leg %d atmospheric correction ..." % (ileg+1))
            print(f"Date: {date_s}, Time: {time_start:.3f}-{time_end:.3f}h, Alt: {alt_avg:.2f}km")
            if iter==0:
                data_dropsonde = data_dropsonde_legs[ileg] if data_dropsonde_legs else None
                prepare_atmospheric_profile(_fdir_general_, date_s, case_tag, ileg, date, time_start, time_end,
                                            alt_avg, data_dropsonde,
                                            cld_leg, levels=levels,
                                            mod_extent=[np.round(np.nanmin(cld_leg['lon']), 2), 
                                                        np.round(np.nanmax(cld_leg['lon']), 2),
                                                        np.round(np.nanmin(cld_leg['lat']), 2),
                                                        np.round(np.nanmax(cld_leg['lat']), 2)],
                                            zpt_filedir=f'{_fdir_general_}/zpt/{date_s}'
                                            )
            # =================================================================================
            
            
            # write out the surface albedo
            #/----------------------------------------------------------------------------\#
            os.makedirs(f'{_fdir_general_}/sfc_alb', exist_ok=True)
            iter_0_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_0.dat'
            if iter == 0:
                
                alb_wvl = np.concatenate(([348.0], cld_leg['ssfr_zen_wvl'], [2050.]))
                alb_avg = np.nanmean(cld_leg['ssfr_nad']/cld_leg['ssfr_zen'], axis=0)
                # print("cld_leg['ssfr_nad']:", np.nanmean(cld_leg['ssfr_nad'], axis=0))
                # print("cld_leg['ssfr_zen']:", np.nanmean(cld_leg['ssfr_zen'], axis=0))
                
                if np.all(np.isnan(alb_avg)):
                    raise ValueError(f"All nadir/zenith ratios are NaN for leg {ileg+1}, cannot compute average albedo")
                alb_avg[alb_avg<0.0] = 0.0
                alb_avg[alb_avg>1.0] = 1.0
                # alb_avg[np.isnan(alb_avg)] = 0.0
                
                if np.any(np.isnan(alb_avg)):
                    s = pd.Series(alb_avg)
                    s_mask = np.isnan(alb_avg)
                    # Fills NaN with the value immediately preceding it
                    s_ffill = s.fillna(method='ffill', limit=2)
                    s_ffill = s_ffill.fillna(method='bfill', limit=2)
                    while np.any(np.isnan(s_ffill)):
                        s_ffill = s_ffill.fillna(method='ffill', limit=2)
                        s_ffill = s_ffill.fillna(method='bfill', limit=2)
                        
                    alb_avg[s_mask] = s_ffill[s_mask]
                        
                alb_avg_extend = np.concatenate(([alb_avg[0]], alb_avg, [alb_avg[-1]]))

                write_2col_file(iter_0_fname, alb_wvl, alb_avg_extend,
                                header=('# SSFR derived sfc albedo\n'
                                        '# wavelength (nm)      albedo (unitless)\n'))
                
                # plt.close('all')
                # plt.figure(figsize=(8, 5))
                # plt.plot(alb_wvl, np.nanmean(cld_leg['ssfr_nad'], axis=0), label='Nadir Radiance')
                # plt.plot(alb_wvl, np.nanmean(cld_leg['ssfr_zen'], axis=0), label='Zenith Radiance')
                # plt.plot(alb_wvl, np.nanmean(cld_leg['ssfr_toa'], axis=0), label='Top-of-Atmosphere')
                # plt.xlabel('Wavelength (nm)')
                # plt.ylabel('Radiance (W/m$^2$/sr/nm)')
                # plt.show()
                
                # plt.close('all')
                # plt.figure(figsize=(8, 5))
                # plt.plot(alb_wvl, alb_avg, label='Derived Surface Albedo')
                # plt.xlabel('Wavelength (nm)')
                # plt.ylabel('Albedo (unitless)')
                # sys.exit()
            else:
                alb_avg = pd.read_csv(iter_0_fname, delim_whitespace=True, comment='#', header=None).iloc[:, 1].values
                alb_avg = alb_avg[1:-1]  # remove the extended edges
            #\----------------------------------------------------------------------------/#
            
            atm_z_grid = levels
            z_list = atm_z_grid
            atm_z_grid_str = ' '.join(['%.3f' % z for z in atm_z_grid])

          
            flux_output = np.zeros(np.count_nonzero(leg_masks[ileg]))
            
            for ix in range(1):
                flux_key_all = []
                
                flux_down_results = []
                flux_down_dir_results = []
                flux_down_diff_results = []
                flux_up_results = []
                output_wvl_results = []
                
                flux_key = np.zeros_like(flux_output, dtype=object)
                cloudy = 0
                clear = 0
                
                # rt initialization
                #/----------------------------------------------------------------------------\#
                lrt_cfg = copy.deepcopy(er3t.rtm.lrt.get_lrt_cfg())
                
                lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.dat')
                # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')
                lrt_cfg['solar_file'] = solar_file
                # lrt_cfg['solar_file'] = lrt_cfg['solar_file'].replace('kurudz_0.1nm.dat', 'kurudz_1.0nm.dat')
                import platform
                # run less streams on Mac for testing, higher resolution on Linux cluster
                if platform.system() == 'Darwin':
                    lrt_cfg['number_of_streams'] = 4
                elif platform.system() == 'Linux':
                    lrt_cfg['number_of_streams'] = 4 if (final_sim and not clear_sky) else 8
                lrt_cfg['mol_abs_param'] = 'reptran coarse'
                # lrt_cfg['mol_abs_param'] = f'reptran medium'
                albedo_file = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_{iter}.dat'
                final_alb_wvl = None
                final_alb = None
                if final_sim:
                    alb_data = np.loadtxt(albedo_file, comments='#')
                    alb_wvl_for_ext = alb_data[:, 0]
                    alb_val_for_ext = alb_data[:, 1]
                    final_alb_wvl, final_alb = alb_extention(alb_wvl_for_ext, alb_val_for_ext, clear_sky=clear_sky)
                    final_alb = np.clip(final_alb, 0.0, 1.0)
                    albedo_file = (
                        final_extension_albedo_name
                    )
                    write_2col_file(
                        albedo_file,
                        final_alb_wvl,
                        final_alb,
                        header=(f'# SSFR final extended sfc albedo {date_s} iteration {iter}\n'
                                '# wavelength (nm)      albedo (unitless)\n'),
                    )
                input_dict_extra_general = {
                                    'crs_model': 'rayleigh Bodhaine29',
                                    'albedo_file': albedo_file,
                                    'mol_file': 'CH4 %s' % os.path.join(zpt_filedir, f'ch4_profiles_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.dat'),
                                    'wavelength_grid_file': wavelength_grid_file,
                                    'atm_z_grid': atm_z_grid_str,
                                    # 'no_scattering':'mol',
                                    # 'no_absorption':'mol',
                                    }
            
                
                Nx_effective = len(effective_wvl)
                mute_list = ['albedo', 'wavelength', 'spline', 'slit_function_file']
                #/----------------------------------------------------------------------------/#

                
                inits_rad = []
                flux_key_ix = []
                output_list = []
                
                if not clear_sky:
                    input_dict_extra = copy.deepcopy(input_dict_extra_general)
                    if manual_cloud:
                        cer_x = manual_cloud_cer
                        cwp_x = manual_cloud_cwp
                        cth_x = manual_cloud_cth
                        cbh_x = manual_cloud_cbh
                        cot_x = manual_cloud_cot
                        cgt_x = cth_x - cbh_x
                        cloud_source = 'manual'
                    else:
                        cot_x = np.nanmean(cld_leg['cot'])
                        cwp_x = np.nanmean(cld_leg['cwp'])
                        cer_x = np.nanmean(cld_leg['cer'])
                        cth_x = np.nanmean(cld_leg['cth'])
                        cbh_x = np.nanmean(cld_leg['cbh'])
                        cgt_x = np.nanmean(cld_leg['cgt'])
                        cloud_source = 'satellite'

                    has_cloud = (
                        cot_x >= 0.1
                        and np.isfinite(cwp_x)
                        and np.isfinite(cth_x)
                        and np.isfinite(cbh_x)
                        and np.isfinite(cgt_x)
                        and cgt_x > 0.0
                    )
                    if has_cloud:
                        cth_ind_cld = bisect.bisect_left(z_list, cth_x)
                        cbh_ind_cld = bisect.bisect_left(z_list, cbh_x)
                        cloud_altitude = z_list[cbh_ind_cld:cth_ind_cld+1]
                        if len(cloud_altitude) == 0:
                            message = (
                                f'No atmospheric levels selected for {cloud_source} cloud layer '
                                f'{cbh_x:.3f}-{cth_x:.3f} km in leg {ileg+1}. '
                                f'Check the custom levels for {date_s}_{case_tag}.'
                            )
                            if manual_cloud:
                                raise ValueError(message)
                            print(f'Warning: {message} Treating leg as clear.')
                            cld_cfg = None
                            dict_key = f'clear {alt_avg:.2f}'
                            clear += 1
                        else:
                            fname_cld = f'{fdir_tmp}/cld_{ix:04d}_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.txt'
                            if os.path.exists(fname_cld):
                                os.remove(fname_cld)
                            cld_cfg = er3t.rtm.lrt.get_cld_cfg()
                            cloudy += 1
                            cld_cfg['cloud_file'] = fname_cld
                            cld_cfg['cloud_altitude'] = cloud_altitude
                            cld_cfg['cloud_effective_radius']  = cer_x
                            cld_cfg['liquid_water_content'] = cwp_x*1000/(cgt_x*1000) # convert kg/m^2 to g/m^3
                            cld_cfg['cloud_optical_thickness'] = cot_x

                            if manual_cloud:
                                print(
                                    f'Using manual cloud for leg {ileg+1}: '
                                    f'CBH={cbh_x:.3f} km, CTH={cth_x:.3f} km, '
                                    f'COT={cot_x:.3f}, CER={cer_x:.3f} um, '
                                    f'CWP={cwp_x:.5f} kg/m^2, '
                                    f'levels={cloud_altitude[0]:.3f}-{cloud_altitude[-1]:.3f} km '
                                    f'({len(cloud_altitude)} points)'
                                )

                            dict_key_arr = np.concatenate(([cld_cfg['cloud_optical_thickness']], [cld_cfg['cloud_effective_radius']], cld_cfg['cloud_altitude'], [alt_avg]))
                            dict_key = '_'.join([f'{i:.3f}' for i in dict_key_arr])
                    else:
                        if manual_cloud:
                            raise ValueError(
                                f'Invalid manual cloud for leg {ileg+1}: '
                                f'CBH={cbh_x}, CTH={cth_x}, COT={cot_x}, CWP={cwp_x}.'
                            )
                        cld_cfg = None
                        dict_key = f'clear {alt_avg:.2f}'
                        clear += 1
                else:
                    cld_cfg = None
                    dict_key = f'clear {alt_avg:.2f}'
                    cot_x = 0.0
                    cwp_x = 0.0
                    cer_x = 0.0
                    cth_x = 0.0
                    cbh_x = 0.0
                    cgt_x = 0.0
                    input_dict_extra = copy.deepcopy(input_dict_extra_general)
                    clear += 1
                flux_key[ix] = dict_key
                
                if (cld_cfg is None) and (dict_key in flux_key_all):
                    flux_key_ix.append(dict_key)
                elif (cld_cfg is not None) and (dict_key in flux_key_all):
                    flux_key_ix.append(dict_key)
                else:
                    input_dict_extra_alb = copy.deepcopy(input_dict_extra)
                    init = er3t.rtm.lrt.lrt_init_mono_flx(
                            input_file  = f'{fdir_tmp}/input_{ix:04d}_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.txt',
                            output_file = f'{fdir_tmp}/output_{ix:04d}_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.txt',
                            date        = date,
                            # surface_albedo=0.08,
                            solar_zenith_angle = sza_avg,
                            Nx = Nx_effective,
                            output_altitude    = [0, alt_avg, 'toa'],
                            input_dict_extra   = input_dict_extra_alb.copy(),
                            mute_list          = mute_list,
                            lrt_cfg            = lrt_cfg,
                            cld_cfg            = cld_cfg,
                            aer_cfg            = None,
                            )
                    #\----------------------------------------------------------------------------/#

                    inits_rad.append(copy.deepcopy(init))
                    output_list.append(f'{fdir_tmp}/output_{ix:04d}_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.txt')
                    flux_key_all.append(dict_key)
                    flux_key_ix.append(dict_key)
                    
            # # Run RT
            print(f"Start running libratran calculations for {output_csv_name.replace('.csv', '')} ")
            # #/----------------------------------------------------------------------------\#
            import platform
            if platform.system() == 'Darwin':
                ##### run several libratran calculations in parallel
                if len(inits_rad) > 0:
                    print('Running libratran calculations ...')
                    run_uvspec_inits(inits_rad)
                    for i in range(len(inits_rad)):
                        output_wvl, flux_down, flux_down_dir, flux_down_diff, flux_up = read_uvspec_flux_output(inits_rad[i])
                        output_wvl_results.append(output_wvl)
                        flux_down_results.append(flux_down)
                        flux_down_dir_results.append(flux_down_dir)
                        flux_down_diff_results.append(flux_down_diff)
                        flux_up_results.append(flux_up)
            ##### run several libratran calculations one by one
            
            elif platform.system() == 'Linux':
                if len(inits_rad) > 0:
                    print('Running libratran calculations ...')
                    run_uvspec_inits(inits_rad)
                    for i in range(len(inits_rad)):
                        output_wvl, flux_down, flux_down_dir, flux_down_diff, flux_up = read_uvspec_flux_output(inits_rad[i])
                        output_wvl_results.append(output_wvl)
                        flux_down_results.append(flux_down)
                        flux_down_dir_results.append(flux_down_dir)
                        flux_down_diff_results.append(flux_down_diff)
                        flux_up_results.append(flux_up)
            # #\----------------------------------------------------------------------------/#
            ###### delete input, output, cld txt files
            # for prefix in ['input', 'output', 'cld']:
            #     for filename in glob.glob(os.path.join(fdir_tmp, f'{prefix}_*.txt')):
            #         os.remove(filename)
            ###### delete atmospheric profile files for lw


            if output_wvl_results:
                output_wvl = output_wvl_results[0]
                if output_wvl.size != effective_wvl.size or not np.allclose(output_wvl, effective_wvl):
                    print(
                        f'Using uvspec output wavelength grid ({output_wvl.size} points) '
                        f'instead of requested grid ({effective_wvl.size} points).'
                    )
                    effective_wvl = output_wvl

            flux_down_results = np.array(flux_down_results)
            flux_down_dir_results = np.array(flux_down_dir_results)
            flux_down_diff_results = np.array(flux_down_diff_results)
            flux_up_results = np.array(flux_up_results)

            if final_sim:
                fup_mean = np.nanmean(cld_leg['ssfr_nad'], axis=0)
                fdn_mean = np.nanmean(cld_leg['ssfr_zen'], axis=0)
                fup_std = np.nanstd(cld_leg['ssfr_nad'], axis=0)
                fdn_std = np.nanstd(cld_leg['ssfr_zen'], axis=0)

                f_ssfr_fup_mean = interp1d(cld_leg['ssfr_zen_wvl'], fup_mean, bounds_error=False, fill_value=np.nan)
                f_ssfr_fdn_mean = interp1d(cld_leg['ssfr_zen_wvl'], fdn_mean, bounds_error=False, fill_value=np.nan)
                f_ssfr_fup_std = interp1d(cld_leg['ssfr_zen_wvl'], fup_std, bounds_error=False, fill_value=np.nan)
                f_ssfr_fdn_std = interp1d(cld_leg['ssfr_zen_wvl'], fdn_std, bounds_error=False, fill_value=np.nan)

                output_dict = {
                    'wvl': effective_wvl,
                    'final_iter': np.full(effective_wvl.shape, iter, dtype=float),
                    'final_status': np.full(effective_wvl.shape, final_status, dtype=object),
                    'final_closure_met': np.full(effective_wvl.shape, final_status == 'closure_passed', dtype=bool),
                    'time_start': np.full(effective_wvl.shape, time_start, dtype=float),
                    'time_end': np.full(effective_wvl.shape, time_end, dtype=float),
                    'alt_km': np.full(effective_wvl.shape, alt_avg, dtype=float),
                    'sza': np.full(effective_wvl.shape, sza_avg, dtype=float),
                    'ssfr_fup_mean': f_ssfr_fup_mean(effective_wvl),
                    'ssfr_fdn_mean': f_ssfr_fdn_mean(effective_wvl),
                    'ssfr_fup_std': f_ssfr_fup_std(effective_wvl),
                    'ssfr_fdn_std': f_ssfr_fdn_std(effective_wvl),
                    'sfc_alb_final': np.interp(effective_wvl, final_alb_wvl, final_alb, left=np.nan, right=np.nan),
                    'simu_fup_sfc_final': np.nanmean(flux_up_results[:, :, 0], axis=0),
                    'simu_fdn_sfc_final': np.nanmean(flux_down_results[:, :, 0], axis=0),
                    'simu_fdn_sfc_direct_final': np.nanmean(flux_down_dir_results[:, :, 0], axis=0),
                    'simu_fdn_sfc_diff_final': np.nanmean(flux_down_diff_results[:, :, 0], axis=0),
                    'simu_fup_p3_final': np.nanmean(flux_up_results[:, :, 1], axis=0),
                    'simu_fdn_p3_final': np.nanmean(flux_down_results[:, :, 1], axis=0),
                    'simu_fdn_p3_direct_final': np.nanmean(flux_down_dir_results[:, :, 1], axis=0),
                    'simu_fdn_p3_diff_final': np.nanmean(flux_down_diff_results[:, :, 1], axis=0),
                    'simu_fup_toa_final': np.nanmean(flux_up_results[:, :, -1], axis=0),
                    'simu_fdn_toa_final': np.nanmean(flux_down_results[:, :, -1], axis=0),
                }
                output_df = pd.DataFrame(output_dict)
                output_df.to_csv(output_csv_name, index=False)
                print(f"Saving final extended spectral simulation to {output_csv_name}")

                del output_dict, output_df
                del flux_down_results, flux_down_dir_results, flux_down_diff_results, flux_up_results
                cld_legs[ileg] = None
                del fup_mean, fdn_mean, fup_std, fdn_std, cld_leg
                gc.collect()
                continue
            
            for flux_dn in [flux_down_results, flux_down_dir_results, flux_down_diff_results, flux_up_results]:
                for iz in range(3):
                    for iset in range(flux_down_results.shape[0]):
                        flux_dn[iset, :, iz] = ssfr_slit_convolve(effective_wvl, flux_dn[iset, :, iz], wvl_joint=950)
            
            
            # simulated fluxes at surface
            Fup_sfc = flux_up_results[:, :, 0]
            Fdn_sfc = flux_down_results[:, :, 0]
            Fdn_sfc_direct = flux_down_dir_results[:, :, 0]
            Fdn_sfc_diff = flux_down_diff_results[:, :, 0]

            # simulated fluxes at p3 altitude
            Fup_p3 = flux_up_results[:, :, 1]
            Fdn_p3 = flux_down_results[:, :, 1]
            Fdn_p3_diff_ratio = flux_down_diff_results[:, :, 1] / flux_down_results[:, :, 1]
            
            # simulated fluxes at toa
            Fup_toa = flux_up_results[:, :, -1]
            Fdn_toa = flux_down_results[:, :, -1]
            
            # interpolate the simulated fluxes to ssfr wavelength grid
            p3_up_to_dn_ratio = Fup_p3 / Fdn_p3
            p3_up_to_dn_ratio_mean = np.nanmean(p3_up_to_dn_ratio, axis=0)
            f_p3_up_to_dn_ratio_mean = interp1d(effective_wvl, p3_up_to_dn_ratio_mean, bounds_error=False, fill_value=np.nan)
            p3_up_to_dn_ratio_mean = f_p3_up_to_dn_ratio_mean(cld_leg['ssfr_zen_wvl'])

            f_Fup_sfc_mean = interp1d(effective_wvl, np.nanmean(Fup_sfc, axis=0), bounds_error=False, fill_value=np.nan)
            Fup_sfc_mean_interp = f_Fup_sfc_mean(cld_leg['ssfr_zen_wvl'])
            f_Fdn_sfc_mean = interp1d(effective_wvl, np.nanmean(Fdn_sfc, axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_sfc_mean_interp = f_Fdn_sfc_mean(cld_leg['ssfr_zen_wvl'])
            f_Fdn_sfc_direct_mean = interp1d(effective_wvl, np.nanmean(Fdn_sfc_direct, axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_sfc_direct_mean_interp = f_Fdn_sfc_direct_mean(cld_leg['ssfr_zen_wvl'])
            f_Fdn_sfc_diff_mean = interp1d(effective_wvl, np.nanmean(Fdn_sfc_diff, axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_sfc_diff_mean_interp = f_Fdn_sfc_diff_mean(cld_leg['ssfr_zen_wvl'])
            
            f_Fup_p3_mean = interp1d(effective_wvl, np.nanmean(Fup_p3, axis=0), bounds_error=False, fill_value=np.nan)
            Fup_p3_mean_interp = f_Fup_p3_mean(cld_leg['ssfr_zen_wvl'])
            f_Fdn_p3_mean = interp1d(effective_wvl, np.nanmean(Fdn_p3, axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_p3_mean_interp = f_Fdn_p3_mean(cld_leg['ssfr_zen_wvl'])
            
            f_Fdn_p3_direct_mean = interp1d(effective_wvl, np.nanmean(flux_down_dir_results[:, :, 1], axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_p3_direct_mean_interp = f_Fdn_p3_direct_mean(cld_leg['ssfr_zen_wvl'])
            f_Fdn_p3_diff_mean = interp1d(effective_wvl, np.nanmean(flux_down_diff_results[:, :, 1], axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_p3_diff_mean_interp = f_Fdn_p3_diff_mean(cld_leg['ssfr_zen_wvl'])
            
            f_Fdn_p3_diff_ratio_mean = interp1d(effective_wvl, np.nanmean(Fdn_p3_diff_ratio, axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_p3_diff_ratio_mean_interp = f_Fdn_p3_diff_ratio_mean(cld_leg['ssfr_zen_wvl'])
            
            f_Fup_toa_mean = interp1d(effective_wvl, np.nanmean(Fup_toa, axis=0), bounds_error=False, fill_value=np.nan)
            Fup_toa_mean_interp = f_Fup_toa_mean(cld_leg['ssfr_zen_wvl'])
            f_Fdn_toa_mean = interp1d(effective_wvl, np.nanmean(Fdn_toa, axis=0), bounds_error=False, fill_value=np.nan)
            Fdn_toa_mean_interp = f_Fdn_toa_mean(cld_leg['ssfr_zen_wvl'])

            
            # SSFR observation
            fup_mean = np.nanmean(cld_leg['ssfr_nad'], axis=0)
            fdn_mean = np.nanmean(cld_leg['ssfr_zen'], axis=0)
            fup_std = np.nanstd(cld_leg['ssfr_nad'], axis=0)
            fdn_std = np.nanstd(cld_leg['ssfr_zen'], axis=0)
            
            # hsr1
            hsr1_dn_dif_ratio_mean = np.nanmean(cld_leg['hsr1_dif']/cld_leg['hsr1_tot'], axis=0)
            
            # surface albedo correction following Odell's correction method
            corr_up = Fup_p3_mean_interp / fup_mean
            corr_dn = Fdn_p3_mean_interp / fdn_mean
            
            alb_wvl = cld_leg['ssfr_zen_wvl']
            
            if iter == 0:
                alb_obs = np.nanmean(cld_leg['ssfr_nad']/cld_leg['ssfr_zen'], axis=0)
                alb_obs[alb_obs<0.0] = 0.0
                alb_obs[alb_obs>1.0] = 1.0
            else:
                alb_file = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_{iter}.dat'
                alb_data = np.loadtxt(alb_file, comments='#', skiprows=2)
                alb_obs = alb_data[:, 1][1:-1]
            
            alb_corr = alb_obs * (corr_dn/corr_up)
            alb_corr[:4] = alb_corr[4]
            alb_corr[alb_corr<0.0] = 0.0
            alb_corr[alb_corr>1.0] = 1.0
            
            if iter < 3:
                alb_corr_mask = gas_abs_masking(alb_wvl, alb_corr, alt=alt_avg)
                alb_corr[np.isnan(alb_corr)] = alb_corr_mask[np.isnan(alb_corr)]
                # alb_ice_fit = ice_alb_fitting(alb_wvl, alb_corr, alt=alt_avg)
                alb_ice_fit = snowice_alb_fitting(alb_wvl, alb_corr, alt=alt_avg, clear_sky=clear_sky)
            else:
                alb_corr_mask = np.full_like(alb_corr, np.nan, dtype=float)
                if np.any(np.isnan(alb_corr)):
                    if np.any(np.isfinite(alb_corr)):
                        alb_corr_series = pd.Series(alb_corr)
                        alb_corr_filled = alb_corr_series.ffill(limit=2).bfill(limit=2)
                        while np.any(np.isnan(alb_corr_filled)):
                            alb_corr_filled = alb_corr_filled.ffill(limit=2).bfill(limit=2)
                        alb_corr[np.isnan(alb_corr)] = np.array(alb_corr_filled)[np.isnan(alb_corr)]
                    else:
                        alb_corr = alb_obs.copy()
                alb_ice_fit = alb_corr.copy()
            
 
            heading_saa_diff = heading_avg - saa_avg
            if heading_saa_diff < 0:
                heading_saa_diff += 360.0
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            toa_mean = np.nanmean(cld_leg['ssfr_toa'], axis=0)
            ax.plot(cld_leg['ssfr_zen_wvl'], toa_mean, '--', color='gray', linewidth=1.0, label='TOA')
            ax.plot(cld_leg['ssfr_zen_wvl'], fup_mean, '--', linewidth=1.0, color='royalblue', label='SSFR upward')
            ax.fill_between(cld_leg['ssfr_zen_wvl'],
                            fup_mean-fup_std,
                            fup_mean+fup_std, color='paleturquoise', alpha=0.75)
            ax.plot(cld_leg['ssfr_zen_wvl'], fdn_mean, '--', linewidth=1.0, color='orange', label='SSFR downward')
            ax.fill_between(cld_leg['ssfr_zen_wvl'],
                            fdn_mean-fdn_std,
                            fdn_mean+fdn_std, color='bisque', alpha=0.75)
            ax.plot(effective_wvl, np.nanmean(Fup_p3, axis=0), color='green', linewidth=2.0, label='Simulation upward')
            ax.plot(effective_wvl, np.nanmean(Fdn_p3, axis=0), color='red', linewidth=2.0, label='Simulation downward')
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Flux (W m$^{-2}$ nm$^{-1}$)', fontsize=12)
            ax.set_xlim(cld_leg['ssfr_zen_wvl'][0], cld_leg['ssfr_zen_wvl'][-1])
            ymin, ymax = ax.get_ylim()
            ax.set_ylim([0, ymax])
            ax.legend()
            if iter == 0:
                ax.set_title(f'{date_s} {time_start:.3f}-{time_end:.3f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo = SSFR upward/downward ratio', fontsize=10)
            elif iter == 1:
                ax.set_title(f'{date_s} {time_start:.3f}-{time_end:.3f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo updated (Odell)', fontsize=10)
            elif iter == 2:
                ax.set_title(f'{date_s} {time_start:.3f}-{time_end:.3f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo updated (fit)', fontsize=10)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_time_%.2f-%.2f_alt-%.2fkm_flux_iteration_%d.png' % (date_s, date_s, case_tag, time_start, time_end, alt_avg, iter), bbox_inches='tight', dpi=150)
            # plt.show()
            

            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            toa_mean = np.nanmean(cld_leg['ssfr_toa'], axis=0)
            l1 = ax.plot(cld_leg['ssfr_zen_wvl'], fdn_mean/toa_mean, '--', linewidth=1.0, color='orange', label='SSFR downward/TOA')
            ax.fill_between(cld_leg['ssfr_zen_wvl'],
                            (fdn_mean-fdn_std)/toa_mean,
                            (fdn_mean+fdn_std)/toa_mean, color='bisque', alpha=0.75)
            
            l2 = ax.plot(cld_leg['ssfr_zen_wvl'], Fdn_p3_mean_interp/toa_mean, color='red', linewidth=2.0, label='Simulation downward/TOA')
            l5 = ax.plot(cld_leg['ssfr_zen_wvl'], Fdn_toa_mean_interp/toa_mean, color='green', linewidth=2.0, label='Simulation TOA dn/TOA')
            # l6 = ax.plot(cld_leg['ssfr_zen_wvl'], fdn_mean_low_dif/toa_mean, color='purple', linestyle=':', linewidth=1.0, label='SSFR downward low-diff/TOA')
            # l7 = ax.plot(cld_leg['ssfr_zen_wvl'], fdn_sim_high_dif/toa_mean, color='brown', linestyle='-.', linewidth=1.0, label='Simulation downward high-diff/TOA')
            
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Downward flux / TOA ratio ', fontsize=12)
            
            ax2 = ax.twinx()
            l3 = ax2.plot(cld_leg['hsr1_wvl'], hsr1_dn_dif_ratio_mean, color='brown', linestyle='--', linewidth=1.0, label='HSR-1 diffuse ratio')
            l4 = ax2.plot(cld_leg['ssfr_zen_wvl'], Fdn_p3_diff_ratio_mean_interp, color='magenta', linestyle='-', linewidth=1.0, label='Simulation diffuse ratio')
            
            
            ax2.set_ylabel('Downward diffuse ratio', fontsize=12)
            ax2.set_ylim([0, 0.4])
            
            ax.set_xlim(cld_leg['ssfr_zen_wvl'][0], cld_leg['ssfr_zen_wvl'][-1])
            # plot horizontal line at y=1.0
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)
            
            ax.set_ylim([0.75, 1.15])
            
            # lns = l1 + l2 + l5 + l6 + l7 + l3 + l4
            lns = l1 + l2 + l5 + l3 + l4
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, fontsize=8, loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            if iter == 0:
                ax.set_title(f'{date_s} {time_start:.3f}-{time_end:.3f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo = SSFR upward/downward ratio', fontsize=10)
            elif iter == 1:
                ax.set_title(f'{date_s} {time_start:.3f}-{time_end:.3f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo updated (Odell)', fontsize=10)
            elif iter == 2:
                ax.set_title(f'{date_s} {time_start:.3f}-{time_end:.3f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo updated (fit)', fontsize=10)
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_time_%.2f-%.2f_alt-%.2fkm_toa_dnflux_toa_ratio_iteration_%d.png' % (date_s, date_s, case_tag, time_start, time_end, alt_avg, iter), bbox_inches='tight', dpi=150)

            
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            ax.plot(alb_wvl, alb_avg, label='SSFR upward/downward ratio')
            ax.plot(alb_wvl, alb_corr, label='updated albedo (Odele)')
            ax.plot(alb_wvl, alb_ice_fit, label='updated albedo (fit)')
            # fill between wavelengths where T_total < 0.05
            ax.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(alb_corr_mask), color='gray', alpha=0.2, label='Mask Gas absorption bands')
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Albedo', fontsize=12)
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlim(cld_leg['ssfr_zen_wvl'][0], cld_leg['ssfr_zen_wvl'][-1])
            ax.legend(fontsize=10)
            ax.set_title(f'{date_s} {time_start:.3f}-{time_end:.3f} Alt {alt_avg:.2f}km')
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_time_%.2f-%.2f_alt-%.2fkm_albedo_iteration_%d.png' % (date_s, date_s, case_tag, time_start, time_end, alt_avg, iter), bbox_inches='tight', dpi=150)

        
            (
                fup_total_rmse,
                fup_average_rmse,
                fup_relative_rmse,
                fup_broadband_bias,
                fup_flux_weighted_relative_rmse,
            ) = flux_rmse_metrics(
                fup_mean,
                Fup_p3_mean_interp,
            )
            (
                fdn_total_rmse,
                fdn_average_rmse,
                fdn_relative_rmse,
                fdn_broadband_bias,
                fdn_flux_weighted_relative_rmse,
            ) = flux_rmse_metrics(
                fdn_mean,
                Fdn_p3_mean_interp,
            )

            output_dict = {
                'wvl': cld_leg['ssfr_zen_wvl'],
                'ssfr_fup_mean': fup_mean,
                'ssfr_fdn_mean': fdn_mean,
                'ssfr_fup_std': fup_std,
                'ssfr_fdn_std': fdn_std,
                'simu_fup_sfc_mean': Fup_sfc_mean_interp,
                'simu_fdn_sfc_mean': Fdn_sfc_mean_interp,
                'simu_fdn_sfc_direct_mean': Fdn_sfc_direct_mean_interp,
                'simu_fdn_sfc_diff_mean': Fdn_sfc_diff_mean_interp,
                'simu_fup_mean': Fup_p3_mean_interp,
                'simu_fdn_mean': Fdn_p3_mean_interp,
                'simu_fup_toa_mean': Fup_toa_mean_interp,
                'toa_mean': toa_mean,
                'corr_factor': (corr_dn/corr_up),
                'fup_total_rmse': fup_total_rmse,
                'fup_average_rmse': fup_average_rmse,
                'fup_relative_rmse': fup_relative_rmse,
                'fup_broadband_bias': fup_broadband_bias,
                'fup_flux_weighted_relative_rmse': fup_flux_weighted_relative_rmse,
                'fdn_total_rmse': fdn_total_rmse,
                'fdn_average_rmse': fdn_average_rmse,
                'fdn_relative_rmse': fdn_relative_rmse,
                'fdn_broadband_bias': fdn_broadband_bias,
                'fdn_flux_weighted_relative_rmse': fdn_flux_weighted_relative_rmse,
            }
            
            output_df = pd.DataFrame(output_dict)
            output_df.to_csv(output_csv_name, index=False)
            if iter == 0:
                alb_wvl_extend = np.concatenate(([348.0], alb_wvl, [2050.0]))
                alb_corr_extend = np.concatenate(([alb_corr[0]], alb_corr, [alb_corr[-1]]))
                alb_ice_fit_extend = np.concatenate(([alb_ice_fit[0]], alb_ice_fit, [alb_ice_fit[-1]]))
                # write out the new surface albedo
                #/----------------------------------------------------------------------------\#                    
                alb_avg_update3 = alb_corr_extend.copy()
                alb_avg_update3_nonnan_first_ind = np.where(~np.isnan(alb_avg_update3))[0][0]
                alb_avg_update3[:alb_avg_update3_nonnan_first_ind] = alb_avg_update3[alb_avg_update3_nonnan_first_ind]
                alb_avg_update3_nonnan_last_ind = np.where(~np.isnan(alb_avg_update3))[0][-1]
                alb_avg_update3[alb_avg_update3_nonnan_last_ind:] = alb_avg_update3[alb_avg_update3_nonnan_last_ind]
                write_2col_file(filename=os.path.join(f'{_fdir_general_}/sfc_alb', f'sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_1.dat'),
                                wvl=alb_wvl_extend,
                                val=alb_avg_update3,
                                header=(f'# SSFR atmospheric corrected sfc albedo {date_s}\n'
                                        '# wavelength (nm)      albedo (unitless)\n'
                                        )
                                )
                    
                alb_avg_update4 = alb_ice_fit_extend.copy()
                alb_avg_update4_nonnan_first_ind = np.where(~np.isnan(alb_avg_update4))[0][0]
                alb_avg_update4[:alb_avg_update4_nonnan_first_ind] = alb_avg_update4[alb_avg_update4_nonnan_first_ind]
                alb_avg_update4_nonnan_last_ind = np.where(~np.isnan(alb_avg_update4))[0][-1]
                alb_avg_update4[alb_avg_update4_nonnan_last_ind:] = alb_avg_update4[alb_avg_update4_nonnan_last_ind]
                write_2col_file(filename=os.path.join(f'{_fdir_general_}/sfc_alb', f'sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_2.dat'),
                                wvl=alb_wvl_extend,
                                val=alb_avg_update4,
                                header=(f'# SSFR atmospheric corrected sfc albedo {date_s} with smooth fitting\n'
                                        '# wavelength (nm)      albedo (unitless)\n'
                                        )
                                )
                #\----------------------------------------------------------------------------/#
                del alb_avg_update3, alb_avg_update4
            if iter >= 2:
                alb_wvl_extend = np.concatenate(([348.0], alb_wvl, [2050.0]))
                alb_ice_fit_extend = np.concatenate(([alb_ice_fit[0]], alb_ice_fit, [alb_ice_fit[-1]]))

                alb_avg_update5 = alb_ice_fit_extend.copy()
                alb_avg_update5_nonnan_first_ind = np.where(~np.isnan(alb_avg_update5))[0][0]
                alb_avg_update5[:alb_avg_update5_nonnan_first_ind] = alb_avg_update5[alb_avg_update5_nonnan_first_ind]
                alb_avg_update5_nonnan_last_ind = np.where(~np.isnan(alb_avg_update5))[0][-1]
                alb_avg_update5[alb_avg_update5_nonnan_last_ind:] = alb_avg_update5[alb_avg_update5_nonnan_last_ind]
                next_iter = iter + 1
                write_2col_file(filename=os.path.join(f'{_fdir_general_}/sfc_alb', f'sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_{next_iter}.dat'),
                                wvl=alb_wvl_extend,
                                val=alb_avg_update5,
                                header=(f'# SSFR atmospheric corrected sfc albedo {date_s} with iter {iter} simulation smooth fitting\n'
                                        '# wavelength (nm)      albedo (unitless)\n'
                                        )
                                )
                del alb_avg_update5, next_iter
            if iter > 0:
                # write out the new simulated p3 level upward to downward ratio
                #/----------------------------------------------------------------------------\#
                p3_up_to_dn_ratio_update = p3_up_to_dn_ratio_mean
                p3_up_to_dn_ratio_update_nonnan_first_ind = np.where(~np.isnan(p3_up_to_dn_ratio_update))[0][0]
                p3_up_to_dn_ratio_update[:p3_up_to_dn_ratio_update_nonnan_first_ind] = alb_avg[p3_up_to_dn_ratio_update_nonnan_first_ind]
                p3_up_to_dn_ratio_update_nonnan_last_ind = np.where(~np.isnan(p3_up_to_dn_ratio_update))[0][-1]
                p3_up_to_dn_ratio_update[p3_up_to_dn_ratio_update_nonnan_last_ind:] = p3_up_to_dn_ratio_update[p3_up_to_dn_ratio_update_nonnan_last_ind]
                write_2col_file(filename=os.path.join(f'{_fdir_general_}/sfc_alb', f'p3_up_dn_ratio_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}_{iter}.dat'),
                                wvl=alb_wvl,
                                val=p3_up_to_dn_ratio_update,
                                header=(f'# SSFR atmospheric corrected sfc albedo {date_s} iteration {iter+1}\n'
                                        '# wavelength (nm)      albedo (unitless)\n'
                                        )
                                )
                #\----------------------------------------------------------------------------/#

            del output_dict, output_df
            cld_legs[ileg] = None
            del cld_leg, Fup_sfc, Fdn_sfc, Fdn_sfc_direct, Fdn_sfc_diff
            del Fup_p3, Fdn_p3
            del Fup_sfc_mean_interp, Fdn_sfc_mean_interp
            del Fdn_sfc_direct_mean_interp, Fdn_sfc_diff_mean_interp
            del Fup_p3_mean_interp, Fdn_p3_mean_interp
            del Fup_toa_mean_interp, Fdn_toa_mean_interp
            del fup_total_rmse, fup_average_rmse, fup_relative_rmse
            del fup_broadband_bias, fup_flux_weighted_relative_rmse
            del fdn_total_rmse, fdn_average_rmse, fdn_relative_rmse
            del fdn_broadband_bias, fdn_flux_weighted_relative_rmse
            del fup_mean, fdn_mean, fup_std, fdn_std
            del toa_mean
            del alb_avg, alb_ice_fit

            gc.collect()
        else:
            cld_legs[ileg] = None
            del cld_leg
            gc.collect()

    print("Finished libratran calculations.")  
    #\----------------------------------------------------------------------------/#

    return

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def flux_rmse_metrics(obs, sim):
    """Return RMSE plus energy-balance and flux-weighted relative diagnostics."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(sim)
    if not np.any(valid):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    diff = sim[valid] - obs[valid]
    total_rmse = np.sqrt(np.sum(diff**2))
    average_rmse = np.sqrt(np.mean(diff**2))
    obs_sum = np.sum(obs[valid])
    relative_rmse = total_rmse / obs_sum if obs_sum != 0 else np.nan
    broadband_bias = np.sum(diff) / obs_sum if obs_sum != 0 else np.nan

    weighted_valid = valid & (obs > 0)
    if np.any(weighted_valid):
        obs_weighted = obs[weighted_valid]
        sim_weighted = sim[weighted_valid]
        obs_weighted_sum = np.sum(obs_weighted)
        if obs_weighted_sum != 0:
            weights = obs_weighted / obs_weighted_sum
            fractional_error = (sim_weighted - obs_weighted) / obs_weighted
            flux_weighted_relative_rmse = np.sqrt(np.sum(weights * fractional_error**2))
        else:
            flux_weighted_relative_rmse = np.nan
    else:
        flux_weighted_relative_rmse = np.nan

    return (
        total_rmse,
        average_rmse,
        relative_rmse,
        broadband_bias,
        flux_weighted_relative_rmse,
    )


if __name__ == '__main__':
    try:
        from .runner import run_cases
    except ImportError:
        from runner import run_cases

    CASE_ID = 'case_043'
    ITERATIONS = range(1)
    run_cases(flt_trk_atm_corr, case_id=CASE_ID, iterations=ITERATIONS)
