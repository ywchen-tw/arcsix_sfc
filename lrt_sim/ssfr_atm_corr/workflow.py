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
import csv
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

import copy
import shutil
import datetime
import logging
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import bisect
import gc
import functools
import multiprocessing as mp
# mpl.use('Agg')


import er3t

# from util.util import *
# from util.arcsix_atm import prepare_atmospheric_profile
from util import *


MIN_FINAL_ITERATION = 2

try:
    from .settings import *
    from .helpers import find_h2o_6_end, gas_abs_masking, write_2col_file
    from .postfit import (
        apply_postfit_correction,
        postfit_h2o_6_fit_end,
        postfit_h2o_6_mask_end,
        use_full_postfit_window,
    )
    from .setup import (
        default_atm_levels,
        load_cloud_observation_leg,
        load_nearest_dropsonde,
        run_uvspec_inits,
        split_tmhr_ranges,
        write_final_sw_support_files,
        write_ssfr_support_files,
    )
except ImportError:
    from settings import *
    from helpers import find_h2o_6_end, gas_abs_masking, write_2col_file
    from postfit import (
        apply_postfit_correction,
        postfit_h2o_6_fit_end,
        postfit_h2o_6_mask_end,
        use_full_postfit_window,
    )
    from setup import (
        default_atm_levels,
        load_cloud_observation_leg,
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


def interp_nan(x, y, x_new):
    """Linearly interpolate one spectrum, returning NaN outside the source range."""
    return np.interp(x_new, x, y, left=np.nan, right=np.nan)


def fill_nan_nearest(values):
    """Fill NaN gaps with nearest finite spectral neighbors."""
    filled = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(filled)
    if not np.any(missing):
        return filled

    finite = np.flatnonzero(~missing)
    if finite.size == 0:
        return filled

    missing_indices = np.flatnonzero(missing)
    right_pos = np.searchsorted(finite, missing_indices)
    left_pos = np.clip(right_pos - 1, 0, finite.size - 1)
    right_pos = np.clip(right_pos, 0, finite.size - 1)
    left = finite[left_pos]
    right = finite[right_pos]
    use_right = missing_indices - left > right - missing_indices
    nearest = np.where(use_right, right, left)
    filled[missing] = filled[nearest]
    return filled


def extend_edges_with_nearest_finite(values):
    """Fill only leading/trailing NaNs using the nearest finite values."""
    filled = np.asarray(values, dtype=float).copy()
    finite = np.flatnonzero(np.isfinite(filled))
    if finite.size == 0:
        return filled

    first = finite[0]
    last = finite[-1]
    filled[:first] = filled[first]
    filled[last + 1:] = filled[last]
    return filled


def write_column_csv(filename, columns):
    """Write equal-length column arrays without constructing a DataFrame."""
    headers = list(columns)
    raw_arrays = [np.asarray(columns[name]) for name in headers]
    if not raw_arrays:
        raise ValueError('No columns to write.')

    n_rows = next((array.shape[0] for array in raw_arrays if array.ndim > 0), None)
    if n_rows is None:
        raise ValueError('At least one CSV column must be non-scalar.')

    arrays = [
        np.full(n_rows, array.item()) if array.ndim == 0 else array
        for array in raw_arrays
    ]
    mismatched = [name for name, array in zip(headers, arrays) if array.shape[0] != n_rows]
    if mismatched:
        raise ValueError(f'CSV columns have mismatched lengths: {", ".join(mismatched)}')

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(zip(*arrays))


_LEG_POOL_MAXTASKSPERCHILD = 8


def _leg_worker_init():
    """Force a non-interactive matplotlib backend inside pool workers."""
    import matplotlib
    matplotlib.use('Agg', force=True)


def _run_legs_parallel(leg_items, leg_common, workers):
    """Fan out independent sub-legs across a process pool (memory ~ workers x one leg)."""
    worker = functools.partial(_process_leg, **leg_common)
    tasks = [(ileg, ts, te) for ileg, (ts, te) in leg_items]
    ctx = mp.get_context('fork' if platform.system() == 'Linux' else 'spawn')
    with ctx.Pool(
        processes=workers,
        initializer=_leg_worker_init,
        maxtasksperchild=_LEG_POOL_MAXTASKSPERCHILD,
    ) as pool:
        pool.starmap(worker, tasks)


def _process_leg(ileg, selected_time_start, selected_time_end, *, date, case_tag,
                 clear_sky, overwrite_lrt, iter, final_sim, final_status,
                 final_extension_rt, manual_cloud, manual_cloud_cer, manual_cloud_cwp,
                 manual_cloud_cth, manual_cloud_cbh, manual_cloud_cot, levels,
                 effective_wvl, wavelength_grid_file, solar_file, data_dropsonde_legs):
    """Run one flight-track sub-leg: prep, libRadtran, post-process, plots, outputs."""
    date_s = date.strftime("%Y%m%d")
    zpt_filedir = f'{_fdir_general_}/zpt/{date_s}'
    cld_leg = load_cloud_observation_leg(
        _fdir_general_,
        _mission_,
        _platform_,
        date_s,
        case_tag,
        selected_time_start,
        selected_time_end,
    )
    if cld_leg.get('skip'):
        print(
            f"Leg {ileg+1}: flagged as skip during preprocessing "
            f"({cld_leg.get('skip_reason', '')}); skipping."
        )
        del cld_leg
        return
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
        del cld_leg, ssfr_zen_flux, ssfr_nad_flux
        gc.collect()
        return

    # Skip legs where too many wavelength channels are all-NaN in the time-mean.
    # A channel is NaN in nanmean only when every time step at that wavelength is
    # invalid. The ffill/bfill fill-loop below would propagate a constant value
    # across those channels, producing a flat spectrum that corrupts snowice fitting.
    _MAX_NAN_WVL_FRAC = 0.3
    zen_nan_frac = np.mean(~np.isfinite(np.nanmean(ssfr_zen_flux, axis=0)))
    nad_nan_frac = np.mean(~np.isfinite(np.nanmean(ssfr_nad_flux, axis=0)))
    if zen_nan_frac > _MAX_NAN_WVL_FRAC or nad_nan_frac > _MAX_NAN_WVL_FRAC:
        print(
            f"Skipping leg {ileg+1} ({time_start:.3f}-{time_end:.3f}h): "
            f"too many NaN wavelengths in SSFR "
            f"(zen {zen_nan_frac:.0%}, nad {nad_nan_frac:.0%})"
        )
        del cld_leg, ssfr_zen_flux, ssfr_nad_flux
        gc.collect()
        return

    del ssfr_zen_flux, ssfr_nad_flux

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
    atm_profile_file = os.path.join(
        zpt_filedir,
        f'atm_profiles_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.dat',
    )
    ch4_profile_file = os.path.join(
        zpt_filedir,
        f'ch4_profiles_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.dat',
    )

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
            print(
                f"Warning: final-extension RT output exists but adjusted extended "
                f"albedo is missing: {final_extension_albedo_name}. "
                "Run processing before the final-extension RT pass."
            )
        if not final_extension_rt:
            print(
                f"Final native-grid products prepared for leg {ileg+1}; "
                "skipping final-extension RT in the regular workflow."
            )
            return
        if not os.path.exists(final_extension_albedo_name):
            raise FileNotFoundError(
                f"Missing adjusted final-extension albedo: {final_extension_albedo_name}\n"
                "Run processing first so it can extend the adjusted albedo, then "
                "rerun with final_extension_rt=True."
            )


    if not os.path.exists(output_csv_name) or overwrite_lrt:
        print("Start leg %d atmospheric correction ..." % (ileg+1))
        print(f"Date: {date_s}, Time: {time_start:.3f}-{time_end:.3f}h, Alt: {alt_avg:.2f}km")
        if iter==0:
            if os.path.exists(atm_profile_file) and os.path.exists(ch4_profile_file):
                print(
                    f"Using existing atmospheric profile files for "
                    f"{date_s} {time_start:.3f}-{time_end:.3f}h Alt {alt_avg:.2f}km"
                )
            else:
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
                alb_avg = fill_nan_nearest(alb_avg)

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
            alb_avg = np.loadtxt(iter_0_fname, comments='#')[:, 1]
            alb_avg = alb_avg[1:-1]  # remove the extended edges
        #\----------------------------------------------------------------------------/#

        atm_z_grid = levels
        z_list = atm_z_grid
        atm_z_grid_str = ' '.join(['%.3f' % z for z in atm_z_grid])

        lrt_cfg = copy.deepcopy(er3t.rtm.lrt.get_lrt_cfg())
        lrt_cfg['atmosphere_file'] = atm_profile_file
        lrt_cfg['solar_file'] = solar_file
        if platform.system() == 'Darwin':
            lrt_cfg['number_of_streams'] = 4
        elif platform.system() == 'Linux':
            lrt_cfg['number_of_streams'] = 4 if (final_sim and not clear_sky) else 8
        lrt_cfg['mol_abs_param'] = 'reptran coarse'

        albedo_file = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_{iter}.dat'
        final_alb_wvl = None
        final_alb = None
        if final_sim:
            albedo_file = final_extension_albedo_name
            alb_data = np.loadtxt(albedo_file, comments='#')
            final_alb_wvl = alb_data[:, 0]
            final_alb = np.clip(alb_data[:, 1], 0.0, 1.0)
        input_dict_extra = {
            'crs_model': 'rayleigh Bodhaine29',
            'albedo_file': albedo_file,
            'mol_file': 'CH4 %s' % ch4_profile_file,
            'wavelength_grid_file': wavelength_grid_file,
            'atm_z_grid': atm_z_grid_str,
        }

        cld_cfg = None
        if not clear_sky:
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
                else:
                    fname_cld = f'{fdir_tmp}/cld_0000_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.txt'
                    try:
                        os.remove(fname_cld)
                    except FileNotFoundError:
                        pass
                    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
                    cld_cfg['cloud_file'] = fname_cld
                    cld_cfg['cloud_altitude'] = cloud_altitude
                    cld_cfg['cloud_effective_radius'] = cer_x
                    cld_cfg['liquid_water_content'] = cwp_x / cgt_x
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
            elif manual_cloud:
                raise ValueError(
                    f'Invalid manual cloud for leg {ileg+1}: '
                    f'CBH={cbh_x}, CTH={cth_x}, COT={cot_x}, CWP={cwp_x}.'
                )

        init = er3t.rtm.lrt.lrt_init_mono_flx(
            input_file=f'{fdir_tmp}/input_0000_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.txt',
            output_file=f'{fdir_tmp}/output_0000_{date_s}_{case_tag}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.txt',
            date=date,
            solar_zenith_angle=sza_avg,
            Nx=len(effective_wvl),
            output_altitude=[0, alt_avg, 'toa'],
            input_dict_extra=input_dict_extra.copy(),
            mute_list=['albedo', 'wavelength', 'spline', 'slit_function_file'],
            lrt_cfg=lrt_cfg,
            cld_cfg=cld_cfg,
            aer_cfg=None,
        )

        print(f"Start running libratran calculations for {output_csv_name.replace('.csv', '')} ")
        print('Running libratran calculations ...')
        run_uvspec_inits([init])
        output_wvl, flux_down, flux_down_dir, flux_down_diff, flux_up = read_uvspec_flux_output(init)
        if output_wvl.size != effective_wvl.size or not np.allclose(output_wvl, effective_wvl):
            print(
                f'Using uvspec output wavelength grid ({output_wvl.size} points) '
                f'instead of requested grid ({effective_wvl.size} points).'
            )
            effective_wvl = output_wvl

        if final_sim:
            fup_mean = np.nanmean(cld_leg['ssfr_nad'], axis=0)
            fdn_mean = np.nanmean(cld_leg['ssfr_zen'], axis=0)
            fup_std = np.nanstd(cld_leg['ssfr_nad'], axis=0)
            fdn_std = np.nanstd(cld_leg['ssfr_zen'], axis=0)

            output_dict = {
                'wvl': effective_wvl,
                'final_iter': np.full(effective_wvl.shape, iter, dtype=float),
                'final_status': np.full(effective_wvl.shape, final_status, dtype=object),
                'final_closure_met': np.full(effective_wvl.shape, final_status == 'closure_passed', dtype=bool),
                'time_start': np.full(effective_wvl.shape, time_start, dtype=float),
                'time_end': np.full(effective_wvl.shape, time_end, dtype=float),
                'alt_km': np.full(effective_wvl.shape, alt_avg, dtype=float),
                'sza': np.full(effective_wvl.shape, sza_avg, dtype=float),
                'ssfr_fup_mean': interp_nan(cld_leg['ssfr_zen_wvl'], fup_mean, effective_wvl),
                'ssfr_fdn_mean': interp_nan(cld_leg['ssfr_zen_wvl'], fdn_mean, effective_wvl),
                'ssfr_fup_std': interp_nan(cld_leg['ssfr_zen_wvl'], fup_std, effective_wvl),
                'ssfr_fdn_std': interp_nan(cld_leg['ssfr_zen_wvl'], fdn_std, effective_wvl),
                'sfc_alb_final': np.interp(effective_wvl, final_alb_wvl, final_alb, left=np.nan, right=np.nan),
                'simu_fup_sfc_final': flux_up[:, 0],
                'simu_fdn_sfc_final': flux_down[:, 0],
                'simu_fdn_sfc_direct_final': flux_down_dir[:, 0],
                'simu_fdn_sfc_diff_final': flux_down_diff[:, 0],
                'simu_fup_p3_final': flux_up[:, 1],
                'simu_fdn_p3_final': flux_down[:, 1],
                'simu_fdn_p3_direct_final': flux_down_dir[:, 1],
                'simu_fdn_p3_diff_final': flux_down_diff[:, 1],
                'simu_fup_toa_final': flux_up[:, -1],
                'simu_fdn_toa_final': flux_down[:, -1],
            }
            write_column_csv(output_csv_name, output_dict)
            print(f"Saving final extended spectral simulation to {output_csv_name}")

            del output_dict
            del flux_down, flux_down_dir, flux_down_diff, flux_up
            del fup_mean, fdn_mean, fup_std, fdn_std, cld_leg
            gc.collect()
            return

        for flux_arr in (flux_down, flux_down_dir, flux_down_diff, flux_up):
            for iz in range(flux_arr.shape[1]):
                flux_arr[:, iz] = ssfr_slit_convolve(effective_wvl, flux_arr[:, iz], wvl_joint=950)


        # simulated fluxes at surface
        Fup_sfc = flux_up[:, 0]
        Fdn_sfc = flux_down[:, 0]
        Fdn_sfc_direct = flux_down_dir[:, 0]
        Fdn_sfc_diff = flux_down_diff[:, 0]

        # simulated fluxes at p3 altitude
        Fup_p3 = flux_up[:, 1]
        Fdn_p3 = flux_down[:, 1]
        Fdn_p3_direct = flux_down_dir[:, 1]
        Fdn_p3_diff = flux_down_diff[:, 1]
        Fdn_p3_diff_ratio = Fdn_p3_diff / Fdn_p3

        # simulated fluxes at toa
        Fup_toa = flux_up[:, -1]
        Fdn_toa = flux_down[:, -1]

        # interpolate the simulated fluxes to ssfr wavelength grid
        p3_up_to_dn_ratio = Fup_p3 / Fdn_p3
        target_wvl = cld_leg['ssfr_zen_wvl']
        p3_up_to_dn_ratio_mean = interp_nan(effective_wvl, p3_up_to_dn_ratio, target_wvl)
        Fup_sfc_mean_interp = interp_nan(effective_wvl, Fup_sfc, target_wvl)
        Fdn_sfc_mean_interp = interp_nan(effective_wvl, Fdn_sfc, target_wvl)
        Fdn_sfc_direct_mean_interp = interp_nan(effective_wvl, Fdn_sfc_direct, target_wvl)
        Fdn_sfc_diff_mean_interp = interp_nan(effective_wvl, Fdn_sfc_diff, target_wvl)
        Fup_p3_mean_interp = interp_nan(effective_wvl, Fup_p3, target_wvl)
        Fdn_p3_mean_interp = interp_nan(effective_wvl, Fdn_p3, target_wvl)
        Fdn_p3_direct_mean_interp = interp_nan(effective_wvl, Fdn_p3_direct, target_wvl)
        Fdn_p3_diff_mean_interp = interp_nan(effective_wvl, Fdn_p3_diff, target_wvl)
        Fdn_p3_diff_ratio_mean_interp = interp_nan(effective_wvl, Fdn_p3_diff_ratio, target_wvl)
        Fup_toa_mean_interp = interp_nan(effective_wvl, Fup_toa, target_wvl)
        Fdn_toa_mean_interp = interp_nan(effective_wvl, Fdn_toa, target_wvl)


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

        # corr_dn: always mask — downwelling always traverses the full atmospheric column
        corr_dn_masked = gas_abs_masking(alb_wvl, corr_dn, alt=999.0)
        corr_dn_filled = np.where(np.isnan(corr_dn_masked), 1.0, corr_dn_masked)

        # Full masking is the current default. The settings switch retains the
        # reduced low-altitude corr_up mask as an optional future experiment.
        corr_up_masked = gas_abs_masking(
            alb_wvl,
            corr_up,
            alt=alt_avg,
            altitude_dependent=ALTITUDE_DEPENDENT_GAS_MASKING,
        )
        corr_up_filled = np.where(np.isnan(corr_up_masked), 1.0, corr_up_masked)

        alb_corr = alb_obs * (corr_dn_filled / corr_up_filled)
        alb_corr[:4] = alb_corr[4]
        alb_corr[alb_corr < 0.0] = 0.0
        alb_corr[alb_corr > 1.0] = 1.0

        adaptive_h2o_6_end = find_h2o_6_end(alb_wvl, alb_corr)
        if date_s == '20240603':
            mask_h2o_6_end = postfit_h2o_6_mask_end(date_s, adaptive_h2o_6_end)
            fit_h2o_6_end = postfit_h2o_6_fit_end(date_s, mask_h2o_6_end)
        else:
            mask_h2o_6_end = postfit_h2o_6_mask_end('20240807', adaptive_h2o_6_end)
            fit_h2o_6_end = postfit_h2o_6_fit_end('20240807', mask_h2o_6_end)
        if mask_h2o_6_end != h2o_6_end:
            suffix = ' date override' if mask_h2o_6_end != adaptive_h2o_6_end else ''
            print(
                f'Extending H2O-6 mask end from {h2o_6_end:.1f} '
                f'to {mask_h2o_6_end:.1f} nm.{suffix}'
            )
        alb_corr_mask = gas_abs_masking(
            alb_wvl,
            alb_corr,
            alt=alt_avg,
            h2o_6_end_override=mask_h2o_6_end,
        )
        alb_corr[np.isnan(alb_corr)] = alb_corr_mask[np.isnan(alb_corr)]
        if np.any(np.isnan(alb_corr)):
            if np.any(np.isfinite(alb_corr)):
                alb_corr = fill_nan_nearest(alb_corr)
            else:
                alb_corr = alb_obs.copy()

        try:
            # Keep fitting/smoothing gas absorption bands for every iteration,
            # including iter >= 3 and whichever iteration is later copied to final.
            alb_ice_fit = snowice_alb_fitting(
                alb_wvl,
                alb_corr,
                alt=alt_avg,
                clear_sky=clear_sky,
                h2o_6_end=fit_h2o_6_end,
            )
            if np.all(~np.isfinite(alb_ice_fit)):
                raise ValueError('snow/ice fitted albedo is all non-finite')
            alb_ice_fit = apply_postfit_correction(
                alb_wvl,
                alb_ice_fit,
                date_s=date_s,
                alt=alt_avg,
                clear_sky=clear_sky,
                preserve_outside_window=not use_full_postfit_window(date_s),
                window_start=fit_h2o_6_end,
                window_end=h2o_7_start,
            )
        except Exception as err:
            print(
                f'Warning: snow/ice albedo fitting failed for {date_s} '
                f'{time_start:.3f}-{time_end:.3f} iter {iter}: {err}. '
                'Using corrected albedo without fitted gas-band smoothing.'
            )
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
        ax.plot(effective_wvl, Fup_p3, color='green', linewidth=2.0, label='Simulation upward')
        ax.plot(effective_wvl, Fdn_p3, color='red', linewidth=2.0, label='Simulation downward')
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
        elif iter >= 2:
            ax.set_title(f'{date_s} {time_start:.3f}-{time_end:.3f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo updated (fit)', fontsize=10)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_time_%.2f-%.2f_alt-%.2fkm_flux_iteration_%d.png' % (date_s, date_s, case_tag, time_start, time_end, alt_avg, iter), bbox_inches='tight', dpi=150)
        plt.close(fig)
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
        elif iter >= 2:
            ax.set_title(f'{date_s} {time_start:.3f}-{time_end:.3f} Alt {alt_avg:.2f}km heading-saa {heading_saa_diff:.1f} deg\nAlbedo updated (fit)', fontsize=10)
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_time_%.2f-%.2f_alt-%.2fkm_toa_dnflux_toa_ratio_iteration_%d.png' % (date_s, date_s, case_tag, time_start, time_end, alt_avg, iter), bbox_inches='tight', dpi=150)
        plt.close(fig)


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
        plt.close(fig)


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

        write_column_csv(output_csv_name, output_dict)
        if iter == 0:
            alb_wvl_extend = np.concatenate(([348.0], alb_wvl, [2050.0]))
            alb_corr_extend = np.concatenate(([alb_corr[0]], alb_corr, [alb_corr[-1]]))
            alb_ice_fit_extend = np.concatenate(([alb_ice_fit[0]], alb_ice_fit, [alb_ice_fit[-1]]))
            # write out the new surface albedo
            #/----------------------------------------------------------------------------\#                    
            alb_avg_update3 = extend_edges_with_nearest_finite(alb_corr_extend)
            write_2col_file(filename=os.path.join(f'{_fdir_general_}/sfc_alb', f'sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_1.dat'),
                            wvl=alb_wvl_extend,
                            val=alb_avg_update3,
                            header=(f'# SSFR atmospheric corrected sfc albedo {date_s}\n'
                                    '# wavelength (nm)      albedo (unitless)\n'
                                    )
                            )

            alb_avg_update4 = extend_edges_with_nearest_finite(alb_ice_fit_extend)
            write_2col_file(filename=os.path.join(f'{_fdir_general_}/sfc_alb', f'sfc_alb_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km_iter_2.dat'),
                            wvl=alb_wvl_extend,
                            val=alb_avg_update4,
                            header=(f'# SSFR atmospheric corrected sfc albedo {date_s} with smooth fitting\n'
                                    '# wavelength (nm)      albedo (unitless)\n'
                                    )
                            )
            #\----------------------------------------------------------------------------/#
            del alb_avg_update3, alb_avg_update4
        if iter >= 1:
            alb_wvl_extend = np.concatenate(([348.0], alb_wvl, [2050.0]))
            alb_ice_fit_extend = np.concatenate(([alb_ice_fit[0]], alb_ice_fit, [alb_ice_fit[-1]]))

            alb_avg_update5 = extend_edges_with_nearest_finite(alb_ice_fit_extend)
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
            p3_up_to_dn_ratio_update = extend_edges_with_nearest_finite(p3_up_to_dn_ratio_mean)
            write_2col_file(filename=os.path.join(f'{_fdir_general_}/sfc_alb', f'p3_up_dn_ratio_{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}_{iter}.dat'),
                            wvl=alb_wvl,
                            val=p3_up_to_dn_ratio_update,
                            header=(f'# SSFR atmospheric corrected sfc albedo {date_s} iteration {iter+1}\n'
                                    '# wavelength (nm)      albedo (unitless)\n'
                                    )
                            )
            #\----------------------------------------------------------------------------/#

        del output_dict
        del cld_leg, Fup_sfc, Fdn_sfc, Fdn_sfc_direct, Fdn_sfc_diff
        del Fup_p3, Fdn_p3, Fdn_p3_direct, Fdn_p3_diff
        del flux_down, flux_down_dir, flux_down_diff, flux_up
        del Fup_sfc_mean_interp, Fdn_sfc_mean_interp
        del Fdn_sfc_direct_mean_interp, Fdn_sfc_diff_mean_interp
        del Fup_p3_mean_interp, Fdn_p3_mean_interp
        del Fdn_p3_direct_mean_interp, Fdn_p3_diff_mean_interp
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
        del cld_leg
        gc.collect()



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
                     final_extension_rt=False,
                     workers=1,
                    ):
    
    log = logging.getLogger("lrt")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    if final_sim and iter < MIN_FINAL_ITERATION:
        raise ValueError(
            f'Refusing final simulation for {date_s}_{case_tag} from iteration {iter}. '
            f'Final products require iter >= {MIN_FINAL_ITERATION} because iter_1 has '
            'not gone through smooth/fitting.'
        )
    
    os.makedirs(f'fig/{date_s}', exist_ok=True)
    
    tmhr_ranges_select = split_tmhr_ranges(tmhr_ranges_select, simulation_interval)

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

    if final_sim and final_extension_rt:
        effective_wvl = write_final_sw_support_files()
        wavelength_grid_file = os.path.abspath('wvl_grid_final_sw.dat')
        solar_file = os.path.abspath('arcsix_ssfr_solar_flux_raw_final.dat')
    elif final_sim:
        effective_wvl = np.array([])
        wavelength_grid_file = None
        solar_file = None
    else:
        effective_wvl = write_ssfr_support_files(iter=iter, clear_sky=clear_sky)
        wavelength_grid_file = os.path.abspath('wvl_grid_test.dat')
        solar_file = os.path.abspath('arcsix_ssfr_solar_flux_raw.dat')
            

    leg_common = dict(
        date=date, case_tag=case_tag, clear_sky=clear_sky, overwrite_lrt=overwrite_lrt,
        iter=iter, final_sim=final_sim, final_status=final_status,
        final_extension_rt=final_extension_rt, manual_cloud=manual_cloud,
        manual_cloud_cer=manual_cloud_cer, manual_cloud_cwp=manual_cloud_cwp,
        manual_cloud_cth=manual_cloud_cth, manual_cloud_cbh=manual_cloud_cbh,
        manual_cloud_cot=manual_cloud_cot, levels=levels, effective_wvl=effective_wvl,
        wavelength_grid_file=wavelength_grid_file, solar_file=solar_file,
        data_dropsonde_legs=data_dropsonde_legs,
    )
    leg_items = list(enumerate(tmhr_ranges_select))
    if workers and workers > 1 and len(leg_items) > 1:
        _run_legs_parallel(leg_items, leg_common, workers)
    else:
        for ileg, (selected_time_start, selected_time_end) in leg_items:
            _process_leg(ileg, selected_time_start, selected_time_end, **leg_common)
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

    CASE_ID = 'case_015'
    ITERATIONS = range(1)
    run_cases(flt_trk_atm_corr, case_id=CASE_ID, iterations=ITERATIONS)
