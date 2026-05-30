"""Post-process SSFR atmospheric-correction products.

This module consumes products written by ``ssfr_atm_corr.workflow``:

* ``*_final.csv`` on the native SSFR wavelength grid.
* ``*_final_extension.csv`` on the extended 300-4000 nm grid.
* ``sfc_alb_*_final.dat`` on the native albedo grid.
* ``sfc_alb_*_final_extension.dat`` on the extended albedo grid.
"""

import datetime
import gc
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

_LRT_SIM_ROOT = str(Path(__file__).resolve().parents[1])
if _LRT_SIM_ROOT not in sys.path:
    sys.path.insert(0, _LRT_SIM_ROOT)

import numpy as np
import pandas as pd

from util import FlightConfig

try:
    from .case_catalog import get_case
    from .settings import _fdir_data_, _fdir_general_
    from .setup import load_cloud_observation_legs, split_tmhr_ranges
except ImportError:
    from case_catalog import get_case
    from settings import _fdir_data_, _fdir_general_
    from setup import load_cloud_observation_legs, split_tmhr_ranges


def make_default_config():
    """Return the default ARCSIX P3B data-location config."""
    return FlightConfig(
        mission='ARCSIX',
        platform='P3B',
        data_root=_fdir_data_,
        root_mac=_fdir_general_,
        root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data',
    )


def read_two_column_file(filename, value_name):
    """Read a two-column wavelength/value text file."""
    df = pd.read_csv(filename, sep=r'\s+', comment='#', names=['wvl', value_name])
    return df['wvl'].to_numpy(), df[value_name].to_numpy()


def maybe_read_two_column_file(filename, value_name):
    """Read a two-column file if it exists, otherwise return ``(None, None)``."""
    if not os.path.exists(filename):
        return None, None
    return read_two_column_file(filename, value_name)


def weighted_broadband_albedo(albedo, weight, wvl):
    """Return wavelength-integrated albedo weighted by spectral irradiance."""
    valid = np.isfinite(albedo) & np.isfinite(weight) & np.isfinite(wvl)
    if np.count_nonzero(valid) < 2:
        return np.nan
    denominator = np.trapz(weight[valid], x=wvl[valid])
    if denominator == 0:
        return np.nan
    return np.trapz(albedo[valid] * weight[valid], x=wvl[valid]) / denominator


def stack_or_object(values):
    """Stack equal-shaped arrays; otherwise keep an object array."""
    if not values:
        return np.array([])
    shapes = [np.shape(value) for value in values]
    if all(shape == shapes[0] for shape in shapes):
        return np.stack(values)
    return np.array(values, dtype=object)


def first_nonempty_array(records, key):
    """Return the first non-empty array stored under key in records."""
    for record in records:
        value = record[key]
        if np.size(value) > 0:
            return value
    return np.array([])


def native_albedo_values(wvl, alb, native_wvl):
    """Map padded native albedo files onto the native CSV wavelength grid."""
    if wvl is None or alb is None:
        return np.full(native_wvl.shape, np.nan, dtype=float)
    if len(wvl) == len(native_wvl) + 2:
        return alb[1:-1]
    if len(wvl) == len(native_wvl) and np.allclose(wvl, native_wvl):
        return alb
    return np.interp(native_wvl, wvl, alb, left=np.nan, right=np.nan)


def process_atm_corr_case(
    date=datetime.datetime(2024, 5, 31),
    tmhr_ranges_select=None,
    case_tag='default',
    config: Optional[FlightConfig] = None,
    simulation_interval=None,
    clear_sky=True,
    output_dir=None,
):
    """Collect final atmospheric-correction products for one case."""
    if tmhr_ranges_select is None:
        tmhr_ranges_select = [[14.10, 14.27]]
    if config is None:
        config = make_default_config()
    if output_dir is None:
        output_dir = f'{_fdir_general_}/sfc_alb_combined_smooth_450nm'

    date_s = date.strftime("%Y%m%d")
    tmhr_ranges_select = split_tmhr_ranges(tmhr_ranges_select, simulation_interval)
    sky_tag = 'clear' if clear_sky else 'sat_cloud'
    fdir_lrt = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_{sky_tag}'
    fdir_alb = f'{_fdir_general_}/sfc_alb'

    cld_legs = load_cloud_observation_legs(
        _fdir_general_,
        config.mission,
        config.platform,
        date_s,
        case_tag,
        tmhr_ranges_select,
    )

    records = []
    skipped = []
    for ileg, cld_leg in enumerate(cld_legs):
        time_start, time_end = cld_leg['time'][0], cld_leg['time'][-1]
        alt_avg = np.round(np.nanmean(cld_leg['alt']), 2)
        stem_time = f'{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km'
        stem_alb = f'{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km'

        final_csv = f'{fdir_lrt}/ssfr_simu_flux_{stem_time}_final.csv'
        final_extension_csv = f'{fdir_lrt}/ssfr_simu_flux_{stem_time}_final_extension.csv'
        if not os.path.exists(final_csv):
            skipped.append((ileg, final_csv))
            continue

        df_final = pd.read_csv(final_csv)
        native_wvl = df_final['wvl'].to_numpy()
        toa_native = df_final['toa_mean'].to_numpy() if 'toa_mean' in df_final else df_final['simu_fdn_toa_mean'].to_numpy()

        alb_paths = {
            'initial': f'{fdir_alb}/sfc_alb_{stem_alb}_iter_0.dat',
            'corrected': f'{fdir_alb}/sfc_alb_{stem_alb}_iter_1.dat',
            'fitted_baseline': f'{fdir_alb}/sfc_alb_{stem_alb}_iter_2.dat',
            'final': f'{fdir_alb}/sfc_alb_{stem_alb}_final.dat',
            'final_extension': f'{fdir_alb}/sfc_alb_{stem_alb}_final_extension.dat',
        }
        albedo_native = {}
        broadband_native = {}
        for key in ('initial', 'corrected', 'fitted_baseline', 'final'):
            alb_wvl, alb = maybe_read_two_column_file(alb_paths[key], 'alb')
            albedo_native[key] = native_albedo_values(alb_wvl, alb, native_wvl)
            broadband_native[key] = weighted_broadband_albedo(albedo_native[key], toa_native, native_wvl)

        extension_wvl = np.array([])
        albedo_extension = np.array([])
        broadband_extension = np.nan
        df_extension = None
        if os.path.exists(final_extension_csv):
            df_extension = pd.read_csv(final_extension_csv)
            ext_alb_wvl, ext_alb = maybe_read_two_column_file(alb_paths['final_extension'], 'alb')
            if ext_alb_wvl is not None:
                extension_wvl = ext_alb_wvl
                albedo_extension = ext_alb
                if 'simu_fdn_toa_final' in df_extension and len(df_extension) == len(extension_wvl):
                    extension_weight = df_extension['simu_fdn_toa_final'].to_numpy()
                else:
                    extension_weight = np.ones_like(extension_wvl)
                broadband_extension = weighted_broadband_albedo(albedo_extension, extension_weight, extension_wvl)

        records.append({
            'leg_index': ileg,
            'time_start': time_start,
            'time_end': time_end,
            'alt_avg': alt_avg,
            'lon_avg': np.nanmean(cld_leg['lon']),
            'lat_avg': np.nanmean(cld_leg['lat']),
            'native_wvl': native_wvl,
            'alb_initial': albedo_native['initial'],
            'alb_corrected': albedo_native['corrected'],
            'alb_fitted_baseline': albedo_native['fitted_baseline'],
            'alb_final': albedo_native['final'],
            'broadband_alb_initial': broadband_native['initial'],
            'broadband_alb_corrected': broadband_native['corrected'],
            'broadband_alb_fitted_baseline': broadband_native['fitted_baseline'],
            'broadband_alb_final': broadband_native['final'],
            'extension_wvl': extension_wvl,
            'alb_final_extension': albedo_extension,
            'broadband_alb_final_extension': broadband_extension,
            'final_csv': final_csv,
            'final_extension_csv': final_extension_csv if os.path.exists(final_extension_csv) else None,
            'albedo_files': alb_paths,
        })
        gc.collect()

    if not records:
        missing = '\n'.join(path for _, path in skipped)
        raise FileNotFoundError(f'No final atmospheric-correction products found for {date_s} {case_tag}.\n{missing}')

    native_wvl = records[0]['native_wvl']
    output = {
        'date': date_s,
        'case_tag': case_tag,
        'tmhr_ranges_select': tmhr_ranges_select,
        'native_wvl': native_wvl,
        'time_start': np.array([record['time_start'] for record in records]),
        'time_end': np.array([record['time_end'] for record in records]),
        'alt_avg': np.array([record['alt_avg'] for record in records]),
        'lon_avg': np.array([record['lon_avg'] for record in records]),
        'lat_avg': np.array([record['lat_avg'] for record in records]),
        'alb_initial': stack_or_object([record['alb_initial'] for record in records]),
        'alb_corrected': stack_or_object([record['alb_corrected'] for record in records]),
        'alb_fitted_baseline': stack_or_object([record['alb_fitted_baseline'] for record in records]),
        'alb_final': stack_or_object([record['alb_final'] for record in records]),
        'broadband_alb_initial': np.array([record['broadband_alb_initial'] for record in records]),
        'broadband_alb_corrected': np.array([record['broadband_alb_corrected'] for record in records]),
        'broadband_alb_fitted_baseline': np.array([record['broadband_alb_fitted_baseline'] for record in records]),
        'broadband_alb_final': np.array([record['broadband_alb_final'] for record in records]),
        'extension_wvl': first_nonempty_array(records, 'extension_wvl'),
        'alb_final_extension': stack_or_object([record['alb_final_extension'] for record in records]),
        'broadband_alb_final_extension': np.array([record['broadband_alb_final_extension'] for record in records]),
        'records': records,
        'skipped': skipped,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_file = (
        f'{output_dir}/sfc_alb_update_{date_s}_{case_tag}_'
        f'time_{tmhr_ranges_select[0][0]:.3f}_{tmhr_ranges_select[-1][-1]:.3f}.pkl'
    )
    with open(output_file, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved processed atmospheric-correction product to {output_file}")
    return output_file


def process_catalog_case(config, case_id, output_dir=None):
    """Process one active atmospheric-correction catalog case by id."""
    case = get_case(case_id)
    year, month, day = [int(part) for part in case['date'].split('-')]
    return process_atm_corr_case(
        date=datetime.datetime(year, month, day),
        tmhr_ranges_select=case['tmhr_ranges_select'],
        case_tag=case['case_tag'],
        config=config,
        simulation_interval=case['simulation_interval'],
        clear_sky=case['clear_sky'],
        output_dir=output_dir,
    )
