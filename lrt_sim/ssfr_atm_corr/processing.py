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
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
for _path in (_REPO_ROOT, _LRT_SIM_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd

from util import FlightConfig

try:
    from .case_catalog import get_case
    from .helpers import gas_abs_masking
    from .settings import _fdir_data_, _fdir_general_, gas_bands
    from .setup import load_cloud_observation_legs, split_tmhr_ranges
except ImportError:
    from case_catalog import get_case
    from helpers import gas_abs_masking
    from settings import _fdir_data_, _fdir_general_, gas_bands
    from setup import load_cloud_observation_legs, split_tmhr_ranges


# TEMPORARY: CURC maintenance prevents rerunning final simulations/albedo copies.
# Prefer iter_2 as the final albedo stand-in even when stale ``sfc_alb_*_final.dat``
# files exist. Remove this override and use real final albedo products after the
# workflow can be rerun.
TEMPORARY_FINAL_ALBEDO_ITER = 2


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


def dataframe_weight(df, preferred_columns, expected_length):
    """Return the first available finite spectral weight column from a DataFrame."""
    for column in preferred_columns:
        if column in df and len(df[column]) == expected_length:
            weight = pd.to_numeric(df[column], errors='coerce').to_numpy()
            if np.any(np.isfinite(weight)):
                return weight
    return np.ones(expected_length, dtype=float)


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


def concatenate_record_arrays(records, key):
    """Concatenate one-dimensional arrays stored in each record."""
    arrays = [np.asarray(record[key]) for record in records if np.size(record[key]) > 0]
    if not arrays:
        return np.array([])
    return np.concatenate(arrays)


def repeat_spectral_by_time(records, spectral_key, default_wvl):
    """Repeat one leg-mean spectrum for every sample time in each leg."""
    repeated = []
    for record in records:
        n_time = len(record['time'])
        spectrum = np.asarray(record[spectral_key], dtype=float)
        if spectrum.size == 0:
            spectrum = np.full(default_wvl.shape, np.nan, dtype=float)
        repeated.append(np.repeat(spectrum[np.newaxis, :], n_time, axis=0))
    if not repeated:
        return np.empty((0, len(default_wvl)))
    return np.vstack(repeated)


def repeat_scalar_by_time(records, scalar_key):
    """Repeat one leg-mean scalar for every sample time in each leg."""
    repeated = [
        np.full(len(record['time']), record[scalar_key], dtype=float)
        for record in records
    ]
    if not repeated:
        return np.array([])
    return np.concatenate(repeated)


def native_albedo_values(wvl, alb, native_wvl):
    """Map padded native albedo files onto the native CSV wavelength grid."""
    if wvl is None or alb is None:
        return np.full(native_wvl.shape, np.nan, dtype=float)
    if len(wvl) == len(native_wvl) + 2:
        return alb[1:-1]
    if len(wvl) == len(native_wvl) and np.allclose(wvl, native_wvl):
        return alb
    return np.interp(native_wvl, wvl, alb, left=np.nan, right=np.nan)


def all_nonfinite(values):
    """Return True when an array has no finite values."""
    values = np.asarray(values, dtype=float)
    return values.size == 0 or np.all(~np.isfinite(values))


def final_iteration_from_extension(df_extension):
    """Return final iteration encoded in a final-extension CSV, if present."""
    if df_extension is None or 'final_iter' not in df_extension:
        return None
    values = pd.to_numeric(df_extension['final_iter'], errors='coerce').dropna().unique()
    if len(values) == 0:
        return None
    return int(round(float(values[0])))


def plot_gas_absorption_bands(ax, wvl=None, alt=None):
    """Shade configured gas absorption bands on an albedo axis."""
    if wvl is not None and alt is not None:
        wvl = np.asarray(wvl, dtype=float)
        masked = gas_abs_masking(wvl, np.zeros_like(wvl, dtype=float), alt=alt)
        ax.fill_between(
            wvl,
            -0.05,
            1.05,
            where=np.isnan(masked),
            color='gray',
            alpha=0.3,
        )
        return

    for band_start, band_end in gas_bands:
        ax.axvspan(band_start, band_end, color='gray', alpha=0.3)


def plot_native_albedo_diagnostic(record, series_keys, labels, colors, filename, title):
    """Plot one native-grid leg diagnostic from stored albedo spectra."""
    import matplotlib.pyplot as plt

    wvl = np.asarray(record['native_wvl'], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label, color in zip(series_keys, labels, colors):
        values = np.asarray(record[key], dtype=float)
        if values.size != wvl.size or np.all(~np.isfinite(values)):
            continue
        ax.plot(wvl, values, '-', color=color, label=label)

    plot_gas_absorption_bands(ax, wvl=wvl, alt=record.get('alt_avg'))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(350, 2000)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_mean_final_albedo(date_s, case_tag, output, fig_dir):
    """Plot mean final native-grid albedo across all processed legs."""
    import matplotlib.pyplot as plt

    wvl = np.asarray(output['native_wvl'], dtype=float)
    alb_final = np.asarray(output['alb_final'], dtype=float)
    if alb_final.ndim != 2 or alb_final.shape[1] != wvl.size:
        return None

    os.makedirs(fig_dir, exist_ok=True)
    alb_avg = np.nanmean(alb_final, axis=0)
    alb_std = np.nanstd(alb_final, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wvl, alb_avg, '-', color='blue', label='Mean albedo (final)')
    ax.fill_between(wvl, alb_avg - alb_std, alb_avg + alb_std, color='blue', alpha=0.1)
    plot_gas_absorption_bands(ax)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(350, 2000)
    ax.set_title(f'Surface Albedo (final) for {date_s} {case_tag}', fontsize=13)
    fig.tight_layout()

    filename = f'{fig_dir}/arcsix_albedo_{date_s}_{case_tag}.png'
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return filename


def plot_broadband_albedo_map(date_s, case_tag, output, fig_dir):
    """Plot broadband final albedo by leg location, with Cartesian fallback."""
    import matplotlib.pyplot as plt

    lon_avg = np.asarray(output['lon_avg'], dtype=float)
    lat_avg = np.asarray(output['lat_avg'], dtype=float)
    lon_all = np.asarray(output['lon_all'], dtype=float)
    lat_all = np.asarray(output['lat_all'], dtype=float)
    color_vals = np.asarray(output['broadband_alb_final'], dtype=float)
    if lon_avg.size == 0 or lat_avg.size == 0:
        return None

    os.makedirs(fig_dir, exist_ok=True)
    try:
        import cartopy
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        plt.close('all')
        central_lon = float(np.nanmean(lon_avg)) if lon_avg.size > 0 else 0.0
        proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(1, 1, 1, projection=proj)
        ax.coastlines(resolution='50m', linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', zorder=0)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        if np.all(~np.isfinite(color_vals)):
            ax.scatter(
                lon_avg,
                lat_avg,
                s=40,
                c='red',
                transform=ccrs.PlateCarree(),
                zorder=3,
                edgecolor='k',
                label='Flight legs',
            )
        else:
            sc = ax.scatter(
                lon_avg,
                lat_avg,
                s=40,
                c=color_vals,
                cmap='viridis',
                transform=ccrs.PlateCarree(),
                zorder=3,
                edgecolor='k',
            )
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
            cbar.set_label('Broadband Albedo (final)', fontsize=10)

        if lon_all.size > 0 and lat_all.size > 0:
            ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)
            lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
            lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
            pad_lon = max(0.5, (lon_max - lon_min) * 0.2)
            pad_lat = max(0.5, (lat_max - lat_min) * 0.2)
            ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())

        ax.set_title(f'Polar projection (North) - Flight legs {date_s} {case_tag}', fontsize=12)
        fig.tight_layout()
        filename = f'{fig_dir}/{date_s}_{case_tag}_broadband_albedo_vs_longitude_polar_projection.png'
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return filename
    except Exception as err:
        print(f'Cartopy polar plot failed for {date_s} {case_tag}; falling back to Cartesian scatter. Error: {err}')

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    if np.all(~np.isfinite(color_vals)):
        ax.scatter(lon_avg, lat_avg, c='red', s=40, edgecolor='k')
    else:
        sc = ax.scatter(lon_avg, lat_avg, c=color_vals, cmap='viridis', s=40, edgecolor='k')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Broadband Albedo (final)', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Flight legs {date_s} {case_tag}')
    fig.tight_layout()
    filename = f'{fig_dir}/{date_s}_{case_tag}_broadband_albedo_vs_longitude_cartesian.png'
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return filename


def plot_processing_outputs(date_s, case_tag, records, output, fig_dir='fig', plot_every=1):
    """Generate legacy-style post-processing plots for one case."""
    plot_files = []
    leg_fig_dir = f'{fig_dir}/{date_s}'
    summary_fig_dir = f'{fig_dir}/sfc_alb_corr_lonlat'
    os.makedirs(leg_fig_dir, exist_ok=True)

    for i, record in enumerate(records):
        if plot_every and i % plot_every != 0:
            continue
        stem = (
            f'arcsix_albedo_{date_s}_{case_tag}_'
            f'{record["time_start"]:.3f}-{record["time_end"]:.3f}_'
            f'{record["alt_avg"]:.2f}km_leg-{record["leg_index"]:03d}'
        )
        title_base = (
            f'Surface Albedo {date_s} {case_tag} '
            f'time {record["time_start"]:.3f}-{record["time_end"]:.3f}, '
            f'alt {record["alt_avg"]:.2f} km'
        )

        iter012_file = f'{leg_fig_dir}/{stem}_iter0_iter1_iter2.png'
        plot_native_albedo_diagnostic(
            record,
            ('alb_initial', 'alb_corrected', 'alb_fitted_baseline'),
            ('iter 0', 'iter 1', 'iter 2'),
            ('black', 'blue', 'green'),
            iter012_file,
            f'{title_base}\niter 0, iter 1, iter 2',
        )
        plot_files.append(iter012_file)

        final_label = 'final'
        if record.get('temporary_final_albedo'):
            final_label = f'final (TEMP iter {TEMPORARY_FINAL_ALBEDO_ITER})'

        iter01final_file = f'{leg_fig_dir}/{stem}_iter0_iter1_final.png'
        plot_native_albedo_diagnostic(
            record,
            ('alb_initial', 'alb_corrected', 'alb_final'),
            ('iter 0', 'iter 1', final_label),
            ('black', 'blue', 'red'),
            iter01final_file,
            f'{title_base}\niter 0, iter 1, final',
        )
        plot_files.append(iter01final_file)

    mean_file = plot_mean_final_albedo(date_s, case_tag, output, summary_fig_dir)
    if mean_file is not None:
        plot_files.append(mean_file)

    map_file = plot_broadband_albedo_map(date_s, case_tag, output, summary_fig_dir)
    if map_file is not None:
        plot_files.append(map_file)

    return plot_files


def process_atm_corr_case(
    date=datetime.datetime(2024, 5, 31),
    tmhr_ranges_select=None,
    case_tag='default',
    config: Optional[FlightConfig] = None,
    simulation_interval=None,
    clear_sky=True,
    output_dir=None,
    make_plots=True,
    fig_dir='fig',
    plot_every=1,
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
        if cld_leg.get('skip'):
            skipped.append((ileg, f'skip-flagged: {cld_leg.get("skip_reason", "")}'))
            continue
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
        native_weight = dataframe_weight(
            df_final,
            ('simu_fdn_sfc_mean', 'simu_fdn_sfc_final', 'toa_mean', 'simu_fdn_toa_mean'),
            len(native_wvl),
        )

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
            broadband_native[key] = weighted_broadband_albedo(albedo_native[key], native_weight, native_wvl)

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
                extension_weight = dataframe_weight(
                    df_extension,
                    ('simu_fdn_sfc_final', 'simu_fdn_sfc_mean', 'simu_fdn_toa_final'),
                    len(extension_wvl),
                )
                broadband_extension = weighted_broadband_albedo(albedo_extension, extension_weight, extension_wvl)
            elif {'wvl', 'sfc_alb_final'}.issubset(df_extension.columns):
                extension_wvl = df_extension['wvl'].to_numpy()
                albedo_extension = df_extension['sfc_alb_final'].to_numpy()
                extension_weight = dataframe_weight(
                    df_extension,
                    ('simu_fdn_sfc_final', 'simu_fdn_sfc_mean', 'simu_fdn_toa_final'),
                    len(extension_wvl),
                )
                broadband_extension = weighted_broadband_albedo(albedo_extension, extension_weight, extension_wvl)

        final_iter = final_iteration_from_extension(df_extension)
        final_albedo_source = alb_paths['final']
        temporary_final_albedo = False
        final_albedo_warning = None
        fallback_albedo = None
        temporary_iter_path = f'{fdir_alb}/sfc_alb_{stem_alb}_iter_{TEMPORARY_FINAL_ALBEDO_ITER}.dat'
        fallback_wvl, fallback_alb = maybe_read_two_column_file(temporary_iter_path, 'alb')
        if fallback_wvl is not None:
            fallback_albedo = native_albedo_values(fallback_wvl, fallback_alb, native_wvl)
            final_albedo_source = temporary_iter_path
            temporary_final_albedo = True
            if os.path.exists(alb_paths['final']):
                final_albedo_warning = (
                    f'TEMPORARY FINAL ALBEDO OVERRIDE: ignoring existing {alb_paths["final"]}; '
                    f'using iter_{TEMPORARY_FINAL_ALBEDO_ITER} instead. '
                    'Replace with real final albedo after rerunning workflow.'
                )
            else:
                final_albedo_warning = (
                    f'TEMPORARY FINAL ALBEDO: missing {alb_paths["final"]}; '
                    f'using iter_{TEMPORARY_FINAL_ALBEDO_ITER} instead. '
                    'Replace with real final albedo after rerunning workflow.'
                )
            print(f'WARNING: {final_albedo_warning}')
        elif all_nonfinite(albedo_native['final']):
            if final_iter is not None and final_iter != TEMPORARY_FINAL_ALBEDO_ITER:
                final_iter_path = f'{fdir_alb}/sfc_alb_{stem_alb}_iter_{final_iter}.dat'
                fallback_wvl, fallback_alb = maybe_read_two_column_file(final_iter_path, 'alb')
                if fallback_wvl is not None:
                    fallback_albedo = native_albedo_values(fallback_wvl, fallback_alb, native_wvl)
                    final_albedo_source = final_iter_path
                    final_albedo_warning = (
                        f'Missing {alb_paths["final"]}; using final_iter={final_iter} '
                        'from final-extension CSV. Replace with real final albedo when available.'
                    )
                    print(f'WARNING: {final_albedo_warning}')
            if fallback_albedo is None and not all_nonfinite(albedo_extension):
                fallback_albedo = np.interp(native_wvl, extension_wvl, albedo_extension, left=np.nan, right=np.nan)
                final_albedo_source = final_extension_csv
                final_albedo_warning = (
                    f'Missing {alb_paths["final"]}; interpolated sfc_alb_final from '
                    'final-extension CSV. Replace with real native final albedo when available.'
                )
                print(f'WARNING: {final_albedo_warning}')
        if fallback_albedo is not None:
            albedo_native['final'] = fallback_albedo
            broadband_native['final'] = weighted_broadband_albedo(albedo_native['final'], native_weight, native_wvl)

        records.append({
            'leg_index': ileg,
            'final_iter': final_iter,
            'final_albedo_source': final_albedo_source,
            'temporary_final_albedo': temporary_final_albedo,
            'final_albedo_warning': final_albedo_warning,
            'time': np.asarray(cld_leg['time']),
            'time_start': time_start,
            'time_end': time_end,
            'alt': np.asarray(cld_leg['alt']),
            'alt_avg': alt_avg,
            'lon': np.asarray(cld_leg['lon']),
            'lon_avg': np.nanmean(cld_leg['lon']),
            'lat': np.asarray(cld_leg['lat']),
            'lat_avg': np.nanmean(cld_leg['lat']),
            'sza': np.asarray(cld_leg['sza']),
            'saa': np.asarray(cld_leg['saa']),
            'kt19_sfc_T': np.asarray(
                cld_leg.get('kt19_sfc_T', np.full_like(cld_leg['time'], np.nan, dtype=float))
            ),
            'ssfr_fdn': np.asarray(cld_leg['ssfr_zen']),
            'ssfr_fup': np.asarray(cld_leg['ssfr_nad']),
            'ssfr_toa': np.asarray(cld_leg['ssfr_toa']),
            'ssfr_icing': np.asarray(cld_leg.get('ssfr_icing', np.zeros(len(cld_leg['time']), dtype=bool))),
            'ssfr_icing_pre': np.asarray(cld_leg.get('ssfr_icing_pre', np.zeros(len(cld_leg['time']), dtype=bool))),
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
        'final_iter': np.array([
            np.nan if record['final_iter'] is None else record['final_iter']
            for record in records
        ]),
        'final_albedo_source': np.array([record['final_albedo_source'] for record in records], dtype=object),
        'temporary_final_albedo': np.array([record['temporary_final_albedo'] for record in records], dtype=bool),
        'final_albedo_warnings': [
            record['final_albedo_warning']
            for record in records
            if record['final_albedo_warning'] is not None
        ],
        'records': records,
        'skipped': skipped,
    }
    output.update({
        # Compatibility aliases for older CRE/ERA5 plotting scripts.
        'wvl': output['native_wvl'],
        'alb_iter0': output['alb_initial'],
        'alb_iter1': output['alb_corrected'],
        'alb_iter2': output['alb_final'],
        'broadband_alb_iter0': output['broadband_alb_initial'],
        'broadband_alb_iter1': output['broadband_alb_corrected'],
        'broadband_alb_iter2': output['broadband_alb_final'],
        'broadband_alb_iter0_filter': output['broadband_alb_initial'],
        'broadband_alb_iter1_filter': output['broadband_alb_corrected'],
        'broadband_alb_iter2_filter': output['broadband_alb_final'],
        'ext_wvl': output['extension_wvl'],
        'alb_iter2_ext_all': repeat_spectral_by_time(records, 'alb_final_extension', output['extension_wvl']),
        'broadband_alb_iter2_ext_all': repeat_scalar_by_time(records, 'broadband_alb_final_extension'),
        'time_all': concatenate_record_arrays(records, 'time'),
        'lon_all': concatenate_record_arrays(records, 'lon'),
        'lat_all': concatenate_record_arrays(records, 'lat'),
        'alt_all': concatenate_record_arrays(records, 'alt'),
        'sza_all': concatenate_record_arrays(records, 'sza'),
        'saa_all': concatenate_record_arrays(records, 'saa'),
        'kt19_sfc_T_all': concatenate_record_arrays(records, 'kt19_sfc_T'),
        'fdn_all': concatenate_record_arrays(records, 'ssfr_fdn'),
        'fup_all': concatenate_record_arrays(records, 'ssfr_fup'),
        'toa_expand_all': concatenate_record_arrays(records, 'ssfr_toa'),
        'icing_all': concatenate_record_arrays(records, 'ssfr_icing'),
        'icing_pre_all': concatenate_record_arrays(records, 'ssfr_icing_pre'),
        'alb_iter1_all': repeat_spectral_by_time(records, 'alb_corrected', output['native_wvl']),
        'alb_iter2_all': repeat_spectral_by_time(records, 'alb_final', output['native_wvl']),
        'broadband_alb_iter1_all': repeat_scalar_by_time(records, 'broadband_alb_corrected'),
        'broadband_alb_iter2_all': repeat_scalar_by_time(records, 'broadband_alb_final'),
        'broadband_alb_iter1_all_filter': repeat_scalar_by_time(records, 'broadband_alb_corrected'),
        'broadband_alb_iter2_all_filter': repeat_scalar_by_time(records, 'broadband_alb_final'),
    })

    if make_plots:
        output['plot_files'] = plot_processing_outputs(
            date_s,
            case_tag,
            records,
            output,
            fig_dir=fig_dir,
            plot_every=plot_every,
        )
    else:
        output['plot_files'] = []

    os.makedirs(output_dir, exist_ok=True)
    output_file = (
        f'{output_dir}/sfc_alb_update_{date_s}_{case_tag}_'
        f'time_{tmhr_ranges_select[0][0]:.3f}_{tmhr_ranges_select[-1][-1]:.3f}.pkl'
    )
    with open(output_file, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved processed atmospheric-correction product to {output_file}")
    return output_file


def process_catalog_case(config, case_id, output_dir=None, make_plots=True, fig_dir='fig', plot_every=1):
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
        make_plots=make_plots,
        fig_dir=fig_dir,
        plot_every=plot_every,
    )
