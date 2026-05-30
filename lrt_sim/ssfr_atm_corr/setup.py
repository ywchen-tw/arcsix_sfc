"""Setup helpers for the SSFR atmospheric-correction workflow."""

import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path

_LRT_SIM_ROOT = str(Path(__file__).resolve().parents[1])
if _LRT_SIM_ROOT not in sys.path:
    sys.path.insert(0, _LRT_SIM_ROOT)

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from util import closest_indices, dropsonde_time_loc_list, gaussian, read_ict_dropsonde, ssfr_slit_convolve

try:
    from .helpers import write_2col_file
except ImportError:
    from helpers import write_2col_file


def split_tmhr_ranges(tmhr_ranges_select, simulation_interval):
    """Split selected time ranges into shorter intervals."""
    if simulation_interval is None:
        return tmhr_ranges_select

    tmhr_ranges_select_new = []
    for lo, hi in tmhr_ranges_select:
        t_start = lo
        while t_start < hi and t_start < (hi - 0.0167 / 6):
            t_end = min(t_start + simulation_interval / 60.0, hi)
            tmhr_ranges_select_new.append([t_start, t_end])
            t_start = t_end
    return tmhr_ranges_select_new


def default_atm_levels():
    return np.concatenate((
        np.arange(0, 0.26, 0.05),
        np.arange(0.3, 1.0, 0.1),
        np.arange(1.0, 2.1, 0.2),
        np.arange(2.5, 4.1, 0.5),
        np.arange(5.0, 10.1, 2.5),
        np.array([15, 20, 30.0, 40.0, 50.0]),
    ))


def load_nearest_dropsonde(fdir_general, date, tmhr_ranges_select, log):
    """Load the nearest dropsonde profile for each selected flight time window."""
    dropsonde_file_list, dropsonde_date_list, dropsonde_tmhr_list, _, _ = dropsonde_time_loc_list(
        dir_dropsonde=f'{fdir_general}/dropsonde'
    )

    date_select = dropsonde_date_list == date.date()
    if np.sum(date_select) == 0:
        print(f"No dropsonde data found for date {date.strftime('%Y-%m-%d')}")
        log.info(f"No dropsonde data found for date {date.strftime('%Y-%m-%d')}")
        return [None for _ in tmhr_ranges_select]

    dropsonde_tmhr_array = np.array(dropsonde_tmhr_list)[date_select]
    dropsonde_files = np.array(dropsonde_file_list)[date_select]
    mid_tmhr = np.array([np.mean(rng) for rng in tmhr_ranges_select])
    dropsonde_idx = closest_indices(dropsonde_tmhr_array, mid_tmhr)

    data_dropsonde_legs = []
    data_dropsonde_by_file = {}
    for time_range, idx in zip(tmhr_ranges_select, dropsonde_idx):
        dropsonde_file = dropsonde_files[idx]
        if dropsonde_file not in data_dropsonde_by_file:
            _, data_dropsonde_by_file[dropsonde_file] = read_ict_dropsonde(
                dropsonde_file,
                encoding='utf-8',
                na_values=[-9999999, -777, -888],
            )
        data_dropsonde_legs.append(data_dropsonde_by_file[dropsonde_file])
        log.info(
            "Using dropsonde file for %.3f-%.3fh: %s",
            time_range[0],
            time_range[1],
            dropsonde_file,
        )

    return data_dropsonde_legs


def lrt_wavelength_grid(clear_sky):
    if platform.system() == 'Darwin':
        if clear_sky:
            return np.arange(350, 2000.1, 2.5)
        return np.arange(350, 2000.1, 10.0)
    if platform.system() == 'Linux':
        return np.arange(350, 2000.1, 1.0)
    return np.arange(350, 2000.1, 2.5)


def lrt_final_sw_wavelength_grid():
    if platform.system() == 'Darwin':
        return np.arange(300, 4000.1, 10.0)
    if platform.system() == 'Linux':
        return np.arange(300, 4000.1, 1.0)
    return np.arange(300, 4000.1, 10.0)


def write_final_sw_support_files():
    """Write support files for the final 300-4000 nm spectral SW run."""
    final_wvl = lrt_final_sw_wavelength_grid()
    write_2col_file(
        'wvl_grid_final_sw.dat',
        final_wvl,
        np.zeros_like(final_wvl),
        header=('# SSFR final SW wavelength grid file\n'
                '# wavelength (nm)\n'),
    )

    df_solor = pd.read_csv('CU_composite_solar_processed.dat', sep=r'\s+', header=None)
    wvl_solar = np.array(df_solor.iloc[:, 0])
    flux_solar = np.array(df_solor.iloc[:, 1])

    f_interp = interp1d(wvl_solar, flux_solar, kind='linear', bounds_error=False, fill_value=0.0)
    wvl_solar_interp = np.arange(250, 4550.1, 1.0)
    flux_solar_interp = f_interp(wvl_solar_interp)
    mask = wvl_solar_interp <= 4500

    write_2col_file(
        'arcsix_ssfr_solar_flux_raw_final.dat',
        wvl_solar_interp[mask],
        flux_solar_interp[mask],
        header=('# SSFR final SW solar flux without slit function convolution\n'
                '# wavelength (nm)      flux (mW/m^2/nm)\n'),
    )

    return final_wvl


def write_ssfr_support_files(iter, clear_sky):
    """Write slit-function, wavelength-grid, and solar-flux files."""
    xx = np.linspace(-12, 12, 241)
    yy_gaussian_vis = gaussian(xx, 0, 3.8251)
    yy_gaussian_nir = gaussian(xx, 0, 4.5046)

    xx_wvl_grid = lrt_wavelength_grid(clear_sky)

    if iter == 0:
        write_2col_file(
            'vis_0.1nm_update.dat',
            xx,
            yy_gaussian_vis,
            header=('# SSFR Silicon slit function\n'
                    '# wavelength (nm)      relative intensity\n'),
        )
        write_2col_file(
            'nir_0.1nm_update.dat',
            xx,
            yy_gaussian_nir,
            header=('# SSFR InGaAs slit function\n'
                    '# wavelength (nm)      relative intensity\n'),
        )
        write_2col_file(
            'wvl_grid_test.dat',
            xx_wvl_grid,
            np.zeros_like(xx_wvl_grid),
            header=('# SSFR Wavelength grid test file\n'
                    '# wavelength (nm)\n'),
        )

    wvl_solar_vis = np.arange(300, 950.1, 1.0)
    wvl_solar_nir = np.arange(951, 2500.1, 1.0)
    wvl_solar_coarse = np.concatenate([wvl_solar_vis, wvl_solar_nir])
    effective_wvl = wvl_solar_coarse[
        np.logical_and(wvl_solar_coarse >= xx_wvl_grid.min(), wvl_solar_coarse <= xx_wvl_grid.max())
    ]

    if iter == 0:
        df_solor = pd.read_csv('CU_composite_solar_processed.dat', sep=r'\s+', header=None)
        wvl_solar = np.array(df_solor.iloc[:, 0])
        flux_solar = np.array(df_solor.iloc[:, 1])

        f_interp = interp1d(wvl_solar, flux_solar, kind='linear', bounds_error=False, fill_value=0.0)
        wvl_solar_interp = np.arange(250, 2550.1, 1.0)
        flux_solar_interp = f_interp(wvl_solar_interp)

        mask = wvl_solar_interp <= 2500
        wvl_solar = wvl_solar_interp[mask]
        flux_solar = flux_solar_interp[mask]

        assert (xx[1] - xx[0]) - (wvl_solar[1] - wvl_solar[0]) < 1e-3

        flux_solar_convolved = ssfr_slit_convolve(wvl_solar, flux_solar, wvl_joint=950)

        write_2col_file(
            'arcsix_ssfr_solar_flux_raw.dat',
            wvl_solar,
            flux_solar,
            header=('# SSFR version solar flux without slit function convolution\n'
                    '# wavelength (nm)      flux (mW/m^2/nm)\n'),
        )
        write_2col_file(
            'arcsix_ssfr_solar_flux_slit.dat',
            wvl_solar,
            flux_solar_convolved,
            header=('# SSFR version solar flux with slit function convolution\n'
                    '# wavelength (nm)      flux (mW/m^2/nm)\n'),
        )

    return effective_wvl


def load_cloud_observation_legs(fdir_general, mission, platform_name, date_s, case_tag, tmhr_ranges_select):
    fdir_cld_obs_info = f'{fdir_general}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)

    cld_legs = []
    for time_start, time_end in tmhr_ranges_select:
        fname_cld_obs_info = (
            '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl'
            % (
                fdir_cld_obs_info,
                mission.lower(),
                platform_name.lower(),
                date_s,
                case_tag,
                time_start,
                time_end,
            )
        )
        print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
        if not os.path.exists(fname_cld_obs_info):
            raise FileNotFoundError(
                f'Missing preprocessed cloud-observation file: {fname_cld_obs_info}\n'
                'Run the SSFR atmospheric-correction preprocessing first, for example:\n'
                f'  python3 -m lrt_sim.ssfr_atm_corr.preprocess_runner <case_id>\n'
                'or, from inside lrt_sim:\n'
                f'  python3 -m ssfr_atm_corr.preprocess_runner <case_id>'
            )
        with open(fname_cld_obs_info, 'rb') as f:
            cld_legs.append(pickle.load(f))
    return cld_legs


def run_uvspec_inits(inits):
    """Run libRadtran directly, bypassing er3t.lrt_run wrappers."""
    try:
        from tqdm import tqdm
        iterator = tqdm(inits, total=len(inits))
    except ImportError:
        iterator = inits

    for init in iterator:
        for filename in (init.input_file, init.output_file):
            parent = os.path.dirname(filename)
            if parent:
                os.makedirs(parent, exist_ok=True)

        with open(init.input_file, 'w') as f:
            for key, value in init.input_dict.items():
                if key not in init.mute_list:
                    f.write('%-20s %s\n' % (key, value))

            if init.input_dict_extra is not None:
                for key, value in init.input_dict_extra.items():
                    if key not in init.mute_list:
                        key_write = key[:key.index('_add')] if '_add' in key else key
                        f.write('%-20s %s\n' % (key_write, value))

            f.write('quiet')

        command = f'{init.executable_file} < {init.input_file} > {init.output_file}'
        print(f'Run command: {command}')
        with open(init.input_file, 'r') as stdin, open(init.output_file, 'w') as stdout:
            subprocess.run(
                [init.executable_file],
                stdin=stdin,
                stdout=stdout,
                check=True,
            )
