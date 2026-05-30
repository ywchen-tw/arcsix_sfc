"""Preprocess SSFR flight legs for the atmospheric-correction workflow.

This module creates the ``flt_cld_obs_info`` pickle files consumed by
``ssfr_atm_corr.workflow``.
"""

import datetime
import gc
import logging
import os
import pickle
import platform
import sys
from pathlib import Path
from typing import Optional

_LRT_SIM_ROOT = str(Path(__file__).resolve().parents[1])
if _LRT_SIM_ROOT not in sys.path:
    sys.path.insert(0, _LRT_SIM_ROOT)

if platform.system() == 'Linux':
    sys.path.append("/projects/yuch8913/arcsix_sfc/lrt_sim/")

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d

from util import FlightConfig, load_h5, nearest_indices, read_ict_kt19, solar_interpolation_func

try:
    from .case_catalog import cloud_observation_file, get_case
    from .helpers import ssfr_flags
    from .qc_plotting import ssfr_time_series_plot
    from .settings import _fdir_data_, _fdir_general_, _mission_, _platform_
    from .setup import split_tmhr_ranges, write_ssfr_support_files
except ImportError:
    from case_catalog import cloud_observation_file, get_case
    from helpers import ssfr_flags
    from qc_plotting import ssfr_time_series_plot
    from settings import _fdir_data_, _fdir_general_, _mission_, _platform_
    from setup import split_tmhr_ranges, write_ssfr_support_files


def make_default_config():
    """Return the default ARCSIX P3B data-location config."""
    return FlightConfig(
        mission='ARCSIX',
        platform='P3B',
        data_root=_fdir_data_,
        root_mac=_fdir_general_,
        root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data',
    )


def load_marli(config, date_s):
    """Load MARLI data if available for this date."""
    if date_s == '20240603':
        return {'time': np.array([]), 'Alt': np.array([]), 'H': np.array([]), 'T': np.array([]), 'LSR': np.array([]), 'WVMR': np.array([])}

    try:
        with Dataset(str(config.marli(date_s))) as ds:
            return {var: ds.variables[var][:] for var in ("time", "Alt", "H", "T", "LSR", "WVMR")}
    except FileNotFoundError:
        print(f"No MARLI data found for date {date_s}")
        return {'time': np.array([]), 'Alt': np.array([]), 'H': np.array([]), 'T': np.array([]), 'LSR': np.array([]), 'WVMR': np.array([])}


def load_kt19(config, date_s):
    """Load KT19 data if available for this date."""
    try:
        print("Loading KT19 data from:", config.kt19(date_s))
        _, data_kt19 = read_ict_kt19(config.kt19(date_s))
        return data_kt19
    except FileNotFoundError:
        print(f"No KT19 data found for date {date_s}")
        return {'tmhr': np.array([]), 'ir_sfc_T': np.array([])}


def collect_cloud_observation_legs(
    date=datetime.datetime(2024, 5, 31),
    tmhr_ranges_select=None,
    case_tag='default',
    config: Optional[FlightConfig] = None,
    simulation_interval=None,
    clear_sky=True,
    overwrite=False,
    plot_qc=True,
    manual_cloud=False,
    manual_cloud_cer=14.4,
    manual_cloud_cwp=0.06013,
    manual_cloud_cth=0.945,
    manual_cloud_cbh=0.344,
    manual_cloud_cot=6.26,
):
    """Create cloud-observation pickle files for one atmospheric-correction case."""
    if tmhr_ranges_select is None:
        tmhr_ranges_select = [[14.10, 14.27]]
    if config is None:
        config = make_default_config()

    log = logging.getLogger("ssfr_atm_corr.preprocess")
    date_s = date.strftime("%Y%m%d")
    log.info("Starting preprocessing for %s %s", date_s, case_tag)

    os.makedirs(f'fig/{date_s}', exist_ok=True)
    tmhr_ranges_select = split_tmhr_ranges(tmhr_ranges_select, simulation_interval)

    data_hsk = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    data_marli = load_marli(config, date_s)
    data_kt19 = load_kt19(config, date_s)

    if plot_qc:
        ssfr_time_series_plot(
            data_hsk,
            data_ssfr,
            data_hsr1,
            tmhr_ranges_select,
            date_s,
            case_tag,
            pitch_roll_thres=3.0,
        )

    # Ensure the slit-convolved solar spectrum used for SSFR TOA is available.
    write_ssfr_support_files(iter=0, clear_sky=clear_sky)
    flux_solar_interp = solar_interpolation_func(
        solar_flux_file='arcsix_ssfr_solar_flux_slit.dat',
        date=date,
    )

    t_hsk = np.asarray(data_hsk["tmhr"])
    t_ssfr = data_ssfr['time'] / 3600.0
    t_hsr1 = data_hsr1['time'] / 3600.0
    t_marli = data_marli['time']
    t_kt19 = data_kt19['tmhr']

    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)

    output_files = []
    for ileg, (time_start, time_end) in enumerate(tmhr_ranges_select):
        fname_pkl = cloud_observation_file(
            _fdir_general_,
            _mission_,
            _platform_,
            date_s,
            case_tag,
            time_start,
            time_end,
        )
        output_files.append(fname_pkl)
        if os.path.exists(fname_pkl) and not overwrite:
            print(f"Preprocessed cloud observation exists, skipping: {fname_pkl}")
            continue

        mask = (t_hsk >= time_start) & (t_hsk <= time_end)
        if not np.any(mask):
            print(f"No housekeeping samples for leg {ileg + 1}: {time_start:.3f}-{time_end:.3f}h; skipping.")
            continue

        times_leg = t_hsk[mask]
        print(f"Preprocessing leg {ileg + 1}: time range {times_leg.min():.3f}-{times_leg.max():.3f}h")

        sel_ssfr = nearest_indices(t_hsk, mask, t_ssfr)
        sel_hsr1 = nearest_indices(t_hsk, mask, t_hsr1)
        sel_marli = nearest_indices(t_hsk, mask, t_marli) if len(t_marli) > 0 else None
        sel_kt19 = nearest_indices(t_hsk, mask, t_kt19) if len(t_kt19) > 0 else None

        leg = {
            "time": times_leg,
            "alt": data_hsk["alt"][mask] / 1000.0,
            "heading": data_hsk["ang_hed"][mask],
            "hsr1_tot": data_hsr1["f_dn_tot"][sel_hsr1],
            "hsr1_dif": data_hsr1["f_dn_dif"][sel_hsr1],
            "hsr1_wvl": data_hsr1["wvl_dn_tot"],
            "lon": data_hsk["lon"][mask],
            "lat": data_hsk["lat"][mask],
            "sza": data_hsk["sza"][mask],
            "saa": data_hsk["saa"][mask],
        }

        if sel_marli is not None:
            marli_wvmr = data_marli["WVMR"][sel_marli, :]
            marli_wvmr[marli_wvmr == 9999] = np.nan
            marli_wvmr[marli_wvmr > 50] = np.nan
            marli_wvmr[marli_wvmr < 0] = 0
            marli_h = data_marli["H"][...]
            marli_mask = np.any(np.isfinite(marli_wvmr), axis=0)
            leg.update({
                "marli_h": marli_h[marli_mask],
                "marli_wvmr": np.nanmean(marli_wvmr[:, marli_mask], axis=0),
            })
        else:
            leg.update({"marli_h": None, "marli_wvmr": None})

        if sel_kt19 is not None:
            leg["kt19_sfc_T"] = data_kt19['ir_sfc_T'][sel_kt19]
        else:
            leg["kt19_sfc_T"] = np.full_like(times_leg, np.nan, dtype=float)

        if clear_sky:
            leg.update({
                "cot": np.full_like(leg['lon'], np.nan),
                "cer": np.full_like(leg['lon'], np.nan),
                "cwp": np.full_like(leg['lon'], np.nan),
                "cth": np.full_like(leg['lon'], np.nan),
                "cgt": np.full_like(leg['lon'], np.nan),
                "cbh": np.full_like(leg['lon'], np.nan),
            })
        elif manual_cloud:
            leg.update({
                "cot": np.full_like(leg['lon'], manual_cloud_cot),
                "cer": np.full_like(leg['lon'], manual_cloud_cer),
                "cwp": np.full_like(leg['lon'], manual_cloud_cwp),
                "cth": np.full_like(leg['lon'], manual_cloud_cth),
                "cgt": np.full_like(leg['lon'], manual_cloud_cth - manual_cloud_cbh),
                "cbh": np.full_like(leg['lon'], manual_cloud_cbh),
            })
        else:
            raise NotImplementedError("Automatic cloud retrieval is not implemented in SSFR preprocessing yet.")

        sza_hsk = data_hsk['sza'][mask]
        ssfr_zen_flux = data_ssfr['f_dn'][sel_ssfr, :].copy()
        ssfr_nad_flux = data_ssfr['f_up'][sel_ssfr, :].copy()
        ssfr_zen_wvl = data_ssfr['wvl_dn']
        ssfr_nad_wvl = data_ssfr['wvl_up']
        ssfr_zen_toa = flux_solar_interp(ssfr_zen_wvl) * np.cos(np.deg2rad(sza_hsk))[:, np.newaxis]

        ssfr_nad_flux_interp = np.full_like(ssfr_zen_flux, np.nan, dtype=float)
        for j in range(ssfr_nad_flux.shape[0]):
            f_nad_flux_interp = interp1d(
                ssfr_nad_wvl,
                ssfr_nad_flux[j, :],
                axis=0,
                bounds_error=False,
                fill_value='extrapolate',
            )
            ssfr_nad_flux_interp[j, :] = f_nad_flux_interp(ssfr_zen_wvl)

        pitch_roll_mask = np.sqrt(data_hsk["ang_pit"][mask]**2 + data_hsk["ang_rol"][mask]**2) < 3.0
        ssfr_zen_flux[~pitch_roll_mask, :] = np.nan
        ssfr_nad_flux_interp[~pitch_roll_mask, :] = np.nan
        ssfr_zen_toa[~pitch_roll_mask, :] = np.nan

        icing = (data_ssfr['flag'] & ssfr_flags.camera_icing) != 0
        icing_pre = (data_ssfr['flag'] & ssfr_flags.camera_icing_pre) != 0
        alp_ang_pit_rol_issue = (data_ssfr['flag'] & ssfr_flags.alp_ang_pit_rol_issue) != 0
        alp_ang_pit_rol_issue_leg = alp_ang_pit_rol_issue[sel_ssfr]
        ssfr_zen_flux[alp_ang_pit_rol_issue_leg, :] = np.nan
        ssfr_nad_flux_interp[alp_ang_pit_rol_issue_leg, :] = np.nan

        leg['ssfr_zen'] = ssfr_zen_flux
        leg['ssfr_nad'] = ssfr_nad_flux_interp
        leg['ssfr_zen_wvl'] = ssfr_zen_wvl
        leg['ssfr_nad_wvl'] = ssfr_zen_wvl
        leg['ssfr_toa'] = ssfr_zen_toa
        leg['ssfr_icing'] = icing[sel_ssfr]
        leg['ssfr_icing_pre'] = icing_pre[sel_ssfr]

        with open(fname_pkl, 'wb') as f:
            pickle.dump(leg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved preprocessed cloud observation to {fname_pkl}")

        del leg, sel_ssfr, sel_hsr1
        gc.collect()

    return output_files


def preprocess_catalog_case(config, case_id, overwrite=False, plot_qc=True):
    """Preprocess one active catalog case by id."""
    case = get_case(case_id)
    year, month, day = [int(part) for part in case['date'].split('-')]
    return collect_cloud_observation_legs(
        date=datetime.datetime(year, month, day),
        tmhr_ranges_select=case['tmhr_ranges_select'],
        case_tag=case['case_tag'],
        config=config,
        simulation_interval=case['simulation_interval'],
        clear_sky=case['clear_sky'],
        overwrite=overwrite,
        plot_qc=plot_qc,
        manual_cloud=case['manual_cloud'],
        manual_cloud_cer=case['manual_cloud_cer'] or 0.0,
        manual_cloud_cwp=case['manual_cloud_cwp'] or 0.0,
        manual_cloud_cth=case['manual_cloud_cth'] or 0.0,
        manual_cloud_cbh=case['manual_cloud_cbh'] or 0.0,
        manual_cloud_cot=case['manual_cloud_cot'] or 0.0,
    )
