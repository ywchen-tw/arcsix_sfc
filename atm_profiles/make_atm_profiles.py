"""Generate atmospheric profiles (atm + ch4) for the atm_profile_targets catalog.

Standalone driver: it produces the ``atm_profiles_*.dat`` / ``ch4_profiles_*.dat``
files that ``ssfr_atm_corr.workflow`` normally writes, but WITHOUT collecting the
SSFR flux pickle. For each target it loads only housekeeping (HSK) to fix the
lat/lon box and altitude, optionally loads MARLI water vapor and the nearest
dropsonde, then calls ``prepare_atmospheric_profile`` directly.

The MODIS-07 download inside ``prepare_atmospheric_profile`` requires a valid
Earthdata/LAADS token.

Usage:
    conda run -n er3t_env python make_atm_profiles.py                # all targets
    conda run -n er3t_env python make_atm_profiles.py --ids atm_001,atm_003
    conda run -n er3t_env python make_atm_profiles.py --overwrite
"""

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless: importing the pipeline pulls in matplotlib

# --- bootstrap: make the lrt_sim pipeline importable and resolve relative data paths ---
_LRT_SIM = Path(__file__).resolve().parents[1] / 'lrt_sim'
for _p in (str(_LRT_SIM), str(_LRT_SIM / 'ssfr_atm_corr')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# settings.py defines _fdir_general_='../data' etc. relative to lrt_sim/; chdir so
# the MODIS download, climatology, and cached sat-data resolve as in the workflow.
os.chdir(_LRT_SIM)

import numpy as np

from ssfr_atm_corr.preprocess import make_default_config, load_marli
from ssfr_atm_corr.setup import default_atm_levels, load_nearest_dropsonde
from ssfr_atm_corr.settings import _fdir_general_
from util import load_h5, nearest_indices, prepare_atmospheric_profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atm_profile_targets import (
    ATM_PROFILE_TARGETS,
    DEFAULT_CASE_TAG,
    HALF_WINDOW_MIN,
    get_target,
    target_window,
)

log = logging.getLogger("atm_profiles")


def _marli_leg(config, date_s, t_hsk, mask):
    """Return (marli_h, marli_wvmr) for the leg, or (None, None) if unavailable.

    Mirrors the MARLI handling in ssfr_atm_corr.preprocess.
    """
    data_marli = load_marli(config, date_s)
    t_marli = data_marli['time']
    if len(t_marli) == 0:
        return None, None
    sel_marli = nearest_indices(t_hsk, mask, t_marli)
    marli_wvmr = data_marli["WVMR"][sel_marli, :]
    marli_wvmr[marli_wvmr == 9999] = np.nan
    marli_wvmr[marli_wvmr > 50] = np.nan
    marli_wvmr[marli_wvmr < 0] = 0
    marli_h = data_marli["H"][...]
    marli_mask = np.any(np.isfinite(marli_wvmr), axis=0)
    return marli_h[marli_mask], np.nanmean(marli_wvmr[:, marli_mask], axis=0)


def make_one(target, ileg, half_window_min=HALF_WINDOW_MIN, case_tag=DEFAULT_CASE_TAG,
             overwrite=False):
    """Generate the atm + ch4 profile files for one catalog target."""
    year, month, day = [int(part) for part in target['date'].split('-')]
    date = datetime.datetime(year, month, day)
    date_s = date.strftime("%Y%m%d")
    # Fold the observation time (HH:MM -> HHMM) into case_tag so it appears in the
    # output filenames, e.g. atm_profiles_20240725_sigma_atm_1142_...dat
    case_tag = f"{case_tag}_{target['time'].replace(':', '')}"
    win_start, win_end = target_window(target, half_window_min)

    config = make_default_config()
    data_hsk = load_h5(config.hsk(date_s))
    t_hsk = np.asarray(data_hsk["tmhr"])
    mask = (t_hsk >= win_start) & (t_hsk <= win_end)
    if not np.any(mask):
        raise ValueError(
            f"No HSK samples in window {win_start:.3f}-{win_end:.3f}h "
            f"for {date_s} ({target['id']})"
        )

    times_leg = t_hsk[mask]
    lon = data_hsk["lon"][mask]
    lat = data_hsk["lat"][mask]
    alt = data_hsk["alt"][mask] / 1000.0  # km
    if not np.any(np.isfinite(lon) & np.isfinite(lat)):
        raise ValueError(f"No valid lat/lon in window for {date_s} ({target['id']})")

    time_start, time_end = float(times_leg[0]), float(times_leg[-1])
    alt_avg = np.round(np.nanmean(alt), 2)
    mod_extent = [
        np.round(np.nanmin(lon), 2), np.round(np.nanmax(lon), 2),
        np.round(np.nanmin(lat), 2), np.round(np.nanmax(lat), 2),
    ]

    zpt_filedir = f'{_fdir_general_}/zpt/{date_s}'
    os.makedirs(zpt_filedir, exist_ok=True)
    atm_file = (f'{zpt_filedir}/atm_profiles_{date_s}_{case_tag}_'
                f'{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.dat')
    ch4_file = (f'{zpt_filedir}/ch4_profiles_{date_s}_{case_tag}_'
                f'{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km.dat')
    if os.path.exists(atm_file) and os.path.exists(ch4_file) and not overwrite:
        print(f"[{target['id']}] profiles exist, skipping: {os.path.basename(atm_file)}")
        return atm_file, ch4_file

    marli_h, marli_wvmr = _marli_leg(config, date_s, t_hsk, mask)
    data_dropsonde = load_nearest_dropsonde(
        _fdir_general_, date, [[time_start, time_end]], log
    )[0]

    cld_leg = {'marli_h': marli_h, 'marli_wvmr': marli_wvmr, 'lon': lon, 'lat': lat}

    print(
        f"[{target['id']}] {date_s} {target['site']} "
        f"{time_start:.3f}-{time_end:.3f}h alt {alt_avg:.2f}km extent {mod_extent}"
    )
    prepare_atmospheric_profile(
        _fdir_general_, date_s, case_tag, ileg, date, time_start, time_end,
        alt_avg, data_dropsonde, cld_leg,
        levels=default_atm_levels(),
        mod_extent=mod_extent,
        zpt_filedir=zpt_filedir,
    )
    print(f"[{target['id']}] wrote {os.path.basename(atm_file)} + ch4")
    return atm_file, ch4_file


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', help='comma-separated target ids (default: all)')
    parser.add_argument('--case-tag', default=DEFAULT_CASE_TAG)
    parser.add_argument('--half-window-min', type=float, default=HALF_WINDOW_MIN)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.ids:
        selected = [get_target(tid.strip()) for tid in args.ids.split(',')]
    else:
        selected = list(ATM_PROFILE_TARGETS)

    # Stable ileg from full-catalog position keeps intermediate zpt files distinct.
    ileg_by_id = {t['id']: i for i, t in enumerate(ATM_PROFILE_TARGETS)}

    ok, failed = [], []
    for target in selected:
        try:
            make_one(
                target, ileg_by_id[target['id']],
                half_window_min=args.half_window_min,
                case_tag=args.case_tag,
                overwrite=args.overwrite,
            )
            ok.append(target['id'])
        except Exception as err:
            failed.append((target['id'], f"{type(err).__name__}: {err}"))
            print(f"[{target['id']}] FAILED: {type(err).__name__}: {err}")

    print(f"\nDone. {len(ok)}/{len(selected)} succeeded.")
    if failed:
        print("Failures:")
        for tid, msg in failed:
            print(f"  {tid}: {msg}")


if __name__ == '__main__':
    main()
