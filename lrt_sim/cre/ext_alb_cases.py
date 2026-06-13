"""Declarative catalog of extended surface-albedo cases for CRE, plus a generator.

Each entry describes how to build one libRadtran surface-albedo ``.dat`` from the
combined atmospheric-correction product (``sfc_alb_combined_spring_summer.pkl``):
a date, a flight time window, the altitude used in the filename, and an optional
scale factor. The generator selects the matching rows, averages the **already
extended** (300-4000 nm) atmospheric-corrected albedo (no re-extension), and
writes the file to ``data/sfc_alb_cre/``.

To add an albedo file, append an entry to ``EXT_ALB_CASES`` and run::

    cd lrt_sim
    conda run -n er3t_env python -m cre.ext_alb_cases            # dry-run preview
    conda run -n er3t_env python -m cre.ext_alb_cases --apply    # write (+ backup)

Case fields
-----------
date        : str   'YYYYMMDD'
time_range  : tuple (t_start, t_end) decimal-hour UTC — selection window and the
                    t0/t1 used in the output filename
alt         : float altitude in km, used only for the filename
scale       : float optional multiplier applied to the albedo (default 1.0)
case_tag    : str   optional extra filter (exact combined-product case tag)

Output filename
---------------
``sfc_alb_{date}_{t0:.3f}_{t1:.3f}_{alt:.2f}km_cre_alb[_scale_{scale}X].dat``
"""

import argparse
import datetime
import os
import pickle
import shutil
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_LRT_SIM_ROOT = str(_THIS_FILE.parents[1])
_REPO_ROOT = str(_THIS_FILE.parents[2])
for _path in (_REPO_ROOT, _LRT_SIM_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd

if __package__:
    from .cre_sim import (
        combined_product_file,
        mean_extended_albedo,
        select_combined_rows,
        write_2col_file,
    )
else:
    from cre_sim import (
        combined_product_file,
        mean_extended_albedo,
        select_combined_rows,
        write_2col_file,
    )

try:
    from ssfr_atm_corr.settings import _fdir_general_
except ImportError:
    from lrt_sim.ssfr_atm_corr.settings import _fdir_general_


# Solar flux used to weight the broadband albedo (slit-convolved SSFR solar flux).
SOLAR_FLUX_FILE = os.path.join(_LRT_SIM_ROOT, 'arcsix_ssfr_solar_flux_slit.dat')


def load_solar_flux(solar_file=SOLAR_FLUX_FILE):
    """Return (wavelength_nm, flux) from a two-column solar-flux .dat file."""
    data = np.loadtxt(solar_file, comments='#')
    return data[:, 0], data[:, 1]


def solar_weighted_broadband(ext_wvl, alb, solar_wvl=None, solar_flux=None,
                             solar_file=SOLAR_FLUX_FILE):
    """Solar-flux-weighted broadband albedo: ∫ alb·F dλ / ∫ F dλ.

    The solar flux is interpolated onto ``ext_wvl`` (zero outside its range), so
    the integral is effectively over the shortwave band the solar file covers.
    """
    if solar_wvl is None:
        solar_wvl, solar_flux = load_solar_flux(solar_file)
    flux = np.interp(ext_wvl, solar_wvl, solar_flux, left=0.0, right=0.0)
    denom = np.trapz(flux, ext_wvl)
    if denom <= 0:
        return np.nan
    return float(np.trapz(np.asarray(alb) * flux, ext_wvl) / denom)


# ---------------------------------------------------------------------------
# Case catalog — add entries here to generate more albedo files.
# ---------------------------------------------------------------------------
EXT_ALB_CASES = [
    {'date': '20240528', 'time_range': (15.610, 17.404), 'alt': 0.22},
    {'date': '20240603', 'time_range': (14.711, 14.868), 'alt': 0.34},
    {'date': '20240605', 'time_range': (12.422, 13.812), 'alt': 5.80},
    {'date': '20240605', 'time_range': (14.258, 15.036), 'alt': 0.11},
    {'date': '20240605', 'time_range': (15.535, 15.918), 'alt': 0.44},
    {'date': '20240606', 'time_range': (16.250, 16.950), 'alt': 0.50},
    {'date': '20240607', 'time_range': (15.336, 15.761), 'alt': 0.12},
    {'date': '20240611', 'time_range': (14.968, 15.347), 'alt': 0.16},
    {'date': '20240611', 'time_range': (15.347, 16.113), 'alt': 0.17},
    {'date': '20240613', 'time_range': (13.704, 13.812), 'alt': 0.21},
    {'date': '20240613', 'time_range': (14.109, 14.140), 'alt': 0.11},
    {'date': '20240613', 'time_range': (15.834, 15.883), 'alt': 0.12},
    {'date': '20240613', 'time_range': (16.043, 16.067), 'alt': 0.16},
    {'date': '20240613', 'time_range': (16.550, 17.581), 'alt': 0.22},
    {'date': '20240725', 'time_range': (15.094, 15.300), 'alt': 0.11},
    {'date': '20240725', 'time_range': (15.881, 15.903), 'alt': 0.33},
    {'date': '20240801', 'time_range': (13.843, 14.351), 'alt': 0.11},
    {'date': '20240807', 'time_range': (13.344, 13.761), 'alt': 0.13},
    {'date': '20240807', 'time_range': (15.472, 15.921), 'alt': 0.11},
    {'date': '20240808', 'time_range': (13.212, 13.345), 'alt': 0.12},
    {'date': '20240809', 'time_range': (13.376, 13.600), 'alt': 0.12},
    {'date': '20240809', 'time_range': (16.029, 16.224), 'alt': 0.11},
    # scaled variants (sensitivity tests)
    {'date': '20240528', 'time_range': (15.610, 17.404), 'alt': 0.22, 'scale': 0.99},
    {'date': '20240808', 'time_range': (15.314, 15.497), 'alt': 0.12, 'scale': 0.97},
    {'date': '20240808', 'time_range': (15.314, 15.497), 'alt': 0.12, 'scale': 1.012},
    # Not in the current combined product (generator will skip with a warning):
    # {'date': '20240531', 'time_range': (13.839, 15.180), 'alt': 5.61},
    # {'date': '20240603', 'time_range': (13.620, 13.750), 'alt': 0.32},  # cloudy_atm_corr_1
]


def case_output_name(case):
    """Return the .dat filename for one case, matching the legacy convention."""
    t0, t1 = case['time_range']
    base = f"sfc_alb_{case['date']}_{t0:.3f}_{t1:.3f}_{case['alt']:.2f}km_cre_alb"
    scale = case.get('scale', 1.0)
    if scale != 1.0:
        base += f"_scale_{scale}X"
    return base + ".dat"


def generate_ext_alb_files(cases=EXT_ALB_CASES, out_dir=None, time_pad=0.001,
                           apply=False, backup=True, combined_file=None,
                           csv_path=None, csv_only=False, solar_file=SOLAR_FLUX_FILE):
    """Generate albedo .dat files for ``cases`` from the combined product.

    Loads the combined pickle once and, for each case, averages the extended
    albedo and computes its solar-flux-weighted broadband albedo (weight =
    ``solar_file``). A summary CSV (filename, date, window, alt, scale, n_rows,
    broadband_albedo) is written next to the albedo files.

    ``apply=False`` (default) previews only. ``apply=True`` writes the .dat files
    (existing ones backed up first when ``backup=True``) and the CSV.
    ``csv_only=True`` writes just the CSV, leaving the .dat files untouched.
    """
    if out_dir is None:
        out_dir = f'{_fdir_general_}/sfc_alb_cre'
    if combined_file is None:
        combined_file = combined_product_file()
    if csv_path is None:
        csv_path = os.path.join(out_dir, 'ext_alb_broadband.csv')

    print(f"Loading combined product: {combined_file}")
    if not os.path.exists(combined_file):
        print("  combined product not found — nothing to do.")
        return
    with open(combined_file, 'rb') as f:
        combined_data = pickle.load(f)
    solar_wvl, solar_flux = load_solar_flux(solar_file)

    os.makedirs(out_dir, exist_ok=True)
    backup_dir = None
    if apply and backup:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(out_dir, f'backup_{stamp}')

    written, skipped, summary = [], [], []
    for case in cases:
        name = case_output_name(case)
        out_path = os.path.join(out_dir, name)
        t0, t1 = case['time_range']
        time_range = (t0 - time_pad, t1 + time_pad)
        scale = case.get('scale', 1.0)

        sel = select_combined_rows(
            case['date'], time_range=time_range,
            case_tag=case.get('case_tag'), combined_data=combined_data,
        )
        n_rows = 0 if sel is None else len(sel['time'])
        ext_wvl, mean_alb = mean_extended_albedo(
            case['date'], time_range=time_range,
            case_tag=case.get('case_tag'), combined_data=combined_data,
        )
        if ext_wvl is None:
            skipped.append(f"{name}  (no combined rows for {case['date']} {time_range})")
            continue

        out_alb = np.clip(mean_alb * scale, 0.0, 1.0)
        broadband = solar_weighted_broadband(ext_wvl, out_alb, solar_wvl, solar_flux)
        tag = (f"{name}  [n={n_rows}, broadband={broadband:.4f}"
               + (f", scale={scale}" if scale != 1.0 else "") + "]")

        if apply:
            if backup and os.path.exists(out_path):
                os.makedirs(backup_dir, exist_ok=True)
                shutil.copy2(out_path, os.path.join(backup_dir, name))
            write_2col_file(
                out_path, ext_wvl, out_alb,
                header=('# SSFR atmospheric-corrected extended sfc albedo '
                        '(mean over flight window from combined product)\n'
                        '# wavelength (nm)      albedo (unitless)\n'),
            )
        written.append(tag)
        summary.append({
            'filename': name,
            'date': case['date'],
            't0': t0,
            't1': t1,
            'alt_km': case['alt'],
            'scale': scale,
            'n_rows': n_rows,
            'broadband_albedo': broadband,
        })

    if apply:
        action = 'Wrote'
    elif csv_only:
        action = 'Computed (CSV only)'
    else:
        action = 'Would write'
    print(f"\n=== {action}: {len(written)} albedo files ===")
    for t in written:
        print(f"  {t}")
    if skipped:
        print(f"\n--- skipped, no combined rows ({len(skipped)}) ---")
        for t in skipped:
            print(f"  skip: {t}")
    if apply and backup_dir and os.path.isdir(backup_dir):
        print(f"\nExisting files backed up to: {backup_dir}")

    if summary and (apply or csv_only):
        pd.DataFrame(summary).to_csv(csv_path, index=False)
        print(f"\nBroadband summary CSV written: {csv_path}")
    elif not apply:
        print("\n(dry run — re-run with --apply to write the .dat files + CSV, "
              "or --csv-only to write just the CSV)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true',
                        help='Write the .dat files + broadband CSV (default: dry-run preview).')
    parser.add_argument('--csv-only', action='store_true',
                        help='Write only the broadband CSV; leave the .dat files untouched.')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not back up existing files before overwriting.')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default data/sfc_alb_cre).')
    parser.add_argument('--csv-path', default=None,
                        help='Broadband CSV path (default <out-dir>/ext_alb_broadband.csv).')
    parser.add_argument('--time-pad', type=float, default=0.001,
                        help='Hours added to each side of the time window to cover '
                             'rounding (default 0.001 h ~ 3.6 s).')
    args = parser.parse_args()
    generate_ext_alb_files(
        out_dir=args.out_dir,
        time_pad=args.time_pad,
        apply=args.apply,
        backup=not args.no_backup,
        csv_path=args.csv_path,
        csv_only=args.csv_only,
    )


if __name__ == '__main__':
    main()
