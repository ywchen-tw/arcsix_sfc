"""Quick CURC smoke test for the CRE simulation: SW+LW, one SZA, one LWP.

Drives ``cre_sim`` directly (mirroring ``process_cre_case``) but collapses the
geometry to a single solar zenith angle and the cloud sweep to a single liquid
water path, and reuses an existing atmospheric profile instead of building one
via MODIS. This makes it a fast end-to-end check that libRadtran runs on the
cluster.

Run from ``lrt_sim/`` on a compute node (not the login node)::

    conda run -n er3t_env python -m cre.test_cre_curc \
        --atm-file atm_profiles_20240603_cloudy_atm_corr_2_14.711_14.868_0.30km.dat \
        --sza 65 --lwp 100

The matching CH4 profile (``ch4_profiles_...``) is required and is derived
automatically from ``--atm-file`` unless ``--ch4-file`` is given.
"""

import argparse
import datetime
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
for _path in (str(_THIS_FILE.parents[2]), str(_THIS_FILE.parents[1])):
    if _path not in sys.path:
        sys.path.insert(0, _path)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case-id', default='case_004',
                    help='Catalog case id (cloud microphysics / levels / time window).')
    ap.add_argument('--sza', type=float, default=65.0,
                    help='Single solar zenith angle [deg].')
    ap.add_argument('--lwp', type=float, default=100.0,
                    help='Single cloud water path [g/m^2].')
    ap.add_argument('--atm-file', required=True,
                    help='Atmospheric profile: bare name resolved under '
                         'data/zpt/<date>/, or an absolute/relative path. '
                         'CH4 profile derived automatically unless --ch4-file given.')
    ap.add_argument('--ch4-file', default=None,
                    help='Matching CH4 profile (overrides the auto-derived name).')
    ap.add_argument('--workers', type=int, default=None,
                    help='libRadtran pool size. Default: cpu-2 on Mac, full cpu on Linux.')
    args = ap.parse_args()

    # Deferred so ``--help`` works without the full er3t/libRadtran stack
    # (matches the import pattern in cre_runner.py).
    from cre.cre_sim import make_default_config, cre_sim, find_catalog_case

    case = find_catalog_case(args.case_id)
    year, month, day = (int(part) for part in case['date'].split('-'))

    cloud_kwargs = {
        key: case[key]
        for key in (
            'manual_cloud', 'manual_cloud_cer', 'manual_cloud_cwp',
            'manual_cloud_cth', 'manual_cloud_cbh', 'manual_cloud_cot',
        )
        if key in case
    }

    config = make_default_config()
    for lw in (False, True):  # SW then LW
        print(f"=== CRE {'LW' if lw else 'SW'}  case={args.case_id} "
              f"sza={args.sza} lwp={args.lwp} g/m^2 ===")
        cre_sim(
            date=datetime.datetime(year, month, day),
            tmhr_ranges_select=case['tmhr_ranges_select'],
            case_tag=case['case_tag'],
            config=config,
            levels=case.get('levels'),
            simulation_interval=case.get('simulation_interval'),
            clear_sky=case.get('clear_sky', True),
            overwrite_lrt=True,
            lw=lw,
            sza_list=[args.sza],
            cwp_list_g=[args.lwp],
            manual_atm_file=args.atm_file,
            manual_ch4_file=args.ch4_file,
            workers=args.workers,
            **cloud_kwargs,
        )
    print("Test finished.")


if __name__ == '__main__':
    main()
