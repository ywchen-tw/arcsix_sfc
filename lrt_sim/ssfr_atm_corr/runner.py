"""Runnable SSFR atmospheric-correction cases.

Keep case selection here so the main correction module can stay focused on the
workflow implementation.
"""

import os
import sys
import argparse
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_LRT_SIM_ROOT = str(_THIS_FILE.parents[1])
_REPO_ROOT = str(_THIS_FILE.parents[2])
for _path in (_REPO_ROOT, _LRT_SIM_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if __package__:
    from .case_catalog import run_catalog_case
    from .settings import _fdir_data_, _fdir_general_
else:
    from case_catalog import run_catalog_case
    from settings import _fdir_data_, _fdir_general_


DEFAULT_CASE_ID = 'case_029'

CLEAR_SKY_CASE_ID_LIST = [
    'case_029', 'case_030', 'case_031', 'case_034', 'case_035',
    'case_036', 'case_038', 'case_041', 'case_042', 'case_043',
    'case_047', 'case_051', 'case_052', 'case_054', 'case_055',
    'case_056', 'case_057', 'case_058', 
    # 'case_061', 'case_062', 'case_063', 'case_067', 
    'case_069',
]

CLOUDY_CASE_ID_LIST = [
    'case_032', 'case_033', 'case_039', 'case_044', 'case_045',
    'case_046', 'case_048', 'case_049', 'case_059', 'case_060',
    'case_064', 'case_065', 'case_066', 'case_068',
]

SPIRAL_CASE_ID_LIST = [
    'case_037', 'case_040', 'case_050', 'case_053',
]

CASE_ID_LIST = sorted(CLEAR_SKY_CASE_ID_LIST + CLOUDY_CASE_ID_LIST + SPIRAL_CASE_ID_LIST)


def make_default_config():
    from util import FlightConfig

    return FlightConfig(
        mission='ARCSIX',
        platform='P3B',
        data_root=_fdir_data_,
        root_mac=_fdir_general_,
        root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data',
    )


def run_cases(
    flt_trk_atm_corr,
    case_id=DEFAULT_CASE_ID,
    case_ids=None,
    overwrite_lrt=True,
    rerun_simulation=False,
    iterations=range(3),
    closure_check=True,
    closure_thresholds=None,
    min_closure_iteration=2,
    max_additional_iterations=5,
    run_final_sim=True,
    run_final_extension_rt=False,
    skip_missing_cloud_observations=True,
):
    """Run one or more surface-albedo atmospheric-correction catalog cases."""
    os.makedirs('./fig', exist_ok=True)
    config = make_default_config()

    if case_ids is None:
        case_ids = [case_id]

    # IMPORTANT: run arcsix_gas_insitu.py first to generate gas files for each date.
    for selected_case_id in case_ids:
        print(f"Running track case: {selected_case_id}")
        run_catalog_case(
            flt_trk_atm_corr,
            config,
            selected_case_id,
            overwrite_lrt=overwrite_lrt,
            rerun_simulation=rerun_simulation,
            iterations=iterations,
            closure_check=closure_check,
            closure_thresholds=closure_thresholds,
            min_closure_iteration=min_closure_iteration,
            max_additional_iterations=max_additional_iterations,
            run_final_sim=run_final_sim,
            run_final_extension_rt=run_final_extension_rt,
            skip_missing_cloud_observations=skip_missing_cloud_observations,
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Run SSFR atmospheric-correction catalog cases.')
    parser.add_argument(
        'case_ids',
        nargs='*',
        help=(
            'Case IDs to run. Use case_### catalog IDs. '
            f'Defaults to {DEFAULT_CASE_ID}.'
        ),
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all clear-sky, cloudy, and spiral-like track cases.',
    )
    parser.add_argument(
        '--track-all',
        action='store_true',
        help='Run all track cases: clear-sky, cloudy, and spiral-like.',
    )
    parser.add_argument(
        '--clear-sky-all',
        action='store_true',
        help='Run all clear-sky track cases.',
    )
    parser.add_argument(
        '--cloudy-all',
        action='store_true',
        help='Run all cloudy track cases.',
    )
    parser.add_argument(
        '--spiral-all',
        action='store_true',
        help='Run all spiral-like track cases.',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=8,
        help='Number of track-workflow iterations to allow. Default: 8.',
    )
    parser.add_argument(
        '--no-overwrite-lrt',
        action='store_true',
        help='Do not overwrite existing libRadtran products.',
    )
    parser.add_argument(
        '--rerun-simulation',
        '--overwrite-simulation',
        action='store_true',
        help=(
            'Force rerun/overwrite of existing simulation products before final outputs. '
            'This bypasses resume shortcuts that would otherwise reuse existing CSVs.'
        ),
    )
    parser.add_argument(
        '--final-extension-rt',
        action='store_true',
        help=(
            'Run the extended-grid final RT pass from existing native final products. '
            'Run processing first so adjusted *_final_extension.dat albedo files exist.'
        ),
    )
    return parser.parse_args()


def split_selected_case_ids(selected_case_ids):
    """Validate selected IDs and return track workflow case IDs."""
    track_case_ids = []
    unknown_case_ids = []

    for case_id in selected_case_ids:
        if case_id in CASE_ID_LIST:
            track_case_ids.append(case_id)
        else:
            unknown_case_ids.append(case_id)

    if unknown_case_ids:
        valid_text = ', '.join(CASE_ID_LIST)
        unknown_text = ', '.join(unknown_case_ids)
        raise ValueError(f'Unknown case ID(s): {unknown_text}. Valid IDs: {valid_text}')

    return track_case_ids


def main():
    args = parse_args()
    print(f"Runner argv: {' '.join(sys.argv)}")
    print(f"Parsed positional case_ids: {args.case_ids}")

    if args.all:
        selected_case_ids = CASE_ID_LIST
    elif args.track_all or args.clear_sky_all or args.cloudy_all or args.spiral_all:
        selected_case_ids = []
        if args.track_all:
            selected_case_ids.extend(CASE_ID_LIST)
        if args.clear_sky_all:
            selected_case_ids.extend(CLEAR_SKY_CASE_ID_LIST)
        if args.cloudy_all:
            selected_case_ids.extend(CLOUDY_CASE_ID_LIST)
        if args.spiral_all:
            selected_case_ids.extend(SPIRAL_CASE_ID_LIST)
        selected_case_ids = list(dict.fromkeys(selected_case_ids))
    elif args.case_ids:
        selected_case_ids = args.case_ids
    else:
        selected_case_ids = [DEFAULT_CASE_ID]

    track_case_ids = split_selected_case_ids(selected_case_ids)
    overwrite_lrt = not args.no_overwrite_lrt
    if args.rerun_simulation:
        overwrite_lrt = True
    if track_case_ids:
        print(f"Selected track case(s): {', '.join(track_case_ids)}")

    if track_case_ids:
        if __package__:
            from .workflow import flt_trk_atm_corr
        else:
            from workflow import flt_trk_atm_corr

        run_cases(
            flt_trk_atm_corr,
            case_ids=track_case_ids,
            overwrite_lrt=overwrite_lrt,
            rerun_simulation=args.rerun_simulation,
            iterations=range(args.iterations),
            run_final_extension_rt=args.final_extension_rt,
        )

if __name__ == '__main__':
    main()
