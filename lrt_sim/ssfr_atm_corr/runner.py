"""Runnable SSFR atmospheric-correction cases.

Keep case selection here so the main correction module can stay focused on the
workflow implementation.
"""

import os
import sys
import datetime
import argparse
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_LRT_SIM_ROOT = str(_THIS_FILE.parents[1])
_REPO_ROOT = str(_THIS_FILE.parents[2])
for _path in (_REPO_ROOT, _LRT_SIM_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if __package__:
    from .case_catalog import SPIRAL_CASE_CATALOG, run_catalog_case
    from .settings import _fdir_data_, _fdir_general_
else:
    from case_catalog import SPIRAL_CASE_CATALOG, run_catalog_case
    from settings import _fdir_data_, _fdir_general_


DEFAULT_CASE_ID = 'case_029'

DEFAULT_LEVEL_CASE_ID_LIST = [
    'case_029', 'case_030', 'case_031', 'case_034', 'case_035',
    'case_036', 'case_037', 'case_038', 'case_040', 'case_041',
    'case_042', 'case_043', 'case_047', 'case_050', 'case_051',
    'case_052', 'case_053', 'case_054', 'case_055', 'case_056',
    'case_057', 'case_058', 'case_061', 'case_062', 'case_063',
    'case_067', 'case_069',
]

CUSTOM_LEVEL_CASE_ID_LIST = [
    'case_032', 'case_033', 'case_039', 'case_044', 'case_045',
    'case_046', 'case_048', 'case_049', 'case_059', 'case_060',
    'case_064', 'case_065', 'case_066', 'case_068',
]

CASE_ID_LIST = sorted(DEFAULT_LEVEL_CASE_ID_LIST + CUSTOM_LEVEL_CASE_ID_LIST)

SPIRAL_CASE_ID_LIST = [
    'spiral_001', 'spiral_002', 'spiral_003', 'spiral_004', 'spiral_005',
]


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
    iterations=range(3),
    closure_check=True,
    closure_thresholds=None,
    min_closure_iteration=2,
    max_additional_iterations=5,
    run_final_sim=True,
    skip_missing_cloud_observations=True,
):
    """Run one or more surface-albedo atmospheric-correction catalog cases."""
    os.makedirs('./fig', exist_ok=True)
    config = make_default_config()

    if case_ids is None:
        case_ids = [case_id]

    # IMPORTANT: run arcsix_gas_insitu.py first to generate gas files for each date.
    for selected_case_id in case_ids:
        run_catalog_case(
            flt_trk_atm_corr,
            config,
            selected_case_id,
            overwrite_lrt=overwrite_lrt,
            iterations=iterations,
            closure_check=closure_check,
            closure_thresholds=closure_thresholds,
            min_closure_iteration=min_closure_iteration,
            max_additional_iterations=max_additional_iterations,
            run_final_sim=run_final_sim,
            skip_missing_cloud_observations=skip_missing_cloud_observations,
        )


def run_spiral_cases(atm_corr_spiral_plot, spiral_case_ids=None):
    """Run one or more legacy spiral-plot catalog cases."""
    os.makedirs('./fig', exist_ok=True)
    config = make_default_config()

    if spiral_case_ids is None:
        spiral_case_ids = SPIRAL_CASE_ID_LIST

    spiral_cases = {case['id']: case for case in SPIRAL_CASE_CATALOG}
    for spiral_case_id in spiral_case_ids:
        case = spiral_cases[spiral_case_id]
        year, month, day = [int(part) for part in case['date'].split('-')]
        atm_corr_spiral_plot(
            date=datetime.datetime(year, month, day),
            tmhr_ranges_select=case['tmhr_ranges_select'],
            case_tag=case['case_tag'],
            config=config,
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Run SSFR atmospheric-correction catalog cases.')
    parser.add_argument(
        'case_ids',
        nargs='*',
        help=(
            'Case IDs to run. Use case_### for track cases and spiral_### for spiral cases. '
            f'Defaults to {DEFAULT_CASE_ID}.'
        ),
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all track and spiral cases.',
    )
    parser.add_argument(
        '--track-all',
        action='store_true',
        help='Run all track cases.',
    )
    parser.add_argument(
        '--spiral-all',
        action='store_true',
        help='Run all spiral cases.',
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
    return parser.parse_args()


def split_selected_case_ids(selected_case_ids):
    """Split selected IDs into track and spiral case IDs."""
    track_case_ids = []
    spiral_case_ids = []
    unknown_case_ids = []

    for case_id in selected_case_ids:
        if case_id in CASE_ID_LIST:
            track_case_ids.append(case_id)
        elif case_id in SPIRAL_CASE_ID_LIST:
            spiral_case_ids.append(case_id)
        else:
            unknown_case_ids.append(case_id)

    if unknown_case_ids:
        valid_text = ', '.join(CASE_ID_LIST + SPIRAL_CASE_ID_LIST)
        unknown_text = ', '.join(unknown_case_ids)
        raise ValueError(f'Unknown case ID(s): {unknown_text}. Valid IDs: {valid_text}')

    return track_case_ids, spiral_case_ids


def main():
    args = parse_args()

    if args.all:
        selected_case_ids = CASE_ID_LIST + SPIRAL_CASE_ID_LIST
    elif args.track_all or args.spiral_all:
        selected_case_ids = []
        if args.track_all:
            selected_case_ids.extend(CASE_ID_LIST)
        if args.spiral_all:
            selected_case_ids.extend(SPIRAL_CASE_ID_LIST)
    elif args.case_ids:
        selected_case_ids = args.case_ids
    else:
        selected_case_ids = [DEFAULT_CASE_ID]

    track_case_ids, spiral_case_ids = split_selected_case_ids(selected_case_ids)
    overwrite_lrt = not args.no_overwrite_lrt

    if track_case_ids:
        if __package__:
            from .workflow import flt_trk_atm_corr
        else:
            from workflow import flt_trk_atm_corr

        run_cases(
            flt_trk_atm_corr,
            case_ids=track_case_ids,
            overwrite_lrt=overwrite_lrt,
            iterations=range(args.iterations),
        )

    if spiral_case_ids:
        if __package__:
            from .spiral import atm_corr_spiral_plot
        else:
            from spiral import atm_corr_spiral_plot

        run_spiral_cases(
            atm_corr_spiral_plot,
            spiral_case_ids=spiral_case_ids,
        )


if __name__ == '__main__':
    main()
