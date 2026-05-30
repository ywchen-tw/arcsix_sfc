"""Runnable preprocessing for SSFR atmospheric-correction cases."""

import argparse
import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_LRT_SIM_ROOT = str(_THIS_FILE.parents[1])
_REPO_ROOT = str(_THIS_FILE.parents[2])
for _path in (_REPO_ROOT, _LRT_SIM_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if __package__:
    from .runner import CASE_ID_LIST, DEFAULT_CASE_ID, SPIRAL_CASE_ID_LIST
else:
    from runner import CASE_ID_LIST, DEFAULT_CASE_ID, SPIRAL_CASE_ID_LIST


def run_preprocess_cases(case_id=DEFAULT_CASE_ID, case_ids=None, overwrite=False, plot_qc=True):
    """Preprocess one or more catalog cases for atmospheric correction."""
    if __package__:
        from .preprocess import make_default_config, preprocess_catalog_case, preprocess_spiral_catalog_case
    else:
        from preprocess import make_default_config, preprocess_catalog_case, preprocess_spiral_catalog_case

    os.makedirs('./fig', exist_ok=True)
    config = make_default_config()
    if case_ids is None:
        case_ids = [case_id]

    for selected_case_id in case_ids:
        if selected_case_id in CASE_ID_LIST:
            preprocess_catalog_case(
                config,
                selected_case_id,
                overwrite=overwrite,
                plot_qc=plot_qc,
            )
        elif selected_case_id in SPIRAL_CASE_ID_LIST:
            preprocess_spiral_catalog_case(
                config,
                selected_case_id,
                overwrite=overwrite,
                plot_qc=plot_qc,
            )
        else:
            valid_text = ', '.join(CASE_ID_LIST + SPIRAL_CASE_ID_LIST)
            raise ValueError(f'Unknown case ID: {selected_case_id}. Valid IDs: {valid_text}')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess SSFR atmospheric-correction catalog cases.')
    parser.add_argument(
        'case_ids',
        nargs='*',
        help='Case IDs to preprocess. Defaults to DEFAULT_CASE_ID unless --all is set.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Preprocess all track and spiral case IDs.',
    )
    parser.add_argument(
        '--track-all',
        action='store_true',
        help='Preprocess all track case IDs.',
    )
    parser.add_argument(
        '--spiral-all',
        action='store_true',
        help='Preprocess all spiral case IDs.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing flt_cld_obs_info pickle files.',
    )
    parser.add_argument(
        '--no-qc-plot',
        action='store_true',
        help='Skip SSFR/HSR1 time-series QC plots.',
    )
    return parser.parse_args()


if __name__ == '__main__':
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

    run_preprocess_cases(
        case_ids=selected_case_ids,
        overwrite=args.overwrite,
        plot_qc=not args.no_qc_plot,
    )
