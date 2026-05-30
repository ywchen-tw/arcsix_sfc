"""Runnable post-processing for SSFR atmospheric-correction products."""

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
    from .runner import CASE_ID_LIST, DEFAULT_CASE_ID
else:
    from runner import CASE_ID_LIST, DEFAULT_CASE_ID


def run_processing_cases(case_id=DEFAULT_CASE_ID, case_ids=None, output_dir=None):
    """Process one or more atmospheric-correction catalog cases."""
    if __package__:
        from .processing import make_default_config, process_catalog_case
    else:
        from processing import make_default_config, process_catalog_case

    os.makedirs('./fig', exist_ok=True)
    config = make_default_config()
    if case_ids is None:
        case_ids = [case_id]

    for selected_case_id in case_ids:
        process_catalog_case(config, selected_case_id, output_dir=output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Post-process SSFR atmospheric-correction catalog cases.')
    parser.add_argument(
        'case_ids',
        nargs='*',
        help='Case IDs to process. Defaults to DEFAULT_CASE_ID unless --all is set.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all case IDs listed in ssfr_atm_corr.runner.CASE_ID_LIST.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Optional output directory for processed pickle files.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.all:
        selected_case_ids = CASE_ID_LIST
    elif args.case_ids:
        selected_case_ids = args.case_ids
    else:
        selected_case_ids = [DEFAULT_CASE_ID]

    run_processing_cases(
        case_ids=selected_case_ids,
        output_dir=args.output_dir,
    )
