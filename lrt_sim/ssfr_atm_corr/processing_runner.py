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
    from .runner import (
        CASE_ID_LIST,
        CLEAR_SKY_CASE_ID_LIST,
        CLOUDY_CASE_ID_LIST,
        DEFAULT_CASE_ID,
        SPIRAL_CASE_ID_LIST,
    )
else:
    from runner import (
        CASE_ID_LIST,
        CLEAR_SKY_CASE_ID_LIST,
        CLOUDY_CASE_ID_LIST,
        DEFAULT_CASE_ID,
        SPIRAL_CASE_ID_LIST,
    )


def run_processing_cases(
    case_id=DEFAULT_CASE_ID,
    case_ids=None,
    output_dir=None,
    make_plots=True,
    fig_dir='fig',
    plot_every=1,
    force_row_extension=False,
):
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
        if selected_case_id not in CASE_ID_LIST:
            valid_text = ', '.join(CASE_ID_LIST)
            raise ValueError(f'Unknown track case ID: {selected_case_id}. Valid IDs: {valid_text}')
        process_catalog_case(
            config,
            selected_case_id,
            output_dir=output_dir,
            make_plots=make_plots,
            fig_dir=fig_dir,
            plot_every=plot_every,
            force_row_extension=force_row_extension,
        )


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
        help='Process all clear-sky, cloudy, and spiral-like track case IDs.',
    )
    parser.add_argument(
        '--clear-sky-all',
        action='store_true',
        help='Process all clear-sky track case IDs.',
    )
    parser.add_argument(
        '--cloudy-all',
        action='store_true',
        help='Process all cloudy track case IDs.',
    )
    parser.add_argument(
        '--spiral-all',
        action='store_true',
        help='Process all spiral-like track case IDs.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Optional output directory for processed pickle files.',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Write processed pickle files without generating diagnostic plots.',
    )
    parser.add_argument(
        '--fig-dir',
        default='fig',
        help='Base directory for diagnostic plots. Default: fig.',
    )
    parser.add_argument(
        '--plot-every',
        type=int,
        default=1,
        help='Plot every Nth leg for per-leg diagnostics. Default: 1.',
    )
    parser.add_argument(
        '--force-row-extension',
        action='store_true',
        help='Force alb_extention(...) for every 1-second row instead of using final-extension template scaling.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.all:
        selected_case_ids = CASE_ID_LIST
    elif args.clear_sky_all or args.cloudy_all or args.spiral_all:
        selected_case_ids = []
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

    run_processing_cases(
        case_ids=selected_case_ids,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        fig_dir=args.fig_dir,
        plot_every=args.plot_every,
        force_row_extension=args.force_row_extension,
    )
