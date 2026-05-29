"""Runnable SSFR atmospheric-correction cases.

Keep case selection here so the main correction module can stay focused on the
workflow implementation.
"""

import os
import sys
import datetime
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_LRT_SIM_ROOT = str(_THIS_FILE.parents[1])
_REPO_ROOT = str(_THIS_FILE.parents[2])
for _path in (_REPO_ROOT, _LRT_SIM_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_IN_LRT_SIM_PACKAGE = bool(__package__) and __package__.startswith('lrt_sim.')

if __package__:
    from .case_catalog import SPIRAL_CASE_CATALOG, run_catalog_case
    from .settings import _fdir_data_, _fdir_general_
else:
    from case_catalog import SPIRAL_CASE_CATALOG, run_catalog_case
    from settings import _fdir_data_, _fdir_general_


DEFAULT_CASE_ID = 'case_057'

CASE_ID_LIST = [
    'case_029', 'case_030', 'case_031', 'case_034',
    'case_035', 'case_036', 'case_037', 'case_038', 'case_040',
    'case_041', 'case_042', 'case_043', 'case_047', 'case_050',
    'case_051', 'case_052', 'case_053', 'case_054', 'case_055',
    'case_056', 'case_057', 'case_058', 'case_061', 'case_062',
    'case_063', 'case_067', 'case_069',
]

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


if __name__ == '__main__':
    RUN_TRACK_CASES = True
    RUN_SPIRAL_CASES = False

    TRACK_CASE_IDS = [DEFAULT_CASE_ID]
    SPIRAL_CASE_IDS = SPIRAL_CASE_ID_LIST
    ITERATIONS = range(8)
    OVERWRITE_LRT = True

    if RUN_TRACK_CASES:
        if __package__:
            from .workflow import flt_trk_atm_corr
        else:
            from workflow import flt_trk_atm_corr

        run_cases(
            flt_trk_atm_corr,
            case_ids=TRACK_CASE_IDS,
            overwrite_lrt=OVERWRITE_LRT,
            iterations=ITERATIONS,
        )

    if RUN_SPIRAL_CASES:
        if _IN_LRT_SIM_PACKAGE:
            from ..ssfr_atm_corr_plot import atm_corr_spiral_plot
        else:
            from ssfr_atm_corr_plot import atm_corr_spiral_plot

        run_spiral_cases(
            atm_corr_spiral_plot,
            spiral_case_ids=SPIRAL_CASE_IDS,
        )
