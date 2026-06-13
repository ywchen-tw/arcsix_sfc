"""Runnable cloud-radiative-effect (CRE) simulation for ARCSIX cases.

Reads surface albedo from the combined atmospheric-correction product
(``sfc_alb_combined_spring_summer.pkl``) and cloud microphysics from the shared
``ssfr_atm_corr`` case catalog, then runs libRadtran CRE simulations.
"""

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
    from .cre_cases import CRE_CASE_IDS, DEFAULT_CRE_CASE_ID, MANUAL_ALB_SWEEP
else:
    from cre_cases import CRE_CASE_IDS, DEFAULT_CRE_CASE_ID, MANUAL_ALB_SWEEP


def _lw_modes(mode):
    """Map a --mode string to the list of lw flags to run."""
    return {'sw': [False], 'lw': [True], 'both': [False, True]}[mode]


def run_cre_cases(
    case_id=DEFAULT_CRE_CASE_ID,
    case_ids=None,
    mode='both',
    overwrite_lrt=False,
    manual_alb=None,
    sza_list=None,
    manual_atm_file=None,
    manual_ch4_file=None,
):
    """Run CRE simulations for one or more catalog cases in SW and/or LW."""
    if __package__:
        from .cre_sim import make_default_config, process_cre_case
    else:
        from cre_sim import make_default_config, process_cre_case

    os.makedirs('./fig', exist_ok=True)
    config = make_default_config()
    if case_ids is None:
        case_ids = [case_id]

    for selected_case_id in case_ids:
        for lw in _lw_modes(mode):
            print(f"=== CRE {'LW' if lw else 'SW'} for {selected_case_id} ===")
            process_cre_case(
                config,
                selected_case_id,
                lw=lw,
                manual_alb=manual_alb,
                overwrite_lrt=overwrite_lrt,
                sza_list=sza_list,
                manual_atm_file=manual_atm_file,
                manual_ch4_file=manual_ch4_file,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--case-id', default=DEFAULT_CRE_CASE_ID,
        help=f'Single catalog case id to run. Defaults to {DEFAULT_CRE_CASE_ID}.',
    )
    parser.add_argument(
        '--case-ids', nargs='+', default=None,
        help='Multiple catalog case ids to run (overrides --case-id).',
    )
    parser.add_argument(
        '--all', action='store_true',
        help=f'Run every CRE case: {", ".join(CRE_CASE_IDS)}.',
    )
    parser.add_argument(
        '--mode', choices=['sw', 'lw', 'both'], default='both',
        help='Run shortwave, longwave, or both. Defaults to both.',
    )
    parser.add_argument(
        '--overwrite-lrt', action='store_true',
        help='Re-run libRadtran even if output files already exist.',
    )
    parser.add_argument(
        '--sza', nargs='+', type=float, default=None,
        help='Explicit solar-zenith-angle list (deg). Pass one value for a quick '
             '1-geometry test, e.g. --sza 50. Default uses the built-in grid.',
    )
    parser.add_argument(
        '--atm-file', default=None,
        help='Reuse an existing atmospheric profile (skips MODIS-based creation). '
             'A bare filename is resolved under data/zpt/<date>/; a path is used as-is.',
    )
    parser.add_argument(
        '--ch4-file', default=None,
        help='Matching CH4 profile. If omitted, derived from --atm-file by '
             'replacing "atm_profiles" with "ch4_profiles".',
    )
    parser.add_argument(
        '--manual-alb-sweep', action='store_true',
        help='Use the cross-case manual albedo spectra sweep from cre_cases.',
    )
    args = parser.parse_args()

    case_ids = args.case_ids
    if args.all:
        case_ids = list(CRE_CASE_IDS)

    manual_alb = MANUAL_ALB_SWEEP if args.manual_alb_sweep else None

    run_cre_cases(
        case_id=args.case_id,
        case_ids=case_ids,
        mode=args.mode,
        overwrite_lrt=args.overwrite_lrt,
        manual_alb=manual_alb,
        sza_list=args.sza,
        manual_atm_file=args.atm_file,
        manual_ch4_file=args.ch4_file,
    )


if __name__ == '__main__':
    main()
