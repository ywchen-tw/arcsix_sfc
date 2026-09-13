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
    from .cre_cases import CRE_CASE_IDS, CRE_SZA_CHUNKS, DEFAULT_CRE_CASE_ID, MANUAL_ALB_SWEEP, sza_chunk
else:
    from cre_cases import CRE_CASE_IDS, CRE_SZA_CHUNKS, DEFAULT_CRE_CASE_ID, MANUAL_ALB_SWEEP, sza_chunk


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
    include_sza_avg=False,
    manual_atm_file=None,
    manual_ch4_file=None,
    workers=None,
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
                include_sza_avg=include_sza_avg,
                manual_atm_file=manual_atm_file,
                manual_ch4_file=manual_ch4_file,
                workers=workers,
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
        '--sza-chunk', type=int, default=None, metavar='INDEX',
        help='Run one SZA chunk from cre_cases.CRE_SZA_CHUNKS instead of the full '
             f'grid ({len(CRE_SZA_CHUNKS)} chunks, 0-{len(CRE_SZA_CHUNKS) - 1}). Splits a '
             'case into several smaller cluster jobs; the chunks together cover the '
             'full grid plus the case-mean SZA. Overrides --sza/--sza-avg.',
    )
    parser.add_argument(
        '--sza-avg', action='store_true',
        help='Also run the case-mean SZA alongside an explicit --sza list. An explicit '
             'list otherwise replaces the grid entirely and drops the case mean, which '
             'cre_plot requires. Implied by the relevant --sza-chunk.',
    )
    parser.add_argument(
        '--manual-alb-sweep', action='store_true',
        help='Use the cross-case manual albedo spectra sweep from cre_cases.',
    )
    parser.add_argument(
        '--manual-alb', default=None,
        help='Run a single surface-albedo .dat (filename under data/sfc_alb_cre/) '
             'for the case. One albedo per invocation; pairs with the SLURM job '
             'array. Ignored if --manual-alb-sweep is given.',
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='libRadtran process-pool size for the SZA x CWP sweep. '
             'Default: cpu-2 on Mac, full cpu count on Linux.',
    )
    args = parser.parse_args()

    case_ids = args.case_ids
    if args.all:
        case_ids = list(CRE_CASE_IDS)

    manual_alb = MANUAL_ALB_SWEEP if args.manual_alb_sweep else args.manual_alb

    sza_list = args.sza
    include_sza_avg = args.sza_avg
    if args.sza_chunk is not None:
        sza_list, include_sza_avg = sza_chunk(args.sza_chunk)
        print(f"SZA chunk {args.sza_chunk}/{len(CRE_SZA_CHUNKS) - 1}: {sza_list}"
              f"{' + case-mean' if include_sza_avg else ''}")

    run_cre_cases(
        case_id=args.case_id,
        case_ids=case_ids,
        mode=args.mode,
        overwrite_lrt=args.overwrite_lrt,
        manual_alb=manual_alb,
        sza_list=sza_list,
        include_sza_avg=include_sza_avg,
        manual_atm_file=args.atm_file,
        manual_ch4_file=args.ch4_file,
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
