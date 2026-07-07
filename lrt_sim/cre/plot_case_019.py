"""Generate CRE plots for case_019 (2024-06-13 cloudy_atm_corr_1).

Thin driver around cre_plot.plot_cre_case(). Reads the per-SZA flux CSVs that
cre_sim wrote and builds the combined CRE figures under fig/20240613/.

case_019's CRE sweep is still running on CURC, so a partial run only plots the
albedos whose SZA CSVs are already on disk (missing ones are skipped, not fatal).
Re-run as more albedos finish.

Run: python cre/plot_case_019.py  (from anywhere; it chdirs to the lrt_sim root
so ../data and fig/ resolve the same way cre_plot's __main__ expects).
"""

import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_LRT_SIM_ROOT = str(_THIS_FILE.parents[1])   # .../lrt_sim
_REPO_ROOT = str(_THIS_FILE.parents[2])      # repo root
for _path in (_REPO_ROOT, _LRT_SIM_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# _fdir_general_ is the cwd-relative '../data' on Mac and fig/ is written to the
# cwd, so pin the working directory to the lrt_sim root regardless of launch dir.
os.chdir(_LRT_SIM_ROOT)

from cre.cre_plot import plot_cre_case, make_default_config
from cre.cre_cases import MANUAL_ALB_SWEEP

os.makedirs('./fig', exist_ok=True)
config = make_default_config()

plot_cre_case(
    config,
    'case_019',
    manual_alb=MANUAL_ALB_SWEEP,
    overwrite_lrt=False,      # plotting never re-runs libRadtran
    force_rebuild=True,       # rebuild the aggregate cache from the per-SZA CSVs
    # obs_alb_file=None -> observation albedo = combined case mean (default);
    # case_019's own albedo is index 6 of MANUAL_ALB_SWEEP (broadband ~0.676).
)
print('DONE case_019')
