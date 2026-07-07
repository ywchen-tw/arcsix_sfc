"""Generate CRE plots for case_004 (2024-06-03 cloudy_atm_corr_2).

Thin driver around cre_plot.plot_cre_case(). Reads the per-SZA flux CSVs that
cre_sim wrote and builds the combined CRE figures under fig/20240603/.

Run: python cre/plot_case_004.py  (from anywhere; it chdirs to the lrt_sim root
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
    'case_004',
    manual_alb=MANUAL_ALB_SWEEP,
    overwrite_lrt=False,      # plotting never re-runs libRadtran
    force_rebuild=True,       # rebuild the aggregate cache from the per-SZA CSVs
    # Observation = case_004 peak 2-min albedo (broadband ~0.758), matching the
    # cre_plot __main__ default for this case.
    obs_alb_file='sfc_alb_20240603_14.716_14.749_0.34km_cre_alb.dat',
)
print('DONE case_004')
