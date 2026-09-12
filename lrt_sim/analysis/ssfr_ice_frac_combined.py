"""Combine per-flight camera ice-fraction/radiance products into one pickle.

Reads every ``cam_icefrac_rad_*.nc`` file under ``{_fdir_general_}/cam_icefrac_rad``
and concatenates them into ``ice_frac_all.pkl`` in the same directory. That pickle is
the camera ice-fraction input consumed by
``lrt_sim.ssfr_atm_corr.combined.combined_atm_corr`` (collocation) and by
``ssfr_ice_frac_alb_analysis.analyze_ice_frac_alb``.

Ported from ``legacy/ssfr_ice_frac_combined_ori.py``; the original is kept untouched
for reference.
"""

import os
import sys
import platform

# Anchor everything to the lrt_sim/ project dir (parent of this analysis/ folder),
# so the script runs the same regardless of the current working directory.
_BASE_DIR_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../lrt_sim
sys.path.insert(0, _BASE_DIR_)   # make `ssfr_atm_corr` and `util` resolvable
os.chdir(_BASE_DIR_)             # keep relative paths (../data, ./fig, ./tmp) valid

if platform.system() == 'Linux':
    sys.path.append("/projects/yuch8913/arcsix_sfc/lrt_sim/")

import glob
import logging
import pickle

import numpy as np
from netCDF4 import Dataset

from ssfr_atm_corr.settings import _fdir_general_


def combined_ice_frac():
    """Concatenate all camera ice-fraction files into a single ice_frac_all.pkl."""
    log = logging.getLogger("ice frac combined")

    output_dir = f'{_fdir_general_}/cam_icefrac_rad'
    # glob all ice fraction files, e.g. cam_icefrac_rad_20240603_144240_145205.nc
    ice_frac_files = sorted(glob.glob(f'{output_dir}/cam_icefrac_rad_*.nc'))
    print(f"Found {len(ice_frac_files)} ice fraction files for combination.")

    # read each file and combine data into a larger dictionary
    date_all = []
    time_all = []
    ice_frac_all = []
    hdrf_thres_all = []
    nad_hdrf_all = []
    nad_rad_all = []

    for ice_frac_file in ice_frac_files:
        print(f"Processing camera ice-fraction file: {ice_frac_file}")
        date_s = int(ice_frac_file.split('_')[-3])
        with Dataset(ice_frac_file, 'r') as nc:
            time_s = nc['tmhr'][:]
            ice_frac = nc['ice_fraction'][:]
            hdrf_thres = nc['hdrf_threshold'][:]
            nad_hdrf = nc['nadir_hdrf'][:]
            nad_rad = nc['nadir_radiance'][:]

            date_all.extend([date_s] * len(time_s))
            time_all.extend(time_s.tolist())
            ice_frac_all.extend(ice_frac.tolist())
            hdrf_thres_all.extend([hdrf_thres] * len(time_s))
            nad_hdrf_all.extend(nad_hdrf.tolist())
            nad_rad_all.extend(nad_rad.tolist())

    date_all = np.array(date_all)
    time_all = np.array(time_all)
    ice_frac_all = np.array(ice_frac_all)
    hdrf_thres_all = np.array(hdrf_thres_all)
    nad_hdrf_all = np.array(nad_hdrf_all)
    nad_rad_all = np.array(nad_rad_all)

    output_all_dict = {
        'date': date_all,
        'time': time_all,
        'ice_frac': ice_frac_all,
        'hdrf_thres': hdrf_thres_all,
        'nad_hdrf': nad_hdrf_all,
        'nad_rad': nad_rad_all,
    }

    combined_output_file = f'{output_dir}/ice_frac_all.pkl'
    with open(combined_output_file, 'wb') as f:
        pickle.dump(output_all_dict, f)
    print(f"Combined ice fraction data saved to {combined_output_file}")


if __name__ == '__main__':

    combined_ice_frac()
