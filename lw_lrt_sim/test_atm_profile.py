"""
Test for prepare_atmospheric_profile in lw_sim/util/atm_profile.py.

Uses:
  - MODIS MOD07_L2.A2024151.1625 (May 30, 2024, 16:25 UTC)
  - Climatology files in data/climatology/
  - Output written to data/test_atm_profiles/

Acknowledgement
---------------
util/atm_profile.py is a standalone replication of atmospheric-profile
preparation logic originally implemented in EaR³T
(Education and Research 3D Radiative Transfer Toolbox):

  Authors : Vikas Nataraja, Yu-Wen Chen, Ken Hirata, Hong Chen, Sebastian Schmidt
  GitHub  : https://github.com/hong-chen/er3t
  Paper   : Chen et al. (2023), AMT, doi:10.5194/amt-16-1971-2023
  License : GNU GPLv3

Original sources:
  er3t.pre.atm.create_modis_dropsonde_atm       → er3t/pre/atm/
  er3t.pre.atm.modis_dropsonde_arcsix_atmmod    → er3t/pre/atm/
"""

import sys
import datetime
import numpy as np

sys.path.insert(0, 'util')
from atm_profile import prepare_atmospheric_profile

# Custom altitude levels (km): fine resolution near surface, coarser aloft
levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                   7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])

prepare_atmospheric_profile(
    fdir_data   = 'data',
    case_tag    = 'test',
    date        = datetime.date(2024, 5, 30),   # day 151 of 2024
    sfc_alt_avg = 0.0, # in km
    fname_mod07 = 'data/MOD07_L2.A2024151.1625.061.2024152014730.hdf',
    levels      = levels,
    # Note: extent format is [lon_west, lon_east, lat_south, lat_north]
    # The default [-60, -80, 82.4, 84.6] has lon_west/east swapped — always pass explicitly.
    mod_extent  = [-80.0, -60.0, 82.4, 84.6],
    zpt_filedir = 'output/test_atm_profiles',
    fname_std_atm = 'data/atmmod/afglss.dat', # can be found in libRadtran/data/atmmod
    plot        = True,
)

print('Done! Output written to output/test_atm_profiles')
